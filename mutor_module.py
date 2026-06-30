"""MuToR-inspired register-token auxiliary MTP loss.

Reference: "Multi-Token Prediction Needs Registers" (Gerontopoulos et al. 2025,
arXiv:2505.10518). The paper interleaves learnable register tokens into the
training sequence with custom attention masking so registers see only prior
real tokens, and real tokens ignore registers; each register at offset d is
tasked with predicting the token d steps ahead.

Our approximation (MuToR-lite): instead of full sequence interleaving (which
would require 4D attention-mask surgery on HF Qwen3_5Moe's generated causal
mask), we pick random positions p in the already-computed hidden-state stream
and predict token[p+d] from hidden[p] + offset_embed(d). The offset embedding
signals "which horizon" to the shared lm_head — analogous to how a real
register would be distinguished by its learned token embedding.

This preserves the core MuToR signal (multi-horizon supervision through an
offset-conditioned head) while skipping the attention-mask surgery. For Phase 2
the cost is:
    - one Embedding(d_max+1, H): ~5·H params (negligible)
    - K register CE losses per forward pass (K ≈ 32 typical)
    - zero inference cost — the head is discarded
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MuToRHead(nn.Module):
    """Learnable offset embeddings used to distinguish register predictions at
    different horizons."""
    def __init__(self, hidden_size: int, d_max: int = 4) -> None:
        super().__init__()
        self.d_max = d_max
        self.offset_emb = nn.Embedding(d_max + 1, hidden_size)
        nn.init.normal_(self.offset_emb.weight, std=0.02)
        with torch.no_grad():
            self.offset_emb.weight[0].zero_()  # d=0 is unused / no-op slot


def mutor_loss(
    hidden: torch.Tensor,           # [B, T, H] — backbone final hidden
    y: torch.Tensor,                # [B, T] — shifted labels, y[p] = x_full[p+1]
    lm_head: nn.Linear,             # shared vocab projection (reused, not retrained)
    head: MuToRHead,
    num_registers: int = 32,
    d_min: int = 2,
    d_max: int | None = None,
    chunk: int = 1024,
    rng: torch.Generator | None = None,
) -> tuple[torch.Tensor, int]:
    """Sample K positions + offsets per batch elt, compute mean CE over the
    K register predictions. Predictions are `lm_head(hidden[p] + offset_emb(d))`;
    targets are `y[p + d - 1]` (token `d` steps past `x[p]`).

    Returns (mean_loss, n_valid). If no valid positions exist, returns a zero
    tensor with requires_grad=False.
    """
    if d_max is None:
        d_max = head.d_max
    device = hidden.device
    B, T, H = hidden.shape

    # Ensure p + d - 1 < T for all sampled (p, d). We clamp so sampling is
    # always in-range with d up to d_max.
    max_valid_p = T - d_max
    if max_valid_p < 0 or num_registers <= 0:
        return hidden.new_zeros((), dtype=torch.float32), 0
    K = int(min(num_registers, max_valid_p + 1))

    gen = rng
    if gen is None:
        pos_bt = torch.randint(0, max_valid_p + 1, (B, K), device=device)
        d_bt = torch.randint(d_min, d_max + 1, (B, K), device=device)
    else:
        pos_bt = torch.randint(0, max_valid_p + 1, (B, K), device=device, generator=gen)
        d_bt = torch.randint(d_min, d_max + 1, (B, K), device=device, generator=gen)

    # Gather hidden[p] at chosen positions
    idx_bt = pos_bt.unsqueeze(-1).expand(-1, -1, H)
    h_reg = torch.gather(hidden, 1, idx_bt)
    # Offset embedding is broadcast over B
    h_reg = h_reg + head.offset_emb(d_bt)

    # Target token ids: y[p + d - 1]
    tgt_idx = pos_bt + d_bt - 1
    tgt = torch.gather(y, 1, tgt_idx)

    # Flat chunked CE (fp32 materialization like chunked_lmhead_ce)
    flat_h = h_reg.reshape(-1, H)
    flat_t = tgt.reshape(-1)
    N = flat_h.shape[0]
    W = lm_head.weight
    bias = getattr(lm_head, "bias", None)
    total = hidden.new_zeros((), dtype=torch.float32)
    for i in range(0, N, chunk):
        end = min(i + chunk, N)
        logits = F.linear(flat_h[i:end].to(W.dtype), W, bias).float()
        total = total + F.cross_entropy(logits, flat_t[i:end], reduction="sum")
    return total / max(1, N), N
