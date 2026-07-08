"""FSP-BCE: handcrafted future-summary auxiliary loss.

Reference: "Beyond Multi-Token Prediction: Pretraining LLMs with Future
Summaries" (Mahajan et al. 2025, arXiv:2510.14751 — Beyond Multi-Token Prediction: Pretraining LLMs with…). The handcrafted variant
(FSP-BCE) predicts, at each position t, a multi-hot "bag of tokens appearing
in the next tau positions" as a distributional future summary, via weighted
binary cross-entropy.

Loss: L_FSP = sum_p w_i · BCE( sigmoid(lm_head(hidden[p] + fsp_bias))_i,
                                1{token i in y[p+1 : p+tau]} )

where `fsp_bias` is a small learnable per-dim bias that tells the shared
lm_head "you are in FSP mode, not NTP mode" and keeps the two losses from
pulling the head in exactly the same direction.

Memory discipline: full per-position [B, T, V] target is V=151K × B × T →
huge. We sample K positions per sequence and build a [B, K, V] sparse multi-hot
target on-the-fly. At K=64, B=1, V=151K this is ~10 MB in bf16 — fine.

Optional tf-idf weighting over vocab can be loaded from a precomputed file
(idf_weights.pt). Default is uniform weighting (no file needed).
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .safe_torch import safe_torch_load


class FSPHead(nn.Module):
    """Per-dim bias that re-uses the shared lm_head for FSP-BCE.

    Adds `hidden_size` trainable scalars — a tiny 2k-param tensor for a 2048-H
    model. Enough to let the model distinguish "predict next token" from
    "predict bag of future tokens" without a separate V-way projection.
    """
    def __init__(self, hidden_size: int, idf_path: Optional[str] = None,
                 vocab_size: Optional[int] = None) -> None:
        super().__init__()
        self.fsp_bias = nn.Parameter(torch.zeros(hidden_size))
        # Optional vocab-level reweighting; registered as a buffer so it moves
        # with the module but is not a trainable parameter.
        if idf_path and Path(idf_path).is_file():
            w = safe_torch_load(idf_path, map_location="cpu")
            if not isinstance(w, torch.Tensor):
                raise ValueError(f"{idf_path}: expected a 1D tensor of vocab weights")
            if vocab_size is not None and w.numel() != vocab_size:
                raise ValueError(
                    f"{idf_path}: weight size {w.numel()} != vocab_size {vocab_size}")
            self.register_buffer("vocab_w", w.to(torch.float32))
        else:
            self.register_buffer("vocab_w", torch.tensor([]), persistent=False)

    def has_vocab_weights(self) -> bool:
        return self.vocab_w.numel() > 0


def build_future_multihot(
    y: torch.Tensor,            # [B, T]
    positions: torch.Tensor,    # [B, K] — sampled positions p
    tau: int,
    vocab_size: int,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Build a [B, K, V] multi-hot target: entry (b,k,v) = 1 iff token v
    appears in y[b, positions[b,k] : positions[b,k] + tau - 1].

    Note: y[p] corresponds to x_full[p+1], so y[p : p + tau - 1] is the range
    of tokens at absolute positions [p+1, p+tau-1]. We use this as the
    "future summary" window. (The paper uses [p+2, p+tau]; close enough for
    our purposes — one position offset.)
    """
    B, T = y.shape
    K = positions.shape[1]
    device = y.device
    # Slice into y at each sampled position; slices of length tau-1.
    # We clamp the end so near-tail positions don't over-read.
    target = torch.zeros((B, K, vocab_size), dtype=dtype, device=device)
    w = min(tau - 1, T)
    # Build index tensor of [B, K, w] token ids from y
    # Position + offset table: [K, w]
    off = torch.arange(1, w + 1, device=device).unsqueeze(0).expand(K, -1)  # [K, w]
    # [B, K, w] indices into y's T dimension, clamped to T-1
    idx = (positions.unsqueeze(-1) + off).clamp(max=T - 1)   # [B, K, w]
    tok = torch.gather(y.unsqueeze(1).expand(-1, K, -1), 2, idx)  # [B, K, w]
    # Scatter 1s into target at (batch, register, token_id)
    target.scatter_(2, tok, 1.0)
    return target


def fsp_loss(
    hidden: torch.Tensor,           # [B, T, H]
    y: torch.Tensor,                # [B, T]
    lm_head: nn.Linear,
    head: FSPHead,
    num_positions: int = 64,
    tau: int = 12,
    chunk: int = 128,
    rng: Optional[torch.Generator] = None,
) -> tuple[torch.Tensor, int]:
    """Sample K positions per batch, build the future-multihot target, and
    compute mean BCE-with-logits (optionally reweighted by vocab_w). Returns
    (mean_loss, n_valid).
    """
    B, T, H = hidden.shape
    V = lm_head.weight.shape[0]
    device = hidden.device

    max_valid_p = T - tau
    if max_valid_p < 0 or num_positions <= 0:
        return hidden.new_zeros((), dtype=torch.float32), 0
    K = int(min(num_positions, max_valid_p + 1))

    if rng is None:
        pos_bt = torch.randint(0, max_valid_p + 1, (B, K), device=device)
    else:
        pos_bt = torch.randint(0, max_valid_p + 1, (B, K), device=device, generator=rng)

    target = build_future_multihot(y, pos_bt, tau, V, dtype=torch.float32)  # [B, K, V]

    # Gather hidden[p] and add the learnable FSP bias
    idx_bt = pos_bt.unsqueeze(-1).expand(-1, -1, H)
    h_sel = torch.gather(hidden, 1, idx_bt)        # [B, K, H]
    h_sel = h_sel + head.fsp_bias                  # broadcast over [B, K]

    # Chunked matmul through lm_head to avoid a full [B, K, V] bf16 tensor
    flat_h = h_sel.reshape(-1, H)
    flat_t = target.reshape(-1, V)
    N = flat_h.shape[0]
    total = hidden.new_zeros((), dtype=torch.float32)
    W = lm_head.weight
    bias = getattr(lm_head, "bias", None)
    has_w = head.has_vocab_weights()
    vw = head.vocab_w if has_w else None
    for i in range(0, N, chunk):
        end = min(i + chunk, N)
        logits = F.linear(flat_h[i:end].to(W.dtype), W, bias).float()   # [chunk, V]
        tc = flat_t[i:end]
        # BCE with logits, sum over vocab then mean over positions
        if has_w:
            loss_chunk = F.binary_cross_entropy_with_logits(
                logits, tc, weight=vw, reduction="sum")
        else:
            loss_chunk = F.binary_cross_entropy_with_logits(
                logits, tc, reduction="sum")
        # Normalize by vocab size so per-position loss is not V-scaled
        total = total + loss_chunk / V
    return total / max(1, N), N
