"""Parallel-heads MTP auxiliary loss (Gloeckle et al. 2024 style).

Reference: "Better & Faster Large Language Models via Multi-token Prediction"
(arXiv:2404.19737 — Better and Faster LLMs via Multi-token Prediction). The paper adds K independent transformer heads that each
read the backbone's final hidden state h_t and predict token x_{t+k} through
a shared unembedding matrix; all losses are summed.

Rationale for keeping this around even when not actively used:
    - Gradient-decoupling mechanism (mtp-planning 2026) is specific to
      parallel heads; may help Phase 2 backbone reasoning more than chained.
    - Tree-verification speculative decoding (Medusa/EAGLE family) is natural
      with parallel candidates; chained produces linear candidates only.
    - Complements chained MTP (can stack). Chained gives richer per-horizon
      signal; parallel gives cleaner per-horizon gradient.

What this module is NOT:
    - A replacement for the chained MTP block. Both can coexist — enable each
      independently with separate weights.
    - A full Qwen3_5MoeDecoderLayer per head. That would roughly quadruple
      MTP memory (K heads × one full MoE layer each). For our memory budget
      we default to a residual SwiGLU MLP per head. Upgrade path to a full
      layer is clean if we later decide the MLP variant is insufficient.

Architecture (per head, --parallel-head-type mlp):
    h' = h + SiLU(W_gate·RMSNorm(h)) ⊙ W_up·RMSNorm(h) @ W_down.T
    logits_k = lm_head(h')
    loss_k   = CE(logits_k[:, :T-h_k+1], y[:, h_k-1 : T])

Param cost for H=2048 expansion=4 per head: ~50M params. K=3 heads = 150M.
Bias-free, matches Qwen3's FFN style.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class _RMSNorm(nn.Module):
    """Local RMSNorm so we don't pull a HF dependency in. Identical numerics
    to the standard LLaMA/Qwen RMSNorm (no bias, no centering)."""
    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # fp32 reduction for numerical stability
        orig_dtype = x.dtype
        x32 = x.float()
        var = x32.pow(2).mean(dim=-1, keepdim=True)
        x32 = x32 * torch.rsqrt(var + self.eps)
        return (x32 * self.weight.float()).to(orig_dtype)


class ParallelMTPHead(nn.Module):
    """Single residual SwiGLU head for one prediction horizon."""
    def __init__(self, hidden_size: int, expansion: int = 4,
                 eps: float = 1e-5) -> None:
        super().__init__()
        inter = hidden_size * expansion
        self.norm = _RMSNorm(hidden_size, eps=eps)
        self.gate_proj = nn.Linear(hidden_size, inter, bias=False)
        self.up_proj = nn.Linear(hidden_size, inter, bias=False)
        self.down_proj = nn.Linear(inter, hidden_size, bias=False)
        # Small-init the down_proj so residual dominates at step 0 — the head
        # effectively starts as identity, which is the NTP baseline behavior.
        nn.init.normal_(self.down_proj.weight, std=0.02 / (2 * hidden_size) ** 0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        return x + self.down_proj(F.silu(self.gate_proj(h)) * self.up_proj(h))


class ParallelHeads(nn.Module):
    """Collection of K independent MTP heads, one per horizon."""
    def __init__(self, hidden_size: int, horizons: list[int],
                 expansion: int = 4, eps: float = 1e-5) -> None:
        super().__init__()
        if any(h < 2 for h in horizons):
            raise ValueError(f"parallel horizons must be >= 2 (h=1 is NTP); got {horizons}")
        self.horizons = list(horizons)
        self.heads = nn.ModuleList([
            ParallelMTPHead(hidden_size, expansion=expansion, eps=eps)
            for _ in horizons
        ])

    def forward_one(self, idx: int, hidden: torch.Tensor) -> torch.Tensor:
        return self.heads[idx](hidden)


def parallel_heads_loss(
    hidden: torch.Tensor,           # [B, T, H] — backbone final hidden
    y: torch.Tensor,                # [B, T] — shifted labels, y[p] = x_full[p+1]
    lm_head: nn.Linear,
    heads: ParallelHeads,
    weights: list[float],
    chunk: int = 2048,
) -> tuple[torch.Tensor, int]:
    """Compute sum_k w_k * CE(lm_head(head_k(hidden[:, :T-h_k+1])),
                               y[:, h_k-1 : T])

    Each head k targets offset h_k (a horizon ≥ 2). Position-p's prediction of
    that head is of token at abs-position p+h_k = y[p + h_k - 1].

    Returns (summed_loss, total_tokens_over_heads).
    """
    B, T, H = hidden.shape
    if len(weights) != len(heads.horizons):
        raise ValueError(
            f"parallel weights ({len(weights)}) != horizons ({len(heads.horizons)})")
    total = hidden.new_zeros((), dtype=torch.float32)
    total_n = 0
    W = lm_head.weight
    bias = getattr(lm_head, "bias", None)

    for k, (h_k, w) in enumerate(zip(heads.horizons, weights)):
        seg = T - h_k + 1
        if seg <= 0:
            continue
        out_k = heads.forward_one(k, hidden[:, :seg])     # [B, seg, H]
        labels_k = y[:, h_k - 1 : h_k - 1 + seg]          # [B, seg]

        flat_h = out_k.reshape(-1, H)
        flat_t = labels_k.reshape(-1)
        N = flat_h.shape[0]
        head_sum = hidden.new_zeros((), dtype=torch.float32)
        for i in range(0, N, chunk):
            end = min(i + chunk, N)
            logits = F.linear(flat_h[i:end].to(W.dtype), W, bias).float()
            head_sum = head_sum + F.cross_entropy(logits, flat_t[i:end], reduction="sum")
        total = total + w * (head_sum / max(1, N))
        total_n += N
    return total, total_n
