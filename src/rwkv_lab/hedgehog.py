"""Portable pieces of "Attention to Mamba" (arXiv:2604.14191, a DRAFT) as standalone,
reusable components — NOT grafted into the RWKV-7 gated-delta kernel.

Two ideas from that recipe are architecture-agnostic and worth having on the shelf:

1. HedgehogFeatureMap — the learned softmax feature map φ(x) = softmax([xW; −xW]) that turns
   a query/key into a positive, sum-to-1 feature so that φ(q)·φ(k)ᵀ approximates a softmax
   attention row. This is the Hedgehog kernel (Zhang et al.) used as the paper's Stage-1
   learnable kernel on Q,K.

2. attn_map_ce_loss — the paper's Stage-1 objective: cross-entropy between the teacher's
   softmax-attention weight distribution and the student's (linear-attention) weight
   distribution, computed per row with the backbone frozen. A cheap attention-MAP match
   (vs RADLADS' hidden-state alignment).

WHY STANDALONE (honest scope): the paper targets a Mamba/HedgeMamba hybrid, and its φ-on-Q/K
does not compose cleanly with our RWKV-7 gated-delta *recurrence* — the delta-rule removal key
must share a basis with the write key (same issue that made RoPE placement subtle), and a
softmax feature map on an L2-normalised delta key is semantically muddy. Grafting it into the
careful-zone kernel for a draft paper is low-ROI and risky, so these live here as tested
utilities for a future linear-attention student that exposes an explicit φ(q)φ(k)ᵀ matrix.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class HedgehogFeatureMap(nn.Module):
    """φ(x) = softmax([xW; −xW], dim=-1). x [..., d] → [..., d] (two halves of d/2, softmaxed
    jointly). Output is non-negative and sums to 1 along the feature dim, so φ(q)·φ(k)ᵀ ≥ 0 and
    behaves like an (un-normalised) attention weight. The two halves are initialised identically
    (paper's `[check]`), i.e. W is shared and the sign is the only difference."""

    def __init__(self, d: int) -> None:
        super().__init__()
        if d % 2:
            raise ValueError("HedgehogFeatureMap needs even d")
        self.w = nn.Linear(d, d // 2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.w(x)
        return F.softmax(torch.cat([z, -z], dim=-1), dim=-1)


def linear_attn_map(phi_q: torch.Tensor, phi_k: torch.Tensor) -> torch.Tensor:
    """Row-normalised causal linear-attention weight matrix A[t,s] ∝ φ(q_t)·φ(k_s), s≤t.
    phi_q, phi_k: [B,H,T,f]. Returns [B,H,T,T] (rows sum to 1 over s≤t)."""
    scores = torch.einsum("bhtf,bhsf->bhts", phi_q, phi_k)     # [B,H,T,T]
    causal = torch.tril(torch.ones(scores.shape[-2:], device=scores.device, dtype=torch.bool))
    scores = scores.masked_fill(~causal, 0.0)
    return scores / (scores.sum(-1, keepdim=True) + 1e-9)


def attn_map_ce_loss(teacher_attn: torch.Tensor, phi_q: torch.Tensor, phi_k: torch.Tensor) -> torch.Tensor:
    """Attention-map CE (paper Eq.5): −Σ_s A_teacher[t,s]·log A_student[t,s], mean over (b,h,t).
    teacher_attn [B,H,T,T] causal softmax rows; phi_q/phi_k [B,H,T,f] student features."""
    a_s = linear_attn_map(phi_q, phi_k).clamp_min(1e-9)
    return -(teacher_attn * a_s.log()).sum(-1).mean()
