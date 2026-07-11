"""Dense-to-sparse attention transfer oracles inspired by RTPurbo.

The method calibrates retrieval heads, fits 16-D query/key indexers to dense
attention with KL, selects a query-dependent top-p support, then distills the
sparse student on the teacher's top logits.  See *Full Attention Strikes Back*
(2026), https://arxiv.org/abs/2605.16928 and community lead
https://discord.com/channels/992359628979568762/992362722035507270/1508496234585784390
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def retrieval_head_scores(attention: torch.Tensor, early: slice, late: slice) -> torch.Tensor:
    """Attention mass from a repeated late needle to its early copy; [...,H,Q,K]."""
    if attention.ndim < 4:
        raise ValueError("attention must end in [heads, query, key]")
    mass = attention[..., late, :][..., early].sum(dim=(-1, -2))
    while mass.ndim > 1:
        mass = mass.mean(0)
    return mass


class LowDimAttentionIndexer(nn.Module):
    """Per-head pre-RoPE Q/K projections used only to choose sparse support."""
    def __init__(self, n_heads: int, head_dim: int, index_dim: int = 16):
        super().__init__()
        self.q_proj = nn.Parameter(torch.empty(n_heads, head_dim, index_dim))
        self.k_proj = nn.Parameter(torch.empty(n_heads, head_dim, index_dim))
        nn.init.orthogonal_(self.q_proj.flatten(0, 1))
        nn.init.orthogonal_(self.k_proj.flatten(0, 1))

    def scores(self, q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        # q [...,H,Q,D], k [...,H,K,D] -> [...,H,Q,K]
        qp = torch.einsum("...hqd,hdi->...hqi", q, self.q_proj)
        kp = torch.einsum("...hkd,hdi->...hki", k, self.k_proj)
        return torch.einsum("...hqi,...hki->...hqk", qp, kp) / qp.shape[-1] ** 0.5

    def calibration_loss(self, q: torch.Tensor, k: torch.Tensor,
                         dense_scores: torch.Tensor) -> torch.Tensor:
        approx = self.scores(q, k)
        return F.kl_div(F.log_softmax(approx.float(), -1),
                        F.softmax(dense_scores.float(), -1), reduction="batchmean")


def dynamic_top_p_mask(scores: torch.Tensor, p: float = 0.9, *, causal: bool = False,
                       query_offset: int | None = None) -> torch.Tensor:
    """Small exact oracle retaining the minimum keys whose probability mass reaches p."""
    if not 0 < p <= 1:
        raise ValueError("p must be in (0,1]")
    if causal:
        q, k = scores.shape[-2:]
        # Decode commonly supplies only the last Q queries against K cached keys.
        # In that case their absolute first position is K-Q, not zero.
        offset = k - q if query_offset is None else query_offset
        invalid = torch.arange(k, device=scores.device) > (offset + torch.arange(
            q, device=scores.device))[:, None]
        scores = scores.masked_fill(invalid, -torch.inf)
    probs = F.softmax(scores.float(), dim=-1)
    ordered, indices = probs.sort(dim=-1, descending=True)
    cumulative = ordered.cumsum(-1)
    keep_sorted = (cumulative - ordered) < p
    keep = torch.zeros_like(keep_sorted).scatter(-1, indices, keep_sorted)
    return keep


def sparse_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                     indexer: LowDimAttentionIndexer, *, p: float = 0.9,
                     causal: bool = False,
                     query_offset: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    support = dynamic_top_p_mask(indexer.scores(q, k), p, causal=causal,
                                 query_offset=query_offset)
    exact = torch.einsum("...hqd,...hkd->...hqk", q, k) / q.shape[-1] ** 0.5
    weights = F.softmax(exact.masked_fill(~support, -torch.inf).float(), -1).to(v.dtype)
    return torch.einsum("...hqk,...hkd->...hqd", weights, v), support


def top_logit_distillation(student: torch.Tensor, teacher: torch.Tensor,
                           top_n: int = 10, temperature: float = 1.0) -> torch.Tensor:
    """RTPurbo stage-two KL restricted to the dense teacher's top logits."""
    n = min(top_n, teacher.shape[-1])
    idx = teacher.topk(n, dim=-1).indices
    t = teacher.gather(-1, idx).float() / temperature
    s = student.gather(-1, idx).float() / temperature
    return F.kl_div(F.log_softmax(s, -1), F.softmax(t, -1), reduction="batchmean") * temperature**2
