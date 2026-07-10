"""Exact contribution tracing for linear recurrent state writes.

Reference: Anthropic, "Circuit Tracing: Revealing Computational Graphs in
Language Models" (Attribution Graphs methods, 2025),
https://transformer-circuits.pub/2025/attribution-graphs/methods.html.

The cited work constructs sparse attribution graphs for Transformer features.
RWKV's linear recurrent state admits a stronger local primitive: each write's
contribution to a later read can be propagated exactly through the decay
product. These edges form the recurrent backbone onto which learned feature
transcoders can later attach; the implementation below makes no causal claim
beyond this algebraically exact decomposition.
"""
from __future__ import annotations

from dataclasses import dataclass
import torch


@dataclass(frozen=True)
class RecurrentEdge:
    write_position: int
    read_position: int
    contribution: torch.Tensor


def trace_linear_recurrence(reads: torch.Tensor, keys: torch.Tensor,
                            values: torch.Tensor, decays: torch.Tensor) -> list[RecurrentEdge]:
    """Trace ``S_t = decay_t*S_(t-1) + key_t outer value_t; y_t=read_t@S_t``.

    Inputs are ``[time,key_dim]``, ``[time,key_dim]``, ``[time,value_dim]`` and
    either ``[time]`` or ``[time,key_dim]``. The sum of edges ending at each
    read position exactly reconstructs its recurrent output (up to fp error).
    """

    if reads.shape != keys.shape or reads.ndim != 2 or values.shape[0] != reads.shape[0]:
        raise ValueError("incompatible recurrent trace geometry")
    T, K = reads.shape
    if decays.shape not in ((T,), (T, K)):
        raise ValueError("decays must be [time] or [time,key_dim]")
    d = decays[:, None] if decays.ndim == 1 else decays
    edges: list[RecurrentEdge] = []
    for t in range(T):
        survival = torch.ones(K, device=reads.device, dtype=reads.dtype)
        for i in range(t, -1, -1):
            if i < t:
                survival = survival * d[i + 1].to(reads.dtype)
            coefficient = (reads[t] * keys[i] * survival).sum()
            edges.append(RecurrentEdge(i, t, coefficient * values[i]))
    return edges


def aggregate_edges(edges: list[RecurrentEdge], time: int, value_dim: int, *,
                    device=None, dtype=None) -> torch.Tensor:
    out = torch.zeros(time, value_dim, device=device, dtype=dtype)
    for edge in edges:
        out[edge.read_position] += edge.contribution.to(device=out.device, dtype=out.dtype)
    return out
