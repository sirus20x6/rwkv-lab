"""EAGLE-3-style draft heads and lossless speculative verification.

Reference: Li et al., "EAGLE-3: Scaling up Inference Acceleration of Large
Language Models via Training-Time Test", arXiv:2503.01840,
https://arxiv.org/abs/2503.01840.

EAGLE-3 fuses low/middle/high target features and predicts tokens directly,
then verifies a tree of candidates with the target model. This module adopts
the feature-fusion draft architecture and exposes conservative greedy
verification. Rejection always falls back to the target token, so output is
exactly target-greedy even when the draft is poor.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
import torch
import torch.nn as nn


class EAGLE3DraftHead(nn.Module):
    """Fuse three target-layer features and emit several direct token drafts."""

    def __init__(self, d_model: int, vocab_size: int, *, draft_steps: int = 4):
        super().__init__()
        self.draft_steps = int(draft_steps)
        self.fuse = nn.Linear(3 * d_model, d_model, bias=False)
        self.norm = nn.RMSNorm(d_model)
        self.token_heads = nn.ModuleList(nn.Linear(d_model, vocab_size, bias=False)
                                         for _ in range(self.draft_steps))

    def forward(self, low: torch.Tensor, middle: torch.Tensor,
                high: torch.Tensor) -> torch.Tensor:
        if low.shape != middle.shape or low.shape != high.shape:
            raise ValueError("low/middle/high features must have identical shapes")
        fused = self.norm(self.fuse(torch.cat((low, middle, high), dim=-1)))
        return torch.stack([head(fused) for head in self.token_heads], dim=-2)


def verify_greedy_draft(prefix: Sequence[int], draft: Sequence[int], *,
                        target_next: Callable[[list[int]], int]) -> tuple[list[int], int]:
    """Accept matching draft tokens and append the target token on first rejection.

    Returns ``(new_tokens, accepted_count)``. Calling repeatedly produces the
    exact same stream as ordinary greedy target decoding.
    """

    context, output, accepted = list(prefix), [], 0
    for candidate in draft:
        target = int(target_next(context))
        if int(candidate) != target:
            output.append(target)
            break
        output.append(target); context.append(target); accepted += 1
    else:
        # A fully accepted draft needs one target call to keep decoding progress measurable.
        target = int(target_next(context)); output.append(target)
    return output, accepted


def topk_tree(logits: torch.Tensor, *, branches: int = 2) -> tuple[torch.Tensor, torch.Tensor]:
    """Return top-k token/value branches for each direct draft position."""

    if logits.ndim < 2:
        raise ValueError("logits must end in [draft_steps,vocab]")
    values, tokens = logits.topk(min(int(branches), logits.shape[-1]), dim=-1)
    return tokens, values
