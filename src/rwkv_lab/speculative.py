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
from dataclasses import asdict, dataclass
import time
import torch
import torch.nn as nn


class EAGLE3DraftHead(nn.Module):
    """Fuse three target-layer features and emit several direct token drafts."""

    def __init__(self, d_model: int, vocab_size: int, *, draft_steps: int = 4):
        super().__init__()
        self.draft_steps = int(draft_steps)
        self.vocab_size = int(vocab_size)
        self.fuse = nn.Linear(3 * d_model, d_model, bias=False)
        self.norm = nn.RMSNorm(d_model)
        # One packed vocabulary GEMM avoids one kernel launch per draft position.
        self.token_head = nn.Linear(d_model, self.draft_steps * self.vocab_size, bias=False)

    def forward(self, low: torch.Tensor, middle: torch.Tensor,
                high: torch.Tensor) -> torch.Tensor:
        if low.shape != middle.shape or low.shape != high.shape:
            raise ValueError("low/middle/high features must have identical shapes")
        fused = self.norm(self.fuse(torch.cat((low, middle, high), dim=-1)))
        return self.token_head(fused).unflatten(-1, (self.draft_steps, self.vocab_size))

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        # Migrate pre-fusion checkpoints containing token_heads.{i}.weight.
        packed_key = prefix + "token_head.weight"
        legacy = [state_dict.get(prefix + f"token_heads.{i}.weight")
                  for i in range(self.draft_steps)]
        if packed_key not in state_dict and all(value is not None for value in legacy):
            state_dict[packed_key] = torch.cat(legacy, dim=0)
            for i in range(self.draft_steps):
                state_dict.pop(prefix + f"token_heads.{i}.weight", None)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)


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


def verify_greedy_draft_batched(
    prefix: Sequence[int],
    draft: Sequence[int],
    *,
    target_verify: Callable[[list[int], list[int]], Sequence[int]],
) -> tuple[list[int], int]:
    """Verify a whole draft with one target-model call.

    ``target_verify(prefix, draft)`` must return ``len(draft) + 1`` greedy target
    predictions: the token after ``prefix``, then the predictions after each
    successively appended draft token.  This is the serving primitive used by
    EAGLE-3's tree verifier (Li et al., 2025, arXiv:2503.01840): one target
    forward validates several proposed positions.  The conservative accept rule
    makes the returned stream exactly target-greedy.
    """

    proposed = [int(token) for token in draft]
    if not proposed:
        raise ValueError("batched verification needs at least one draft token")
    target = [int(token) for token in target_verify(list(prefix), proposed)]
    if len(target) != len(proposed) + 1:
        raise ValueError("target_verify must return len(draft) + 1 predictions")
    accepted = 0
    for candidate, expected in zip(proposed, target):
        if candidate != expected:
            return proposed[:accepted] + [expected], accepted
        accepted += 1
    return proposed + [target[-1]], accepted


@dataclass(frozen=True)
class SpeculativeDecodeStats:
    generated_tokens: int
    accepted_tokens: int
    drafted_tokens: int
    target_calls: int
    draft_calls: int
    seconds: float
    tokens_per_second: float
    acceptance_rate: float


def speculative_greedy_decode(
    prefix: Sequence[int],
    *,
    max_new: int,
    draft_steps: int,
    draft_propose: Callable[[list[int], int], Sequence[int]],
    target_verify: Callable[[list[int], list[int]], Sequence[int]],
    stop_token: int | None = None,
) -> tuple[list[int], SpeculativeDecodeStats]:
    """Exact greedy speculative decoding with explicit serving counters."""

    if not prefix or max_new < 1 or draft_steps < 1:
        raise ValueError("speculative decode needs a prefix and positive token budgets")
    context, output = [int(token) for token in prefix], []
    accepted_total = drafted_total = target_calls = draft_calls = 0
    started = time.perf_counter()
    while len(output) < max_new:
        width = min(int(draft_steps), max_new - len(output))
        draft = [int(token) for token in draft_propose(context, width)][:width]
        draft_calls += 1
        if not draft:
            raise ValueError("draft_propose returned no tokens")
        drafted_total += len(draft)
        verified, accepted = verify_greedy_draft_batched(
            context, draft, target_verify=target_verify)
        target_calls += 1
        accepted_total += accepted
        for token in verified:
            if len(output) >= max_new:
                break
            output.append(token)
            context.append(token)
            if stop_token is not None and token == stop_token:
                elapsed = time.perf_counter() - started
                stats = SpeculativeDecodeStats(
                    len(output), accepted_total, drafted_total, target_calls, draft_calls,
                    elapsed, len(output) / max(elapsed, 1e-12),
                    accepted_total / max(drafted_total, 1),
                )
                return output, stats
    elapsed = time.perf_counter() - started
    return output, SpeculativeDecodeStats(
        len(output), accepted_total, drafted_total, target_calls, draft_calls,
        elapsed, len(output) / max(elapsed, 1e-12),
        accepted_total / max(drafted_total, 1),
    )


def qualify_speculative_greedy(
    prefix: Sequence[int],
    *,
    max_new: int,
    draft_steps: int,
    target_next: Callable[[list[int]], int],
    draft_propose: Callable[[list[int], int], Sequence[int]],
    target_verify: Callable[[list[int], list[int]], Sequence[int]],
    stop_token: int | None = None,
    minimum_speedup: float = 1.02,
) -> dict:
    """Require exact tokens before reporting an EAGLE serving path as adopted."""

    context, baseline = [int(token) for token in prefix], []
    started = time.perf_counter()
    for _ in range(max_new):
        token = int(target_next(context))
        baseline.append(token)
        context.append(token)
        if stop_token is not None and token == stop_token:
            break
    baseline_seconds = time.perf_counter() - started
    candidate, stats = speculative_greedy_decode(
        prefix, max_new=max_new, draft_steps=draft_steps,
        draft_propose=draft_propose, target_verify=target_verify, stop_token=stop_token)
    exact = candidate == baseline
    speedup = baseline_seconds / max(stats.seconds, 1e-12)
    return {
        "schema": "rwkv-lab.speculative-serving-qualification.v1",
        "exact_tokens": exact,
        "baseline_seconds": baseline_seconds,
        "candidate_seconds": stats.seconds,
        "speedup": speedup,
        "minimum_speedup": minimum_speedup,
        "stats": asdict(stats),
        "adopted": bool(exact and speedup >= minimum_speedup),
    }


def topk_tree(logits: torch.Tensor, *, branches: int = 2) -> tuple[torch.Tensor, torch.Tensor]:
    """Return top-k token/value branches for each direct draft position."""

    if logits.ndim < 2:
        raise ValueError("logits must end in [draft_steps,vocab]")
    values, tokens = logits.topk(min(int(branches), logits.shape[-1]), dim=-1)
    return tokens, values
