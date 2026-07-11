"""Bounded summary-conditioned iterative reasoning.

Reference: Wu et al., "Reasoning Cache: Continual Improvement Over Long
Horizons via Short-Horizon RL", arXiv:2602.03773,
https://arxiv.org/abs/2602.03773. The paper replaces ordinary decoding with
alternating response generation and summarization during training and inference.

This module supplies the budget/accounting contract. It never executes generated
code and never promotes a checkpoint; verifiers remain in ``rlvr_train`` and
Adamaton owns isolated generated-code verification.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import time
from typing import Callable, Any


@dataclass(frozen=True)
class ReasoningCacheStep:
    iteration: int
    response: str
    summary: str
    reward: float | None
    response_tokens: int
    summary_tokens: int


@dataclass
class ReasoningCacheResult:
    schema: str
    steps: list[ReasoningCacheStep]
    total_tokens: int
    elapsed_seconds: float
    stop_reason: str

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "steps": [asdict(step) for step in self.steps]}


def run_reasoning_cache(prompt: str, *,
                        generate: Callable[[str, str, int], tuple[str, int]],
                        summarize: Callable[[str, str, int], tuple[str, int]],
                        verify: Callable[[str], float] | None = None,
                        max_iterations: int = 4, max_total_tokens: int = 16_384,
                        max_seconds: float = 0.0, target_reward: float | None = None) -> ReasoningCacheResult:
    if max_iterations < 1 or max_total_tokens < 1 or max_seconds < 0:
        raise ValueError("reasoning-cache budgets are invalid")
    started = time.perf_counter(); summary = ""; steps = []; used = 0
    stop = "iteration_budget"
    for iteration in range(max_iterations):
        if max_seconds and time.perf_counter() - started >= max_seconds:
            stop = "time_budget"; break
        remaining = max_total_tokens - used
        if remaining <= 0:
            stop = "token_budget"; break
        response, response_tokens = generate(prompt, summary, remaining)
        response_tokens = int(response_tokens)
        if response_tokens < 0 or response_tokens > remaining:
            raise ValueError("generator reported invalid token usage")
        used += response_tokens
        reward = float(verify(response)) if verify is not None else None
        if target_reward is not None and reward is not None and reward >= target_reward:
            steps.append(ReasoningCacheStep(iteration, response, summary, reward,
                                            response_tokens, 0))
            stop = "target_reward"; break
        remaining = max_total_tokens - used
        if remaining <= 0:
            steps.append(ReasoningCacheStep(iteration, response, summary, reward,
                                            response_tokens, 0))
            stop = "token_budget"; break
        next_summary, summary_tokens = summarize(response, summary, remaining)
        summary_tokens = int(summary_tokens)
        if summary_tokens < 0 or summary_tokens > remaining:
            raise ValueError("summarizer reported invalid token usage")
        used += summary_tokens
        summary = next_summary
        steps.append(ReasoningCacheStep(iteration, response, summary, reward,
                                        response_tokens, summary_tokens))
    return ReasoningCacheResult("rwkv-lab.reasoning-cache.v1", steps, used,
                                time.perf_counter() - started, stop)


def reasoning_cache_training_pairs(prompt: str, result: ReasoningCacheResult) -> list[dict[str, str]]:
    """Materialize auditable short-horizon response/summary supervision pairs."""
    pairs, previous = [], ""
    for step in result.steps:
        pairs.append({"kind": "response", "prompt": prompt, "cache": previous,
                      "target": step.response})
        if step.summary_tokens:
            pairs.append({"kind": "summary", "prompt": prompt, "cache": previous,
                          "response": step.response, "target": step.summary})
            previous = step.summary
    return pairs
