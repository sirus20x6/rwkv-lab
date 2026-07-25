"""Small, reusable training-loop speedups with deterministic semantics."""
from __future__ import annotations

import copy
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch


@dataclass(frozen=True)
class ContextStage:
    """One context-curriculum stage."""

    start_fraction: float
    seq_len: int


def parse_context_curriculum(spec: str, *, max_seq_len: int) -> tuple[ContextStage, ...]:
    """Parse ``fraction:seq_len`` stages, e.g. ``0:256,0.33:512,0.67:1024``."""
    if not spec:
        return ()
    stages: list[ContextStage] = []
    for item in spec.split(","):
        try:
            fraction_text, length_text = item.strip().split(":", 1)
            stage = ContextStage(float(fraction_text), int(length_text))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "context curriculum must be comma-separated fraction:seq_len stages"
            ) from exc
        if not 0.0 <= stage.start_fraction < 1.0:
            raise ValueError("context-curriculum fractions must be in [0, 1)")
        if not 0 < stage.seq_len <= max_seq_len:
            raise ValueError(
                f"context-curriculum lengths must be in [1, {max_seq_len}]"
            )
        stages.append(stage)
    if not stages or stages[0].start_fraction != 0.0:
        raise ValueError("context curriculum must start at fraction 0")
    if any(a.start_fraction >= b.start_fraction for a, b in zip(stages, stages[1:])):
        raise ValueError("context-curriculum fractions must be strictly increasing")
    if any(a.seq_len > b.seq_len for a, b in zip(stages, stages[1:])):
        raise ValueError("context-curriculum sequence lengths must be non-decreasing")
    return tuple(stages)


def context_batch_for_step(
    stages: tuple[ContextStage, ...],
    *,
    step: int,
    total_steps: int,
    max_seq_len: int,
    base_batch: int,
) -> tuple[int, int]:
    """Return ``(seq_len, batch)`` while holding the base token budget constant."""
    if not stages:
        return max_seq_len, base_batch
    if total_steps <= 0:
        raise ValueError("context curriculum requires a positive step horizon")
    progress = min(max(step / total_steps, 0.0), 1.0)
    active = stages[0]
    for stage in stages[1:]:
        if progress < stage.start_fraction:
            break
        active = stage
    token_budget = max_seq_len * base_batch
    return active.seq_len, max(1, round(token_budget / active.seq_len))


class AsyncCPUBatchPrefetcher:
    """Build and pin one CPU batch ahead without advancing RNG speculatively.

    The producer receives a private NumPy generator cloned from ``rng``. The
    caller's generator commits the resulting state only when ``next()`` consumes
    the batch. A checkpoint taken while a future batch is pending therefore
    resumes exactly as if prefetching were disabled.
    """

    def __init__(
        self,
        producer: Callable[[np.random.Generator], torch.Tensor],
        rng: np.random.Generator,
        *,
        pin_memory: bool,
    ):
        self._producer = producer
        self._rng = rng
        self._pin_memory = bool(pin_memory)
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="rwkv-batch")
        self._closed = False
        self._future: Future[tuple[torch.Tensor, dict]] = self._submit()

    def _submit(self) -> Future[tuple[torch.Tensor, dict]]:
        state = copy.deepcopy(self._rng.bit_generator.state)

        def produce() -> tuple[torch.Tensor, dict]:
            local_rng = np.random.default_rng()
            local_rng.bit_generator.state = state
            batch = self._producer(local_rng)
            if self._pin_memory and not batch.is_pinned():
                batch = batch.pin_memory()
            return batch, copy.deepcopy(local_rng.bit_generator.state)

        return self._pool.submit(produce)

    def next(self) -> torch.Tensor:
        if self._closed:
            raise RuntimeError("batch prefetcher is closed")
        batch, committed_state = self._future.result()
        self._rng.bit_generator.state = committed_state
        self._future = self._submit()
        return batch

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._future.cancel()
        self._pool.shutdown(wait=False, cancel_futures=True)

    def __enter__(self) -> "AsyncCPUBatchPrefetcher":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()


__all__ = [
    "AsyncCPUBatchPrefetcher",
    "ContextStage",
    "context_batch_for_step",
    "parse_context_curriculum",
]
