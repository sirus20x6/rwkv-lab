"""Small, reusable training-loop speedups with deterministic semantics."""
from __future__ import annotations

import copy
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Self

import numpy as np
import torch

from rwkv_lab.training_runtime.curricula import (
    ContextStage,
    context_batch_for_stages,
    parse_context_stages,
)


def parse_context_curriculum(spec: str, *, max_seq_len: int) -> tuple[ContextStage, ...]:
    """Compatibility facade for the typed context-curriculum parser."""
    return parse_context_stages(spec, maximum_sequence_length=max_seq_len)


def context_batch_for_step(
    stages: tuple[ContextStage, ...],
    *,
    step: int,
    total_steps: int,
    max_seq_len: int,
    base_batch: int,
) -> tuple[int, int]:
    """Compatibility facade for the typed curriculum state machine."""
    return context_batch_for_stages(
        stages,
        step=step,
        total_steps=total_steps,
        maximum_sequence_length=max_seq_len,
        base_batch_size=base_batch,
    )


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

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_exc) -> None:
        self.close()


__all__ = [
    "AsyncCPUBatchPrefetcher",
    "ContextStage",
    "context_batch_for_step",
    "parse_context_curriculum",
]
