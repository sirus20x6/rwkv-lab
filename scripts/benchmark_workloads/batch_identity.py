#!/usr/bin/env python3
"""Identity of the batches a benchmark run actually trained on.

Ordering and content drift are what a prefetching or worker-parallel input
pipeline introduces, and neither is visible in a throughput number. A loader
that returns the right batches in the wrong order, or resumes on the wrong
cursor, still reports a perfectly good steps-per-second.

These digests exist so `ordering_parity` and `content_parity` in the
qualification evidence can be MEASURED by comparing two arms, rather than
asserted as a constant true beside a speed win.

Every workload emits them, including the synthetic ones. A fixture that
compares a run against itself then reports parity it genuinely observed.
"""

from __future__ import annotations

import hashlib
from typing import Any, Iterable

EMPTY_CHAIN = "sha256:" + hashlib.sha256(b"").hexdigest()


def batch_digest(step_index: int, *tensors: Any) -> str:
    """Digest exactly what one step trains on, bound to its position.

    The step index is bound in deliberately. A fixture whose batch content
    repeats with a short period — small document sets do — would otherwise
    produce a chain that cannot tell step 0 from step 2, and a reordering
    across that period would pass unnoticed.

    Shapes are included so a padding or mask change cannot preserve the digest
    by coincidence of having the same underlying bytes.
    """
    accumulator = hashlib.sha256()
    accumulator.update(int(step_index).to_bytes(8, "big", signed=True))
    accumulator.update(b"\0")
    for tensor in tensors:
        materialized = tensor.detach().to("cpu").contiguous()
        accumulator.update(str(tuple(materialized.shape)).encode())
        accumulator.update(b"\0")
        accumulator.update(str(materialized.dtype).encode())
        accumulator.update(b"\0")
        accumulator.update(materialized.numpy().tobytes())
        accumulator.update(b"\0")
    return "sha256:" + accumulator.hexdigest()


def chain_digests(step_digests: Iterable[str]) -> str:
    """Fold per-step digests into one value covering the whole sequence."""
    chain = EMPTY_CHAIN
    for step_digest in step_digests:
        chain = "sha256:" + hashlib.sha256(
            (chain + "\0" + step_digest).encode()).hexdigest()
    return chain
