#!/usr/bin/env python3
"""One portable language-model training step, measured.

This is the workload behind the portable fixtures. It is deliberately small
and CPU-only, but it is not synthetic in the way a bare GEMM is: it drives the
repository's real registered training components -- the fp32-master AdamW, the
linear-warmup-cosine schedule, and the linear-head cross-entropy objective --
so a portable baseline moves when those actually change.

Run as a subprocess by scripts/run_benchmark_fixture.py, one process per
phase, and emits a single JSON report on stdout. Nothing here decides whether
a result qualifies.
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import time


def parse_bucket(bucket: str) -> tuple[int, int]:
    """'seq512xbatch8' -> (512, 8). Shapes come from the fixture, not here."""
    try:
        sequence, batch = bucket.split("x")
        return int(sequence.removeprefix("seq")), int(batch.removeprefix("batch"))
    except ValueError as error:
        raise SystemExit(f"unsupported shape bucket {bucket!r}") from error


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True,
                        choices=["cold", "warmup", "timed"])
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=8)
    arguments = parser.parse_args()

    # Import inside main so the cold phase pays the real import cost, which is
    # a genuine part of a fresh worker's startup.
    import torch
    from torch import nn

    import rwkv_lab.training_components as components

    sequence_length, batch_size = parse_bucket(arguments.bucket)
    torch.manual_seed(arguments.seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    vocabulary, width = 512, 128

    model = nn.Sequential(
        nn.Embedding(vocabulary, width),
        nn.LayerNorm(width),
        nn.Linear(width, width),
        nn.GELU(),
        nn.LayerNorm(width),
    )
    head = nn.Linear(width, vocabulary, bias=False)
    parameters = list(model.parameters()) + list(head.parameters())

    optimizer = components.build_registered_optimizer(
        components.OptimizerImplementation.FP32_MASTER_ADAMW_V1,
        parameters,
        components.AdamWConfiguration(learning_rate=1e-3),
    )
    schedule = components.build_registered_schedule(
        components.ScheduleImplementation.LINEAR_WARMUP_COSINE_V1,
        optimizer,
        components.LinearWarmupCosineConfiguration(
            warmup_steps=2, max_steps=max(4, arguments.steps)),
    )
    objective = components.build_registered_objective(
        components.ObjectiveImplementation.LINEAR_HEAD_CROSS_ENTROPY_V1,
        components.LinearHeadCrossEntropyConfiguration(),
    )

    step_seconds: list[float] = []
    input_wait = 0.0
    loss_value = float("nan")
    generator = torch.Generator().manual_seed(arguments.seed)

    for _ in range(arguments.steps):
        # Input wait is measured separately so a throughput win that merely
        # moved cost into the loader is visible rather than hidden.
        input_started = time.perf_counter()
        tokens = torch.randint(0, vocabulary, (batch_size, sequence_length),
                               generator=generator)
        targets = torch.randint(0, vocabulary, (batch_size, sequence_length),
                                generator=generator)
        input_wait += time.perf_counter() - input_started

        step_started = time.perf_counter()
        hidden = model(tokens)
        loss = objective(hidden, head, targets)
        loss.backward()
        optimizer.step()
        schedule.step()
        optimizer.zero_grad(set_to_none=True)
        step_seconds.append(time.perf_counter() - step_started)
        loss_value = float(loss.detach())

    step_seconds.sort()
    median = step_seconds[len(step_seconds) // 2]
    # Peak RSS is the portable stand-in for device peak memory. It is the
    # honest measure available without an accelerator, and it is reported under
    # its own name rather than pretending to be allocator statistics.
    peak_memory = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024

    json.dump({
        "median_step_seconds": median,
        "steps_per_second": 1.0 / median if median > 0 else 0.0,
        "peak_memory_bytes": peak_memory,
        "peak_memory_kind": "process_max_rss",
        "input_wait_seconds": input_wait,
        "final_loss": loss_value,
        "quality_metric": "cross_entropy",
        "sequence_length": sequence_length,
        "batch_size": batch_size,
        "accelerator": bool(os.environ.get("CUDA_VISIBLE_DEVICES")),
    }, sys.stdout)
    return 0


if __name__ == "__main__":
    sys.exit(main())
