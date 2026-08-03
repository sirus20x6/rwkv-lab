#!/usr/bin/env python3
"""One accelerator language-model training step, measured.

This is the CUDA counterpart to portable_lm_step.py. It drives the same
registered optimizer, schedule, and objective, but it requires a real CUDA
device and reports CUDA allocator telemetry. It never falls back to the CPU.

Run as a subprocess by scripts/run_benchmark_fixture.py, one process per
phase, and emits a single JSON report on stdout. Nothing here decides whether
a result qualifies.
"""

from __future__ import annotations

import argparse
import json
import sys
import time

# Run as a script the sibling module is already importable; loaded by file
# path from a test it is not, so make both work.
import os.path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from batch_identity import batch_digest, chain_digests  # noqa: E402


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
    parser.add_argument("--compile", action="store_true", dest="compile_step")
    parser.add_argument(
        "--compile-mode",
        choices=["default", "reduce-overhead"],
        default="default",
    )
    parser.add_argument("--fallback-bucket")
    arguments = parser.parse_args()

    # Import inside main so the cold phase pays the real import cost, which is
    # a genuine part of a fresh worker's startup.
    import torch
    from torch import nn

    if not torch.cuda.is_available():
        raise SystemExit(
            "CUDA is required for the accelerator benchmark workload; "
            "no CUDA device is available")

    import rwkv_lab.training_components as components

    sequence_length, batch_size = parse_bucket(arguments.bucket)
    device_index = torch.cuda.current_device()
    device = torch.device("cuda", device_index)
    torch.manual_seed(arguments.seed)
    torch.cuda.manual_seed_all(arguments.seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    vocabulary, width = 512, 128

    model = nn.Sequential(
        nn.Embedding(vocabulary, width),
        nn.LayerNorm(width),
        nn.Linear(width, width),
        nn.GELU(),
        nn.LayerNorm(width),
    ).to(device)
    head = nn.Linear(width, vocabulary, bias=False).to(device)
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
        # This fixture validates the accelerator protocol rather than an
        # optional flash-attn installation, so use the registered deterministic
        # PyTorch path on every supported CUDA stack.
        components.LinearHeadCrossEntropyConfiguration(prefer_fused=False),
    )

    step_seconds: list[float] = []
    input_wait = 0.0
    loss_value = float("nan")
    generator = torch.Generator(device=device).manual_seed(arguments.seed)

    def training_step(tokens, targets):
        hidden = model(tokens)
        loss = objective(hidden, head, targets)
        loss.backward()
        gradient_norm = torch.stack([
            torch.linalg.vector_norm(parameter.grad.detach().to(torch.float64))
            for parameter in parameters
            if parameter.grad is not None
        ]).sum()
        optimizer.step()
        schedule.step()
        optimizer.zero_grad(set_to_none=True)
        return loss.detach(), gradient_norm.detach()

    measured_step = training_step
    if arguments.compile_step:
        measured_step = torch.compile(
            training_step,
            dynamic=False,
            mode=arguments.compile_mode,
        )

    final_gradient_norm = float("nan")
    step_digests: list[str] = []

    torch.cuda.reset_peak_memory_stats(device)
    for step_index in range(arguments.steps):
        # Input wait includes completion of device-side input generation rather
        # than measuring only the time needed to enqueue it.
        input_started = time.perf_counter()
        tokens = torch.randint(
            0,
            vocabulary,
            (batch_size, sequence_length),
            generator=generator,
            device=device,
        )
        targets = torch.randint(
            0,
            vocabulary,
            (batch_size, sequence_length),
            generator=generator,
            device=device,
        )
        torch.cuda.synchronize(device)
        input_wait += time.perf_counter() - input_started
        # Digested after the fence, so the bytes read back are the completed
        # device-side batch rather than a partially enqueued one.
        step_digests.append(batch_digest(step_index, tokens, targets))

        # CUDA launches are asynchronous. These fences make the interval
        # contain completed device work, not merely queued launches.
        torch.cuda.synchronize(device)
        step_started = time.perf_counter()
        loss, gradient_norm = measured_step(tokens, targets)
        torch.cuda.synchronize(device)
        step_seconds.append(time.perf_counter() - step_started)
        loss_value = float(loss.detach())
        final_gradient_norm = float(gradient_norm.detach())

    fallback = None
    if arguments.fallback_bucket:
        if arguments.fallback_bucket == arguments.bucket:
            raise SystemExit("fallback bucket must be outside the compiled shape")
        fallback_sequence, fallback_batch = parse_bucket(
            arguments.fallback_bucket)
        fallback_tokens = torch.randint(
            0,
            vocabulary,
            (fallback_batch, fallback_sequence),
            generator=generator,
            device=device,
        )
        fallback_targets = torch.randint(
            0,
            vocabulary,
            (fallback_batch, fallback_sequence),
            generator=generator,
            device=device,
        )
        torch.cuda.synchronize(device)
        fallback_started = time.perf_counter()
        fallback_loss, fallback_gradient_norm = measured_step(
            fallback_tokens, fallback_targets)
        torch.cuda.synchronize(device)
        fallback_seconds = time.perf_counter() - fallback_started
        fallback = {
            "bucket": arguments.fallback_bucket,
            "step_seconds": fallback_seconds,
            "result_fingerprint": {
                "final_loss": float(fallback_loss.detach()),
                "gradient_norm_sum": float(fallback_gradient_norm.detach()),
            },
        }

    first_step = step_seconds[0]
    sorted_step_seconds = sorted(step_seconds)
    median = sorted_step_seconds[len(sorted_step_seconds) // 2]
    training_step_seconds = sum(step_seconds)
    measured_total = input_wait + training_step_seconds
    peak_memory = int(torch.cuda.max_memory_allocated(device))
    capability = torch.cuda.get_device_capability(device)

    json.dump({
        "median_step_seconds": median,
        "first_step_seconds": first_step,
        "steps_per_second": 1.0 / median if median > 0 else 0.0,
        "peak_memory_bytes": peak_memory,
        "peak_memory_kind": "cuda_max_memory_allocated",
        "input_wait_seconds": input_wait,
        "training_step_seconds": training_step_seconds,
        "input_wait_ratio": input_wait / measured_total if measured_total else 0.0,
        # Synthesized on device; see portable_lm_step.py. The interval above is
        # tensor construction, not dataloader stall, and no host-to-device copy
        # is being measured because the data never leaves the device.
        "input_pipeline": "synthetic_in_process",
        "input_workers": 0,
        "input_prefetch_depth": 0,
        "input_pipeline_mode": "serial",
        "batch_sequence_digest": chain_digests(step_digests),
        "step_batch_digests": step_digests,
        "final_loss": loss_value,
        "result_fingerprint": {
            "final_loss": loss_value,
            "gradient_norm_sum": final_gradient_norm,
        },
        "quality_metric": "cross_entropy",
        "sequence_length": sequence_length,
        "batch_size": batch_size,
        "accelerator": True,
        "accelerator_device_name": torch.cuda.get_device_name(device),
        "accelerator_capability": list(capability),
        "compiled": arguments.compile_step,
        "compile_mode": (
            arguments.compile_mode if arguments.compile_step else None),
        "fallback": fallback,
    }, sys.stdout)
    return 0


if __name__ == "__main__":
    sys.exit(main())
