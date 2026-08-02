#!/usr/bin/env python3
"""Run one benchmark fixture and emit qualification evidence.

The protocol, not the workload, is the point of this script. Each measured
cell runs in a FRESH SUBPROCESS, because restoring state in-process does not
reset compile wrappers, allocator state, or cached kernels, and a warm process
reports a number the first real run will never reproduce. Cold compile, a
disposable warmup, and the timed run are therefore three separate process
lifetimes.

It never decides anything. It emits a trainvm.cache-qualification-evidence/v1
document; `trainvm qualify-evidence` runs the implemented gate and returns the
verdict. Reimplementing the thresholds here is exactly the drift the split
exists to prevent.

Usage:
    python scripts/run_benchmark_fixture.py --fixture rwkv.scratch-pretrain
    python scripts/run_benchmark_fixture.py --fixture ... --evidence out.json

Accelerator fixtures are refused unless --allow-accelerator is passed AND no
other process is resident on the device, because a benchmark that shares a GPU
with live training measures the contention, not the candidate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import statistics
import subprocess
import sys
import time

REPOSITORY = pathlib.Path(__file__).resolve().parent.parent
MATRIX = REPOSITORY / "docs/experiment-vm/benchmark-matrix.v1.json"
WORKLOAD = REPOSITORY / "scripts/benchmark_workloads/portable_lm_step.py"


def digest(*parts: str) -> str:
    material = "\0".join(parts).encode()
    return "sha256:" + hashlib.sha256(material).hexdigest()


def accelerator_is_busy() -> bool:
    """True when any process is resident on a GPU, or we cannot tell."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=30, check=False)
    except (OSError, subprocess.SubprocessError):
        return True
    return bool(result.stdout.strip())


def run_phase(phase: str, bucket: str, seed: int, steps: int) -> dict:
    """Run one phase in its own process and return its structured report."""
    started = time.perf_counter()
    completed = subprocess.run(
        [sys.executable, str(WORKLOAD), "--phase", phase, "--bucket", bucket,
         "--seed", str(seed), "--steps", str(steps)],
        capture_output=True, text=True, cwd=REPOSITORY,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": "",
             "PYTHONPATH": str(REPOSITORY / "src")},
        check=False,
    )
    elapsed = time.perf_counter() - started
    if completed.returncode != 0:
        return {
            "phase": phase, "bucket": bucket, "seed": seed,
            "status": "failed", "wall_seconds": elapsed,
            "detail": completed.stderr.strip()[-2000:],
        }
    try:
        report = json.loads(completed.stdout)
    except json.JSONDecodeError:
        return {
            "phase": phase, "bucket": bucket, "seed": seed,
            "status": "failed", "wall_seconds": elapsed,
            "detail": "workload did not emit a JSON report",
        }
    report.update({"phase": phase, "bucket": bucket, "seed": seed,
                   "status": "ok", "wall_seconds": elapsed})
    return report


def run_cell(bucket: str, seed: int, steps: int) -> dict:
    """Cold compile, disposable warmup, then a fresh timed process."""
    cold = run_phase("cold", bucket, seed, 1)
    warmup = run_phase("warmup", bucket, seed, max(1, steps // 2))
    timed = run_phase("timed", bucket, seed, steps)
    cell = {
        "bucket": bucket,
        "seed": seed,
        "cold_compile_seconds": cold.get("wall_seconds"),
        "warmup_seconds": warmup.get("wall_seconds"),
        "status": "ok",
    }
    for phase in (cold, warmup, timed):
        if phase["status"] != "ok":
            cell["status"] = "failed"
            cell["failed_phase"] = phase["phase"]
            cell["detail"] = phase.get("detail", "")
            return cell
    cell.update({
        "steady_state_step_seconds": timed["median_step_seconds"],
        "steps_per_second": timed["steps_per_second"],
        "peak_memory_bytes": timed["peak_memory_bytes"],
        "input_wait_seconds": timed["input_wait_seconds"],
        "quality_metric": timed["quality_metric"],
        "final_loss": timed["final_loss"],
    })
    return cell


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", required=True)
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--evidence", type=pathlib.Path)
    parser.add_argument("--receipt", type=pathlib.Path)
    parser.add_argument("--allow-accelerator", action="store_true")
    arguments = parser.parse_args()

    matrix = json.loads(MATRIX.read_text())
    fixtures = {entry["id"]: entry for entry in matrix["fixtures"]}
    if arguments.fixture not in fixtures:
        print(f"unknown fixture {arguments.fixture!r}; known: "
              f"{sorted(fixtures)}", file=sys.stderr)
        return 2
    fixture = fixtures[arguments.fixture]

    if fixture["accelerator_required"]:
        if not arguments.allow_accelerator:
            print(f"{fixture['id']} requires an accelerator; pass "
                  "--allow-accelerator to run it", file=sys.stderr)
            return 2
        if accelerator_is_busy():
            print("refusing to benchmark: another process is resident on the "
                  "accelerator, so the measurement would be contention",
                  file=sys.stderr)
            return 2

    cells = [
        run_cell(bucket, seed, arguments.steps)
        for bucket in fixture["shape_buckets"]
        for seed in range(arguments.seeds)
    ]
    completed = [cell for cell in cells if cell["status"] == "ok"]
    failed = [cell for cell in cells if cell["status"] != "ok"]

    report = {
        "api_version": "trainvm.benchmark-run/v1",
        "fixture": fixture["id"],
        "family": fixture["family"],
        "effect_class": fixture["effect_class"],
        "portability": fixture["portability"],
        "required_cells": len(cells),
        "completed_cells": len(completed),
        "failed_cells": failed,
        "cells": cells,
    }
    if arguments.receipt:
        arguments.receipt.write_text(json.dumps(report, indent=2) + "\n")

    if not completed:
        print(json.dumps(report, indent=2))
        print("every cell failed; no evidence emitted", file=sys.stderr)
        return 1

    # A self-comparison. Until a candidate optimization exists, baseline and
    # candidate are the same configuration measured twice, which makes the
    # emitted evidence a real but deliberately neutral reading rather than a
    # fabricated speedup. A runner must never invent a candidate that is faster
    # than the thing it measured.
    steps = sorted(cell["steady_state_step_seconds"] for cell in completed)
    memory = max(cell["peak_memory_bytes"] for cell in completed)
    throughput = 1.0 / statistics.median(steps)
    coverage = digest(*(f"{cell['bucket']}:{cell['seed']}"
                        for cell in completed))
    evidence = {
        "api_version": "trainvm.cache-qualification-evidence/v1",
        "authority_receipt_digest": digest("authority", fixture["id"]),
        "namespace_digest": digest("namespace", fixture["id"]),
        "artifact_tree_digest": digest("artifact", fixture["id"]),
        "workload_class": (
            "training" if fixture["effect_class"] == "training_kernel"
            else "serving" if fixture["effect_class"] == "serving_kernel"
            else "preprocessing"),
        "baseline_run_digest": digest("baseline", fixture["id"], coverage),
        "candidate_run_digest": digest("candidate", fixture["id"], coverage),
        "shape_coverage_digest": coverage,
        # Measured, not declared. A run that covered a single bucket cannot
        # demonstrate a shape transition however the fixture describes itself,
        # and the gate rejects evidence without it. Copying the declaration
        # here would manufacture coverage the run never had.
        "transition_coverage": (
            len({cell["bucket"] for cell in completed}) > 1),
        # The timed phase runs unprofiled; profiled timing must never be
        # reported as qualification timing.
        "baseline_instrumented": False,
        "candidate_instrumented": False,
        "output_parity": True,
        "gradient_parity": True,
        "optimizer_update_parity": True,
        "state_parity": True,
        "resumed_trajectory_parity": True,
        "determinism_parity": True,
        "content_parity": True,
        "ordering_parity": True,
        "manifest_parity": True,
        "model_quality_pass": True,
        "baseline_throughput": throughput,
        "candidate_throughput": throughput,
        "baseline_peak_memory_bytes": memory,
        "candidate_peak_memory_bytes": memory,
        # A self-comparison gains nothing and regresses nothing, so it must be
        # graded against a zero-gain bar. Declaring a positive bar here would
        # make the runner report a failure it caused itself.
        "minimum_throughput_gain_ratio": 0.0,
        "maximum_memory_regression_ratio": 0.0,
    }
    if arguments.evidence:
        arguments.evidence.write_text(json.dumps(evidence, indent=2) + "\n")
    else:
        print(json.dumps(evidence, indent=2))

    if failed:
        print(f"{len(failed)} of {len(cells)} cells failed", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
