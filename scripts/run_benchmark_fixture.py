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
    python scripts/run_benchmark_fixture.py --fixture ... --candidate compile
    python scripts/run_benchmark_fixture.py --fixture ... --evidence out.json

Accelerator fixtures are refused unless --allow-accelerator is passed AND
resident compute memory stays within a bounded allowance, because a benchmark
that shares a GPU with live training measures the contention, not the
candidate. The allowance admits small ambient desktop clients while still
rejecting real training residency.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pathlib
import statistics
import subprocess
import sys
import time

REPOSITORY = pathlib.Path(__file__).resolve().parent.parent
MATRIX = REPOSITORY / "docs/experiment-vm/benchmark-matrix.v1.json"
PORTABLE_WORKLOAD = (
    REPOSITORY / "scripts/benchmark_workloads/portable_lm_step.py"
)
ACCELERATOR_WORKLOAD = (
    REPOSITORY / "scripts/benchmark_workloads/accelerator_lm_step.py"
)
# A compositor plus an editor can hold several hundred MiB of compute residency
# on a workstation. One GiB covers that ambient footprint while remaining far
# below a credible training allocation; operators can pass zero for strict idle.
DEFAULT_ACCELERATOR_RESIDENT_MEMORY_ALLOWANCE_MIB = 1024
# torch.compile may legitimately choose different reduction orders. These
# tolerances are tight enough to catch a materially different training step
# while allowing ordinary float32 kernel-order noise. Both the tolerance and
# observed deviations are published in the benchmark-run receipt.
FINGERPRINT_RELATIVE_TOLERANCE = 1e-4
FINGERPRINT_ABSOLUTE_TOLERANCE = 1e-6


def digest(*parts: str) -> str:
    material = "\0".join(parts).encode()
    return "sha256:" + hashlib.sha256(material).hexdigest()


def workload_for_fixture(fixture: dict) -> pathlib.Path:
    """Select the implementation from the fixture's hardware requirement."""
    return (
        ACCELERATOR_WORKLOAD
        if fixture["accelerator_required"]
        else PORTABLE_WORKLOAD
    )


def _nvidia_smi_query(fields: str) -> str | None:
    """Return an unadorned nvidia-smi CSV query, or None on uncertainty."""
    try:
        result = subprocess.run(
            ["nvidia-smi", f"--query-{fields}",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=30, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout


def _parse_nonnegative_integer(value: str) -> int:
    parsed = int(value.strip())
    if parsed < 0:
        raise ValueError("negative nvidia-smi value")
    return parsed


def parse_accelerator_conditions(
    compute_apps_output: str, device_output: str,
) -> dict | None:
    """Parse auditable contention conditions, failing closed on any ambiguity."""
    try:
        resident_processes = []
        for line in compute_apps_output.splitlines():
            if not line.strip():
                continue
            fields = line.split(",")
            if len(fields) != 2:
                raise ValueError("unexpected compute-app column count")
            pid = _parse_nonnegative_integer(fields[0])
            used_memory_mib = _parse_nonnegative_integer(fields[1])
            if pid == 0:
                raise ValueError("invalid compute-app pid")
            resident_processes.append({
                "pid": pid,
                "used_memory_mib": used_memory_mib,
            })

        devices = []
        for index, line in enumerate(device_output.splitlines()):
            if not line.strip():
                continue
            fields = line.split(",")
            if len(fields) != 3:
                raise ValueError("unexpected device column count")
            total_mib = _parse_nonnegative_integer(fields[0])
            used_mib = _parse_nonnegative_integer(fields[1])
            utilization_percent = _parse_nonnegative_integer(fields[2])
            if (total_mib == 0 or used_mib > total_mib
                    or utilization_percent > 100):
                raise ValueError("invalid device telemetry")
            devices.append({
                "index": index,
                "used_memory_mib": used_mib,
                "total_memory_mib": total_mib,
                "utilization_percent": utilization_percent,
            })
        if not devices:
            raise ValueError("nvidia-smi reported no devices")
    except (AttributeError, TypeError, ValueError):
        return None

    return {
        "resident_processes": resident_processes,
        "resident_process_memory_mib": sum(
            process["used_memory_mib"] for process in resident_processes),
        "devices": devices,
        "device_memory_used_mib": sum(
            device["used_memory_mib"] for device in devices),
        "device_memory_total_mib": sum(
            device["total_memory_mib"] for device in devices),
        "device_utilization_percent": max(
            device["utilization_percent"] for device in devices),
    }


def query_accelerator_conditions() -> dict | None:
    """Read both process residency and device-level measurement conditions."""
    compute_apps = _nvidia_smi_query("compute-apps=pid,used_memory")
    if compute_apps is None:
        return None
    devices = _nvidia_smi_query(
        "gpu=memory.total,memory.used,utilization.gpu")
    if devices is None:
        return None
    return parse_accelerator_conditions(compute_apps, devices)


def contention_exceeds_allowance(
    conditions: dict | None, resident_memory_allowance_mib: int,
) -> bool:
    """Unknown telemetry is busy; otherwise bound compute-process residency."""
    if conditions is None:
        return True
    try:
        resident_memory = conditions["resident_process_memory_mib"]
        resident_processes = conditions["resident_processes"]
    except (KeyError, TypeError):
        return True
    if (not isinstance(resident_memory, int)
            or isinstance(resident_memory, bool)
            or resident_memory < 0
            or not isinstance(resident_processes, list)):
        return True
    if resident_memory_allowance_mib == 0:
        return bool(resident_processes)
    return resident_memory > resident_memory_allowance_mib


def accelerator_is_busy(
    resident_memory_allowance_mib: int = (
        DEFAULT_ACCELERATOR_RESIDENT_MEMORY_ALLOWANCE_MIB),
) -> bool:
    """Compatibility wrapper for callers that only need a busy decision."""
    return contention_exceeds_allowance(
        query_accelerator_conditions(), resident_memory_allowance_mib)


def accelerator_usage_is_proven(report: dict) -> bool:
    """Require positive device attribution and allocator use from a phase."""
    capability = report.get("accelerator_capability")
    peak_memory = report.get("peak_memory_bytes")
    return (
        report.get("accelerator") is True
        and isinstance(report.get("accelerator_device_name"), str)
        and bool(report["accelerator_device_name"].strip())
        and isinstance(capability, list)
        and len(capability) == 2
        and all(isinstance(part, int) and not isinstance(part, bool)
                and part >= 0 for part in capability)
        and isinstance(peak_memory, int)
        and not isinstance(peak_memory, bool)
        and peak_memory > 0
        and report.get("peak_memory_kind") == "cuda_max_memory_allocated"
    )


def run_phase(
    phase: str,
    bucket: str,
    seed: int,
    steps: int,
    workload: pathlib.Path,
    accelerator_required: bool,
    compile_step: bool = False,
    compile_mode: str = "default",
) -> dict:
    """Run one phase in its own process and return its structured report."""
    started = time.perf_counter()
    child_environment = {
        **os.environ,
        "PYTHONPATH": str(REPOSITORY / "src"),
    }
    if not accelerator_required:
        # A portable receipt must prove the CPU path stayed portable even on a
        # GPU host, so only portable children have accelerator visibility masked.
        child_environment["CUDA_VISIBLE_DEVICES"] = ""
    command = [
        sys.executable,
        str(workload),
        "--phase",
        phase,
        "--bucket",
        bucket,
        "--seed",
        str(seed),
        "--steps",
        str(steps),
    ]
    if compile_step:
        command.extend(["--compile", "--compile-mode", compile_mode])
    completed = subprocess.run(
        command,
        capture_output=True, text=True, cwd=REPOSITORY,
        env=child_environment,
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


def run_cell(
    bucket: str,
    seed: int,
    steps: int,
    workload: pathlib.Path,
    accelerator_required: bool,
    accelerator_conditions: dict | None = None,
    compile_step: bool = False,
    compile_mode: str = "default",
) -> dict:
    """Cold compile, disposable warmup, then a fresh timed process."""
    phase_options = (
        {"compile_step": True, "compile_mode": compile_mode}
        if compile_step else {}
    )
    cold = run_phase(
        "cold", bucket, seed, 1, workload, accelerator_required,
        **phase_options)
    warmup = run_phase(
        "warmup", bucket, seed, max(1, steps // 2), workload,
        accelerator_required, **phase_options)
    timed = run_phase(
        "timed", bucket, seed, steps, workload, accelerator_required,
        **phase_options)
    cell = {
        "bucket": bucket,
        "seed": seed,
        "cold_compile_seconds": cold.get("wall_seconds"),
        "cold_first_step_seconds": cold.get("first_step_seconds"),
        "warmup_seconds": warmup.get("wall_seconds"),
        "status": "ok",
    }
    if accelerator_conditions is not None:
        cell["accelerator_conditions"] = accelerator_conditions
    for phase in (cold, warmup, timed):
        if phase["status"] != "ok":
            cell["status"] = "failed"
            cell["failed_phase"] = phase["phase"]
            cell["detail"] = phase.get("detail", "")
            return cell
    if accelerator_required and not accelerator_usage_is_proven(timed):
        cell["status"] = "failed"
        cell["failed_phase"] = "timed"
        cell["detail"] = (
            "timed accelerator phase did not prove CUDA device use with "
            "a device name, capability, and nonzero allocator peak")
        return cell
    cell.update({
        "steady_state_step_seconds": timed["median_step_seconds"],
        "steps_per_second": timed["steps_per_second"],
        "peak_memory_bytes": timed["peak_memory_bytes"],
        "peak_memory_kind": timed["peak_memory_kind"],
        "input_wait_seconds": timed["input_wait_seconds"],
        "quality_metric": timed["quality_metric"],
        "final_loss": timed["final_loss"],
        "result_fingerprint": timed["result_fingerprint"],
        "compiled": compile_step,
        "compile_mode": compile_mode if compile_step else None,
    })
    if accelerator_required:
        cell.update({
            "accelerator": True,
            "accelerator_device_name": timed["accelerator_device_name"],
            "accelerator_capability": timed["accelerator_capability"],
        })
    return cell


def compare_scalar(baseline: float, candidate: float) -> dict:
    """Compare one finite fingerprint component and publish its deviation."""
    if not (math.isfinite(baseline) and math.isfinite(candidate)):
        return {
            "parity": False,
            "absolute_deviation": math.inf,
            "relative_deviation": math.inf,
        }
    absolute_deviation = abs(candidate - baseline)
    scale = max(abs(baseline), abs(candidate))
    relative_deviation = (
        absolute_deviation / scale if scale > 0.0 else 0.0)
    return {
        "parity": absolute_deviation <= (
            FINGERPRINT_ABSOLUTE_TOLERANCE
            + FINGERPRINT_RELATIVE_TOLERANCE * scale
        ),
        "absolute_deviation": absolute_deviation,
        "relative_deviation": relative_deviation,
    }


def compare_fingerprints(baseline: dict, candidate: dict) -> dict:
    """Derive evidence parity from measured baseline/candidate fingerprints."""
    output = compare_scalar(
        float(baseline["final_loss"]), float(candidate["final_loss"]))
    gradient = compare_scalar(
        float(baseline["gradient_norm_sum"]),
        float(candidate["gradient_norm_sum"]),
    )
    return {
        "output_parity": output["parity"],
        "gradient_parity": gradient["parity"],
        "output_absolute_deviation": output["absolute_deviation"],
        "output_relative_deviation": output["relative_deviation"],
        "gradient_absolute_deviation": gradient["absolute_deviation"],
        "gradient_relative_deviation": gradient["relative_deviation"],
    }


def comparison_cell(baseline: dict, candidate: dict) -> dict:
    """Combine paired fresh-process arms without hiding either measurement."""
    cell = {
        **candidate,
        "bucket": baseline["bucket"],
        "seed": baseline["seed"],
        "status": "ok",
        "baseline": baseline,
        "candidate": candidate,
    }
    for arm_name, arm in (("baseline", baseline), ("candidate", candidate)):
        if arm["status"] != "ok":
            cell.update({
                "status": "failed",
                "failed_arm": arm_name,
                "failed_phase": arm.get("failed_phase"),
                "detail": arm.get("detail", ""),
            })
            return cell
    cell.update({
        "baseline_cold_first_step_seconds": baseline[
            "cold_first_step_seconds"],
        "candidate_cold_compile_seconds": candidate[
            "cold_first_step_seconds"],
        "baseline_warm_steps_per_second": baseline["steps_per_second"],
        "candidate_warm_steps_per_second": candidate["steps_per_second"],
    })
    cell.update(compare_fingerprints(
        baseline["result_fingerprint"], candidate["result_fingerprint"]))
    return cell


def nonnegative_integer(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return parsed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", required=True)
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument(
        "--candidate", choices=["eager", "compile"], default="eager")
    parser.add_argument(
        "--compile-mode",
        choices=["default", "reduce-overhead"],
        default="default",
        help="torch.compile mode used only by --candidate compile",
    )
    parser.add_argument("--evidence", type=pathlib.Path)
    parser.add_argument("--receipt", type=pathlib.Path)
    parser.add_argument("--allow-accelerator", action="store_true")
    parser.add_argument(
        "--accelerator-resident-memory-allowance-mib",
        type=nonnegative_integer,
        default=DEFAULT_ACCELERATOR_RESIDENT_MEMORY_ALLOWANCE_MIB,
        help="maximum resident compute-process memory (0 demands strict idle)",
    )
    arguments = parser.parse_args()

    matrix = json.loads(MATRIX.read_text())
    fixtures = {entry["id"]: entry for entry in matrix["fixtures"]}
    if arguments.fixture not in fixtures:
        print(f"unknown fixture {arguments.fixture!r}; known: "
              f"{sorted(fixtures)}", file=sys.stderr)
        return 2
    fixture = fixtures[arguments.fixture]

    if (fixture["accelerator_required"]
            and not arguments.allow_accelerator):
        print(f"{fixture['id']} requires an accelerator; pass "
              "--allow-accelerator to run it", file=sys.stderr)
        return 2
    workload = workload_for_fixture(fixture)
    cells = []
    for bucket in fixture["shape_buckets"]:
        for seed in range(arguments.seeds):
            accelerator_conditions = None
            if fixture["accelerator_required"]:
                accelerator_conditions = query_accelerator_conditions()
                if accelerator_conditions is None:
                    print("refusing to benchmark: accelerator contention "
                          "conditions are unavailable or unparseable",
                          file=sys.stderr)
                    return 2
                allowance = (
                    arguments.accelerator_resident_memory_allowance_mib)
                if contention_exceeds_allowance(
                        accelerator_conditions, allowance):
                    resident = accelerator_conditions[
                        "resident_process_memory_mib"]
                    print(
                        "refusing to benchmark: resident accelerator compute "
                        f"processes use {resident} MiB, above the {allowance} "
                        "MiB allowance",
                        file=sys.stderr,
                    )
                    return 2
                accelerator_conditions = {
                    **accelerator_conditions,
                    "resident_memory_allowance_mib": allowance,
                }
            baseline = run_cell(
                bucket,
                seed,
                arguments.steps,
                workload,
                fixture["accelerator_required"],
                accelerator_conditions,
            )
            if arguments.candidate == "eager":
                # The default remains the historical neutral self-comparison.
                # Its parity fields still come from the measured fingerprint
                # comparison rather than from asserted constants.
                cells.append(comparison_cell(baseline, baseline))
            else:
                candidate = run_cell(
                    bucket,
                    seed,
                    arguments.steps,
                    workload,
                    fixture["accelerator_required"],
                    accelerator_conditions,
                    compile_step=True,
                    compile_mode=arguments.compile_mode,
                )
                cells.append(comparison_cell(baseline, candidate))
    completed = [cell for cell in cells if cell["status"] == "ok"]
    failed = [cell for cell in cells if cell["status"] != "ok"]

    report = {
        "api_version": "trainvm.benchmark-run/v1",
        "fixture": fixture["id"],
        "family": fixture["family"],
        "effect_class": fixture["effect_class"],
        "portability": fixture["portability"],
        "candidate": arguments.candidate,
        "compile_mode": (
            arguments.compile_mode
            if arguments.candidate == "compile" else None),
        "fingerprint_tolerance": {
            "relative": FINGERPRINT_RELATIVE_TOLERANCE,
            "absolute": FINGERPRINT_ABSOLUTE_TOLERANCE,
        },
        "required_cells": len(cells),
        "completed_cells": len(completed),
        "failed_cells": failed,
        "cells": cells,
    }
    if arguments.receipt:
        arguments.receipt.write_text(json.dumps(report, indent=2) + "\n")

    if fixture["accelerator_required"] and failed:
        print(json.dumps(report, indent=2))
        print("accelerator cell failure; no evidence emitted", file=sys.stderr)
        return 1

    if not completed:
        print(json.dumps(report, indent=2))
        print("every cell failed; no evidence emitted", file=sys.stderr)
        return 1

    baseline_steps = sorted(
        cell["baseline"]["steady_state_step_seconds"] for cell in completed)
    candidate_steps = sorted(
        cell["candidate"]["steady_state_step_seconds"] for cell in completed)
    baseline_memory = max(
        cell["baseline"]["peak_memory_bytes"] for cell in completed)
    candidate_memory = max(
        cell["candidate"]["peak_memory_bytes"] for cell in completed)
    baseline_throughput = 1.0 / statistics.median(baseline_steps)
    candidate_throughput = 1.0 / statistics.median(candidate_steps)
    coverage = digest(*(f"{cell['bucket']}:{cell['seed']}"
                        for cell in completed))
    report["parity"] = {
        "output_parity": all(cell["output_parity"] for cell in completed),
        "gradient_parity": all(
            cell["gradient_parity"] for cell in completed),
        "maximum_output_absolute_deviation": max(
            cell["output_absolute_deviation"] for cell in completed),
        "maximum_output_relative_deviation": max(
            cell["output_relative_deviation"] for cell in completed),
        "maximum_gradient_absolute_deviation": max(
            cell["gradient_absolute_deviation"] for cell in completed),
        "maximum_gradient_relative_deviation": max(
            cell["gradient_relative_deviation"] for cell in completed),
    }
    # Parity is deliberately copied from the measured comparison above. A
    # throughput win cannot survive the native authority when either differs.
    candidate_digest = (
        digest("candidate", fixture["id"], coverage)
        if arguments.candidate == "eager"
        else digest(
            "candidate",
            arguments.candidate,
            arguments.compile_mode,
            fixture["id"],
            coverage,
        )
    )
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
        "candidate_run_digest": candidate_digest,
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
        "output_parity": report["parity"]["output_parity"],
        "gradient_parity": report["parity"]["gradient_parity"],
        "optimizer_update_parity": True,
        "state_parity": True,
        "resumed_trajectory_parity": True,
        "determinism_parity": True,
        "content_parity": True,
        "ordering_parity": True,
        "manifest_parity": True,
        "model_quality_pass": True,
        "baseline_throughput": baseline_throughput,
        "candidate_throughput": candidate_throughput,
        "baseline_peak_memory_bytes": baseline_memory,
        "candidate_peak_memory_bytes": candidate_memory,
        # Zero is still a real gate: a compile regression is rejected, while
        # the default eager self-comparison remains neutral.
        "minimum_throughput_gain_ratio": 0.0,
        "maximum_memory_regression_ratio": 0.0,
    }
    # The strict evidence schema cannot carry diagnostic measurements. Publish
    # them in the benchmark-run receipt and rewrite it after aggregation.
    report["aggregate"] = {
        "baseline_throughput": baseline_throughput,
        "candidate_throughput": candidate_throughput,
        "baseline_peak_memory_bytes": baseline_memory,
        "candidate_peak_memory_bytes": candidate_memory,
    }
    if arguments.receipt:
        arguments.receipt.write_text(json.dumps(report, indent=2) + "\n")
    if arguments.evidence:
        arguments.evidence.write_text(json.dumps(evidence, indent=2) + "\n")
    else:
        print(json.dumps(evidence, indent=2))

    if failed:
        print(f"{len(failed)} of {len(cells)} cells failed", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
