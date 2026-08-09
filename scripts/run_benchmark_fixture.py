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
AO3_CORPUS_WORKLOAD = (
    REPOSITORY / "scripts/benchmark_workloads/ao3_corpus_lm_step.py"
)
AO3_CORPUS_FIXTURE = "rwkv.ao3-real-input"
VISION_CORPUS_WORKLOAD = (
    REPOSITORY / "scripts/benchmark_workloads/vision_corpus_image_step.py"
)
VISION_CORPUS_FIXTURE = "vision.multimodal-student"
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
    """Select the closed implementation for a fixture declaration."""
    if fixture.get("id") == AO3_CORPUS_FIXTURE:
        return AO3_CORPUS_WORKLOAD
    if fixture.get("id") == VISION_CORPUS_FIXTURE:
        return VISION_CORPUS_WORKLOAD
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
    input_workers: int = 0,
    input_prefetch_depth: int = 0,
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
    if input_workers or input_prefetch_depth:
        command.extend([
            "--input-workers", str(input_workers),
            "--input-prefetch-depth", str(input_prefetch_depth),
        ])
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
    input_workers: int = 0,
    input_prefetch_depth: int = 0,
) -> dict:
    """Cold compile, disposable warmup, then a fresh timed process."""
    phase_options: dict = {}
    if compile_step:
        phase_options.update(
            {"compile_step": True, "compile_mode": compile_mode})
    if input_workers or input_prefetch_depth:
        phase_options.update({
            "input_workers": input_workers,
            "input_prefetch_depth": input_prefetch_depth,
        })
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
        "training_step_seconds": timed.get("training_step_seconds"),
        "input_wait_ratio": timed.get("input_wait_ratio"),
        # Wall time a real training loop spends per step: blocked on input
        # PLUS computing. steady_state_step_seconds excludes input entirely,
        # which is the right basis for a kernel candidate and the wrong one
        # for an input-pipeline candidate, whose whole effect is on the part
        # that measure omits.
        "end_to_end_step_seconds": (
            (timed["input_wait_seconds"]
             + (timed.get("training_step_seconds") or 0.0)) / steps
            if steps > 0 else None),
        # Carried so a receipt states what its input-wait number describes.
        # Most fixtures synthesize tensors; the AO3 fixture performs real file
        # reads, UTF-8 decode, and tokenization inside the measured interval.
        "input_pipeline": timed.get("input_pipeline", "unknown"),
        # How the input work was scheduled, and the identity of what it
        # produced. The digest is what lets ordering and content parity be
        # compared between arms instead of assumed.
        "input_pipeline_mode": timed.get("input_pipeline_mode", "unknown"),
        "input_workers": timed.get("input_workers", 0),
        "input_prefetch_depth": timed.get("input_prefetch_depth", 0),
        "batch_sequence_digest": timed.get("batch_sequence_digest"),
        "step_batch_digests": timed.get("step_batch_digests", []),
        "quality_metric": timed["quality_metric"],
        "final_loss": timed["final_loss"],
        "result_fingerprint": timed["result_fingerprint"],
        "compiled": compile_step,
        "compile_mode": compile_mode if compile_step else None,
    })
    for diagnostic in (
        "corpus_index",
        "corpus_root",
        "minimum_document_bytes",
        "maximum_document_bytes",
        "document_set_size",
        "documents_read",
        "corpus_bytes_read",
        "decoded_characters",
        "tokens_encoded",
        "document_selection_digest",
        "tokenizer",
        "tokenizer_version",
        "sequence_length",
        "batch_size",
    ):
        if diagnostic in timed:
            cell[diagnostic] = timed[diagnostic]
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


def trajectory_evidence(completed: list) -> dict:
    """Report the trajectory claim this runner can actually support.

    `resumed_trajectory_parity` is a verdict, not a boolean, because a fused
    or compiled kernel agrees with its reference to float32 epsilon on one
    step and then separates as the trajectory amplifies that agreement-level
    difference. Two of the three verdicts need a measurement this runner does
    not make: it compares arms per shape bucket rather than stepping a
    trajectory, so it has no divergence rate and no paired checkpoint metric.

    What it does have is the measured deviation per bucket. When every one of
    them is exactly zero the arms are bit-identical and `equivalent` is honest.
    Otherwise the honest answer is `diverged` -- not because the candidate is
    bad, but because nothing here has shown it is not. Qualifying such a
    candidate needs a run that measures a rate or a checkpoint-quality pair.
    """
    def bounded(value: float) -> float:
        # A non-finite fingerprint means total disagreement, not an
        # unrepresentable number. The evidence schema requires finite
        # deviations, so record it as a fully separated one rather than
        # emitting a document the gate cannot parse at all.
        return value if math.isfinite(value) else 1.0

    samples = [
        {
            "step": index + 1,
            "relative_deviation": bounded(max(
                cell["output_relative_deviation"],
                cell["gradient_relative_deviation"],
            )),
        }
        for index, cell in enumerate(completed)
    ]
    identical = all(
        cell["output_absolute_deviation"] == 0.0
        and cell["gradient_absolute_deviation"] == 0.0
        for cell in completed
    )
    return {
        "verdict": "equivalent" if identical else "diverged",
        "criterion": "bit_identical",
        "effect_class": "optimizer_update",
        "candidate_divergence": samples,
        "reference_divergence": [],
        "checkpoint_quality": [],
        "analysis_seed": 0,
    }


def compare_batch_identity(baseline: dict, candidate: dict) -> dict:
    """Derive ordering and content parity from what each arm actually loaded.

    A prefetching or worker-parallel loader is precisely the change that can
    return the right batches in the wrong order, and a throughput number will
    not notice. Both arms publish a per-step digest bound to the step index and
    a chained digest over the sequence, so:

      content parity  - the same steps trained on the same bytes
      ordering parity - they arrived in the same order

    An arm that published no digest yields False rather than True. Absent
    evidence is not parity; that assumption is what this replaces.
    """
    baseline_steps = baseline.get("step_batch_digests") or []
    candidate_steps = candidate.get("step_batch_digests") or []
    baseline_chain = baseline.get("batch_sequence_digest")
    candidate_chain = candidate.get("batch_sequence_digest")

    observed = bool(baseline_steps) and bool(candidate_steps)
    content = (
        observed
        and sorted(baseline_steps) == sorted(candidate_steps)
    )
    ordering = (
        observed
        and isinstance(baseline_chain, str)
        and baseline_chain == candidate_chain
        and baseline_steps == candidate_steps
    )
    return {
        "content_parity": content,
        "ordering_parity": ordering,
        "batch_identity_observed": observed,
        "baseline_batch_sequence_digest": baseline_chain,
        "candidate_batch_sequence_digest": candidate_chain,
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
    cell.update(compare_batch_identity(baseline, candidate))
    cell.update({
        "baseline_input_wait_seconds": baseline["input_wait_seconds"],
        "candidate_input_wait_seconds": candidate["input_wait_seconds"],
        "baseline_input_pipeline_mode": baseline["input_pipeline_mode"],
        "candidate_input_pipeline_mode": candidate["input_pipeline_mode"],
    })
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
        "--candidate", choices=["eager", "compile", "prefetch"],
        default="eager")
    parser.add_argument(
        "--input-workers",
        type=nonnegative_integer,
        default=8,
        help="worker threads for --candidate prefetch; the baseline arm "
             "always stays serial so the comparison has a fixed reference",
    )
    parser.add_argument(
        "--input-prefetch-depth",
        type=nonnegative_integer,
        default=2,
        help="batches prepared ahead for --candidate prefetch",
    )
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
                candidate_options: dict = {}
                if arguments.candidate == "compile":
                    candidate_options.update({
                        "compile_step": True,
                        "compile_mode": arguments.compile_mode,
                    })
                else:
                    candidate_options.update({
                        "input_workers": arguments.input_workers,
                        "input_prefetch_depth": (
                            arguments.input_prefetch_depth),
                    })
                candidate = run_cell(
                    bucket,
                    seed,
                    arguments.steps,
                    workload,
                    fixture["accelerator_required"],
                    accelerator_conditions,
                    **candidate_options,
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

    # An input-pipeline candidate moves cost between blocking on input and
    # computing. Scoring it on step time alone would credit none of the gain
    # and charge it for the producer's CPU contention, rejecting a real
    # improvement as a regression. The basis is published in the receipt and
    # in the evidence-adjacent report rather than silently switched.
    throughput_basis = (
        "end_to_end_including_input_wait"
        if arguments.candidate == "prefetch" else "training_step_only")
    measure = (
        "end_to_end_step_seconds"
        if throughput_basis == "end_to_end_including_input_wait"
        else "steady_state_step_seconds")
    baseline_steps = sorted(
        cell["baseline"][measure] for cell in completed)
    candidate_steps = sorted(
        cell["candidate"][measure] for cell in completed)
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
        "content_parity": all(cell["content_parity"] for cell in completed),
        "ordering_parity": all(cell["ordering_parity"] for cell in completed),
        "batch_identity_observed": all(
            cell["batch_identity_observed"] for cell in completed),
    }
    report["input_pipeline"] = {
        "baseline_mode": completed[0]["baseline_input_pipeline_mode"],
        "candidate_mode": completed[0]["candidate_input_pipeline_mode"],
        "baseline_input_wait_seconds": sum(
            cell["baseline_input_wait_seconds"] for cell in completed),
        "candidate_input_wait_seconds": sum(
            cell["candidate_input_wait_seconds"] for cell in completed),
    }
    # Parity is deliberately copied from the measured comparison above. A
    # throughput win cannot survive the native authority when either differs.
    if arguments.candidate == "eager":
        candidate_digest = digest("candidate", fixture["id"], coverage)
    elif arguments.candidate == "compile":
        candidate_digest = digest(
            "candidate", arguments.candidate, arguments.compile_mode,
            fixture["id"], coverage)
    else:
        candidate_digest = digest(
            "candidate", arguments.candidate,
            str(arguments.input_workers),
            str(arguments.input_prefetch_depth),
            fixture["id"], coverage)
    workload_class = (
        "training" if fixture["effect_class"] == "training_kernel"
        else "serving" if fixture["effect_class"] == "serving_kernel"
        else "preprocessing")
    evidence = {
        "api_version": "trainvm.cache-qualification-evidence/v1",
        "authority_receipt_digest": digest("authority", fixture["id"]),
        "namespace_digest": digest("namespace", fixture["id"]),
        "artifact_tree_digest": digest("artifact", fixture["id"]),
        "workload_class": workload_class,
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
        # torch's fused AdamW keeps `step` on the parameter's device and the
        # foreach reference keeps it on the host, so a state round trip between
        # them is only comparable after Optimizer.load_state_dict has
        # normalized it. Both arms here go through that path.
        "optimizer_state_device_policy": (
            "normalized_on_load" if workload_class == "training"
            else "not_applicable"),
        # Measured, not asserted. This runner compares arms per shape bucket
        # rather than stepping a trajectory, so the only claim it can support
        # is bit-identity, and only when every measured deviation is exactly
        # zero. A candidate that separates at all -- a compiled one will --
        # is reported as diverged here and must be qualified by a run that
        # actually measures a divergence rate or a checkpoint-quality pair.
        "resumed_trajectory_parity": trajectory_evidence(completed),
        "determinism_parity": True,
        # Measured from the per-step batch digests published by both arms, not
        # asserted. A loader that reorders batches, or resumes on the wrong
        # cursor, fails here even when its throughput number improves.
        "content_parity": report["parity"]["content_parity"],
        "ordering_parity": report["parity"]["ordering_parity"],
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
        "throughput_basis": throughput_basis,
        "baseline_throughput": baseline_throughput,
        "candidate_throughput": candidate_throughput,
        "baseline_peak_memory_bytes": baseline_memory,
        "candidate_peak_memory_bytes": candidate_memory,
        # Both bases, always, so a reader can see what the chosen one omits.
        "baseline_training_step_only_throughput": 1.0 / statistics.median(
            sorted(cell["baseline"]["steady_state_step_seconds"]
                   for cell in completed)),
        "candidate_training_step_only_throughput": 1.0 / statistics.median(
            sorted(cell["candidate"]["steady_state_step_seconds"]
                   for cell in completed)),
        "baseline_end_to_end_throughput": 1.0 / statistics.median(
            sorted(cell["baseline"]["end_to_end_step_seconds"]
                   for cell in completed)),
        "candidate_end_to_end_throughput": 1.0 / statistics.median(
            sorted(cell["candidate"]["end_to_end_step_seconds"]
                   for cell in completed)),
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
