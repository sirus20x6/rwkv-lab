"""The benchmark runner must measure honestly and decide nothing.

The properties worth pinning are the ones that would let a bad number look
good: phases must run in separate processes, an accelerator fixture must prove
device use and refuse to run beside live training, and the emitted evidence
must be exactly what the native gate accepts rather than a shape the runner
invented.
"""

from __future__ import annotations

import importlib.util
import json
import pathlib
import shutil
import subprocess
import sys

import pytest

REPOSITORY = pathlib.Path(__file__).resolve().parents[1]
RUNNER = REPOSITORY / "scripts/run_benchmark_fixture.py"
MATRIX = REPOSITORY / "docs/experiment-vm/benchmark-matrix.v1.json"

# Three buckets, so the run can actually demonstrate a shape transition.
PORTABLE_FIXTURE = "rwkv.scratch-pretrain"
# One bucket, which by construction cannot demonstrate one.
SINGLE_BUCKET_FIXTURE = "finetune.optimizer-arm"
ACCELERATOR_FIXTURE = "rwkv.pretrained-continuation"

MODULE_SPEC = importlib.util.spec_from_file_location(
    "run_benchmark_fixture", RUNNER)
assert MODULE_SPEC is not None and MODULE_SPEC.loader is not None
benchmark_runner = importlib.util.module_from_spec(MODULE_SPEC)
MODULE_SPEC.loader.exec_module(benchmark_runner)


def run_runner(*arguments: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(RUNNER), *arguments],
        capture_output=True, text=True, cwd=REPOSITORY, check=False,
        timeout=900,
    )


def find_trainvm() -> str:
    candidates = [
        shutil.which("trainvm"),
        str(REPOSITORY / "trainvm/build/trainvm"),
        str(REPOSITORY / "trainvm/build-verify/trainvm"),
    ]
    binary = next(
        (candidate for candidate in candidates
         if candidate and pathlib.Path(candidate).exists()),
        None,
    )
    if binary is None:
        pytest.skip("trainvm binary not built; gate verdict not exercised")
    return binary


def test_unknown_fixture_is_refused():
    result = run_runner("--fixture", "no.such-fixture")
    assert result.returncode == 2
    assert "unknown fixture" in result.stderr


def test_accelerator_fixture_refuses_without_explicit_opt_in():
    accelerator_fixture = next(
        entry["id"] for entry in json.loads(MATRIX.read_text())["fixtures"]
        if entry["accelerator_required"]
    )
    result = run_runner("--fixture", accelerator_fixture)
    assert result.returncode == 2
    assert "requires an accelerator" in result.stderr


def test_workload_selection_follows_fixture_accelerator_requirement():
    assert benchmark_runner.workload_for_fixture(
        {"accelerator_required": False}
    ) == benchmark_runner.PORTABLE_WORKLOAD
    assert benchmark_runner.workload_for_fixture(
        {"accelerator_required": True}
    ) == benchmark_runner.ACCELERATOR_WORKLOAD


def test_ambient_accelerator_residency_is_within_default_allowance():
    conditions = benchmark_runner.parse_accelerator_conditions(
        "1101, 376\n2202, 211\n",
        "97887, 6912, 5\n",
    )
    assert conditions is not None
    assert conditions["resident_process_memory_mib"] == 587
    assert conditions["device_memory_used_mib"] == 6912
    assert conditions["device_memory_total_mib"] == 97887
    assert conditions["device_utilization_percent"] == 5
    assert not benchmark_runner.contention_exceeds_allowance(
        conditions,
        benchmark_runner.DEFAULT_ACCELERATOR_RESIDENT_MEMORY_ALLOWANCE_MIB,
    )
    assert benchmark_runner.contention_exceeds_allowance(conditions, 0)


def test_large_accelerator_residency_exceeds_default_allowance():
    conditions = benchmark_runner.parse_accelerator_conditions(
        "3303, 8192\n",
        "97887, 14336, 82\n",
    )
    assert conditions is not None
    assert benchmark_runner.contention_exceeds_allowance(
        conditions,
        benchmark_runner.DEFAULT_ACCELERATOR_RESIDENT_MEMORY_ALLOWANCE_MIB,
    )
    assert benchmark_runner.contention_exceeds_allowance(
        {},
        benchmark_runner.DEFAULT_ACCELERATOR_RESIDENT_MEMORY_ALLOWANCE_MIB,
    )


def test_unparseable_accelerator_conditions_fail_closed():
    conditions = benchmark_runner.parse_accelerator_conditions(
        "not-a-pid, unknown\n",
        "97887, 6912, 5\n",
    )
    assert conditions is None
    assert benchmark_runner.contention_exceeds_allowance(
        conditions,
        benchmark_runner.DEFAULT_ACCELERATOR_RESIDENT_MEMORY_ALLOWANCE_MIB,
    )


def test_missing_accelerator_usage_marker_prevents_evidence(
    tmp_path, monkeypatch, capsys,
):
    def phase_without_device_attribution(
        phase, bucket, seed, steps, workload, accelerator_required,
    ):
        assert workload == benchmark_runner.ACCELERATOR_WORKLOAD
        assert accelerator_required is True
        return {
            "phase": phase,
            "bucket": bucket,
            "seed": seed,
            "status": "ok",
            "wall_seconds": 0.01,
            "median_step_seconds": 0.01,
            "steps_per_second": 100.0,
            "peak_memory_bytes": 1024,
            "peak_memory_kind": "cuda_max_memory_allocated",
            "input_wait_seconds": 0.0,
            "quality_metric": "cross_entropy",
            "final_loss": 1.0,
            "accelerator": True,
            # Device name and CUDA capability are deliberately missing.
        }

    evidence = tmp_path / "must-not-exist.json"
    monkeypatch.setattr(
        benchmark_runner, "query_accelerator_conditions",
        lambda: {
            "resident_processes": [],
            "resident_process_memory_mib": 0,
            "devices": [{
                "index": 0,
                "used_memory_mib": 0,
                "total_memory_mib": 97887,
                "utilization_percent": 0,
            }],
            "device_memory_used_mib": 0,
            "device_memory_total_mib": 97887,
            "device_utilization_percent": 0,
        },
    )
    monkeypatch.setattr(
        benchmark_runner, "run_phase", phase_without_device_attribution)
    monkeypatch.setattr(sys, "argv", [
        str(RUNNER),
        "--fixture", ACCELERATOR_FIXTURE,
        "--allow-accelerator",
        "--seeds", "1",
        "--steps", "1",
        "--evidence", str(evidence),
    ])

    assert benchmark_runner.main() == 1
    assert not evidence.exists()
    assert "accelerator cell failure; no evidence emitted" in capsys.readouterr().err


@pytest.mark.slow
def test_portable_fixture_emits_evidence_the_gate_accepts(tmp_path):
    evidence = tmp_path / "evidence.json"
    receipt = tmp_path / "receipt.json"
    result = run_runner(
        "--fixture", PORTABLE_FIXTURE, "--seeds", "1", "--steps", "2",
        "--evidence", str(evidence), "--receipt", str(receipt))
    assert result.returncode == 0, result.stderr

    report = json.loads(receipt.read_text())
    assert report["completed_cells"] == report["required_cells"] >= 1
    assert not report["failed_cells"]
    cell = report["cells"][0]
    # Cold compile is a separate process lifetime from the timed run, so it
    # must be reported separately and must not be folded into step time.
    assert cell["cold_compile_seconds"] > 0
    assert cell["warmup_seconds"] > 0
    assert cell["steady_state_step_seconds"] > 0
    assert cell["peak_memory_bytes"] > 0
    assert cell["input_wait_seconds"] >= 0

    document = json.loads(evidence.read_text())
    assert document["api_version"] == "trainvm.cache-qualification-evidence/v1"
    # A self-comparison must not claim a speedup it did not measure.
    assert document["baseline_throughput"] == document["candidate_throughput"]
    assert document["minimum_throughput_gain_ratio"] == 0.0
    # Qualification timing must never come from an instrumented run.
    assert document["baseline_instrumented"] is False
    assert document["candidate_instrumented"] is False

    binary = find_trainvm()
    verdict = subprocess.run(
        [binary, "qualify-evidence"], input=evidence.read_text(),
        capture_output=True, text=True, check=False)
    assert verdict.returncode == 0, verdict.stdout + verdict.stderr
    assert json.loads(verdict.stdout)["qualified"] is True


@pytest.mark.slow
def test_single_bucket_run_cannot_claim_transition_coverage(tmp_path):
    """A one-bucket run must be rejected, not quietly credited.

    The fixture that drives this declares no curriculum transition, and the
    run covers one shape. Reporting transition coverage from the declaration
    rather than from what was measured would manufacture evidence.
    """
    evidence = tmp_path / "evidence.json"
    result = run_runner(
        "--fixture", SINGLE_BUCKET_FIXTURE, "--seeds", "1", "--steps", "2",
        "--evidence", str(evidence))
    assert result.returncode == 0, result.stderr
    document = json.loads(evidence.read_text())
    assert document["transition_coverage"] is False

    binary = find_trainvm()
    verdict = subprocess.run(
        [binary, "qualify-evidence"], input=evidence.read_text(),
        capture_output=True, text=True, check=False)
    assert verdict.returncode == 3, verdict.stdout + verdict.stderr
    assert "missing_transition_coverage" in json.loads(
        verdict.stdout)["rejection_reasons"]


@pytest.mark.slow
@pytest.mark.gpu
def test_accelerator_fixture_emits_evidence_the_gate_accepts(tmp_path):
    cuda_probe = subprocess.run(
        [sys.executable, "-c", "import torch; print(torch.cuda.is_available())"],
        capture_output=True,
        text=True,
        check=False,
    )
    if cuda_probe.returncode != 0 or cuda_probe.stdout.strip() != "True":
        pytest.skip("CUDA is not available")
    if shutil.which("nvidia-smi") is None:
        pytest.skip("nvidia-smi is not available")

    evidence = tmp_path / "accelerator-evidence.json"
    receipt = tmp_path / "accelerator-receipt.json"
    result = run_runner(
        "--fixture", ACCELERATOR_FIXTURE,
        "--allow-accelerator",
        "--seeds", "1",
        "--steps", "2",
        "--evidence", str(evidence),
        "--receipt", str(receipt),
    )
    assert result.returncode == 0, result.stdout + result.stderr

    report = json.loads(receipt.read_text())
    assert report["completed_cells"] == report["required_cells"] >= 1
    assert not report["failed_cells"]
    for cell in report["cells"]:
        assert cell["accelerator"] is True
        assert cell["accelerator_device_name"]
        assert len(cell["accelerator_capability"]) == 2
        assert cell["peak_memory_bytes"] > 0
        assert cell["peak_memory_kind"] == "cuda_max_memory_allocated"
        conditions = cell["accelerator_conditions"]
        assert conditions["resident_process_memory_mib"] <= conditions[
            "resident_memory_allowance_mib"]
        assert conditions["device_memory_total_mib"] > 0
        assert conditions["device_memory_used_mib"] >= 0
        assert 0 <= conditions["device_utilization_percent"] <= 100

    binary = find_trainvm()
    verdict = subprocess.run(
        [binary, "qualify-evidence"],
        input=evidence.read_text(),
        capture_output=True,
        text=True,
        check=False,
    )
    assert verdict.returncode == 0, verdict.stdout + verdict.stderr
    assert json.loads(verdict.stdout)["qualified"] is True
