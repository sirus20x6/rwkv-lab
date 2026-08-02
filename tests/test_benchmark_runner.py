"""The benchmark runner must measure honestly and decide nothing.

The properties worth pinning are the ones that would let a bad number look
good: phases must run in separate processes, an accelerator fixture must
refuse to run beside live training, and the emitted evidence must be exactly
what the native gate accepts rather than a shape the runner invented.
"""

from __future__ import annotations

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


def run_runner(*arguments: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(RUNNER), *arguments],
        capture_output=True, text=True, cwd=REPOSITORY, check=False,
        timeout=900,
    )


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

    binary = shutil.which("trainvm") or str(
        REPOSITORY / "trainvm/build-verify/trainvm")
    if not pathlib.Path(binary).exists():
        pytest.skip("trainvm binary not built; gate acceptance not exercised")
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

    binary = shutil.which("trainvm") or str(
        REPOSITORY / "trainvm/build-verify/trainvm")
    if not pathlib.Path(binary).exists():
        pytest.skip("trainvm binary not built; gate rejection not exercised")
    verdict = subprocess.run(
        [binary, "qualify-evidence"], input=evidence.read_text(),
        capture_output=True, text=True, check=False)
    assert verdict.returncode == 3, verdict.stdout + verdict.stderr
    assert "missing_transition_coverage" in json.loads(
        verdict.stdout)["rejection_reasons"]
