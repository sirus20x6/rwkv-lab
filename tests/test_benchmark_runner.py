"""The benchmark runner must measure honestly and decide nothing.

The properties worth pinning are the ones that would let a bad number look
good: phases must run in separate processes, an accelerator fixture must prove
device use and refuse to run beside live training, and the emitted evidence
must be exactly what the native gate accepts rather than a shape the runner
invented.
"""

from __future__ import annotations

import datetime
import importlib.util
import json
import os
import pathlib
import shutil
import socket
import sqlite3
import subprocess
import sys

import pytest

# tests/ is on sys.path because this directory holds the rootdir conftest.
import trainvm_binary
import ztok_binary

REPOSITORY = pathlib.Path(__file__).resolve().parents[1]
RUNNER = REPOSITORY / "scripts/run_benchmark_fixture.py"
MATRIX = REPOSITORY / "docs/experiment-vm/benchmark-matrix.v1.json"
PORTABLE_WORKLOAD = (
    REPOSITORY / "scripts/benchmark_workloads/portable_lm_step.py"
)
AO3_CORPUS_WORKLOAD = (
    REPOSITORY / "scripts/benchmark_workloads/ao3_corpus_lm_step.py"
)
QUALIFIED_EVIDENCE = (
    REPOSITORY
    / "docs/experiment-vm/examples/qualification-evidence.qualified.v1.json"
)

# Three buckets, so the run can actually demonstrate a shape transition.
PORTABLE_FIXTURE = "rwkv.scratch-pretrain"
# One bucket, which by construction cannot demonstrate one.
SINGLE_BUCKET_FIXTURE = "finetune.optimizer-arm"
ACCELERATOR_FIXTURE = "rwkv.pretrained-continuation"
AO3_CORPUS_FIXTURE = "rwkv.ao3-real-input"

MODULE_SPEC = importlib.util.spec_from_file_location(
    "run_benchmark_fixture", RUNNER)
assert MODULE_SPEC is not None and MODULE_SPEC.loader is not None
benchmark_runner = importlib.util.module_from_spec(MODULE_SPEC)
MODULE_SPEC.loader.exec_module(benchmark_runner)

AO3_MODULE_SPEC = importlib.util.spec_from_file_location(
    "ao3_corpus_lm_step", AO3_CORPUS_WORKLOAD)
assert AO3_MODULE_SPEC is not None and AO3_MODULE_SPEC.loader is not None
ao3_workload = importlib.util.module_from_spec(AO3_MODULE_SPEC)
sys.modules[AO3_MODULE_SPEC.name] = ao3_workload
AO3_MODULE_SPEC.loader.exec_module(ao3_workload)


def run_runner(*arguments: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(RUNNER), *arguments],
        capture_output=True, text=True, cwd=REPOSITORY, check=False,
        timeout=900,
    )


def find_trainvm() -> str:
    """The gate binary is the one built from this checkout, or none at all.

    See tests/trainvm_binary.py for why PATH is never consulted: preferring a
    global install meant these graders could judge new source with an old
    binary, on exactly the hosts where someone is reading the verdict. The
    chosen binary is also printed in the pytest header on every run, so a
    surprising verdict can be attributed to the thing that produced it.
    """
    binary, description = trainvm_binary.resolve_trainvm()
    if binary is None:
        pytest.skip(f"trainvm gate binary {description}")
    print(f"grading with trainvm binary: {binary} ({description})")
    return binary


def test_gate_binary_is_never_taken_from_path(tmp_path, monkeypatch):
    """A trainvm on PATH must not be able to grade this checkout.

    The rigged PATH here is the whole test: `shutil.which` is asserted to find
    the decoy first, so the resolver is genuinely being offered it and is
    genuinely declining. Before this change the decoy would have been returned
    and used to judge qualification evidence.
    """
    decoy_directory = tmp_path / "bin"
    decoy_directory.mkdir()
    decoy = decoy_directory / "trainvm"
    decoy.write_text("#!/bin/sh\necho 'not the binary under test'\nexit 1\n")
    decoy.chmod(0o755)
    monkeypatch.setenv("PATH", str(decoy_directory))
    monkeypatch.delenv("TRAINVM_BINARY", raising=False)
    monkeypatch.delenv("TRAINVM_BUILD_DIR", raising=False)

    assert shutil.which("trainvm") == str(decoy)

    binary, description = trainvm_binary.resolve_trainvm()
    assert binary != str(decoy)
    assert str(decoy_directory) not in (binary or "")
    if binary is not None:
        assert pathlib.Path(binary).is_relative_to(REPOSITORY), description


def test_explicit_binary_request_that_cannot_be_honoured_is_loud(
        tmp_path, monkeypatch):
    """Asking for a specific binary and not getting it must not become a skip."""
    monkeypatch.setenv("TRAINVM_BINARY", str(tmp_path / "absent"))
    with pytest.raises(trainvm_binary.TrainvmBinaryError):
        trainvm_binary.resolve_trainvm()


def test_report_header_records_which_binary_would_grade(monkeypatch):
    """Every run states the grader, including when there is none."""
    monkeypatch.delenv("TRAINVM_BINARY", raising=False)
    line = trainvm_binary.report_line()
    assert line.startswith("trainvm gate binary: ")
    binary, _ = trainvm_binary.resolve_trainvm()
    assert (binary or "none") in line


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
    assert benchmark_runner.workload_for_fixture({
        "id": AO3_CORPUS_FIXTURE,
        "accelerator_required": False,
    }) == benchmark_runner.AO3_CORPUS_WORKLOAD


def test_run_phase_selects_eager_or_compile_candidate(monkeypatch):
    commands = []

    def completed(command, **_kwargs):
        commands.append(command)
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=json.dumps({"median_step_seconds": 1.0}),
            stderr="",
        )

    monkeypatch.setattr(benchmark_runner.subprocess, "run", completed)
    eager_report = benchmark_runner.run_phase(
        "timed",
        "seq8xbatch1",
        0,
        2,
        benchmark_runner.PORTABLE_WORKLOAD,
        False,
    )
    report = benchmark_runner.run_phase(
        "timed",
        "seq8xbatch1",
        0,
        2,
        benchmark_runner.PORTABLE_WORKLOAD,
        False,
        compile_step=True,
        compile_mode="reduce-overhead",
    )
    assert eager_report["status"] == report["status"] == "ok"
    assert "--compile" not in commands[0]
    assert "--compile" in commands[1]
    assert commands[1][-2:] == ["--compile-mode", "reduce-overhead"]


@pytest.mark.parametrize(
    ("candidate", "output_parity", "gradient_parity"),
    [
        (
            {"final_loss": 2.000001, "gradient_norm_sum": 8.000004},
            True,
            True,
        ),
        (
            {"final_loss": 2.01, "gradient_norm_sum": 8.1},
            False,
            False,
        ),
    ],
)
def test_fingerprint_comparison_is_measured(
    candidate, output_parity, gradient_parity,
):
    comparison = benchmark_runner.compare_fingerprints(
        {"final_loss": 2.0, "gradient_norm_sum": 8.0}, candidate)
    assert comparison["output_parity"] is output_parity
    assert comparison["gradient_parity"] is gradient_parity
    assert comparison["output_relative_deviation"] >= 0.0
    assert comparison["gradient_relative_deviation"] >= 0.0


def test_parity_failure_is_rejected_by_native_authority():
    comparison = benchmark_runner.compare_fingerprints(
        {"final_loss": 2.0, "gradient_norm_sum": 8.0},
        {"final_loss": 2.01, "gradient_norm_sum": 8.0},
    )
    document = json.loads(QUALIFIED_EVIDENCE.read_text())
    document["output_parity"] = comparison["output_parity"]
    binary = find_trainvm()
    verdict = subprocess.run(
        [binary, "qualify-evidence"],
        input=json.dumps(document),
        capture_output=True,
        text=True,
        check=False,
    )
    assert verdict.returncode == 3, verdict.stdout + verdict.stderr
    assert "output_parity_failed" in json.loads(
        verdict.stdout)["rejection_reasons"]


GPU_GRADER_OBSERVATIONS = (
    REPOSITORY / "docs/experiment-vm/gpu-grader-observations.v1.json"
)


def record_gpu_grader_observation(
    grader: str, fixture: str, device_name: str, measurement: dict,
) -> None:
    """Write down that this grader produced a number, here, just now.

    Rewrites the committed receipt in place. The caller is expected to commit
    the result: the file being in the tree is what makes "never ran" visible,
    and a run on a host without an accelerator leaves it untouched, which is
    the state the receipt exists to expose.
    """
    document = json.loads(GPU_GRADER_OBSERVATIONS.read_text(encoding="utf-8"))
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True,
        cwd=REPOSITORY, check=False)
    document["graders"][grader] = {
        "last_observed": datetime.datetime.now(
            datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "commit": revision.stdout.strip() or "unknown",
        "host": socket.gethostname(),
        "accelerator_device_name": device_name,
        "fixture": fixture,
        "measurement": measurement,
    }
    GPU_GRADER_OBSERVATIONS.write_text(
        json.dumps(document, indent=2) + "\n", encoding="utf-8")


def gpu_marked_graders_in_this_module() -> set[str]:
    """The gpu-marked test functions this module defines, by name."""
    return {
        name for name, value in globals().items()
        if name.startswith("test_") and callable(value)
        and any(mark.name == "gpu"
                for mark in getattr(value, "pytestmark", []))
    }


def test_every_gpu_grader_records_when_it_last_produced_a_number():
    """"Never ran" and "ran and passed" must not be the same colour.

    Both GPU graders skip on a host without an accelerator, and a skip is
    green, so their result carries no information about whether either has
    ever run. This receipt carries it instead. The check is that the receipt
    enumerates exactly the gpu-marked graders in this module, so adding one
    without a receipt entry fails here rather than being silently unobserved.

    Deliberately not a staleness assertion. An age threshold would fail in
    hosted CI, which has no accelerator and can never satisfy it, for a reason
    no author of any pull request could fix - and it would fire on changes
    that have nothing to do with the graders. What a reviewer needs is the
    date and the commit, stated; not a red check about the calendar.
    """
    document = json.loads(GPU_GRADER_OBSERVATIONS.read_text(encoding="utf-8"))
    assert document["api_version"] == "trainvm.gpu-grader-observations/v1"
    recorded = document["graders"]
    assert set(recorded) == gpu_marked_graders_in_this_module(), (
        "the observation receipt and the gpu-marked graders have drifted "
        "apart; every gpu grader needs an entry, even an unobserved one")
    for grader, entry in recorded.items():
        observed = entry["last_observed"]
        if observed is None:
            # A grader that has never produced a number is a legitimate
            # recorded state. It just must not be able to hide.
            assert set(entry) == {"last_observed"}, grader
            continue
        assert observed.endswith("Z") and len(observed) == 20, grader
        datetime.datetime.strptime(observed, "%Y-%m-%dT%H:%M:%SZ")
        assert entry["commit"] and entry["host"], grader
        assert entry["accelerator_device_name"], grader
        assert entry["fixture"], grader
        measurement = entry["measurement"]
        assert measurement["steady_state_step_seconds"] > 0, grader
        assert isinstance(measurement["qualified"], bool), grader


def measured_conditions(compute_apps: str, *device_samples: str) -> dict:
    """Parsed conditions carrying the ok status a live query would set."""
    conditions = benchmark_runner.parse_accelerator_conditions(
        compute_apps, device_samples[0], device_samples[1:])
    assert conditions is not None
    return {"status": benchmark_runner.NVIDIA_SMI_OK, **conditions}


def test_idle_desktop_residency_does_not_count_as_contention():
    """The exact host state this guard used to refuse, and why it must not.

    Measured 2026-08-09: an idle logged-in workstation holds 1057 MiB of
    compute residency (Wayland compositor 491, an inference daemon 550, a
    Steam helper 16) at 0-3% utilisation. Running this repository's own
    accelerator workload against 41.7 GiB of resident-but-quiet memory moved
    its median step by 1.5%, inside the idle sample spread. Residency is not
    what the benchmark is sensitive to, so it must not be what bounds it.
    """
    conditions = measured_conditions(
        "7432, 491\n1723668, 550\n3387531, 16\n",
        "97887, 6452, 0\n",
    )
    assert conditions["resident_process_memory_mib"] == 1057
    assert conditions["device_memory_free_mib"] == 97887 - 6452
    verdict, explanation = benchmark_runner.classify_accelerator_conditions(
        conditions)
    assert verdict == benchmark_runner.ACCELERATOR_AVAILABLE, explanation

    # Forty times the retired 1024 MiB allowance, still quiet, still runnable.
    heavily_resident = measured_conditions(
        "7432, 491\n1723668, 550\n999999, 42691\n",
        "97887, 48000, 1\n",
    )
    assert benchmark_runner.classify_accelerator_conditions(
        heavily_resident)[0] == benchmark_runner.ACCELERATOR_AVAILABLE


def test_small_process_saturating_compute_is_refused():
    """The false negative the retired residency bound admitted.

    Measured 2026-08-09: a process holding 900 MiB — inside the old 1024 MiB
    allowance on a headless host — while saturating compute slowed the
    accelerator workload's median step by 1.40x. A residency bound would have
    graded that device. A utilisation bound refuses it.
    """
    conditions = measured_conditions("908461, 900\n", "97887, 1814, 100\n")
    assert conditions["resident_process_memory_mib"] == 900
    verdict, explanation = benchmark_runner.classify_accelerator_conditions(
        conditions)
    assert verdict == benchmark_runner.ACCELERATOR_CONTENDED
    assert "busy" in explanation and "100%" in explanation


def test_the_three_situations_are_reported_as_three_situations():
    """No GPU, a busy GPU, and an idle GPU short of memory are not one thing."""
    absent = benchmark_runner.classify_accelerator_conditions(
        {"status": benchmark_runner.NVIDIA_SMI_ABSENT})
    assert absent[0] == benchmark_runner.ACCELERATOR_ABSENT
    assert "no accelerator on this host" in absent[1]
    assert "waiting will not change it" in absent[1]

    busy = benchmark_runner.classify_accelerator_conditions(
        measured_conditions("4404, 60000\n", "97887, 64000, 96\n"))
    assert busy[0] == benchmark_runner.ACCELERATOR_CONTENDED
    assert "utilisation 96%" in busy[1]

    starved = benchmark_runner.classify_accelerator_conditions(
        measured_conditions("4404, 96000\n", "97887, 97000, 0\n"))
    assert starved[0] == benchmark_runner.ACCELERATOR_MEMORY_EXHAUSTED
    assert "idle" in starved[1]
    assert "887 MiB of 97887 MiB is free" in starved[1]

    # The three explanations must not be interchangeable prose.
    assert len({absent[1], busy[1], starved[1]}) == 3


def test_unparseable_and_absent_telemetry_are_told_apart():
    """Both refuse, but only one of them is a broken driver."""
    assert benchmark_runner.parse_accelerator_conditions(
        "not-a-pid, unknown\n", "97887, 6912, 5\n") is None
    unavailable = benchmark_runner.classify_accelerator_conditions(None)
    assert unavailable[0] == benchmark_runner.ACCELERATOR_TELEMETRY_UNAVAILABLE

    failed = benchmark_runner.classify_accelerator_conditions(
        {"status": benchmark_runner.NVIDIA_SMI_FAILED})
    assert failed[0] == benchmark_runner.ACCELERATOR_TELEMETRY_UNAVAILABLE
    assert "treating the device as busy" in failed[1]

    # Missing or nonsensical fields still fail closed rather than being read.
    for broken in ({"status": benchmark_runner.NVIDIA_SMI_OK},
                   {"status": benchmark_runner.NVIDIA_SMI_OK,
                    "device_utilization_percent": True,
                    "device_memory_free_mib": 4096,
                    "device_memory_total_mib": 97887,
                    "resident_process_memory_mib": 0,
                    "resident_processes": []}):
        assert benchmark_runner.classify_accelerator_conditions(broken)[0] == (
            benchmark_runner.ACCELERATOR_TELEMETRY_UNAVAILABLE)


def test_utilisation_is_the_maximum_over_samples():
    """A quiet instant inside a busy run must not read as a free device."""
    conditions = measured_conditions(
        "4404, 4096\n",
        "97887, 8192, 0\n",
        "97887, 8192, 99\n",
        "97887, 8192, 0\n",
    )
    assert conditions["device_utilization_percent"] == 99
    assert conditions["utilization_samples"] == [[0], [99], [0]]
    assert benchmark_runner.classify_accelerator_conditions(
        conditions)[0] == benchmark_runner.ACCELERATOR_CONTENDED


def test_free_memory_is_the_least_free_device_not_the_sum():
    """One busy device on a multi-GPU host must not be hidden by a free one."""
    conditions = measured_conditions(
        "4404, 40000\n",
        "40960, 40000, 0\n40960, 100, 0\n",
    )
    assert conditions["device_memory_free_mib"] == 960
    assert benchmark_runner.classify_accelerator_conditions(
        conditions)[0] == benchmark_runner.ACCELERATOR_MEMORY_EXHAUSTED


def test_residency_allowance_is_opt_in_and_only_ever_tightens():
    """Kept for an operator who wants strict idle; never the default."""
    conditions = measured_conditions(
        "7432, 491\n1723668, 550\n3387531, 16\n", "97887, 6452, 0\n")
    assert benchmark_runner.classify_accelerator_conditions(
        conditions)[0] == benchmark_runner.ACCELERATOR_AVAILABLE
    assert benchmark_runner.classify_accelerator_conditions(
        conditions, resident_memory_allowance_mib=1024)[0] == (
            benchmark_runner.ACCELERATOR_RESIDENCY_OVER_ALLOWANCE)
    strict = benchmark_runner.classify_accelerator_conditions(
        conditions, resident_memory_allowance_mib=0)
    assert strict[0] == benchmark_runner.ACCELERATOR_RESIDENCY_OVER_ALLOWANCE
    assert "strict idle" in strict[1]
    # A generous allowance cannot admit a device the utilisation bound refuses.
    busy = measured_conditions("908461, 900\n", "97887, 1814, 100\n")
    assert benchmark_runner.classify_accelerator_conditions(
        busy, resident_memory_allowance_mib=99999)[0] == (
            benchmark_runner.ACCELERATOR_CONTENDED)


def test_no_flag_can_grade_a_contended_device():
    """There must be no way to spend an argument and measure contention.

    The guard is a refusal, not a warning. Every accelerator-facing argument
    the parser accepts is checked here: none of them may turn a contended
    device into a graded one.
    """
    source = RUNNER.read_text(encoding="utf-8")
    for forbidden in ("--force", "--ignore-contention", "--skip-guard",
                      "--no-accelerator-guard"):
        assert forbidden not in source, (
            f"{forbidden} would grade a contended device")
    busy = measured_conditions("908461, 900\n", "97887, 1814, 100\n")
    for utilization_allowance in (0, 10, 50, 99, 100, 1000):
        for free_memory in (0, 2048, 99999):
            for residency in (None, 0, 1024, 10 ** 9):
                verdict, _ = benchmark_runner.classify_accelerator_conditions(
                    busy, utilization_allowance, free_memory, residency)
                if utilization_allowance >= 100:
                    # Only an allowance that admits a fully saturated device
                    # can pass it, which is a setting that says what it does.
                    continue
                assert verdict != benchmark_runner.ACCELERATOR_AVAILABLE


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
        lambda *_arguments, **_keywords: {
            "status": benchmark_runner.NVIDIA_SMI_OK,
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
            "device_memory_free_mib": 97887,
            "device_utilization_percent": 0,
            "utilization_samples": [[0]],
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
def test_compiled_candidate_falls_back_cleanly_on_unseen_shape(tmp_path):
    environment = {
        **os.environ,
        "PYTHONPATH": str(REPOSITORY / "src"),
        "CUDA_VISIBLE_DEVICES": "",
        "TORCHINDUCTOR_CACHE_DIR": str(tmp_path / "inductor-cache"),
    }

    def run_workload(*arguments):
        return subprocess.run(
            [sys.executable, str(PORTABLE_WORKLOAD), *arguments],
            capture_output=True,
            text=True,
            cwd=REPOSITORY,
            env=environment,
            check=False,
            timeout=300,
        )

    common = [
        "--phase", "timed",
        "--bucket", "seq8xbatch2",
        "--fallback-bucket", "seq11xbatch1",
        "--seed", "0",
        "--steps", "2",
    ]
    eager = run_workload(*common)
    compiled = run_workload(*common, "--compile")
    assert eager.returncode == 0, eager.stdout + eager.stderr
    assert compiled.returncode == 0, compiled.stdout + compiled.stderr

    eager_report = json.loads(eager.stdout)
    compiled_report = json.loads(compiled.stdout)
    fallback = compiled_report["fallback"]
    assert fallback["bucket"] == "seq11xbatch1"
    assert fallback["step_seconds"] > 0.0
    assert benchmark_runner.compare_fingerprints(
        eager_report["result_fingerprint"],
        compiled_report["result_fingerprint"],
    )["output_parity"] is True
    fallback_parity = benchmark_runner.compare_fingerprints(
        eager_report["fallback"]["result_fingerprint"],
        fallback["result_fingerprint"],
    )
    assert fallback_parity["output_parity"] is True
    assert fallback_parity["gradient_parity"] is True


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
        # The cell must carry the conditions it was measured under, and they
        # must be the conditions the guard actually admitted it on.
        assert conditions["contention_verdict"] == (
            benchmark_runner.ACCELERATOR_AVAILABLE)
        assert conditions["device_utilization_percent"] <= conditions[
            "utilization_allowance_percent"]
        assert conditions["device_memory_free_mib"] >= conditions[
            "free_memory_requirement_mib"]
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

    record_gpu_grader_observation(
        "test_accelerator_fixture_emits_evidence_the_gate_accepts",
        ACCELERATOR_FIXTURE,
        report["cells"][0]["accelerator_device_name"],
        {
            "steady_state_step_seconds": report["cells"][0][
                "steady_state_step_seconds"],
            "qualified": True,
        },
    )


@pytest.mark.slow
@pytest.mark.gpu
def test_compiled_accelerator_candidate_is_judged_by_authority(tmp_path):
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

    evidence = tmp_path / "compiled-evidence.json"
    receipt = tmp_path / "compiled-receipt.json"
    result = run_runner(
        "--fixture", ACCELERATOR_FIXTURE,
        "--candidate", "compile",
        "--allow-accelerator",
        "--seeds", "1",
        "--steps", "4",
        "--evidence", str(evidence),
        "--receipt", str(receipt),
    )
    assert result.returncode == 0, result.stdout + result.stderr

    report = json.loads(receipt.read_text())
    assert report["candidate"] == "compile"
    assert report["completed_cells"] == report["required_cells"] >= 1
    assert not report["failed_cells"]
    for cell in report["cells"]:
        assert cell["candidate"]["compiled"] is True
        assert cell["candidate_cold_compile_seconds"] > 0.0
        assert cell["baseline_warm_steps_per_second"] > 0.0
        assert cell["candidate_warm_steps_per_second"] > 0.0
        assert cell["output_relative_deviation"] >= 0.0
        assert cell["gradient_relative_deviation"] >= 0.0

    binary = find_trainvm()
    verdict = subprocess.run(
        [binary, "qualify-evidence"],
        input=evidence.read_text(),
        capture_output=True,
        text=True,
        check=False,
    )
    assert verdict.returncode in (0, 3), verdict.stdout + verdict.stderr
    authority_receipt = json.loads(verdict.stdout)
    assert authority_receipt["evidence"] == json.loads(evidence.read_text())
    assert authority_receipt["qualified"] is (verdict.returncode == 0)

    record_gpu_grader_observation(
        "test_compiled_accelerator_candidate_is_judged_by_authority",
        ACCELERATOR_FIXTURE,
        report["cells"][0]["accelerator_device_name"],
        {
            "steady_state_step_seconds": report["cells"][0][
                "steady_state_step_seconds"],
            "qualified": authority_receipt["qualified"],
        },
    )


def test_ao3_document_selection_is_deterministic(tmp_path):
    corpus_root = tmp_path / "corpus"
    corpus_root.mkdir()
    index = tmp_path / "ao3.sqlite3"
    with sqlite3.connect(index) as connection:
        connection.execute(
            "CREATE TABLE final_selection ("
            "path TEXT, file_present INTEGER, source TEXT)"
        )
        for number in range(16):
            relative = f"ao3_17/document-{number:02d}.txt"
            document = corpus_root / relative
            document.parent.mkdir(exist_ok=True)
            document.write_text(
                f"real deterministic document {number}\n", encoding="utf-8")
            connection.execute(
                "INSERT INTO final_selection(path, file_present, source) "
                "VALUES (?, 1, 'old17')",
                (relative,),
            )

    first = ao3_workload.select_documents(
        index, corpus_root, seed=29, count=6, minimum_document_bytes=1)
    second = ao3_workload.select_documents(
        index, corpus_root, seed=29, count=6, minimum_document_bytes=1)

    assert [document.relative_path for document in first] == [
        document.relative_path for document in second]
    assert [document.rowid for document in first] == [
        document.rowid for document in second]


@pytest.mark.slow
def test_ao3_workload_refuses_an_absent_corpus_without_fallback(tmp_path):
    environment = {
        **os.environ,
        "PYTHONPATH": str(REPOSITORY / "src"),
        "CUDA_VISIBLE_DEVICES": "",
        "MOE_BENCHMARK_AO3_INDEX": str(tmp_path / "missing.sqlite3"),
        "MOE_BENCHMARK_AO3_ROOT": str(tmp_path / "missing-corpus"),
        # Selection must fail before tokenizer construction; an existing file
        # keeps this test specifically about the absent corpus path.
        "MOE_BENCHMARK_RWKV_VOCAB": str(PORTABLE_WORKLOAD),
    }
    result = subprocess.run(
        [
            sys.executable,
            str(AO3_CORPUS_WORKLOAD),
            "--phase", "timed",
            "--bucket", "seq8xbatch1",
            "--seed", "0",
            "--steps", "1",
        ],
        capture_output=True,
        text=True,
        cwd=REPOSITORY,
        env=environment,
        check=False,
        timeout=30,
    )
    assert result.returncode != 0
    assert "AO3 corpus index does not exist" in result.stderr
    assert "no synthetic fallback is available" in result.stderr
    assert "synthetic_in_process" not in result.stdout


@pytest.mark.slow
def test_ao3_fixture_carries_real_input_measurements_into_receipt(tmp_path):
    required = (
        ao3_workload.DEFAULT_CORPUS_INDEX,
        ao3_workload.DEFAULT_CORPUS_ROOT,
        ao3_workload.DEFAULT_TOKENIZER_VOCAB,
    )
    # Both skips below name what was missing. They used to say only "host AO3
    # corpus or RWKV tokenizer is not available" and "ztok is not importable",
    # which is the same silence test_world_vocab.py was fixed for: a dependency
    # on another repository failing open, with nothing in the output saying
    # which dependency or where it was looked for. Unlike the parity assertion
    # this one cannot be vendored away -- it needs the multi-gigabyte host AO3
    # corpus and ztok's compiled Python binding, and there is deliberately no
    # synthetic fallback -- so naming the gap is the whole of the fix here.
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        pytest.skip("host AO3 corpus or RWKV tokenizer is not available: "
                    + ", ".join(missing))
    if importlib.util.find_spec("ztok") is None:
        pytest.skip("ztok's Python binding is not importable "
                    "(it lives in the ztok repository, not this checkout); "
                    + ztok_binary.python_binding_report_line())

    receipt = tmp_path / "ao3-receipt.json"
    result = run_runner(
        "--fixture", AO3_CORPUS_FIXTURE,
        "--seeds", "1",
        "--steps", "1",
        "--receipt", str(receipt),
    )
    assert result.returncode == 0, result.stdout + result.stderr

    report = json.loads(receipt.read_text())
    assert report["completed_cells"] == report["required_cells"] == 3
    assert not report["failed_cells"]
    for cell in report["cells"]:
        assert cell["input_pipeline"] == ao3_workload.INPUT_PIPELINE
        assert cell["input_wait_seconds"] > 0.0
        assert 0.0 < cell["input_wait_ratio"] < 1.0
        assert cell["documents_read"] == cell["batch_size"]
        assert cell["corpus_bytes_read"] >= (
            cell["documents_read"]
            * ao3_workload.MINIMUM_DOCUMENT_BYTES
        )
        assert cell["corpus_bytes_read"] <= (
            cell["documents_read"]
            * ao3_workload.MAXIMUM_DOCUMENT_BYTES
        )
        assert cell["tokens_encoded"] > 0
        assert cell["document_selection_digest"].startswith("sha256:")


def test_every_workload_declares_its_input_pipeline_source():
    """An input-wait number is meaningless until it says what it measured.

    The portable and accelerator workloads synthesize tensors in process. The
    AO3 workload reads, decodes, and tokenizes real documents. Naming each
    source is what stops a later input-pipeline optimization from claiming a
    gain against a number that describes something else, so declarations are
    pinned per workload rather than globally.
    """
    workloads = sorted(
        (REPOSITORY / "scripts" / "benchmark_workloads").glob("*_lm_step.py"))
    assert workloads, "no benchmark workloads found"
    expected = {
        "accelerator_lm_step.py": "synthetic_in_process",
        "ao3_corpus_lm_step.py": "ao3_raw_utf8_decode_ztok",
        "portable_lm_step.py": "synthetic_in_process",
    }
    assert {workload.name for workload in workloads} == set(expected)
    for workload in workloads:
        source = workload.read_text(encoding="utf-8")
        assert '"input_pipeline"' in source, (
            f"{workload.name} reports input wait without declaring its source")
        assert f'"{expected[workload.name]}"' in source, (
            f"{workload.name} must declare its actual input source")
