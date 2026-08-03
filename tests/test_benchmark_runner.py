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
import os
import pathlib
import shutil
import sqlite3
import subprocess
import sys

import pytest

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
    if not all(path.exists() for path in required):
        pytest.skip("host AO3 corpus or RWKV tokenizer is not available")
    if importlib.util.find_spec("ztok") is None:
        pytest.skip("ztok is not importable")

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
