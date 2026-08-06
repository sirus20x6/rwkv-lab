"""The no-code authoring release matrix must fail closed as it evolves."""

from __future__ import annotations

import copy
import json
import pathlib
import subprocess
import sys

import pytest

REPOSITORY = pathlib.Path(__file__).resolve().parents[1]
VALIDATOR = REPOSITORY / "scripts/validate_no_code_authoring_matrix.py"
MATRIX = (
    REPOSITORY
    / "docs"
    / "experiment-vm"
    / "no-code-authoring-matrix.v1.json"
)


def run_validator(path: pathlib.Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(VALIDATOR), str(path)],
        capture_output=True,
        text=True,
        cwd=REPOSITORY,
        check=False,
    )


@pytest.fixture
def matrix() -> dict:
    return json.loads(MATRIX.read_text(encoding="utf-8"))


def write(tmp_path: pathlib.Path, document: dict) -> pathlib.Path:
    path = tmp_path / "no-code-authoring-matrix.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def scenario(document: dict, family: str) -> dict:
    return next(item for item in document["scenarios"] if item["family"] == family)


def test_checked_in_matrix_is_valid():
    result = run_validator(MATRIX)
    assert result.returncode == 0, result.stdout + result.stderr


def test_every_required_family_must_remain_covered(tmp_path, matrix):
    matrix["scenarios"] = [
        item for item in matrix["scenarios"] if item["family"] != "mageflow"
    ]
    result = run_validator(write(tmp_path, matrix))
    assert result.returncode == 1
    assert "missing required families" in result.stdout


def test_qwen_must_vary_every_expensive_authoring_axis(tmp_path, matrix):
    qwen = scenario(matrix, "hf_multimodal_sft")
    qwen["variation_axes"].remove("lora_rank")
    qwen["variants"] = [
        item for item in qwen["variants"] if item["axis"] != "lora_rank"
    ]
    result = run_validator(write(tmp_path, matrix))
    assert result.returncode == 1
    assert "lora_rank" in result.stdout


def test_declared_variant_must_actually_override_configuration(tmp_path, matrix):
    variant = scenario(matrix, "rwkv_lm")["variants"][0]
    variant["override_fields"] = []
    result = run_validator(write(tmp_path, matrix))
    assert result.returncode == 1
    assert "override_fields must be non-empty" in result.stdout


def test_matrix_cannot_carry_executable_authority(tmp_path, matrix):
    scenario(matrix, "transformer_lm")["command"] = ["python", "train.py"]
    result = run_validator(write(tmp_path, matrix))
    assert result.returncode == 1
    assert "reserved key 'command'" in result.stdout


def test_matrix_cannot_carry_a_measured_result(tmp_path, matrix):
    scenario(matrix, "rwkv_lm")["measured_authoring_seconds"] = 3.0
    result = run_validator(write(tmp_path, matrix))
    assert result.returncode == 1
    assert "reserved key 'measured_authoring_seconds'" in result.stdout


@pytest.mark.parametrize(
    "mutation",
    [
        "repository_source",
        "adapter_registry",
        "handler_dispatch",
        "sealed_worker_deployment",
        "dashboard_source",
        "service_configuration",
    ],
)
def test_every_code_build_and_deploy_mutation_is_forbidden(
    tmp_path, matrix, mutation
):
    modified = copy.deepcopy(matrix)
    modified["forbidden_mutations"].remove(mutation)
    result = run_validator(write(tmp_path, modified))
    assert result.returncode == 1
    assert mutation in result.stdout


def test_required_evidence_cannot_drop_immediate_dashboard_visibility(
    tmp_path, matrix
):
    matrix["required_evidence"].remove("immediate_dashboard_registration")
    result = run_validator(write(tmp_path, matrix))
    assert result.returncode == 1
    assert "immediate_dashboard_registration" in result.stdout


def test_unsupported_request_must_fail_at_an_explicit_boundary(tmp_path, matrix):
    scenario(matrix, "mageflow")["unsupported_request_disposition"] = "ignored"
    result = run_validator(write(tmp_path, matrix))
    assert result.returncode == 1
    assert "new_implementation_required" in result.stdout
