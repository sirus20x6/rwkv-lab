"""The benchmark matrix validator must reject the drifts it exists to catch.

Without these, the validator could pass vacuously and nobody would notice: a
gate that never fires looks identical to a document that is always correct.
"""

from __future__ import annotations

import json
import pathlib
import subprocess
import sys

import pytest

REPOSITORY = pathlib.Path(__file__).resolve().parents[1]
VALIDATOR = REPOSITORY / "scripts/validate_benchmark_matrix.py"
MATRIX = REPOSITORY / "docs/experiment-vm/benchmark-matrix.v1.json"


def run_validator(path: pathlib.Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(VALIDATOR), str(path)],
        capture_output=True, text=True, cwd=REPOSITORY, check=False,
    )


@pytest.fixture
def matrix() -> dict:
    return json.loads(MATRIX.read_text())


def write(tmp_path: pathlib.Path, document: dict) -> pathlib.Path:
    path = tmp_path / "benchmark-matrix.json"
    path.write_text(json.dumps(document))
    return path


def test_checked_in_matrix_is_valid():
    result = run_validator(MATRIX)
    assert result.returncode == 0, result.stdout + result.stderr


def test_serving_fixture_cannot_claim_gradient_parity(tmp_path, matrix):
    for fixture in matrix["fixtures"]:
        if fixture["effect_class"] == "serving_kernel":
            fixture["parity_evidence"].append("gradient_parity")
    result = run_validator(write(tmp_path, matrix))
    assert result.returncode == 1
    assert "which serving_kernel cannot" in result.stdout


def test_training_fixture_cannot_drop_required_parity(tmp_path, matrix):
    for fixture in matrix["fixtures"]:
        if fixture["effect_class"] == "training_kernel":
            fixture["parity_evidence"].remove("resumed_trajectory_parity")
            break
    result = run_validator(write(tmp_path, matrix))
    assert result.returncode == 1
    assert "omits required" in result.stdout


def test_portable_fixture_cannot_require_an_accelerator(tmp_path, matrix):
    for fixture in matrix["fixtures"]:
        if fixture["portability"] == "portable":
            fixture["accelerator_required"] = True
            break
    result = run_validator(write(tmp_path, matrix))
    assert result.returncode == 1
    assert "cannot require an accelerator" in result.stdout


def test_every_required_family_must_be_covered(tmp_path, matrix):
    matrix["fixtures"] = [
        fixture for fixture in matrix["fixtures"]
        if fixture["family"] != "distillation"
    ]
    result = run_validator(write(tmp_path, matrix))
    assert result.returncode == 1
    assert "distillation" in result.stdout


def test_fixture_cannot_carry_executable_authority(tmp_path, matrix):
    matrix["fixtures"][0]["argv"] = ["python", "-m", "anything"]
    result = run_validator(write(tmp_path, matrix))
    assert result.returncode == 1
    assert "reserved key" in result.stdout


def test_fixture_cannot_carry_a_measured_result(tmp_path, matrix):
    matrix["fixtures"][0]["measured_step_time"] = 0.25
    result = run_validator(write(tmp_path, matrix))
    assert result.returncode == 1
    assert "reserved key" in result.stdout


def test_matrix_must_exercise_a_curriculum_transition(tmp_path, matrix):
    for fixture in matrix["fixtures"]:
        fixture["declares_curriculum_transition"] = False
    result = run_validator(write(tmp_path, matrix))
    assert result.returncode == 1
    assert "curriculum-stage transition" in result.stdout


def test_quality_gate_is_required_before_a_speed_claim(tmp_path, matrix):
    matrix["fixtures"][0]["quality_gate"] = ""
    result = run_validator(write(tmp_path, matrix))
    assert result.returncode == 1
    assert "quality gate is required" in result.stdout
