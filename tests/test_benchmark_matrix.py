"""The benchmark matrix validator must reject the drifts it exists to catch.

Without these, the validator could pass vacuously and nobody would notice: a
gate that never fires looks identical to a document that is always correct.
"""

from __future__ import annotations

import copy
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


# --- the words the verdict line prints ------------------------------------
#
# Asserted on the message, never on the exit code. The validator is green with
# either spelling of these sentences, so an exit-code test proves nothing about
# them.

sys.path.insert(0, str(REPOSITORY))
from scripts.validate_benchmark_matrix import population_summary  # noqa: E402


def verdict(stdout: str) -> str:
    lines = [line for line in stdout.splitlines()
             if line.startswith("benchmark matrix:")]
    assert len(lines) == 1, f"expected exactly one verdict line:\n{stdout}"
    return lines[0]


def test_the_verdict_line_says_the_transition_count_is_families(matrix):
    """The clause used to sit after "fixtures ... families" and read as fixtures.

    It counts family names, so one fixture and ten fixtures in the same family
    both print the same number. The expected values are read out of the
    document, so adding a fixture moves this test with the change.
    """
    line = verdict(run_validator(MATRIX).stdout)
    families = {fixture["family"] for fixture in matrix["fixtures"]}
    transitions = {
        fixture["family"] for fixture in matrix["fixtures"]
        if fixture.get("declares_curriculum_transition")
    }
    assert f"{len(matrix['fixtures'])} fixtures over {len(families)} families" in line
    assert (f"{len(transitions)} of those families declaring a "
            "curriculum-stage transition") in line, line


def test_the_transition_count_does_not_move_with_fixtures_in_one_family(
    tmp_path, matrix
):
    """The regression proof: this sentence would have caught the original slip.

    A second transition-declaring fixture in a family that already has one adds
    a fixture and no family. Under the old wording the number sat where a
    reader took it for fixtures and did not move; now it says which it is.
    """
    original = verdict(run_validator(MATRIX).stdout)
    twin = copy.deepcopy(next(
        fixture for fixture in matrix["fixtures"]
        if fixture.get("declares_curriculum_transition")))
    twin["id"] = twin["id"] + "-twin"
    matrix["fixtures"].append(twin)

    line = verdict(run_validator(write(tmp_path, matrix)).stdout)
    transitions = len({
        fixture["family"] for fixture in matrix["fixtures"]
        if fixture.get("declares_curriculum_transition")})
    assert f"{len(matrix['fixtures'])} fixtures" in line, line
    assert (f"{transitions} of those families declaring a "
            "curriculum-stage transition") in line, line
    assert original.split("fixtures over", 1)[1] == line.split("fixtures over", 1)[1]


def test_a_fixture_with_no_family_key_is_not_counted_as_a_family(tmp_path, matrix):
    """`families.add(family)` was unconditional, so a missing key counted as one.

    That is a computation defect rather than a wording one: the file already
    knew, subtracting `{None}` again before reporting unreviewed families, so
    the only place `None` ever reached was the count this line prints.
    """
    orphan = copy.deepcopy(matrix["fixtures"][0])
    orphan["id"] = orphan["id"] + "-orphan"
    orphan.pop("family")
    orphan["declares_curriculum_transition"] = False
    matrix["fixtures"].append(orphan)

    families = {fixture["family"] for fixture in matrix["fixtures"]
                if "family" in fixture}
    result = run_validator(write(tmp_path, matrix))
    line = verdict(result.stdout)
    assert result.returncode == 0, result.stdout + result.stderr
    assert f"{len(matrix['fixtures'])} fixtures over {len(families)} families" in line


@pytest.mark.parametrize(
    ("fixtures", "families", "transitions", "expected"),
    [
        (
            0, set(), set(),
            "0 fixtures over 0 families, 0 of those families declaring a "
            "curriculum-stage transition",
        ),
        (
            1, {"rwkv_scratch"}, {"rwkv_scratch"},
            "1 fixture over 1 family, 1 of those families declaring a "
            "curriculum-stage transition",
        ),
        (
            5, {"rwkv_scratch", "serving"}, {"serving"},
            "5 fixtures over 2 families, 1 of those families declaring a "
            "curriculum-stage transition",
        ),
    ],
    ids=["none", "one", "several"],
)
def test_the_verdict_line_reads_correctly_at_every_count(
    fixtures, families, transitions, expected
):
    """Singular and plural, at 0, 1 and n.

    "1 fixtures over 1 families" is the shape of error that makes a reader
    distrust the rest of the line.
    """
    assert population_summary(fixtures, families, transitions) == expected
