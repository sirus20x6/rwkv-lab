"""The coverage gate's two halves must agree on what a test file is.

`scripts/ci_coverage_gate.py` catches a test file that contributes nothing to
the marker expression CI actually runs. It does that by comparing two
populations, and they were derived differently: the node-id regex accepts
nested `tests/**` paths, while the on-disk glob was non-recursive.

That disagreement is silent **in the flattering direction**. A file only the
regex could see lands in `covered` and never in `on_disk`, so the verdict line
prints a covered total larger than the population it is measured against —
which reads as better coverage than the tree has, from the gate whose whole
purpose is to notice missing coverage.

The tests are arranged around what the two halves must agree on:

    a nested file is counted        the glob recurses
    a nested file is DEMANDED       recursing means nested tests are gated too,
                                    which is the half narrowing the regex would
                                    have lost
    covered beyond on_disk fails    the invariant a reader otherwise checks by
                                    eye against the verdict line
    the verdict line carries it     the count is the visible symptom

Every case builds a directory literally named `tests` and runs from its parent:
the gate's node-id regex is anchored on that prefix, so a differently-named
fixture directory would collect nothing and every test here would pass while
measuring nothing.
"""

from __future__ import annotations

import pathlib
import subprocess
import sys

import pytest

REPOSITORY = pathlib.Path(__file__).resolve().parents[1]
GATE = REPOSITORY / "scripts/ci_coverage_gate.py"

A_REAL_TEST = "def test_contributes():\n    assert True\n"
NO_TESTS = "VALUE = 1  # imports cleanly, collects nothing\n"


@pytest.fixture
def tree(tmp_path: pathlib.Path) -> pathlib.Path:
    (tmp_path / "tests").mkdir()
    return tmp_path


def write(tree: pathlib.Path, relative: str, body: str) -> None:
    path = tree / "tests" / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")


def run(tree: pathlib.Path, *arguments: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(GATE), "--tests-dir", "tests", *arguments],
        cwd=tree, capture_output=True, text=True, check=False)


def verdict(stdout: str) -> str:
    return stdout.strip().splitlines()[-1]


def test_a_flat_contributing_file_passes(tree):
    write(tree, "test_flat.py", A_REAL_TEST)
    result = run(tree)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "1 covered" in verdict(result.stdout)
    assert "1 test files on disk" in verdict(result.stdout)


def test_a_nested_contributing_file_is_counted_on_disk(tree):
    """The defect: the regex saw it, the glob did not.

    Before recursing, this printed `1 covered ... 0 test files on disk` — a
    covered total larger than the population, from a passing gate.
    """
    write(tree, "native/test_nested.py", A_REAL_TEST)
    result = run(tree)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "1 covered" in verdict(result.stdout)
    assert "1 test files on disk" in verdict(result.stdout), (
        "a nested test file contributed tests but was not counted on disk")


def test_a_nested_file_contributing_nothing_is_demanded(tree):
    """This is what recursing buys, and what narrowing the regex would lose.

    Narrowing would have made a nested file invisible to the gate: neither
    counted as covered nor required to contribute. A whole nested directory
    could then contribute nothing and the gate would never say so — the
    failure it exists to catch, relocated rather than fixed.
    """
    write(tree, "test_flat.py", A_REAL_TEST)
    write(tree, "native/test_silent.py", NO_TESTS)
    result = run(tree)
    assert result.returncode == 1
    assert "NO COVERAGE: tests/native/test_silent.py" in result.stdout


def test_a_covered_file_the_gate_does_not_count_fails(tree):
    """`covered` must never exceed the population it is measured against.

    Triggered the way it would really happen: a project configuring a
    `python_files` pattern the gate's own glob does not match. pytest collects
    the file, the node-id regex accepts it, and the on-disk glob cannot see it.
    """
    (tree / "pytest.ini").write_text(
        "[pytest]\npython_files = check_*.py\n", encoding="utf-8")
    write(tree, "test_flat.py", A_REAL_TEST)  # not collected under this pattern
    write(tree, "check_other.py", A_REAL_TEST)

    result = run(tree)
    assert result.returncode == 1
    assert "UNACCOUNTED: tests/check_other.py" in result.stdout
    assert "larger than the population" in result.stdout


def test_the_verdict_line_reports_the_counts_it_compared(tree):
    """The count is the visible symptom, so it is asserted rather than the code."""
    write(tree, "test_a.py", A_REAL_TEST)
    write(tree, "deep/test_b.py", A_REAL_TEST)
    line = verdict(run(tree).stdout)
    assert "2 covered" in line
    assert "2 test files on disk" in line
    assert "PASSED" in line


def test_an_empty_tests_directory_is_not_a_pass_by_accident(tree):
    """Nothing to cover is green, but the numbers must say so plainly.

    Asserted because `0 covered ... 0 test files on disk` and a healthy tree
    must not print the same thing, and because a gate that silently tolerates
    an empty population is the shape this repository keeps finding.
    """
    result = run(tree)
    assert result.returncode == 0, result.stdout + result.stderr
    line = verdict(result.stdout)
    assert "0 covered" in line and "0 test files on disk" in line
