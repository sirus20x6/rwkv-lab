"""A job that exists to execute assertions must fail when it executes none.

`scripts/ci_marked_suite_ran.py` exists because the `fla dependency tests` job
reported `1 skipped, 2154 deselected` -- zero passed -- and was green. Its one
test took `pytest.importorskip("fla.models.utils")`, which failed because
`fla-core` declares only torch and einops while the import needs triton.

The test's own safeguard was real but watched the wrong thing: "pytest exits 5
if the marker disappears" covers *collection*, and the test was collected
normally and then skipped at *runtime*. pytest exits 0 for a run that skips
everything.

Tests are arranged by mechanism:

    all-skipped fails          the run that produced the card
    a passing run passes       the gate must not simply be red forever
    an errored run fails       "not skipped" is not the same as "ran"; asserting
                               on passed is what separates them
    a missing report fails     pytest not reaching the point of writing one is a
                               harder failure than an empty run, not a lesser one
    --at-least is honoured     the bound is a real parameter, not decoration
    the tally is verdict-neutral   the last line must not read as a pass
    a flat <testsuite> root parses  pytest emits both shapes
"""

from __future__ import annotations

import pathlib
import subprocess
import sys

import pytest

REPOSITORY = pathlib.Path(__file__).resolve().parents[1]
GATE = REPOSITORY / "scripts/ci_marked_suite_ran.py"


def report(path: pathlib.Path, *, tests: int, skipped: int = 0,
           failures: int = 0, errors: int = 0, flat: bool = False) -> pathlib.Path:
    suite = (f'<testsuite name="pytest" tests="{tests}" failures="{failures}" '
             f'errors="{errors}" skipped="{skipped}"></testsuite>')
    path.write_text(suite if flat else f"<testsuites>{suite}</testsuites>",
                    encoding="utf-8")
    return path


def run(path: pathlib.Path, *arguments: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(GATE), str(path), *arguments],
        capture_output=True, text=True, check=False)


def test_a_run_that_skipped_everything_fails(tmp_path):
    """The exact shape of the run that produced card-a702bb37."""
    result = run(report(tmp_path / "r.xml", tests=1, skipped=1))
    assert result.returncode == 1
    assert "0 test(s) passed" in result.stdout
    assert "FAILED" in result.stdout


def test_a_run_that_passed_is_green(tmp_path):
    """A gate that cannot go green is a permanently red job, which gets ignored."""
    result = run(report(tmp_path / "r.xml", tests=1))
    assert result.returncode == 0, result.stdout
    assert "PASSED" in result.stdout


def test_an_errored_run_fails_too(tmp_path):
    """An errored test is *not skipped*, and must still not read as a run.

    This is why the gate asserts on passed rather than on `skipped == 0`. The
    weaker rule would call a job that errored on every test a success.
    """
    result = run(report(tmp_path / "r.xml", tests=1, errors=1))
    assert result.returncode == 1
    assert "0 test(s) passed" in result.stdout


def test_a_failing_run_is_not_counted_as_having_run(tmp_path):
    result = run(report(tmp_path / "r.xml", tests=1, failures=1))
    assert result.returncode == 1


def test_a_missing_report_fails_rather_than_being_skipped(tmp_path):
    """pytest not reaching the point of writing a report is the harder failure."""
    result = run(tmp_path / "absent.xml")
    assert result.returncode == 1
    assert "no JUnit report" in result.stdout


@pytest.mark.parametrize("passed,bound,expected", [(1, 1, 0), (1, 2, 1), (3, 2, 0)])
def test_the_bound_is_honoured(tmp_path, passed, bound, expected):
    """--at-least is a real parameter.

    Asserted across the boundary in both directions so a gate hardcoding 1
    internally, and ignoring the argument, cannot pass this file.
    """
    result = run(report(tmp_path / "r.xml", tests=passed), "--at-least", str(bound))
    assert result.returncode == expected, result.stdout


def test_the_tally_reads_true_on_a_failing_run(tmp_path):
    """verdict_line supplies PASSED/FAILED, so the detail must be neutral."""
    verdict = run(report(tmp_path / "r.xml", tests=1, skipped=1)).stdout.strip().splitlines()[-1]
    assert "FAILED" in verdict
    assert "1 or more had to pass" in verdict


def test_a_flat_testsuite_root_is_parsed(tmp_path):
    """pytest emits <testsuite> as the root in some versions, <testsuites> in others.

    A reader that handles only one silently measures zero tests for the other,
    which would make this gate fail every run for the wrong reason -- and a
    gate that is wrong most of the time gets bypassed.
    """
    result = run(report(tmp_path / "r.xml", tests=1, flat=True))
    assert result.returncode == 0, result.stdout


def test_the_label_appears_in_the_failure(tmp_path):
    """The message has to say which job asserted nothing, not just that one did."""
    result = run(report(tmp_path / "r.xml", tests=1, skipped=1),
                 "--label", "the fla dependency job")
    assert "the fla dependency job" in result.stdout
