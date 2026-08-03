"""A gate's last line must say which way the gate went.

These assert the property the helper exists for, not its wording: a reader who
sees only the final line, with no context and no colour, cannot mistake a
failing run for a passing one. Asserting the exact string would pin the
phrasing and prove nothing about that property.
"""

from __future__ import annotations

import re
import subprocess
import sys

import pytest

from scripts.gate_verdict import verdict_line

DETAIL = "128 covered, 9 explained exclusions, 138 test files on disk"

# Every gate that uses the helper. If one is added without a line here it keeps
# whatever summary it had, which is the drift this file exists to prevent.
GATES = (
    "scripts/ci_coverage_gate.py",
    "scripts/validate_benchmark_matrix.py",
    "scripts/validate_experiment_documents.py",
    "scripts/validate_native_ci_exclusions.py",
)

# scripts/acceptance.sh is deliberately absent: it already ends in
# "ACCEPTANCE FAILED" or "acceptance passed", and it drives real suites on a
# real host, so running it from a unit test is not appropriate.


def test_the_failing_and_passing_summaries_are_different():
    """The whole point: one line, two outcomes, no ambiguity."""
    passed = verdict_line("coverage gate", [], DETAIL)
    failed = verdict_line("coverage gate", ["a problem"], DETAIL)
    assert passed != failed


def test_the_failing_summary_is_not_a_prefix_or_suffix_of_the_passing_one():
    """Guards against a fix that only appends or prepends a word.

    A truncated or wrapped line must still be distinguishable, so the two forms
    have to diverge early rather than share a long common run.
    """
    passed = verdict_line("coverage gate", [], DETAIL)
    failed = verdict_line("coverage gate", ["a problem"], DETAIL)
    assert not passed.startswith(failed) and not failed.startswith(passed)
    assert not passed.endswith(failed) and not failed.endswith(passed)
    shared = 0
    while shared < min(len(passed), len(failed)) and passed[shared] == failed[shared]:
        shared += 1
    assert shared <= len("coverage gate: "), (
        f"the two forms agree for their first {shared} characters, so a "
        f"reader skimming the start of the line learns nothing")


def test_the_failing_summary_states_a_verdict_and_a_count():
    failed = verdict_line("coverage gate", ["one", "two"], DETAIL)
    assert "FAILED" in failed
    assert "2 problems" in failed


def test_a_single_problem_is_not_reported_in_the_plural():
    assert "1 problem above" in verdict_line("gate", ["only one"], DETAIL)
    assert "1 problems" not in verdict_line("gate", ["only one"], DETAIL)


def test_the_passing_summary_does_not_contain_the_failing_word():
    """A passing line containing FAILED anywhere would defeat a skim or a grep."""
    passed = verdict_line("coverage gate", [], DETAIL)
    assert "FAILED" not in passed
    assert "PASSED" in passed


def test_the_detail_survives_in_both_forms():
    """The tally is still useful; stating a verdict must not cost it."""
    for problems in ([], ["a problem"]):
        assert DETAIL in verdict_line("coverage gate", problems, DETAIL)


@pytest.mark.parametrize("gate", GATES)
def test_the_wired_gate_ends_with_a_verdict_line(gate):
    """End to end, against the real script rather than the helper alone.

    A gate could import the helper and still print something else afterwards;
    this runs it and looks at the line that is actually last. Gates run in
    their passing configuration, so this pins the passing form -- the failing
    form is proven against the helper above, since making a real gate fail here
    would mean corrupting checked-in fixtures.

    Parametrized rather than looped so a gate that cannot run in this
    environment skips on its own instead of taking the others with it.
    """
    completed = subprocess.run(
        [sys.executable, gate],
        capture_output=True, text=True, check=False,
    )
    # Not every job installs every gate's dependencies -- the CPU job has no
    # jsonschema, which only the schema-golden job pip-installs. A gate that
    # dies at import prints nothing, and asserting against that empty output
    # would be an assertion about the environment rather than about the code.
    missing = re.search(r"ModuleNotFoundError: No module named '([^']+)'",
                        completed.stderr)
    if missing and not completed.stdout.strip():
        pytest.skip(f"{gate} needs {missing.group(1)}, absent in this job")

    lines = completed.stdout.strip().splitlines()
    assert lines, (
        f"{gate} printed nothing on stdout (exit {completed.returncode}); "
        f"stderr was: {completed.stderr.strip()[:400]}")
    last = lines[-1]
    assert "PASSED" in last or "FAILED" in last, (
        f"{gate} ends with {last!r}, which states no verdict")
    assert (completed.returncode == 0) == ("PASSED" in last), (
        f"{gate} exited {completed.returncode} but its last line says "
        f"{last!r}; the summary and the exit code disagree")
