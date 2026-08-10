"""A gate's last line must say which way the gate went.

These assert the property the helper exists for, not its wording: a reader who
sees only the final line, with no context and no colour, cannot mistake a
failing run for a passing one. Asserting the exact string would pin the
phrasing and prove nothing about that property.
"""

from __future__ import annotations

import pathlib
import re
import subprocess
import sys

import pytest

from scripts.gate_verdict import verdict_line

REPOSITORY = pathlib.Path(__file__).resolve().parents[1]

DETAIL = "128 covered, 9 explained exclusions, 138 test files on disk"

# Every gate that uses the helper AND runs with no arguments, executed below.
#
# This list used to claim it held "every gate that uses the helper". It held
# four of twelve. The other eight kept whatever summary they had, which is the
# exact drift this file exists to prevent -- and the list could not say so,
# because nothing compared it against the scripts that import verdict_line.
# `test_every_gate_using_the_helper_is_accounted_for` does that now, so an
# unlisted gate fails rather than being quietly uncovered.
GATES = (
    "scripts/ci_catalog_doc_counts_gate.py",
    "scripts/ci_compatibility_pin_gate.py",
    "scripts/ci_contract_caller_gate.py",
    "scripts/ci_coverage_gate.py",
    "scripts/ci_gpu_observation_gate.py",
    "scripts/ci_native_host_path_gate.py",
    "scripts/ci_step_zero_arming_gate.py",
    "scripts/ci_unwired_module_gate.py",
    "scripts/validate_benchmark_matrix.py",
    "scripts/validate_experiment_documents.py",
    "scripts/validate_native_ci_exclusions.py",
    "scripts/validate_no_code_authoring_matrix.py",
)

# Gates the parametrization above cannot run, because they require arguments:
# `[sys.executable, gate]` with none would test argparse's usage exit, not the
# verdict property. Each names the test that asserts the property instead, and
# `test_an_argument_taking_gate_is_asserted_somewhere` checks that file exists
# and mentions a verdict -- so "it is covered elsewhere" cannot become a claim
# nobody checks, which is the failure this whole file is about. That check found
# one such claim false on its first run: nothing in
# tests/test_no_code_authoring_receipt.py asserted a verdict until it did.
#
# Building argv fixtures for these here was considered and rejected: one of them
# (`print_step_zero_arming_pin.py --check`) needs a built trainvm binary, so the
# fixture would either be skipped in the schema job or drag an eight-minute C++26
# build into a unit test. A pointer that is itself checked is the cheaper
# instrument and does not degrade to a skip.
NEEDS_ARGUMENTS = {
    "scripts/ci_skip_reason_gate.py": "tests/test_skip_reason_gate.py",
    "scripts/print_disposition_digests.py": "tests/test_disposition_digests.py",
    "scripts/print_step_zero_arming_pin.py": "tests/test_step_zero_arming_pin.py",
    "scripts/validate_no_code_authoring_receipt.py": "tests/test_no_code_authoring_receipt.py",
}

# scripts/acceptance.sh is deliberately absent: it already ends in
# "ACCEPTANCE FAILED" or "acceptance passed", and it drives real suites on a
# real host, so running it from a unit test is not appropriate.


def scripts_using_the_helper() -> set[str]:
    """Every script that imports verdict_line, read from the tree.

    Derived rather than listed, because a hand-maintained census of a
    hand-maintained list has the same defect one level up.
    """
    found = set()
    for path in sorted((REPOSITORY / "scripts").glob("*.py")):
        if "from scripts.gate_verdict import verdict_line" in path.read_text(
                encoding="utf-8"):
            found.add(f"scripts/{path.name}")
    return found


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
    # Anchored to this checkout rather than to the invocation directory: a
    # bare relative path runs whichever gate the caller's cwd happens to hold,
    # which is the same "graded by a file outside the tree under test" mistake
    # that let a PATH-resolved trainvm judge the benchmark evidence.
    completed = subprocess.run(
        [sys.executable, str(REPOSITORY / gate)],
        capture_output=True, text=True, check=False, cwd=REPOSITORY,
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


def test_every_gate_using_the_helper_is_accounted_for():
    """A new gate must join one list or fail here.

    This is the assertion the old comment stood in for. It claimed the tuple
    held every gate using the helper; it held four of twelve, and nothing
    compared the two, so eight gates were uncovered while the file read as
    exhaustive. That is the same shape as a receipt that describes a different
    commit than the one it is served at.

    Both directions are checked. An unlisted script fails, which is the drift.
    A listed script that no longer imports the helper also fails, so a stale
    entry cannot sit here implying coverage that has moved.
    """
    using = scripts_using_the_helper()
    accounted = set(GATES) | set(NEEDS_ARGUMENTS)

    unlisted = sorted(using - accounted)
    assert not unlisted, (
        f"these scripts import verdict_line and are in neither GATES nor "
        f"NEEDS_ARGUMENTS: {unlisted}. Add to GATES if it runs with no "
        f"arguments; otherwise add to NEEDS_ARGUMENTS naming the test that "
        f"asserts its verdict line.")

    stale = sorted(accounted - using)
    assert not stale, (
        f"these are listed here but no longer import verdict_line: {stale}. "
        f"Remove them; an entry implying coverage that has moved is worse than "
        f"no entry.")


@pytest.mark.parametrize("gate,asserting_test", sorted(NEEDS_ARGUMENTS.items()))
def test_an_argument_taking_gate_is_asserted_somewhere(gate, asserting_test):
    """The pointer in NEEDS_ARGUMENTS has to lead somewhere real.

    "Covered elsewhere" is the kind of claim that survives the thing it names
    being deleted. This does not assert the other file is *correct* -- only
    that it exists and asserts a verdict, which is the part that would rot
    silently, and which was already false for one of these four.
    """
    assert (REPOSITORY / gate).exists(), f"{gate} is listed but does not exist"

    target = REPOSITORY / asserting_test
    assert target.exists(), (
        f"{gate} points at {asserting_test} for its verdict-line assertion, "
        f"and that file does not exist")

    text = target.read_text(encoding="utf-8")
    assert "PASSED" in text or "FAILED" in text, (
        f"{asserting_test} is named as where {gate}'s verdict line is asserted, "
        f"but it mentions neither PASSED nor FAILED")
