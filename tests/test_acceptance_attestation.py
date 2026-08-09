"""The acceptance attestation must record runs honestly, including failed ones.

Four native ctest suites are excluded from the hosted CI job and covered only by
`scripts/acceptance.sh` on a real host. Nothing schedules that script, and its
receipt is git-ignored, so for the life of the repository "acceptance passed
last night" and "acceptance has never been run" were indistinguishable.

`docs/experiment-vm/acceptance-attestations.v1.jsonl` is the durable half of the
fix: one appended line per run. These tests pin the properties that make it
worth having, because each of them is easy to lose in a well-meaning edit.
"""

from __future__ import annotations

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from scripts.append_acceptance_attestation import (  # noqa: E402
    API_VERSION,
    NOT_RUN,
    attestation,
    excluded_suite_results,
    junit_statuses,
)

REPOSITORY = pathlib.Path(__file__).resolve().parent.parent
LOG = REPOSITORY / "docs/experiment-vm/acceptance-attestations.v1.jsonl"
EXCLUSIONS = REPOSITORY / "docs/experiment-vm/native-ci-exclusions.v1.json"
ACCEPTANCE = REPOSITORY / "scripts/acceptance.sh"

REQUIRED_FIELDS = {
    "api_version",
    "ran_at",
    "commit",
    "dirty_worktree",
    "host",
    "result",
    "hosted_ci_excluded_suites",
}


def exclusions() -> dict:
    return json.loads(EXCLUSIONS.read_text(encoding="utf-8"))


def receipt(suites: list[dict], **overrides) -> dict:
    document = {
        "api_version": "trainvm.acceptance/v1",
        "commit": "0" * 40,
        "dirty_worktree": False,
        "generated_at": "2026-08-09T12:00:00Z",
        "suites": suites,
    }
    document.update(overrides)
    return document


def test_the_log_is_tracked_and_every_line_is_a_complete_record():
    """A half-written line is worse than none: it reads as evidence."""
    assert LOG.is_file(), (
        f"{LOG.relative_to(REPOSITORY)} must exist even when empty. An absent "
        "file is indistinguishable from a repository that never had the "
        "mechanism; an empty tracked file says plainly that no run is attested."
    )
    for number, text in enumerate(LOG.read_text(encoding="utf-8").splitlines(), 1):
        if not text.strip():
            continue
        line = json.loads(text)  # one JSON object per line, no exceptions
        assert line["api_version"] == API_VERSION, f"line {number}"
        missing = REQUIRED_FIELDS - set(line)
        assert not missing, f"line {number} is missing {sorted(missing)}"
        assert line["result"] in ("passed", "failed"), f"line {number}"


def test_a_failed_run_is_still_attested():
    """A green-only log makes an absent line ambiguous all over again."""
    line = attestation(
        receipt([
            {"name": "native-ctest", "status": "FAILED", "detail": ""},
            {"name": "go", "status": "passed", "detail": ""},
        ]),
        exclusions(),
        {"rwkv_lab_worker_artifact": "failed"},
        gpu_requested=False,
        host="somehost",
    )
    assert line["result"] == "failed"
    assert line["failed_suites"] == ["native-ctest"]
    assert "rwkv_lab_worker_artifact" in line["hosted_ci_excluded_failed"]


def test_every_declared_exclusion_gets_a_status():
    """The four suites the hosted job skips are the point of the record.

    The names are read from the pinned declaration rather than copied, so a
    fifth exclusion is attested the day it is declared.
    """
    declared = {entry["suite"] for entry in exclusions()["exclusions"]}
    assert declared, "the exclusion declaration is empty"
    results = excluded_suite_results(exclusions(), {}, native_ctest_status="passed")
    assert set(results) == declared


def test_a_skipped_native_build_reports_not_run_rather_than_passed():
    """A host with no GCC 16 runs none of the four. Saying so beats silence."""
    results = excluded_suite_results(exclusions(), {}, native_ctest_status="skipped")
    assert set(results.values()) == {NOT_RUN}

    line = attestation(
        receipt([{"name": "native-ctest", "status": "skipped", "detail": ""}]),
        exclusions(),
        {},
        gpu_requested=False,
        host="somehost",
    )
    # The overall run did not fail -- nothing failed -- but no reader can
    # mistake it for evidence that the excluded suites passed.
    assert line["result"] == "passed"
    assert set(line["hosted_ci_excluded_suites"].values()) == {NOT_RUN}
    assert line["hosted_ci_excluded_failed"] == []


def test_a_dirty_worktree_is_carried_into_the_line():
    """A run against modified source attests the modification, not the commit."""
    line = attestation(
        receipt([{"name": "go", "status": "passed", "detail": ""}], dirty_worktree=True),
        exclusions(),
        {},
        gpu_requested=False,
        host="somehost",
    )
    assert line["dirty_worktree"] is True


def test_junit_failures_are_not_read_as_passes(tmp_path):
    """The one parsing error worth ruling out is a failure read as a pass."""
    junit = tmp_path / "native-ctest.xml"
    junit.write_text(
        '<?xml version="1.0"?>\n'
        '<testsuite name="ctest" tests="3">\n'
        '  <testcase name="passing_suite" status="run"/>\n'
        '  <testcase name="failing_suite" status="fail">'
        '<failure message="boom"/></testcase>\n'
        '  <testcase name="skipped_suite" status="run"><skipped/></testcase>\n'
        "</testsuite>\n",
        encoding="utf-8",
    )
    assert junit_statuses(junit) == {
        "passing_suite": "passed",
        "failing_suite": "failed",
        "skipped_suite": "skipped",
    }


def test_missing_junit_is_not_an_exception():
    """Acceptance must finish and attest even when ctest wrote nothing."""
    assert junit_statuses(pathlib.Path("/nonexistent/native-ctest.xml")) == {}


def test_acceptance_appends_and_never_truncates():
    """Overwriting would delete the history that makes staleness legible."""
    text = ACCEPTANCE.read_text(encoding="utf-8")
    assert "append_acceptance_attestation.py" in text, (
        "acceptance.sh must write the attestation itself; a step a human has to "
        "remember is a step that gets forgotten"
    )
    assert '--log "$attestation_log"' in text
    # The receipt is redirected with `>`; the attestation must never be.
    assert ">$attestation_log" not in text
    assert "> $attestation_log" not in text
