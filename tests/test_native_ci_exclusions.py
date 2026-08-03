"""The native ctest exclusion set must not be able to drift silently.

Four native suites are skipped on the hosted runner. That is defensible, but
the reasons lived only in a workflow comment that nothing checked against the
regex beside it. Two defects have already accumulated behind exactly this gap:
a pinned adapter-profile count went stale because the only suite checking it is
on the list, and both catalog digests drifted because their validation ran only
where nothing was watching.

These tests assert the pin actually fires, in both directions. A pin that
cannot fail is worse than none, because it reads as coverage.
"""

from __future__ import annotations

import json
import pathlib

import pytest

from scripts.validate_native_ci_exclusions import failures

REPOSITORY = pathlib.Path(__file__).resolve().parent.parent
DECLARATION = REPOSITORY / "docs/experiment-vm/native-ci-exclusions.v1.json"
WORKFLOW = REPOSITORY / ".github/workflows/tests.yml"


@pytest.fixture
def declaration() -> dict:
    return json.loads(DECLARATION.read_text(encoding="utf-8"))


@pytest.fixture
def workflow() -> str:
    return WORKFLOW.read_text(encoding="utf-8")


def test_checked_in_declaration_matches_the_workflow(declaration, workflow):
    assert failures(declaration, workflow) == []


def test_an_undeclared_exclusion_fails(declaration, workflow):
    """Adding a fifth exclusion must fail until somebody states why."""
    widened = workflow.replace(
        '|rwkv_lab_worker_artifact)$"',
        '|rwkv_lab_worker_artifact|host_startup_audit_tests)$"')
    assert widened != workflow, "workflow no longer contains the expected pattern"
    problems = failures(declaration, widened)
    assert any("host_startup_audit_tests" in problem for problem in problems)


def test_a_dropped_declaration_fails(declaration, workflow):
    """Removing a reason while the workflow still skips the suite must fail."""
    declaration["exclusions"] = [
        entry for entry in declaration["exclusions"]
        if entry["suite"] != "host_resources_tests"
    ]
    problems = failures(declaration, workflow)
    assert any("host_resources_tests" in problem for problem in problems)


def test_a_stale_declaration_fails(declaration, workflow):
    """A suite the workflow stopped excluding must not linger as declared.

    Removing an exclusion changes what CI proves just as much as adding one, so
    it fails until the declaration is updated too.
    """
    narrowed = workflow.replace("|host_resources_tests", "")
    assert narrowed != workflow
    problems = failures(declaration, narrowed)
    assert any("no longer excludes" in problem for problem in problems)


def test_an_exclusion_without_a_stated_reason_fails(declaration, workflow):
    """A reason is the entire point; a blank one is an undeclared exclusion."""
    declaration["exclusions"][0]["reason"] = "flaky"
    problems = failures(declaration, workflow)
    assert any("stated sentence" in problem for problem in problems)


def test_authority_is_evidence_only(declaration, workflow):
    """This file records scope; the workflow grants it."""
    declaration["authority"] = "ci_scope_authority"
    problems = failures(declaration, workflow)
    assert any("authority" in problem for problem in problems)
