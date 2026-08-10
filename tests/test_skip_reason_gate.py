"""The skip-reason gate has to bite, and the checked-in baseline has to be real.

Two halves, tested separately:

* the gate refuses a novel reason, an over-budget count and a pattern loose
  enough to swallow the next new reason, and
* the declaration on disk actually describes this repository -- it validates,
  and every reason a real run produced is covered by it.

The first half is what stops this from being decorative. A pin that cannot fail
is worse than no pin, and this repository has already shipped an integrity gate
that printed PASSED because its check silently failed.
"""

from __future__ import annotations

import json
import pathlib
import subprocess
import sys

import pytest

from scripts.ci_skip_reason_gate import (
    DECLARATION,
    declaration_problems,
    main,
    normalise,
    pattern_shape,
)

REPOSITORY = pathlib.Path(__file__).resolve().parents[1]


def junit(path: pathlib.Path, reasons: list[str], passed: int = 3) -> pathlib.Path:
    """Write a JUnit report with the given skip reasons.

    Deliberately built from the same element shape pytest emits -- a
    ``<skipped message=...>`` inside a ``<testcase>`` -- rather than from a
    hand-rolled structure the production parser would never meet. A fixture
    that constructs something the real writer does not is the shape that let a
    tuple/list mismatch survive six days of green CI here.
    """
    from xml.sax.saxutils import quoteattr

    lines = ['<?xml version="1.0" encoding="utf-8"?>', "<testsuites><testsuite>"]
    for index in range(passed):
        lines.append(f'<testcase classname="tests.t" name="ok{index}" />')
    for index, reason in enumerate(reasons):
        lines.append(
            f'<testcase classname="tests.t" name="s{index}">'
            f'<skipped type="pytest.skip" message={quoteattr(reason)}>'
            "</skipped></testcase>")
    lines.append("</testsuite></testsuites>")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def declaration(tmp_path: pathlib.Path, **overrides) -> pathlib.Path:
    document = {
        "api_version": "trainvm.skip-reason-baseline/v1",
        "authority": "ci_scope_evidence_only",
        "environments": [
            {"name": "bounded", "count_authority": "enforced", "max_skipped": 2,
             "why": "a fixed runner whose skip set is reproducible",
             "observed": "2 skipped in the run this fixture stands for"},
            {"name": "unbounded", "count_authority": "reported",
             "why": "a real host whose asset set differs from the next one"},
        ],
        "reasons": [
            {"pattern": "^CUDA is not available$",
             "why": "the runner exposes no accelerator device"},
            {"pattern": "^probe asset is not present on this host: .+$",
             "why": "the asset path is interpolated into the reason"},
        ],
    }
    document.update(overrides)
    path = tmp_path / "baseline.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def run(tmp_path, reasons, environment="bounded", **overrides) -> int:
    return main([
        str(junit(tmp_path / "report.xml", reasons)),
        "--environment", environment,
        "--declaration", str(declaration(tmp_path, **overrides)),
    ])


# --- the gate bites -------------------------------------------------------

def test_a_declared_reason_passes(tmp_path):
    assert run(tmp_path, ["CUDA is not available"]) == 0


def test_a_novel_reason_fails(tmp_path, capsys):
    """The half the card calls the more valuable one."""
    assert run(tmp_path, ["the display driver went missing overnight"]) == 1
    assert "NEW SKIP REASON" in capsys.readouterr().err


def test_a_novel_reason_fails_even_where_the_count_is_only_reported(tmp_path):
    """Novel-reason detection is environment-independent, by design.

    The declared set is a property of the source, so an environment that
    cannot pin a count still has no business producing a reason nobody wrote.
    """
    assert run(tmp_path, ["something nobody declared"], environment="unbounded") == 1


def test_a_reason_that_merely_contains_a_declared_one_is_novel(tmp_path):
    """Anchoring is load-bearing: containment is not a match."""
    assert run(tmp_path, ["CUDA is not available, and neither is the host"]) == 1


def test_an_interpolated_tail_matches_its_declared_pattern(tmp_path):
    assert run(tmp_path, ["probe asset is not present on this host: /srv/x"]) == 0
    assert run(tmp_path, ["probe asset is not present on this host: /srv/y"]) == 0


def test_exceeding_the_pinned_count_fails_even_with_no_novel_reason(tmp_path, capsys):
    reasons = ["CUDA is not available"] * 3  # bound is 2
    assert run(tmp_path, reasons) == 1
    assert "over its pinned bound" in capsys.readouterr().err


def test_the_same_over_budget_run_passes_where_the_count_is_reported(tmp_path):
    """The per-environment half: a host-dependent count is not a verdict."""
    assert run(tmp_path, ["CUDA is not available"] * 9, environment="unbounded") == 0


def test_an_undeclared_environment_fails_rather_than_passing_silently(tmp_path):
    """A typo in --environment must not disable the check.

    This is the failure the gate exists to prevent, aimed at itself: an unknown
    name that fell through to "nothing to enforce" would keep printing PASSED
    while checking nothing.
    """
    assert run(tmp_path, ["CUDA is not available"], environment="typo") == 1


def test_a_report_with_no_test_cases_is_a_failure_not_an_empty_skip_set(tmp_path):
    """An empty or truncated XML must not read as "nothing was skipped"."""
    path = tmp_path / "empty.xml"
    path.write_text("<testsuites><testsuite/></testsuites>", encoding="utf-8")
    assert main([
        str(path), "--environment", "bounded",
        "--declaration", str(declaration(tmp_path)),
    ]) == 1


def test_a_missing_report_is_a_failure(tmp_path):
    assert main([
        str(tmp_path / "absent.xml"), "--environment", "bounded",
        "--declaration", str(declaration(tmp_path)),
    ]) == 1


def test_the_receipt_records_the_count_and_every_distinct_reason(tmp_path):
    receipt = tmp_path / "receipt.json"
    assert main([
        str(junit(tmp_path / "r.xml", ["CUDA is not available"] * 2)),
        "--environment", "bounded",
        "--declaration", str(declaration(tmp_path)),
        "--receipt", str(receipt),
    ]) == 0
    document = json.loads(receipt.read_text())
    assert document["skipped"] == 2
    assert document["reasons"] == {"CUDA is not available": 2}
    assert document["environment"] == "bounded"


def test_the_receipt_is_written_even_when_the_gate_fails(tmp_path):
    """A receipt only written on success would document nothing worth reading."""
    receipt = tmp_path / "receipt.json"
    assert main([
        str(junit(tmp_path / "r.xml", ["nobody declared this one"])),
        "--environment", "bounded",
        "--declaration", str(declaration(tmp_path)),
        "--receipt", str(receipt),
    ]) == 1
    assert json.loads(receipt.read_text())["novel_reasons"] == [
        "nobody declared this one"]


# --- the declaration cannot be widened into uselessness -------------------

@pytest.mark.parametrize("pattern", [
    "^.*$",
    "^.+$",
    "^CUDA.*$",
    "^could not import .*$",
])
def test_a_pattern_loose_enough_to_absorb_a_new_reason_is_refused(tmp_path, pattern):
    problems = declaration_problems(json.loads(declaration(
        tmp_path,
        reasons=[{"pattern": pattern, "why": "widened to make a red gate green"}],
    ).read_text()))
    assert problems


@pytest.mark.parametrize("pattern", [
    "^temporarily disabled while we work out what is wrong$",
    "^skipped$",
])
def test_a_canary_reason_is_refused_even_as_an_exact_entry(tmp_path, pattern):
    """The canaries earn their keep only against patterns nothing else catches.

    These carry no wildcard, so the literal-count rule exempts them; the canary
    list is the only thing that refuses "declare the placeholder and move on".
    """
    problems = declaration_problems(json.loads(declaration(
        tmp_path,
        reasons=[{"pattern": pattern,
                  "why": "declaring a placeholder rather than a condition"}],
    ).read_text()))
    assert any("canary" in problem for problem in problems)


def test_an_unanchored_pattern_is_refused(tmp_path):
    problems = declaration_problems(json.loads(declaration(
        tmp_path,
        reasons=[{"pattern": "CUDA is not available on this particular host",
                  "why": "a substring match would swallow longer reasons"}],
    ).read_text()))
    assert any("anchored" in problem for problem in problems)


def test_a_short_pattern_with_no_wildcard_is_allowed(tmp_path):
    """`^SM120 required$` matches one string; its length says nothing."""
    assert declaration_problems(json.loads(declaration(
        tmp_path,
        reasons=[{"pattern": "^SM120 required$",
                  "why": "the device is not of the required architecture"}],
    ).read_text())) == []


def test_a_reason_with_no_stated_why_is_refused(tmp_path):
    problems = declaration_problems(json.loads(declaration(
        tmp_path, reasons=[{"pattern": "^SM120 required$", "why": "because"}],
    ).read_text()))
    assert problems


def test_an_enforced_environment_without_a_bound_is_refused(tmp_path):
    problems = declaration_problems(json.loads(declaration(
        tmp_path,
        environments=[{"name": "x", "count_authority": "enforced",
                       "why": "a fixed runner whose skip set is reproducible",
                       "observed": "measured in the run this fixture stands for"}],
    ).read_text()))
    assert any("max_skipped" in problem for problem in problems)


def test_an_enforced_bound_that_names_no_run_is_refused(tmp_path):
    """A bound nobody measured is a guess wearing a pin's clothes."""
    problems = declaration_problems(json.loads(declaration(
        tmp_path,
        environments=[{"name": "x", "count_authority": "enforced",
                       "max_skipped": 7,
                       "why": "a fixed runner whose skip set is reproducible"}],
    ).read_text()))
    assert any("observed" in problem for problem in problems)


def test_a_reported_environment_needs_no_measurement(tmp_path):
    """The whole point of 'reported': there is no number to justify."""
    assert declaration_problems(json.loads(declaration(
        tmp_path,
        environments=[{"name": "x", "count_authority": "reported",
                       "why": "a real host whose asset set differs from the next"}],
    ).read_text())) == []


def test_an_invalid_declaration_fails_the_gate_rather_than_grading_against_it(tmp_path):
    assert main([
        str(junit(tmp_path / "r.xml", ["CUDA is not available"])),
        "--environment", "bounded",
        "--declaration", str(declaration(tmp_path, api_version="something/v9")),
    ]) == 1


def test_pattern_shape_counts_literals_and_wildcards_separately():
    assert pattern_shape("^CUDA is not available$") == (21, 0)
    literals, wildcards = pattern_shape("^probe asset: .+$")
    assert wildcards and literals == len("probe asset: ")


def test_normalisation_folds_whitespace_and_nothing_else():
    """A reason wrapped across two source lines is the same reason.

    The bound is deliberately tight: anything that folded more than whitespace
    could collapse a genuinely novel reason into a declared one, which is the
    failure that would make the novel-reason half decorative.
    """
    assert normalise("multi line\n   reason") == "multi line reason"
    assert normalise("CUDA is unavailable") != normalise("CUDA is not available")


# --- the checked-in declaration describes this repository ------------------

def test_the_checked_in_declaration_validates():
    assert declaration_problems(
        json.loads(DECLARATION.read_text(encoding="utf-8"))) == []


def test_every_declared_environment_is_reachable_from_a_runner():
    """An environment nobody passes --environment for is a dead pin."""
    workflow = (REPOSITORY / ".github/workflows/tests.yml").read_text(encoding="utf-8")
    acceptance = (REPOSITORY / "scripts/acceptance.sh").read_text(encoding="utf-8")
    document = json.loads(DECLARATION.read_text(encoding="utf-8"))
    for environment in document["environments"]:
        name = environment["name"]
        assert name in workflow or name in acceptance, (
            f"{name} is declared but no runner passes --environment {name}, so "
            "its bound grades nothing")


def test_the_gate_states_a_verdict_in_both_directions(tmp_path):
    """The same property tests/test_gate_verdict.py pins for the other gates.

    It cannot live there: every gate in that file runs with no arguments, and
    this one needs a report and an environment.
    """
    outcomes = {}
    for label, reasons in (("pass", ["CUDA is not available"]),
                           ("fail", ["nobody declared this one"])):
        completed = subprocess.run(
            [sys.executable, str(REPOSITORY / "scripts/ci_skip_reason_gate.py"),
             str(junit(tmp_path / f"{label}.xml", reasons)),
             "--environment", "bounded",
             "--declaration", str(declaration(tmp_path))],
            capture_output=True, text=True, check=False, cwd=REPOSITORY)
        last = completed.stdout.strip().splitlines()[-1]
        assert (completed.returncode == 0) == ("PASSED" in last), (
            f"exit {completed.returncode} but the last line says {last!r}")
        outcomes[label] = last
    assert "FAILED" in outcomes["fail"] and "PASSED" in outcomes["pass"]
