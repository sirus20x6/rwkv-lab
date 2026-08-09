#!/usr/bin/env python3
"""Pin the native ctest exclusion set so it cannot grow silently.

The native job runs in a container on a hosted runner and skips four suites
that fail there specifically. The reasons were written in a workflow comment,
which nothing checked against the regex beside it, so the two could drift and a
fifth exclusion could be added without anybody stating why.

This compares the workflow's `ctest -E` alternation against the declared set in
docs/experiment-vm/native-ci-exclusions.v1.json and fails on any difference in
either direction. Removing an exclusion is as much a change of what CI proves
as adding one, so both fail until declared.

It also prints the skipped suites and their reasons, so a reader of the CI log
can see what did not run instead of inferring it from a passing job.

And it prints where the excluded suites last actually ran. The declaration says
they are covered by `scripts/acceptance.sh` on a real host, but nothing
schedules that script, so "covered" only ever meant "runnable". Echoing the last
line of the attestation log puts the answer in the CI log of every run: either a
commit and a date a reader can compare against `git log`, or a plain statement
that no run has ever been attested. It is reported, never enforced -- no hosted
runner can produce a real-host attestation, so a gate here could only ever be
red, and the whole reason these four are excluded is that a permanently red
signal is one everybody learns to ignore.
"""

from __future__ import annotations

import json
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from scripts.gate_verdict import verdict_line  # noqa: E402

REPOSITORY = pathlib.Path(__file__).resolve().parent.parent
DECLARATION = REPOSITORY / "docs/experiment-vm/native-ci-exclusions.v1.json"
API_VERSION = "trainvm.native-ci-exclusions/v1"
AUTHORITY = "ci_scope_evidence_only"

# The alternation the native job passes to ctest, e.g.
#   -E "^(alpha|beta)$"
EXCLUSION_PATTERN = re.compile(r'-E\s+"\^\(([^)]*)\)\$"')


def failures(declaration: dict, workflow_text: str) -> list[str]:
    problems: list[str] = []

    if declaration.get("api_version") != API_VERSION:
        problems.append(f"api_version must be {API_VERSION!r}")
    if declaration.get("authority") != AUTHORITY:
        problems.append(
            f"authority must be {AUTHORITY!r}: this file records scope, it does "
            "not grant it")

    entries = declaration.get("exclusions")
    if not isinstance(entries, list) or not entries:
        problems.append("exclusions must be a non-empty list")
        return problems

    declared: list[str] = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            problems.append(f"exclusion {index} is not an object")
            continue
        suite = entry.get("suite")
        reason = entry.get("reason")
        if not isinstance(suite, str) or not suite.strip():
            problems.append(f"exclusion {index} has no suite name")
            continue
        # A reason is the entire point. An exclusion nobody justified is the
        # thing this file exists to prevent.
        if not isinstance(reason, str) or len(reason.strip()) < 16:
            problems.append(f"{suite}: reason must be a stated sentence")
        declared.append(suite)

    if len(set(declared)) != len(declared):
        problems.append("declared exclusions contain a duplicate suite")

    matches = EXCLUSION_PATTERN.findall(workflow_text)
    if len(matches) != 1:
        problems.append(
            f"expected exactly one ctest -E exclusion expression in the "
            f"workflow, found {len(matches)}")
        return problems

    workflow_suites = [part for part in matches[0].split("|") if part]

    missing = sorted(set(workflow_suites) - set(declared))
    extra = sorted(set(declared) - set(workflow_suites))
    if missing:
        problems.append(
            "the workflow excludes suites that are not declared with a reason: "
            + ", ".join(missing))
    if extra:
        problems.append(
            "declared exclusions that the workflow no longer excludes: "
            + ", ".join(extra))

    return problems


def attestation_report(declaration: dict) -> list[str]:
    """Say when the excluded suites last ran for real, or that they never have.

    Reported, not enforced -- see the module docstring. The failure modes here
    are all "cannot tell", and every one of them prints a sentence saying so
    rather than falling silent, because silence is the state this log exists to
    end.
    """
    relative = declaration.get("attested_by")
    if not relative:
        return ["  no attestation log is declared; nothing records whether the "
                "above have ever run"]

    log = REPOSITORY / relative
    if not log.is_file():
        return [f"  MISSING attestation log {relative}: no run is recorded"]

    lines = [line for line in log.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        return [
            f"  {relative} is empty: NO real-host acceptance has ever been "
            "attested, so the suites above are unverified on every host",
        ]

    try:
        last = json.loads(lines[-1])
    except json.JSONDecodeError:
        return [f"  {relative} ends in a line that is not JSON; cannot report"]

    report = [
        f"  last real-host acceptance: {last.get('result', '?')} at "
        f"{str(last.get('commit', '?'))[:12]} on {last.get('host', '?')} "
        f"({last.get('ran_at', '?')}), {len(lines)} run(s) attested",
    ]
    if last.get("dirty_worktree"):
        report.append(
            "  that run was against a DIRTY worktree, so it attests the "
            "modifications rather than the commit named")
    for suite, status in (last.get("hosted_ci_excluded_suites") or {}).items():
        report.append(f"    {suite}: {status}")
    report.append(
        "  compare that commit against git log; a distant or absent one means "
        "the suites above are unchecked right now")
    return report


def main() -> int:
    declaration = json.loads(DECLARATION.read_text(encoding="utf-8"))
    workflow = REPOSITORY / declaration.get("workflow", ".github/workflows/tests.yml")
    problems = failures(declaration, workflow.read_text(encoding="utf-8"))

    for problem in problems:
        print(f"FAIL: {problem}", file=sys.stderr)

    entries = declaration["exclusions"]
    if not problems:
        for entry in entries:
            print(f"  NOT RUN in hosted CI: {entry['suite']} — {entry['reason']}")
        print(f"  all of the above are covered by {declaration['covered_by']}")
        for line in attestation_report(declaration):
            print(line)

    # Last, and after the per-suite listing, so the verdict is the line a
    # reader sees when the log is truncated. Previously the failing path wrote
    # only to stderr and returned early, so a stdout-only reader saw silence on
    # failure and a suite listing on success -- distinguishable, but neither
    # line said which had happened.
    print(verdict_line(
        "native ctest exclusions",
        problems,
        f"{len(entries)} declared and pinned to {declaration['workflow']}",
    ))
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
