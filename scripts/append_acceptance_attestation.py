#!/usr/bin/env python3
"""Append one line recording that a real-host acceptance run happened.

`.github/workflows/tests.yml` excludes four native ctest suites from the hosted
job because they fail on a hosted runner specifically, and says they are covered
by `scripts/acceptance.sh` on a real host instead. Nothing invoked that script:
no workflow, no Makefile, nothing in CMake. Its receipt goes to
`evidence/acceptance.json`, and `/evidence/` is git-ignored. So there was no
record anywhere that the four had ever run, and "acceptance passed last night"
and "acceptance has never been run" produced byte-identical repositories.

This closes the second half of that gap -- the attestation. It does not cause
the runs; a person still types the command.

A receipt and an attestation are not the same artifact, and committing the first
is the mistake this one exists to avoid. A receipt is a snapshot of the current
state, so it is served to every reader at whatever commit they happen to have
checked out, and one spent six days confidently describing a dirty worktree
before it was git-ignored (c0433c0). A line here is a historical fact -- on this
date, at this commit, on this host, these suites did this -- which stays true
however far main moves afterwards. Facts append; snapshots rot.

Three properties follow from that, and each is load-bearing:

* **Append, never overwrite.** A single-line "last run" file destroys the very
  history that makes staleness legible, and makes every run collide with every
  other run in the same line. JSON Lines appends.
* **Record failures.** A log that only grows on success is a green-only
  instrument: an absent line would mean "never ran" or "ran and failed" with no
  way to tell, which is the defect this whole exercise is about.
* **Staleness must be readable.** Each line carries its commit and timestamp, so
  `git log` answers "how far has main moved since acceptance last ran" without
  any additional machinery. Nothing here fails a build for a stale attestation;
  that would be a red check nobody without hardware access could fix, and the
  workflow already explains why a permanently red job is worse than an honest
  gap. Visibility first.

Usage:
    python scripts/append_acceptance_attestation.py \
        --receipt evidence/acceptance.json \
        --exclusions docs/experiment-vm/native-ci-exclusions.v1.json \
        --ctest-junit evidence/native-ctest.xml \
        --log docs/experiment-vm/acceptance-attestations.v1.jsonl \
        --gpu-requested 0
"""

from __future__ import annotations

import argparse
import json
import pathlib
import platform
import xml.etree.ElementTree as ElementTree

API_VERSION = "trainvm.acceptance-attestation/v1"

# Statuses used for the individually named suites. "not_run" is deliberately
# distinct from "failed": a host without GCC 16 skips the whole native build,
# and reporting that as a failure would be as misleading as omitting it.
NOT_RUN = "not_run"
UNKNOWN = "unknown"


def suite_statuses(receipt: dict) -> dict[str, str]:
    """Flatten the receipt's suite list into name -> status."""
    return {
        entry["name"]: entry["status"]
        for entry in receipt.get("suites", [])
        if isinstance(entry, dict) and "name" in entry
    }


def junit_statuses(junit: pathlib.Path) -> dict[str, str]:
    """Read per-test results out of ctest's JUnit output.

    ctest marks a case with a `status` attribute and, for anything other than a
    pass, a `<failure>` or `<skipped>` child. Both are checked because the
    attribute's spelling has varied across CMake versions and a case that is
    read as passed when it failed is the one error mode worth ruling out.
    """
    if not junit.is_file():
        return {}
    try:
        root = ElementTree.parse(junit).getroot()
    except ElementTree.ParseError:
        return {}

    results: dict[str, str] = {}
    for case in root.iter("testcase"):
        name = case.get("name")
        if not name:
            continue
        if case.find("failure") is not None or case.find("error") is not None:
            results[name] = "failed"
        elif case.find("skipped") is not None:
            results[name] = "skipped"
        elif (case.get("status") or "run") in ("fail", "failed"):
            results[name] = "failed"
        elif (case.get("status") or "run") in ("notrun", "disabled"):
            results[name] = NOT_RUN
        else:
            results[name] = "passed"
    return results


def excluded_suite_results(
    exclusions: dict,
    junit: dict[str, str],
    native_ctest_status: str | None,
) -> dict[str, str]:
    """Status for each suite the hosted CI job does not run.

    The names come from the pinned exclusion declaration rather than a copy kept
    here, so adding an exclusion extends the attestation automatically. A copy
    would drift, and drifting is what the declaration exists to prevent.
    """
    results: dict[str, str] = {}
    for entry in exclusions.get("exclusions", []):
        suite = entry.get("suite")
        if not isinstance(suite, str):
            continue
        if suite in junit:
            results[suite] = junit[suite]
        elif native_ctest_status in (None, "skipped"):
            # The native build never happened, so the suite did not run. Saying
            # so beats both silence and a fabricated pass.
            results[suite] = NOT_RUN
        else:
            # ctest ran but this suite is not in its output: it was filtered
            # out, renamed, or deleted. Any of those is a real thing to notice.
            results[suite] = UNKNOWN
    return results


def attestation(
    receipt: dict,
    exclusions: dict,
    junit: dict[str, str],
    gpu_requested: bool,
    host: str,
) -> dict:
    suites = suite_statuses(receipt)
    failed = sorted(name for name, status in suites.items() if status == "FAILED")
    excluded = excluded_suite_results(exclusions, junit, suites.get("native-ctest"))
    excluded_failed = sorted(
        name for name, status in excluded.items() if status == "failed"
    )

    # Ordered so the first fields of a `tail -1` are the ones a human reads:
    # when, against what, and whether it worked.
    return {
        "api_version": API_VERSION,
        "ran_at": receipt.get("generated_at"),
        "commit": receipt.get("commit"),
        "dirty_worktree": bool(receipt.get("dirty_worktree")),
        "host": host,
        "result": "failed" if failed else "passed",
        "gpu_requested": bool(gpu_requested),
        "failed_suites": failed,
        "hosted_ci_excluded_suites": excluded,
        "hosted_ci_excluded_failed": excluded_failed,
        "suites": suites,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", required=True)
    parser.add_argument("--exclusions", required=True)
    parser.add_argument("--ctest-junit", required=True)
    parser.add_argument("--log", required=True)
    parser.add_argument("--gpu-requested", default="0")
    arguments = parser.parse_args()

    receipt = json.loads(pathlib.Path(arguments.receipt).read_text(encoding="utf-8"))
    exclusions = json.loads(
        pathlib.Path(arguments.exclusions).read_text(encoding="utf-8")
    )
    junit = junit_statuses(pathlib.Path(arguments.ctest_junit))

    line = attestation(
        receipt,
        exclusions,
        junit,
        arguments.gpu_requested not in ("0", "", "false", "False"),
        platform.node() or "unknown",
    )

    log = pathlib.Path(arguments.log)
    log.parent.mkdir(parents=True, exist_ok=True)
    # Append mode, and one compact line, so two runs can never overwrite each
    # other and a merge of two branches that each ran acceptance keeps both
    # facts rather than choosing between them.
    with log.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(line, sort_keys=False, separators=(",", ":")) + "\n")

    print(
        f"attested: {line['result']} at {line['commit']} on {line['host']}"
        f" -> {arguments.log}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
