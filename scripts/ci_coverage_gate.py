#!/usr/bin/env python3
"""Fail when a test file contributes no collected tests to the CI run.

CI previously executed a hand-maintained list of test files. A list like that
degrades silently: a new suite is simply never named, and the run stays green
while covering less than it did before. The workflow now runs the whole
directory, so the remaining failure mode is a file that exists but contributes
nothing -- an import guard that skips everything, a renamed marker, or a suite
whose tests were all deleted.

This gate collects the real run and reports those files. Anything genuinely
expected to contribute nothing must say so, with a reason, in
tests/coverage_exclusions.json.

Usage:
    python scripts/ci_coverage_gate.py [-m MARKEXPR] [--tests-dir tests]
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import subprocess
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from scripts.gate_verdict import verdict_line  # noqa: E402

EXCLUSIONS_NAME = "coverage_exclusions.json"


def collect(tests_dir: pathlib.Path, mark_expression: str) -> set[str]:
    """Return the set of test files that contribute at least one test."""
    command = [
        sys.executable, "-m", "pytest", str(tests_dir),
        "--collect-only", "-q", "-p", "no:cacheprovider",
    ]
    if mark_expression:
        command += ["-m", mark_expression]
    completed = subprocess.run(
        command, capture_output=True, text=True, check=False,
    )
    if completed.returncode not in (0, 5):  # 5 == no tests collected
        sys.stderr.write(completed.stdout)
        sys.stderr.write(completed.stderr)
        raise SystemExit(
            f"collection failed with exit code {completed.returncode}"
        )
    # Each collected node id starts with the file path.
    node = re.compile(r"^(tests/[\w/]+\.py)::")
    return {
        match.group(1)
        for line in completed.stdout.splitlines()
        if (match := node.match(line.strip()))
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tests-dir", default="tests")
    parser.add_argument(
        "-m", "--mark-expression", default="not gpu",
        help="pytest marker expression the CI job actually runs",
    )
    arguments = parser.parse_args()

    tests_dir = pathlib.Path(arguments.tests_dir)
    if not tests_dir.is_dir():
        raise SystemExit(f"{tests_dir} is not a directory")

    exclusions_path = tests_dir / EXCLUSIONS_NAME
    exclusions: dict[str, str] = {}
    if exclusions_path.exists():
        document = json.loads(exclusions_path.read_text())
        exclusions = dict(document.get("excluded", {}))

    on_disk = {
        str(path) for path in sorted(tests_dir.glob("test_*.py"))
    }
    covered = collect(tests_dir, arguments.mark_expression)
    uncovered = sorted(on_disk - covered - set(exclusions))
    stale = sorted(set(exclusions) - on_disk)

    for path in uncovered:
        print(f"NO COVERAGE: {path} contributes no tests to '"
              f"{arguments.mark_expression}' and is not an explained exclusion")
    for path in stale:
        print(f"STALE EXCLUSION: {path} is listed in {exclusions_path} "
              "but no longer exists")

    print(verdict_line(
        "coverage gate",
        uncovered + stale,
        f"{len(covered)} covered, {len(exclusions)} explained exclusions, "
        f"{len(on_disk)} test files on disk",
    ))
    return 1 if uncovered or stale else 0


if __name__ == "__main__":
    raise SystemExit(main())
