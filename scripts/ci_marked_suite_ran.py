#!/usr/bin/env python3
"""Fail when a marker-scoped job asserted nothing.

The `fla dependency tests` job exists to run the one assertion the CPU job
cannot: `test_stack_installed_legacy_fla_cache_preserves_layer_states`, which
drives the *installed* `fla` Cache rather than the protocol-only fake. It
installs `flash-linear-attention` and runs `pytest -m "fla and not gpu"`.

On 2026-08-10 that job reported:

    1 skipped, 2154 deselected, 8 warnings in 16.80s

Zero passed. The import inside `pytest.importorskip("fla.models.utils")` was
failing, the skip was taken, and the job was green having asserted nothing --
for as long as anyone can establish.

The test carries a comment explaining why it believed this could not happen:

    the `fla` marker is what stops that skip from silently deleting the
    coverage: the fla-dependency CI job installs the real package and runs
    `-m fla`, where pytest exits 5 (no tests collected) if this marker ever
    disappears.

That reasoning is sound and covers a real failure -- a *deleted marker*. It
does not cover the failure that happened, because the test was collected
normally and then skipped at runtime. pytest exits 0 for a run that skips
everything, so the safeguard was watching collection while the coverage
drained out of execution.

This gate is the missing half: a job whose entire purpose is to execute
something must fail when it executes nothing. It deliberately asserts on
*passed*, not on "not skipped" -- a test that errors is also not skipped, and
an errored run must not read as a run.

Why a separate gate rather than an argument to the skip-reason gate
-------------------------------------------------------------------
`scripts/ci_skip_reason_gate.py` already reads the same JUnit XML in this job,
and its bound of 1 for `github-hosted-fla` was measured from exactly the broken
run described above -- it faithfully recorded the hole rather than closing it,
which is correct behaviour for a countdown. But it answers "were the skips
declared and within budget", and the question here is "did anything run at
all". Those come apart precisely when a budget is set from a broken baseline,
which is the situation that produced this file, so folding one into the other
would make each less legible.

Usage:
    python scripts/ci_marked_suite_ran.py <junit.xml> --at-least N --label TEXT
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import xml.etree.ElementTree as ET

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from scripts.gate_verdict import verdict_line  # noqa: E402


def counts(report: pathlib.Path) -> tuple[int, int, int, int]:
    """Return (total, failures, errors, skipped) from a pytest JUnit report.

    Read from the XML rather than from pytest's terminal output on purpose:
    this repository's rich reporter is parsed wrongly by naive readers, and a
    harness that mis-parses the terminal reports success for everything.
    """
    root = ET.parse(report).getroot()
    suites = [root] if root.tag == "testsuite" else list(root)
    total = failures = errors = skipped = 0
    for suite in suites:
        total += int(suite.get("tests", 0))
        failures += int(suite.get("failures", 0))
        errors += int(suite.get("errors", 0))
        skipped += int(suite.get("skipped", 0))
    return total, failures, errors, skipped


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", help="pytest --junitxml path")
    parser.add_argument(
        "--at-least", type=int, default=1,
        help="minimum number of tests that must PASS (default 1)")
    parser.add_argument(
        "--label", default="this job",
        help="what the job exists to run, quoted in the failure")
    arguments = parser.parse_args()

    report = pathlib.Path(arguments.report)
    problems: list[str] = []

    if not report.exists():
        # Not a skip. A missing report means pytest did not get far enough to
        # write one, which is a harder failure than an empty run, and reading
        # it as "nothing to check" is how this whole class of defect survives.
        problems.append(
            f"{report}: no JUnit report, so it cannot be shown that {arguments.label} "
            f"ran anything. pytest did not reach the point of writing one.")
        print(f"FAIL: {problems[0]}")
        print(verdict_line("marked suite ran", problems, f"0 tests recorded in {report}"))
        return 1

    total, failures, errors, skipped = counts(report)
    passed = total - failures - errors - skipped

    if passed < arguments.at_least:
        problems.append(
            f"{report}: {passed} test(s) passed, fewer than the {arguments.at_least} "
            f"{arguments.label} exists to run ({total} collected, {skipped} skipped, "
            f"{failures} failed, {errors} errored). A job that asserts nothing is "
            f"green for the same reason a job that asserts everything is: nothing "
            f"failed. If the skip is legitimate, the honest change is to delete the "
            f"job or state why it may run empty -- not to lower this bound.")

    for problem in problems:
        print(f"FAIL: {problem}")

    print(verdict_line(
        "marked suite ran",
        problems,
        f"{passed} passed, {skipped} skipped, {failures} failed, {errors} errored "
        f"of {total} collected; {arguments.at_least} or more had to pass",
    ))
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
