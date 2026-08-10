#!/usr/bin/env python3
"""Bound the runtime skips inside collected test files.

`scripts/ci_coverage_gate.py` catches a test *file* that contributes nothing,
and `scripts/validate_native_ci_exclusions.py` catches a native *suite* that CI
stops running. Between those two sits the case neither sees: a file that
collects twenty tests, every one of which calls `pytest.skip()` at runtime. It
collected, so it is covered; it ran, so the job is green; and a run reporting
`1264 passed, 10 skipped` can drift to `1000 passed, 274 skipped` with every
gate still passing.

That drift is plausible here rather than theoretical, because of what the
reasons are. Every skip in this repository is runtime-conditional on a *host or
environment* fact -- a CUDA device, `nvidia-smi`, an installed asset, an
importable optional package. All of those change without anybody touching the
code, so the count can move on its own and nothing records that it did.

This gate reads a run's JUnit XML -- structured, emitted by the same pytest
invocation that produced the result, never the human terminal report -- and
compares what it finds against
`docs/experiment-vm/skip-reason-baseline.v1.json`, following the precedent of
`native-ci-exclusions.v1.json`: a checked-in declaration, one stated reason per
entry, changeable only on purpose.

Two halves, deliberately unequal:

* **A novel reason fails, everywhere.** The declared set is the union of every
  reason string the *source* can produce, so it is a property of the code and
  not of the machine. Observing fewer of them than are declared is normal and
  is only reported. Observing one nobody declared means a skip path exists that
  no human has looked at, and that is worth exactly one glance. This half is
  enforced in every environment.
* **The count is bounded per environment.** A count is a property of the host,
  so a single number cannot fit both a hosted runner with no GPU and a
  workstation with every asset installed. Each environment declares its own
  bound, or declares in the file that its count is reported rather than
  enforced and says why. An unknown environment name is a failure, not a
  silent pass -- otherwise a typo in `--environment` disables the check while
  the gate keeps printing PASSED.

Normalisation is whitespace only: runs of whitespace collapse to one space and
the ends are stripped, so a reason wrapped across two source lines matches the
same entry as its one-line form. Nothing else is folded away. Reasons vary by
interpolation at the front (`f"{gate} needs ..."`), the middle and the end
(`f"probe asset is not present on this host: {asset}"`), so entries are
anchored regular expressions rather than prefixes -- a prefix cannot express
the first case, and a prefix short enough to try would swallow every reason
that starts the same way. Because a pattern that matches too much is the one
failure that would make this gate decorative, each declared pattern must be
fully anchored, must carry at least MINIMUM_LITERAL_CHARACTERS literal
characters outside its wildcards, and must fail to match every canary string
below.

Usage:
    python scripts/ci_skip_reason_gate.py evidence/python-cpu-3.12.xml \\
        --environment github-hosted-cpu [--receipt evidence/skip-receipt.json]
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
import re
import subprocess
import sys
import xml.etree.ElementTree as ElementTree

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from scripts.gate_verdict import verdict_line  # noqa: E402

REPOSITORY = pathlib.Path(__file__).resolve().parent.parent
DECLARATION = REPOSITORY / "docs/experiment-vm/skip-reason-baseline.v1.json"
API_VERSION = "trainvm.skip-reason-baseline/v1"
AUTHORITY = "ci_scope_evidence_only"

# A pattern that carries a wildcard has to be specific enough that a genuinely
# new reason cannot land inside it. Twenty-four literal characters is roughly
# "one stated clause"; the patterns that would not clear it -- `^.*$`,
# `^CUDA.*$` -- are exactly the ones that would quietly absorb the next reason
# somebody writes.
#
# A pattern with no wildcard at all is exempt, and that is not a loophole: it
# matches one string and nothing else, so its length says nothing about how
# much it could swallow. Several real reasons here are genuinely short
# (`SM120 required`, `CUDA unavailable`) and a blanket minimum would only push
# them into being padded or wildcarded, which is worse.
MINIMUM_LITERAL_CHARACTERS = 24

# Strings no declared pattern may match. These are not a proof of specificity
# and are not meant to be; they are the cheap end of it, catching an entry that
# was widened to make a red gate go green.
CANARIES = (
    "",
    "skipped",
    "TODO",
    "temporarily disabled while we work out what is wrong",
    "a reason nobody has ever written down before",
)


def pattern_shape(pattern: str) -> tuple[int, int]:
    """Return (literal codepoints, wildcard elements) for a pattern.

    Walks the parsed regular expression rather than the source text, so `\\.`
    counts as one literal character and `[a-z]+` counts as none. Anchors are
    neither: `^` and `$` match no character and widen nothing.

    This reads a private stdlib module (`re._parser`, `sre_parse` before 3.11).
    Deliberately unguarded: if it ever stops importing this gate must fail
    loudly rather than fall back to a weaker check and keep printing PASSED,
    which is the exact shape of failure it exists to catch.
    """
    import re._parser as parser  # noqa: PLC0415  (private; see docstring)

    def walk(sequence) -> tuple[int, int]:
        literals = wildcards = 0
        for operation, argument in sequence:
            name = str(operation).rsplit(".", 1)[-1]
            if name == "LITERAL":
                literals += 1
            elif name == "AT":  # ^ and $
                continue
            elif name in {"MAX_REPEAT", "MIN_REPEAT"}:
                _, _, body = argument
                body_literals, body_wildcards = walk(body)
                # A repeated literal guarantees one occurrence, no more.
                literals += min(body_literals, 1)
                wildcards += body_wildcards + 1
            elif name in {"SUBPATTERN", "ATOMIC_GROUP"}:
                body = argument[-1] if isinstance(argument, tuple) else argument
                body_literals, body_wildcards = walk(body)
                literals += body_literals
                wildcards += body_wildcards
            elif name == "BRANCH":
                _, branches = argument
                walked = [walk(branch) for branch in branches]
                # The weakest branch is the one a loose entry would take.
                literals += min((entry[0] for entry in walked), default=0)
                wildcards += sum(entry[1] for entry in walked) + 1
            else:
                wildcards += 1
        return literals, wildcards

    return walk(parser.parse(pattern))


def normalise(reason: str) -> str:
    """Collapse whitespace. Nothing else -- see the module docstring."""
    return re.sub(r"\s+", " ", reason).strip()


def skips_in(paths: list[pathlib.Path]) -> tuple[list[str], list[str]]:
    """Return (normalised reasons, problems) from JUnit XML reports.

    One entry per skipped test case, not per distinct reason, so the caller can
    count both. A report that names no test cases at all is a problem: an empty
    or truncated XML would otherwise read as "nothing was skipped", which is
    the same silence this gate exists to end.
    """
    reasons: list[str] = []
    problems: list[str] = []
    for path in paths:
        if not path.is_file():
            problems.append(f"no JUnit report at {path}: nothing to check")
            continue
        try:
            root = ElementTree.parse(path).getroot()
        except ElementTree.ParseError as error:
            problems.append(f"{path} is not parseable JUnit XML: {error}")
            continue
        cases = root.iter("testcase")
        seen_case = False
        for case in cases:
            seen_case = True
            for skipped in case.iter("skipped"):
                message = skipped.get("message")
                if message is None:
                    message = skipped.text or ""
                reasons.append(normalise(message))
        if not seen_case:
            problems.append(
                f"{path} names no test cases; a report with no cases cannot "
                "show a skip, so it must not be read as 'nothing was skipped'")
    return reasons, problems


def declaration_problems(declaration: dict) -> list[str]:
    """Validate the pin itself, before believing anything it says."""
    problems: list[str] = []

    if declaration.get("api_version") != API_VERSION:
        problems.append(f"api_version must be {API_VERSION!r}")
    if declaration.get("authority") != AUTHORITY:
        problems.append(
            f"authority must be {AUTHORITY!r}: this file records what a run "
            "skipped, it does not decide what runs")

    entries = declaration.get("reasons")
    if not isinstance(entries, list) or not entries:
        problems.append("reasons must be a non-empty list")
        entries = []

    seen: set[str] = set()
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            problems.append(f"reason {index} is not an object")
            continue
        pattern = entry.get("pattern")
        why = entry.get("why")
        if not isinstance(pattern, str) or not pattern:
            problems.append(f"reason {index} declares no pattern")
            continue
        if pattern in seen:
            problems.append(f"{pattern}: declared twice")
        seen.add(pattern)
        # A reason nobody justified is the thing this file exists to prevent.
        if not isinstance(why, str) or len(why.strip()) < 16:
            problems.append(f"{pattern}: why must be a stated sentence")
        if not (pattern.startswith("^") and pattern.endswith("$")):
            problems.append(
                f"{pattern}: must be anchored ^...$ so it cannot match a "
                "longer reason that merely contains it")
            continue
        try:
            compiled = re.compile(pattern)
        except re.error as error:
            problems.append(f"{pattern}: is not a valid expression ({error})")
            continue
        literals, wildcards = pattern_shape(pattern)
        if wildcards and literals < MINIMUM_LITERAL_CHARACTERS:
            problems.append(
                f"{pattern}: carries a wildcard behind only {literals} literal "
                f"characters, under the {MINIMUM_LITERAL_CHARACTERS} required; "
                "a pattern this loose would absorb the next new reason instead "
                "of reporting it")
        for canary in CANARIES:
            if compiled.fullmatch(canary):
                problems.append(
                    f"{pattern}: matches the canary {canary!r}, so it is too "
                    "broad to distinguish a novel reason")

    environments = declaration.get("environments")
    if not isinstance(environments, list) or not environments:
        problems.append("environments must be a non-empty list")
        environments = []
    names: set[str] = set()
    for index, environment in enumerate(environments):
        if not isinstance(environment, dict):
            problems.append(f"environment {index} is not an object")
            continue
        name = environment.get("name")
        if not isinstance(name, str) or not name:
            problems.append(f"environment {index} has no name")
            continue
        if name in names:
            problems.append(f"{name}: declared twice")
        names.add(name)
        authority = environment.get("count_authority")
        if authority not in {"enforced", "reported"}:
            problems.append(
                f"{name}: count_authority must be 'enforced' or 'reported'")
        elif authority == "enforced":
            if not isinstance(environment.get("max_skipped"), int):
                problems.append(f"{name}: enforced count needs max_skipped")
        why = environment.get("why")
        if not isinstance(why, str) or len(why.strip()) < 16:
            problems.append(f"{name}: why must be a stated sentence")

    return problems


def commit() -> str:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPOSITORY,
            capture_output=True, text=True, check=False)
    except OSError:
        return "unknown"
    return completed.stdout.strip() or "unknown"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "reports", nargs="+", type=pathlib.Path,
        help="JUnit XML written by the run being graded (pytest --junitxml)")
    parser.add_argument(
        "--environment", required=True,
        help="which declared environment produced these reports")
    parser.add_argument(
        "--receipt", type=pathlib.Path, default=None,
        help="write the run's skip receipt here")
    parser.add_argument("--declaration", type=pathlib.Path, default=DECLARATION)
    arguments = parser.parse_args(argv)

    declaration = json.loads(arguments.declaration.read_text(encoding="utf-8"))
    problems = declaration_problems(declaration)
    if problems:
        # Nothing below can be trusted against a pin that does not validate,
        # so this returns rather than grading a run against it.
        for problem in problems:
            print(f"FAIL: {problem}", file=sys.stderr)
        print(verdict_line("skip reason gate", problems,
                           "the baseline declaration itself is invalid"))
        return 1

    environments = {entry["name"]: entry for entry in declaration["environments"]}
    environment = environments.get(arguments.environment)
    if environment is None:
        problem = (
            f"unknown environment {arguments.environment!r}; declare it in "
            f"{arguments.declaration} alongside "
            f"{', '.join(sorted(environments))}")
        print(f"FAIL: {problem}", file=sys.stderr)
        print(verdict_line("skip reason gate", [problem],
                           "no environment to grade against"))
        return 1

    reasons, problems = skips_in(arguments.reports)
    counted = collections.Counter(reasons)
    patterns = [
        (entry["pattern"], re.compile(entry["pattern"]))
        for entry in declaration["reasons"]
    ]

    novel: list[str] = []
    matched_patterns: set[str] = set()
    for reason in sorted(counted):
        hit = [text for text, compiled in patterns if compiled.fullmatch(reason)]
        if hit:
            matched_patterns.update(hit)
        else:
            novel.append(reason)

    for reason in novel:
        problems.append(
            f"NEW SKIP REASON: {reason!r} ({counted[reason]} test(s)) matches "
            "no declared pattern")

    total = len(reasons)
    authority = environment["count_authority"]
    bound = environment.get("max_skipped")
    if authority == "enforced" and total > bound:
        problems.append(
            f"{arguments.environment} skipped {total} tests, over its pinned "
            f"bound of {bound}; raise the pin deliberately or find out what "
            "stopped running")

    # The listing comes before the verdict for the same reason the native
    # exclusion gate's does: a reader of the CI log should see what did not run
    # rather than infer it from a passing job.
    print(f"  environment: {arguments.environment} ({authority} count"
          + (f", bound {bound}" if authority == "enforced" else "") + ")")
    print(f"  {total} skipped test(s), {len(counted)} distinct reason(s)")
    for reason, count in sorted(counted.items(), key=lambda item: (-item[1], item[0])):
        print(f"    x{count:<4} {reason}")
    unobserved = sorted({text for text, _ in patterns} - matched_patterns)
    if unobserved:
        # Reported, never enforced. The declared set is the union across every
        # bound environment, so a machine that has the asset simply does not
        # fire that skip, and requiring every entry to be observed -- the shape
        # the native exclusion gate uses -- would be wrong here for that
        # reason. Listed rather than counted because this is also the only
        # place an entry that no longer corresponds to any skip site becomes
        # visible.
        print(f"  {len(unobserved)} declared reason(s) this environment did "
              "not produce, which is expected:")
        for pattern in unobserved:
            print(f"    {pattern}")
    for problem in problems:
        print(f"FAIL: {problem}", file=sys.stderr)

    if arguments.receipt is not None:
        arguments.receipt.parent.mkdir(parents=True, exist_ok=True)
        arguments.receipt.write_text(json.dumps({
            "api_version": "trainvm.skip-receipt/v1",
            "commit": commit(),
            "environment": arguments.environment,
            "reports": [str(path) for path in arguments.reports],
            "skipped": total,
            "distinct_reasons": len(counted),
            "reasons": {reason: count for reason, count in sorted(counted.items())},
            "novel_reasons": novel,
        }, indent=2) + "\n", encoding="utf-8")
        print(f"  receipt: {arguments.receipt}")

    print(verdict_line(
        "skip reason gate", problems,
        # Neutral wording on purpose: an earlier draft said "all declared in
        # ..." here, which the failing form then printed verbatim beside the
        # word FAILED -- the detail claiming the opposite of the verdict.
        f"{total} skipped, {len(counted)} distinct reason(s), graded against "
        f"{arguments.declaration.name} as {arguments.environment}"))
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
