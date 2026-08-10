#!/usr/bin/env python3
"""Fail when a native module has no production includer.

A passing suite says the code works. It never says anything uses it. Four
modules were found in this repository, in one session, that were fully
implemented, fully tested, green, and reached by nothing outside their own
tests:

    LinuxCacheEvidencePublisher    every probe snapshot in the tree was written
                                   by a test (closed by 8ca3f10)
    worker runtime evidence hop    serve()/main.cpp passed a null authority, so
                                   every real message was refused (deployment
                                   half landed in d89cb13)
    WorkerExecutionPhaseCancelled  raise, the only except, and both reads all
                                   inside execution_phases.py (card-0eabe3e2)
    conversion_publication         its header is included by exactly two files,
                                   its own .cpp and its test (card-1cbcb5c9)

Each took a card sweep to find. The detector is one graph traversal, so CI
owns it instead.

This is deliberately the same gate as scripts/ci_coverage_gate.py pointed at a
different graph: that one fails when a *test file* contributes nothing to the
run, this one fails when a *module* contributes nothing to the product. A
second mechanism for the same shape of question would itself be the defect this
repository keeps finding.

Direct includes are not enough, and the difference is large
----------------------------------------------------------
The cheap form of this check -- ``git grep -l "<module>.hpp" -- trainvm/``, and
if the only hits are the module's own ``.cpp`` and its ``_tests.cpp`` then
nothing includes it -- is what found ``conversion_publication``. As a gate it
is unusable. Headers here include other headers (``service.hpp`` includes 13),
so a module reached only *transitively* looks unwired to a direct-include
count. Measured on this tree at the commit that added this file:

    direct includes      34 headers look unwired
    transitive reach      5 headers actually are

29 of 34 would have been false positives -- 85% -- and a gate that is wrong
five times out of six on its first run is one nobody trusts again. So this
walks the header-to-header edges to fixpoint.

The traversal validates against three of the four instances above. The two that
have since been wired (``linux_cache_evidence``, ``worker_runtime_evidence``)
are reported as reachable; ``conversion_publication``, which has not, is
reported as unwired. A direct-include count gets all three right too, along
with 29 things that are fine.

What counts as production
-------------------------
A translation unit under ``trainvm/src/``. Test translation units are, by
enumerated pattern rather than by substring:

    trainvm/tests/*.cpp        the native suites
    trainvm/benchmarks/*.cpp   built EXCLUDE_FROM_ALL, run by hand

Substring matching on "test" is how the count for ``execution_phase`` was
inflated by 14 during the investigation that produced this file: an exclusion
covering ``tests/`` and ``trainvm/tests/`` silently classified Go ``*_test.go``
files as production. The patterns are listed above and enumerated in
``TEST_ROOTS`` so the classification is visible rather than inferred.

A module's own ``.cpp`` never counts as its own includer. That is the whole
point: ``conversion_publication.cpp`` includes ``conversion_publication.hpp``
and always will.

Generated and vendored code is out of scope by construction, not by filter. The
graph reads only ``trainvm/include/trainvm/*.hpp`` and the three source roots
above. The generated protobuf bindings land in ``trainvm/build/generated``,
which is gitignored and in none of them, and ``dashboard/web/static/vendor/``
is not C++.

Why this fails rather than warns
--------------------------------
Four instances survived in a repository with careful reviewers and a strong
test culture, which is evidence that a non-blocking signal gets ignored. The
native-CI exclusion declaration makes the same argument from the other side --
"a permanently red job is one everybody learns to ignore" -- and a permanently
yellow one is ignored from day one.

Why there is an allowlist
-------------------------
Not every unwired module is a defect. A module landed deliberately ahead of its
consumer is legitimate: the worker runtime evidence hop was exactly that, and
refusing the message was the *correct* behaviour while the deployment half was
missing. The allowlist entry is what turns "unwired" from an accident into a
recorded decision with a reason and, where one exists, the card that will
remove it.

It is a countdown, not a configuration. An entry whose module has since gained
a production includer FAILS, so the list can only shrink -- the same shape as
``UNMAPPED_INTERCEPTION`` in tests/test_step_zero_interception_enumeration.py.

The Python half of this question is NOT shipped here, and that is measured
---------------------------------------------------------------------------
The card that asked for this named the Python analogue: a symbol exported from
``__init__.py`` whose every importer is under ``tests/``. It was implemented
and measured before being dropped, because at this tree's numbers it cannot
fail:

    symbol level, rwkv_lab.trainvm_worker      79 of 117 exports unwired
    symbol level, rwkv_lab.trainvm_adapters    17 of  18 exports unwired
    module level, everything under src/        69 of 231 modules unwired

Those are not 165 defects. ``rwkv_lab.trainvm_worker`` is a documented SDK
surface (docs/experiment-vm/PYTHON_WORKER_SDK.md) whose consumers are meant to
sit outside this repository, and ``src/rwkv_lab`` is a research lever library
where implementing a lever ahead of its use is the point. A gate that is 90%
allowlist is an instrument tuned to produce a comfortable number, which is the
defect class the card belongs to. Filed as card-d198cc09 rather than shipped
weak.

Usage:
    python scripts/ci_unwired_module_gate.py [--repository .]
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from scripts.gate_verdict import verdict_line  # noqa: E402

API_VERSION = "trainvm.unwired-module-exclusions/v1"
AUTHORITY = "ci_scope_evidence_only"
DECLARATION = "docs/experiment-vm/unwired-module-exclusions.v1.json"

HEADER_ROOT = "trainvm/include"
PRODUCTION_ROOT = "trainvm/src"
# Enumerated, not inferred. See the module docstring: substring matching on
# "test" is how a previous count of this same question went wrong.
TEST_ROOTS = ("trainvm/tests", "trainvm/benchmarks")

# Only quoted, project-internal includes. A system include cannot name a module
# in this tree.
INCLUDE = re.compile(r'^\s*#\s*include\s*"(trainvm/[\w./]+\.hpp)"', re.M)

# A reason short enough to be a label is not a reason. Matches the threshold
# scripts/validate_native_ci_exclusions.py already applies to its own entries.
MINIMUM_REASON = 16


def read(path: pathlib.Path) -> str:
    return path.read_text(encoding="utf-8")


def includes(path: pathlib.Path) -> set[str]:
    return set(INCLUDE.findall(read(path)))


def own_source(header: str) -> str:
    """The .cpp that implements a header, which is never its own includer."""
    return f"{PRODUCTION_ROOT}/{pathlib.PurePosixPath(header).stem}.cpp"


def unwired_modules(repository: pathlib.Path) -> tuple[list[str], dict[str, int]]:
    """Return the headers no production translation unit reaches, and a tally.

    Reachability is transitive through header-to-header includes, computed to
    fixpoint. A header reached only via an intermediate header is wired, which
    is the case a direct-include count gets wrong 29 times on this tree.
    """
    header_root = repository / HEADER_ROOT
    headers = {
        path.relative_to(header_root).as_posix(): path
        for path in sorted(header_root.rglob("*.hpp"))
    }
    header_edges = {name: includes(path) for name, path in headers.items()}

    def reachable_from(seeds: set[str]) -> set[str]:
        seen: set[str] = set()
        pending = list(seeds)
        while pending:
            current = pending.pop()
            if current in seen:
                continue
            seen.add(current)
            pending.extend(header_edges.get(current, ()))
        return seen

    production = sorted((repository / PRODUCTION_ROOT).glob("*.cpp"))
    tests = [
        path
        for root in TEST_ROOTS
        for path in sorted((repository / root).glob("*.cpp"))
    ]

    reach = {
        source.relative_to(repository).as_posix(): reachable_from(includes(source))
        for source in production
    }

    unwired = []
    for header in sorted(headers):
        reachers = {
            source for source, seen in reach.items() if header in seen
        } - {own_source(header)}
        if not reachers:
            unwired.append(header)

    tally = {
        "headers": len(headers),
        "production": len(production),
        "tests": len(tests),
    }
    return unwired, tally


def declaration_problems(declaration: dict) -> list[str]:
    problems: list[str] = []
    if declaration.get("api_version") != API_VERSION:
        problems.append(f"api_version must be {API_VERSION!r}")
    if declaration.get("authority") != AUTHORITY:
        problems.append(
            f"authority must be {AUTHORITY!r}: this file records scope, it "
            "does not grant it")
    return problems


def allowed(declaration: dict) -> tuple[dict[str, str], list[str]]:
    """Return module -> reason, plus any problems with the entries themselves."""
    problems: list[str] = []
    entries = declaration.get("exclusions")
    if not isinstance(entries, list) or not entries:
        return {}, ["exclusions must be a non-empty list"]

    reasons: dict[str, str] = {}
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            problems.append(f"exclusion {index} is not an object")
            continue
        module = entry.get("module")
        reason = entry.get("reason")
        if not isinstance(module, str) or not module.strip():
            problems.append(f"exclusion {index} has no module")
            continue
        # A reason is the entire point. An exclusion nobody justified is the
        # thing this file exists to prevent.
        if not isinstance(reason, str) or len(reason.strip()) < MINIMUM_REASON:
            problems.append(f"{module}: reason must be a stated sentence")
        if module in reasons:
            problems.append(f"{module}: declared twice")
        reasons[module] = reason if isinstance(reason, str) else ""
    return reasons, problems


def staleness(
    reasons: dict[str, str], unwired: list[str], header_root: pathlib.Path
) -> tuple[list[str], list[str]]:
    """Return allowlist entries that stopped being true, in both directions.

    This is what makes the list a countdown rather than a configuration. An
    entry for a module that has since gained a production includer must be
    removed, or the list would accumulate excuses that no longer describe
    anything; an entry naming a module that no longer exists is the same
    failure with the file deleted instead of wired.
    """
    still_unwired = set(unwired)
    now_wired = sorted(
        module for module in reasons
        if module not in still_unwired and (header_root / module).is_file()
    )
    missing = sorted(
        module for module in reasons if not (header_root / module).is_file()
    )
    return now_wired, missing


def population_summary(
    tally: dict[str, int],
    unwired: list[str],
    undeclared: list[str],
    reasons: dict[str, str],
    stale: int,
) -> str:
    """Say which population each number counts, in the line that gets quoted.

    This sentence used to read ``{len(unwired)} unwired, {len(reasons)}
    explained exclusions``. ``unwired`` is *every* header no production
    translation unit reaches, explained and unexplained alike; ``reasons`` is
    the allowlist, which ``staleness()`` forces to be a subset of it. So on a
    passing run the two are the same five modules counted twice, and the line
    reads as a five-module backlog standing beside five excuses. The number a
    reader wants -- unwired and *unexplained* -- is a third variable,
    ``undeclared``, which never reached this line at all. It is zero on a
    passing run, which is exactly when one number under two names is
    indistinguishable from the truth, and exactly why it is printed.

    The three header counts partition every header in the tree, in this order,
    so they always sum to the total and a reader can check that they do:

    - unwired and unexplained: the failure this gate exists to catch;
    - unwired but explained: an allowlist entry states why, a recorded decision;
    - wired: some production translation unit reaches it, transitively or not.

    The allowlist is reported separately because it is a different population
    -- a countdown of entries, not a count of headers -- and an entry that
    stopped applying is the other way this gate fails.
    """
    headers = tally["headers"]
    unexplained = len(undeclared)
    explained = len(unwired) - unexplained
    wired = headers - len(unwired)
    entries = len(reasons)
    # "translation units" stays plural unconditionally: it is the head noun of
    # a coordination over two counts, and any coordination containing a zero
    # takes the plural, so there is no count at which the singular is right.
    return (
        f"{headers} {'header' if headers == 1 else 'headers'} over "
        f"{tally['production']} production and {tally['tests']} test "
        "translation units; "
        f"{unexplained} unwired and unexplained, "
        f"{explained} unwired but explained in {DECLARATION}, "
        f"{wired} wired; {entries} allowlist "
        f"{'entry' if entries == 1 else 'entries'}, {stale} no longer applicable"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repository",
        default=str(pathlib.Path(__file__).resolve().parent.parent),
        help="repository root to analyse",
    )
    arguments = parser.parse_args()
    repository = pathlib.Path(arguments.repository).resolve()

    declaration_path = repository / DECLARATION
    if not declaration_path.is_file():
        print(f"FAIL: missing declaration {DECLARATION}")
        print(verdict_line(
            "unwired module gate", ["missing declaration"],
            "0 modules analysed"))
        return 1

    declaration = json.loads(read(declaration_path))
    problems = declaration_problems(declaration)
    reasons, entry_problems = allowed(declaration)
    problems += entry_problems

    # Printed here rather than only counted. Proving this gate bites caught it
    # doing the thing gate_verdict.py exists to prevent, from the other side:
    # a malformed entry produced "FAILED -- 1 problem above" with nothing above
    # it, so the verdict named a problem the log never stated.
    for problem in problems:
        print(f"FAIL: {problem}")

    unwired, tally = unwired_modules(repository)
    header_root = repository / HEADER_ROOT

    undeclared = [header for header in unwired if header not in reasons]
    now_wired, missing = staleness(reasons, unwired, header_root)

    for header in undeclared:
        print(f"UNWIRED: {header} is reached by no translation unit under "
              f"{PRODUCTION_ROOT}/ other than its own implementation, and is "
              f"not an explained exclusion in {DECLARATION}")
    for module in now_wired:
        print(f"STALE EXCLUSION: {module} now has a production includer; "
              f"remove its entry from {DECLARATION}")
    for module in missing:
        print(f"STALE EXCLUSION: {module} is listed in {DECLARATION} but no "
              "longer exists")

    if not (undeclared or now_wired or missing):
        for header in unwired:
            print(f"  UNWIRED but explained: {header} — {reasons[header].strip()}")

    problems += undeclared + now_wired + missing
    print(verdict_line(
        "unwired module gate",
        problems,
        population_summary(
            tally, unwired, undeclared, reasons, len(now_wired) + len(missing)
        ),
    ))
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
