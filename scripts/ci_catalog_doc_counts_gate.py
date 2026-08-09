#!/usr/bin/env python3
"""Fail when a number stated in COMPATIBILITY_CATALOG.md stops being true.

The document opened with:

    The reviewed v1 inventory contains 156 effect-specific records over 145
    unique source files.

156 was right. 145 was wrong by ten, and had been for long enough that nobody
knows when it stopped being right -- the catalog kept gaining referenced
sources and the sentence did not follow. The same document says "155
referenced sources" twice further down, so it had been disagreeing with itself
in plain sight.

The instructive part is *which* number survived. 156 is the length of
`kReviewedWorkflowIds`, a compiled array: a wrong 156 does not render as a
stale sentence, it fails to build. 145 lived only in prose, so nothing held it.
The number that was checked stayed true and the number that was written down
did not, in the same sentence, over the same catalog. That is the entire
argument for this file: a figure quoted in a document is an assertion, and an
assertion nothing evaluates decays.

So the fix is not "145 -> 155". A corrected number with nothing holding it is
the same defect one commit later, and this document would have been correct at
some point too.

What it checks
--------------
Every figure the document states about the catalog, not the one that was
noticed. A gate that pins the number somebody already found, while three
others drift beside it, is an instrument tuned to produce a comfortable
result. The claims are enumerated in CLAIMS below; at the commit that added
this file that is eight numbers across four sentences, of which exactly one
was wrong.

Two sides, from two places
--------------------------
The claimed value is parsed out of the prose. The true value is computed from
the catalog. That separation is the whole check: a gate that recomputes both
sides from the same source always agrees with itself and measures nothing, so
it stays green while the document says whatever it likes.

Concretely, the failure this must catch is not only "somebody typed the wrong
number" but "somebody added a source and the document did not move" -- which is
how 145 got wrong in the first place. The second is only detectable because the
document side is read as text.

A claim that no longer matches is a FAILURE
-------------------------------------------
Prose gets reworded, and a pattern that quietly stops matching turns this file
into a gate that passes because it checked nothing. Every claim must match
exactly once; zero matches and two matches both fail, and say which claim.

That is deliberately the noisier direction. Rewording a sentence this file
watches costs one red run and a pattern edit. The alternative costs nothing
and silently returns the document to the state that produced this card.

Why here and not in the native suite
------------------------------------
The card suggested a compiled constant beside `kReviewedWorkflowIds`, which
would work. It would also only run behind an ~8 minute GCC 16 build that the
PR-tier classifier can skip, to check a fact about a Markdown file that needs
no compiler -- and the specific pull request that gets this wrong is a
documentation edit, which is the one least likely to be waiting on a native
build. Same reasoning `scripts/ci_unwired_module_gate.py` and the disposition
pin check are already wired into the seconds-fast schema job.

It costs no re-pinning traffic either, which matters while several agents are
regenerating per-file pins in compatibility_catalog.cpp: this touches neither
that file nor any digest.

Usage:
    python scripts/ci_catalog_doc_counts_gate.py [--repository .] [--write]

`--write` splices the computed values back into the document, replacing only
the matched digits and moving nothing else -- the same "regenerate rather than
hand-edit" shape as `trainvm print-catalog-digests --write`. Without it the
gate only reports and returns an exit code.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from scripts.gate_verdict import verdict_line  # noqa: E402

DOCUMENT = "docs/experiment-vm/COMPATIBILITY_CATALOG.md"
CATALOG = "docs/experiment-vm/compatibility-workflows.v1.json"
DISPOSITIONS = "docs/experiment-vm/source-dispositions.{scope}.v1.json"

# One entry per figure the document states about the catalog.
#
# `pattern` must contain exactly one group named `n`; only those digits are
# compared, and only those digits are rewritten by --write. The surrounding
# text is matched so the claim is identified by what it asserts rather than by
# position -- line numbers move, and a gate keyed on a line number silently
# starts checking a different sentence.
#
# `truth` names a key of the measurements dict computed below.
CLAIMS: tuple[tuple[str, str, str], ...] = (
    (
        "inventory entry count",
        r"reviewed v1 inventory contains (?P<n>\d+) effect-specific records",
        "entries",
    ),
    (
        "inventory unique source count",
        r"effect-specific records over (?P<n>\d+)\s+unique source files",
        "unique_sources",
    ),
    (
        "referenced source count",
        r"every edit to any of the (?P<n>\d+)\s+referenced sources",
        "unique_sources",
    ),
    (
        "scripts dispositions also referenced here",
        r"(?P<n>\d+) of the \d+ `scripts` disposition entries",
        "scripts_overlap",
    ),
    (
        "scripts disposition entry count",
        r"\d+ of the (?P<n>\d+) `scripts` disposition entries",
        "scripts_entries",
    ),
    (
        "rwkv-lab dispositions also referenced here",
        r"(?P<n>\d+) of the \d+ `rwkv-lab`",
        "rwkv_lab_overlap",
    ),
    (
        "rwkv-lab disposition entry count",
        r"\d+ of the (?P<n>\d+) `rwkv-lab`",
        "rwkv_lab_entries",
    ),
    (
        "pin count",
        r"There are (?P<n>\d+) pins",
        "unique_sources",
    ),
)


def read(path: pathlib.Path) -> str:
    return path.read_text(encoding="utf-8")


def unique_source_paths(catalog: dict) -> set[str]:
    """Every distinct path the entries reference.

    Read off the entries rather than off `source_digests`, even though the two
    are required to be equal. Taking the count from the pin list would make
    this gate agree with a catalog whose pins and entries had come apart, and
    "unique source files" is a claim about the entries.
    """
    paths: set[str] = set()
    for entry in catalog.get("entries", []):
        paths.update(entry.get("source_paths", []))
    return paths


def catalog_problems(catalog: dict) -> list[str]:
    """The invariants this gate's own measurement depends on.

    `source_digests` is required to pin exactly the referenced paths -- the
    native loader refuses the catalog otherwise. It is restated here because
    the whole point of running in the Python job is that no C++ has run: the
    document's "155 pins" and its "155 unique source files" are only the same
    number while that holds, and a gate that assumes its own premise is how a
    figure goes unchecked in the first place.
    """
    problems: list[str] = []
    pinned = [pin.get("source_path") for pin in catalog.get("source_digests", [])]
    duplicates = sorted({path for path in pinned if pinned.count(path) > 1})
    for path in duplicates:
        problems.append(f"{CATALOG}: source_digests pins {path} more than once")
    referenced = unique_source_paths(catalog)
    for path in sorted(referenced - set(pinned)):
        problems.append(
            f"{CATALOG}: source_digests does not pin referenced source {path}")
    for path in sorted(set(pinned) - referenced):
        problems.append(
            f"{CATALOG}: source_digests pins {path}, which no entry references")
    return problems


def disposition_paths(repository: pathlib.Path, scope: str) -> list[str]:
    path = repository / DISPOSITIONS.format(scope=scope)
    document = json.loads(read(path))
    return [entry["source_path"] for entry in document.get("entries", [])]


def measure(repository: pathlib.Path) -> tuple[dict[str, int], list[str]]:
    catalog = json.loads(read(repository / CATALOG))
    referenced = unique_source_paths(catalog)
    scripts = disposition_paths(repository, "scripts")
    rwkv_lab = disposition_paths(repository, "rwkv-lab")
    measurements = {
        "entries": len(catalog.get("entries", [])),
        "unique_sources": len(referenced),
        "scripts_entries": len(scripts),
        "scripts_overlap": len(referenced & set(scripts)),
        "rwkv_lab_entries": len(rwkv_lab),
        "rwkv_lab_overlap": len(referenced & set(rwkv_lab)),
    }
    return measurements, catalog_problems(catalog)


def evaluate(
    document: str, measurements: dict[str, int]
) -> tuple[list[str], list[tuple[re.Match[str], int]]]:
    """Return the problems, and the (match, correct value) pairs --write needs."""
    problems: list[str] = []
    repairs: list[tuple[re.Match[str], int]] = []
    for label, pattern, key in CLAIMS:
        matches = list(re.finditer(pattern, document))
        if len(matches) != 1:
            problems.append(
                f"{DOCUMENT}: the {label} claim matched {len(matches)} times, "
                f"expected exactly one. The sentence this gate reads was "
                f"reworded or removed; update the pattern in CLAIMS "
                f"({pattern!r}) so the number stays checked. Its value is "
                f"{measurements[key]}.")
            continue
        match = matches[0]
        claimed = int(match.group("n"))
        correct = measurements[key]
        if claimed != correct:
            problems.append(
                f"{DOCUMENT}: {label} says {claimed}, the catalog holds "
                f"{correct}")
            repairs.append((match, correct))
    return problems, repairs


def rewrite(document: str, repairs: list[tuple[re.Match[str], int]]) -> str:
    """Replace only the matched digits, right to left so offsets stay valid."""
    for match, correct in sorted(repairs, key=lambda pair: -pair[0].start("n")):
        document = (
            document[: match.start("n")] + str(correct) + document[match.end("n"):]
        )
    return document


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repository",
        default=str(pathlib.Path(__file__).resolve().parent.parent),
        help="repository root to analyse")
    parser.add_argument(
        "--write", action="store_true",
        help="splice the computed values into the document")
    arguments = parser.parse_args()
    repository = pathlib.Path(arguments.repository).resolve()

    document_path = repository / DOCUMENT
    measurements, problems = measure(repository)
    document = read(document_path)
    claim_problems, repairs = evaluate(document, measurements)
    problems += claim_problems

    for problem in problems:
        print(f"FAIL: {problem}")

    if arguments.write and repairs:
        document_path.write_text(rewrite(document, repairs), encoding="utf-8")
        print(f"WROTE: {len(repairs)} corrected figures into {DOCUMENT}")
        # Still a failure. --write is a fixer, not a way to make CI green: the
        # run that had to correct the document is the run that should be red,
        # and the next one passes.

    print(verdict_line(
        "catalog document counts gate",
        problems,
        f"{len(CLAIMS)} stated figures checked against {measurements['entries']} "
        f"entries over {measurements['unique_sources']} unique source files",
    ))
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
