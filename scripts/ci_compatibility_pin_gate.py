#!/usr/bin/env python3
"""Fail when a compatibility catalog's per-source pins no longer match the tree.

The native loader already enforces this, and enforces it well: it fails closed
and names the file. On PR #186 it said

    trainvm: compatibility source src/rwkv_lab/vision_native_train.py does not
    match its pinned bytes

at step 98 of 99 -- after 85 translation units had compiled and four static
libraries and an executable had linked. The check was right. What was wrong was
how much apparatus its cheapest question needed: comparing 155 sha256 values
against a stored list is sub-second work, and it was reachable only through an
8-14 minute C++26 build. An instrument that expensive to consult gets skipped,
and skipping it is what produced the red build.

So this runs in the seconds-fast schema job, beside the three gates that are
already there for exactly this reason: `validate_native_ci_exclusions.py`,
`ci_unwired_module_gate.py` and the `print_disposition_digests.py --check` loop.
The sibling catalogs had a cheap pin check and this one did not; line 4 of
`print_disposition_digests.py` says so in as many words.

What it deliberately does NOT do
--------------------------------
It does not recompute the source tree digest, and it does not recompute the
classification surface digest. Both are folds -- domain strings, separators, an
ordering rule -- and a Python mirror of a C++ fold is a second implementation
that has to be kept in agreement with the first. That coupling is not
hypothetical here: `print_disposition_digests.py` mirrors the disposition fold,
and PR #180 had to sequence its work around the hazard. `9e27ea1` checked and
found no such mirror exists for the compatibility catalog. Adding one to save a
build would be trading a slow check for a class of silent disagreement.

The per-file pins need no mirror. Each `source_digests` entry is
`{source_path, source_sha256}` and verifying it is `sha256(open(path).read())`
-- no digest-of-digests, no separator convention, no ordering that can drift.
And it loses nothing in practice, because both folds are computed from the same
file bytes: any edit that moves the surface digest necessarily moves that file's
sha256, which is caught here. The per-file pins strictly dominate as a drift
detector; the folds are what the pins are *for*, and they stay the C++
implementation's business.

The other invariants below (coverage, uniqueness, sortedness, spelling, no
symlinks) are restatements of what `compatibility_catalog.cpp` validates before
it hashes anything. They are all set and string comparisons -- the same
"nothing to drift" property -- and each is a way for the binary to refuse a
catalog this gate would otherwise call clean.

Scope is read off the tree, not off a list
------------------------------------------
A catalog is recognised by filename glob AND by `api_version`, and disagreement
between the two is a FAILURE rather than a skip. Either signal alone is
escapable: keyed only on the api_version, editing one string drops a document
out of scope silently; keyed only on the filename, a document named anything
else is never seen. `tests/test_experiment_schema_authority.py` hardened the
sibling gate this way after the same reasoning.

Usage:
    python scripts/ci_compatibility_pin_gate.py [--repository .]

There is no --write. Regenerating is `trainvm print-catalog-digests <catalog>
. --write`, which also refreshes the two digests this gate does not compute, so
a second writer here could only produce a document the binary still refuses.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from scripts.gate_verdict import verdict_line  # noqa: E402

CATALOG_DIRECTORY = "docs/experiment-vm"
CATALOG_GLOB = "compatibility-workflows*.v1.json"
API_VERSION = "trainvm.compatibility-workflows/v1"


def source_sha256(path: pathlib.Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def is_lowercase_sha256(value: str) -> bool:
    """The spelling rule from compatibility_catalog.cpp's is_lowercase_sha256."""
    if not value.startswith("sha256:"):
        return False
    body = value[len("sha256:"):]
    return len(body) == 64 and all(c in "0123456789abcdef" for c in body)


def path_spelling_problem(value: str) -> str | None:
    """Mirror of validate_source_path_spelling(), which is pure string work.

    A path the loader refuses to open is a path this gate must not silently
    hash through some other resolution -- `../` and an absolute path both read
    fine from Python and are rejected outright by the C++.
    """
    if not value:
        return "is empty"
    relative = pathlib.PurePosixPath(value)
    if relative.is_absolute():
        return "is not repository-relative"
    parts = relative.parts
    if any(part in (".", "..", "") for part in parts):
        return "is not normalized (contains '.' or '..')"
    if value != str(relative):
        return "is not spelled normally"
    return None


def symlink_problem(root: pathlib.Path, relative: str) -> str | None:
    """The loader opens sources with RESOLVE_NO_SYMLINKS beneath the root.

    Without this, a source replaced by a symlink to another file would hash to
    whatever it points at and pass here, while the binary refuses to open it at
    all -- the gate would be green for a catalog the runtime rejects, which is
    the one outcome that makes a pre-flight worse than none.
    """
    walked = root
    for part in pathlib.PurePosixPath(relative).parts:
        walked = walked / part
        if walked.is_symlink():
            return f"resolves through a symlink at {walked.relative_to(root)}"
    return None


def referenced_paths(catalog: dict) -> set[str]:
    paths: set[str] = set()
    for entry in catalog.get("entries", []):
        paths.update(entry.get("source_paths", []))
    return paths


def check_catalog(
    catalog: dict, name: str, root: pathlib.Path
) -> tuple[list[str], int]:
    """Return (problems, pins checked) for one recognised catalog."""
    problems: list[str] = []
    pins = catalog.get("source_digests", [])
    if not pins:
        problems.append(f"{name}: source_digests pins nothing")
        return problems, 0

    pinned_paths = [pin.get("source_path", "") for pin in pins]
    seen: set[str] = set()
    for path in pinned_paths:
        if path in seen:
            problems.append(
                f"{name}: compatibility source_digests must pin each path "
                f"once: {path}")
        seen.add(path)

    if pinned_paths != sorted(pinned_paths):
        problems.append(
            f"{name}: compatibility source_digests must be sorted by "
            f"source_path")

    referenced = referenced_paths(catalog)
    for path in sorted(referenced - seen):
        problems.append(
            f"{name}: compatibility source_digests does not pin referenced "
            f"source {path}")
    for path in sorted(seen - referenced):
        problems.append(
            f"{name}: compatibility source_digests pins {path}, which no "
            f"entry references")

    for pin in pins:
        path = pin.get("source_path", "")
        pinned = pin.get("source_sha256", "")
        spelling = path_spelling_problem(path)
        if spelling is not None:
            problems.append(f"{name}: source_path {path!r} {spelling}")
            continue
        if not is_lowercase_sha256(pinned):
            problems.append(
                f"{name}: compatibility source_sha256 must be lowercase "
                f"sha256 for {path}")
            continue
        symlinked = symlink_problem(root, path)
        if symlinked is not None:
            problems.append(f"{name}: compatibility source {path} {symlinked}")
            continue
        absolute = root / path
        if not absolute.is_file():
            problems.append(
                f"{name}: compatibility source {path} is missing from the "
                f"worktree")
            continue
        # The binary's own sentence, verbatim, so a log line from this gate and
        # a log line from `trainvm` are searchable as the same failure.
        if source_sha256(absolute) != pinned:
            problems.append(
                f"{name}: compatibility source {path} does not match its "
                f"pinned bytes")
    return problems, len(pins)


def collect(root: pathlib.Path) -> tuple[list[tuple[str, dict]], list[str]]:
    """Recognise catalogs by filename AND api_version; disagreement fails."""
    directory = root / CATALOG_DIRECTORY
    problems: list[str] = []
    recognised: list[tuple[str, dict]] = []

    by_name = sorted(directory.rglob(CATALOG_GLOB))
    named = {path.resolve() for path in by_name}

    for path in by_name:
        name = str(path.relative_to(root))
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            problems.append(f"{name}: is not readable JSON ({error})")
            continue
        if not isinstance(document, dict):
            problems.append(f"{name}: is not a JSON object")
            continue
        version = document.get("api_version")
        if version != API_VERSION:
            problems.append(
                f"{name}: is named like a compatibility catalog but declares "
                f"api_version {version!r}, not {API_VERSION!r}. A document "
                f"cannot leave this gate's scope by editing one string")
            continue
        recognised.append((name, document))

    # The other direction. A catalog renamed out of the glob keeps its
    # api_version, and nothing else in this directory carries that string.
    for path in sorted(directory.rglob("*.json")):
        if path.resolve() in named:
            continue
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        if isinstance(document, dict) and document.get("api_version") == API_VERSION:
            problems.append(
                f"{path.relative_to(root)}: declares api_version "
                f"{API_VERSION!r} but is not named {CATALOG_GLOB}, so the "
                f"pins in it would go unchecked")
    return recognised, problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repository",
        default=str(pathlib.Path(__file__).resolve().parent.parent),
        help="repository root the source_path entries are relative to")
    arguments = parser.parse_args()
    root = pathlib.Path(arguments.repository).resolve()

    recognised, problems = collect(root)
    if not recognised and not problems:
        problems.append(
            f"no {CATALOG_GLOB} under {root / CATALOG_DIRECTORY}; this gate "
            f"would otherwise pass having checked nothing")

    pins = 0
    for name, document in recognised:
        catalog_problems, counted = check_catalog(document, name, root)
        problems += catalog_problems
        pins += counted

    for problem in problems:
        print(f"FAIL: {problem}")

    print(verdict_line(
        "compatibility source pins",
        problems,
        f"{pins} per-source pins across {len(recognised)} "
        f"{'catalog' if len(recognised) == 1 else 'catalogs'}; the tree and "
        f"classification-surface digests are checked by trainvm, not here",
    ))
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
