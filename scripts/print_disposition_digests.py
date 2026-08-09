#!/usr/bin/env python3
"""Compute the content pins a source-disposition catalog declares.

The compatibility catalog has `trainvm print-catalog-digests`. This one had
nothing: its 165 per-source hashes and its tree digest were maintained by hand,
so refreshing them after an ordinary source edit meant hashing files yourself
and hoping. That is most of why re-pinning reads as a chore rather than as the
review it is supposed to be, and a chore gets done without the review.

It reports rather than checks by default, so it still tells you when the
checked-in values disagree -- that is the point of running it. `--check` turns
the same comparison into an exit code for CI.

The algorithm is the one in trainvm/src/source_disposition_catalog.cpp. This is
a replication, which is a thing that can silently drift from what it replicates.
The guard used to be indirect -- both implementations were compared against a
`source_tree_digest` stored in each catalog, so they were only cross-checked
through that value, and only at the moment somebody regenerated it. That stored
line was also the one line every concurrent change to a scope had to rewrite,
which conflicted four pull requests in a single evening.

The line is gone and the guard is now direct: `--tree-digest` drives this
implementation over entries handed to it from outside, and
trainvm/tests/source_disposition_catalog_tests.cpp runs both implementations
over the same entries and compares the answers with no stored value between
them.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from scripts.gate_verdict import verdict_line  # noqa: E402

REPOSITORY = pathlib.Path(__file__).resolve().parent.parent
TREE_DOMAIN = b"trainvm.source-disposition-tree/v1"


def source_sha256(path: pathlib.Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def tree_digest(entries: list[dict]) -> str:
    """Mirror of trainvm::source_tree_digest() in source_disposition_catalog.cpp.

    Entry order is significant and is the catalog's own order, not sorted here:
    the C++ folds entries in the order the document lists them.

    Kept honest by source_disposition_catalog_tests.cpp, which drives this
    function and the C++ one over the same entries via `--tree-digest` and
    requires the two answers to be equal.
    """
    material = bytearray(TREE_DOMAIN)
    for entry in entries:
        material += b"\0" + entry["source_path"].encode()
        material += b"\0" + entry["source_sha256"].encode()
    return "sha256:" + hashlib.sha256(bytes(material)).hexdigest()


def recompute(document: dict, root: pathlib.Path) -> tuple[list[str], dict]:
    """Return (drifted source paths, the document with pins refreshed)."""
    refreshed = json.loads(json.dumps(document))
    drifted: list[str] = []
    for entry in refreshed["entries"]:
        path = root / entry["source_path"]
        if not path.is_file():
            drifted.append(f"{entry['source_path']} (missing from the worktree)")
            continue
        actual = source_sha256(path)
        if actual != entry["source_sha256"]:
            drifted.append(entry["source_path"])
        entry["source_sha256"] = actual
    return drifted, refreshed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "catalog", type=pathlib.Path,
        help="a docs/experiment-vm/source-dispositions.*.v1.json document, or "
             "with --tree-digest a JSON array of entry objects ('-' for stdin)")
    parser.add_argument(
        "--tree-digest", action="store_true",
        help="read a bare JSON array of {source_path, source_sha256} objects "
             "and print the tree digest of exactly those entries. This is how "
             "the native suite drives this implementation over the same "
             "entries it drives the C++ one over, so the two are compared "
             "directly rather than through a stored value")
    parser.add_argument(
        "--root", type=pathlib.Path, default=REPOSITORY,
        help="repository root the source_path entries are relative to")
    parser.add_argument(
        "--check", action="store_true",
        help="exit non-zero when the checked-in pins are stale, instead of "
             "printing what they should be")
    parser.add_argument(
        "--write", action="store_true",
        help="rewrite the catalog in place with the recomputed pins")
    arguments = parser.parse_args()

    if arguments.tree_digest:
        text = (sys.stdin.read() if str(arguments.catalog) == "-"
                else arguments.catalog.read_text(encoding="utf-8"))
        entries = json.loads(text)
        if not isinstance(entries, list):
            raise SystemExit("--tree-digest expects a JSON array of entries")
        print(tree_digest(entries))
        return 0

    document = json.loads(arguments.catalog.read_text(encoding="utf-8"))
    drifted, refreshed = recompute(document, arguments.root)
    computed = tree_digest(refreshed["entries"])
    # A catalog must not store this. It is derivable from entries, so a stored
    # copy carries no information, and it is the one line two independent
    # changes to the same scope are both forced to rewrite. Refusing a document
    # that still has one keeps a stale value from sitting there looking
    # authoritative -- the C++ loader refuses it too, for the same reason.
    stored = document.get("source_tree_digest")
    refreshed.pop("source_tree_digest", None)

    for path in drifted:
        print(f"DRIFTED: {path}")
    if stored is not None:
        print(f"STORED TREE DIGEST: {stored}")
        print("STORED TREE DIGEST: catalogs must not store one; it is derived "
              "from entries at validation time")

    problems = list(drifted) + ([] if stored is None else ["stored tree digest"])

    if arguments.write:
        arguments.catalog.write_text(
            json.dumps(refreshed, indent=2) + "\n", encoding="utf-8")
        print(f"rewrote {arguments.catalog} with {len(refreshed['entries'])} "
              f"source pins")

    print(verdict_line(
        f"disposition pins ({arguments.catalog.name})",
        problems,
        f"{len(refreshed['entries'])} sources, tree digest {computed}",
    ))
    return 1 if (arguments.check and problems) else 0


if __name__ == "__main__":
    raise SystemExit(main())
