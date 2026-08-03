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
a replication, which is a thing that can silently drift from what it replicates,
so `--check` against an unmodified checkout is the guard: if this file ever
stops agreeing with the C++, that command fails and says so.
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
    """Mirror of source_tree_digest() in source_disposition_catalog.cpp.

    Entry order is significant and is the catalog's own order, not sorted here:
    the C++ folds entries in the order the document lists them.
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
        help="a docs/experiment-vm/source-dispositions.*.v1.json document")
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

    document = json.loads(arguments.catalog.read_text(encoding="utf-8"))
    drifted, refreshed = recompute(document, arguments.root)
    computed = tree_digest(refreshed["entries"])
    declared = document.get("source_tree_digest", "")
    refreshed["source_tree_digest"] = computed

    for path in drifted:
        print(f"DRIFTED: {path}")
    if declared and computed != declared:
        print(f"TREE DIGEST: declared {declared}")
        print(f"TREE DIGEST: computed {computed}")

    problems = list(drifted) + ([] if computed == declared else ["tree digest"])

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
