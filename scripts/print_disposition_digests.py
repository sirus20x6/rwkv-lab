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

Refusals, which are not drift
-----------------------------
The C++ loader decides two things before it hashes anything: `source_path` is
run through `validate_relative_path` (normalized, repository-relative), and the
file is opened `O_NOFOLLOW`, so a source replaced by a symlink cannot be read at
all. Those decisions have to hold here too, because the loader's byte check only
runs when `load_file` is handed a repository root and the native suite guards
that behind `TRAINVM_LEGACY_SOURCE_ROOT`, unset in CI: in a hosted run this
script is the only thing verifying those sources' bytes, so its resolution rules
are the effective contract. A checker that follows a link answers "are these the
bytes at the end of this path" rather than "are these the bytes that were
reviewed", and the two coincide right up until they matter.

A refusal is reported as REFUSED, exits non-zero with or without `--check`, and
suppresses `--write` -- repinning cannot repair it, and pinning through a link
would record the target's bytes as though they were the reviewed ones.
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


def path_spelling_problem(value: str) -> str | None:
    """Mirror of validate_relative_path() in source_disposition_catalog.cpp.

    The C++ runs that on every entry's `source_path` before it hashes anything,
    so a document this script pins through is not necessarily a document the
    loader will accept. Pure string work on both sides -- nothing here can drift
    the way a fold can. The sentence is the native one verbatim so one grep
    finds either report.
    """
    if not value or len(value) > 1024:
        return "source_path must be bounded and nonempty"
    relative = pathlib.PurePosixPath(value)
    if relative.is_absolute() or any(
            part in ("", ".", "..") for part in relative.parts) or (
            value != str(relative)):
        return "source_path must be a normalized repository-relative path"
    return None


def symlink_problem(root: pathlib.Path, relative: str) -> str | None:
    """The loader opens sources with O_NOFOLLOW, so a link is not a source.

    Without this, a classified source replaced by a symlink hashes to whatever
    it points at and reads as clean here, while the binary refuses to open it at
    all -- the pin would answer "are these the bytes at the end of this link"
    rather than "are these the bytes that were reviewed". Same sentence as
    scripts/ci_compatibility_pin_gate.py's check, for the same grep.
    """
    walked = root
    for part in pathlib.PurePosixPath(relative).parts:
        walked = walked / part
        if walked.is_symlink():
            return f"resolves through a symlink at {walked.relative_to(root)}"
    return None


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


def recompute(
    document: dict, root: pathlib.Path
) -> tuple[list[str], list[str], dict]:
    """Return (drifted source paths, refused entries, the refreshed document).

    Drift and refusal are separate because they are answered differently: a
    drifted pin is what `--write` exists to repair, while a refused entry is one
    the native loader would not accept at all, so pinning it would write a value
    the binary can never agree with. Nothing refused gets a recomputed pin.
    """
    refreshed = json.loads(json.dumps(document))
    drifted: list[str] = []
    refused: list[str] = []
    for entry in refreshed["entries"]:
        source_path = entry["source_path"]
        spelling = path_spelling_problem(source_path)
        if spelling is not None:
            refused.append(f"{source_path!r}: {spelling}")
            continue
        symlinked = symlink_problem(root, source_path)
        if symlinked is not None:
            refused.append(f"{source_path} {symlinked}")
            continue
        path = root / source_path
        if not path.is_file():
            drifted.append(f"{source_path} (missing from the worktree)")
            continue
        actual = source_sha256(path)
        if actual != entry["source_sha256"]:
            drifted.append(source_path)
        entry["source_sha256"] = actual
    return drifted, refused, refreshed


def digest_summary(declared: str, computed: str, count: int) -> str:
    """Say which tree digest the number is, in the line that gets quoted.

    This sentence used to read `{n} sources, tree digest {computed}`, where
    `computed` is the fold over the pins recomputed from the bytes on disk --
    the digest the catalog *ought* to declare, not the one it does. On a
    passing run the two are equal and the label costs nothing. On a drifted run
    they differ, and the drifted run is the one whose line gets pasted into a
    card; a reader records the recomputed value as the catalog's current tree
    digest, and this repository already has a history of plausible-looking
    digests that match nothing.

    So the two are printed side by side exactly when they differ, each named by
    where it came from. `as read` rather than `declares` because `--write` may
    already have replaced the pins on disk by the time this line prints.
    """
    noun = "source" if count == 1 else "sources"
    if declared == computed:
        return f"{count} {noun}, tree digest {computed}"
    return (
        f"{count} {noun}; as read, the catalog's pins fold to {declared}; "
        f"the bytes on disk fold to {computed}"
    )


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
    drifted, refused, refreshed = recompute(document, arguments.root)
    computed = tree_digest(refreshed["entries"])
    # The fold over the pins as checked in, which is what the C++ loader will
    # compute from this document. It differs from `computed` exactly when a
    # source's bytes have moved, which is the case whose line gets quoted.
    declared = tree_digest(document["entries"])
    # A catalog must not store this. It is derivable from entries, so a stored
    # copy carries no information, and it is the one line two independent
    # changes to the same scope are both forced to rewrite. Refusing a document
    # that still has one keeps a stale value from sitting there looking
    # authoritative -- the C++ loader refuses it too, for the same reason.
    stored = document.get("source_tree_digest")
    refreshed.pop("source_tree_digest", None)

    for path in drifted:
        print(f"DRIFTED: {path}")
    for problem in refused:
        print(f"REFUSED: {problem}")
    if stored is not None:
        print(f"STORED TREE DIGEST: {stored}")
        print("STORED TREE DIGEST: catalogs must not store one; it is derived "
              "from entries at validation time")

    problems = (list(drifted) + list(refused) +
                ([] if stored is None else ["stored tree digest"]))

    if arguments.write:
        # Refusals are not repairable by repinning. Writing here would produce a
        # document that looks freshly pinned and that the loader still rejects
        # -- and for a symlinked source it would pin the link target's bytes as
        # though they were the reviewed ones, which is the failure this refuses.
        if refused:
            print(f"REFUSED: not rewriting {arguments.catalog}; "
                  f"{len(refused)} entries the loader would reject")
        else:
            arguments.catalog.write_text(
                json.dumps(refreshed, indent=2) + "\n", encoding="utf-8")
            print(f"rewrote {arguments.catalog} with "
                  f"{len(refreshed['entries'])} source pins")

    print(verdict_line(
        f"disposition pins ({arguments.catalog.name})",
        problems,
        digest_summary(declared, computed, len(refreshed["entries"])),
    ))
    if refused:
        return 1
    return 1 if (arguments.check and problems) else 0


if __name__ == "__main__":
    raise SystemExit(main())
