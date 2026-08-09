"""The disposition-pin generator, checked from the side that needs no compiler.

scripts/print_disposition_digests.py replicates source_tree_digest() from
trainvm/src/source_disposition_catalog.cpp. A replication can drift from the
thing it replicates, and a drifted generator is worse than none.

The agreement between the two implementations is NOT asserted here. It cannot
be: this job has no C++ build. It is asserted in
trainvm/tests/source_disposition_catalog_tests.cpp, which drives both over the
same entries. That test replaced an arrangement where each implementation was
compared against a `source_tree_digest` stored in every catalog -- which
cross-checked them only transitively, only when somebody regenerated, and cost
one guaranteed conflict per concurrent change to a scope.

What is left here is everything that does not need a compiler: the algorithm's
own properties, the per-file content pins, and the refusal to accept a catalog
that still stores a tree digest.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import pathlib
import subprocess
import sys

import pytest

REPOSITORY = pathlib.Path(__file__).resolve().parents[1]
GENERATOR = REPOSITORY / "scripts/print_disposition_digests.py"
CATALOGS = sorted(
    (REPOSITORY / "docs/experiment-vm").glob("source-dispositions.*.v1.json"))

_spec = importlib.util.spec_from_file_location("print_disposition_digests",
                                               GENERATOR)
assert _spec is not None and _spec.loader is not None
generator = importlib.util.module_from_spec(_spec)
sys.modules["print_disposition_digests"] = generator
_spec.loader.exec_module(generator)


def test_there_are_catalogs_to_check():
    """Guards the whole file against becoming vacuous if the glob stops matching."""
    assert CATALOGS, "no source-disposition catalogs found to verify"


@pytest.mark.parametrize("catalog", CATALOGS, ids=lambda p: p.name)
def test_no_catalog_stores_a_tree_digest(catalog):
    """The serialization point must stay gone.

    A stored tree digest is derivable from `entries`, so it adds nothing, and
    it is the one line two independent changes to the same scope are both
    forced to rewrite -- four pull requests conflicted on it in one evening
    while touching disjoint work. Nothing stops it being pasted back except
    this assertion and the C++ loader's exact key set.
    """
    document = json.loads(catalog.read_text(encoding="utf-8"))
    assert "source_tree_digest" not in document


def test_the_tree_digest_is_a_pure_function_of_the_entry_pins():
    """The fold reads source_path and source_sha256 and nothing else.

    This is the property that makes storing it pointless, so it is worth an
    assertion rather than an argument: everything else in an entry can change
    without moving the digest.
    """
    entries = [{"source_path": "scripts/a.py", "source_sha256": "sha256:" + "a" * 64,
                "class": "executable_operation", "effects": ["read_source"]}]
    stripped = [{"source_path": entries[0]["source_path"],
                 "source_sha256": entries[0]["source_sha256"]}]
    assert generator.tree_digest(entries) == generator.tree_digest(stripped)


def test_the_fold_is_framed_so_a_split_cannot_be_moved():
    """("ab", "c") and ("a", "bc") concatenate alike and must not collide.

    This is what the NUL framing buys. A length-prefixed or unseparated fold
    would pass every other test in this file and fail here.
    """
    left = [{"source_path": "ab", "source_sha256": "c"}]
    right = [{"source_path": "a", "source_sha256": "bc"}]
    assert generator.tree_digest(left) != generator.tree_digest(right)


@pytest.mark.parametrize("catalog", CATALOGS, ids=lambda p: p.name)
def test_every_checked_in_source_hash_matches_the_file_on_disk(catalog):
    document = json.loads(catalog.read_text(encoding="utf-8"))
    drifted, _ = generator.recompute(document, REPOSITORY)
    assert drifted == []


def test_a_mutated_source_hash_is_reported_as_drift(tmp_path):
    """The generator must NOTICE staleness, not quietly recompute over it."""
    document = json.loads(CATALOGS[0].read_text(encoding="utf-8"))
    mutated = copy.deepcopy(document)
    mutated["entries"][0]["source_sha256"] = "sha256:" + "0" * 64
    drifted, refreshed = generator.recompute(mutated, REPOSITORY)
    assert drifted == [mutated["entries"][0]["source_path"]]
    # And it repairs it, so --write produces something the C++ will accept.
    assert refreshed["entries"][0]["source_sha256"] == \
        document["entries"][0]["source_sha256"]


def test_the_tree_digest_depends_on_entry_order():
    """Order is significant in the C++ fold, so it must be here too.

    If this passed with the order swapped, the replication would be folding a
    set rather than a sequence and could produce a digest the native suite
    rejects for a reason nothing here would explain.
    """
    document = json.loads(CATALOGS[0].read_text(encoding="utf-8"))
    entries = document["entries"]
    assert len(entries) >= 2
    swapped = [entries[1], entries[0], *entries[2:]]
    assert generator.tree_digest(swapped) != generator.tree_digest(entries)


def test_the_cli_computes_a_tree_digest_over_entries_handed_to_it(tmp_path):
    """The affordance the native agreement test drives.

    If this mode breaks, the cross-language check in
    source_disposition_catalog_tests.cpp cannot run at all -- and it lives
    behind an ~8 minute C++ build, so it is worth failing in seconds here.
    """
    entries = [{"source_path": "scripts/a.py", "source_sha256": "sha256:" + "a" * 64},
               {"source_path": "scripts/b.py", "source_sha256": "sha256:" + "b" * 64}]
    request = tmp_path / "entries.json"
    request.write_text(json.dumps(entries), encoding="utf-8")
    completed = subprocess.run(
        [sys.executable, str(GENERATOR), str(request), "--tree-digest"],
        capture_output=True, text=True, check=False)
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == generator.tree_digest(entries)


def test_check_mode_fails_on_a_catalog_that_stores_a_tree_digest(tmp_path):
    """End to end, through the real CLI, with a real non-zero exit."""
    document = json.loads(CATALOGS[0].read_text(encoding="utf-8"))
    document["source_tree_digest"] = generator.tree_digest(document["entries"])
    stale = tmp_path / "stale.json"
    stale.write_text(json.dumps(document, indent=2), encoding="utf-8")
    completed = subprocess.run(
        [sys.executable, str(GENERATOR), str(stale), "--check",
         "--root", str(REPOSITORY)],
        capture_output=True, text=True, check=False)
    assert completed.returncode == 1, completed.stdout
    assert "FAILED" in completed.stdout.strip().splitlines()[-1]


def test_check_mode_passes_on_the_real_catalogs():
    for catalog in CATALOGS:
        completed = subprocess.run(
            [sys.executable, str(GENERATOR), str(catalog), "--check",
             "--root", str(REPOSITORY)],
            capture_output=True, text=True, check=False)
        last = completed.stdout.strip().splitlines()[-1]
        assert completed.returncode == 0, completed.stdout
        assert "PASSED" in last
