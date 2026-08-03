"""The disposition-pin generator must agree with the C++ that enforces it.

scripts/print_disposition_digests.py replicates source_tree_digest() from
trainvm/src/source_disposition_catalog.cpp. A replication can drift from the
thing it replicates, and a drifted generator is worse than none: it would hand
you a digest the native suite then rejects, or -- far worse -- agree by accident
on the catalogs you happen to test and disagree on the one you re-pin.

These pin it in both directions: it reproduces every checked-in digest, and it
notices a mutated pin rather than recomputing over it silently.
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
def test_the_generator_reproduces_the_checked_in_tree_digest(catalog):
    """The replication agrees with the C++ on every catalog, not just one.

    Agreement on a single catalog could be luck; the three here differ in size
    (3, 128 and 165 entries) and in scope, so agreeing on all three is evidence
    the fold itself matches rather than one document happening to line up.
    """
    document = json.loads(catalog.read_text(encoding="utf-8"))
    assert generator.tree_digest(document["entries"]) == \
        document["source_tree_digest"]


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


def test_check_mode_fails_on_a_stale_catalog(tmp_path):
    """End to end, through the real CLI, with a real non-zero exit."""
    document = json.loads(CATALOGS[0].read_text(encoding="utf-8"))
    document["source_tree_digest"] = "sha256:" + "0" * 64
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
