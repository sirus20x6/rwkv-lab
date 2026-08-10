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
import shutil
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
    drifted, refused, _ = generator.recompute(document, REPOSITORY)
    assert drifted == []
    assert refused == []


def test_a_mutated_source_hash_is_reported_as_drift(tmp_path):
    """The generator must NOTICE staleness, not quietly recompute over it."""
    document = json.loads(CATALOGS[0].read_text(encoding="utf-8"))
    mutated = copy.deepcopy(document)
    mutated["entries"][0]["source_sha256"] = "sha256:" + "0" * 64
    drifted, _, refreshed = generator.recompute(mutated, REPOSITORY)
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


@pytest.fixture()
def sandbox(tmp_path):
    """A repository holding one real catalog and exactly the sources it pins.

    The dashboard catalog is used because it is the small one (4 entries) and
    because it is the catalog the native suite already loads WITH a repository
    root, so the two instruments are pointed at the same document.
    """
    catalog = REPOSITORY / "docs/experiment-vm/source-dispositions.dashboard.v1.json"
    root = tmp_path / "repository"
    document = json.loads(catalog.read_text(encoding="utf-8"))
    for entry in document["entries"]:
        destination = root / entry["source_path"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPOSITORY / entry["source_path"], destination)
    stored = root / "catalog.json"
    stored.parent.mkdir(parents=True, exist_ok=True)
    stored.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    return root, stored


def _run(sandbox, *extra):
    root, stored = sandbox
    return subprocess.run(
        [sys.executable, str(GENERATOR), str(stored), "--root", str(root), *extra],
        capture_output=True, text=True, check=False)


def test_the_sandbox_is_green_before_it_is_broken(sandbox):
    """Otherwise every refusal test below could be passing for another reason."""
    completed = _run(sandbox, "--check")
    assert completed.returncode == 0, completed.stdout
    assert "PASSED" in completed.stdout.strip().splitlines()[-1]


def _replace_with_symlink(sandbox):
    """Swap a pinned source for a symlink to byte-identical content."""
    root, stored = sandbox
    document = json.loads(stored.read_text(encoding="utf-8"))
    relative = document["entries"][0]["source_path"]
    source = root / relative
    twin = root / "identical-twin.py"
    twin.write_bytes(source.read_bytes())
    source.unlink()
    source.symlink_to(twin)
    assert generator.source_sha256(source) == \
        document["entries"][0]["source_sha256"], \
        "the fixture must be invisible to a checker that only hashes"
    return relative


def test_a_source_replaced_by_a_symlink_to_identical_bytes_is_refused(sandbox):
    """The case a naive hash cannot see, and the reason the C++ uses O_NOFOLLOW.

    The bytes at the end of the link are the reviewed bytes, so every digest
    still matches. What has changed is that the reviewed file is gone.
    """
    relative = _replace_with_symlink(sandbox)
    completed = _run(sandbox, "--check")
    assert completed.returncode == 1, completed.stdout
    assert f"REFUSED: {relative} resolves through a symlink" in completed.stdout
    assert "FAILED" in completed.stdout.strip().splitlines()[-1]


def test_a_symlinked_parent_directory_is_refused(sandbox):
    """The link need not be the leaf; the loader walks the whole path."""
    root, stored = sandbox
    document = json.loads(stored.read_text(encoding="utf-8"))
    relative = pathlib.PurePosixPath(document["entries"][0]["source_path"])
    parent = root / relative.parent
    moved = root / "elsewhere"
    parent.rename(moved)
    parent.symlink_to(moved)
    completed = _run(sandbox, "--check")
    assert completed.returncode == 1, completed.stdout
    assert (f"resolves through a symlink at {relative.parent}"
            in completed.stdout), completed.stdout


def test_a_refusal_is_not_silenced_by_leaving_check_off(sandbox):
    """Report mode reports; a document the loader rejects still exits non-zero."""
    _replace_with_symlink(sandbox)
    completed = _run(sandbox)
    assert completed.returncode == 1, completed.stdout
    assert "resolves through a symlink" in completed.stdout


def test_write_refuses_to_pin_through_a_symlink(sandbox):
    """--write must not record the link target's bytes as the reviewed ones."""
    _root, stored = sandbox
    before = stored.read_text(encoding="utf-8")
    _replace_with_symlink((_root, stored))
    completed = _run((_root, stored), "--write")
    assert completed.returncode == 1, completed.stdout
    assert f"REFUSED: not rewriting {stored}" in completed.stdout
    assert stored.read_text(encoding="utf-8") == before


def test_write_still_repairs_an_ordinary_drifted_pin(sandbox):
    """The refusal must not cost --write its actual job."""
    root, stored = sandbox
    document = json.loads(stored.read_text(encoding="utf-8"))
    relative = document["entries"][0]["source_path"]
    (root / relative).write_text("# edited\n", encoding="utf-8")
    completed = _run((root, stored), "--write")
    assert completed.returncode == 0, completed.stdout
    assert f"rewrote {stored}" in completed.stdout
    rewritten = json.loads(stored.read_text(encoding="utf-8"))
    assert rewritten["entries"][0]["source_sha256"] == \
        generator.source_sha256(root / relative)


@pytest.mark.parametrize("spelling", [
    "/dashboard/app.py",
    "../dashboard/app.py",
    "dashboard/../dashboard/app.py",
    "./dashboard/app.py",
    "dashboard//app.py",
    "dashboard/app.py/",
    "",
])
def test_an_unnormalized_source_path_is_refused(sandbox, spelling):
    """validate_relative_path() in the C++ refuses each of these outright.

    Python resolves `..` and `.` happily and reads straight through an absolute
    path, so without this the script would hash a file outside the root and
    call the catalog clean.
    """
    root, stored = sandbox
    document = json.loads(stored.read_text(encoding="utf-8"))
    document["entries"][0]["source_path"] = spelling
    stored.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    completed = _run((root, stored), "--check")
    assert completed.returncode == 1, completed.stdout
    assert "REFUSED:" in completed.stdout
    assert ("must be a normalized repository-relative path" in completed.stdout
            or "must be bounded and nonempty" in completed.stdout), \
        completed.stdout


def test_an_escaping_source_path_is_refused_before_it_is_hashed(tmp_path, sandbox):
    """The concrete harm: `../` reaching a file outside the root entirely."""
    root, stored = sandbox
    outside = tmp_path / "outside.py"
    outside.write_text("# not in the repository\n", encoding="utf-8")
    document = json.loads(stored.read_text(encoding="utf-8"))
    document["entries"][0]["source_path"] = "../outside.py"
    document["entries"][0]["source_sha256"] = generator.source_sha256(outside)
    stored.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    completed = _run((root, stored), "--check")
    assert completed.returncode == 1, completed.stdout
    assert "must be a normalized repository-relative path" in completed.stdout


# --- the words the verdict line prints ------------------------------------
#
# Asserted on the message, never on the exit code: on the passing path the two
# digests are equal, so the label costs nothing and changes no outcome, and on
# the failing path the exit code is already 1 with either spelling.


def _verdict(stdout: str) -> str:
    lines = [line for line in stdout.splitlines()
             if line.startswith("disposition pins (")]
    assert len(lines) == 1, f"expected exactly one verdict line:\n{stdout}"
    return lines[0]


def test_a_clean_catalog_names_one_tree_digest(sandbox):
    """When the two folds agree there is only one digest and no ambiguity."""
    completed = _run(sandbox, "--check")
    verdict = _verdict(completed.stdout)
    document = json.loads(sandbox[1].read_text(encoding="utf-8"))
    expected = generator.tree_digest(document["entries"])
    assert f"4 sources, tree digest {expected}" in verdict, verdict


def test_a_drifted_catalog_says_which_digest_is_which(sandbox):
    """The regression proof: this sentence would have caught the original slip.

    The line printed `tree digest {computed}` -- the fold over pins recomputed
    from the bytes on disk, which is the digest the catalog *ought* to declare.
    On a drifted run that is not the catalog's own digest, and the drifted run
    is the one whose line gets quoted into a card. Both are now named by where
    they came from.
    """
    root, stored = sandbox
    declared = generator.tree_digest(
        json.loads(stored.read_text(encoding="utf-8"))["entries"])
    relative = json.loads(stored.read_text(encoding="utf-8"))["entries"][0][
        "source_path"]
    (root / relative).write_text("# edited\n", encoding="utf-8")

    completed = _run((root, stored), "--check")
    assert completed.returncode == 1, completed.stdout
    verdict = _verdict(completed.stdout)
    document = json.loads(stored.read_text(encoding="utf-8"))
    _, _, refreshed = generator.recompute(document, root)
    computed = generator.tree_digest(refreshed["entries"])

    assert declared != computed, "the fixture did not move a pin"
    assert f"as read, the catalog's pins fold to {declared}" in verdict, verdict
    assert f"the bytes on disk fold to {computed}" in verdict, verdict
    assert f"tree digest {computed}" not in verdict, verdict


@pytest.mark.parametrize(
    ("count", "expected"),
    [
        (0, "0 sources, tree digest sha256:abc"),
        (1, "1 source, tree digest sha256:abc"),
        (3, "3 sources, tree digest sha256:abc"),
    ],
    ids=["none", "one", "several"],
)
def test_the_verdict_line_reads_correctly_at_every_count(count, expected):
    """Singular and plural, at 0, 1 and n.

    "1 sources" is the shape of error that makes a reader distrust the digest
    printed beside it, and this count moves whenever a source is classified.
    """
    assert generator.digest_summary("sha256:abc", "sha256:abc", count) == expected


def test_the_drifted_form_is_also_singular_at_one():
    assert generator.digest_summary("sha256:a", "sha256:b", 1) == (
        "1 source; as read, the catalog's pins fold to sha256:a; "
        "the bytes on disk fold to sha256:b")


def test_check_mode_passes_on_the_real_catalogs():
    for catalog in CATALOGS:
        completed = subprocess.run(
            [sys.executable, str(GENERATOR), str(catalog), "--check",
             "--root", str(REPOSITORY)],
            capture_output=True, text=True, check=False)
        last = completed.stdout.strip().splitlines()[-1]
        assert completed.returncode == 0, completed.stdout
        assert "PASSED" in last
