"""The figures COMPATIBILITY_CATALOG.md states are checked against the catalog.

scripts/ci_catalog_doc_counts_gate.py exists because "156 records over 145
unique source files" was wrong by ten and nothing evaluated it. The interesting
half of that is not the wrong number, it is that the gate replacing it must be
able to go red for the reason the original figure went wrong: a source was
added and the prose did not follow.

So the tests below are arranged around distinct mechanisms rather than around
distinct wordings:

    claimed != computed          the plain disagreement
    computed moves with the tree the failure that produced the card -- the true
                                 value is recomputed, not restated, so adding a
                                 source turns a previously-correct sentence red
    claim no longer matches      rewording must fail, not silently stop checking
    pins match referenced paths  the premise the unique-source count rests on
    --write splices only digits  the fixer

Each one is separately mutation-tested; see the PR that added this file for the
table of which tests redden per mutation.
"""

from __future__ import annotations

import importlib.util
import json
import pathlib
import shutil
import subprocess
import sys

import pytest

REPOSITORY = pathlib.Path(__file__).resolve().parents[1]
GATE = REPOSITORY / "scripts/ci_catalog_doc_counts_gate.py"
DOCUMENT = "docs/experiment-vm/COMPATIBILITY_CATALOG.md"
CATALOG = "docs/experiment-vm/compatibility-workflows.v1.json"

_spec = importlib.util.spec_from_file_location("ci_catalog_doc_counts_gate", GATE)
assert _spec is not None and _spec.loader is not None
gate = importlib.util.module_from_spec(_spec)
sys.modules["ci_catalog_doc_counts_gate"] = gate
_spec.loader.exec_module(gate)


def run(repository: pathlib.Path, *arguments: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(GATE), "--repository", str(repository), *arguments],
        capture_output=True, text=True, check=False)


@pytest.fixture
def tree(tmp_path: pathlib.Path) -> pathlib.Path:
    """A copy of just the documents the gate reads.

    Copying the whole repository would make every test in this file slow enough
    to be skipped over; the gate opens four files and nothing else.
    """
    root = tmp_path / "repository"
    (root / "docs/experiment-vm").mkdir(parents=True)
    for name in sorted((REPOSITORY / "docs/experiment-vm").glob(
            "source-dispositions.*.v1.json")):
        shutil.copy(name, root / "docs/experiment-vm" / name.name)
    shutil.copy(REPOSITORY / CATALOG, root / CATALOG)
    shutil.copy(REPOSITORY / DOCUMENT, root / DOCUMENT)
    return root


def load(root: pathlib.Path) -> dict:
    return json.loads((root / CATALOG).read_text(encoding="utf-8"))


def store(root: pathlib.Path, catalog: dict) -> None:
    (root / CATALOG).write_text(json.dumps(catalog, indent=2), encoding="utf-8")


def document_of(root: pathlib.Path) -> str:
    return (root / DOCUMENT).read_text(encoding="utf-8")


def test_the_checked_in_document_agrees_with_the_checked_in_catalog():
    result = run(REPOSITORY)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "PASSED" in result.stdout


def test_every_claim_matches_the_document_exactly_once():
    """Guards this file against the gate quietly checking nothing.

    A CLAIMS table whose patterns all miss would make the gate fail loudly, but
    a table that became EMPTY would make it pass on any document at all. The
    count is asserted rather than merely `> 0` so deleting a claim is a
    deliberate edit here as well as there.
    """
    assert len(gate.CLAIMS) == 8
    document = document_of(REPOSITORY)
    for label, pattern, _ in gate.CLAIMS:
        import re
        assert len(re.findall(pattern, document)) == 1, label


def test_a_stale_prose_count_fails_and_names_both_numbers(tree):
    """The state this card was filed in: the document said 145, the tree held 155."""
    document = document_of(tree).replace(
        "over 155 unique source files", "over 145 unique source files")
    assert "over 145" in document, "the sentence under test was not rewritten"
    (tree / DOCUMENT).write_text(document, encoding="utf-8")

    result = run(tree)
    assert result.returncode == 1
    assert ("inventory unique source count says 145, the catalog holds 155"
            in result.stdout)
    assert "FAILED" in result.stdout


def test_adding_a_source_moves_the_true_count_and_reddens_a_correct_document(tree):
    """The failure that produced the card, reproduced forward.

    Nothing in the document changes here. The catalog gains one referenced
    source, exactly as it did on its way from 145 to 155, and the sentence that
    was correct a moment ago has to go red -- otherwise the gate is only
    checking that somebody typed carefully once.
    """
    assert run(tree).returncode == 0, "the copied tree must start clean"

    catalog = load(tree)
    added = "scripts/a_source_the_catalog_did_not_reference.py"
    catalog["entries"][0]["source_paths"].append(added)
    catalog["source_digests"].append(
        {"source_path": added, "source_sha256": "sha256:" + "0" * 64})
    store(tree, catalog)

    result = run(tree)
    assert result.returncode == 1
    assert ("inventory unique source count says 155, the catalog holds 156"
            in result.stdout)
    # The other two sentences quoting the same figure move together, which is
    # the point of enumerating them rather than pinning the one that was found.
    assert "referenced source count says 155, the catalog holds 156" in result.stdout
    assert "pin count says 155, the catalog holds 156" in result.stdout


def test_removing_an_entry_moves_the_entry_count(tree):
    """The other half of the header sentence, which was right all along.

    156 survived because it is a compiled array length. That protection lives
    in the native suite behind an eight-minute build, so this asserts the
    document's copy of it is watched here too.
    """
    catalog = load(tree)
    removed = catalog["entries"].pop()
    still_referenced = {
        path for entry in catalog["entries"] for path in entry["source_paths"]}
    catalog["source_digests"] = [
        pin for pin in catalog["source_digests"]
        if pin["source_path"] in still_referenced]
    store(tree, catalog)
    assert removed

    result = run(tree)
    assert result.returncode == 1
    assert "inventory entry count says 156, the catalog holds 155" in result.stdout


def test_a_reworded_claim_fails_rather_than_silently_unchecking(tree):
    """A pattern that stops matching must be red, not absent.

    This is the mode that turns a gate into decoration: the sentence is edited,
    the regex misses, every remaining claim passes, and the log says PASSED
    while one figure is no longer checked by anything.
    """
    document = document_of(tree).replace("There are 155 pins", "There are 155 hashes")
    assert "155 hashes" in document
    (tree / DOCUMENT).write_text(document, encoding="utf-8")

    result = run(tree)
    assert result.returncode == 1
    assert "the pin count claim matched 0 times, expected exactly one" in result.stdout
    # The correct value is still reported, so the person editing the pattern
    # does not have to go and compute it.
    assert "Its value is 155." in result.stdout


def test_a_duplicated_claim_sentence_also_fails(tree):
    """Two matches is as ambiguous as none -- the gate must not pick one."""
    document = document_of(tree).replace(
        "There are 155 pins", "There are 155 pins. There are 155 pins", 1)
    (tree / DOCUMENT).write_text(document, encoding="utf-8")

    result = run(tree)
    assert result.returncode == 1
    assert "the pin count claim matched 2 times, expected exactly one" in result.stdout


def test_pins_that_stop_covering_the_referenced_paths_fail(tree):
    """The premise "155 pins" and "155 unique source files" are the same number.

    The native loader refuses a catalog whose `source_digests` and referenced
    paths disagree, but no C++ runs in the job this gate lives in, so the
    premise is restated rather than assumed.
    """
    catalog = load(tree)
    orphan = catalog["source_digests"].pop()
    store(tree, catalog)

    result = run(tree)
    assert result.returncode == 1
    assert (f"source_digests does not pin referenced source "
            f"{orphan['source_path']}" in result.stdout)


def test_a_pin_no_entry_references_fails(tree):
    catalog = load(tree)
    catalog["source_digests"].append(
        {"source_path": "scripts/unreferenced.py",
         "source_sha256": "sha256:" + "0" * 64})
    store(tree, catalog)

    result = run(tree)
    assert result.returncode == 1
    assert ("source_digests pins scripts/unreferenced.py, which no entry "
            "references" in result.stdout)


def test_a_disposition_overlap_figure_is_checked_against_both_catalogs(tree):
    """The 65-of-129 and 68-of-165 figures are catalog measurements too.

    Filed as claims rather than left as prose for the same reason as the rest:
    they were correct when written, and nothing was evaluating them.
    """
    scope = tree / "docs/experiment-vm/source-dispositions.scripts.v1.json"
    document = json.loads(scope.read_text(encoding="utf-8"))
    document["entries"].append(
        {"source_path": "scripts/not_in_the_compatibility_catalog.py"})
    scope.write_text(json.dumps(document, indent=2), encoding="utf-8")

    result = run(tree)
    assert result.returncode == 1
    assert ("scripts disposition entry count says 129, the catalog holds 130"
            in result.stdout)
    # The overlap did not move: the added path is referenced by no entry.
    assert "dispositions also referenced here" not in result.stdout


def test_write_corrects_only_the_matched_digits(tree):
    """--write is the `print-catalog-digests --write` shape: regenerate, do not retype."""
    before = document_of(tree)
    (tree / DOCUMENT).write_text(
        before.replace("over 155 unique", "over 42 unique"), encoding="utf-8")

    written = run(tree, "--write")
    assert written.returncode == 1, "the run that had to correct the document is red"
    assert "WROTE: 1 corrected figures" in written.stdout
    assert document_of(tree) == before, "only the digits may move"

    assert run(tree).returncode == 0


def test_write_leaves_an_unmatched_claim_alone(tree):
    """A claim the gate cannot locate is a human's problem, not a splice target."""
    document = document_of(tree).replace("There are 155 pins", "There are 155 hashes")
    (tree / DOCUMENT).write_text(document, encoding="utf-8")

    result = run(tree, "--write")
    assert result.returncode == 1
    assert "WROTE" not in result.stdout
    assert document_of(tree) == document
