"""A checked-in experiment document cannot name an adapter that does not exist.

`experiments/` holds the documents someone would actually launch, and nothing
validated it: `validate_experiment_documents.py` scans
`docs/experiment-vm/examples/`. Nine of the twenty-one JSON documents there name
an adapter that exists nowhere on main -- six ztok, three Qwen caption -- and
both implementations live on unmerged branches.

Those nine are preserved history rather than defects, so the gate carries a
countdown rather than refusing them. The tests are arranged around what would
make that countdown hollow:

    an unknown adapter fails           the drift, for a document nobody listed
    a listed-but-resolvable entry fails  the countdown can only shrink
    a listed-but-deleted path fails    an exemption for a file that is gone
    an empty scope fails               checking nothing is not passing
    builtin adapters resolve           trainvm.core is not in the rwkv-lab pin
    adapters nest at several depths    a path-specific reader would see none
"""

from __future__ import annotations

import json
import pathlib
import shutil
import subprocess
import sys

import pytest

REPOSITORY = pathlib.Path(__file__).resolve().parents[1]
GATE = REPOSITORY / "scripts/ci_experiment_adapter_gate.py"
PIN = "docs/experiment-vm/step-zero-arming.v1.json"
UNRUNNABLE = "docs/experiment-vm/unrunnable-experiment-documents.v1.json"


def run(repository: pathlib.Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(GATE), "--repository", str(repository)],
        capture_output=True, text=True, check=False)


@pytest.fixture
def tree(tmp_path: pathlib.Path) -> pathlib.Path:
    """A minimal repository: the real registry pin, an empty countdown, no documents.

    The real pin is copied rather than synthesised so the set of declared
    adapters is the one production resolves against; inventing it here would
    make these tests agree with a fixture instead of with the registry.
    """
    root = tmp_path / "repository"
    (root / "docs/experiment-vm").mkdir(parents=True)
    (root / "experiments").mkdir(parents=True)
    shutil.copy(REPOSITORY / PIN, root / PIN)
    write_countdown(root, [])
    return root


def write_document(root: pathlib.Path, name: str, body: dict) -> str:
    path = root / "experiments" / name
    path.write_text(json.dumps(body), encoding="utf-8")
    return f"experiments/{name}"


def write_countdown(root: pathlib.Path, documents: list[dict]) -> None:
    (root / UNRUNNABLE).write_text(
        json.dumps({"api_version": "trainvm.unrunnable-experiment-documents/v1",
                    "documents": documents}, indent=2), encoding="utf-8")


def test_the_checked_in_tree_passes():
    """The invariant this ships enforcing, against the real repository."""
    result = run(REPOSITORY)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "PASSED" in result.stdout


def test_an_unknown_adapter_fails(tree):
    write_document(tree, "new.json", {"spec": {"invoke": {"adapter": "rwkv-lab.nope"}}})
    result = run(tree)
    assert result.returncode == 1
    assert "rwkv-lab.nope" in result.stdout
    assert "does not declare" in result.stdout
    assert "FAILED" in result.stdout


def test_a_declared_adapter_passes(tree):
    """A real key from the pin, so this cannot pass by the gate checking nothing."""
    declared = json.loads((REPOSITORY / PIN).read_text())["profiles"][0]["adapter"]
    write_document(tree, "ok.json", {"spec": {"invoke": {"adapter": declared}}})
    assert run(tree).returncode == 0


def test_a_builtin_adapter_resolves(tree):
    """trainvm.core is implemented by the authority and is not in the rwkv-lab pin.

    Without this it would read as missing, and ten of the repository's documents
    name it -- the gate would have started red for the wrong reason.
    """
    write_document(tree, "core.json", {"spec": {"invoke": {"adapter": "trainvm.core"}}})
    assert run(tree).returncode == 0


def test_a_listed_document_is_tolerated(tree):
    path = write_document(tree, "old.json", {"spec": {"invoke": {"adapter": "rwkv-lab.gone"}}})
    write_countdown(tree, [{"path": path, "adapter": "rwkv-lab.gone", "why": "unmerged"}])
    result = run(tree)
    assert result.returncode == 0, result.stdout
    assert "1 recorded as preserved" in result.stdout


def test_a_listed_document_that_now_resolves_fails(tree):
    """The countdown can only shrink.

    Porting an implementation must force the entry's deletion. Otherwise a
    stale exemption sits there implying a document cannot run when it can, and
    nobody deletes it later.
    """
    declared = json.loads((REPOSITORY / PIN).read_text())["profiles"][0]["adapter"]
    path = write_document(tree, "ported.json", {"spec": {"invoke": {"adapter": declared}}})
    write_countdown(tree, [{"path": path, "adapter": declared, "why": "stale"}])
    result = run(tree)
    assert result.returncode == 1
    assert "now resolves" in result.stdout


def test_a_listed_document_that_no_longer_exists_fails(tree):
    write_countdown(tree, [{"path": "experiments/deleted.json", "adapter": "x", "why": "gone"}])
    result = run(tree)
    assert result.returncode == 1
    assert "does not exist" in result.stdout


def test_an_empty_scope_fails(tree):
    """Checking nothing is not passing; a renamed directory must not retire this."""
    result = run(tree)
    assert result.returncode == 1
    assert "nothing was checked" in result.stdout


def test_adapters_are_found_at_any_depth(tree):
    """These documents nest adapters at several depths.

    A reader keyed to one path would see none in an unexpected shape, which is
    indistinguishable from a document that names none -- the gate would pass
    while looking at nothing.
    """
    write_document(tree, "deep.json", {
        "spec": {"workflow": {"nodes": [
            {"invoke": {"training": {"adapter": "rwkv-lab.buried"}}}]}}})
    result = run(tree)
    assert result.returncode == 1
    assert "rwkv-lab.buried" in result.stdout


def test_unparseable_json_fails_rather_than_being_skipped(tree):
    (tree / "experiments/broken.json").write_text("{not json", encoding="utf-8")
    result = run(tree)
    assert result.returncode == 1
    assert "not valid JSON" in result.stdout


def test_the_shipped_countdown_holds_exactly_the_unrunnable_documents():
    """Pins the starting size so an entry cannot be added without a deliberate edit.

    Asserted as an exact count rather than `> 0`: this list only being allowed
    to shrink is the whole design, and a test that tolerates growth would not
    notice it growing.
    """
    document = json.loads((REPOSITORY / UNRUNNABLE).read_text(encoding="utf-8"))
    entries = document["documents"]
    assert len(entries) == 9
    for entry in entries:
        assert (REPOSITORY / entry["path"]).exists(), entry["path"]
        assert entry["why"].strip(), f"{entry['path']} has no stated reason"
