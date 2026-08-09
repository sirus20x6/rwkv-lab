"""What `scripts/ci_compatibility_pin_gate.py` must catch, and must not miss.

The gate's whole claim is that it cannot be green for a catalog the native
loader would refuse. A gate that recomputes both sides from the same place
always agrees with itself, so the tests below mutate the two sides
independently and require each mutation to redden a *different* assertion:

  * change the file, leave the pin      -> the pin check fails, naming the path
  * regenerate the pin                  -> PASSES (so the gate is not just red)
  * revert the file, keep the new pin   -> the pin check fails again

If the third passed, the gate would be reading the pin out of the file it is
meant to be checking.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import re
import shutil
import subprocess
import sys

import pytest

REPOSITORY = pathlib.Path(__file__).resolve().parent.parent
GATE = REPOSITORY / "scripts/ci_compatibility_pin_gate.py"
CATALOG = "docs/experiment-vm/compatibility-workflows.v1.json"
CATALOG_GLOB = "compatibility-workflows*.v1.json"
API_VERSION = "trainvm.compatibility-workflows/v1"


def _run(root: pathlib.Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(GATE), "--repository", str(root)],
        capture_output=True, text=True, check=False, cwd=str(REPOSITORY))


def _load(root: pathlib.Path) -> dict:
    return json.loads((root / CATALOG).read_text(encoding="utf-8"))


def _store(root: pathlib.Path, catalog: dict) -> None:
    (root / CATALOG).write_text(
        json.dumps(catalog, indent=2) + "\n", encoding="utf-8")


@pytest.fixture()
def sandbox(tmp_path: pathlib.Path) -> pathlib.Path:
    """A repository holding the catalog and exactly the sources it pins.

    Copying the pinned files rather than the whole tree keeps this to 155 small
    reads, and means a test that mutates one is mutating a copy -- no test here
    can touch the working tree it was launched from.
    """
    root = tmp_path / "repository"
    (root / "docs").mkdir(parents=True)
    shutil.copytree(REPOSITORY / "docs/experiment-vm",
                    root / "docs/experiment-vm")
    catalog = _load(root)
    for pin in catalog["source_digests"]:
        destination = root / pin["source_path"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPOSITORY / pin["source_path"], destination)
    return root


def _classified_source(catalog: dict) -> str:
    """A path the catalog pins, chosen deterministically."""
    return catalog["source_digests"][0]["source_path"]


def test_the_gate_passes_on_the_repository_as_checked_in() -> None:
    """If this is ever red, the working tree really has an unregenerated pin."""
    completed = _run(REPOSITORY)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert completed.stdout.strip().splitlines()[-1].startswith(
        "compatibility source pins: PASSED")


def test_the_gate_counts_every_pin_in_every_catalog_on_disk() -> None:
    """Scope stated over the tree, not over a list.

    The verdict line is the only thing most readers see, so the number in it
    has to be a measurement rather than a slogan: it must equal the pins
    actually on disk, or a catalog could go unread with the gate still
    printing a confident PASSED.
    """
    directory = REPOSITORY / "docs/experiment-vm"
    on_disk = sorted(directory.rglob(CATALOG_GLOB))
    assert on_disk, f"no {CATALOG_GLOB} under {directory}; this test is vacuous"
    pins = sum(
        len(json.loads(path.read_text(encoding="utf-8"))["source_digests"])
        for path in on_disk)

    completed = _run(REPOSITORY)
    summary = completed.stdout.strip().splitlines()[-1]
    counted = re.search(r"(\d+) per-source pins across (\d+) catalog", summary)
    assert counted is not None, f"the verdict states no tally: {summary}"
    assert int(counted.group(1)) == pins, summary
    assert int(counted.group(2)) == len(on_disk), summary


def test_editing_a_classified_source_without_regenerating_turns_it_red(
        sandbox: pathlib.Path) -> None:
    """PR #186's failure, reproduced without a compiler.

    The message must be the binary's own sentence: the point of the gate is
    that a developer greps the same string whichever instrument reported it.
    """
    catalog = _load(sandbox)
    path = _classified_source(catalog)
    (sandbox / path).write_bytes(
        (sandbox / path).read_bytes() + b"\n# drift\n")

    completed = _run(sandbox)
    assert completed.returncode != 0, completed.stdout
    assert (f"compatibility source {path} does not match its pinned bytes"
            in completed.stdout), completed.stdout


def test_regenerating_the_pin_makes_it_green_again(
        sandbox: pathlib.Path) -> None:
    """The other half of the proof: the gate is not merely always red.

    `trainvm print-catalog-digests --write` is what an author actually runs;
    what it writes for this field is the file's sha256, which is what is
    written here. Driving the built binary would make this test require an
    ~8 minute C++26 build to assert a fact about a JSON field -- the exact
    cost the gate exists to remove.
    """
    catalog = _load(sandbox)
    path = _classified_source(catalog)
    (sandbox / path).write_bytes(
        (sandbox / path).read_bytes() + b"\n# drift\n")
    assert _run(sandbox).returncode != 0

    catalog["source_digests"][0]["source_sha256"] = "sha256:" + hashlib.sha256(
        (sandbox / path).read_bytes()).hexdigest()
    _store(sandbox, catalog)

    completed = _run(sandbox)
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_reverting_the_source_while_keeping_the_new_pin_turns_it_red(
        sandbox: pathlib.Path) -> None:
    """The test that proves the two sides are read from two places.

    If the gate recomputed the pin from the file it is checking, this state --
    original bytes, regenerated pin -- would be indistinguishable from a clean
    tree and would pass.
    """
    catalog = _load(sandbox)
    path = _classified_source(catalog)
    original = (sandbox / path).read_bytes()
    (sandbox / path).write_bytes(original + b"\n# drift\n")
    catalog["source_digests"][0]["source_sha256"] = "sha256:" + hashlib.sha256(
        (sandbox / path).read_bytes()).hexdigest()
    _store(sandbox, catalog)
    assert _run(sandbox).returncode == 0

    (sandbox / path).write_bytes(original)

    completed = _run(sandbox)
    assert completed.returncode != 0, (
        f"the gate accepted a pin computed from bytes that are no longer on "
        f"disk, so it is reading the pin from the file it checks:\n"
        f"{completed.stdout}")
    assert (f"compatibility source {path} does not match its pinned bytes"
            in completed.stdout), completed.stdout


def test_a_deleted_classified_source_is_named_rather_than_skipped(
        sandbox: pathlib.Path) -> None:
    """A pin whose file is gone must fail, not fall through the hash check."""
    catalog = _load(sandbox)
    path = _classified_source(catalog)
    (sandbox / path).unlink()

    completed = _run(sandbox)
    assert completed.returncode != 0, completed.stdout
    assert f"compatibility source {path} is missing" in completed.stdout


def test_a_source_replaced_by_a_symlink_is_refused(
        sandbox: pathlib.Path) -> None:
    """The loader opens sources with RESOLVE_NO_SYMLINKS; this must agree.

    Hashing through the link would report the target's bytes, so a catalog
    could be green here and refused by the binary at startup -- a pre-flight
    that disagrees with the thing it flies ahead of is worse than none.
    """
    catalog = _load(sandbox)
    path = _classified_source(catalog)
    target = sandbox / "linked_source.py"
    target.write_bytes((sandbox / path).read_bytes())
    (sandbox / path).unlink()
    (sandbox / path).symlink_to(target)

    completed = _run(sandbox)
    assert completed.returncode != 0, (
        f"a pinned source replaced by a symlink to identical bytes was "
        f"accepted:\n{completed.stdout}")
    assert "symlink" in completed.stdout, completed.stdout


def test_an_unpinned_referenced_source_turns_it_red(
        sandbox: pathlib.Path) -> None:
    """Coverage, in the direction the loader checks it.

    Dropping a pin must not read as "one fewer thing to verify".
    """
    catalog = _load(sandbox)
    dropped = catalog["source_digests"].pop(0)["source_path"]
    _store(sandbox, catalog)

    completed = _run(sandbox)
    assert completed.returncode != 0, completed.stdout
    assert f"does not pin referenced source {dropped}" in completed.stdout


def test_a_pin_no_entry_references_turns_it_red(
        sandbox: pathlib.Path) -> None:
    """And the other direction: a pin for a path nothing classifies."""
    catalog = _load(sandbox)
    catalog["source_digests"].append({
        "source_path": "zz_unreferenced.py",
        "source_sha256": "sha256:" + "0" * 64,
    })
    _store(sandbox, catalog)

    completed = _run(sandbox)
    assert completed.returncode != 0, completed.stdout
    assert ("pins zz_unreferenced.py, which no entry references"
            in completed.stdout), completed.stdout


def test_unsorted_pins_turn_it_red(sandbox: pathlib.Path) -> None:
    """The loader compares computed pins to stored ones positionally.

    It hashes `all_source_paths` in sorted order and compares index by index,
    so an unsorted list makes it blame whichever path happens to line up.
    Refusing the ordering here keeps this gate from reporting a mismatch on a
    file that is fine.
    """
    catalog = _load(sandbox)
    catalog["source_digests"].reverse()
    _store(sandbox, catalog)

    completed = _run(sandbox)
    assert completed.returncode != 0, completed.stdout
    assert "must be sorted by source_path" in completed.stdout


def test_a_pin_that_is_not_a_lowercase_sha256_turns_it_red(
        sandbox: pathlib.Path) -> None:
    """The loader refuses the spelling before it hashes anything.

    Without this the gate compares an uppercase or truncated digest against a
    lowercase one, fails on the *bytes*, and blames a file that is fine -- the
    right verdict for the wrong reason, which sends the author to regenerate a
    pin that was never the problem.
    """
    catalog = _load(sandbox)
    path = _classified_source(catalog)
    catalog["source_digests"][0]["source_sha256"] = (
        catalog["source_digests"][0]["source_sha256"].upper())
    _store(sandbox, catalog)

    completed = _run(sandbox)
    assert completed.returncode != 0, completed.stdout
    assert f"must be lowercase sha256 for {path}" in completed.stdout, (
        completed.stdout)


def test_a_catalog_cannot_leave_the_gate_by_renaming_its_api_version(
        sandbox: pathlib.Path) -> None:
    """Recognition is filename AND api_version; disagreement is a failure."""
    catalog = _load(sandbox)
    catalog["api_version"] = "trainvm.something-else/v1"
    _store(sandbox, catalog)

    completed = _run(sandbox)
    assert completed.returncode != 0, (
        f"a compatibility catalog left this gate's scope by editing one "
        f"string:\n{completed.stdout}")
    assert "api_version" in completed.stdout


def test_a_catalog_cannot_leave_the_gate_by_being_renamed(
        sandbox: pathlib.Path) -> None:
    """The mirror case: the api_version is carried, the filename is not."""
    catalog = _load(sandbox)
    (sandbox / CATALOG).unlink()
    (sandbox / "docs/experiment-vm/compat-workflows-renamed.json").write_text(
        json.dumps(catalog, indent=2) + "\n", encoding="utf-8")

    completed = _run(sandbox)
    assert completed.returncode != 0, (
        f"a compatibility catalog left this gate's scope by being renamed:\n"
        f"{completed.stdout}")
    assert "compat-workflows-renamed.json" in completed.stdout


def test_a_directory_holding_no_catalog_fails_rather_than_passing_empty(
        sandbox: pathlib.Path) -> None:
    """Zero recognised catalogs is a failure, not a clean run.

    `PASSED -- 0 pins` is the shape of a gate that has stopped measuring
    anything while still being read as evidence.
    """
    (sandbox / CATALOG).unlink()

    completed = _run(sandbox)
    assert completed.returncode != 0, completed.stdout
    assert "would otherwise pass having checked nothing" in completed.stdout


def test_the_gate_does_not_recompute_either_fold(sandbox: pathlib.Path) -> None:
    """The design constraint, asserted rather than merely written down.

    A Python mirror of a C++ fold is a second implementation that must be kept
    in agreement with the first, and PR #180 already paid for that coupling on
    the disposition catalogs. Corrupting the two stored fold digests must leave
    this gate green: they are the native loader's business, and a future
    edit that quietly starts checking them here reintroduces the hazard.
    """
    catalog = _load(sandbox)
    assert "classification_surface_digest" in catalog
    catalog["classification_surface_digest"] = "sha256:" + "1" * 64
    catalog["source_tree_digest"] = "sha256:" + "2" * 64
    _store(sandbox, catalog)

    completed = _run(sandbox)
    assert completed.returncode == 0, (
        f"this gate now depends on a digest it does not own; either it "
        f"mirrors a C++ fold, or it reads a value the catalog must not "
        f"store:\n{completed.stdout}")
