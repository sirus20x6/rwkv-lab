"""A refused runtime closure has to say which part of the tree moved.

`materialize_trainvm_worker_deployment.py` computes the runtime closure once per
materialization and refuses to replace an existing closure whose bytes differ.
That refusal is correct and stays. What it did not do was say *why* the second
computation disagreed, and the bare message —

    FileExistsError: refusing to replace changed runtime closure: <path>

— reads like nondeterminism in the digest, inside a directory that `mkdtemp`
had just created and nothing could have staled. `card-ce0c12ae` was filed on
exactly that reading.

It was not nondeterminism. Measured on the training host over the real
19,969-file / 8.1 GB tree: three consecutive closure builds produced
byte-identical documents, three consecutive materializations into a single
output directory produced byte-identical closures, and the full native suite
passed 82/82 with `rwkv_lab_worker_artifact` among them. The one observed
failure coincided with a `pacman -Syu` that rewrote 2,373 files under the
scanned `site-packages` and relinked libraries the ELF graph pins. Both closures
were accurate; they described two different trees.

So the defect worth fixing is diagnostic, not arithmetic — and it is worth
fixing precisely because this suite is excluded from hosted CI, where a red line
nobody can explain is a red line everybody learns to ignore. These tests run in
hosted CI, over documents built by hand, so the explanation is covered even
though the suite that produces it is not.
"""

from __future__ import annotations

import importlib.util
import json
import pathlib
import sys

import pytest

REPOSITORY = pathlib.Path(__file__).resolve().parent.parent
MATERIALIZER_PATH = (
    REPOSITORY / "scripts" / "materialize_trainvm_worker_deployment.py"
)


def _load_materializer():
    specification = importlib.util.spec_from_file_location(
        "materialize_trainvm_worker_deployment", MATERIALIZER_PATH
    )
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    # The materializer imports its sibling artifact builder by bare name, the
    # way it is invoked from CMake; put `scripts/` on the path as that caller
    # does rather than editing the script to suit a test.
    scripts = str(MATERIALIZER_PATH.parent)
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    specification.loader.exec_module(module)
    return module


materializer = _load_materializer()


def _document(files, *, objects=(), extensions=(), cuda=None, version="v4"):
    return {
        "api_version": f"trainvm.python-bootstrap-runtime-closure/{version}",
        "distributions": [{"name": "torch", "version": "2.9.0"}],
        "files": list(files),
        "native": {
            "cuda": cuda or {"driver_identity": "610.43.03", "sonames": []},
            "kernel_registry": {
                "digest": "sha256:0",
                "extensions": list(extensions),
                "roots": ["/usr/lib/python3.14/site-packages"],
            },
            "objects": list(objects),
        },
    }


def _entry(path, sha="sha256:aaaa", size=10):
    return {"kind": "regular", "mode": 0o644, "path": path, "sha256": sha, "size": size}


def _report(before, after) -> str:
    return materializer._closure_disagreement(
        json.dumps(before).encode(), json.dumps(after).encode()
    )


def test_a_changed_file_is_named_with_its_path():
    library = "/usr/lib/python3.14/site-packages/torch/lib/libtorch_cuda.so"
    report = _report(
        _document([_entry(library, sha="sha256:before")]),
        _document([_entry(library, sha="sha256:after")]),
    )
    assert library in report
    assert "files: 0 added, 0 removed, 1 changed" in report
    # The reader has to be told this is a moving tree, not a moving digest.
    assert "moved between the two closure computations" in report


def test_an_added_and_a_removed_file_are_distinguished():
    kept = _entry("/usr/lib/python3.14/site-packages/kept.py")
    report = _report(
        _document([kept, _entry("/usr/lib/python3.14/site-packages/gone.py")]),
        _document([kept, _entry("/usr/lib/python3.14/site-packages/new.py")]),
    )
    assert "files: 1 added, 1 removed, 0 changed" in report
    assert "added /usr/lib/python3.14/site-packages/new.py" in report
    assert "removed /usr/lib/python3.14/site-packages/gone.py" in report
    assert "kept.py" not in report


def test_an_upgrade_sized_difference_is_truncated_rather_than_dumped():
    """A `pacman -Syu` moves thousands of paths; the report stays readable."""

    before = _document([_entry(f"/usr/lib/p/{index}.py") for index in range(500)])
    after = _document(
        [_entry(f"/usr/lib/p/{index}.py", sha="sha256:new") for index in range(500)]
    )
    report = _report(before, after)
    assert "files: 0 added, 0 removed, 500 changed" in report
    named = [line for line in report.splitlines() if line.strip().startswith("changed ")]
    assert len(named) == materializer.DISAGREEMENT_EXAMPLES
    assert f"and {500 - materializer.DISAGREEMENT_EXAMPLES} more changed" in report


def test_a_relinked_library_is_reported_from_the_elf_graph():
    linked = {
        "dependencies": [{"needed": "libcudart.so.12", "resolved": "/usr/lib/libcudart.so.12"}],
        "path": "/usr/lib/python3.14/site-packages/flash_attn_2_cuda.so",
        "search": ["/usr/lib"],
        "soname": "flash_attn_2_cuda.so",
    }
    relinked = dict(linked)
    relinked["dependencies"] = [{"needed": "libcudart.so.12", "resolved": None}]
    report = _report(
        _document([], objects=[linked]), _document([], objects=[relinked])
    )
    assert "native.objects: 0 added, 0 removed, 1 changed" in report
    assert "flash_attn_2_cuda.so" in report


def test_a_changed_extension_is_reported_from_the_kernel_registry():
    path = "/usr/lib/python3.14/site-packages/demo_ext.so"
    report = _report(
        _document([], extensions=[{"path": path, "sha256": "sha256:one"}]),
        _document([], extensions=[{"path": path, "sha256": "sha256:two"}]),
    )
    assert "native.kernel_registry.extensions: 0 added, 0 removed, 1 changed" in report
    assert path in report


def test_a_moved_driver_identity_is_named():
    report = _report(
        _document([], cuda={"driver_identity": "610.43.03", "sonames": []}),
        _document([], cuda={"driver_identity": "610.44.01", "sonames": []}),
    )
    assert "native.cuda" in report
    assert "610.44.01" in report


def test_a_schema_bump_is_reported_before_anything_else():
    report = _report(
        _document([_entry("/usr/lib/a.py")], version="v3"),
        _document([_entry("/usr/lib/a.py")], version="v4"),
    )
    assert "api_version" in report.splitlines()[3]


def test_unreadable_documents_do_not_raise_out_of_the_diagnosis():
    """The report runs on a failure path; it must not replace one error with another."""

    report = materializer._closure_disagreement(b"not json", b"{}")
    assert "not both readable JSON" in report


def test_the_refusal_keeps_its_original_first_line(tmp_path):
    """The explanation is added to the refusal, not substituted for it.

    Anything already grepping for the original sentence — a log scan, another
    agent's notes — keeps working, and the closure path stays on the first line
    where a truncated log will still show it.
    """

    closure = tmp_path / "bootstrap-runtime-closure-368197e4b35d1632.json"
    message = materializer._changed_closure_message(
        closure,
        json.dumps(_document([_entry("/usr/lib/a.py")])).encode(),
        json.dumps(_document([_entry("/usr/lib/a.py", sha="sha256:moved")])).encode(),
    )
    assert message.splitlines()[0] == (
        f"refusing to replace changed runtime closure: {closure}"
    )
    assert "/usr/lib/a.py" in message
