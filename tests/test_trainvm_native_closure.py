"""A changed shared library, extension, or driver must not verify quietly.

The Python file closure pins what ``import`` finds. What the process then
*executes* is chosen afterwards by ``ld.so``, from a DT_NEEDED list and a
search order the manifest never recorded — so an extension could be pinned byte
for byte while the ``libcudart.so.12`` behind it was repointed, shadowed, or
replaced, and the closure digest would not move.

This file is the evidence that the native half now discriminates. It is
deliberately built the way ``tests/test_engram_entry_points.py`` is: the
fixtures **construct** the condition under test rather than hoping the host
happens to be in it. Every ELF here is written byte by byte into a temporary
tree, so the assertions say the same thing on the training host — which has a
CUDA stack, an NVIDIA driver, and flash-attn — as on a hosted runner that has
none of them.

Each rejection test is a regression reintroduced on purpose:

* the SONAME target is repointed to a different file,
* a library is planted earlier in an object's search order,
* an extension's bytes change,
* an extension appears on the import path that the closure never named,
* the CUDA driver identity moves.

Presence of a digest is not the property; *discrimination* is. A test that only
asserted a native section had been computed would pass against every one of
those five.

Honest scope: the ELF fixtures are synthetic, and no real ``flash_attn``,
``bitsandbytes``, ``deepspeed`` or ``fla`` wheel is installed on any machine
this runs on. What is asserted is the mechanism that covers them — they are
ordinary compiled extensions with ordinary DT_NEEDED lists, and they enter the
closure through the same two doors these fixtures use. The end-to-end test
against a real interpreter is ``rwkv_lab_worker_artifact`` in the native suite,
which is excluded from hosted CI; that is exactly why the discrimination is
asserted here instead.
"""

from __future__ import annotations

import importlib.util
import json
import os
import pathlib
import struct
import sys

import pytest

from rwkv_lab import trainvm_runtime_guard as guard

REPOSITORY = pathlib.Path(__file__).resolve().parent.parent
BUILDER_PATH = REPOSITORY / "scripts" / "build_trainvm_runtime_closure.py"


def _load_builder():
    specification = importlib.util.spec_from_file_location(
        "build_trainvm_runtime_closure", BUILDER_PATH
    )
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


builder = _load_builder()


# ---------------------------------------------------------------------------
# A minimal ELF64 writer. Real enough for the loader's questions — SONAME,
# NEEDED, RUNPATH — and nothing else. PT_LOAD is laid out with p_vaddr equal to
# p_offset so the address-to-offset mapping the reader performs is exercised
# rather than bypassed by being trivially zero.
# ---------------------------------------------------------------------------

_PT_LOAD = 1
_PT_DYNAMIC = 2
_DT_NULL = 0
_DT_NEEDED = 1
_DT_STRTAB = 5
_DT_STRSZ = 10
_DT_SONAME = 14
_DT_RUNPATH = 29
_LOAD_BASE = 0x1000


def write_elf(
    path: pathlib.Path,
    *,
    soname: str = "",
    needed: tuple[str, ...] = (),
    runpath: tuple[str, ...] = (),
    filler: bytes = b"",
) -> pathlib.Path:
    """Write a shared object carrying exactly the dynamic entries given."""

    table = bytearray(b"\x00")
    offsets: dict[str, int] = {}

    def intern(value: str) -> int:
        if value not in offsets:
            offsets[value] = len(table)
            table.extend(value.encode("utf-8") + b"\x00")
        return offsets[value]

    entries: list[tuple[int, int]] = []
    if soname:
        entries.append((_DT_SONAME, intern(soname)))
    for name in needed:
        entries.append((_DT_NEEDED, intern(name)))
    if runpath:
        entries.append((_DT_RUNPATH, intern(":".join(runpath))))
    # DT_STRTAB and DT_STRSZ are patched below, once the layout is known.
    entries.append((_DT_STRTAB, 0))
    entries.append((_DT_STRSZ, len(table)))
    entries.append((_DT_NULL, 0))

    header_size = 64
    program_size = 56 * 2
    dynamic_offset = header_size + program_size
    dynamic_size = len(entries) * 16
    string_offset = dynamic_offset + dynamic_size
    entries = [
        (_DT_STRTAB, _LOAD_BASE + string_offset) if tag == _DT_STRTAB else (tag, value)
        for tag, value in entries
    ]
    total = string_offset + len(table) + len(filler)

    image = bytearray()
    image.extend(b"\x7fELF\x02\x01\x01\x00" + b"\x00" * 8)
    image.extend(struct.pack("<HHI", 3, 0x3E, 1))  # ET_DYN, x86-64, version 1
    image.extend(struct.pack("<QQQ", 0, header_size, 0))  # entry, phoff, shoff
    image.extend(struct.pack("<IHHHHHH", 0, header_size, 56, 2, 64, 0, 0))
    image.extend(
        struct.pack(
            "<IIQQQQQQ", _PT_LOAD, 5, 0, _LOAD_BASE, _LOAD_BASE, total, total, 0x1000
        )
    )
    image.extend(
        struct.pack(
            "<IIQQQQQQ",
            _PT_DYNAMIC,
            6,
            dynamic_offset,
            _LOAD_BASE + dynamic_offset,
            _LOAD_BASE + dynamic_offset,
            dynamic_size,
            dynamic_size,
            8,
        )
    )
    for tag, value in entries:
        image.extend(struct.pack("<qQ", tag, value))
    image.extend(table)
    image.extend(filler)
    assert len(image) == total
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(bytes(image))
    # Explicit rather than umask-dependent: the closure builder refuses a
    # group- or world-writable file, so a runner with umask 002 would otherwise
    # fail these tests for a reason that has nothing to do with them.
    os.chmod(path, 0o644)
    return path


# ---------------------------------------------------------------------------
# Fixture tree and the two halves of the pipeline driven over it.
# ---------------------------------------------------------------------------


class Closure:
    """A built native section plus the file index the guard verifies against."""

    def __init__(self, native: dict, paths: dict[str, dict]) -> None:
        self.native = native
        self.paths = paths

    def verify(self) -> None:
        # Same order and same fail-closed wrapper as
        # `verify_embedded_runtime_closure`: the file pass, then the native
        # pass, with an OSError from a vanished file surfacing as a rejection
        # rather than as a traceback out of the guard.
        try:
            for path in sorted(self.paths):
                guard._verify_entry(self.paths[path])
            guard._verify_native(self.native, self.paths)
        except guard.RuntimeClosureError:
            raise
        except (KeyError, OSError, TypeError, ValueError) as error:
            raise guard.RuntimeClosureError(
                "runtime closure verification failed closed"
            ) from error


@pytest.fixture
def tree(tmp_path, monkeypatch):
    """An import path with one extension, over a library in its own RUNPATH."""

    site = tmp_path / "site-packages"
    libraries = tmp_path / "libs"
    write_elf(
        libraries / "libdemo.so.1.0",
        soname="libdemo.so.1",
        filler=b"first implementation",
    )
    write_elf(
        libraries / "libdemo.so.2.0",
        soname="libdemo.so.1",
        filler=b"second implementation",
    )
    (libraries / "libdemo.so.1").symlink_to("libdemo.so.1.0")
    write_elf(
        site / "demo_ext.cpython-312-x86_64-linux-gnu.so",
        soname="demo_ext.so",
        needed=("libdemo.so.1",),
        runpath=(str(libraries),),
        filler=b"compiled extension v1",
    )
    monkeypatch.setattr(builder, "_native_roots", lambda: [str(site)])
    monkeypatch.setattr(guard, "native_roots", lambda: [str(site)])
    monkeypatch.delenv("LD_LIBRARY_PATH", raising=False)
    return tmp_path


def build(tree: pathlib.Path) -> Closure:
    site = tree / "site-packages"
    paths: dict[str, dict] = {}
    # Seeded the way `build()` seeds: from files already in the closure, which
    # never contains a dangling link.
    seeds = [
        str(item) for item in sorted(site.rglob("*.so")) if os.path.isfile(item)
    ]
    for seed in seeds:
        builder._add_path(paths, pathlib.Path(seed))
    native = builder._scan_native(paths, seeds)
    return Closure(native, paths)


# ---------------------------------------------------------------------------
# The ELF reader.
# ---------------------------------------------------------------------------


def test_elf_reader_recovers_soname_needed_and_runpath(tmp_path):
    path = write_elf(
        tmp_path / "object.so",
        soname="libobject.so.3",
        needed=("libone.so.1", "libtwo.so.2"),
        runpath=("/opt/one", "/opt/two"),
    )
    record = builder.elf_dynamic_path(path)
    assert record == {
        "needed": ["libone.so.1", "libtwo.so.2"],
        "rpath": [],
        "runpath": ["/opt/one", "/opt/two"],
        "soname": "libobject.so.3",
    }


def test_elf_reader_ignores_a_file_that_is_not_an_object(tmp_path):
    path = tmp_path / "plain.so"
    path.write_bytes(b"not an ELF file, but named like one" * 4)
    os.chmod(path, 0o644)
    assert builder.elf_dynamic_path(path) is None


def test_elf_reader_refuses_a_truncated_dynamic_segment(tmp_path):
    path = write_elf(tmp_path / "object.so", soname="libobject.so.1")
    image = bytearray(path.read_bytes())
    # Point DT_STRTAB at an address inside no loaded segment. Reporting this as
    # "no dependencies" would turn a corrupt object into a clean closure entry.
    image[64 + 56 * 2 + 16 : 64 + 56 * 2 + 32] = struct.pack(
        "<qQ", 5, 0xDEAD_0000
    )
    path.write_bytes(bytes(image))
    with pytest.raises(builder.ElfError):
        builder.elf_dynamic(bytes(path.read_bytes()))


# ---------------------------------------------------------------------------
# The closure verifies when nothing moved. Everything below reintroduces one
# regression against this baseline.
# ---------------------------------------------------------------------------


def test_an_unchanged_native_closure_verifies(tree):
    closure = build(tree)
    closure.verify()
    assert closure.native["objects"], "the extension and its library are objects"
    assert len(closure.native["kernel_registry"]["extensions"]) == 1


def test_a_repointed_soname_target_is_rejected(tree):
    """The classic one: same name, same search order, different file."""

    closure = build(tree)
    closure.verify()
    link = tree / "libs" / "libdemo.so.1"
    link.unlink()
    link.symlink_to("libdemo.so.2.0")
    with pytest.raises(guard.RuntimeClosureError) as failure:
        closure.verify()
    assert "symlink target changed" in str(failure.value)


def test_a_replaced_library_is_rejected(tree):
    closure = build(tree)
    closure.verify()
    write_elf(
        tree / "libs" / "libdemo.so.1.0",
        soname="libdemo.so.1",
        filler=b"third implementation",
    )
    with pytest.raises(guard.RuntimeClosureError) as failure:
        closure.verify()
    assert "content changed" in str(failure.value)


def test_a_shadowing_library_is_rejected(tree, monkeypatch):
    """No pinned byte changes here, and the loaded library still changes.

    A directory ahead of the recorded one in the object's search order gains a
    file with the same SONAME. Every digest in the manifest still matches; only
    re-running the resolution notices.
    """

    shadow = tree / "shadow"
    shadow.mkdir()
    monkeypatch.setenv("LD_LIBRARY_PATH", str(shadow))
    closure = build(tree)
    closure.verify()
    write_elf(
        shadow / "libdemo.so.1", soname="libdemo.so.1", filler=b"shadow implementation"
    )
    with pytest.raises(guard.RuntimeClosureError) as failure:
        closure.verify()
    assert "shared library resolution changed" in str(failure.value)


def test_a_removed_library_is_rejected(tree):
    closure = build(tree)
    closure.verify()
    (tree / "libs" / "libdemo.so.1").unlink()
    with pytest.raises(guard.RuntimeClosureError):
        closure.verify()


def test_a_changed_search_order_environment_is_rejected(tree, monkeypatch):
    closure = build(tree)
    closure.verify()
    monkeypatch.setenv("LD_LIBRARY_PATH", str(tree / "elsewhere"))
    with pytest.raises(guard.RuntimeClosureError) as failure:
        closure.verify()
    assert "LD_LIBRARY_PATH changed" in str(failure.value)


def test_a_changed_extension_is_rejected(tree):
    closure = build(tree)
    closure.verify()
    write_elf(
        tree / "site-packages" / "demo_ext.cpython-312-x86_64-linux-gnu.so",
        soname="demo_ext.so",
        needed=("libdemo.so.1",),
        runpath=(str(tree / "libs"),),
        # Same length as the original, so what is caught is the content digest
        # and not the size — a size check alone would pass a same-size swap.
        filler=b"compiled extension v2",
    )
    with pytest.raises(guard.RuntimeClosureError) as failure:
        closure.verify()
    assert "content changed" in str(failure.value)


def test_an_unrecorded_extension_on_the_import_path_is_rejected(tree):
    """The one a re-verification of recorded entries alone cannot see.

    Nothing the manifest names has changed. A new compiled extension is simply
    sitting on the import path, importable, registering whatever it registers.
    """

    closure = build(tree)
    closure.verify()
    write_elf(
        tree / "site-packages" / "other_ext.cpython-312-x86_64-linux-gnu.so",
        soname="other_ext.so",
    )
    with pytest.raises(guard.RuntimeClosureError) as failure:
        closure.verify()
    assert "does not match the import path" in str(failure.value)


def test_a_removed_extension_is_rejected(tree):
    closure = build(tree)
    closure.verify()
    (tree / "site-packages" / "demo_ext.cpython-312-x86_64-linux-gnu.so").unlink()
    with pytest.raises(guard.RuntimeClosureError):
        closure.verify()


def test_a_changed_cuda_driver_identity_is_rejected(tree, tmp_path, monkeypatch):
    """The driver is in the kernel, so no pinned byte can carry its identity."""

    version = tmp_path / "nvidia-version"
    version.write_text("NVRM version: 610.43.03\n")
    monkeypatch.setattr(builder, "NVIDIA_DRIVER_VERSION_FILE", str(version))
    monkeypatch.setattr(guard, "NVIDIA_DRIVER_VERSION_FILE", str(version))
    closure = build(tree)
    assert closure.native["cuda"]["driver_version"] == "NVRM version: 610.43.03"
    closure.verify()
    version.write_text("NVRM version: 611.00.00\n")
    with pytest.raises(guard.RuntimeClosureError) as failure:
        closure.verify()
    assert "CUDA driver identity changed" in str(failure.value)


def test_a_host_that_grew_a_driver_is_rejected(tree, tmp_path, monkeypatch):
    absent = tmp_path / "no-such-driver"
    monkeypatch.setattr(builder, "NVIDIA_DRIVER_VERSION_FILE", str(absent))
    monkeypatch.setattr(guard, "NVIDIA_DRIVER_VERSION_FILE", str(absent))
    closure = build(tree)
    assert closure.native["cuda"]["driver_version"] is None
    closure.verify()
    absent.write_text("NVRM version: 610.43.03\n")
    with pytest.raises(guard.RuntimeClosureError) as failure:
        closure.verify()
    assert "CUDA driver identity changed" in str(failure.value)


def test_cuda_dependencies_are_named_in_the_receipt(tmp_path, monkeypatch):
    """flash-attn, bitsandbytes and DeepSpeed enter through this door.

    None of them is installed on any machine this test runs on, so what is
    asserted is the classification an extension of theirs would receive: a
    DT_NEEDED on the CUDA runtime is recorded as a CUDA identity rather than as
    an anonymous library, whether or not it resolves on this host.
    """

    site = tmp_path / "site-packages"
    write_elf(
        site / "flash_attn_2_cuda.cpython-312-x86_64-linux-gnu.so",
        soname="flash_attn_2_cuda.so",
        needed=("libcudart.so.12", "libc.so.6"),
    )
    monkeypatch.setattr(builder, "_native_roots", lambda: [str(site)])
    monkeypatch.delenv("LD_LIBRARY_PATH", raising=False)
    closure = build(tmp_path)
    assert closure.native["cuda"]["sonames"] == ["libcudart.so.12"]


# ---------------------------------------------------------------------------
# Cold, as distinct from rejected. The guard refuses a changed runtime; the
# digest is what makes a changed runtime ineligible to reuse a compiler cache.
# ---------------------------------------------------------------------------


def _digest_of(closure: Closure) -> str:
    body = {
        "api_version": builder.SCHEMA,
        "distributions": [],
        "files": [closure.paths[name] for name in sorted(closure.paths)],
        "native": closure.native,
        "python": {},
        "root_distributions": [],
    }
    return builder._digest(builder._canonical(body))


def test_a_changed_library_makes_the_closure_digest_cold(tree):
    """`closure_digest` is `bootstrap_runtime_closure_fingerprint` downstream.

    `trainvm/src/cache_namespace_authority.cpp` requires the cache probe's
    `runtime_closure_fingerprint` to equal the sealed launch's, and that value
    is this digest — so a digest that moves is a cache namespace that cannot be
    reused, which is what "cold" means here.
    """

    before = _digest_of(build(tree))
    link = tree / "libs" / "libdemo.so.1"
    link.unlink()
    link.symlink_to("libdemo.so.2.0")
    assert _digest_of(build(tree)) != before


def test_an_unchanged_tree_rebuilds_to_the_same_digest(tree):
    assert _digest_of(build(tree)) == _digest_of(build(tree))


# ---------------------------------------------------------------------------
# Builder and guard hold separate copies of the import-path walk, because the
# builder runs inside a sealed interpreter that cannot import this package and
# the guard may import nothing outside the standard library. Nothing but this
# test stops the two from drifting apart.
# ---------------------------------------------------------------------------


def test_the_two_import_path_walks_agree(tmp_path):
    site = tmp_path / "site-packages"
    write_elf(site / "one.cpython-312-x86_64-linux-gnu.so", soname="one.so")
    write_elf(site / "nested" / "libbundled.so.4", soname="libbundled.so.4")
    write_elf(site / "__pycache__" / "hidden.so", soname="hidden.so")
    (site / "dangling.so").symlink_to("nowhere.so")
    (site / "notes.txt").write_text("not an object")
    discovered = builder.native_objects([str(site)])
    assert discovered == guard.native_objects([str(site)])
    assert str(site / "__pycache__" / "hidden.so") not in discovered
    assert str(site / "notes.txt") not in discovered
    assert str(site / "dangling.so") in discovered
    assert str(site / "nested" / "libbundled.so.4") in discovered


def test_a_dangling_import_path_object_that_starts_resolving_is_rejected(
    tmp_path, monkeypatch
):
    site = tmp_path / "site-packages"
    site.mkdir(parents=True)
    (site / "late.so").symlink_to("arriving.so")
    monkeypatch.setattr(builder, "_native_roots", lambda: [str(site)])
    monkeypatch.setattr(guard, "native_roots", lambda: [str(site)])
    monkeypatch.delenv("LD_LIBRARY_PATH", raising=False)
    closure = build(tmp_path)
    assert closure.native["kernel_registry"]["extensions"] == [
        {"path": str(site / "late.so"), "sha256": None}
    ]
    closure.verify()
    write_elf(site / "arriving.so", soname="arriving.so")
    with pytest.raises(guard.RuntimeClosureError) as failure:
        closure.verify()
    assert "became loadable" in str(failure.value)


# ---------------------------------------------------------------------------
# Wiring. The native section has to survive the whole document, not only the
# two functions above: schema, exact field sets, canonical bytes, and the three
# independent copies of the manifest validation.
# ---------------------------------------------------------------------------


def test_every_manifest_validator_agrees_on_the_field_set():
    artifact_specification = importlib.util.spec_from_file_location(
        "build_trainvm_worker_artifact",
        REPOSITORY / "scripts" / "build_trainvm_worker_artifact.py",
    )
    assert artifact_specification is not None and artifact_specification.loader
    artifact = importlib.util.module_from_spec(artifact_specification)
    artifact_specification.loader.exec_module(artifact)
    assert (
        builder.SCHEMA
        == guard.RUNTIME_CLOSURE_SCHEMA
        == artifact.RUNTIME_CLOSURE_SCHEMA
        == "trainvm.python-bootstrap-runtime-closure/v3"
    )
    source = (REPOSITORY / "scripts" / "build_trainvm_worker_artifact.py").read_text()
    assert '"native",' in source, "the artifact builder must require the native section"


@pytest.mark.skipif(
    sys.platform != "linux", reason="the runtime closure is a Linux deployment artifact"
)
def test_the_builder_emits_a_verifiable_document_for_this_interpreter(tmp_path):
    """One end-to-end pass over the real interpreter, not a fixture tree.

    Everything above drives `_scan_native` and `_verify_native` directly. This
    one runs `build()` over the running Python and checks the document it emits
    is the shape the guard's field-set literals demand — which is the assertion
    that fails if a native field is added to one side only.
    """

    # The import-path inventory is pointed at an empty directory here. Walking
    # and digesting a development site-packages costs minutes and asserts
    # nothing this test is about; the tree fixtures above cover the registry.
    # Everything else — the stdlib closure, the interpreter's own ELF graph,
    # DT_NEEDED resolution against the real loader configuration, the CUDA
    # driver identity — runs against this machine as it is.
    roots = tmp_path / "empty-site-packages"
    roots.mkdir()
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(builder, "_native_roots", lambda: [str(roots)])
    try:
        document = builder.build(("packaging",))
    finally:
        monkeypatch.undo()
    assert set(document) == {
        "api_version",
        "closure_digest",
        "distributions",
        "files",
        "native",
        "python",
        "root_distributions",
    }
    assert document["api_version"] == guard.RUNTIME_CLOSURE_SCHEMA
    body = dict(document)
    body.pop("closure_digest")
    assert document["closure_digest"] == builder._digest(builder._canonical(body))
    native = document["native"]
    assert set(native) == {
        "cuda",
        "kernel_registry",
        "ld_library_path",
        "loader_configuration",
        "objects",
        "system_search",
    }
    # The interpreter is an object in its own closure, and its dependencies are
    # pinned files rather than names.
    index = {entry["path"]: entry for entry in document["files"]}
    assert any(
        os.path.realpath(sys.executable) == item["path"] for item in native["objects"]
    )
    for item in native["objects"]:
        assert item["path"] in index
        for dependency in item["dependencies"]:
            assert dependency["resolved"] is None or dependency["resolved"] in index
    # Canonical JSON, because that is the form the digest is taken over.
    assert json.loads(builder._canonical(document).decode("utf-8")) == document
