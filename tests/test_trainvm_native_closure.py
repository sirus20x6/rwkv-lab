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
closure through the same two doors these fixtures use. A whole document does go
out of ``build()`` and back through ``verify_embedded_runtime_closure()`` here,
but over a substituted runtime: a hosted runner's Python tool cache is
group-writable and the builder refuses — correctly — to seal it. The pass over
a real interpreter is ``rwkv_lab_worker_artifact`` in the native suite, which is
excluded from hosted CI; that is exactly why the discrimination is asserted here
instead.
"""

from __future__ import annotations

import importlib.util
import json
import os
import pathlib
import struct
import sys
import zipfile

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


def build_id_note(description: bytes) -> bytes:
    """An `NT_GNU_BUILD_ID` note as /sys/module/<name>/notes/ serves it."""

    return (
        len(b"GNU\x00").to_bytes(4, sys.byteorder)
        + len(description).to_bytes(4, sys.byteorder)
        + (3).to_bytes(4, sys.byteorder)
        + b"GNU\x00"
        + description
    )


# The line this host actually serves. It ends in the user and host that
# compiled the module, which is the whole reason the identity is not this line.
_REAL_SHAPE = (
    "NVRM version: NVIDIA UNIX Open Kernel Module for x86_64  {version}  "
    "Release Build  ({builder})"
)


@pytest.fixture
def driver(tmp_path, monkeypatch):
    """A driver identity this test owns, on both copies at once.

    Both the guard and the builder are pointed at the same pair of files, so a
    test that moves one moves it for both -- and neither reads the real host,
    which would make every assertion here depend on whether the machine
    running the suite happens to have an NVIDIA driver loaded.
    """

    version = tmp_path / "nvidia-version"
    note = tmp_path / "nvidia-build-id"

    def write(*, driver_version: str | None, build: str = "root@sealed", note_bytes=b""):
        if driver_version is None:
            version.unlink(missing_ok=True)
        else:
            version.write_text(
                _REAL_SHAPE.format(version=driver_version, builder=build) + "  \n"
            )
        if note_bytes:
            note.write_bytes(note_bytes)
        else:
            note.unlink(missing_ok=True)

    for module in (builder, guard):
        monkeypatch.setattr(module, "NVIDIA_DRIVER_VERSION_FILE", str(version))
        monkeypatch.setattr(module, "NVIDIA_MODULE_BUILD_ID_FILE", str(note))
    return write


def test_a_changed_cuda_driver_identity_is_rejected(tree, driver):
    """The driver is in the kernel, so no pinned byte can carry its identity."""

    driver(driver_version="610.43.03", note_bytes=build_id_note(b"\x11" * 20))
    closure = build(tree)
    assert closure.native["cuda"]["driver_identity"] == (
        "610.43.03+gnu-build-id:" + "11" * 20
    )
    closure.verify()
    driver(driver_version="611.00.00", note_bytes=build_id_note(b"\x11" * 20))
    with pytest.raises(guard.RuntimeClosureError) as failure:
        closure.verify()
    assert "CUDA driver identity changed" in str(failure.value)


def test_a_rebuilt_driver_of_the_same_version_is_accepted(tree, driver):
    """The defect this pair replaced: a rebuild is not a driver change.

    Same version, same module bytes, different machine did the compiling. The
    v3 identity was the whole `/proc/driver/nvidia/version` line, so this
    refused the host and took every cache namespace cold with it for a
    bit-for-bit compatible driver.
    """

    driver(
        driver_version="610.43.03",
        build="root@neuromancer",
        note_bytes=build_id_note(b"\x22" * 20),
    )
    closure = build(tree)
    sealed = closure.native["cuda"]["driver_report"]
    driver(
        driver_version="610.43.03",
        build="builder@some-distribution-farm",
        note_bytes=build_id_note(b"\x22" * 20),
    )
    closure.verify()
    # The line that moved is still recorded -- it is evidence, so it is kept
    # and simply not compared.
    assert "neuromancer" in sealed
    assert sealed != guard.driver_report()


def test_a_rebuild_that_changed_the_module_is_rejected(tree, driver):
    """And the other direction, which is why the version alone is not enough.

    Same version string, different artifact: an open kernel module compiled
    with different flags is not the module the closure was sealed against, and
    the build ID is the part that notices.
    """

    driver(driver_version="610.43.03", note_bytes=build_id_note(b"\x33" * 20))
    closure = build(tree)
    closure.verify()
    driver(driver_version="610.43.03", note_bytes=build_id_note(b"\x44" * 20))
    with pytest.raises(guard.RuntimeClosureError) as failure:
        closure.verify()
    assert "CUDA driver identity changed" in str(failure.value)


def test_a_driver_identity_is_a_legal_downstream_identity(driver):
    """`fixed_public_identity` in trainvm/src/cache_namespace.cpp accepts it.

    The identity reaches the cache namespace claim through
    `compute_compatibility_digest`, and that validator allows only
    `[A-Za-z0-9._+:-]`. The v3 whole-line form contained a space, `(`, `)` and
    `@`, so it could not have been claimed at all.
    """

    allowed = set(
        "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-+:"
    )
    driver(driver_version="610.43.03", note_bytes=build_id_note(b"\x55" * 20))
    identity = guard.driver_identity()
    assert identity is not None
    assert not set(identity) - allowed
    assert set(guard.driver_report() or "") - allowed


def test_a_driver_without_a_build_id_falls_back_to_the_version(tree, driver):
    """A module linked with `--build-id=none` still has a usable identity."""

    driver(driver_version="610.43.03")
    closure = build(tree)
    assert closure.native["cuda"]["driver_identity"] == "610.43.03"
    closure.verify()
    driver(driver_version="610.43.03", note_bytes=build_id_note(b"\x66" * 20))
    with pytest.raises(guard.RuntimeClosureError) as failure:
        closure.verify()
    assert "CUDA driver identity changed" in str(failure.value)


def test_a_malformed_build_id_note_is_ignored_rather_than_trusted(driver):
    driver(driver_version="610.43.03")
    assert guard._module_build_id() is None
    note = pathlib.Path(guard.NVIDIA_MODULE_BUILD_ID_FILE)
    for broken in (
        b"",
        b"\x00" * 8,
        build_id_note(b"")[:10],
        # Right shape, wrong note type: NT_GNU_ABI_TAG, not a build ID.
        (4).to_bytes(4, sys.byteorder)
        + (16).to_bytes(4, sys.byteorder)
        + (1).to_bytes(4, sys.byteorder)
        + b"GNU\x00"
        + b"\x07" * 16,
        # A description longer than the note delivers.
        build_id_note(b"\x01" * 20)[:-4],
    ):
        note.write_bytes(broken)
        assert guard._module_build_id() is None, broken
        assert builder._module_build_id() is None, broken
        assert guard.driver_identity() == "610.43.03"


def test_a_host_that_grew_a_driver_is_rejected(tree, driver):
    driver(driver_version=None)
    closure = build(tree)
    assert closure.native["cuda"]["driver_identity"] is None
    assert closure.native["cuda"]["driver_report"] is None
    closure.verify()
    driver(driver_version="610.43.03", note_bytes=build_id_note(b"\x77" * 20))
    with pytest.raises(guard.RuntimeClosureError) as failure:
        closure.verify()
    assert "CUDA driver identity changed" in str(failure.value)


def test_the_two_driver_identity_readings_agree(driver):
    """The third thing the two copies must not drift on.

    `native_objects` has `test_the_two_import_path_walks_agree`; this is the
    same protection for the driver identity, which `card-ed669b66` moved in
    both copies at once. Extended here rather than read a third time: a third
    reading of a fact two copies already state is exactly the defect being
    fixed.
    """

    cases = [
        {"driver_version": "610.43.03", "note_bytes": build_id_note(b"\x88" * 20)},
        {"driver_version": "610.43.03", "build": "someone@elsewhere"},
        {"driver_version": "611.0", "note_bytes": build_id_note(b"\x99" * 8)},
        {"driver_version": None},
    ]
    for case in cases:
        driver(**case)
        assert builder.driver_identity() == guard.driver_identity(), case
        assert builder.driver_report() == guard.driver_report(), case
        assert builder._module_build_id() == guard._module_build_id(), case
    # An unparseable line keeps its whole self rather than dropping the driver
    # out of the identity, and both copies have to make that same choice.
    pathlib.Path(guard.NVIDIA_DRIVER_VERSION_FILE).write_text("NVRM version: rolling\n")
    assert builder.driver_identity() == guard.driver_identity() == "NVRM version: rolling"


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
        == "trainvm.python-bootstrap-runtime-closure/v4"
    )
    source = (REPOSITORY / "scripts" / "build_trainvm_worker_artifact.py").read_text()
    assert '"native",' in source, "the artifact builder must require the native section"


class _FakeDistribution:
    """Just enough of `importlib.metadata.Distribution` for `build()`."""

    def __init__(self, name: str, version: str, root: pathlib.Path, names) -> None:
        self.metadata = {"Name": name}
        self.version = version
        self.files = [pathlib.PurePath(item) for item in names]
        self._root = root

    def locate_file(self, relative):
        return self._root / relative


def _sealed_document(tmp_path, monkeypatch) -> dict:
    """A whole closure document over a tree this test owns end to end.

    The real interpreter cannot be used for this: a GitHub hosted runner's
    Python tool cache is group-writable, and the builder refuses — correctly —
    to seal a runtime that another member of the group can rewrite. Substituting
    the inputs keeps the assertion runnable on every machine while `build()`,
    the digest derivation, the ELF scan, and the guard all stay real.
    """

    runtime = tmp_path / "runtime"
    write_elf(runtime / "libsupport.so.1", soname="libsupport.so.1")
    write_elf(
        runtime / "python-fake",
        soname="python-fake",
        needed=("libsupport.so.1",),
        runpath=(str(runtime),),
    )
    (runtime / "module.py").write_text("VALUE = 1\n")
    os.chmod(runtime / "module.py", 0o644)
    roots = tmp_path / "empty-site-packages"
    roots.mkdir()
    monkeypatch.setattr(sys, "executable", str(runtime / "python-fake"))
    monkeypatch.setattr(
        builder, "_stdlib_files", lambda: [runtime / "module.py"]
    )
    monkeypatch.setattr(
        builder,
        "_distribution_closure",
        lambda roots_: [
            _FakeDistribution("demo", "1.0", runtime, ["libsupport.so.1"])
        ],
    )
    monkeypatch.setattr(builder, "_native_roots", lambda: [str(roots)])
    monkeypatch.setattr(guard, "native_roots", lambda: [str(roots)])
    monkeypatch.delenv("LD_LIBRARY_PATH", raising=False)
    return builder.build(("demo",))


def test_a_built_document_verifies_through_the_real_guard(tmp_path, monkeypatch):
    """`build()` out, `verify_embedded_runtime_closure()` in, over a zip.

    This is the assertion that fails if a native field is added to one side
    only: the guard compares the manifest's field set exactly, so a builder that
    emits a key the guard does not list — or the reverse — is rejected here.
    """

    document = _sealed_document(tmp_path, monkeypatch)
    archive = tmp_path / "worker.pyz"
    with zipfile.ZipFile(archive, "w") as package:
        package.writestr(
            guard.RUNTIME_CLOSURE_MEMBER, builder._canonical(document) + b"\n"
        )
    assert (
        guard.verify_embedded_runtime_closure(str(archive))
        == document["closure_digest"]
    )


def test_a_built_document_has_the_field_set_every_validator_demands(
    tmp_path, monkeypatch
):
    document = _sealed_document(tmp_path, monkeypatch)
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
    assert set(document["native"]) == {
        "cuda",
        "kernel_registry",
        "ld_library_path",
        "loader_configuration",
        "objects",
        "system_search",
    }
    # The interpreter is an object in its own closure, and its dependency is a
    # pinned file rather than a name.
    index = {entry["path"]: entry for entry in document["files"]}
    interpreter = os.path.realpath(sys.executable)
    objects = {item["path"]: item for item in document["native"]["objects"]}
    assert interpreter in objects and interpreter in index
    assert objects[interpreter]["dependencies"] == [
        {
            "needed": "libsupport.so.1",
            "resolved": str(tmp_path / "runtime" / "libsupport.so.1"),
        }
    ]
    assert objects[interpreter]["dependencies"][0]["resolved"] in index
    # Canonical JSON, because that is the form the digest is taken over.
    assert json.loads(builder._canonical(document).decode("utf-8")) == document


@pytest.mark.skipif(
    sys.platform != "linux", reason="the runtime closure is a Linux deployment artifact"
)
def test_the_builder_emits_a_verifiable_document_for_this_interpreter(tmp_path):
    """The same pass over the REAL interpreter, where it can be sealed at all.

    Skips rather than fails where the running Python's own runtime is group- or
    world-writable — a hosted runner's tool cache is, and refusing to seal it is
    the builder behaving correctly, not a defect in this test. The two tests
    above carry the shape assertions on every machine; this one is what
    exercises the actual stdlib, the actual loader configuration, and the actual
    CUDA driver on a host where the closure is meaningful.
    """

    roots = tmp_path / "empty-site-packages"
    roots.mkdir()
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(builder, "_native_roots", lambda: [str(roots)])
    try:
        document = builder.build(("packaging",))
    except ValueError as error:
        if "group/world writable" not in str(error):
            raise
        pytest.skip(f"this interpreter's runtime cannot be sealed: {error}")
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
