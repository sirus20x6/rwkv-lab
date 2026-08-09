#!/usr/bin/env python3
"""Build the deterministic pre-dispatch runtime closure for TrainVM workers.

The closure started as a Python file closure: the interpreter's standard
library plus every installed file of a transitive distribution set, each pinned
by size, mode and content digest. That is enough to pin what Python *imports*
and nothing about what the dynamic loader then pulls in underneath it.

A compiled extension is a Python file only in the sense that ``import`` finds
it. What it executes is decided afterwards by ``ld.so``, from the extension's
own ``DT_NEEDED`` list and a search order the manifest never recorded. So
``flash_attn_2_cuda...so`` could be pinned byte for byte while the
``libcudart.so.12`` it binds to was repointed, replaced, or shadowed by a
directory earlier in its search order, and the closure digest would not move.

This module closes that gap. It walks the ELF graph reachable from the closure
(``objects``), resolves every ``DT_NEEDED`` name through the same ordered search
the loader would use, and pins each resolution — so the SONAME target is a
manifest entry with its own digest, and repointing a ``libcudart.so.12``
symlink is a rejected identity rather than an invisible one. It records the
CUDA driver identity separately, because the driver is not a file the closure
can own. And it inventories every native object reachable on the import path as
a kernel registry, so an extension appearing, vanishing, or being relinked
moves ``kernel_registry_digest``.

All of it folds into ``closure_digest``, which is the value the deployment
carries as ``bootstrap_runtime_closure_fingerprint`` and which
``CacheNamespaceClaimEvidence.runtime_closure_fingerprint`` binds. So a changed
library is cold for compiler/JIT cache reuse by construction, and
``rwkv_lab.trainvm_runtime_guard`` rejects it before the worker imports any
third-party code.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import importlib.metadata
import json
import mmap
import os
import stat
import struct
import sys
import sysconfig
import tempfile
from collections import deque
from pathlib import Path
from typing import Any

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

SCHEMA = "trainvm.python-bootstrap-runtime-closure/v3"
DEFAULT_ROOT_DISTRIBUTIONS = (
    "grpcio",
    "pillow",
    "protobuf",
    "torch",
)

# Reading the ELF header of every closure file is affordable (the builder
# already reads each file in full to digest it), but a malformed or hostile
# file must not be able to make the scan unbounded.
MAXIMUM_DYNAMIC_ENTRIES = 4096
MAXIMUM_NEEDED = 512
MAXIMUM_SEARCH_DIRECTORIES = 256
MAXIMUM_OBJECTS = 8192
MAXIMUM_NATIVE_OBJECTS = 32768

# The default trusted directories glibc consults after the cache. Recorded
# explicitly rather than assumed, so the guard re-walks the same list.
DEFAULT_SYSTEM_SEARCH = ("/lib64", "/usr/lib64", "/lib", "/usr/lib")

# A file on the import path that the dynamic loader can load. Extension modules
# end in `.so` after their ABI tag; bundled shared libraries carry a version
# suffix after it.
NATIVE_SUFFIXES = (".so",)

# Sonames whose identity is the compute stack rather than ordinary system
# libraries. Classification only: every one of them is pinned as a file like
# any other, and this list decides what the receipt calls out by name.
CUDA_SONAME_PREFIXES = (
    "libcublas",
    "libcuda.so",
    "libcudart",
    "libcudnn",
    "libcufft",
    "libcufile",
    "libcurand",
    "libcusolver",
    "libcusparse",
    "libnccl",
    "libnvJitLink",
    "libnvidia-",
    "libnvrtc",
    "libnvToolsExt",
)

NVIDIA_DRIVER_VERSION_FILE = "/proc/driver/nvidia/version"


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")


def _digest(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            value.update(block)
    return "sha256:" + value.hexdigest()


def _entry(path: Path) -> dict[str, Any]:
    path = Path(os.path.abspath(path))
    metadata = path.lstat()
    mode = stat.S_IMODE(metadata.st_mode)
    if stat.S_ISLNK(metadata.st_mode):
        return {
            "kind": "symlink",
            "mode": mode,
            "path": str(path),
            "target": os.readlink(path),
        }
    if mode & (stat.S_IWGRP | stat.S_IWOTH):
        raise ValueError(f"runtime closure path is group/world writable: {path}")
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"runtime closure path is not regular or symlink: {path}")
    return {
        "kind": "regular",
        "mode": mode,
        "path": str(path),
        "sha256": _sha256(path),
        "size": metadata.st_size,
    }


def _add_path(paths: dict[str, dict[str, Any]], path: Path) -> None:
    path = Path(os.path.abspath(path))
    if path.is_dir():
        return
    entry = _entry(path)
    previous = paths.setdefault(str(path), entry)
    if previous != entry:
        raise ValueError(f"runtime closure path changed while scanning: {path}")
    if entry["kind"] == "symlink":
        target = path.resolve(strict=True)
        _add_path(paths, target)


def _distribution_closure(
    root_distributions: tuple[str, ...],
) -> list[importlib.metadata.Distribution]:
    pending = deque(root_distributions)
    selected: dict[str, importlib.metadata.Distribution] = {}
    while pending:
        requested = pending.popleft()
        if requested in selected:
            continue
        distribution = importlib.metadata.distribution(requested)
        name = canonicalize_name(distribution.metadata["Name"])
        selected[name] = distribution
        for raw in distribution.requires or ():
            requirement = Requirement(raw)
            if requirement.marker is not None and not requirement.marker.evaluate():
                continue
            dependency = canonicalize_name(requirement.name)
            if dependency not in selected:
                pending.append(dependency)
    return [selected[name] for name in sorted(selected)]


def _stdlib_files() -> list[Path]:
    stdlib = Path(sysconfig.get_paths()["stdlib"]).resolve(strict=True)
    purelib = Path(sysconfig.get_paths()["purelib"]).resolve(strict=True)
    platlib = Path(sysconfig.get_paths()["platlib"]).resolve(strict=True)
    excluded = {purelib, platlib}
    files: list[Path] = []
    for root, directories, names in os.walk(stdlib, followlinks=False):
        root_path = Path(root)
        directories[:] = sorted(
            directory
            for directory in directories
            if root_path / directory not in excluded
            and directory not in {"__pycache__", "site-packages", "dist-packages"}
        )
        for name in sorted(names):
            if name.endswith((".pyc", ".pyo")):
                continue
            path = root_path / name
            if path.is_file() or path.is_symlink():
                files.append(path)
    library = sysconfig.get_config_var("LDLIBRARY")
    library_directory = sysconfig.get_config_var("LIBDIR")
    if library and library_directory:
        candidate = Path(library_directory) / library
        if candidate.exists():
            files.append(candidate)
    if sys.prefix != sys.base_prefix:
        configuration = Path(sys.prefix) / "pyvenv.cfg"
        if configuration.is_file():
            files.append(configuration)
    return files


class ElfError(ValueError):
    pass


def _unpack(order: str, layout: str, data: bytes, offset: int) -> tuple[Any, ...]:
    size = struct.calcsize(order + layout)
    if offset < 0 or offset + size > len(data):
        raise ElfError("ELF structure runs past the end of the file")
    return struct.unpack_from(order + layout, data, offset)


def _elf_strings(data: bytes, offset: int, size: int) -> bytes:
    if offset < 0 or size < 0 or offset + size > len(data):
        raise ElfError("ELF string table runs past the end of the file")
    return data[offset : offset + size]


def _elf_string(table: bytes, index: int) -> str:
    if index < 0 or index >= len(table):
        raise ElfError("ELF string index is outside its table")
    end = table.find(b"\x00", index)
    if end < 0:
        raise ElfError("ELF string is unterminated")
    return table[index:end].decode("utf-8", errors="surrogateescape")


def elf_dynamic(data: bytes) -> dict[str, Any] | None:
    """Read SONAME, NEEDED, RPATH and RUNPATH out of an ELF image.

    Returns None when the bytes are not an ELF object or carry no dynamic
    segment — a static binary and a data file are both "nothing to resolve".
    A file that *is* ELF but whose dynamic segment is malformed raises, because
    silently treating it as dependency-free is the failure this exists to stop.
    """

    if len(data) < 64 or data[:4] != b"\x7fELF":
        return None
    bitness = data[4]
    endianness = data[5]
    if bitness not in (1, 2) or endianness not in (1, 2):
        raise ElfError("ELF identification is not a supported class or encoding")
    order = "<" if endianness == 1 else ">"
    if bitness == 2:
        (phoff,) = _unpack(order, "Q", data, 0x20)
        (phentsize, phnum) = _unpack(order, "HH", data, 0x36)
        phlayout = "IIQQQQQQ"
        offset_index, vaddr_index, filesz_index = 2, 3, 5
        dynlayout = "qQ"
    else:
        (phoff,) = _unpack(order, "I", data, 0x1C)
        (phentsize, phnum) = _unpack(order, "HH", data, 0x2A)
        phlayout = "IIIIIIII"
        offset_index, vaddr_index, filesz_index = 1, 2, 4
        dynlayout = "iI"
    if phnum == 0 or phentsize < struct.calcsize(order + phlayout):
        return None
    loads: list[tuple[int, int, int]] = []
    dynamic: tuple[int, int] | None = None
    for index in range(min(phnum, 0xFFFF)):
        header = _unpack(order, phlayout, data, phoff + index * phentsize)
        kind = header[0]
        offset = header[offset_index]
        vaddr = header[vaddr_index]
        filesz = header[filesz_index]
        if kind == 1:  # PT_LOAD
            loads.append((vaddr, offset, filesz))
        elif kind == 2:  # PT_DYNAMIC
            dynamic = (offset, filesz)
    if dynamic is None:
        return None

    def to_offset(address: int) -> int:
        for vaddr, offset, filesz in loads:
            if vaddr <= address < vaddr + filesz:
                return offset + (address - vaddr)
        raise ElfError("ELF dynamic address is not inside any loaded segment")

    entry_size = struct.calcsize(order + dynlayout)
    dynamic_offset, dynamic_size = dynamic
    strtab_address: int | None = None
    strtab_size: int | None = None
    entries: list[tuple[int, int]] = []
    for index in range(min(dynamic_size // entry_size, MAXIMUM_DYNAMIC_ENTRIES)):
        tag, value = _unpack(order, dynlayout, data, dynamic_offset + index * entry_size)
        if tag == 0:  # DT_NULL
            break
        if tag == 5:  # DT_STRTAB
            strtab_address = value
        elif tag == 10:  # DT_STRSZ
            strtab_size = value
        entries.append((tag, value))
    if strtab_address is None or strtab_size is None:
        raise ElfError("ELF dynamic segment declares no string table")
    table = _elf_strings(data, to_offset(strtab_address), strtab_size)
    needed: list[str] = []
    soname = ""
    rpath: list[str] = []
    runpath: list[str] = []
    for tag, value in entries:
        if tag == 1:  # DT_NEEDED
            if len(needed) >= MAXIMUM_NEEDED:
                raise ElfError("ELF object declares an unbounded dependency list")
            needed.append(_elf_string(table, value))
        elif tag == 14:  # DT_SONAME
            soname = _elf_string(table, value)
        elif tag == 15:  # DT_RPATH
            rpath.extend(_elf_string(table, value).split(":"))
        elif tag == 29:  # DT_RUNPATH
            runpath.extend(_elf_string(table, value).split(":"))
    if any("\x00" in name or not name for name in needed):
        raise ElfError("ELF dependency name is empty or embeds a NUL")
    return {
        "needed": needed,
        "rpath": [item for item in rpath if item],
        "runpath": [item for item in runpath if item],
        "soname": soname,
    }


def _expand_tokens(entry: str, origin: str, machine: str, library: str) -> str | None:
    """Expand the loader's dynamic-string tokens, or refuse the entry.

    An entry carrying a token this builder does not model would otherwise be
    recorded as a literal directory that never exists, quietly turning a real
    search directory into a no-op and hiding whatever the loader actually
    picked. Refusing is the honest outcome.
    """

    for token, replacement in (
        ("$ORIGIN", origin),
        ("${ORIGIN}", origin),
        ("$LIB", library),
        ("${LIB}", library),
        ("$PLATFORM", machine),
        ("${PLATFORM}", machine),
    ):
        entry = entry.replace(token, replacement)
    if "$" in entry:
        return None
    return os.path.normpath(entry)


def _system_search() -> tuple[list[str], list[str]]:
    """The trusted directories, read from ld.so.conf rather than assumed.

    Returns the ordered directories and the configuration files they were read
    from. The guard does not recompute this list; the configuration files are
    pinned into the closure instead, so an edit to `/etc/ld.so.conf` or one of
    its includes — which is how a directory gets inserted ahead of another —
    is a content change on a manifest entry rather than something the guard
    would have to re-derive and could disagree about.

    This deliberately reads the configuration and not ``/etc/ld.so.cache``. The
    cache is a derived index whose contents can disagree with the directories
    on disk between an install and an ``ldconfig`` run, and the property the
    guard needs is "which directories are consulted, in what order" — which is
    the configuration, not the index. The consequence, stated rather than
    hidden: a library reachable only through a stale cache entry resolves here
    as absent.
    """

    directories: list[str] = []
    seen: set[str] = set()

    def add(candidate: str) -> None:
        normalized = os.path.normpath(candidate)
        if (
            normalized.startswith("/")
            and normalized not in seen
            and len(directories) < MAXIMUM_SEARCH_DIRECTORIES
        ):
            seen.add(normalized)
            directories.append(normalized)

    pending = deque(["/etc/ld.so.conf"])
    visited: set[str] = set()
    configurations: list[str] = []
    while pending:
        configuration = pending.popleft()
        if configuration in visited or len(visited) > MAXIMUM_SEARCH_DIRECTORIES:
            continue
        visited.add(configuration)
        try:
            text = Path(configuration).read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        configurations.append(configuration)
        for raw in text.splitlines():
            line = raw.split("#", 1)[0].strip()
            if not line:
                continue
            if line.startswith("include "):
                pattern = line[len("include ") :].strip()
                if not pattern.startswith("/"):
                    pattern = os.path.join(
                        os.path.dirname(configuration), pattern
                    )
                pending.extend(sorted(glob.glob(pattern)))
                continue
            add(line)
    for default in DEFAULT_SYSTEM_SEARCH:
        add(default)
    return directories, sorted(configurations)


def _object_search(
    record: dict[str, Any],
    path: Path,
    ld_library_path: tuple[str, ...],
    system: tuple[str, ...],
) -> list[str]:
    """The ordered directories the loader would consult for this object.

    glibc's order is DT_RPATH (only when DT_RUNPATH is absent), then
    LD_LIBRARY_PATH, then DT_RUNPATH, then the trusted directories.
    """

    origin = str(path.parent)
    machine = os.uname().machine
    library = "lib64" if struct.calcsize("P") == 8 else "lib"
    directories: list[str] = []
    seen: set[str] = set()

    def extend(entries: tuple[str, ...] | list[str]) -> None:
        for entry in entries:
            expanded = _expand_tokens(entry, origin, machine, library)
            if (
                expanded is None
                or not expanded.startswith("/")
                or expanded in seen
                or len(directories) >= MAXIMUM_SEARCH_DIRECTORIES
            ):
                continue
            seen.add(expanded)
            directories.append(expanded)

    if not record["runpath"]:
        extend(record["rpath"])
    extend(ld_library_path)
    extend(record["runpath"])
    extend(system)
    return directories


def _resolve(search: list[str], needed: str) -> str | None:
    """First directory in loader order that holds `needed`, as the loader sees it.

    Returned as ``<directory>/<soname>`` — the path the loader opens, not its
    fully resolved target. Recording the link rather than the target is what
    makes a repointed ``libcudart.so.12`` symlink visible: the manifest pins the
    link's target string, and following it also pins the file it names.
    """

    if "/" in needed:
        return needed if os.path.lexists(needed) else None
    for directory in search:
        candidate = os.path.join(directory, needed)
        if os.path.lexists(candidate):
            return candidate
    return None


def _native_roots() -> list[str]:
    roots: list[str] = []
    for name in ("purelib", "platlib"):
        try:
            root = str(Path(sysconfig.get_paths()[name]).resolve(strict=True))
        except (KeyError, OSError):
            continue
        if root not in roots:
            roots.append(root)
    return sorted(roots)


def native_objects(roots: list[str]) -> list[str]:
    """Every loadable native object on the import path, whoever installed it.

    Scoped to the directories rather than to the closure's distributions on
    purpose. An extension that no distribution claims is exactly the one worth
    noticing: it is importable, it registers whatever kernels it registers, and
    a per-distribution inventory would not see it at all.
    """

    found: list[str] = []
    for root in roots:
        for directory, subdirectories, names in os.walk(root, followlinks=False):
            subdirectories[:] = sorted(
                name for name in subdirectories if name != "__pycache__"
            )
            for name in sorted(names):
                if not any(
                    name.endswith(suffix) or f"{suffix}." in name
                    for suffix in NATIVE_SUFFIXES
                ):
                    continue
                candidate = os.path.join(directory, name)
                if os.path.isfile(candidate) or os.path.islink(candidate):
                    found.append(candidate)
                    if len(found) > MAXIMUM_NATIVE_OBJECTS:
                        raise ValueError(
                            "native object inventory is unbounded on the import path"
                        )
    return sorted(set(found))


def driver_version() -> str | None:
    """The NVIDIA kernel driver identity, which no file digest can pin.

    The driver is loaded into the kernel, not installed into the closure, so a
    driver upgrade changes what every CUDA library binds to while leaving every
    pinned byte identical. Recorded here so the guard can compare it, and null
    on a host with no NVIDIA driver so the absence is itself pinned.
    """

    try:
        with open(NVIDIA_DRIVER_VERSION_FILE, encoding="utf-8", errors="replace") as f:
            line = f.readline(4096).strip()
    except OSError:
        return None
    return line or None


def _is_cuda(soname: str) -> bool:
    name = os.path.basename(soname)
    return any(name.startswith(prefix) for prefix in CUDA_SONAME_PREFIXES)


def elf_dynamic_path(path: Path) -> dict[str, Any] | None:
    """`elf_dynamic` over a file, without reading a multi-gigabyte library in.

    ``libtorch_cuda.so`` is over a gigabyte; reading each candidate whole to
    look at its first four bytes would cost more memory than the rest of the
    build put together. The magic test is four bytes and the mapping is lazy.
    """

    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC)
    except OSError:
        return None
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_size < 64:
            return None
        if os.pread(descriptor, 4, 0) != b"\x7fELF":
            return None
        with mmap.mmap(descriptor, 0, prot=mmap.PROT_READ) as data:
            return elf_dynamic(data)
    except (OSError, ValueError):
        return None
    finally:
        os.close(descriptor)


def _scan_native(
    paths: dict[str, dict[str, Any]], seeds: list[str]
) -> dict[str, Any]:
    ld_library_path = tuple(
        entry
        for entry in os.environ.get("LD_LIBRARY_PATH", "").split(":")
        if entry
    )
    directories, configurations = _system_search()
    system = tuple(directories)
    for configuration in configurations:
        _add_path(paths, Path(configuration))
    roots = _native_roots()
    registry = native_objects(roots)
    pending = deque(os.path.abspath(seed) for seed in [*seeds, *registry])
    objects: dict[str, dict[str, Any]] = {}
    cuda_sonames: set[str] = set()
    while pending:
        key = pending.popleft()
        if key in objects:
            continue
        record = elf_dynamic_path(Path(key))
        if record is None:
            continue
        if len(objects) >= MAXIMUM_OBJECTS:
            raise ValueError("runtime closure ELF graph is unbounded")
        search = _object_search(record, Path(key), ld_library_path, system)
        dependencies = []
        for needed in sorted(set(record["needed"])):
            resolved = _resolve(search, needed)
            if _is_cuda(needed):
                cuda_sonames.add(os.path.basename(needed))
            dependencies.append({"needed": needed, "resolved": resolved})
            if resolved is not None:
                # Pin the path the loader opens AND the file it ends at. They
                # differ whenever a search directory is itself a symlink
                # (/lib64 -> usr/lib), where lstat sees a regular file under
                # the un-resolved name and no link entry records the hop.
                _add_path(paths, Path(resolved))
                target = os.path.realpath(resolved)
                if target != os.path.abspath(resolved):
                    _add_path(paths, Path(target))
                pending.append(target)
        objects[key] = {
            "dependencies": dependencies,
            "path": key,
            "search": search,
            "soname": record["soname"],
        }
    extensions = []
    for item in registry:
        entry = paths.get(item)
        if entry is not None and entry["kind"] == "regular":
            extensions.append({"path": item, "sha256": entry["sha256"]})
        elif os.path.isfile(item):
            # Not claimed by any closure distribution, and importable anyway.
            # Pinned here rather than merely listed, so its bytes are as
            # answerable as a distribution's own.
            extensions.append({"path": item, "sha256": _sha256(Path(item))})
        else:
            # A dangling symlink, or a link to a directory. It is on the import
            # path and it loads nothing today; a null digest pins exactly that,
            # so the day it starts resolving to a real object is a rejection
            # rather than a silent new dependency.
            extensions.append({"path": item, "sha256": None})
    return {
        "cuda": {
            "driver_version": driver_version(),
            "sonames": sorted(cuda_sonames),
        },
        "kernel_registry": {
            "digest": _digest(_canonical(extensions)),
            "extensions": extensions,
            "roots": roots,
        },
        "ld_library_path": list(ld_library_path),
        "loader_configuration": configurations,
        "objects": [objects[name] for name in sorted(objects)],
        "system_search": list(system),
    }


def build(root_distributions: tuple[str, ...]) -> dict[str, Any]:
    paths: dict[str, dict[str, Any]] = {}
    distributions = _distribution_closure(root_distributions)
    for path in _stdlib_files():
        _add_path(paths, path)
    identities = []
    for distribution in distributions:
        name = canonicalize_name(distribution.metadata["Name"])
        identities.append({"name": name, "version": distribution.version})
        files = distribution.files
        if files is None:
            raise ValueError(f"distribution has no installed file inventory: {name}")
        for relative in files:
            path = Path(distribution.locate_file(relative))
            if path.exists() or path.is_symlink():
                _add_path(paths, path)
    # The interpreter is the object that loads every other one, so it belongs in
    # the closure rather than only in the deployment's separate
    # executable_fingerprint. Adding it here also pins the venv symlink chain
    # that reaches it.
    _add_path(paths, Path(sys.executable))
    # The ELF graph is walked after the Python file closure is complete, and it
    # adds to it: a resolved DT_NEEDED target outside any distribution becomes a
    # pinned file like any other. Seeded from what the closure already holds, so
    # the interpreter, libpython, and every extension module are entry points.
    native = _scan_native(
        paths,
        [
            *[
                entry["path"]
                for entry in paths.values()
                if entry["kind"] == "regular"
            ],
            os.path.realpath(sys.executable),
        ],
    )
    body = {
        "api_version": SCHEMA,
        "distributions": identities,
        "files": [paths[name] for name in sorted(paths)],
        "native": native,
        "python": {
            "cache_tag": sys.implementation.cache_tag,
            "implementation": sys.implementation.name,
            "platform": sysconfig.get_platform(),
            "prefix": sys.prefix,
            "version": ".".join(str(value) for value in sys.version_info[:3]),
        },
        "root_distributions": list(root_distributions),
    }
    return {**body, "closure_digest": _digest(_canonical(body))}


def _publish(output: Path, data: bytes) -> None:
    output = output.absolute()
    output.parent.mkdir(mode=0o750, parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, 0o600)
        os.replace(temporary, output)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--distribution",
        action="append",
        default=[],
        help="root installed distribution to include recursively (repeatable)",
    )
    arguments = parser.parse_args()
    requested = arguments.distribution or list(DEFAULT_ROOT_DISTRIBUTIONS)
    root_distributions = tuple(
        sorted({canonicalize_name(name) for name in requested})
    )
    if not root_distributions or any(not name for name in root_distributions):
        raise ValueError("runtime closure roots must be nonempty distributions")
    document = build(root_distributions)
    data = _canonical(document) + b"\n"
    _publish(arguments.output, data)
    print(
        json.dumps(
            {
                "schema": SCHEMA,
                "output": str(arguments.output.absolute()),
                "closure_digest": document["closure_digest"],
                "manifest_sha256": _digest(data),
                "root_distributions": list(root_distributions),
                "distribution_count": len(document["distributions"]),
                "file_count": len(document["files"]),
                "cuda_driver_version": document["native"]["cuda"][
                    "driver_version"
                ],
                "elf_object_count": len(document["native"]["objects"]),
                "kernel_registry_digest": document["native"]["kernel_registry"][
                    "digest"
                ],
                "native_object_count": len(
                    document["native"]["kernel_registry"]["extensions"]
                ),
                "unresolved_dependency_count": sum(
                    1
                    for item in document["native"]["objects"]
                    for dependency in item["dependencies"]
                    if dependency["resolved"] is None
                ),
                "total_bytes": sum(
                    entry.get("size", 0) for entry in document["files"]
                ),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
