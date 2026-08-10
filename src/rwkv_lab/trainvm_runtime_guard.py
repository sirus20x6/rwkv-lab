"""Pre-import verification for an authority-bound Python runtime closure."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import sys
import sysconfig
import time
import zipfile
from pathlib import Path
from typing import Any

RUNTIME_CLOSURE_MEMBER = "TRAINVM_RUNTIME_CLOSURE.json"
RUNTIME_CLOSURE_SCHEMA = "trainvm.python-bootstrap-runtime-closure/v4"
MAXIMUM_MANIFEST_BYTES = 32 * 1024 * 1024
MAXIMUM_FILES = 100_000
MAXIMUM_TOTAL_BYTES = 32 * 1024 * 1024 * 1024
MAXIMUM_OBJECTS = 8192
MAXIMUM_NATIVE_OBJECTS = 32768

# Kept identical to scripts/build_trainvm_runtime_closure.py. The two are not
# shared code: the builder runs inside the adapter's own sealed interpreter,
# where this package is not importable, and this module must import nothing
# outside the standard library because it runs before the closure is trusted.
# tests/test_trainvm_native_closure.py drives both over the same tree and fails
# when they disagree, which is the protection a shared import would have given.
NATIVE_SUFFIXES = (".so",)
NVIDIA_DRIVER_VERSION_FILE = "/proc/driver/nvidia/version"
NVIDIA_MODULE_BUILD_ID_FILE = "/sys/module/nvidia/notes/.note.gnu.build-id"
_NT_GNU_BUILD_ID = 3
_DRIVER_VERSION_TOKEN = re.compile(r"\d+(?:\.\d+)+")


class RuntimeClosureError(RuntimeError):
    pass


# Ancestors that are writable by the worker only because the worker owns them.
# Recorded rather than raised (see _require_nonwritable_ancestors) and reported
# by self_owned_closure_ancestors(), so a deployment that has given up
# write-protection of its own runtime says so out loud instead of looking
# identical to one that still has it.
_self_owned_ancestors: set[str] = set()


def self_owned_closure_ancestors() -> list[str]:
    """Closure ancestors the worker could rewrite, because it owns them.

    Empty when the worker runs as an identity distinct from the one owning its
    code, which is the configuration this guard can actually enforce.
    """
    return sorted(_self_owned_ancestors)


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")


def _digest(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _sha256_file(path: str, expected_size: int) -> str:
    flags = os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size != expected_size:
            raise RuntimeClosureError("runtime closure file identity changed")
        value = hashlib.sha256()
        total = 0
        while block := os.read(descriptor, 1 << 20):
            total += len(block)
            value.update(block)
        after = os.fstat(descriptor)
        if (
            total != expected_size
            or before.st_dev != after.st_dev
            or before.st_ino != after.st_ino
            or before.st_size != after.st_size
            or before.st_mtime_ns != after.st_mtime_ns
            or before.st_ctime_ns != after.st_ctime_ns
        ):
            raise RuntimeClosureError("runtime closure file changed while hashing")
        return "sha256:" + value.hexdigest()
    finally:
        os.close(descriptor)


def _moment(nanoseconds: int) -> str:
    seconds, remainder = divmod(nanoseconds, 1_000_000_000)
    stamp = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(seconds))
    return f"{stamp}.{remainder // 1_000_000:03d}Z"


def _sealing_time_ns(archive_path: str) -> int | None:
    """When this closure was sealed onto this host.

    Read for one purpose: phrasing a rejection that has already been decided.
    Never load-bearing, for the same reason `driver_report` is not -- an mtime
    is writable by anyone who can write the file, so nothing may be *accepted*
    on its strength. Returning None costs the reader a sentence and costs the
    guard nothing.
    """

    try:
        return os.stat(archive_path).st_mtime_ns
    except OSError:
        return None


def _attribution(path: str, sealed_ns: int | None) -> str:
    """Say whether a digest that moved moved *after* the closure was sealed.

    A closure names every native object on its import path, so an unrelated
    host package rewritten mid-session reds it on a change that never touched
    Python -- and the bare message names a file the author has no reason to
    recognise. Every natural reading of that is wrong, and the worst of them
    ("the check is flaky, rerun it") is the habit this repository can least
    afford in a suite that hosted CI does not run.

    The two cases are distinguishable and mean opposite things. A file written
    after the seal changed under the run; a file untouched since the seal never
    matched the closure at all. Neither answer changes the verdict -- the object
    is rejected either way -- so a backdated mtime buys an attacker a misleading
    sentence and no acceptance.
    """

    if sealed_ns is None:
        return " (sealing time unreadable, so the change cannot be placed in time)"
    try:
        written_ns = os.lstat(path).st_mtime_ns
    except OSError:
        return " (file timestamp unreadable, so the change cannot be placed in time)"
    if written_ns > sealed_ns:
        return (
            f" (rewritten {_moment(written_ns)}, after this closure was sealed at "
            f"{_moment(sealed_ns)}: something changed this file on the host after "
            f"the seal, which is what a package manager run during a session looks "
            f"like -- re-seal the closure; this is not evidence about the change "
            f"under test)"
        )
    return (
        f" (last written {_moment(written_ns)}, not since this closure was sealed at "
        f"{_moment(sealed_ns)}: these bytes were already in place at the seal and "
        f"still do not match it, so this is a real mismatch and not host churn)"
    )


def _require_nonwritable_ancestors(path: Path) -> None:
    """Refuse a closure the worker could rewrite between verification and use.

    The check assumes the worker runs as an identity that does not own its own
    code. Where that holds it is the whole point of this guard, and it keeps its
    full force. Where the worker *is* the owner it is unsatisfiable rather than
    merely unmet: no arrangement of permissions makes a directory unwritable by
    the uid that owns it, short of handing the closure to a different uid. An
    ancestor owned by the worker is therefore reported as an explicit,
    named-in-the-receipt absence of the property, not silently tolerated and not
    treated as tampering. On a single-owner host the guarantee is unobtainable:
    anything able to write as that uid is already that uid.
    """
    current = path.parent
    while True:
        if os.access(current, os.W_OK, effective_ids=True):
            if current.stat().st_uid == os.geteuid():
                _self_owned_ancestors.add(str(current))
                return
            raise RuntimeClosureError(
                f"runtime closure has a path ancestor writable by another "
                f"identity: {current}"
            )
        if current == current.parent:
            return
        current = current.parent


def _verify_entry(entry: dict[str, Any], sealed_ns: int | None) -> None:
    path_value = entry.get("path")
    kind = entry.get("kind")
    if (
        not isinstance(path_value, str)
        or not path_value.startswith("/")
        or "\x00" in path_value
        or os.path.normpath(path_value) != path_value
        or kind not in {"regular", "symlink"}
    ):
        raise RuntimeClosureError("runtime closure path entry is malformed")
    path = Path(path_value)
    _require_nonwritable_ancestors(path)
    metadata = path.lstat()
    mode = stat.S_IMODE(metadata.st_mode)
    if mode != entry.get("mode"):
        raise RuntimeClosureError("runtime closure permissions changed or are unsafe")
    if kind == "symlink":
        if set(entry) != {"kind", "mode", "path", "target"} or not stat.S_ISLNK(
            metadata.st_mode
        ):
            raise RuntimeClosureError("runtime closure symlink identity changed")
        target = entry.get("target")
        if not isinstance(target, str) or os.readlink(path) != target:
            raise RuntimeClosureError("runtime closure symlink target changed")
        return
    if set(entry) != {"kind", "mode", "path", "sha256", "size"}:
        raise RuntimeClosureError("runtime closure regular entry shape changed")
    size = entry.get("size")
    sha256 = entry.get("sha256")
    if not stat.S_ISREG(metadata.st_mode):
        raise RuntimeClosureError(f"runtime closure entry is not a regular file: {path}")
    # Group- or world-writable is a real misconfiguration and stays fatal: it is
    # reachable by identities that have no claim on this runtime, and it is
    # fixable with chmod.
    if mode & (stat.S_IWGRP | stat.S_IWOTH):
        raise RuntimeClosureError(
            f"runtime closure file is group- or world-writable: {path}"
        )
    # Writability by the worker itself is the same unobtainable property as in
    # _require_nonwritable_ancestors, and is recorded the same way. It was
    # previously folded into the content comparison below, so a worker that
    # owned its own runtime reported "file content changed" for a file whose
    # bytes matched the manifest exactly — sending the reader to hunt for
    # tampering that had not occurred.
    if os.access(path, os.W_OK, effective_ids=True):
        if metadata.st_uid != os.geteuid():
            raise RuntimeClosureError(
                f"runtime closure file is writable by another identity: {path}"
            )
        _self_owned_ancestors.add(path_value)
    if (
        not isinstance(size, int)
        or isinstance(size, bool)
        or size < 0
        or not isinstance(sha256, str)
    ):
        raise RuntimeClosureError("runtime closure entry size or digest is malformed")
    if _sha256_file(path_value, size) != sha256:
        raise RuntimeClosureError(
            f"runtime closure file content changed: {path}"
            f"{_attribution(path_value, sealed_ns)}"
        )


def native_roots() -> list[str]:
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
    """Every loadable native object on the import path, discovered not assumed.

    This walks the directories rather than reading the manifest's list back,
    which is the whole point: an extension that appeared since the closure was
    built is invisible to any check that only re-verifies recorded entries, and
    it is importable and registers whatever it registers all the same.
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
                        raise RuntimeClosureError(
                            "native object inventory is unbounded on the import path"
                        )
    return sorted(set(found))


def driver_report() -> str | None:
    """The kernel's own description of the NVIDIA driver, verbatim.

    Evidence, never an identity. The line ends in the user and host that
    compiled the module -- ``(root@neuromancer)`` -- so comparing it for
    equality refuses a host for having rebuilt its driver rather than for
    running a different one. Recorded in the manifest beside the identity, and
    quoted in the rejection, so a human debugging one can still see both the
    build the closure was sealed against and the build in front of them.
    """

    try:
        with open(NVIDIA_DRIVER_VERSION_FILE, encoding="utf-8", errors="replace") as f:
            line = f.readline(4096).strip()
    except OSError:
        return None
    return line or None


def _module_build_id() -> str | None:
    """The GNU build ID of the loaded ``nvidia`` module, from its ELF note.

    The build ID is the linker's hash of the module's own contents, so it
    answers the question the build host was standing in for: is this the same
    artifact? Read from sysfs rather than from the module file on disk, which
    is ``/lib/modules/<release>/updates/dkms/nvidia.ko.zst`` on a DKMS host --
    compressed, at a path that depends on how the driver was packaged, and not
    necessarily the artifact the kernel actually has loaded. The sysfs note is
    a fixed path, a plain read, and world-readable.

    None when the kernel exposes no usable note, which is not an error: a
    module linked with ``--build-id=none`` has none to expose, and the identity
    falls back to the version token alone.
    """

    try:
        with open(NVIDIA_MODULE_BUILD_ID_FILE, "rb") as stream:
            note = stream.read(4096)
    except OSError:
        return None
    if len(note) < 12:
        return None
    name_size = int.from_bytes(note[0:4], sys.byteorder)
    description_size = int.from_bytes(note[4:8], sys.byteorder)
    start = 12 + (name_size + 3) // 4 * 4
    if (
        int.from_bytes(note[8:12], sys.byteorder) != _NT_GNU_BUILD_ID
        or note[12 : 12 + name_size] != b"GNU\x00"
        or not 8 <= description_size <= 64
        or len(note) < start + description_size
    ):
        return None
    return note[start : start + description_size].hex()


def driver_identity() -> str | None:
    """What has to still hold for a compiled artifact to be valid on this host.

    The driver's version token, plus the loaded module's build ID when the
    kernel exposes one.

    Deliberately *not* the whole first line of /proc/driver/nvidia/version.
    That line carries the build user and host, so a DKMS rebuild, or the same
    driver version built on another machine, moves it while the driver stays
    bit-for-bit compatible -- and everything keyed on this identity refuses or
    goes cold for a non-reason: this guard rejects the host, and the worker's
    cache namespace changes underneath ``compute_compatibility_digest``. The
    whole line is not even a legal identity downstream: it contains spaces,
    parentheses and ``@``, none of which ``fixed_public_identity`` in
    trainvm/src/cache_namespace.cpp accepts.

    The version token alone would be too weak to stand on its own -- "same
    version string, different build" is a real failure, and an open kernel
    module compiled elsewhere with different flags is not obviously the same
    artifact. The build ID closes that: a rebuild producing identical bytes
    reads as identical, one producing different bytes reads as different, and
    both directions are what we actually mean.

    None on a host with no NVIDIA driver, which is pinned as firmly as a
    version string is.
    """

    report = driver_report()
    if report is None:
        return None
    found = _DRIVER_VERSION_TOKEN.search(report)
    # A line whose shape this does not recognise keeps the whole line rather
    # than dropping the driver out of the identity: too strict is recoverable
    # by resealing, too permissive is not noticed at all.
    version = found.group(0) if found else report
    build_id = _module_build_id()
    return version if build_id is None else f"{version}+gnu-build-id:{build_id}"


def _resolve(search: list[str], needed: str) -> str | None:
    if "/" in needed:
        return needed if os.path.lexists(needed) else None
    for directory in search:
        candidate = os.path.join(directory, needed)
        if os.path.lexists(candidate):
            return candidate
    return None


def _sha256_whole(path: str) -> str:
    value = hashlib.sha256()
    with open(path, "rb") as stream:
        while block := stream.read(1 << 20):
            value.update(block)
    return "sha256:" + value.hexdigest()


def _verify_native_object(
    item: Any, index: dict[str, dict[str, Any]], pinned: set[str]
) -> str:
    if (
        not isinstance(item, dict)
        or set(item) != {"dependencies", "path", "search", "soname"}
        or not isinstance(item.get("path"), str)
        or not isinstance(item.get("soname"), str)
        or not isinstance(item.get("search"), list)
        or not isinstance(item.get("dependencies"), list)
    ):
        raise RuntimeClosureError("runtime closure ELF object is malformed")
    path = item["path"]
    search = item["search"]
    if any(
        not isinstance(directory, str) or not directory.startswith("/")
        for directory in search
    ):
        raise RuntimeClosureError("runtime closure ELF search path is malformed")
    if path not in pinned:
        raise RuntimeClosureError(
            f"runtime closure ELF object is not a pinned file: {path}"
        )
    seen: list[str] = []
    for dependency in item["dependencies"]:
        if (
            not isinstance(dependency, dict)
            or set(dependency) != {"needed", "resolved"}
            or not isinstance(dependency.get("needed"), str)
            or not dependency["needed"]
        ):
            raise RuntimeClosureError(
                "runtime closure ELF dependency is malformed"
            )
        needed = dependency["needed"]
        resolved = dependency["resolved"]
        if resolved is not None and not isinstance(resolved, str):
            raise RuntimeClosureError(
                "runtime closure ELF dependency target is malformed"
            )
        seen.append(needed)
        # Re-run the loader's decision rather than trusting the recorded
        # answer. Content digests cannot see this: planting a library in a
        # directory that comes earlier in this object's search order changes
        # which file is loaded while every pinned byte stays identical.
        if _resolve(search, needed) != resolved:
            raise RuntimeClosureError(
                f"runtime closure shared library resolution changed: "
                f"{path} needs {needed}"
            )
        if resolved is not None and resolved not in index and resolved not in pinned:
            raise RuntimeClosureError(
                f"runtime closure shared library is not pinned: {resolved}"
            )
    if seen != sorted(set(seen)):
        raise RuntimeClosureError(
            "runtime closure ELF dependencies are not canonical"
        )
    return path


def _verify_kernel_registry(
    registry: Any, index: dict[str, dict[str, Any]], sealed_ns: int | None
) -> set[str]:
    if (
        not isinstance(registry, dict)
        or set(registry) != {"digest", "extensions", "roots"}
        or not isinstance(registry.get("digest"), str)
        or not isinstance(registry.get("extensions"), list)
        or registry.get("roots") != native_roots()
    ):
        raise RuntimeClosureError("runtime closure kernel registry is malformed")
    extensions = registry["extensions"]
    if len(extensions) > MAXIMUM_NATIVE_OBJECTS:
        raise RuntimeClosureError("runtime closure kernel registry is unbounded")
    if registry["digest"] != _digest(_canonical(extensions)):
        raise RuntimeClosureError("runtime closure kernel registry digest is invalid")
    recorded: list[str] = []
    answerable: set[str] = set()
    for extension in extensions:
        if (
            not isinstance(extension, dict)
            or set(extension) != {"path", "sha256"}
            or not isinstance(extension.get("path"), str)
            or not extension["path"].startswith("/")
        ):
            raise RuntimeClosureError(
                "runtime closure kernel registry entry is malformed"
            )
        path = extension["path"]
        sha256 = extension["sha256"]
        recorded.append(path)
        if sha256 is None:
            if os.path.isfile(path):
                raise RuntimeClosureError(
                    f"runtime closure kernel registry entry became loadable: {path}"
                )
            continue
        if not isinstance(sha256, str):
            raise RuntimeClosureError(
                "runtime closure kernel registry digest is malformed"
            )
        answerable.add(path)
        entry = index.get(path)
        if entry is not None and entry["kind"] == "regular":
            # Already verified byte for byte by the file pass; agreeing with it
            # is what makes the registry a view of the closure rather than a
            # second, independently forgeable claim about the same file.
            if entry["sha256"] != sha256:
                raise RuntimeClosureError(
                    f"runtime closure kernel registry disagrees with the file "
                    f"closure: {path}"
                )
            continue
        if _sha256_whole(path) != sha256:
            raise RuntimeClosureError(
                f"runtime closure kernel registry object changed: {path}"
                f"{_attribution(path, sealed_ns)}"
            )
    if recorded != sorted(set(recorded)):
        raise RuntimeClosureError(
            "runtime closure kernel registry is not canonical"
        )
    if native_objects(registry["roots"]) != recorded:
        raise RuntimeClosureError(
            "runtime closure kernel registry does not match the import path"
        )
    return answerable


def _verify_native(
    native: Any, index: dict[str, dict[str, Any]], sealed_ns: int | None
) -> None:
    """Verify what the dynamic loader would do, not only what Python imports.

    Everything here is a property no file digest can carry on its own: which
    file a SONAME resolves to, which kernel driver the CUDA libraries will bind
    to, and what is on the import path that the manifest never named.
    """

    if not isinstance(native, dict) or set(native) != {
        "cuda",
        "kernel_registry",
        "ld_library_path",
        "loader_configuration",
        "objects",
        "system_search",
    }:
        raise RuntimeClosureError("runtime closure native section is malformed")
    search_path = native["ld_library_path"]
    if search_path != [
        entry for entry in os.environ.get("LD_LIBRARY_PATH", "").split(":") if entry
    ]:
        raise RuntimeClosureError("runtime closure LD_LIBRARY_PATH changed")
    system = native["system_search"]
    if not isinstance(system, list) or any(
        not isinstance(directory, str) or not directory.startswith("/")
        for directory in system
    ):
        raise RuntimeClosureError("runtime closure system search path is malformed")
    configuration = native["loader_configuration"]
    if (
        not isinstance(configuration, list)
        or configuration != sorted(set(configuration))
        or any(item not in index for item in configuration)
    ):
        # The guard does not re-derive `system_search`; it requires the files
        # that produce it to be pinned, so editing one is a content change
        # caught by the file pass.
        raise RuntimeClosureError(
            "runtime closure loader configuration is not pinned"
        )
    # The registry runs first because it establishes which import-path objects
    # are answerable for their own bytes. An object on the import path that no
    # closure distribution claims is still pinned — by the registry rather than
    # by `files` — and an ELF object may cite either.
    pinned = _verify_kernel_registry(native["kernel_registry"], index, sealed_ns)
    pinned.update(
        path for path, entry in index.items() if entry["kind"] == "regular"
    )
    objects = native["objects"]
    if not isinstance(objects, list) or len(objects) > MAXIMUM_OBJECTS:
        raise RuntimeClosureError("runtime closure ELF object list is invalid")
    paths = [_verify_native_object(item, index, pinned) for item in objects]
    if paths != sorted(set(paths)):
        raise RuntimeClosureError("runtime closure ELF objects are not canonical")
    cuda = native["cuda"]
    if (
        not isinstance(cuda, dict)
        or set(cuda) != {"driver_identity", "driver_report", "sonames"}
        or not isinstance(cuda["sonames"], list)
        or cuda["sonames"] != sorted(set(cuda["sonames"]))
    ):
        raise RuntimeClosureError("runtime closure CUDA identity is malformed")
    recorded_driver = cuda["driver_identity"]
    recorded_report = cuda["driver_report"]
    if (recorded_driver is not None and not isinstance(recorded_driver, str)) or (
        recorded_report is not None and not isinstance(recorded_report, str)
    ):
        raise RuntimeClosureError("runtime closure CUDA driver identity is malformed")
    # The driver is in the kernel, not in the closure, so nothing else here can
    # notice it moving. Null is pinned as firmly as an identity string: a host
    # that grew a driver since the closure was sealed is a changed host.
    #
    # Only `driver_identity` is compared. `driver_report` is carried for the
    # human reading a rejection and is deliberately never load-bearing -- it
    # names the machine that compiled the module, and equality on that is the
    # defect this pair replaced.
    observed = driver_identity()
    if recorded_driver != observed:
        raise RuntimeClosureError(
            "runtime closure CUDA driver identity changed: sealed against "
            f"{recorded_driver!r} ({recorded_report!r}), host reports "
            f"{observed!r} ({driver_report()!r})"
        )


def verify_embedded_runtime_closure(archive_path: str | None = None) -> str:
    """Verify the deployment environment before importing any third-party code."""

    archive_path = archive_path or sys.argv[0]
    # The archive's own mtime is when this closure was sealed onto this host.
    # The manifest cannot carry the answer: the artifact builder writes every
    # zip member at a fixed 1980 timestamp so the build is byte-reproducible,
    # which is worth more than a self-describing seal time. Explanatory only --
    # see _attribution.
    sealed_ns = _sealing_time_ns(archive_path)
    try:
        with zipfile.ZipFile(archive_path) as archive:
            info = archive.getinfo(RUNTIME_CLOSURE_MEMBER)
            if info.file_size <= 0 or info.file_size > MAXIMUM_MANIFEST_BYTES:
                raise RuntimeClosureError("runtime closure manifest size is invalid")
            raw = archive.read(info)
        if not raw.endswith(b"\n"):
            raise RuntimeClosureError("runtime closure manifest is not canonical")
        document = json.loads(raw)
        if raw != _canonical(document) + b"\n" or not isinstance(document, dict):
            raise RuntimeClosureError("runtime closure manifest is not canonical")
        if set(document) != {
            "api_version",
            "closure_digest",
            "distributions",
            "files",
            "native",
            "python",
            "root_distributions",
        }:
            raise RuntimeClosureError("runtime closure manifest fields are not exact")
        body = dict(document)
        closure_digest = body.pop("closure_digest")
        python = body.get("python")
        distributions = body.get("distributions")
        files = body.get("files")
        roots = body.get("root_distributions")
        if (
            body.get("api_version") != RUNTIME_CLOSURE_SCHEMA
            or closure_digest != _digest(_canonical(body))
            or not isinstance(python, dict)
            or python
            != {
                "cache_tag": sys.implementation.cache_tag,
                "implementation": sys.implementation.name,
                "platform": sysconfig.get_platform(),
                "prefix": sys.prefix,
                "version": ".".join(str(value) for value in sys.version_info[:3]),
            }
            or not isinstance(distributions, list)
            or not isinstance(files, list)
            or not isinstance(roots, list)
            or not roots
            or not files
            or len(files) > MAXIMUM_FILES
        ):
            raise RuntimeClosureError("runtime closure manifest semantics are invalid")
        identities: list[str] = []
        for item in distributions:
            if (
                not isinstance(item, dict)
                or set(item) != {"name", "version"}
                or not isinstance(item.get("name"), str)
                or not item["name"]
                or not isinstance(item.get("version"), str)
                or not item["version"]
            ):
                raise RuntimeClosureError(
                    "runtime closure distribution identity is malformed"
                )
            identities.append(item["name"])
        if identities != sorted(set(identities)):
            raise RuntimeClosureError(
                "runtime closure distributions are not canonical"
            )
        if (
            any(not isinstance(name, str) or not name for name in roots)
            or roots != sorted(set(roots))
            or not set(roots).issubset(identities)
        ):
            raise RuntimeClosureError(
                "runtime closure root distributions are not canonical"
            )
        paths: list[str] = []
        index: dict[str, dict[str, Any]] = {}
        total = 0
        for entry in files:
            if not isinstance(entry, dict):
                raise RuntimeClosureError("runtime closure entry is not an object")
            _verify_entry(entry, sealed_ns)
            paths.append(entry["path"])
            index[entry["path"]] = entry
            if entry["kind"] == "regular":
                total += entry["size"]
                if total > MAXIMUM_TOTAL_BYTES:
                    raise RuntimeClosureError(
                        "runtime closure byte total is unbounded"
                    )
        if paths != sorted(set(paths)):
            raise RuntimeClosureError("runtime closure paths are not canonical")
        _verify_native(body.get("native"), index, sealed_ns)
        return closure_digest
    except RuntimeClosureError:
        raise
    except (KeyError, OSError, TypeError, ValueError, zipfile.BadZipFile) as error:
        raise RuntimeClosureError("runtime closure verification failed closed") from error


__all__ = [
    "MAXIMUM_FILES",
    "MAXIMUM_MANIFEST_BYTES",
    "MAXIMUM_NATIVE_OBJECTS",
    "MAXIMUM_OBJECTS",
    "MAXIMUM_TOTAL_BYTES",
    "RUNTIME_CLOSURE_MEMBER",
    "RUNTIME_CLOSURE_SCHEMA",
    "RuntimeClosureError",
    "driver_identity",
    "driver_report",
    "native_objects",
    "native_roots",
    "self_owned_closure_ancestors",
    "verify_embedded_runtime_closure",
]
