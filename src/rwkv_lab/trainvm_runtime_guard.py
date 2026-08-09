"""Pre-import verification for an authority-bound Python runtime closure."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import sys
import sysconfig
import zipfile
from pathlib import Path
from typing import Any

RUNTIME_CLOSURE_MEMBER = "TRAINVM_RUNTIME_CLOSURE.json"
RUNTIME_CLOSURE_SCHEMA = "trainvm.python-bootstrap-runtime-closure/v2"
MAXIMUM_MANIFEST_BYTES = 32 * 1024 * 1024
MAXIMUM_FILES = 100_000
MAXIMUM_TOTAL_BYTES = 32 * 1024 * 1024 * 1024


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


def _verify_entry(entry: dict[str, Any]) -> None:
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
        raise RuntimeClosureError(f"runtime closure file content changed: {path}")


def verify_embedded_runtime_closure(archive_path: str | None = None) -> str:
    """Verify the deployment environment before importing any third-party code."""

    archive_path = archive_path or sys.argv[0]
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
        total = 0
        for entry in files:
            if not isinstance(entry, dict):
                raise RuntimeClosureError("runtime closure entry is not an object")
            _verify_entry(entry)
            paths.append(entry["path"])
            if entry["kind"] == "regular":
                total += entry["size"]
                if total > MAXIMUM_TOTAL_BYTES:
                    raise RuntimeClosureError(
                        "runtime closure byte total is unbounded"
                    )
        if paths != sorted(set(paths)):
            raise RuntimeClosureError("runtime closure paths are not canonical")
        return closure_digest
    except RuntimeClosureError:
        raise
    except (KeyError, OSError, TypeError, ValueError, zipfile.BadZipFile) as error:
        raise RuntimeClosureError("runtime closure verification failed closed") from error


__all__ = [
    "MAXIMUM_FILES",
    "MAXIMUM_MANIFEST_BYTES",
    "MAXIMUM_TOTAL_BYTES",
    "RUNTIME_CLOSURE_MEMBER",
    "RUNTIME_CLOSURE_SCHEMA",
    "RuntimeClosureError",
    "self_owned_closure_ancestors",
    "verify_embedded_runtime_closure",
]
