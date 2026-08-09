"""Descriptor-pinned authority for mutable trainer input paths."""

from __future__ import annotations

import hashlib
import os
import stat
import struct
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path

INPUT_CONTENT_ROOT_API_VERSION = "trainvm.input-content-root/v1"
MAXIMUM_FILE_COUNT = 10_000_000
MAXIMUM_TOTAL_BYTES = 16 * 1024**5
MAXIMUM_DIRECTORY_DEPTH = 128
MAXIMUM_NAME_BYTES = 4096
MAXIMUM_PATH_BYTES = 4096
MAXIMUM_ROOT_COUNT = 256

_FILE_DOMAIN = b"trainvm.input-content.file/v1\0"
_DIRECTORY_DOMAIN = b"trainvm.input-content.directory/v1\0"
_IDENTITY_FIELDS = {
    "api_version",
    "path",
    "kind",
    "file_count",
    "total_bytes",
    "tree_sha256",
}
_READ_FLAGS = os.O_RDONLY | os.O_CLOEXEC | os.O_NONBLOCK
_DIRECTORY_FLAGS = _READ_FLAGS | os.O_DIRECTORY


class ContentAuthorityError(ValueError):
    """The declared input content is malformed, unsafe, or no longer exact."""


@dataclass(frozen=True, slots=True)
class InputContentRootIdentity:
    api_version: str
    path: str
    kind: str
    file_count: int
    total_bytes: int
    tree_sha256: str


@dataclass(frozen=True, slots=True)
class _MeasuredNode:
    kind: str
    file_count: int
    total_bytes: int
    digest: bytes


def _require_nofollow() -> int:
    nofollow = getattr(os, "O_NOFOLLOW", None)
    if not isinstance(nofollow, int) or nofollow == 0:
        raise ContentAuthorityError("this platform cannot enforce O_NOFOLLOW")
    return nofollow


def _canonical_path(path: object) -> str:
    try:
        raw = os.fspath(path)  # type: ignore[arg-type]
    except TypeError as error:
        raise ContentAuthorityError("content root path is not path-like") from error
    if not isinstance(raw, str):
        raise ContentAuthorityError("content root path must be text")
    try:
        encoded = raw.encode("utf-8", errors="strict")
    except UnicodeEncodeError as error:
        raise ContentAuthorityError("content root path is not valid UTF-8") from error
    if (
        not encoded
        or len(encoded) > MAXIMUM_PATH_BYTES
        or "\0" in raw
        or raw.startswith("//")
        or not os.path.isabs(raw)
        or os.path.normpath(raw) != raw
    ):
        raise ContentAuthorityError(
            "content root path must be absolute, normalized, and bounded"
        )
    return raw


def _name_bytes(name: str) -> bytes:
    try:
        encoded = name.encode("utf-8", errors="strict")
    except UnicodeEncodeError as error:
        raise ContentAuthorityError(
            "directory entry name is not valid UTF-8"
        ) from error
    if not encoded or len(encoded) > MAXIMUM_NAME_BYTES:
        raise ContentAuthorityError("directory entry name exceeds its bound")
    return encoded


def _stat_identity(
    value: os.stat_result,
) -> tuple[int, int, int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_nlink,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _same_stat(left: os.stat_result, right: os.stat_result) -> bool:
    return _stat_identity(left) == _stat_identity(right)


def _namespace_component_identity(
    value: os.stat_result,
) -> tuple[int, int, int, int, int]:
    """Identity of a path component *above* the measured root.

    A directory's link count, size, and timestamps all change whenever an
    unrelated entry is created or removed inside it. Ancestors are shared with
    the rest of the machine -- ``/tmp`` most obviously -- so comparing those
    fields reports ordinary concurrent activity beside the root as a
    substitution. Only the device/inode pair and the ownership and mode that
    policy depends on identify the component. Content stability *inside* the
    root is a separate question, still checked by :func:`_same_stat`.
    """

    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_uid,
        value.st_gid,
    )


def _same_namespace_component(
    left: os.stat_result, right: os.stat_result
) -> bool:
    return _namespace_component_identity(left) == _namespace_component_identity(
        right
    )


def _checked_namespace_component(
    name: str, *, dir_fd: int, expected: os.stat_result
) -> os.stat_result:
    observed = os.stat(name, dir_fd=dir_fd, follow_symlinks=False)
    if not _same_namespace_component(observed, expected):
        raise ContentAuthorityError("content path was substituted while measured")
    return observed


def _checked_stat(
    name: str, *, dir_fd: int, expected: os.stat_result
) -> os.stat_result:
    observed = os.stat(name, dir_fd=dir_fd, follow_symlinks=False)
    if not _same_stat(observed, expected):
        raise ContentAuthorityError("content path was substituted while measured")
    return observed


def _read_file(
    descriptor: int,
    before: os.stat_result,
    *,
    parent_descriptor: int | None,
    name: str | None,
) -> _MeasuredNode:
    if not stat.S_ISREG(before.st_mode):
        raise ContentAuthorityError("content tree contains a non-regular file")
    if before.st_size < 0 or before.st_size > MAXIMUM_TOTAL_BYTES:
        raise ContentAuthorityError("content file exceeds the byte bound")
    content = hashlib.sha256()
    observed_size = 0
    while True:
        chunk = os.read(descriptor, 1024 * 1024)
        if not chunk:
            break
        observed_size += len(chunk)
        if observed_size > before.st_size or observed_size > MAXIMUM_TOTAL_BYTES:
            raise ContentAuthorityError("content file changed size while measured")
        content.update(chunk)
    after = os.fstat(descriptor)
    if observed_size != before.st_size or not _same_stat(before, after):
        raise ContentAuthorityError("content file changed while measured")
    if parent_descriptor is not None and name is not None:
        _checked_stat(name, dir_fd=parent_descriptor, expected=before)
    material = _FILE_DOMAIN + struct.pack(">Q", observed_size) + content.digest()
    return _MeasuredNode(
        kind="file",
        file_count=1,
        total_bytes=observed_size,
        digest=hashlib.sha256(material).digest(),
    )


def _directory_names(descriptor: int) -> list[tuple[bytes, str]]:
    names = [(_name_bytes(name), name) for name in os.listdir(descriptor)]
    names.sort(key=lambda item: item[0])
    return names


def _measure_child(parent_descriptor: int, name: str, depth: int) -> _MeasuredNode:
    if depth > MAXIMUM_DIRECTORY_DEPTH:
        raise ContentAuthorityError("content tree exceeds its depth bound")
    before = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
    if stat.S_ISLNK(before.st_mode):
        raise ContentAuthorityError("content tree contains a symbolic link")
    if stat.S_ISREG(before.st_mode):
        flags = _READ_FLAGS
    elif stat.S_ISDIR(before.st_mode):
        flags = _DIRECTORY_FLAGS
    else:
        raise ContentAuthorityError("content tree contains a special node")
    descriptor = os.open(name, flags | _require_nofollow(), dir_fd=parent_descriptor)
    try:
        opened = os.fstat(descriptor)
        if not _same_stat(before, opened):
            raise ContentAuthorityError("content path changed while opened")
        if stat.S_ISREG(opened.st_mode):
            return _read_file(
                descriptor,
                opened,
                parent_descriptor=parent_descriptor,
                name=name,
            )
        measured = _read_directory(descriptor, opened, depth=depth)
        after = os.fstat(descriptor)
        if not _same_stat(opened, after):
            raise ContentAuthorityError("content directory changed while measured")
        _checked_stat(name, dir_fd=parent_descriptor, expected=opened)
        return measured
    finally:
        os.close(descriptor)


def _read_directory(
    descriptor: int, before: os.stat_result, *, depth: int
) -> _MeasuredNode:
    if not stat.S_ISDIR(before.st_mode):
        raise ContentAuthorityError("content node is not a directory")
    children = _directory_names(descriptor)
    material = bytearray(_DIRECTORY_DOMAIN)
    file_count = 0
    total_bytes = 0
    for name_bytes, name in children:
        child = _measure_child(descriptor, name, depth + 1)
        file_count += child.file_count
        total_bytes += child.total_bytes
        if file_count > MAXIMUM_FILE_COUNT:
            raise ContentAuthorityError("content tree exceeds its file-count bound")
        if total_bytes > MAXIMUM_TOTAL_BYTES:
            raise ContentAuthorityError("content tree exceeds its byte bound")
        material.extend(struct.pack(">I", len(name_bytes)))
        material.extend(name_bytes)
        material.extend(b"f" if child.kind == "file" else b"d")
        material.extend(child.digest)
    if _directory_names(descriptor) != children:
        raise ContentAuthorityError("content directory entries changed while measured")
    return _MeasuredNode(
        kind="directory",
        file_count=file_count,
        total_bytes=total_bytes,
        digest=hashlib.sha256(material).digest(),
    )


def _open_root(
    path: str,
) -> tuple[int, os.stat_result, list[tuple[int, str, os.stat_result]]]:
    nofollow = _require_nofollow()
    descriptors: list[int] = []
    links: list[tuple[int, str, os.stat_result]] = []
    current = os.open("/", _DIRECTORY_FLAGS | nofollow)
    descriptors.append(current)
    components = [part for part in path.split("/") if part]
    if not components:
        return current, os.fstat(current), links
    try:
        for index, name in enumerate(components):
            _name_bytes(name)
            before = os.stat(name, dir_fd=current, follow_symlinks=False)
            final = index + 1 == len(components)
            if stat.S_ISLNK(before.st_mode):
                raise ContentAuthorityError("content root traverses a symbolic link")
            if not final and not stat.S_ISDIR(before.st_mode):
                raise ContentAuthorityError("content root ancestor is not a directory")
            if final and stat.S_ISREG(before.st_mode):
                flags = _READ_FLAGS
            elif stat.S_ISDIR(before.st_mode):
                flags = _DIRECTORY_FLAGS
            else:
                raise ContentAuthorityError("content root is a special node")
            child = os.open(name, flags | nofollow, dir_fd=current)
            descriptors.append(child)
            opened = os.fstat(child)
            if not _same_namespace_component(before, opened):
                raise ContentAuthorityError("content root changed while opened")
            links.append((current, name, opened))
            current = child
        return current, os.fstat(current), links
    except Exception:
        for descriptor in reversed(descriptors):
            os.close(descriptor)
        raise


def _close_root(links: Sequence[tuple[int, str, os.stat_result]], root: int) -> None:
    descriptors = {root}
    descriptors.update(parent for parent, _, _ in links)
    for descriptor in descriptors:
        os.close(descriptor)


def measure_input_content_root(
    path: str | os.PathLike[str],
) -> InputContentRootIdentity:
    """Measure one exact regular file or nonempty directory content root."""

    canonical = _canonical_path(path)
    try:
        descriptor, opened, links = _open_root(canonical)
        try:
            if stat.S_ISREG(opened.st_mode):
                parent, name, _ = links[-1]
                measured = _read_file(
                    descriptor,
                    opened,
                    parent_descriptor=parent,
                    name=name,
                )
            elif stat.S_ISDIR(opened.st_mode):
                measured = _read_directory(descriptor, opened, depth=0)
                after = os.fstat(descriptor)
                if not _same_stat(opened, after):
                    raise ContentAuthorityError(
                        "content root directory changed while measured"
                    )
            else:
                raise ContentAuthorityError("content root is not a file or directory")
            for parent, name, expected in reversed(links):
                _checked_namespace_component(
                    name, dir_fd=parent, expected=expected
                )
        finally:
            _close_root(links, descriptor)
    except ContentAuthorityError:
        raise
    except (OSError, OverflowError, struct.error) as error:
        raise ContentAuthorityError(
            f"content root could not be measured safely: {canonical!r}: {error}"
        ) from error
    if measured.kind == "directory" and measured.file_count == 0:
        raise ContentAuthorityError("content root directory is empty")
    return InputContentRootIdentity(
        api_version=INPUT_CONTENT_ROOT_API_VERSION,
        path=canonical,
        kind=measured.kind,
        file_count=measured.file_count,
        total_bytes=measured.total_bytes,
        tree_sha256="sha256:" + measured.digest.hex(),
    )


def _strict_uint(value: object, *, label: str, minimum: int, maximum: int) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise ContentAuthorityError(f"content root {label} is invalid")
    return value


def _decode_identity(raw: object) -> InputContentRootIdentity:
    if not isinstance(raw, Mapping) or set(raw) != _IDENTITY_FIELDS:
        raise ContentAuthorityError("content root identity fields are not exact")
    api_version = raw.get("api_version")
    raw_path = raw.get("path")
    kind = raw.get("kind")
    digest = raw.get("tree_sha256")
    if type(api_version) is not str or api_version != INPUT_CONTENT_ROOT_API_VERSION:
        raise ContentAuthorityError("content root api_version is unsupported")
    if type(raw_path) is not str:
        raise ContentAuthorityError("content root path is not text")
    path = _canonical_path(raw_path)
    if type(kind) is not str or kind not in {"file", "directory"}:
        raise ContentAuthorityError("content root kind is invalid")
    if (
        type(digest) is not str
        or len(digest) != 71
        or not digest.startswith("sha256:")
        or any(character not in "0123456789abcdef" for character in digest[7:])
    ):
        raise ContentAuthorityError("content root digest is invalid")
    return InputContentRootIdentity(
        api_version=api_version,
        path=path,
        kind=kind,
        file_count=_strict_uint(
            raw.get("file_count"),
            label="file_count",
            minimum=1,
            maximum=MAXIMUM_FILE_COUNT,
        ),
        total_bytes=_strict_uint(
            raw.get("total_bytes"),
            label="total_bytes",
            minimum=0,
            maximum=MAXIMUM_TOTAL_BYTES,
        ),
        tree_sha256=digest,
    )


def _paths_overlap(left: str, right: str) -> bool:
    left_path = Path(left)
    right_path = Path(right)
    return (
        left_path == right_path
        or left_path in right_path.parents
        or right_path in left_path.parents
    )


def verify_input_content_roots(raw: object) -> tuple[InputContentRootIdentity, ...]:
    """Strictly decode and remeasure a canonical list of input roots."""

    if type(raw) is not list or not 1 <= len(raw) <= MAXIMUM_ROOT_COUNT:
        raise ContentAuthorityError("input content roots must be a nonempty list")
    decoded = tuple(_decode_identity(item) for item in raw)
    for left, right in pairwise(decoded):
        if left.path.encode("utf-8") >= right.path.encode("utf-8"):
            raise ContentAuthorityError(
                "input content roots are not strictly path-sorted: "
                f"{right.path!r} does not sort after {left.path!r}"
            )
    for index, left in enumerate(decoded):
        for right in decoded[index + 1 :]:
            if _paths_overlap(left.path, right.path):
                raise ContentAuthorityError(
                    "input content root paths overlap: "
                    f"{left.path!r} and {right.path!r} are not disjoint"
                )
    verified: list[InputContentRootIdentity] = []
    for declared in decoded:
        measured = measure_input_content_root(declared.path)
        if measured != declared:
            raise ContentAuthorityError(
                f"input content root identity no longer matches: {declared.path!r}"
            )
        verified.append(measured)
    return tuple(verified)


__all__ = [
    "INPUT_CONTENT_ROOT_API_VERSION",
    "ContentAuthorityError",
    "InputContentRootIdentity",
    "measure_input_content_root",
    "verify_input_content_roots",
]
