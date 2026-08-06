"""Family-neutral immutable directory artifact publication.

This is the non-checkpoint counterpart to :mod:`checkpoint`.  A trainer may
stage a completed adapter, report bundle, or other declared output only inside
its run directory.  The publisher snapshots regular files without following
symlinks, binds every byte into a canonical manifest, atomically promotes a
content-addressed revision, and only then announces it to TrainVM.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import stat
import tempfile
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol
from urllib.parse import unquote, urlsplit

from ._canonical import (
    CanonicalJsonError,
    canonical_dumps,
    canonical_loads,
    exact_fields,
    is_bounded_text,
    is_digest,
    is_uint64,
    sha256_digest,
)

try:
    from trainvm.v1 import trainvm_pb2 as wire
except ImportError as error:  # pragma: no cover - installation contract
    raise RuntimeError(
        "TrainVM artifact publication requires the 'trainvm-worker' project extra"
    ) from error


IMMUTABLE_TREE_SCHEMA = "trainvm.immutable-tree.v1"
MAXIMUM_ARTIFACT_FILES = 131_072
MAXIMUM_ARTIFACT_BYTES = 16 * 1024 * 1024 * 1024 * 1024
MAXIMUM_ARTIFACT_MANIFEST_BYTES = 128 * 1024 * 1024
MAXIMUM_RELATIVE_PATH_BYTES = 4096
_KINDS = {
    "path": wire.ARTIFACT_KIND_PATH,
    "dataset": wire.ARTIFACT_KIND_DATASET,
    "metrics": wire.ARTIFACT_KIND_METRICS,
    "report": wire.ARTIFACT_KIND_REPORT,
    "opaque": wire.ARTIFACT_KIND_OPAQUE,
}
_ARTIFACT_FIELDS = frozenset(
    {
        "artifact_id",
        "logical_name",
        "kind",
        "schema",
        "uri",
        "size_bytes",
        "fingerprint_algorithm",
        "fingerprint",
        "complete",
        "producer_node_id",
        "producer_attempt_id",
        "parent_artifact_ids",
        "published_at_ns",
    }
)
_MANIFEST_FIELDS = frozenset(
    {
        "schema",
        "artifact_schema",
        "artifact_kind",
        "invocation_digest",
        "producer",
        "payload_directory",
        "file_count",
        "payload_size_bytes",
        "objects",
        "parent_artifact_ids",
        "canonical_tree_digest",
    }
)


class ArtifactPublicationError(RuntimeError):
    pass


class _ArtifactSession(Protocol):
    bootstrap: object
    invocation: object

    def artifact(self, **values: object) -> int: ...


@dataclass(frozen=True, slots=True)
class ArtifactPublicationRequest:
    source_directory: str | os.PathLike[str]
    output_name: str
    parent_artifact_ids: tuple[str, ...] = ()
    optimizer_step: int | None = None


@dataclass(frozen=True, slots=True)
class PublishedArtifact:
    artifact_id: str
    manifest_path: Path
    manifest_sha256: str
    payload_size_bytes: int
    file_count: int
    worker_sequence: int


@dataclass(frozen=True, slots=True)
class ResolvedArtifactObject:
    relative_path: str
    sha256: str
    size_bytes: int


@dataclass(frozen=True, slots=True)
class ResolvedInputArtifact:
    artifact_id: str
    logical_name: str
    kind: str
    schema: str
    manifest_path: Path
    payload_directory: Path
    manifest_sha256: str
    payload_size_bytes: int
    file_count: int
    parent_artifact_ids: tuple[str, ...]
    canonical_tree_digest: str
    objects: tuple[ResolvedArtifactObject, ...]


@dataclass(frozen=True, slots=True)
class _SourceFile:
    source: Path
    relative_path: str
    device: int
    inode: int
    size: int
    modified_ns: int


def _text(value: object, label: str, maximum: int = 256) -> str:
    if not is_bounded_text(value, maximum) or any(
        character in value for character in ("\x00", "\n", "\r", "\u2028", "\u2029")
    ):
        raise ArtifactPublicationError(f"artifact {label} is invalid")
    assert isinstance(value, str)
    return value


def _within(path: Path, roots: tuple[Path, ...]) -> bool:
    return any(path == root or root in path.parents for root in roots)


def _roots(values: object, label: str) -> tuple[Path, ...]:
    if not isinstance(values, (tuple, list)) or not values:
        raise ArtifactPublicationError(f"workspace {label} is missing")
    roots: list[Path] = []
    for value in values:
        if not isinstance(value, str) or not Path(value).is_absolute():
            raise ArtifactPublicationError(
                f"workspace {label} contains an invalid root"
            )
        try:
            root = Path(value).resolve(strict=True)
        except OSError as error:
            raise ArtifactPublicationError(
                f"workspace {label} root is unavailable"
            ) from error
        if not root.is_dir():
            raise ArtifactPublicationError(
                f"workspace {label} root is not a directory"
            )
        roots.append(root)
    return tuple(roots)


def _relative_text(path: Path) -> str:
    value = path.as_posix()
    if (
        not value
        or value.startswith("/")
        or value in {".", ".."}
        or ".." in path.parts
        or len(value.encode("utf-8")) > MAXIMUM_RELATIVE_PATH_BYTES
        or any(
            character in value
            for character in ("\x00", "\n", "\r", "\u2028", "\u2029")
        )
    ):
        raise ArtifactPublicationError("artifact relative path is invalid")
    return value


def _scan_tree(root: Path) -> tuple[_SourceFile, ...]:
    files: list[_SourceFile] = []

    def visit(directory: Path, relative: Path) -> None:
        try:
            entries = sorted(os.scandir(directory), key=lambda item: item.name)
        except OSError as error:
            raise ArtifactPublicationError(
                "artifact directory could not be enumerated"
            ) from error
        for entry in entries:
            child_relative = relative / entry.name
            try:
                info = entry.stat(follow_symlinks=False)
            except OSError as error:
                raise ArtifactPublicationError(
                    "artifact entry identity is unavailable"
                ) from error
            if stat.S_ISLNK(info.st_mode):
                raise ArtifactPublicationError("artifact tree contains a symlink")
            if stat.S_ISDIR(info.st_mode):
                visit(Path(entry.path), child_relative)
                continue
            if not stat.S_ISREG(info.st_mode):
                raise ArtifactPublicationError(
                    "artifact tree contains a nonregular entry"
                )
            files.append(
                _SourceFile(
                    source=Path(entry.path),
                    relative_path=_relative_text(child_relative),
                    device=info.st_dev,
                    inode=info.st_ino,
                    size=info.st_size,
                    modified_ns=info.st_mtime_ns,
                )
            )
            if len(files) > MAXIMUM_ARTIFACT_FILES:
                raise ArtifactPublicationError(
                    "artifact tree exceeds the file-count bound"
                )

    visit(root, Path())
    if not files:
        raise ArtifactPublicationError("artifact directory contains no files")
    if sum(item.size for item in files) > MAXIMUM_ARTIFACT_BYTES:
        raise ArtifactPublicationError("artifact tree exceeds the byte bound")
    return tuple(files)


def _same_source(left: _SourceFile, right: _SourceFile) -> bool:
    return (
        left.relative_path,
        left.device,
        left.inode,
        left.size,
        left.modified_ns,
    ) == (
        right.relative_path,
        right.device,
        right.inode,
        right.size,
        right.modified_ns,
    )


def _copy_file(source: _SourceFile, destination: Path) -> str:
    destination.parent.mkdir(mode=0o750, parents=True, exist_ok=True)
    source_flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        source_descriptor = os.open(source.source, source_flags)
    except OSError as error:
        raise ArtifactPublicationError(
            "artifact source could not be opened safely"
        ) from error
    digest = hashlib.sha256()
    copied = 0
    try:
        before = os.fstat(source_descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_dev != source.device
            or before.st_ino != source.inode
            or before.st_size != source.size
            or before.st_mtime_ns != source.modified_ns
        ):
            raise ArtifactPublicationError(
                "artifact source changed before it was frozen"
            )
        destination_descriptor = os.open(
            destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o440
        )
        try:
            while chunk := os.read(source_descriptor, 4 * 1024 * 1024):
                copied += len(chunk)
                if copied > source.size:
                    raise ArtifactPublicationError(
                        "artifact source grew while it was frozen"
                    )
                digest.update(chunk)
                view = memoryview(chunk)
                while view:
                    written = os.write(destination_descriptor, view)
                    if written <= 0:
                        raise ArtifactPublicationError(
                            "artifact destination stopped accepting bytes"
                        )
                    view = view[written:]
            os.fsync(destination_descriptor)
        finally:
            os.close(destination_descriptor)
        after = os.fstat(source_descriptor)
        if copied != source.size or (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ) != (
            source.device,
            source.inode,
            source.size,
            source.modified_ns,
        ):
            raise ArtifactPublicationError(
                "artifact source changed while it was frozen"
            )
    finally:
        os.close(source_descriptor)
    return "sha256:" + digest.hexdigest()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _verify_revision(
    revision: Path,
    manifest_bytes: bytes,
    objects: tuple[Mapping[str, object], ...],
) -> None:
    if revision.is_symlink():
        raise ArtifactPublicationError("artifact revision is invalid")
    try:
        resolved_revision = revision.resolve(strict=True)
    except OSError as error:
        raise ArtifactPublicationError("artifact revision is unavailable") from error
    if not resolved_revision.is_dir():
        raise ArtifactPublicationError("artifact revision is invalid")
    manifest = resolved_revision / "manifest.json"
    if (
        not manifest.is_file()
        or manifest.is_symlink()
        or manifest.read_bytes() != manifest_bytes
    ):
        raise ArtifactPublicationError(
            "artifact revision identity exists with different bytes"
        )
    payload = resolved_revision / "payload"
    if payload.is_symlink():
        raise ArtifactPublicationError("artifact revision payload is invalid")
    try:
        payload = payload.resolve(strict=True)
    except OSError as error:
        raise ArtifactPublicationError(
            "artifact revision payload is unavailable"
        ) from error
    if payload.parent != resolved_revision or not payload.is_dir():
        raise ArtifactPublicationError("artifact revision payload is invalid")
    observed = _scan_tree(payload)
    if len(observed) != len(objects):
        raise ArtifactPublicationError("artifact revision payload was mutated")
    for source, expected in zip(observed, objects, strict=True):
        if (
            source.relative_path != expected.get("relative_path")
            or source.size != expected.get("size_bytes")
        ):
            raise ArtifactPublicationError("artifact revision payload was mutated")
        raw = _read_regular_file_stably(source.source, source.size)
        if sha256_digest(raw) != expected.get("sha256"):
            raise ArtifactPublicationError("artifact revision payload was mutated")


def _read_regular_file_stably(path: Path, maximum_bytes: int) -> bytes:
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except OSError as error:
        raise ArtifactPublicationError("artifact manifest is unavailable") from error
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size > maximum_bytes:
            raise ArtifactPublicationError("artifact manifest is invalid")
        chunks: list[bytes] = []
        observed = 0
        while chunk := os.read(descriptor, min(1024 * 1024, maximum_bytes + 1)):
            observed += len(chunk)
            if observed > maximum_bytes:
                raise ArtifactPublicationError("artifact manifest exceeds its bound")
            chunks.append(chunk)
        after = os.fstat(descriptor)
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ):
            raise ArtifactPublicationError("artifact manifest changed while read")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _manifest_path_from_uri(uri: object) -> Path:
    if not is_bounded_text(uri, 4096):
        raise ArtifactPublicationError("artifact URI is invalid")
    assert isinstance(uri, str)
    parsed = urlsplit(uri)
    if (
        parsed.scheme != "file"
        or parsed.netloc
        or parsed.query
        or parsed.fragment
        or not parsed.path.startswith("/")
    ):
        raise ArtifactPublicationError("artifact URI is not a local file URI")
    try:
        decoded = unquote(parsed.path, errors="strict")
    except UnicodeDecodeError as error:
        raise ArtifactPublicationError("artifact URI is not UTF-8") from error
    path = Path(decoded)
    if (
        not path.is_absolute()
        or path != Path(os.path.normpath(path))
        or path.as_uri() != uri
        or path.name != "manifest.json"
    ):
        raise ArtifactPublicationError("artifact manifest URI is noncanonical")
    return path


def resolve_input_artifact(
    invocation: object,
    input_name: str,
    *,
    required_kind: str,
    required_schema: str,
) -> ResolvedInputArtifact:
    """Resolve and fully verify one controller-selected immutable input artifact."""

    inputs = getattr(invocation, "inputs", None)
    workspace = getattr(invocation, "workspace", None)
    if not isinstance(inputs, Mapping) or not isinstance(workspace, Mapping):
        raise ArtifactPublicationError("artifact input authority is invalid")
    artifact = inputs.get(input_name)
    if not isinstance(artifact, Mapping):
        raise ArtifactPublicationError("artifact input descriptor is missing")
    try:
        exact_fields(artifact, _ARTIFACT_FIELDS)
    except CanonicalJsonError as error:
        raise ArtifactPublicationError(
            "artifact input descriptor fields are inexact"
        ) from error
    parents = artifact.get("parent_artifact_ids")
    valid_parents = (
        isinstance(parents, (tuple, list))
        and len(parents) == len(set(parents))
        and all(is_bounded_text(parent, 1024) for parent in parents)
    )
    if (
        not is_bounded_text(artifact.get("artifact_id"), 1024)
        or not is_bounded_text(artifact.get("logical_name"), 1024)
        or artifact.get("kind") != required_kind
        or artifact.get("schema") != required_schema
        or required_kind not in _KINDS
        or not is_uint64(artifact.get("size_bytes"))
        or artifact.get("fingerprint_algorithm") != "manifest_sha256"
        or not is_digest(artifact.get("fingerprint"))
        or artifact.get("complete") is not True
        or not is_bounded_text(artifact.get("producer_node_id"), 1024)
        or not is_bounded_text(artifact.get("producer_attempt_id"), 1024)
        or not valid_parents
        or not is_uint64(artifact.get("published_at_ns"))
    ):
        raise ArtifactPublicationError("artifact input descriptor is invalid")

    manifest_path = _manifest_path_from_uri(artifact.get("uri"))
    if manifest_path.is_symlink():
        raise ArtifactPublicationError("artifact manifest cannot be a symlink")
    raw_roots = tuple(workspace.get("allowed_read_roots") or ()) + tuple(
        workspace.get("allowed_write_roots") or ()
    )
    roots = _roots(raw_roots, "artifact input roots")
    try:
        manifest_path = manifest_path.resolve(strict=True)
    except OSError as error:
        raise ArtifactPublicationError("artifact manifest is unavailable") from error
    if not _within(manifest_path, roots):
        raise ArtifactPublicationError("artifact manifest escaped workspace authority")
    raw = _read_regular_file_stably(
        manifest_path, MAXIMUM_ARTIFACT_MANIFEST_BYTES
    )
    if sha256_digest(raw) != artifact.get("fingerprint"):
        raise ArtifactPublicationError("artifact manifest fingerprint disagrees")
    try:
        manifest = canonical_loads(
            raw, maximum_bytes=MAXIMUM_ARTIFACT_MANIFEST_BYTES
        )
        exact_fields(manifest, _MANIFEST_FIELDS)
    except CanonicalJsonError as error:
        raise ArtifactPublicationError("artifact manifest is not canonical") from error

    producer = manifest.get("producer")
    objects = manifest.get("objects")
    manifest_parents = manifest.get("parent_artifact_ids")
    tree_digest = manifest.get("canonical_tree_digest")
    body = dict(manifest)
    del body["canonical_tree_digest"]
    if (
        manifest.get("schema") != IMMUTABLE_TREE_SCHEMA
        or manifest.get("artifact_schema") != required_schema
        or manifest.get("artifact_kind") != required_kind
        or not is_digest(manifest.get("invocation_digest"))
        or not isinstance(producer, dict)
        or set(producer) != {"run_id", "node_id", "attempt_id"}
        or not is_bounded_text(producer.get("run_id"), 1024)
        or producer.get("node_id") != artifact.get("producer_node_id")
        or producer.get("attempt_id") != artifact.get("producer_attempt_id")
        or manifest.get("payload_directory") != "payload"
        or not is_uint64(manifest.get("file_count"), positive=True)
        or not is_uint64(manifest.get("payload_size_bytes"))
        or not isinstance(objects, list)
        or len(objects) != manifest.get("file_count")
        or len(objects) > MAXIMUM_ARTIFACT_FILES
        or manifest_parents != list(parents)
        or not is_digest(tree_digest)
        or sha256_digest(canonical_dumps(body)) != tree_digest
    ):
        raise ArtifactPublicationError("artifact manifest semantics are invalid")
    total_size = 0
    seen: set[str] = set()
    resolved_objects: list[ResolvedArtifactObject] = []
    for item in objects:
        if not isinstance(item, dict) or set(item) != {
            "relative_path",
            "sha256",
            "size_bytes",
        }:
            raise ArtifactPublicationError("artifact object record is invalid")
        relative = item.get("relative_path")
        size = item.get("size_bytes")
        try:
            valid_relative = (
                isinstance(relative, str)
                and _relative_text(Path(relative)) == relative
            )
        except ArtifactPublicationError:
            valid_relative = False
        if (
            not valid_relative
            or relative in seen
            or not is_digest(item.get("sha256"))
            or not is_uint64(size)
        ):
            raise ArtifactPublicationError("artifact object identity is invalid")
        assert isinstance(relative, str) and isinstance(size, int)
        seen.add(relative)
        total_size += size
        resolved_objects.append(
            ResolvedArtifactObject(
                relative_path=relative,
                sha256=str(item["sha256"]),
                size_bytes=size,
            )
        )
        if total_size > MAXIMUM_ARTIFACT_BYTES:
            raise ArtifactPublicationError("artifact payload exceeds its byte bound")
    if (
        total_size != manifest.get("payload_size_bytes")
        or artifact.get("size_bytes") != total_size + len(raw)
    ):
        raise ArtifactPublicationError("artifact size authority is inconsistent")
    revision = manifest_path.parent
    _verify_revision(revision, raw, tuple(objects))
    payload = (revision / "payload").resolve(strict=True)
    return ResolvedInputArtifact(
        artifact_id=str(artifact["artifact_id"]),
        logical_name=str(artifact["logical_name"]),
        kind=required_kind,
        schema=required_schema,
        manifest_path=manifest_path,
        payload_directory=payload,
        manifest_sha256=str(artifact["fingerprint"]),
        payload_size_bytes=total_size,
        file_count=len(objects),
        parent_artifact_ids=tuple(parents),
        canonical_tree_digest=str(tree_digest),
        objects=tuple(resolved_objects),
    )


def read_input_artifact_file(
    artifact: ResolvedInputArtifact,
    relative_path: str,
    *,
    maximum_bytes: int,
) -> bytes:
    """Read one declared artifact object and recheck its immutable identity."""

    try:
        normalized = _relative_text(Path(relative_path))
    except ArtifactPublicationError as error:
        raise ArtifactPublicationError("artifact object request is invalid") from error
    expected = next(
        (item for item in artifact.objects if item.relative_path == normalized),
        None,
    )
    if expected is None:
        raise ArtifactPublicationError("artifact object is not in its manifest")
    if maximum_bytes < 0 or expected.size_bytes > maximum_bytes:
        raise ArtifactPublicationError("artifact object exceeds its read bound")
    candidate = artifact.payload_directory / normalized
    if candidate.is_symlink():
        raise ArtifactPublicationError("artifact object cannot be a symlink")
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as error:
        raise ArtifactPublicationError("artifact object is unavailable") from error
    if (
        resolved != candidate
        or artifact.payload_directory not in resolved.parents
        or not resolved.is_file()
    ):
        raise ArtifactPublicationError("artifact object escaped its payload")
    raw = _read_regular_file_stably(resolved, maximum_bytes)
    if (
        len(raw) != expected.size_bytes
        or sha256_digest(raw) != expected.sha256
    ):
        raise ArtifactPublicationError("artifact object identity disagrees")
    return raw


def load_input_artifact_json(
    artifact: ResolvedInputArtifact,
    relative_path: str,
    *,
    maximum_bytes: int,
) -> Mapping[str, object]:
    """Load one canonical JSON object from a verified immutable artifact."""

    raw = read_input_artifact_file(
        artifact, relative_path, maximum_bytes=maximum_bytes
    )
    try:
        return canonical_loads(raw, maximum_bytes=maximum_bytes)
    except CanonicalJsonError as error:
        raise ArtifactPublicationError(
            "artifact JSON object is not canonical"
        ) from error


class ImmutableArtifactPublisher:
    """Freeze one authority-declared non-checkpoint directory output."""

    def __init__(self, session: _ArtifactSession, *, output_name: str):
        self._session = session
        self._output_name = _text(output_name, "output name")
        if self._output_name in {".", ".."} or "/" in self._output_name:
            raise ArtifactPublicationError(
                "artifact output name must be one path component"
            )
        invocation = session.invocation
        workspace = getattr(invocation, "workspace", None)
        publishes = getattr(invocation, "publishes", None)
        if not isinstance(workspace, Mapping) or not isinstance(publishes, Mapping):
            raise ArtifactPublicationError(
                "worker artifact publication contract is invalid"
            )
        output = publishes.get(self._output_name)
        declaration = output.get("declaration") if isinstance(output, Mapping) else None
        kind = declaration.get("type") if isinstance(declaration, Mapping) else None
        if (
            not isinstance(output, Mapping)
            or not isinstance(declaration, Mapping)
            or kind not in _KINDS
            or declaration.get("immutability") != "immutable"
            or declaration.get("fingerprint") != "manifest_sha256"
        ):
            raise ArtifactPublicationError(
                "artifact output declaration is incompatible with immutable snapshots"
            )
        self._kind_name = str(kind)
        self._kind = _KINDS[self._kind_name]
        self._artifact_schema = _text(
            declaration.get("schema"), "artifact schema", 512
        )
        self._logical_name = _text(output.get("logical_name"), "logical name")
        write_roots = _roots(workspace.get("allowed_write_roots"), "allowed_write_roots")
        run_directory = workspace.get("run_directory")
        if not isinstance(run_directory, str) or not Path(run_directory).is_absolute():
            raise ArtifactPublicationError("workspace run directory is invalid")
        try:
            self._run_root = Path(run_directory).resolve(strict=True)
        except OSError as error:
            raise ArtifactPublicationError(
                "workspace run directory is unavailable"
            ) from error
        if not _within(self._run_root, write_roots):
            raise ArtifactPublicationError(
                "workspace run directory is outside declared write roots"
            )
        revision_root = self._run_root
        for component in ("trainvm_artifacts", "outputs", self._output_name):
            candidate = revision_root / component
            try:
                candidate.mkdir(mode=0o750)
            except FileExistsError:
                pass
            except OSError as error:
                raise ArtifactPublicationError(
                    "artifact publication root is unavailable"
                ) from error
            if candidate.is_symlink():
                raise ArtifactPublicationError(
                    "artifact publication root contains a symlink"
                )
            try:
                resolved_candidate = candidate.resolve(strict=True)
            except OSError as error:
                raise ArtifactPublicationError(
                    "artifact publication root is unavailable"
                ) from error
            if (
                not resolved_candidate.is_dir()
                or resolved_candidate.parent != revision_root
            ):
                raise ArtifactPublicationError(
                    "artifact publication root escaped the run workspace"
                )
            revision_root = resolved_candidate
        self._revision_root = revision_root
        if not _within(self._revision_root, write_roots):
            raise ArtifactPublicationError(
                "artifact publication root is outside declared write roots"
            )

    def publish(
        self,
        source_directory: str | os.PathLike[str],
        *,
        parent_artifact_ids: Iterable[str] = (),
        progress: Callable[[], None] | None = None,
        optimizer_step: int | None = None,
    ) -> PublishedArtifact:
        source = Path(source_directory)
        if not source.is_absolute():
            raise ArtifactPublicationError(
                "artifact source directory must be absolute"
            )
        try:
            source = source.resolve(strict=True)
        except OSError as error:
            raise ArtifactPublicationError(
                "artifact source directory is unavailable"
            ) from error
        if not source.is_dir() or not _within(source, (self._run_root,)):
            raise ArtifactPublicationError(
                "artifact source is not a directory inside the run workspace"
            )
        if (
            source == self._run_root
            or source == self._revision_root
            or self._revision_root in source.parents
            or source in self._revision_root.parents
        ):
            raise ArtifactPublicationError(
                "artifact source cannot be the publication namespace"
            )
        parents = tuple(parent_artifact_ids)
        if len(parents) != len(set(parents)) or any(
            not is_bounded_text(parent, 1024)
            or any(character in parent for character in ("\u2028", "\u2029"))
            for parent in parents
        ):
            raise ArtifactPublicationError("artifact parent identities are invalid")

        snapshot = _scan_tree(source)
        temporary = Path(tempfile.mkdtemp(prefix="snapshot-", dir=self._revision_root))
        try:
            payload = temporary / "payload"
            payload.mkdir(mode=0o750)
            objects: list[Mapping[str, object]] = []
            for item in snapshot:
                if progress is not None:
                    progress()
                digest = _copy_file(item, payload / item.relative_path)
                objects.append(
                    {
                        "relative_path": item.relative_path,
                        "sha256": digest,
                        "size_bytes": item.size,
                    }
                )
            rescanned = _scan_tree(source)
            if len(snapshot) != len(rescanned) or any(
                not _same_source(left, right)
                for left, right in zip(snapshot, rescanned, strict=True)
            ):
                raise ArtifactPublicationError(
                    "artifact tree changed while it was frozen"
                )
            payload_size = sum(item.size for item in snapshot)
            bootstrap = self._session.bootstrap
            invocation_digest = getattr(self._session.invocation, "invocation_digest", None)
            if not is_digest(invocation_digest):
                raise ArtifactPublicationError(
                    "artifact publisher has no sealed invocation identity"
                )
            body = {
                "schema": IMMUTABLE_TREE_SCHEMA,
                "artifact_schema": self._artifact_schema,
                "artifact_kind": self._kind_name,
                "invocation_digest": invocation_digest,
                "producer": {
                    "run_id": _text(bootstrap.run_id, "producer run ID", 1024),
                    "node_id": _text(bootstrap.node_id, "producer node ID", 1024),
                    "attempt_id": _text(
                        bootstrap.attempt_id, "producer attempt ID", 1024
                    ),
                },
                "payload_directory": "payload",
                "file_count": len(objects),
                "payload_size_bytes": payload_size,
                "objects": objects,
                "parent_artifact_ids": list(parents),
            }
            tree_digest = sha256_digest(canonical_dumps(body))
            manifest_bytes = canonical_dumps(
                {**body, "canonical_tree_digest": tree_digest}
            )
            if len(manifest_bytes) > MAXIMUM_ARTIFACT_MANIFEST_BYTES:
                raise ArtifactPublicationError(
                    "artifact manifest exceeds its byte bound"
                )
            manifest_sha256 = sha256_digest(manifest_bytes)
            artifact_id = "artifact-" + hashlib.sha256(
                canonical_dumps(
                    [
                        bootstrap.run_id,
                        bootstrap.node_id,
                        bootstrap.attempt_id,
                        self._output_name,
                        tree_digest,
                    ]
                )
            ).hexdigest()
            manifest = temporary / "manifest.json"
            descriptor = os.open(
                manifest, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o440
            )
            try:
                with os.fdopen(descriptor, "wb") as destination:
                    destination.write(manifest_bytes)
                    destination.flush()
                    os.fsync(destination.fileno())
            except BaseException:
                try:
                    os.close(descriptor)
                except OSError:
                    pass
                raise
            for directory, directories, _files in os.walk(payload, topdown=False):
                for name in directories:
                    os.chmod(Path(directory) / name, 0o550)
                os.chmod(directory, 0o550)
                _fsync_directory(Path(directory))
            os.chmod(temporary, 0o550)
            _fsync_directory(temporary)
            revision = self._revision_root / artifact_id
            try:
                os.rename(temporary, revision)
                _fsync_directory(self._revision_root)
            except OSError:
                _verify_revision(revision, manifest_bytes, tuple(objects))
            manifest = revision / "manifest.json"
        finally:
            if temporary.exists():
                for directory, directories, files in os.walk(temporary):
                    os.chmod(directory, 0o750)
                    for name in directories:
                        os.chmod(Path(directory) / name, 0o750)
                    for name in files:
                        os.chmod(Path(directory) / name, 0o640)
                shutil.rmtree(temporary)

        sequence = self._session.artifact(
            artifact_id=artifact_id,
            logical_name=self._logical_name,
            kind=self._kind,
            schema=self._artifact_schema,
            uri=manifest.resolve(strict=True).as_uri(),
            size_bytes=payload_size + len(manifest_bytes),
            fingerprint_algorithm="manifest_sha256",
            fingerprint=manifest_sha256,
            parent_artifact_ids=parents,
            optimizer_step=optimizer_step,
            wait=True,
        )
        return PublishedArtifact(
            artifact_id=artifact_id,
            manifest_path=manifest,
            manifest_sha256=manifest_sha256,
            payload_size_bytes=payload_size,
            file_count=len(objects),
            worker_sequence=sequence,
        )


def publish_artifact_requests(
    session: _ArtifactSession,
    requests: Iterable[ArtifactPublicationRequest],
    *,
    progress: Callable[[], None] | None = None,
) -> tuple[PublishedArtifact, ...]:
    publishers: dict[str, ImmutableArtifactPublisher] = {}
    published: list[PublishedArtifact] = []
    for request in requests:
        publisher = publishers.get(request.output_name)
        if publisher is None:
            publisher = ImmutableArtifactPublisher(
                session, output_name=request.output_name
            )
            publishers[request.output_name] = publisher
        published.append(
            publisher.publish(
                request.source_directory,
                parent_artifact_ids=request.parent_artifact_ids,
                optimizer_step=request.optimizer_step,
                progress=progress,
            )
        )
    return tuple(published)


__all__ = [
    "IMMUTABLE_TREE_SCHEMA",
    "ArtifactPublicationError",
    "ArtifactPublicationRequest",
    "ImmutableArtifactPublisher",
    "PublishedArtifact",
    "publish_artifact_requests",
]
