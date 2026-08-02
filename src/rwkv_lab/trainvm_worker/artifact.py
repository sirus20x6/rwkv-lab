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

from ._canonical import canonical_dumps, is_bounded_text, is_digest, sha256_digest

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


@dataclass(frozen=True, slots=True)
class PublishedArtifact:
    artifact_id: str
    manifest_path: Path
    manifest_sha256: str
    payload_size_bytes: int
    file_count: int
    worker_sequence: int


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
        digest = hashlib.sha256()
        descriptor = os.open(
            source.source, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        )
        try:
            while chunk := os.read(descriptor, 4 * 1024 * 1024):
                digest.update(chunk)
        finally:
            os.close(descriptor)
        if "sha256:" + digest.hexdigest() != expected.get("sha256"):
            raise ArtifactPublicationError("artifact revision payload was mutated")


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
