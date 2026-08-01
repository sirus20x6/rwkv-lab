"""Immutable checkpoint snapshots published through the TrainVM artifact boundary.

Trainers own checkpoint contents and resume semantics.  This module owns the
family-neutral publication mechanics: confining a completed checkpoint to the
authority workspace, copying it out of rolling trainer retention, hashing the
tree, atomically promoting an immutable revision, and announcing that revision
to the controller.
"""

from __future__ import annotations

import fcntl
import hashlib
import os
import shutil
import stat
import tempfile
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from ._canonical import canonical_dumps, is_bounded_text, sha256_digest

try:
    from trainvm.v1 import trainvm_pb2 as wire
except ImportError as error:  # pragma: no cover - installation contract
    raise RuntimeError(
        "TrainVM checkpoint publication requires the 'trainvm-worker' project extra"
    ) from error


CHECKPOINT_SNAPSHOT_SCHEMA = "trainvm.checkpoint-snapshot.v1"
MAXIMUM_CHECKPOINT_FILES = 131_072
MAXIMUM_CHECKPOINT_BYTES = 16 * 1024 * 1024 * 1024 * 1024
MAXIMUM_RELATIVE_PATH_BYTES = 4096
_RESUME_GRADES = frozenset({"terminal_checkpoint", "compatible", "exact"})
_STATE_COMPONENTS = frozenset(
    {
        "component_composition",
        "control_revision",
        "curriculum",
        "data_cursor",
        "expert_routing",
        "gradient_scaler",
        "lr_schedule",
        "model",
        "optimizer",
        "parameter_routing",
        "rng_accelerator",
        "rng_numpy",
        "rng_python",
        "rng_torch",
        "topology",
        "weight_decay_schedule",
    }
)
_FICLONE = 0x40049409


class CheckpointPublicationError(RuntimeError):
    pass


class _ArtifactSession(Protocol):
    bootstrap: object
    invocation: object

    def artifact(self, **values: object) -> int: ...


@dataclass(frozen=True, slots=True)
class CheckpointPublicationRequest:
    source_directory: str | os.PathLike[str]
    optimizer_step: int
    resume_grade: str
    state_components: tuple[str, ...]
    output_name: str = "checkpoint"
    parent_artifact_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class PublishedCheckpoint:
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
        character in value for character in ("\u2028", "\u2029")
    ):
        raise CheckpointPublicationError(f"checkpoint {label} is invalid")
    assert isinstance(value, str)
    return value


def _roots(values: object, label: str) -> tuple[Path, ...]:
    if not isinstance(values, (tuple, list)) or not values:
        raise CheckpointPublicationError(f"workspace {label} is missing")
    result: list[Path] = []
    for value in values:
        if not isinstance(value, str) or not Path(value).is_absolute():
            raise CheckpointPublicationError(
                f"workspace {label} contains an invalid root"
            )
        try:
            root = Path(value).resolve(strict=True)
        except OSError as error:
            raise CheckpointPublicationError(
                f"workspace {label} root is unavailable"
            ) from error
        if not root.is_dir():
            raise CheckpointPublicationError(
                f"workspace {label} root is not a directory"
            )
        result.append(root)
    return tuple(result)


def _within(path: Path, roots: tuple[Path, ...]) -> bool:
    return any(path == root or root in path.parents for root in roots)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _relative_text(path: Path) -> str:
    value = path.as_posix()
    encoded = value.encode("utf-8")
    if (
        not value
        or value.startswith("/")
        or value in {".", ".."}
        or ".." in path.parts
        or not encoded
        or len(encoded) > MAXIMUM_RELATIVE_PATH_BYTES
        or any(
            character in value
            for character in ("\x00", "\n", "\r", "\u2028", "\u2029")
        )
    ):
        raise CheckpointPublicationError("checkpoint relative path is invalid")
    return value


def _scan_tree(root: Path) -> tuple[_SourceFile, ...]:
    files: list[_SourceFile] = []

    def visit(directory: Path, relative: Path) -> None:
        try:
            entries = sorted(os.scandir(directory), key=lambda item: item.name)
        except OSError as error:
            raise CheckpointPublicationError(
                "checkpoint directory could not be enumerated"
            ) from error
        for entry in entries:
            child_relative = relative / entry.name
            try:
                info = entry.stat(follow_symlinks=False)
            except OSError as error:
                raise CheckpointPublicationError(
                    "checkpoint entry identity is unavailable"
                ) from error
            if stat.S_ISLNK(info.st_mode):
                raise CheckpointPublicationError("checkpoint tree contains a symlink")
            if stat.S_ISDIR(info.st_mode):
                visit(Path(entry.path), child_relative)
                continue
            if not stat.S_ISREG(info.st_mode):
                raise CheckpointPublicationError(
                    "checkpoint tree contains a nonregular entry"
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
            if len(files) > MAXIMUM_CHECKPOINT_FILES:
                raise CheckpointPublicationError(
                    "checkpoint tree exceeds the file-count bound"
                )

    visit(root, Path())
    if not files:
        raise CheckpointPublicationError("checkpoint directory contains no files")
    total = sum(item.size for item in files)
    if total > MAXIMUM_CHECKPOINT_BYTES:
        raise CheckpointPublicationError("checkpoint tree exceeds the byte bound")
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


def _copy_file(
    source: _SourceFile, destination: Path, progress: Callable[[], None] | None
) -> str:
    destination.parent.mkdir(mode=0o750, parents=True, exist_ok=True)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        source_descriptor = os.open(source.source, flags)
    except OSError as error:
        raise CheckpointPublicationError(
            "checkpoint source could not be opened safely"
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
            raise CheckpointPublicationError(
                "checkpoint source changed before it was frozen"
            )
        destination_descriptor = os.open(
            destination, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o440
        )
        try:
            cloned = False
            try:
                fcntl.ioctl(destination_descriptor, _FICLONE, source_descriptor)
                cloned = True
            except OSError:
                # Reflink is only a storage/performance optimization. The
                # descriptor-copy path below has the same immutable result.
                os.ftruncate(destination_descriptor, 0)
                os.lseek(destination_descriptor, 0, os.SEEK_SET)
            while chunk := os.read(source_descriptor, 4 * 1024 * 1024):
                if progress is not None:
                    progress()
                copied += len(chunk)
                if copied > source.size:
                    raise CheckpointPublicationError(
                        "checkpoint source grew while it was frozen"
                    )
                digest.update(chunk)
                if not cloned:
                    view = memoryview(chunk)
                    while view:
                        written = os.write(destination_descriptor, view)
                        if written <= 0:
                            raise CheckpointPublicationError(
                                "checkpoint destination stopped accepting bytes"
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
            raise CheckpointPublicationError(
                "checkpoint source changed while it was frozen"
            )
    finally:
        os.close(source_descriptor)
    return "sha256:" + digest.hexdigest()


def _verify_existing_revision(
    revision: Path,
    manifest_bytes: bytes,
    objects: Iterable[Mapping[str, object]],
) -> None:
    manifest_path = revision / "manifest.json"
    if (
        not manifest_path.is_file()
        or manifest_path.is_symlink()
        or manifest_path.read_bytes() != manifest_bytes
    ):
        raise CheckpointPublicationError(
            "checkpoint revision identity exists with different bytes"
        )
    payload = revision / "payload"
    if payload.is_symlink():
        raise CheckpointPublicationError("checkpoint revision payload is invalid")
    try:
        payload = payload.resolve(strict=True)
    except OSError as error:
        raise CheckpointPublicationError(
            "checkpoint revision payload is unavailable"
        ) from error
    if payload.parent != revision.resolve(strict=True) or not payload.is_dir():
        raise CheckpointPublicationError("checkpoint revision payload is invalid")
    observed = _scan_tree(payload)
    expected = tuple(objects)
    if len(observed) != len(expected):
        raise CheckpointPublicationError("checkpoint revision payload was mutated")
    for source, item in zip(observed, expected, strict=True):
        expected_path = item.get("relative_path")
        expected_size = item.get("size_bytes")
        expected_digest = item.get("sha256")
        if (
            source.relative_path != expected_path
            or source.size != expected_size
            or not isinstance(expected_digest, str)
        ):
            raise CheckpointPublicationError("checkpoint revision payload was mutated")
        digest = hashlib.sha256()
        with source.source.open("rb") as input_file:
            while chunk := input_file.read(4 * 1024 * 1024):
                digest.update(chunk)
        if "sha256:" + digest.hexdigest() != expected_digest:
            raise CheckpointPublicationError("checkpoint revision payload was mutated")


class CheckpointPublisher:
    """Freeze completed trainer state and publish one immutable checkpoint."""

    def __init__(self, session: _ArtifactSession, *, output_name: str = "checkpoint"):
        self._session = session
        self._output_name = _text(output_name, "output name")
        invocation = session.invocation
        workspace = getattr(invocation, "workspace", None)
        publishes = getattr(invocation, "publishes", None)
        if not isinstance(workspace, Mapping) or not isinstance(publishes, Mapping):
            raise CheckpointPublicationError(
                "worker checkpoint publication contract is invalid"
            )
        output = publishes.get(self._output_name)
        if not isinstance(output, Mapping):
            raise CheckpointPublicationError(
                f"invocation does not declare output {self._output_name!r}"
            )
        declaration = output.get("declaration")
        if (
            not isinstance(declaration, Mapping)
            or declaration.get("type") != "checkpoint"
            or declaration.get("immutability") != "immutable"
            or declaration.get("fingerprint") != "manifest_sha256"
        ):
            raise CheckpointPublicationError(
                "checkpoint output declaration is incompatible with immutable snapshots"
            )
        self._checkpoint_schema = _text(
            declaration.get("schema"), "artifact schema", 512
        )
        self._logical_name = _text(output.get("logical_name"), "logical name")
        self._write_roots = _roots(
            workspace.get("allowed_write_roots"), "allowed_write_roots"
        )
        run_directory = workspace.get("run_directory")
        if not isinstance(run_directory, str) or not Path(run_directory).is_absolute():
            raise CheckpointPublicationError("workspace run directory is invalid")
        try:
            self._run_root = Path(run_directory).resolve(strict=True)
        except OSError as error:
            raise CheckpointPublicationError(
                "workspace run directory is unavailable"
            ) from error
        if not _within(self._run_root, self._write_roots):
            raise CheckpointPublicationError(
                "workspace run directory is outside declared write roots"
            )
        self._revision_root = (
            self._run_root / "trainvm_artifacts" / "checkpoints" / self._output_name
        )
        self._revision_root.mkdir(mode=0o750, parents=True, exist_ok=True)
        if not _within(self._revision_root.resolve(strict=True), self._write_roots):
            raise CheckpointPublicationError(
                "checkpoint publication root is outside declared write roots"
            )

    def publish(
        self,
        source_directory: str | os.PathLike[str],
        *,
        optimizer_step: int,
        resume_grade: str,
        state_components: Iterable[str],
        parent_artifact_ids: Iterable[str] = (),
        progress: Callable[[int], None] | None = None,
    ) -> PublishedCheckpoint:
        if (
            not isinstance(optimizer_step, int)
            or isinstance(optimizer_step, bool)
            or not 0 <= optimizer_step < 1 << 64
        ):
            raise CheckpointPublicationError(
                "checkpoint optimizer step is outside uint64"
            )
        if resume_grade not in _RESUME_GRADES:
            raise CheckpointPublicationError("checkpoint resume grade is unsupported")
        components = tuple(state_components)
        if (
            not components
            or len(components) != len(set(components))
            or tuple(sorted(components)) != components
            or any(component not in _STATE_COMPONENTS for component in components)
        ):
            raise CheckpointPublicationError(
                "checkpoint state components must be a sorted closed set"
            )
        parents = tuple(parent_artifact_ids)
        if len(parents) != len(set(parents)) or any(
            not is_bounded_text(parent, 1024)
            or any(character in parent for character in ("\u2028", "\u2029"))
            for parent in parents
        ):
            raise CheckpointPublicationError("checkpoint parent identities are invalid")
        source = Path(source_directory)
        if not source.is_absolute():
            raise CheckpointPublicationError(
                "checkpoint source directory must be absolute"
            )
        try:
            source = source.resolve(strict=True)
        except OSError as error:
            raise CheckpointPublicationError(
                "checkpoint source directory is unavailable"
            ) from error
        if not source.is_dir() or not _within(source, (self._run_root,)):
            raise CheckpointPublicationError(
                "checkpoint source is not a directory inside the run workspace"
            )
        if source == self._revision_root or self._revision_root in source.parents:
            raise CheckpointPublicationError(
                "checkpoint source cannot be the publication namespace"
            )

        snapshot = _scan_tree(source)
        if progress is not None:
            progress(optimizer_step)
        temporary = Path(
            tempfile.mkdtemp(prefix="snapshot-", dir=self._revision_root)
        )
        try:
            payload_root = temporary / "payload"
            payload_root.mkdir(mode=0o750)
            objects: list[dict[str, object]] = []
            for item in snapshot:
                destination = payload_root / item.relative_path
                digest = _copy_file(
                    item,
                    destination,
                    (
                        (lambda: progress(optimizer_step))
                        if progress is not None
                        else None
                    ),
                )
                objects.append(
                    {
                        "relative_path": item.relative_path,
                        "sha256": digest,
                        "size_bytes": item.size,
                    }
                )
            rescanned = _scan_tree(source)
            if len(rescanned) != len(snapshot) or any(
                not _same_source(left, right)
                for left, right in zip(snapshot, rescanned, strict=True)
            ):
                raise CheckpointPublicationError(
                    "checkpoint tree changed while it was frozen"
                )
            payload_size = sum(item.size for item in snapshot)
            bootstrap = self._session.bootstrap
            body = {
                "schema": CHECKPOINT_SNAPSHOT_SCHEMA,
                "checkpoint_schema": self._checkpoint_schema,
                "producer": {
                    "run_id": _text(bootstrap.run_id, "producer run ID", 1024),
                    "node_id": _text(bootstrap.node_id, "producer node ID", 1024),
                    "attempt_id": _text(
                        bootstrap.attempt_id, "producer attempt ID", 1024
                    ),
                },
                "optimizer_step": optimizer_step,
                "resume_grade": resume_grade,
                "state_components": list(components),
                "payload_directory": "payload",
                "file_count": len(objects),
                "payload_size_bytes": payload_size,
                "objects": objects,
                "parent_artifact_ids": list(parents),
            }
            tree_digest = sha256_digest(canonical_dumps(body))
            manifest = {**body, "canonical_tree_digest": tree_digest}
            manifest_bytes = canonical_dumps(manifest)
            manifest_sha256 = sha256_digest(manifest_bytes)
            artifact_seed = canonical_dumps(
                [
                    bootstrap.run_id,
                    bootstrap.node_id,
                    bootstrap.attempt_id,
                    self._output_name,
                    tree_digest,
                ]
            )
            artifact_id = "checkpoint-" + hashlib.sha256(artifact_seed).hexdigest()
            revision = self._revision_root / artifact_id
            manifest_path = temporary / "manifest.json"
            descriptor = os.open(
                manifest_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o440
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
            for directory, directories, _files in os.walk(payload_root, topdown=False):
                for name in directories:
                    os.chmod(Path(directory) / name, 0o550)
                os.chmod(directory, 0o550)
                _fsync_directory(Path(directory))
            os.chmod(temporary, 0o550)
            _fsync_directory(temporary)
            try:
                os.rename(temporary, revision)
                _fsync_directory(self._revision_root)
            except OSError:
                _verify_existing_revision(revision, manifest_bytes, objects)
            manifest_path = revision / "manifest.json"
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
            kind=wire.ARTIFACT_KIND_CHECKPOINT,
            schema=self._checkpoint_schema,
            uri=manifest_path.resolve(strict=True).as_uri(),
            size_bytes=payload_size + len(manifest_bytes),
            fingerprint_algorithm="manifest_sha256",
            fingerprint=manifest_sha256,
            parent_artifact_ids=parents,
            wait=True,
        )
        return PublishedCheckpoint(
            artifact_id=artifact_id,
            manifest_path=manifest_path,
            manifest_sha256=manifest_sha256,
            payload_size_bytes=payload_size,
            file_count=len(objects),
            worker_sequence=sequence,
        )


def publish_checkpoint_requests(
    session: _ArtifactSession,
    requests: Iterable[CheckpointPublicationRequest],
    *,
    progress: Callable[[int], None] | None = None,
) -> tuple[PublishedCheckpoint, ...]:
    """Publish handler results without giving family trainers transport authority."""

    publishers: dict[str, CheckpointPublisher] = {}
    results: list[PublishedCheckpoint] = []
    for request in requests:
        publisher = publishers.get(request.output_name)
        if publisher is None:
            publisher = CheckpointPublisher(session, output_name=request.output_name)
            publishers[request.output_name] = publisher
        results.append(
            publisher.publish(
                request.source_directory,
                optimizer_step=request.optimizer_step,
                resume_grade=request.resume_grade,
                state_components=request.state_components,
                parent_artifact_ids=request.parent_artifact_ids,
                progress=progress,
            )
        )
    return tuple(results)


__all__ = [
    "CHECKPOINT_SNAPSHOT_SCHEMA",
    "CheckpointPublicationError",
    "CheckpointPublicationRequest",
    "CheckpointPublisher",
    "PublishedCheckpoint",
    "publish_checkpoint_requests",
]
