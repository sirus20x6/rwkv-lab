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
from types import SimpleNamespace
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
        "TrainVM checkpoint publication requires the 'trainvm-worker' project extra"
    ) from error


CHECKPOINT_SNAPSHOT_SCHEMA = "trainvm.checkpoint-snapshot.v1"
MAXIMUM_CHECKPOINT_FILES = 131_072
MAXIMUM_CHECKPOINT_BYTES = 16 * 1024 * 1024 * 1024 * 1024
MAXIMUM_RELATIVE_PATH_BYTES = 4096
MAXIMUM_CHECKPOINT_MANIFEST_BYTES = 128 * 1024 * 1024
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
        "optimizer_groups",
        "parameter_routing",
        "plateau_state",
        "rng_accelerator",
        "rng_numpy",
        "rng_python",
        "rng_torch",
        "topology",
        "weight_decay_schedule",
    }
)
_FICLONE = 0x40049409
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
class ResolvedResumeCheckpoint:
    artifact_id: str
    manifest_path: Path
    payload_directory: Path
    optimizer_step: int
    resume_grade: str
    state_components: tuple[str, ...]
    canonical_tree_digest: str


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


def _read_stable_regular_file(path: Path, maximum_bytes: int) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise CheckpointPublicationError(
            "resume checkpoint manifest could not be opened safely"
        ) from error
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_size <= 0
            or before.st_size > maximum_bytes
        ):
            raise CheckpointPublicationError(
                "resume checkpoint manifest size or type is invalid"
            )
        chunks: list[bytes] = []
        total = 0
        while chunk := os.read(descriptor, min(4 * 1024 * 1024, maximum_bytes + 1)):
            total += len(chunk)
            if total > maximum_bytes:
                raise CheckpointPublicationError(
                    "resume checkpoint manifest exceeds its byte bound"
                )
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
            raise CheckpointPublicationError(
                "resume checkpoint manifest changed while read"
            )
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def resolve_resume_checkpoint(invocation: object) -> ResolvedResumeCheckpoint | None:
    """Verify and resolve the controller-selected immutable resume snapshot."""

    resume = getattr(invocation, "resume", None)
    if resume is None:
        return None
    workspace = getattr(invocation, "workspace", None)
    node_id = getattr(invocation, "node_id", None)
    attempt_id = getattr(invocation, "attempt_id", None)
    if (
        not isinstance(resume, Mapping)
        or not isinstance(workspace, Mapping)
        or not is_bounded_text(node_id, 1024)
        or not is_bounded_text(attempt_id, 1024)
    ):
        raise CheckpointPublicationError(
            "resume checkpoint invocation authority is invalid"
        )
    checkpoint = resume.get("checkpoint")
    if not isinstance(checkpoint, Mapping):
        raise CheckpointPublicationError(
            "resume checkpoint artifact authority is invalid"
        )
    uri = checkpoint.get("uri")
    if not isinstance(uri, str):
        raise CheckpointPublicationError("resume checkpoint URI is invalid")
    parsed = urlsplit(uri)
    if (
        parsed.scheme != "file"
        or parsed.netloc
        or parsed.query
        or parsed.fragment
        or not parsed.path.startswith("/")
    ):
        raise CheckpointPublicationError(
            "resume checkpoint must use a local absolute file URI"
        )
    decoded_path = unquote(parsed.path, errors="strict")
    manifest_path = Path(decoded_path)
    if (
        not manifest_path.is_absolute()
        or manifest_path != Path(os.path.normpath(manifest_path))
    ):
        raise CheckpointPublicationError(
            "resume checkpoint manifest path is noncanonical"
        )
    if manifest_path.is_symlink():
        raise CheckpointPublicationError(
            "resume checkpoint manifest cannot be a symlink"
        )
    read_roots = workspace.get("allowed_read_roots")
    write_roots = workspace.get("allowed_write_roots")
    if not isinstance(read_roots, (tuple, list)) or not isinstance(
        write_roots, (tuple, list)
    ):
        raise CheckpointPublicationError(
            "resume checkpoint workspace roots are invalid"
        )
    roots = _roots(tuple(read_roots) + tuple(write_roots), "resume roots")
    try:
        manifest_path = manifest_path.resolve(strict=True)
    except OSError as error:
        raise CheckpointPublicationError(
            "resume checkpoint manifest is unavailable"
        ) from error
    if not _within(manifest_path, roots):
        raise CheckpointPublicationError(
            "resume checkpoint manifest escaped workspace authority"
        )
    raw = _read_stable_regular_file(
        manifest_path, MAXIMUM_CHECKPOINT_MANIFEST_BYTES
    )
    fingerprint = checkpoint.get("fingerprint")
    if (
        checkpoint.get("fingerprint_algorithm") != "manifest_sha256"
        or not is_digest(fingerprint)
        or sha256_digest(raw) != fingerprint
    ):
        raise CheckpointPublicationError(
            "resume checkpoint manifest fingerprint disagrees with authority"
        )
    try:
        manifest = canonical_loads(
            raw, maximum_bytes=MAXIMUM_CHECKPOINT_MANIFEST_BYTES
        )
        exact_fields(
            manifest,
            frozenset(
                {
                    "schema", "checkpoint_schema", "producer", "optimizer_step",
                    "resume_grade", "state_components", "payload_directory",
                    "file_count", "payload_size_bytes", "objects",
                    "parent_artifact_ids", "canonical_tree_digest",
                }
            ),
        )
    except CanonicalJsonError as error:
        raise CheckpointPublicationError(
            "resume checkpoint manifest is not canonical"
        ) from error
    tree_digest = manifest.get("canonical_tree_digest")
    body = dict(manifest)
    del body["canonical_tree_digest"]
    producer = manifest.get("producer")
    components = manifest.get("state_components")
    objects = manifest.get("objects")
    parents = manifest.get("parent_artifact_ids")
    authority_parents = checkpoint.get("parent_artifact_ids")
    optimizer_step = manifest.get("optimizer_step")
    resume_grade = manifest.get("resume_grade")
    valid_components = (
        isinstance(components, list)
        and bool(components)
        and all(isinstance(component, str) for component in components)
        and components == sorted(set(components))
        and all(component in _STATE_COMPONENTS for component in components)
    )
    valid_parents = (
        isinstance(parents, list)
        and all(is_bounded_text(parent, 1024) for parent in parents)
        and len(parents) == len(set(parents))
        and parents == authority_parents
    )
    if (
        manifest.get("schema") != CHECKPOINT_SNAPSHOT_SCHEMA
        or manifest.get("checkpoint_schema") != checkpoint.get("schema")
        or not isinstance(producer, dict)
        or set(producer) != {"run_id", "node_id", "attempt_id"}
        or producer.get("run_id") != getattr(invocation, "run_id", None)
        or producer.get("node_id") != node_id
        or producer.get("attempt_id") != checkpoint.get("producer_attempt_id")
        or producer.get("attempt_id") == attempt_id
        or optimizer_step != resume.get("optimizer_step")
        or not is_uint64(optimizer_step)
        or resume_grade not in {"compatible", "exact"}
        or not valid_components
        or manifest.get("payload_directory") != "payload"
        or not is_uint64(manifest.get("file_count"))
        or not is_uint64(manifest.get("payload_size_bytes"))
        or not isinstance(objects, list)
        or len(objects) != manifest.get("file_count")
        or len(objects) > MAXIMUM_CHECKPOINT_FILES
        or not valid_parents
        or not is_digest(tree_digest)
        or sha256_digest(canonical_dumps(body)) != tree_digest
    ):
        raise CheckpointPublicationError(
            "resume checkpoint manifest semantics are invalid"
        )
    total_size = 0
    seen_paths: set[str] = set()
    for item in objects:
        if not isinstance(item, dict) or set(item) != {
            "relative_path", "sha256", "size_bytes"
        }:
            raise CheckpointPublicationError(
                "resume checkpoint object record is invalid"
            )
        relative = item.get("relative_path")
        size = item.get("size_bytes")
        if (
            not isinstance(relative, str)
            or _relative_text(Path(relative)) != relative
            or relative in seen_paths
            or not is_uint64(size)
            or not is_digest(item.get("sha256"))
        ):
            raise CheckpointPublicationError(
                "resume checkpoint object identity is invalid"
            )
        seen_paths.add(relative)
        total_size += size
        if total_size > MAXIMUM_CHECKPOINT_BYTES:
            raise CheckpointPublicationError(
                "resume checkpoint payload exceeds its byte bound"
            )
    if total_size != manifest.get("payload_size_bytes"):
        raise CheckpointPublicationError(
            "resume checkpoint payload size is inconsistent"
        )
    if checkpoint.get("size_bytes") != total_size + len(raw):
        raise CheckpointPublicationError(
            "resume checkpoint artifact size disagrees with its manifest"
        )
    revision = manifest_path.parent
    _verify_existing_revision(revision, raw, objects)
    payload = (revision / "payload").resolve(strict=True)
    return ResolvedResumeCheckpoint(
        artifact_id=str(checkpoint.get("artifact_id")),
        manifest_path=manifest_path,
        payload_directory=payload,
        optimizer_step=optimizer_step,
        resume_grade=resume_grade,
        state_components=tuple(components),
        canonical_tree_digest=tree_digest,
    )


def resolve_input_checkpoint(
    invocation: object,
    input_name: str,
    *,
    required_schema: str | None = None,
) -> ResolvedResumeCheckpoint:
    """Verify a checkpoint artifact selected as an ordinary node input.

    The immutable snapshot verifier remains the same one used for resume.  This
    bridge only obtains the manifest's optimizer step through a confined,
    bounded pre-read, then supplies a synthetic consumer identity so the resume
    verifier can enforce producer lineage without requiring the producer and
    consumer to be the same workflow node.
    """

    inputs = getattr(invocation, "inputs", None)
    workspace = getattr(invocation, "workspace", None)
    run_id = getattr(invocation, "run_id", None)
    if (
        not isinstance(inputs, Mapping)
        or not isinstance(workspace, Mapping)
        or not is_bounded_text(run_id, 1024)
    ):
        raise CheckpointPublicationError(
            "checkpoint input invocation authority is invalid"
        )
    checkpoint = inputs.get(input_name)
    if not isinstance(checkpoint, Mapping):
        raise CheckpointPublicationError("checkpoint input descriptor is missing")
    try:
        exact_fields(checkpoint, _ARTIFACT_FIELDS)
    except CanonicalJsonError as error:
        raise CheckpointPublicationError(
            "checkpoint input descriptor fields are inexact"
        ) from error
    schema = checkpoint.get("schema")
    parents = checkpoint.get("parent_artifact_ids")
    valid_parents = (
        isinstance(parents, (tuple, list))
        and len(parents) == len(set(parents))
        and all(is_bounded_text(parent, 1024) for parent in parents)
    )
    if (
        not is_bounded_text(checkpoint.get("artifact_id"), 1024)
        or not is_bounded_text(checkpoint.get("logical_name"), 1024)
        or checkpoint.get("kind") != "checkpoint"
        or not is_bounded_text(schema, 512)
        or (required_schema is not None and schema != required_schema)
        or not is_uint64(checkpoint.get("size_bytes"))
        or checkpoint.get("fingerprint_algorithm") != "manifest_sha256"
        or not is_digest(checkpoint.get("fingerprint"))
        or checkpoint.get("complete") is not True
        or not is_bounded_text(checkpoint.get("producer_node_id"), 1024)
        or not is_bounded_text(checkpoint.get("producer_attempt_id"), 1024)
        or not valid_parents
        or not is_uint64(checkpoint.get("published_at_ns"))
    ):
        raise CheckpointPublicationError("checkpoint input descriptor is invalid")
    uri = checkpoint.get("uri")
    if not isinstance(uri, str):
        raise CheckpointPublicationError("checkpoint input URI is invalid")
    parsed = urlsplit(uri)
    if (
        parsed.scheme != "file"
        or parsed.netloc
        or parsed.query
        or parsed.fragment
        or not parsed.path.startswith("/")
    ):
        raise CheckpointPublicationError(
            "checkpoint input must use a local absolute file URI"
        )
    manifest_path = Path(unquote(parsed.path, errors="strict"))
    if (
        not manifest_path.is_absolute()
        or manifest_path != Path(os.path.normpath(manifest_path))
        or manifest_path.is_symlink()
    ):
        raise CheckpointPublicationError(
            "checkpoint input manifest path is noncanonical"
        )
    read_roots = workspace.get("allowed_read_roots")
    write_roots = workspace.get("allowed_write_roots")
    if not isinstance(read_roots, (tuple, list)) or not isinstance(
        write_roots, (tuple, list)
    ):
        raise CheckpointPublicationError(
            "checkpoint input workspace roots are invalid"
        )
    roots = _roots(tuple(read_roots) + tuple(write_roots), "checkpoint input roots")
    try:
        manifest_path = manifest_path.resolve(strict=True)
    except OSError as error:
        raise CheckpointPublicationError(
            "checkpoint input manifest is unavailable"
        ) from error
    if not _within(manifest_path, roots):
        raise CheckpointPublicationError(
            "checkpoint input manifest escaped workspace authority"
        )
    raw = _read_stable_regular_file(
        manifest_path, MAXIMUM_CHECKPOINT_MANIFEST_BYTES
    )
    try:
        preview = canonical_loads(
            raw, maximum_bytes=MAXIMUM_CHECKPOINT_MANIFEST_BYTES
        )
    except CanonicalJsonError as error:
        raise CheckpointPublicationError(
            "checkpoint input manifest is not canonical"
        ) from error
    optimizer_step = preview.get("optimizer_step")
    if not is_uint64(optimizer_step):
        raise CheckpointPublicationError(
            "checkpoint input optimizer step is invalid"
        )
    producer_attempt = str(checkpoint.get("producer_attempt_id"))
    consumer_attempt = "checkpoint-input-consumer"
    if consumer_attempt == producer_attempt:
        consumer_attempt += "-next"
    bridge = SimpleNamespace(
        resume={"checkpoint": checkpoint, "optimizer_step": optimizer_step},
        workspace=workspace,
        run_id=run_id,
        node_id=checkpoint.get("producer_node_id"),
        attempt_id=consumer_attempt,
    )
    resolved = resolve_resume_checkpoint(bridge)
    if resolved is None:  # pragma: no cover - bridge always supplies resume
        raise CheckpointPublicationError("checkpoint input resolution disappeared")
    return resolved


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
    "ResolvedResumeCheckpoint",
    "publish_checkpoint_requests",
    "resolve_input_checkpoint",
    "resolve_resume_checkpoint",
]
