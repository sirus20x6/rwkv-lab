"""Family-neutral immutable evaluation examples and step-zero launch evidence."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from ._canonical import canonical_dumps, is_bounded_text, is_digest, sha256_digest

try:
    from trainvm.v1 import trainvm_pb2 as wire
except ImportError as error:  # pragma: no cover
    raise RuntimeError(
        "TrainVM eval publication requires the 'trainvm-worker' project extra"
    ) from error


EVAL_EXAMPLES_SCHEMA = "rwkv-lab.eval-examples.v1"
MAXIMUM_EXAMPLES = 512
MAXIMUM_PARTS = 32
MAXIMUM_MANIFEST_BYTES = 60 * 1024
MAXIMUM_MEDIA_BYTES = 2 * 1024 * 1024 * 1024


class EvalExamplesError(RuntimeError):
    pass


class _ArtifactSession(Protocol):
    bootstrap: object
    invocation: object

    def artifact(self, **values: object) -> int: ...


@dataclass(frozen=True, slots=True)
class EvalMedia:
    kind: str
    path: str | os.PathLike[str]
    media_type: str
    expected_sha256: str | None = None


@dataclass(frozen=True, slots=True)
class EvalEvidencePart:
    kind: str
    text: str | None = None
    media: EvalMedia | None = None
    schema: str | None = None
    value: Mapping[str, Any] | tuple[Any, ...] | list[Any] | None = None


@dataclass(frozen=True, slots=True)
class EvalExample:
    example_id: str
    heldout_item_id: str
    heldout_item_digest: str
    input: tuple[EvalEvidencePart, ...]
    target: tuple[EvalEvidencePart, ...]
    prediction: tuple[EvalEvidencePart, ...]


@dataclass(frozen=True, slots=True)
class EvalExamplesPublicationRequest:
    output_name: str
    optimizer_step: int
    series_id: str
    identity_field: str
    identities_digest: str
    selector_digest: str
    evaluator_component_digest: str
    metric_names: tuple[str, ...]
    checkpoint_artifact_id: str
    checkpoint_manifest_digest: str
    policy_digest: str
    examples: tuple[EvalExample, ...]
    parent_artifact_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PublishedEvalExamples:
    artifact_id: str
    manifest_path: Path
    manifest_sha256: str
    optimizer_step: int
    worker_sequence: int


def _text(value: object, label: str, maximum: int = 1024) -> str:
    if not is_bounded_text(value, maximum) or any(
        character in value for character in ("\x00", "\n", "\r", "\u2028", "\u2029")
    ):
        raise EvalExamplesError(f"eval-examples {label} is invalid")
    assert isinstance(value, str)
    return value


def _digest(value: object, label: str) -> str:
    if not isinstance(value, str) or not is_digest(value):
        raise EvalExamplesError(f"eval-examples {label} is not a sha256 digest")
    return value


def _roots(values: object, label: str) -> tuple[Path, ...]:
    if not isinstance(values, (tuple, list)) or not values:
        raise EvalExamplesError(f"workspace {label} is missing")
    result: list[Path] = []
    for value in values:
        if not isinstance(value, str) or not Path(value).is_absolute():
            raise EvalExamplesError(f"workspace {label} contains an invalid root")
        root = Path(value).resolve(strict=True)
        if not root.is_dir():
            raise EvalExamplesError(f"workspace {label} contains a non-directory")
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


class EvalExamplesPublisher:
    def __init__(self, session: _ArtifactSession, *, output_name: str) -> None:
        self._session = session
        self._output_name = _text(output_name, "output name", 256)
        invocation = session.invocation
        workspace = getattr(invocation, "workspace", None)
        publishes = getattr(invocation, "publishes", None)
        if not isinstance(workspace, Mapping) or not isinstance(publishes, Mapping):
            raise EvalExamplesError("worker invocation publication contract is invalid")
        publication = publishes.get(self._output_name)
        declaration = (
            publication.get("declaration") if isinstance(publication, Mapping) else None
        )
        if (
            not isinstance(publication, Mapping)
            or not isinstance(declaration, Mapping)
            or declaration.get("type") != "eval_examples"
            or declaration.get("schema") != EVAL_EXAMPLES_SCHEMA
            or declaration.get("immutability") != "append_only"
            or declaration.get("fingerprint") != "manifest_sha256"
        ):
            raise EvalExamplesError(
                "output is not a declared eval-examples v1 artifact"
            )
        self._logical_name = _text(publication.get("logical_name"), "logical name", 256)
        self._read_roots = _roots(
            workspace.get("allowed_read_roots"), "allowed_read_roots"
        )
        self._write_roots = _roots(
            workspace.get("allowed_write_roots"), "allowed_write_roots"
        )
        run_directory = workspace.get("run_directory")
        if not isinstance(run_directory, str) or not Path(run_directory).is_absolute():
            raise EvalExamplesError("workspace run_directory is invalid")
        run_root = Path(run_directory).resolve(strict=True)
        if not _within(run_root, self._write_roots):
            raise EvalExamplesError(
                "workspace run_directory is outside write authority"
            )
        self._root = run_root / "trainvm_artifacts" / "eval_examples"
        self._objects = self._root / "objects" / "sha256"
        self._revisions = self._root / "revisions"
        self._objects.mkdir(mode=0o750, parents=True, exist_ok=True)
        self._revisions.mkdir(mode=0o750, parents=True, exist_ok=True)

    def _freeze_media(self, media: EvalMedia) -> dict[str, object]:
        if media.kind not in {"image", "video", "audio"}:
            raise EvalExamplesError("eval media kind is unsupported")
        media_type = _text(media.media_type, "media type", 256)
        source = Path(media.path)
        if not source.is_absolute() or source.is_symlink():
            raise EvalExamplesError("eval media path is not an absolute regular path")
        source = source.resolve(strict=True)
        if not source.is_file() or not _within(
            source, self._read_roots + self._write_roots
        ):
            raise EvalExamplesError("eval media path is outside read authority")
        temporary_fd, temporary_name = tempfile.mkstemp(
            prefix="object-", dir=self._objects
        )
        temporary = Path(temporary_name)
        digest = hashlib.sha256()
        total = 0
        try:
            with (
                os.fdopen(temporary_fd, "wb") as output,
                source.open("rb") as input_file,
            ):
                while chunk := input_file.read(1024 * 1024):
                    total += len(chunk)
                    if total > MAXIMUM_MEDIA_BYTES:
                        raise EvalExamplesError("eval media exceeds its byte bound")
                    digest.update(chunk)
                    output.write(chunk)
                output.flush()
                os.fsync(output.fileno())
            if total == 0:
                raise EvalExamplesError("eval media is empty")
            actual = "sha256:" + digest.hexdigest()
            if media.expected_sha256 is not None and media.expected_sha256 != actual:
                raise EvalExamplesError("eval media disagrees with expected digest")
            destination_dir = self._objects / digest.hexdigest()[:2]
            destination_dir.mkdir(mode=0o750, exist_ok=True)
            destination = destination_dir / digest.hexdigest()
            try:
                os.link(temporary, destination)
                os.chmod(destination, 0o440)
                _fsync_directory(destination_dir)
            except FileExistsError:
                if (
                    destination.is_symlink()
                    or not destination.is_file()
                    or destination.stat().st_size != total
                ):
                    raise EvalExamplesError("content-addressed eval media was mutated")
                existing_digest = hashlib.sha256()
                with destination.open("rb") as existing:
                    while chunk := existing.read(1024 * 1024):
                        existing_digest.update(chunk)
                if existing_digest.digest() != digest.digest():
                    raise EvalExamplesError("content-addressed eval media was mutated")
            return {
                "kind": media.kind,
                "path": f"objects/{digest.hexdigest()}",
                "media_type": media_type,
                "sha256": actual,
                "size_bytes": total,
            }
        finally:
            temporary.unlink(missing_ok=True)

    def _part(self, part: EvalEvidencePart) -> dict[str, object]:
        if part.kind == "text":
            if (
                part.media is not None
                or part.schema is not None
                or part.value is not None
            ):
                raise EvalExamplesError("eval text part has incompatible fields")
            return {"kind": "text", "text": _text(part.text, "text", 16 * 1024)}
        if part.kind in {"image", "video", "audio"}:
            if (
                part.text is not None
                or part.schema is not None
                or part.value is not None
                or part.media is None
                or part.media.kind != part.kind
            ):
                raise EvalExamplesError("eval media part has incompatible fields")
            return self._freeze_media(part.media)
        if part.kind == "structured":
            if part.text is not None or part.media is not None or part.value is None:
                raise EvalExamplesError("eval structured part has incompatible fields")
            value = json.loads(canonical_dumps(part.value))
            if (
                not isinstance(value, (dict, list))
                or not value
                or len(canonical_dumps(value)) > 16 * 1024
            ):
                raise EvalExamplesError("eval structured value is invalid")
            return {
                "kind": "structured",
                "schema": _text(part.schema, "structured schema", 256),
                "value": value,
            }
        raise EvalExamplesError("eval evidence part kind is unsupported")

    def _parts(
        self, values: tuple[EvalEvidencePart, ...], label: str
    ) -> list[dict[str, object]]:
        if not 0 < len(values) <= MAXIMUM_PARTS:
            raise EvalExamplesError(f"eval example {label} is empty or too large")
        return [self._part(value) for value in values]

    def publish(self, request: EvalExamplesPublicationRequest) -> PublishedEvalExamples:
        if request.output_name != self._output_name:
            raise EvalExamplesError("eval request output does not match publisher")
        if (
            isinstance(request.optimizer_step, bool)
            or not isinstance(request.optimizer_step, int)
            or not 0 <= request.optimizer_step < 1 << 64
        ):
            raise EvalExamplesError("eval optimizer step is invalid")
        if not 0 < len(request.examples) <= MAXIMUM_EXAMPLES:
            raise EvalExamplesError("eval example count is outside its bound")
        metric_names = tuple(
            _text(value, "metric name", 256) for value in request.metric_names
        )
        if not metric_names or len(metric_names) != len(set(metric_names)):
            raise EvalExamplesError("eval metric identities are empty or duplicated")
        parent_artifact_ids = tuple(
            _text(value, "parent artifact id") for value in request.parent_artifact_ids
        )
        if (
            not parent_artifact_ids
            or len(parent_artifact_ids) != len(set(parent_artifact_ids))
            or request.checkpoint_artifact_id not in parent_artifact_ids
        ):
            raise EvalExamplesError(
                "eval checkpoint must be one unique declared artifact parent"
            )
        seen_examples: set[str] = set()
        seen_heldout: set[str] = set()
        examples: list[dict[str, object]] = []
        for value in request.examples:
            example_id = _text(value.example_id, "example id")
            heldout_id = _text(value.heldout_item_id, "heldout item id", 4096)
            if example_id in seen_examples or heldout_id in seen_heldout:
                raise EvalExamplesError("eval example identities must be unique")
            seen_examples.add(example_id)
            seen_heldout.add(heldout_id)
            examples.append(
                {
                    "example_id": example_id,
                    "heldout_item_id": heldout_id,
                    "heldout_item_digest": _digest(
                        value.heldout_item_digest, "heldout item digest"
                    ),
                    "input": self._parts(value.input, "input"),
                    "target": self._parts(value.target, "target"),
                    "prediction": self._parts(value.prediction, "prediction"),
                }
            )
        bootstrap = self._session.bootstrap
        body: dict[str, object] = {
            "api_version": EVAL_EXAMPLES_SCHEMA,
            "run_id": _text(bootstrap.run_id, "run id"),
            "node_id": _text(bootstrap.node_id, "node id"),
            "attempt_id": _text(bootstrap.attempt_id, "attempt id"),
            "optimizer_step": request.optimizer_step,
            "step_domain": "optimizer_step",
            "series_id": _text(request.series_id, "series id"),
            "heldout": {
                "identity_field": _text(request.identity_field, "identity field", 256),
                "identities_digest": _digest(
                    request.identities_digest, "identities digest"
                ),
                "selector_digest": _digest(request.selector_digest, "selector digest"),
            },
            "evaluator": {
                "component_digest": _digest(
                    request.evaluator_component_digest, "evaluator component digest"
                ),
                "metric_names": list(metric_names),
            },
            "checkpoint": {
                "artifact_id": _text(
                    request.checkpoint_artifact_id, "checkpoint artifact id"
                ),
                "manifest_digest": _digest(
                    request.checkpoint_manifest_digest, "checkpoint manifest digest"
                ),
            },
            "policy_digest": _digest(request.policy_digest, "policy digest"),
            "examples": examples,
        }
        manifest = {
            **body,
            "canonical_manifest_digest": sha256_digest(canonical_dumps(body)),
        }
        manifest_bytes = canonical_dumps(manifest)
        if len(manifest_bytes) > MAXIMUM_MANIFEST_BYTES:
            raise EvalExamplesError(
                "eval-examples manifest exceeds the worker message bound"
            )
        manifest_sha256 = sha256_digest(manifest_bytes)
        artifact_id = (
            "eval-examples-"
            + hashlib.sha256(
                canonical_dumps(
                    [
                        bootstrap.run_id,
                        bootstrap.node_id,
                        request.optimizer_step,
                        manifest_sha256,
                    ]
                )
            ).hexdigest()
        )
        revision = self._revisions / artifact_id
        temporary = Path(tempfile.mkdtemp(prefix="revision-", dir=self._revisions))
        try:
            for example in examples:
                for field in ("input", "target", "prediction"):
                    for part in example[field]:
                        relative = part.get("path")
                        digest_value = part.get("sha256")
                        if relative is None:
                            continue
                        assert isinstance(relative, str)
                        assert isinstance(digest_value, str)
                        digest_hex = digest_value.removeprefix("sha256:")
                        source = self._objects / digest_hex[:2] / digest_hex
                        destination = temporary / relative
                        destination.parent.mkdir(
                            mode=0o750, parents=True, exist_ok=True
                        )
                        try:
                            os.link(source, destination)
                        except FileExistsError as error:
                            raise EvalExamplesError(
                                "duplicate eval media payload path"
                            ) from error
                        os.chmod(destination, 0o440)
            objects_directory = temporary / "objects"
            if objects_directory.exists():
                os.chmod(objects_directory, 0o550)
            path = temporary / "manifest.json"
            descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o440)
            with os.fdopen(descriptor, "wb") as output:
                output.write(manifest_bytes)
                output.flush()
                os.fsync(output.fileno())
            os.chmod(temporary, 0o550)
            _fsync_directory(temporary)
            try:
                os.rename(temporary, revision)
                _fsync_directory(self._revisions)
            except OSError:
                existing = revision / "manifest.json"
                if not existing.is_file() or existing.read_bytes() != manifest_bytes:
                    raise EvalExamplesError("eval revision identity collision")
        finally:
            if temporary.exists():
                os.chmod(temporary, 0o750)
                shutil.rmtree(temporary)
        manifest_path = revision / "manifest.json"
        sequence = self._session.artifact(
            artifact_id=artifact_id,
            logical_name=self._logical_name,
            kind=wire.ARTIFACT_KIND_EVAL_EXAMPLES,
            schema=EVAL_EXAMPLES_SCHEMA,
            uri=manifest_path.resolve(strict=True).as_uri(),
            size_bytes=len(manifest_bytes),
            fingerprint_algorithm="manifest_sha256",
            fingerprint=manifest_sha256,
            parent_artifact_ids=parent_artifact_ids,
            optimizer_step=request.optimizer_step,
            canonical_manifest_json=manifest_bytes,
            wait=True,
        )
        return PublishedEvalExamples(
            artifact_id,
            manifest_path,
            manifest_sha256,
            request.optimizer_step,
            sequence,
        )


__all__ = [
    "EVAL_EXAMPLES_SCHEMA",
    "EvalEvidencePart",
    "EvalExample",
    "EvalExamplesError",
    "EvalExamplesPublicationRequest",
    "EvalExamplesPublisher",
    "EvalMedia",
    "PublishedEvalExamples",
]
