from __future__ import annotations

import hashlib
import os
import shutil
import stat
import tempfile
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from typing import Protocol

from ._canonical import canonical_dumps, is_bounded_text, is_digest, sha256_digest

try:
    from trainvm.v1 import trainvm_pb2 as wire
except ImportError as error:  # pragma: no cover - installation contract
    raise RuntimeError(
        "TrainVM eval galleries require the 'trainvm-worker' project extra"
    ) from error


EVAL_GALLERY_SCHEMA = "rwkv-lab.eval-gallery.v2"
MAXIMUM_GALLERY_ITEMS = 512
MAXIMUM_IMAGE_BYTES = 128 * 1024 * 1024
MAXIMUM_SAMPLING_ATTRIBUTES = 64
_FORMAT_SUFFIX = {"PNG": ".png", "JPEG": ".jpg", "GIF": ".gif", "WEBP": ".webp"}


class EvalGalleryError(ValueError):
    pass


class _ArtifactSession(Protocol):
    bootstrap: object
    invocation: object

    def artifact(self, **values: object) -> int: ...


@dataclass(frozen=True, slots=True)
class GalleryImage:
    path: str | os.PathLike[str]
    expected_sha256: str | None = None


@dataclass(frozen=True, slots=True)
class EvalGalleryItem:
    item_id: str
    heldout_item_id: str
    heldout_manifest_digest: str
    prompt_or_condition_digest: str
    generated: GalleryImage
    seed: int
    sampling_attributes: Mapping[str, str]
    target: GalleryImage | None = None
    source: GalleryImage | None = None


@dataclass(frozen=True, slots=True)
class PublishedEvalGallery:
    artifact_id: str
    manifest_path: Path
    manifest_sha256: str
    size_bytes: int
    worker_sequence: int


def _within(path: Path, roots: tuple[Path, ...]) -> bool:
    return any(path == root or root in path.parents for root in roots)


def _digest_hex(value: str) -> str:
    if not is_digest(value):
        raise EvalGalleryError("gallery identity must be a canonical sha256: digest")
    return value[7:]


def _text(value: object, label: str, maximum: int = 512) -> str:
    if not is_bounded_text(value, maximum):
        raise EvalGalleryError(f"{label} must be nonempty bounded single-line text")
    assert isinstance(value, str)
    return value


def _resolve_roots(values: object, label: str) -> tuple[Path, ...]:
    if not isinstance(values, (tuple, list)) or not values:
        raise EvalGalleryError(f"workspace {label} must declare at least one root")
    roots: list[Path] = []
    for raw in values:
        if not isinstance(raw, str) or not Path(raw).is_absolute():
            raise EvalGalleryError(f"workspace {label} contains a non-absolute root")
        try:
            resolved = Path(raw).resolve(strict=True)
        except OSError as error:
            raise EvalGalleryError(f"workspace {label} root is unavailable") from error
        if not resolved.is_dir():
            raise EvalGalleryError(f"workspace {label} root is not a directory")
        roots.append(resolved)
    return tuple(roots)


def _open_confined(path: str | os.PathLike[str], roots: tuple[Path, ...]):
    candidate = Path(path)
    if not candidate.is_absolute():
        raise EvalGalleryError("gallery image path must be absolute")
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as error:
        raise EvalGalleryError("gallery image is unavailable") from error
    if not _within(resolved, roots):
        raise EvalGalleryError("gallery image is outside declared read/write roots")
    source = None
    try:
        source = resolved.open("rb")
        descriptor_path = Path(f"/proc/self/fd/{source.fileno()}").resolve(strict=True)
        info = os.fstat(source.fileno())
    except OSError as error:
        if source is not None:
            source.close()
        raise EvalGalleryError("gallery image could not be opened safely") from error
    if not _within(descriptor_path, roots) or not stat.S_ISREG(info.st_mode):
        source.close()
        raise EvalGalleryError(
            "gallery image resolved outside authority or is not regular"
        )
    if info.st_size <= 0 or info.st_size > MAXIMUM_IMAGE_BYTES:
        source.close()
        raise EvalGalleryError("gallery image size is outside the publication bound")
    return source


def _validate_image(path: Path) -> str:
    try:
        from PIL import Image
    except ImportError as error:  # pragma: no cover - dependency contract
        raise EvalGalleryError(
            "Pillow is required to validate eval gallery images"
        ) from error
    try:
        with Image.open(path) as image:
            image_format = image.format
            width, height = image.size
            if image_format not in _FORMAT_SUFFIX or width <= 0 or height <= 0:
                raise EvalGalleryError("gallery object is not a supported raster image")
            if width > 65_536 or height > 65_536 or width * height > 268_435_456:
                raise EvalGalleryError(
                    "gallery raster dimensions exceed the decode bound"
                )
            image.verify()
    except EvalGalleryError:
        raise
    except Exception as error:
        raise EvalGalleryError(
            "gallery object failed raster decode validation"
        ) from error
    return _FORMAT_SUFFIX[image_format]


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


class EvalGalleryPublisher:
    """Validate, freeze, and publish one immutable qualitative eval revision."""

    def __init__(
        self, session: _ArtifactSession, *, output_name: str = "eval_gallery"
    ) -> None:
        self._session = session
        self._output_name = _text(output_name, "gallery output name", 256)
        invocation = session.invocation
        workspace = invocation.workspace
        publishes = invocation.publishes
        if not isinstance(workspace, Mapping) or not isinstance(publishes, Mapping):
            raise EvalGalleryError(
                "worker invocation workspace/publishes contract is invalid"
            )
        publication = publishes.get(self._output_name)
        if not isinstance(publication, Mapping):
            raise EvalGalleryError(
                f"invocation does not declare output {self._output_name!r}"
            )
        declaration = publication.get("declaration")
        logical_name = publication.get("logical_name")
        if (
            not isinstance(declaration, Mapping)
            or declaration.get("type") != "image_gallery"
            or declaration.get("schema") != EVAL_GALLERY_SCHEMA
            or declaration.get("immutability") != "append_only"
            or declaration.get("fingerprint") != "manifest_sha256"
        ):
            raise EvalGalleryError(
                "gallery output declaration is incompatible with eval-gallery v2"
            )
        self._logical_name = _text(logical_name, "gallery logical name", 256)
        self._read_roots = _resolve_roots(
            workspace.get("allowed_read_roots"), "allowed_read_roots"
        )
        self._write_roots = _resolve_roots(
            workspace.get("allowed_write_roots"), "allowed_write_roots"
        )
        run_directory = workspace.get("run_directory")
        if not isinstance(run_directory, str) or not Path(run_directory).is_absolute():
            raise EvalGalleryError("workspace run_directory must be absolute")
        run_root = Path(run_directory).resolve(strict=True)
        if not _within(run_root, self._write_roots):
            raise EvalGalleryError(
                "workspace run_directory is outside declared write roots"
            )
        self._publication_root = run_root / "trainvm_artifacts" / "eval_gallery"
        self._publication_root.mkdir(mode=0o750, parents=True, exist_ok=True)
        resolved_publication_root = self._publication_root.resolve(strict=True)
        if not _within(resolved_publication_root, self._write_roots):
            raise EvalGalleryError(
                "gallery publication root escaped declared write roots"
            )
        self._publication_root = resolved_publication_root
        self._object_root = self._publication_root / "objects" / "sha256"
        self._revision_root = self._publication_root / "revisions"
        self._object_root.mkdir(mode=0o750, parents=True, exist_ok=True)
        self._revision_root.mkdir(mode=0o750, parents=True, exist_ok=True)

    def _freeze_image(self, value: GalleryImage) -> tuple[str, str]:
        roots = self._read_roots + self._write_roots
        staging = self._object_root / ".staging"
        staging.mkdir(mode=0o750, exist_ok=True)
        temporary_fd, temporary_name = tempfile.mkstemp(prefix="object-", dir=staging)
        temporary = Path(temporary_name)
        digest = hashlib.sha256()
        try:
            with (
                os.fdopen(temporary_fd, "wb") as destination,
                _open_confined(value.path, roots) as source,
            ):
                total = 0
                while chunk := source.read(1024 * 1024):
                    total += len(chunk)
                    if total > MAXIMUM_IMAGE_BYTES:
                        raise EvalGalleryError(
                            "gallery image exceeds publication size bound"
                        )
                    digest.update(chunk)
                    destination.write(chunk)
                destination.flush()
                os.fsync(destination.fileno())
            actual_digest = "sha256:" + digest.hexdigest()
            if (
                value.expected_sha256 is not None
                and value.expected_sha256 != actual_digest
            ):
                raise EvalGalleryError(
                    "gallery source image disagrees with its expected digest"
                )
            suffix = _validate_image(temporary)
            destination_dir = self._object_root / digest.hexdigest()[:2]
            destination_dir.mkdir(mode=0o750, exist_ok=True)
            destination = destination_dir / (digest.hexdigest() + suffix)
            try:
                os.link(temporary, destination)
                os.chmod(destination, 0o440)
                _fsync_directory(destination_dir)
            except FileExistsError:
                if destination.is_symlink() or not destination.is_file():
                    raise EvalGalleryError(
                        "content-addressed gallery object is not regular"
                    )
                info = destination.stat()
                if info.st_size <= 0 or info.st_size > MAXIMUM_IMAGE_BYTES:
                    raise EvalGalleryError(
                        "content-addressed gallery object size is invalid"
                    )
                existing_hash = hashlib.sha256()
                with destination.open("rb") as existing_file:
                    while chunk := existing_file.read(1024 * 1024):
                        existing_hash.update(chunk)
                existing = existing_hash.hexdigest()
                if existing != digest.hexdigest():
                    raise EvalGalleryError(
                        "content-addressed gallery object was mutated"
                    )
            return destination.resolve(strict=True).as_uri(), actual_digest
        finally:
            temporary.unlink(missing_ok=True)

    def _item_document(self, item: EvalGalleryItem) -> dict[str, object]:
        item_id = _text(item.item_id, "gallery item ID")
        heldout_item_id = _text(item.heldout_item_id, "held-out item ID")
        _digest_hex(item.heldout_manifest_digest)
        _digest_hex(item.prompt_or_condition_digest)
        if (
            not isinstance(item.seed, int)
            or isinstance(item.seed, bool)
            or not 0 <= item.seed < 1 << 64
        ):
            raise EvalGalleryError("gallery seed must be uint64")
        if (
            not isinstance(item.sampling_attributes, Mapping)
            or len(item.sampling_attributes) > MAXIMUM_SAMPLING_ATTRIBUTES
        ):
            raise EvalGalleryError("gallery sampling attributes are invalid")
        attributes: dict[str, str] = {}
        for key, value in item.sampling_attributes.items():
            attributes[_text(key, "sampling attribute key", 128)] = _text(
                value, "sampling attribute value", 1024
            )
        generated_uri, generated_digest = self._freeze_image(item.generated)
        document: dict[str, object] = {
            "generated_object_sha256": generated_digest,
            "generated_object_uri": generated_uri,
            "heldout_item_id": heldout_item_id,
            "heldout_manifest_digest": item.heldout_manifest_digest,
            "item_id": item_id,
            "prompt_or_condition_digest": item.prompt_or_condition_digest,
            "sampling_attributes": attributes,
            "seed": item.seed,
        }
        if item.target is not None:
            target_uri, target_digest = self._freeze_image(item.target)
            document["target_object_sha256"] = target_digest
            document["target_object_uri"] = target_uri
        if item.source is not None:
            source_uri, source_digest = self._freeze_image(item.source)
            document["source_object_sha256"] = source_digest
            document["source_object_uri"] = source_uri
        return document

    def publish(
        self,
        *,
        step: int,
        step_domain: str,
        checkpoint_manifest_digest: str,
        evaluator_profile_digest: str,
        use_policy_digest: str,
        items: Iterable[EvalGalleryItem],
        parent_artifact_ids: Iterable[str],
    ) -> PublishedEvalGallery:
        if (
            not isinstance(step, int)
            or isinstance(step, bool)
            or not 0 <= step < 1 << 64
        ):
            raise EvalGalleryError("gallery step must be uint64")
        step_domain = _text(step_domain, "gallery step domain", 128)
        for value in (
            checkpoint_manifest_digest,
            evaluator_profile_digest,
            use_policy_digest,
        ):
            _digest_hex(value)
        parents = tuple(islice(parent_artifact_ids, 1025))
        if (
            not parents
            or len(parents) > 1024
            or any(not is_bounded_text(parent, 1024) for parent in parents)
            or len(set(parents)) != len(parents)
        ):
            raise EvalGalleryError(
                "gallery parent artifact identities must be unique and nonempty"
            )
        item_values = tuple(islice(items, MAXIMUM_GALLERY_ITEMS + 1))
        if not 0 < len(item_values) <= MAXIMUM_GALLERY_ITEMS:
            raise EvalGalleryError("gallery item count is outside publication bound")
        seen: set[str] = set()
        item_documents: list[dict[str, object]] = []
        for item in item_values:
            document = self._item_document(item)
            item_id = str(document["item_id"])
            if item_id in seen:
                raise EvalGalleryError(f"duplicate gallery item ID: {item_id}")
            seen.add(item_id)
            item_documents.append(document)
        bootstrap = self._session.bootstrap
        body: dict[str, object] = {
            "api_version": EVAL_GALLERY_SCHEMA,
            "attempt_id": _text(bootstrap.attempt_id, "producer attempt", 1024),
            "checkpoint_manifest_digest": checkpoint_manifest_digest,
            "evaluator_profile_digest": evaluator_profile_digest,
            "items": item_documents,
            "node_id": _text(bootstrap.node_id, "producer node", 1024),
            "run_id": _text(bootstrap.run_id, "producer run", 1024),
            "step": step,
            "step_domain": step_domain,
            "use_policy_digest": use_policy_digest,
        }
        canonical_manifest_digest = sha256_digest(canonical_dumps(body))
        manifest = {**body, "canonical_manifest_digest": canonical_manifest_digest}
        manifest_bytes = canonical_dumps(manifest)
        manifest_sha256 = sha256_digest(manifest_bytes)
        artifact_seed = canonical_dumps(
            [
                bootstrap.run_id,
                bootstrap.node_id,
                bootstrap.attempt_id,
                step,
                canonical_manifest_digest,
            ]
        )
        artifact_id = "eval-gallery-" + hashlib.sha256(artifact_seed).hexdigest()
        revision = self._revision_root / artifact_id
        temporary = Path(tempfile.mkdtemp(prefix="revision-", dir=self._revision_root))
        try:
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
            os.chmod(temporary, 0o550)
            _fsync_directory(temporary)
            try:
                os.rename(temporary, revision)
                _fsync_directory(self._revision_root)
            except OSError:
                existing = revision / "manifest.json"
                if not existing.is_file() or existing.read_bytes() != manifest_bytes:
                    raise EvalGalleryError(
                        "gallery revision identity already exists with different bytes"
                    )
            manifest_path = revision / "manifest.json"
        finally:
            if temporary.exists():
                os.chmod(temporary, 0o750)
                shutil.rmtree(temporary)
        sequence = self._session.artifact(
            artifact_id=artifact_id,
            logical_name=self._logical_name,
            kind=wire.ARTIFACT_KIND_IMAGE_GALLERY,
            schema=EVAL_GALLERY_SCHEMA,
            uri=manifest_path.resolve(strict=True).as_uri(),
            size_bytes=len(manifest_bytes),
            fingerprint_algorithm="manifest_sha256",
            fingerprint=manifest_sha256,
            parent_artifact_ids=parents,
            wait=True,
        )
        return PublishedEvalGallery(
            artifact_id=artifact_id,
            manifest_path=manifest_path,
            manifest_sha256=manifest_sha256,
            size_bytes=len(manifest_bytes),
            worker_sequence=sequence,
        )
