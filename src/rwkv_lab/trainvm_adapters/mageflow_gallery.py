"""Lower completed MageFlow qualitative evals into TrainVM publications."""

from __future__ import annotations

import hashlib
import json
import os
import stat
from collections.abc import Mapping
from pathlib import Path

from rwkv_lab.trainvm_worker import (
    EvalGalleryItem,
    EvalGalleryPublicationRequest,
    GalleryImage,
    WorkerInvocation,
)


class MageFlowGalleryResultError(ValueError):
    pass


_MAXIMUM_GALLERY_RESULT_BYTES = 4 * 1024 * 1024
_MAXIMUM_GALLERY_ITEMS = 512
_MAXIMUM_HELDOUT_MANIFEST_BYTES = 1024 * 1024 * 1024
_USE_POLICY = {
    "api_version": "rwkv-lab.mageflow-private-heldout-use-policy/v1",
    "heldout_membership": "manifest_bound",
    "purpose": "qualitative_training_evaluation",
    "target_pairing": "required",
}


def _sha256_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _canonical_digest(value: object) -> str:
    return _sha256_bytes(
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    )


def _read_regular_file(path: Path, *, maximum_bytes: int) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise MageFlowGalleryResultError("gallery result file is unavailable") from error
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_size <= 0
            or before.st_size > maximum_bytes
        ):
            raise MageFlowGalleryResultError("gallery result file is invalid")
        chunks: list[bytes] = []
        total = 0
        while chunk := os.read(descriptor, 64 * 1024):
            total += len(chunk)
            if total > maximum_bytes:
                raise MageFlowGalleryResultError("gallery result file is oversized")
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
            raise MageFlowGalleryResultError("gallery result changed while read")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _regular_file_digest(path: Path, *, maximum_bytes: int) -> str:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise MageFlowGalleryResultError("held-out manifest is unavailable") from error
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_size <= 0
            or before.st_size > maximum_bytes
        ):
            raise MageFlowGalleryResultError("held-out manifest is invalid")
        digest = hashlib.sha256()
        total = 0
        while chunk := os.read(descriptor, 1024 * 1024):
            total += len(chunk)
            if total > maximum_bytes:
                raise MageFlowGalleryResultError("held-out manifest is oversized")
            digest.update(chunk)
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
            raise MageFlowGalleryResultError("held-out manifest changed while read")
        return "sha256:" + digest.hexdigest()
    finally:
        os.close(descriptor)


def _gallery_document(path: Path, *, step: int) -> Mapping[str, object]:
    try:
        value = json.loads(
            _read_regular_file(
                path, maximum_bytes=_MAXIMUM_GALLERY_RESULT_BYTES
            )
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise MageFlowGalleryResultError("gallery result is malformed") from error
    if not isinstance(value, Mapping):
        raise MageFlowGalleryResultError("gallery result is not an object")
    items = value.get("items")
    if (
        value.get("complete") is not True
        or value.get("step") != step
        or not isinstance(items, list)
        or not 0 < len(items) <= _MAXIMUM_GALLERY_ITEMS
    ):
        raise MageFlowGalleryResultError("gallery result is incomplete")
    return value


def _item(value: object, *, heldout_manifest_digest: str) -> EvalGalleryItem:
    if not isinstance(value, Mapping):
        raise MageFlowGalleryResultError("gallery item is not an object")
    image_id = value.get("image_id")
    route = value.get("route")
    prompt = value.get("prompt")
    generated = value.get("image")
    target = value.get("target_image")
    seed = value.get("seed")
    attributes = value.get("sampling_attributes")
    if (
        not isinstance(image_id, str)
        or not image_id
        or not isinstance(route, str)
        or not route
        or not isinstance(prompt, str)
        or not prompt
        or not isinstance(generated, str)
        or not Path(generated).is_absolute()
        or not isinstance(target, str)
        or not Path(target).is_absolute()
        or not isinstance(seed, int)
        or isinstance(seed, bool)
        or not 0 <= seed < 1 << 64
        or not isinstance(attributes, Mapping)
        or not attributes
        or any(
            not isinstance(key, str)
            or not key
            or not isinstance(item, str)
            or not item
            for key, item in attributes.items()
        )
    ):
        raise MageFlowGalleryResultError("gallery item semantics are invalid")
    return EvalGalleryItem(
        item_id=f"{route}:{image_id}",
        heldout_item_id=image_id,
        heldout_manifest_digest=heldout_manifest_digest,
        prompt_or_condition_digest=_sha256_bytes(prompt.encode("utf-8")),
        generated=GalleryImage(generated),
        target=GalleryImage(target),
        seed=seed,
        sampling_attributes=dict(attributes),
    )


def completed_mageflow_gallery_request(
    invocation: WorkerInvocation,
    run_directory: Path,
    eval_manifest: Path,
    *,
    step: int,
    checkpoint_request_index: int | None = None,
    checkpoint_manifest_digest: str | None = None,
    parent_artifact_ids: tuple[str, ...] = (),
) -> EvalGalleryPublicationRequest | None:
    """Return the exact terminal gallery selected by the invocation, if declared."""

    publishes = getattr(invocation, "publishes", None)
    if not isinstance(publishes, Mapping) or "eval_gallery" not in publishes:
        return None
    invocation_digest = getattr(invocation, "invocation_digest", None)
    if not isinstance(invocation_digest, str):
        raise MageFlowGalleryResultError("worker invocation digest is unavailable")
    if (checkpoint_request_index is None) == (checkpoint_manifest_digest is None):
        raise MageFlowGalleryResultError(
            "gallery must select exactly one checkpoint binding"
        )
    if checkpoint_manifest_digest is not None and not parent_artifact_ids:
        raise MageFlowGalleryResultError(
            "external checkpoint gallery requires parent artifact lineage"
        )
    heldout_manifest_digest = _regular_file_digest(
        eval_manifest, maximum_bytes=_MAXIMUM_HELDOUT_MANIFEST_BYTES
    )
    document = _gallery_document(
        run_directory / "eval_samples" / f"step_{step:08d}.json",
        step=step,
    )
    items = tuple(
        _item(value, heldout_manifest_digest=heldout_manifest_digest)
        for value in document["items"]
    )
    item_ids = tuple(item.item_id for item in items)
    if len(item_ids) != len(set(item_ids)):
        raise MageFlowGalleryResultError("gallery item identities are not unique")
    return EvalGalleryPublicationRequest(
        output_name="eval_gallery",
        step=step,
        step_domain="optimizer_step",
        evaluator_profile_digest=invocation_digest,
        use_policy_digest=_canonical_digest(
            {
                **_USE_POLICY,
                "prompt_screening": (
                    "pinned_model_content_gate"
                    if document.get("prompt_screening") is True
                    else "private_training_eval_bypass"
                ),
            }
        ),
        items=items,
        parent_artifact_ids=parent_artifact_ids,
        checkpoint_manifest_digest=checkpoint_manifest_digest,
        checkpoint_request_index=checkpoint_request_index,
    )


__all__ = [
    "MageFlowGalleryResultError",
    "completed_mageflow_gallery_request",
]
