from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from urllib.parse import unquote, urlparse

import pytest
from PIL import Image

from rwkv_lab.trainvm_worker import (
    EVAL_GALLERY_SCHEMA,
    EvalGalleryError,
    EvalGalleryItem,
    EvalGalleryPublisher,
    GalleryImage,
)


def digest(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


class FakeGallerySession:
    def __init__(self, root) -> None:
        self.bootstrap = SimpleNamespace(
            run_id="run-1", node_id="eval", attempt_id="eval@3"
        )
        self.invocation = SimpleNamespace(
            workspace=MappingProxyType(
                {
                    "run_directory": str(root),
                    "allowed_read_roots": (str(root),),
                    "allowed_write_roots": (str(root),),
                }
            ),
            publishes=MappingProxyType(
                {
                    "eval_gallery": MappingProxyType(
                        {
                            "logical_name": "eval_gallery",
                            "declaration": MappingProxyType(
                                {
                                    "type": "image_gallery",
                                    "schema": EVAL_GALLERY_SCHEMA,
                                    "immutability": "append_only",
                                    "fingerprint": "manifest_sha256",
                                }
                            ),
                        }
                    )
                }
            ),
        )
        self.artifacts: list[dict[str, object]] = []

    def artifact(self, **values: object) -> int:
        self.artifacts.append(values)
        return 17


def write_image(path, color: tuple[int, int, int]) -> bytes:
    Image.new("RGB", (8, 6), color).save(path, format="PNG")
    return path.read_bytes()


def gallery_item(generated, target) -> EvalGalleryItem:
    return EvalGalleryItem(
        item_id="item-1",
        heldout_item_id="photo-1",
        heldout_manifest_digest="sha256:" + "a" * 64,
        prompt_or_condition_digest="sha256:" + "b" * 64,
        generated=GalleryImage(generated),
        target=GalleryImage(target),
        seed=42,
        sampling_attributes={"route": "photo", "sampler": "euler"},
    )


def publish(publisher: EvalGalleryPublisher, item: EvalGalleryItem):
    return publisher.publish(
        step=125,
        step_domain="optimizer_step",
        checkpoint_manifest_digest="sha256:" + "c" * 64,
        evaluator_profile_digest="sha256:" + "d" * 64,
        use_policy_digest="sha256:" + "e" * 64,
        items=(item,),
        parent_artifact_ids=("checkpoint-125",),
    )


def test_eval_gallery_publisher_freezes_and_atomically_publishes_side_by_side(
    tmp_path,
) -> None:
    generated_path = tmp_path / "generated.png"
    target_path = tmp_path / "target.png"
    generated = write_image(generated_path, (250, 20, 30))
    target = write_image(target_path, (10, 220, 40))
    session = FakeGallerySession(tmp_path)
    publisher = EvalGalleryPublisher(session)

    result = publish(publisher, gallery_item(generated_path, target_path))
    assert result.worker_sequence == 17
    assert result.manifest_path.is_file()
    assert stat_mode(result.manifest_path) == 0o440
    assert session.artifacts == [
        {
            "artifact_id": result.artifact_id,
            "logical_name": "eval_gallery",
            "kind": 4,
            "schema": EVAL_GALLERY_SCHEMA,
            "uri": result.manifest_path.resolve().as_uri(),
            "size_bytes": result.size_bytes,
            "fingerprint_algorithm": "sha256",
            "fingerprint": result.manifest_sha256,
            "parent_artifact_ids": ("checkpoint-125",),
            "wait": True,
        }
    ]
    manifest_bytes = result.manifest_path.read_bytes()
    assert result.manifest_sha256 == digest(manifest_bytes)
    manifest = json.loads(manifest_bytes)
    assert manifest["step"] == 125
    assert manifest["canonical_manifest_digest"].startswith("sha256:")
    (item,) = manifest["items"]
    assert item["generated_object_sha256"] == digest(generated)
    assert item["target_object_sha256"] == digest(target)
    frozen_generated = uri_path(item["generated_object_uri"])
    frozen_target = uri_path(item["target_object_uri"])
    assert frozen_generated.read_bytes() == generated
    assert frozen_target.read_bytes() == target
    assert stat_mode(frozen_generated) == 0o440

    generated_path.write_bytes(b"changed after publication")
    assert frozen_generated.read_bytes() == generated

    replay = publish(publisher, gallery_item(frozen_generated, frozen_target))
    assert replay.artifact_id == result.artifact_id
    assert replay.manifest_path == result.manifest_path


def stat_mode(path) -> int:
    return os.stat(path).st_mode & 0o777


def uri_path(value: str):
    parsed = urlparse(value)
    assert parsed.scheme == "file" and not parsed.netloc
    return Path(unquote(parsed.path))


def test_eval_gallery_rejects_outside_paths_and_expected_digest_mismatch(
    tmp_path,
) -> None:
    generated_path = tmp_path / "generated.png"
    target_path = tmp_path / "target.png"
    write_image(generated_path, (1, 2, 3))
    write_image(target_path, (4, 5, 6))
    session = FakeGallerySession(tmp_path)
    publisher = EvalGalleryPublisher(session)
    outside = tmp_path.parent / f"{tmp_path.name}-outside.png"
    write_image(outside, (7, 8, 9))
    try:
        with pytest.raises(EvalGalleryError, match="outside declared"):
            publish(publisher, gallery_item(outside, target_path))
    finally:
        outside.unlink(missing_ok=True)

    item = gallery_item(generated_path, target_path)
    item = EvalGalleryItem(
        item_id=item.item_id,
        heldout_item_id=item.heldout_item_id,
        heldout_manifest_digest=item.heldout_manifest_digest,
        prompt_or_condition_digest=item.prompt_or_condition_digest,
        generated=GalleryImage(generated_path, "sha256:" + "0" * 64),
        target=item.target,
        seed=item.seed,
        sampling_attributes=item.sampling_attributes,
    )
    with pytest.raises(EvalGalleryError, match="expected digest"):
        publish(publisher, item)


def test_eval_gallery_requires_exact_declared_output_contract(tmp_path) -> None:
    session = FakeGallerySession(tmp_path)
    session.invocation = SimpleNamespace(
        workspace=session.invocation.workspace,
        publishes={
            "eval_gallery": {
                "logical_name": "eval_gallery",
                "declaration": {
                    "type": "image_gallery",
                    "schema": "wrong.v1",
                    "immutability": "append_only",
                    "fingerprint": "manifest_sha256",
                },
            }
        },
    )
    with pytest.raises(EvalGalleryError, match="incompatible"):
        EvalGalleryPublisher(session)
