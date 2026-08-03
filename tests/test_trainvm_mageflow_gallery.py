from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import pytest

from rwkv_lab.trainvm_adapters.mageflow_gallery import (
    MageFlowGalleryResultError,
    completed_mageflow_gallery_request,
)


def _digest(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _write_gallery(run_directory, items) -> None:
    result = run_directory / "eval_samples" / "step_00000019.json"
    result.parent.mkdir(parents=True)
    result.write_text(
        json.dumps({"complete": True, "step": 19, "items": items}),
        encoding="utf-8",
    )


def _item(tmp_path, *, image_id="photo-1"):
    return {
        "image_id": image_id,
        "image": str(tmp_path / "generated.png"),
        "target_image": str(tmp_path / "target.png"),
        "prompt": "a held-out portrait",
        "route": "photo",
        "seed": 123,
        "sampling_attributes": {
            "cfg": "4",
            "route": "photo",
            "sampler": "mage_flow_rectified_flow",
            "steps": "28",
        },
    }


def test_completed_gallery_request_binds_manifest_condition_and_checkpoint(tmp_path):
    manifest = tmp_path / "eval.jsonl"
    manifest_bytes = b'{"image":"target.png","caption":"portrait"}\n'
    manifest.write_bytes(manifest_bytes)
    run_directory = tmp_path / "run"
    _write_gallery(run_directory, [_item(tmp_path)])
    invocation = SimpleNamespace(
        invocation_digest="sha256:" + "a" * 64,
        publishes={"eval_gallery": {}},
    )

    request = completed_mageflow_gallery_request(
        invocation,
        run_directory,
        manifest,
        step=19,
        checkpoint_request_index=0,
    )

    assert request is not None
    assert request.checkpoint_request_index == 0
    assert request.checkpoint_manifest_digest is None
    assert request.evaluator_profile_digest == invocation.invocation_digest
    assert request.items[0].heldout_manifest_digest == _digest(manifest_bytes)
    assert request.items[0].prompt_or_condition_digest == _digest(
        b"a held-out portrait"
    )
    assert request.items[0].target is not None
    assert request.items[0].item_id == "photo:photo-1"


def test_completed_gallery_request_is_optional_but_rejects_duplicate_items(tmp_path):
    manifest = tmp_path / "eval.jsonl"
    manifest.write_text("{}\n", encoding="utf-8")
    run_directory = tmp_path / "run"
    item = _item(tmp_path)
    _write_gallery(run_directory, [item, item])
    invocation = SimpleNamespace(
        invocation_digest="sha256:" + "a" * 64,
        publishes={"eval_gallery": {}},
    )

    with pytest.raises(MageFlowGalleryResultError, match="not unique"):
        completed_mageflow_gallery_request(
            invocation,
            run_directory,
            manifest,
            step=19,
            checkpoint_request_index=0,
        )

    invocation.publishes = {}
    assert (
        completed_mageflow_gallery_request(
            invocation,
            run_directory,
            manifest,
            step=19,
            checkpoint_request_index=0,
        )
        is None
    )
