from __future__ import annotations

from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import pytest
from PIL import Image

from rwkv_lab.trainvm_worker import (
    CHECKPOINT_SNAPSHOT_SCHEMA,
    EVAL_GALLERY_SCHEMA,
    CheckpointPublicationRequest,
    EvalGalleryItem,
    EvalGalleryPublicationRequest,
    GalleryImage,
    WorkerPublicationError,
    WorkerPublicationRuntime,
)


class FakeSession:
    def __init__(self, root: Path) -> None:
        self.bootstrap = SimpleNamespace(
            run_id="run-live", node_id="train", attempt_id="train@1"
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
                    "checkpoint": MappingProxyType(
                        {
                            "logical_name": "checkpoint",
                            "declaration": MappingProxyType(
                                {
                                    "type": "checkpoint",
                                    "schema": "rwkv-lab.live-checkpoint.v1",
                                    "immutability": "immutable",
                                    "fingerprint": "manifest_sha256",
                                }
                            ),
                        }
                    ),
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
                    ),
                }
            ),
        )
        self.artifacts: list[dict[str, object]] = []

    def artifact(self, **values: object) -> int:
        self.artifacts.append(values)
        return len(self.artifacts)


def test_live_publication_freezes_checkpoint_before_bound_gallery(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "checkpoint-12"
    checkpoint.mkdir()
    checkpoint.joinpath("state.pt").write_bytes(b"model+optimizer+rng")
    generated = tmp_path / "generated.png"
    target = tmp_path / "target.png"
    Image.new("RGB", (8, 8), (20, 40, 60)).save(generated)
    Image.new("RGB", (8, 8), (70, 50, 30)).save(target)
    session = FakeSession(tmp_path)
    runtime = WorkerPublicationRuntime(session)

    published_checkpoint, published_gallery = runtime.publish_eval_revision(
        CheckpointPublicationRequest(
            checkpoint,
            optimizer_step=12,
            resume_grade="compatible",
            state_components=("model", "optimizer", "rng_torch"),
        ),
        EvalGalleryPublicationRequest(
            output_name="eval_gallery",
            step=12,
            step_domain="optimizer_step",
            evaluator_profile_digest="sha256:" + "a" * 64,
            use_policy_digest="sha256:" + "b" * 64,
            items=(
                EvalGalleryItem(
                    item_id="photo:heldout-1",
                    heldout_item_id="heldout-1",
                    heldout_manifest_digest="sha256:" + "c" * 64,
                    prompt_or_condition_digest="sha256:" + "d" * 64,
                    generated=GalleryImage(generated),
                    target=GalleryImage(target),
                    seed=7,
                    sampling_attributes={"route": "photo"},
                ),
            ),
            checkpoint_request_index=0,
        ),
    )

    assert [artifact["schema"] for artifact in session.artifacts] == [
        "rwkv-lab.live-checkpoint.v1",
        EVAL_GALLERY_SCHEMA,
    ]
    assert session.artifacts[1]["parent_artifact_ids"] == (
        published_checkpoint.artifact_id,
    )
    assert runtime.published_checkpoints == (published_checkpoint,)
    assert runtime.published_galleries == (published_gallery,)
    assert published_checkpoint.manifest_path.name == "manifest.json"
    assert CHECKPOINT_SNAPSHOT_SCHEMA in published_checkpoint.manifest_path.read_text()


def test_live_publication_rejects_step_mismatch_before_filesystem_work(
    tmp_path: Path,
) -> None:
    session = FakeSession(tmp_path)
    runtime = WorkerPublicationRuntime(session)
    with pytest.raises(WorkerPublicationError, match="steps disagree"):
        runtime.publish_eval_revision(
            CheckpointPublicationRequest(
                tmp_path,
                optimizer_step=8,
                resume_grade="compatible",
                state_components=("model",),
            ),
            EvalGalleryPublicationRequest(
                output_name="eval_gallery",
                step=9,
                step_domain="optimizer_step",
                evaluator_profile_digest="sha256:" + "a" * 64,
                use_policy_digest="sha256:" + "b" * 64,
                items=(),
                checkpoint_request_index=0,
            ),
        )
    assert session.artifacts == []
