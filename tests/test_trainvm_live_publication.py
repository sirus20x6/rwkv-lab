from __future__ import annotations

import json
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

def _gallery_item(tmp_path: Path, name: str) -> EvalGalleryItem:
    generated = tmp_path / f"{name}-generated.png"
    target = tmp_path / f"{name}-target.png"
    Image.new("RGB", (8, 8), (20, 40, 60)).save(generated)
    Image.new("RGB", (8, 8), (70, 50, 30)).save(target)
    return EvalGalleryItem(
        item_id=f"photo:{name}",
        heldout_item_id=name,
        heldout_manifest_digest="sha256:" + "c" * 64,
        prompt_or_condition_digest="sha256:" + "d" * 64,
        generated=GalleryImage(generated),
        target=GalleryImage(target),
        seed=7,
        sampling_attributes={"route": "photo"},
    )


def _checkpoint(tmp_path: Path, step: int) -> CheckpointPublicationRequest:
    directory = tmp_path / f"checkpoint-{step}"
    directory.mkdir(exist_ok=True)
    directory.joinpath("state.pt").write_bytes(b"model+optimizer+rng")
    return CheckpointPublicationRequest(
        directory,
        optimizer_step=step,
        resume_grade="compatible",
        state_components=("model", "optimizer", "rng_torch"),
    )


def _gallery(
    tmp_path: Path,
    step: int,
    *,
    items: tuple[EvalGalleryItem, ...] | None = None,
    checkpoint_request_index: int | None = 0,
    checkpoint_manifest_digest: str | None = None,
) -> EvalGalleryPublicationRequest:
    return EvalGalleryPublicationRequest(
        output_name="eval_gallery",
        step=step,
        step_domain="optimizer_step",
        evaluator_profile_digest="sha256:" + "a" * 64,
        use_policy_digest="sha256:" + "b" * 64,
        items=(items if items is not None else (_gallery_item(tmp_path, f"h{step}"),)),
        checkpoint_request_index=checkpoint_request_index,
        checkpoint_manifest_digest=checkpoint_manifest_digest,
    )


def test_live_publication_refuses_a_gallery_that_selects_another_checkpoint(
    tmp_path: Path,
) -> None:
    """A live gallery is bound to the checkpoint frozen beside it, index 0."""

    session = FakeSession(tmp_path)
    runtime = WorkerPublicationRuntime(session)
    with pytest.raises(WorkerPublicationError) as error:
        runtime.publish_eval_revision(
            _checkpoint(tmp_path, 12),
            _gallery(tmp_path, 12, checkpoint_request_index=1),
        )
    assert "must select its paired checkpoint" in str(error.value)
    assert session.artifacts == []


def test_live_publication_refuses_a_gallery_carrying_its_own_checkpoint_digest(
    tmp_path: Path,
) -> None:
    """The digest is the runtime's to write; a supplied one would go unchecked."""

    session = FakeSession(tmp_path)
    runtime = WorkerPublicationRuntime(session)
    with pytest.raises(WorkerPublicationError) as error:
        runtime.publish_eval_revision(
            _checkpoint(tmp_path, 12),
            _gallery(
                tmp_path,
                12,
                checkpoint_manifest_digest="sha256:" + "e" * 64,
            ),
        )
    assert "cannot provide a checkpoint digest" in str(error.value)
    assert session.artifacts == []


def test_live_publication_refuses_a_non_increasing_step(tmp_path: Path) -> None:
    """Revisions are append-only, so a repeated or earlier step is refused."""

    session = FakeSession(tmp_path)
    runtime = WorkerPublicationRuntime(session)
    runtime.publish_eval_revision(_checkpoint(tmp_path, 12), _gallery(tmp_path, 12))
    published = len(session.artifacts)
    with pytest.raises(WorkerPublicationError) as error:
        runtime.publish_eval_revision(_checkpoint(tmp_path, 12), _gallery(tmp_path, 12))
    assert "strictly increasing" in str(error.value)
    assert len(session.artifacts) == published
    assert len(runtime.published_galleries) == 1


def test_live_publication_accepts_a_later_step_after_an_earlier_one(
    tmp_path: Path,
) -> None:
    """The increasing-step guard must not refuse ordinary forward progress."""

    session = FakeSession(tmp_path)
    runtime = WorkerPublicationRuntime(session)
    runtime.publish_eval_revision(_checkpoint(tmp_path, 12), _gallery(tmp_path, 12))
    runtime.publish_eval_revision(_checkpoint(tmp_path, 24), _gallery(tmp_path, 24))
    assert [gallery.artifact_id for gallery in runtime.published_galleries] == [
        session.artifacts[1]["artifact_id"],
        session.artifacts[3]["artifact_id"],
    ]
    assert session.artifacts[3]["parent_artifact_ids"] == (
        runtime.published_checkpoints[1].artifact_id,
    )


def test_live_publication_refuses_untyped_requests(tmp_path: Path) -> None:
    session = FakeSession(tmp_path)
    runtime = WorkerPublicationRuntime(session)
    with pytest.raises(WorkerPublicationError) as checkpoint_error:
        runtime.publish_eval_revision(object(), _gallery(tmp_path, 12))
    assert "typed checkpoint request" in str(checkpoint_error.value)
    with pytest.raises(WorkerPublicationError) as gallery_error:
        runtime.publish_eval_revision(_checkpoint(tmp_path, 12), object())
    assert "typed gallery request" in str(gallery_error.value)
    assert session.artifacts == []


def test_live_publication_reports_progress_for_both_artifacts(
    tmp_path: Path,
) -> None:
    session = FakeSession(tmp_path)
    observed: list[tuple[str, int]] = []
    runtime = WorkerPublicationRuntime(
        session,
        checkpoint_progress=lambda step: observed.append(("checkpoint", step)),
        gallery_progress=lambda step: observed.append(("gallery", step)),
    )
    runtime.publish_eval_revision(_checkpoint(tmp_path, 12), _gallery(tmp_path, 12))
    assert ("checkpoint", 12) in observed
    assert ("gallery", 12) in observed
    assert observed.index(("checkpoint", 12)) < observed.index(("gallery", 12))


class FakeInvocation:
    def __init__(self, *, publishes: tuple[str, ...]) -> None:
        self.publishes = {name: {"logical_name": name} for name in publishes}
        self.inputs: dict[str, object] = {}
        self.invocation_digest = "sha256:" + "f" * 64


def _write_gallery_document(run_directory: Path, step: int) -> None:
    generated = run_directory / f"generated-{step}.png"
    target = run_directory / f"target-{step}.png"
    Image.new("RGB", (8, 8), (10, 20, 30)).save(generated)
    Image.new("RGB", (8, 8), (30, 20, 10)).save(target)
    samples = run_directory / "eval_samples"
    samples.mkdir(exist_ok=True)
    samples.joinpath(f"step_{step:08d}.json").write_text(
        json.dumps(
            {
                "complete": True,
                "step": step,
                "prompt_screening": True,
                "items": [
                    {
                        "image_id": f"heldout-{step}",
                        "route": "photo",
                        "prompt": "a held-out prompt",
                        "image": str(generated),
                        "target_image": str(target),
                        "seed": 7,
                        "sampling_attributes": {"route": "photo"},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def test_mageflow_live_publisher_is_absent_without_its_three_preconditions(
    tmp_path: Path,
) -> None:
    """No runtime, no eval manifest, or no declared gallery output means no callback."""

    from rwkv_lab.trainvm_adapters.handlers import _mageflow_live_eval_publisher

    runtime = WorkerPublicationRuntime(FakeSession(tmp_path))
    manifest = tmp_path / "eval.jsonl"
    manifest.write_text("{}\n", encoding="utf-8")
    declared = FakeInvocation(publishes=("checkpoint", "eval_gallery"))
    assert (
        _mageflow_live_eval_publisher(
            declared, None, tmp_path, manifest, state_components=("model",)
        )
        is None
    )
    assert (
        _mageflow_live_eval_publisher(
            declared, runtime, tmp_path, None, state_components=("model",)
        )
        is None
    )
    assert (
        _mageflow_live_eval_publisher(
            FakeInvocation(publishes=("checkpoint",)),
            runtime,
            tmp_path,
            manifest,
            state_components=("model",),
        )
        is None
    )
    assert (
        _mageflow_live_eval_publisher(
            declared, runtime, tmp_path, manifest, state_components=("model",)
        )
        is not None
    )


def test_mageflow_live_publisher_freezes_a_same_step_revision(tmp_path: Path) -> None:
    """The callback a MageFlow trainer receives publishes checkpoint then gallery."""

    from rwkv_lab.trainvm_adapters.handlers import _mageflow_live_eval_publisher

    session = FakeSession(tmp_path)
    runtime = WorkerPublicationRuntime(session)
    manifest = tmp_path / "eval.jsonl"
    manifest.write_text('{"image": "x"}\n', encoding="utf-8")
    _write_gallery_document(tmp_path, 40)
    checkpoint_directory = tmp_path / "checkpoint-40"
    checkpoint_directory.mkdir()
    checkpoint_directory.joinpath("state.pt").write_bytes(b"model")

    publish = _mageflow_live_eval_publisher(
        FakeInvocation(publishes=("checkpoint", "eval_gallery")),
        runtime,
        tmp_path,
        manifest,
        state_components=("model", "optimizer", "rng_torch"),
    )
    assert publish is not None
    publish(checkpoint_directory, 40)

    assert [artifact["schema"] for artifact in session.artifacts] == [
        "rwkv-lab.live-checkpoint.v1",
        EVAL_GALLERY_SCHEMA,
    ]
    assert len(runtime.published_checkpoints) == 1
    assert session.artifacts[1]["parent_artifact_ids"] == (
        runtime.published_checkpoints[0].artifact_id,
    )
