from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from urllib.parse import unquote, urlparse

import pytest

from rwkv_lab.trainvm_worker import (
    CHECKPOINT_SNAPSHOT_SCHEMA,
    CheckpointPublicationError,
    CheckpointPublicationRequest,
    CheckpointPublisher,
    PublishedCheckpoint,
    publish_checkpoint_requests,
    resolve_resume_checkpoint,
)
from rwkv_lab.trainvm_worker.checkpoint import _STATE_COMPONENTS


def digest(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def test_dashboard_checkpoint_state_vocabulary_matches_worker_authority() -> None:
    source = (
        Path(__file__).parents[1]
        / "dashboard/internal/server/trainvm_checkpoints.go"
    ).read_text(encoding="utf-8")
    start = source.index("func validCheckpointStateComponent")
    end = source.index("func checkpointCanonicalDigest", start)
    dashboard_components = frozenset(re.findall(r'"([a-z][a-z0-9_]*)"', source[start:end]))
    assert dashboard_components == _STATE_COMPONENTS


class FakeCheckpointSession:
    def __init__(self, root: Path) -> None:
        self.bootstrap = SimpleNamespace(
            run_id="run-1", node_id="train", attempt_id="train@2"
        )
        declaration = MappingProxyType(
            {
                "type": "checkpoint",
                "schema": "rwkv-lab.test-checkpoint.v1",
                "immutability": "immutable",
                "fingerprint": "manifest_sha256",
            }
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
                        {"logical_name": "checkpoint", "declaration": declaration}
                    )
                }
            ),
        )
        self.artifacts: list[dict[str, object]] = []

    def artifact(self, **values: object) -> int:
        self.artifacts.append(values)
        return len(self.artifacts)


def uri_path(value: str) -> Path:
    parsed = urlparse(value)
    assert parsed.scheme == "file" and not parsed.netloc
    return Path(unquote(parsed.path))


def make_checkpoint(root: Path) -> Path:
    checkpoint = root / "checkpoint-00000012"
    (checkpoint / "adapter").mkdir(parents=True)
    (checkpoint / "adapter" / "weights.safetensors").write_bytes(b"weights")
    (checkpoint / "trainer-state.pt").write_bytes(b"optimizer+rng")
    (checkpoint / "state.json").write_text('{"step":12}\n', encoding="utf-8")
    return checkpoint


def test_checkpoint_is_copied_hashed_promoted_and_published(tmp_path: Path) -> None:
    checkpoint = make_checkpoint(tmp_path)
    session = FakeCheckpointSession(tmp_path)
    publisher = CheckpointPublisher(session)
    progress: list[int] = []

    result = publisher.publish(
        checkpoint,
        optimizer_step=12,
        resume_grade="compatible",
        state_components=(
            "component_composition",
            "data_cursor",
            "model",
            "optimizer",
            "optimizer_groups",
            "plateau_state",
            "rng_torch",
        ),
        parent_artifact_ids=("base-model-1",),
        progress=progress.append,
    )

    manifest_bytes = result.manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    assert manifest["schema"] == CHECKPOINT_SNAPSHOT_SCHEMA
    assert manifest["checkpoint_schema"] == "rwkv-lab.test-checkpoint.v1"
    assert manifest["optimizer_step"] == 12
    assert manifest["resume_grade"] == "compatible"
    assert manifest["file_count"] == 3
    assert result.file_count == 3
    assert result.manifest_sha256 == digest(manifest_bytes)
    assert progress and set(progress) == {12}
    assert session.artifacts == [
        {
            "artifact_id": result.artifact_id,
            "logical_name": "checkpoint",
            "kind": 2,
            "schema": "rwkv-lab.test-checkpoint.v1",
            "uri": result.manifest_path.resolve().as_uri(),
            "size_bytes": result.payload_size_bytes + len(manifest_bytes),
            "fingerprint_algorithm": "manifest_sha256",
            "fingerprint": result.manifest_sha256,
            "parent_artifact_ids": ("base-model-1",),
            "optimizer_step": 12,
            "wait": True,
        }
    ]
    payload = result.manifest_path.parent / manifest["payload_directory"]
    assert (payload / "adapter" / "weights.safetensors").read_bytes() == b"weights"
    assert os.stat(payload / "adapter" / "weights.safetensors").st_mode & 0o777 == 0o440
    replay = publisher.publish(
        checkpoint,
        optimizer_step=12,
        resume_grade="compatible",
        state_components=(
            "component_composition",
            "data_cursor",
            "model",
            "optimizer",
            "optimizer_groups",
            "plateau_state",
            "rng_torch",
        ),
        parent_artifact_ids=("base-model-1",),
    )
    assert replay.artifact_id == result.artifact_id
    assert replay.manifest_path == result.manifest_path
    checkpoint.joinpath("adapter", "weights.safetensors").write_bytes(b"mutated")
    assert (payload / "adapter" / "weights.safetensors").read_bytes() == b"weights"


def test_checkpoint_requests_keep_transport_outside_family_handler(
    tmp_path: Path,
) -> None:
    checkpoint = make_checkpoint(tmp_path)
    session = FakeCheckpointSession(tmp_path)
    (result,) = publish_checkpoint_requests(
        session,
        (
            CheckpointPublicationRequest(
                checkpoint,
                optimizer_step=12,
                resume_grade="terminal_checkpoint",
                state_components=("model", "optimizer", "rng_torch"),
            ),
        ),
    )
    assert result.worker_sequence == 1


def test_checkpoint_publisher_rejects_escape_symlink_and_bad_claims(
    tmp_path: Path,
) -> None:
    checkpoint = make_checkpoint(tmp_path)
    session = FakeCheckpointSession(tmp_path)
    publisher = CheckpointPublisher(session)
    outside = tmp_path.parent / f"{tmp_path.name}-outside"
    outside.mkdir()
    try:
        with pytest.raises(CheckpointPublicationError, match="inside the run"):
            publisher.publish(
                outside,
                optimizer_step=1,
                resume_grade="compatible",
                state_components=("model",),
            )
    finally:
        outside.rmdir()

    checkpoint.joinpath("escape").symlink_to(tmp_path / "state-outside")
    with pytest.raises(CheckpointPublicationError, match="symlink"):
        publisher.publish(
            checkpoint,
            optimizer_step=12,
            resume_grade="compatible",
            state_components=("model",),
        )
    checkpoint.joinpath("escape").unlink()

    with pytest.raises(CheckpointPublicationError, match="sorted closed set"):
        publisher.publish(
            checkpoint,
            optimizer_step=12,
            resume_grade="exact",
            state_components=("optimizer", "model"),
        )
    with pytest.raises(CheckpointPublicationError, match="resume grade"):
        publisher.publish(
            checkpoint,
            optimizer_step=12,
            resume_grade="exact_candidate",
            state_components=("model",),
        )


def test_checkpoint_declaration_must_be_immutable_manifest_fingerprinted(
    tmp_path: Path,
) -> None:
    session = FakeCheckpointSession(tmp_path)
    declaration = dict(session.invocation.publishes["checkpoint"]["declaration"])
    declaration["immutability"] = "append_only"
    session.invocation = SimpleNamespace(
        workspace=session.invocation.workspace,
        publishes={
            "checkpoint": {
                "logical_name": "checkpoint",
                "declaration": declaration,
            }
        },
    )
    with pytest.raises(CheckpointPublicationError, match="incompatible"):
        CheckpointPublisher(session)


def resume_invocation(
    session: FakeCheckpointSession,
    result: PublishedCheckpoint,
    *,
    attempt_id: str = "train@3",
) -> SimpleNamespace:
    manifest_path = result.manifest_path
    manifest_bytes = manifest_path.read_bytes()
    checkpoint = {
        "artifact_id": result.artifact_id,
        "logical_name": "checkpoint",
        "kind": "checkpoint",
        "schema": "rwkv-lab.test-checkpoint.v1",
        "uri": manifest_path.resolve().as_uri(),
        "size_bytes": result.payload_size_bytes + len(manifest_bytes),
        "fingerprint_algorithm": "manifest_sha256",
        "fingerprint": result.manifest_sha256,
        "complete": True,
        "producer_node_id": "train",
        "producer_attempt_id": "train@2",
        "parent_artifact_ids": ["base-model-1"],
        "published_at_ns": 1,
    }
    return SimpleNamespace(
        run_id="run-1",
        node_id="train",
        attempt_id=attempt_id,
        workspace=session.invocation.workspace,
        resume={
            "api_version": "trainvm.resume-checkpoint/v1",
            "checkpoint": checkpoint,
            "optimizer_step": 12,
            "pause_command_id": "pause-1",
            "resume_command_id": "resume-1",
        },
    )


def test_controller_selected_resume_checkpoint_is_rehashed_before_use(
    tmp_path: Path,
) -> None:
    checkpoint = make_checkpoint(tmp_path)
    session = FakeCheckpointSession(tmp_path)
    result = CheckpointPublisher(session).publish(
        checkpoint,
        optimizer_step=12,
        resume_grade="exact",
        state_components=("data_cursor", "model", "optimizer", "rng_torch"),
        parent_artifact_ids=("base-model-1",),
    )
    invocation = resume_invocation(session, result)

    resolved = resolve_resume_checkpoint(invocation)

    assert resolved is not None
    assert resolved.artifact_id == result.artifact_id
    assert resolved.manifest_sha256 == result.manifest_sha256
    assert resolved.optimizer_step == 12
    assert resolved.payload_directory.name == "payload"
    assert resolved.state_components == (
        "data_cursor",
        "model",
        "optimizer",
        "rng_torch",
    )

    payload = resolved.payload_directory / "trainer-state.pt"
    payload.chmod(0o640)
    payload.write_bytes(b"tampered")
    with pytest.raises(CheckpointPublicationError, match="mutated"):
        resolve_resume_checkpoint(invocation)


def test_resume_checkpoint_rejects_authority_manifest_lineage_mismatch(
    tmp_path: Path,
) -> None:
    checkpoint = make_checkpoint(tmp_path)
    session = FakeCheckpointSession(tmp_path)
    result = CheckpointPublisher(session).publish(
        checkpoint,
        optimizer_step=12,
        resume_grade="compatible",
        state_components=("model", "optimizer", "rng_torch"),
        parent_artifact_ids=("base-model-1",),
    )
    invocation = resume_invocation(session, result)
    invocation.resume["checkpoint"]["parent_artifact_ids"] = ["forged-parent"]

    with pytest.raises(CheckpointPublicationError, match="semantics"):
        resolve_resume_checkpoint(invocation)
