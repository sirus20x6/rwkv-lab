from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from urllib.parse import unquote, urlparse

import pytest

from rwkv_lab.trainvm_worker import (
    CHECKPOINT_SNAPSHOT_SCHEMA,
    CheckpointPublicationError,
    CheckpointPublicationRequest,
    CheckpointPublisher,
    publish_checkpoint_requests,
)


def digest(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


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
