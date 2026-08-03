from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import pytest

from rwkv_lab.trainvm_worker import (
    IMMUTABLE_TREE_SCHEMA,
    ArtifactPublicationError,
    ArtifactPublicationRequest,
    ImmutableArtifactPublisher,
    publish_artifact_requests,
    read_input_artifact_file,
    resolve_input_artifact,
)


class FakeArtifactSession:
    def __init__(self, root: Path) -> None:
        self.bootstrap = SimpleNamespace(
            run_id="run-1", node_id="posttrain", attempt_id="posttrain@1"
        )
        self.invocation = SimpleNamespace(
            invocation_digest="sha256:" + "a" * 64,
            workspace=MappingProxyType(
                {
                    "run_directory": str(root),
                    "allowed_write_roots": (str(root),),
                }
            ),
            publishes=MappingProxyType(
                {
                    "adapter": MappingProxyType(
                        {
                            "logical_name": "posttrained_adapter",
                            "declaration": MappingProxyType(
                                {
                                    "type": "opaque",
                                    "schema": "rwkv-lab.posttraining-output.v1",
                                    "immutability": "immutable",
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
        return len(self.artifacts)


def test_directory_artifact_is_frozen_and_protocol_published(tmp_path: Path) -> None:
    source = tmp_path / "posttraining-output"
    (source / "adapter").mkdir(parents=True)
    (source / "adapter" / "weights.safetensors").write_bytes(b"weights")
    (source / "posttrain-result.json").write_text("{}\n", encoding="utf-8")
    session = FakeArtifactSession(tmp_path)

    (published,) = publish_artifact_requests(
        session,
        (ArtifactPublicationRequest(source, "adapter", ("base-checkpoint",)),),
    )

    manifest_bytes = published.manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    assert manifest["schema"] == IMMUTABLE_TREE_SCHEMA
    assert manifest["artifact_schema"] == "rwkv-lab.posttraining-output.v1"
    assert manifest["artifact_kind"] == "opaque"
    assert manifest["file_count"] == 2
    assert published.manifest_sha256 == (
        "sha256:" + hashlib.sha256(manifest_bytes).hexdigest()
    )
    assert session.artifacts[0]["kind"] == 7
    assert session.artifacts[0]["schema"] == "rwkv-lab.posttraining-output.v1"
    assert session.artifacts[0]["parent_artifact_ids"] == ("base-checkpoint",)
    payload = published.manifest_path.parent / "payload"
    assert (payload / "adapter" / "weights.safetensors").read_bytes() == b"weights"
    (source / "adapter" / "weights.safetensors").write_bytes(b"changed")
    assert (payload / "adapter" / "weights.safetensors").read_bytes() == b"weights"


def test_directory_artifact_rejects_symlinks_and_wrong_declaration(
    tmp_path: Path,
) -> None:
    source = tmp_path / "output"
    source.mkdir()
    (source / "result.json").write_text("{}", encoding="utf-8")
    (source / "escape").symlink_to(tmp_path / "outside")
    session = FakeArtifactSession(tmp_path)
    publisher = ImmutableArtifactPublisher(session, output_name="adapter")
    with pytest.raises(ArtifactPublicationError, match="publication namespace"):
        publisher.publish(tmp_path)
    with pytest.raises(ArtifactPublicationError, match="symlink"):
        publisher.publish(source)

    declaration = dict(session.invocation.publishes["adapter"]["declaration"])
    declaration["type"] = "checkpoint"
    session.invocation = SimpleNamespace(
        workspace=session.invocation.workspace,
        publishes={
            "adapter": {
                "logical_name": "posttrained_adapter",
                "declaration": declaration,
            }
        },
    )
    with pytest.raises(ArtifactPublicationError, match="incompatible"):
        ImmutableArtifactPublisher(session, output_name="adapter")


def _input_descriptor(session, published) -> dict[str, object]:
    announced = session.artifacts[0]
    return {
        "artifact_id": published.artifact_id,
        "logical_name": announced["logical_name"],
        "kind": "opaque",
        "schema": announced["schema"],
        "uri": announced["uri"],
        "size_bytes": announced["size_bytes"],
        "fingerprint_algorithm": announced["fingerprint_algorithm"],
        "fingerprint": announced["fingerprint"],
        "complete": True,
        "producer_node_id": session.bootstrap.node_id,
        "producer_attempt_id": session.bootstrap.attempt_id,
        "parent_artifact_ids": list(announced["parent_artifact_ids"]),
        "published_at_ns": 1,
    }


def test_input_artifact_resolution_reverifies_descriptor_manifest_and_payload(
    tmp_path: Path,
) -> None:
    source = tmp_path / "output"
    source.mkdir()
    (source / "result.json").write_text('{"score":1}\n', encoding="utf-8")
    session = FakeArtifactSession(tmp_path)
    published = ImmutableArtifactPublisher(session, output_name="adapter").publish(
        source, parent_artifact_ids=("parent-1",)
    )
    descriptor = _input_descriptor(session, published)
    invocation = SimpleNamespace(
        inputs=MappingProxyType({"candidate": MappingProxyType(descriptor)}),
        workspace=MappingProxyType(
            {
                "allowed_read_roots": (str(tmp_path),),
                "allowed_write_roots": (str(tmp_path),),
            }
        ),
    )

    resolved = resolve_input_artifact(
        invocation,
        "candidate",
        required_kind="opaque",
        required_schema="rwkv-lab.posttraining-output.v1",
    )

    assert resolved.artifact_id == published.artifact_id
    assert resolved.parent_artifact_ids == ("parent-1",)
    assert (resolved.payload_directory / "result.json").read_text() == (
        '{"score":1}\n'
    )
    assert read_input_artifact_file(
        resolved, "result.json", maximum_bytes=1024
    ) == b'{"score":1}\n'

    payload = resolved.payload_directory / "result.json"
    payload.chmod(0o640)
    payload.write_text('{"score":2}\n', encoding="utf-8")
    with pytest.raises(ArtifactPublicationError, match="mutated"):
        resolve_input_artifact(
            invocation,
            "candidate",
            required_kind="opaque",
            required_schema="rwkv-lab.posttraining-output.v1",
        )


def test_input_artifact_resolution_rejects_descriptor_substitution(
    tmp_path: Path,
) -> None:
    source = tmp_path / "output"
    source.mkdir()
    (source / "result.json").write_text("{}", encoding="utf-8")
    session = FakeArtifactSession(tmp_path)
    published = ImmutableArtifactPublisher(session, output_name="adapter").publish(
        source
    )
    descriptor = _input_descriptor(session, published)
    descriptor["producer_node_id"] = "different-node"
    invocation = SimpleNamespace(
        inputs={"candidate": descriptor},
        workspace={"allowed_read_roots": (str(tmp_path),)},
    )
    with pytest.raises(ArtifactPublicationError, match="semantics"):
        resolve_input_artifact(
            invocation,
            "candidate",
            required_kind="opaque",
            required_schema="rwkv-lab.posttraining-output.v1",
        )


def test_artifact_publication_namespace_rejects_path_and_symlink_escape(
    tmp_path: Path,
) -> None:
    session = FakeArtifactSession(tmp_path)
    session.invocation = SimpleNamespace(
        invocation_digest=session.invocation.invocation_digest,
        workspace=session.invocation.workspace,
        publishes={"../adapter": session.invocation.publishes["adapter"]},
    )
    with pytest.raises(ArtifactPublicationError, match="one path component"):
        ImmutableArtifactPublisher(session, output_name="../adapter")

    target = tmp_path / "alternate-publication-root"
    target.mkdir()
    (tmp_path / "trainvm_artifacts").symlink_to(target, target_is_directory=True)
    session = FakeArtifactSession(tmp_path)
    with pytest.raises(ArtifactPublicationError, match="contains a symlink"):
        ImmutableArtifactPublisher(session, output_name="adapter")
