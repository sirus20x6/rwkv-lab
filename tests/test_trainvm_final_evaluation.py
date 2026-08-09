from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from rwkv_lab.trainvm_worker import (
    FINAL_EVALUATION_SCHEMA,
    FinalEvaluationPublicationError,
    FinalEvaluationPublicationRequest,
    FinalEvaluationPublisher,
)


def sha256(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


class FakeSession:
    def __init__(self, run_directory) -> None:
        self.bootstrap = SimpleNamespace(
            run_id="run-1", node_id="train", attempt_id="train@1"
        )
        self.invocation = SimpleNamespace(
            workspace={"run_directory": str(run_directory)},
            publishes={
                "final_evaluation": {
                    "logical_name": "final_evaluation",
                    "finalization_policy_digest": "sha256:" + "a" * 64,
                    "declaration": {
                        "type": "report",
                        "schema": FINAL_EVALUATION_SCHEMA,
                        "immutability": "immutable",
                        "fingerprint": "manifest_sha256",
                    },
                }
            },
        )
        self.artifacts: list[dict[str, object]] = []

    def artifact(self, **values: object) -> int:
        self.artifacts.append(values)
        return 19


def request() -> FinalEvaluationPublicationRequest:
    output_receipts = (
        {
            "output_name": "checkpoint",
            "artifact_id": "checkpoint-10",
            "artifact_fingerprint": "sha256:" + "b" * 64,
        },
        {
            "output_name": "eval_gallery",
            "artifact_id": "gallery-10",
            "artifact_fingerprint": "sha256:" + "c" * 64,
        },
        {
            "output_name": "test_eval",
            "artifact_id": "test-10",
            "artifact_fingerprint": "sha256:" + "d" * 64,
        },
    )
    return FinalEvaluationPublicationRequest(
        optimizer_step=10,
        checkpoint_artifact_id="checkpoint-10",
        checkpoint_fingerprint="sha256:" + "b" * 64,
        required_members=("member-a", "member-b"),
        member_context_digests={
            "member-a": "sha256:" + "e" * 64,
            "member-b": "sha256:" + "f" * 64,
        },
        records=(
            {
                "member_id": "member-a",
                "context_digest": "sha256:" + "e" * 64,
                "attempt": 1,
                "disposition": "success",
                "result_digest": "sha256:" + "1" * 64,
            },
            {
                "member_id": "member-b",
                "context_digest": "sha256:" + "f" * 64,
                "attempt": 1,
                "disposition": "success",
                "result_digest": "sha256:" + "2" * 64,
            },
        ),
        output_receipts=output_receipts,
        required_scalars=(
            {"metric_name": "eval.loss", "step_domain": "optimizer_step"},
        ),
        parent_artifact_ids=("checkpoint-10", "gallery-10", "test-10"),
    )


def test_final_evaluation_publisher_seals_canonical_controller_bound_closure(
    tmp_path,
) -> None:
    session = FakeSession(tmp_path)
    published = FinalEvaluationPublisher(session).publish(request())

    assert published.worker_sequence == 19
    manifest_bytes = published.manifest_path.read_bytes()
    assert published.manifest_sha256 == sha256(manifest_bytes)
    assert published.manifest_path.stat().st_mode & 0o777 == 0o440
    manifest = json.loads(manifest_bytes)
    assert manifest["policy_digest"] == "sha256:" + "a" * 64
    assert manifest["membership_count"] == 2
    assert manifest["resolved_member_count"] == 2
    assert session.artifacts == [
        {
            "artifact_id": published.artifact_id,
            "logical_name": "final_evaluation",
            "kind": 6,
            "schema": FINAL_EVALUATION_SCHEMA,
            "uri": published.manifest_path.resolve().as_uri(),
            "size_bytes": len(manifest_bytes),
            "fingerprint_algorithm": "manifest_sha256",
            "fingerprint": published.manifest_sha256,
            "parent_artifact_ids": (
                "checkpoint-10",
                "gallery-10",
                "test-10",
            ),
            "optimizer_step": 10,
            "wait": True,
        }
    ]


def test_final_evaluation_publisher_rejects_worker_selected_parent_set(
    tmp_path,
) -> None:
    candidate = request()
    candidate = replace(candidate, parent_artifact_ids=("checkpoint-10",))
    with pytest.raises(FinalEvaluationPublicationError, match="exactly equal"):
        FinalEvaluationPublisher(FakeSession(tmp_path)).publish(candidate)
