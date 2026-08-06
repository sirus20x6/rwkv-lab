from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import pytest

from rwkv_lab.trainvm_worker import (
    EVAL_EXAMPLES_SCHEMA,
    EvalEvidencePart,
    EvalExample,
    EvalExamplesError,
    EvalExamplesPublicationRequest,
    EvalExamplesPublisher,
    EvalMedia,
)


def digest(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


class FakeSession:
    def __init__(self, root: Path) -> None:
        self.bootstrap = SimpleNamespace(
            run_id="run-1", node_id="train", attempt_id="train@1"
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
                    "eval_examples": MappingProxyType(
                        {
                            "logical_name": "eval_examples",
                            "declaration": MappingProxyType(
                                {
                                    "type": "eval_examples",
                                    "schema": EVAL_EXAMPLES_SCHEMA,
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
        return 11


def request(media: Path) -> EvalExamplesPublicationRequest:
    heldout = b"heldout row"
    return EvalExamplesPublicationRequest(
        output_name="eval_examples",
        optimizer_step=0,
        series_id="fixed-validation",
        identity_field="sample_id",
        identities_digest=digest(b"sample-1"),
        selector_digest=digest(b"selector"),
        evaluator_component_digest=digest(b"evaluator"),
        metric_names=("eval.loss",),
        checkpoint_artifact_id="checkpoint-0",
        checkpoint_manifest_digest=digest(b"checkpoint"),
        policy_digest=digest(b"policy"),
        examples=(
            EvalExample(
                example_id="sample-1",
                heldout_item_id="row-1",
                heldout_item_digest=digest(heldout),
                input=(EvalEvidencePart(kind="text", text="describe image"),),
                target=(EvalEvidencePart(kind="text", text="teacher caption"),),
                prediction=(
                    EvalEvidencePart(
                        kind="image",
                        media=EvalMedia("image", media, "image/png"),
                    ),
                ),
            ),
        ),
        parent_artifact_ids=("checkpoint-0", "heldout-snapshot"),
    )


def test_eval_examples_publisher_freezes_relative_content_and_binds_wire_contract(
    tmp_path: Path,
) -> None:
    media = tmp_path / "generated.png"
    content = b"not-decoded-by-contract\x00png"
    media.write_bytes(content)
    session = FakeSession(tmp_path)

    result = EvalExamplesPublisher(session, output_name="eval_examples").publish(
        request(media)
    )

    assert result.worker_sequence == 11
    manifest_bytes = result.manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    part = manifest["examples"][0]["prediction"][0]
    assert part["path"] == f"objects/{hashlib.sha256(content).hexdigest()}"
    assert not Path(part["path"]).is_absolute()
    assert (result.manifest_path.parent / part["path"]).read_bytes() == content
    assert part["sha256"] == digest(content)
    assert session.artifacts == [
        {
            "artifact_id": result.artifact_id,
            "logical_name": "eval_examples",
            "kind": 8,
            "schema": EVAL_EXAMPLES_SCHEMA,
            "uri": result.manifest_path.resolve().as_uri(),
            "size_bytes": len(manifest_bytes),
            "fingerprint_algorithm": "manifest_sha256",
            "fingerprint": digest(manifest_bytes),
            "parent_artifact_ids": ("checkpoint-0", "heldout-snapshot"),
            "optimizer_step": 0,
            "canonical_manifest_json": manifest_bytes,
            "wait": True,
        }
    ]

    media.write_bytes(b"mutated source")
    assert (result.manifest_path.parent / part["path"]).read_bytes() == content


def test_eval_examples_rejects_empty_examples_and_escaped_media(tmp_path: Path) -> None:
    source = tmp_path / "media"
    source.write_bytes(b"content")
    publisher = EvalExamplesPublisher(
        FakeSession(tmp_path), output_name="eval_examples"
    )
    empty = request(source)
    empty = replace(empty, examples=())
    with pytest.raises(EvalExamplesError, match="example count"):
        publisher.publish(empty)

    outside = tmp_path.parent / "outside-eval-examples"
    outside.write_bytes(b"outside")
    try:
        with pytest.raises(EvalExamplesError, match="outside read authority"):
            publisher.publish(request(outside))
    finally:
        outside.unlink(missing_ok=True)


def test_eval_examples_rejects_invalid_parents_and_corrupt_content_store(
    tmp_path: Path,
) -> None:
    source = tmp_path / "media"
    content = b"content"
    source.write_bytes(content)
    publisher = EvalExamplesPublisher(
        FakeSession(tmp_path), output_name="eval_examples"
    )
    missing_checkpoint = replace(
        request(source), parent_artifact_ids=("heldout-snapshot",)
    )
    with pytest.raises(EvalExamplesError, match="checkpoint.*artifact parent"):
        publisher.publish(missing_checkpoint)

    digest_hex = hashlib.sha256(content).hexdigest()
    object_path = (
        tmp_path
        / "trainvm_artifacts"
        / "eval_examples"
        / "objects"
        / "sha256"
        / digest_hex[:2]
        / digest_hex
    )
    object_path.parent.mkdir(parents=True, exist_ok=True)
    object_path.write_bytes(b"corrupt")
    with pytest.raises(EvalExamplesError, match="content-addressed.*mutated"):
        publisher.publish(request(source))
