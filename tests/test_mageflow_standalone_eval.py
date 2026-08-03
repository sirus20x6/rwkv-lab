from __future__ import annotations

import json
from contextlib import nullcontext
from dataclasses import asdict
from types import SimpleNamespace

import pytest

from rwkv_lab.mage_flow_terminal_train import TerminalExpertTrainConfig
from rwkv_lab.mage_flow_tread_looping import TreadLoopConfig
from rwkv_lab.trainvm_adapters.content_authority import measure_input_content_root
from rwkv_lab.trainvm_adapters.handlers import _mageflow_standalone_eval
from rwkv_lab.trainvm_adapters.mageflow_eval import (
    MageFlowStandaloneEvalConfig,
    MageFlowStandaloneEvalError,
    resolve_mageflow_evaluation,
)


def _row(image: str, domain: str = "animation") -> dict[str, object]:
    return {
        "image": image,
        "image_id": "heldout-1",
        "domain": domain,
        "source": "test",
        "caption": "a held-out frame",
        "conditioning_text": "a held-out frame",
        "conditioning_kind": "human",
        "is_captioned": True,
        "training_scope": "expert_and_selected_shared",
        "train_width": 512,
        "train_height": 512,
        "latent_tokens": 1024,
    }


def _checkpoint(tmp_path, manifest, model):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    expert = checkpoint / "animation.safetensors"
    expert.write_bytes(b"expert")
    config = TerminalExpertTrainConfig(
        domain="animation",
        train_manifest=str(manifest),
        eval_manifest=str(manifest),
        expert_checkpoint=str(expert),
        output_dir=str(tmp_path / "old-run"),
        model_path=str(model),
        repa_enabled=False,
        tread_factored_looping=TreadLoopConfig().to_dict(),
    )
    fingerprint = "checkpoint-fingerprint"
    (checkpoint / "run_contract.json").write_text(
        json.dumps(
            {
                "schema": "rwkv-lab.mage-flow-terminal-train.v3",
                "fingerprint": fingerprint,
                "config": asdict(config),
            }
        ),
        encoding="utf-8",
    )
    (checkpoint / "checkpoint.json").write_text(
        json.dumps(
            {
                "schema": "rwkv-lab.mage-flow-terminal-train.v3",
                "domain": "animation",
                "step": 41,
                "fingerprint": fingerprint,
                "expert": expert.name,
                "experts": {"animation": expert.name},
                "shared_backbone": None,
                "repa_projection": None,
                "tread_loop_controller": None,
                "run_contract": "run_contract.json",
            }
        ),
        encoding="utf-8",
    )
    return checkpoint


def test_checkpoint_contract_resolves_eval_without_training_or_resume(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    manifest = tmp_path / "eval.jsonl"
    manifest.write_text(
        json.dumps(_row(str(tmp_path / "image.png"))) + "\n", encoding="utf-8"
    )
    checkpoint = _checkpoint(tmp_path, manifest, model)
    request = MageFlowStandaloneEvalConfig(
        domain="animation",
        eval_manifest=str(manifest),
        model_path=str(model),
        eval_microbatch_size=2,
        eval_examples=1,
        attention_backend="flash4",
    )

    resolved = resolve_mageflow_evaluation(
        request,
        checkpoint,
        tmp_path / "new-eval",
        optimizer_step=41,
    )

    assert resolved.optimizer_step == 41
    assert resolved.expert_checkpoint == checkpoint / "animation.safetensors"
    assert resolved.trainer_config.train_manifest == str(manifest)
    assert resolved.trainer_config.eval_manifest == str(manifest)
    assert resolved.trainer_config.output_dir == str(tmp_path / "new-eval")
    assert resolved.trainer_config.resume_from is None
    assert resolved.trainer_config.encoder_cache_mode == "off"
    assert resolved.trainer_config.gradient_checkpointing is False
    assert resolved.trainer_config.attention_backend == "flash4"


def test_checkpoint_eval_rejects_authority_step_mismatch_and_path_escape(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    manifest = tmp_path / "eval.jsonl"
    manifest.write_text(
        json.dumps(_row(str(tmp_path / "image.png"))) + "\n", encoding="utf-8"
    )
    checkpoint = _checkpoint(tmp_path, manifest, model)
    request = MageFlowStandaloneEvalConfig(
        domain="animation", eval_manifest=str(manifest), model_path=str(model)
    )
    with pytest.raises(MageFlowStandaloneEvalError, match="authority"):
        resolve_mageflow_evaluation(
            request, checkpoint, tmp_path / "eval", optimizer_step=42
        )

    document = json.loads((checkpoint / "checkpoint.json").read_text())
    document["experts"]["animation"] = "../outside.safetensors"
    (checkpoint / "checkpoint.json").write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(MageFlowStandaloneEvalError, match="path is invalid"):
        resolve_mageflow_evaluation(
            request, checkpoint, tmp_path / "eval", optimizer_step=41
        )


def test_checkpoint_eval_requires_every_enabled_architecture_state(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    manifest = tmp_path / "eval.jsonl"
    manifest.write_text(
        json.dumps(_row(str(tmp_path / "image.png"))) + "\n", encoding="utf-8"
    )
    checkpoint = _checkpoint(tmp_path, manifest, model)
    run_contract = json.loads((checkpoint / "run_contract.json").read_text())
    run_contract["config"]["repa_enabled"] = True
    (checkpoint / "run_contract.json").write_text(
        json.dumps(run_contract), encoding="utf-8"
    )
    request = MageFlowStandaloneEvalConfig(
        domain="animation", eval_manifest=str(manifest), model_path=str(model)
    )

    with pytest.raises(MageFlowStandaloneEvalError, match="REPA projection"):
        resolve_mageflow_evaluation(
            request, checkpoint, tmp_path / "eval", optimizer_step=41
        )

    with pytest.raises(MageFlowStandaloneEvalError, match="inexact"):
        MageFlowStandaloneEvalConfig.from_mapping(
            {
                "domain": "animation",
                "eval_manifest": str(manifest),
                "model_path": str(model),
                "optimizer": "forbidden",
            }
        )


def test_standalone_eval_handler_publishes_checkpoint_bound_gallery_and_result(
    tmp_path, monkeypatch
):
    read_root = tmp_path / "read"
    model = read_root / "model"
    checkpoint = read_root / "checkpoint"
    model.mkdir(parents=True)
    checkpoint.mkdir()
    image = read_root / "image.png"
    image.write_bytes(b"image")
    manifest = read_root / "eval.jsonl"
    manifest.write_text(json.dumps(_row(str(image))) + "\n", encoding="utf-8")
    run_directory = tmp_path / "write" / "run"
    run_directory.mkdir(parents=True)
    resolved_checkpoint = SimpleNamespace(
        artifact_id="checkpoint-41",
        manifest_sha256="sha256:" + "b" * 64,
        optimizer_step=41,
        payload_directory=checkpoint,
    )
    monkeypatch.setattr(
        "rwkv_lab.trainvm_adapters.handlers.resolve_input_checkpoint",
        lambda *_args, **_kwargs: resolved_checkpoint,
    )
    monkeypatch.setattr(
        "rwkv_lab.trainvm_adapters.handlers.resolve_mageflow_evaluation",
        lambda *_args, **_kwargs: SimpleNamespace(),
    )

    def evaluate(_evaluation, output_directory):
        generated = output_directory / "generated.png"
        generated.write_bytes(b"generated")
        gallery = output_directory / "eval_samples" / "step_00000041.json"
        gallery.parent.mkdir()
        gallery.write_text(
            json.dumps(
                {
                    "complete": True,
                    "step": 41,
                    "items": [
                        {
                            "image_id": "heldout-1",
                            "image": str(generated),
                            "target_image": str(image),
                            "prompt": "a held-out frame",
                            "route": "animation",
                            "seed": 123,
                            "sampling_attributes": {
                                "cfg": "4",
                                "route": "animation",
                                "sampler": "mage_flow_rectified_flow",
                                "steps": "28",
                            },
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        return {"eval/primary_loss": 0.25, "eval/gallery_samples": 1}

    monkeypatch.setattr(
        "rwkv_lab.trainvm_adapters.handlers.evaluate_mageflow_checkpoint", evaluate
    )
    published_metrics = []
    observability = SimpleNamespace(
        keepalive=lambda *_args: nullcontext(),
        publish_if_declared=lambda name, value, **kwargs: published_metrics.append(
            (name, value, kwargs)
        ),
    )
    invocation = SimpleNamespace(
        inputs={
            "config": {
                "domain": "animation",
                "eval_manifest": str(manifest),
                "model_path": str(model),
            },
            "checkpoint": {},
        },
        workspace={
            "run_directory": str(run_directory),
            "allowed_read_roots": [str(read_root)],
            "input_content_roots": [asdict(measure_input_content_root(read_root))],
            "allowed_write_roots": [str(run_directory.parent)],
        },
        publishes={"eval_gallery": {}, "result": {}},
        resume=None,
        node_id="evaluate_animation",
        attempt_id="evaluate_animation@1",
        invocation_digest="sha256:" + "a" * 64,
    )

    result = _mageflow_standalone_eval(
        invocation,
        None,
        observability=observability,
        controls=SimpleNamespace(effective_values={}),
    )

    assert result.event_type == "operation.completed"
    assert result.optimizer_step == 41
    assert result.payload["primary_loss"] == pytest.approx(0.25)
    assert result.checkpoint_requests == ()
    assert result.artifact_requests[0].parent_artifact_ids == ("checkpoint-41",)
    report = json.loads(
        (result.artifact_requests[0].source_directory / "result.json").read_text()
    )
    assert report["checkpoint_manifest_digest"] == "sha256:" + "b" * 64
    gallery = result.eval_gallery_requests[0]
    assert gallery.checkpoint_request_index is None
    assert gallery.checkpoint_manifest_digest == "sha256:" + "b" * 64
    assert gallery.parent_artifact_ids == ("checkpoint-41",)
    assert {item[0] for item in published_metrics} == {
        "eval/gallery_samples",
        "eval/primary_loss",
    }
