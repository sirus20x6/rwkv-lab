from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import pytest
import torch

from rwkv_lab.training_runtime.optimizers import (
    AdamWNoDecayConfiguration,
    OptimizerImplementation,
    build_registered_optimizer,
)
from rwkv_lab.training_runtime.schedules import (
    LinearWarmupCosineConfiguration,
    ScheduleImplementation,
    build_registered_schedule,
    linear_warmup_cosine_multiplier,
)
from rwkv_lab.training_runtime.weight_decay_schedules import (
    ConstantWeightDecayConfiguration,
    WeightDecayScheduleImplementation,
    build_registered_weight_decay_schedule,
)

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "docs/experiment-vm/examples"
PARITY = json.loads((EXAMPLES / "qwen-caption-corrected-v3.parity.v1.json").read_text())
PROFILES = json.loads(
    (EXAMPLES / "hf-multimodal-sft.recipe-profiles.v1.json").read_text()
)["recipes"]
PROFILE = next(profile for profile in PROFILES if profile["key"]["version"] == "2")
INSTANCE = json.loads(
    (EXAMPLES / "qwen-caption-lora-r256.recipe-instance.v1.json").read_text()
)


def _components() -> dict[str, object]:
    return PROFILE["template_document"]["spec"]["workflow"]["nodes"]["train"]["invoke"][
        "training"
    ]["components"]


def _legacy_multiplier(
    step: int, *, warmup: int, maximum: int, minimum: float
) -> float:
    if step < warmup:
        return max(1.0e-8, step / float(warmup))
    progress = min(1.0, max(0.0, (step - warmup) / float(maximum - warmup)))
    return minimum + (1.0 - minimum) * 0.5 * (1.0 + math.cos(math.pi * progress))


def test_compact_recipe_preserves_corrected_v3_semantic_contract() -> None:
    components = _components()
    identity = PARITY["identity"]
    data = PARITY["data"]
    trainability = PARITY["trainability"]
    optimization = PARITY["optimization"]
    cadence = PARITY["cadence"]

    assert INSTANCE["recipe"] == {"name": "hf_multimodal_sft", "version": "2"}
    assert INSTANCE["overrides"] == {
        "model.path": identity["model_path"],
        "data.root": data["dataset_root"],
        "model.target_manifest": str(Path(data["dataset_root"]) / "lora-targets.json"),
        "data.batch_size": optimization["micro_batch_size"],
        "trainability.lora_rank": trainability["rank"],
        "trainability.lora_alpha": trainability["alpha"],
        "hyperparameters.learning_rate": optimization["learning_rate"],
        "hyperparameters.gradient_accumulation": optimization["gradient_accumulation"],
        "hyperparameters.maximum_steps": optimization["maximum_steps"],
        "evaluation.qualitative_sample_count": data["held_out_count"],
        "data.qualitative_manifest_name": data["held_out_manifest"],
        "data.qualitative_manifest_sha256": "sha256:"
        + data["held_out_manifest_sha256"],
        "evaluation.qualitative_every_steps": cadence["evaluation_every_steps"],
        "evaluation.full_every_steps": cadence["evaluation_every_steps"],
        "checkpointing.every_steps": cadence["checkpoint_every_steps"],
    }

    loader = components["model_loader"]["configuration"]
    assert components["model_loader"]["key"] == {
        "category": "model_loader",
        "name": "hf_multimodal",
        "version": "2.0.0",
    }
    assert loader["attention_implementation"] == identity["attention_implementation"]
    assert loader["experts_implementation"] == identity["experts_implementation"]
    assert loader["exact_checkpoint"] is True
    assert loader["local_files_only"] is True

    lora = components["trainability"]["configuration"]
    assert (lora["rank"], lora["alpha"], lora["dropout"]) == (
        trainability["rank"],
        trainability["alpha"],
        trainability["dropout"],
    )
    assert lora["required_policy_digest"] == trainability["target_policy_digest"]

    optimizer = components["optimizer"]["configuration"]
    assert optimizer == {
        "learning_rate": optimization["learning_rate"],
        "beta1": optimization["beta1"],
        "beta2": optimization["beta2"],
        "epsilon": optimization["epsilon"],
        "foreach": optimization["foreach"],
        "fused": optimization["fused"],
    }
    assert (
        components["weight_decay"]["configuration"]["weight_decay"]
        == (optimization["weight_decay"])
    )
    assert (
        components["sample_mapping"]["configuration"]["target_column"]
        == (data["target_column"])
    )
    assert components["sample_mapping"]["configuration"]["append_eos"] is True
    held_out = components["qualitative_samples"]["configuration"]
    assert held_out == {
        "manifest_name": data["held_out_manifest"],
        "manifest_sha256": "sha256:" + data["held_out_manifest_sha256"],
        "identity_field": "id",
        "sample_count": data["held_out_count"],
    }
    renderer = components["artifact_renderer"]
    assert renderer["key"]["name"] == "caption_triplet"
    assert PROFILE["template_document"]["spec"]["recovery"]["exact_resume"] is True


def test_frozen_qwen_assets_match_parity_contract_when_present() -> None:
    model = Path(PARITY["identity"]["model_path"])
    dataset = Path(PARITY["data"]["dataset_root"])
    if not model.is_dir() or not dataset.is_dir():
        pytest.skip("production Qwen assets are not installed on this test host")

    assert (
        hashlib.sha256((model / "config.json").read_bytes()).hexdigest()
        == (PARITY["identity"]["model_config_sha256"])
    )
    assert (
        hashlib.sha256(
            (model / "model.safetensors.index.json").read_bytes()
        ).hexdigest()
        == PARITY["identity"]["weight_index_sha256"]
    )
    manifest = json.loads((dataset / "manifest.json").read_text())
    assert manifest["dataset_digest"] == PARITY["data"]["dataset_digest"]
    assert manifest["counts"] == PARITY["data"]["split_counts"]

    held_out = dataset / PARITY["data"]["held_out_manifest"]
    assert (
        hashlib.sha256(held_out.read_bytes()).hexdigest()
        == (PARITY["data"]["held_out_manifest_sha256"])
    )
    identities = tuple(
        json.loads(line)["id"] for line in held_out.read_text().splitlines()[:10]
    )
    encoded = json.dumps(identities, ensure_ascii=False, separators=(",", ":")).encode()
    assert (
        "sha256:" + hashlib.sha256(encoded).hexdigest()
        == (PARITY["data"]["held_out_identities_sha256"])
    )
    targets = dataset / "lora-targets.json"
    assert (
        hashlib.sha256(targets.read_bytes()).hexdigest()
        == (PARITY["trainability"]["target_manifest_sha256"])
    )


def test_legacy_and_composed_schedule_trajectories_are_exact() -> None:
    values = PARITY["optimization"]
    configuration = LinearWarmupCosineConfiguration(
        warmup_steps=values["warmup_steps"],
        max_steps=values["maximum_steps"],
        minimum_ratio=values["minimum_ratio"],
    )
    tolerance = PARITY["tolerances"]["schedule_absolute"]
    for record in PARITY["bounded_schedule_trajectory"]:
        step = record["step"]
        legacy = _legacy_multiplier(
            step,
            warmup=values["warmup_steps"],
            maximum=values["maximum_steps"],
            minimum=values["minimum_ratio"],
        )
        composed = linear_warmup_cosine_multiplier(step, configuration)
        assert legacy == pytest.approx(record["multiplier"], abs=tolerance, rel=0)
        assert composed == pytest.approx(legacy, abs=tolerance, rel=0)


def test_legacy_and_composed_optimizers_match_over_bounded_steps() -> None:
    values = PARITY["optimization"]
    legacy_parameter = torch.nn.Parameter(torch.tensor([1.25, -0.75, 0.5]))
    composed_parameter = torch.nn.Parameter(legacy_parameter.detach().clone())
    common = {
        "lr": values["learning_rate"],
        "betas": (values["beta1"], values["beta2"]),
        "eps": values["epsilon"],
        "weight_decay": values["weight_decay"],
        "foreach": values["foreach"],
        "fused": values["fused"],
    }
    legacy = torch.optim.AdamW((legacy_parameter,), **common)
    composed = build_registered_optimizer(
        OptimizerImplementation.TORCH_ADAMW_NO_DECAY_V2,
        (composed_parameter,),
        AdamWNoDecayConfiguration(
            learning_rate=values["learning_rate"],
            beta1=values["beta1"],
            beta2=values["beta2"],
            epsilon=values["epsilon"],
            foreach=values["foreach"],
            fused=values["fused"],
        ),
    )
    build_registered_weight_decay_schedule(
        WeightDecayScheduleImplementation.CONSTANT_V1,
        composed,
        ConstantWeightDecayConfiguration(values["weight_decay"]),
    )
    legacy_schedule = torch.optim.lr_scheduler.LambdaLR(
        legacy,
        lambda step: _legacy_multiplier(
            step,
            warmup=values["warmup_steps"],
            maximum=values["maximum_steps"],
            minimum=values["minimum_ratio"],
        ),
    )
    composed_schedule = build_registered_schedule(
        ScheduleImplementation.LINEAR_WARMUP_COSINE_V1,
        composed,
        LinearWarmupCosineConfiguration(
            values["warmup_steps"], values["maximum_steps"], values["minimum_ratio"]
        ),
    )

    for step in range(1, 9):
        gradient = torch.tensor([0.1 * step, -0.03 * step, 0.02 * (step + 1)])
        legacy_parameter.grad = gradient.clone()
        composed_parameter.grad = gradient.clone()
        legacy.step()
        composed.step()
        legacy_schedule.step()
        composed_schedule.step()
        legacy.zero_grad(set_to_none=True)
        composed.zero_grad(set_to_none=True)

    tolerance = PARITY["tolerances"]["bounded_cpu_optimizer_absolute"]
    torch.testing.assert_close(
        composed_parameter, legacy_parameter, atol=tolerance, rtol=0
    )
    assert composed_schedule.get_last_lr() == pytest.approx(
        legacy_schedule.get_last_lr(), abs=1e-15, rel=0
    )
