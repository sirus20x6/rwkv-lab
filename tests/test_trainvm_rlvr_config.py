from __future__ import annotations

from dataclasses import fields

import pytest

from rwkv_lab.trainvm_adapters.rlvr import RLVRTrainConfig


def minimum_config() -> dict[str, object]:
    return {
        "checkpoint": "/sealed/base.pt",
        "output_dir": "/run/rlvr",
        "vocab": "/sealed/vocab.txt",
    }


def test_rlvr_config_is_exact_and_normalizes_sequences() -> None:
    config = RLVRTrainConfig(
        **minimum_config(),
        rollout_devices=["cuda:0", "cuda:1"],
        device="cuda:0",
        curriculum_stages=[1, 2, 3],
        verifier_executable="/sealed/verifier",
        verifier_arguments=["--profile", "math"],
    )
    assert config.rollout_devices == ("cuda:0", "cuda:1")
    assert config.curriculum_stages == (1, 2, 3)
    assert config.verifier_arguments == ("--profile", "math")
    assert {field.name for field in fields(config)} == {
        "checkpoint", "output_dir", "vocab", "tasks", "heldout_tasks",
        "algorithm", "steps", "prompts_per_step", "group_size", "epochs",
        "max_new_tokens", "rollout_engine", "rollout_devices", "temperature",
        "eval_temperature", "top_p", "top_k", "stop_token", "learning_rate",
        "weight_decay", "optimizer", "warmup_steps", "max_gradient_norm",
        "clip_low", "clip_high", "kl_coefficient", "reference",
        "reference_checkpoint", "train_tasks", "eval_tasks", "difficulty",
        "curriculum_stages", "sft_steps", "sft_batch_size",
        "sft_learning_rate", "preflight_prompts", "minimum_preflight_reward",
        "maximum_preflight_reward", "minimum_preflight_active_groups",
        "eval_every", "eval_prompts", "eval_group_size",
        "minimum_heldout_delta", "confidence", "bootstrap_samples",
        "require_confidence", "maximum_family_regression",
        "maximum_rollout_tokens", "maximum_train_seconds", "save_every",
        "verifier_executable", "verifier_arguments", "verifier_timeout",
        "log_samples", "seed", "device", "use_ema",
    }


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("group_size", 1, "group_size"),
        ("algorithm", "ppo", "algorithm"),
        ("confidence", 1.0, "confidence"),
        ("minimum_preflight_reward", 0.99, "minimum_preflight_reward"),
        ("device", "cpu", "device"),
        ("verifier_arguments", ["--unsafe"], "verifier_arguments"),
    ],
)
def test_rlvr_config_rejects_invalid_values(
    field: str, value: object, message: str
) -> None:
    values = minimum_config()
    values[field] = value
    with pytest.raises(ValueError, match=message):
        RLVRTrainConfig(**values)


def test_rlvr_config_rejects_unknown_fields() -> None:
    with pytest.raises(TypeError):
        RLVRTrainConfig(**minimum_config(), shell_command="curl example.test")
