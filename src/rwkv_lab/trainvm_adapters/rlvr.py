"""Closed TrainVM configuration for one bounded RWKV RLVR candidate run."""

from __future__ import annotations

import math
from dataclasses import dataclass


def _path(value: object, label: str, *, optional: bool = False) -> str:
    if optional and value == "":
        return ""
    if not isinstance(value, str) or not value or "\x00" in value:
        raise ValueError(f"{label} must be a nonempty path string")
    return value


def _integer(value: object, label: str, minimum: int, maximum: int) -> int:
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or not minimum <= value <= maximum
    ):
        raise ValueError(f"{label} must be an integer in [{minimum}, {maximum}]")
    return value


def _finite(value: object, label: str, minimum: float, maximum: float) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or not minimum <= float(value) <= maximum
    ):
        raise ValueError(f"{label} must be finite in [{minimum}, {maximum}]")
    return float(value)


def _string_tuple(value: object, label: str) -> tuple[str, ...]:
    if not isinstance(value, (tuple, list)) or any(
        not isinstance(item, str) or not item or "\x00" in item for item in value
    ):
        raise ValueError(f"{label} must be a list of nonempty strings")
    return tuple(value)


def _integer_tuple(value: object, label: str) -> tuple[int, ...]:
    if not isinstance(value, (tuple, list)):
        raise TypeError(f"{label} must be a list of integers")
    result = tuple(value)
    for item in result:
        _integer(item, label, 1, 1_000_000)
    return result


@dataclass(frozen=True, slots=True)
class RLVRTrainConfig:
    checkpoint: str
    output_dir: str
    vocab: str
    tasks: str = ""
    heldout_tasks: str = ""
    algorithm: str = "gspo"
    steps: int = 100
    prompts_per_step: int = 2
    group_size: int = 8
    epochs: int = 1
    max_new_tokens: int = 64
    rollout_engine: str = "auto"
    rollout_devices: tuple[str, ...] = ()
    temperature: float = 1.0
    eval_temperature: float = 0.7
    top_p: float = 1.0
    top_k: int = 0
    stop_token: int = 1
    learning_rate: float = 1.0e-6
    weight_decay: float = 0.0
    optimizer: str = "adamw"
    warmup_steps: int = 10
    max_gradient_norm: float = 1.0
    clip_low: float = 0.2
    clip_high: float = -1.0
    kl_coefficient: float = 0.01
    reference: str = "initial"
    reference_checkpoint: str = ""
    train_tasks: int = 4096
    eval_tasks: int = 256
    difficulty: int = 2
    curriculum_stages: tuple[int, ...] = ()
    sft_steps: int = 0
    sft_batch_size: int = 2
    sft_learning_rate: float = 2.0e-5
    preflight_prompts: int = 0
    minimum_preflight_reward: float = 0.01
    maximum_preflight_reward: float = 0.99
    minimum_preflight_active_groups: int = 1
    eval_every: int = 10
    eval_prompts: int = 32
    eval_group_size: int = 4
    minimum_heldout_delta: float = 0.01
    confidence: float = 0.95
    bootstrap_samples: int = 10_000
    require_confidence: bool = True
    maximum_family_regression: float = 0.0
    maximum_rollout_tokens: int = 0
    maximum_train_seconds: float = 0.0
    save_every: int = 10
    verifier_executable: str = ""
    verifier_arguments: tuple[str, ...] = ()
    verifier_timeout: float = 10.0
    log_samples: int = 2
    seed: int = 0
    device: str = "cuda"
    use_ema: bool = False

    def __post_init__(self) -> None:
        for label in ("checkpoint", "output_dir", "vocab"):
            _path(getattr(self, label), label)
        for label in (
            "tasks",
            "heldout_tasks",
            "reference_checkpoint",
            "verifier_executable",
        ):
            _path(getattr(self, label), label, optional=True)
        object.__setattr__(
            self,
            "rollout_devices",
            _string_tuple(self.rollout_devices, "rollout_devices"),
        )
        object.__setattr__(
            self,
            "verifier_arguments",
            _string_tuple(self.verifier_arguments, "verifier_arguments"),
        )
        object.__setattr__(
            self,
            "curriculum_stages",
            _integer_tuple(self.curriculum_stages, "curriculum_stages"),
        )
        if self.algorithm not in {"gspo", "dr_grpo", "dapo"}:
            raise ValueError("algorithm must be gspo, dr_grpo, or dapo")
        if self.rollout_engine not in {"auto", "recurrent", "batched"}:
            raise ValueError("rollout_engine must be auto, recurrent, or batched")
        if self.optimizer not in {"adamw", "adamw8bit", "paged-adamw8bit"}:
            raise ValueError("optimizer is not supported")
        if self.reference not in {"initial", "rollout", "none"}:
            raise ValueError("reference must be initial, rollout, or none")
        if self.verifier_arguments and not self.verifier_executable:
            raise ValueError("verifier_arguments require verifier_executable")
        for label, minimum in (
            ("steps", 1),
            ("prompts_per_step", 1),
            ("group_size", 2),
            ("epochs", 1),
            ("max_new_tokens", 1),
            ("top_k", 0),
            ("stop_token", 0),
            ("warmup_steps", 0),
            ("train_tasks", 1),
            ("eval_tasks", 1),
            ("difficulty", 1),
            ("sft_steps", 0),
            ("sft_batch_size", 1),
            ("preflight_prompts", 0),
            ("minimum_preflight_active_groups", 1),
            ("eval_every", 0),
            ("eval_prompts", 1),
            ("eval_group_size", 1),
            ("bootstrap_samples", 0),
            ("maximum_rollout_tokens", 0),
            ("save_every", 0),
            ("log_samples", 0),
            ("seed", 0),
        ):
            _integer(getattr(self, label), label, minimum, 1_000_000_000)
        for label, minimum, maximum in (
            ("temperature", 1.0e-12, 1000.0),
            ("eval_temperature", 1.0e-12, 1000.0),
            ("top_p", 1.0e-12, 1.0),
            ("learning_rate", 1.0e-12, 1.0),
            ("weight_decay", 0.0, 10.0),
            ("max_gradient_norm", 1.0e-12, 1.0e12),
            ("clip_low", 0.0, 1000.0),
            ("kl_coefficient", 0.0, 1.0e6),
            ("sft_learning_rate", 1.0e-12, 1.0),
            ("minimum_preflight_reward", 0.0, 1.0),
            ("maximum_preflight_reward", 0.0, 1.0),
            ("minimum_heldout_delta", -1.0, 1.0),
            ("confidence", 1.0e-12, 1.0 - 1.0e-12),
            ("maximum_family_regression", 0.0, 1.0),
            ("maximum_train_seconds", 0.0, 1.0e12),
            ("verifier_timeout", 1.0e-6, 1.0e6),
        ):
            _finite(getattr(self, label), label, minimum, maximum)
        if self.clip_high != -1.0:
            _finite(self.clip_high, "clip_high", 0.0, 1000.0)
        if self.minimum_preflight_reward >= self.maximum_preflight_reward:
            raise ValueError(
                "minimum_preflight_reward must be below maximum_preflight_reward"
            )
        if not isinstance(self.require_confidence, bool) or not isinstance(
            self.use_ema, bool
        ):
            raise TypeError("boolean configuration fields must be booleans")
        if self.device != "cuda" and not (
            isinstance(self.device, str)
            and self.device.startswith("cuda:")
            and self.device[5:].isascii()
            and self.device[5:].isdigit()
        ):
            raise ValueError("device must be cuda or cuda:N")
        if self.rollout_devices and self.rollout_devices[0] != self.device:
            raise ValueError("first rollout device must match device")


__all__ = ["RLVRTrainConfig"]
