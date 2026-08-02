"""Closed TrainVM configuration for the native RWKV vision-head arm."""

from __future__ import annotations

import math
from dataclasses import dataclass


def _path(value: object, label: str) -> str:
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


@dataclass(frozen=True, slots=True)
class VisionNativeHeadConfig:
    baseline_checkpoint: str
    compressor_checkpoint: str
    train_manifest: str
    eval_manifest: str
    moon_cache: str
    fusion_cache: str
    moonvit_checkpoint: str
    siglip2_model: str
    dinov2_model: str
    sam_model: str
    vocab: str
    output_dir: str
    steps: int = 5000
    batch_size: int = 8
    workers: int = 8
    learning_rate: float = 2.0e-4
    weight_decay: float = 0.01
    max_gradient_norm: float = 1.0
    eval_every: int = 100
    eval_examples: int = 64
    checkpoint_every: int = 50
    seed: int = 20260716
    device: str = "cuda"

    def __post_init__(self) -> None:
        for label in (
            "baseline_checkpoint",
            "compressor_checkpoint",
            "train_manifest",
            "eval_manifest",
            "moon_cache",
            "fusion_cache",
            "moonvit_checkpoint",
            "siglip2_model",
            "dinov2_model",
            "sam_model",
            "vocab",
            "output_dir",
        ):
            _path(getattr(self, label), label)
        for label, minimum in (
            ("steps", 1),
            ("batch_size", 1),
            ("workers", 0),
            ("eval_every", 1),
            ("eval_examples", 1),
            ("checkpoint_every", 1),
            ("seed", 0),
        ):
            _integer(getattr(self, label), label, minimum, 1_000_000_000)
        _finite(self.learning_rate, "learning_rate", 1.0e-12, 1.0)
        _finite(self.weight_decay, "weight_decay", 0.0, 10.0)
        _finite(self.max_gradient_norm, "max_gradient_norm", 1.0e-12, 1.0e12)
        if self.eval_every > self.steps or self.checkpoint_every > self.steps:
            raise ValueError("eval/checkpoint cadence cannot exceed steps")
        if self.device != "cuda":
            raise ValueError("vision native-head v1 requires device cuda")


__all__ = ["VisionNativeHeadConfig"]
