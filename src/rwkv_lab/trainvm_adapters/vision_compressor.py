"""Closed TrainVM configuration for canonical multi-teacher vision compression."""

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


@dataclass(frozen=True, slots=True)
class VisionTeacherCompressorConfig:
    train_manifest: str
    eval_manifest: str
    moon_cache: str
    fusion_cache: str
    output_dir: str
    moonvit_checkpoint: str
    siglip2_model: str
    dinov2_model: str
    sam_model: str
    steps: int = 6000
    batch_size: int = 8
    workers: int = 8
    learning_rate: float = 2.0e-4
    weight_decay: float = 0.05
    teacher_dropout: float = 0.15
    relational_weight: float = 0.2
    variance_weight: float = 0.05
    covariance_weight: float = 0.01
    diversity_weight: float = 0.05
    max_gradient_norm: float = 1.0
    eval_every: int = 250
    checkpoint_every: int = 250
    log_every: int = 10
    seed: int = 20260716
    init_from: str = ""
    device: str = "cuda"

    def __post_init__(self) -> None:
        for label in (
            "train_manifest",
            "eval_manifest",
            "moon_cache",
            "fusion_cache",
            "output_dir",
            "moonvit_checkpoint",
            "siglip2_model",
            "dinov2_model",
            "sam_model",
        ):
            _path(getattr(self, label), label)
        _path(self.init_from, "init_from", optional=True)
        _integer(self.steps, "steps", 1, 1_000_000_000)
        _integer(self.batch_size, "batch_size", 1, 1_048_576)
        _integer(self.workers, "workers", 0, 4096)
        _finite(self.learning_rate, "learning_rate", 1.0e-12, 1.0)
        _finite(self.weight_decay, "weight_decay", 0.0, 10.0)
        _finite(self.teacher_dropout, "teacher_dropout", 0.0, 1.0)
        if self.teacher_dropout >= 1.0:
            raise ValueError("teacher_dropout must be less than one")
        for label in (
            "relational_weight",
            "variance_weight",
            "covariance_weight",
            "diversity_weight",
        ):
            _finite(getattr(self, label), label, 0.0, 1.0e6)
        _finite(self.max_gradient_norm, "max_gradient_norm", 1.0e-12, 1.0e12)
        for label in ("eval_every", "checkpoint_every", "log_every"):
            _integer(getattr(self, label), label, 1, self.steps)
        _integer(self.seed, "seed", 0, (1 << 63) - 1)
        if self.device != "cuda" and not (
            isinstance(self.device, str)
            and self.device.startswith("cuda:")
            and self.device[5:].isascii()
            and self.device[5:].isdigit()
        ):
            raise ValueError("device must be cuda or cuda:N")


__all__ = ["VisionTeacherCompressorConfig"]
