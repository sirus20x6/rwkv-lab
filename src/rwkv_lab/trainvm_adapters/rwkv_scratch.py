"""Closed typed configuration for the baseline scratch-RWKV TrainVM adapter."""

from __future__ import annotations

import math
from dataclasses import dataclass


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
        or not math.isfinite(value)
        or not minimum <= value <= maximum
    ):
        raise ValueError(f"{label} must be finite in [{minimum}, {maximum}]")
    return float(value)


@dataclass(frozen=True, slots=True)
class RWKVScratchTrainConfig:
    """Intentionally small v1 surface for reproducible native scratch training.

    Research levers remain separate adapter versions until their topology,
    optimizer state, and checkpoint identity are represented declaratively.
    """

    data: str
    output_dir: str
    steps: int
    d_model: int = 512
    n_layers: int = 6
    head_size: int = 64
    sequence_length: int = 512
    batch_size: int = 16
    gradient_accumulation_steps: int = 1
    learning_rate: float = 6.0e-4
    weight_decay: float = 0.1
    max_gradient_norm: float = 1.0
    warmup_steps: int = 100
    minimum_learning_rate: float = 0.0
    cooldown_fraction: float = 0.2
    cooldown_power: float = 2.0
    validation_windows: int = 40
    eval_every_steps: int = 50
    log_every_steps: int = 10
    seed: int = 0
    resume: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.data, str) or not self.data:
            raise ValueError("data must be a nonempty path")
        if not isinstance(self.output_dir, str) or not self.output_dir:
            raise ValueError("output_dir must be a nonempty path")
        if self.resume is not None and (
            not isinstance(self.resume, str) or not self.resume
        ):
            raise ValueError("resume must be a nonempty path when present")
        _integer(self.steps, "steps", 1, 1_000_000_000)
        _integer(self.d_model, "d_model", 64, 65_536)
        _integer(self.n_layers, "n_layers", 1, 4_096)
        _integer(self.head_size, "head_size", 16, 1_024)
        if self.d_model % self.head_size:
            raise ValueError("d_model must be divisible by head_size")
        _integer(self.sequence_length, "sequence_length", 2, 1_048_576)
        _integer(self.batch_size, "batch_size", 1, 1_048_576)
        _integer(
            self.gradient_accumulation_steps,
            "gradient_accumulation_steps",
            1,
            65_536,
        )
        learning_rate = _finite(
            self.learning_rate, "learning_rate", 1.0e-12, 1.0
        )
        _finite(self.weight_decay, "weight_decay", 0.0, 10.0)
        _finite(self.max_gradient_norm, "max_gradient_norm", 1.0e-12, 1.0e9)
        _integer(self.warmup_steps, "warmup_steps", 0, self.steps)
        minimum = _finite(
            self.minimum_learning_rate,
            "minimum_learning_rate",
            0.0,
            learning_rate,
        )
        if minimum > learning_rate:
            raise ValueError("minimum_learning_rate cannot exceed learning_rate")
        cooldown = _finite(
            self.cooldown_fraction, "cooldown_fraction", 0.0, 1.0
        )
        if cooldown == 0.0:
            raise ValueError("cooldown_fraction must be positive")
        _finite(self.cooldown_power, "cooldown_power", 1.0e-12, 64.0)
        _integer(self.validation_windows, "validation_windows", 1, 1_000_000)
        _integer(self.eval_every_steps, "eval_every_steps", 1, self.steps)
        _integer(self.log_every_steps, "log_every_steps", 1, self.steps)
        _integer(self.seed, "seed", 0, (1 << 63) - 1)

    def trainer_arguments(
        self,
        *,
        data: str,
        output_dir: str,
        checkpoint: str,
        resume: str | None,
    ) -> list[str]:
        values = {
            "--data": data,
            "--out": output_dir,
            "--save": checkpoint,
            "--steps": self.steps,
            "--d-model": self.d_model,
            "--n-layers": self.n_layers,
            "--head-size": self.head_size,
            "--seq-len": self.sequence_length,
            "--batch": self.batch_size,
            "--grad-accum": self.gradient_accumulation_steps,
            "--lr": self.learning_rate,
            "--weight-decay": self.weight_decay,
            "--grad-clip": self.max_gradient_norm,
            "--warmup": self.warmup_steps,
            "--powercool-min-lr": self.minimum_learning_rate,
            "--powercool-cooldown-fraction": self.cooldown_fraction,
            "--powercool-power": self.cooldown_power,
            "--val-windows": self.validation_windows,
            "--eval-every": self.eval_every_steps,
            "--log-every": self.log_every_steps,
            "--seed": self.seed,
        }
        arguments: list[str] = []
        for flag, value in values.items():
            arguments.extend((flag, str(value)))
        arguments.extend(("--optimizer", "adamw"))
        arguments.extend(("--lr-schedule", "powercool"))
        arguments.extend(("--distributed", "none"))
        if resume is not None:
            arguments.extend(("--resume", resume))
        return arguments


__all__ = ["RWKVScratchTrainConfig"]
