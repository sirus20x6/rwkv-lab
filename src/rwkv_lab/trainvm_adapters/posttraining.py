"""Closed TrainVM configuration for restart-only RWKV post-training.

The legacy command exposes a broad research surface.  This v1 contract keeps
the useful SFT, preference, and reward objectives while making every path and
resource-affecting option explicit and bounded.  Interrupted attempts restart
from their immutable base checkpoint and dataset; v1 intentionally makes no
checkpoint-resume claim.
"""

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
class RWKVPostTrainConfig:
    checkpoint: str
    data: str
    output_dir: str
    objective: str = "sft"
    adapter_name: str = "posttrain"
    rank: int = 16
    alpha: float = 32.0
    targets: tuple[str, ...] = ()
    steps: int = 100
    batch_size: int = 2
    learning_rate: float = 2.0e-4
    minimum_learning_rate_ratio: float = 0.1
    warmup_steps: int = 0
    weight_decay: float = 0.01
    max_gradient_norm: float = 1.0
    beta: float = 0.1
    gamma: float = 1.0
    max_length: int = 2048
    seed: int = 0
    device: str = "auto"
    template: str = ""
    eval_data: str = ""
    token_cache: str = ""
    max_train_tokens: int = 0
    packing: str = "audit"
    base_quantization: str = "none"
    quant_block_size: int = 64
    quant_backend: str = "auto"
    activation_offload: bool = False
    log_every: int = 10

    def __post_init__(self) -> None:
        for label in ("checkpoint", "data", "output_dir"):
            _path(getattr(self, label), label)
        for label in ("template", "eval_data", "token_cache"):
            _path(getattr(self, label), label, optional=True)
        if self.objective not in {
            "sft",
            "dpo",
            "kto",
            "orpo",
            "simpo",
            "reward",
            "prm",
        }:
            raise ValueError("objective is not a registered post-training objective")
        if (
            not isinstance(self.adapter_name, str)
            or not self.adapter_name
            or len(self.adapter_name.encode("utf-8")) > 128
        ):
            raise ValueError("adapter_name must be a bounded nonempty string")
        if not isinstance(self.targets, (tuple, list)):
            raise TypeError("targets must be a sequence")
        targets = tuple(self.targets)
        if (
            len(targets) > 64
            or len(targets) != len(set(targets))
            or any(
                not isinstance(value, str)
                or not value
                or len(value.encode("utf-8")) > 256
                for value in targets
            )
        ):
            raise ValueError("targets must be unique bounded nonempty strings")
        object.__setattr__(self, "targets", targets)
        _integer(self.rank, "rank", 1, 65_536)
        _finite(self.alpha, "alpha", 1.0e-12, 1.0e9)
        _integer(self.steps, "steps", 1, 1_000_000_000)
        _integer(self.batch_size, "batch_size", 1, 1_048_576)
        _finite(self.learning_rate, "learning_rate", 1.0e-12, 1.0)
        _finite(
            self.minimum_learning_rate_ratio,
            "minimum_learning_rate_ratio",
            0.0,
            1.0,
        )
        _integer(self.warmup_steps, "warmup_steps", 0, self.steps)
        _finite(self.weight_decay, "weight_decay", 0.0, 10.0)
        _finite(self.max_gradient_norm, "max_gradient_norm", 1.0e-12, 1.0e12)
        _finite(self.beta, "beta", 1.0e-12, 1.0e6)
        _finite(self.gamma, "gamma", 0.0, 1.0e6)
        _integer(self.max_length, "max_length", 2, 1_048_576)
        _integer(self.seed, "seed", 0, (1 << 63) - 1)
        if self.device not in {"auto", "cpu", "cuda"} and not (
            isinstance(self.device, str) and self.device.startswith("cuda:")
        ):
            raise ValueError("device must be auto, cpu, cuda, or cuda:N")
        _integer(self.max_train_tokens, "max_train_tokens", 0, (1 << 63) - 1)
        if self.packing not in {"off", "audit", "reset"}:
            raise ValueError("packing must be off, audit, or reset")
        if self.base_quantization not in {"none", "nf4"}:
            raise ValueError("base_quantization must be none or nf4")
        _integer(self.quant_block_size, "quant_block_size", 16, 65_536)
        if self.quant_backend not in {"auto", "portable", "torchao"}:
            raise ValueError("quant_backend must be auto, portable, or torchao")
        if not isinstance(self.activation_offload, bool):
            raise TypeError("activation_offload must be boolean")
        _integer(self.log_every, "log_every", 1, self.steps)


__all__ = ["RWKVPostTrainConfig"]
