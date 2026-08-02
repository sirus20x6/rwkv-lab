from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol

import torch

from rwkv_lab.training_components import rebase_learning_rate_schedule
from rwkv_lab.trainvm_worker import Scalar, WorkerControlRuntime


class MageFlowControlError(ValueError):
    pass


class _MutableMageFlowConfig(Protocol):
    learning_rate: float
    eval_every: int
    caption_dropout: float
    mixed_precision: str


_SUPPORTED_CONTROLS = frozenset(
    {"learning_rate", "eval_every", "caption_dropout", "mixed_precision"}
)


def _number(value: object, label: str, *, maximum: float | None = None) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
        or (maximum is not None and value >= maximum)
    ):
        bound = f" and less than {maximum}" if maximum is not None else ""
        raise MageFlowControlError(f"{label} must be finite, positive{bound}")
    return float(value)


def _dropout(value: object) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not 0 <= value < 1
    ):
        raise MageFlowControlError("caption_dropout must be finite in [0, 1)")
    return float(value)


def _positive_integer(value: object, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise MageFlowControlError(f"{label} must be a positive integer")
    return value


def _precision(value: object) -> str:
    if value not in {"bf16", "fp16"}:
        raise MageFlowControlError("mixed_precision is not supported")
    return str(value)


def _validated_values(values: Mapping[str, object]) -> dict[str, Scalar]:
    unknown = set(values) - _SUPPORTED_CONTROLS
    if unknown:
        raise MageFlowControlError("MageFlow control catalog contains an unknown key")
    validated: dict[str, Scalar] = {}
    for key, value in values.items():
        if key == "learning_rate":
            validated[key] = _number(value, key)
        elif key == "eval_every":
            validated[key] = _positive_integer(value, key)
        elif key == "caption_dropout":
            validated[key] = _dropout(value)
        else:
            validated[key] = _precision(value)
    return validated


def lower_initial_mageflow_controls(
    config: _MutableMageFlowConfig, runtime: WorkerControlRuntime
) -> None:
    """Overlay authority controls before model/optimizer construction."""

    initial = _validated_values(runtime.effective_values)
    for key, value in initial.items():
        setattr(config, key, value)


@dataclass(slots=True)
class MageFlowMutableControls:
    """Topology adapter for MageFlow controls; schedule math stays category-owned."""

    config: _MutableMageFlowConfig
    scheduler: torch.optim.lr_scheduler.LRScheduler
    runtime: WorkerControlRuntime

    def __post_init__(self) -> None:
        initial = _validated_values(self.runtime.effective_values)
        expected: dict[str, Scalar] = {
            "learning_rate": float(self.config.learning_rate),
            "eval_every": self.config.eval_every,
            "caption_dropout": float(self.config.caption_dropout),
            "mixed_precision": self.config.mixed_precision,
        }
        if any(expected[key] != value for key, value in initial.items()):
            raise MageFlowControlError(
                "authority controls disagree with MageFlow configuration"
            )

    def apply(
        self,
        effective: Mapping[str, Scalar],
        assignments: Mapping[str, Scalar],
    ) -> None:
        _validated_values(effective)
        selected = _validated_values(assignments)
        if "mixed_precision" in selected:
            raise MageFlowControlError(
                "mixed_precision requires a replacement MageFlow worker"
            )
        if (
            "caption_dropout" in selected
            and float(selected["caption_dropout"]) > 0
            and self.config.caption_dropout == 0
            and getattr(self.config, "encoder_cache_mode", "off") != "off"
        ):
            raise MageFlowControlError(
                "caption_dropout cannot enable an absent cached null condition"
            )

        # Validation above is complete before any mutation. Schedule rebasing
        # validates its whole optimizer state before changing scalar fields.
        if "learning_rate" in selected:
            new_learning_rate = float(selected["learning_rate"])
            rebase_learning_rate_schedule(
                self.scheduler,
                old_base_learning_rate=float(self.config.learning_rate),
                new_base_learning_rate=new_learning_rate,
            )
            self.config.learning_rate = new_learning_rate
        if "eval_every" in selected:
            self.config.eval_every = int(selected["eval_every"])
        if "caption_dropout" in selected:
            self.config.caption_dropout = float(selected["caption_dropout"])


__all__ = [
    "MageFlowControlError",
    "MageFlowMutableControls",
    "lower_initial_mageflow_controls",
]
