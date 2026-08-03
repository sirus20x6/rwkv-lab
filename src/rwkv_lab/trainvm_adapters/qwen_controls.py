from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any

import torch

from rwkv_lab.training_components import rebase_learning_rate_schedule
from rwkv_lab.trainvm_worker import Scalar, WorkerControlRuntime


class QwenControlError(ValueError):
    pass


_SUPPORTED_CONTROLS = frozenset({"learning_rate", "eval_every"})


def _validated(values: Mapping[str, object]) -> dict[str, Scalar]:
    if set(values) - _SUPPORTED_CONTROLS:
        raise QwenControlError("Qwen control catalog contains an unknown key")
    result: dict[str, Scalar] = {}
    for key, value in values.items():
        if key == "learning_rate":
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise QwenControlError("learning_rate must be finite and positive")
            result[key] = float(value)
        elif not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise QwenControlError("eval_every must be a positive integer")
        else:
            result[key] = value
    return result


def lower_initial_qwen_controls(config: Any, runtime: WorkerControlRuntime) -> Any:
    """Return a frozen config with authority controls applied before construction."""

    initial = _validated(runtime.effective_values)
    changes = dict(initial)
    if "learning_rate" in changes:
        ratio = float(changes["learning_rate"]) / float(config.learning_rate)
        changes["min_learning_rate"] = float(config.min_learning_rate * ratio)
    return replace(config, **changes)


@dataclass(slots=True)
class QwenMutableControls:
    scheduler: torch.optim.lr_scheduler.LRScheduler
    runtime: WorkerControlRuntime
    learning_rate: float
    eval_every: int
    constructed_base_learning_rate: float

    def __post_init__(self) -> None:
        initial = _validated(self.runtime.effective_values)
        expected: dict[str, Scalar] = {
            "learning_rate": float(self.learning_rate),
            "eval_every": self.eval_every,
        }
        if any(expected[key] != value for key, value in initial.items()):
            raise QwenControlError("authority controls disagree with Qwen configuration")
        if self.constructed_base_learning_rate != self.learning_rate:
            rebase_learning_rate_schedule(
                self.scheduler,
                old_base_learning_rate=self.constructed_base_learning_rate,
                new_base_learning_rate=self.learning_rate,
            )

    def apply(
        self,
        effective: Mapping[str, Scalar],
        assignments: Mapping[str, Scalar],
    ) -> None:
        _validated(effective)
        selected = _validated(assignments)
        if "learning_rate" in selected:
            new_learning_rate = float(selected["learning_rate"])
            rebase_learning_rate_schedule(
                self.scheduler,
                old_base_learning_rate=self.learning_rate,
                new_base_learning_rate=new_learning_rate,
            )
            self.learning_rate = new_learning_rate
        if "eval_every" in selected:
            self.eval_every = int(selected["eval_every"])


__all__ = [
    "QwenControlError",
    "QwenMutableControls",
    "lower_initial_qwen_controls",
]
