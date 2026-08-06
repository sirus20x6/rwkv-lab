from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any

from .resolved import resolved_component_parts


class EvaluatorImplementation(str, Enum):
    SCALAR_LOSS_V1 = "rwkv_lab.evaluator.scalar_loss.v1"


@dataclass(frozen=True, slots=True)
class ScalarLossEvaluatorConfiguration:
    split_slot: str
    metrics: tuple[str, ...]
    reduction: str = "weighted_mean"
    maximum_examples: int = 0

    def __post_init__(self) -> None:
        if (
            not self.split_slot
            or not self.split_slot[0].isalpha()
            or any(
                not character.isascii()
                or not (character.isalnum() or character in {"_", "-", ".", ":"})
                for character in self.split_slot
            )
        ):
            raise ValueError("evaluator split slot must be a symbolic identity")
        if not self.metrics or tuple(sorted(set(self.metrics))) != self.metrics:
            raise ValueError("evaluator metrics must be a nonempty canonical set")
        if any(
            not isinstance(metric, str)
            or not metric
            or not metric[0].isascii()
            or not metric[0].isalpha()
            or any(
                not character.isascii()
                or not (character.isalnum() or character in {"_", "-", ".", ":"})
                for character in metric
            )
            for metric in self.metrics
        ):
            raise ValueError("evaluator metrics must be symbolic identities")
        if self.reduction not in {"mean", "sum", "weighted_mean"}:
            raise ValueError("scalar evaluator reduction is unsupported")
        if (
            not isinstance(self.maximum_examples, int)
            or isinstance(self.maximum_examples, bool)
            or not 0 <= self.maximum_examples <= 1_000_000_000
        ):
            raise ValueError("maximum_examples must be an integer in range")

    @classmethod
    def from_resolved(
        cls, configuration: Mapping[str, Any]
    ) -> ScalarLossEvaluatorConfiguration:
        if set(configuration) != {
            "split_slot",
            "metrics",
            "reduction",
            "maximum_examples",
        }:
            raise ValueError("resolved scalar evaluator configuration is inexact")
        return cls(
            split_slot=configuration["split_slot"],
            metrics=tuple(configuration["metrics"]),
            reduction=configuration["reduction"],
            maximum_examples=configuration["maximum_examples"],
        )


@dataclass(frozen=True, slots=True)
class ScalarLossEvaluator:
    configuration: ScalarLossEvaluatorConfiguration

    def reduce(self, values: Sequence[float], weights: Sequence[float] = ()) -> float:
        if not values or any(not math.isfinite(value) for value in values):
            raise ValueError("scalar evaluation requires finite nonempty results")
        if self.configuration.reduction == "sum":
            return float(sum(values))
        if self.configuration.reduction == "mean":
            return float(sum(values) / len(values))
        if len(weights) != len(values) or any(
            not math.isfinite(weight) or weight <= 0 for weight in weights
        ):
            raise ValueError(
                "weighted scalar evaluation requires positive aligned weights"
            )
        return float(
            sum(v * w for v, w in zip(values, weights, strict=True)) / sum(weights)
        )


def evaluator_from_resolved_component(
    component: Mapping[str, Any],
) -> ScalarLossEvaluator:
    implementation, configuration = resolved_component_parts(component, "evaluator")
    if implementation != EvaluatorImplementation.SCALAR_LOSS_V1:
        raise ValueError("resolved evaluator implementation is not allowlisted")
    return ScalarLossEvaluator(
        ScalarLossEvaluatorConfiguration.from_resolved(configuration)
    )


__all__ = [
    "EvaluatorImplementation",
    "ScalarLossEvaluator",
    "ScalarLossEvaluatorConfiguration",
    "evaluator_from_resolved_component",
]
