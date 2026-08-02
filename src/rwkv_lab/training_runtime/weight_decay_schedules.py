from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any

import torch

from .resolved import resolved_component_parts


class WeightDecayScheduleImplementation(str, Enum):
    CONSTANT_V1 = "rwkv_lab.weight_decay_schedule.constant.v1"


@dataclass(frozen=True, slots=True)
class ConstantWeightDecayConfiguration:
    weight_decay: float

    def __post_init__(self) -> None:
        if (
            isinstance(self.weight_decay, bool)
            or not isinstance(self.weight_decay, (int, float))
            or not math.isfinite(self.weight_decay)
            or self.weight_decay < 0
        ):
            raise ValueError("weight_decay must be finite and nonnegative")

    @classmethod
    def from_resolved(
        cls, configuration: Mapping[str, Any]
    ) -> ConstantWeightDecayConfiguration:
        if set(configuration) != {"weight_decay"}:
            raise ValueError(
                "resolved constant weight-decay configuration has missing or unknown fields"
            )
        return cls(**configuration)


class ConstantWeightDecaySchedule:
    """Stateless optimizer-step decay policy independent of optimizer math."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        configuration: ConstantWeightDecayConfiguration,
    ) -> None:
        self.optimizer = optimizer
        self.configuration = configuration
        self.step(0)

    def step(self, optimizer_step: int) -> None:
        if (
            not isinstance(optimizer_step, int)
            or isinstance(optimizer_step, bool)
            or optimizer_step < 0
        ):
            raise ValueError("weight-decay schedule step must be nonnegative")
        for group in self.optimizer.param_groups:
            multiplier = group.get("weight_decay_multiplier", 1.0)
            if (
                isinstance(multiplier, bool)
                or not isinstance(multiplier, (int, float))
                or not math.isfinite(multiplier)
                or multiplier < 0
            ):
                raise ValueError(
                    "optimizer weight_decay_multiplier must be finite and nonnegative"
                )
            group["weight_decay"] = self.configuration.weight_decay * multiplier

    def state_dict(self) -> dict[str, object]:
        return {}

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        if state:
            raise ValueError("constant weight-decay schedule state must be empty")


def build_registered_weight_decay_schedule(
    implementation: WeightDecayScheduleImplementation,
    optimizer: torch.optim.Optimizer,
    configuration: ConstantWeightDecayConfiguration,
) -> ConstantWeightDecaySchedule:
    if implementation is not WeightDecayScheduleImplementation.CONSTANT_V1:
        raise ValueError(
            f"unsupported weight-decay schedule implementation: {implementation!r}"
        )
    if not isinstance(configuration, ConstantWeightDecayConfiguration):
        raise TypeError("constant weight decay requires its typed configuration")
    return ConstantWeightDecaySchedule(optimizer, configuration)


def weight_decay_schedule_from_resolved_component(
    component: Mapping[str, Any], optimizer: torch.optim.Optimizer
) -> ConstantWeightDecaySchedule:
    implementation, configuration = resolved_component_parts(
        component, "weight_decay_schedule"
    )
    try:
        selected = WeightDecayScheduleImplementation(implementation)
    except ValueError as error:
        raise ValueError(
            "resolved weight-decay schedule implementation is not allowlisted"
        ) from error
    return build_registered_weight_decay_schedule(
        selected,
        optimizer,
        ConstantWeightDecayConfiguration.from_resolved(configuration),
    )


__all__ = [
    "ConstantWeightDecayConfiguration",
    "ConstantWeightDecaySchedule",
    "WeightDecayScheduleImplementation",
    "build_registered_weight_decay_schedule",
    "weight_decay_schedule_from_resolved_component",
]
