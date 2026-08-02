from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from functools import partial
from typing import Any

import torch

from rwkv_lab.powercool import powercool_lr

from .resolved import resolved_component_parts


class ScheduleImplementation(str, Enum):
    LINEAR_WARMUP_COSINE_V1 = "rwkv_lab.schedule.linear_warmup_cosine.v1"
    POWERCOOL_V1 = "rwkv_lab.schedule.powercool.v1"


@dataclass(frozen=True, slots=True)
class LinearWarmupCosineConfiguration:
    warmup_steps: int
    max_steps: int
    minimum_ratio: float = 0.1

    def __post_init__(self) -> None:
        if (
            not isinstance(self.warmup_steps, int)
            or isinstance(self.warmup_steps, bool)
            or self.warmup_steps < 0
        ):
            raise ValueError("warmup_steps must be a nonnegative integer")
        if (
            not isinstance(self.max_steps, int)
            or isinstance(self.max_steps, bool)
            or self.max_steps < 1
        ):
            raise ValueError("max_steps must be a positive integer")
        if (
            isinstance(self.minimum_ratio, bool)
            or not isinstance(self.minimum_ratio, (int, float))
            or not math.isfinite(self.minimum_ratio)
            or not 0 <= self.minimum_ratio <= 1
        ):
            raise ValueError("minimum_ratio must be finite and in [0, 1]")

    @classmethod
    def from_resolved(
        cls, configuration: Mapping[str, Any]
    ) -> LinearWarmupCosineConfiguration:
        if set(configuration) != {"warmup_steps", "max_steps", "minimum_ratio"}:
            raise ValueError(
                "resolved schedule configuration has missing or unknown fields"
            )
        return cls(**configuration)


@dataclass(frozen=True, slots=True)
class PowerCoolConfiguration:
    warmup_steps: int
    max_steps: int
    minimum_ratio: float = 0.0
    cooldown_fraction: float = 0.2
    power: float = 2.0

    def __post_init__(self) -> None:
        if (
            not isinstance(self.warmup_steps, int)
            or isinstance(self.warmup_steps, bool)
            or self.warmup_steps < 0
        ):
            raise ValueError("warmup_steps must be a nonnegative integer")
        if (
            not isinstance(self.max_steps, int)
            or isinstance(self.max_steps, bool)
            or self.max_steps < 1
        ):
            raise ValueError("max_steps must be a positive integer")
        if self.warmup_steps > self.max_steps:
            raise ValueError("warmup_steps cannot exceed max_steps")
        for name, value in (
            ("minimum_ratio", self.minimum_ratio),
            ("cooldown_fraction", self.cooldown_fraction),
            ("power", self.power),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise ValueError(f"{name} must be finite")
        if not 0 <= self.minimum_ratio <= 1:
            raise ValueError("minimum_ratio must be in [0, 1]")
        if not 0 < self.cooldown_fraction <= 1:
            raise ValueError("cooldown_fraction must be in (0, 1]")
        if self.power <= 0:
            raise ValueError("power must be positive")

    @classmethod
    def from_resolved(cls, configuration: Mapping[str, Any]) -> PowerCoolConfiguration:
        if set(configuration) != {
            "warmup_steps",
            "max_steps",
            "minimum_ratio",
            "cooldown_fraction",
            "power",
        }:
            raise ValueError(
                "resolved PowerCool configuration has missing or unknown fields"
            )
        return cls(**configuration)


def linear_warmup_cosine_multiplier(
    step: int, configuration: LinearWarmupCosineConfiguration
) -> float:
    """Pure optimizer-step schedule shared by all registered workers."""

    if not isinstance(step, int) or isinstance(step, bool) or step < 0:
        raise ValueError("schedule step must be a nonnegative integer")
    if configuration.warmup_steps and step < configuration.warmup_steps:
        return max(1.0e-8, step / float(configuration.warmup_steps))
    progress = (step - configuration.warmup_steps) / float(
        max(1, configuration.max_steps - configuration.warmup_steps)
    )
    progress = min(1.0, max(0.0, progress))
    return configuration.minimum_ratio + (1.0 - configuration.minimum_ratio) * 0.5 * (
        1.0 + math.cos(math.pi * progress)
    )


def powercool_multiplier(step: int, configuration: PowerCoolConfiguration) -> float:
    """Pure PowerCool multiplier over the zero-based optimizer-step domain."""

    return powercool_lr(
        step,
        peak_lr=1.0,
        min_lr=configuration.minimum_ratio,
        total_steps=configuration.max_steps,
        warmup_steps=configuration.warmup_steps,
        cooldown_fraction=configuration.cooldown_fraction,
        power=configuration.power,
    )


def build_registered_schedule(
    implementation: ScheduleImplementation,
    optimizer: torch.optim.Optimizer,
    configuration: LinearWarmupCosineConfiguration | PowerCoolConfiguration,
) -> torch.optim.lr_scheduler.LRScheduler:
    """Construct one allowlisted schedule over the optimizer-step domain."""

    if implementation is ScheduleImplementation.LINEAR_WARMUP_COSINE_V1:
        if not isinstance(configuration, LinearWarmupCosineConfiguration):
            raise TypeError("linear-warmup-cosine requires its typed configuration")
        multiplier = partial(
            linear_warmup_cosine_multiplier, configuration=configuration
        )
    elif implementation is ScheduleImplementation.POWERCOOL_V1:
        if not isinstance(configuration, PowerCoolConfiguration):
            raise TypeError("PowerCool requires its typed configuration")
        multiplier = partial(powercool_multiplier, configuration=configuration)
    else:
        raise ValueError(f"unsupported schedule implementation: {implementation!r}")
    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


def rebase_learning_rate_schedule(
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    *,
    old_base_learning_rate: float,
    new_base_learning_rate: float,
) -> None:
    """Atomically preserve schedule phase and parameter-group LR ratios."""

    for label, value in (
        ("old base learning rate", old_base_learning_rate),
        ("new base learning rate", new_base_learning_rate),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value <= 0
        ):
            raise ValueError(f"{label} must be finite and positive")
    groups = scheduler.optimizer.param_groups
    if len(scheduler.base_lrs) != len(groups) or not groups:
        raise ValueError("schedule and optimizer parameter groups disagree")
    last_lrs = scheduler.get_last_lr()
    if len(last_lrs) != len(groups):
        raise ValueError("schedule last-learning-rate state is inconsistent")
    for label, values in (
        ("base", scheduler.base_lrs),
        ("last", last_lrs),
    ):
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0
            for value in values
        ):
            raise ValueError(f"schedule {label} learning-rate state is invalid")
    for group in groups:
        if "initial_lr" not in group:
            raise ValueError("optimizer group has no schedule base learning rate")
        for field in ("lr", "initial_lr"):
            value = group[field]
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
            ):
                raise ValueError(f"optimizer group {field} is invalid")

    ratio = float(new_base_learning_rate / old_base_learning_rate)
    scheduler.base_lrs = [float(value * ratio) for value in scheduler.base_lrs]
    for group in groups:
        group["initial_lr"] = float(group["initial_lr"] * ratio)
        group["lr"] = float(group["lr"] * ratio)
    # get_last_lr() reports this checkpointed cache rather than reading the
    # optimizer groups, so move it in the same trainer callback.
    scheduler._last_lr = [float(value * ratio) for value in last_lrs]


def schedule_from_resolved_component(
    component: Mapping[str, Any], optimizer: torch.optim.Optimizer
) -> torch.optim.lr_scheduler.LRScheduler:
    selected, typed_configuration = schedule_configuration_from_resolved_component(
        component
    )
    return build_registered_schedule(selected, optimizer, typed_configuration)


def schedule_configuration_from_resolved_component(
    component: Mapping[str, Any],
) -> tuple[
    ScheduleImplementation,
    LinearWarmupCosineConfiguration | PowerCoolConfiguration,
]:
    implementation, configuration = resolved_component_parts(
        component, "learning_rate_schedule"
    )
    try:
        selected = ScheduleImplementation(implementation)
    except ValueError as error:
        raise ValueError(
            "resolved schedule implementation is not allowlisted"
        ) from error
    typed_configuration = (
        LinearWarmupCosineConfiguration.from_resolved(configuration)
        if selected is ScheduleImplementation.LINEAR_WARMUP_COSINE_V1
        else PowerCoolConfiguration.from_resolved(configuration)
    )
    return selected, typed_configuration
