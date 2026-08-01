"""Closed tensor-runtime implementations for TrainVM training components.

TrainVM owns component keys, reflected schemas, family compatibility, and the
content-addressed composition lock.  This module owns only the small PyTorch
construction boundary that cannot live in the C++ control plane.  Symbolic
implementation IDs are closed enums; they are never import strings.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable, Mapping

import torch

from rwkv_lab.mage_flow_optimizations import FP32MasterAdamW


class OptimizerImplementation(str, Enum):
    TORCH_ADAMW_V1 = "rwkv_lab.optimizer.torch_adamw.v1"
    FP32_MASTER_ADAMW_V1 = "rwkv_lab.optimizer.fp32_master_adamw.v1"


class ScheduleImplementation(str, Enum):
    LINEAR_WARMUP_COSINE_V1 = "rwkv_lab.schedule.linear_warmup_cosine.v1"


@dataclass(frozen=True, slots=True)
class AdamWConfiguration:
    learning_rate: float
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1.0e-8
    weight_decay: float = 0.01
    foreach: bool = True

    def __post_init__(self) -> None:
        if any(
            isinstance(value, bool) or not isinstance(value, (int, float))
            for value in (
                self.learning_rate,
                self.beta1,
                self.beta2,
                self.epsilon,
                self.weight_decay,
            )
        ):
            raise TypeError("AdamW numeric fields must be numbers, not booleans")
        numeric = (
            self.learning_rate,
            self.beta1,
            self.beta2,
            self.epsilon,
            self.weight_decay,
        )
        if not all(math.isfinite(value) for value in numeric):
            raise ValueError("AdamW configuration must be finite")
        if self.learning_rate <= 0:
            raise ValueError("AdamW learning_rate must be positive")
        if not 0 <= self.beta1 < 1 or not 0 <= self.beta2 < 1:
            raise ValueError("AdamW beta values must be in [0, 1)")
        if self.epsilon <= 0 or self.weight_decay < 0:
            raise ValueError("AdamW epsilon must be positive and decay nonnegative")
        if not isinstance(self.foreach, bool):
            raise TypeError("AdamW foreach must be boolean")

    @classmethod
    def from_resolved(cls, configuration: Mapping[str, Any]) -> "AdamWConfiguration":
        expected = {
            "learning_rate",
            "beta1",
            "beta2",
            "epsilon",
            "weight_decay",
            "foreach",
        }
        if set(configuration) != expected:
            raise ValueError("resolved AdamW configuration has missing or unknown fields")
        return cls(**configuration)


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
    ) -> "LinearWarmupCosineConfiguration":
        if set(configuration) != {"warmup_steps", "max_steps", "minimum_ratio"}:
            raise ValueError("resolved schedule configuration has missing or unknown fields")
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
    return configuration.minimum_ratio + (
        1.0 - configuration.minimum_ratio
    ) * 0.5 * (1.0 + math.cos(math.pi * progress))


def build_registered_optimizer(
    implementation: OptimizerImplementation,
    parameters: Iterable[torch.nn.Parameter] | Iterable[Mapping[str, Any]],
    configuration: AdamWConfiguration,
) -> torch.optim.Optimizer:
    """Construct one allowlisted optimizer implementation from typed values."""

    kwargs = {
        "lr": configuration.learning_rate,
        "betas": (configuration.beta1, configuration.beta2),
        "eps": configuration.epsilon,
        "weight_decay": configuration.weight_decay,
        "foreach": configuration.foreach,
    }
    if implementation is OptimizerImplementation.TORCH_ADAMW_V1:
        return torch.optim.AdamW(parameters, **kwargs)
    if implementation is OptimizerImplementation.FP32_MASTER_ADAMW_V1:
        return FP32MasterAdamW(parameters, **kwargs)
    raise ValueError(f"unsupported optimizer implementation: {implementation!r}")


def build_registered_schedule(
    implementation: ScheduleImplementation,
    optimizer: torch.optim.Optimizer,
    configuration: LinearWarmupCosineConfiguration,
) -> torch.optim.lr_scheduler.LRScheduler:
    """Construct one allowlisted schedule over the optimizer-step domain."""

    if implementation is not ScheduleImplementation.LINEAR_WARMUP_COSINE_V1:
        raise ValueError(f"unsupported schedule implementation: {implementation!r}")
    return torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: linear_warmup_cosine_multiplier(step, configuration),
    )


def _resolved_component_parts(
    component: Mapping[str, Any], expected_category: str
) -> tuple[str, Mapping[str, Any]]:
    if set(component) != {"configuration", "descriptor", "descriptor_digest"}:
        raise ValueError("resolved component envelope has missing or unknown fields")
    descriptor = component["descriptor"]
    configuration = component["configuration"]
    if not isinstance(descriptor, Mapping) or not isinstance(configuration, Mapping):
        raise TypeError("resolved component descriptor and configuration must be objects")
    key = descriptor.get("key")
    implementation = descriptor.get("implementation")
    if (
        not isinstance(key, Mapping)
        or key.get("category") != expected_category
        or not isinstance(implementation, str)
    ):
        raise ValueError("resolved component category or implementation is invalid")
    return implementation, configuration


def optimizer_from_resolved_component(
    component: Mapping[str, Any],
    parameters: Iterable[torch.nn.Parameter] | Iterable[Mapping[str, Any]],
) -> torch.optim.Optimizer:
    implementation, configuration = _resolved_component_parts(component, "optimizer")
    try:
        selected = OptimizerImplementation(implementation)
    except ValueError as error:
        raise ValueError("resolved optimizer implementation is not allowlisted") from error
    return build_registered_optimizer(
        selected, parameters, AdamWConfiguration.from_resolved(configuration)
    )


def schedule_from_resolved_component(
    component: Mapping[str, Any], optimizer: torch.optim.Optimizer
) -> torch.optim.lr_scheduler.LRScheduler:
    implementation, configuration = _resolved_component_parts(
        component, "learning_rate_schedule"
    )
    try:
        selected = ScheduleImplementation(implementation)
    except ValueError as error:
        raise ValueError("resolved schedule implementation is not allowlisted") from error
    return build_registered_schedule(
        selected,
        optimizer,
        LinearWarmupCosineConfiguration.from_resolved(configuration),
    )


def supported_implementation_ids() -> frozenset[str]:
    return frozenset(
        implementation.value
        for implementation in (*OptimizerImplementation, *ScheduleImplementation)
    )


def supported_worker_capabilities() -> frozenset[str]:
    return frozenset(
        {
            "optimizer.torch_adamw.v1",
            "optimizer.fp32_master_adamw.v1",
            "schedule.linear_warmup_cosine.v1",
        }
    )


__all__ = [
    "AdamWConfiguration",
    "LinearWarmupCosineConfiguration",
    "OptimizerImplementation",
    "ScheduleImplementation",
    "build_registered_optimizer",
    "build_registered_schedule",
    "linear_warmup_cosine_multiplier",
    "optimizer_from_resolved_component",
    "schedule_from_resolved_component",
    "supported_implementation_ids",
    "supported_worker_capabilities",
]
