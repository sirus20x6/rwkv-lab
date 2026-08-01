"""Closed tensor-runtime implementations for TrainVM training components.

TrainVM owns component keys, reflected schemas, family compatibility, and the
content-addressed composition lock.  This module owns only the small PyTorch
construction boundary that cannot live in the C++ control plane.  Symbolic
implementation IDs are closed enums; they are never import strings.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any

import torch

from rwkv_lab.mage_flow_optimizations import FP32MasterAdamW
from rwkv_lab.training_parameter_routing import (
    ParameterRoute,
    ParameterRoutingResult,
    route_trainable_parameters,
)


class OptimizerImplementation(str, Enum):
    TORCH_ADAMW_V1 = "rwkv_lab.optimizer.torch_adamw.v1"
    FP32_MASTER_ADAMW_V1 = "rwkv_lab.optimizer.fp32_master_adamw.v1"


class ScheduleImplementation(str, Enum):
    LINEAR_WARMUP_COSINE_V1 = "rwkv_lab.schedule.linear_warmup_cosine.v1"


class ParameterRouterImplementation(str, Enum):
    MAGEFLOW_APPEARANCE_EXPERT_V1 = (
        "rwkv_lab.parameter_router.mageflow_appearance_expert.v1"
    )
    MAGEFLOW_TERMINAL_EXPERT_V1 = (
        "rwkv_lab.parameter_router.mageflow_terminal_expert.v1"
    )


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
    def from_resolved(cls, configuration: Mapping[str, Any]) -> AdamWConfiguration:
        expected = {
            "learning_rate",
            "beta1",
            "beta2",
            "epsilon",
            "weight_decay",
            "foreach",
        }
        if set(configuration) != expected:
            raise ValueError(
                "resolved AdamW configuration has missing or unknown fields"
            )
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
    ) -> LinearWarmupCosineConfiguration:
        if set(configuration) != {"warmup_steps", "max_steps", "minimum_ratio"}:
            raise ValueError(
                "resolved schedule configuration has missing or unknown fields"
            )
        return cls(**configuration)


@dataclass(frozen=True, slots=True)
class AppearanceExpertRoutingConfiguration:
    shared_backbone_multiplier: float = 0.5

    def __post_init__(self) -> None:
        _validate_positive_multiplier(
            self.shared_backbone_multiplier, "shared_backbone_multiplier"
        )

    @classmethod
    def from_resolved(
        cls, configuration: Mapping[str, Any]
    ) -> AppearanceExpertRoutingConfiguration:
        if set(configuration) != {"shared_backbone_multiplier"}:
            raise ValueError(
                "resolved appearance routing configuration has missing or unknown fields"
            )
        return cls(**configuration)


@dataclass(frozen=True, slots=True)
class TerminalExpertRoutingConfiguration:
    shared_backbone_multiplier: float = 0.5
    repa_projection_multiplier: float = 1.0

    def __post_init__(self) -> None:
        _validate_positive_multiplier(
            self.shared_backbone_multiplier, "shared_backbone_multiplier"
        )
        _validate_positive_multiplier(
            self.repa_projection_multiplier, "repa_projection_multiplier"
        )

    @classmethod
    def from_resolved(
        cls, configuration: Mapping[str, Any]
    ) -> TerminalExpertRoutingConfiguration:
        if set(configuration) != {
            "shared_backbone_multiplier",
            "repa_projection_multiplier",
        }:
            raise ValueError(
                "resolved terminal routing configuration has missing or unknown fields"
            )
        return cls(**configuration)


def _validate_positive_multiplier(value: Any, name: str) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ValueError(f"{name} must be positive and finite")


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


def build_registered_parameter_routing(
    implementation: ParameterRouterImplementation,
    named_parameters: Iterable[tuple[str, torch.nn.Parameter]],
    role_parameter_ids: Mapping[str, frozenset[int]],
    *,
    base_learning_rate: float,
    configuration: (
        AppearanceExpertRoutingConfiguration | TerminalExpertRoutingConfiguration
    ),
) -> ParameterRoutingResult:
    """Resolve model-owned roles through one closed routing implementation."""

    named = list(named_parameters)
    all_parameter_ids = frozenset(id(parameter) for _, parameter in named)
    if implementation is ParameterRouterImplementation.MAGEFLOW_APPEARANCE_EXPERT_V1:
        if not isinstance(configuration, AppearanceExpertRoutingConfiguration):
            raise TypeError(
                "appearance router requires appearance routing configuration"
            )
        if set(role_parameter_ids) != {"expert"}:
            raise ValueError(
                "appearance router requires exactly the expert ownership role"
            )
        expert_ids = role_parameter_ids["expert"]
        routes = (
            ParameterRoute("experts", 1.0, expert_ids),
            ParameterRoute(
                "shared_backbone",
                configuration.shared_backbone_multiplier,
                all_parameter_ids - expert_ids,
            ),
        )
    elif implementation is ParameterRouterImplementation.MAGEFLOW_TERMINAL_EXPERT_V1:
        if not isinstance(configuration, TerminalExpertRoutingConfiguration):
            raise TypeError("terminal router requires terminal routing configuration")
        if set(role_parameter_ids) != {"expert", "repa"}:
            raise ValueError(
                "terminal router requires exactly expert and REPA ownership roles"
            )
        expert_ids = role_parameter_ids["expert"]
        repa_ids = role_parameter_ids["repa"]
        routes = (
            ParameterRoute("terminal_expert", 1.0, expert_ids, required=True),
            ParameterRoute(
                "shared_backbone",
                configuration.shared_backbone_multiplier,
                all_parameter_ids - expert_ids - repa_ids,
            ),
            ParameterRoute(
                "vae_repa_projection",
                configuration.repa_projection_multiplier,
                repa_ids,
                required=bool(repa_ids),
            ),
        )
    else:
        raise ValueError(
            f"unsupported parameter-router implementation: {implementation!r}"
        )
    return route_trainable_parameters(
        named, routes, base_learning_rate=base_learning_rate
    )


def _resolved_component_parts(
    component: Mapping[str, Any], expected_category: str
) -> tuple[str, Mapping[str, Any]]:
    if set(component) != {"configuration", "descriptor", "descriptor_digest"}:
        raise ValueError("resolved component envelope has missing or unknown fields")
    descriptor = component["descriptor"]
    configuration = component["configuration"]
    if not isinstance(descriptor, Mapping) or not isinstance(configuration, Mapping):
        raise TypeError(
            "resolved component descriptor and configuration must be objects"
        )
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
        raise ValueError(
            "resolved optimizer implementation is not allowlisted"
        ) from error
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
        raise ValueError(
            "resolved schedule implementation is not allowlisted"
        ) from error
    return build_registered_schedule(
        selected,
        optimizer,
        LinearWarmupCosineConfiguration.from_resolved(configuration),
    )


def parameter_routing_from_resolved_component(
    component: Mapping[str, Any],
    named_parameters: Iterable[tuple[str, torch.nn.Parameter]],
    role_parameter_ids: Mapping[str, frozenset[int]],
    *,
    base_learning_rate: float,
) -> ParameterRoutingResult:
    implementation, configuration = _resolved_component_parts(
        component, "parameter_router"
    )
    try:
        selected = ParameterRouterImplementation(implementation)
    except ValueError as error:
        raise ValueError(
            "resolved parameter-router implementation is not allowlisted"
        ) from error
    typed_configuration: (
        AppearanceExpertRoutingConfiguration | TerminalExpertRoutingConfiguration
    )
    if selected is ParameterRouterImplementation.MAGEFLOW_APPEARANCE_EXPERT_V1:
        typed_configuration = AppearanceExpertRoutingConfiguration.from_resolved(
            configuration
        )
    else:
        typed_configuration = TerminalExpertRoutingConfiguration.from_resolved(
            configuration
        )
    return build_registered_parameter_routing(
        selected,
        named_parameters,
        role_parameter_ids,
        base_learning_rate=base_learning_rate,
        configuration=typed_configuration,
    )


def supported_implementation_ids() -> frozenset[str]:
    return frozenset(
        implementation.value
        for implementation in (
            *OptimizerImplementation,
            *ScheduleImplementation,
            *ParameterRouterImplementation,
        )
    )


def supported_worker_capabilities() -> frozenset[str]:
    return frozenset(
        {
            "optimizer.torch_adamw.v1",
            "optimizer.fp32_master_adamw.v1",
            "schedule.linear_warmup_cosine.v1",
            "parameter_router.mageflow_appearance_expert.v1",
            "parameter_router.mageflow_terminal_expert.v1",
        }
    )


__all__ = [
    "AdamWConfiguration",
    "AppearanceExpertRoutingConfiguration",
    "LinearWarmupCosineConfiguration",
    "OptimizerImplementation",
    "ParameterRouterImplementation",
    "ScheduleImplementation",
    "TerminalExpertRoutingConfiguration",
    "build_registered_optimizer",
    "build_registered_parameter_routing",
    "build_registered_schedule",
    "linear_warmup_cosine_multiplier",
    "optimizer_from_resolved_component",
    "parameter_routing_from_resolved_component",
    "schedule_from_resolved_component",
    "supported_implementation_ids",
    "supported_worker_capabilities",
]
