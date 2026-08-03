from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any

import torch

from rwkv_lab.training_parameter_routing import (
    ParameterRoute,
    ParameterRoutingResult,
    route_trainable_parameters,
)

from .resolved import resolved_component_parts


class ParameterRouterImplementation(str, Enum):
    RWKV_MATRIX_OPTIMIZER_V1 = (
        "rwkv_lab.parameter_router.rwkv_matrix_optimizer.v1"
    )
    MAGEFLOW_FULL_BACKBONE_V1 = (
        "rwkv_lab.parameter_router.mageflow_full_backbone.v1"
    )
    MAGEFLOW_APPEARANCE_EXPERT_V1 = (
        "rwkv_lab.parameter_router.mageflow_appearance_expert.v1"
    )
    MAGEFLOW_TERMINAL_EXPERT_V1 = (
        "rwkv_lab.parameter_router.mageflow_terminal_expert.v1"
    )


def _validate_positive_multiplier(value: Any, name: str) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ValueError(f"{name} must be positive and finite")


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
class FullBackboneRoutingConfiguration:
    """The full NR-MMDiT path owns every trainable transformer parameter."""

    @classmethod
    def from_resolved(
        cls, configuration: Mapping[str, Any]
    ) -> FullBackboneRoutingConfiguration:
        if configuration:
            raise ValueError(
                "resolved full-backbone routing configuration has unknown fields"
            )
        return cls()


@dataclass(frozen=True, slots=True)
class RWKVMatrixOptimizerRoutingConfiguration:
    fallback_multiplier: float = 0.1

    def __post_init__(self) -> None:
        _validate_positive_multiplier(
            self.fallback_multiplier, "fallback_multiplier"
        )
        if self.fallback_multiplier > 1000:
            raise ValueError("fallback_multiplier exceeds its declared bound")

    @classmethod
    def from_resolved(
        cls, configuration: Mapping[str, Any]
    ) -> RWKVMatrixOptimizerRoutingConfiguration:
        if set(configuration) != {"fallback_multiplier"}:
            raise ValueError(
                "resolved RWKV matrix routing configuration has missing or unknown fields"
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


def build_registered_parameter_routing(
    implementation: ParameterRouterImplementation,
    named_parameters: Iterable[tuple[str, torch.nn.Parameter]],
    role_parameter_ids: Mapping[str, frozenset[int]],
    *,
    base_learning_rate: float,
    configuration: (
        AppearanceExpertRoutingConfiguration
        | FullBackboneRoutingConfiguration
        | RWKVMatrixOptimizerRoutingConfiguration
        | TerminalExpertRoutingConfiguration
    ),
) -> ParameterRoutingResult:
    """Resolve model-owned roles through one closed routing implementation."""

    named = list(named_parameters)
    all_parameter_ids = frozenset(id(parameter) for _, parameter in named)
    if implementation is ParameterRouterImplementation.RWKV_MATRIX_OPTIMIZER_V1:
        if not isinstance(configuration, RWKVMatrixOptimizerRoutingConfiguration):
            raise TypeError(
                "RWKV matrix router requires its typed routing configuration"
            )
        if set(role_parameter_ids) != {"muon"}:
            raise ValueError("RWKV matrix router requires exactly the muon ownership role")
        muon_ids = role_parameter_ids["muon"]
        routes = (
            ParameterRoute("muon", 1.0, muon_ids, required=True),
            ParameterRoute(
                "adam_fallback",
                configuration.fallback_multiplier,
                all_parameter_ids - muon_ids,
                required=True,
            ),
        )
    elif implementation is ParameterRouterImplementation.MAGEFLOW_FULL_BACKBONE_V1:
        if not isinstance(configuration, FullBackboneRoutingConfiguration):
            raise TypeError(
                "full-backbone router requires full-backbone routing configuration"
            )
        if role_parameter_ids:
            raise ValueError("full-backbone router does not accept partial ownership roles")
        routes = (ParameterRoute("full_backbone", 1.0, all_parameter_ids, required=True),)
    elif implementation is ParameterRouterImplementation.MAGEFLOW_APPEARANCE_EXPERT_V1:
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


def parameter_routing_from_resolved_component(
    component: Mapping[str, Any],
    named_parameters: Iterable[tuple[str, torch.nn.Parameter]],
    role_parameter_ids: Mapping[str, frozenset[int]],
    *,
    base_learning_rate: float,
) -> ParameterRoutingResult:
    implementation, configuration = resolved_component_parts(
        component, "parameter_router"
    )
    try:
        selected = ParameterRouterImplementation(implementation)
    except ValueError as error:
        raise ValueError(
            "resolved parameter-router implementation is not allowlisted"
        ) from error
    typed_configuration: (
        AppearanceExpertRoutingConfiguration
        | FullBackboneRoutingConfiguration
        | RWKVMatrixOptimizerRoutingConfiguration
        | TerminalExpertRoutingConfiguration
    )
    if selected is ParameterRouterImplementation.RWKV_MATRIX_OPTIMIZER_V1:
        typed_configuration = RWKVMatrixOptimizerRoutingConfiguration.from_resolved(
            configuration
        )
    elif selected is ParameterRouterImplementation.MAGEFLOW_FULL_BACKBONE_V1:
        typed_configuration = FullBackboneRoutingConfiguration.from_resolved(
            configuration
        )
    elif selected is ParameterRouterImplementation.MAGEFLOW_APPEARANCE_EXPERT_V1:
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
