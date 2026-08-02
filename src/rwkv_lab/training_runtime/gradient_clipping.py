from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any

import torch

from .resolved import resolved_component_parts


class GradientClippingImplementation(str, Enum):
    GLOBAL_NORM_V1 = "rwkv_lab.gradient_clipping.global_norm.v1"


@dataclass(frozen=True, slots=True)
class GlobalNormClippingConfiguration:
    max_norm: float
    norm_type: float = 2.0
    error_if_nonfinite: bool = False

    def __post_init__(self) -> None:
        for name, value in (("max_norm", self.max_norm), ("norm_type", self.norm_type)):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"{name} must be positive and finite")
        if not isinstance(self.error_if_nonfinite, bool):
            raise TypeError("error_if_nonfinite must be boolean")

    @classmethod
    def from_resolved(
        cls, configuration: Mapping[str, Any]
    ) -> GlobalNormClippingConfiguration:
        if set(configuration) != {"max_norm", "norm_type", "error_if_nonfinite"}:
            raise ValueError(
                "resolved global-norm clipping configuration has missing or unknown fields"
            )
        return cls(**configuration)


def build_registered_gradient_clipping(
    implementation: GradientClippingImplementation,
    parameters: Iterable[torch.Tensor],
    configuration: GlobalNormClippingConfiguration,
) -> torch.Tensor:
    """Apply one allowlisted clipping policy to one explicit parameter group."""

    if implementation is not GradientClippingImplementation.GLOBAL_NORM_V1:
        raise ValueError(
            f"unsupported gradient-clipping implementation: {implementation!r}"
        )
    if not isinstance(configuration, GlobalNormClippingConfiguration):
        raise TypeError("global-norm clipping requires its typed configuration")
    return torch.nn.utils.clip_grad_norm_(
        tuple(parameters),
        max_norm=configuration.max_norm,
        norm_type=configuration.norm_type,
        error_if_nonfinite=configuration.error_if_nonfinite,
    )


def gradient_clipping_from_resolved_component(
    component: Mapping[str, Any], parameters: Iterable[torch.Tensor]
) -> torch.Tensor:
    implementation, configuration = resolved_component_parts(
        component, "gradient_clipping"
    )
    try:
        selected = GradientClippingImplementation(implementation)
    except ValueError as error:
        raise ValueError(
            "resolved gradient-clipping implementation is not allowlisted"
        ) from error
    return build_registered_gradient_clipping(
        selected,
        parameters,
        GlobalNormClippingConfiguration.from_resolved(configuration),
    )


__all__ = [
    "GlobalNormClippingConfiguration",
    "GradientClippingImplementation",
    "build_registered_gradient_clipping",
    "gradient_clipping_from_resolved_component",
]
