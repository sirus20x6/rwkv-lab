from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any

import torch

from .resolved import resolved_component_parts


class NormalizationImplementation(str, Enum):
    LAYER_NORM_V1 = "rwkv_lab.normalization.layer_norm.v1"


@dataclass(frozen=True, slots=True)
class LayerNormConfiguration:
    epsilon: float = 1.0e-5

    def __post_init__(self) -> None:
        if (
            isinstance(self.epsilon, bool)
            or not isinstance(self.epsilon, (int, float))
            or not math.isfinite(self.epsilon)
            or not 1.0e-12 <= self.epsilon <= 1.0
        ):
            raise ValueError("LayerNorm epsilon must be finite in [1e-12, 1]")

    @classmethod
    def from_resolved(
        cls, configuration: Mapping[str, Any]
    ) -> LayerNormConfiguration:
        if set(configuration) != {"epsilon"}:
            raise ValueError(
                "resolved LayerNorm configuration has missing or unknown fields"
            )
        return cls(**configuration)


class LayerNormFactory:
    """Construct affine LayerNorm sites without owning their model parameters."""

    def __init__(self, configuration: LayerNormConfiguration) -> None:
        self.configuration = configuration

    def __call__(self, normalized_shape: int | tuple[int, ...]) -> torch.nn.LayerNorm:
        if isinstance(normalized_shape, bool) or not isinstance(
            normalized_shape, (int, tuple)
        ):
            raise TypeError("LayerNorm shape must be an integer or integer tuple")
        return torch.nn.LayerNorm(
            normalized_shape,
            eps=float(self.configuration.epsilon),
            elementwise_affine=True,
            bias=True,
        )

    def state_dict(self) -> dict[str, object]:
        return {}

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        if state:
            raise ValueError("LayerNorm factory state must be empty")


def build_registered_normalization(
    implementation: NormalizationImplementation,
    configuration: LayerNormConfiguration,
) -> LayerNormFactory:
    if implementation is not NormalizationImplementation.LAYER_NORM_V1:
        raise ValueError(f"unsupported normalization implementation: {implementation!r}")
    if not isinstance(configuration, LayerNormConfiguration):
        raise TypeError("LayerNorm requires its typed configuration")
    return LayerNormFactory(configuration)


def normalization_from_resolved_component(
    component: Mapping[str, Any],
) -> LayerNormFactory:
    implementation, configuration = resolved_component_parts(
        component, "normalization"
    )
    try:
        selected = NormalizationImplementation(implementation)
    except ValueError as error:
        raise ValueError(
            "resolved normalization implementation is not allowlisted"
        ) from error
    return build_registered_normalization(
        selected, LayerNormConfiguration.from_resolved(configuration)
    )


__all__ = [
    "LayerNormConfiguration",
    "LayerNormFactory",
    "NormalizationImplementation",
    "build_registered_normalization",
    "normalization_from_resolved_component",
]
