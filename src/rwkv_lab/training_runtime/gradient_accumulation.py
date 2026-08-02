from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any

import torch

from .resolved import resolved_component_parts


class GradientAccumulationImplementation(str, Enum):
    FIXED_V1 = "rwkv_lab.gradient_accumulation.fixed.v1"


@dataclass(frozen=True, slots=True)
class FixedGradientAccumulationConfiguration:
    microbatches_per_optimizer_step: int

    def __post_init__(self) -> None:
        if (
            not isinstance(self.microbatches_per_optimizer_step, int)
            or isinstance(self.microbatches_per_optimizer_step, bool)
            or not 1 <= self.microbatches_per_optimizer_step <= 65_536
        ):
            raise ValueError(
                "microbatches_per_optimizer_step must be an integer in [1, 65536]"
            )

    @classmethod
    def from_resolved(
        cls, configuration: Mapping[str, Any]
    ) -> FixedGradientAccumulationConfiguration:
        if set(configuration) != {"microbatches_per_optimizer_step"}:
            raise ValueError(
                "resolved fixed accumulation configuration has missing or unknown fields"
            )
        return cls(**configuration)


class FixedGradientAccumulation:
    """One optimizer-step-local accumulation policy.

    The v1 state grade is deliberately stateless: supported consumers resume
    only at optimizer-step boundaries. A later safe-point-capable policy must
    persist its microbatch cursor and accumulated gradients explicitly.
    """

    def __init__(self, configuration: FixedGradientAccumulationConfiguration) -> None:
        self.configuration = configuration

    @property
    def microbatches_per_optimizer_step(self) -> int:
        return self.configuration.microbatches_per_optimizer_step

    def microbatch_indices(self) -> range:
        return range(self.microbatches_per_optimizer_step)

    def scale_loss(self, loss: torch.Tensor) -> torch.Tensor:
        if not isinstance(loss, torch.Tensor) or loss.numel() != 1:
            raise ValueError("gradient accumulation requires one scalar loss tensor")
        if self.microbatches_per_optimizer_step == 1:
            return loss
        return loss / self.microbatches_per_optimizer_step

    def state_dict(self) -> dict[str, object]:
        return {}

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        if state:
            raise ValueError("fixed step-boundary accumulation state must be empty")


def build_registered_gradient_accumulation(
    implementation: GradientAccumulationImplementation,
    configuration: FixedGradientAccumulationConfiguration,
) -> FixedGradientAccumulation:
    if implementation is not GradientAccumulationImplementation.FIXED_V1:
        raise ValueError(
            f"unsupported gradient accumulation implementation: {implementation!r}"
        )
    if not isinstance(configuration, FixedGradientAccumulationConfiguration):
        raise TypeError("fixed accumulation requires its typed configuration")
    return FixedGradientAccumulation(configuration)


def gradient_accumulation_from_resolved_component(
    component: Mapping[str, Any],
) -> FixedGradientAccumulation:
    implementation, configuration = resolved_component_parts(
        component, "gradient_accumulation"
    )
    try:
        selected = GradientAccumulationImplementation(implementation)
    except ValueError as error:
        raise ValueError(
            "resolved gradient accumulation implementation is not allowlisted"
        ) from error
    return build_registered_gradient_accumulation(
        selected,
        FixedGradientAccumulationConfiguration.from_resolved(configuration),
    )


__all__ = [
    "FixedGradientAccumulation",
    "FixedGradientAccumulationConfiguration",
    "GradientAccumulationImplementation",
    "build_registered_gradient_accumulation",
    "gradient_accumulation_from_resolved_component",
]
