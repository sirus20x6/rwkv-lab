from __future__ import annotations

from collections.abc import Mapping
from enum import Enum
from typing import Any

import torch
import torch.nn.functional as F

from .resolved import resolved_component_parts


class ActivationImplementation(str, Enum):
    SQUARED_RELU_V1 = "rwkv_lab.activation.squared_relu.v1"
    SILU_V1 = "rwkv_lab.activation.silu.v1"


class RegisteredActivation:
    def __init__(self, implementation: ActivationImplementation) -> None:
        self.implementation = implementation

    @property
    def symbolic_name(self) -> str:
        return {
            ActivationImplementation.SQUARED_RELU_V1: "squared_relu",
            ActivationImplementation.SILU_V1: "silu",
        }[self.implementation]

    def __call__(self, value: torch.Tensor) -> torch.Tensor:
        if not isinstance(value, torch.Tensor):
            raise TypeError("activation input must be a tensor")
        if self.implementation is ActivationImplementation.SQUARED_RELU_V1:
            return F.relu(value).square()
        if self.implementation is ActivationImplementation.SILU_V1:
            return F.silu(value)
        raise ValueError(f"unsupported activation: {self.implementation!r}")

    def install(self, module: torch.nn.Module) -> None:
        setter = getattr(module, "set_activation", None)
        if not isinstance(module, torch.nn.Module) or not callable(setter):
            raise TypeError("activation installation point is unsupported")
        setter(self.symbolic_name)

    def state_dict(self) -> dict[str, object]:
        return {}

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        if state:
            raise ValueError("registered activation state must be empty")


def build_registered_activation(
    implementation: ActivationImplementation,
) -> RegisteredActivation:
    if not isinstance(implementation, ActivationImplementation):
        raise TypeError("activation implementation must be allowlisted")
    return RegisteredActivation(implementation)


def activation_from_resolved_component(
    component: Mapping[str, Any],
) -> RegisteredActivation:
    implementation, configuration = resolved_component_parts(component, "activation")
    if configuration:
        raise ValueError("resolved activation configuration must be empty")
    try:
        selected = ActivationImplementation(implementation)
    except ValueError as error:
        raise ValueError("resolved activation implementation is not allowlisted") from error
    return build_registered_activation(selected)


__all__ = [
    "ActivationImplementation",
    "RegisteredActivation",
    "activation_from_resolved_component",
    "build_registered_activation",
]
