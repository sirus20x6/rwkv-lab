from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any

import torch

from .resolved import resolved_component_parts


class ActivationMemoryImplementation(str, Enum):
    HF_GRADIENT_CHECKPOINTING_V1 = (
        "rwkv_lab.activation_memory.hf_gradient_checkpointing.v1"
    )


@dataclass(frozen=True, slots=True)
class HFGradientCheckpointingConfiguration:
    use_reentrant: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.use_reentrant, bool):
            raise TypeError("gradient checkpointing use_reentrant must be boolean")

    @classmethod
    def from_resolved(
        cls, configuration: Mapping[str, Any]
    ) -> HFGradientCheckpointingConfiguration:
        if set(configuration) != {"use_reentrant"}:
            raise ValueError(
                "resolved HF gradient-checkpointing configuration is inexact"
            )
        return cls(**configuration)


@dataclass(frozen=True, slots=True)
class HFGradientCheckpointing:
    configuration: HFGradientCheckpointingConfiguration

    def apply(self, model: torch.nn.Module) -> None:
        enable = getattr(model, "gradient_checkpointing_enable", None)
        if not callable(enable):
            raise TypeError("HF model does not support gradient checkpointing")
        enable(
            gradient_checkpointing_kwargs={
                "use_reentrant": self.configuration.use_reentrant
            }
        )
        if getattr(model, "is_gradient_checkpointing", None) is not True:
            raise ValueError("HF model did not confirm gradient checkpointing")

    def component_state(self) -> Mapping[str, bool]:
        return MappingProxyType(
            {
                "enabled": True,
                "use_reentrant": self.configuration.use_reentrant,
            }
        )

    def restore_component_state(
        self, state: Mapping[str, Any], model: torch.nn.Module
    ) -> None:
        if dict(state) != dict(self.component_state()):
            raise ValueError("HF gradient-checkpointing resume state disagrees")
        if getattr(model, "is_gradient_checkpointing", None) is not True:
            raise ValueError("HF gradient checkpointing is disabled during resume")


def activation_memory_from_resolved_component(
    component: Mapping[str, Any],
) -> HFGradientCheckpointing:
    implementation, configuration = resolved_component_parts(
        component, "activation_memory"
    )
    if implementation != (
        ActivationMemoryImplementation.HF_GRADIENT_CHECKPOINTING_V1
    ):
        raise ValueError("resolved activation-memory implementation is not allowlisted")
    return HFGradientCheckpointing(
        HFGradientCheckpointingConfiguration.from_resolved(configuration)
    )


__all__ = [
    "ActivationMemoryImplementation",
    "HFGradientCheckpointing",
    "HFGradientCheckpointingConfiguration",
    "activation_memory_from_resolved_component",
]
