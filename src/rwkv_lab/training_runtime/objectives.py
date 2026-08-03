from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any

import torch

from rwkv_lab.fused_ce import lmhead_cross_entropy

from .resolved import resolved_component_parts


class ObjectiveImplementation(str, Enum):
    LINEAR_HEAD_CROSS_ENTROPY_V1 = (
        "rwkv_lab.objective.linear_head_cross_entropy.v1"
    )


@dataclass(frozen=True, slots=True)
class LinearHeadCrossEntropyConfiguration:
    chunk_size: int = 2048
    prefer_fused: bool = True

    def __post_init__(self) -> None:
        if (
            not isinstance(self.chunk_size, int)
            or isinstance(self.chunk_size, bool)
            or not 1 <= self.chunk_size <= 1_048_576
        ):
            raise ValueError("cross-entropy chunk_size must be in [1, 1048576]")
        if not isinstance(self.prefer_fused, bool):
            raise TypeError("cross-entropy prefer_fused must be boolean")

    @classmethod
    def from_resolved(
        cls, configuration: Mapping[str, Any]
    ) -> LinearHeadCrossEntropyConfiguration:
        if set(configuration) != {"chunk_size", "prefer_fused"}:
            raise ValueError(
                "resolved linear-head cross-entropy configuration has missing or unknown fields"
            )
        return cls(**configuration)


class LinearHeadCrossEntropyObjective:
    """Mean token CE over a declared linear head with a bounded fallback."""

    def __init__(self, configuration: LinearHeadCrossEntropyConfiguration) -> None:
        self.configuration = configuration

    def __call__(
        self,
        hidden: torch.Tensor,
        linear_head: torch.nn.Module,
        labels: torch.Tensor,
        *,
        ignore_index: int | None = None,
    ) -> torch.Tensor:
        if not isinstance(hidden, torch.Tensor) or hidden.ndim < 2:
            raise ValueError("linear-head cross entropy requires hidden activations")
        if not isinstance(labels, torch.Tensor) or labels.shape != hidden.shape[:-1]:
            raise ValueError("linear-head cross entropy labels disagree with hidden shape")
        if not isinstance(linear_head, torch.nn.Module) or not isinstance(
            getattr(linear_head, "weight", None), torch.Tensor
        ):
            raise TypeError("linear-head cross entropy requires a weighted module")
        if ignore_index is not None and (
            not isinstance(ignore_index, int) or isinstance(ignore_index, bool)
        ):
            raise ValueError("linear-head cross entropy ignore_index must be integer")
        return lmhead_cross_entropy(
            hidden,
            linear_head,
            labels,
            chunk=self.configuration.chunk_size,
            fused=self.configuration.prefer_fused,
            ignore_index=ignore_index,
        )

    def state_dict(self) -> dict[str, object]:
        return {}

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        if state:
            raise ValueError("linear-head cross-entropy objective state must be empty")


def build_registered_objective(
    implementation: ObjectiveImplementation,
    configuration: LinearHeadCrossEntropyConfiguration,
) -> LinearHeadCrossEntropyObjective:
    if implementation is not ObjectiveImplementation.LINEAR_HEAD_CROSS_ENTROPY_V1:
        raise ValueError(f"unsupported objective implementation: {implementation!r}")
    if not isinstance(configuration, LinearHeadCrossEntropyConfiguration):
        raise TypeError("linear-head cross entropy requires its typed configuration")
    return LinearHeadCrossEntropyObjective(configuration)


def objective_from_resolved_component(
    component: Mapping[str, Any],
) -> LinearHeadCrossEntropyObjective:
    implementation, configuration = resolved_component_parts(component, "objective")
    try:
        selected = ObjectiveImplementation(implementation)
    except ValueError as error:
        raise ValueError("resolved objective implementation is not allowlisted") from error
    return build_registered_objective(
        selected,
        LinearHeadCrossEntropyConfiguration.from_resolved(configuration),
    )


__all__ = [
    "LinearHeadCrossEntropyConfiguration",
    "LinearHeadCrossEntropyObjective",
    "ObjectiveImplementation",
    "build_registered_objective",
    "objective_from_resolved_component",
]
