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
    CACHED_REFERENCE_DPO_V1 = "rwkv_lab.objective.cached_reference_dpo.v1"


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


def _column(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value.encode("utf-8")) > 256
        or any(character in value for character in "\x00\r\n")
    ):
        raise ValueError(f"{label} must be a bounded nonempty column name")
    return value


@dataclass(frozen=True, slots=True)
class CachedReferenceDPOConfiguration:
    """Exact cached-reference DPO fields plus the held-out CE audit policy."""

    beta: float = 0.1
    label_smoothing: float = 0.0
    chosen_column: str = "chosen"
    rejected_column: str = "rejected"
    reference_chosen_logp_column: str = "reference_chosen_logp_sum"
    reference_rejected_logp_column: str = "reference_rejected_logp_sum"
    reference_chosen_token_ids_column: str = "reference_chosen_token_ids"
    reference_rejected_token_ids_column: str = "reference_rejected_token_ids"
    launch_audit_examples: int = 16
    reference_drift_tolerance: float = 2.0
    evaluation_chunk_size: int = 2048
    evaluation_prefer_fused: bool = True

    def __post_init__(self) -> None:
        if (
            not isinstance(self.beta, (float, int))
            or isinstance(self.beta, bool)
            or not 0.0 < float(self.beta) <= 10.0
        ):
            raise ValueError("DPO beta must be in (0, 10]")
        if (
            not isinstance(self.label_smoothing, (float, int))
            or isinstance(self.label_smoothing, bool)
            or not 0.0 <= float(self.label_smoothing) < 0.5
        ):
            raise ValueError("DPO label_smoothing must be in [0, 0.5)")
        columns = (
            self.chosen_column,
            self.rejected_column,
            self.reference_chosen_logp_column,
            self.reference_rejected_logp_column,
            self.reference_chosen_token_ids_column,
            self.reference_rejected_token_ids_column,
        )
        for index, value in enumerate(columns):
            _column(value, f"DPO column {index}")
        if len(set(columns)) != len(columns):
            raise ValueError("DPO columns must be distinct")
        if (
            not isinstance(self.launch_audit_examples, int)
            or isinstance(self.launch_audit_examples, bool)
            or not 1 <= self.launch_audit_examples <= 1024
        ):
            raise ValueError("DPO launch_audit_examples must be in [1, 1024]")
        if (
            not isinstance(self.reference_drift_tolerance, (float, int))
            or isinstance(self.reference_drift_tolerance, bool)
            or not 0.0 < float(self.reference_drift_tolerance) <= 100.0
        ):
            raise ValueError("DPO reference_drift_tolerance must be in (0, 100]")
        LinearHeadCrossEntropyConfiguration(
            chunk_size=self.evaluation_chunk_size,
            prefer_fused=self.evaluation_prefer_fused,
        )

    @classmethod
    def from_resolved(
        cls, configuration: Mapping[str, Any]
    ) -> CachedReferenceDPOConfiguration:
        fields = {
            "beta",
            "label_smoothing",
            "chosen_column",
            "rejected_column",
            "reference_chosen_logp_column",
            "reference_rejected_logp_column",
            "reference_chosen_token_ids_column",
            "reference_rejected_token_ids_column",
            "launch_audit_examples",
            "reference_drift_tolerance",
            "evaluation_chunk_size",
            "evaluation_prefer_fused",
        }
        if set(configuration) != fields:
            raise ValueError(
                "resolved cached-reference DPO configuration has missing or unknown fields"
            )
        return cls(**configuration)


class CachedReferenceDPOObjective:
    def __init__(self, configuration: CachedReferenceDPOConfiguration) -> None:
        self.configuration = configuration
        self.evaluation_objective = LinearHeadCrossEntropyObjective(
            LinearHeadCrossEntropyConfiguration(
                chunk_size=configuration.evaluation_chunk_size,
                prefer_fused=configuration.evaluation_prefer_fused,
            )
        )

    def __call__(
        self,
        policy_chosen: torch.Tensor,
        policy_rejected: torch.Tensor,
        reference_chosen: torch.Tensor,
        reference_rejected: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        values = (policy_chosen, policy_rejected, reference_chosen, reference_rejected)
        if any(not isinstance(value, torch.Tensor) for value in values):
            raise TypeError("DPO log-probabilities must be tensors")
        if any(value.shape != policy_chosen.shape for value in values[1:]):
            raise ValueError("DPO log-probabilities must have identical shapes")
        beta = float(self.configuration.beta)
        margin = beta * (
            (policy_chosen - reference_chosen)
            - (policy_rejected - reference_rejected)
        )
        smoothing = float(self.configuration.label_smoothing)
        loss = (
            -(1.0 - smoothing) * torch.nn.functional.logsigmoid(margin)
            - smoothing * torch.nn.functional.logsigmoid(-margin)
        )
        return loss.mean(), margin.detach()

    def state_dict(self) -> dict[str, object]:
        return {}

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        if state:
            raise ValueError("cached-reference DPO objective state must be empty")


def build_registered_objective(
    implementation: ObjectiveImplementation,
    configuration: (
        LinearHeadCrossEntropyConfiguration | CachedReferenceDPOConfiguration
    ),
) -> LinearHeadCrossEntropyObjective | CachedReferenceDPOObjective:
    if implementation is ObjectiveImplementation.LINEAR_HEAD_CROSS_ENTROPY_V1:
        if not isinstance(configuration, LinearHeadCrossEntropyConfiguration):
            raise TypeError("linear-head cross entropy requires its typed configuration")
        return LinearHeadCrossEntropyObjective(configuration)
    if implementation is ObjectiveImplementation.CACHED_REFERENCE_DPO_V1:
        if not isinstance(configuration, CachedReferenceDPOConfiguration):
            raise TypeError("cached-reference DPO requires its typed configuration")
        return CachedReferenceDPOObjective(configuration)
    raise ValueError(f"unsupported objective implementation: {implementation!r}")


def objective_from_resolved_component(
    component: Mapping[str, Any],
) -> LinearHeadCrossEntropyObjective | CachedReferenceDPOObjective:
    implementation, configuration = resolved_component_parts(component, "objective")
    try:
        selected = ObjectiveImplementation(implementation)
    except ValueError as error:
        raise ValueError("resolved objective implementation is not allowlisted") from error
    typed = (
        CachedReferenceDPOConfiguration.from_resolved(configuration)
        if selected is ObjectiveImplementation.CACHED_REFERENCE_DPO_V1
        else LinearHeadCrossEntropyConfiguration.from_resolved(configuration)
    )
    return build_registered_objective(selected, typed)


__all__ = [
    "CachedReferenceDPOConfiguration",
    "CachedReferenceDPOObjective",
    "LinearHeadCrossEntropyConfiguration",
    "LinearHeadCrossEntropyObjective",
    "ObjectiveImplementation",
    "build_registered_objective",
    "objective_from_resolved_component",
]
