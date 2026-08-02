from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any

import torch

from .resolved import resolved_component_parts


class PrecisionImplementation(str, Enum):
    BF16_PARAMETERS_FP32_REDUCTIONS_V1 = (
        "rwkv_lab.precision.bf16_parameters_fp32_reductions.v1"
    )


@dataclass(frozen=True, slots=True)
class BFloat16PrecisionConfiguration:
    parameter_dtype: str = "bfloat16"
    compute_dtype: str = "bfloat16"
    reduction_dtype: str = "float32"
    gradient_scaling: bool = False

    def __post_init__(self) -> None:
        expected = {
            "parameter_dtype": "bfloat16",
            "compute_dtype": "bfloat16",
            "reduction_dtype": "float32",
            "gradient_scaling": False,
        }
        observed = {
            "parameter_dtype": self.parameter_dtype,
            "compute_dtype": self.compute_dtype,
            "reduction_dtype": self.reduction_dtype,
            "gradient_scaling": self.gradient_scaling,
        }
        if observed != expected:
            raise ValueError(
                "BF16/FP32 precision configuration must match its fixed v1 semantics"
            )

    @classmethod
    def from_resolved(
        cls, configuration: Mapping[str, Any]
    ) -> BFloat16PrecisionConfiguration:
        if set(configuration) != {
            "parameter_dtype",
            "compute_dtype",
            "reduction_dtype",
            "gradient_scaling",
        }:
            raise ValueError(
                "resolved BF16/FP32 precision configuration has missing or unknown fields"
            )
        return cls(**configuration)


class BFloat16PrecisionPolicy:
    """BF16 parameter/compute policy with explicit FP32 reduction semantics."""

    def __init__(self, configuration: BFloat16PrecisionConfiguration) -> None:
        self.configuration = configuration

    @property
    def parameter_dtype(self) -> torch.dtype:
        return torch.bfloat16

    @property
    def compute_dtype(self) -> torch.dtype:
        return torch.bfloat16

    @property
    def reduction_dtype(self) -> torch.dtype:
        return torch.float32

    def convert_module(
        self, module: torch.nn.Module, device: torch.device | str
    ) -> torch.nn.Module:
        if not isinstance(module, torch.nn.Module):
            raise TypeError("precision conversion requires a torch module")
        return module.to(device=device, dtype=self.parameter_dtype)

    def reduce(self, tensor: torch.Tensor) -> torch.Tensor:
        if not isinstance(tensor, torch.Tensor):
            raise TypeError("precision reduction requires a tensor")
        return tensor.to(dtype=self.reduction_dtype)

    def state_dict(self) -> dict[str, object]:
        return {}

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        if state:
            raise ValueError("unscaled BF16 precision policy state must be empty")


def build_registered_precision_policy(
    implementation: PrecisionImplementation,
    configuration: BFloat16PrecisionConfiguration,
) -> BFloat16PrecisionPolicy:
    if implementation is not PrecisionImplementation.BF16_PARAMETERS_FP32_REDUCTIONS_V1:
        raise ValueError(f"unsupported precision implementation: {implementation!r}")
    if not isinstance(configuration, BFloat16PrecisionConfiguration):
        raise TypeError("BF16 precision requires its typed configuration")
    return BFloat16PrecisionPolicy(configuration)


def precision_policy_from_resolved_component(
    component: Mapping[str, Any],
) -> BFloat16PrecisionPolicy:
    implementation, configuration = resolved_component_parts(component, "precision")
    try:
        selected = PrecisionImplementation(implementation)
    except ValueError as error:
        raise ValueError("resolved precision implementation is not allowlisted") from error
    return build_registered_precision_policy(
        selected,
        BFloat16PrecisionConfiguration.from_resolved(configuration),
    )


__all__ = [
    "BFloat16PrecisionConfiguration",
    "BFloat16PrecisionPolicy",
    "PrecisionImplementation",
    "build_registered_precision_policy",
    "precision_policy_from_resolved_component",
]
