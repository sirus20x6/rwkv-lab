from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any

import torch

from rwkv_lab.training_optimizers import FP32MasterAdamW

from .resolved import resolved_component_parts


class OptimizerImplementation(str, Enum):
    TORCH_ADAMW_V1 = "rwkv_lab.optimizer.torch_adamw.v1"
    FP32_MASTER_ADAMW_V1 = "rwkv_lab.optimizer.fp32_master_adamw.v1"
    TORCH_ADAMW_NO_DECAY_V2 = "rwkv_lab.optimizer.torch_adamw_no_decay.v2"
    FP32_MASTER_ADAMW_NO_DECAY_V2 = (
        "rwkv_lab.optimizer.fp32_master_adamw_no_decay.v2"
    )
    TORCH_SPARSE_ADAM_V1 = "rwkv_lab.optimizer.torch_sparse_adam.v1"


@dataclass(frozen=True, slots=True)
class AdamWConfiguration:
    learning_rate: float
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1.0e-8
    weight_decay: float = 0.01
    foreach: bool = True
    fused: bool = False

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
        if not isinstance(self.foreach, bool) or not isinstance(self.fused, bool):
            raise TypeError("AdamW foreach and fused flags must be boolean")
        if self.foreach and self.fused:
            raise ValueError("AdamW foreach and fused modes are mutually exclusive")

    @classmethod
    def from_resolved(cls, configuration: Mapping[str, Any]) -> AdamWConfiguration:
        expected = {
            "learning_rate",
            "beta1",
            "beta2",
            "epsilon",
            "weight_decay",
            "foreach",
            "fused",
        }
        if set(configuration) != expected:
            raise ValueError(
                "resolved AdamW configuration has missing or unknown fields"
            )
        return cls(**configuration)


@dataclass(frozen=True, slots=True)
class AdamWNoDecayConfiguration:
    """AdamW mechanics with decay delegated to its independent schedule."""

    learning_rate: float
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1.0e-8
    foreach: bool = True
    fused: bool = False

    def __post_init__(self) -> None:
        # Reuse the mature scalar/backend checks while forcing the optimizer's
        # own decay to zero. The separately resolved decay component installs
        # the effective value before the first update.
        AdamWConfiguration(
            learning_rate=self.learning_rate,
            beta1=self.beta1,
            beta2=self.beta2,
            epsilon=self.epsilon,
            weight_decay=0.0,
            foreach=self.foreach,
            fused=self.fused,
        )

    @classmethod
    def from_resolved(
        cls, configuration: Mapping[str, Any]
    ) -> AdamWNoDecayConfiguration:
        expected = {
            "learning_rate",
            "beta1",
            "beta2",
            "epsilon",
            "foreach",
            "fused",
        }
        if set(configuration) != expected:
            raise ValueError(
                "resolved no-decay AdamW configuration has missing or unknown fields"
            )
        return cls(**configuration)


@dataclass(frozen=True, slots=True)
class SparseAdamConfiguration:
    learning_rate: float
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1.0e-8

    def __post_init__(self) -> None:
        AdamWConfiguration(
            learning_rate=self.learning_rate,
            beta1=self.beta1,
            beta2=self.beta2,
            epsilon=self.epsilon,
            weight_decay=0.0,
            foreach=False,
            fused=False,
        )

    @classmethod
    def from_resolved(cls, configuration: Mapping[str, Any]) -> SparseAdamConfiguration:
        if set(configuration) != {"learning_rate", "beta1", "beta2", "epsilon"}:
            raise ValueError(
                "resolved SparseAdam configuration has missing or unknown fields"
            )
        return cls(**configuration)


def build_registered_optimizer(
    implementation: OptimizerImplementation,
    parameters: Iterable[torch.nn.Parameter] | Iterable[Mapping[str, Any]],
    configuration: (
        AdamWConfiguration | AdamWNoDecayConfiguration | SparseAdamConfiguration
    ),
) -> torch.optim.Optimizer:
    """Construct one allowlisted optimizer implementation from typed values."""

    if implementation is OptimizerImplementation.TORCH_SPARSE_ADAM_V1:
        if not isinstance(configuration, SparseAdamConfiguration):
            raise TypeError("Torch SparseAdam v1 requires its sparse configuration")
        return torch.optim.SparseAdam(
            parameters,
            lr=configuration.learning_rate,
            betas=(configuration.beta1, configuration.beta2),
            eps=configuration.epsilon,
        )
    kwargs = {
        "lr": configuration.learning_rate,
        "betas": (configuration.beta1, configuration.beta2),
        "eps": configuration.epsilon,
        "weight_decay": (
            configuration.weight_decay
            if isinstance(configuration, AdamWConfiguration)
            else 0.0
        ),
        "foreach": configuration.foreach,
        "fused": configuration.fused,
    }
    if implementation is OptimizerImplementation.TORCH_ADAMW_V1:
        if not isinstance(configuration, AdamWConfiguration):
            raise TypeError("Torch AdamW v1 requires its legacy configuration")
        return torch.optim.AdamW(parameters, **kwargs)
    if implementation is OptimizerImplementation.FP32_MASTER_ADAMW_V1:
        if not isinstance(configuration, AdamWConfiguration):
            raise TypeError("FP32-master AdamW v1 requires its legacy configuration")
        return FP32MasterAdamW(parameters, **kwargs)
    if implementation is OptimizerImplementation.TORCH_ADAMW_NO_DECAY_V2:
        if not isinstance(configuration, AdamWNoDecayConfiguration):
            raise TypeError("no-decay Torch AdamW requires its v2 configuration")
        return torch.optim.AdamW(parameters, **kwargs)
    if implementation is OptimizerImplementation.FP32_MASTER_ADAMW_NO_DECAY_V2:
        if not isinstance(configuration, AdamWNoDecayConfiguration):
            raise TypeError("no-decay FP32-master AdamW requires its v2 configuration")
        return FP32MasterAdamW(parameters, **kwargs)
    raise ValueError(f"unsupported optimizer implementation: {implementation!r}")


def optimizer_from_resolved_component(
    component: Mapping[str, Any],
    parameters: Iterable[torch.nn.Parameter] | Iterable[Mapping[str, Any]],
) -> torch.optim.Optimizer:
    implementation, configuration = resolved_component_parts(component, "optimizer")
    try:
        selected = OptimizerImplementation(implementation)
    except ValueError as error:
        raise ValueError(
            "resolved optimizer implementation is not allowlisted"
        ) from error
    if selected is OptimizerImplementation.TORCH_SPARSE_ADAM_V1:
        typed_configuration = SparseAdamConfiguration.from_resolved(configuration)
    elif selected in {
            OptimizerImplementation.TORCH_ADAMW_NO_DECAY_V2,
            OptimizerImplementation.FP32_MASTER_ADAMW_NO_DECAY_V2,
    }:
        typed_configuration = AdamWNoDecayConfiguration.from_resolved(configuration)
    else:
        typed_configuration = AdamWConfiguration.from_resolved(configuration)
    return build_registered_optimizer(selected, parameters, typed_configuration)
