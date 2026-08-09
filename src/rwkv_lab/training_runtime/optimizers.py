from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, fields
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
    SPECTRAL_MUON_NO_DECAY_V1 = (
        "rwkv_lab.optimizer.spectral_muon_no_decay.v1"
    )


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


@dataclass(frozen=True, slots=True)
class SpectralMuonConfiguration:
    """Closed optimizer mechanics; adapters still own parameter topology."""

    learning_rate: float
    momentum: float = 0.95
    nesterov: bool = False
    ns_steps: int = 5
    cubic: bool = False
    spectral_power: float = 0.0
    power_method: str = "eigh"
    second_moment: bool = False
    sm_beta2: float = 0.99
    sm_eps: float = 1.0e-8
    equilibrate: str = "none"
    plus_norm: str = "none"
    row_uniform: bool = False
    mona: bool = False
    mona_beta: float = 0.9
    mona_alpha: float = 0.1
    scale: float = 0.4
    ddc_strength: float = 0.0
    ddc_mode: str = "both"
    rsav: bool = False
    rsav_c: float = 1.0
    rsav_cap: float = 0.2
    rsav_relax: float = 0.0
    tile_size: int = 0
    da_muon: bool = False
    da_eta_max: float = 0.01
    da_r0: float = 1.0e-3
    aro: bool = False
    aro_sink_iters: int = 5
    aro_compile: bool = False
    batched: bool = False
    compile_ns: bool = False
    row_update_floor: float = 0.0
    radial_brake: float = 0.0
    radius_pin: bool = False
    cautious_weight_decay: bool = False
    adam_update_interval: int = 1
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_eps: float = 1.0e-8

    def __post_init__(self) -> None:
        boolean_fields = (
            "nesterov",
            "cubic",
            "second_moment",
            "row_uniform",
            "mona",
            "rsav",
            "da_muon",
            "aro",
            "aro_compile",
            "batched",
            "compile_ns",
            "radius_pin",
            "cautious_weight_decay",
        )
        if any(not isinstance(getattr(self, name), bool) for name in boolean_fields):
            raise TypeError("SpectralMuon boolean fields must be boolean")
        integer_fields = (
            "ns_steps",
            "tile_size",
            "aro_sink_iters",
            "adam_update_interval",
        )
        if any(
            not isinstance(getattr(self, name), int)
            or isinstance(getattr(self, name), bool)
            for name in integer_fields
        ):
            raise TypeError("SpectralMuon integer fields must be integers")
        if self.ns_steps < 1 or self.aro_sink_iters < 1 or self.adam_update_interval < 1:
            raise ValueError("SpectralMuon iteration counts must be positive")
        if (
            self.ns_steps > 64
            or self.aro_sink_iters > 64
            or self.adam_update_interval > 65_536
            or not 0 <= self.tile_size <= 65_536
        ):
            raise ValueError("SpectralMuon iteration/tile counts exceed their bounds")
        numeric_fields = tuple(
            field.name
            for field in fields(self)
            if field.name not in boolean_fields
            and field.name not in integer_fields
            and field.name
            not in {"power_method", "equilibrate", "plus_norm", "ddc_mode"}
        )
        if any(
            isinstance(getattr(self, name), bool)
            or not isinstance(getattr(self, name), (int, float))
            or not math.isfinite(getattr(self, name))
            for name in numeric_fields
        ):
            raise TypeError("SpectralMuon numeric fields must be finite numbers")
        if self.power_method not in {"eigh", "svd"}:
            raise ValueError("SpectralMuon power_method is unsupported")
        if self.equilibrate not in {"none", "R", "C", "RC"}:
            raise ValueError("SpectralMuon equilibrate is unsupported")
        if self.plus_norm not in {"none", "row", "col"}:
            raise ValueError("SpectralMuon plus_norm is unsupported")
        if self.ddc_mode not in {"row", "col", "both"}:
            raise ValueError("SpectralMuon ddc_mode is unsupported")
        if not 0 < self.learning_rate <= 1 or not 0 < self.scale <= 100:
            raise ValueError("SpectralMuon learning_rate or scale is outside its bounds")
        if not 0 <= self.momentum < 1 or not 0 <= self.sm_beta2 < 1:
            raise ValueError("SpectralMuon momentum and second moment must be in [0, 1)")
        if not 0 <= self.mona_beta < 1:
            raise ValueError("SpectralMuon MONA beta must be in [0, 1)")
        if not -10 <= self.mona_alpha <= 10:
            raise ValueError("SpectralMuon MONA alpha is outside its bounds")
        if not 0 <= self.spectral_power <= 1:
            raise ValueError("SpectralMuon spectral_power must be in [0, 1]")
        if not 0 <= self.ddc_strength <= 1:
            raise ValueError("SpectralMuon ddc_strength must be in [0, 1]")
        if (
            not 0 < self.rsav_c <= 1.0e12
            or not 0 <= self.rsav_cap <= 1
            or not 0 <= self.rsav_relax <= 1
        ):
            raise ValueError("SpectralMuon RSAV configuration is outside its bounds")
        if (
            not 0 < self.da_eta_max <= 1
            or not 0 < self.da_r0 <= 1.0e12
            or not 0 <= self.row_update_floor <= 100
        ):
            raise ValueError("SpectralMuon distance/floor values are outside their bounds")
        if not 0 <= self.radial_brake <= 1:
            raise ValueError("SpectralMuon radial brake is outside its bounds")
        if not 0 <= self.adam_beta1 < 1 or not 0 <= self.adam_beta2 < 1:
            raise ValueError("SpectralMuon Adam betas must be in [0, 1)")
        if not 0 < self.sm_eps <= 1 or not 0 < self.adam_eps <= 1:
            raise ValueError("SpectralMuon epsilon values are outside their bounds")

    @classmethod
    def from_resolved(
        cls, configuration: Mapping[str, Any]
    ) -> SpectralMuonConfiguration:
        if set(configuration) != {field.name for field in fields(cls)}:
            raise ValueError(
                "resolved SpectralMuon configuration has missing or unknown fields"
            )
        return cls(**configuration)


def _spectral_muon_groups(
    parameters: Iterable[torch.nn.Parameter] | Iterable[Mapping[str, Any]],
    *,
    learning_rate: float,
) -> list[Mapping[str, Any]]:
    """Preserve the caller's Muon/fallback split instead of inventing one.

    SpectralMuon's mechanics are closed here, but which tensors take the
    orthogonalised update is a *topology* decision the adapter's parameter
    router owns. A group that does not say is refused rather than defaulted,
    because a silent default would route every tensor down one arm and the run
    would still complete.
    """

    values = list(parameters)
    if not values:
        raise ValueError("SpectralMuon requires parameters")
    if all(isinstance(value, Mapping) for value in values):
        groups = []
        for value in values:
            group = dict(value)
            if not isinstance(group.get("use_muon"), bool):
                raise TypeError(
                    "SpectralMuon parameter groups must explicitly select use_muon"
                )
            group.setdefault("lr", learning_rate)
            groups.append(group)
        return groups
    if any(isinstance(value, Mapping) for value in values):
        raise TypeError("SpectralMuon parameters and parameter groups cannot be mixed")
    return [{"params": values, "lr": learning_rate, "use_muon": False}]


def build_registered_optimizer(
    implementation: OptimizerImplementation,
    parameters: Iterable[torch.nn.Parameter] | Iterable[Mapping[str, Any]],
    configuration: (
        AdamWConfiguration
        | AdamWNoDecayConfiguration
        | SparseAdamConfiguration
        | SpectralMuonConfiguration
    ),
) -> torch.optim.Optimizer:
    """Construct one allowlisted optimizer implementation from typed values."""

    if implementation is OptimizerImplementation.SPECTRAL_MUON_NO_DECAY_V1:
        if not isinstance(configuration, SpectralMuonConfiguration):
            raise TypeError("SpectralMuon v1 requires its typed configuration")
        from rwkv_lab.spectral_muon import SpectralMuon

        values = asdict(configuration)
        learning_rate = float(values.pop("learning_rate"))
        values["adam_betas"] = (
            values.pop("adam_beta1"),
            values.pop("adam_beta2"),
        )
        # The no-decay grade is the registration, not a default: decay is an
        # independently selected weight-decay-schedule component, so the
        # optimizer must never apply one of its own.
        values["weight_decay"] = 0.0
        return SpectralMuon(
            _spectral_muon_groups(parameters, learning_rate=learning_rate),
            **values,
        )

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
    if selected is OptimizerImplementation.SPECTRAL_MUON_NO_DECAY_V1:
        typed_configuration = SpectralMuonConfiguration.from_resolved(configuration)
    elif selected is OptimizerImplementation.TORCH_SPARSE_ADAM_V1:
        typed_configuration = SparseAdamConfiguration.from_resolved(configuration)
    elif selected in {
            OptimizerImplementation.TORCH_ADAMW_NO_DECAY_V2,
            OptimizerImplementation.FP32_MASTER_ADAMW_NO_DECAY_V2,
    }:
        typed_configuration = AdamWNoDecayConfiguration.from_resolved(configuration)
    else:
        typed_configuration = AdamWConfiguration.from_resolved(configuration)
    return build_registered_optimizer(selected, parameters, typed_configuration)
