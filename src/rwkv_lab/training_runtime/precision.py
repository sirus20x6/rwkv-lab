from __future__ import annotations

import importlib.util
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
    FP8_SCALED_V1 = "rwkv_lab.precision.fp8_scaled.v1"
    NVFP4_MICROSCALING_V1 = "rwkv_lab.precision.nvfp4_microscaling.v1"


class PrecisionUnavailableError(RuntimeError):
    """A selected scaled precision policy cannot execute on this worker."""


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


_DEFAULT_ELIGIBLE_PROJECTIONS = (
    "*attention*.weight",
    "*attn*.weight",
    "*feed_forward*.weight",
    "*ffn*.weight",
    "*expert*.weight",
)
_DEFAULT_EXCLUDED_PARAMETERS = (
    "*embedding*.weight",
    "head.weight",
    "*.head.weight",
    "*lm_head*.weight",
    "norm.weight",
    "*.norm*.weight",
)


def _validate_projection_surface(
    eligible: tuple[str, ...], excluded: tuple[str, ...]
) -> None:
    for field_name, patterns in (
        ("eligible_projection_patterns", eligible),
        ("excluded_parameter_patterns", excluded),
    ):
        if not isinstance(patterns, tuple) or not patterns:
            raise ValueError(f"scaled precision {field_name} must be a non-empty tuple")
        if any(
            not isinstance(pattern, str) or not pattern.strip() for pattern in patterns
        ):
            raise ValueError(
                f"scaled precision {field_name} must contain non-empty strings"
            )
        if len(set(patterns)) != len(patterns):
            raise ValueError(
                f"scaled precision {field_name} must not contain duplicates"
            )
    if set(eligible).intersection(excluded):
        raise ValueError(
            "scaled precision eligible and excluded parameter patterns must not overlap"
        )


def _validate_state_domains(
    observed: tuple[str, ...], expected: tuple[str, ...]
) -> None:
    if not isinstance(observed, tuple) or observed != expected:
        raise ValueError(
            "scaled precision checkpoint_state_domains disagree with scaling and "
            "master-weight declarations"
        )


@dataclass(frozen=True, slots=True)
class Float8PrecisionConfiguration:
    backend: str = "transformer_engine"
    format: str = "float8_e4m3fn"
    scaling_strategy: str = "delayed_per_tensor"
    amax_history_length: int = 16
    retain_fp32_master_weights: bool = True
    checkpoint_state_domains: tuple[str, ...] = (
        "scale_factors",
        "amax_history",
        "master_weights",
    )
    eligible_projection_patterns: tuple[str, ...] = _DEFAULT_ELIGIBLE_PROJECTIONS
    excluded_parameter_patterns: tuple[str, ...] = _DEFAULT_EXCLUDED_PARAMETERS

    def __post_init__(self) -> None:
        if self.backend not in {"transformer_engine", "torchao"}:
            raise ValueError("FP8 backend must be 'transformer_engine' or 'torchao'")
        if self.format != "float8_e4m3fn":
            raise ValueError("FP8 format must be float8_e4m3fn")
        expected_strategy = {
            "transformer_engine": "delayed_per_tensor",
            "torchao": "dynamic_per_tensor",
        }[self.backend]
        if self.scaling_strategy != expected_strategy:
            raise ValueError(
                f"FP8 {self.backend} scaling_strategy must be {expected_strategy!r}"
            )
        if not isinstance(self.amax_history_length, int) or isinstance(
            self.amax_history_length, bool
        ):
            raise TypeError("FP8 amax_history_length must be an integer")
        if self.scaling_strategy == "delayed_per_tensor":
            if not 1 <= self.amax_history_length <= 4096:
                raise ValueError("delayed FP8 amax_history_length must be in [1, 4096]")
        elif self.amax_history_length != 0:
            raise ValueError("dynamic FP8 amax_history_length must be zero")
        if not isinstance(self.retain_fp32_master_weights, bool):
            raise TypeError("FP8 retain_fp32_master_weights must be boolean")
        expected_domains = ["scale_factors"]
        if self.scaling_strategy == "delayed_per_tensor":
            expected_domains.append("amax_history")
        if self.retain_fp32_master_weights:
            expected_domains.append("master_weights")
        _validate_state_domains(self.checkpoint_state_domains, tuple(expected_domains))
        _validate_projection_surface(
            self.eligible_projection_patterns, self.excluded_parameter_patterns
        )

    @classmethod
    def from_resolved(
        cls, configuration: Mapping[str, Any]
    ) -> Float8PrecisionConfiguration:
        expected = {
            "backend",
            "format",
            "scaling_strategy",
            "amax_history_length",
            "retain_fp32_master_weights",
            "checkpoint_state_domains",
            "eligible_projection_patterns",
            "excluded_parameter_patterns",
        }
        if set(configuration) != expected:
            raise ValueError(
                "resolved FP8 precision configuration has missing or unknown fields"
            )
        normalized = dict(configuration)
        for field_name in (
            "checkpoint_state_domains",
            "eligible_projection_patterns",
            "excluded_parameter_patterns",
        ):
            value = normalized[field_name]
            if not isinstance(value, (list, tuple)):
                raise TypeError(f"resolved FP8 {field_name} must be a sequence")
            normalized[field_name] = tuple(value)
        return cls(**normalized)


@dataclass(frozen=True, slots=True)
class NVFP4PrecisionConfiguration:
    backend: str = "transformer_engine"
    format: str = "nvfp4"
    scaling_strategy: str = "microscaling_per_block"
    block_size: int = 16
    retain_fp32_master_weights: bool = True
    checkpoint_state_domains: tuple[str, ...] = (
        "scale_factors",
        "master_weights",
    )
    eligible_projection_patterns: tuple[str, ...] = _DEFAULT_ELIGIBLE_PROJECTIONS
    excluded_parameter_patterns: tuple[str, ...] = _DEFAULT_EXCLUDED_PARAMETERS

    def __post_init__(self) -> None:
        if self.backend != "transformer_engine":
            raise ValueError("NVFP4 backend must be 'transformer_engine'")
        if self.format != "nvfp4":
            raise ValueError("NVFP4 format must be nvfp4")
        if self.scaling_strategy != "microscaling_per_block":
            raise ValueError("NVFP4 scaling_strategy must be 'microscaling_per_block'")
        if (
            not isinstance(self.block_size, int)
            or isinstance(self.block_size, bool)
            or self.block_size != 16
        ):
            raise ValueError("NVFP4 microscaling block_size must be 16")
        if not isinstance(self.retain_fp32_master_weights, bool):
            raise TypeError("NVFP4 retain_fp32_master_weights must be boolean")
        expected_domains = ["scale_factors"]
        if self.retain_fp32_master_weights:
            expected_domains.append("master_weights")
        _validate_state_domains(self.checkpoint_state_domains, tuple(expected_domains))
        _validate_projection_surface(
            self.eligible_projection_patterns, self.excluded_parameter_patterns
        )

    @classmethod
    def from_resolved(
        cls, configuration: Mapping[str, Any]
    ) -> NVFP4PrecisionConfiguration:
        expected = {
            "backend",
            "format",
            "scaling_strategy",
            "block_size",
            "retain_fp32_master_weights",
            "checkpoint_state_domains",
            "eligible_projection_patterns",
            "excluded_parameter_patterns",
        }
        if set(configuration) != expected:
            raise ValueError(
                "resolved NVFP4 precision configuration has missing or unknown fields"
            )
        normalized = dict(configuration)
        for field_name in (
            "checkpoint_state_domains",
            "eligible_projection_patterns",
            "excluded_parameter_patterns",
        ):
            value = normalized[field_name]
            if not isinstance(value, (list, tuple)):
                raise TypeError(f"resolved NVFP4 {field_name} must be a sequence")
            normalized[field_name] = tuple(value)
        return cls(**normalized)


def _clone_tensor_mapping(value: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {name: tensor.detach().clone() for name, tensor in value.items()}


class _ScaledPrecisionPolicy:
    label: str

    def __init__(
        self, configuration: Float8PrecisionConfiguration | NVFP4PrecisionConfiguration
    ) -> None:
        self.configuration = configuration
        self._state = self._default_state()

    @property
    def parameter_dtype(self) -> torch.dtype:
        # Scaled GEMMs retain recoverable live parameters; the backend owns the
        # low-precision operands and their checkpointed scale state.
        return torch.bfloat16

    @property
    def reduction_dtype(self) -> torch.dtype:
        return torch.float32

    def reduce(self, tensor: torch.Tensor) -> torch.Tensor:
        if not isinstance(tensor, torch.Tensor):
            raise TypeError("precision reduction requires a tensor")
        return tensor.to(dtype=self.reduction_dtype)

    def _default_state(self) -> dict[str, dict[str, torch.Tensor]]:
        state = {"scale_factors": {"__policy__": torch.ones((), dtype=torch.float32)}}
        if "amax_history" in self.configuration.checkpoint_state_domains:
            assert isinstance(self.configuration, Float8PrecisionConfiguration)
            state["amax_history"] = {
                "__policy__": torch.zeros(
                    self.configuration.amax_history_length, dtype=torch.float32
                )
            }
        if "master_weights" in self.configuration.checkpoint_state_domains:
            state["master_weights"] = {}
        return state

    def state_dict(self) -> dict[str, object]:
        return {
            domain: _clone_tensor_mapping(values)
            for domain, values in self._state.items()
        }

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        expected_domains = set(self.configuration.checkpoint_state_domains)
        if set(state) != expected_domains:
            raise ValueError(
                f"{self.label} precision state must contain exactly "
                f"{sorted(expected_domains)!r}"
            )
        validated: dict[str, dict[str, torch.Tensor]] = {}
        for domain in self.configuration.checkpoint_state_domains:
            values = state[domain]
            if not isinstance(values, Mapping):
                raise TypeError(
                    f"{self.label} precision {domain} state must be a mapping"
                )
            tensors: dict[str, torch.Tensor] = {}
            for name, tensor in values.items():
                if not isinstance(name, str) or not name:
                    raise ValueError(
                        f"{self.label} precision {domain} names must be non-empty strings"
                    )
                if not isinstance(tensor, torch.Tensor):
                    raise TypeError(
                        f"{self.label} precision {domain} values must be tensors"
                    )
                if tensor.dtype is not torch.float32:
                    raise ValueError(
                        f"{self.label} precision {domain} tensors must be float32"
                    )
                if not bool(torch.isfinite(tensor).all()):
                    raise ValueError(
                        f"{self.label} precision {domain} tensors must be finite"
                    )
                tensors[name] = tensor.detach().clone()
            validated[domain] = tensors
        scales = validated["scale_factors"]
        if not scales:
            raise ValueError(
                f"{self.label} precision state must retain at least one scale factor"
            )
        if any(not bool((scale > 0).all()) for scale in scales.values()):
            raise ValueError(f"{self.label} precision scale factors must be positive")
        if "amax_history" in validated:
            assert isinstance(self.configuration, Float8PrecisionConfiguration)
            histories = validated["amax_history"]
            if set(histories) != set(scales):
                raise ValueError(
                    "FP8 amax history keys must exactly match scale-factor keys"
                )
            if any(
                history.numel() != self.configuration.amax_history_length
                for history in histories.values()
            ):
                raise ValueError(
                    "FP8 amax history tensors must match amax_history_length"
                )
        self._state = validated


class Float8PrecisionPolicy(_ScaledPrecisionPolicy):
    """Checkpoint codec and declared surface for backend-owned FP8 GEMMs."""

    label = "FP8"

    def __init__(self, configuration: Float8PrecisionConfiguration) -> None:
        super().__init__(configuration)

    def convert_module(
        self, module: torch.nn.Module, device: torch.device | str
    ) -> torch.nn.Module:
        if not isinstance(module, torch.nn.Module):
            raise TypeError("precision conversion requires a torch module")
        _require_scaled_precision_available(self.label, self.configuration.backend)
        raise PrecisionUnavailableError(
            "FP8 backend module conversion is registered but not execution-qualified"
        )


class NVFP4PrecisionPolicy(_ScaledPrecisionPolicy):
    """Checkpoint codec and declared surface for backend-owned NVFP4 GEMMs."""

    label = "NVFP4"

    def __init__(self, configuration: NVFP4PrecisionConfiguration) -> None:
        super().__init__(configuration)

    def convert_module(
        self, module: torch.nn.Module, device: torch.device | str
    ) -> torch.nn.Module:
        if not isinstance(module, torch.nn.Module):
            raise TypeError("precision conversion requires a torch module")
        _require_scaled_precision_available(self.label, self.configuration.backend)
        raise PrecisionUnavailableError(
            "NVFP4 backend module conversion is registered but not execution-qualified"
        )


# SM120 is the only capability qualified by this card. A minimum capability
# floor is intentionally not inferred from architecture numbering.
_QUALIFIED_CAPABILITIES = frozenset({(12, 0)})


def _require_cuda_capability(label: str) -> None:
    if not torch.cuda.is_available():
        raise PrecisionUnavailableError(
            f"{label} precision requires CUDA, but torch.cuda.is_available() is false"
        )
    capability = tuple(torch.cuda.get_device_capability(torch.cuda.current_device()))
    if capability not in _QUALIFIED_CAPABILITIES:
        found = f"sm_{capability[0]}{capability[1]}"
        qualified = ", ".join(
            f"sm_{major}{minor}" for major, minor in sorted(_QUALIFIED_CAPABILITIES)
        )
        raise PrecisionUnavailableError(
            f"{label} precision has insufficient CUDA capability {found}; "
            f"qualified capabilities: {qualified}"
        )


def _require_scaled_precision_available(label: str, backend: str) -> None:
    if importlib.util.find_spec(backend) is None:
        raise PrecisionUnavailableError(
            f"{label} precision requires backend package {backend!r}, "
            "but it is not installed"
        )
    _require_cuda_capability(label)


def build_registered_precision_policy(
    implementation: PrecisionImplementation,
    configuration: (
        BFloat16PrecisionConfiguration
        | Float8PrecisionConfiguration
        | NVFP4PrecisionConfiguration
    ),
) -> BFloat16PrecisionPolicy | Float8PrecisionPolicy | NVFP4PrecisionPolicy:
    if implementation is PrecisionImplementation.BF16_PARAMETERS_FP32_REDUCTIONS_V1:
        if not isinstance(configuration, BFloat16PrecisionConfiguration):
            raise TypeError("BF16 precision requires its typed configuration")
        return BFloat16PrecisionPolicy(configuration)
    if implementation is PrecisionImplementation.FP8_SCALED_V1:
        if not isinstance(configuration, Float8PrecisionConfiguration):
            raise TypeError("FP8 precision requires its typed configuration")
        _require_scaled_precision_available("FP8", configuration.backend)
        return Float8PrecisionPolicy(configuration)
    if implementation is PrecisionImplementation.NVFP4_MICROSCALING_V1:
        if not isinstance(configuration, NVFP4PrecisionConfiguration):
            raise TypeError("NVFP4 precision requires its typed configuration")
        _require_scaled_precision_available("NVFP4", configuration.backend)
        return NVFP4PrecisionPolicy(configuration)
    raise ValueError(f"unsupported precision implementation: {implementation!r}")


def precision_policy_from_resolved_component(
    component: Mapping[str, Any],
) -> BFloat16PrecisionPolicy | Float8PrecisionPolicy | NVFP4PrecisionPolicy:
    implementation, configuration = resolved_component_parts(component, "precision")
    try:
        selected = PrecisionImplementation(implementation)
    except ValueError as error:
        raise ValueError(
            "resolved precision implementation is not allowlisted"
        ) from error
    if selected is PrecisionImplementation.BF16_PARAMETERS_FP32_REDUCTIONS_V1:
        typed_configuration = BFloat16PrecisionConfiguration.from_resolved(
            configuration
        )
    elif selected is PrecisionImplementation.FP8_SCALED_V1:
        typed_configuration = Float8PrecisionConfiguration.from_resolved(configuration)
    else:
        typed_configuration = NVFP4PrecisionConfiguration.from_resolved(configuration)
    return build_registered_precision_policy(selected, typed_configuration)


__all__ = [
    "BFloat16PrecisionConfiguration",
    "BFloat16PrecisionPolicy",
    "Float8PrecisionConfiguration",
    "Float8PrecisionPolicy",
    "NVFP4PrecisionConfiguration",
    "NVFP4PrecisionPolicy",
    "PrecisionImplementation",
    "PrecisionUnavailableError",
    "build_registered_precision_policy",
    "precision_policy_from_resolved_component",
]
