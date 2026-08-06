from __future__ import annotations

import fnmatch
import hashlib
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any

import torch


class TrainabilityImplementation(str, Enum):
    FULL_V1 = "rwkv_lab.trainability.full.v1"
    FROZEN_V1 = "rwkv_lab.trainability.frozen.v1"
    NAMED_RULES_V1 = "rwkv_lab.trainability.named_rules.v1"
    LORA_V1 = "rwkv_lab.trainability.lora.v1"


@dataclass(frozen=True, slots=True)
class FullTrainabilityConfiguration:
    @classmethod
    def from_resolved(cls, value: dict[str, Any]) -> FullTrainabilityConfiguration:
        if value:
            raise ValueError("full trainability configuration must be empty")
        return cls()


@dataclass(frozen=True, slots=True)
class FrozenTrainabilityConfiguration:
    @classmethod
    def from_resolved(cls, value: dict[str, Any]) -> FrozenTrainabilityConfiguration:
        if value:
            raise ValueError("frozen trainability configuration must be empty")
        return cls()


def _patterns(values: Any, label: str) -> tuple[str, ...]:
    if not isinstance(values, (tuple, list)) or len(values) > 256:
        raise ValueError(f"{label} must be a bounded selector list")
    result = tuple(values)
    if any(not isinstance(item, str) or not item or len(item) > 512 for item in result):
        raise ValueError(f"{label} contains an invalid selector")
    if tuple(sorted(set(result))) != result:
        raise ValueError(f"{label} must be a canonical sorted selector set")
    return result


@dataclass(frozen=True, slots=True)
class NamedRulesTrainabilityConfiguration:
    default_trainable: bool
    freeze_patterns: tuple[str, ...]
    unfreeze_patterns: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.default_trainable, bool):
            raise TypeError("default_trainable must be boolean")
        _patterns(self.freeze_patterns, "freeze_patterns")
        _patterns(self.unfreeze_patterns, "unfreeze_patterns")

    @classmethod
    def from_resolved(
        cls, value: dict[str, Any]
    ) -> NamedRulesTrainabilityConfiguration:
        expected = {"default_trainable", "freeze_patterns", "unfreeze_patterns"}
        if set(value) != expected or not isinstance(value["default_trainable"], bool):
            raise ValueError("named-rules trainability configuration is inexact")
        return cls(
            default_trainable=value["default_trainable"],
            freeze_patterns=_patterns(value["freeze_patterns"], "freeze_patterns"),
            unfreeze_patterns=_patterns(
                value["unfreeze_patterns"], "unfreeze_patterns"
            ),
        )


@dataclass(frozen=True, slots=True)
class LoraTrainabilityConfiguration:
    rank: int
    alpha: int
    dropout: float
    target_selectors: tuple[str, ...]
    modules_to_save: tuple[str, ...]
    bias: str
    merge_on_completion: bool

    def __post_init__(self) -> None:
        _patterns(self.target_selectors, "target_selectors")
        _patterns(self.modules_to_save, "modules_to_save")
        if not isinstance(self.rank, int) or not 1 <= self.rank <= 4096:
            raise ValueError("LoRA rank is out of range")
        if not isinstance(self.alpha, int) or not 1 <= self.alpha <= 65536:
            raise ValueError("LoRA alpha is out of range")
        if not isinstance(self.dropout, (int, float)) or not 0 <= self.dropout < 1:
            raise ValueError("LoRA dropout is out of range")
        if not self.target_selectors:
            raise ValueError("LoRA requires at least one target selector")
        if self.bias not in {"none", "all", "lora_only"}:
            raise ValueError("LoRA bias policy is invalid")
        if not isinstance(self.merge_on_completion, bool):
            raise TypeError("LoRA merge policy is invalid")

    @classmethod
    def from_resolved(cls, value: dict[str, Any]) -> LoraTrainabilityConfiguration:
        expected = {
            "rank",
            "alpha",
            "dropout",
            "target_selectors",
            "modules_to_save",
            "bias",
            "merge_on_completion",
        }
        if set(value) != expected:
            raise ValueError("LoRA trainability configuration is inexact")
        return cls(
            rank=value["rank"],
            alpha=value["alpha"],
            dropout=value["dropout"],
            target_selectors=_patterns(value["target_selectors"], "target_selectors"),
            modules_to_save=_patterns(value["modules_to_save"], "modules_to_save"),
            bias=value["bias"],
            merge_on_completion=value["merge_on_completion"],
        )


TrainabilityConfiguration = (
    FullTrainabilityConfiguration
    | FrozenTrainabilityConfiguration
    | NamedRulesTrainabilityConfiguration
    | LoraTrainabilityConfiguration
)


def _manifest(model: torch.nn.Module) -> str:
    lines = [
        f"{name}\t{tuple(parameter.shape)}\t{parameter.dtype}\t{int(parameter.requires_grad)}"
        for name, parameter in sorted(model.named_parameters())
    ]
    return "sha256:" + hashlib.sha256("\n".join(lines).encode()).hexdigest()


def _matches(
    names: tuple[str, ...], patterns: tuple[str, ...], label: str
) -> frozenset[str]:
    selected: set[str] = set()
    for pattern in patterns:
        matches = {name for name in names if fnmatch.fnmatchcase(name, pattern)}
        if not matches:
            raise ValueError(f"{label} selector {pattern!r} matched no tensors")
        selected.update(matches)
    return frozenset(selected)


def resolve_lora_targets(
    model: torch.nn.Module, selectors: tuple[str, ...]
) -> tuple[str, ...]:
    module_names = tuple(sorted(name for name, _ in model.named_modules() if name))
    return tuple(sorted(_matches(module_names, selectors, "LoRA target")))


@dataclass(frozen=True, slots=True)
class TrainabilityResult:
    model: torch.nn.Module
    trainable_parameter_names: tuple[str, ...]
    trainable_parameter_manifest: str
    adapter_state_manifest: str | None = None
    merged: bool = False

    def component_state(self) -> MappingProxyType[str, str | bool]:
        state: dict[str, str | bool] = {
            "trainable_parameter_manifest": self.trainable_parameter_manifest,
        }
        if self.adapter_state_manifest is not None:
            state["adapter_state_manifest"] = self.adapter_state_manifest
            state["merged"] = self.merged
        return MappingProxyType(state)


@dataclass(frozen=True, slots=True)
class RegisteredTrainability:
    implementation: TrainabilityImplementation
    configuration: TrainabilityConfiguration

    def apply(self, model: torch.nn.Module) -> TrainabilityResult:
        names = tuple(name for name, _ in model.named_parameters())
        if not names:
            raise ValueError("model exposes no named parameters")
        if self.implementation is TrainabilityImplementation.FULL_V1:
            for parameter in model.parameters():
                parameter.requires_grad_(True)
        elif self.implementation is TrainabilityImplementation.FROZEN_V1:
            for parameter in model.parameters():
                parameter.requires_grad_(False)
        elif self.implementation is TrainabilityImplementation.NAMED_RULES_V1:
            configuration = self.configuration
            assert isinstance(configuration, NamedRulesTrainabilityConfiguration)
            frozen = _matches(names, configuration.freeze_patterns, "freeze")
            unfrozen = _matches(names, configuration.unfreeze_patterns, "unfreeze")
            for name, parameter in model.named_parameters():
                trainable = configuration.default_trainable
                if name in frozen:
                    trainable = False
                if name in unfrozen:
                    trainable = True
                parameter.requires_grad_(trainable)
        else:
            configuration = self.configuration
            assert isinstance(configuration, LoraTrainabilityConfiguration)
            targets = resolve_lora_targets(model, configuration.target_selectors)
            modules_to_save = (
                tuple(
                    sorted(
                        _matches(
                            tuple(
                                sorted(
                                    name for name, _ in model.named_modules() if name
                                )
                            ),
                            configuration.modules_to_save,
                            "modules_to_save",
                        )
                    )
                )
                if configuration.modules_to_save
                else None
            )
            from peft import LoraConfig, get_peft_model

            peft_configuration = LoraConfig(
                r=configuration.rank,
                lora_alpha=configuration.alpha,
                lora_dropout=float(configuration.dropout),
                target_modules=list(targets),
                modules_to_save=list(modules_to_save) if modules_to_save else None,
                bias=configuration.bias,
            )
            model = get_peft_model(model, peft_configuration)

        trainable = tuple(
            name
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        )
        manifest = _manifest(model)
        adapter_manifest = (
            manifest
            if self.implementation is TrainabilityImplementation.LORA_V1
            else None
        )
        return TrainabilityResult(
            model=model,
            trainable_parameter_names=trainable,
            trainable_parameter_manifest=manifest,
            adapter_state_manifest=adapter_manifest,
        )

    def merge_for_export(self, result: TrainabilityResult) -> torch.nn.Module:
        configuration = self.configuration
        if not isinstance(configuration, LoraTrainabilityConfiguration):
            return result.model
        if not configuration.merge_on_completion:
            return result.model
        merge = getattr(result.model, "merge_and_unload", None)
        if not callable(merge):
            raise TypeError("LoRA model cannot honor merge_on_completion")
        return merge()


def build_registered_trainability(
    implementation: TrainabilityImplementation,
    configuration: TrainabilityConfiguration,
) -> RegisteredTrainability:
    expected = {
        TrainabilityImplementation.FULL_V1: FullTrainabilityConfiguration,
        TrainabilityImplementation.FROZEN_V1: FrozenTrainabilityConfiguration,
        TrainabilityImplementation.NAMED_RULES_V1: NamedRulesTrainabilityConfiguration,
        TrainabilityImplementation.LORA_V1: LoraTrainabilityConfiguration,
    }[implementation]
    if not isinstance(configuration, expected):
        raise TypeError("trainability implementation and configuration disagree")
    return RegisteredTrainability(implementation, configuration)


def trainability_from_resolved_component(
    component: dict[str, Any],
) -> RegisteredTrainability:
    if set(component) != {"configuration", "descriptor", "descriptor_digest"}:
        raise ValueError("resolved trainability envelope has unknown fields")
    descriptor = component["descriptor"]
    if descriptor["key"]["category"] != "trainability":
        raise ValueError("resolved component is not a trainability policy")
    implementation = TrainabilityImplementation(descriptor["implementation"])
    configuration_type = {
        TrainabilityImplementation.FULL_V1: FullTrainabilityConfiguration,
        TrainabilityImplementation.FROZEN_V1: FrozenTrainabilityConfiguration,
        TrainabilityImplementation.NAMED_RULES_V1: NamedRulesTrainabilityConfiguration,
        TrainabilityImplementation.LORA_V1: LoraTrainabilityConfiguration,
    }[implementation]
    configuration = configuration_type.from_resolved(dict(component["configuration"]))
    return build_registered_trainability(implementation, configuration)
