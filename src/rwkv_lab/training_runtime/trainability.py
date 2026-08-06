from __future__ import annotations

import fnmatch
import hashlib
import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any

import torch


class TrainabilityImplementation(str, Enum):
    FULL_V1 = "rwkv_lab.trainability.full.v1"
    FROZEN_V1 = "rwkv_lab.trainability.frozen.v1"
    NAMED_RULES_V1 = "rwkv_lab.trainability.named_rules.v1"
    LORA_V1 = "rwkv_lab.trainability.lora.v1"
    LORA_TARGET_MANIFEST_V2 = "rwkv_lab.trainability.lora_target_manifest.v2"


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


@dataclass(frozen=True, slots=True)
class LoraTargetManifestConfiguration:
    rank: int
    alpha: int
    dropout: float
    target_manifest_path: str
    manifest_schema: str
    required_policy_digest: str
    bias: str
    merge_on_completion: bool

    def __post_init__(self) -> None:
        LoraTrainabilityConfiguration(
            self.rank,
            self.alpha,
            self.dropout,
            ("manifest-placeholder",),
            (),
            self.bias,
            self.merge_on_completion,
        )
        if not Path(self.target_manifest_path).is_file():
            raise ValueError("LoRA target manifest must be an existing file")
        if not isinstance(self.manifest_schema, str) or not self.manifest_schema:
            raise ValueError("LoRA target manifest schema must be nonempty")
        if not (
            isinstance(self.required_policy_digest, str)
            and len(self.required_policy_digest) == 71
            and self.required_policy_digest.startswith("sha256:")
            and all(
                character in "0123456789abcdef"
                for character in self.required_policy_digest[7:]
            )
        ):
            raise ValueError("LoRA target policy digest must be a lowercase sha256")

    @classmethod
    def from_resolved(
        cls, value: dict[str, Any]
    ) -> LoraTargetManifestConfiguration:
        if set(value) != {
            "rank",
            "alpha",
            "dropout",
            "target_manifest_path",
            "manifest_schema",
            "required_policy_digest",
            "bias",
            "merge_on_completion",
        }:
            raise ValueError("LoRA target-manifest configuration is inexact")
        return cls(**value)


@dataclass(frozen=True, slots=True)
class LoraTargetReceipt:
    targets: tuple[str, ...]
    manifest_file_digest: str
    producer_target_digest: str
    policy_digest: str
    model_config_sha256: str
    weight_index_sha256: str


def load_lora_target_receipt(
    path: Path,
    *,
    manifest_schema: str | None = None,
    required_policy_digest: str | None = None,
) -> LoraTargetReceipt:
    encoded = path.read_bytes()
    document = json.loads(encoded)
    expected = {
        "architecture",
        "model_config_sha256",
        "model_type",
        "policy",
        "schema",
        "target_count",
        "target_digest",
        "targets",
        "weight_index_sha256",
    }
    if not isinstance(document, dict) or set(document) != expected:
        raise ValueError("LoRA target manifest is inexact")
    raw_targets = document["targets"]
    targets = tuple(raw_targets) if isinstance(raw_targets, list) else ()
    policy = document["policy"]
    lowercase_sha256 = lambda value: (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )
    policy_digest = "sha256:" + hashlib.sha256(
        json.dumps(policy, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if (
        not isinstance(document["schema"], str)
        or not document["schema"]
        or not isinstance(document["architecture"], list)
        or not document["architecture"]
        or any(
            not isinstance(item, str) or not item
            for item in document["architecture"]
        )
        or not isinstance(document["model_type"], str)
        or not document["model_type"]
        or not lowercase_sha256(document["model_config_sha256"])
        or not lowercase_sha256(document["weight_index_sha256"])
        or not lowercase_sha256(document["target_digest"])
        or not isinstance(raw_targets, list)
        or isinstance(document["target_count"], bool)
        or not isinstance(document["target_count"], int)
        or document["target_count"] != len(targets)
        or not targets
        or any(not isinstance(target, str) or not target for target in targets)
        or tuple(sorted(set(targets))) != targets
        or not isinstance(policy, dict)
        or not policy
        or any(
            not isinstance(key, str)
            or not key
            or not isinstance(value, str)
            or not value
            for key, value in policy.items()
        )
        or (
            manifest_schema is not None
            and document["schema"] != manifest_schema
        )
        or (
            required_policy_digest is not None
            and policy_digest != required_policy_digest
        )
    ):
        raise ValueError("LoRA target manifest receipt is invalid")
    return LoraTargetReceipt(
        targets,
        "sha256:" + hashlib.sha256(encoded).hexdigest(),
        "sha256:" + document["target_digest"],
        policy_digest,
        document["model_config_sha256"],
        document["weight_index_sha256"],
    )


def preflight_lora_target_manifest(
    model_path: Path,
    manifest_path: Path,
    *,
    manifest_schema: str | None = None,
    required_policy_digest: str | None = None,
) -> LoraTargetReceipt:
    receipt = load_lora_target_receipt(
        manifest_path,
        manifest_schema=manifest_schema,
        required_policy_digest=required_policy_digest,
    )
    config = model_path / "config.json"
    index = model_path / "model.safetensors.index.json"
    if (
        hashlib.sha256(config.read_bytes()).hexdigest()
        != receipt.model_config_sha256
        or hashlib.sha256(index.read_bytes()).hexdigest()
        != receipt.weight_index_sha256
    ):
        raise ValueError("LoRA target manifest names a different model snapshot")
    index_document = json.loads(index.read_text(encoding="utf-8"))
    weight_map = index_document.get("weight_map")
    if not isinstance(weight_map, dict):
        raise TypeError("HF safetensor index has an invalid weight map")
    modules = {
        name.rsplit(".", 1)[0]
        for name in weight_map
        if name.endswith((".weight", ".bias"))
    }
    missing = tuple(target for target in receipt.targets if target not in modules)
    if missing:
        raise ValueError("LoRA target manifest names missing snapshot modules")
    return receipt


TrainabilityConfiguration = (
    FullTrainabilityConfiguration
    | FrozenTrainabilityConfiguration
    | NamedRulesTrainabilityConfiguration
    | LoraTrainabilityConfiguration
    | LoraTargetManifestConfiguration
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


def preflight_lora_targets_from_snapshot(
    model_path: Path, selectors: tuple[str, ...]
) -> tuple[str, ...]:
    """Resolve LoRA selectors from safetensor metadata without loading weights."""

    index_path = model_path / "model.safetensors.index.json"
    single_path = model_path / "model.safetensors"
    if index_path.is_file():
        document = json.loads(index_path.read_text(encoding="utf-8"))
        weight_map = document.get("weight_map") if isinstance(document, dict) else None
        if not isinstance(weight_map, dict) or any(
            not isinstance(name, str) or not isinstance(shard, str)
            for name, shard in weight_map.items()
        ):
            raise ValueError("HF safetensor index has an invalid weight map")
        tensor_names = tuple(weight_map)
    elif single_path.is_file():
        from safetensors import safe_open

        with safe_open(single_path, framework="pt", device="cpu") as payload:
            tensor_names = tuple(payload.keys())
    else:
        raise ValueError(
            "LoRA preflight requires model.safetensors or its sharded index"
        )
    module_names = tuple(
        sorted(
            {
                name.rsplit(".", 1)[0]
                for name in tensor_names
                if name.endswith((".weight", ".bias")) and "." in name
            }
        )
    )
    if not module_names:
        raise ValueError("HF safetensor metadata exposes no module candidates")
    return tuple(sorted(_matches(module_names, selectors, "LoRA target")))


@dataclass(frozen=True, slots=True)
class TrainabilityResult:
    model: torch.nn.Module
    trainable_parameter_names: tuple[str, ...]
    trainable_parameter_manifest: str
    adapter_backed: bool = False
    target_receipt: LoraTargetReceipt | None = None

    def component_state(
        self,
        *,
        adapter_state_manifest: str | None = None,
        merged: bool = False,
    ) -> MappingProxyType[str, str | bool]:
        state: dict[str, str | bool] = {
            "trainable_parameter_manifest": self.trainable_parameter_manifest,
        }
        if self.adapter_backed:
            if not (
                isinstance(adapter_state_manifest, str)
                and len(adapter_state_manifest) == 71
                and adapter_state_manifest.startswith("sha256:")
                and all(
                    character in "0123456789abcdef"
                    for character in adapter_state_manifest[7:]
                )
            ):
                raise ValueError(
                    "LoRA checkpoint state requires the persisted adapter manifest digest"
                )
            state["adapter_state_manifest"] = adapter_state_manifest
            state["merged"] = merged
            if self.target_receipt is not None:
                state.update(
                    {
                        "target_manifest_file_digest": self.target_receipt.manifest_file_digest,
                        "producer_target_digest": self.target_receipt.producer_target_digest,
                        "producer_digest_status": "producer_claim_unrecomputed",
                        "target_policy_digest": self.target_receipt.policy_digest,
                    }
                )
        elif adapter_state_manifest is not None or merged:
            raise ValueError("non-adapter trainability has no adapter checkpoint state")
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
            assert isinstance(
                configuration,
                (LoraTargetManifestConfiguration, LoraTrainabilityConfiguration),
            )
            target_receipt = (
                load_lora_target_receipt(
                    Path(configuration.target_manifest_path),
                    manifest_schema=configuration.manifest_schema,
                    required_policy_digest=configuration.required_policy_digest,
                )
                if isinstance(configuration, LoraTargetManifestConfiguration)
                else None
            )
            if target_receipt is not None:
                module_names = frozenset(name for name, _ in model.named_modules() if name)
                missing = tuple(
                    target for target in target_receipt.targets if target not in module_names
                )
                if missing:
                    raise ValueError("LoRA target manifest names missing model modules")
                targets = target_receipt.targets
            else:
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
                if isinstance(configuration, LoraTrainabilityConfiguration)
                and configuration.modules_to_save
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
        return TrainabilityResult(
            model=model,
            trainable_parameter_names=trainable,
            trainable_parameter_manifest=manifest,
            adapter_backed=self.implementation
            in {
                TrainabilityImplementation.LORA_V1,
                TrainabilityImplementation.LORA_TARGET_MANIFEST_V2,
            },
            target_receipt=(target_receipt if "target_receipt" in locals() else None),
        )

    def merge_for_export(self, result: TrainabilityResult) -> torch.nn.Module:
        configuration = self.configuration
        if not isinstance(
            configuration,
            (LoraTargetManifestConfiguration, LoraTrainabilityConfiguration),
        ):
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
        TrainabilityImplementation.LORA_TARGET_MANIFEST_V2: LoraTargetManifestConfiguration,
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
        TrainabilityImplementation.LORA_TARGET_MANIFEST_V2: LoraTargetManifestConfiguration,
    }[implementation]
    configuration = configuration_type.from_resolved(dict(component["configuration"]))
    return build_registered_trainability(implementation, configuration)
