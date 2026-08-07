from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any, Protocol

_MAXIMUM_AUXILIARY_IDENTITY_BYTES = 1024 * 1024


class ModelLoaderImplementation(str, Enum):
    HF_CAUSAL_V1 = "rwkv_lab.model_loader.hf_causal.v1"
    HF_MULTIMODAL_V1 = "rwkv_lab.model_loader.hf_multimodal.v1"


@dataclass(frozen=True, slots=True)
class HuggingFaceModelConfiguration:
    model_path: str
    checkpoint_fingerprint: str
    revision: str = "main"
    local_files_only: bool = True
    trust_remote_code: bool = False
    attention_implementation: str = "sdpa"
    quantization: str = "none"
    exact_checkpoint: bool = True
    # Prefixes of checkpoint keys this configuration knowingly declines to load.
    # A blanket exact_checkpoint gate can only be satisfied or switched off
    # wholesale, so one genuinely ignorable family (an unused prediction head,
    # say) pushes a deployment into switching the gate off entirely — and then
    # an entire parameter family can go unbound without anything failing.
    # Naming the exclusions keeps everything else fail-closed.
    ignorable_unexpected_prefixes: tuple[str, ...] = ()
    # Prefixes of model parameters that MUST be populated from the checkpoint.
    # A family listed here that loses even one tensor fails the load by name,
    # rather than contributing anonymous entries to a missing-key count.
    required_tensor_families: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        path = Path(self.model_path)
        if not path.is_absolute() or not path.exists():
            raise ValueError("model_path must name an existing absolute asset")
        if not (
            len(self.checkpoint_fingerprint) == 71
            and self.checkpoint_fingerprint.startswith("sha256:")
            and all(
                character in "0123456789abcdef"
                for character in self.checkpoint_fingerprint[7:]
            )
        ):
            raise ValueError("checkpoint_fingerprint must be a lowercase sha256 digest")
        if self.attention_implementation not in {
            "eager",
            "sdpa",
            "flash_attention_2",
        }:
            raise ValueError("unsupported attention implementation")
        if self.quantization not in {"none", "4bit", "8bit"}:
            raise ValueError("unsupported model quantization")
        for prefixes, label in (
            (self.ignorable_unexpected_prefixes, "ignorable unexpected"),
            (self.required_tensor_families, "required tensor family"),
        ):
            if len(set(prefixes)) != len(prefixes):
                raise ValueError(f"{label} prefixes must be unique")
            if any(not prefix for prefix in prefixes):
                raise ValueError(f"{label} prefixes must be non-empty")

    @classmethod
    def from_resolved(cls, value: dict[str, Any]) -> HuggingFaceModelConfiguration:
        expected = {
            "model_path",
            "checkpoint_fingerprint",
            "revision",
            "local_files_only",
            "trust_remote_code",
            "attention_implementation",
            "quantization",
            "exact_checkpoint",
            "ignorable_unexpected_prefixes",
            "required_tensor_families",
        }
        if set(value) != expected:
            raise ValueError("resolved Hugging Face model configuration is inexact")
        resolved = dict(value)
        for field in ("ignorable_unexpected_prefixes", "required_tensor_families"):
            resolved[field] = tuple(resolved[field])
        return cls(**resolved)


@dataclass(frozen=True, slots=True)
class ModelLoadReceipt:
    implementation: str
    checkpoint_fingerprint: str
    model_class: str
    checkpoint_tensor_count: int
    auxiliary_class: str
    auxiliary_fingerprint: str
    missing_keys: tuple[str, ...]
    unexpected_keys: tuple[str, ...]
    mismatched_keys: tuple[str, ...]
    error_messages: tuple[str, ...]
    # Checkpoint keys dropped under a declared exclusion, recorded so an
    # exclusion is auditable evidence rather than an invisible allowance.
    ignored_unexpected_keys: tuple[str, ...]
    # Per required family: how many of the model's parameters in that family
    # came from the checkpoint, out of how many the model has.
    family_binding: tuple[tuple[str, int, int], ...]
    exact: bool

    def canonical_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def digest(self) -> str:
        encoded = json.dumps(
            self.canonical_dict(), separators=(",", ":"), sort_keys=True
        ).encode()
        return "sha256:" + hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class LoadedModel:
    model: Any
    tokenizer_or_processor: Any
    receipt: ModelLoadReceipt

    def component_state(self) -> MappingProxyType[str, str]:
        return MappingProxyType(
            {
                "base_checkpoint_fingerprint": self.receipt.checkpoint_fingerprint,
                "load_receipt_digest": self.receipt.digest,
            }
        )


class _TransformersFacade(Protocol):
    AutoModelForCausalLM: Any
    AutoModelForImageTextToText: Any
    AutoTokenizer: Any
    AutoProcessor: Any


def _loading_arguments(configuration: HuggingFaceModelConfiguration) -> dict[str, Any]:
    arguments: dict[str, Any] = {
        "revision": configuration.revision,
        "local_files_only": configuration.local_files_only,
        "trust_remote_code": configuration.trust_remote_code,
        "attn_implementation": configuration.attention_implementation,
        "output_loading_info": True,
    }
    if configuration.quantization == "4bit":
        arguments["load_in_4bit"] = True
    elif configuration.quantization == "8bit":
        arguments["load_in_8bit"] = True
    return arguments


def _receipt(
    implementation: ModelLoaderImplementation,
    configuration: HuggingFaceModelConfiguration,
    model: Any,
    loading_info: dict[str, Any],
    auxiliary: Any,
) -> ModelLoadReceipt:
    missing = tuple(sorted(loading_info.get("missing_keys", ())))
    all_unexpected = tuple(sorted(loading_info.get("unexpected_keys", ())))
    mismatched = tuple(
        sorted(str(item) for item in loading_info.get("mismatched_keys", ()))
    )
    errors = tuple(str(item) for item in loading_info.get("error_msgs", ()))

    ignored = tuple(
        key
        for key in all_unexpected
        if key.startswith(configuration.ignorable_unexpected_prefixes)
    )
    unexpected = tuple(key for key in all_unexpected if key not in set(ignored))

    # A family is source-bound only when every parameter the model declares
    # under that prefix was populated from the checkpoint. Counting from the
    # model's own state_dict means a family that is absent entirely — the
    # checkpoint nesting it somewhere the class never looks — is a binding of
    # zero rather than a silently satisfied constraint.
    missing_set = set(missing)
    binding: list[tuple[str, int, int]] = []
    unbound: list[str] = []
    for family in configuration.required_tensor_families:
        declared = [key for key in model.state_dict() if key.startswith(family)]
        bound = [key for key in declared if key not in missing_set]
        binding.append((family, len(bound), len(declared)))
        if not declared or len(bound) != len(declared):
            unbound.append(f"{family} ({len(bound)}/{len(declared)} bound)")

    exact = not (missing or unexpected or mismatched or errors)
    if configuration.exact_checkpoint and not exact:
        raise RuntimeError(
            "Hugging Face checkpoint did not load exactly: "
            f"{len(missing)} missing, {len(unexpected)} unexpected "
            f"({len(ignored)} ignored by declaration), "
            f"{len(mismatched)} mismatched, {len(errors)} errors"
        )
    if unbound:
        raise RuntimeError(
            "required frozen base tensor families are not source-bound: "
            + ", ".join(unbound)
        )
    return ModelLoadReceipt(
        implementation=implementation.value,
        checkpoint_fingerprint=configuration.checkpoint_fingerprint,
        model_class=f"{type(model).__module__}.{type(model).__qualname__}",
        checkpoint_tensor_count=len(model.state_dict()),
        auxiliary_class=f"{type(auxiliary).__module__}.{type(auxiliary).__qualname__}",
        auxiliary_fingerprint=_auxiliary_fingerprint(auxiliary),
        missing_keys=missing,
        unexpected_keys=unexpected,
        mismatched_keys=mismatched,
        error_messages=errors,
        ignored_unexpected_keys=ignored,
        family_binding=tuple(binding),
        exact=exact,
    )


def _identity_value(value: Any, *, depth: int = 0) -> Any:
    """Reduce processor/tokenizer configuration to bounded canonical JSON.

    Hugging Face processors expose ordinary mappings plus token wrapper objects.
    Token wrappers have stable string forms; arbitrary runtime objects do not enter
    the receipt.  This keeps the model-loader receipt generic while ensuring a
    changed chat template, special-token policy, tokenizer configuration, or image
    preprocessing configuration changes exact-resume identity.
    """

    if depth > 8:
        raise ValueError("Hugging Face auxiliary identity is too deeply nested")
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Mapping):
        if len(value) > 4096 or any(not isinstance(key, str) for key in value):
            raise ValueError("Hugging Face auxiliary identity mapping is invalid")
        return {
            key: _identity_value(item, depth=depth + 1)
            for key, item in sorted(value.items())
        }
    if isinstance(value, (list, tuple)):
        if len(value) > 4096:
            raise ValueError("Hugging Face auxiliary identity list is oversized")
        return [_identity_value(item, depth=depth + 1) for item in value]
    token_fields = (
        "content",
        "single_word",
        "lstrip",
        "rstrip",
        "normalized",
        "special",
    )
    if all(hasattr(value, name) for name in token_fields):
        token = {name: getattr(value, name) for name in token_fields}
        if not isinstance(token["content"], str) or any(
            not isinstance(token[name], bool) for name in token_fields[1:]
        ):
            raise ValueError("Hugging Face token-wrapper identity is invalid")
        return {"added_token": token}
    raise ValueError("Hugging Face auxiliary identity contains a noncanonical object")


def _configuration(value: Any) -> Mapping[str, Any]:
    to_dict = getattr(value, "to_dict", None)
    if not callable(to_dict):
        return {}
    configuration = to_dict()
    if not isinstance(configuration, Mapping):
        raise TypeError("Hugging Face auxiliary to_dict result is not a mapping")
    return configuration


def _auxiliary_fingerprint(auxiliary: Any) -> str:
    tokenizer = getattr(auxiliary, "tokenizer", auxiliary)
    image_processor = getattr(auxiliary, "image_processor", None)
    chat_template = getattr(auxiliary, "chat_template", None)
    if chat_template is None:
        chat_template = getattr(tokenizer, "chat_template", None)
    body = {
        "api_version": "rwkv-lab.hf-auxiliary-identity/v1",
        "auxiliary_class": f"{type(auxiliary).__module__}.{type(auxiliary).__qualname__}",
        "auxiliary_configuration": _configuration(auxiliary),
        "chat_template": chat_template,
        "image_processor_class": (
            f"{type(image_processor).__module__}.{type(image_processor).__qualname__}"
            if image_processor is not None
            else None
        ),
        "image_processor_configuration": (
            _configuration(image_processor) if image_processor is not None else {}
        ),
        "special_tokens_map": getattr(tokenizer, "special_tokens_map", {}),
        "tokenizer_class": f"{type(tokenizer).__module__}.{type(tokenizer).__qualname__}",
        "tokenizer_configuration": _configuration(tokenizer),
    }
    encoded = json.dumps(
        _identity_value(body),
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    if not encoded or len(encoded) > _MAXIMUM_AUXILIARY_IDENTITY_BYTES:
        raise ValueError("Hugging Face auxiliary identity exceeds its byte bound")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class RegisteredModelLoader:
    implementation: ModelLoaderImplementation
    configuration: HuggingFaceModelConfiguration

    def load(
        self, *, transformers_module: _TransformersFacade | None = None
    ) -> LoadedModel:
        if transformers_module is None:
            import transformers as transformers_module  # type: ignore[no-redef]

        arguments = _loading_arguments(self.configuration)
        if self.implementation is ModelLoaderImplementation.HF_CAUSAL_V1:
            factory = transformers_module.AutoModelForCausalLM
            auxiliary_factory = transformers_module.AutoTokenizer
        else:
            factory = transformers_module.AutoModelForImageTextToText
            auxiliary_factory = transformers_module.AutoProcessor
        loaded = factory.from_pretrained(self.configuration.model_path, **arguments)
        if not isinstance(loaded, tuple) or len(loaded) != 2:
            raise RuntimeError("Hugging Face loader did not return loading attestation")
        model, loading_info = loaded
        if not isinstance(loading_info, dict):
            raise TypeError("Hugging Face loading attestation is malformed")
        auxiliary = auxiliary_factory.from_pretrained(
            self.configuration.model_path,
            revision=self.configuration.revision,
            local_files_only=self.configuration.local_files_only,
            trust_remote_code=self.configuration.trust_remote_code,
        )
        return LoadedModel(
            model=model,
            tokenizer_or_processor=auxiliary,
            receipt=_receipt(
                self.implementation,
                self.configuration,
                model,
                loading_info,
                auxiliary,
            ),
        )


def build_registered_model_loader(
    implementation: ModelLoaderImplementation,
    configuration: HuggingFaceModelConfiguration,
) -> RegisteredModelLoader:
    return RegisteredModelLoader(implementation, configuration)


def model_loader_from_resolved_component(
    component: dict[str, Any],
) -> RegisteredModelLoader:
    if set(component) != {"configuration", "descriptor", "descriptor_digest"}:
        raise ValueError("resolved model-loader envelope has unknown fields")
    descriptor = component["descriptor"]
    implementation = ModelLoaderImplementation(descriptor["implementation"])
    if descriptor["key"]["category"] != "model_loader":
        raise ValueError("resolved component is not a model loader")
    configuration = HuggingFaceModelConfiguration.from_resolved(
        dict(component["configuration"])
    )
    return build_registered_model_loader(implementation, configuration)
