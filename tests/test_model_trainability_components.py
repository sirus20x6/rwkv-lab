from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import pytest
import torch

from rwkv_lab.training_components import (
    FrozenTrainabilityConfiguration,
    HuggingFaceModelConfiguration,
    LoraTrainabilityConfiguration,
    ModelLoaderImplementation,
    NamedRulesTrainabilityConfiguration,
    TrainabilityImplementation,
    TrainabilityResult,
    build_registered_model_loader,
    build_registered_trainability,
    model_loader_from_resolved_component,
    resolve_lora_targets,
    trainability_from_resolved_component,
)
from rwkv_lab.trainvm_worker import (
    TrainingCompositionError,
    load_resolved_training_composition,
)


def _digest(value: object) -> str:
    encoded = json.dumps(value, separators=(",", ":"), sort_keys=True).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


class TinyQwen(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.language_model = torch.nn.ModuleDict(
            {
                "layers": torch.nn.ModuleList(
                    [
                        torch.nn.ModuleDict(
                            {
                                "self_attn": torch.nn.ModuleDict(
                                    {
                                        "q_proj": torch.nn.Linear(4, 4),
                                        "k_proj": torch.nn.Linear(4, 4),
                                    }
                                ),
                                "mlp": torch.nn.Linear(4, 4),
                            }
                        )
                    ]
                )
            }
        )
        self.vision_tower = torch.nn.Linear(4, 4)


class _AutoFactory:
    loading_info: ClassVar[dict[str, object]] = {
        "missing_keys": [],
        "unexpected_keys": [],
        "mismatched_keys": [],
        "error_msgs": [],
    }

    @classmethod
    def from_pretrained(cls, *_args, **kwargs):
        assert kwargs["output_loading_info"] is True
        return TinyQwen(), dict(cls.loading_info)


class _AuxiliaryFactory:
    @classmethod
    def from_pretrained(cls, *_args, **_kwargs):
        return SimpleNamespace(
            chat_template="{{ messages }}",
            special_tokens_map={"eos_token": "<eos>"},
        )


def _model_configuration(model_path: Path) -> HuggingFaceModelConfiguration:
    return HuggingFaceModelConfiguration(
        model_path=str(model_path),
        checkpoint_fingerprint="sha256:" + "a" * 64,
    )


@pytest.mark.parametrize(
    ("implementation", "model_attribute", "auxiliary_attribute"),
    [
        (
            ModelLoaderImplementation.HF_CAUSAL_V1,
            "AutoModelForCausalLM",
            "AutoTokenizer",
        ),
        (
            ModelLoaderImplementation.HF_MULTIMODAL_V1,
            "AutoModelForImageTextToText",
            "AutoProcessor",
        ),
        (
            ModelLoaderImplementation.HF_MULTIMODAL_V2,
            "AutoModelForImageTextToText",
            "AutoProcessor",
        ),
    ],
)
def test_hugging_face_loaders_produce_exact_checkpoint_receipts(
    tmp_path: Path,
    implementation: ModelLoaderImplementation,
    model_attribute: str,
    auxiliary_attribute: str,
) -> None:
    facade = SimpleNamespace(
        **{model_attribute: _AutoFactory, auxiliary_attribute: _AuxiliaryFactory}
    )
    loaded = build_registered_model_loader(
        implementation, _model_configuration(tmp_path)
    ).load(transformers_module=facade)

    assert loaded.receipt.exact
    assert loaded.receipt.checkpoint_tensor_count == len(loaded.model.state_dict())
    assert loaded.receipt.auxiliary_fingerprint.startswith("sha256:")
    assert loaded.component_state() == {
        "base_checkpoint_fingerprint": "sha256:" + "a" * 64,
        "load_receipt_digest": loaded.receipt.digest,
    }


def test_hugging_face_load_receipt_binds_processor_template_identity(
    tmp_path: Path,
) -> None:
    class ChangedAuxiliaryFactory:
        template = "first"

        @classmethod
        def from_pretrained(cls, *_args, **_kwargs):
            return SimpleNamespace(
                chat_template=cls.template,
                special_tokens_map={"eos_token": "<eos>"},
            )

    facade = SimpleNamespace(
        AutoModelForImageTextToText=_AutoFactory,
        AutoProcessor=ChangedAuxiliaryFactory,
    )
    loader = build_registered_model_loader(
        ModelLoaderImplementation.HF_MULTIMODAL_V1,
        _model_configuration(tmp_path),
    )
    first = loader.load(transformers_module=facade)
    ChangedAuxiliaryFactory.template = "changed"
    second = loader.load(transformers_module=facade)
    assert first.receipt.auxiliary_fingerprint != second.receipt.auxiliary_fingerprint
    assert first.receipt.digest != second.receipt.digest


def test_hugging_face_loader_passes_only_declared_expert_backend(
    tmp_path: Path,
) -> None:
    seen: list[dict[str, object]] = []

    class RecordingFactory(_AutoFactory):
        @classmethod
        def from_pretrained(cls, *_args, **kwargs):
            seen.append(dict(kwargs))
            return super().from_pretrained(*_args, **kwargs)

    facade = SimpleNamespace(
        AutoModelForImageTextToText=RecordingFactory,
        AutoProcessor=_AuxiliaryFactory,
    )
    automatic = build_registered_model_loader(
        ModelLoaderImplementation.HF_MULTIMODAL_V1,
        _model_configuration(tmp_path),
    )
    automatic_receipt = automatic.load(transformers_module=facade).receipt
    assert "experts_implementation" not in seen[-1]
    assert "load_configuration_digest" not in automatic_receipt.canonical_dict()

    grouped = build_registered_model_loader(
        ModelLoaderImplementation.HF_MULTIMODAL_V2,
        HuggingFaceModelConfiguration(
            model_path=str(tmp_path),
            checkpoint_fingerprint="sha256:" + "a" * 64,
            experts_implementation="grouped_mm",
        ),
    )
    grouped_receipt = grouped.load(transformers_module=facade).receipt
    assert seen[-1]["experts_implementation"] == "grouped_mm"
    assert grouped_receipt.canonical_dict()["load_configuration_digest"] == (
        grouped_receipt.load_configuration_digest
    )
    assert grouped_receipt.load_configuration_digest != (
        automatic_receipt.load_configuration_digest
    )
    assert grouped_receipt.digest != automatic_receipt.digest

    with pytest.raises(ValueError, match="experts implementation"):
        HuggingFaceModelConfiguration(
            model_path=str(tmp_path),
            checkpoint_fingerprint="sha256:" + "a" * 64,
            experts_implementation="unbounded_import",
        )

    grouped_configuration = HuggingFaceModelConfiguration(
        model_path=str(tmp_path),
        checkpoint_fingerprint="sha256:" + "a" * 64,
        experts_implementation="grouped_mm",
    )
    for legacy in (
        ModelLoaderImplementation.HF_CAUSAL_V1,
        ModelLoaderImplementation.HF_MULTIMODAL_V1,
    ):
        with pytest.raises(ValueError, match="versioned multimodal loader"):
            build_registered_model_loader(legacy, grouped_configuration)


def test_hugging_face_auxiliary_identity_rejects_process_local_object_repr(
    tmp_path: Path,
) -> None:
    class UnstableAuxiliaryFactory:
        @classmethod
        def from_pretrained(cls, *_args, **_kwargs):
            return SimpleNamespace(
                chat_template="stable",
                special_tokens_map={"eos_token": object()},
            )

    facade = SimpleNamespace(
        AutoModelForImageTextToText=_AutoFactory,
        AutoProcessor=UnstableAuxiliaryFactory,
    )
    loader = build_registered_model_loader(
        ModelLoaderImplementation.HF_MULTIMODAL_V1,
        _model_configuration(tmp_path),
    )
    with pytest.raises(ValueError, match="noncanonical object"):
        loader.load(transformers_module=facade)


def test_exact_checkpoint_loader_rejects_loading_drift(tmp_path: Path) -> None:
    class Drifted(_AutoFactory):
        loading_info: ClassVar[dict[str, object]] = {
            **_AutoFactory.loading_info,
            "missing_keys": ["lm_head.weight"],
        }

    facade = SimpleNamespace(
        AutoModelForCausalLM=Drifted, AutoTokenizer=_AuxiliaryFactory
    )
    loader = build_registered_model_loader(
        ModelLoaderImplementation.HF_CAUSAL_V1,
        _model_configuration(tmp_path),
    )
    with pytest.raises(RuntimeError, match="did not load exactly"):
        loader.load(transformers_module=facade)


def test_named_rules_freeze_and_unfreeze_qwen_tensors_deterministically() -> None:
    model = TinyQwen()
    policy = build_registered_trainability(
        TrainabilityImplementation.NAMED_RULES_V1,
        NamedRulesTrainabilityConfiguration(
            default_trainable=False,
            freeze_patterns=(),
            unfreeze_patterns=("language_model.layers.*.self_attn.*",),
        ),
    )
    result = policy.apply(model)
    assert result.trainable_parameter_names
    assert all("self_attn" in name for name in result.trainable_parameter_names)
    assert result.component_state()["trainable_parameter_manifest"].startswith(
        "sha256:"
    )

    frozen = build_registered_trainability(
        TrainabilityImplementation.FROZEN_V1,
        FrozenTrainabilityConfiguration(),
    ).apply(model)
    assert frozen.trainable_parameter_names == ()


def test_qwen_lora_target_selection_is_exact_and_unmatched_patterns_fail() -> None:
    model = TinyQwen()
    assert resolve_lora_targets(
        model, ("language_model.layers.*.self_attn.*_proj",)
    ) == (
        "language_model.layers.0.self_attn.k_proj",
        "language_model.layers.0.self_attn.q_proj",
    )
    with pytest.raises(ValueError, match="matched no tensors"):
        resolve_lora_targets(model, ("language_model.*.does_not_exist",))

    with pytest.raises(ValueError, match="canonical sorted selector set"):
        LoraTrainabilityConfiguration.from_resolved(
            {
                "rank": 256,
                "alpha": 512,
                "dropout": 0.0,
                "target_selectors": ["z*", "a*"],
                "modules_to_save": [],
                "bias": "none",
                "merge_on_completion": False,
            }
        )


def test_lora_checkpoint_contract_requires_persisted_adapter_manifest() -> None:
    result = TrainabilityResult(
        model=TinyQwen(),
        trainable_parameter_names=("adapter.weight",),
        trainable_parameter_manifest="sha256:" + "a" * 64,
        adapter_backed=True,
    )
    with pytest.raises(ValueError, match="persisted adapter manifest"):
        result.component_state()
    assert result.component_state(adapter_state_manifest="sha256:" + "b" * 64) == {
        "adapter_state_manifest": "sha256:" + "b" * 64,
        "merged": False,
        "trainable_parameter_manifest": "sha256:" + "a" * 64,
    }


def test_resolved_component_string_lists_and_resume_state_are_exact(
    tmp_path: Path,
) -> None:
    root = Path(__file__).resolve().parents[1]
    registry = json.loads(
        (root / "docs/experiment-vm/examples/training-components.v1.json").read_text()
    )
    requested = {
        "model": (
            "model_loader",
            "hf_multimodal",
            {
                "attention_implementation": "sdpa",
                "checkpoint_fingerprint": "sha256:" + "b" * 64,
                "exact_checkpoint": True,
                "local_files_only": True,
                "model_path": str(tmp_path),
                "quantization": "none",
                "revision": "main",
                "trust_remote_code": False,
            },
        ),
        "trainability": (
            "trainability",
            "lora",
            {
                "alpha": 512,
                "bias": "none",
                "dropout": 0.0,
                "merge_on_completion": False,
                "modules_to_save": [],
                "rank": 256,
                "target_selectors": ["language_model.layers.*.self_attn.*_proj"],
            },
        ),
    }
    components = {}
    for slot, (category, name, configuration) in requested.items():
        descriptor = next(
            item
            for item in registry["components"]
            if item["key"]["category"] == category and item["key"]["name"] == name
        )
        components[slot] = {
            "configuration": configuration,
            "descriptor": descriptor,
            "descriptor_digest": _digest(descriptor),
        }
    body = {
        "api_version": "trainvm.resolved-training-composition/v1",
        "components": components,
        "model_family": "transformer",
        "registry_digest": _digest(registry),
    }
    resolved = load_resolved_training_composition(
        {**body, "composition_digest": _digest(body)}
    )
    selectors = resolved.require("trainability", category="trainability").configuration[
        "target_selectors"
    ]
    assert selectors == ("language_model.layers.*.self_attn.*_proj",)

    valid_state = {
        "model": {
            "base_checkpoint_fingerprint": "sha256:" + "b" * 64,
            "load_receipt_digest": "sha256:" + "c" * 64,
        },
        "trainability": {
            "adapter_state_manifest": "sha256:" + "d" * 64,
            "merged": False,
            "trainable_parameter_manifest": "sha256:" + "e" * 64,
        },
    }
    assert (
        resolved.validate_resume_state(valid_state)["trainability"]["merged"] is False
    )
    del valid_state["trainability"]["adapter_state_manifest"]
    with pytest.raises(TrainingCompositionError, match="incomplete"):
        resolved.validate_resume_state(valid_state)


def test_resolved_component_factories_dispatch_without_workload_imports(
    tmp_path: Path,
) -> None:
    root = Path(__file__).resolve().parents[1]
    descriptors = json.loads(
        (root / "docs/experiment-vm/examples/training-components.v1.json").read_text()
    )["components"]
    model_descriptor = next(
        item for item in descriptors if item["key"]["name"] == "hf_multimodal"
    )
    policy_descriptor = next(
        item
        for item in descriptors
        if item["key"]["category"] == "trainability" and item["key"]["name"] == "lora"
    )
    loader = model_loader_from_resolved_component(
        {
            "configuration": {
                "attention_implementation": "sdpa",
                "checkpoint_fingerprint": "sha256:" + "a" * 64,
                "exact_checkpoint": True,
                "local_files_only": True,
                "model_path": str(tmp_path),
                "quantization": "none",
                "revision": "main",
                "trust_remote_code": False,
            },
            "descriptor": model_descriptor,
            "descriptor_digest": _digest(model_descriptor),
        }
    )
    policy = trainability_from_resolved_component(
        {
            "configuration": {
                "alpha": 512,
                "bias": "none",
                "dropout": 0.0,
                "merge_on_completion": False,
                "modules_to_save": [],
                "rank": 256,
                "target_selectors": ["language_model.layers.*.self_attn.*_proj"],
            },
            "descriptor": policy_descriptor,
            "descriptor_digest": _digest(policy_descriptor),
        }
    )
    assert loader.implementation is ModelLoaderImplementation.HF_MULTIMODAL_V1
    assert policy.implementation is TrainabilityImplementation.LORA_V1


def test_resolved_model_loader_rejects_cross_version_configuration(
    tmp_path: Path,
) -> None:
    root = Path(__file__).resolve().parents[1]
    descriptors = json.loads(
        (root / "docs/experiment-vm/examples/training-components.v1.json").read_text()
    )["components"]
    descriptor = next(
        item
        for item in descriptors
        if item["key"]
        == {"category": "model_loader", "name": "hf_multimodal", "version": "1.0.0"}
    )
    with pytest.raises(ValueError, match="does not match its implementation"):
        model_loader_from_resolved_component(
            {
                "configuration": {
                    "attention_implementation": "sdpa",
                    "checkpoint_fingerprint": "sha256:" + "a" * 64,
                    "exact_checkpoint": True,
                    "experts_implementation": "grouped_mm",
                    "local_files_only": True,
                    "model_path": str(tmp_path),
                    "quantization": "none",
                    "revision": "main",
                    "trust_remote_code": False,
                },
                "descriptor": descriptor,
                "descriptor_digest": _digest(descriptor),
            }
        )
