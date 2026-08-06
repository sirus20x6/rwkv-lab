"""One component-driven Hugging Face causal/multimodal SFT engine.

The operation deliberately owns only the stable Hugging Face forward boundary.
Model loading, trainability, data identity, batching policy, optimizer mechanics,
evaluation cadence, and checkpoint policy remain registered components.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import shutil
import stat
import tempfile
import time
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from typing import Any

import torch

from rwkv_lab.training_components import (
    AssistantConversationMapperConfiguration,
    AssistantOnlyMapperConfiguration,
    CausalTokensMapperConfiguration,
    ImageCaptionProcessorConfiguration,
    JsonlFrozenImageSplitsConfiguration,
    JsonlImageCaptionConfiguration,
    MappedSample,
    PaddedCollatorConfiguration,
    ProcessedSample,
    RawSample,
)

_MULTIMODAL_TENSOR_KEYS = frozenset(
    {
        "aspect_ratio_ids",
        "aspect_ratio_mask",
        "attention_mask",
        "cross_attention_mask",
        "image_grid_thw",
        "image_sizes",
        "input_ids",
        "pixel_attention_mask",
        "pixel_values",
        "pixel_values_videos",
        "position_ids",
        "rope_deltas",
        "second_per_grid_ts",
        "token_type_ids",
        "video_grid_thw",
    }
)
_ENGINE_STATE_SCHEMA = "rwkv-lab.hf-multimodal-sft-state.v1"
_PENDING_PUBLICATION_DIGEST = (
    "sha256:"
    + hashlib.sha256(b"rwkv-lab.checkpoint-publication-pending/v1").hexdigest()
)


class HFMultimodalSFTError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class HFTrainingStack:
    loaded: Any
    trainability: Any
    trainability_result: Any
    precision: Any
    activation_memory: Any
    model: torch.nn.Module
    optimizer: torch.optim.Optimizer
    learning_rate_schedule: Any
    weight_decay_schedule: Any
    objective: Any
    accumulation: Any


def initialize_training_stack(
    components: Any,
    device: torch.device | str,
    *,
    transformers_module: Any | None = None,
    loaded: Any | None = None,
    precision: Any | None = None,
) -> HFTrainingStack:
    """Initialize in the only safe order for adapter-backed precision policy."""

    if loaded is None:
        loader = components.model_loader(slot="model_loader")
        loaded = loader.load(transformers_module=transformers_module)
    if not loaded.receipt.exact:
        raise HFMultimodalSFTError("HF engine requires an exact base-load receipt")
    trainability = components.trainability()
    trainability_result = trainability.apply(loaded.model)
    if not trainability_result.trainable_parameter_names:
        raise HFMultimodalSFTError("HF training composition produced zero trainables")
    precision = precision or components.precision()
    model = precision.convert_module(trainability_result.model, device)
    activation_memory = components.activation_memory()
    activation_memory.apply(model)
    trainable = tuple(
        parameter for parameter in model.parameters() if parameter.requires_grad
    )
    if not trainable:
        raise HFMultimodalSFTError(
            "HF precision conversion produced zero trainable parameters"
        )
    optimizer = components.optimizer(trainable)
    learning_rate_schedule = components.learning_rate_schedule(optimizer)
    weight_decay_schedule = components.weight_decay_schedule(optimizer)
    return HFTrainingStack(
        loaded=loaded,
        trainability=trainability,
        trainability_result=trainability_result,
        precision=precision,
        activation_memory=activation_memory,
        model=model,
        optimizer=optimizer,
        learning_rate_schedule=learning_rate_schedule,
        weight_decay_schedule=weight_decay_schedule,
        objective=components.objective(),
        accumulation=components.gradient_accumulation(),
    )


def _json_tree(value: Any, *, depth: int = 0) -> Any:
    if depth > 32:
        raise HFMultimodalSFTError("canonical engine state is too deeply nested")
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise HFMultimodalSFTError("canonical engine state is non-finite")
        return value
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise HFMultimodalSFTError("canonical engine state has a non-string key")
        return {
            key: _json_tree(item, depth=depth + 1)
            for key, item in sorted(value.items())
        }
    if isinstance(value, (tuple, list)):
        return [_json_tree(item, depth=depth + 1) for item in value]
    raise HFMultimodalSFTError(
        f"canonical engine state contains unsupported {type(value).__name__}"
    )


def _digest(value: object) -> str:
    encoded = json.dumps(
        _json_tree(value),
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _is_digest(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 71
        and value.startswith("sha256:")
        and all(character in "0123456789abcdef" for character in value[7:])
    )


_FINAL_CONTEXT_COMPONENT_SLOTS = (
    "artifact_renderer",
    "data",
    "evaluator",
    "generation_policy",
    "processor",
    "sample_mapping",
    "test_split",
)


def _final_member_context_digest(sample: ProcessedSample, components: Any) -> str:
    """Bind one member to the exact immutable transform/evaluator semantics."""

    descriptors = components.composition.components
    try:
        component_digests = {
            slot: descriptors[slot].descriptor_digest
            for slot in _FINAL_CONTEXT_COMPONENT_SLOTS
        }
    except (AttributeError, KeyError) as error:
        raise HFMultimodalSFTError(
            "final member context lacks a resolved component descriptor"
        ) from error
    if any(not _is_digest(value) for value in component_digests.values()):
        raise HFMultimodalSFTError(
            "final member context contains an invalid component digest"
        )
    return _digest(
        {
            "api_version": "rwkv-lab.hf-final-member-context/v1",
            "components": component_digests,
            "member_id": sample.sample_id,
            "raw_sample": sample.values,
        }
    )


def _prompt_and_target(
    sample: ProcessedSample,
    configuration: (
        AssistantConversationMapperConfiguration
        | AssistantOnlyMapperConfiguration
    ),
) -> tuple[str | None, str, str]:
    if isinstance(configuration, AssistantConversationMapperConfiguration):
        system_prompt = configuration.system_prompt
        prompt = configuration.user_prompt
    else:
        system_prompt = None
        prompt = (
            sample.values[configuration.prompt_column]
            if configuration.prompt_column
            else configuration.fixed_prompt
        )
    target = sample.values[configuration.target_column]
    if (
        not isinstance(prompt, str)
        or not prompt.strip()
        or not isinstance(target, str)
        or not target.strip()
    ):
        raise HFMultimodalSFTError(
            "assistant-only sample requires nonempty aligned prompt and target text"
        )
    return system_prompt, prompt, target


def _conversation(
    system_prompt: str | None,
    prompt: str,
    target: str | None = None,
) -> list[dict[str, object]]:
    messages: list[dict[str, object]] = []
    if system_prompt is not None:
        messages.append({"role": "system", "content": system_prompt})
    messages.append(
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ],
        }
    )
    if target is not None:
        messages.append({"role": "assistant", "content": target})
    return messages


def _template(processor: Any, messages: object, *, generation: bool) -> str:
    render = getattr(processor, "apply_chat_template", None)
    if not callable(render):
        raise HFMultimodalSFTError(
            "multimodal processor has no apply_chat_template boundary"
        )
    value = render(
        messages,
        tokenize=False,
        add_generation_prompt=generation,
    )
    if not isinstance(value, str) or not value:
        raise HFMultimodalSFTError("multimodal chat template returned invalid text")
    return value


def _token_ids(tokenizer: Any, text: str) -> tuple[int, ...]:
    if callable(tokenizer):
        encoded = tokenizer(text, add_special_tokens=False)
        values = encoded.get("input_ids") if isinstance(encoded, Mapping) else None
    else:
        encode = getattr(tokenizer, "encode", None)
        values = encode(text, add_special_tokens=False) if callable(encode) else None
    if not isinstance(values, (tuple, list)) or any(
        not isinstance(item, int) or isinstance(item, bool) or item < 0
        for item in values
    ):
        raise HFMultimodalSFTError("processor tokenizer returned invalid token IDs")
    return tuple(values)


@dataclass(frozen=True, slots=True)
class HFForwardBatch:
    tensors: Mapping[str, torch.Tensor]
    sample_ids: tuple[str, ...]
    targets: tuple[str, ...]
    source_images: tuple[Any, ...]
    supervised_tokens: int

    def to(self, device: torch.device) -> HFForwardBatch:
        return HFForwardBatch(
            MappingProxyType(
                {
                    key: value.to(device, non_blocking=True)
                    for key, value in self.tensors.items()
                }
            ),
            self.sample_ids,
            self.targets,
            self.source_images,
            self.supervised_tokens,
        )


@dataclass(frozen=True, slots=True)
class HFForwardBatchCodec:
    """Turn registered processed samples into one allowlisted HF tensor mapping."""

    processor_or_tokenizer: Any
    mapper_configuration: (
        AssistantConversationMapperConfiguration
        | AssistantOnlyMapperConfiguration
        | CausalTokensMapperConfiguration
    )
    processor_configuration: ImageCaptionProcessorConfiguration | Any
    collator_configuration: PaddedCollatorConfiguration

    @property
    def multimodal(self) -> bool:
        return isinstance(
            self.mapper_configuration,
            (
                AssistantConversationMapperConfiguration,
                AssistantOnlyMapperConfiguration,
            ),
        )

    def encode(self, samples: Sequence[ProcessedSample]) -> HFForwardBatch:
        if not samples:
            raise HFMultimodalSFTError("HF forward batch cannot be empty")
        if self.multimodal:
            return self._encode_multimodal(samples)
        return self._encode_causal(samples)

    def _encode_causal(self, samples: Sequence[ProcessedSample]) -> HFForwardBatch:
        configuration = self.mapper_configuration
        assert isinstance(configuration, CausalTokensMapperConfiguration)
        mapped: list[MappedSample] = []
        for sample in samples:
            tokens = sample.values.get(configuration.token_column)
            if (
                not isinstance(tokens, list)
                or not tokens
                or any(
                    not isinstance(token, int)
                    or isinstance(token, bool)
                    or token < 0
                    for token in tokens
                )
            ):
                raise HFMultimodalSFTError("causal sample contains invalid token IDs")
            maximum_tokens = min(
                configuration.maximum_tokens,
                self.collator_configuration.maximum_sequence_length,
            )
            if len(tokens) > maximum_tokens:
                raise HFMultimodalSFTError(
                    "causal sample exceeds the declared token policy"
                )
            values = tuple(tokens)
            mapped.append(MappedSample(sample.sample_id, values, values))
        maximum = min(
            max(item.token_length for item in mapped),
            self.collator_configuration.maximum_sequence_length,
        )
        multiple = self.collator_configuration.pad_to_multiple
        length = min(
            ((maximum + multiple - 1) // multiple) * multiple,
            self.collator_configuration.maximum_sequence_length,
        )
        input_ids = torch.full(
            (len(mapped), length),
            self.collator_configuration.pad_token_id,
            dtype=torch.long,
        )
        labels = torch.full(
            (len(mapped), length),
            self.collator_configuration.label_pad_token_id,
            dtype=torch.long,
        )
        attention = torch.zeros((len(mapped), length), dtype=torch.long)
        for index, item in enumerate(mapped):
            count = len(item.input_ids)
            input_ids[index, :count] = torch.tensor(item.input_ids)
            labels[index, :count] = torch.tensor(item.labels)
            attention[index, :count] = 1
        return HFForwardBatch(
            MappingProxyType(
                {
                    "attention_mask": attention,
                    "input_ids": input_ids,
                    "labels": labels,
                }
            ),
            tuple(item.sample_id for item in mapped),
            tuple("" for _ in mapped),
            (),
            int((labels != self.collator_configuration.label_pad_token_id).sum()),
        )

    def _encode_multimodal(self, samples: Sequence[ProcessedSample]) -> HFForwardBatch:
        mapper = self.mapper_configuration
        if not isinstance(
            mapper,
            (
                AssistantConversationMapperConfiguration,
                AssistantOnlyMapperConfiguration,
            ),
        ) or not isinstance(
            self.processor_configuration, ImageCaptionProcessorConfiguration
        ):
            raise HFMultimodalSFTError(
                "multimodal codec requires image-caption processor and assistant mapper"
            )
        processor = self.processor_or_tokenizer
        tokenizer = getattr(processor, "tokenizer", None)
        if tokenizer is None:
            raise HFMultimodalSFTError("multimodal processor exposes no tokenizer")
        padding_side = getattr(tokenizer, "padding_side", None)
        if padding_side not in {"left", "right"}:
            raise HFMultimodalSFTError(
                "multimodal assistant masking requires an explicit padding side"
            )
        prompts: list[str] = []
        full_texts: list[str] = []
        raw_prompts: list[tuple[int, ...]] = []
        raw_full: list[tuple[int, ...]] = []
        images: list[Any] = []
        targets: list[str] = []
        for sample in samples:
            if sample.image is None:
                raise HFMultimodalSFTError("multimodal sample has no decoded image")
            system_prompt, prompt, target = _prompt_and_target(sample, mapper)
            rendered_prompt = _template(
                processor,
                _conversation(system_prompt, prompt),
                generation=True,
            )
            rendered_full = _template(
                processor,
                _conversation(system_prompt, prompt, target),
                generation=False,
            )
            if not rendered_full.startswith(rendered_prompt):
                raise HFMultimodalSFTError(
                    "chat template has no stable assistant-only target boundary"
                )
            prompt_ids = _token_ids(tokenizer, rendered_prompt)
            full_ids = _token_ids(tokenizer, rendered_full)
            eos = getattr(tokenizer, "eos_token_id", None)
            if (
                full_ids[: len(prompt_ids)] != prompt_ids
                or len(full_ids) > mapper.maximum_tokens
            ):
                raise HFMultimodalSFTError(
                    "rendered assistant target is misaligned or exceeds token policy"
                )
            if mapper.append_eos:
                if not isinstance(eos, int) or not full_ids or full_ids[-1] != eos:
                    raise HFMultimodalSFTError(
                        "append_eos requires the chat template to end in the exact EOS token"
                    )
            elif isinstance(eos, int) and full_ids and full_ids[-1] == eos:
                raise HFMultimodalSFTError(
                    "chat template added EOS despite the disabled append_eos policy"
                )
            prompts.append(rendered_prompt)
            full_texts.append(rendered_full)
            raw_prompts.append(prompt_ids)
            raw_full.append(full_ids)
            images.append(sample.image)
            targets.append(target)
        if not callable(processor):
            raise HFMultimodalSFTError("multimodal processor is not callable")
        encoded = processor(
            text=full_texts,
            images=images,
            padding=True,
            pad_to_multiple_of=self.collator_configuration.pad_to_multiple,
            truncation=False,
            return_tensors="pt",
        )
        if not isinstance(encoded, Mapping) or set(encoded).difference(
            _MULTIMODAL_TENSOR_KEYS
        ):
            raise HFMultimodalSFTError(
                "multimodal processor emitted an unsupported forward field"
            )
        tensors = dict(encoded)
        if any(not isinstance(value, torch.Tensor) for value in tensors.values()):
            raise HFMultimodalSFTError(
                "multimodal processor emitted a non-tensor forward value"
            )
        input_ids = tensors.get("input_ids")
        attention = tensors.get("attention_mask")
        if (
            not isinstance(input_ids, torch.Tensor)
            or input_ids.ndim != 2
            or input_ids.shape[0] != len(samples)
            or not isinstance(attention, torch.Tensor)
            or attention.shape != input_ids.shape
            or input_ids.shape[1] > self.collator_configuration.maximum_sequence_length
        ):
            raise HFMultimodalSFTError(
                "multimodal processor produced incompatible token tensors"
            )
        labels = torch.full_like(
            input_ids, self.collator_configuration.label_pad_token_id
        )
        for index, (prompt_ids, full_ids) in enumerate(
            zip(raw_prompts, raw_full, strict=True)
        ):
            token_count = int(attention[index].sum())
            token_start = 0 if padding_side == "right" else input_ids.shape[1] - token_count
            token_end = token_start + token_count
            if token_count > min(
                mapper.maximum_tokens,
                self.collator_configuration.maximum_sequence_length,
            ):
                raise HFMultimodalSFTError(
                    "image-token-expanded sequence exceeds the declared token policy"
                )
            expansion = token_count - len(full_ids)
            boundary = token_start + len(prompt_ids) + expansion
            if not token_start < boundary < token_end:
                raise HFMultimodalSFTError(
                    "image-token expansion erased the assistant target"
                )
            suffix = torch.tensor(
                full_ids[len(prompt_ids) :],
                dtype=input_ids.dtype,
                device=input_ids.device,
            )
            if not torch.equal(input_ids[index, boundary:token_end], suffix):
                raise HFMultimodalSFTError(
                    "image-token expansion changed the assistant target suffix"
                )
            labels[index, boundary:token_end] = input_ids[index, boundary:token_end]
        tensors["labels"] = labels
        supervised = int(
            (labels != self.collator_configuration.label_pad_token_id).sum()
        )
        if supervised < len(samples):
            raise HFMultimodalSFTError("multimodal batch has empty assistant targets")
        return HFForwardBatch(
            MappingProxyType(tensors),
            tuple(sample.sample_id for sample in samples),
            tuple(targets),
            tuple(images),
            supervised,
        )

    def generation_tensors(
        self, samples: Sequence[ProcessedSample]
    ) -> Mapping[str, torch.Tensor]:
        if not self.multimodal:
            raise HFMultimodalSFTError(
                "qualitative generation is not defined for pretokenized causal data"
            )
        mapper = self.mapper_configuration
        assert isinstance(
            mapper,
            (
                AssistantConversationMapperConfiguration,
                AssistantOnlyMapperConfiguration,
            ),
        )
        processor = self.processor_or_tokenizer
        prompts: list[str] = []
        images: list[Any] = []
        for sample in samples:
            system_prompt, prompt, _ = _prompt_and_target(sample, mapper)
            prompts.append(
                _template(
                    processor,
                    _conversation(system_prompt, prompt),
                    generation=True,
                )
            )
            if sample.image is None:
                raise HFMultimodalSFTError("qualitative sample has no decoded image")
            images.append(sample.image)
        encoded = processor(
            text=prompts,
            images=images,
            padding=True,
            pad_to_multiple_of=self.collator_configuration.pad_to_multiple,
            truncation=False,
            return_tensors="pt",
        )
        if not isinstance(encoded, Mapping) or set(encoded).difference(
            _MULTIMODAL_TENSOR_KEYS
        ):
            raise HFMultimodalSFTError(
                "generation processor emitted an unsupported forward field"
            )
        if any(not isinstance(value, torch.Tensor) for value in encoded.values()):
            raise HFMultimodalSFTError("generation processor emitted non-tensor data")
        return MappingProxyType(dict(encoded))


@dataclass(slots=True)
class HFDataRuntime:
    """Deterministic split/sampler/batching runtime shared by both HF routes."""

    training_pipeline: Any
    evaluation_pipeline: Any
    training_records: Mapping[str, RawSample]
    evaluation_records: Mapping[str, RawSample]
    test_records: Mapping[str, RawSample]
    qualitative_ids: tuple[str, ...]
    split_membership_digest: str
    test_membership_digest: str | None = None
    qualitative_identities_digest: str | None = None
    _processed_pending: dict[str, ProcessedSample] = field(default_factory=dict)
    _ready_batches: deque[tuple[str, ...]] = field(default_factory=deque)

    @classmethod
    def build(cls, components: Any) -> HFDataRuntime:
        training = components.data_pipeline(split_slot="split")
        evaluation = components.data_pipeline(split_slot="evaluation_split")
        training.validate_schema()
        evaluation.validate_schema()
        # Both routes consume authority-measured content-root tree identities,
        # checked against workspace.input_content_roots by the operation
        # handler. The component state carries that same fingerprint.
        frozen = isinstance(
            training.source.configuration, JsonlFrozenImageSplitsConfiguration
        )
        if frozen:
            test = components.data_pipeline(split_slot="test_split")
            test.validate_schema()
            raw = training.source.records_for_split("train")
            evaluation_raw = evaluation.source.records_for_split("validation")
            test_raw = test.source.records_for_split("test")
        else:
            test = None
            raw = tuple(training.source.records())
            evaluation_raw = tuple(evaluation.source.records())
            test_raw = ()
        if len(raw) < 2 or len({sample.sample_id for sample in raw}) != len(raw):
            raise HFMultimodalSFTError(
                "HF data source requires at least two uniquely identified records"
            )
        if not frozen and tuple(sample.sample_id for sample in evaluation_raw) != tuple(
            sample.sample_id for sample in raw
        ):
            raise HFMultimodalSFTError(
                "training and evaluation pipelines disagree on source membership"
            )
        identities = tuple(sample.sample_id for sample in raw)
        evaluation_identities = tuple(sample.sample_id for sample in evaluation_raw)
        test_identities = tuple(sample.sample_id for sample in test_raw)
        training_split = training.split_selector.select(identities)
        evaluation_split = evaluation.split_selector.select(
            evaluation_identities if frozen else identities
        )
        test_split = (
            test.split_selector.select(test_identities) if test is not None else None
        )
        selected_sets = (
            set(training_split.selected_ids),
            set(evaluation_split.selected_ids),
            set(test_split.selected_ids) if test_split is not None else set(),
        )
        if frozen and (
            any(
                len(selected) != len(expected)
                for selected, expected in zip(
                    selected_sets,
                    (identities, evaluation_identities, test_identities),
                    strict=True,
                )
            )
            or selected_sets[0] & selected_sets[1]
            or selected_sets[0] & selected_sets[2]
            or selected_sets[1] & selected_sets[2]
        ):
            raise HFMultimodalSFTError("frozen data manifests have cross-split IDs")
        if not frozen and (
            training_split.membership_digest != evaluation_split.membership_digest
            or selected_sets[0] & selected_sets[1]
            or selected_sets[0] | selected_sets[1] != set(identities)
        ):
            raise HFMultimodalSFTError(
                "training and evaluation split components are not complementary"
            )
        by_id = {sample.sample_id: sample for sample in raw}
        evaluation_by_id = {sample.sample_id: sample for sample in evaluation_raw}
        test_by_id = {sample.sample_id: sample for sample in test_raw}
        training.sampler.bind(training_split.selected_ids)
        qualitative = components.qualitative_samples()
        count = qualitative.configuration.sample_count
        qualitative_ids = tuple(evaluation_split.selected_ids[:count])
        binding = qualitative.bind(
            qualitative_ids, selector_digest=evaluation_split.membership_digest
        )
        measured_identities_digest = getattr(binding, "identities_digest", None)
        return cls(
            training_pipeline=training,
            evaluation_pipeline=evaluation,
            training_records=MappingProxyType(
                {sample_id: by_id[sample_id] for sample_id in training_split.selected_ids}
            ),
            evaluation_records=MappingProxyType(
                {
                    sample_id: (evaluation_by_id if frozen else by_id)[sample_id]
                    for sample_id in evaluation_split.selected_ids
                }
            ),
            test_records=MappingProxyType(
                {
                    sample_id: test_by_id[sample_id]
                    for sample_id in (
                        test_split.selected_ids if test_split is not None else ()
                    )
                }
            ),
            qualitative_ids=qualitative_ids,
            split_membership_digest=evaluation_split.membership_digest,
            test_membership_digest=(
                test_split.membership_digest if test_split is not None else None
            ),
            qualitative_identities_digest=measured_identities_digest,
        )

    @staticmethod
    def _image_root(pipeline: Any) -> Path | None:
        configuration = pipeline.source.configuration
        return (
            Path(configuration.image_root)
            if isinstance(
                configuration,
                (JsonlFrozenImageSplitsConfiguration, JsonlImageCaptionConfiguration),
            )
            else None
        )

    def _process(self, pipeline: Any, sample: RawSample) -> ProcessedSample:
        return pipeline.processor.process(
            sample, image_root=self._image_root(pipeline)
        )

    @staticmethod
    def _measurement(sample: ProcessedSample) -> int:
        if sample.image_size is not None:
            return sample.image_size[0] * sample.image_size[1]
        if sample.token_length is not None:
            return sample.token_length
        raise HFMultimodalSFTError("processed sample has no batching measurement")

    def next_training_samples(self) -> tuple[ProcessedSample, ...]:
        if self._ready_batches:
            identifiers = self._ready_batches.popleft()
            return tuple(
                self._process(self.training_pipeline, self.training_records[item])
                for item in identifiers
            )
        while True:
            sample_id = self.training_pipeline.sampler.take(1)[0]
            if sample_id in self._processed_pending:
                # A bucket that cannot fill inside one epoch is flushed before
                # the next epoch can repeat an identity.
                flushed = self.training_pipeline.batching.flush()
                self._ready_batches.extend(flushed)
                if not self._ready_batches:
                    raise HFMultimodalSFTError(
                        "batching policy dropped every pending training sample"
                    )
                identifiers = self._ready_batches.popleft()
                result = tuple(self._processed_pending.pop(item) for item in identifiers)
                return result
            processed = self._process(
                self.training_pipeline, self.training_records[sample_id]
            )
            self._processed_pending[sample_id] = processed
            emitted = self.training_pipeline.batching.add(
                sample_id, measurement=self._measurement(processed)
            )
            if emitted:
                return tuple(self._processed_pending.pop(item) for item in emitted)

    def evaluation_samples(self, *, maximum: int = 0) -> tuple[ProcessedSample, ...]:
        identifiers = tuple(self.evaluation_records)
        if maximum:
            identifiers = identifiers[:maximum]
        return tuple(
            self._process(self.evaluation_pipeline, self.evaluation_records[item])
            for item in identifiers
        )

    def test_samples(self, *, maximum: int = 0) -> tuple[ProcessedSample, ...]:
        identifiers = tuple(self.test_records)
        if maximum:
            identifiers = identifiers[:maximum]
        return tuple(
            self._process(self.evaluation_pipeline, self.test_records[item])
            for item in identifiers
        )

    def qualitative_samples(self) -> tuple[ProcessedSample, ...]:
        return tuple(
            self._process(self.evaluation_pipeline, self.evaluation_records[item])
            for item in self.qualitative_ids
        )

    def runtime_state(self) -> Mapping[str, Any]:
        return MappingProxyType(
            {
                "api_version": "rwkv-lab.hf-data-runtime/v1",
                "ready_batches": [list(batch) for batch in self._ready_batches],
            }
        )

    def component_state(self) -> dict[str, Mapping[str, Any]]:
        state = {
            "data": self.training_pipeline.source.component_state(),
            "sampler": self.training_pipeline.sampler.component_state(),
            "batching": self.training_pipeline.batching.component_state(),
        }
        if self.qualitative_identities_digest is not None:
            state["qualitative_samples"] = {
                "identities_digest": self.qualitative_identities_digest,
                "selector_digest": self.split_membership_digest,
            }
        return state

    def restore(
        self,
        component_state: Mapping[str, Any],
        runtime_state: Mapping[str, Any],
    ) -> None:
        qualitative_state = component_state.get("qualitative_samples")
        if self.qualitative_identities_digest is not None and qualitative_state != {
            "identities_digest": self.qualitative_identities_digest,
            "selector_digest": self.split_membership_digest,
        }:
            raise HFMultimodalSFTError(
                "qualitative held-out binding differs from exact resume"
            )
        self.training_pipeline.source.restore_component_state(component_state["data"])
        self.training_pipeline.sampler.restore_component_state(
            component_state["sampler"]
        )
        batching_state = component_state["batching"]
        pending = batching_state.get("pending_sample_ids")
        if not isinstance(pending, (tuple, list)):
            raise HFMultimodalSFTError("batching resume pending identities are invalid")
        measurements: dict[str, int] = {}
        self._processed_pending.clear()
        for sample_id in pending:
            if sample_id not in self.training_records:
                raise HFMultimodalSFTError(
                    "batching resume identity is outside training membership"
                )
            processed = self._process(
                self.training_pipeline, self.training_records[sample_id]
            )
            self._processed_pending[sample_id] = processed
            measurements[sample_id] = self._measurement(processed)
        self.training_pipeline.batching.restore_component_state(
            batching_state, measurements=measurements
        )
        if set(runtime_state) != {"api_version", "ready_batches"} or runtime_state[
            "api_version"
        ] != "rwkv-lab.hf-data-runtime/v1":
            raise HFMultimodalSFTError("HF data runtime resume state is inexact")
        ready = runtime_state["ready_batches"]
        if not isinstance(ready, (tuple, list)):
            raise HFMultimodalSFTError("HF ready-batch resume state is invalid")
        parsed: deque[tuple[str, ...]] = deque()
        for batch in ready:
            if (
                not isinstance(batch, (tuple, list))
                or not batch
                or len(set(batch)) != len(batch)
                or any(item not in self.training_records for item in batch)
            ):
                raise HFMultimodalSFTError("HF ready-batch identity is invalid")
            parsed.append(tuple(batch))
        self._ready_batches = parsed


@dataclass(frozen=True, slots=True)
class HFEngineState:
    optimizer_step: int
    composition_digest: str
    model_load_receipt_digest: str
    processor_fingerprint: str
    component_state: Mapping[str, Any]
    controls_state: Mapping[str, Any]
    runtime_state: Mapping[str, Any] = field(
        default_factory=lambda: MappingProxyType({})
    )
    qualitative_baselines: Mapping[str, str] = field(
        default_factory=lambda: MappingProxyType({})
    )
    microbatch_in_optimizer_step: int = 0

    def canonical_document(self) -> Mapping[str, Any]:
        if (
            not isinstance(self.optimizer_step, int)
            or isinstance(self.optimizer_step, bool)
            or self.optimizer_step < 0
            or not isinstance(self.microbatch_in_optimizer_step, int)
            or isinstance(self.microbatch_in_optimizer_step, bool)
            or self.microbatch_in_optimizer_step != 0
        ):
            raise HFMultimodalSFTError(
                "exact checkpoints require a valid optimizer-step accumulation boundary"
            )
        if not isinstance(self.runtime_state, Mapping) or not isinstance(
            self.qualitative_baselines, Mapping
        ):
            raise HFMultimodalSFTError("engine runtime state mappings are invalid")
        if any(
            not isinstance(key, str)
            or not key
            or not isinstance(value, str)
            or not value.strip()
            or len(value.encode("utf-8")) > 16 * 1024
            for key, value in self.qualitative_baselines.items()
        ):
            raise HFMultimodalSFTError("engine qualitative baselines are invalid")
        body = {
            "api_version": _ENGINE_STATE_SCHEMA,
            "component_state": _json_tree(self.component_state),
            "composition_digest": self.composition_digest,
            "controls_state": _json_tree(self.controls_state),
            "model_load_receipt_digest": self.model_load_receipt_digest,
            "microbatch_in_optimizer_step": self.microbatch_in_optimizer_step,
            "optimizer_step": self.optimizer_step,
            "processor_fingerprint": self.processor_fingerprint,
            "qualitative_baselines": _json_tree(self.qualitative_baselines),
            "runtime_state": _json_tree(self.runtime_state),
        }
        return MappingProxyType({**body, "state_digest": _digest(body)})


def _staged_objects(
    directory: Path, *, exclude: frozenset[str] = frozenset()
) -> tuple[dict[str, object], ...]:
    """Hash a tree through no-follow directory descriptors.

    The descriptor walk prevents a child symlink or a path-swap between stat
    and open from entering an exact checkpoint receipt.
    """

    objects: list[dict[str, object]] = []
    nofollow = getattr(os, "O_NOFOLLOW", 0)

    def walk(parent: int, prefix: str) -> None:
        try:
            names = sorted(os.listdir(parent))
        except OSError as error:
            raise HFMultimodalSFTError(
                "checkpoint staging tree cannot be enumerated"
            ) from error
        for name in names:
            relative = f"{prefix}/{name}" if prefix else name
            if relative in exclude:
                continue
            try:
                before = os.stat(name, dir_fd=parent, follow_symlinks=False)
            except OSError as error:
                raise HFMultimodalSFTError(
                    "checkpoint staging object cannot be inspected"
                ) from error
            if stat.S_ISLNK(before.st_mode):
                raise HFMultimodalSFTError("checkpoint staging contains a symlink")
            if stat.S_ISDIR(before.st_mode):
                child = os.open(
                    name, os.O_RDONLY | os.O_DIRECTORY | nofollow, dir_fd=parent
                )
                try:
                    opened = os.fstat(child)
                    if (opened.st_dev, opened.st_ino) != (
                        before.st_dev,
                        before.st_ino,
                    ):
                        raise HFMultimodalSFTError(
                            "checkpoint staging directory changed during inspection"
                        )
                    walk(child, relative)
                finally:
                    os.close(child)
                continue
            if not stat.S_ISREG(before.st_mode):
                raise HFMultimodalSFTError(
                    "checkpoint staging contains a non-file"
                )
            descriptor = os.open(name, os.O_RDONLY | nofollow, dir_fd=parent)
            digest = hashlib.sha256()
            size = 0
            try:
                opened = os.fstat(descriptor)
                if (
                    not stat.S_ISREG(opened.st_mode)
                    or (opened.st_dev, opened.st_ino)
                    != (before.st_dev, before.st_ino)
                ):
                    raise HFMultimodalSFTError(
                        "checkpoint staging file changed during inspection"
                    )
                while chunk := os.read(descriptor, 4 * 1024 * 1024):
                    digest.update(chunk)
                    size += len(chunk)
                after = os.fstat(descriptor)
                if size != opened.st_size or after.st_size != opened.st_size:
                    raise HFMultimodalSFTError(
                        "checkpoint staging file changed while hashing"
                    )
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            objects.append(
                {
                    "relative_path": relative,
                    "sha256": "sha256:" + digest.hexdigest(),
                    "size_bytes": size,
                }
            )

    root = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | nofollow)
    try:
        if not stat.S_ISDIR(os.fstat(root).st_mode):
            raise HFMultimodalSFTError("checkpoint staging root is not a directory")
        walk(root, "")
    finally:
        os.close(root)
    return tuple(sorted(objects, key=lambda item: str(item["relative_path"])))


def _object_matches(data: bytes, expected: Mapping[str, Any]) -> bool:
    return expected.get("size_bytes") == len(data) and expected.get(
        "sha256"
    ) == "sha256:" + hashlib.sha256(data).hexdigest()


def _read_staged_json(
    directory: Path,
    relative_path: str,
    *,
    expected: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if "/" in relative_path or relative_path in {"", ".", ".."}:
        raise HFMultimodalSFTError("checkpoint JSON path is invalid")
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    root = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | nofollow)
    descriptor = -1
    try:
        descriptor = os.open(relative_path, os.O_RDONLY | nofollow, dir_fd=root)
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_size > 1024 * 1024:
            raise HFMultimodalSFTError("checkpoint JSON object is invalid")
        pieces: list[bytes] = []
        remaining = info.st_size
        while remaining:
            chunk = os.read(descriptor, min(remaining, 64 * 1024))
            if not chunk:
                raise HFMultimodalSFTError("checkpoint JSON object was truncated")
            pieces.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise HFMultimodalSFTError("checkpoint JSON object changed while reading")
        encoded = b"".join(pieces)
        if expected is not None and not _object_matches(encoded, expected):
            raise HFMultimodalSFTError(
                "checkpoint JSON object disagrees with its receipt"
            )
        value = json.loads(encoded)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise HFMultimodalSFTError("checkpoint JSON object is invalid") from error
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        os.close(root)
    if not isinstance(value, dict):
        raise HFMultimodalSFTError("checkpoint JSON object must be a mapping")
    return value


def _open_staged_binary(
    directory: Path, relative_path: str, *, expected: Mapping[str, Any]
) -> Any:
    if "/" in relative_path or relative_path in {"", ".", ".."}:
        raise HFMultimodalSFTError("checkpoint binary path is invalid")
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    root = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | nofollow)
    try:
        descriptor = os.open(relative_path, os.O_RDONLY | nofollow, dir_fd=root)
    finally:
        os.close(root)
    info = os.fstat(descriptor)
    if not stat.S_ISREG(info.st_mode):
        os.close(descriptor)
        raise HFMultimodalSFTError("checkpoint binary object is invalid")
    digest = hashlib.sha256()
    size = 0
    while chunk := os.read(descriptor, 4 * 1024 * 1024):
        digest.update(chunk)
        size += len(chunk)
    if (
        expected.get("size_bytes") != size
        or expected.get("sha256") != "sha256:" + digest.hexdigest()
    ):
        os.close(descriptor)
        raise HFMultimodalSFTError(
            "checkpoint binary object disagrees with its receipt"
        )
    os.lseek(descriptor, 0, os.SEEK_SET)
    return os.fdopen(descriptor, "rb")


def _exclusive_output(directory: Path, name: str) -> Any:
    descriptor = os.open(
        directory / name,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o440,
    )
    return os.fdopen(descriptor, "wb")


def _save_safetensors_exclusive(
    directory: Path, name: str, tensors: Mapping[str, torch.Tensor]
) -> None:
    from safetensors.torch import save_file

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".trainable-", suffix=".safetensors", dir=directory
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    destination = directory / name
    try:
        save_file(dict(tensors), temporary)
        descriptor = os.open(
            temporary, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        )
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.link(temporary, destination, follow_symlinks=False)
        os.chmod(destination, 0o440, follow_symlinks=False)
    finally:
        temporary.unlink(missing_ok=True)


def _rng_documents() -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    python_state = random.getstate()
    if (
        not isinstance(python_state, tuple)
        or len(python_state) != 3
        or not isinstance(python_state[1], tuple)
    ):
        raise HFMultimodalSFTError("Python RNG returned an unsupported state")
    document: dict[str, Any] = {
        "api_version": "rwkv-lab.hf-multimodal-sft-rng/v1",
        "python": {
            "version": python_state[0],
            "state": list(python_state[1]),
            "gaussian": python_state[2],
        },
        "numpy": None,
    }
    try:
        import numpy

        numpy_state = numpy.random.get_state()
        document["numpy"] = {
            "algorithm": numpy_state[0],
            "state": numpy_state[1].tolist(),
            "position": numpy_state[2],
            "has_gaussian": numpy_state[3],
            "cached_gaussian": numpy_state[4],
        }
    except ImportError:
        pass
    tensors = {"torch_cpu": torch.get_rng_state().cpu()}
    if torch.cuda.is_available():
        tensors.update(
            {
                f"torch_cuda_{index}": value.cpu()
                for index, value in enumerate(torch.cuda.get_rng_state_all())
            }
        )
    return document, tensors


def _restore_rng(document: Mapping[str, Any], tensors: Mapping[str, Any]) -> None:
    if set(document) != {"api_version", "numpy", "python"} or document.get(
        "api_version"
    ) != "rwkv-lab.hf-multimodal-sft-rng/v1":
        raise HFMultimodalSFTError("exact checkpoint RNG document is inexact")
    python_state = document.get("python")
    if not isinstance(python_state, dict) or set(python_state) != {
        "gaussian",
        "state",
        "version",
    }:
        raise HFMultimodalSFTError("exact checkpoint Python RNG state is invalid")
    values = python_state["state"]
    if not isinstance(values, list) or any(
        not isinstance(value, int) or isinstance(value, bool) for value in values
    ):
        raise HFMultimodalSFTError("exact checkpoint Python RNG words are invalid")
    torch_cpu = tensors.get("torch_cpu")
    if not isinstance(torch_cpu, torch.Tensor) or torch_cpu.dtype != torch.uint8:
        raise HFMultimodalSFTError("exact checkpoint torch RNG state is invalid")
    numpy_state = document.get("numpy")
    try:
        cuda_keys = sorted(
            (key for key in tensors if key.startswith("torch_cuda_")),
            key=lambda key: int(key.removeprefix("torch_cuda_")),
        )
    except ValueError as error:
        raise HFMultimodalSFTError("CUDA RNG tensor key is invalid") from error
    expected_tensor_keys = {"torch_cpu", *cuda_keys}
    if set(tensors) != expected_tensor_keys:
        raise HFMultimodalSFTError("exact checkpoint RNG tensor set is invalid")
    random.setstate(
        (python_state["version"], tuple(values), python_state["gaussian"])
    )
    torch.set_rng_state(torch_cpu)
    if numpy_state is not None:
        if not isinstance(numpy_state, dict) or set(numpy_state) != {
            "algorithm",
            "cached_gaussian",
            "has_gaussian",
            "position",
            "state",
        }:
            raise HFMultimodalSFTError("exact checkpoint NumPy RNG state is invalid")
        import numpy

        numpy.random.set_state(
            (
                numpy_state["algorithm"],
                numpy.asarray(numpy_state["state"], dtype=numpy.uint32),
                numpy_state["position"],
                numpy_state["has_gaussian"],
                numpy_state["cached_gaussian"],
            )
        )
    if cuda_keys:
        if not torch.cuda.is_available():
            raise HFMultimodalSFTError("CUDA RNG state cannot be restored without CUDA")
        expected = [f"torch_cuda_{index}" for index in range(torch.cuda.device_count())]
        if cuda_keys != expected:
            raise HFMultimodalSFTError("CUDA RNG device count disagrees")
        torch.cuda.set_rng_state_all([tensors[key] for key in expected])


def _fsync_directories(directory: Path) -> None:
    for path in sorted(
        (item for item in directory.rglob("*") if item.is_dir()),
        key=lambda item: len(item.parts),
        reverse=True,
    ):
        descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def stage_exact_checkpoint(
    directory: Path,
    *,
    model: Any,
    optimizer: torch.optim.Optimizer,
    learning_rate_schedule: Any,
    weight_decay_schedule: Any,
    precision: Any,
    state: HFEngineState,
    sealed_evidence: Mapping[str, Path] | None = None,
) -> str:
    """Write a fresh complete payload for the existing immutable publisher.

    A caller must derive a new attempt-local path for every staging attempt.
    Existing or interrupted paths are never deleted or reused.
    """

    # The format intentionally excludes in-flight gradients. Validate that the
    # caller is at an optimizer boundary before creating any filesystem state.
    state.canonical_document()
    try:
        directory.mkdir(mode=0o750, parents=False, exist_ok=False)
    except FileExistsError as error:
        raise HFMultimodalSFTError(
            "checkpoint staging path already exists and cannot be overwritten"
        ) from error
    trainable_state = {
        name: parameter.detach().cpu().contiguous()
        for name, parameter in sorted(model.named_parameters())
        if parameter.requires_grad
    }
    if not trainable_state:
        raise HFMultimodalSFTError("checkpoint contains no trainable model state")
    _save_safetensors_exclusive(
        directory, "trainable-state.safetensors", trainable_state
    )
    with _exclusive_output(directory, "optimizer.pt") as output:
        torch.save(
            {
                "optimizer": optimizer.state_dict(),
                "learning_rate_schedule": learning_rate_schedule.state_dict(),
                "weight_decay_schedule": weight_decay_schedule.state_dict(),
                "precision": precision.state_dict(),
            },
            output,
        )
        output.flush()
        os.fsync(output.fileno())
    rng_document, rng_tensors = _rng_documents()
    rng_encoded = json.dumps(
        rng_document,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    with _exclusive_output(directory, "rng.json") as output:
        output.write(rng_encoded)
        output.flush()
        os.fsync(output.fileno())
    with _exclusive_output(directory, "rng-tensors.pt") as output:
        torch.save(rng_tensors, output)
        output.flush()
        os.fsync(output.fileno())
    for name, source in sorted((sealed_evidence or {}).items()):
        if not name or Path(name).name != name or not source.is_dir():
            raise HFMultimodalSFTError("sealed checkpoint evidence is invalid")
        shutil.copytree(source, directory / name, copy_function=shutil.copy2)
    document = dict(state.canonical_document())
    encoded = json.dumps(
        document,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    descriptor = os.open(
        directory / "engine-state.json",
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o440,
    )
    with os.fdopen(descriptor, "wb") as output:
        output.write(encoded)
        output.flush()
        os.fsync(output.fileno())
    objects = _staged_objects(directory)
    completion = {
        "api_version": "rwkv-lab.hf-multimodal-sft-staging/v1",
        "engine_state": document["state_digest"],
        "model_load_receipt": state.model_load_receipt_digest,
        "objects_digest": _digest(objects),
        "processor": state.processor_fingerprint,
    }
    completion = {**completion, "completion_digest": _digest(completion)}
    receipt = directory / "staging-complete.json"
    descriptor = os.open(
        receipt,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o440,
    )
    with os.fdopen(descriptor, "wb") as output:
        output.write(
            json.dumps(
                completion,
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
        )
        output.flush()
        os.fsync(output.fileno())
    _fsync_directories(directory)
    return str(completion["completion_digest"])


def restore_exact_checkpoint(
    directory: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    learning_rate_schedule: Any,
    weight_decay_schedule: Any,
    precision: Any,
    composition: Any,
    expected_composition_digest: str,
    expected_model_load_receipt_digest: str,
    expected_processor_fingerprint: str,
    controls_state_validator: Any,
) -> HFEngineState:
    """Restore a publisher-verified payload with exact identity and slot coverage."""
    try:
        root_info = directory.lstat()
    except OSError as error:
        raise HFMultimodalSFTError("exact checkpoint payload is incomplete") from error
    if not stat.S_ISDIR(root_info.st_mode) or stat.S_ISLNK(root_info.st_mode):
        raise HFMultimodalSFTError("exact checkpoint payload is incomplete")
    completion = _read_staged_json(directory, "staging-complete.json")
    if set(completion) != {
        "api_version",
        "completion_digest",
        "engine_state",
        "model_load_receipt",
        "objects_digest",
        "processor",
    }:
        raise HFMultimodalSFTError("exact checkpoint completion receipt is inexact")
    completion_body = dict(completion)
    completion_digest = completion_body.pop("completion_digest")
    objects = _staged_objects(
        directory, exclude=frozenset({"staging-complete.json"})
    )
    object_receipts = {
        str(item["relative_path"]): item for item in objects
    }
    if (
        completion["api_version"]
        != "rwkv-lab.hf-multimodal-sft-staging/v1"
        or completion_digest != _digest(completion_body)
        or completion["objects_digest"] != _digest(objects)
        or completion["model_load_receipt"]
        != expected_model_load_receipt_digest
        or completion["processor"] != expected_processor_fingerprint
    ):
        raise HFMultimodalSFTError("exact checkpoint completion receipt disagrees")
    required_objects = {
        "engine-state.json",
        "optimizer.pt",
        "rng-tensors.pt",
        "rng.json",
        "trainable-state.safetensors",
    }
    if not required_objects.issubset(object_receipts):
        raise HFMultimodalSFTError("exact checkpoint object set is incomplete")
    document = _read_staged_json(
        directory,
        "engine-state.json",
        expected=object_receipts["engine-state.json"],
    )
    if set(document) != {
        "api_version",
        "component_state",
        "composition_digest",
        "controls_state",
        "microbatch_in_optimizer_step",
        "model_load_receipt_digest",
        "optimizer_step",
        "processor_fingerprint",
        "qualitative_baselines",
        "runtime_state",
        "state_digest",
    }:
        raise HFMultimodalSFTError("exact checkpoint engine state is inexact")
    body = dict(document)
    state_digest = body.pop("state_digest")
    if (
        document["api_version"] != _ENGINE_STATE_SCHEMA
        or state_digest != _digest(body)
        or completion["engine_state"] != state_digest
        or document["composition_digest"] != expected_composition_digest
        or document["model_load_receipt_digest"]
        != expected_model_load_receipt_digest
        or document["processor_fingerprint"] != expected_processor_fingerprint
    ):
        raise HFMultimodalSFTError("exact checkpoint identity disagrees")
    component_state = document["component_state"]
    controls_state = document["controls_state"]
    if not isinstance(component_state, dict) or not isinstance(controls_state, dict):
        raise HFMultimodalSFTError("exact checkpoint state mappings are invalid")
    restored_state = HFEngineState(
        optimizer_step=document["optimizer_step"],
        composition_digest=document["composition_digest"],
        model_load_receipt_digest=document["model_load_receipt_digest"],
        processor_fingerprint=document["processor_fingerprint"],
        component_state=component_state,
        controls_state=controls_state,
        runtime_state=document["runtime_state"],
        qualitative_baselines=document["qualitative_baselines"],
        microbatch_in_optimizer_step=document["microbatch_in_optimizer_step"],
    )
    if dict(restored_state.canonical_document()) != document:
        raise HFMultimodalSFTError("exact checkpoint engine state is noncanonical")
    composition.validate_resume_state(component_state)
    controls_state_validator(controls_state)
    with _open_staged_binary(
        directory,
        "trainable-state.safetensors",
        expected=object_receipts["trainable-state.safetensors"],
    ) as source:
        from safetensors import safe_open

        with safe_open(
            f"/proc/self/fd/{source.fileno()}", framework="pt", device="cpu"
        ) as payload:
            names = tuple(payload.keys())
            trainable = {name: payload.get_tensor(name) for name in names}
    expected_parameters = {
        name: parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    if not isinstance(trainable, dict) or set(trainable) != set(expected_parameters):
        raise HFMultimodalSFTError("exact checkpoint trainable tensor set disagrees")
    for name, parameter in expected_parameters.items():
        value = trainable[name]
        if (
            not isinstance(value, torch.Tensor)
            or value.shape != parameter.shape
            or value.dtype != parameter.dtype
        ):
            raise HFMultimodalSFTError(
                f"exact checkpoint trainable tensor {name!r} disagrees"
            )
        parameter.data.copy_(value.to(device=parameter.device))
    with _open_staged_binary(
        directory, "optimizer.pt", expected=object_receipts["optimizer.pt"]
    ) as source:
        mechanics = torch.load(source, map_location="cpu", weights_only=True)
    if not isinstance(mechanics, dict) or set(mechanics) != {
        "optimizer",
        "learning_rate_schedule",
        "weight_decay_schedule",
        "precision",
    }:
        raise HFMultimodalSFTError("exact checkpoint mechanics state is inexact")
    optimizer.load_state_dict(mechanics["optimizer"])
    learning_rate_schedule.load_state_dict(mechanics["learning_rate_schedule"])
    weight_decay_schedule.load_state_dict(mechanics["weight_decay_schedule"])
    precision.load_state_dict(mechanics["precision"])
    rng_document = _read_staged_json(
        directory, "rng.json", expected=object_receipts["rng.json"]
    )
    with _open_staged_binary(
        directory,
        "rng-tensors.pt",
        expected=object_receipts["rng-tensors.pt"],
    ) as source:
        rng_tensors = torch.load(source, map_location="cpu", weights_only=True)
    if not isinstance(rng_tensors, dict):
        raise HFMultimodalSFTError("exact checkpoint RNG tensors are invalid")
    _restore_rng(rng_document, rng_tensors)
    return restored_state


def scalar_loss(outputs: Any) -> torch.Tensor:
    loss = getattr(outputs, "loss", None)
    if not isinstance(loss, torch.Tensor) or loss.ndim != 0 or not torch.isfinite(loss):
        raise HFMultimodalSFTError("HF forward result has no finite scalar loss")
    return loss


def component_causal_loss(
    model: torch.nn.Module,
    objective: Any,
    tensors: Mapping[str, torch.Tensor],
    *,
    ignore_index: int,
) -> torch.Tensor:
    """Apply the registered unshifted objective with an explicit causal shift."""

    labels = tensors.get("labels")
    if not isinstance(labels, torch.Tensor) or labels.ndim != 2:
        raise HFMultimodalSFTError("HF batch has no rank-two labels")
    arguments = {key: value for key, value in tensors.items() if key != "labels"}
    outputs = model(
        **arguments,
        use_cache=False,
        output_hidden_states=True,
        return_dict=True,
    )
    hidden_states = getattr(outputs, "hidden_states", None)
    if not isinstance(hidden_states, (tuple, list)) or not hidden_states:
        raise HFMultimodalSFTError(
            "HF model did not expose hidden states for the registered objective"
        )
    hidden = hidden_states[-1]
    output_embeddings = getattr(model, "get_output_embeddings", None)
    head = output_embeddings() if callable(output_embeddings) else None
    if not isinstance(hidden, torch.Tensor) or not isinstance(head, torch.nn.Module):
        raise HFMultimodalSFTError(
            "HF model has no hidden-state/output-head objective boundary"
        )
    if hidden.shape[:-1] != labels.shape or hidden.shape[1] < 2:
        raise HFMultimodalSFTError("HF hidden states disagree with labels")
    loss = objective(
        hidden[..., :-1, :],
        head,
        labels[..., 1:],
        ignore_index=ignore_index,
    )
    if not isinstance(loss, torch.Tensor) or loss.ndim != 0 or not torch.isfinite(loss):
        raise HFMultimodalSFTError("registered objective returned invalid loss")
    return loss


def weighted_loss(values: Sequence[tuple[float, int]]) -> float:
    if not values or any(
        not math.isfinite(loss) or tokens < 1 for loss, tokens in values
    ):
        raise HFMultimodalSFTError("evaluation produced invalid weighted losses")
    total = sum(tokens for _, tokens in values)
    return sum(loss * tokens for loss, tokens in values) / total


def normalize_token_mean_gradients(
    parameters: Sequence[torch.nn.Parameter], total_supervised_tokens: int
) -> None:
    if (
        not isinstance(total_supervised_tokens, int)
        or isinstance(total_supervised_tokens, bool)
        or total_supervised_tokens < 1
    ):
        raise HFMultimodalSFTError("token-mean gradient denominator is invalid")
    for parameter in parameters:
        if parameter.grad is not None:
            parameter.grad.div_(total_supervised_tokens)


def _autocast(precision: Any, device: torch.device) -> Any:
    dtype = precision.compute_dtype
    if dtype not in {torch.float16, torch.bfloat16}:
        return torch.autocast(device_type=device.type, enabled=False)
    return torch.autocast(device_type=device.type, dtype=dtype)


def _batch_size(pipeline: Any) -> int:
    configuration = pipeline.batching.configuration
    value = getattr(configuration, "batch_size", None)
    if isinstance(value, int):
        return value
    values = getattr(configuration, "batch_sizes", None)
    if isinstance(values, tuple) and values:
        return min(values)
    raise HFMultimodalSFTError("HF batching component has no bounded batch size")


def evaluate_hf_scalar_loss(
    *,
    stack: HFTrainingStack,
    codec: HFForwardBatchCodec,
    data: HFDataRuntime,
    evaluator: Any,
    device: torch.device,
    maximum_examples: int,
    samples: Sequence[ProcessedSample] | None = None,
) -> float:
    samples = (
        tuple(samples)
        if samples is not None
        else data.evaluation_samples(maximum=maximum_examples)
    )
    size = _batch_size(data.evaluation_pipeline)
    values: list[float] = []
    weights: list[float] = []
    training = stack.model.training
    stack.model.eval()
    try:
        with torch.no_grad():
            for start in range(0, len(samples), size):
                batch = codec.encode(samples[start : start + size]).to(device)
                with _autocast(stack.precision, device):
                    loss = component_causal_loss(
                        stack.model,
                        stack.objective,
                        batch.tensors,
                        ignore_index=codec.collator_configuration.label_pad_token_id,
                    )
                values.append(float(stack.precision.reduce(loss).cpu()))
                weights.append(float(batch.supervised_tokens))
    finally:
        stack.model.train(training)
    evaluator_values = tuple(values)
    if not evaluator_values:
        raise HFMultimodalSFTError("HF scalar evaluation selected no examples")
    return evaluator.reduce(values, weights)


def _decode(tokenizer: Any, values: torch.Tensor | Sequence[int]) -> str:
    decode = getattr(tokenizer, "decode", None)
    if not callable(decode):
        raise HFMultimodalSFTError("HF tokenizer has no decode boundary")
    if isinstance(values, torch.Tensor):
        values = values.detach().cpu().tolist()
    text = decode(values, skip_special_tokens=True)
    if not isinstance(text, str) or not text.strip():
        raise HFMultimodalSFTError("HF qualitative generation decoded empty text")
    return text.strip()


def generate_hf_captions(
    *,
    stack: HFTrainingStack,
    codec: HFForwardBatchCodec,
    samples: Sequence[ProcessedSample],
    device: torch.device,
    maximum_new_tokens: int = 128,
    use_cache: bool = True,
) -> tuple[tuple[str, str, str], ...]:
    """Return (teacher target, generated text, prompt digest) for fixed samples."""

    model = stack.model
    generate = getattr(model, "generate", None)
    if not callable(generate):
        raise HFMultimodalSFTError("HF model has no qualitative generate boundary")
    training = model.training
    model.eval()
    result: list[tuple[str, str, str]] = []
    try:
        with torch.no_grad(), _autocast(stack.precision, device):
            if codec.multimodal:
                tensors = {
                    key: value.to(device)
                    for key, value in codec.generation_tensors(samples).items()
                }
                output = generate(
                    **tensors,
                    max_new_tokens=maximum_new_tokens,
                    do_sample=False,
                    use_cache=use_cache,
                )
                sequences = getattr(output, "sequences", output)
                if not isinstance(sequences, torch.Tensor) or sequences.ndim != 2:
                    raise HFMultimodalSFTError(
                        "HF qualitative generation returned invalid sequences"
                    )
                tokenizer = getattr(codec.processor_or_tokenizer, "tokenizer", None)
                width = tensors["input_ids"].shape[1]
                mapper = codec.mapper_configuration
                assert isinstance(
                    mapper,
                    (
                        AssistantConversationMapperConfiguration,
                        AssistantOnlyMapperConfiguration,
                    ),
                )
                for index, sample in enumerate(samples):
                    system_prompt, prompt, target = _prompt_and_target(sample, mapper)
                    result.append(
                        (
                            target.strip(),
                            _decode(tokenizer, sequences[index, width:]),
                            _digest(
                                {
                                    "system_prompt": system_prompt,
                                    "user_prompt": prompt,
                                }
                            ),
                        )
                    )
            else:
                mapper = codec.mapper_configuration
                assert isinstance(mapper, CausalTokensMapperConfiguration)
                tokenizer = codec.processor_or_tokenizer
                for sample in samples:
                    tokens = sample.values[mapper.token_column]
                    if not isinstance(tokens, list) or len(tokens) < 2:
                        raise HFMultimodalSFTError(
                            "causal qualitative sample requires at least two tokens"
                        )
                    boundary = max(1, len(tokens) // 2)
                    prefix = torch.tensor(tokens[:boundary], device=device).unsqueeze(0)
                    attention = torch.ones_like(prefix)
                    output = generate(
                        input_ids=prefix,
                        attention_mask=attention,
                        max_new_tokens=maximum_new_tokens,
                        do_sample=False,
                        use_cache=use_cache,
                    )
                    sequences = getattr(output, "sequences", output)
                    if not isinstance(sequences, torch.Tensor):
                        raise HFMultimodalSFTError(
                            "causal qualitative generation returned invalid sequences"
                        )
                    result.append(
                        (
                            _decode(tokenizer, tokens[boundary:]),
                            _decode(tokenizer, sequences[0, boundary:]),
                            _digest({"tokens": tokens[:boundary]}),
                        )
                    )
    finally:
        model.train(training)
    return tuple(result)


def _checkpoint_model_state_digest(
    *, model_load_receipt: str, checkpoint_artifact_id: str, manifest_digest: str
) -> str:
    if (
        not _is_digest(model_load_receipt)
        or not checkpoint_artifact_id
        or not _is_digest(manifest_digest)
    ):
        raise HFMultimodalSFTError("checkpoint model-state identity is invalid")
    return _digest(
        {
            "api_version": "rwkv-lab.hf-checkpoint-model-state/v1",
            "checkpoint_artifact_id": checkpoint_artifact_id,
            "checkpoint_manifest_digest": manifest_digest,
            "model_load_receipt": model_load_receipt,
        }
    )


def _component_state(
    *,
    components: Any,
    stack: HFTrainingStack,
    data: HFDataRuntime,
    checkpoint_policy_state: Mapping[str, Any],
    optimizer_step: int,
) -> dict[str, Any]:
    names = tuple(stack.trainability_result.trainable_parameter_names)
    parameter_manifest = _digest(names)
    trainability_state = stack.trainability_result.component_state(
        adapter_state_manifest=(
            parameter_manifest if stack.trainability_result.adapter_backed else None
        )
    )
    state: dict[str, Any] = {
        **data.component_state(),
        "model_loader": stack.loaded.component_state(),
        "optimizer": {"parameter_state_manifest": parameter_manifest},
        "trainability": trainability_state,
        "activation_memory": stack.activation_memory.component_state(),
        "learning_rate": {"cursor": optimizer_step},
        "checkpoint_policy": dict(checkpoint_policy_state),
    }
    return dict(components.composition.validate_resume_state(state))


def _source_image_path(data: HFDataRuntime, sample: ProcessedSample) -> Path | None:
    configuration = data.evaluation_pipeline.source.configuration
    processor = data.evaluation_pipeline.processor.configuration
    if not isinstance(
        configuration,
        (JsonlFrozenImageSplitsConfiguration, JsonlImageCaptionConfiguration),
    ) or not isinstance(processor, ImageCaptionProcessorConfiguration):
        return None
    value = sample.values[processor.image_column]
    if not isinstance(value, str):
        raise HFMultimodalSFTError("qualitative image path is invalid")
    path = (Path(configuration.image_root) / value).resolve(strict=True)
    path.relative_to(Path(configuration.image_root).resolve(strict=True))
    return path


def _text_card(path: Path, text: str) -> None:
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (1024, 576), "#111827")
    drawing = ImageDraw.Draw(image)
    lines: list[str] = []
    remaining = text
    while remaining:
        lines.append(remaining[:100])
        remaining = remaining[100:]
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o440,
    )
    with os.fdopen(descriptor, "wb") as output:
        drawing.multiline_text((32, 32), "\n".join(lines[:20]), fill="white", spacing=8)
        image.save(output, format="PNG")
        output.flush()
        os.fsync(output.fileno())


def _write_test_caption_evidence(
    *,
    directory: Path,
    stack: HFTrainingStack,
    codec: HFForwardBatchCodec,
    samples: Sequence[ProcessedSample],
    device: torch.device,
    step: int,
    model_load_receipt: str,
    checkpoint_artifact_id: str,
    checkpoint_manifest_digest: str,
    model_state_digest: str,
    split_membership_digest: str,
    decode_policy_digest: str,
    model_state_mode: str,
    maximum_new_tokens: int,
    generation_batch_size: int,
    use_cache: bool,
    progress: Callable[[int, int, int, float], None] | None = None,
) -> Path:
    directory.mkdir(mode=0o750, exist_ok=True)
    if model_state_mode not in {"base_adapters_disabled", "trained"}:
        raise HFMultimodalSFTError("test evidence model-state mode is invalid")
    if (
        not checkpoint_artifact_id
        or not _is_digest(checkpoint_manifest_digest)
        or not _is_digest(model_state_digest)
    ):
        raise HFMultimodalSFTError("test evidence model-state digest is invalid")
    path = directory / "captions.jsonl"
    partial_path = directory / "captions.partial.jsonl"
    identity_path = directory / "identity.json"
    sample_ids = tuple(sample.sample_id for sample in samples)
    if not sample_ids or len(set(sample_ids)) != len(sample_ids):
        raise HFMultimodalSFTError("test caption evidence IDs are empty or duplicate")
    identity = {
        "api_version": "rwkv-lab.hf-test-caption-evidence-identity/v1",
        "checkpoint_artifact_id": checkpoint_artifact_id,
        "checkpoint_manifest_digest": checkpoint_manifest_digest,
        "decode_policy_digest": decode_policy_digest,
        "model_load_receipt": model_load_receipt,
        "model_state_digest": model_state_digest,
        "model_state_mode": model_state_mode,
        "records": len(samples),
        "sample_ids_digest": _digest(sample_ids),
        "split_membership_digest": split_membership_digest,
        "step": step,
    }
    identity_encoded = json.dumps(
        identity, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    identity_digest = "sha256:" + hashlib.sha256(identity_encoded).hexdigest()
    if identity_path.exists():
        if identity_path.read_bytes() != identity_encoded:
            raise HFMultimodalSFTError(
                "partial test caption evidence identity is stale or mismatched"
            )
    else:
        with _exclusive_output(directory, "identity.json") as output:
            output.write(identity_encoded)
            output.flush()
            os.fsync(output.fileno())
    receipt_path = directory / "receipt.json"
    if receipt_path.is_file() and path.is_file():
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        encoded_final = path.read_bytes()
        try:
            final_records = tuple(
                json.loads(line) for line in encoded_final.splitlines()
            )
        except json.JSONDecodeError as error:
            raise HFMultimodalSFTError(
                "final test caption evidence is malformed"
            ) from error
        final_records_valid = (
            encoded_final.endswith(b"\n")
            and len(final_records) == len(samples)
            and all(
                isinstance(record, dict)
                and set(record)
                == {
                    "error_code",
                    "prompt_digest",
                    "sample_id",
                    "status",
                    "step",
                    "target",
                    "text",
                    "wall_time_ms",
                }
                and record["sample_id"] == expected_id
                and record["status"] == "complete"
                and record["error_code"] is None
                and isinstance(record["step"], int)
                and not isinstance(record["step"], bool)
                and record["step"] == step
                and isinstance(record["wall_time_ms"], (int, float))
                and not isinstance(record["wall_time_ms"], bool)
                and math.isfinite(float(record["wall_time_ms"]))
                and float(record["wall_time_ms"]) >= 0.0
                and all(
                    isinstance(record[field], str) and record[field].strip()
                    for field in ("prompt_digest", "target", "text")
                )
                and len(record["prompt_digest"]) == 71
                and record["prompt_digest"].startswith("sha256:")
                and all(
                    character in "0123456789abcdef"
                    for character in record["prompt_digest"][7:]
                )
                for record, expected_id in zip(
                    final_records, sample_ids, strict=True
                )
            )
        )
        if (
            set(receipt)
            != {
                "api_version",
                "captions_sha256",
                "checkpoint_artifact_id",
                "checkpoint_manifest_digest",
                "decode_policy_digest",
                "failures",
                "identity_sha256",
                "model_load_receipt",
                "model_state_digest",
                "model_state_mode",
                "records",
                "split_membership_digest",
                "step",
                "total_wall_time_ms",
            }
            or receipt.get("api_version")
            != "rwkv-lab.hf-test-caption-evidence/v1"
            or receipt.get("records") != len(samples)
            or receipt.get("failures") != 0
            or receipt.get("identity_sha256") != identity_digest
            or receipt.get("checkpoint_artifact_id") != checkpoint_artifact_id
            or receipt.get("checkpoint_manifest_digest")
            != checkpoint_manifest_digest
            or receipt.get("model_load_receipt") != model_load_receipt
            or receipt.get("model_state_digest") != model_state_digest
            or receipt.get("model_state_mode") != model_state_mode
            or receipt.get("split_membership_digest") != split_membership_digest
            or receipt.get("decode_policy_digest") != decode_policy_digest
            or receipt.get("step") != step
            or isinstance(receipt.get("total_wall_time_ms"), bool)
            or not isinstance(receipt.get("total_wall_time_ms"), (int, float))
            or not math.isfinite(float(receipt["total_wall_time_ms"]))
            or float(receipt["total_wall_time_ms"]) < 0.0
            or receipt.get("captions_sha256")
            != "sha256:" + hashlib.sha256(encoded_final).hexdigest()
            or not final_records_valid
        ):
            raise HFMultimodalSFTError("final test caption evidence is inconsistent")
        partial_path.unlink(missing_ok=True)
        directory_descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
        return directory

    allowed_ids = frozenset(sample_ids)
    latest: dict[str, dict[str, Any]] = {}
    total_wall_time_ms = 0.0
    failures = 0
    if partial_path.exists():
        encoded = partial_path.read_bytes()
        valid_end = 0
        chunks = encoded.splitlines(keepends=True)
        for offset, chunk in enumerate(chunks):
            complete_line = chunk.endswith(b"\n")
            try:
                record = json.loads(chunk)
            except json.JSONDecodeError as error:
                if offset != len(chunks) - 1:
                    raise HFMultimodalSFTError(
                        "partial test caption evidence is malformed before EOF"
                    ) from error
                with partial_path.open("r+b") as output:
                    output.truncate(valid_end)
                    output.flush()
                    os.fsync(output.fileno())
                break
            if (
                not isinstance(record, dict)
                or set(record)
                != {
                    "error_code",
                    "prompt_digest",
                    "sample_id",
                    "status",
                    "step",
                    "target",
                    "text",
                    "wall_time_ms",
                }
                or record.get("sample_id") not in allowed_ids
                or not isinstance(record.get("sample_id"), str)
                or isinstance(record.get("step"), bool)
                or not isinstance(record.get("step"), int)
                or record.get("step") != step
                or record.get("status") not in {"complete", "failed"}
                or isinstance(record.get("wall_time_ms"), bool)
                or not isinstance(record.get("wall_time_ms"), (int, float))
                or not math.isfinite(float(record["wall_time_ms"]))
                or float(record["wall_time_ms"]) < 0.0
            ):
                raise HFMultimodalSFTError(
                    "partial test caption evidence record is inexact"
                )
            total_wall_time_ms += float(record.get("wall_time_ms", 0.0))
            if record["status"] == "complete":
                if not all(
                    isinstance(record[field], str) and record[field].strip()
                    for field in ("prompt_digest", "target", "text")
                ) or record["error_code"] is not None or not (
                    len(record["prompt_digest"]) == 71
                    and record["prompt_digest"].startswith("sha256:")
                    and all(
                        character in "0123456789abcdef"
                        for character in record["prompt_digest"][7:]
                    )
                ):
                    raise HFMultimodalSFTError(
                        "partial completed caption evidence is empty"
                    )
            elif not (
                isinstance(record["error_code"], str)
                and record["error_code"].strip()
                and record["prompt_digest"] == ""
                and record["target"] == ""
                and record["text"] == ""
            ):
                raise HFMultimodalSFTError(
                    "partial failed caption evidence is inconsistent"
                )
            latest[record["sample_id"]] = record
            valid_end += len(chunk)
            if not complete_line:
                with partial_path.open("ab") as output:
                    output.write(b"\n")
                    output.flush()
                    os.fsync(output.fileno())

    records: list[dict[str, Any] | None] = [None] * len(samples)
    for index, sample in enumerate(samples):
        candidate = latest.get(sample.sample_id)
        if candidate is not None and candidate["status"] == "complete":
            records[index] = candidate
    completed = sum(record is not None for record in records)
    failures = sum(record["status"] == "failed" for record in latest.values())
    if progress is not None:
        progress(completed, len(samples), failures, total_wall_time_ms)
    resolution_buckets: dict[tuple[int, int] | None, list[int]] = {}
    for index, sample in enumerate(samples):
        if records[index] is not None:
            continue
        key = (
            (
                (sample.image_size[0] + 255) // 256,
                (sample.image_size[1] + 255) // 256,
            )
            if sample.image_size is not None
            else None
        )
        resolution_buckets.setdefault(key, []).append(index)
    batches = tuple(
        tuple(indices[start : start + generation_batch_size])
        for indices in resolution_buckets.values()
        for start in range(0, len(indices), generation_batch_size)
    )
    partial_descriptor = os.open(
        partial_path,
        os.O_WRONLY | os.O_CREAT | os.O_APPEND | getattr(os, "O_NOFOLLOW", 0),
        0o640,
    )
    with os.fdopen(partial_descriptor, "ab") as partial:
        for indices in batches:
            batch = tuple(samples[index] for index in indices)
            started = time.perf_counter()
            try:
                generated = generate_hf_captions(
                    stack=stack,
                    codec=codec,
                    samples=batch,
                    device=device,
                    maximum_new_tokens=maximum_new_tokens,
                    use_cache=use_cache,
                )
                if len(generated) != len(batch):
                    raise HFMultimodalSFTError(
                        "test caption batch returned the wrong record count"
                    )
                status = "complete"
                error_code = None
            except Exception:  # noqa: BLE001 - persist the failure, then fence
                generated = tuple(("", "", "") for _ in batch)
                status = "failed"
                error_code = "generation_failed"
                failures += len(batch)
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            total_wall_time_ms += elapsed_ms
            per_item_ms = elapsed_ms / len(batch)
            for index, sample, evidence in zip(
                indices, batch, generated, strict=True
            ):
                target, current, prompt_digest = evidence
                record = {
                    "error_code": error_code,
                    "prompt_digest": prompt_digest,
                    "sample_id": sample.sample_id,
                    "status": status,
                    "step": step,
                    "target": target,
                    "text": current,
                    "wall_time_ms": per_item_ms,
                }
                partial.write(
                    json.dumps(
                        record,
                        allow_nan=False,
                        ensure_ascii=False,
                        separators=(",", ":"),
                        sort_keys=True,
                    ).encode("utf-8")
                    + b"\n"
                )
                latest[sample.sample_id] = record
                if status == "complete":
                    records[index] = record
            partial.flush()
            os.fsync(partial.fileno())
            failures = sum(
                record["status"] == "failed" for record in latest.values()
            )
            if status == "failed":
                raise HFMultimodalSFTError(
                    f"HF test caption gate recorded {failures} failed generations"
                )
            completed += len(batch)
            if progress is not None:
                progress(completed, len(samples), failures, total_wall_time_ms)
    encoded_records = bytearray()
    for record in records:
        if record is None:
            raise HFMultimodalSFTError("test caption evidence is incomplete")
        encoded_records.extend(
            json.dumps(
                record,
                allow_nan=False,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
            + b"\n"
        )
    if path.exists():
        if path.read_bytes() != encoded_records:
            raise HFMultimodalSFTError(
                "unsealed final test caption evidence is inconsistent"
            )
    else:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=".captions-complete-", suffix=".jsonl", dir=directory
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as output:
                output.write(encoded_records)
                output.flush()
                os.fsync(output.fileno())
            os.chmod(temporary, 0o440, follow_symlinks=False)
            os.replace(temporary, path)
            directory_descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory_descriptor)
            finally:
                os.close(directory_descriptor)
        finally:
            temporary.unlink(missing_ok=True)
    file_digest = "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
    receipt = {
        "api_version": "rwkv-lab.hf-test-caption-evidence/v1",
        "captions_sha256": file_digest,
        "checkpoint_artifact_id": checkpoint_artifact_id,
        "checkpoint_manifest_digest": checkpoint_manifest_digest,
        "decode_policy_digest": decode_policy_digest,
        "failures": failures,
        "identity_sha256": identity_digest,
        "model_load_receipt": model_load_receipt,
        "model_state_digest": model_state_digest,
        "model_state_mode": model_state_mode,
        "records": len(records),
        "split_membership_digest": split_membership_digest,
        "step": step,
        "total_wall_time_ms": total_wall_time_ms,
    }
    with _exclusive_output(directory, "receipt.json") as output:
        output.write(
            json.dumps(
                receipt,
                allow_nan=False,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
        )
        output.flush()
        os.fsync(output.fileno())
    partial_path.unlink(missing_ok=True)
    directory_descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory_descriptor)
    finally:
        os.close(directory_descriptor)
    return directory


def _bundle_test_caption_evidence(
    *,
    directory: Path,
    baseline: Path,
    final: Path,
    split_membership_digest: str,
    baseline_checkpoint_artifact_id: str,
    baseline_checkpoint_manifest_digest: str,
    final_checkpoint_artifact_id: str,
    final_checkpoint_manifest_digest: str,
    final_model_state_digest: str,
) -> Path:
    if (
        not baseline_checkpoint_artifact_id
        or not final_checkpoint_artifact_id
        or not _is_digest(baseline_checkpoint_manifest_digest)
        or not _is_digest(final_checkpoint_manifest_digest)
        or not _is_digest(final_model_state_digest)
    ):
        raise HFMultimodalSFTError("test evidence bundle identity is invalid")
    sources = {"baseline": baseline, "final": final}
    sealed_names = frozenset(
        {"captions.jsonl", "identity.json", "receipt.json", "scalar-evaluation.json"}
    )
    nofollow = getattr(os, "O_NOFOLLOW", 0)

    def regular_bytes(parent: int, name: str) -> bytes:
        try:
            descriptor = os.open(name, os.O_RDONLY | nofollow, dir_fd=parent)
        except OSError as error:
            raise HFMultimodalSFTError(
                "sealed test caption evidence cannot be opened without following links"
            ) from error
        try:
            before = os.fstat(descriptor)
            if not stat.S_ISREG(before.st_mode):
                raise HFMultimodalSFTError(
                    "sealed test caption evidence contains a special node"
                )
            chunks: list[bytes] = []
            size = 0
            while chunk := os.read(descriptor, 4 * 1024 * 1024):
                chunks.append(chunk)
                size += len(chunk)
            after = os.fstat(descriptor)
            if (
                size != before.st_size
                or after.st_size != before.st_size
                or (after.st_dev, after.st_ino) != (before.st_dev, before.st_ino)
            ):
                raise HFMultimodalSFTError(
                    "sealed test caption evidence changed while reading"
                )
            return b"".join(chunks)
        finally:
            os.close(descriptor)

    def sealed_bytes_from_descriptor(root: int) -> dict[str, bytes]:
        try:
            names = tuple(os.listdir(root))
        except OSError as error:
            raise HFMultimodalSFTError(
                "test caption evidence bundle is incomplete"
            ) from error
        if set(names) != sealed_names:
            raise HFMultimodalSFTError(
                "sealed test caption evidence has an inexact file layout"
            )
        return {name: regular_bytes(root, name) for name in sorted(sealed_names)}

    def sealed_bytes(root: Path) -> dict[str, bytes]:
        try:
            descriptor = os.open(root, os.O_RDONLY | os.O_DIRECTORY | nofollow)
        except OSError as error:
            raise HFMultimodalSFTError(
                "sealed test caption evidence cannot be a symlink"
            ) from error
        try:
            return sealed_bytes_from_descriptor(descriptor)
        finally:
            os.close(descriptor)

    def json_mapping(contents: bytes, label: str) -> dict[str, Any]:
        try:
            document = json.loads(contents)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise HFMultimodalSFTError(
                f"sealed {label} evidence is malformed"
            ) from error
        if not isinstance(document, dict):
            raise HFMultimodalSFTError(f"sealed {label} evidence is not a mapping")
        return document

    def caption_records(contents: bytes) -> tuple[dict[str, Any], ...]:
        if not contents.endswith(b"\n"):
            raise HFMultimodalSFTError("sealed caption evidence is incomplete")
        try:
            records = tuple(json.loads(line) for line in contents.splitlines())
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise HFMultimodalSFTError("sealed caption evidence is malformed") from error
        if any(not isinstance(record, dict) for record in records):
            raise HFMultimodalSFTError("sealed caption evidence is malformed")
        return records

    def validate_source(
        name: str, objects: Mapping[str, bytes]
    ) -> tuple[dict[str, Any], tuple[dict[str, Any], ...]]:
        evidence = json_mapping(objects["receipt.json"], f"{name} receipt")
        identity = json_mapping(objects["identity.json"], f"{name} identity")
        scalar = json_mapping(
            objects["scalar-evaluation.json"], f"{name} scalar receipt"
        )
        records = caption_records(objects["captions.jsonl"])
        if (
            set(evidence)
            != {
                "api_version",
                "captions_sha256",
                "checkpoint_artifact_id",
                "checkpoint_manifest_digest",
                "decode_policy_digest",
                "failures",
                "identity_sha256",
                "model_load_receipt",
                "model_state_digest",
                "model_state_mode",
                "records",
                "split_membership_digest",
                "step",
                "total_wall_time_ms",
            }
            or set(identity)
            != {
                "api_version",
                "checkpoint_artifact_id",
                "checkpoint_manifest_digest",
                "decode_policy_digest",
                "model_load_receipt",
                "model_state_digest",
                "model_state_mode",
                "records",
                "sample_ids_digest",
                "split_membership_digest",
                "step",
            }
            or set(scalar)
            != {
                "api_version",
                "coverage",
                "loss",
                "records",
                "split_membership_digest",
                "step",
            }
            or evidence["api_version"] != "rwkv-lab.hf-test-caption-evidence/v1"
            or identity["api_version"]
            != "rwkv-lab.hf-test-caption-evidence-identity/v1"
            or scalar["api_version"] != "rwkv-lab.hf-test-scalar-evidence/v1"
            or scalar["coverage"] != "full"
            or not isinstance(evidence["records"], int)
            or isinstance(evidence["records"], bool)
            or not isinstance(evidence["step"], int)
            or isinstance(evidence["step"], bool)
            or not isinstance(evidence["total_wall_time_ms"], (int, float))
            or isinstance(evidence["total_wall_time_ms"], bool)
            or not math.isfinite(float(evidence["total_wall_time_ms"]))
            or float(evidence["total_wall_time_ms"]) < 0.0
            or not isinstance(scalar["loss"], (int, float))
            or isinstance(scalar["loss"], bool)
            or not math.isfinite(float(scalar["loss"]))
            or evidence["captions_sha256"]
            != "sha256:" + hashlib.sha256(objects["captions.jsonl"]).hexdigest()
            or evidence["identity_sha256"]
            != "sha256:" + hashlib.sha256(objects["identity.json"]).hexdigest()
            or evidence["failures"] != 0
            or evidence["records"] != len(records)
            or evidence["records"] != identity["records"]
            or evidence["records"] != scalar["records"]
            or evidence["split_membership_digest"] != split_membership_digest
            or identity["split_membership_digest"] != split_membership_digest
            or scalar["split_membership_digest"] != split_membership_digest
            or evidence["step"] != identity["step"]
            or evidence["step"] != scalar["step"]
            or evidence["checkpoint_artifact_id"]
            != identity["checkpoint_artifact_id"]
            or evidence["checkpoint_manifest_digest"]
            != identity["checkpoint_manifest_digest"]
            or evidence["decode_policy_digest"] != identity["decode_policy_digest"]
            or evidence["model_load_receipt"] != identity["model_load_receipt"]
            or evidence["model_state_digest"] != identity["model_state_digest"]
            or evidence["model_state_mode"] != identity["model_state_mode"]
        ):
            raise HFMultimodalSFTError(
                f"sealed {name} evidence receipts are inconsistent"
            )
        expected_mode = "base_adapters_disabled" if name == "baseline" else "trained"
        if (
            evidence["model_state_mode"] != expected_mode
            or (name == "baseline" and evidence["step"] != 0)
            or (name == "final" and evidence["step"] <= 0)
        ):
            raise HFMultimodalSFTError(
                f"sealed {name} evidence has the wrong model state"
            )
        sample_ids = tuple(record.get("sample_id") for record in records)
        if (
            any(
                set(record)
                != {
                    "error_code",
                    "prompt_digest",
                    "sample_id",
                    "status",
                    "step",
                    "target",
                    "text",
                    "wall_time_ms",
                }
                or record["status"] != "complete"
                or record["error_code"] is not None
                or not isinstance(record["step"], int)
                or isinstance(record["step"], bool)
                or record["step"] != evidence["step"]
                or not isinstance(record["wall_time_ms"], (int, float))
                or isinstance(record["wall_time_ms"], bool)
                or not math.isfinite(float(record["wall_time_ms"]))
                or float(record["wall_time_ms"]) < 0.0
                or not all(
                    isinstance(record[field], str) and record[field].strip()
                    for field in ("prompt_digest", "sample_id", "target", "text")
                )
                for record in records
            )
            or len(set(sample_ids)) != len(sample_ids)
            or identity["sample_ids_digest"] != _digest(sample_ids)
        ):
            raise HFMultimodalSFTError(
                f"sealed {name} caption identities are inconsistent"
            )
        return identity, records

    source_objects = {name: sealed_bytes(source) for name, source in sources.items()}
    validated = {
        name: validate_source(name, objects) for name, objects in source_objects.items()
    }
    baseline_identity, baseline_records = validated["baseline"]
    final_identity, final_records = validated["final"]
    if (
        baseline_identity["checkpoint_artifact_id"]
        != baseline_checkpoint_artifact_id
        or baseline_identity["checkpoint_manifest_digest"]
        != baseline_checkpoint_manifest_digest
        or final_identity["checkpoint_artifact_id"] != final_checkpoint_artifact_id
        or final_identity["checkpoint_manifest_digest"]
        != final_checkpoint_manifest_digest
        or final_identity["model_state_digest"] != final_model_state_digest
        or any(
            baseline_identity[field] != final_identity[field]
            for field in (
                "decode_policy_digest",
                "model_load_receipt",
                "records",
                "sample_ids_digest",
                "split_membership_digest",
            )
        )
        or tuple(
            (record["sample_id"], record["prompt_digest"], record["target"])
            for record in baseline_records
        )
        != tuple(
            (record["sample_id"], record["prompt_digest"], record["target"])
            for record in final_records
        )
    ):
        raise HFMultimodalSFTError(
            "baseline and final test evidence are not identity-aligned"
        )
    receipt = {
        "api_version": "rwkv-lab.hf-test-caption-evidence-bundle/v1",
        "baseline_checkpoint_artifact_id": baseline_checkpoint_artifact_id,
        "baseline_checkpoint_manifest_digest": baseline_checkpoint_manifest_digest,
        "final_checkpoint_artifact_id": final_checkpoint_artifact_id,
        "final_checkpoint_manifest_digest": final_checkpoint_manifest_digest,
        "final_model_state_digest": final_model_state_digest,
        "split_membership_digest": split_membership_digest,
    }
    # Compute the immutable receipt from the sealed sources, not the destination.
    receipt.update(
        {
            "baseline_receipt_sha256": "sha256:"
            + hashlib.sha256(source_objects["baseline"]["receipt.json"]).hexdigest(),
            "baseline_scalar_receipt_sha256": "sha256:"
            + hashlib.sha256(
                source_objects["baseline"]["scalar-evaluation.json"]
            ).hexdigest(),
            "final_receipt_sha256": "sha256:"
            + hashlib.sha256(source_objects["final"]["receipt.json"]).hexdigest(),
            "final_scalar_receipt_sha256": "sha256:"
            + hashlib.sha256(
                source_objects["final"]["scalar-evaluation.json"]
            ).hexdigest(),
        }
    )
    encoded = json.dumps(receipt, separators=(",", ":"), sort_keys=True).encode(
        "utf-8"
    )

    def exact_bundle(root: Path) -> bool:
        try:
            root_descriptor = os.open(
                root, os.O_RDONLY | os.O_DIRECTORY | nofollow
            )
        except OSError:
            return False
        try:
            if set(os.listdir(root_descriptor)) != {
                "baseline",
                "final",
                "receipt.json",
            }:
                return False
            destination_objects: dict[str, dict[str, bytes]] = {}
            for name in ("baseline", "final"):
                child = os.open(
                    name,
                    os.O_RDONLY | os.O_DIRECTORY | nofollow,
                    dir_fd=root_descriptor,
                )
                try:
                    destination_objects[name] = sealed_bytes_from_descriptor(child)
                finally:
                    os.close(child)
            if regular_bytes(root_descriptor, "receipt.json") != encoded:
                return False
        except (HFMultimodalSFTError, OSError):
            return False
        finally:
            os.close(root_descriptor)
        for name in ("baseline", "final"):
            objects = destination_objects[name]
            if objects != source_objects[name]:
                return False
            if (
                "sha256:" + hashlib.sha256(objects["receipt.json"]).hexdigest()
                != receipt[f"{name}_receipt_sha256"]
                or "sha256:"
                + hashlib.sha256(objects["scalar-evaluation.json"]).hexdigest()
                != receipt[f"{name}_scalar_receipt_sha256"]
            ):
                return False
            try:
                validate_source(name, objects)
            except HFMultimodalSFTError:
                return False
        return True

    if directory.exists():
        if not exact_bundle(directory):
            raise HFMultimodalSFTError(
                "existing test caption evidence bundle is inconsistent"
            )
        return directory

    directory.parent.mkdir(mode=0o750, parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{directory.name}-", dir=directory.parent)
    )
    try:
        for name, objects in source_objects.items():
            evidence = staging / name
            evidence.mkdir(mode=0o750)
            for file_name, contents in objects.items():
                with _exclusive_output(evidence, file_name) as output:
                    output.write(contents)
                    output.flush()
                    os.fsync(output.fileno())
        with _exclusive_output(staging, "receipt.json") as output:
            output.write(encoded)
            output.flush()
            os.fsync(output.fileno())
        _fsync_directories(staging)
        os.rename(staging, directory)
        parent_descriptor = os.open(directory.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(parent_descriptor)
        finally:
            os.close(parent_descriptor)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return directory


def _write_test_scalar_receipt(
    *,
    directory: Path,
    loss: float,
    records: int,
    split_membership_digest: str,
    step: int,
) -> None:
    if not math.isfinite(loss) or records < 1:
        raise HFMultimodalSFTError("full test scalar evidence is invalid")
    receipt = {
        "api_version": "rwkv-lab.hf-test-scalar-evidence/v1",
        "coverage": "full",
        "loss": loss,
        "records": records,
        "split_membership_digest": split_membership_digest,
        "step": step,
    }
    encoded = json.dumps(
        receipt,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    path = directory / "scalar-evaluation.json"
    if path.exists():
        if path.read_bytes() != encoded:
            raise HFMultimodalSFTError(
                "existing full test scalar evidence is inconsistent"
            )
        return
    with _exclusive_output(directory, "scalar-evaluation.json") as output:
        output.write(encoded)
        output.flush()
        os.fsync(output.fileno())


def _quarantine_test_evidence(directory: Path) -> str:
    expected = {
        "captions.jsonl",
        "identity.json",
        "receipt.json",
        "scalar-evaluation.json",
    }
    try:
        root_info = directory.lstat()
        if stat.S_ISLNK(root_info.st_mode) or not stat.S_ISDIR(root_info.st_mode):
            raise HFMultimodalSFTError(
                "private test-evidence quarantine root is invalid"
            )
        entries = tuple(directory.iterdir())
    except OSError as error:
        raise HFMultimodalSFTError(
            "private test-evidence quarantine is unavailable"
        ) from error
    if {entry.name for entry in entries} != expected:
        raise HFMultimodalSFTError(
            "private test-evidence quarantine has an inexact layout"
        )
    for entry in entries:
        info = entry.lstat()
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
            raise HFMultimodalSFTError(
                "private test-evidence quarantine contains a non-file"
            )
        os.chmod(entry, 0o400, follow_symlinks=False)
    os.chmod(directory, 0o700, follow_symlinks=False)
    os.chmod(directory.parent, 0o700, follow_symlinks=False)
    return _digest(_staged_objects(directory))


def run_hf_multimodal_sft(
    *,
    invocation: Any,
    components: Any,
    run_directory: Path,
    controls: Any,
    observability: Any,
    step_profiler: Any,
    resume_directory: Path | None,
    resume_parent_artifact_ids: tuple[str, ...] = (),
    resume_checkpoint_manifest_digest: str | None = None,
    device: torch.device | str | None = None,
) -> int:
    """Execute one exact, component-owned HF SFT lifecycle."""

    from rwkv_lab.trainvm_worker import (
        ArtifactPublicationRequest,
        CheckpointPublicationRequest,
        EvalGalleryItem,
        EvalGalleryPublicationRequest,
        FinalEvaluationPublicationRequest,
        GalleryImage,
        PublishedCheckpoint,
    )

    run_directory.mkdir(mode=0o750, parents=True, exist_ok=True)
    quarantine_root = run_directory / ".private-test-quarantine"
    quarantine_root.mkdir(mode=0o700, exist_ok=True)
    quarantine_info = quarantine_root.lstat()
    if stat.S_ISLNK(quarantine_info.st_mode) or not stat.S_ISDIR(
        quarantine_info.st_mode
    ):
        raise HFMultimodalSFTError("private test-evidence quarantine is invalid")
    os.chmod(quarantine_root, 0o700, follow_symlinks=False)
    quarantine_baseline = quarantine_root / "baseline"
    controls.poll_initialization()
    selected_device = torch.device(
        device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    data = HFDataRuntime.build(components)
    policy = components.checkpoint_policy()
    evaluator = components.evaluator()
    evaluation_schedule = components.evaluation_schedule()
    if not evaluation_schedule.configuration.final:
        raise HFMultimodalSFTError(
            "HF operation requires final scalar and qualitative evaluation"
        )
    renderer = components.artifact_renderer()
    generation_policy = components.generation_policy()
    maximum_new_tokens = generation_policy.configuration.maximum_new_tokens
    use_generation_cache = generation_policy.configuration.use_cache
    generation_batch_size = generation_policy.configuration.generation_batch_size
    padding_side = generation_policy.configuration.padding_side
    decode_policy_digest = generation_policy.digest
    maximum_steps = components.learning_rate_configuration()[1].max_steps
    loader = components.model_loader(slot="model_loader")
    loaded = loader.load()
    if not loaded.receipt.exact:
        raise HFMultimodalSFTError("HF engine requires an exact base-load receipt")
    tokenizer = getattr(
        loaded.tokenizer_or_processor,
        "tokenizer",
        loaded.tokenizer_or_processor,
    )
    try:
        tokenizer.padding_side = padding_side
    except (AttributeError, TypeError) as error:
        raise HFMultimodalSFTError(
            "HF tokenizer cannot apply the declared generation padding policy"
        ) from error
    if getattr(tokenizer, "padding_side", None) != padding_side:
        raise HFMultimodalSFTError(
            "HF tokenizer disagrees with the declared generation padding policy"
        )
    codec = HFForwardBatchCodec(
        loaded.tokenizer_or_processor,
        data.training_pipeline.mapper.configuration,
        data.training_pipeline.processor.configuration,
        data.training_pipeline.collator.configuration,
    )
    precision = components.precision()
    inference_model = precision.convert_module(loaded.model, selected_device)
    def reject_live_controls(
        _candidate: Mapping[str, Any], assignments: Mapping[str, Any]
    ) -> None:
        if assignments:
            raise HFMultimodalSFTError("HF SFT v1 exposes no live scalar controls")

    preoptimizer_qualitative: tuple[tuple[str, str, str], ...] | None = None
    if resume_directory is None:
        launch_decision = evaluation_schedule.for_step(0)
        if not (
            launch_decision.launch_gate
            and launch_decision.qualitative
            and launch_decision.full_scalar
        ):
            raise HFMultimodalSFTError("HF operation lacks a complete step-zero gate")
        controls.evaluation(0, reject_live_controls)
        keepalive = getattr(observability, "keepalive", None)
        phase = keepalive(0, "launch_eval") if callable(keepalive) else nullcontext()
        with phase:
            inference_stack = SimpleNamespace(model=inference_model, precision=precision)
            preoptimizer_qualitative = generate_hf_captions(
                stack=inference_stack,
                codec=codec,
                samples=data.qualitative_samples(),
                device=selected_device,
                maximum_new_tokens=maximum_new_tokens,
                use_cache=use_generation_cache,
            )
    stack = initialize_training_stack(
        components,
        selected_device,
        loaded=loaded,
        precision=precision,
    )
    baseline_test_evidence: Path | None = None
    baseline_complete = False
    baseline_evidence_digest = "sha256:" + "0" * 64
    baseline_checkpoint_artifact_id: str | None = None
    baseline_checkpoint_manifest_digest: str | None = None
    launch_gallery_complete = False
    zero_digest = "sha256:" + "0" * 64
    checkpoint_policy_state: Mapping[str, Any] = policy.component_state(
        last_published_step=0,
        publication_manifest=zero_digest,
        retention_manifest=zero_digest,
        atomic_publication_complete=True,
    )
    published_steps: list[int] = []
    publication_pending = False
    baselines: dict[str, str] = {}
    step = 0
    if resume_directory is not None:
        restored = restore_exact_checkpoint(
            resume_directory,
            model=stack.model,
            optimizer=stack.optimizer,
            learning_rate_schedule=stack.learning_rate_schedule,
            weight_decay_schedule=stack.weight_decay_schedule,
            precision=stack.precision,
            composition=components.composition,
            expected_composition_digest=components.composition.composition_digest,
            expected_model_load_receipt_digest=stack.loaded.receipt.digest,
            expected_processor_fingerprint=stack.loaded.receipt.auxiliary_fingerprint,
            controls_state_validator=controls.verify_checkpoint_state,
        )
        if set(restored.runtime_state) != {
            "baseline_checkpoint_artifact_id",
            "baseline_checkpoint_manifest_digest",
            "baseline_complete",
            "baseline_evidence_digest",
            "checkpoint_policy",
            "data",
            "finalization_pending",
            "launch_gallery_complete",
            "publication_pending",
            "published_steps",
        }:
            raise HFMultimodalSFTError("HF operation runtime resume state is inexact")
        data.restore(restored.component_state, restored.runtime_state["data"])
        stack.activation_memory.restore_component_state(
            restored.component_state["activation_memory"], stack.model
        )
        if restored.runtime_state["checkpoint_policy"] != restored.component_state[
            "checkpoint_policy"
        ]:
            raise HFMultimodalSFTError(
                "checkpoint policy runtime and component state disagree"
            )
        checkpoint_policy_state = restored.runtime_state["checkpoint_policy"]
        restored_published_steps = restored.runtime_state["published_steps"]
        if not isinstance(restored_published_steps, list):
            raise HFMultimodalSFTError(
                "checkpoint publication history is malformed"
            )
        published_steps = list(restored_published_steps)
        publication_pending = restored.runtime_state["publication_pending"]
        baselines = dict(restored.qualitative_baselines)
        step = restored.optimizer_step
        candidate_baseline = quarantine_baseline
        baseline_complete = restored.runtime_state["baseline_complete"]
        baseline_evidence_digest = restored.runtime_state[
            "baseline_evidence_digest"
        ]
        baseline_checkpoint_artifact_id = restored.runtime_state[
            "baseline_checkpoint_artifact_id"
        ]
        baseline_checkpoint_manifest_digest = restored.runtime_state[
            "baseline_checkpoint_manifest_digest"
        ]
        launch_gallery_complete = restored.runtime_state[
            "launch_gallery_complete"
        ]
        if (
            not isinstance(baseline_complete, bool)
            or not isinstance(launch_gallery_complete, bool)
            or not isinstance(publication_pending, bool)
            or not publication_pending
            or (baseline_complete and not launch_gallery_complete)
            or not published_steps
            or published_steps[-1] != step
            or published_steps != sorted(set(published_steps))
            or step > maximum_steps
            or any(
                not isinstance(value, int)
                or isinstance(value, bool)
                or value < 0
                for value in published_steps
            )
            or resume_checkpoint_manifest_digest is None
            or not _is_digest(baseline_evidence_digest)
            or not isinstance(
                restored.runtime_state["finalization_pending"], bool
            )
            or restored.runtime_state["finalization_pending"]
            != (step == maximum_steps)
            or len(resume_parent_artifact_ids) != 1
        ):
            raise HFMultimodalSFTError("launch gate completion state is invalid")
        expected_pending_policy = policy.component_state(
            last_published_step=step,
            publication_manifest=_PENDING_PUBLICATION_DIGEST,
            retention_manifest=_digest(policy.retained_steps(published_steps)),
            atomic_publication_complete=True,
        )
        if checkpoint_policy_state != expected_pending_policy:
            raise HFMultimodalSFTError(
                "exact resume lacks a canonical pending checkpoint publication"
            )
        checkpoint_policy_state = policy.component_state(
            last_published_step=step,
            publication_manifest=resume_checkpoint_manifest_digest,
            retention_manifest=_digest(policy.retained_steps(published_steps)),
            atomic_publication_complete=True,
        )
        publication_pending = False
        if baseline_checkpoint_artifact_id is None:
            if step != 0:
                raise HFMultimodalSFTError(
                    "trained resume lacks its step-zero checkpoint anchor"
                )
            baseline_checkpoint_artifact_id = resume_parent_artifact_ids[0]
            baseline_checkpoint_manifest_digest = resume_checkpoint_manifest_digest
        if (
            not isinstance(baseline_checkpoint_artifact_id, str)
            or not baseline_checkpoint_artifact_id
            or not _is_digest(baseline_checkpoint_manifest_digest)
        ):
            raise HFMultimodalSFTError("step-zero checkpoint anchor is invalid")
        if data.test_records and baseline_complete and (
            not candidate_baseline.is_dir()
            or _quarantine_test_evidence(candidate_baseline)
            != baseline_evidence_digest
        ):
            raise HFMultimodalSFTError(
                "exact resume lacks its private test-baseline quarantine"
            )
        baseline_test_evidence = (
            candidate_baseline
            if data.test_records and baseline_complete
            else None
        )

    checkpoint_sequence = 0
    def stage_checkpoint_request() -> Any:
        nonlocal checkpoint_sequence
        checkpoint_sequence += 1
        projected_published_steps = list(published_steps)
        if not projected_published_steps or projected_published_steps[-1] != step:
            projected_published_steps.append(step)
        pending_policy_state = policy.component_state(
            last_published_step=step,
            publication_manifest=_PENDING_PUBLICATION_DIGEST,
            retention_manifest=_digest(
                policy.retained_steps(projected_published_steps)
            ),
            atomic_publication_complete=True,
        )
        state = HFEngineState(
            optimizer_step=step,
            composition_digest=components.composition.composition_digest,
            model_load_receipt_digest=stack.loaded.receipt.digest,
            processor_fingerprint=stack.loaded.receipt.auxiliary_fingerprint,
            component_state=_component_state(
                components=components,
                stack=stack,
                data=data,
                checkpoint_policy_state=pending_policy_state,
                optimizer_step=step,
            ),
            controls_state=controls.checkpoint_state(),
            runtime_state={
                "baseline_checkpoint_artifact_id": baseline_checkpoint_artifact_id,
                "baseline_checkpoint_manifest_digest": (
                    baseline_checkpoint_manifest_digest
                ),
                "checkpoint_policy": dict(pending_policy_state),
                "baseline_complete": baseline_complete,
                "baseline_evidence_digest": baseline_evidence_digest,
                "data": dict(data.runtime_state()),
                "finalization_pending": step == maximum_steps,
                "launch_gallery_complete": launch_gallery_complete,
                "publication_pending": True,
                "published_steps": projected_published_steps,
            },
            qualitative_baselines=baselines,
        )
        staging = run_directory / (
            f"hf-checkpoint-step-{step}-attempt-{invocation.attempt_id}-"
            f"{checkpoint_sequence}"
        )
        stage_exact_checkpoint(
            staging,
            model=stack.model,
            optimizer=stack.optimizer,
            learning_rate_schedule=stack.learning_rate_schedule,
            weight_decay_schedule=stack.weight_decay_schedule,
            precision=stack.precision,
            state=state,
        )
        request = CheckpointPublicationRequest(
            source_directory=staging,
            optimizer_step=step,
            resume_grade="exact",
            state_components=(
                "component_composition",
                "control_revision",
                "data_cursor",
                "lr_schedule",
                "model",
                "optimizer",
                "rng_accelerator",
                "rng_numpy",
                "rng_python",
                "rng_torch",
                "weight_decay_schedule",
            ),
            parent_artifact_ids=resume_parent_artifact_ids,
        )
        return request

    def record_published_checkpoint(published: Any) -> None:
        nonlocal checkpoint_policy_state, publication_pending
        if not published_steps or published_steps[-1] != step:
            published_steps.append(step)
        checkpoint_policy_state = policy.component_state(
            last_published_step=step,
            publication_manifest=published.manifest_sha256,
            retention_manifest=_digest(policy.retained_steps(published_steps)),
            atomic_publication_complete=True,
        )
        publication_pending = False

    def publish_gallery(
        checkpoint: Any, generated: Sequence[tuple[str, str, str]]
    ) -> Any:
        samples = data.qualitative_samples()
        items: list[Any] = []
        cards = run_directory / f"eval-cards-step-{step}"
        if any(_source_image_path(data, sample) is None for sample in samples):
            cards.mkdir(mode=0o750, exist_ok=True)
        for index, (sample, evidence) in enumerate(zip(samples, generated, strict=True)):
            teacher, current, prompt_digest = evidence
            baseline = baselines[sample.sample_id]
            rendered = renderer.render(
                sample_identity=sample.sample_id,
                step=step,
                evidence={
                    "teacher_target": teacher,
                    "baseline": baseline,
                    "current": current,
                },
            )
            values = rendered["evidence"]
            if any(len(str(value).encode("utf-8")) > 1024 for value in values.values()):
                raise HFMultimodalSFTError(
                    "caption evidence exceeds the gallery protocol text bound"
                )
            source_path = _source_image_path(data, sample)
            if source_path is None:
                source_path = cards / f"{index:04d}.png"
                if not source_path.exists():
                    _text_card(source_path, current)
            image = GalleryImage(source_path)
            items.append(
                EvalGalleryItem(
                    item_id=f"{sample.sample_id}:step:{step}",
                    heldout_item_id=sample.sample_id,
                    heldout_manifest_digest=data.split_membership_digest,
                    prompt_or_condition_digest=prompt_digest,
                    generated=image,
                    target=image,
                    source=image,
                    seed=0,
                    sampling_attributes={
                        "teacher_target": str(values["teacher_target"]),
                        "baseline": str(values["baseline"]),
                        "current": str(values["current"]),
                        "ordered_identities_digest": (
                            data.qualitative_identities_digest
                            or _digest(data.qualitative_ids)
                        ),
                    },
                )
            )
        return controls.publish_evaluation_gallery(
            EvalGalleryPublicationRequest(
                output_name="eval_gallery",
                step=step,
                step_domain="optimizer_step",
                evaluator_profile_digest=_digest(
                    {
                        "evaluator": components.composition.components[
                            "evaluator"
                        ].descriptor_digest,
                        "renderer": components.composition.components[
                            "artifact_renderer"
                        ].descriptor_digest,
                    }
                ),
                use_policy_digest=decode_policy_digest,
                items=tuple(items),
            ),
            checkpoint=checkpoint,
        )

    if step == 0:
        decision = evaluation_schedule.for_step(0)
        if not (decision.launch_gate and decision.qualitative and decision.full_scalar):
            raise HFMultimodalSFTError("HF operation lacks a complete step-zero gate")
        def base_model_context() -> Any:
            disable_adapter = getattr(stack.model, "disable_adapter", None)
            if stack.trainability_result.adapter_backed:
                if not callable(disable_adapter):
                    raise HFMultimodalSFTError(
                        "adapter-backed baseline cannot prove adapters are disabled"
                    )
                return disable_adapter()
            return nullcontext()

        qualitative: tuple[tuple[str, str, str], ...] | None = None
        if not baselines:
            if preoptimizer_qualitative is None:
                raise HFMultimodalSFTError("step-zero caption gate was not completed")
            baselines.update(
                {
                    sample_id: current
                    for sample_id, (_, current, _) in zip(
                        data.qualitative_ids, preoptimizer_qualitative, strict=True
                    )
                }
            )
            qualitative = generate_hf_captions(
                stack=stack,
                codec=codec,
                samples=data.qualitative_samples(),
                device=selected_device,
                maximum_new_tokens=maximum_new_tokens,
                use_cache=use_generation_cache,
            )
            if any(
                base_teacher != current_teacher or base_prompt != current_prompt
                for (base_teacher, _base, base_prompt), (
                    current_teacher,
                    _current,
                    current_prompt,
                ) in zip(preoptimizer_qualitative, qualitative, strict=True)
            ):
                raise HFMultimodalSFTError(
                    "step-zero teacher, baseline, and current identities are misaligned"
                )
            eval_loss = evaluate_hf_scalar_loss(
                stack=stack,
                codec=codec,
                data=data,
                evaluator=evaluator,
                device=selected_device,
                maximum_examples=evaluator.configuration.maximum_examples,
            )
            observability.publish_if_declared("eval.loss", eval_loss, step=0)
        elif not launch_gallery_complete:
            controls.evaluation(0, reject_live_controls)
            qualitative = generate_hf_captions(
                stack=stack,
                codec=codec,
                samples=data.qualitative_samples(),
                device=selected_device,
                maximum_new_tokens=maximum_new_tokens,
                use_cache=use_generation_cache,
            )

        if not launch_gallery_complete:
            if qualitative is None:
                raise HFMultimodalSFTError("step-zero gallery evidence is unavailable")
            request = stage_checkpoint_request()
            checkpoint = controls.publish_policy_checkpoint(request)
            record_published_checkpoint(checkpoint)
            baseline_checkpoint_artifact_id = checkpoint.artifact_id
            baseline_checkpoint_manifest_digest = checkpoint.manifest_sha256
            publish_gallery(checkpoint, qualitative)
            launch_gallery_complete = True
        if not launch_gallery_complete:
            raise HFMultimodalSFTError("step-zero gallery gate was not completed")

        if data.test_records and not baseline_complete:
            if (
                baseline_checkpoint_artifact_id is None
                or baseline_checkpoint_manifest_digest is None
            ):
                raise HFMultimodalSFTError(
                    "test baseline lacks its step-zero checkpoint anchor"
                )
            base_model_state_digest = _checkpoint_model_state_digest(
                model_load_receipt=loaded.receipt.digest,
                checkpoint_artifact_id=baseline_checkpoint_artifact_id,
                manifest_digest=baseline_checkpoint_manifest_digest,
            )
            controls.evaluation(0, reject_live_controls)
            keepalive = getattr(observability, "keepalive", None)
            phase = (
                keepalive(0, "baseline_eval")
                if callable(keepalive)
                else nullcontext()
            )
            with phase, base_model_context():
                baseline_test_evidence = _write_test_caption_evidence(
                    directory=quarantine_baseline,
                    stack=stack,
                    codec=codec,
                    samples=data.test_samples(),
                    device=selected_device,
                    step=0,
                    model_load_receipt=loaded.receipt.digest,
                    checkpoint_artifact_id=baseline_checkpoint_artifact_id,
                    checkpoint_manifest_digest=(
                        baseline_checkpoint_manifest_digest
                    ),
                    model_state_digest=base_model_state_digest,
                    split_membership_digest=str(data.test_membership_digest),
                    decode_policy_digest=decode_policy_digest,
                    model_state_mode="base_adapters_disabled",
                    maximum_new_tokens=maximum_new_tokens,
                    generation_batch_size=generation_batch_size,
                    use_cache=use_generation_cache,
                    progress=lambda completed, total, failed, elapsed_ms: (
                        observability.publish_if_declared(
                            "eval.caption_items_completed", completed, step=0
                        ),
                        observability.publish_if_declared(
                            "eval.caption_items_total", total, step=0
                        ),
                        observability.publish_if_declared(
                            "eval.caption_items_failed", failed, step=0
                        ),
                        observability.publish_if_declared(
                            "eval.caption_wall_seconds", elapsed_ms / 1000.0, step=0
                        ),
                    ),
                )
                test_loss = evaluate_hf_scalar_loss(
                    stack=stack,
                    codec=codec,
                    data=data,
                    evaluator=evaluator,
                    device=selected_device,
                    maximum_examples=len(data.test_records),
                    samples=data.test_samples(),
                )
            _write_test_scalar_receipt(
                directory=baseline_test_evidence,
                loss=test_loss,
                records=len(data.test_records),
                split_membership_digest=str(data.test_membership_digest),
                step=0,
            )
            baseline_evidence_digest = _quarantine_test_evidence(
                baseline_test_evidence
            )
            baseline_complete = True
            request = stage_checkpoint_request()
            checkpoint = controls.publish_policy_checkpoint(request)
            record_published_checkpoint(checkpoint)
        elif not data.test_records:
            baseline_complete = True

    final_checkpoint: Any | None = None
    if resume_directory is not None and step == maximum_steps:
        assert resume_checkpoint_manifest_digest is not None
        final_checkpoint = PublishedCheckpoint(
            artifact_id=resume_parent_artifact_ids[0],
            manifest_path=resume_directory,
            manifest_sha256=resume_checkpoint_manifest_digest,
            payload_size_bytes=0,
            file_count=0,
            worker_sequence=0,
        )

    stack.model.train()
    stack.optimizer.zero_grad(set_to_none=True)
    while step < maximum_steps:
        started = time.perf_counter()
        tokens = 0
        losses: list[tuple[float, int]] = []
        for _ in stack.accumulation.microbatch_indices():
            controls.microbatch(step + 1, reject_live_controls)
            with step_profiler.input_wait():
                samples = data.next_training_samples()
            batch = codec.encode(samples).to(selected_device)
            with _autocast(stack.precision, selected_device):
                loss = component_causal_loss(
                    stack.model,
                    stack.objective,
                    batch.tensors,
                    ignore_index=codec.collator_configuration.label_pad_token_id,
                )
            # The registered objective is a token mean. Backpropagating each
            # microbatch token sum and normalizing once preserves that exact
            # objective when buckets contain unequal supervised-token counts.
            (loss * batch.supervised_tokens).backward()
            losses.append(
                (
                    float(stack.precision.reduce(loss.detach()).cpu()),
                    batch.supervised_tokens,
                )
            )
            tokens += batch.supervised_tokens
        trainable_parameters = tuple(
            parameter for parameter in stack.model.parameters() if parameter.requires_grad
        )
        normalize_token_mean_gradients(trainable_parameters, tokens)
        components.gradient_clipping(
            trainable_parameters
        )
        stack.optimizer.step()
        step += 1
        stack.learning_rate_schedule.step()
        stack.weight_decay_schedule.step(step)
        stack.optimizer.zero_grad(set_to_none=True)
        step_profiler.step(step)
        elapsed = max(time.perf_counter() - started, 1.0e-9)
        observability.publish_if_declared(
            "train.loss", weighted_loss(losses), step=step
        )
        observability.publish_if_declared(
            "train.tokens_per_second", tokens / elapsed, step=step
        )
        observability.optimizer_step(step, "train")
        controls.optimizer_step(step, reject_live_controls)

        decision = evaluation_schedule.for_step(step, final=step == maximum_steps)
        if decision.full_scalar or decision.qualitative:
            controls.evaluation(step, reject_live_controls)
        if decision.full_scalar:
            value = evaluate_hf_scalar_loss(
                stack=stack,
                codec=codec,
                data=data,
                evaluator=evaluator,
                device=selected_device,
                maximum_examples=evaluator.configuration.maximum_examples,
            )
            observability.publish_if_declared("eval.loss", value, step=step)
        policy_due = policy.due(step, final=step == maximum_steps)
        checkpoint = None
        request = None
        if policy_due or decision.qualitative or controls.checkpoint_boundary_requested:
            controls.checkpoint(step, reject_live_controls)
        boundary = controls.checkpoint_boundary_requested
        if policy_due or decision.qualitative or boundary:
            request = stage_checkpoint_request()
        if controls.checkpoint_completion_requested:
            assert request is not None
            checkpoint = controls.publish_requested_checkpoint(request)
            if checkpoint is not None:
                record_published_checkpoint(checkpoint)
        if (policy_due or decision.qualitative) and checkpoint is None:
            assert request is not None
            checkpoint = controls.publish_policy_checkpoint(request)
            record_published_checkpoint(checkpoint)
        if step == maximum_steps:
            if checkpoint is None:
                raise HFMultimodalSFTError(
                    "final evaluation lacks its exact checkpoint anchor"
                )
            final_checkpoint = checkpoint
        elif decision.qualitative:
            qualitative = generate_hf_captions(
                stack=stack,
                codec=codec,
                samples=data.qualitative_samples(),
                device=selected_device,
                maximum_new_tokens=maximum_new_tokens,
                use_cache=use_generation_cache,
            )
            publish_gallery(checkpoint, qualitative)

    if step != maximum_steps or final_checkpoint is None:
        raise HFMultimodalSFTError("HF operation did not reach finalization")
    if (
        not data.test_records
        or baseline_test_evidence is None
        or baseline_checkpoint_artifact_id is None
        or baseline_checkpoint_manifest_digest is None
    ):
        raise HFMultimodalSFTError(
            "final test evidence lacks its private baseline/checkpoint anchor"
        )
    final_decision = evaluation_schedule.for_step(step, final=True)
    if not (final_decision.full_scalar and final_decision.qualitative):
        raise HFMultimodalSFTError("HF finalization is not fully evaluable")
    controls.evaluation(step, reject_live_controls)
    final_model_state_digest = _checkpoint_model_state_digest(
        model_load_receipt=stack.loaded.receipt.digest,
        checkpoint_artifact_id=final_checkpoint.artifact_id,
        manifest_digest=final_checkpoint.manifest_sha256,
    )
    test_value = evaluate_hf_scalar_loss(
        stack=stack,
        codec=codec,
        data=data,
        evaluator=evaluator,
        device=selected_device,
        maximum_examples=len(data.test_records),
        samples=data.test_samples(),
    )
    keepalive = getattr(observability, "keepalive", None)
    phase = (
        keepalive(step, "final_eval") if callable(keepalive) else nullcontext()
    )

    def publish_final_progress(
        completed: int,
        total: int,
        failed: int,
        elapsed_ms: float,
    ) -> None:
        observability.publish_if_declared(
            "eval.caption_items_completed", completed, step=step
        )
        observability.publish_if_declared(
            "eval.caption_items_total", total, step=step
        )
        observability.publish_if_declared(
            "eval.caption_items_failed", failed, step=step
        )
        observability.publish_if_declared(
            "eval.caption_wall_seconds", elapsed_ms / 1000.0, step=step
        )

    with phase:
        test_evidence = _write_test_caption_evidence(
            directory=run_directory / f"test-caption-evidence-step-{step}",
            stack=stack,
            codec=codec,
            samples=data.test_samples(),
            device=selected_device,
            step=step,
            model_load_receipt=stack.loaded.receipt.digest,
            checkpoint_artifact_id=final_checkpoint.artifact_id,
            checkpoint_manifest_digest=final_checkpoint.manifest_sha256,
            model_state_digest=final_model_state_digest,
            split_membership_digest=str(data.test_membership_digest),
            decode_policy_digest=decode_policy_digest,
            model_state_mode="trained",
            maximum_new_tokens=maximum_new_tokens,
            generation_batch_size=generation_batch_size,
            use_cache=use_generation_cache,
            progress=publish_final_progress,
        )
    _write_test_scalar_receipt(
        directory=test_evidence,
        loss=test_value,
        records=len(data.test_records),
        split_membership_digest=str(data.test_membership_digest),
        step=step,
    )
    bundle = _bundle_test_caption_evidence(
        directory=run_directory / f"test-caption-bundle-step-{step}",
        baseline=baseline_test_evidence,
        final=test_evidence,
        split_membership_digest=str(data.test_membership_digest),
        baseline_checkpoint_artifact_id=baseline_checkpoint_artifact_id,
        baseline_checkpoint_manifest_digest=baseline_checkpoint_manifest_digest,
        final_checkpoint_artifact_id=final_checkpoint.artifact_id,
        final_checkpoint_manifest_digest=final_checkpoint.manifest_sha256,
        final_model_state_digest=final_model_state_digest,
    )
    published_test_evaluation = controls.publish_artifact(
        ArtifactPublicationRequest(
            source_directory=bundle,
            output_name="test_eval",
            parent_artifact_ids=(
                baseline_checkpoint_artifact_id,
                final_checkpoint.artifact_id,
            ),
            optimizer_step=step,
        )
    )
    observability.publish_if_declared("eval.test_loss", test_value, step=step)
    qualitative = generate_hf_captions(
        stack=stack,
        codec=codec,
        samples=data.qualitative_samples(),
        device=selected_device,
        maximum_new_tokens=maximum_new_tokens,
        use_cache=use_generation_cache,
    )
    published_gallery = publish_gallery(final_checkpoint, qualitative)
    final_caption_records = tuple(
        json.loads(line)
        for line in (test_evidence / "captions.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
    )
    samples_by_id = {sample.sample_id: sample for sample in data.test_samples()}
    canonical_members = tuple(sorted(samples_by_id))
    records_by_id = {
        str(record["sample_id"]): record for record in final_caption_records
    }
    if set(records_by_id) != set(canonical_members):
        raise HFMultimodalSFTError(
            "final closure membership disagrees with sealed test evidence"
        )
    final_records = tuple(
        {
            "member_id": member,
            "context_digest": _final_member_context_digest(
                samples_by_id[member], components
            ),
            "attempt": 1,
            "disposition": "success",
            "result_digest": _digest(records_by_id[member]["text"]),
        }
        for member in canonical_members
    )
    observable_eval_scalars = {
        str(metric["name"])
        for metric in invocation.observability.get("metrics", ())
        if isinstance(metric, Mapping)
        and isinstance(metric.get("name"), str)
        and metric.get("step_domain") == "optimizer_step"
    }
    evaluator_metric_names = tuple(
        name if str(name).startswith("eval.") else f"eval.{name}"
        for name in evaluator.configuration.metrics
    )
    if any(name not in observable_eval_scalars for name in evaluator_metric_names):
        raise HFMultimodalSFTError(
            "final evaluator metric lacks an exact observable scalar"
        )
    required_scalars = tuple(
        {"metric_name": name, "step_domain": "optimizer_step"}
        for name in sorted(evaluator_metric_names)
    )
    output_receipts = tuple(
        sorted(
            (
                {
                    "output_name": "checkpoint",
                    "artifact_id": final_checkpoint.artifact_id,
                    "artifact_fingerprint": final_checkpoint.manifest_sha256,
                },
                {
                    "output_name": "eval_gallery",
                    "artifact_id": published_gallery.artifact_id,
                    "artifact_fingerprint": published_gallery.manifest_sha256,
                },
                {
                    "output_name": "test_eval",
                    "artifact_id": published_test_evaluation.artifact_id,
                    "artifact_fingerprint": published_test_evaluation.manifest_sha256,
                },
            ),
            key=lambda receipt: receipt["output_name"],
        )
    )
    controls.publish_final_evaluation(
        FinalEvaluationPublicationRequest(
            optimizer_step=step,
            checkpoint_artifact_id=final_checkpoint.artifact_id,
            checkpoint_fingerprint=final_checkpoint.manifest_sha256,
            required_members=canonical_members,
            member_context_digests={
                member: _final_member_context_digest(
                    samples_by_id[member], components
                )
                for member in canonical_members
            },
            records=final_records,
            output_receipts=output_receipts,
            required_scalars=required_scalars,
            parent_artifact_ids=tuple(
                receipt["artifact_id"] for receipt in output_receipts
            ),
        )
    )
    return step


__all__ = [
    "HFDataRuntime",
    "HFEngineState",
    "HFForwardBatch",
    "HFForwardBatchCodec",
    "HFMultimodalSFTError",
    "HFTrainingStack",
    "component_causal_loss",
    "initialize_training_stack",
    "normalize_token_mean_gradients",
    "restore_exact_checkpoint",
    "run_hf_multimodal_sft",
    "scalar_loss",
    "stage_exact_checkpoint",
    "weighted_loss",
]
