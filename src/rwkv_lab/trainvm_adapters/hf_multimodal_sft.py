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
import stat
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import torch

from rwkv_lab.training_components import (
    AssistantOnlyMapperConfiguration,
    CausalTokensMapperConfiguration,
    ImageCaptionProcessorConfiguration,
    MappedSample,
    PaddedCollatorConfiguration,
    ProcessedSample,
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


class HFMultimodalSFTError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class HFTrainingStack:
    loaded: Any
    trainability: Any
    trainability_result: Any
    precision: Any
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
) -> HFTrainingStack:
    """Initialize in the only safe order for adapter-backed precision policy."""

    loader = components.model_loader(slot="model_loader")
    loaded = loader.load(transformers_module=transformers_module)
    if not loaded.receipt.exact:
        raise HFMultimodalSFTError("HF engine requires an exact base-load receipt")
    trainability = components.trainability()
    trainability_result = trainability.apply(loaded.model)
    if not trainability_result.trainable_parameter_names:
        raise HFMultimodalSFTError("HF training composition produced zero trainables")
    precision = components.precision()
    model = precision.convert_module(trainability_result.model, device)
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
        model=model,
        optimizer=optimizer,
        learning_rate_schedule=learning_rate_schedule,
        weight_decay_schedule=weight_decay_schedule,
        objective=components.objective(),
        accumulation=components.gradient_accumulation(),
    )


def _digest(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _prompt_and_target(
    sample: ProcessedSample, configuration: AssistantOnlyMapperConfiguration
) -> tuple[str, str]:
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
    return prompt, target


def _conversation(prompt: str, target: str | None = None) -> list[dict[str, object]]:
    messages: list[dict[str, object]] = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ],
        }
    ]
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
        AssistantOnlyMapperConfiguration | CausalTokensMapperConfiguration
    )
    processor_configuration: ImageCaptionProcessorConfiguration | Any
    collator_configuration: PaddedCollatorConfiguration

    @property
    def multimodal(self) -> bool:
        return isinstance(self.mapper_configuration, AssistantOnlyMapperConfiguration)

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
        if not isinstance(mapper, AssistantOnlyMapperConfiguration) or not isinstance(
            self.processor_configuration, ImageCaptionProcessorConfiguration
        ):
            raise HFMultimodalSFTError(
                "multimodal codec requires image-caption processor and assistant mapper"
            )
        processor = self.processor_or_tokenizer
        tokenizer = getattr(processor, "tokenizer", None)
        if tokenizer is None:
            raise HFMultimodalSFTError("multimodal processor exposes no tokenizer")
        if getattr(tokenizer, "padding_side", "right") != "right":
            raise HFMultimodalSFTError(
                "multimodal assistant masking requires right padding"
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
            prompt, target = _prompt_and_target(sample, mapper)
            rendered_prompt = _template(
                processor, _conversation(prompt), generation=True
            )
            rendered_full = _template(
                processor, _conversation(prompt, target), generation=False
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
            if token_count > min(
                mapper.maximum_tokens,
                self.collator_configuration.maximum_sequence_length,
            ):
                raise HFMultimodalSFTError(
                    "image-token-expanded sequence exceeds the declared token policy"
                )
            expansion = token_count - len(full_ids)
            boundary = len(prompt_ids) + expansion
            if not 0 < boundary < token_count:
                raise HFMultimodalSFTError(
                    "image-token expansion erased the assistant target"
                )
            suffix = torch.tensor(
                full_ids[len(prompt_ids) :],
                dtype=input_ids.dtype,
                device=input_ids.device,
            )
            if not torch.equal(input_ids[index, boundary:token_count], suffix):
                raise HFMultimodalSFTError(
                    "image-token expansion changed the assistant target suffix"
                )
            labels[index, boundary:token_count] = input_ids[index, boundary:token_count]
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
        assert isinstance(mapper, AssistantOnlyMapperConfiguration)
        processor = self.processor_or_tokenizer
        prompts: list[str] = []
        images: list[Any] = []
        for sample in samples:
            prompt, _ = _prompt_and_target(sample, mapper)
            prompts.append(_template(processor, _conversation(prompt), generation=True))
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


@dataclass(frozen=True, slots=True)
class HFEngineState:
    optimizer_step: int
    composition_digest: str
    model_load_receipt_digest: str
    processor_fingerprint: str
    component_state: Mapping[str, Any]
    controls_state: Mapping[str, Any]
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
        body = {
            "api_version": _ENGINE_STATE_SCHEMA,
            "component_state": dict(self.component_state),
            "composition_digest": self.composition_digest,
            "controls_state": dict(self.controls_state),
            "model_load_receipt_digest": self.model_load_receipt_digest,
            "microbatch_in_optimizer_step": self.microbatch_in_optimizer_step,
            "optimizer_step": self.optimizer_step,
            "processor_fingerprint": self.processor_fingerprint,
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
    model_directory = directory / "model"
    save_pretrained = getattr(model, "save_pretrained", None)
    if not callable(save_pretrained):
        raise HFMultimodalSFTError("HF model cannot publish save_pretrained state")
    save_pretrained(model_directory, safe_serialization=True)
    trainable_state = {
        name: parameter.detach().cpu().clone()
        for name, parameter in sorted(model.named_parameters())
        if parameter.requires_grad
    }
    if not trainable_state:
        raise HFMultimodalSFTError("checkpoint contains no trainable model state")
    with _exclusive_output(directory, "trainable-state.pt") as output:
        torch.save(trainable_state, output)
        output.flush()
        os.fsync(output.fileno())
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
        "trainable-state.pt",
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
        microbatch_in_optimizer_step=document["microbatch_in_optimizer_step"],
    )
    if dict(restored_state.canonical_document()) != document:
        raise HFMultimodalSFTError("exact checkpoint engine state is noncanonical")
    composition.validate_resume_state(component_state)
    controls_state_validator(controls_state)
    with _open_staged_binary(
        directory,
        "trainable-state.pt",
        expected=object_receipts["trainable-state.pt"],
    ) as source:
        trainable = torch.load(source, map_location="cpu", weights_only=True)
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


__all__ = [
    "HFEngineState",
    "HFForwardBatch",
    "HFForwardBatchCodec",
    "HFMultimodalSFTError",
    "HFTrainingStack",
    "component_causal_loss",
    "initialize_training_stack",
    "restore_exact_checkpoint",
    "scalar_loss",
    "stage_exact_checkpoint",
    "weighted_loss",
]
