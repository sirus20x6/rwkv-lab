from __future__ import annotations

import hashlib
import json
import random
from contextlib import nullcontext
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from rwkv_lab.training_components import (
    AssistantOnlyMapperConfiguration,
    CachedReferenceDPOConfiguration,
    CachedReferenceDPOObjective,
    CausalTokensMapperConfiguration,
    CollatorImplementation,
    ImageCaptionProcessorConfiguration,
    JsonlFrozenTokenSplitsConfiguration,
    LinearHeadCrossEntropyConfiguration,
    LinearHeadCrossEntropyObjective,
    PackedTokenCollatorConfiguration,
    PaddedCollatorConfiguration,
    ProcessedSample,
    RegisteredCollator,
)
from rwkv_lab.training_runtime.activation_memory import (
    HFGradientCheckpointing,
    HFGradientCheckpointingConfiguration,
)
from rwkv_lab.training_runtime.generation_policies import (
    GreedyGenerationConfiguration,
)
from rwkv_lab.training_runtime.trainability import (
    load_lora_target_receipt,
    preflight_lora_target_manifest,
    preflight_lora_targets_from_snapshot,
)
from rwkv_lab.trainvm_adapters.hf_multimodal_sft import (
    HFEngineState,
    HFForwardBatchCodec,
    HFMultimodalSFTError,
    backward_cached_dpo_pair,
    component_cached_dpo_loss,
    component_causal_loss,
    initialize_training_stack,
    normalize_token_mean_gradients,
    restore_exact_checkpoint,
    run_hf_multimodal_sft,
    stage_exact_checkpoint,
)


class FakeTokenizer:
    eos_token_id = 2

    def __init__(self, padding_side: str = "right") -> None:
        self.padding_side = padding_side

    def __call__(self, text: str, *, add_special_tokens: bool = False):
        del add_special_tokens
        return {"input_ids": [ord(character) for character in text]}


class FakeProcessor:
    def __init__(self, padding_side: str = "right") -> None:
        self.tokenizer = FakeTokenizer(padding_side)

    def apply_chat_template(
        self, messages, *, tokenize: bool, add_generation_prompt: bool
    ) -> str:
        assert tokenize is False
        prompt = messages[0]["content"][1]["text"]
        prefix = f"¤{prompt}|"
        if add_generation_prompt:
            return prefix
        return prefix + messages[1]["content"] + chr(self.tokenizer.eos_token_id)

    def __call__(
        self,
        *,
        text,
        images,
        padding: bool,
        pad_to_multiple_of: int,
        truncation: bool,
        return_tensors: str,
    ):
        assert padding and not truncation and return_tensors == "pt"
        rows = []
        for item in text:
            raw = self.tokenizer(item)["input_ids"]
            marker = raw.index(ord("¤"))
            rows.append(raw[:marker] + [900, 901] + raw[marker + 1 :])
        maximum = max(len(row) for row in rows)
        length = ((maximum + pad_to_multiple_of - 1) // pad_to_multiple_of) * (
            pad_to_multiple_of
        )
        ids = torch.zeros((len(rows), length), dtype=torch.long)
        attention = torch.zeros_like(ids)
        for index, row in enumerate(rows):
            start = 0 if self.tokenizer.padding_side == "right" else length - len(row)
            ids[index, start : start + len(row)] = torch.tensor(row)
            attention[index, start : start + len(row)] = 1
        return {
            "input_ids": ids,
            "attention_mask": attention,
            "pixel_values": torch.tensor(images, dtype=torch.float32).reshape(-1, 1),
            "image_grid_thw": torch.ones((len(rows), 3), dtype=torch.long),
        }


def _codec(*, padding_side: str = "right") -> HFForwardBatchCodec:
    return HFForwardBatchCodec(
        FakeProcessor(padding_side),
        AssistantOnlyMapperConfiguration(
            prompt_column="",
            fixed_prompt="describe",
            target_column="caption",
            maximum_tokens=256,
            append_eos=True,
        ),
        ImageCaptionProcessorConfiguration(
            image_column="image",
            caption_columns=("caption",),
            minimum_pixels=1,
            maximum_pixels=1024,
            maximum_edge=32,
            oversize_policy="reject",
        ),
        PaddedCollatorConfiguration(
            pad_token_id=0,
            label_pad_token_id=-100,
            pad_to_multiple=8,
            maximum_sequence_length=256,
        ),
    )


def _sample(sample_id: str, image: float, caption: str) -> ProcessedSample:
    return ProcessedSample(
        sample_id=sample_id,
        ordinal=int(sample_id[-1]),
        values={"image": f"{sample_id}.png", "caption": caption},
        image=image,
        image_size=(1, 1),
    )


@pytest.mark.parametrize("padding_side", ["left", "right"])
def test_multimodal_codec_keeps_image_target_identity_and_masks_prompt(
    padding_side: str,
) -> None:
    batch = _codec(padding_side=padding_side).encode(
        (_sample("sample-1", 1.0, "red"), _sample("sample-2", 7.0, "blue"))
    )
    assert batch.sample_ids == ("sample-1", "sample-2")
    assert batch.targets == ("red", "blue")
    assert batch.source_images == (1.0, 7.0)
    assert batch.tensors["pixel_values"].flatten().tolist() == [1.0, 7.0]
    labels = batch.tensors["labels"]
    assert (labels != -100).sum(dim=1).tolist() == [4, 5]
    for row, target in zip(labels, ("red" + chr(2), "blue" + chr(2)), strict=True):
        assert row[row != -100].tolist() == [ord(character) for character in target]


def test_batched_generation_policy_requires_explicit_left_padding() -> None:
    with pytest.raises(ValueError, match="left padding"):
        GreedyGenerationConfiguration(128, 4, True, "right")
    assert GreedyGenerationConfiguration(128, 4, True, "left").padding_side == "left"


def _causal_codec(*, mapper_maximum: int = 8, collator_maximum: int = 8):
    return HFForwardBatchCodec(
        object(),
        CausalTokensMapperConfiguration(
            token_column="tokens", maximum_tokens=mapper_maximum
        ),
        object(),
        PaddedCollatorConfiguration(
            pad_token_id=0,
            label_pad_token_id=-100,
            pad_to_multiple=1,
            maximum_sequence_length=collator_maximum,
        ),
    )


@pytest.mark.parametrize("tokens", [[1, True], [1, -1], [1, 2.5]])
def test_causal_codec_rejects_noncanonical_token_ids(tokens) -> None:
    sample = ProcessedSample("sample", 0, {"tokens": tokens})
    with pytest.raises(HFMultimodalSFTError, match="invalid token IDs"):
        _causal_codec().encode((sample,))


def test_causal_codec_rejects_over_limit_instead_of_truncating() -> None:
    sample = ProcessedSample("sample", 0, {"tokens": [1, 2, 3, 4]})
    with pytest.raises(HFMultimodalSFTError, match="exceeds"):
        _causal_codec(mapper_maximum=4, collator_maximum=3).encode((sample,))


def test_causal_codec_uses_registered_packed_token_collator() -> None:
    codec = HFForwardBatchCodec(
        object(),
        CausalTokensMapperConfiguration("tokens", 8),
        object(),
        RegisteredCollator(
            CollatorImplementation.PACKED_TOKENS_V1,
            PackedTokenCollatorConfiguration(0, -100, 1, 8, 2),
        ),
    )
    batch = codec.encode(
        (
            ProcessedSample("first", 0, {"tokens": [10, 11]}),
            ProcessedSample("second", 1, {"tokens": [20, 21, 22]}),
        )
    )

    assert len(batch.sample_ids) == 1
    assert batch.sample_ids[0].startswith("sha256:")
    assert batch.tensors["input_ids"].tolist() == [
        [10, 11, 2, 20, 21, 22, 0, 0]
    ]
    assert batch.supervised_tokens == 6


class TinyCausalModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(9)
        self.embedding = torch.nn.Embedding(1024, 12)
        self.head = torch.nn.Linear(12, 1024, bias=False)

    def get_output_embeddings(self):
        return self.head

    def forward(
        self,
        *,
        input_ids,
        attention_mask,
        pixel_values,
        image_grid_thw,
        labels=None,
        use_cache=False,
        output_hidden_states=False,
        return_dict=True,
    ):
        del attention_mask, image_grid_thw, use_cache, return_dict
        hidden = self.embedding(input_ids) + pixel_values[:, None, :]
        result = SimpleNamespace(hidden_states=(hidden,))
        if labels is not None:
            logits = self.head(hidden[:, :-1])
            result.loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                labels[:, 1:].reshape(-1),
                ignore_index=-100,
            )
        assert output_hidden_states or labels is not None
        return result


def test_registered_objective_matches_hf_causal_shift_and_is_image_sensitive() -> None:
    batch = _codec().encode((_sample("sample-1", 1.0, "red"),))
    model = TinyCausalModel()
    objective = LinearHeadCrossEntropyObjective(
        LinearHeadCrossEntropyConfiguration(chunk_size=32, prefer_fused=False)
    )
    actual = component_causal_loss(model, objective, batch.tensors, ignore_index=-100)
    expected = model(**batch.tensors, output_hidden_states=True).loss
    assert float(actual.detach()) == pytest.approx(float(expected.detach()))

    changed = _codec().encode((_sample("sample-1", 9.0, "red"),))
    changed_loss = component_causal_loss(
        model, objective, changed.tensors, ignore_index=-100
    )
    assert float(changed_loss.detach()) != pytest.approx(float(actual.detach()))


def test_cached_reference_dpo_binds_exact_frozen_suffixes_and_updates_policy() -> None:
    chosen = "specific red subject"
    rejected = "generic subject"
    values = {
        "image": "sample-1.png",
        "caption": chosen,
        "chosen": chosen,
        "rejected": rejected,
        "reference_chosen_logp_sum": -9.0,
        "reference_rejected_logp_sum": -7.0,
        "reference_chosen_token_ids": [
            *(ord(character) for character in chosen),
            2,
        ],
        "reference_rejected_token_ids": [
            *(ord(character) for character in rejected),
            2,
        ],
    }
    sample = ProcessedSample("sample-1", 0, values, image=1.0, image_size=(1, 1))
    configuration = CachedReferenceDPOConfiguration(
        evaluation_chunk_size=32,
        evaluation_prefer_fused=False,
    )
    objective = CachedReferenceDPOObjective(configuration)
    batch = _codec().encode_preference((sample,), configuration)
    assert batch.chosen.targets == (chosen,)
    assert batch.rejected.targets == (rejected,)
    assert batch.reference_chosen.tolist() == [-9.0]
    model = TinyCausalModel()
    low_memory_model = deepcopy(model)
    loss, margin, policy_chosen, policy_rejected = component_cached_dpo_loss(
        model, objective, batch, ignore_index=-100
    )
    assert loss.ndim == margin.ndim == 0 or margin.shape == (1,)
    assert policy_chosen.shape == policy_rejected.shape == (1,)
    loss.backward()
    assert model.embedding.weight.grad is not None
    assert torch.isfinite(model.embedding.weight.grad).all()
    low_memory_loss, low_memory_margin, *_ = backward_cached_dpo_pair(
        low_memory_model, objective, batch, ignore_index=-100
    )
    assert low_memory_loss == pytest.approx(loss.detach())
    assert torch.equal(low_memory_margin, margin)
    for expected, actual in zip(
        model.parameters(), low_memory_model.parameters(), strict=True
    ):
        assert actual.grad is not None
        assert expected.grad is not None
        assert torch.allclose(actual.grad, expected.grad, rtol=2e-5, atol=2e-6)

    bad_values = dict(values)
    bad_values["reference_chosen_token_ids"] = [123]
    bad = ProcessedSample("sample-1", 0, bad_values, image=1.0, image_size=(1, 1))
    with pytest.raises(HFMultimodalSFTError, match="frozen reference ledger"):
        _codec().encode_preference((bad,), configuration)


def test_unequal_microbatches_match_one_concatenated_token_mean_gradient() -> None:
    parameter = torch.nn.Parameter(torch.tensor(0.7))
    first = ((parameter * torch.tensor([1.0, 3.0]) - 1.0) ** 2).mean()
    second = ((parameter * torch.tensor([2.0, 4.0, 5.0, 7.0]) + 0.5) ** 2).mean()
    (first * 2).backward()
    (second * 4).backward()
    normalize_token_mean_gradients((parameter,), 6)
    actual = parameter.grad.detach().clone()

    reference = torch.nn.Parameter(torch.tensor(0.7))
    expected = torch.cat(
        (
            (reference * torch.tensor([1.0, 3.0]) - 1.0) ** 2,
            (reference * torch.tensor([2.0, 4.0, 5.0, 7.0]) + 0.5) ** 2,
        )
    ).mean()
    expected.backward()
    assert torch.equal(actual, reference.grad)


def test_stack_initialization_applies_precision_after_trainability() -> None:
    events: list[str] = []

    class Loader:
        def load(self, *, transformers_module=None):
            del transformers_module
            events.append("load")
            return SimpleNamespace(
                model=torch.nn.Linear(2, 2),
                receipt=SimpleNamespace(exact=True),
            )

    class Trainability:
        def apply(self, model):
            events.append("trainability")
            model.register_parameter(
                "lora_parameter", torch.nn.Parameter(torch.ones(2, dtype=torch.float32))
            )
            return SimpleNamespace(
                model=model,
                trainable_parameter_names=("lora_parameter",),
            )

    class Precision:
        compute_dtype = torch.bfloat16

        def convert_module(self, model, device):
            events.append("precision")
            assert model.lora_parameter.dtype is torch.float32
            return model.to(device=device, dtype=torch.bfloat16)

    class ActivationMemory:
        def apply(self, model):
            events.append("activation_memory")

    class Components:
        def model_loader(self, *, slot):
            assert slot == "model_loader"
            return Loader()

        def trainability(self):
            return Trainability()

        def precision(self):
            return Precision()

        def activation_memory(self):
            return ActivationMemory()

        def optimizer(self, parameters):
            events.append("optimizer")
            parameters = tuple(parameters)
            assert any(parameter.dtype is torch.bfloat16 for parameter in parameters)
            return torch.optim.SGD(parameters, lr=0.1)

        def learning_rate_schedule(self, optimizer):
            return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _step: 1.0)

        def weight_decay_schedule(self, optimizer):
            return SimpleNamespace(optimizer=optimizer)

        def objective(self):
            return object()

        def gradient_accumulation(self):
            return object()

    initialize_training_stack(Components(), "cpu")
    assert events == [
        "load",
        "trainability",
        "precision",
        "activation_memory",
        "optimizer",
    ]


def test_hf_gradient_checkpointing_receipts_exact_resume_policy() -> None:
    class Model(torch.nn.Module):
        is_gradient_checkpointing = False

        def gradient_checkpointing_enable(self, *, gradient_checkpointing_kwargs):
            assert gradient_checkpointing_kwargs == {"use_reentrant": False}
            self.is_gradient_checkpointing = True

    model = Model()
    policy = HFGradientCheckpointing(
        HFGradientCheckpointingConfiguration(use_reentrant=False)
    )
    policy.apply(model)
    assert policy.component_state() == {
        "enabled": True,
        "use_reentrant": False,
    }
    policy.restore_component_state(policy.component_state(), model)
    with pytest.raises(ValueError, match="resume state"):
        policy.restore_component_state(
            {"enabled": True, "use_reentrant": True}, model
        )


def test_lora_selector_preflight_uses_snapshot_index_without_loading_weights(
    tmp_path,
) -> None:
    index = {
        "metadata": {"total_size": 123},
        "weight_map": {
            "model.language_model.layers.0.self_attn.q_proj.weight": (
                "model-00001-of-00002.safetensors"
            ),
            "model.language_model.layers.1.linear_attn.in_proj_qkv.weight": (
                "model-00002-of-00002.safetensors"
            ),
        },
    }
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(index), encoding="utf-8"
    )
    selected = preflight_lora_targets_from_snapshot(
        tmp_path,
        ("model.language_model.layers.*.self_attn.*_proj",),
    )
    assert selected == (
        "model.language_model.layers.0.self_attn.q_proj",
    )
    with pytest.raises(ValueError, match="matched no tensors"):
        preflight_lora_targets_from_snapshot(
            tmp_path,
            ("language_model.layers.*.self_attn.*_proj",),
        )


def test_exact_lora_target_manifest_is_bound_to_snapshot_metadata(tmp_path) -> None:
    import hashlib

    config = b'{"model_type":"fixture"}'
    index = json.dumps(
        {
            "weight_map": {
                "lm_head.weight": "model.safetensors",
                "model.layers.0.q_proj.weight": "model.safetensors",
            }
        },
        sort_keys=True,
    ).encode()
    (tmp_path / "config.json").write_bytes(config)
    (tmp_path / "model.safetensors.index.json").write_bytes(index)
    policy = {"attention": "adapted", "vision": "frozen"}
    policy_digest = "sha256:" + hashlib.sha256(
        json.dumps(policy, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    manifest = tmp_path / "targets.json"
    manifest.write_text(
        json.dumps(
            {
                "architecture": ["Fixture"],
                "model_config_sha256": hashlib.sha256(config).hexdigest(),
                "model_type": "fixture",
                "policy": policy,
                "schema": "fixture.generic-targets.v9",
                "target_count": 2,
                "target_digest": "a" * 64,
                "targets": ["lm_head", "model.layers.0.q_proj"],
                "weight_index_sha256": hashlib.sha256(index).hexdigest(),
            }
        ),
        encoding="utf-8",
    )
    receipt = preflight_lora_target_manifest(
        tmp_path,
        manifest,
        manifest_schema="fixture.generic-targets.v9",
        required_policy_digest=policy_digest,
    )
    assert receipt.targets == ("lm_head", "model.layers.0.q_proj")
    assert receipt.producer_target_digest == "sha256:" + "a" * 64
    assert (
        load_lora_target_receipt(
            manifest,
            manifest_schema="fixture.generic-targets.v9",
            required_policy_digest=policy_digest,
        )
        == receipt
    )

    changed = json.loads(manifest.read_text(encoding="utf-8"))
    changed["targets"][1] = "model.layers.0.missing"
    manifest.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="missing snapshot modules"):
        preflight_lora_target_manifest(tmp_path, manifest)


def _engine_state() -> HFEngineState:
    return HFEngineState(
        optimizer_step=0,
        composition_digest="sha256:" + "a" * 64,
        model_load_receipt_digest="sha256:" + "b" * 64,
        processor_fingerprint="sha256:" + "c" * 64,
        component_state={},
        controls_state={},
    )


class SaveableModel(torch.nn.Linear):
    def save_pretrained(self, directory, *, safe_serialization: bool):
        assert safe_serialization
        directory.mkdir()
        (directory / "model.safetensors").write_bytes(b"weights")


class StatelessSchedule:
    def state_dict(self):
        return {}

    def load_state_dict(self, state):
        assert state == {}


def test_checkpoint_staging_refuses_preexisting_path_without_mutation(tmp_path) -> None:
    destination = tmp_path / "checkpoint-step0-attempt1"
    destination.mkdir()
    sentinel = destination / "sentinel"
    sentinel.write_text("owned", encoding="utf-8")
    model = SaveableModel(2, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    with pytest.raises(HFMultimodalSFTError, match="cannot be overwritten"):
        stage_exact_checkpoint(
            destination,
            model=model,
            optimizer=optimizer,
            learning_rate_schedule=StatelessSchedule(),
            weight_decay_schedule=StatelessSchedule(),
            precision=StatelessSchedule(),
            state=_engine_state(),
        )
    assert sentinel.read_text(encoding="utf-8") == "owned"


def test_interrupted_checkpoint_staging_is_never_reused(tmp_path) -> None:
    class InterruptedModel(SaveableModel):
        interrupted = False

        def named_parameters(self, *args, **kwargs):
            if self.interrupted:
                raise RuntimeError("interrupted")
            return super().named_parameters(*args, **kwargs)

    destination = tmp_path / "checkpoint-step0-attempt1"
    model = InterruptedModel(2, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    model.interrupted = True
    with pytest.raises(RuntimeError, match="interrupted"):
        stage_exact_checkpoint(
            destination,
            model=model,
            optimizer=optimizer,
            learning_rate_schedule=StatelessSchedule(),
            weight_decay_schedule=StatelessSchedule(),
            precision=StatelessSchedule(),
            state=_engine_state(),
        )
    assert destination.is_dir()
    assert not (destination / "staging-complete.json").exists()
    with pytest.raises(HFMultimodalSFTError, match="cannot be overwritten"):
        stage_exact_checkpoint(
            destination,
            model=SaveableModel(2, 2),
            optimizer=optimizer,
            learning_rate_schedule=StatelessSchedule(),
            weight_decay_schedule=StatelessSchedule(),
            precision=StatelessSchedule(),
            state=_engine_state(),
        )


def test_complete_checkpoint_staging_is_durable_and_receipted(tmp_path) -> None:
    destination = tmp_path / "checkpoint-step0-attempt1"
    model = SaveableModel(2, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    digest = stage_exact_checkpoint(
        destination,
        model=model,
        optimizer=optimizer,
        learning_rate_schedule=StatelessSchedule(),
        weight_decay_schedule=StatelessSchedule(),
        precision=StatelessSchedule(),
        state=_engine_state(),
    )
    assert digest.startswith("sha256:")
    assert (destination / "engine-state.json").is_file()
    assert (destination / "optimizer.pt").is_file()
    assert (destination / "trainable-state.safetensors").is_file()
    assert not (destination / "trainable-state.pt").exists()
    assert not (destination / "model").exists()
    assert (destination / "rng.json").is_file()
    assert (destination / "rng-tensors.pt").is_file()
    assert (destination / "staging-complete.json").is_file()


def _restore(destination, model, optimizer):
    class Composition:
        def validate_resume_state(self, state):
            assert state == {}

    return restore_exact_checkpoint(
        destination,
        model=model,
        optimizer=optimizer,
        learning_rate_schedule=StatelessSchedule(),
        weight_decay_schedule=StatelessSchedule(),
        precision=StatelessSchedule(),
        composition=Composition(),
        expected_composition_digest="sha256:" + "a" * 64,
        expected_model_load_receipt_digest="sha256:" + "b" * 64,
        expected_processor_fingerprint="sha256:" + "c" * 64,
        controls_state_validator=lambda state: state == {},
    )


def test_exact_checkpoint_round_trip_validates_completion_before_restore(tmp_path):
    destination = tmp_path / "checkpoint-step0-attempt1"
    model = SaveableModel(2, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    expected = model.weight.detach().clone()
    stage_exact_checkpoint(
        destination,
        model=model,
        optimizer=optimizer,
        learning_rate_schedule=StatelessSchedule(),
        weight_decay_schedule=StatelessSchedule(),
        precision=StatelessSchedule(),
        state=_engine_state(),
    )
    model.weight.data.zero_()
    restored = _restore(destination, model, optimizer)
    assert restored.optimizer_step == 0
    assert torch.equal(model.weight, expected)


def test_exact_checkpoint_resume_reproduces_uninterrupted_trajectory(tmp_path):
    import numpy

    class Composition:
        def validate_resume_state(self, state):
            assert state == {}

    class Precision(StatelessSchedule):
        pass

    def update(model, optimizer, schedule):
        optimizer.zero_grad(set_to_none=True)
        draws = (
            random.random(),
            float(numpy.random.random()),
            float(torch.rand(())),
        )
        inputs = torch.randn(4, 3) + sum(draws)
        target = torch.randn(4, 2)
        loss = F.mse_loss(model(inputs), target)
        loss.backward()
        optimizer.step()
        schedule.step()
        return draws, float(loss.detach())

    random.seed(17)
    numpy.random.seed(17)
    torch.manual_seed(17)
    uninterrupted = torch.nn.Linear(3, 2)
    uninterrupted_optimizer = torch.optim.AdamW(
        uninterrupted.parameters(), lr=0.02
    )
    uninterrupted_schedule = torch.optim.lr_scheduler.StepLR(
        uninterrupted_optimizer, step_size=1, gamma=0.8
    )
    update(uninterrupted, uninterrupted_optimizer, uninterrupted_schedule)
    checkpoint = tmp_path / "checkpoint-step1-attempt1"
    state = HFEngineState(
        optimizer_step=1,
        composition_digest="sha256:" + "a" * 64,
        model_load_receipt_digest="sha256:" + "b" * 64,
        processor_fingerprint="sha256:" + "c" * 64,
        component_state={},
        controls_state={},
        runtime_state={"data_cursor": 1},
    )
    stage_exact_checkpoint(
        checkpoint,
        model=uninterrupted,
        optimizer=uninterrupted_optimizer,
        learning_rate_schedule=uninterrupted_schedule,
        weight_decay_schedule=StatelessSchedule(),
        precision=Precision(),
        state=state,
    )
    expected_draws, expected_loss = update(
        uninterrupted, uninterrupted_optimizer, uninterrupted_schedule
    )

    torch.manual_seed(999)
    resumed = torch.nn.Linear(3, 2)
    resumed_optimizer = torch.optim.AdamW(resumed.parameters(), lr=0.02)
    resumed_schedule = torch.optim.lr_scheduler.StepLR(
        resumed_optimizer, step_size=1, gamma=0.8
    )
    restored = restore_exact_checkpoint(
        checkpoint,
        model=resumed,
        optimizer=resumed_optimizer,
        learning_rate_schedule=resumed_schedule,
        weight_decay_schedule=StatelessSchedule(),
        precision=Precision(),
        composition=Composition(),
        expected_composition_digest=state.composition_digest,
        expected_model_load_receipt_digest=state.model_load_receipt_digest,
        expected_processor_fingerprint=state.processor_fingerprint,
        controls_state_validator=lambda value: value == {},
    )
    actual_draws, actual_loss = update(
        resumed, resumed_optimizer, resumed_schedule
    )

    assert restored.optimizer_step == 1
    assert restored.runtime_state == {"data_cursor": 1}
    assert actual_draws == expected_draws
    assert actual_loss == expected_loss
    for expected, actual in zip(
        uninterrupted.parameters(), resumed.parameters(), strict=True
    ):
        assert torch.equal(actual, expected)
    assert resumed_optimizer.state_dict()["param_groups"] == (
        uninterrupted_optimizer.state_dict()["param_groups"]
    )
    for expected_state, actual_state in zip(
        uninterrupted_optimizer.state.values(),
        resumed_optimizer.state.values(),
        strict=True,
    ):
        assert set(actual_state) == set(expected_state)
        for name, expected in expected_state.items():
            actual = actual_state[name]
            if isinstance(expected, torch.Tensor):
                assert torch.equal(actual, expected)
            else:
                assert actual == expected


def test_exact_checkpoint_rejects_changed_object_before_model_mutation(tmp_path):
    destination = tmp_path / "checkpoint-step0-attempt1"
    model = SaveableModel(2, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    stage_exact_checkpoint(
        destination,
        model=model,
        optimizer=optimizer,
        learning_rate_schedule=StatelessSchedule(),
        weight_decay_schedule=StatelessSchedule(),
        precision=StatelessSchedule(),
        state=_engine_state(),
    )
    model.weight.data.fill_(17)
    before = model.weight.detach().clone()
    mechanics = destination / "optimizer.pt"
    corrupted = bytearray(mechanics.read_bytes())
    corrupted[len(corrupted) // 2] ^= 1
    mechanics.chmod(0o640)
    mechanics.write_bytes(corrupted)
    with pytest.raises(HFMultimodalSFTError, match="completion receipt disagrees"):
        _restore(destination, model, optimizer)
    assert torch.equal(model.weight, before)


def test_exact_checkpoint_rejects_symlink_object(tmp_path):
    destination = tmp_path / "checkpoint-step0-attempt1"
    model = SaveableModel(2, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    stage_exact_checkpoint(
        destination,
        model=model,
        optimizer=optimizer,
        learning_rate_schedule=StatelessSchedule(),
        weight_decay_schedule=StatelessSchedule(),
        precision=StatelessSchedule(),
        state=_engine_state(),
    )
    rng = destination / "rng.json"
    outside = tmp_path / "outside.json"
    outside.write_bytes(rng.read_bytes())
    rng.unlink()
    rng.symlink_to(outside)
    with pytest.raises(HFMultimodalSFTError, match="symlink"):
        _restore(destination, model, optimizer)


def test_exact_checkpoint_refuses_mid_accumulation_boundary(tmp_path):
    state = HFEngineState(
        optimizer_step=0,
        composition_digest="sha256:" + "a" * 64,
        model_load_receipt_digest="sha256:" + "b" * 64,
        processor_fingerprint="sha256:" + "c" * 64,
        component_state={},
        controls_state={},
        microbatch_in_optimizer_step=1,
    )
    model = SaveableModel(2, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    with pytest.raises(HFMultimodalSFTError, match="accumulation boundary"):
        stage_exact_checkpoint(
            tmp_path / "checkpoint",
            model=model,
            optimizer=optimizer,
            learning_rate_schedule=StatelessSchedule(),
            weight_decay_schedule=StatelessSchedule(),
            precision=StatelessSchedule(),
            state=state,
        )
    assert not (tmp_path / "checkpoint").exists()


def _engine_harness(tmp_path, *, schedule_configuration=None, maximum_steps=1):
    """One self-contained fake object graph for the generic HF SFT engine.

    Extracted so more than one test can drive the same engine over the same
    doubles. The cadence and the run length are the only parameters: every test
    that varies them is asserting something about evaluation scheduling against
    an otherwise byte-identical training setup.
    """

    from rwkv_lab.training_runtime.data_pipeline import (
        BatchingImplementation,
        DeterministicSamplerConfiguration,
        FixedBatchingConfiguration,
        RawSample,
        RegisteredBatching,
        RegisteredSampler,
        SamplerImplementation,
        SplitSelection,
    )
    from rwkv_lab.training_runtime.evaluation_schedules import (
        EvaluationSchedule,
        EvaluationScheduleConfiguration,
    )
    from rwkv_lab.training_runtime.gradient_accumulation import (
        FixedGradientAccumulation,
        FixedGradientAccumulationConfiguration,
    )
    from rwkv_lab.training_runtime.objectives import (
        LinearHeadCrossEntropyConfiguration,
        LinearHeadCrossEntropyObjective,
    )

    class Tokenizer:
        def decode(self, values, *, skip_special_tokens):
            assert skip_special_tokens
            return " ".join(str(value) for value in values)

    timeline: list[str] = []
    generation_weights: list[torch.Tensor] = []

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            torch.manual_seed(4)
            self.embedding = torch.nn.Embedding(32, 8)
            self.head = torch.nn.Linear(8, 32, bias=False)

        def get_output_embeddings(self):
            return self.head

        def forward(
            self,
            *,
            input_ids,
            attention_mask,
            use_cache,
            output_hidden_states,
            return_dict,
        ):
            del attention_mask, use_cache, output_hidden_states, return_dict
            return SimpleNamespace(hidden_states=(self.embedding(input_ids),))

        def generate(self, *, input_ids, attention_mask, **_kwargs):
            del attention_mask
            timeline.append("generate")
            generation_weights.append(self.head.weight.detach().clone())
            suffix = torch.full(
                (input_ids.shape[0], 1), 7, dtype=torch.long, device=input_ids.device
            )
            return torch.cat((input_ids, suffix), dim=1)

        def save_pretrained(self, directory, *, safe_serialization):
            assert safe_serialization
            directory.mkdir()
            (directory / "model.safetensors").write_bytes(b"tiny")

    records = (
        RawSample("a", 0, {"tokens": [1, 2, 3, 4]}),
        RawSample("b", 1, {"tokens": [2, 3, 4, 5]}),
        RawSample("held", 2, {"tokens": [5, 6, 8, 9]}),
        RawSample("test", 3, {"tokens": [6, 7, 8, 9]}),
    )
    dataset_root = tmp_path / "frozen-data"
    dataset_root.mkdir()
    for name in ("manifest.json", "train.jsonl", "validation.jsonl", "test.jsonl"):
        (dataset_root / name).write_text("{}\n", encoding="utf-8")
    frozen_configuration = JsonlFrozenTokenSplitsConfiguration(
        dataset_root=str(dataset_root),
        content_fingerprint="sha256:" + "d" * 64,
        declared_columns=("id", "split", "tokens"),
        token_column="tokens",
        id_column="id",
    )

    class Source:
        configuration = frozen_configuration

        def records_for_split(self, split):
            selected = {
                "train": records[:2],
                "validation": records[2:3],
                "test": records[3:],
            }
            return selected[split]

        def component_state(self):
            return {"content_fingerprint": "sha256:" + "d" * 64, "cursor": 0}

        def restore_component_state(self, state):
            assert state == self.component_state()

    consumed: dict[str, list[str]] = {"train": [], "validation": [], "test": []}

    class Processor:
        configuration = object()

        def __init__(self, selection):
            self.selection = selection

        def process(self, sample, *, image_root):
            assert image_root is None
            consumed[self.selection].append(sample.sample_id)
            return ProcessedSample(
                sample.sample_id,
                sample.ordinal,
                sample.values,
                token_length=len(sample.values["tokens"]),
            )

    class Split:
        def __init__(self, selection):
            self.selection = selection

        def select(self, identities):
            selected = tuple(identities)
            digest_character = {"train": "d", "validation": "e", "test": "f"}[
                self.selection
            ]
            return SplitSelection(
                selected, (), "sha256:" + digest_character * 64
            )

    class Pipeline:
        def __init__(self, selection):
            self.source = Source()
            self.processor = Processor(selection)
            self.mapper = SimpleNamespace(
                configuration=CausalTokensMapperConfiguration(
                    token_column="tokens", maximum_tokens=8
                )
            )
            self.collator = SimpleNamespace(
                configuration=PaddedCollatorConfiguration(
                    pad_token_id=0,
                    label_pad_token_id=-100,
                    pad_to_multiple=1,
                    maximum_sequence_length=8,
                )
            )
            self.sampler = RegisteredSampler(
                SamplerImplementation.DETERMINISTIC_V1,
                DeterministicSamplerConfiguration(seed=1, shuffle=False),
            )
            self.batching = RegisteredBatching(
                BatchingImplementation.FIXED_V1,
                FixedBatchingConfiguration(
                    batch_size=1, drop_last=False, prefetch_workers=0
                ),
            )
            self.split_selector = Split(selection)

        def validate_schema(self):
            return None

    model = Model()
    initial = model.head.weight.detach().clone()

    class Receipt:
        exact = True
        digest = "sha256:" + "a" * 64
        auxiliary_fingerprint = "sha256:" + "b" * 64

    class Loaded:
        tokenizer_or_processor = Tokenizer()
        receipt = Receipt()

        def __init__(self, loaded_model):
            self.model = loaded_model

        def component_state(self):
            return {
                "base_checkpoint_fingerprint": "sha256:" + "c" * 64,
                "load_receipt_digest": self.receipt.digest,
            }

    class TrainabilityResult:
        adapter_backed = False

        def __init__(self, result_model):
            self.model = result_model
            self.trainable_parameter_names = tuple(
                name for name, _ in result_model.named_parameters()
            )

        def component_state(self, *, adapter_state_manifest=None):
            assert adapter_state_manifest is None
            return {"trainable_parameter_manifest": "sha256:" + "f" * 64}

    class Precision:
        compute_dtype = torch.float32

        def convert_module(self, loaded_model, device):
            timeline.append("device_move")
            return loaded_model.to(device)

        def reduce(self, value):
            return value.float()

        def state_dict(self):
            return {}

        def load_state_dict(self, state):
            assert state == {}

    class Schedule:
        def __init__(self, optimizer):
            self.optimizer = optimizer

        def step(self, *_args):
            return None

        def state_dict(self):
            return {}

        def load_state_dict(self, state):
            assert state == {}

    class Composition:
        composition_digest = "sha256:" + "1" * 64

        def __init__(self):
            self.components = {
                "evaluator": SimpleNamespace(
                    descriptor_digest="sha256:" + "2" * 64
                ),
                "artifact_renderer": SimpleNamespace(
                    descriptor_digest="sha256:" + "3" * 64
                ),
                **{
                    slot: SimpleNamespace(descriptor_digest="sha256:" + "6" * 64)
                    for slot in (
                        "data",
                        "generation_policy",
                        "processor",
                        "qualitative_samples",
                        "sample_mapping",
                        "test_split",
                    )
                },
            }

        def validate_resume_state(self, state):
            return state

    class Components:
        composition = Composition()

        def __init__(self):
            self.training = Pipeline("train")
            self.evaluation = Pipeline("validation")
            self.test = Pipeline("test")

        def data_pipeline(self, *, split_slot):
            timeline.append(f"data_pipeline:{split_slot}")
            return {
                "split": self.training,
                "evaluation_split": self.evaluation,
                "test_split": self.test,
            }[split_slot]

        def model_loader(self, *, slot):
            assert slot == "model_loader"
            def load(**_kwargs):
                timeline.append("model_load")
                return Loaded(model)

            return SimpleNamespace(load=load)

        def trainability(self):
            def apply(item):
                with torch.no_grad():
                    item.head.weight.add_(0.01)
                return TrainabilityResult(item)

            return SimpleNamespace(apply=apply)

        def precision(self):
            return Precision()

        def activation_memory(self):
            return SimpleNamespace(
                apply=lambda _model: None,
                component_state=lambda: {
                    "enabled": True,
                    "use_reentrant": False,
                },
                restore_component_state=lambda state, _model: state,
            )

        def optimizer(self, parameters):
            timeline.append("optimizer")
            return torch.optim.SGD(tuple(parameters), lr=0.05)

        def learning_rate_schedule(self, optimizer):
            return Schedule(optimizer)

        def weight_decay_schedule(self, optimizer):
            return Schedule(optimizer)

        def objective(self):
            return LinearHeadCrossEntropyObjective(
                LinearHeadCrossEntropyConfiguration(
                    chunk_size=32, prefer_fused=False
                )
            )

        def gradient_accumulation(self):
            return FixedGradientAccumulation(
                FixedGradientAccumulationConfiguration(
                    microbatches_per_optimizer_step=1
                )
            )

        def qualitative_samples(self):
            return SimpleNamespace(
                configuration=SimpleNamespace(sample_count=1, identity_field="id"),
                select=lambda population, selector_digest, dataset_root: SimpleNamespace(
                    identities=tuple(population[:1]),
                    identities_digest="sha256:" + "4" * 64,
                    selector_digest=selector_digest,
                ),
            )

        def checkpoint_policy(self):
            return SimpleNamespace(
                component_state=lambda **values: {
                    key: values[key]
                    for key in (
                        "last_published_step",
                        "publication_manifest",
                        "retention_manifest",
                    )
                },
                due=lambda step, final: final,
                retained_steps=lambda steps: tuple(sorted(set(steps))),
            )

        def evaluator(self):
            return SimpleNamespace(
                configuration=SimpleNamespace(
                    maximum_examples=0, metrics=("test_loss",)
                ),
                reduce=lambda values, weights: sum(
                    v * w for v, w in zip(values, weights, strict=True)
                )
                / sum(weights),
            )

        def evaluation_schedule(self):
            return EvaluationSchedule(
                schedule_configuration
                or EvaluationScheduleConfiguration(
                    defer_full_scalar=False,
                    full_step_zero=True,
                    final=True,
                )
            )

        def artifact_renderer(self):
            return SimpleNamespace(
                render=lambda **values: {
                    "evidence": values["evidence"],
                    "step": values["step"],
                }
            )

        def generation_policy(self):
            return SimpleNamespace(
                configuration=SimpleNamespace(
                    maximum_new_tokens=128,
                    generation_batch_size=1,
                    padding_side="left",
                    use_cache=True,
                ),
                digest="sha256:" + "5" * 64,
            )

        def learning_rate_configuration(self):
            return None, SimpleNamespace(max_steps=maximum_steps)

        def gradient_clipping(self, parameters):
            return torch.nn.utils.clip_grad_norm_(tuple(parameters), 1.0)

    class Controls:
        checkpoint_boundary_requested = False
        checkpoint_completion_requested = False

        def __init__(
            self,
            *,
            fail_gallery=False,
            fail_artifact=False,
            fail_before_step=0,
            patch=None,
            patch_at=None,
        ):
            self.events = []
            self.safe_points = []
            self.trajectory = []
            self.fail_gallery = fail_gallery
            self.fail_artifact = fail_artifact
            # Interrupts the loop at the start of the named optimizer step, so
            # a test can resume from a checkpoint that is genuinely mid-run
            # rather than from the launch or the final one.
            self.fail_before_step = fail_before_step
            # Delivers one control patch at exactly one (phase, step) safe
            # point, so a test can show which safe points accept a live
            # cadence change and which refuse it.
            self.patch = dict(patch or {})
            self.patch_at = patch_at

        def _assignments(self, phase, step):
            # One-shot, like a real control command: the authority pops each
            # patch once, and a safe point can be re-entered at the same step.
            if self.patch_at == (phase, step):
                self.patch_at = None
                return dict(self.patch)
            return {}

        def checkpoint_state(self):
            return {"effective_control_revision": 0, "effective_controls": {}}

        def poll_initialization(self):
            timeline.append("initialization")
            self.safe_points.append(("initialization", 0))

        def microbatch(self, step, applier):
            if self.fail_before_step and step == self.fail_before_step:
                raise RuntimeError("simulated crash before optimizer step")
            self.safe_points.append(("microbatch", step))
            applier({}, self._assignments("microbatch", step))

        def optimizer_step(self, step, applier):
            # Mutation sentinel. The real controller refuses an optimizer step
            # past the attempt baseline until the typed evidence is durable, so
            # an engine that reached this point without having published it is
            # already wrong here rather than only in production.
            assert any(
                kind == "eval_examples" for kind, *_ in self.events
            ), "optimizer step reached before attempt-baseline eval-examples"
            self.safe_points.append(("optimizer_step", step))
            # Sampled immediately after the optimizer mutated, so a trajectory
            # comparison sees the exact post-update state at every step.
            self.trajectory.append(
                (
                    step,
                    model.head.weight.detach().clone(),
                    model.embedding.weight.detach().clone(),
                    torch.random.get_rng_state().clone(),
                    random.getstate(),
                )
            )
            applier({}, self._assignments("optimizer_step", step))

        def evaluation(self, step, applier):
            self.safe_points.append(("evaluation", step))
            applier({}, self._assignments("evaluation", step))

        def checkpoint(self, step, applier):
            self.safe_points.append(("checkpoint", step))
            applier({}, self._assignments("checkpoint", step))

        def verify_checkpoint_state(self, state):
            assert state == self.checkpoint_state()

        def publish_policy_checkpoint(self, request):
            self.events.append(("checkpoint", request, model.head.weight.detach().clone()))
            return SimpleNamespace(
                manifest_sha256="sha256:" + str(len(self.events)) * 64,
                artifact_id=f"checkpoint-{len(self.events)}",
            )

        def publish_evaluation_gallery(self, request, *, checkpoint):
            self.events.append(("gallery", request, model.head.weight.detach().clone()))
            if self.fail_gallery:
                self.fail_gallery = False
                raise RuntimeError("simulated crash after launch checkpoint")
            return SimpleNamespace(
                artifact_id="gallery", manifest_sha256="sha256:" + "7" * 64
            )

        def publish_evaluation_examples(self, request):
            self.events.append(
                ("eval_examples", request, model.head.weight.detach().clone())
            )
            return SimpleNamespace(
                artifact_id=f"eval-examples-{request.optimizer_step}",
                manifest_sha256="sha256:" + "e" * 64,
            )

        def publish_artifact(self, request):
            if self.fail_artifact:
                self.fail_artifact = False
                raise RuntimeError("simulated crash after final checkpoint")
            timeline.append("test_artifact")
            self.events.append(("artifact", request, model.head.weight.detach().clone()))
            return SimpleNamespace(
                artifact_id="test-eval", manifest_sha256="sha256:" + "8" * 64
            )

        def publish_final_evaluation(self, request):
            self.events.append(
                ("final_evaluation", request, model.head.weight.detach().clone())
            )

        def publish_requested_checkpoint(self, request):
            raise AssertionError(request)

    class Observability:
        def __init__(self):
            self.metrics = []

        def publish_if_declared(self, name, value, *, step):
            if name == "eval.test_loss":
                timeline.append("test_metric")
            self.metrics.append((name, value, step))

        def optimizer_step(self, step, phase):
            return None

    return SimpleNamespace(
        Components=Components,
        Controls=Controls,
        Observability=Observability,
        consumed=consumed,
        generation_weights=generation_weights,
        initial=initial,
        model=model,
        timeline=timeline,
    )


def test_generic_causal_loop_publishes_step_zero_before_optimizer_mutation(tmp_path):
    harness = _engine_harness(tmp_path)
    Components = harness.Components
    Controls = harness.Controls
    Observability = harness.Observability
    model = harness.model
    initial = harness.initial
    timeline = harness.timeline
    generation_weights = harness.generation_weights

    failed_controls = Controls(fail_gallery=True)
    failed_observability = Observability()
    with pytest.raises(RuntimeError, match="crash after launch checkpoint"):
        run_hf_multimodal_sft(
            invocation=SimpleNamespace(attempt_id="attempt-crash"),
            components=Components(),
            run_directory=tmp_path / "run",
            controls=failed_controls,
            observability=failed_observability,
            step_profiler=SimpleNamespace(
                input_wait=nullcontext, step=lambda _step: None
            ),
            resume_directory=None,
            device="cpu",
        )
    launch_checkpoint = failed_controls.events[0][1].source_directory
    assert [event[0] for event in failed_controls.events] == [
        "checkpoint",
        "gallery",
    ]

    controls = Controls(fail_artifact=True)
    observability = Observability()
    with pytest.raises(RuntimeError, match="crash after final checkpoint"):
        run_hf_multimodal_sft(
            invocation=SimpleNamespace(attempt_id="attempt-1"),
            components=Components(),
            run_directory=tmp_path / "run",
            controls=controls,
            observability=observability,
            step_profiler=SimpleNamespace(
                input_wait=nullcontext, step=lambda _step: None
            ),
            resume_directory=launch_checkpoint,
            resume_parent_artifact_ids=("checkpoint-crash",),
            resume_checkpoint_manifest_digest="sha256:" + "1" * 64,
            device="cpu",
        )
    final_checkpoint = controls.events[4][1].source_directory
    final_checkpoint_state = json.loads(
        (final_checkpoint / "engine-state.json").read_text(encoding="utf-8")
    )
    assert final_checkpoint_state["runtime_state"]["finalization_pending"] is True
    before_eval_only_recovery = model.head.weight.detach().clone()
    final_controls = Controls()
    final_observability = Observability()
    step = run_hf_multimodal_sft(
        invocation=SimpleNamespace(
            attempt_id="attempt-finalize",
            observability={
                "metrics": (
                    {"name": "eval.test_loss", "step_domain": "optimizer_step"},
                )
            },
        ),
        components=Components(),
        run_directory=tmp_path / "run",
        controls=final_controls,
        observability=final_observability,
        step_profiler=SimpleNamespace(input_wait=nullcontext, step=lambda _step: None),
        resume_directory=final_checkpoint,
        resume_parent_artifact_ids=("checkpoint-5",),
        resume_checkpoint_manifest_digest="sha256:" + "5" * 64,
        device="cpu",
    )
    assert step == 1
    assert torch.equal(model.head.weight, before_eval_only_recovery)
    # The universal typed evidence lands with the launch checkpoint, before the
    # loop can reach an optimizer mutation, and a resumed attempt republishes it
    # at its own baseline rather than leaning on the previous attempt's.
    assert [event[0] for event in controls.events] == [
        "checkpoint",
        "gallery",
        "eval_examples",
        "checkpoint",
        "checkpoint",
    ]
    assert [event[0] for event in final_controls.events] == [
        "checkpoint",
        "eval_examples",
        "artifact",
        "gallery",
        "final_evaluation",
    ]
    closure_request = final_controls.events[2][1]
    expected_context = {
        "api_version": "rwkv-lab.hf-final-member-context/v1",
        "components": {
            "artifact_renderer": "sha256:" + "3" * 64,
            "data": "sha256:" + "6" * 64,
            "evaluator": "sha256:" + "2" * 64,
            "generation_policy": "sha256:" + "6" * 64,
            "processor": "sha256:" + "6" * 64,
            "sample_mapping": "sha256:" + "6" * 64,
            "test_split": "sha256:" + "6" * 64,
        },
        "member_id": "test",
    }
    expected_context_digest = "sha256:" + hashlib.sha256(
        json.dumps(expected_context, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert closure_request.member_context_digests == {
        "test": expected_context_digest
    }
    assert not torch.equal(generation_weights[0], generation_weights[1])
    assert torch.equal(failed_controls.events[0][2], generation_weights[1])
    assert torch.equal(controls.events[0][2], generation_weights[2])
    step_zero_item = controls.events[1][1].items[0]
    assert (
        step_zero_item.sampling_attributes["baseline"]
        == step_zero_item.sampling_attributes["current"]
    )
    frozen_eval_manifest = step_zero_item.sampling_attributes["eval_manifest_digest"]
    assert frozen_eval_manifest.startswith("sha256:")
    assert step_zero_item.sampling_attributes == {
        "teacher_target": "8 9",
        "baseline": "7",
        "current": "7",
        "ordered_identities_digest": "sha256:" + "4" * 64,
        "eval_manifest_digest": frozen_eval_manifest,
    }
    final_item = final_controls.events[3][1].items[0]
    assert final_item.heldout_item_id == step_zero_item.heldout_item_id
    assert final_item.prompt_or_condition_digest == (
        step_zero_item.prompt_or_condition_digest
    )
    # Same frozen evaluation manifest at both ends of the run: the step-zero
    # baseline and the final gallery are comparable because they were read
    # from the identical declared subset with the identical decode policy.
    assert final_item.sampling_attributes == {
        "teacher_target": "8 9",
        "baseline": "7",
        "current": "7",
        "ordered_identities_digest": "sha256:" + "4" * 64,
        "eval_manifest_digest": frozen_eval_manifest,
    }
    assert controls.events[1][1].evaluator_profile_digest == frozen_eval_manifest
    assert (
        final_controls.events[3][1].evaluator_profile_digest == frozen_eval_manifest
    )
    assert not torch.equal(model.head.weight, initial)
    assert timeline.index("generate") < timeline.index("optimizer")
    assert timeline.index("initialization") < timeline.index("data_pipeline:split")
    assert timeline.index("data_pipeline:test_split") < timeline.index("model_load")
    assert timeline.index("model_load") < timeline.index("device_move")
    assert len(generation_weights) == 7
    assert torch.equal(generation_weights[0], initial)
    assert not torch.equal(generation_weights[-1], initial)
    assert step_zero_item.sampling_attributes["ordered_identities_digest"] == (
        "sha256:" + "4" * 64
    )
    checkpoint_state = json.loads(
        (
            controls.events[0][1].source_directory / "engine-state.json"
        ).read_text(encoding="utf-8")
    )
    assert checkpoint_state["component_state"]["qualitative_samples"] == {
        "identities_digest": "sha256:" + "4" * 64,
        "selector_digest": "sha256:" + "e" * 64,
    }
    assert checkpoint_state["runtime_state"]["launch_gallery_complete"] is False
    assert checkpoint_state["runtime_state"]["publication_pending"] is True
    assert checkpoint_state["runtime_state"]["published_steps"] == [0]
    replayed_checkpoint_state = json.loads(
        (
            controls.events[0][1].source_directory / "engine-state.json"
        ).read_text(encoding="utf-8")
    )
    assert replayed_checkpoint_state["runtime_state"]["publication_pending"] is True
    assert replayed_checkpoint_state["runtime_state"]["published_steps"] == [0]
    pending_manifest = replayed_checkpoint_state["runtime_state"][
        "checkpoint_policy"
    ]["publication_manifest"]
    assert pending_manifest.startswith("sha256:")
    assert pending_manifest != "sha256:" + "1" * 64
    assert (
        replayed_checkpoint_state["runtime_state"]["checkpoint_policy"]
        == replayed_checkpoint_state["component_state"]["checkpoint_policy"]
    )
    baseline_checkpoint_state = json.loads(
        (
            controls.events[3][1].source_directory / "engine-state.json"
        ).read_text(encoding="utf-8")
    )
    assert baseline_checkpoint_state["runtime_state"]["baseline_complete"] is True
    assert baseline_checkpoint_state["runtime_state"][
        "baseline_checkpoint_artifact_id"
    ] == "checkpoint-1"
    assert (
        baseline_checkpoint_state["runtime_state"]["launch_gallery_complete"]
        is True
    )
    assert not (
        controls.events[3][1].source_directory / "sealed-test-baseline"
    ).exists()
    quarantined_baseline = tmp_path / "run" / ".private-test-quarantine" / "baseline"
    assert quarantined_baseline.is_dir()
    assert quarantined_baseline.stat().st_mode & 0o077 == 0
    assert all(path.stat().st_mode & 0o277 == 0 for path in quarantined_baseline.iterdir())
    # The declared milestone plan is published before any milestone is paid
    # for, and the first loss the run reports is still the step-zero baseline.
    assert [name for name, _value, _step in failed_observability.metrics[:4]] == [
        "eval.planned_scalar_full_milestones",
        "eval.planned_scalar_probe_milestones",
        "eval.planned_qualitative_milestones",
        "eval.cadence_revision",
    ]
    assert all(step == 0 for _name, _value, step in failed_observability.metrics[:4])
    first_loss = next(
        (name, value, step)
        for name, value, step in failed_observability.metrics
        if name.endswith("loss")
    )
    assert first_loss[0] == "eval.loss"
    assert first_loss[2] == 0
    assert all(
        not (name == "eval.test_loss" and at_step == 0)
        for name, _value, at_step in (
            *failed_observability.metrics,
            *observability.metrics,
        )
    )
    final_test_metrics = [
        (value, at_step)
        for name, value, at_step in final_observability.metrics
        if name == "eval.test_loss"
    ]
    assert len(final_test_metrics) == 1
    assert final_test_metrics[0][1] == 1
    assert final_controls.events[2][1].output_name == "test_eval"
    assert final_controls.events[2][1].parent_artifact_ids == (
        "checkpoint-1",
        "checkpoint-5",
    )
    bundle_receipt = json.loads(
        (
            tmp_path / "run" / "test-caption-bundle-step-1" / "receipt.json"
        ).read_text(encoding="utf-8")
    )
    assert bundle_receipt["baseline_checkpoint_artifact_id"] == "checkpoint-1"
    assert bundle_receipt["final_checkpoint_artifact_id"] == "checkpoint-5"
    assert timeline.index("test_artifact") < timeline.index("test_metric")
    assert controls.safe_points == [
        ("initialization", 0),
        ("evaluation", 0),
        ("evaluation", 0),
        ("microbatch", 1),
        ("optimizer_step", 1),
        ("evaluation", 1),
        ("checkpoint", 1),
        ("evaluation", 1),
    ]


def test_checkpoint_evidence_identity_never_reads_live_model_tensors():
    import rwkv_lab.trainvm_adapters.hf_multimodal_sft as engine

    assert not hasattr(engine, "_trainable_tensor_digest")
    assert engine._checkpoint_model_state_digest(
        model_load_receipt="sha256:" + "a" * 64,
        checkpoint_artifact_id="checkpoint-exact",
        manifest_digest="sha256:" + "b" * 64,
    ).startswith("sha256:")


def test_test_caption_evidence_batches_by_resolution_and_restores_manifest_order(
    tmp_path, monkeypatch
):
    import rwkv_lab.trainvm_adapters.hf_multimodal_sft as engine

    samples = tuple(
        ProcessedSample(
            sample_id,
            ordinal,
            {"caption": f"target-{sample_id}"},
            image=object(),
            image_size=size,
        )
        for ordinal, (sample_id, size) in enumerate(
            (
                ("a", (512, 512)),
                ("b", (1024, 512)),
                ("c", (500, 500)),
                ("d", (1000, 500)),
                ("e", (510, 510)),
            )
        )
    )
    calls: list[tuple[str, ...]] = []

    def generate(*, samples, maximum_new_tokens, use_cache, **_values):
        assert maximum_new_tokens == 768
        assert use_cache is True
        identities = tuple(sample.sample_id for sample in samples)
        calls.append(identities)
        return tuple(
            (
                f"target-{sample.sample_id}",
                f"generated-{sample.sample_id}",
                "sha256:" + str(sample.ordinal) * 64,
            )
            for sample in samples
        )

    monkeypatch.setattr(engine, "generate_hf_captions", generate)
    directory = engine._write_test_caption_evidence(
        directory=tmp_path / "evidence",
        stack=object(),
        codec=object(),
        samples=samples,
        device=torch.device("cpu"),
        step=0,
        model_load_receipt="sha256:" + "a" * 64,
        checkpoint_artifact_id="checkpoint-baseline",
        checkpoint_manifest_digest="sha256:" + "0" * 64,
        model_state_digest="sha256:" + "1" * 64,
        split_membership_digest="sha256:" + "b" * 64,
        decode_policy_digest="sha256:" + "c" * 64,
        model_state_mode="base_adapters_disabled",
        maximum_new_tokens=768,
        generation_batch_size=2,
        use_cache=True,
    )
    records = tuple(
        json.loads(line)
        for line in (directory / "captions.jsonl").read_text().splitlines()
    )
    assert calls == [("a", "c"), ("e",), ("b", "d")]
    assert tuple(record["sample_id"] for record in records) == (
        "a",
        "b",
        "c",
        "d",
        "e",
    )
    assert tuple(record["text"] for record in records) == tuple(
        f"generated-{sample.sample_id}" for sample in samples
    )
    engine._write_test_scalar_receipt(
        directory=directory,
        loss=1.25,
        records=len(samples),
        split_membership_digest="sha256:" + "b" * 64,
        step=0,
    )
    engine._write_test_scalar_receipt(
        directory=directory,
        loss=1.25,
        records=len(samples),
        split_membership_digest="sha256:" + "b" * 64,
        step=0,
    )
    with pytest.raises(HFMultimodalSFTError, match="scalar evidence is inconsistent"):
        engine._write_test_scalar_receipt(
            directory=directory,
            loss=1.5,
            records=len(samples),
            split_membership_digest="sha256:" + "b" * 64,
            step=0,
        )
    final = engine._write_test_caption_evidence(
        directory=tmp_path / "final",
        stack=object(),
        codec=object(),
        samples=samples,
        device=torch.device("cpu"),
        step=745,
        model_load_receipt="sha256:" + "a" * 64,
        checkpoint_artifact_id="checkpoint-final",
        checkpoint_manifest_digest="sha256:" + "9" * 64,
        model_state_digest="sha256:" + "2" * 64,
        split_membership_digest="sha256:" + "b" * 64,
        decode_policy_digest="sha256:" + "c" * 64,
        model_state_mode="trained",
        maximum_new_tokens=768,
        generation_batch_size=2,
        use_cache=True,
    )
    engine._write_test_scalar_receipt(
        directory=final,
        loss=0.75,
        records=len(samples),
        split_membership_digest="sha256:" + "b" * 64,
        step=745,
    )
    bundle_identity = {
        "baseline_checkpoint_artifact_id": "checkpoint-baseline",
        "baseline_checkpoint_manifest_digest": "sha256:" + "0" * 64,
        "final_checkpoint_artifact_id": "checkpoint-final",
        "final_checkpoint_manifest_digest": "sha256:" + "9" * 64,
        "final_model_state_digest": "sha256:" + "2" * 64,
    }
    bundle = engine._bundle_test_caption_evidence(
        directory=tmp_path / "bundle",
        baseline=directory,
        final=final,
        split_membership_digest="sha256:" + "b" * 64,
        **bundle_identity,
    )
    bundle_receipt = json.loads((bundle / "receipt.json").read_text())
    assert bundle_receipt["split_membership_digest"] == "sha256:" + "b" * 64
    assert (bundle / "baseline" / "captions.jsonl").is_file()
    assert (bundle / "final" / "captions.jsonl").is_file()
    assert (
        engine._bundle_test_caption_evidence(
            directory=tmp_path / "bundle",
            baseline=directory,
            final=final,
            split_membership_digest="sha256:" + "b" * 64,
            **bundle_identity,
        )
        == bundle
    )
    (directory / "unexpected.txt").write_text("not sealed", encoding="utf-8")
    with pytest.raises(HFMultimodalSFTError, match="inexact file layout"):
        engine._bundle_test_caption_evidence(
            directory=tmp_path / "bundle-source-extra",
            baseline=directory,
            final=final,
            split_membership_digest="sha256:" + "b" * 64,
            **bundle_identity,
        )
    (directory / "unexpected.txt").unlink()
    baseline_link = tmp_path / "baseline-link"
    baseline_link.symlink_to(directory, target_is_directory=True)
    with pytest.raises(HFMultimodalSFTError, match="cannot be a symlink"):
        engine._bundle_test_caption_evidence(
            directory=tmp_path / "bundle-source-link",
            baseline=baseline_link,
            final=final,
            split_membership_digest="sha256:" + "b" * 64,
            **bundle_identity,
        )
    identity_path = directory / "identity.json"
    identity_contents = identity_path.read_bytes()
    identity_path.unlink()
    identity_path.write_bytes(identity_contents + b" ")
    with pytest.raises(HFMultimodalSFTError, match="receipts are inconsistent"):
        engine._bundle_test_caption_evidence(
            directory=tmp_path / "bundle-tampered-identity",
            baseline=directory,
            final=final,
            split_membership_digest="sha256:" + "b" * 64,
            **bundle_identity,
        )
    identity_path.unlink()
    identity_path.write_bytes(identity_contents)
    final_captions_path = final / "captions.jsonl"
    final_receipt_path = final / "receipt.json"
    final_captions_contents = final_captions_path.read_bytes()
    final_receipt_contents = final_receipt_path.read_bytes()
    final_records = tuple(
        json.loads(line) for line in final_captions_contents.splitlines()
    )
    final_records[0]["target"] = "misaligned-teacher-target"
    tampered_captions = b"".join(
        json.dumps(
            record,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        + b"\n"
        for record in final_records
    )
    final_receipt = json.loads(final_receipt_contents)
    final_receipt["captions_sha256"] = (
        "sha256:" + hashlib.sha256(tampered_captions).hexdigest()
    )
    final_captions_path.unlink()
    final_captions_path.write_bytes(tampered_captions)
    final_receipt_path.unlink()
    final_receipt_path.write_text(
        json.dumps(final_receipt, separators=(",", ":"), sort_keys=True),
        encoding="utf-8",
    )
    with pytest.raises(HFMultimodalSFTError, match="not identity-aligned"):
        engine._bundle_test_caption_evidence(
            directory=tmp_path / "bundle-misaligned-target",
            baseline=directory,
            final=final,
            split_membership_digest="sha256:" + "b" * 64,
            **bundle_identity,
        )
    final_captions_path.unlink()
    final_captions_path.write_bytes(final_captions_contents)
    final_receipt_path.unlink()
    final_receipt_path.write_bytes(final_receipt_contents)
    (bundle / "unexpected.txt").write_text("not sealed", encoding="utf-8")
    with pytest.raises(HFMultimodalSFTError, match="bundle is inconsistent"):
        engine._bundle_test_caption_evidence(
            directory=bundle,
            baseline=directory,
            final=final,
            split_membership_digest="sha256:" + "b" * 64,
            **bundle_identity,
        )
    (bundle / "unexpected.txt").unlink()
    identity = bundle / "baseline" / "identity.json"
    identity.unlink()
    identity.symlink_to(directory / "identity.json")
    with pytest.raises(HFMultimodalSFTError, match="bundle is inconsistent"):
        engine._bundle_test_caption_evidence(
            directory=bundle,
            baseline=directory,
            final=final,
            split_membership_digest="sha256:" + "b" * 64,
            **bundle_identity,
        )


def test_test_caption_evidence_retries_failed_ids_and_collapses_latest_records(
    tmp_path, monkeypatch
) -> None:
    import rwkv_lab.trainvm_adapters.hf_multimodal_sft as engine

    samples = tuple(
        ProcessedSample(
            sample_id,
            ordinal,
            {"caption": f"target-{sample_id}"},
            image=object(),
            image_size=(512, 512),
        )
        for ordinal, sample_id in enumerate(("a", "b"))
    )
    attempts = 0

    def generate(*, samples, **_values):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("transient generation failure")
        return tuple(
            (
                f"target-{sample.sample_id}",
                f"generated-{attempts}-{sample.sample_id}",
                "sha256:" + f"{sample.ordinal:064x}",
            )
            for sample in samples
        )

    monkeypatch.setattr(engine, "generate_hf_captions", generate)
    arguments = {
        "directory": tmp_path / "evidence",
        "stack": object(),
        "codec": object(),
        "samples": samples,
        "device": torch.device("cpu"),
        "step": 0,
        "model_load_receipt": "sha256:" + "a" * 64,
        "checkpoint_artifact_id": "checkpoint-baseline",
        "checkpoint_manifest_digest": "sha256:" + "0" * 64,
        "model_state_digest": "sha256:" + "1" * 64,
        "split_membership_digest": "sha256:" + "b" * 64,
        "decode_policy_digest": "sha256:" + "c" * 64,
        "model_state_mode": "base_adapters_disabled",
        "maximum_new_tokens": 32,
        "generation_batch_size": 2,
        "use_cache": True,
    }
    with pytest.raises(engine.HFMultimodalSFTError, match="failed generations"):
        engine._write_test_caption_evidence(**arguments)
    failed = tuple(
        json.loads(line)
        for line in (
            tmp_path / "evidence" / "captions.partial.jsonl"
        ).read_text().splitlines()
    )
    assert [record["status"] for record in failed] == ["failed", "failed"]

    # A torn final append is the only malformed input tolerated during recovery.
    with (tmp_path / "evidence" / "captions.partial.jsonl").open("ab") as output:
        output.write(b'{"sample_id":"torn"')
    directory = engine._write_test_caption_evidence(**arguments)
    assert not (directory / "captions.partial.jsonl").exists()
    final = tuple(
        json.loads(line)
        for line in (directory / "captions.jsonl").read_text().splitlines()
    )
    assert tuple(record["sample_id"] for record in final) == ("a", "b")
    assert tuple(record["text"] for record in final) == (
        "generated-2-a",
        "generated-2-b",
    )
    assert json.loads((directory / "receipt.json").read_text())["failures"] == 0


def test_all_error_final_audit_cannot_seal_completion_evidence(
    tmp_path, monkeypatch
) -> None:
    import rwkv_lab.trainvm_adapters.hf_multimodal_sft as engine

    samples = tuple(
        ProcessedSample(
            sample_id,
            ordinal,
            {"caption": f"target-{sample_id}"},
            image=object(),
            image_size=(512, 512),
        )
        for ordinal, sample_id in enumerate(("a", "b", "c"))
    )
    monkeypatch.setattr(
        engine,
        "generate_hf_captions",
        lambda **_values: (_ for _ in ()).throw(
            RuntimeError("cudaErrorLaunchTimeout")
        ),
    )
    directory = tmp_path / "final-audit"
    with pytest.raises(engine.HFMultimodalSFTError, match="failed generations"):
        engine._write_test_caption_evidence(
            directory=directory,
            stack=object(),
            codec=object(),
            samples=samples,
            device=torch.device("cpu"),
            step=745,
            model_load_receipt="sha256:" + "a" * 64,
            checkpoint_artifact_id="checkpoint-final",
            checkpoint_manifest_digest="sha256:" + "b" * 64,
            model_state_digest="sha256:" + "c" * 64,
            split_membership_digest="sha256:" + "d" * 64,
            decode_policy_digest="sha256:" + "e" * 64,
            model_state_mode="trained",
            maximum_new_tokens=768,
            generation_batch_size=3,
            use_cache=True,
        )
    failed = tuple(
        json.loads(line)
        for line in (directory / "captions.partial.jsonl").read_text().splitlines()
    )
    assert len(failed) == len(samples)
    assert all(record["status"] == "failed" for record in failed)
    assert all(record["error_code"] == "generation_failed" for record in failed)
    assert not (directory / "captions.jsonl").exists()
    assert not (directory / "receipt.json").exists()


def test_test_caption_evidence_rejects_stale_resume_identity(
    tmp_path, monkeypatch
) -> None:
    import rwkv_lab.trainvm_adapters.hf_multimodal_sft as engine

    sample = ProcessedSample(
        "a", 0, {"caption": "target"}, image=object(), image_size=(512, 512)
    )
    monkeypatch.setattr(
        engine,
        "generate_hf_captions",
        lambda **_values: (("target", "generated", "sha256:" + "d" * 64),),
    )
    arguments = {
        "directory": tmp_path / "evidence",
        "stack": object(),
        "codec": object(),
        "samples": (sample,),
        "device": torch.device("cpu"),
        "step": 0,
        "model_load_receipt": "sha256:" + "a" * 64,
        "checkpoint_artifact_id": "checkpoint-baseline",
        "checkpoint_manifest_digest": "sha256:" + "0" * 64,
        "model_state_digest": "sha256:" + "1" * 64,
        "split_membership_digest": "sha256:" + "b" * 64,
        "decode_policy_digest": "sha256:" + "c" * 64,
        "model_state_mode": "base_adapters_disabled",
        "maximum_new_tokens": 32,
        "generation_batch_size": 1,
        "use_cache": True,
    }
    engine._write_test_caption_evidence(**arguments)
    arguments["decode_policy_digest"] = "sha256:" + "e" * 64
    with pytest.raises(engine.HFMultimodalSFTError, match="stale or mismatched"):
        engine._write_test_caption_evidence(**arguments)
    arguments["decode_policy_digest"] = "sha256:" + "c" * 64
    arguments["model_state_digest"] = "sha256:" + "f" * 64
    with pytest.raises(engine.HFMultimodalSFTError, match="stale or mismatched"):
        engine._write_test_caption_evidence(**arguments)


def test_test_caption_evidence_resumes_after_base_exception_without_replay(
    tmp_path, monkeypatch
) -> None:
    import rwkv_lab.trainvm_adapters.hf_multimodal_sft as engine

    samples = tuple(
        ProcessedSample(
            sample_id,
            ordinal,
            {"caption": f"target-{sample_id}"},
            image=object(),
            image_size=(512, 512),
        )
        for ordinal, sample_id in enumerate(("a", "b", "c", "d"))
    )
    first_calls: list[tuple[str, ...]] = []

    def interrupted(*, samples, **_values):
        identities = tuple(sample.sample_id for sample in samples)
        first_calls.append(identities)
        if identities == ("c", "d"):
            raise KeyboardInterrupt
        return tuple(
            (
                f"target-{sample.sample_id}",
                f"generated-{sample.sample_id}",
                "sha256:" + f"{sample.ordinal:064x}",
            )
            for sample in samples
        )

    monkeypatch.setattr(engine, "generate_hf_captions", interrupted)
    arguments = {
        "directory": tmp_path / "evidence",
        "stack": object(),
        "codec": object(),
        "samples": samples,
        "device": torch.device("cpu"),
        "step": 0,
        "model_load_receipt": "sha256:" + "a" * 64,
        "checkpoint_artifact_id": "checkpoint-baseline",
        "checkpoint_manifest_digest": "sha256:" + "0" * 64,
        "model_state_digest": "sha256:" + "1" * 64,
        "split_membership_digest": "sha256:" + "b" * 64,
        "decode_policy_digest": "sha256:" + "c" * 64,
        "model_state_mode": "base_adapters_disabled",
        "maximum_new_tokens": 32,
        "generation_batch_size": 2,
        "use_cache": True,
    }
    with pytest.raises(KeyboardInterrupt):
        engine._write_test_caption_evidence(**arguments)
    assert first_calls == [("a", "b"), ("c", "d")]

    resumed_calls: list[tuple[str, ...]] = []

    def resumed(*, samples, **_values):
        resumed_calls.append(tuple(sample.sample_id for sample in samples))
        return tuple(
            (
                f"target-{sample.sample_id}",
                f"generated-{sample.sample_id}",
                "sha256:" + f"{sample.ordinal:064x}",
            )
            for sample in samples
        )

    monkeypatch.setattr(engine, "generate_hf_captions", resumed)
    directory = engine._write_test_caption_evidence(**arguments)
    assert resumed_calls == [("c", "d")]
    records = tuple(
        json.loads(line)
        for line in (directory / "captions.jsonl").read_text().splitlines()
    )
    assert tuple(record["sample_id"] for record in records) == (
        "a",
        "b",
        "c",
        "d",
    )
    assert not (directory / "captions.partial.jsonl").exists()
