from __future__ import annotations

import json
import random
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from rwkv_lab.training_components import (
    AssistantOnlyMapperConfiguration,
    CausalTokensMapperConfiguration,
    ImageCaptionProcessorConfiguration,
    LinearHeadCrossEntropyConfiguration,
    LinearHeadCrossEntropyObjective,
    PaddedCollatorConfiguration,
    ProcessedSample,
)
from rwkv_lab.training_runtime.activation_memory import (
    HFGradientCheckpointing,
    HFGradientCheckpointingConfiguration,
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
    component_causal_loss,
    initialize_training_stack,
    normalize_token_mean_gradients,
    restore_exact_checkpoint,
    run_hf_multimodal_sft,
    stage_exact_checkpoint,
)


class FakeTokenizer:
    padding_side = "right"
    eos_token_id = 2

    def __call__(self, text: str, *, add_special_tokens: bool = False):
        del add_special_tokens
        return {"input_ids": [ord(character) for character in text]}


class FakeProcessor:
    def __init__(self) -> None:
        self.tokenizer = FakeTokenizer()

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
            ids[index, : len(row)] = torch.tensor(row)
            attention[index, : len(row)] = 1
        return {
            "input_ids": ids,
            "attention_mask": attention,
            "pixel_values": torch.tensor(images, dtype=torch.float32).reshape(-1, 1),
            "image_grid_thw": torch.ones((len(rows), 3), dtype=torch.long),
        }


def _codec() -> HFForwardBatchCodec:
    return HFForwardBatchCodec(
        FakeProcessor(),
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


def test_multimodal_codec_keeps_image_target_identity_and_masks_prompt() -> None:
    batch = _codec().encode(
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
    manifest = tmp_path / "targets.json"
    manifest.write_text(
        json.dumps(
            {
                "architecture": ["Fixture"],
                "model_config_sha256": hashlib.sha256(config).hexdigest(),
                "model_type": "fixture",
                    "policy": {
                        "fused_moe_experts": "frozen",
                        "mtp": "not_loaded",
                        "vision": "frozen",
                        "multimodal_projector": "frozen",
                        "router": "frozen",
                    },
                    "schema": "rwkv-lab.qwen-caption-lora-targets.v1",
                "target_count": 2,
                "target_digest": "a" * 64,
                "targets": ["lm_head", "model.layers.0.q_proj"],
                "weight_index_sha256": hashlib.sha256(index).hexdigest(),
            }
        ),
        encoding="utf-8",
    )
    receipt = preflight_lora_target_manifest(tmp_path, manifest)
    assert receipt.targets == ("lm_head", "model.layers.0.q_proj")
    assert receipt.producer_target_digest == "sha256:" + "a" * 64
    assert load_lora_target_receipt(manifest) == receipt

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


def test_generic_causal_loop_publishes_step_zero_before_optimizer_mutation(tmp_path):
    from collections import namedtuple

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
    )

    class Source:
        configuration = object()

        def records(self):
            return iter(records)

        def component_state(self):
            return {"content_fingerprint": "sha256:" + "d" * 64, "cursor": 0}

        def restore_component_state(self, state):
            assert state == self.component_state()

    class Processor:
        configuration = object()

        def process(self, sample, *, image_root):
            assert image_root is None
            return ProcessedSample(
                sample.sample_id,
                sample.ordinal,
                sample.values,
                token_length=len(sample.values["tokens"]),
            )

    class Split:
        def __init__(self, held_out):
            self.held_out = held_out

        def select(self, identities):
            selected = ("held",) if self.held_out else ("a", "b")
            return SplitSelection(selected, ("held",), "sha256:" + "e" * 64)

    class Pipeline:
        def __init__(self, held_out):
            self.source = Source()
            self.processor = Processor()
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
            self.split_selector = Split(held_out)

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
            }

        def validate_resume_state(self, state):
            return state

    Decision = namedtuple(
        "Decision", "qualitative full_scalar launch_gate defer_full_scalar"
    )

    class Components:
        composition = Composition()

        def __init__(self):
            self.training = Pipeline(False)
            self.evaluation = Pipeline(True)

        def data_pipeline(self, *, split_slot):
            return self.evaluation if split_slot == "evaluation_split" else self.training

        def model_loader(self, *, slot):
            assert slot == "model_loader"
            return SimpleNamespace(load=lambda **_kwargs: Loaded(model))

        def trainability(self):
            return SimpleNamespace(apply=lambda item: TrainabilityResult(item))

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
                configuration=SimpleNamespace(sample_count=1),
                bind=lambda identities, selector_digest: SimpleNamespace(
                    identities=tuple(identities),
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
                configuration=SimpleNamespace(maximum_examples=0),
                reduce=lambda values, weights: sum(
                    v * w for v, w in zip(values, weights, strict=True)
                )
                / sum(weights),
            )

        def evaluation_schedule(self):
            return SimpleNamespace(
                for_step=lambda step, final=False: Decision(
                    qualitative=step == 0 or final,
                    full_scalar=step == 0 or final,
                    launch_gate=step == 0,
                    defer_full_scalar=False,
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
                        use_cache=True,
                ),
                digest="sha256:" + "5" * 64,
            )

        def learning_rate_configuration(self):
            return None, SimpleNamespace(max_steps=1)

        def gradient_clipping(self, parameters):
            return torch.nn.utils.clip_grad_norm_(tuple(parameters), 1.0)

    class Controls:
        checkpoint_boundary_requested = False
        checkpoint_completion_requested = False

        def __init__(self):
            self.events = []
            self.safe_points = []

        def checkpoint_state(self):
            return {"effective_control_revision": 0, "effective_controls": {}}

        def poll_initialization(self):
            self.safe_points.append(("initialization", 0))

        def microbatch(self, step, applier):
            self.safe_points.append(("microbatch", step))
            applier({}, {})

        def optimizer_step(self, step, applier):
            self.safe_points.append(("optimizer_step", step))
            applier({}, {})

        def evaluation(self, step, applier):
            self.safe_points.append(("evaluation", step))
            applier({}, {})

        def checkpoint(self, step, applier):
            self.safe_points.append(("checkpoint", step))
            applier({}, {})

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
            return SimpleNamespace(artifact_id="gallery")

        def publish_requested_checkpoint(self, request):
            raise AssertionError(request)

    class Observability:
        def __init__(self):
            self.metrics = []

        def publish_if_declared(self, name, value, *, step):
            self.metrics.append((name, value, step))

        def optimizer_step(self, step, phase):
            return None

    controls = Controls()
    observability = Observability()
    step = run_hf_multimodal_sft(
        invocation=SimpleNamespace(attempt_id="attempt-1"),
        components=Components(),
        run_directory=tmp_path / "run",
        controls=controls,
        observability=observability,
        step_profiler=SimpleNamespace(
            input_wait=nullcontext, step=lambda _step: None
        ),
        resume_directory=None,
        device="cpu",
    )
    assert step == 1
    assert [event[0] for event in controls.events] == [
        "checkpoint",
        "gallery",
        "checkpoint",
        "gallery",
    ]
    assert torch.equal(controls.events[0][2], initial)
    step_zero_item = controls.events[1][1].items[0]
    assert (
        step_zero_item.sampling_attributes["baseline"]
        == step_zero_item.sampling_attributes["current"]
    )
    assert not torch.equal(model.head.weight, initial)
    assert timeline.index("generate") < timeline.index("optimizer")
    assert len(generation_weights) == 2
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
    assert observability.metrics[0][0] == "eval.loss"
    assert observability.metrics[0][2] == 0
    assert controls.safe_points == [
        ("initialization", 0),
        ("evaluation", 0),
        ("microbatch", 1),
        ("optimizer_step", 1),
        ("evaluation", 1),
        ("checkpoint", 1),
    ]


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
        split_membership_digest="sha256:" + "b" * 64,
        decode_policy_digest="sha256:" + "c" * 64,
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
    final = engine._write_test_caption_evidence(
        directory=tmp_path / "final",
        stack=object(),
        codec=object(),
        samples=samples,
        device=torch.device("cpu"),
        step=745,
        model_load_receipt="sha256:" + "a" * 64,
        split_membership_digest="sha256:" + "b" * 64,
        decode_policy_digest="sha256:" + "c" * 64,
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
    bundle = engine._bundle_test_caption_evidence(
        directory=tmp_path / "bundle",
        baseline=directory,
        final=final,
        split_membership_digest="sha256:" + "b" * 64,
    )
    bundle_receipt = json.loads((bundle / "receipt.json").read_text())
    assert bundle_receipt["split_membership_digest"] == "sha256:" + "b" * 64
    assert (bundle / "baseline" / "captions.jsonl").is_file()
    assert (bundle / "final" / "captions.jsonl").is_file()
