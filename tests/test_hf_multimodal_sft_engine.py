from __future__ import annotations

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
from rwkv_lab.trainvm_adapters.hf_multimodal_sft import (
    HFEngineState,
    HFForwardBatchCodec,
    HFMultimodalSFTError,
    component_causal_loss,
    initialize_training_stack,
    restore_exact_checkpoint,
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

    class Components:
        def model_loader(self, *, slot):
            assert slot == "model_loader"
            return Loader()

        def trainability(self):
            return Trainability()

        def precision(self):
            return Precision()

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
    assert events == ["load", "trainability", "precision", "optimizer"]


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
        def save_pretrained(self, directory, *, safe_serialization: bool):
            super().save_pretrained(directory, safe_serialization=safe_serialization)
            raise RuntimeError("interrupted")

    destination = tmp_path / "checkpoint-step0-attempt1"
    model = InterruptedModel(2, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
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


def test_checkpoint_staging_rejects_preplanted_sibling_object(tmp_path):
    class PreplantingModel(SaveableModel):
        def save_pretrained(self, directory, *, safe_serialization: bool):
            super().save_pretrained(directory, safe_serialization=safe_serialization)
            (directory.parent / "rng.json").write_text("planted", encoding="utf-8")

    destination = tmp_path / "checkpoint-step0-attempt1"
    model = PreplantingModel(2, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    with pytest.raises(FileExistsError):
        stage_exact_checkpoint(
            destination,
            model=model,
            optimizer=optimizer,
            learning_rate_schedule=StatelessSchedule(),
            weight_decay_schedule=StatelessSchedule(),
            precision=StatelessSchedule(),
            state=_engine_state(),
        )
    assert (destination / "rng.json").read_text(encoding="utf-8") == "planted"
    assert not (destination / "staging-complete.json").exists()


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
