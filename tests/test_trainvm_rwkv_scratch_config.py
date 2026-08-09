import os
from types import MappingProxyType

import pytest
import torch

from rwkv_lab.rwkv_pretrain import (
    build_rwkv_text_eval_examples,
    initialize_model_weights_from_checkpoint,
)
from rwkv_lab.trainvm_adapters.rwkv_scratch import (
    RWKVScratchTrainConfig,
    RWKVTextEvalPolicy,
)


class _UnsafeCheckpointPayload:
    def __init__(self, sentinel: str) -> None:
        self.sentinel = sentinel

    def __reduce__(self):
        return os.system, (f"touch {self.sentinel}",)


def test_rwkv_scratch_config_has_closed_baseline_training_surface() -> None:
    config = RWKVScratchTrainConfig(
        steps=200,
        learning_rate=3.0e-4,
        minimum_learning_rate=3.0e-5,
    )
    arguments = config.trainer_arguments(
        data="/data/tokens.bin",
        output_dir="/runs/one",
        checkpoint="/runs/one/checkpoint-final/state.pt",
        resume=None,
    )
    assert arguments[arguments.index("--optimizer") + 1] == "adamw"
    assert arguments[arguments.index("--lr-schedule") + 1] == "powercool"
    assert arguments[arguments.index("--distributed") + 1] == "none"
    assert arguments[arguments.index("--powercool-min-lr") + 1] == "3e-05"
    assert not any("muon" in value for value in arguments)


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"d_model": 510}, "divisible"),
        ({"cooldown_fraction": 0.0}, "positive"),
        ({"minimum_learning_rate": 1.0}, "minimum_learning_rate"),
        ({"warmup_steps": 201}, "warmup_steps"),
    ],
)
def test_rwkv_scratch_config_rejects_unrepresented_or_invalid_semantics(
    updates, message
) -> None:
    values = {
        "steps": 200,
    }
    values.update(updates)
    with pytest.raises(ValueError, match=message):
        RWKVScratchTrainConfig(**values)


def test_rwkv_scratch_config_constructor_rejects_unknown_research_switches() -> None:
    with pytest.raises(TypeError, match="unexpected keyword"):
        RWKVScratchTrainConfig(
            steps=200,
            arbitrary_module="user.code",
        )


def test_continuation_initializes_weights_but_not_optimizer_state(tmp_path) -> None:
    source = torch.nn.Linear(3, 2)
    with torch.no_grad():
        source.weight.fill_(4.0)
        source.bias.fill_(5.0)
    checkpoint = tmp_path / "scratch-state.pt"
    torch.save(
        {
            "model": source.state_dict(),
            "opt": {"source-only": True},
            "step": 73,
            "component_composition_digest": "sha256:" + "a" * 64,
        },
        checkpoint,
    )
    continued = torch.nn.Linear(3, 2)

    assert initialize_model_weights_from_checkpoint(continued, str(checkpoint)) == 73
    assert torch.equal(continued.weight, source.weight)
    assert torch.equal(continued.bias, source.bias)
    fresh_optimizer = torch.optim.AdamW(continued.parameters())
    assert fresh_optimizer.state == {}


def test_continuation_rejects_non_tensor_and_pickle_payloads(tmp_path) -> None:
    target = torch.nn.Linear(3, 2)
    non_tensor = tmp_path / "non-tensor.pt"
    torch.save({"model": {"weight": "not-a-tensor"}, "step": 0}, non_tensor)
    with pytest.raises(TypeError, match="tensor-only"):
        initialize_model_weights_from_checkpoint(target, str(non_tensor))

    sentinel = tmp_path / "pickle-executed"
    malicious = tmp_path / "malicious.pt"
    torch.save(
        {"model": {"weight": _UnsafeCheckpointPayload(str(sentinel))}, "step": 0},
        malicious,
    )
    with pytest.raises(Exception, match="Weights only load failed"):
        initialize_model_weights_from_checkpoint(target, str(malicious))
    assert not sentinel.exists()


def test_continuation_and_exact_resume_flags_are_mutually_exclusive() -> None:
    config = RWKVScratchTrainConfig(
        steps=200,
        initial_checkpoint="/sealed/scratch-state.pt",
    )
    with pytest.raises(ValueError, match="mutually exclusive"):
        config.trainer_arguments(
            data="/data/tokens.bin",
            output_dir="/runs/one",
            checkpoint="/runs/one/checkpoint-final/state.pt",
            resume="/controller/resume/state.pt",
        )


def test_continuation_lowers_to_model_only_initialization() -> None:
    config = RWKVScratchTrainConfig(
        steps=200,
        initial_checkpoint="/sealed/scratch-state.pt",
    )
    arguments = config.trainer_arguments(
        data="/data/tokens.bin",
        output_dir="/runs/continued",
        checkpoint="/runs/continued/checkpoint-final/state.pt",
        resume=None,
    )

    assert arguments[arguments.index("--init-checkpoint") + 1] == (
        "/sealed/scratch-state.pt"
    )
    assert "--resume" not in arguments


def _text_eval_policy(**updates) -> RWKVTextEvalPolicy:
    values = {
        "heldout_tokens": MappingProxyType({"held": (3, 4, 5)}),
        "identity_field": "id",
        "identities_digest": "sha256:" + "1" * 64,
        "selector_digest": "sha256:" + "2" * 64,
        "evaluator_component_digest": "sha256:" + "3" * 64,
        "metric_names": ("perplexity", "validation_loss"),
        "generation_policy_digest": "sha256:" + "4" * 64,
    }
    values.update(updates)
    return RWKVTextEvalPolicy(**values)


def test_text_eval_identity_and_targets_are_stable_across_steps() -> None:
    class PredictionModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.scores = torch.nn.Parameter(torch.zeros(8))

        def forward(self, input_ids):
            return self.scores.view(1, 1, -1).expand(
                input_ids.shape[0], input_ids.shape[1], -1
            )

    model = PredictionModel()
    with torch.no_grad():
        model.scores[1] = 1
    baseline = build_rwkv_text_eval_examples(
        model, _text_eval_policy(), device="cpu", optimizer_step=0
    )[0]
    with torch.no_grad():
        model.scores[2] = 2
    later = build_rwkv_text_eval_examples(
        model, _text_eval_policy(), device="cpu", optimizer_step=7
    )[0]

    assert baseline.example_id == later.example_id
    assert baseline.heldout_item_id == later.heldout_item_id
    assert baseline.heldout_item_digest == later.heldout_item_digest
    assert baseline.input == later.input
    assert baseline.target == later.target
    assert baseline.prediction != later.prediction


@pytest.mark.parametrize(
    "updates",
    [
        {"selector_digest": "not-a-digest"},
        {"metric_names": ("validation_loss", "perplexity")},
        {"metric_names": ("perplexity", "perplexity")},
    ],
)
def test_text_eval_policy_rejects_noncanonical_provenance(updates) -> None:
    with pytest.raises(ValueError):
        _text_eval_policy(**updates)
