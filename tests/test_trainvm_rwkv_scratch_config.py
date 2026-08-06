import pytest
import torch

from rwkv_lab.rwkv_pretrain import initialize_model_weights_from_checkpoint
from rwkv_lab.trainvm_adapters.rwkv_scratch import RWKVScratchTrainConfig


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
