import pytest

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
