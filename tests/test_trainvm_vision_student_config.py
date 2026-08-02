from __future__ import annotations

import pytest

from rwkv_lab.trainvm_adapters.vision_student import VisionRWKVStudentConfig


def minimum_config() -> dict[str, object]:
    return {
        "baseline_checkpoint": "/sealed/baseline.pt",
        "compressor_checkpoint": "/sealed/compressor.pt",
        "native_head_checkpoint": "/sealed/native.pt",
        "train_manifest": "/sealed/train.jsonl",
        "eval_manifest": "/sealed/eval.jsonl",
        "moon_cache": "/sealed/moon-cache",
        "fusion_cache": "/sealed/fusion-cache",
        "moonvit_checkpoint": "/sealed/moonvit.safetensors",
        "siglip2_model": "/sealed/siglip2",
        "dinov2_model": "/sealed/dinov2",
        "sam_model": "/sealed/sam",
        "vocab": "/sealed/vocab.txt",
        "output_dir": "/run/student",
    }


def test_vision_student_config_closes_architecture_and_loss_surface() -> None:
    config = VisionRWKVStudentConfig(**minimum_config())
    assert config.image_size == 512
    assert config.grid_size == 16
    assert config.caption_weight == pytest.approx(0.1)
    assert config.checkpoint_blocks is True


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("steps", 0),
        ("image_size", 510),
        ("hidden_size", 2050),
        ("caption_weight", float("nan")),
        ("checkpoint_blocks", 1),
        ("device", "cpu"),
    ],
)
def test_vision_student_config_rejects_invalid_values(
    field: str, value: object
) -> None:
    values = minimum_config()
    values[field] = value
    with pytest.raises((TypeError, ValueError)):
        VisionRWKVStudentConfig(**values)


def test_vision_student_config_rejects_unknown_fields() -> None:
    with pytest.raises(TypeError):
        VisionRWKVStudentConfig(**minimum_config(), teacher_command="python")
