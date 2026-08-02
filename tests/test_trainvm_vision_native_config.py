from __future__ import annotations

import pytest

from rwkv_lab.trainvm_adapters.vision_native import VisionNativeHeadConfig


def minimum_config() -> dict[str, object]:
    return {
        "baseline_checkpoint": "/sealed/baseline.pt",
        "compressor_checkpoint": "/sealed/compressor.pt",
        "train_manifest": "/sealed/train.jsonl",
        "eval_manifest": "/sealed/eval.jsonl",
        "moon_cache": "/sealed/moon-cache",
        "fusion_cache": "/sealed/fusion-cache",
        "moonvit_checkpoint": "/sealed/moonvit.safetensors",
        "siglip2_model": "/sealed/siglip2",
        "dinov2_model": "/sealed/dinov2",
        "sam_model": "/sealed/sam",
        "vocab": "/sealed/vocab.txt",
        "output_dir": "/run/native-head",
    }


def test_vision_native_config_accepts_the_exact_closed_surface() -> None:
    config = VisionNativeHeadConfig(**minimum_config())
    assert config.steps == 5000
    assert config.device == "cuda"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("steps", 0),
        ("batch_size", True),
        ("learning_rate", float("nan")),
        ("eval_every", 5001),
        ("device", "cuda:1"),
    ],
)
def test_vision_native_config_rejects_invalid_values(
    field: str, value: object
) -> None:
    values = minimum_config()
    values[field] = value
    with pytest.raises(ValueError):
        VisionNativeHeadConfig(**values)


def test_vision_native_config_rejects_unknown_fields() -> None:
    with pytest.raises(TypeError):
        VisionNativeHeadConfig(**minimum_config(), arbitrary_command="python tool.py")
