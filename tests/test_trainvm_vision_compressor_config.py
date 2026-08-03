from __future__ import annotations

from dataclasses import fields

import pytest

from rwkv_lab.trainvm_adapters.vision_compressor import (
    VisionTeacherCompressorConfig,
)


def config(**updates: object) -> VisionTeacherCompressorConfig:
    values = {
        "train_manifest": "/data/train.jsonl",
        "eval_manifest": "/data/eval.jsonl",
        "moon_cache": "/cache/moon",
        "fusion_cache": "/cache/fusion",
        "output_dir": "/runs/compressor",
        "moonvit_checkpoint": "/models/moon.safetensors",
        "siglip2_model": "/models/siglip2",
        "dinov2_model": "/models/dinov2",
        "sam_model": "/models/sam",
    }
    values.update(updates)
    return VisionTeacherCompressorConfig(**values)


def test_vision_compressor_config_is_closed_and_immutable() -> None:
    value = config()
    assert value.device == "cuda"
    assert value.steps == 6000
    assert {field.name for field in fields(value)} == {
        "train_manifest",
        "eval_manifest",
        "moon_cache",
        "fusion_cache",
        "output_dir",
        "moonvit_checkpoint",
        "siglip2_model",
        "dinov2_model",
        "sam_model",
        "steps",
        "batch_size",
        "workers",
        "learning_rate",
        "weight_decay",
        "teacher_dropout",
        "relational_weight",
        "variance_weight",
        "covariance_weight",
        "diversity_weight",
        "max_gradient_norm",
        "eval_every",
        "checkpoint_every",
        "log_every",
        "seed",
        "init_from",
        "device",
    }
    with pytest.raises(TypeError):
        VisionTeacherCompressorConfig(**{**values(value), "unknown": True})


def values(value: VisionTeacherCompressorConfig) -> dict[str, object]:
    return {field.name: getattr(value, field.name) for field in fields(value)}


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"workers": -1}, "workers"),
        ({"teacher_dropout": 1.0}, "less than one"),
        ({"log_every": 6001}, "log_every"),
        ({"learning_rate": float("nan")}, "learning_rate"),
        ({"device": "cpu"}, "device"),
        ({"init_from": "bad\x00path"}, "init_from"),
    ],
)
def test_vision_compressor_config_rejects_unbounded_or_ambiguous_values(
    updates: dict[str, object], message: str
) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        config(**updates)
