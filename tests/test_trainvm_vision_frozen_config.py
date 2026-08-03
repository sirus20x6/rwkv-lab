import pytest

from rwkv_lab.trainvm_adapters.vision_frozen import VisionFrozenAdapterConfig


def _base() -> dict:
    return {
        "arm": "moonvit",
        "train_manifests": ["/sealed/train.jsonl"],
        "eval_manifests": ["/sealed/eval.jsonl"],
        "rwkv_checkpoint": "/sealed/rwkv.pt",
        "moonvit_checkpoint": "/sealed/moonvit.safetensors",
        "vocab": "/sealed/vocab.txt",
        "moon_cache": "/sealed/moon-cache",
    }


def test_vision_frozen_config_canonicalizes_manifest_and_layer_lists() -> None:
    config = VisionFrozenAdapterConfig(**_base())

    assert config.train_manifests == ("/sealed/train.jsonl",)
    assert config.moonvit_tap_layers == (8, 17, 26)
    assert config.deep_vision_layers == (8, 16, 24)


def test_vision_frozen_config_requires_all_compressor_inputs() -> None:
    values = _base() | {"arm": "compressor"}

    with pytest.raises(ValueError, match="compressor_checkpoint"):
        VisionFrozenAdapterConfig(**values)


def test_vision_frozen_config_rejects_compressor_inputs_on_moonvit_arm() -> None:
    values = _base() | {"compressor_checkpoint": "/sealed/compressor.pt"}

    with pytest.raises(ValueError, match="compressor-only"):
        VisionFrozenAdapterConfig(**values)


def test_vision_frozen_config_rejects_incoherent_batch_bounds() -> None:
    values = _base() | {"batch_size": 2, "min_batch_size": 3}

    with pytest.raises(ValueError, match="min_batch_size"):
        VisionFrozenAdapterConfig(**values)
