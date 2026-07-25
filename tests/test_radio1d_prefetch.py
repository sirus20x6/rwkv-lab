from types import SimpleNamespace

import torch
from PIL import Image

import rwkv_lab.vision_train as vision_train
from rwkv_lab.radio1d_rwkv import RadioFeatureProjector


def _fake_radio(tmp_path, count=2):
    rows = []
    for index in range(count):
        source = tmp_path / f"image-{index}.png"
        Image.new("RGB", (64, 64), (index, 0, 0)).save(source)
        rows.append({"image": str(source), "tokens": [1, 2], "prompt_len": 1,
                     "_radio_tiles": 1})
    tower = SimpleNamespace(
        radio_revision="revision",
        radio_max_detail_tiles=48,
        radio_tile_batch=8,
        radio_adaptive_token_threshold=12,
    )
    return rows, tower


def test_radio_prefetch_never_encodes_misses_off_main_thread(tmp_path, monkeypatch):
    rows, tower = _fake_radio(tmp_path)
    calls = []

    def encode(_tower, tiles, *, batch_size, num_tokens):
        calls.append((len(tiles), batch_size, num_tokens))
        values = torch.arange(len(tiles), dtype=torch.bfloat16)
        return values[:, None, None].expand(-1, num_tokens, 2560).clone()

    monkeypatch.setattr(vision_train, "encode_radio_tiles", encode)
    vision_train._FEATURE_MEMORY_CACHE.clear()
    try:
        result = vision_train.prefetch_training_batch(
            rows, tower, RadioFeatureProjector(), tmp_path / "cache", None)
        assert calls == []
        assert result.ready == 0
        assert result.generated == 0
        assert result.disk_hits == result.resident_hits == 0

        cached = vision_train.cached_radio_features(
            rows, tower, tmp_path / "cache", revision="revision",
            max_detail_tiles=48, tile_batch=8, adaptive_token_threshold=12)
        assert cached[0][0][0, 0, 0].item() == 0
        assert cached[1][0][0, 0, 0].item() == 1
        assert len(calls) == 1

        warm = vision_train.prefetch_training_batch(
            rows, tower, RadioFeatureProjector(), tmp_path / "cache", None)
        assert warm.ready == warm.resident_hits == 2
        assert warm.generated == warm.disk_hits == 0
    finally:
        vision_train._FEATURE_MEMORY_CACHE.clear()


def test_radio_prefetch_rejects_mixed_tile_counts_before_encoding(
        tmp_path, monkeypatch):
    rows, tower = _fake_radio(tmp_path)
    Image.new("RGB", (512, 2048), "blue").save(rows[1]["image"])
    rows[1]["_radio_tiles"] = 7
    called = False

    def encode(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("mixed buckets must fail before RADIO")

    monkeypatch.setattr(vision_train, "encode_radio_tiles", encode)
    vision_train._FEATURE_MEMORY_CACHE.clear()
    try:
        import pytest
        with pytest.raises(ValueError, match="share one tile count"):
            vision_train.prefetch_training_batch(
                rows, tower, RadioFeatureProjector(), tmp_path / "cache", None)
        assert not called
    finally:
        vision_train._FEATURE_MEMORY_CACHE.clear()
