import torch

from rwkv_lab.vision_teacher_compressor import (
    CanonicalTeacherCompressor, CompressorConfig, EpochBatchSampler,
    compressor_loss, split_cached_features, teacher_keep_mask)


def tiny_config():
    return CompressorConfig(tokens=8, latent_width=64, stem_width=8,
                            layers=1, heads=4, ff_mult=2)


def test_compressor_shapes_and_finite_loss(monkeypatch):
    import rwkv_lab.vision_teacher_compressor as module
    monkeypatch.setattr(module, "STREAM_WIDTHS", (16, 16, 16, 12, 8, 4))
    model = CanonicalTeacherCompressor(tiny_config())
    streams = [torch.randn(2, 8, width) for width in module.STREAM_WIDTHS]
    keep = teacher_keep_mask(2, len(streams), 0.9, torch.device("cpu"))
    latent, predictions = model(streams, keep)
    loss, metrics = compressor_loss(
        latent, predictions, streams, relational_weight=0.2,
        variance_weight=0.05, covariance_weight=0.01,
        diversity_weight=0.05)
    assert latent.shape == (2, 8, 64)
    assert [item.shape for item in predictions] == [item.shape for item in streams]
    assert torch.isfinite(loss)
    assert keep.any(dim=1).all()
    assert "latent_std" in metrics


def test_split_real_cache_contract():
    streams = split_cached_features(
        torch.ones(2, 3, 128, 4, 1152),
        torch.ones(2, 128, 2176))
    assert [tuple(item.shape) for item in streams] == [
        (2, 128, 4608), (2, 128, 4608), (2, 128, 4608),
        (2, 128, 1152), (2, 128, 768), (2, 128, 256)]


def test_missing_sam_fails_closed():
    fusion = torch.ones(1, 128, 2176)
    fusion[..., 1920:] = 0
    try:
        split_cached_features(torch.ones(1, 3, 128, 4, 1152), fusion)
    except ValueError as error:
        assert "SAM" in str(error)
    else:
        raise AssertionError("zero SAM stream was accepted")


def test_epoch_sampler_resumes_exactly():
    sampler = EpochBatchSampler(11, 4, 123)
    batches = iter(sampler)
    first = next(batches)
    sampler.consumed(len(first))
    state = sampler.state_dict()
    restored = EpochBatchSampler(11, 4, 123)
    restored.load_state_dict(state)
    assert list(restored) == list(batches)


def test_epoch_sampler_rolls_after_short_batch():
    sampler = EpochBatchSampler(5, 4, 3)
    batches = list(sampler)
    for batch in batches:
        sampler.consumed(len(batch))
    assert sampler.state_dict()["epoch"] == 1
    assert sampler.state_dict()["cursor"] == 0


def test_init_from_is_parsed_for_manifest_transfer(tmp_path):
    from rwkv_lab.vision_teacher_compressor import parse_args
    checkpoint = tmp_path / "best.pt"
    args = parse_args(["--data", "train.jsonl", "--eval-data", "eval.jsonl",
                       "--moon-cache", "moon", "--fusion-cache", "fusion",
                       "--out", "run", "--init-from", str(checkpoint)])
    assert args.init_from == checkpoint
