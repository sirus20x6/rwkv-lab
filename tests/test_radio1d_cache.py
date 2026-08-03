import threading
import os
import shutil
from concurrent.futures import ThreadPoolExecutor

import torch
from PIL import Image

from rwkv_lab.radio1d_cache import (
    CACHE_SCHEMA, LEGACY_CACHE_SCHEMA, cache_is_current, cache_path, load_cache,
    make_metadata, save_cache,
)
from rwkv_lab.radio1d_rwkv import build_radio_tiles


def test_atomic_cache_roundtrip_and_revision_validation(tmp_path):
    source = tmp_path / "image.png"
    Image.new("RGB", (64, 64), "purple").save(source)
    tiles = build_radio_tiles(Image.open(source))
    metadata = make_metadata(source, "pinned-revision", tiles)
    features = torch.randn(len(tiles), 256, 2560, dtype=torch.bfloat16)
    target = cache_path(tmp_path / "cache", source)
    save_cache(target, metadata, features)
    restored_metadata, restored = load_cache(target)
    assert restored_metadata == metadata
    assert torch.equal(restored, features)
    assert cache_is_current(target, source, "pinned-revision")
    assert not cache_is_current(target, source, "different-revision")
    assert not list(target.parent.glob(".*.tmp"))


def test_exact_sha_allows_reusing_cache_across_materialized_paths(tmp_path):
    source = tmp_path / "original.png"
    clone = tmp_path / "rematerialized.png"
    Image.new("RGB", (64, 64), "purple").save(source)
    shutil.copyfile(source, clone)
    os.utime(clone, ns=(clone.stat().st_atime_ns,
                        source.stat().st_mtime_ns + 1_000_000))
    tiles = build_radio_tiles(Image.open(source))
    metadata = make_metadata(source, "revision", tiles)
    target = cache_path(tmp_path / "cache", source)
    save_cache(target, metadata, torch.zeros(
        len(tiles), 256, 2560, dtype=torch.bfloat16))

    assert not cache_is_current(target, clone, "revision")
    assert cache_is_current(
        target, clone, "revision", source_sha256=metadata.source_sha256)
    assert not cache_is_current(
        target, clone, "revision", source_sha256="0" * 64)


def test_same_process_concurrent_writers_use_private_temporaries(tmp_path):
    source = tmp_path / "image.png"
    Image.new("RGB", (64, 64), "purple").save(source)
    tiles = build_radio_tiles(Image.open(source))
    metadata = make_metadata(source, "pinned-revision", tiles)
    features = torch.randn(len(tiles), 256, 2560, dtype=torch.bfloat16)
    target = cache_path(tmp_path / "cache", source)
    barrier = threading.Barrier(2)

    def write() -> None:
        barrier.wait()
        save_cache(target, metadata, features)

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(write) for _ in range(2)]
        for future in futures:
            future.result()

    restored_metadata, restored = load_cache(target)
    assert restored_metadata == metadata
    assert torch.equal(restored, features)
    assert not list(target.parent.glob(".*.tmp"))


def test_cache_rejects_shortened_radio_token_sequence(tmp_path):
    import pytest
    source = tmp_path / "image.png"
    Image.new("RGB", (64, 64), "purple").save(source)
    tiles = build_radio_tiles(Image.open(source))
    metadata = make_metadata(source, "revision", tiles)
    with pytest.raises(ValueError, match="cached features"):
        save_cache(tmp_path / "bad.safetensors", metadata,
                   torch.randn(1, 128, 2560, dtype=torch.bfloat16))


def test_long_grid_keeps_256_by_default_and_rejects_mismatched_cache(tmp_path):
    import json
    import pytest
    from safetensors.torch import save_file

    source = tmp_path / "tall.png"
    Image.new("RGB", (512, 6656), "purple").save(source)
    tiles = build_radio_tiles(Image.open(source))
    assert len(tiles) > 12
    # Large images now keep the full native budget; halving them made total
    # tokens fall as images grew.
    assert make_metadata(source, "revision", tiles).tokens_per_tile == 256

    # The compact tier still applies when a caller asks for it, and a cache
    # written under a different policy must still be rejected.
    metadata = make_metadata(source, "revision", tiles,
                             adaptive_token_threshold=12)
    assert metadata.schema == CACHE_SCHEMA
    assert metadata.tokens_per_tile == 128

    payload = {
        **json.loads(metadata.to_json()),
        "schema": LEGACY_CACHE_SCHEMA,
        "tokens_per_tile": 256,
    }
    target = cache_path(tmp_path / "cache", source)
    target.parent.mkdir(parents=True)
    save_file(
        {"global_tokens": torch.zeros(len(tiles), 256, 2560,
                                      dtype=torch.bfloat16)},
        str(target), metadata={
            "schema": LEGACY_CACHE_SCHEMA,
            "radio_metadata": json.dumps(payload),
        })
    assert not cache_is_current(target, source, "revision")
    with pytest.raises(ValueError, match="adaptive RADIO token policy"):
        load_cache(target)


def test_threshold_change_only_invalidates_entries_whose_width_changes(tmp_path):
    """Moving the knob must not discard entries that are byte-for-byte correct.

    cache_is_current used to also require an exact adaptive_token_threshold
    match. That is redundant with the tokens_per_tile check below it and far
    stricter: raising the threshold from 12 to 49 left 96% of a real 533 GB
    corpus valid in substance while invalidating all 142k entries.
    """
    small = tmp_path / "small.png"
    Image.new("RGB", (256, 256), "purple").save(small)
    tiles = build_radio_tiles(Image.open(small))
    assert len(tiles) <= 12
    metadata = make_metadata(small, "rev", tiles, adaptive_token_threshold=12)
    target = cache_path(tmp_path / "cache", small)
    save_cache(target, metadata,
               torch.zeros(len(tiles), metadata.tokens_per_tile, 2560,
                           dtype=torch.bfloat16))
    # Same resulting width under either threshold -> the entry stays usable.
    assert cache_is_current(target, small, "rev", adaptive_token_threshold=12)
    assert cache_is_current(target, small, "rev", adaptive_token_threshold=49)

    tall = tmp_path / "tall.png"
    Image.new("RGB", (512, 6656), "purple").save(tall)
    long_tiles = build_radio_tiles(Image.open(tall))
    assert len(long_tiles) > 12
    long_meta = make_metadata(tall, "rev", long_tiles, adaptive_token_threshold=12)
    assert long_meta.tokens_per_tile == 128
    long_target = cache_path(tmp_path / "cache", tall)
    save_cache(long_target, long_meta,
               torch.zeros(len(long_tiles), 128, 2560, dtype=torch.bfloat16))
    # Width genuinely differs at the new threshold -> must be regenerated.
    assert cache_is_current(long_target, tall, "rev", adaptive_token_threshold=12)
    assert not cache_is_current(long_target, tall, "rev",
                                adaptive_token_threshold=49)
