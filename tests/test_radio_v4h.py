import json

import pytest
import torch
from PIL import Image

from rwkv_lab.radio1d_rwkv import build_radio_tiles, native_aspect_size
from rwkv_lab.radio_v4h import (
    V4H_HIDDEN_SIZE, V4H_LATTICE, V4H_PATCH, V4HCacheMetadata, cache_path,
    load_v4h_cache, make_v4h_metadata, pool_to_lattice, save_v4h_cache,
    v4h_cache_is_current)


def test_native_aspect_size_is_supported_and_preserves_aspect():
    """RADIO accepts any per-axis multiple of 16, so padding is unnecessary."""
    for width, height in ((1920, 1080), (480, 640), (2480, 3508), (512, 512)):
        h, w = native_aspect_size(width, height)
        assert h % V4H_PATCH == 0 and w % V4H_PATCH == 0
        assert max(h, w) <= 512 and min(h, w) >= V4H_PATCH
        assert abs((w / h) - (width / height)) / (width / height) < 0.06


def test_non_square_tiling_removes_padding_and_stays_batchable():
    """Detail cells must share one size or they cannot be stacked."""
    image = Image.new("RGB", (1920, 1080), (90, 90, 90))
    letterboxed = build_radio_tiles(image, letterbox=True)
    native = build_radio_tiles(image, letterbox=False)

    assert [t.source_box for t in native] == [t.source_box for t in letterboxed]
    assert all(t.image.width % V4H_PATCH == 0 and t.image.height % V4H_PATCH == 0
               for t in native)
    # thumbnail keeps its own aspect; every detail cell shares one size
    details = {t.image.size for t in native if t.role != "thumbnail"}
    assert len(details) == 1
    assert len({t.image.size for t in native}) <= 2

    def tokens(tiles):
        return sum((t.image.height // V4H_PATCH) * (t.image.width // V4H_PATCH)
                   for t in tiles)
    assert tokens(native) < tokens(letterboxed)     # padding really is gone


def test_pool_to_lattice_is_uniform_across_input_aspects():
    """Any grid shape must yield the same token count and layout."""
    pooled = [pool_to_lattice(torch.randn(2, gh, gw, 8))
              for gh, gw in ((32, 32), (23, 32), (32, 23), (12, 5))]
    for value in pooled:
        assert value.shape == (2, V4H_LATTICE ** 2, 8)
    # a horizontal split survives pooling at a non-square input aspect
    grid = torch.zeros(1, 24, 32, 4)
    grid[:, :12] = 1.0
    lattice = pool_to_lattice(grid).reshape(1, V4H_LATTICE, V4H_LATTICE, 4)
    assert torch.allclose(lattice[0, :8], torch.ones_like(lattice[0, :8]))
    assert torch.allclose(lattice[0, 8:], torch.zeros_like(lattice[0, 8:]))


def test_v4h_cache_roundtrip_and_revision_validation(tmp_path):
    source = tmp_path / "image.png"
    Image.new("RGB", (640, 480), "purple").save(source)
    tiles = build_radio_tiles(Image.open(source), letterbox=False)
    metadata = make_v4h_metadata(source, "rev-a", tiles)
    thumb = tiles[0].image
    thumbnail = torch.zeros(1, thumb.height // V4H_PATCH,
                            thumb.width // V4H_PATCH, V4H_HIDDEN_SIZE,
                            dtype=torch.bfloat16)
    details = None
    rest = [t for t in tiles if t.role != "thumbnail"]
    if rest:
        first = rest[0].image
        details = torch.zeros(len(rest), first.height // V4H_PATCH,
                              first.width // V4H_PATCH, V4H_HIDDEN_SIZE,
                              dtype=torch.bfloat16)
    target = cache_path(tmp_path / "cache", source)
    save_v4h_cache(target, metadata, thumbnail, details)

    restored, rt, rd = load_v4h_cache(target)
    assert restored == metadata
    assert torch.equal(rt, thumbnail)
    assert (rd is None) == (details is None)
    assert v4h_cache_is_current(target, source, "rev-a")
    assert not v4h_cache_is_current(target, source, "rev-b")
    assert not list(target.parent.glob(".*.tmp"))     # atomic write left nothing


def test_v4h_cache_rejects_tile_count_disagreement(tmp_path):
    source = tmp_path / "image.png"
    Image.new("RGB", (640, 480), "purple").save(source)
    tiles = build_radio_tiles(Image.open(source), letterbox=False)
    metadata = make_v4h_metadata(source, "rev", tiles)
    target = cache_path(tmp_path / "cache", source)
    # one tile of features for a multi-tile metadata record
    save_v4h_cache(target, metadata,
                   torch.zeros(1, 2, 2, V4H_HIDDEN_SIZE, dtype=torch.bfloat16),
                   None)
    if metadata.tile_count > 1:
        with pytest.raises(ValueError, match="tile count"):
            load_v4h_cache(target)


def test_pool_and_pair_is_a_true_concatenation_of_neighbours():
    """Each 2560-wide token must be [left|right], not an interleave.

    The halves are different spatial cells, so getting the order wrong would
    scramble which channels belong to which location.
    """
    from rwkv_lab.radio_v4h import pool_and_pair
    lattice, channels = 4, 3
    grid = torch.zeros(1, lattice, 2 * lattice, channels)
    for row in range(lattice):
        for column in range(2 * lattice):
            grid[0, row, column] = row * 100 + column
    tokens = pool_and_pair(grid, lattice=lattice, axis="columns")
    assert tokens.shape == (1, lattice * lattice, 2 * channels)
    view = tokens[0].reshape(lattice, lattice, 2, channels)
    for row in range(lattice):
        for column in range(lattice):
            assert torch.allclose(view[row, column, 0], grid[0, row, 2 * column])
            assert torch.allclose(view[row, column, 1], grid[0, row, 2 * column + 1])


def test_pool_and_pair_preserves_left_right_geometry():
    from rwkv_lab.radio_v4h import pool_and_pair
    grid = torch.zeros(1, 8, 16, 2)
    grid[:, :, :8] = 1.0                       # left half of the tile is hot
    lattice = pool_and_pair(grid, lattice=4, axis="columns").reshape(4, 4, 4)
    assert bool((lattice[:, :2] == 1).all())
    assert bool((lattice[:, 2:] == 0).all())


def test_pool_and_pair_reaches_the_lm_width_without_parameters():
    """The whole point: 1280 -> 2560 with no learned projection."""
    from rwkv_lab.radio1d_rwkv import RadioFeatureProjector, RadioRWKVBridge
    from rwkv_lab.radio_v4h import V4H_HIDDEN_SIZE, pool_and_pair

    tokens = pool_and_pair(torch.randn(3, 24, 32, V4H_HIDDEN_SIZE), lattice=16)
    assert tokens.shape == (3, 256, 2 * V4H_HIDDEN_SIZE) == (3, 256, 2560)
    # bridge is byte-identical to the RADIO1D arm, so the encoder is the only
    # variable in the comparison
    paired = RadioRWKVBridge(hidden_size=2560, rank=256, tokens_per_tile=256,
                             max_tiles=49)
    reference = RadioRWKVBridge()
    assert paired.input_projection is None
    assert (sum(p.numel() for p in RadioFeatureProjector(bridge=paired).parameters())
            == sum(p.numel() for p in RadioFeatureProjector(bridge=reference).parameters()))


def test_pool_and_pair_rejects_a_bad_axis():
    from rwkv_lab.radio_v4h import pool_and_pair
    with pytest.raises(ValueError, match="axis"):
        pool_and_pair(torch.randn(1, 8, 8, 4), lattice=4, axis="diagonal")


def test_native_sizing_never_invents_pixels():
    """The whole point of the native path: no upscaling, ever.

    Snapping to the NEAREST multiple of 32 would round 1650 up to 1664 and
    fabricate detail, so the implementation floors instead.
    """
    from rwkv_lab.radio_v4h import V4H_MAX_EDGE, native_image_size
    for width, height in ((683, 683), (1650, 1275), (2480, 3508), (8000, 6000),
                          (640, 640), (300, 200), (1138, 1138)):
        h, w = native_image_size(width, height)
        assert h % 32 == 0 and w % 32 == 0
        assert (w // 16) % 2 == 0, "grid width must be even to pair columns"
        assert h <= height and w <= width, f"upscaled {width}x{height}"
        assert max(h, w) <= V4H_MAX_EDGE


def test_native_sizing_only_downscales_past_the_cap():
    from rwkv_lab.radio_v4h import native_image_size
    # under the cap: within one snap step of the source
    h, w = native_image_size(1024, 768)
    assert 1024 - w < 32 and 768 - h < 32
    # over the cap: scaled down, aspect preserved
    h, w = native_image_size(8000, 6000)
    assert max(h, w) == 2048
    assert abs((w / h) - (8000 / 6000)) / (8000 / 6000) < 0.05


def test_pair_columns_concatenates_neighbours_at_full_resolution():
    from rwkv_lab.radio_v4h import pair_columns
    gh, gw, channels = 3, 6, 2
    grid = torch.zeros(1, gh, gw, channels)
    for row in range(gh):
        for column in range(gw):
            grid[0, row, column] = row * 100 + column
    tokens = pair_columns(grid)
    assert tokens.shape == (1, gh * gw // 2, 2 * channels)
    view = tokens[0].reshape(gh, gw // 2, 2, channels)
    for row in range(gh):
        for column in range(gw // 2):
            assert torch.allclose(view[row, column, 0], grid[0, row, 2 * column])
            assert torch.allclose(view[row, column, 1], grid[0, row, 2 * column + 1])


def test_pair_columns_rejects_an_odd_grid():
    from rwkv_lab.radio_v4h import pair_columns
    with pytest.raises(ValueError, match="even"):
        pair_columns(torch.randn(1, 4, 5, 8))


def test_native_grid_prediction_matches_the_encoder(tmp_path):
    """Sampler bucketing keys on the predicted grid; a mismatch mis-batches."""
    from rwkv_lab.radio_v4h import (V4H_PATCH, build_native_image,
                                    native_grid_for)
    source = tmp_path / "image.png"
    Image.new("RGB", (1650, 1275), "purple").save(source)
    sized, grid = build_native_image(Image.open(source))
    assert grid == (sized.shape[0] // V4H_PATCH, sized.shape[1] // V4H_PATCH)
    assert native_grid_for({"image": str(source)}, root=tmp_path) == grid


def test_native_cache_roundtrip_and_validation(tmp_path):
    from rwkv_lab.radio_v4h import (V4H_HIDDEN_SIZE, cache_path,
                                    load_native_features, load_native_grid,
                                    native_cache_is_current, save_native_cache)
    source = tmp_path / "image.png"
    Image.new("RGB", (640, 480), "purple").save(source)
    grid = torch.zeros(1, 30, 40, V4H_HIDDEN_SIZE, dtype=torch.bfloat16)
    target = cache_path(tmp_path / "cache", source)
    save_native_cache(target, grid, revision="rev-a", source=source)

    restored, shape = load_native_grid(target)
    assert shape == (30, 40) and torch.equal(restored, grid)
    assert native_cache_is_current(target, source, "rev-a")
    assert not native_cache_is_current(target, source, "rev-b")
    assert not list(target.parent.glob(".*.tmp"))

    rows = [{"image": str(source)}]
    tokens, boxes, roles = load_native_features(
        rows, tmp_path / "cache", revision="rev-a", root=tmp_path)[0]
    # per-token, not per-image: geometry rides on each token's own box
    assert tokens.shape == (30 * 20, 2 * V4H_HIDDEN_SIZE)
    assert boxes.shape == (30 * 20, 4)
    assert roles.shape == (30 * 20,) and int(roles.sum()) == 0
    # boxes tile the image exactly once
    area = ((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])).sum()
    assert abs(float(area) - 1.0) < 1e-4


def test_native_load_raises_on_a_miss_rather_than_encoding(tmp_path):
    """Encoding inline would touch CUDA from the prefetch worker's thread."""
    from rwkv_lab.radio_v4h import load_native_features
    source = tmp_path / "absent.png"
    Image.new("RGB", (64, 64), "purple").save(source)
    with pytest.raises(FileNotFoundError, match="native"):
        load_native_features([{"image": str(source)}], tmp_path / "cache",
                             revision="rev", root=tmp_path)


def test_resize_uses_area_down_and_lanczos_up():
    """Direction-aware filters, and an identity fast path when size matches."""
    import numpy as np
    import cv2
    from rwkv_lab.radio_v4h import resize_array

    source = np.random.default_rng(0).integers(0, 255, (64, 80, 3), dtype=np.uint8)
    down = resize_array(source, 32, 40)
    assert down.shape == (32, 40, 3)
    # area by default: cubic/lanczos ring on decimation, costing glyph edges
    assert np.array_equal(down, cv2.resize(source, (40, 32),
                                           interpolation=cv2.INTER_AREA))
    up = resize_array(source, 128, 160)
    assert np.array_equal(up, cv2.resize(source, (160, 128),
                                         interpolation=cv2.INTER_LANCZOS4))
    # no resample at all when the size already matches
    assert resize_array(source, 64, 80) is source
    # bicubic remains selectable
    cubic = resize_array(source, 16, 20, downscale="bicubic")
    assert np.array_equal(cubic, cv2.resize(source, (20, 16),
                                            interpolation=cv2.INTER_CUBIC))
    with pytest.raises(ValueError, match="interpolation"):
        resize_array(source, 32, 40, downscale="nearest-ish")


def test_build_native_image_returns_an_array_at_the_predicted_grid():
    import numpy as np
    from rwkv_lab.radio_v4h import V4H_PATCH, build_native_image

    sized, grid = build_native_image(Image.new("RGB", (1650, 1275), "purple"))
    assert isinstance(sized, np.ndarray) and sized.dtype == np.uint8
    assert sized.shape[2] == 3
    assert grid == (sized.shape[0] // V4H_PATCH, sized.shape[1] // V4H_PATCH)
    assert sized.shape[0] <= 1275 and sized.shape[1] <= 1650      # never upscaled
