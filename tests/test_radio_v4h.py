import json

import pytest
import torch
from PIL import Image

from rwkv_lab.radio1d_rwkv import build_radio_tiles, native_aspect_size
from rwkv_lab.radio_v4h import (
    V4H_HIDDEN_SIZE, V4H_LATTICE, V4H_PATCH, V4HCacheMetadata, cache_path,
    load_v4h_cache, make_v4h_metadata, native_cache_token_count,
    pool_to_lattice, save_native_cache, save_v4h_cache,
    v4h_artifact_fingerprint, v4h_cache_is_current)


def test_v4h_artifact_fingerprint_hashes_contents_not_directory_mtime(
        tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text('{"width":1280}')
    weights = model / "model.safetensors"
    weights.write_bytes(b"weights-a")
    (model / "modeling.py").write_text("VERSION = 1\n")
    first = v4h_artifact_fingerprint(model)

    weights.write_bytes(b"weights-b")
    assert weights.stat().st_size == len(b"weights-a")
    assert v4h_artifact_fingerprint(model) != first


def test_v4h_artifact_fingerprint_cache_invalidates_on_artifact_stat(
        tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    config = model / "config.json"
    config.write_text('{"width":1280}')
    (model / "model.safetensors").write_bytes(b"weights")
    cache = tmp_path / "fingerprint.json"
    first = v4h_artifact_fingerprint(model, fingerprint_cache=cache)
    assert v4h_artifact_fingerprint(model, fingerprint_cache=cache) == first

    config.write_text('{"width":4096}')
    assert v4h_artifact_fingerprint(
        model, fingerprint_cache=cache) != first


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
        assert h % 16 == 0 and w % 32 == 0
        assert (w // 16) % 2 == 0, "grid width must be even to pair columns"
        assert h <= height and w <= width, f"upscaled {width}x{height}"
        assert max(h, w) <= V4H_MAX_EDGE


def test_native_sizing_uses_the_smallest_required_step_per_axis():
    from rwkv_lab.radio_v4h import native_image_size

    # Height does not participate in horizontal channel pairing and must not
    # lose an extra patch merely to satisfy width's even-grid requirement.
    assert native_image_size(640, 496) == (496, 640)
    assert native_image_size(640, 500) == (496, 640)


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


def test_native_cache_token_count_uses_tensor_bytes_not_manifest_geometry(
        tmp_path):
    source = tmp_path / "rotated.jpg"
    Image.new("RGB", (333, 500), "purple").save(source)
    cache = tmp_path / "native.safetensors"
    grid = torch.zeros(1, 20, 30, 4096, dtype=torch.bfloat16)
    save_native_cache(cache, grid, revision="test", source=source)

    assert native_cache_token_count(cache, hidden_size=4096) == 600


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


def test_fused_native_cells_roundtrip_without_width_projection(tmp_path):
    from rwkv_lab.radio_v4h import (
        cache_path, load_native_features, native_cache_is_current,
        save_native_cache)
    source = tmp_path / "image.png"
    Image.new("RGB", (64, 32), "purple").save(source)
    grid = torch.randn(1, 2, 4, 4096, dtype=torch.bfloat16)
    target = cache_path(tmp_path / "cache", source)
    save_native_cache(target, grid, revision="all-adaptors", source=source)

    assert native_cache_is_current(
        target, source, "all-adaptors", hidden_size=4096)
    assert not native_cache_is_current(
        target, source, "all-adaptors", hidden_size=1280)
    tokens, boxes, roles = load_native_features(
        [{"image": str(source)}], tmp_path / "cache",
        revision="all-adaptors", root=tmp_path, hidden_size=4096,
        packing="cells")[0]
    assert tokens.shape == (8, 4096)
    assert torch.equal(tokens, grid.flatten(1, 2)[0])
    assert boxes.shape == (8, 4)
    assert roles.shape == (8,)
    area = ((boxes[:, 2] - boxes[:, 0])
            * (boxes[:, 3] - boxes[:, 1])).sum()
    assert abs(float(area) - 1.0) < 1e-5


def test_native_cache_validation_rejects_tensor_metadata_shape_disagreement(
        tmp_path):
    from safetensors.torch import save_file
    from rwkv_lab.radio_v4h import (
        NATIVE_CACHE_SCHEMA, cache_path, native_cache_is_current,
        save_native_cache)

    source = tmp_path / "image.png"
    Image.new("RGB", (64, 32), "purple").save(source)
    target = cache_path(tmp_path / "cache", source)
    save_native_cache(
        target, torch.zeros(1, 2, 4, 4096, dtype=torch.bfloat16),
        revision="all-adaptors", source=source)

    from safetensors import safe_open
    with safe_open(str(target), framework="pt", device="cpu") as handle:
        metadata = handle.metadata()
    save_file(
        {"grid": torch.zeros(1, 1, 8, 4096, dtype=torch.bfloat16)},
        str(target), metadata=metadata)
    assert metadata["schema"] == NATIVE_CACHE_SCHEMA
    assert not native_cache_is_current(
        target, source, "all-adaptors", hidden_size=4096)


def test_all_adaptor_widths_fill_rwkv_7b_exactly():
    """Widths, concat ORDER and every calibration scale, on non-zero heads.

    Zero-valued fake heads make this a shape test only: zeros survive any
    scale and any permutation of two blocks, so a swapped SigLIP2/SAM3 slice or
    a dropped fusion_scale still passed. Each head therefore emits its own
    non-zero constant here.
    """
    from rwkv_lab.radio_v4h_adaptors import (
        DINO_COMPACT_WIDTH, DINO_FUSION_SCALE, FUSED_ADAPTOR_WIDTH,
        SAM3_FUSION_SCALE, SAM3_SPATIAL_WIDTH, SIGLIP2_FUSION_SCALE,
        SIGLIP2_SPATIAL_WIDTH, V4HAdaptorFusion)

    class Constant(torch.nn.Module):
        def __init__(self, width, value):
            super().__init__()
            self.width, self.value = width, value

        def forward(self, value):
            return value.new_full((*value.shape[:-1], self.width), self.value)

    fusion = V4HAdaptorFusion(
        Constant(SIGLIP2_SPATIAL_WIDTH, 2.0), Constant(SAM3_SPATIAL_WIDTH, 3.0),
        Constant(DINO_COMPACT_WIDTH, 5.0))
    output = fusion(torch.randn(2, 7, 1280))
    assert output.shape == (2, 7, FUSED_ADAPTOR_WIDTH)
    # 1536 + 1024 + 1536 lands exactly on the 4096-wide RWKV 7.2B/13.3B token
    assert (FUSED_ADAPTOR_WIDTH
            == SIGLIP2_SPATIAL_WIDTH + SAM3_SPATIAL_WIDTH + DINO_COMPACT_WIDTH
            == 4096)

    bounds = (0, SIGLIP2_SPATIAL_WIDTH,
              SIGLIP2_SPATIAL_WIDTH + SAM3_SPATIAL_WIDTH, FUSED_ADAPTOR_WIDTH)
    for (start, stop), value, scale in zip(
            zip(bounds, bounds[1:]), (2.0, 3.0, 5.0),
            (SIGLIP2_FUSION_SCALE, SAM3_FUSION_SCALE, DINO_FUSION_SCALE)):
        block = output[..., start:stop]
        assert torch.allclose(block, torch.full_like(block, value * scale)), (
            "each teacher must occupy its own slice, scaled by its own "
            "calibration constant")
    # the three scaled constants are mutually distinct, so any two swapped
    # blocks or any missing scale changes at least one slice above
    assert len({round(v * s, 6) for v, s in zip(
        (2.0, 3.0, 5.0),
        (SIGLIP2_FUSION_SCALE, SAM3_FUSION_SCALE, DINO_FUSION_SCALE))}) == 3


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


def test_writer_and_reader_agree_on_the_cache_revision():
    """A revision mismatch presents as an ENTIRELY ABSENT cache, not a version
    error, so it survives verification that hardcodes the writer's own string.
    That is exactly how it reached a launch: 110k entries were valid, and the
    trainer asked for a different revision than the cache script wrote.
    """
    import argparse
    import importlib.util
    import sys
    from pathlib import Path

    from rwkv_lab.radio_v4h import DEFAULT_NATIVE_REVISION

    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "_cache_v4h_native", root / "scripts" / "cache_v4h_native.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    try:
        assert module.DEFAULT_REVISION == DEFAULT_NATIVE_REVISION
    finally:
        sys.modules.pop(spec.name, None)

    source = (root / "src" / "rwkv_lab" / "vision_train.py").read_text()
    assert 'ap.add_argument("--radio-v4h-revision",\n' in source
    assert "default=_DEFAULT_NATIVE_REVISION," in source, (
        "the trainer must take its default from the shared constant, not a literal")


def test_stale_and_absent_cache_entries_report_differently(tmp_path):
    """'Absent' and 'exists but rejected' need distinct messages to diagnose."""
    from rwkv_lab.radio_v4h import (V4H_HIDDEN_SIZE, cache_path,
                                    load_native_features, save_native_cache)
    source = tmp_path / "image.png"
    Image.new("RGB", (640, 480), "purple").save(source)
    rows = [{"image": str(source)}]

    with pytest.raises(FileNotFoundError, match="no native"):
        load_native_features(rows, tmp_path / "cache", revision="rev-a",
                             root=tmp_path)

    save_native_cache(cache_path(tmp_path / "cache", source),
                      torch.zeros(1, 30, 40, V4H_HIDDEN_SIZE, dtype=torch.bfloat16),
                      revision="rev-a", source=source)
    with pytest.raises(FileNotFoundError, match="stale"):
        load_native_features(rows, tmp_path / "cache", revision="rev-b",
                             root=tmp_path)


def test_native_backend_disables_the_tile_quantum_guard():
    """The deep-vision guard and the projector must agree on native mode.

    They disagreed once: is_radio_backend() is true for both RADIO backends,
    so the native path was handed the tiled quantum of 128 and every step
    raised on a prefix length of grid_h*(grid_w//2).
    """
    import argparse
    from rwkv_lab.vision_train import uses_native_prefix

    native = argparse.Namespace(vision_backend="radio_v4h", radio_v4h_native=True)
    lattice = argparse.Namespace(vision_backend="radio_v4h", radio_v4h_native=False)
    tiled = argparse.Namespace(vision_backend="radio1d", radio_v4h_native=True)
    assert uses_native_prefix(native)
    assert not uses_native_prefix(lattice)
    assert not uses_native_prefix(tiled)


def test_project_visual_prefix_is_the_only_native_dispatch():
    """Native vs tiled must be decided in exactly one place.

    multimodal_loss dispatched on native_mode; write_eval_samples called the
    tiled path unconditionally. Training ran natively for 1,000 steps and then
    died at its first qualitative sample. A unit test on either function alone
    would have passed -- the defect was that the two disagreed -- so this
    asserts the decision exists once in the source.
    """
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    source = (root / "src" / "rwkv_lab" / "vision_train.py").read_text()
    assert source.count(".forward_native(") == 1, (
        "native dispatch must live only in project_visual_prefix")
    assert source.count("def project_visual_prefix") == 1


def test_project_visual_prefix_routes_native_and_tiled():
    import torch.nn as nn
    from rwkv_lab.radio1d_rwkv import RadioFeatureProjector
    from rwkv_lab.vision_train import project_visual_prefix

    class Fake(RadioFeatureProjector):
        def __init__(self, native):
            nn.Module.__init__(self)
            self.native_mode = native
            self.seen = []

        def forward_native(self, features):
            self.seen.append("native")
            return "NATIVE"

        def forward(self, features, image_aspect=None):
            self.seen.append(("tiled", image_aspect))
            return "TILED"

    native = Fake(True)
    assert project_visual_prefix(native, "F", image_aspect="A") == "NATIVE"
    assert native.seen == ["native"]

    tiled = Fake(False)
    assert project_visual_prefix(tiled, "F", image_aspect="A") == "TILED"
    assert tiled.seen == [("tiled", "A")]

    class Plain(nn.Module):
        def forward(self, features):
            return "PLAIN"

    assert project_visual_prefix(Plain(), "F", image_aspect="A") == "PLAIN"


def test_fingerprint_rejects_a_size_and_mtime_preserving_replacement(tmp_path):
    """cp -p / rsync -t / tar -p restore both stat fields of the ORIGINAL.

    The cached short-circuit keyed on (size, mtime_ns) alone therefore returned
    the previous fingerprint for a swapped weight file -- and this fingerprint
    is the guard against training on a stale feature cache.
    """
    import os

    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text('{"width":1280}')
    weights = model / "model.safetensors"
    weights.write_bytes(b"weights-a" * 4096)
    cache = tmp_path / "fingerprint.json"
    first = v4h_artifact_fingerprint(model, fingerprint_cache=cache)
    stat = weights.stat()

    weights.write_bytes(b"weights-b" * 4096)
    os.utime(weights, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    assert weights.stat().st_size == stat.st_size
    assert weights.stat().st_mtime_ns == stat.st_mtime_ns
    assert v4h_artifact_fingerprint(model, fingerprint_cache=cache) != first


def test_fingerprint_include_adaptors_covers_the_fused_artifacts(tmp_path):
    from rwkv_lab.radio_v4h_adaptors import (
        DEFAULT_ADAPTOR_CHECKPOINT, DEFAULT_COMPACTOR)

    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text('{"width":1280}')
    (model / "model.safetensors").write_bytes(b"weights")
    native = v4h_artifact_fingerprint(model)

    # Missing fused artifacts name themselves and say why they are required.
    with pytest.raises(FileNotFoundError) as failure:
        v4h_artifact_fingerprint(model, include_adaptors=True)
    message = str(failure.value)
    assert DEFAULT_ADAPTOR_CHECKPOINT in message and DEFAULT_COMPACTOR in message
    assert "teacher adaptor weights" in message

    checkpoint = model / DEFAULT_ADAPTOR_CHECKPOINT
    checkpoint.write_bytes(b"adaptors-a")
    (model / DEFAULT_COMPACTOR).write_bytes(b"compactor")
    fused = v4h_artifact_fingerprint(model, include_adaptors=True)
    assert fused != native

    checkpoint.write_bytes(b"adaptors-b")
    assert v4h_artifact_fingerprint(model, include_adaptors=True) != fused


def test_native_revision_names_the_snapping_convention():
    """A 1500px page snaps to 92 rows under a shared 32 step and 93 under 16/32.

    Both entries claim the same revision unless it is bumped, so one cache
    directory silently holds two incompatible geometries.
    """
    from rwkv_lab.radio_v4h import (
        DEFAULT_NATIVE_REVISION, V4H_NATIVE_HEIGHT_STEP, V4H_NATIVE_WIDTH_STEP,
        native_image_size)

    assert native_image_size(1000, 1500)[0] // 16 == 93
    assert native_image_size(1000, 1500, step=32)[0] // 16 == 92
    assert DEFAULT_NATIVE_REVISION != "c-radiov4-h", (
        "the pre-split revision string cannot describe the split convention")
    assert (f"h{V4H_NATIVE_HEIGHT_STEP}w{V4H_NATIVE_WIDTH_STEP}"
            in DEFAULT_NATIVE_REVISION)


def test_native_cache_records_and_checks_the_snapping_steps(tmp_path):
    from safetensors import safe_open

    from rwkv_lab.radio_v4h import (
        V4H_HIDDEN_SIZE, cache_path, native_cache_is_current, save_native_cache)

    source = tmp_path / "image.png"
    Image.new("RGB", (640, 480), "purple").save(source)
    target = cache_path(tmp_path / "cache", source)
    save_native_cache(
        target, torch.zeros(1, 30, 40, V4H_HIDDEN_SIZE, dtype=torch.bfloat16),
        revision="rev-a", source=source)

    with safe_open(str(target), framework="pt", device="cpu") as handle:
        meta = json.loads(handle.metadata()["radio_metadata"])
    assert (meta["height_step"], meta["width_step"]) == (16, 32)
    assert native_cache_is_current(target, source, "rev-a")
    assert not native_cache_is_current(
        target, source, "rev-a", height_step=32)


def test_native_cache_refuses_a_cross_configuration_overwrite(tmp_path):
    """A 4096-wide writer aimed at a 1280-wide cache must not destroy it.

    cache_path hashes only the source, so both writers address the same file;
    every entry then fails its currency check, is re-encoded, and os.replace
    takes the old one away. One wrong --cache-dir destroyed a 117k-image cache
    with no prompt and no error.
    """
    from rwkv_lab.radio_v4h import (
        V4H_HIDDEN_SIZE, cache_path, load_native_grid, save_native_cache)

    source = tmp_path / "image.png"
    Image.new("RGB", (640, 480), "purple").save(source)
    target = cache_path(tmp_path / "cache", source)
    native = torch.zeros(1, 30, 40, V4H_HIDDEN_SIZE, dtype=torch.bfloat16)
    save_native_cache(target, native, revision="native", source=source)

    fused = torch.ones(1, 30, 40, 4096, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="refusing to overwrite"):
        save_native_cache(target, fused, revision="fused", source=source)
    # a different revision at the same width is equally refused
    with pytest.raises(ValueError, match="refusing to overwrite"):
        save_native_cache(target, native, revision="native-v2", source=source)
    restored, shape = load_native_grid(target)
    assert shape == (30, 40) and restored.shape[-1] == V4H_HIDDEN_SIZE

    # re-encoding under the SAME configuration stays an ordinary overwrite
    save_native_cache(target, native + 1, revision="native", source=source)
    assert bool(load_native_grid(target)[0].eq(1).all())
    # and a deliberate reconfiguration is still possible, explicitly
    save_native_cache(target, fused, revision="fused", source=source,
                      allow_reconfigure=True)
    assert load_native_grid(target)[0].shape[-1] == 4096
    assert not list(target.parent.glob(".*.tmp"))


def test_native_grid_prediction_is_exif_aware(tmp_path):
    """build_native_image transposes first, so the prediction must too.

    Under the old shared 32-pixel step a transposed prediction still produced
    the same cell count; with 16 rows / 32 columns it does not, and the sampler
    then buckets a row against a grid the encoder never wrote.
    """
    from rwkv_lab.radio_v4h import build_native_image, native_grid_for

    source = tmp_path / "rotated.jpg"
    image = Image.new("RGB", (1500, 1008), "purple")
    exif = image.getexif()
    exif[0x0112] = 6                       # rotate 90 degrees on load
    image.save(source, exif=exif)

    with Image.open(source) as stored:
        assert stored.size == (1500, 1008)
        _sized, grid = build_native_image(stored)
    assert native_grid_for({"image": str(source)}, root=tmp_path) == grid
    # the manifest carries the PRE-transpose size, which must not win
    assert native_grid_for(
        {"image": str(source), "width": 1500, "height": 1008},
        root=tmp_path) == grid
    # a row may carry the orientation itself and skip the header read
    assert native_grid_for(
        {"image": str(source), "width": 1500, "height": 1008,
         "exif_orientation": 6}, root=tmp_path) == grid


def test_native_cache_token_count_at_the_lattice_width(tmp_path):
    from rwkv_lab.radio_v4h import V4H_HIDDEN_SIZE, save_native_cache

    source = tmp_path / "image.png"
    Image.new("RGB", (64, 32), "purple").save(source)
    cache = tmp_path / "native.safetensors"
    grid = torch.zeros(1, 30, 40, V4H_HIDDEN_SIZE, dtype=torch.bfloat16)

    # An ordinary revision keeps the header well under one 2560-byte cell, so
    # the metadata-only default and the header read agree exactly. The default
    # is what the trainer's per-row planning loop uses: reading even 8 bytes
    # faults in a readahead window (~1.6 MB of real I/O per entry on the array
    # holding these caches), which over ~110k rows added hours to every startup.
    save_native_cache(cache, grid, revision="c-radiov4-h-native-h16w32",
                      source=source)
    assert native_cache_token_count(cache, hidden_size=V4H_HIDDEN_SIZE) == 1200
    assert native_cache_token_count(
        cache, hidden_size=V4H_HIDDEN_SIZE, exact_header=True) == 1200

    # A pathologically long free-form revision pushes the header past one cell.
    # Only the header read stays exact there; the inferred count is one high.
    # The revision is uniform across a cache directory, so this is a property of
    # the cache rather than of an entry -- probe one entry, don't pay per row.
    pathological = tmp_path / "long_revision.safetensors"
    save_native_cache(pathological, grid, revision="r" * 4096, source=source)
    assert native_cache_token_count(
        pathological, hidden_size=V4H_HIDDEN_SIZE, exact_header=True) == 1200
    assert native_cache_token_count(
        pathological, hidden_size=V4H_HIDDEN_SIZE) == 1201

    for exact in (False, True):
        with pytest.raises(ValueError, match="hidden size must be positive"):
            native_cache_token_count(cache, hidden_size=0, exact_header=exact)

    # Only the header read can validate the width: dividing a known payload
    # leaves a nonzero remainder for a wrong hidden size. The inferred form
    # absorbs the discrepancy into the header it is guessing and returns a
    # plausible wrong count, so it must never be read as evidence that a cache
    # is the width the caller asked for. load_native_features is the authority
    # -- it compares against the real tensor when the row loads.
    with pytest.raises(ValueError, match="file geometry"):
        native_cache_token_count(cache, hidden_size=1279, exact_header=True)
    assert native_cache_token_count(cache, hidden_size=1279) == 1201
    truncated = tmp_path / "short.safetensors"
    truncated.write_bytes(b"abc")
    with pytest.raises(ValueError, match="too short"):
        native_cache_token_count(truncated, hidden_size=V4H_HIDDEN_SIZE,
                                 exact_header=True)
    with pytest.raises(ValueError, match="file geometry"):
        native_cache_token_count(truncated, hidden_size=V4H_HIDDEN_SIZE)


def test_pinned_package_name_is_stable_across_interpreters(tmp_path):
    """PYTHONHASHSEED must not decide which module object the adaptors import.

    ``abs(hash(str(path)))`` is salted per process, so the synthetic package
    name differed between a worker and its parent and each got its own copy of
    the remote classes.
    """
    import hashlib
    import os

    from rwkv_lab.radio_v4h import _pinned_package_name

    model = tmp_path / "model"
    model.mkdir()
    digest = hashlib.sha256(os.fsencode(str(model.resolve()))).hexdigest()
    assert (_pinned_package_name(model, "radio_v4h")
            == f"_rwkv_radio_v4h_{digest[:16]}")
    # a trailing separator is the same directory, hence the same package
    assert (_pinned_package_name(tmp_path / "model" / "", "radio_v4h")
            == _pinned_package_name(model, "radio_v4h"))


def test_remote_adaptor_module_reuses_the_encoder_package(tmp_path):
    """One package per model directory, so the classes are not duplicated."""
    import sys

    from rwkv_lab.radio_v4h import _pinned_package_name
    from rwkv_lab.radio_v4h_adaptors import _remote_module

    model = tmp_path / "model"
    model.mkdir()
    (model / "adaptor_module_factory.py").write_text("class MLP2:\n    pass\n")
    package = _pinned_package_name(model, "radio_v4h")
    try:
        module = _remote_module(model, "adaptor_module_factory.py")
        assert module.__name__ == f"{package}.adaptor_module_factory"
        assert _remote_module(model, "adaptor_module_factory.py") is module
        assert module.MLP2 is sys.modules[module.__name__].MLP2
    finally:
        for name in [n for n in sys.modules if n.startswith(package)]:
            sys.modules.pop(name, None)


@pytest.fixture(scope="module")
def dino_qr_compactor():
    """One real-geometry QR factorization shared by the compactor tests."""
    from rwkv_lab.radio_v4h_adaptors import (
        DINO_SPATIAL_WIDTH, build_dino_qr_compactor)

    torch.manual_seed(20260726)
    head = torch.nn.Module()
    head.final = torch.nn.Sequential(
        torch.nn.Identity(), torch.nn.Linear(1520, DINO_SPATIAL_WIDTH))
    compact, basis = build_dino_qr_compactor(head)
    return head.final[-1], compact, basis


def test_dino_qr_compactor_reconstructs_the_original_affine_map(
        dino_qr_compactor):
    from rwkv_lab.radio_v4h_adaptors import DINO_QR_RANK

    final, compact, basis = dino_qr_compactor
    assert basis.shape == (4096, DINO_QR_RANK)
    latent = torch.randn(4, final.in_features)
    expected = final(latent)
    reconstructed = compact(latent)[:, :DINO_QR_RANK] @ basis.T
    relative = ((reconstructed - expected).norm()
                / expected.norm()).item()
    assert relative < 2e-5, relative
    # inner products survive too: that is what makes the coordinates a faithful
    # substitute for the 4096-wide output rather than a lossy projection
    gram = compact(latent)[:, :DINO_QR_RANK] @ compact(latent)[:, :DINO_QR_RANK].T
    assert torch.allclose(gram, expected @ expected.T, rtol=2e-4, atol=2e-3)


def test_dino_qr_compactor_tail_is_permanently_zero(dino_qr_compactor):
    from rwkv_lab.radio_v4h_adaptors import DINO_COMPACT_WIDTH, DINO_QR_RANK

    _final, compact, _basis = dino_qr_compactor
    assert compact.out_features == DINO_COMPACT_WIDTH
    assert bool(compact.weight[DINO_QR_RANK:].eq(0).all())
    assert bool(compact.bias[DINO_QR_RANK:].eq(0).all())
    tail = compact(torch.randn(3, compact.in_features))[:, DINO_QR_RANK:]
    assert tail.shape[-1] == DINO_COMPACT_WIDTH - DINO_QR_RANK == 15
    assert bool(tail.eq(0).all()), "the padded tail carries no information"
    assert not compact.weight.requires_grad


def test_dino_qr_compactor_rejects_unexpected_heads():
    from rwkv_lab.radio_v4h_adaptors import (
        DINO_SPATIAL_WIDTH, build_dino_qr_compactor)

    def head(final):
        module = torch.nn.Module()
        module.final = torch.nn.Sequential(final)
        return module

    with pytest.raises(TypeError, match="nn.Linear"):
        build_dino_qr_compactor(head(torch.nn.GELU()))
    with pytest.raises(ValueError, match="unexpected DINO final geometry"):
        build_dino_qr_compactor(head(torch.nn.Linear(1024, DINO_SPATIAL_WIDTH)))
    with pytest.raises(ValueError, match="affine rank"):
        build_dino_qr_compactor(
            head(torch.nn.Linear(1520, DINO_SPATIAL_WIDTH)), output_width=1520)


def test_dino_qr_compactor_roundtrips_and_rejects_a_foreign_schema(tmp_path):
    from safetensors.torch import save_file

    from rwkv_lab.radio_v4h_adaptors import (
        load_dino_qr_compactor, save_dino_qr_compactor)

    compact = torch.nn.Linear(6, 8)
    basis = torch.randn(10, 5)
    target = tmp_path / "compactor.safetensors"
    save_dino_qr_compactor(target, compact, basis)

    restored, restored_basis = load_dino_qr_compactor(target)
    assert (restored.in_features, restored.out_features) == (6, 8)
    assert torch.equal(restored.weight, compact.weight)
    assert torch.equal(restored.bias, compact.bias)
    assert torch.equal(restored_basis, basis)
    assert not restored.weight.requires_grad
    assert not list(target.parent.glob(".*.tmp"))

    foreign = tmp_path / "foreign.safetensors"
    save_file({"weight": compact.weight.detach().contiguous()}, str(foreign),
              metadata={"schema": "something-else"})
    with pytest.raises(ValueError, match="unsupported DINO compactor schema"):
        load_dino_qr_compactor(foreign)
