import torch
import pytest
from types import SimpleNamespace

from rwkv_lab.vision_fusion import (
    AlignedFrozenVisionFeatures, FusedVisionPrefix, SamAlignedFrozenFeatures,
    VisionFusionResidual,
    VisionTowerConfig, factor_grid, pool_grid_tokens, pool_tokens,
    sam_cropped_tokens, sam_dense_cache_key, sam_live_cells,
    valid_aligned_feature, valid_sam_dense_feature)


def test_pool_tokens_shape_and_mean_preservation():
    x = torch.arange(2 * 17 * 3, dtype=torch.float32).reshape(2, 17, 3)
    pooled = pool_tokens(x, 5)
    assert pooled.shape == (2, 5, 3)
    assert torch.allclose(pooled.mean(dim=1), x.mean(dim=1), atol=1e-5)


def test_prefix_configuration_matches_context_budget():
    cfg = VisionTowerConfig(siglip_tokens=4, dinov2_tokens=5, sam_tokens=6)
    prefix = FusedVisionPrefix(rwkv_hidden_size=32, config=cfg)
    assert cfg.token_budget == 15
    assert [layer.in_features for layer in prefix.projections] == [768, 768, 256]
    assert prefix.tower_type.shape == (3, 1, 32)


def test_pool_tokens_rejects_zero_budget():
    with pytest.raises(ValueError, match="positive"):
        pool_tokens(torch.ones(1, 4, 3), 0)


def test_fused_prefix_accepts_low_precision_frozen_tower_outputs():
    cfg = VisionTowerConfig(siglip_tokens=2, dinov2_tokens=2, sam_tokens=2)
    prefix = FusedVisionPrefix(rwkv_hidden_size=8, config=cfg)
    prefix.siglip = prefix.dinov2 = prefix.sam = torch.nn.Identity()
    raw = (
        torch.ones(1, 4, 768, dtype=torch.bfloat16),
        torch.ones(1, 4, 768, dtype=torch.bfloat16),
        torch.ones(1, 4, 256, dtype=torch.bfloat16),
    )
    prefix.extract_tower_tokens = lambda images, device: raw

    result = prefix([object()], device="cpu")

    assert result.shape == (1, 6, 8)
    assert result.dtype == torch.float32


def test_aligned_fusion_supports_so400m_width_without_aliasing_base():
    base = AlignedFrozenVisionFeatures(VisionTowerConfig(siglip_width=768))
    so400m = AlignedFrozenVisionFeatures(VisionTowerConfig(siglip_width=1152))
    assert base.width == 1792
    assert so400m.width == 2176
    assert base.cache_fingerprint != so400m.cache_fingerprint
    adapter = VisionFusionResidual(32, rank=8, source_width=so400m.width)
    features = torch.randn(2, 4, so400m.width)
    assert adapter(features).shape == (2, 4, 32)
    assert valid_aligned_feature(features[0], 4, so400m.width)
    assert not valid_aligned_feature(features[0], 4, base.width)


def test_so400m_prefix_projection_uses_configured_width():
    model = FusedVisionPrefix(
        rwkv_hidden_size=32, config=VisionTowerConfig(siglip_width=1152))
    assert [layer.in_features for layer in model.projections] == [1152, 768, 256]


def test_sam_global_fusion_has_bounded_radio_cache_geometry(tmp_path):
    model = SamAlignedFrozenFeatures(tmp_path / "sam", tokens=128)
    assert model.width == 256
    assert model.fusion_tokens == 128
    assert not model.loaded
    assert len(model.cache_fingerprint) == 64


def test_dense_sam_payload_and_cache_key_include_source_identity(tmp_path):
    image = tmp_path / "image.jpg"
    image.write_bytes(b"first")
    feature = torch.ones(256, 64, 64, dtype=torch.bfloat16)
    assert valid_sam_dense_feature(feature)
    assert not valid_sam_dense_feature(feature.flatten(1))
    first = sam_dense_cache_key(image, tower_fingerprint="sam-a")
    second_tower = sam_dense_cache_key(image, tower_fingerprint="sam-b")
    image.write_bytes(b"second-version")
    second_source = sam_dense_cache_key(image, tower_fingerprint="sam-a")
    assert len({first, second_tower, second_source}) == 3


def test_frozen_extractors_cast_processor_pixels_to_tower_dtype():
    class Inputs:
        def __init__(self):
            self.pixel_values = torch.ones(1, 3, 4, 4, dtype=torch.float32)
            # SamProcessor always reports this; the SAM crop depends on it.
            self.reshaped_input_sizes = torch.tensor([[4, 4]])

        def to(self, _device):
            return self

    class Processor:
        def __call__(self, **_kwargs):
            return Inputs()

    class Tower(torch.nn.Module):
        def __init__(self, width):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros((), dtype=torch.bfloat16))
            self.width = width

        def forward(self, *, pixel_values):
            assert pixel_values.dtype == self.anchor.dtype
            return SimpleNamespace(last_hidden_state=torch.ones(
                1, 4, self.width, dtype=self.anchor.dtype))

    class Sam(Tower):
        def get_image_embeddings(self, pixel_values):
            assert pixel_values.dtype == self.anchor.dtype
            return torch.ones(1, 256, 2, 2, dtype=self.anchor.dtype)

    prefix = FusedVisionPrefix()
    prefix.siglip_processor = prefix.dinov2_processor = prefix.sam_processor = Processor()
    prefix.siglip, prefix.dinov2, prefix.sam = Tower(768), Tower(768), Sam(256)
    values = prefix.extract_tower_tokens([object()], device="cpu")
    assert [tuple(value.shape) for value in values] == [
        (1, 4, 768), (1, 4, 768), (1, 4, 256)]


def test_factor_grid_is_the_most_square_factorization():
    assert factor_grid(128) == (8, 16)
    assert factor_grid(256) == (16, 16)
    assert factor_grid(96) == (8, 12)
    assert factor_grid(1) == (1, 1)


def test_pool_grid_tokens_keeps_both_spatial_axes():
    """A vertical split must survive pooling; the 1D sequence pool loses it."""
    grid = torch.zeros(1, 1, 8, 8)
    grid[..., :4] = 1.0            # left half hot, right half cold

    spatial = pool_grid_tokens(grid, 4).reshape(2, 2)
    assert torch.allclose(spatial, torch.tensor([[1.0, 0.0], [1.0, 0.0]]))

    # Flatten-then-pool averages row-major runs, erasing the left/right contrast.
    flat = pool_tokens(grid.flatten(2).transpose(1, 2), 4)[0, :, 0]
    assert torch.allclose(flat, torch.full((4,), 0.5))


def test_sam_live_cells_tracks_the_unpadded_canvas():
    assert sam_live_cells(1024, 1024) == (64, 64)   # square: no padding
    assert sam_live_cells(576, 1024) == (36, 64)    # 16:9: 44% of rows are padding
    assert sam_live_cells(1024, 768) == (64, 48)    # 3:4 portrait
    assert sam_live_cells(1, 1) == (1, 1)


def test_sam_cropped_tokens_never_average_the_padded_canvas():
    """Padding must not reach the pooled feature at any aspect ratio.

    Pooling SAM's full 64x64 grid spends 44% of a 16:9 feature describing the
    zero-padded canvas, producing tokens that are identical for every image of
    that shape — the failure that made the SAM tower contribute nothing.
    """
    dense = torch.full((1, 4, 64, 64), -9.0)        # padding sentinel
    dense[..., :36, :] = 1.0                        # 16:9 live region
    sizes = torch.tensor([[576, 1024]])

    cropped = sam_cropped_tokens(dense, sizes, 128)
    assert cropped.shape == (1, 128, 4)
    assert torch.allclose(cropped, torch.ones_like(cropped))

    contaminated = pool_tokens(dense.flatten(2).transpose(1, 2), 128)
    assert (contaminated < 0).any()                 # the old path pools padding


def test_sam_cropped_tokens_handle_mixed_aspect_batches():
    dense = torch.zeros(2, 4, 64, 64)
    dense[0, :, :36, :] = 1.0                       # landscape row
    dense[1, :, :, :48] = 2.0                       # portrait row
    sizes = torch.tensor([[576, 1024], [1024, 768]])
    pooled = sam_cropped_tokens(dense, sizes, 32)
    assert pooled.shape == (2, 32, 4)
    assert torch.allclose(pooled[0], torch.ones_like(pooled[0]))
    assert torch.allclose(pooled[1], torch.full_like(pooled[1], 2.0))


def test_sam_crop_padding_is_opt_in_and_repartitions_the_cache():
    """The shipped frozen compressor was trained on uncropped SAM features."""
    legacy = VisionTowerConfig()
    assert legacy.sam_crop_padding is False
    cropped = VisionTowerConfig(sam_crop_padding=True)
    assert legacy.fingerprint() != cropped.fingerprint()


def test_sam_fusion_tower_fingerprint_rejects_the_uncropped_generation():
    v2 = SamAlignedFrozenFeatures("models/vision/sam-vit-base", tokens=128)
    assert v2.cache_fingerprint != SamAlignedFrozenFeatures(
        "models/vision/sam-vit-base", tokens=64).cache_fingerprint
