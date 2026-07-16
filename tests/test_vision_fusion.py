import torch
import pytest
from types import SimpleNamespace

from rwkv_lab.vision_fusion import (
    AlignedFrozenVisionFeatures, FusedVisionPrefix, VisionFusionResidual,
    VisionTowerConfig, pool_tokens, sam_dense_cache_key,
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
