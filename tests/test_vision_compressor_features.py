from __future__ import annotations

import torch

from rwkv_lab.vision_compressor_features import (
    CanonicalLatentPrefixProjector,
    FrozenTeacherCompressor,
    NativePrefixIdentity,
    RWKVNativeTeacherCompressor,
)
from rwkv_lab.vision_teacher_compressor import (
    CanonicalTeacherCompressor,
    CompressorConfig,
)


def tiny_config() -> CompressorConfig:
    return CompressorConfig(tokens=2, latent_width=8, stem_width=4,
                            layers=1, heads=2, ff_mult=1)


def test_frozen_compressor_has_no_trainable_parameters(tmp_path):
    model = CanonicalTeacherCompressor(tiny_config())
    frozen = FrozenTeacherCompressor(model, tmp_path / "frozen.pt")
    assert not frozen.training
    assert all(not parameter.requires_grad for parameter in frozen.parameters())


def test_canonical_projector_backprop_stops_at_frozen_latent():
    projector = CanonicalLatentPrefixProjector(
        rwkv_hidden=16, prefix_tokens=2, latent_width=8)
    feature = torch.randn(2, 8)
    projector([feature]).square().mean().backward()
    assert feature.grad is None
    assert all(parameter.grad is not None for parameter in projector.parameters())


def test_native_compressor_owns_rwkv_width_output(tmp_path):
    config = CompressorConfig(tokens=128, latent_width=8, stem_width=4,
                              layers=1, heads=2, ff_mult=1)
    model = CanonicalTeacherCompressor(config)
    frozen = FrozenTeacherCompressor(model, tmp_path / "frozen.pt")
    native = RWKVNativeTeacherCompressor(
        frozen, rwkv_hidden=12, prefix_tokens=64)
    moon = [torch.randn(3, 128, 4, 1152)]
    fusion = [torch.randn(128, 2176)]
    output = native(moon, fusion)
    assert output.shape == (1, 64, 12)
    assert not any(parameter.requires_grad
                   for parameter in native.compressor.parameters())
    assert {id(parameter) for parameter in native.native_parameters()} == {
        id(parameter) for parameter in native.parameters()
        if parameter.requires_grad
    }
    identity = NativePrefixIdentity(64, 12)
    assert torch.equal(identity(list(output.unbind(0))), output)
