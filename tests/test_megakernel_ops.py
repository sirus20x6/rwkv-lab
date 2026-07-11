from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from rwkv_lab.megakernel_ops import (
    layer_norm_six_mix,
    residual_add_layer_norm,
    residual_add_layer_norm_channel_mix,
)

pytestmark = pytest.mark.gpu


def _reference_ln(x, weight, bias, eps):
    return F.layer_norm(
        x.float(), (x.shape[-1],), weight.float(), bias.float(), eps
    ).to(x.dtype)


def test_boundary_ops_functional_fallback_and_guards():
    torch.manual_seed(81)
    x = torch.randn(2, 1, 9)
    update = torch.randn_like(x)
    previous = torch.randn_like(x)
    weight = torch.randn(9)
    bias = torch.randn(9)
    six = torch.randn(6, 9)
    cmix = torch.randn(9)

    norm = _reference_ln(x, weight, bias, 1e-5)
    mixed, shift = layer_norm_six_mix(x, previous, six, weight, bias)
    expected_mixed = norm.unsqueeze(2) + (
        previous.unsqueeze(2) - norm.unsqueeze(2)
    ) * six.view(1, 1, 6, 9)
    torch.testing.assert_close(mixed, expected_mixed)
    torch.testing.assert_close(shift, norm)

    residual_fp32 = x.float() + update.float()
    expected_residual = residual_fp32.to(x.dtype)
    expected_norm = _reference_ln(residual_fp32, weight, bias, 1e-5).to(x.dtype)
    residual, mixed, shift = residual_add_layer_norm_channel_mix(
        x, update, previous, cmix, weight, bias
    )
    torch.testing.assert_close(residual, expected_residual)
    torch.testing.assert_close(shift, expected_norm)
    torch.testing.assert_close(
        mixed, expected_norm + (previous - expected_norm) * cmix.view(1, 1, 9)
    )

    residual, norm = residual_add_layer_norm(x, update, weight, bias)
    torch.testing.assert_close(residual, expected_residual)
    torch.testing.assert_close(norm, expected_norm)

    with pytest.raises(ValueError, match=r"\[batch,1,channels\]"):
        layer_norm_six_mix(x[:, 0], previous, six, weight, bias)
    with pytest.raises(ValueError, match=r"\[6,channels\]"):
        layer_norm_six_mix(x, previous, six[:5], weight, bias)
    with pytest.raises(ValueError, match=r"\[channels\]"):
        residual_add_layer_norm_channel_mix(
            x, update, previous, cmix[:-1], weight, bias
        )


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("channels", [64, 768])
def test_triton_boundaries_match_fp32_oracle(dtype, channels):
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    torch.manual_seed(82 + channels)
    shape = (2, 1, channels)
    x = torch.randn(shape, device="cuda", dtype=dtype)
    update = torch.randn_like(x)
    previous = torch.randn_like(x)
    weight = torch.randn(channels, device="cuda", dtype=dtype)
    bias = torch.randn(channels, device="cuda", dtype=dtype)
    six = torch.randn(6, channels, device="cuda", dtype=dtype)
    cmix = torch.randn(channels, device="cuda", dtype=dtype)

    expected_norm = _reference_ln(x, weight, bias, 6.4e-4)
    expected_six = expected_norm.unsqueeze(2).float() + (
        previous.unsqueeze(2).float() - expected_norm.unsqueeze(2).float()
    ) * six.float().view(1, 1, 6, channels)
    actual_six, actual_shift = layer_norm_six_mix(
        x, previous, six, weight, bias, eps=6.4e-4
    )
    torch.testing.assert_close(actual_six, expected_six.to(dtype), atol=3e-2, rtol=2e-2)
    torch.testing.assert_close(actual_shift, expected_norm, atol=3e-2, rtol=2e-2)

    residual_fp32 = x.float() + update.float()
    expected_residual = residual_fp32.to(dtype)
    expected_norm = _reference_ln(
        residual_fp32, weight, bias, 6.4e-4
    ).to(dtype)
    expected_cmix = expected_norm.float() + (
        previous.float() - expected_norm.float()
    ) * cmix.float().view(1, 1, channels)
    residual, actual_cmix, shift = residual_add_layer_norm_channel_mix(
        x, update, previous, cmix, weight, bias, eps=6.4e-4
    )
    torch.testing.assert_close(residual, expected_residual, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(actual_cmix, expected_cmix.to(dtype), atol=3e-2, rtol=2e-2)
    torch.testing.assert_close(shift, expected_norm, atol=3e-2, rtol=2e-2)

    residual, norm = residual_add_layer_norm(
        x, update, weight, bias, eps=6.4e-4
    )
    torch.testing.assert_close(residual, expected_residual, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(norm, expected_norm, atol=3e-2, rtol=2e-2)


@pytest.mark.gpu
def test_boundary_ops_are_torch_compile_visible():
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    channels = 64
    tensors = [
        torch.randn(1, 1, channels, device="cuda", dtype=torch.bfloat16)
        for _ in range(3)
    ]
    x, update, previous = tensors
    mix = torch.randn(channels, device="cuda", dtype=torch.bfloat16)
    weight = torch.ones(channels, device="cuda", dtype=torch.bfloat16)
    bias = torch.zeros(channels, device="cuda", dtype=torch.bfloat16)

    def boundary(a, b, p, m, w, z):
        return residual_add_layer_norm_channel_mix(a, b, p, m, w, z)[1]

    expected = boundary(x, update, previous, mix, weight, bias)
    compiled = torch.compile(boundary, fullgraph=True)
    actual = compiled(x, update, previous, mix, weight, bias)
    torch.testing.assert_close(actual, expected)
