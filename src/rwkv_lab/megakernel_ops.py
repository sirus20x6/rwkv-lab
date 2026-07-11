"""Portable fused boundaries for one-token RWKV inference.

These operations deliberately stop at GEMM boundaries.  They reduce launch
and memory traffic without baking in a GPU-specific weight layout, and expose
the Triton programs as compiler-visible custom operators for ``torch.compile``.
All normalization and interpolation arithmetic accumulates in fp32; returned
activations retain the input dtype.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

try:  # Triton is supplied by the CUDA PyTorch environment.
    import triton
    import triton.language as tl

    _HAVE_TRITON = True
except Exception:  # pragma: no cover - depends on the installed CUDA stack
    triton = tl = None
    _HAVE_TRITON = False


def _layer_norm_fp32(
    value: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    result = F.layer_norm(
        value.float(), (value.shape[-1],), weight.float(), bias.float(), eps
    )
    return result.to(value.dtype)


def _check_token(value: torch.Tensor, name: str) -> int:
    if value.ndim != 3 or value.shape[1] != 1:
        raise ValueError(f"{name} must have shape [batch,1,channels]")
    if value.shape[-1] == 0:
        raise ValueError(f"{name} channels must be nonzero")
    return value.shape[-1]


def _check_norm_parameters(
    value: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor
) -> None:
    channels = value.shape[-1]
    if weight.shape != (channels,) or bias.shape != (channels,):
        raise ValueError("LayerNorm weight and bias must have shape [channels]")
    if weight.device != value.device or bias.device != value.device:
        raise ValueError("LayerNorm parameters and activations must share one device")


def _use_triton(value: torch.Tensor) -> bool:
    return (
        _HAVE_TRITON
        and value.is_cuda
        and value.dtype in (torch.float16, torch.bfloat16)
        # A single Triton program normalizes one row.  Larger widths need a
        # two-pass reduction and intentionally use the functional fallback.
        and value.shape[-1] <= 16384
    )


if _HAVE_TRITON:
    @triton.jit
    def _ln_six_mix_kernel(
        x_ptr, previous_ptr, mix_ptr, weight_ptr, bias_ptr,
        mixed_ptr, shift_ptr,
        C: tl.constexpr, BLOCK_C: tl.constexpr, EPS: tl.constexpr,
    ):
        row = tl.program_id(0)
        offsets = tl.arange(0, BLOCK_C)
        mask = offsets < C
        base = row * C + offsets
        x = tl.load(x_ptr + base, mask=mask, other=0.0).to(tl.float32)
        mean = tl.sum(x, axis=0) / C
        centered = x - mean
        variance = tl.sum(centered * centered, axis=0) / C
        norm = centered * tl.rsqrt(variance + EPS)
        norm = norm * tl.load(weight_ptr + offsets, mask=mask, other=0.0)
        norm += tl.load(bias_ptr + offsets, mask=mask, other=0.0)
        previous = tl.load(previous_ptr + base, mask=mask, other=0.0).to(tl.float32)
        delta = previous - norm
        for branch in tl.static_range(6):
            coefficient = tl.load(
                mix_ptr + branch * C + offsets, mask=mask, other=0.0
            ).to(tl.float32)
            tl.store(
                mixed_ptr + (row * 6 + branch) * C + offsets,
                norm + delta * coefficient,
                mask=mask,
            )
        tl.store(shift_ptr + base, norm, mask=mask)

    @torch.library.triton_op("rwkv_lab::ln_six_mix", mutates_args={})
    def _ln_six_mix_op(
        x: torch.Tensor,
        previous: torch.Tensor,
        mix: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        eps: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, _, channels = x.shape
        mixed = torch.empty(
            (batch, 1, 6, channels), device=x.device, dtype=x.dtype
        )
        shift = torch.empty_like(x)
        torch.library.wrap_triton(_ln_six_mix_kernel)[(batch,)](
            x, previous, mix, weight, bias, mixed, shift,
            C=channels,
            BLOCK_C=triton.next_power_of_2(channels),
            EPS=float(eps),
            num_warps=8 if channels >= 2048 else 4,
        )
        return mixed, shift

    @triton.jit
    def _add_ln_channel_mix_kernel(
        x_ptr, update_ptr, previous_ptr, mix_ptr, weight_ptr, bias_ptr,
        residual_ptr, mixed_ptr, shift_ptr,
        C: tl.constexpr, BLOCK_C: tl.constexpr, EPS: tl.constexpr,
    ):
        row = tl.program_id(0)
        offsets = tl.arange(0, BLOCK_C)
        mask = offsets < C
        base = row * C + offsets
        # Keeping the add in fp32 is the fused operator's documented
        # semantics and avoids a low-precision residual round trip.
        residual = (
            tl.load(x_ptr + base, mask=mask, other=0.0).to(tl.float32)
            + tl.load(update_ptr + base, mask=mask, other=0.0).to(tl.float32)
        )
        mean = tl.sum(residual, axis=0) / C
        centered = residual - mean
        variance = tl.sum(centered * centered, axis=0) / C
        norm = centered * tl.rsqrt(variance + EPS)
        norm = norm * tl.load(weight_ptr + offsets, mask=mask, other=0.0)
        norm += tl.load(bias_ptr + offsets, mask=mask, other=0.0)
        previous = tl.load(previous_ptr + base, mask=mask, other=0.0).to(tl.float32)
        coefficient = tl.load(mix_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        tl.store(residual_ptr + base, residual, mask=mask)
        tl.store(mixed_ptr + base, norm + (previous - norm) * coefficient, mask=mask)
        tl.store(shift_ptr + base, norm, mask=mask)

    @torch.library.triton_op("rwkv_lab::add_ln_channel_mix", mutates_args={})
    def _add_ln_channel_mix_op(
        x: torch.Tensor,
        update: torch.Tensor,
        previous: torch.Tensor,
        mix: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        eps: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, _, channels = x.shape
        residual = torch.empty_like(x)
        mixed = torch.empty_like(x)
        shift = torch.empty_like(x)
        torch.library.wrap_triton(_add_ln_channel_mix_kernel)[(batch,)](
            x, update, previous, mix, weight, bias,
            residual, mixed, shift,
            C=channels,
            BLOCK_C=triton.next_power_of_2(channels),
            EPS=float(eps),
            num_warps=8 if channels >= 2048 else 4,
        )
        return residual, mixed, shift

    @triton.jit
    def _add_ln_kernel(
        x_ptr, update_ptr, weight_ptr, bias_ptr, residual_ptr, norm_ptr,
        C: tl.constexpr, BLOCK_C: tl.constexpr, EPS: tl.constexpr,
    ):
        row = tl.program_id(0)
        offsets = tl.arange(0, BLOCK_C)
        mask = offsets < C
        base = row * C + offsets
        residual = (
            tl.load(x_ptr + base, mask=mask, other=0.0).to(tl.float32)
            + tl.load(update_ptr + base, mask=mask, other=0.0).to(tl.float32)
        )
        mean = tl.sum(residual, axis=0) / C
        centered = residual - mean
        variance = tl.sum(centered * centered, axis=0) / C
        norm = centered * tl.rsqrt(variance + EPS)
        norm = norm * tl.load(weight_ptr + offsets, mask=mask, other=0.0)
        norm += tl.load(bias_ptr + offsets, mask=mask, other=0.0)
        tl.store(residual_ptr + base, residual, mask=mask)
        tl.store(norm_ptr + base, norm, mask=mask)

    @torch.library.triton_op("rwkv_lab::add_layer_norm", mutates_args={})
    def _add_ln_op(
        x: torch.Tensor,
        update: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        eps: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, _, channels = x.shape
        residual = torch.empty_like(x)
        norm = torch.empty_like(x)
        torch.library.wrap_triton(_add_ln_kernel)[(batch,)](
            x, update, weight, bias, residual, norm,
            C=channels,
            BLOCK_C=triton.next_power_of_2(channels),
            EPS=float(eps),
            num_warps=8 if channels >= 2048 else 4,
        )
        return residual, norm


def layer_norm_six_mix(
    x: torch.Tensor,
    previous: torch.Tensor,
    mix: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    *,
    eps: float = 1e-5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """LayerNorm and create the six RWKV TimeMix branches in one pass.

    ``x`` and ``previous`` are ``[B,1,C]`` normalized-input tokens and
    ``mix`` is ``[6,C]`` ordered by the caller (normally r/w/k/v/a/g).  The
    returned pair is ``(mixed[B,1,6,C], new_shift[B,1,C])``.

    Inspired by Albatross's fused ``ln_mix6`` boundary, while retaining a
    portable contiguous layout and runtime geometry:
    https://github.com/BlinkDL/Albatross/tree/main/faster3b_2606
    """
    channels = _check_token(x, "x")
    if previous.shape != x.shape:
        raise ValueError("previous must match x shape")
    if mix.shape != (6, channels):
        raise ValueError("TimeMix coefficients must have shape [6,channels]")
    _check_norm_parameters(x, weight, bias)
    if any(t.device != x.device for t in (previous, mix)):
        raise ValueError("TimeMix tensors must share one device")
    tensors = tuple(t.contiguous() for t in (x, previous, mix, weight, bias))
    if _use_triton(x):
        return _ln_six_mix_op(*tensors, float(eps))
    normalized = _layer_norm_fp32(x, weight, bias, eps)
    mixed = normalized.unsqueeze(2).float() + (
        previous.unsqueeze(2).float() - normalized.unsqueeze(2).float()
    ) * mix.float().view(1, 1, 6, channels)
    return mixed.to(x.dtype), normalized


def residual_add_layer_norm_channel_mix(
    x: torch.Tensor,
    update: torch.Tensor,
    previous: torch.Tensor,
    mix: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    *,
    eps: float = 1e-5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fuse attention residual add, LayerNorm, and ChannelMix token shift.

    Returns ``(residual, mixed, new_shift)``.  The residual add and LayerNorm
    accumulate in fp32 before their outputs are converted back to ``x.dtype``.
    This portable boundary follows Albatross's ``add_ln_cmix_mix`` idea:
    https://github.com/BlinkDL/Albatross/tree/main/faster3b_2606
    """
    channels = _check_token(x, "x")
    if update.shape != x.shape or previous.shape != x.shape:
        raise ValueError("update and previous must match x shape")
    if mix.shape != (channels,):
        raise ValueError("ChannelMix coefficients must have shape [channels]")
    _check_norm_parameters(x, weight, bias)
    if any(t.device != x.device for t in (update, previous, mix)):
        raise ValueError("ChannelMix tensors must share one device")
    tensors = tuple(
        t.contiguous() for t in (x, update, previous, mix, weight, bias)
    )
    if _use_triton(x):
        return _add_ln_channel_mix_op(*tensors, float(eps))
    residual_fp32 = x.float() + update.float()
    residual = residual_fp32.to(x.dtype)
    normalized = _layer_norm_fp32(residual_fp32, weight, bias, eps).to(x.dtype)
    mixed = normalized.float() + (
        previous.float() - normalized.float()
    ) * mix.float().view(1, 1, channels)
    return residual, mixed.to(x.dtype), normalized


def residual_add_layer_norm(
    x: torch.Tensor,
    update: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    *,
    eps: float = 1e-5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fuse a one-token residual addition and LayerNorm in fp32."""
    _check_token(x, "x")
    if update.shape != x.shape:
        raise ValueError("update must match x shape")
    _check_norm_parameters(x, weight, bias)
    if update.device != x.device:
        raise ValueError("residual tensors must share one device")
    tensors = tuple(t.contiguous() for t in (x, update, weight, bias))
    if _use_triton(x):
        return _add_ln_op(*tensors, float(eps))
    residual_fp32 = x.float() + update.float()
    return (
        residual_fp32.to(x.dtype),
        _layer_norm_fp32(residual_fp32, weight, bias, eps).to(x.dtype),
    )


__all__ = [
    "layer_norm_six_mix",
    "residual_add_layer_norm",
    "residual_add_layer_norm_channel_mix",
]
