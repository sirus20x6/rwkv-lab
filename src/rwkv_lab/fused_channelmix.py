"""Training fast path for RWKV ChannelMix.

The autograd function owns the complete
``linear -> relu² -> linear`` block and saves exactly the tensors needed by its
BF16 backward. Optionally, the up-projection uses a weight copy quantized once
after each optimizer step; unsupported devices transparently use BF16.
"""
from __future__ import annotations

import time

import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
    from triton.tools.tensor_descriptor import TensorDescriptor

    _HAS_TRITON = True
except Exception:  # pragma: no cover - optional CUDA dependency
    triton = tl = TensorDescriptor = None
    _HAS_TRITON = False


if _HAS_TRITON:
    @triton.jit
    def _linear_relu_square_kernel(
        a_desc,
        b_desc,
        pre_desc,
        post_desc,
        dequant_scale,
        M,
        N,
        K,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        NUM_SMS: tl.constexpr,
        USE_FP8: tl.constexpr,
    ):
        start = tl.program_id(0)
        tiles_m = tl.cdiv(M, BLOCK_M)
        tiles_n = tl.cdiv(N, BLOCK_N)
        tile_count = tiles_m * tiles_n
        for tile in tl.range(start, tile_count, NUM_SMS, flatten=True):
            pid_m = tile // tiles_n
            pid_n = tile % tiles_n
            offset_m = pid_m * BLOCK_M
            offset_n = pid_n * BLOCK_N
            acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            for k_tile in range(tl.cdiv(K, BLOCK_K)):
                offset_k = k_tile * BLOCK_K
                a = a_desc.load([offset_m, offset_k])
                b = b_desc.load([offset_n, offset_k])
                acc = tl.dot(a, b.T, acc)
            if USE_FP8:
                acc *= tl.load(dequant_scale)
            value = acc.to(tl.bfloat16)
            pre_desc.store([offset_m, offset_n], value)
            activated = tl.maximum(value, 0.0)
            post_desc.store([offset_m, offset_n], activated * activated)


_DUMMY_SCALE: dict[torch.device, torch.Tensor] = {}


def _triton_linear_relu_square(
    x: torch.Tensor,
    weight: torch.Tensor,
    *,
    x_fp8: torch.Tensor | None = None,
    weight_fp8: torch.Tensor | None = None,
    dequant_scale: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Fused projection and ReLU²; ``None`` means use the portable fallback."""
    use_fp8 = x_fp8 is not None and weight_fp8 is not None
    # The FP8 variant reduces over 128-element K tiles, so the contraction
    # dimension must divide the block size actually being used. Checking a fixed
    # 64 here let a K of, say, 192 reach TensorDescriptor with a mismatched box.
    block_k = 128 if use_fp8 else 64
    if (
        not _HAS_TRITON
        or not x.is_cuda
        or x.ndim != 2
        or weight.ndim != 2
        or x.dtype != torch.bfloat16
        or weight.dtype != torch.bfloat16
        or x.shape[1] != weight.shape[1]
        or x.shape[0] % 128
        or weight.shape[0] % 128
        or x.shape[1] % block_k
        or (use_fp8 and (x_fp8.shape != x.shape
                         or weight_fp8.shape != weight.shape))
    ):
        return None
    M, K = x.shape
    N = weight.shape[0]
    a_value = x_fp8 if use_fp8 else x
    b_value = weight_fp8 if use_fp8 else weight
    pre = torch.empty(M, N, device=x.device, dtype=x.dtype)
    post = torch.empty_like(pre)
    a_desc = TensorDescriptor.from_tensor(a_value, [128, block_k])
    b_desc = TensorDescriptor.from_tensor(b_value, [128, block_k])
    pre_desc = TensorDescriptor.from_tensor(pre, [128, 128])
    post_desc = TensorDescriptor.from_tensor(post, [128, 128])
    sms = torch.cuda.get_device_properties(x.device).multi_processor_count
    if dequant_scale is None:
        dequant_scale = _DUMMY_SCALE.setdefault(
            x.device, torch.ones(1, device=x.device, dtype=torch.float32)
        )
    grid = (min(sms, triton.cdiv(M, 128) * triton.cdiv(N, 128)),)
    _linear_relu_square_kernel[grid](
        a_desc,
        b_desc,
        pre_desc,
        post_desc,
        dequant_scale,
        M,
        N,
        K,
        BLOCK_M=128,
        BLOCK_N=128,
        BLOCK_K=block_k,
        NUM_SMS=sms,
        USE_FP8=use_fp8,
        num_stages=2,
        num_warps=8,
    )
    return pre, post


def can_scaled_mm(value: torch.Tensor) -> bool:
    """Whether the private scaled GEMM primitive is usable on this tensor."""
    return (
        value.is_cuda
        and hasattr(torch, "_scaled_mm")
        and torch.cuda.get_device_capability(value.device)[0] >= 9
        and value.shape[-1] % 16 == 0
    )


@torch.no_grad()
def quantize_weight_fp8(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Tensorwise E4M3 weight copy and dequantization scale."""
    maximum = torch.finfo(torch.float8_e4m3fn).max
    scale = weight.detach().float().abs().amax().clamp_min(1e-12) / maximum
    quantized = (weight.detach() / scale).to(torch.float8_e4m3fn)
    return quantized, scale.reshape(1)


def _scaled_fp8_up(
    x: torch.Tensor,
    weight_fp8: torch.Tensor,
    weight_scale: torch.Tensor,
) -> torch.Tensor:
    maximum = torch.finfo(torch.float8_e4m3fn).max
    input_scale = x.detach().float().abs().amax().clamp_min(1e-12) / maximum
    x_fp8 = (x / input_scale).to(torch.float8_e4m3fn)
    # _scaled_mm expects the logical [K, N] right operand. Keeping it contiguous
    # also avoids a per-call materialization hidden behind a transposed view.
    right = weight_fp8.transpose(0, 1).contiguous()
    return torch._scaled_mm(
        x_fp8,
        right,
        scale_a=input_scale.reshape(1),
        scale_b=weight_scale,
        out_dtype=x.dtype,
        use_fast_accum=True,
    )


class _FusedChannelMix(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, x, key_weight, value_weight, key_fp8, key_scale, use_fp8, use_triton
    ):
        shape = x.shape
        flat = x.reshape(-1, shape[-1])
        fp8_active = bool(use_fp8) and can_scaled_mm(flat)
        fused = None
        if fp8_active and use_triton:
            maximum = torch.finfo(torch.float8_e4m3fn).max
            input_scale = (
                flat.detach().float().abs().amax().clamp_min(1e-12) / maximum
            )
            flat_fp8 = (flat / input_scale).to(torch.float8_e4m3fn)
            fused = _triton_linear_relu_square(
                flat,
                key_weight,
                x_fp8=flat_fp8,
                weight_fp8=key_fp8,
                dequant_scale=(input_scale * key_scale),
            )
        if fused is None and use_triton:
            fused = _triton_linear_relu_square(flat, key_weight)
        if fused is None:
            pre = (
                _scaled_fp8_up(flat, key_fp8, key_scale)
                if fp8_active else F.linear(flat, key_weight)
            )
            post = F.relu(pre).square()
        else:
            pre, post = fused
        output = F.linear(post, value_weight)
        # Save only ``pre``: ``post`` is one elementwise recomputation away, and
        # retaining both would make this block cost one more [tokens, ffn_hidden]
        # activation than the eager path it replaces (eager keeps `mixed` and the
        # ReLU output). Recomputing from ``pre`` — rather than from ``post`` via
        # sqrt — also keeps the backward bitwise-equal to the previous version.
        ctx.save_for_backward(flat, key_weight, value_weight, pre)
        ctx.input_shape = shape
        return output.reshape(*shape[:-1], value_weight.shape[0])

    @staticmethod
    def backward(ctx, grad_output):
        flat, key_weight, value_weight, pre = ctx.saved_tensors
        activated = F.relu(pre)
        post = activated.square()
        grad = grad_output.reshape(-1, grad_output.shape[-1])
        grad_value = grad.transpose(0, 1) @ post
        grad_post = grad @ value_weight
        grad_pre = grad_post * (2.0 * activated)
        grad_key = grad_pre.transpose(0, 1) @ flat
        grad_input = grad_pre @ key_weight
        return (
            grad_input.reshape(ctx.input_shape),
            grad_key,
            grad_value,
            None,
            None,
            None,
            None,
        )


def fused_channel_mix(
    x: torch.Tensor,
    key_weight: torch.Tensor,
    value_weight: torch.Tensor,
    *,
    key_fp8: torch.Tensor | None = None,
    key_scale: torch.Tensor | None = None,
    use_triton: bool = False,
) -> torch.Tensor:
    """Execute the fused training block, using cached FP8 when supplied."""
    use_fp8 = key_fp8 is not None and key_scale is not None
    if not use_fp8:
        key_fp8 = x.new_empty(0)
        key_scale = x.new_ones(1, dtype=torch.float32)
    return _FusedChannelMix.apply(
        x, key_weight, value_weight, key_fp8, key_scale, use_fp8, use_triton
    )


def qualify_channelmix_training(
    module,
    sample: torch.Tensor,
    *,
    allow_cached_fp8: bool,
    repeats: int = 8,
    min_speedup: float = 1.01,
) -> dict:
    """Select the fastest parity-compatible ChannelMix training backend."""
    if not sample.is_cuda:
        module._fused_training = True
        module._cached_fp8_up = False
        module._triton_fused_training = False
        return {"adopted": "fused_bf16", "reason": "CPU portable path"}
    gradient = torch.randn_like(sample)

    def configure(*, fused: bool, cached: bool, triton_fused: bool) -> None:
        if cached and not hasattr(module, "_key_weight_fp8"):
            module.enable_fused_training(
                cached_fp8_up=True, triton_fused=triton_fused
            )
        module._fused_training = fused
        module._cached_fp8_up = cached
        module._triton_fused_training = triton_fused
        if cached:
            module.refresh_fp8_cache()

    def run():
        module.zero_grad(set_to_none=True)
        input_value = sample.detach().clone().requires_grad_(True)
        value = module(input_value)
        value.backward(gradient)
        return (
            value.detach(),
            input_value.grad.detach(),
            *(parameter.grad.detach() for parameter in module.parameters()),
        )

    def measure(*, fused: bool, cached: bool, triton_fused: bool) -> float:
        configure(fused=fused, cached=cached, triton_fused=triton_fused)
        for _ in range(3):
            run()
        torch.cuda.synchronize(sample.device)
        start = time.perf_counter()
        for _ in range(repeats):
            run()
        torch.cuda.synchronize(sample.device)
        return 1000.0 * (time.perf_counter() - start) / repeats

    candidates = {
        "eager": (False, False, False),
        "fused_bf16": (True, False, False),
        "triton_bf16": (True, False, True),
    }
    if allow_cached_fp8:
        candidates.update({
            "cached_fp8": (True, True, False),
            "triton_cached_fp8": (True, True, True),
        })

    configure(fused=False, cached=False, triton_fused=False)
    reference = run()
    rejected = {}
    compatible = {}
    for name, flags in candidates.items():
        try:
            configure(
                fused=flags[0], cached=flags[1], triton_fused=flags[2]
            )
            actual = run()
            relative_errors = [
                (candidate.float() - expected.float()).norm()
                / expected.float().norm().clamp_min(1e-12)
                for candidate, expected in zip(actual, reference)
            ]
            max_error = max(float(error) for error in relative_errors)
            tolerance = 0.08 if flags[1] else 0.015
            if all(torch.isfinite(value).all() for value in actual) and max_error <= tolerance:
                compatible[name] = flags
            else:
                rejected[name] = f"relative_error={max_error:.6g}"
        except Exception as exc:
            rejected[name] = f"{type(exc).__name__}: {exc}"

    # The eager reference must always remain available as the fail-closed path.
    compatible["eager"] = candidates["eager"]
    times = {
        name: measure(fused=fused, cached=cached, triton_fused=triton_fused)
        for name, (fused, cached, triton_fused) in compatible.items()
    }
    baseline = times["eager"]
    fastest = min(times, key=times.get)
    adopted = (
        fastest
        if fastest != "eager" and baseline / times[fastest] >= min_speedup
        else "eager"
    )
    fused, cached, triton_fused = candidates[adopted]
    module._fused_training = fused
    module._cached_fp8_up = cached
    module._triton_fused_training = triton_fused
    if cached:
        module.refresh_fp8_cache()
    module.zero_grad(set_to_none=True)
    return {
        "adopted": adopted,
        "times_ms": times,
        "speedup": baseline / times[adopted],
        "rejected": rejected,
    }


__all__ = [
    "can_scaled_mm",
    "fused_channel_mix",
    "qualify_channelmix_training",
    "quantize_weight_fp8",
]
