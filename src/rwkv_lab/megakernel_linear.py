"""Parity- and performance-gated B1T1 projection kernel candidates.

The row-one GEMV, packed R/K/V projection, and sparse-FFN directions are
informed by Albatross ``faster3b_2606``:
https://github.com/BlinkDL/Albatross/tree/main/faster3b_2606

Albatross's layouts are tuned for particular GPUs and model geometry.  This
module deliberately does not copy them: candidates are shape-specialized,
Triton-autotuned on the local GPU, and remain behind ``qualify_b1t1_kernels``.
Portable eager PyTorch is the fail-closed fallback.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import time
from typing import Any, Callable

import torch
import torch.nn.functional as F

try:  # Triton is supplied by the CUDA PyTorch environment.
    import triton
    import triton.language as tl

    _HAVE_TRITON = hasattr(torch.library, "triton_op")
    _TRITON_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover - depends on the CUDA stack
    triton = tl = None
    _HAVE_TRITON = False
    _TRITON_ERROR = exc


@dataclass(frozen=True)
class PreparedRowOneWeight:
    """An explicitly prepared weight plus metadata needed to invalidate tuning.

    ``out_in`` is ordinary ``nn.Linear`` layout. ``in_out`` is a contiguous
    transpose candidate. Preparing a layout is optional and never mutates the
    source parameter.
    """

    tensor: torch.Tensor
    layout: str
    out_features: int
    in_features: int
    source_signature: str


def prepare_row_one_weight(
    weight: torch.Tensor, *, layout: str = "out_in"
) -> PreparedRowOneWeight:
    if weight.ndim != 2:
        raise ValueError("row-one weight must have shape [out_features,in_features]")
    if layout not in {"out_in", "in_out"}:
        raise ValueError("prepared row-one layout must be 'out_in' or 'in_out'")
    out_features, in_features = weight.shape
    prepared = weight.detach().contiguous()
    if layout == "in_out":
        prepared = prepared.t().contiguous()
    signature = hashlib.sha256(json.dumps({
        "shape": list(weight.shape),
        "dtype": str(weight.dtype),
        "device": str(weight.device),
        "layout": layout,
        "stride": list(prepared.stride()),
    }, sort_keys=True).encode()).hexdigest()
    return PreparedRowOneWeight(
        prepared, layout, out_features, in_features, signature)


def _cuda_candidate_available(tensor: torch.Tensor) -> bool:
    return bool(_HAVE_TRITON and tensor.is_cuda and not torch.is_grad_enabled())


def _empty_bias(x: torch.Tensor) -> torch.Tensor:
    return torch.empty(0, device=x.device, dtype=x.dtype)


if _HAVE_TRITON:
    _GEMV_CONFIGS = [
        triton.Config({"BLOCK_N": 16}, num_warps=4),
        triton.Config({"BLOCK_N": 32}, num_warps=4),
        triton.Config({"BLOCK_N": 64}, num_warps=8),
        triton.Config({"BLOCK_N": 128}, num_warps=8),
    ]

    @triton.autotune(configs=_GEMV_CONFIGS, key=["K", "N"], cache_results=True)
    @triton.jit
    def _row_one_gemv_kernel(
        x_ptr, weight_ptr, bias_ptr, out_ptr,
        K: tl.constexpr, N: tl.constexpr, BLOCK_K: tl.constexpr,
        BLOCK_N: tl.constexpr, TRANSPOSED: tl.constexpr,
        HAS_BIAS: tl.constexpr,
    ):
        batch = tl.program_id(0)
        block_n = tl.program_id(1)
        offs_n = block_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)
        mask_n = offs_n < N
        mask_k = offs_k < K
        x = tl.load(x_ptr + batch * K + offs_k, mask=mask_k, other=0.0).to(tl.float32)
        if TRANSPOSED:
            offsets = offs_k[:, None] * N + offs_n[None, :]
        else:
            offsets = offs_n[None, :] * K + offs_k[:, None]
        weight = tl.load(
            weight_ptr + offsets,
            mask=mask_k[:, None] & mask_n[None, :], other=0.0,
        ).to(tl.float32)
        result = tl.sum(x[:, None] * weight, axis=0)
        if HAS_BIAS:
            result += tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)
        tl.store(out_ptr + batch * N + offs_n,
                 result.to(out_ptr.dtype.element_ty), mask=mask_n)

    def _launch_gemv(
        x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, *,
        n: int, transposed: bool,
    ) -> torch.Tensor:
        batch, _, k = x.shape
        output = torch.empty((batch, 1, n), device=x.device, dtype=x.dtype)
        def grid(meta):
            return batch, triton.cdiv(n, meta["BLOCK_N"])
        torch.library.wrap_triton(_row_one_gemv_kernel)[grid](
            x, weight, bias, output, K=k, N=n,
            BLOCK_K=triton.next_power_of_2(k), TRANSPOSED=transposed,
            HAS_BIAS=bias.numel() != 0,
        )
        return output

    @torch.library.triton_op("rwkv_lab::row_one_linear", mutates_args={})
    def _row_one_linear_op(
        x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor,
    ) -> torch.Tensor:
        return _launch_gemv(x, weight, bias, n=weight.shape[0], transposed=False)

    @torch.library.triton_op("rwkv_lab::row_one_linear_transposed", mutates_args={})
    def _row_one_linear_transposed_op(
        x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor,
    ) -> torch.Tensor:
        return _launch_gemv(x, weight, bias, n=weight.shape[1], transposed=True)

    @torch.library.triton_op("rwkv_lab::packed_rkv", mutates_args={})
    def _packed_rkv_op(
        x: torch.Tensor, weights: torch.Tensor, biases: torch.Tensor,
    ) -> torch.Tensor:
        # One compiler-visible op and one kernel. Real RWKV TimeMix supplies a
        # distinct mixed activation for R, K, and V as [B,1,3,K]; the simpler
        # [B,1,K] form intentionally broadcasts one input to all projections.
        batch, k = x.shape[0], x.shape[-1]
        projections, n, _ = weights.shape
        output = torch.empty(
            (batch, 1, projections, n), device=x.device, dtype=x.dtype)
        def grid(meta):
            return batch, projections, triton.cdiv(n, meta["BLOCK_N"])
        torch.library.wrap_triton(_packed_rkv_kernel)[grid](
            x, weights, biases, output, K=k, N=n,
            BLOCK_K=triton.next_power_of_2(k), DISTINCT_INPUT=x.ndim == 4,
            HAS_BIAS=biases.numel() != 0,
        )
        return output

    @triton.autotune(configs=_GEMV_CONFIGS, key=["K", "N"], cache_results=True)
    @triton.jit
    def _packed_rkv_kernel(
        x_ptr, weight_ptr, bias_ptr, out_ptr,
        K: tl.constexpr, N: tl.constexpr, BLOCK_K: tl.constexpr,
        BLOCK_N: tl.constexpr, DISTINCT_INPUT: tl.constexpr,
        HAS_BIAS: tl.constexpr,
    ):
        batch = tl.program_id(0)
        projection = tl.program_id(1)
        block_n = tl.program_id(2)
        offs_n = block_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)
        mask_n = offs_n < N
        mask_k = offs_k < K
        input_row = batch * K
        if DISTINCT_INPUT:
            input_row = (batch * 3 + projection) * K
        mixed = tl.load(x_ptr + input_row + offs_k, mask=mask_k, other=0.0).to(tl.float32)
        weight_offsets = (
            projection * N * K
            + offs_n[None, :] * K + offs_k[:, None]
        )
        weight = tl.load(
            weight_ptr + weight_offsets,
            mask=mask_n[None, :] & mask_k[:, None], other=0.0,
        ).to(tl.float32)
        result = tl.sum(mixed[:, None] * weight, axis=0)
        if HAS_BIAS:
            result += tl.load(
                bias_ptr + projection * N + offs_n, mask=mask_n, other=0.0
            ).to(tl.float32)
        output_row = (batch * 3 + projection) * N
        tl.store(out_ptr + output_row + offs_n,
                 result.to(out_ptr.dtype.element_ty), mask=mask_n)

    @triton.autotune(configs=_GEMV_CONFIGS, key=["K", "N"], cache_results=True)
    @triton.jit
    def _sqrelu_value_kernel(
        hidden_ptr, weight_ptr, bias_ptr, out_ptr,
        K: tl.constexpr, N: tl.constexpr, BLOCK_K: tl.constexpr,
        BLOCK_N: tl.constexpr, HAS_BIAS: tl.constexpr,
    ):
        batch = tl.program_id(0)
        block_n = tl.program_id(1)
        offs_n = block_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)
        mask_n = offs_n < N
        mask_k = offs_k < K
        hidden = tl.load(
            hidden_ptr + batch * K + offs_k, mask=mask_k, other=0.0).to(tl.float32)
        hidden = tl.maximum(hidden, 0.0)
        hidden *= hidden
        weight = tl.load(
            weight_ptr + offs_n[None, :] * K + offs_k[:, None],
            mask=mask_n[None, :] & mask_k[:, None], other=0.0,
        ).to(tl.float32)
        result = tl.sum(hidden[:, None] * weight, axis=0)
        if HAS_BIAS:
            result += tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)
        tl.store(out_ptr + batch * N + offs_n,
                 result.to(out_ptr.dtype.element_ty), mask=mask_n)

    @torch.library.triton_op("rwkv_lab::ffn_sqrelu_value", mutates_args={})
    def _ffn_sqrelu_value_op(
        x: torch.Tensor, key_weight: torch.Tensor, value_weight: torch.Tensor,
        key_bias: torch.Tensor, value_bias: torch.Tensor,
    ) -> torch.Tensor:
        # The activation is fused into the value projection. The key
        # preactivation remains an intermediate, avoiding the numerically risky
        # zero-skipping approximation used by some sparse FFN experiments.
        hidden = _launch_gemv(
            x, key_weight, key_bias, n=key_weight.shape[0], transposed=False)
        batch, _, k = hidden.shape
        n = value_weight.shape[0]
        output = torch.empty((batch, 1, n), device=x.device, dtype=x.dtype)
        def grid(meta):
            return batch, triton.cdiv(n, meta["BLOCK_N"])
        torch.library.wrap_triton(_sqrelu_value_kernel)[grid](
            hidden, value_weight, value_bias, output,
            K=k, N=n, BLOCK_K=triton.next_power_of_2(k),
            HAS_BIAS=value_bias.numel() != 0,
        )
        return output


def _validate_b1t1(x: torch.Tensor) -> None:
    if x.ndim != 3 or x.shape[1] != 1:
        raise ValueError("row-one candidates require input shape [batch,1,features]")


def row_one_linear(
    x: torch.Tensor,
    weight: torch.Tensor | PreparedRowOneWeight,
    bias: torch.Tensor | None = None,
    *,
    use_candidate: bool = False,
) -> torch.Tensor:
    """Apply a B1T1 linear, using Triton only after external qualification."""
    _validate_b1t1(x)
    prepared = weight if isinstance(weight, PreparedRowOneWeight) else None
    raw = prepared.tensor if prepared is not None else weight
    layout = prepared.layout if prepared is not None else "out_in"
    out_features = prepared.out_features if prepared is not None else raw.shape[0]
    in_features = prepared.in_features if prepared is not None else raw.shape[1]
    if x.shape[-1] != in_features:
        raise ValueError("row-one input and weight feature dimensions differ")
    if bias is not None and bias.shape != (out_features,):
        raise ValueError("row-one bias must match out_features")
    if use_candidate and _cuda_candidate_available(x):
        x = x if x.is_contiguous() else x.contiguous()
        candidate_bias = bias if bias is not None else _empty_bias(x)
        if layout == "in_out":
            return _row_one_linear_transposed_op(x, raw, candidate_bias)
        return _row_one_linear_op(x, raw, candidate_bias)
    eager_weight = raw.t() if layout == "in_out" else raw
    return F.linear(x, eager_weight, bias)


def packed_rkv_projection(
    x: torch.Tensor,
    weights: torch.Tensor,
    biases: torch.Tensor | None = None,
    *,
    use_candidate: bool = False,
) -> torch.Tensor:
    """Project packed R/K/V weights, returning ``[batch,1,3,out]``.

    ``x`` may be one shared ``[batch,1,in]`` activation or the real RWKV
    TimeMix form ``[batch,1,3,in]`` containing distinct R/K/V mixtures.
    """
    distinct_input = x.ndim == 4
    if distinct_input:
        if x.shape[1] != 1 or x.shape[2] != 3:
            raise ValueError(
                "distinct packed R/K/V input must have shape [batch,1,3,in]")
    else:
        _validate_b1t1(x)
    if weights.ndim != 3 or weights.shape[0] != 3:
        raise ValueError("packed R/K/V weights must have shape [3,out,in]")
    if weights.shape[-1] != x.shape[-1]:
        raise ValueError("packed R/K/V input and weight feature dimensions differ")
    if biases is not None and biases.shape != weights.shape[:2]:
        raise ValueError("packed R/K/V bias must have shape [3,out]")
    tensors = (weights,) if biases is None else (weights, biases)
    if any(t.device != x.device or t.dtype != x.dtype for t in tensors):
        raise ValueError("packed R/K/V inputs, weights, and bias must share device/dtype")
    if use_candidate and _cuda_candidate_available(x):
        x = x if x.is_contiguous() else x.contiguous()
        return _packed_rkv_op(
            x, weights.contiguous(),
            biases.contiguous() if biases is not None else _empty_bias(x),
        )
    if distinct_input:
        # Each projection consumes its own TimeMix activation. einsum is the
        # direct batched oracle and preserves the [R,K,V] correspondence.
        result = torch.einsum("btrk,rok->btro", x, weights)
        if biases is not None:
            result = result + biases.view(1, 1, 3, -1)
        return result
    # A single flattened cuBLAS/Inductor projection is the honest shared-input
    # baseline for a packed kernel, rather than three Python-dispatched GEMMs.
    projections, out_features, in_features = weights.shape
    flat_bias = biases.reshape(-1) if biases is not None else None
    return F.linear(
        x, weights.reshape(projections * out_features, in_features), flat_bias
    ).view(x.shape[0], 1, projections, out_features)


def ffn_squared_relu_value(
    x: torch.Tensor,
    key_weight: torch.Tensor,
    value_weight: torch.Tensor,
    key_bias: torch.Tensor | None = None,
    value_bias: torch.Tensor | None = None,
    *,
    use_candidate: bool = False,
) -> torch.Tensor:
    """Exact squared-ReLU FFN candidate; no activation sparsity approximation."""
    _validate_b1t1(x)
    if key_weight.ndim != 2 or value_weight.ndim != 2:
        raise ValueError("FFN weights must be matrices")
    if key_weight.shape[1] != x.shape[-1] or value_weight.shape[1] != key_weight.shape[0]:
        raise ValueError("FFN key/value projection geometry mismatch")
    if use_candidate and _cuda_candidate_available(x):
        x = x if x.is_contiguous() else x.contiguous()
        return _ffn_sqrelu_value_op(
            x, key_weight.contiguous(), value_weight.contiguous(),
            key_bias.contiguous() if key_bias is not None else _empty_bias(x),
            value_bias.contiguous() if value_bias is not None else _empty_bias(x),
        )
    hidden = torch.relu(F.linear(x, key_weight, key_bias)).square()
    return F.linear(hidden, value_weight, value_bias)


def _median_cuda_us(fn: Callable[[], torch.Tensor], *, warmup: int, repeats: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples: list[float] = []
    for _ in range(repeats):
        start, end = torch.cuda.Event(True), torch.cuda.Event(True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        samples.append(float(start.elapsed_time(end)) * 1000.0)
    return sorted(samples)[len(samples) // 2]


def _runtime_metadata(x: torch.Tensor) -> dict[str, Any]:
    if not x.is_cuda:
        return {"device": str(x.device), "torch": torch.__version__, "triton": None}
    props = torch.cuda.get_device_properties(x.device)
    return {
        "device": str(x.device),
        "device_name": props.name,
        "compute_capability": list(torch.cuda.get_device_capability(x.device)),
        "torch": torch.__version__,
        "triton": getattr(triton, "__version__", None),
    }


@torch.no_grad()
def qualify_b1t1_kernels(
    x: torch.Tensor,
    row_weight: torch.Tensor,
    rkv_weights: torch.Tensor,
    key_weight: torch.Tensor,
    value_weight: torch.Tensor,
    *,
    rkv_x: torch.Tensor | None = None,
    row_bias: torch.Tensor | None = None,
    rkv_biases: torch.Tensor | None = None,
    key_bias: torch.Tensor | None = None,
    value_bias: torch.Tensor | None = None,
    prepared_layout: str = "out_in",
    warmup: int = 5,
    repeats: int = 20,
    min_speedup: float = 1.0,
    atol: float = 2e-2,
    rtol: float = 2e-2,
) -> dict[str, Any]:
    """Qualify every candidate independently and fail closed on parity or speed.

    The receipt binds decisions to GPU identity, dtypes, model geometry, and a
    prepared-layout signature. Callers may adopt only entries whose ``adopted``
    flag is true; public APIs otherwise retain eager behavior.
    """
    _validate_b1t1(x)
    rkv_input = x if rkv_x is None else rkv_x
    if rkv_input.ndim == 4:
        if rkv_input.shape[:3] != (x.shape[0], 1, 3):
            raise ValueError("qualification rkv_x must have shape [batch,1,3,in]")
    else:
        _validate_b1t1(rkv_input)
    if rkv_input.shape[-1] != x.shape[-1]:
        raise ValueError("qualification x and rkv_x feature dimensions differ")
    if rkv_input.device != x.device or rkv_input.dtype != x.dtype:
        raise ValueError("qualification x and rkv_x must share device/dtype")
    prepared = prepare_row_one_weight(row_weight, layout=prepared_layout)
    geometry = {
        "batch": x.shape[0], "input": x.shape[-1],
        "row_output": row_weight.shape[0], "rkv_output": rkv_weights.shape[1],
        "ffn_hidden": key_weight.shape[0], "ffn_output": value_weight.shape[0],
        "dtype": str(x.dtype), "prepared_layout": prepared_layout,
        "rkv_input_mode": "distinct" if rkv_input.ndim == 4 else "shared",
        "prepared_signature": prepared.source_signature,
    }
    tuning_identity = {
        "runtime": _runtime_metadata(x),
        "geometry": geometry,
        "search_space": {
            "block_n": [16, 32, 64, 128],
            "warps": [4, 4, 8, 8],
            "weight_layouts": ["out_in", "in_out"],
        },
    }
    tuning_key = hashlib.sha256(
        json.dumps(tuning_identity, sort_keys=True).encode()).hexdigest()
    base: dict[str, Any] = {
        "schema": "rwkv-lab.b1t1-linear-qualification.v1",
        "qualified_at_unix": time.time(),
        "runtime": _runtime_metadata(x),
        "geometry": geometry,
        "tuning": {**tuning_identity["search_space"], "key": tuning_key},
        "source": "https://github.com/BlinkDL/Albatross/tree/main/faster3b_2606",
        "candidates": {},
        "adopted": False,
    }
    if not _cuda_candidate_available(x):
        base["reason"] = "CUDA Triton inference qualification is unavailable"
        return base

    pairs = {
        "row_one": (
            lambda: row_one_linear(x, prepared, row_bias),
            lambda: row_one_linear(x, prepared, row_bias, use_candidate=True),
        ),
        "packed_rkv": (
            lambda: packed_rkv_projection(rkv_input, rkv_weights, rkv_biases),
            lambda: packed_rkv_projection(
                rkv_input, rkv_weights, rkv_biases, use_candidate=True),
        ),
        "ffn_sqrelu_value": (
            lambda: ffn_squared_relu_value(
                x, key_weight, value_weight, key_bias, value_bias),
            lambda: ffn_squared_relu_value(
                x, key_weight, value_weight, key_bias, value_bias,
                use_candidate=True),
        ),
    }
    for name, (eager, candidate) in pairs.items():
        expected = eager()
        actual = candidate()
        repeated = candidate()
        max_error = float((expected.float() - actual.float()).abs().max().item())
        parity = bool(torch.allclose(expected, actual, atol=atol, rtol=rtol))
        deterministic = bool(torch.equal(actual, repeated))
        eager_us = _median_cuda_us(eager, warmup=warmup, repeats=repeats)
        candidate_us = _median_cuda_us(candidate, warmup=warmup, repeats=repeats)
        speedup = eager_us / candidate_us if candidate_us else float("inf")
        adopted = parity and deterministic and speedup >= min_speedup
        base["candidates"][name] = {
            "parity": parity, "deterministic": deterministic,
            "max_abs_error": max_error, "eager_median_us": eager_us,
            "candidate_median_us": candidate_us, "speedup": speedup,
            "minimum_speedup": min_speedup, "adopted": adopted,
        }
    base["adopted"] = all(
        item["adopted"] for item in base["candidates"].values())
    return base
