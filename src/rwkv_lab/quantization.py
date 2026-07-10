"""Portable NF4 frozen-linear backend and QLoRA qualification.

QLoRA introduced training LoRA adapters through a frozen 4-bit NormalFloat base:
https://arxiv.org/abs/2305.14314.  The NF4 codebook below matches the primary bitsandbytes
implementation: https://github.com/bitsandbytes-foundation/bitsandbytes/blob/main/bitsandbytes/functional.py

This PyTorch implementation is a correctness and portability backend.  It stores two 4-bit
indices per byte and one scale per block, but dequantizes for ``F.linear``.  Kernel adoption is a
separate parity-gated concern in ``posttrain_kernels.py``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from torch import nn
import torch.nn.functional as F


NF4_CODEBOOK = (
    -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
    -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
    0.07958029955625534, 0.16093020141124725, 0.24611230194568634,
    0.33791524171829224, 0.44070982933044434, 0.5626170039176941,
    0.7229568362236023, 1.0,
)


def quantize_nf4(weight: torch.Tensor, block_size: int = 64) -> tuple[torch.Tensor, torch.Tensor, int]:
    if not weight.is_floating_point() or block_size <= 0:
        raise ValueError("NF4 needs a floating tensor and positive block size")
    flat = weight.detach().float().flatten()
    original_count = flat.numel()
    padded_count = ((original_count + block_size - 1) // block_size) * block_size
    if padded_count != original_count:
        flat = F.pad(flat, (0, padded_count - original_count))
    blocks = flat.reshape(-1, block_size)
    scales = blocks.abs().amax(-1).clamp_min(torch.finfo(torch.float32).tiny)
    normalized = blocks / scales[:, None]
    code = torch.tensor(NF4_CODEBOOK, device=weight.device, dtype=torch.float32)
    indices = (normalized[..., None] - code).abs().argmin(-1).to(torch.uint8).flatten()
    if indices.numel() % 2:
        indices = F.pad(indices, (0, 1))
    packed = indices[0::2] | (indices[1::2] << 4)
    return packed, scales.to(torch.float16), original_count


def dequantize_nf4(packed: torch.Tensor, scales: torch.Tensor, count: int,
                   block_size: int, *, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    low = packed & 0x0F
    high = packed >> 4
    indices = torch.stack((low, high), -1).flatten().long()
    code = torch.tensor(NF4_CODEBOOK, device=packed.device, dtype=torch.float32)
    values = code[indices].reshape(-1, block_size) * scales.float()[:, None]
    return values.flatten()[:count].to(dtype)


class NF4Linear(nn.Module):
    """Frozen packed-NF4 linear compatible with the native LoRA wrapper."""

    is_quantized_4bit = True

    def __init__(self, linear: nn.Module, *, block_size: int = 64,
                 compute_dtype: torch.dtype = torch.float32):
        super().__init__()
        if not hasattr(linear, "weight"):
            raise TypeError("NF4Linear needs a module with a weight")
        packed, scales, count = quantize_nf4(linear.weight, block_size)
        self.in_features = int(linear.in_features)
        self.out_features = int(linear.out_features)
        self.block_size = int(block_size)
        self.weight_count = int(count)
        self.compute_dtype = compute_dtype
        self.register_buffer("packed_weight", packed)
        self.register_buffer("scales", scales)
        bias = getattr(linear, "bias", None)
        self.register_buffer("bias", bias.detach().clone() if bias is not None else None)

    @property
    def weight(self) -> torch.Tensor:
        return self.dequantized_weight()

    def dequantized_weight(self, *, dtype: torch.dtype | None = None) -> torch.Tensor:
        value = dequantize_nf4(self.packed_weight, self.scales, self.weight_count,
                               self.block_size, dtype=dtype or self.compute_dtype)
        return value.reshape(self.out_features, self.in_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = self.compute_dtype if x.device.type != "cpu" else torch.float32
        weight = self.dequantized_weight(dtype=dtype)
        bias = self.bias.to(dtype) if self.bias is not None else None
        return F.linear(x.to(dtype), weight, bias)

    def storage_bytes(self) -> int:
        tensors = (self.packed_weight, self.scales, self.bias)
        return sum(value.numel() * value.element_size() for value in tensors if value is not None)


def quantize_model_nf4(model: nn.Module, *, block_size: int = 64,
                       exclude: Iterable[str] = ("head",)) -> list[str]:
    """Replace eligible dense linears in place; call before injecting LoRA."""
    excluded = tuple(exclude)
    selected = []
    for path, module in list(model.named_modules()):
        if not path or isinstance(module, NF4Linear) or not isinstance(module, nn.Linear):
            continue
        if any(path == name or path.startswith(name + ".") or path.endswith("." + name)
               for name in excluded):
            continue
        parent_path, _, attr = path.rpartition(".")
        parent = model.get_submodule(parent_path) if parent_path else model
        setattr(parent, attr, NF4Linear(module, block_size=block_size,
                                       compute_dtype=module.weight.dtype))
        selected.append(path)
    if not selected:
        raise ValueError("no dense linear modules were eligible for NF4 quantization")
    return selected


@torch.no_grad()
def dequantize_model_nf4(model: nn.Module) -> list[str]:
    """Materialize every remaining packed base as a standard dense Linear."""
    replaced = []
    for path, module in list(model.named_modules()):
        if not path or not isinstance(module, NF4Linear):
            continue
        dense = nn.Linear(module.in_features, module.out_features, bias=module.bias is not None,
                          device=module.packed_weight.device, dtype=torch.float32)
        dense.weight.copy_(module.dequantized_weight(dtype=torch.float32))
        if module.bias is not None:
            dense.bias.copy_(module.bias.float())
        parent_path, _, attr = path.rpartition(".")
        parent = model.get_submodule(parent_path) if parent_path else model
        setattr(parent, attr, dense)
        replaced.append(path)
    return replaced


def model_storage_bytes(model: nn.Module) -> int:
    return sum(tensor.numel() * tensor.element_size() for tensor in model.state_dict().values())


@dataclass(frozen=True)
class QLoRAQualification:
    schema: str
    max_abs_base_error: float
    zero_init_max_abs: float
    merged_max_abs: float
    gradient_finite: bool
    gradient_nonzero: bool
    dense_bytes: int
    quantized_bytes: int
    compression_ratio: float
    passed: bool


def qualify_linear_qlora(linear: nn.Linear, sample: torch.Tensor, *, block_size: int = 64,
                         rank: int = 4, base_tolerance: float = 0.35,
                         parity_tolerance: float = 2e-5) -> QLoRAQualification:
    """Qualify initialization, gradients, dense merge parity, and actual stored bytes."""
    from rwkv_lab.adapters import LoRALinear

    dense = nn.Linear(linear.in_features, linear.out_features, bias=linear.bias is not None,
                      device=linear.weight.device, dtype=linear.weight.dtype)
    dense.load_state_dict(linear.state_dict())
    quantized = NF4Linear(linear, block_size=block_size, compute_dtype=torch.float32)
    wrapped = LoRALinear(quantized)
    wrapped.add_adapter("qualification", rank=rank, alpha=float(rank))
    with torch.no_grad():
        dense_output = dense(sample)
        quantized_output = quantized(sample)
        zero_output = wrapped(sample)
        base_error = float((dense_output.float() - quantized_output.float()).abs().max())
        zero_error = float((quantized_output.float() - zero_output.float()).abs().max())
    loss = wrapped(sample).float().square().mean()
    loss.backward()
    gradients = [wrapped.adapters["qualification"].A.grad,
                 wrapped.adapters["qualification"].B.grad]
    finite = all(value is not None and torch.isfinite(value).all() for value in gradients)
    nonzero = wrapped.adapters["qualification"].B.grad is not None and bool(
        torch.any(wrapped.adapters["qualification"].B.grad != 0))
    with torch.no_grad():
        branch = wrapped.adapters["qualification"]
        merged_weight = quantized.dequantized_weight() + branch.delta().to(torch.float32)
        merged = F.linear(sample.float(), merged_weight,
                          quantized.bias.float() if quantized.bias is not None else None)
        active = wrapped(sample).float()
        merged_error = float((merged - active).abs().max())
    dense_bytes = linear.weight.numel() * linear.weight.element_size()
    if linear.bias is not None:
        dense_bytes += linear.bias.numel() * linear.bias.element_size()
    quantized_bytes = quantized.storage_bytes()
    ratio = dense_bytes / max(1, quantized_bytes)
    passed = (base_error <= base_tolerance and zero_error <= parity_tolerance and
              merged_error <= parity_tolerance and finite and nonzero and ratio > 1.5)
    return QLoRAQualification("rwkv-lab.qlora-qualification.v1", base_error, zero_error,
                              merged_error, bool(finite), bool(nonzero), dense_bytes,
                              quantized_bytes, ratio, bool(passed))
