"""NVFP4-style simulated quantization-aware pretraining.

References:

* NVIDIA, "Pretraining Large Language Models with NVFP4",
  arXiv:2509.25149, https://arxiv.org/abs/2509.25149.
* NVIDIA, "TetraJet-v2: Accelerating Pretraining with NVFP4",
  arXiv:2510.27527, https://arxiv.org/abs/2510.27527.

The cited systems use Blackwell tensor cores, block scales, randomized Hadamard
transforms, and higher-precision master state. This file implements the
algorithmically faithful *fake-quant* boundary for A/B work: E2M1 values,
per-block scales, optional RHT, and straight-through gradients over bf16/fp32
master weights. It does not claim NVFP4 hardware throughput.
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


_E2M1_POSITIVE = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def _nearest_e2m1(x: torch.Tensor) -> torch.Tensor:
    code = x.new_tensor(_E2M1_POSITIVE)
    mag = x.abs().unsqueeze(-1)
    q = code[(mag - code).abs().argmin(-1)]
    return q.copysign(x)


def nvfp4_fake_quant(x: torch.Tensor, *, block_size: int = 16,
                     stochastic: bool = False, ste: bool = True) -> torch.Tensor:
    """Quantize/dequantize E2M1 with one absmax scale per contiguous block."""

    if block_size <= 0:
        raise ValueError("block_size must be positive")
    shape, flat = x.shape, x.float().reshape(-1)
    pad = (-flat.numel()) % block_size
    if pad:
        flat = F.pad(flat, (0, pad))
    blocks = flat.view(-1, block_size)
    scale = blocks.abs().amax(-1, keepdim=True).clamp_min(1e-12) / 6.0
    normalized = (blocks / scale).clamp(-6, 6)
    if stochastic:
        # Small unbiased dither before nearest-code projection; reproducible under torch RNG.
        normalized = normalized + (torch.rand_like(normalized) - 0.5) / 2
    quantized = (_nearest_e2m1(normalized) * scale).reshape(-1)[:x.numel()].reshape(shape)
    quantized = quantized.to(x.dtype)
    return x + (quantized - x).detach() if ste else quantized


def hadamard_transform(x: torch.Tensor, signs: torch.Tensor | None = None) -> torch.Tensor:
    """Orthonormal randomized Hadamard transform over a power-of-two last dim."""

    n = x.shape[-1]
    if n <= 0 or n & (n - 1):
        raise ValueError("Hadamard dimension must be a positive power of two")
    y = x if signs is None else x * signs.to(device=x.device, dtype=x.dtype)
    y = y.reshape(-1, n)
    h = 1
    while h < n:
        y = y.reshape(-1, n // (2 * h), 2, h)
        a, b = y[:, :, 0].clone(), y[:, :, 1].clone()
        y = torch.stack((a + b, a - b), dim=2).reshape(-1, n)
        h *= 2
    return (y / math.sqrt(n)).reshape(x.shape)


class NVFP4Linear(nn.Linear):
    """Linear layer with fake-NVFP4 operands and normal master parameters."""

    def __init__(self, *args, block_size: int = 16, rht: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.block_size, self.rht = int(block_size), bool(rht)
        if self.rht:
            if self.in_features & (self.in_features - 1):
                raise ValueError("RHT requires power-of-two in_features")
            self.register_buffer("rht_signs", torch.randint(0, 2, (self.in_features,)) * 2 - 1)
        else:
            self.register_buffer("rht_signs", None)

    @classmethod
    def from_linear(cls, layer: nn.Linear, *, block_size: int = 16,
                    rht: bool = False) -> "NVFP4Linear":
        out = cls(layer.in_features, layer.out_features, bias=layer.bias is not None,
                  device=layer.weight.device, dtype=layer.weight.dtype,
                  block_size=block_size, rht=rht)
        out.weight = layer.weight
        if layer.bias is not None:
            out.bias = layer.bias
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.weight
        if self.rht:
            x = hadamard_transform(x, self.rht_signs)
            w = hadamard_transform(w, self.rht_signs)
        qx = nvfp4_fake_quant(x, block_size=self.block_size)
        qw = nvfp4_fake_quant(w, block_size=self.block_size)
        return F.linear(qx, qw, self.bias)


def convert_to_nvfp4_training(module: nn.Module, *, block_size: int = 16,
                              rht: bool = False) -> int:
    """Replace eligible hidden linears in place; return the conversion count."""

    count = 0
    for name, child in list(module.named_children()):
        low = name.lower()
        if isinstance(child, nn.Linear) and not isinstance(child, NVFP4Linear) \
                and not any(k in low for k in ("head", "engram", "de_")):
            use_rht = rht and child.in_features > 0 and not (child.in_features & (child.in_features - 1))
            setattr(module, name, NVFP4Linear.from_linear(child, block_size=block_size, rht=use_rht))
            count += 1
        elif not isinstance(child, NVFP4Linear):
            count += convert_to_nvfp4_training(child, block_size=block_size, rht=rht)
    return count
