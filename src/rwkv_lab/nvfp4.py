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
import time
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


def transformer_engine_nvfp4_status() -> tuple[bool, str]:
    """Return native NVFP4 availability without silently importing a fake path.

    Transformer Engine documents NVFP4 as an SM100+ recipe with E2M1 values,
    hierarchical block/global scales, 2-D weight quantization, and optional RHT:
    https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/features/low_precision_training/nvfp4/nvfp4.html
    """

    try:
        import transformer_engine.pytorch as te
    except Exception as exc:
        return False, f"Transformer Engine unavailable: {exc!r}"
    try:
        status = te.is_nvfp4_available(return_reason=True)
    except Exception as exc:
        return False, f"Transformer Engine cannot query NVFP4 support: {exc!r}"
    if isinstance(status, tuple):
        return bool(status[0]), str(status[1])
    return bool(status), "available" if status else "native NVFP4 unavailable"


class TransformerEngineNVFP4Linear(nn.Module):
    """Blackwell tensor-core NVFP4 linear, kept separate from the fake-quant oracle."""

    accelerated_backend = "transformer_engine_nvfp4"

    def __init__(self, linear: nn.Linear, *, rht: bool = True):
        super().__init__()
        available, reason = transformer_engine_nvfp4_status()
        if not available:
            raise RuntimeError(reason)
        if linear.weight.device.type != "cuda":
            raise ValueError("native NVFP4 requires a CUDA-resident linear")
        import transformer_engine.pytorch as te
        from transformer_engine.common.recipe import NVFP4BlockScaling

        self.in_features = int(linear.in_features)
        self.out_features = int(linear.out_features)
        self.rht = bool(rht)
        self.recipe = NVFP4BlockScaling(disable_rht=not self.rht)
        self.linear = te.Linear(
            self.in_features, self.out_features, bias=linear.bias is not None,
            params_dtype=linear.weight.dtype, device=linear.weight.device,
        )
        with torch.no_grad():
            self.linear.weight.copy_(linear.weight)
            if linear.bias is not None:
                self.linear.bias.copy_(linear.bias)

    @property
    def weight(self) -> torch.Tensor:
        return self.linear.weight

    @property
    def bias(self) -> torch.Tensor | None:
        return self.linear.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        import transformer_engine.pytorch as te
        with te.autocast(enabled=True, recipe=self.recipe):
            return self.linear(x)


def qualify_native_nvfp4(linear: nn.Linear, sample: torch.Tensor, *, rht: bool = True,
                         tolerance: float = 0.35, gradient_tolerance: float = 0.35,
                         minimum_speedup: float = 1.02, repeats: int = 10) -> dict:
    """Adopt Transformer Engine only after fake-quant parity and measured speed."""

    if sample.device.type != "cuda":
        return {"schema": "rwkv-lab.nvfp4-backend-qualification.v1",
                "backend": "transformer_engine", "available": False,
                "adopted": False, "error": "native NVFP4 qualification requires CUDA"}
    reference_linear = nn.Linear(
        linear.in_features, linear.out_features, bias=linear.bias is not None,
        device=linear.weight.device, dtype=linear.weight.dtype)
    reference_linear.load_state_dict(linear.state_dict())
    reference = NVFP4Linear.from_linear(reference_linear, rht=rht)
    try:
        candidate = TransformerEngineNVFP4Linear(linear, rht=rht)
    except Exception as exc:
        return {"schema": "rwkv-lab.nvfp4-backend-qualification.v1",
                "backend": "transformer_engine", "available": False,
                "adopted": False, "error": repr(exc)}

    x0 = sample.detach().clone().requires_grad_(True)
    x1 = sample.detach().clone().requires_grad_(True)
    y0, y1 = reference(x0), candidate(x1)
    output_error = float((y0.detach().float() - y1.detach().float()).abs().max())
    g0 = torch.autograd.grad(y0.float().sum(), (x0, reference.weight), retain_graph=False)
    g1 = torch.autograd.grad(y1.float().sum(), (x1, candidate.weight), retain_graph=False)
    gradient_error = max(float((left.float() - right.float()).abs().max())
                         for left, right in zip(g0, g1))

    def median_ms(module: nn.Module, value: torch.Tensor) -> float:
        for _ in range(3):
            module(value)
        torch.cuda.synchronize(value.device)
        timings = []
        for _ in range(repeats):
            started = time.perf_counter()
            module(value)
            torch.cuda.synchronize(value.device)
            timings.append((time.perf_counter() - started) * 1000)
        return sorted(timings)[len(timings) // 2]

    reference_ms = median_ms(reference, sample)
    candidate_ms = median_ms(candidate, sample)
    speedup = reference_ms / max(candidate_ms, 1e-12)
    parity = output_error <= tolerance and gradient_error <= gradient_tolerance
    performance = speedup >= minimum_speedup
    return {"schema": "rwkv-lab.nvfp4-backend-qualification.v1",
            "backend": "transformer_engine", "available": True,
            "output_max_abs": output_error, "gradient_max_abs": gradient_error,
            "tolerance": tolerance, "gradient_tolerance": gradient_tolerance,
            "reference_ms": reference_ms, "candidate_ms": candidate_ms,
            "speedup": speedup, "minimum_speedup": minimum_speedup,
            "parity_passed": parity, "performance_passed": performance,
            "adopted": bool(parity and performance)}


def convert_to_nvfp4_training(module: nn.Module, *, block_size: int = 16,
                              rht: bool = False, backend: str = "fake") -> int:
    """Replace eligible hidden linears; native requests fail closed if unavailable."""

    if backend not in ("fake", "transformer_engine"):
        raise ValueError("NVFP4 backend must be fake or transformer_engine")
    if backend == "transformer_engine":
        available, reason = transformer_engine_nvfp4_status()
        if not available:
            raise RuntimeError(reason)

    count = 0
    for name, child in list(module.named_children()):
        low = name.lower()
        if isinstance(child, nn.Linear) and not isinstance(child, NVFP4Linear) \
                and not any(k in low for k in ("head", "engram", "de_")):
            if backend == "transformer_engine" and (
                    child.in_features % 16 or child.out_features % 16):
                continue
            use_rht = rht and child.in_features > 0 and not (child.in_features & (child.in_features - 1))
            replacement = (TransformerEngineNVFP4Linear(child, rht=use_rht)
                           if backend == "transformer_engine" else
                           NVFP4Linear.from_linear(child, block_size=block_size, rht=use_rht))
            setattr(module, name, replacement)
            count += 1
        elif not isinstance(child, (NVFP4Linear, TransformerEngineNVFP4Linear)):
            count += convert_to_nvfp4_training(
                child, block_size=block_size, rht=rht, backend=backend)
    return count
