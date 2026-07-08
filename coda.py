"""CODA-inspired throughput helper (arXiv:2605.19269) — the PORTABLE subset only.

Honest scope (from reading the paper): full CODA fuses memory-bound ops (RMSNorm, SwiGLU,
RoPE, cross-entropy, residual add) into the *epilogue* of an adjacent GEMM, on-chip in the
Tensor-Core accumulator registers, via a CuTeDSL/CUTLASS DSL. That register-level, cross-GEMM
fusion is NOT expressible in PyTorch/torch.compile — inductor cannot fuse arbitrary epilogue
work into a cuBLAS GEMM's output tile or across two matmuls. Reproducing it means writing
custom CuTeDSL kernels (their repo), a real GPU-kernel effort, Hopper/Blackwell single-GPU.

What IS portable — and captures the *category* of win (collapsing many small memory-bound
kernels + their global-memory round-trips) — is:
  1. wrap each block's non-attention pointwise chain (RMSNorm → proj → SwiGLU/act, RoPE,
     residual add) in ONE torch.compile region (`mode="max-autotune"`);
  2. use a fused-linear-cross-entropy for the large-vocab head (Liger / Cut-Cross-Entropy)
     so the [B,T,V] logits are never fully materialised.

`compile_module` is a safe wrapper: it torch.compiles a module and, on any failure (no triton,
CPU-only, unsupported op), returns the original module unchanged — so it is always safe to call.
Expect a few-percent to low-double-digit end-to-end win from (1)+(2); the final register-level
slice is the custom-CUDA-only part and is intentionally out of scope here.
"""
from __future__ import annotations

import os
import torch
import torch.nn as nn


def compile_module(module: nn.Module, mode: str = "max-autotune", enabled: bool = True) -> nn.Module:
    """torch.compile `module`, returning the compiled version — or the ORIGINAL module unchanged
    if compilation is disabled/unavailable/fails (CPU, no triton, env opt-out). Never raises."""
    if not enabled or os.environ.get("CODA_NO_COMPILE") == "1":
        return module
    if not hasattr(torch, "compile"):
        return module
    try:
        return torch.compile(module, mode=mode, fullgraph=False, dynamic=False)
    except Exception as e:  # unsupported backend / no GPU / inductor error -> transparent fallback
        print(f"[coda] torch.compile unavailable ({type(e).__name__}: {e}); running eager", flush=True)
        return module


class SwiGLU(nn.Module):
    """SwiGLU MLP whose forward is one pointwise chain (gate/up split → silu(gate)*up → down).
    Wrap an instance in `compile_module(...)` to get inductor to fuse the chain into a single
    kernel (the portable slice of CODA's SwiGLU epilogue). Standard SwiGLU numerics; drop-in."""

    def __init__(self, d_model: int, hidden: int):
        super().__init__()
        self.up = nn.Linear(d_model, 2 * hidden, bias=False)
        self.down = nn.Linear(hidden, d_model, bias=False)

    def forward(self, x):
        g, u = self.up(x).chunk(2, dim=-1)
        return self.down(torch.nn.functional.silu(g) * u)
