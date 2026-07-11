"""Stable triangular inversion oracle and qualification for delta-rule kernels.

Reference: Sobczyk et al., "Fast and Stable Triangular Inversion for Delta-Rule
Linear Transformers", arXiv:2605.21325, https://arxiv.org/abs/2605.21325.
The paper studies direct and iterative inversion in low precision and reports
up to 4.3x NPU speedups. This implementation is a portable correctness oracle;
hardware kernels remain unadopted until ``qualify_triangular_backend`` passes.
"""
from __future__ import annotations

import time
from typing import Callable

import torch


def stable_triangular_inverse(matrix: torch.Tensor, *, method: str = "direct",
                              iterations: int | None = None) -> torch.Tensor:
    if matrix.ndim < 2 or matrix.shape[-1] != matrix.shape[-2]:
        raise ValueError("triangular matrix must be square")
    work_dtype = matrix.dtype if matrix.dtype in (torch.float32, torch.float64) else torch.float32
    lower = torch.tril(matrix.to(work_dtype))
    n = lower.shape[-1]
    if method == "direct":
        eye = torch.eye(n, device=lower.device, dtype=lower.dtype).expand(*lower.shape[:-2], n, n)
        return torch.linalg.solve_triangular(lower, eye, upper=False)
    if method != "neumann":
        raise ValueError("triangular inversion method must be direct or neumann")
    diagonal = lower.diagonal(dim1=-2, dim2=-1)
    if not torch.allclose(diagonal, torch.ones_like(diagonal), atol=1e-6, rtol=1e-6):
        raise ValueError("Neumann triangular inversion requires a unit diagonal")
    strict = lower - torch.eye(n, device=lower.device, dtype=lower.dtype)
    term = torch.eye(n, device=lower.device, dtype=lower.dtype).expand_as(lower).clone()
    inverse = term.clone()
    for _ in range(int(iterations or n - 1)):
        term = -term @ strict
        inverse = inverse + term
    return inverse


def qualify_triangular_backend(matrix: torch.Tensor, candidate: Callable[[torch.Tensor], torch.Tensor],
                               *, tolerance: float = 2e-4, repeats: int = 20,
                               minimum_speedup: float = 1.02) -> dict:
    oracle = stable_triangular_inverse(matrix.double()).float()
    try:
        proposed = candidate(matrix).float()
        error = float((oracle - proposed).abs().max())
        identity = torch.eye(matrix.shape[-1], device=matrix.device).expand_as(matrix)
        residual = float((matrix.float() @ proposed - identity).abs().max())

        def median(fn):
            values = []
            for _ in range(max(1, repeats)):
                started = time.perf_counter(); fn(); values.append(time.perf_counter() - started)
            return float(torch.tensor(values).median())
        oracle_s = median(lambda: stable_triangular_inverse(matrix))
        candidate_s = median(lambda: candidate(matrix))
        speedup = oracle_s / max(candidate_s, 1e-12)
        parity = error <= tolerance and residual <= tolerance * 4
        return {"schema": "rwkv-lab.triangular-delta-qualification.v1",
                "available": True, "parity_passed": parity, "max_error": error,
                "residual": residual, "speedup": speedup,
                "adopted": bool(parity and speedup >= minimum_speedup)}
    except Exception as exc:
        return {"schema": "rwkv-lab.triangular-delta-qualification.v1",
                "available": False, "adopted": False, "error": repr(exc)}
