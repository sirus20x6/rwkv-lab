"""Fail-closed qualification boundary for externally synthesized kernels.

Mirage searches algebraic and schedule transformations across kernel/block/
thread levels and probabilistically verifies candidates: Wu et al. (2024),
https://arxiv.org/abs/2405.05751. Community lead:
https://discord.com/channels/992359628979568762/992362493055881276/1524695625336225852

RWKV-Lab does not execute generated source or shell commands. External tools
must provide an already-imported callable, which is adopted only after output,
gradient, determinism, and median-latency qualification.
"""
from __future__ import annotations

import hashlib
import time
from typing import Callable, Sequence
import torch


def _tensor_leaves(value):
    if isinstance(value, torch.Tensor):
        return [value]
    if isinstance(value, (tuple, list)):
        return [leaf for child in value for leaf in _tensor_leaves(child)]
    if isinstance(value, dict):
        return [leaf for key in sorted(value) for leaf in _tensor_leaves(value[key])]
    raise TypeError(f"kernel outputs must be tensors or tensor containers, got {type(value).__name__}")


def _tree_close(left, right, *, atol, rtol, exact=False):
    a, b = _tensor_leaves(left), _tensor_leaves(right)
    return len(a) == len(b) and all(
        (torch.equal(x, y) if exact else torch.allclose(x, y, atol=atol, rtol=rtol))
        for x, y in zip(a, b))


def qualify_kernel_candidate(reference: Callable, candidate: Callable,
                             probes: Sequence[tuple[torch.Tensor, ...]], *,
                             source: str, repeats: int = 5,
                             atol: float = 1e-5, rtol: float = 1e-4,
                             minimum_speedup: float = 1.02) -> dict:
    if not probes:
        raise ValueError("at least one probe is required")
    exact, gradient, deterministic = True, True, True
    for probe in probes:
        ref_args = tuple(x.detach().clone().requires_grad_(x.is_floating_point()) for x in probe)
        can_args = tuple(x.detach().clone().requires_grad_(x.is_floating_point()) for x in probe)
        expected, actual = reference(*ref_args), candidate(*can_args)
        exact &= _tree_close(actual, expected, atol=atol, rtol=rtol)
        repeat_args = tuple(x.detach().clone() for x in can_args)
        deterministic &= _tree_close(actual, candidate(*repeat_args), atol=0, rtol=0, exact=True)
        ref_leaves, can_leaves = _tensor_leaves(expected), _tensor_leaves(actual)
        ref_loss = sum((x.float().sum() for x in ref_leaves if x.is_floating_point()),
                       ref_args[0].new_zeros((), dtype=torch.float32))
        can_loss = sum((x.float().sum() for x in can_leaves if x.is_floating_point()),
                       can_args[0].new_zeros((), dtype=torch.float32))
        if ref_loss.requires_grad:
            # Fail-closed: a candidate whose output does not participate in
            # autograd (or whose backward raises) is rejected, not raised.
            try:
                ref_loss.backward()
                if not can_loss.requires_grad:
                    gradient = False
                else:
                    can_loss.backward()
                    for a, b in zip(ref_args, can_args):
                        if a.grad is not None:
                            gradient &= b.grad is not None and torch.allclose(a.grad, b.grad, atol=atol, rtol=rtol)
            except RuntimeError:
                gradient = False

    def median_latency(fn):
        warm_args = tuple(x.detach().clone() for x in probes[0])
        for _ in range(2):  # untimed warmup absorbs JIT/compile cost
            fn(*warm_args)
        timings = []
        for _ in range(max(1, repeats)):
            args = tuple(x.detach().clone() for x in probes[0])
            if probes[0][0].is_cuda: torch.cuda.synchronize(probes[0][0].device)
            started = time.perf_counter(); fn(*args)
            if probes[0][0].is_cuda: torch.cuda.synchronize(probes[0][0].device)
            timings.append(time.perf_counter() - started)
        return sorted(timings)[len(timings) // 2]
    ref_s, candidate_s = median_latency(reference), median_latency(candidate)
    speedup = ref_s / max(candidate_s, 1e-12)
    return {"schema": "rwkv-lab.external-kernel-candidate.v1", "source": source,
            "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
            "probes": len(probes), "output_parity": exact, "gradient_parity": gradient,
            "deterministic": deterministic, "reference_seconds": ref_s,
            "candidate_seconds": candidate_s, "speedup": speedup,
            "adopted": bool(exact and gradient and deterministic and speedup >= minimum_speedup)}
