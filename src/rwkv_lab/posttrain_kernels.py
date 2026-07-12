"""Parity-gated post-training kernel and compile qualification.

LoRA math follows https://arxiv.org/abs/2106.09685. Preference formulas remain in
``preference.py`` with their paper citations. No candidate is adopted from timing alone: outputs,
losses, and gradients must pass first. Activation offload uses PyTorch's documented
``torch.autograd.graph.save_on_cpu`` hook.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import asdict, dataclass
import json
import statistics
import time
from typing import Callable

import torch
from torch import nn

from rwkv_lab.adapters import LoRABranch
from rwkv_lab.preference import OutcomeRewardHead, ProcessRewardHead, sequence_logps


@dataclass(frozen=True)
class Qualification:
    name: str
    max_abs: float
    gradient_max_abs: float
    eager_ms: float
    candidate_ms: float
    speedup: float
    parity_passed: bool
    performance_passed: bool
    adopted: bool
    parameter_gradient_max_abs: float = 0.0


def _timed(fn: Callable, inputs: tuple, *, warmup: int = 3, repeats: int = 10) -> float:
    for _ in range(warmup):
        fn(*inputs)
    if any(isinstance(value, torch.Tensor) and value.is_cuda for value in inputs):
        torch.cuda.synchronize()
    values = []
    for _ in range(repeats):
        started = time.perf_counter()
        fn(*inputs)
        if any(isinstance(value, torch.Tensor) and value.is_cuda for value in inputs):
            torch.cuda.synchronize()
        values.append((time.perf_counter() - started) * 1000)
    return statistics.median(values)


def qualify_callable(name: str, eager: Callable, candidate: Callable, inputs: tuple, *,
                     tolerance: float = 2e-5, minimum_speedup: float = 1.02,
                     params: tuple[torch.Tensor, ...] | None = None) -> Qualification:
    detached = tuple(value.detach().clone().requires_grad_(value.requires_grad)
                     if isinstance(value, torch.Tensor) else value for value in inputs)
    eager_out = eager(*inputs)
    candidate_out = candidate(*detached)
    max_abs = float((eager_out.detach().float() - candidate_out.detach().float()).abs().max())
    gradient_error = 0.0
    eager_tensors = [value for value in inputs if isinstance(value, torch.Tensor) and value.requires_grad]
    candidate_tensors = [value for value in detached if isinstance(value, torch.Tensor) and value.requires_grad]
    if eager_tensors:
        eager_grad = torch.autograd.grad(eager_out.float().sum(), eager_tensors, retain_graph=True)
        candidate_grad = torch.autograd.grad(candidate_out.float().sum(), candidate_tensors,
                                             retain_graph=True)
        gradient_error = max(float((left.float() - right.float()).abs().max())
                             for left, right in zip(eager_grad, candidate_grad))
    # Pair each eager parameter with its candidate counterpart. A shared-param
    # candidate (torch.compile wrapper) pairs a param with itself; a candidate
    # with its OWN copied parameters (handwritten kernel) pairs by name —
    # differentiating the union instead would give one-sided None grads and
    # unconditionally fail parity for exactly that case.
    if params is not None:
        param_pairs = [(p, p) for p in params]
    elif isinstance(eager, nn.Module) and isinstance(candidate, nn.Module):
        def _named(mod: nn.Module) -> dict[str, torch.Tensor]:
            return {k.removeprefix("_orig_mod."): v for k, v in mod.named_parameters()
                    if v.requires_grad}
        e_named, c_named = _named(eager), _named(candidate)
        param_pairs = [(e_named[k], c_named[k]) for k in e_named if k in c_named]
    elif isinstance(eager, nn.Module):
        param_pairs = [(p, p) for p in eager.parameters() if p.requires_grad]
    elif isinstance(candidate, nn.Module):
        param_pairs = [(p, p) for p in candidate.parameters() if p.requires_grad]
    else:
        param_pairs = []
    param_gradient_error = 0.0
    if param_pairs and eager_out.requires_grad and candidate_out.requires_grad:
        eager_pgrad = torch.autograd.grad(eager_out.float().sum(),
                                          [e for e, _ in param_pairs],
                                          retain_graph=True, allow_unused=True)
        candidate_pgrad = torch.autograd.grad(candidate_out.float().sum(),
                                              [c for _, c in param_pairs],
                                              retain_graph=True, allow_unused=True)
        for left, right in zip(eager_pgrad, candidate_pgrad):
            if left is None and right is None:
                continue
            if left is None or right is None:
                param_gradient_error = float("inf")
                continue
            param_gradient_error = max(param_gradient_error,
                                       float((left.float() - right.float()).abs().max()))
    eager_ms = _timed(eager, inputs)
    candidate_ms = _timed(candidate, detached)
    speedup = eager_ms / max(candidate_ms, 1e-12)
    parity = (max_abs <= tolerance and gradient_error <= tolerance
              and param_gradient_error <= tolerance)
    performance = speedup >= minimum_speedup
    return Qualification(name, max_abs, gradient_error, eager_ms, candidate_ms, speedup,
                         parity, performance, parity and performance,
                         parameter_gradient_max_abs=param_gradient_error)


def score_preference_pairs(model: nn.Module, chosen_ids: torch.Tensor, chosen_labels: torch.Tensor,
                           rejected_ids: torch.Tensor, rejected_labels: torch.Tensor, *,
                           average: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    """One recurrent model call for both sides of a pair, preserving paired ordering."""
    if chosen_ids.shape != rejected_ids.shape or chosen_labels.shape != rejected_labels.shape:
        raise ValueError("preference pair batches must have matching padded shapes")
    ids = torch.cat((chosen_ids, rejected_ids), 0)
    labels = torch.cat((chosen_labels, rejected_labels), 0)
    scores = sequence_logps(model(ids), labels, average=average)
    count = chosen_ids.shape[0]
    return scores[:count], scores[count:]


def activation_offload(enabled: bool, device: torch.device | str):
    """Offload saved CUDA activations for backward; no-op elsewhere."""
    return (torch.autograd.graph.save_on_cpu(pin_memory=True)
            if enabled and torch.device(device).type == "cuda" else nullcontext())


def _activation_profile(module: nn.Module, sample: torch.Tensor, enabled: bool) -> dict:
    if not sample.is_cuda:
        return {"enabled": enabled, "available": False, "reason": "CUDA required"}
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(sample.device)
    module.zero_grad(set_to_none=True)
    value = sample.detach().clone().requires_grad_(True)
    started = time.perf_counter()
    with activation_offload(enabled, sample.device):
        output = module(value)
        output.float().square().mean().backward()
    torch.cuda.synchronize()
    return {"enabled": enabled, "available": True,
            "elapsed_ms": (time.perf_counter() - started) * 1000,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(sample.device),
            "output_sum": float(output.detach().float().sum())}


def benchmark_suite(device: str = "auto", *, compile_backend: str = "inductor") -> dict:
    device = "cuda" if device == "auto" and torch.cuda.is_available() else (
        "cpu" if device == "auto" else device)
    dtype = torch.float32
    torch.manual_seed(17)
    reports = []

    branch = LoRABranch(256, 256, 16, 32.0, device=device, dtype=dtype).eval()
    with torch.no_grad():
        branch.B.normal_(std=0.01)
    x = torch.randn(4, 128, 256, device=device, dtype=dtype, requires_grad=True)
    try:
        compiled_branch = torch.compile(branch, backend=compile_backend)
        reports.append(qualify_callable("lora_compile", branch, compiled_branch, (x,)))
    except Exception as exc:
        reports.append({"name": "lora_compile", "adopted": False, "error": repr(exc)})

    hidden = torch.randn(4, 128, 256, device=device, dtype=dtype, requires_grad=True)
    response_mask = torch.zeros(4, 128, dtype=torch.bool, device=device)
    response_mask[:, 64:] = True
    outcome = OutcomeRewardHead(256).to(device)
    process = ProcessRewardHead(256).to(device)
    positions = torch.arange(15, 128, 16, device=device)[None].expand(4, -1)
    for name, module, inputs in (("outcome_reward_compile", outcome, (hidden, response_mask)),
                                 ("process_reward_compile", process, (hidden, positions))):
        try:
            compiled = torch.compile(module, backend=compile_backend)
            reports.append(qualify_callable(name, module, compiled, inputs))
        except Exception as exc:
            reports.append({"name": name, "adopted": False, "error": repr(exc)})

    class ToyScorer(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(512, 64)
            self.head = nn.Linear(64, 512)

        def forward(self, ids):
            return self.head(self.embedding(ids))

    scorer = ToyScorer().to(device)
    chosen_ids = torch.randint(1, 512, (8, 96), device=device)
    rejected_ids = torch.randint(1, 512, (8, 96), device=device)
    chosen_labels, rejected_labels = chosen_ids.clone(), rejected_ids.clone()
    chosen_labels[:, :32] = -100
    rejected_labels[:, :32] = -100

    def separate(ci, cl, ri, rl):
        return torch.cat((sequence_logps(scorer(ci), cl), sequence_logps(scorer(ri), rl)))

    def paired(ci, cl, ri, rl):
        return torch.cat(score_preference_pairs(scorer, ci, cl, ri, rl))

    reports.append(qualify_callable("recurrent_paired_scoring", separate, paired,
                                    (chosen_ids, chosen_labels, rejected_ids, rejected_labels)))
    offload_profiles = [_activation_profile(branch, x, False),
                        _activation_profile(branch, x, True)]
    if all(row.get("available") for row in offload_profiles):
        offload_profiles[1]["parity_passed"] = abs(
            offload_profiles[0]["output_sum"] - offload_profiles[1]["output_sum"]) <= 2e-5
        offload_profiles[1]["memory_reduced"] = (
            offload_profiles[1]["peak_allocated_bytes"] < offload_profiles[0]["peak_allocated_bytes"])
        offload_profiles[1]["adopted"] = (offload_profiles[1]["parity_passed"] and
                                           offload_profiles[1]["memory_reduced"])
    serialized = [asdict(value) if isinstance(value, Qualification) else value for value in reports]
    return {"schema": "rwkv-lab.posttrain-kernel-qualification.v1", "device": device,
            "compile_backend": compile_backend, "reports": serialized,
            "activation_offload_profiles": offload_profiles,
            "adopted": [row["name"] for row in serialized if row.get("adopted")],
            "policy": "output and gradient parity, then median speedup >= 1.02"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Qualify post-training kernels and compile profiles")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--compile-backend", default="inductor")
    parser.add_argument("--output", default="")
    args = parser.parse_args()
    report = benchmark_suite(device=args.device, compile_backend=args.compile_backend)
    if args.output:
        from pathlib import Path
        Path(args.output).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
