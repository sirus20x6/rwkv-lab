"""Fail-closed qualification for production training and serving kernels.

Hardware backends remain optional.  A backend is *available* when it can run;
it is *adopted* only when its correctness oracle passes before measured speed.
The component modules contain the primary paper and vendor-API citations.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Sequence

import torch
import torch.nn as nn

from rwkv_lab.nvfp4 import qualify_native_nvfp4, transformer_engine_nvfp4_status
from rwkv_lab.online_memory import OnlineAssociativeMemory, qualify_compiled_online_memory
from rwkv_lab.quantization import qualify_accelerated_nf4
from rwkv_lab.rosa_sam import (HAVE_CUDA as HAVE_CUDA_ROSA,
                               cuda_sam_retrieve_cf, cuda_sam_workspace_bytes,
                               sam_retrieve_cf)
from rwkv_lab.triangular_delta import (qualify_triangular_backend,
                                       stable_triangular_inverse)


def environment_report(device: str) -> dict:
    selected = torch.device(device)
    report = {
        "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
        "device": str(selected), "cuda_available": torch.cuda.is_available(),
    }
    if selected.type == "cuda" and torch.cuda.is_available():
        report.update({
            "device_name": torch.cuda.get_device_name(selected),
            "compute_capability": list(torch.cuda.get_device_capability(selected)),
        })
    available, reason = transformer_engine_nvfp4_status()
    report["transformer_engine_nvfp4"] = {"available": available, "reason": reason}
    return report


def qualify_recurrent_generation(model: nn.Module, prompt_ids: Sequence[int], *,
                                 device: str, max_new: int = 32,
                                 repeats: int = 3) -> dict:
    """Require token-exact recurrent generation before comparing throughput."""

    from rwkv_lab.generate import sample_with_stats

    reason = (model.recurrent_incompatibility() if
              hasattr(model, "recurrent_incompatibility") else
              "model does not expose a recurrent compatibility contract")
    if not hasattr(model, "forward_recurrent") or reason:
        return {"schema": "rwkv-lab.recurrent-serving-qualification.v1",
                "available": False, "adopted": False, "error": reason}
    prefix_rates, recurrent_rates = [], []
    reference = candidate = None
    for repeat in range(max(1, repeats)):
        reference, prefix_stats = sample_with_stats(
            model, list(prompt_ids), max_new=max_new, temperature=0,
            stop_at_sep=False, device=device, seed=17, engine="prefix")
        candidate, recurrent_stats = sample_with_stats(
            model, list(prompt_ids), max_new=max_new, temperature=0,
            stop_at_sep=False, device=device, seed=17, engine="recurrent")
        if repeat:
            prefix_rates.append(prefix_stats["tokens_per_second"])
            recurrent_rates.append(recurrent_stats["tokens_per_second"])
    # With one requested repeat, retain its timing rather than returning an empty sample.
    if not prefix_rates:
        prefix_rates.append(prefix_stats["tokens_per_second"])
        recurrent_rates.append(recurrent_stats["tokens_per_second"])
    prefix_rate = sorted(prefix_rates)[len(prefix_rates) // 2]
    recurrent_rate = sorted(recurrent_rates)[len(recurrent_rates) // 2]
    exact = candidate == reference
    speedup = recurrent_rate / max(prefix_rate, 1e-12)
    return {"schema": "rwkv-lab.recurrent-serving-qualification.v1",
            "available": True, "exact_tokens": exact, "tokens": len(candidate or ()),
            "prefix_tokens_per_second": prefix_rate,
            "recurrent_tokens_per_second": recurrent_rate, "speedup": speedup,
            "adopted": bool(exact and speedup >= 1.02)}


def qualify_cuda_rosa(*, device: str, repeats: int = 3, length: int = 1024,
                      routes: int = 1024, route_width: int = 4,
                      production_length: int = 4096,
                      production_routes: int = 1024) -> dict:
    """Parity, throughput, and long-context memory gate for hard ROSA's online SAM.

    The algorithm and counterfactual tables implement ROSA-Tuning
    (https://arxiv.org/abs/2602.02499); CPU Numba remains the independent oracle.
    """
    selected = torch.device(device)
    schema = "rwkv-lab.rosa-cuda-qualification.v1"
    if selected.type != "cuda" or not HAVE_CUDA_ROSA:
        return {"schema": schema, "available": False, "adopted": False,
                "error": "Numba CUDA ROSA kernel unavailable"}
    import numpy as np
    rng = np.random.default_rng(73)
    alphabet = 1 << route_width
    query = rng.integers(0, alphabet, (1, length, routes), dtype=np.int32)
    key = rng.integers(0, alphabet, (1, length, routes), dtype=np.int32)
    gpu_query = torch.from_numpy(query).to(selected)
    gpu_key = torch.from_numpy(key).to(selected)
    expected = sam_retrieve_cf(query, key, alphabet, route_width)
    cuda_sam_retrieve_cf(gpu_query[:, :8, :4].contiguous(),
                         gpu_key[:, :8, :4].contiguous(), alphabet, route_width)
    torch.cuda.synchronize(selected)

    blocking, native = [], []
    actual = None
    for _ in range(max(1, repeats)):
        started = time.perf_counter()
        q_cpu, k_cpu = gpu_query.cpu().numpy(), gpu_key.cpu().numpy()
        old = tuple(torch.from_numpy(value).to(selected)
                    for value in sam_retrieve_cf(q_cpu, k_cpu, alphabet, route_width))
        torch.cuda.synchronize(selected)
        blocking.append(time.perf_counter() - started)
        started = time.perf_counter()
        actual = cuda_sam_retrieve_cf(gpu_query, gpu_key, alphabet, route_width)
        torch.cuda.synchronize(selected)
        native.append(time.perf_counter() - started)
    exact = all(np.array_equal(want, got.cpu().numpy())
                for want, got in zip(expected, actual))
    blocking_s = sorted(blocking)[len(blocking) // 2]
    native_s = sorted(native)[len(native) // 2]
    speedup = blocking_s / max(native_s, 1e-12)
    workspace = cuda_sam_workspace_bytes(1, length, routes, alphabet)
    production_workspace = cuda_sam_workspace_bytes(
        1, production_length, production_routes, alphabet)
    total_memory = torch.cuda.get_device_properties(selected).total_memory
    memory_passed = production_workspace <= int(total_memory * 0.25)
    return {"schema": schema, "available": True, "exact": exact,
            "blocking_roundtrip_seconds": blocking_s,
            "device_native_seconds": native_s, "speedup": speedup,
            "workspace_bytes": workspace,
            "production_context": production_length,
            "production_routes": production_routes,
            "production_workspace_bytes": production_workspace,
            "production_memory_fraction": production_workspace / total_memory,
            "memory_passed": memory_passed,
            "adopted": bool(exact and speedup >= 1.02 and memory_passed)}


def compare_performance_baseline(current: dict, baseline: dict, *,
                                 max_throughput_regression: float = 0.05,
                                 max_memory_regression: float = 0.10,
                                 max_kernel_regression: float = 0.10) -> dict:
    """Fail a persisted qualification baseline on throughput, memory, or adoption loss."""
    reasons = []

    def walk(value, prefix=""):
        out = {}
        if isinstance(value, dict):
            for key, child in value.items():
                path = f"{prefix}.{key}" if prefix else key
                out.update(walk(child, path))
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            out[prefix] = float(value)
        return out

    now, before = walk(current), walk(baseline)
    compared = []
    for path, old in before.items():
        if path not in now or old <= 0:
            continue
        new = now[path]
        if path.endswith("tokens_per_second"):
            ratio = new / old
            compared.append({"metric": path, "baseline": old, "current": new, "ratio": ratio})
            if ratio < 1 - max_throughput_regression:
                reasons.append(f"{path} throughput regressed {(1-ratio)*100:.1f}%")
        elif path.endswith("device_native_seconds"):
            ratio = old / max(new, 1e-12)
            compared.append({"metric": path, "baseline": old, "current": new, "ratio": ratio})
            if ratio < 1 - max_throughput_regression:
                reasons.append(f"{path} latency regressed {(1-ratio)*100:.1f}%")
        elif path.endswith(("peak_memory_bytes", "production_workspace_bytes")):
            ratio = new / old
            compared.append({"metric": path, "baseline": old, "current": new, "ratio": ratio})
            if ratio > 1 + max_memory_regression:
                reasons.append(f"{path} memory grew {(ratio-1)*100:.1f}%")
        elif path.endswith("cuda_kernel_count"):
            ratio = new / old
            compared.append({"metric": path, "baseline": old, "current": new, "ratio": ratio})
            if ratio > 1 + max_kernel_regression:
                reasons.append(f"{path} kernel count grew {(ratio-1)*100:.1f}%")
    lost = sorted(set(baseline.get("adopted", ())) - set(current.get("adopted", ())))
    if lost:
        reasons.append("previously adopted backends lost: " + ", ".join(lost))
    return {"schema": "rwkv-lab.performance-regression-gate.v1",
            "passed": not reasons, "reasons": reasons, "compared": compared,
            "max_throughput_regression": max_throughput_regression,
            "max_memory_regression": max_memory_regression,
            "max_kernel_regression": max_kernel_regression}


def profile_kernel_evidence(work, *, device: torch.device) -> dict:
    """Count representative operators/CUDA kernels for persisted launch-regression evidence."""
    if device.type != "cuda":
        return {"available": False, "operator_count": 0, "cuda_kernel_count": 0}
    from torch.profiler import ProfilerActivity, profile
    work()
    torch.cuda.synchronize(device)
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as trace:
        work()
        torch.cuda.synchronize(device)
    events = trace.events()
    return {"available": True,
            "operator_count": sum(event.device_type == torch.autograd.DeviceType.CPU
                                  for event in events),
            "cuda_kernel_count": sum(event.device_type == torch.autograd.DeviceType.CUDA
                                     for event in events)}


def qualification_suite(*, device: str = "auto", compile_backend: str = "inductor",
                        repeats: int = 5, checkpoint: str = "",
                        prompt_ids: Sequence[int] = (1, 2, 3, 4),
                        max_new: int = 32) -> dict:
    device = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
    selected = torch.device(device)
    dtype = torch.bfloat16 if selected.type == "cuda" else torch.float32
    torch.manual_seed(41)
    if selected.type == "cuda":
        torch.cuda.reset_peak_memory_stats(selected)
    suite_started = time.perf_counter()

    nf4_linear = nn.Linear(512, 512, bias=False, device=selected, dtype=dtype)
    nf4_sample = torch.randn(16, 512, device=selected, dtype=dtype)
    nf4 = qualify_accelerated_nf4(
        nf4_linear, nf4_sample, repeats=repeats, minimum_speedup=1.02)

    if selected.type == "cuda":
        nv_linear = nn.Linear(1024, 1024, bias=False, device=selected, dtype=torch.bfloat16)
        nv_sample = torch.randn(8, 128, 1024, device=selected, dtype=torch.bfloat16)
        nvfp4 = qualify_native_nvfp4(
            nv_linear, nv_sample, repeats=repeats, minimum_speedup=1.02)
    else:
        nvfp4 = {"schema": "rwkv-lab.nvfp4-backend-qualification.v1",
                 "backend": "transformer_engine", "available": False,
                 "adopted": False, "error": "native NVFP4 requires CUDA"}

    memory = OnlineAssociativeMemory(128, d_memory=32, mode="titans").to(selected, dtype)
    memory_sample = torch.randn(2, 16, 128, device=selected, dtype=dtype)
    online_memory = qualify_compiled_online_memory(
        memory, memory_sample, compile_backend=compile_backend, repeats=repeats,
        tolerance=(2e-2 if dtype == torch.bfloat16 else 2e-5))
    triangular_sample = torch.eye(64, device=selected, dtype=dtype).repeat(4, 1, 1)
    triangular_sample.add_(torch.tril(
        torch.randn_like(triangular_sample) * 0.01, diagonal=-1))
    triangular = qualify_triangular_backend(
        triangular_sample,
        lambda value: stable_triangular_inverse(value, method="neumann"),
        repeats=repeats, tolerance=(2e-2 if dtype == torch.bfloat16 else 2e-4))
    def representative_training_step():
        memory.zero_grad(set_to_none=True)
        sample = memory_sample.detach().requires_grad_(True)
        memory(sample, record_stats=False).square().mean().backward()
    launch_profile = profile_kernel_evidence(representative_training_step, device=selected)
    rosa = qualify_cuda_rosa(device=device, repeats=max(1, min(repeats, 3)))

    recurrent = {"schema": "rwkv-lab.recurrent-serving-qualification.v1",
                 "available": False, "adopted": False,
                 "error": "pass --checkpoint to benchmark a real model"}
    if checkpoint:
        from rwkv_lab.generate import build_from_ckpt
        model, _ = build_from_ckpt(checkpoint, device)
        recurrent = qualify_recurrent_generation(
            model, prompt_ids, device=device, max_new=max_new, repeats=repeats)

    reports = {"nf4": nf4, "nvfp4": nvfp4, "rosa": rosa,
               "triangular_delta": triangular,
               "online_memory": online_memory, "recurrent_decode": recurrent,
               "eagle3": {
                   "available": True, "adopted": False,
                   "qualification_api": "rwkv_lab.speculative.qualify_speculative_greedy",
                   "reason": "adoption requires a trained draft head and target verifier",
               }}
    metrics = {"elapsed_seconds": time.perf_counter() - suite_started,
               "peak_memory_bytes": (torch.cuda.max_memory_allocated(selected)
                                     if selected.type == "cuda" else 0),
               "representative_training_profile": launch_profile}
    return {"schema": "rwkv-lab.production-kernel-qualification.v1",
            "environment": environment_report(device), "reports": reports,
            "metrics": metrics,
            "adopted": [name for name, report in reports.items() if report.get("adopted")],
            "policy": "exact/parity-qualified first; median throughput speedup >= 1.02 second"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Qualify production training/serving kernels")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--compile-backend", default="inductor")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--prompt-ids", default="1,2,3,4")
    parser.add_argument("--max-new", type=int, default=32)
    parser.add_argument("--output", default="")
    parser.add_argument("--baseline", default="",
                        help="prior qualification JSON; exits nonzero on performance regression")
    parser.add_argument("--max-throughput-regression", type=float, default=0.05)
    parser.add_argument("--max-memory-regression", type=float, default=0.10)
    parser.add_argument("--max-kernel-regression", type=float, default=0.10)
    args = parser.parse_args()
    prompt_ids = tuple(int(value) for value in args.prompt_ids.split(",") if value.strip())
    report = qualification_suite(
        device=args.device, compile_backend=args.compile_backend, repeats=args.repeats,
        checkpoint=args.checkpoint, prompt_ids=prompt_ids, max_new=args.max_new)
    if args.baseline:
        baseline = json.loads(Path(args.baseline).read_text())
        report["regression_gate"] = compare_performance_baseline(
            report, baseline,
            max_throughput_regression=args.max_throughput_regression,
            max_memory_regression=args.max_memory_regression,
            max_kernel_regression=args.max_kernel_regression)
    if args.output:
        Path(args.output).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))
    if args.baseline and not report["regression_gate"]["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
