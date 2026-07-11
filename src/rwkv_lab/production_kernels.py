"""Fail-closed qualification for production training and serving kernels.

Hardware backends remain optional.  A backend is *available* when it can run;
it is *adopted* only when its correctness oracle passes before measured speed.
The component modules contain the primary paper and vendor-API citations.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import torch
import torch.nn as nn

from rwkv_lab.nvfp4 import qualify_native_nvfp4, transformer_engine_nvfp4_status
from rwkv_lab.online_memory import OnlineAssociativeMemory, qualify_compiled_online_memory
from rwkv_lab.quantization import qualify_accelerated_nf4


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


def qualification_suite(*, device: str = "auto", compile_backend: str = "inductor",
                        repeats: int = 5, checkpoint: str = "",
                        prompt_ids: Sequence[int] = (1, 2, 3, 4),
                        max_new: int = 32) -> dict:
    device = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
    selected = torch.device(device)
    dtype = torch.bfloat16 if selected.type == "cuda" else torch.float32
    torch.manual_seed(41)

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

    recurrent = {"schema": "rwkv-lab.recurrent-serving-qualification.v1",
                 "available": False, "adopted": False,
                 "error": "pass --checkpoint to benchmark a real model"}
    if checkpoint:
        from rwkv_lab.generate import build_from_ckpt
        model, _ = build_from_ckpt(checkpoint, device)
        recurrent = qualify_recurrent_generation(
            model, prompt_ids, device=device, max_new=max_new, repeats=repeats)

    reports = {"nf4": nf4, "nvfp4": nvfp4,
               "online_memory": online_memory, "recurrent_decode": recurrent,
               "eagle3": {
                   "available": True, "adopted": False,
                   "qualification_api": "rwkv_lab.speculative.qualify_speculative_greedy",
                   "reason": "adoption requires a trained draft head and target verifier",
               }}
    return {"schema": "rwkv-lab.production-kernel-qualification.v1",
            "environment": environment_report(device), "reports": reports,
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
    args = parser.parse_args()
    prompt_ids = tuple(int(value) for value in args.prompt_ids.split(",") if value.strip())
    report = qualification_suite(
        device=args.device, compile_backend=args.compile_backend, repeats=args.repeats,
        checkpoint=args.checkpoint, prompt_ids=prompt_ids, max_new=args.max_new)
    if args.output:
        Path(args.output).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
