"""
Wall-clock comparison of base Qwen3.6-35B-A3B vs our MLA-converted model on:
  - Prefill (single forward over N-token prompt)
  - Autoregressive decode (generate M tokens)

Loads each model in turn (can't fit both on one GPU), runs warmup + 3 trials,
reports median tokens/sec for each phase.
"""
from __future__ import annotations

import argparse
import gc
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from load_converted import load_converted_model


PROMPT_LEN = 1024
DECODE_LEN = 64
WARMUP = 2
TRIALS = 3


def _fresh_gpu() -> None:
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


@torch.no_grad()
def bench(model, input_ids: torch.Tensor, label: str,
          cache_decode: bool = True) -> dict:
    device = input_ids.device

    prefill_times = []
    decode_times = []
    for trial in range(WARMUP + TRIALS):
        _fresh_gpu()

        # Prefill
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = model(input_ids=input_ids, use_cache=cache_decode)
        torch.cuda.synchronize()
        prefill_dt = time.perf_counter() - t0
        last = out.logits[:, -1:, :].argmax(dim=-1)

        # Decode
        decode_dt = None
        if cache_decode:
            past = out.past_key_values
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(DECODE_LEN):
                out = model(input_ids=last, past_key_values=past, use_cache=True)
                past = out.past_key_values
                last = out.logits[:, -1:, :].argmax(dim=-1)
            torch.cuda.synchronize()
            decode_dt = time.perf_counter() - t0

        if trial >= WARMUP:
            prefill_times.append(prefill_dt)
            if decode_dt is not None:
                decode_times.append(decode_dt)
                decode_msg = f"decode={decode_dt*1000:.0f}ms ({DECODE_LEN/decode_dt:.1f} tok/s)"
            else:
                decode_msg = "decode=N/A (MLA KV-cache unsupported)"
            print(f"  [{label}] trial {trial-WARMUP+1}: "
                  f"prefill={prefill_dt*1000:.0f}ms {decode_msg}")

    decode_med = np.median(decode_times) if decode_times else None
    return {
        "prefill_ms": np.median(prefill_times) * 1000,
        "prefill_toks_per_sec": PROMPT_LEN / np.median(prefill_times),
        "decode_ms": None if decode_med is None else decode_med * 1000,
        "decode_toks_per_sec": None if decode_med is None else DECODE_LEN / decode_med,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="/thearray/git/moe-mla/Qwen3.6-35B-A3B")
    ap.add_argument("--patch-dir", default="/thearray/git/moe-mla/converted")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model_dir)
    # Use a random-ish prompt of exactly PROMPT_LEN tokens. Repeat a long doc.
    prompt = "The quick brown fox jumps over the lazy dog. " * 200
    ids = tok(prompt, return_tensors="pt").input_ids[:, :PROMPT_LEN].to("cuda:0")
    print(f"prompt: {ids.shape[1]} tokens, decode: {DECODE_LEN} tokens")
    print()

    # ========== Base ==========
    print("=== Loading BASE Qwen3.6-35B-A3B ===")
    base = AutoModelForCausalLM.from_pretrained(
        args.model_dir, dtype=torch.bfloat16, device_map="cuda:0",
    )
    base.eval()
    base_stats = bench(base, ids, "base")
    del base
    _fresh_gpu()

    # ========== MLA ==========
    print("\n=== Loading MLA-converted model ===")
    mla, _ = load_converted_model(
        model_dir=args.model_dir, patch_dir=args.patch_dir,
        device_map="cuda:0", dtype=torch.bfloat16, freeze_non_mla=False,
    )
    mla.eval()
    mla_stats = bench(mla, ids, "mla", cache_decode=False)

    # ========== Report ==========
    print()
    print(f"{'metric':<28s} {'base':>14s} {'mla':>14s} {'mla/base':>10s}")
    print("-" * 70)
    for k in ("prefill_ms", "prefill_toks_per_sec", "decode_ms", "decode_toks_per_sec"):
        b = base_stats[k]
        m = mla_stats[k]
        if m is None:
            print(f"{k:<28s} {b:>14.2f} {'N/A':>14s} {'N/A':>10s}")
        else:
            ratio = m / b
            print(f"{k:<28s} {b:>14.2f} {m:>14.2f} {ratio:>10.3f}x")


if __name__ == "__main__":
    main()
