#!/usr/bin/env python
"""Eval the ORIGINAL untouched Qwen3.5-9B-Base on the same eval windows the
isolation runs use, and cache it as runs/_baseline.json so the dashboard can show
it as the reference 'first point'. One-shot; the original model never changes.

Uses convert_train.evaluate with the SAME defaults (seed=12345) so the number is
directly comparable to the runs' eval points.
"""
from __future__ import annotations
import sys
sys.modules.setdefault("torchvision", None)
import argparse, json
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM
from build_memory_targets import load_token_stream
from convert_train import evaluate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="Qwen3.5-9B-Base")
    ap.add_argument("--data", default="/thearray/git/babyllm/data/cache/qwen3.6_fwedu_train")
    ap.add_argument("--eval-windows", type=int, default=8)
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--out", default="runs/_baseline.json")
    args = ap.parse_args()

    print(f"loading original {args.model_dir} ...", flush=True)
    m = AutoModelForCausalLM.from_pretrained(
        args.model_dir, dtype=getattr(torch, args.dtype), low_cpu_mem_usage=True).to(args.device).eval()
    tm = getattr(m.model, "language_model", m.model)
    toks = load_token_stream(args.data)
    ev = evaluate(tm, m.lm_head, toks, args.eval_windows, args.seq_len, args.device)
    rec = {"ppl": ev["ppl"], "loss": ev["loss"], "top1_acc": ev["top1_acc"],
           "model_dir": args.model_dir, "data": args.data,
           "eval_windows": args.eval_windows, "seq_len": args.seq_len}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(rec, indent=2))
    print(f"BASELINE ppl={ev['ppl']:.3f} loss={ev['loss']:.4f} top1={ev['top1_acc']:.4f} -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
