"""
PPL comparison on the non-cvevc merged token stream for:
  (1) base Qwen3.6-35B-A3B (untouched)
  (2) MLA-converted model with our latest trained patch loaded

Uses the SAME random windows for both models so the comparison is apples-to-apples.
Loads each model in turn (both won't fit on one GPU).
"""
from __future__ import annotations

import argparse
import gc
import math
import time

import numpy as np
import torch

from train_mla import chunked_ce
from load_converted import load_converted_model
from mla_module import MLAAttention
from safe_torch import safe_torch_load


@torch.no_grad()
def run_eval(model, arr: np.memmap, seq_len: int, n_batches: int,
             seed: int, device: torch.device, label: str) -> tuple[float, float]:
    model.eval()
    rng = np.random.default_rng(seed)
    total_loss = 0.0
    total_tokens = 0
    t0 = time.time()
    for i in range(n_batches):
        max_start = arr.shape[0] - (seq_len + 1)
        start = int(rng.integers(low=0, high=max_start))
        window = arr[start : start + seq_len + 1].astype(np.int64)
        ids = torch.from_numpy(window).unsqueeze(0).to(device)
        x, y = ids[:, :-1], ids[:, 1:]
        logits = model(input_ids=x).logits
        loss = chunked_ce(logits, y) * y.numel()
        total_loss += loss.item()
        total_tokens += y.numel()
        del ids, x, y, logits, loss
        if (i + 1) % 16 == 0:
            print(f"  [{label}] batch {i+1:3d}/{n_batches}  "
                  f"running loss={total_loss/total_tokens:.4f}")
    mean_loss = total_loss / total_tokens
    print(f"  [{label}] done in {time.time()-t0:.0f}s   "
          f"loss={mean_loss:.4f}  ppl={math.exp(mean_loss):.4f}")
    return mean_loss, math.exp(mean_loss)


def _fresh_gpu() -> None:
    gc.collect()
    torch.cuda.empty_cache()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", default="/thearray/data/non_cvevc_tokens.bin")
    ap.add_argument("--model-dir", default="/thearray/git/moe-mla/Qwen3.6-35B-A3B")
    ap.add_argument("--patch-dir", default="/thearray/git/moe-mla/converted")
    ap.add_argument("--trained-ckpt", default="",
                    help="path to ckpt.pt to load into the MLA modules on top of the patch-init. "
                         "If empty, use the patch init state (i.e. untrained MLA).")
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--n-batches", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    arr = np.memmap(args.bin, dtype=np.uint32, mode="r")
    print(f"eval source: {args.bin}  ({arr.shape[0]:,} tokens, "
          f"{arr.shape[0]/1e9:.2f}B)")
    print(f"eval plan:   {args.n_batches} random windows × {args.seq_len} tokens "
          f"= {args.n_batches*args.seq_len:,} tokens/model")
    print()

    # ====== Base ======
    print("=== Loading BASE Qwen3.6-35B-A3B ===")
    from transformers import AutoModelForCausalLM
    base = AutoModelForCausalLM.from_pretrained(
        args.model_dir, dtype=torch.bfloat16, device_map="cuda:0",
    )
    device = next(base.parameters()).device
    base_loss, base_ppl = run_eval(base, arr, args.seq_len, args.n_batches,
                                    args.seed, device, label="base")
    del base
    _fresh_gpu()

    # ====== MLA ======
    print()
    print("=== Loading MLA-converted model ===")
    mla, mla_modules = load_converted_model(
        model_dir=args.model_dir, patch_dir=args.patch_dir,
        device_map="cuda:0", dtype=torch.bfloat16, freeze_non_mla=False,
    )
    if args.trained_ckpt:
        print(f"applying trained checkpoint on top: {args.trained_ckpt}")
        ckpt = safe_torch_load(args.trained_ckpt, map_location="cpu")
        import json
        from pathlib import Path as _P
        fai = json.loads((_P(args.patch_dir) / "manifest.json").read_text())["full_attn_layer_indices"]
        for li, m in zip(fai, mla_modules):
            sd = {k: v.to(device=next(m.parameters()).device, dtype=next(m.parameters()).dtype)
                  for k, v in ckpt["mla_state_dicts"][f"layer_{li}"].items()}
            m.load_state_dict(sd, strict=False)
        print(f"trained state loaded (step {ckpt['step']})")
        del ckpt
        _fresh_gpu()

    mla_loss, mla_ppl = run_eval(mla, arr, args.seq_len, args.n_batches,
                                  args.seed, device, label="mla")
    del mla
    _fresh_gpu()

    # ====== Report ======
    print()
    print(f"{'model':<12s} {'loss':>10s} {'ppl':>10s} {'Δloss vs base':>16s} {'Δppl %':>10s}")
    print("-" * 64)
    print(f"{'base':<12s} {base_loss:>10.4f} {base_ppl:>10.4f}")
    delta_loss = mla_loss - base_loss
    ppl_pct = 100 * (mla_ppl - base_ppl) / base_ppl
    label = "mla (trained)" if args.trained_ckpt else "mla (init)"
    print(f"{label:<12s} {mla_loss:>10.4f} {mla_ppl:>10.4f} {delta_loss:>+16.4f} {ppl_pct:>+10.2f}%")


if __name__ == "__main__":
    main()
