"""
Compute per-layer K_nope / V activation L2 norms for BKV-style MLA conversion.

============================================================================
THIS IS A MANDATORY PRE-STEP FOR `convert.py`. NEVER SKIP IT.
============================================================================

convert.py refuses to run without --bkv-stats. The naive SVD fallback wastes
capacity allocating equal singular-value budget to K_nope and V even when
their activation norms differ by 3:1 (we've measured this on layer 31 of
Qwen3.5-9B). A non-BKV conversion plateaus much higher in recovery training
than a BKV one — we proved this the hard way by discarding an 8-hour 9B run.

If you somehow find yourself converting without BKV: stop, run this, redo.
============================================================================

For each full-attention layer in the ORIGINAL GQA model (plus the MTP block's
attention if present), forward a small calibration set through the model,
capture k_proj and v_proj outputs, and compute the mean per-dim L2 norm
across all tokens. The scalar ratio `||K_nope|| / ||V||` is what BKV uses
to rebalance K vs V before the joint low-rank SVD.

Paper: TransMLA (NeurIPS 2025, arXiv:2502.07864). The rebalancing preserves
the forward pass exactly at full rank (via paired k_proj /= r, k_up_proj *= r)
but changes which principal components survive truncation. Per paper, skipping
this gives ~22% accuracy drop at 93% KV compression; doing it properly: ~1.65%.

Output:
    bkv_stats.pt — dict keyed by layer_idx (int) and "mtp" (str), each value:
        {"ratio": float, "k_nope_norm": float, "v_norm": float,
         "k_nope_norm_per_dim": Tensor[D_nope],
         "v_norm_per_dim":      Tensor[D_v],
         "n_tokens": int}

Usage:
    python build_bkv_stats.py \\
        --calibration wikitext \\
        --nsamples 128 --seqlen 2048 \\
        --out /thearray/git/moe-mla/converted/bkv_stats.pt

    # Then convert with BKV:
    python convert.py --bkv-stats /thearray/git/moe-mla/converted/bkv_stats.pt \\
                      --out-dir converted_bkv
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch


def _load_calibration_wikitext(tokenizer, nsamples: int, seqlen: int) -> torch.Tensor:
    """Pull wikitext-2 raw validation split, tokenize, pack into [nsamples, seqlen]
    non-overlapping windows."""
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
    text = "\n\n".join(s for s in ds["text"] if s.strip())
    ids = tokenizer(text, return_tensors="pt").input_ids[0]
    total = nsamples * seqlen
    if ids.numel() < total:
        # Concat-and-tile if the corpus is short.
        reps = (total + ids.numel() - 1) // ids.numel()
        ids = ids.repeat(reps)[:total]
    else:
        ids = ids[:total]
    return ids.view(nsamples, seqlen)


def _load_calibration_tokens_bin(path: str, total_tokens: int, nsamples: int,
                                 seqlen: int, seed: int) -> torch.Tensor:
    """Sample random non-overlapping windows from our pre-tokenized bin."""
    arr = np.memmap(path, dtype=np.uint32, mode="r")
    n_valid = min(total_tokens, arr.shape[0]) - seqlen - 1
    if n_valid <= 0:
        raise ValueError(f"bin too short for seqlen={seqlen}")
    rng = np.random.default_rng(seed)
    starts = rng.integers(low=0, high=n_valid, size=nsamples)
    offsets = np.arange(seqlen, dtype=np.int64)
    idx = starts[:, None].astype(np.int64) + offsets[None, :]
    out = arr[idx.reshape(-1)].astype(np.int64).reshape(nsamples, seqlen)
    return torch.from_numpy(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="/thearray/git/moe-mla/Qwen3.6-35B-A3B")
    ap.add_argument("--calibration", choices=["wikitext", "tokens-bin"], default="wikitext",
                    help="wikitext = HF wikitext-2-raw-v1 validation (retokenized with Qwen "
                         "tokenizer). tokens-bin = random slice of --tokens-bin (faster, in "
                         "the distribution we train on).")
    ap.add_argument("--tokens-bin", default="/thearray/data/non_cvevc_tokens.bin")
    ap.add_argument("--tokens-bin-total", type=int, default=29_284_583_603)
    ap.add_argument("--nsamples", type=int, default=128)
    ap.add_argument("--seqlen", type=int, default=2048,
                    help="longer = more tokens per sample but more GPU mem. "
                         "For BKV stats 512-2048 is fine; we want statistics, "
                         "not per-token quality.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--include-mtp", type=int, default=1)
    ap.add_argument("--out", default="/thearray/git/moe-mla/converted/bkv_stats.pt")
    args = ap.parse_args()

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]

    model_dir = Path(args.model_dir)
    cfg_full = json.loads((model_dir / "config.json").read_text())
    cfg = cfg_full["text_config"]

    full_attn_idx = [i for i, t in enumerate(cfg["layer_types"]) if t == "full_attention"]
    num_kv_heads = cfg["num_key_value_heads"]
    head_dim = cfg.get("head_dim") or (cfg["hidden_size"] // cfg["num_attention_heads"])
    # Qwen3.5-9B nests partial_rotary_factor inside rope_parameters; 35B has it
    # at both root and inside rope_parameters. Accept either.
    partial_rotary = cfg.get("partial_rotary_factor")
    if partial_rotary is None:
        partial_rotary = cfg.get("rope_parameters", {}).get("partial_rotary_factor", 1.0)
    D_rope = int(head_dim * partial_rotary)
    D_nope = head_dim - D_rope
    # Qwen3 uses rope_position="first" (confirmed via our gqa_config_from_hf).
    # In raw k_proj output, each head's layout is [rope(D_rope) | nope(D_nope)].
    rope_first = True

    print(f"model:  {model_dir}")
    print(f"config: num_kv_heads={num_kv_heads}  head_dim={head_dim}  "
          f"D_rope={D_rope}  D_nope={D_nope}  rope_first={rope_first}")
    print(f"full-attention layers: {full_attn_idx}")
    print(f"calibration: {args.calibration}  nsamples={args.nsamples}  seqlen={args.seqlen}")

    # ---- Load tokenizer + calibration ids ----
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if args.calibration == "wikitext":
        cal_ids = _load_calibration_wikitext(tokenizer, args.nsamples, args.seqlen)
    else:
        cal_ids = _load_calibration_tokens_bin(
            args.tokens_bin, args.tokens_bin_total,
            args.nsamples, args.seqlen, args.seed,
        )
    print(f"calibration ids: {tuple(cal_ids.shape)}  total={cal_ids.numel()} tokens")

    # ---- Load model ----
    print("loading model (this can take ~1 minute)...")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        torch_dtype=dtype,
        device_map=args.device,
        trust_remote_code=True,
        attn_implementation="sdpa",
    )
    model.eval()
    print(f"  loaded in {time.time()-t0:.1f}s")

    # ---- Resolve decoder layers ----
    inner = model.model
    if hasattr(inner, "language_model"):
        inner = inner.language_model
    layers = inner.layers
    num_layers = len(layers)
    print(f"num decoder layers: {num_layers}")

    # ---- Attach hooks on k_proj and v_proj of each targeted layer ----
    # Collect per-dim squared-sum and token count. Post-hook on nn.Linear gives us
    # the raw output before head reshape, with shape [B, T, num_kv_heads*head_dim].
    stats = {}  # layer_idx (int) or "mtp" (str) -> dict
    hooks = []

    def make_hook(key, which):
        # key = layer_idx int or "mtp"; which = "k" or "v"
        def hook(module, inputs, output):
            # output: [B, T, num_kv_heads*head_dim]
            B, T, Dtotal = output.shape
            # reshape into heads; K has [rope|nope], V has just [v] (==head_dim)
            out = output.reshape(-1, num_kv_heads, head_dim)
            if which == "k":
                if rope_first:
                    k_nope = out[:, :, D_rope:]  # [N, n_kv, D_nope]
                else:
                    k_nope = out[:, :, :D_nope]
                # per-dim squared sum across all tokens and heads (flatten heads for "mean" norm)
                flat = k_nope.reshape(-1, D_nope).float()
                stats.setdefault(key, {})
                d = stats[key]
                sq = (flat * flat).sum(dim=0).cpu()
                d["k_sq"] = sq if "k_sq" not in d else d["k_sq"] + sq
                d["k_n"] = d.get("k_n", 0) + flat.shape[0]
            else:  # v
                flat = out.reshape(-1, head_dim).float()
                stats.setdefault(key, {})
                d = stats[key]
                sq = (flat * flat).sum(dim=0).cpu()
                d["v_sq"] = sq if "v_sq" not in d else d["v_sq"] + sq
                d["v_n"] = d.get("v_n", 0) + flat.shape[0]
        return hook

    for li in full_attn_idx:
        attn = layers[li].self_attn
        hooks.append(attn.k_proj.register_forward_hook(make_hook(li, "k")))
        hooks.append(attn.v_proj.register_forward_hook(make_hook(li, "v")))

    mtp = getattr(model, "mtp", None) or getattr(model.model, "mtp", None)
    if mtp is not None and args.include_mtp:
        try:
            mtp_attn = mtp.layers[0].self_attn
            hooks.append(mtp_attn.k_proj.register_forward_hook(make_hook("mtp", "k")))
            hooks.append(mtp_attn.v_proj.register_forward_hook(make_hook("mtp", "v")))
            print("hooked MTP self_attn")
        except (AttributeError, IndexError) as e:
            print(f"skipping MTP: {e}")

    # ---- Forward pass ----
    print(f"forwarding {args.nsamples} samples...")
    t0 = time.time()
    with torch.no_grad():
        for i in range(args.nsamples):
            ids = cal_ids[i:i+1].to(args.device)
            _ = model(input_ids=ids)
            if (i + 1) % 16 == 0:
                print(f"  {i+1}/{args.nsamples}  ({time.time()-t0:.0f}s)")
    print(f"  done in {time.time()-t0:.1f}s")

    for h in hooks:
        h.remove()

    # ---- Compute ratio per layer ----
    out_stats = {}
    for key, d in stats.items():
        k_per_dim = (d["k_sq"] / max(1, d["k_n"])).sqrt()   # [D_nope]
        v_per_dim = (d["v_sq"] / max(1, d["v_n"])).sqrt()   # [D_v]
        k_scalar = float(k_per_dim.mean())
        v_scalar = float(v_per_dim.mean())
        ratio = k_scalar / max(v_scalar, 1e-12)
        out_stats[key] = {
            "ratio":               ratio,
            "k_nope_norm":         k_scalar,
            "v_norm":              v_scalar,
            "k_nope_norm_per_dim": k_per_dim,
            "v_norm_per_dim":      v_per_dim,
            "n_tokens":            int(d["k_n"]),
        }

    # ---- Report + save ----
    print("\nBKV ratios (||K_nope|| / ||V||):")
    for key in sorted(out_stats.keys(), key=lambda x: (isinstance(x, str), x)):
        s = out_stats[key]
        label = f"layer {key}" if isinstance(key, int) else key
        print(f"  {label:>10s}:  k={s['k_nope_norm']:7.4f}  v={s['v_norm']:7.4f}  "
              f"ratio={s['ratio']:6.3f}  ({s['n_tokens']} tokens)")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out_stats, out_path)
    print(f"\nwrote: {out_path}")


if __name__ == "__main__":
    main()
