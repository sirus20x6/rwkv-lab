"""
Convert every full-attention layer in Qwen3.6-35B-A3B from GQA to MLA.

Reads attention tensors directly from the source safetensors shards (avoids
loading the whole 35B model), runs gqa_to_mla_svd per layer, and writes a
self-contained patch containing only the modified attention weights plus the
full MLA/GQA config used. The patch is tiny compared to the base checkpoint.

Use load_converted.py to reconstruct a ready-to-train model:
    original model (from safetensors shards) + patch + MLAAttention modules

Output layout:
    converted/
      patch.safetensors       # modified attention weights, bf16
      manifest.json           # conversion metadata + old keys to drop

============================================================================
DO NOT SKIP BKV. EVER.
============================================================================

`--bkv-stats` is a hard requirement, not a convenience flag. The TransMLA
paper (arXiv:2502.07864, NeurIPS 2025) measures ~22% accuracy drop without
BKV vs ~1.65% with it. We have a real-world receipt: a full 8-hour 9B run
(qwen9b_mla_xsa_muon, h1_ppl 19.15 -> 2.530) was thrown away because we
skipped BKV. The naive SVD wastes capacity on layers where K_nope and V
have wildly different activation norms (we saw 3:1 imbalance at layer 31).

If you are about to run convert.py without --bkv-stats, stop. Run
build_bkv_stats.py first. It takes 15-20 minutes. Always.
============================================================================
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from svd_init import GQAConfig, MLAConfig, gqa_to_mla_svd
from layer_swap import gqa_config_from_hf
from safe_torch import safe_torch_load


ATTN_INPUT_KEYS = ["q_proj.weight", "k_proj.weight", "v_proj.weight",
                   "o_proj.weight", "q_norm.weight", "k_norm.weight"]

MTP_ATTN_PREFIX = "mtp.layers.0.self_attn."


def _backbone_prefix(layer_idx: int) -> str:
    return f"model.language_model.layers.{layer_idx}.self_attn."


def load_attn_sd_by_prefix(model_dir: Path, wmap: dict, prefix: str) -> dict:
    """Load the six attention tensors (q/k/v/o + q_norm/k_norm) under a given
    state-dict key prefix, e.g. 'model.language_model.layers.3.self_attn.' or
    'mtp.layers.0.self_attn.'."""
    sd = {}
    for k in ATTN_INPUT_KEYS:
        full = prefix + k
        with safe_open(model_dir / wmap[full], framework="pt") as f:
            sd[k] = f.get_tensor(full)
    return sd


def load_attn_sd(model_dir: Path, wmap: dict, layer_idx: int) -> dict:
    return load_attn_sd_by_prefix(model_dir, wmap, _backbone_prefix(layer_idx))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="/thearray/git/moe-mla/Qwen3.6-35B-A3B")
    ap.add_argument("--out-dir", default="/thearray/git/moe-mla/converted")
    ap.add_argument("--head-expansion", type=int, default=4,
                    help="multiply original num_heads by this factor")
    ap.add_argument("--kv-lora-rank", type=int, default=1024)
    ap.add_argument("--noise-std", type=float, default=1e-3,
                    help="symmetry-breaking Gaussian noise on duplicated heads")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--include-mtp", type=int, default=1,
                    help="1=also convert the MTP block's self_attn with the same MLA config "
                         "(4x heads etc). 0=leave MTP untouched.")
    ap.add_argument("--bkv-stats", type=str, default="",
                    help="REQUIRED. Path to bkv_stats.pt from build_bkv_stats.py. "
                         "Each full-attention layer uses the measured K/V activation-norm "
                         "ratio to rebalance the joint SVD (TransMLA BKV recipe). "
                         "Leaving this empty falls back to legacy unbalanced SVD, which is "
                         "FORBIDDEN — paper measures ~22pct downstream-acc drop vs ~1.65pct with BKV, "
                         "and we have direct evidence (an 8h 9B run discarded) that the "
                         "non-BKV recovery curve plateaus much higher. Empty path is allowed "
                         "only for archival reproduction of pre-BKV-era runs; new conversions "
                         "must build BKV stats first.")
    ap.add_argument("--allow-no-bkv", action="store_true",
                    help="ESCAPE HATCH for archival repro of pre-BKV runs. Required to bypass the "
                         "hard fail when --bkv-stats is empty. Do NOT use for new conversions.")
    args = ap.parse_args()

    # Hard fail: BKV is mandatory. The escape hatch exists only for reproducing
    # pre-BKV runs verbatim. See the file-level docstring for context.
    if not args.bkv_stats and not args.allow_no_bkv:
        raise SystemExit(
            "ERROR: --bkv-stats is required. Build it first with build_bkv_stats.py "
            "(takes 15-20 minutes), then re-run convert.py. To bypass for archival repro "
            "only, pass --allow-no-bkv. See the convert.py docstring for why this matters."
        )

    model_dir = Path(args.model_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg_full = json.loads((model_dir / "config.json").read_text())
    cfg = cfg_full["text_config"]
    wmap = json.loads((model_dir / "model.safetensors.index.json").read_text())["weight_map"]

    gqa_cfg = gqa_config_from_hf(cfg)
    d = gqa_cfg.head_dim
    # Qwen3.5-9B nests partial_rotary_factor inside rope_parameters; Qwen3.6-35B
    # duplicates it at both text_config root and under rope_parameters. Accept both.
    _prf = cfg.get("partial_rotary_factor")
    if _prf is None:
        _prf = cfg.get("rope_parameters", {}).get("partial_rotary_factor", 1.0)
    D_rope = int(d * _prf)
    D_nope = d - D_rope

    mla_cfg = MLAConfig(
        hidden_size=gqa_cfg.hidden_size,
        num_heads=gqa_cfg.num_q_heads * args.head_expansion,
        qk_nope_head_dim=D_nope,
        qk_rope_head_dim=D_rope,
        v_head_dim=d,
        kv_lora_rank=args.kv_lora_rank,
        has_output_gate=gqa_cfg.has_output_gate,
        has_qk_norm=gqa_cfg.has_qk_norm,
        num_kv_rope_heads=gqa_cfg.num_kv_heads,
    )
    full_attn_idx = [i for i, t in enumerate(cfg["layer_types"]) if t == "full_attention"]

    # Load BKV stats if provided. Format: dict[layer_idx -> {"ratio": float, ...}]
    bkv_stats = None
    if args.bkv_stats:
        bkv_stats = safe_torch_load(args.bkv_stats, map_location="cpu")
        n_covered = sum(1 for li in full_attn_idx if li in bkv_stats)
        print(f"BKV stats: {args.bkv_stats}  ({n_covered}/{len(full_attn_idx)} layers covered)")

    print(f"source:  {model_dir}")
    print(f"output:  {out_dir}")
    print(f"GQA:     Nq={gqa_cfg.num_q_heads} Nkv={gqa_cfg.num_kv_heads} d={d} "
          f"rope={D_rope}+nope={D_nope}  gate={gqa_cfg.has_output_gate}  qk_norm={gqa_cfg.has_qk_norm}")
    print(f"MLA:     Nh={mla_cfg.num_heads} R={mla_cfg.kv_lora_rank} "
          f"Nkr={mla_cfg.num_kv_rope_heads}  noise={args.noise_std}  "
          f"bkv={'on' if bkv_stats else 'off'}")
    print(f"layers:  {full_attn_idx}  ({len(full_attn_idx)} full-attention layers)")
    print()

    patch: dict[str, torch.Tensor] = {}
    dropped_keys: list[str] = []
    total_old_params = 0
    total_new_params = 0

    t0 = time.time()
    for li in full_attn_idx:
        t1 = time.time()
        sd = load_attn_sd(model_dir, wmap, li)
        old_param_count = sum(v.numel() for v in sd.values())
        balance_kv = None
        if bkv_stats is not None and li in bkv_stats:
            balance_kv = float(bkv_stats[li]["ratio"])
        mla_sd = gqa_to_mla_svd(
            sd, gqa_cfg, mla_cfg,
            head_expand_noise_std=args.noise_std,
            rng_seed=args.seed + li,
            balance_kv=balance_kv,
        )
        new_param_count = sum(v.numel() for v in mla_sd.values())
        total_old_params += old_param_count
        total_new_params += new_param_count

        prefix = f"model.language_model.layers.{li}.self_attn."
        for k, v in mla_sd.items():
            patch[prefix + k] = v
        for k in ATTN_INPUT_KEYS:
            dropped_keys.append(prefix + k)

        print(f"  layer {li:2d}: {old_param_count/1e6:7.2f}M -> {new_param_count/1e6:7.2f}M  "
              f"({time.time()-t1:.1f}s)")

    # ---- MTP block ----
    mtp_converted = False
    if args.include_mtp:
        # Verify presence of MTP attention tensors in the checkpoint index
        mtp_present = all((MTP_ATTN_PREFIX + k) in wmap for k in ATTN_INPUT_KEYS)
        if not mtp_present:
            print(f"  (no MTP attention tensors found at prefix '{MTP_ATTN_PREFIX}'; skipping MTP)")
        else:
            t1 = time.time()
            sd = load_attn_sd_by_prefix(model_dir, wmap, MTP_ATTN_PREFIX)
            old_param_count = sum(v.numel() for v in sd.values())
            balance_kv_mtp = None
            if bkv_stats is not None and "mtp" in bkv_stats:
                balance_kv_mtp = float(bkv_stats["mtp"]["ratio"])
            mla_sd = gqa_to_mla_svd(
                sd, gqa_cfg, mla_cfg,
                head_expand_noise_std=args.noise_std,
                rng_seed=args.seed + 10_000,  # distinct seed from backbone layers
                balance_kv=balance_kv_mtp,
            )
            new_param_count = sum(v.numel() for v in mla_sd.values())
            total_old_params += old_param_count
            total_new_params += new_param_count
            for k, v in mla_sd.items():
                patch[MTP_ATTN_PREFIX + k] = v
            for k in ATTN_INPUT_KEYS:
                dropped_keys.append(MTP_ATTN_PREFIX + k)
            mtp_converted = True
            print(f"  MTP     : {old_param_count/1e6:7.2f}M -> {new_param_count/1e6:7.2f}M  "
                  f"({time.time()-t1:.1f}s)")

    print()
    print(f"total: {total_old_params/1e6:.2f}M -> {total_new_params/1e6:.2f}M  "
          f"(+{(total_new_params-total_old_params)/1e6:.2f}M, "
          f"{100*(total_new_params-total_old_params)/total_old_params:+.1f}%)")
    print(f"elapsed: {time.time()-t0:.1f}s")

    # Save patch as bf16 to match the source checkpoint dtype.
    patch_bf16 = {k: v.to(torch.bfloat16).contiguous() for k, v in patch.items()}
    save_file(patch_bf16, out_dir / "patch.safetensors")

    manifest = {
        "source_checkpoint": str(model_dir),
        "full_attn_layer_indices": full_attn_idx,
        "mtp_converted": mtp_converted,
        "mtp_attn_prefix": MTP_ATTN_PREFIX if mtp_converted else None,
        "dropped_keys": dropped_keys,
        "new_keys": sorted(patch.keys()),
        "gqa_config": asdict(gqa_cfg),
        "mla_config": asdict(mla_cfg),
        "conversion_args": {
            "head_expansion": args.head_expansion,
            "kv_lora_rank": args.kv_lora_rank,
            "noise_std": args.noise_std,
            "seed": args.seed,
            "include_mtp": bool(args.include_mtp),
            "bkv_stats": args.bkv_stats or None,
            "bkv_enabled": bkv_stats is not None,
        },
        "param_count_attn_before": total_old_params,
        "param_count_attn_after": total_new_params,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"wrote: {out_dir/'patch.safetensors'}  ({(out_dir/'patch.safetensors').stat().st_size/1e6:.1f} MB)")
    print(f"wrote: {out_dir/'manifest.json'}")


if __name__ == "__main__":
    main()
