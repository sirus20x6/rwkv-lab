"""Load a checkpoint and run eval only (no training) — for diagnosing whether
eval metric differences are sample variance or real model differences."""
from __future__ import annotations

import argparse
import sys

import torch
import numpy as np

sys.path.insert(0, "/thearray/git/moe-mla")
from load_converted import load_converted_model
from train_mla import TrainConfig, multi_horizon_eval, open_tokens
from safe_torch import safe_torch_load


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", required=True)
    ap.add_argument("--eval-batches", type=int, default=8)
    ap.add_argument("--tokens-bin", default="/thearray/data/non_cvevc_tokens.bin")
    ap.add_argument("--total-tokens-in-bin", type=int, default=29284583603)
    ap.add_argument("--model-dir", default=None,
                    help="overrides TrainConfig default (which is the 35B path)")
    ap.add_argument("--patch-dir", default=None)
    ap.add_argument("--rwkv8-deltanet-layers", default=None,
                    help="comma-separated layer indices, e.g. '28,29,30'")
    ap.add_argument("--rwkv8-swap-mode", default="timemix")
    args = ap.parse_args()

    cfg_kwargs = dict(
        tokens_bin=args.tokens_bin,
        total_tokens_in_bin=args.total_tokens_in_bin,
        eval_batches=args.eval_batches,
        install_mtp=1,
        train_mtp_only=1,  # match Phase 1 mode so freeze patterns align
    )
    if args.model_dir is not None:
        cfg_kwargs["model_dir"] = args.model_dir
    if args.patch_dir is not None:
        cfg_kwargs["patch_dir"] = args.patch_dir
    cfg = TrainConfig(**cfg_kwargs)

    model, mla_modules = load_converted_model(
        model_dir=cfg.model_dir, patch_dir=cfg.patch_dir,
        device_map="cuda:0", dtype=torch.bfloat16, freeze_non_mla=True,
        install_mtp=bool(cfg.install_mtp),
        rwkv8_deltanet_layers=args.rwkv8_deltanet_layers,
        rwkv8_swap_mode=args.rwkv8_swap_mode,
    )

    print(f"resuming from: {args.resume}")
    ckpt = safe_torch_load(args.resume, map_location="cpu")
    saved_mla = ckpt["mla_state_dicts"]
    for m in mla_modules:
        key = getattr(m, "_save_key", None)
        if key is None or key not in saved_mla:
            continue
        sd = {k: v.to(device=next(m.parameters()).device,
                      dtype=next(m.parameters()).dtype)
              for k, v in saved_mla[key].items()}
        m.load_state_dict(sd, strict=False)

    if "mtp_extra_state_dict" in ckpt and getattr(model, "mtp_trainer", None) is not None:
        mtp = model.mtp_trainer
        named_params = dict(mtp.named_parameters())
        loaded = 0
        for name, v in ckpt["mtp_extra_state_dict"].items():
            p = named_params.get(name)
            if p is None: continue
            with torch.no_grad():
                p.data.copy_(v.to(device=p.device, dtype=p.dtype))
            loaded += 1
        print(f"mtp_extra: {loaded} tensors loaded")

    if "backbone_state_dict" in ckpt:
        sd = ckpt["backbone_state_dict"]
        named = dict(model.named_parameters())
        loaded, skipped = 0, 0
        for name, v in sd.items():
            p = named.get(name)
            if p is None:
                skipped += 1
                continue
            with torch.no_grad():
                p.data.copy_(v.to(device=p.device, dtype=p.dtype))
            loaded += 1
        n_params = sum(t.numel() for t in sd.values())
        print(f"backbone_state_dict: restored {loaded} tensors "
              f"({n_params/1e9:.2f}B params), skipped {skipped}")
    else:
        print("[warn] no backbone_state_dict in ckpt — using base safetensors backbone")

    arr, train_end, eval_end = open_tokens(cfg)
    eval_start = train_end
    device = next(model.parameters()).device

    results = multi_horizon_eval(model, arr, eval_start, eval_end, cfg, device,
                                 horizons=(1, 2, 3, 4))
    print(f"\neval_batches={args.eval_batches}")
    for k in (1, 2, 3, 4):
        if k in results:
            r = results[k]
            print(f"  h={k}  ppl={r['ppl']:7.3f}  "
                  f"top1={r['top1_acc']*100:6.3f}%  top5={r['top5_acc']*100:6.3f}%  "
                  f"({r['tokens']} tokens)")


if __name__ == "__main__":
    main()
