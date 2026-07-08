#!/usr/bin/env python
"""Fan out the I/O-first isolation conversion across all GDN layers, sequentially.

Per the io_vs_state_conflict finding, an isolated GDN→RWKV layer reaches base ppl
fastest on an I/O-only objective (block + lm_ce; no SMT/DMT, no codec). This driver
runs `convert_train.py` once per GDN layer with that recipe, warm-starting from the
existing per-layer library where available (converted_layers_lib/L##.pt) else
GDN-init, and leaves each layer's ckpt under runs/<out-root>_L##/ for
assemble_looped.py. Single GPU → sequential. Stops on the first failure.

  python drive_isolation.py --dry-run            # preview the 24 commands
  python drive_isolation.py --layers 2,4 --steps 3000
"""
from __future__ import annotations
import argparse
import subprocess
import sys
from pathlib import Path

# The 24 GDN (linear_attention) layers of Qwen3.5-9B (full-attn at 3,7,...,31 are MLA).
GDN_LAYERS = [0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18,
              20, 21, 22, 24, 25, 26, 28, 29, 30]
LIB = Path("Qwen3.5-9B-RWKV/converted_layers_lib")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", default="", help="comma-sep subset; default = all 24 GDN layers")
    ap.add_argument("--model-dir", default="Qwen3.5-9B-Base")
    ap.add_argument("--data", default="/thearray/git/babyllm/data/cache/qwen3.6_fwedu_train")
    ap.add_argument("--out-root", default="runs/iso_io")
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--w-block", type=float, default=20.0)
    ap.add_argument("--w-lmce", type=float, default=1.0)
    ap.add_argument("--warmup-steps", type=int, default=200)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no-warmstart", action="store_true", help="GDN-init every layer (ignore the library)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    layers = [int(x) for x in args.layers.split(",") if x.strip()] or GDN_LAYERS
    py = sys.executable
    done = []
    for L in layers:
        out = f"{args.out_root}_L{L:02d}"
        lib = LIB / f"L{L:02d}.pt"
        warm = (not args.no_warmstart) and lib.exists()
        cmd = [py, "convert_train.py", "--layer", str(L),
               "--model-dir", args.model_dir, "--data", args.data, "--out", out,
               "--optimizer", "schedulefree", "--sf-r", "1.0", "--codec-pretrain", "0",
               "--w-lmce", str(args.w_lmce), "--w-block", str(args.w_block),
               "--w-smt", "0.0", "--w-dmt", "0.0", "--lr", str(args.lr),
               "--steps", str(args.steps), "--seq-len", str(args.seq_len),
               "--warmup-steps", str(args.warmup_steps),
               "--eval-every", str(args.eval_every), "--save-every", str(args.save_every),
               "--device", args.device, "--dtype", "bfloat16"]
        if warm:
            cmd += ["--init-rwkv-ckpt", str(lib)]
        tag = "warm-start from library" if warm else ("GDN-init" if not lib.exists() else "GDN-init (--no-warmstart)")
        print(f"\n=== L{L:2d}  ({tag})  -> {out} ===", flush=True)
        print("   " + " ".join(cmd), flush=True)
        if args.dry_run:
            continue
        rc = subprocess.run(cmd).returncode
        if rc != 0:
            print(f"\n!! L{L} FAILED (exit {rc}). Stopping. Completed: {done}", flush=True)
            sys.exit(1)
        done.append(L)

    if not args.dry_run:
        print(f"\nAll {len(done)} layers converted: {done}", flush=True)
        print(f"Assemble: python -m rwkv_lab.assemble_looped --out Qwen3.5-9B-RWKV/rwkv_layers_isolated.pt "
              f"{args.out_root}_L*/step_*/ckpt.pt", flush=True)


if __name__ == "__main__":
    main()
