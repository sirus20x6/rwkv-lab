#!/usr/bin/env python
"""Assemble independently-converted single layers into one LoopedRWKV artifact.

Input: per-layer files, each one of
  * a convert_train.py checkpoint  {"student": <sd>, "codec": ..., "args": {"layer": L, ...}}
    The student sd may be a bare core OR a LoopedRWKV sd (core.* + gates) — convert_train
    saves the student AS a LoopedRWKV since the loop went native (--loop-count default 4).
    Looped sds are STRIPPED to the bare core here: the isolation-trained loop is
    deliberately discarded (see below) and wrapping it directly would double-prefix
    the keys to core.core.* (a bug that broke every gdn_sweep banked ckpt).
  * a library layer file           {"layer_id": L, "state_dict": <LoopedRWKV sd: core.* + residual_weight + iter_norm.weight>, ...}

Output: the rwkv_layers_looped.pt format consumed by distill_consolidate.py and
the progressive loader:
  {"layers": {L: <LoopedRWKV sd>}, "converted": [...], "loop_count": N,
   "allow_neg_eigval": False, "stream_cursor": 0, "batch": 1}

Wrapping a bare core into LoopedRWKV state:
  * prefix every core key with "core."
  * add residual_weight = zeros(loop_count)  -> loop == single-pass at start
    (LoopedRWKV's identity init; the loop is then trained during joint
    consolidation, which is the only place the full-model trajectory exists)
  * add iter_norm.weight = ones(hidden)      -> RMSNorm default
Library inputs are already LoopedRWKV state and carry their own (core-matched)
residual_weight / iter_norm, so they pass through unchanged.

No training, no GPU — pure state_dict surgery on CPU.

Example:
  python -m rwkv_lab.assemble_looped --loop-count 4 \\
      --out Qwen3.5-9B-RWKV/rwkv_layers_isolated.pt \\
      runs/iso_L*/step_*/ckpt.pt Qwen3.5-9B-RWKV/converted_layers_lib/L00.pt
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import torch


def _wrap_bare(bare: dict, loop_count: int) -> dict:
    """Bare RWKV-core state_dict -> LoopedRWKV state_dict (zero-init loop)."""
    ref = bare.get("receptance.weight")
    if ref is None:  # fall back to any 2D weight to read the hidden dim
        ref = next(v for v in bare.values() if v.ndim == 2)
    hidden = ref.shape[1]
    sd = {f"core.{k}": v for k, v in bare.items()}
    sd["iter_norm.weight"] = torch.ones(hidden, dtype=ref.dtype)
    sd["residual_weight"] = torch.zeros(loop_count, dtype=torch.float32)
    return sd


def _looped_layer(blob, loop_count: int):
    """Return (layer_idx, looped_state_dict) from one input file's loaded blob."""
    if isinstance(blob, dict) and "student" in blob:           # convert_train ckpt
        layer = int(blob["args"]["layer"])
        sd = blob["student"]
        if any(str(k).startswith("core.") for k in sd):
            # Looped convert_train ckpt: strip to the bare core. The isolation-trained
            # loop (gates + iter_norm) is discarded on purpose — the loop is re-learned
            # in joint consolidation, the only place the full-model trajectory exists.
            sd = {str(k)[len("core."):]: v for k, v in sd.items() if str(k).startswith("core.")}
        return layer, _wrap_bare(sd, loop_count)
    if isinstance(blob, dict) and "state_dict" in blob:        # library layer file
        layer = int(blob["layer_id"])
        sd = blob["state_dict"]
        # n_loops is dim 0 of residual_weight ([N] scalar gates, [N,G]/[N,C] finer modes);
        # numel() would mis-read a 2D gate (e.g. [4,64] -> 256).
        n = int(sd["residual_weight"].shape[0]) if "residual_weight" in sd else loop_count
        if n != loop_count:  # fatal: metadata loop_count would disagree with this layer's residual_weight
            raise SystemExit(f"layer {layer}: source residual_weight n_loops {n} != --loop-count {loop_count}; "
                             f"re-run with --loop-count {n} or fix the source")
        return layer, sd
    raise SystemExit(f"unrecognized per-layer file (no 'student' or 'state_dict' key): keys={list(blob)[:8]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+", help="per-layer files or globs (convert_train ckpt.pt or library L##.pt)")
    ap.add_argument("--loop-count", type=int, default=4)
    ap.add_argument("--gate-cap", type=float, default=0.0,
                    help="loop gate soft-cap (--loop-gate-cap) the LIBRARY layers' gates were "
                         "trained with; recorded into the artifact so distill_consolidate "
                         "rebuilds LoopedRWKV with the same effective-gate function. Irrelevant "
                         "for convert ckpts (stripped to zero-init gates).")
    ap.add_argument("--out", required=True)
    ap.add_argument("--allow-neg-eigval", action=argparse.BooleanOptionalAction, default=True)
    args = ap.parse_args()

    paths = []
    for pat in args.inputs:
        m = sorted(glob.glob(pat))
        paths.extend(m if m else [pat])
    if not paths:
        raise SystemExit("no input files matched")

    layers: dict[int, dict] = {}
    for p in paths:
        if not Path(p).exists():
            raise SystemExit(f"input not found: {p}")
        blob = torch.load(p, map_location="cpu", weights_only=False)
        L, sd = _looped_layer(blob, args.loop_count)
        if L in layers:
            raise SystemExit(f"duplicate layer {L} (second from {p})")
        # sanity: every layer must carry the loop params, else distill_consolidate
        # would silently load a bare core and never train the loop.
        assert "residual_weight" in sd and "iter_norm.weight" in sd, f"layer {L} missing loop params"
        assert any(k.startswith("core.") for k in sd), f"layer {L} has no core.* keys"
        layers[L] = sd
        print(f"  + L{L:02d}: {len(sd)} tensors  (from {p})", flush=True)

    conv = sorted(layers)
    out_blob = {
        "layers": layers,
        "converted": conv,
        "loop_count": args.loop_count,
        "gate_cap": float(args.gate_cap),
        "allow_neg_eigval": bool(args.allow_neg_eigval),
        "stream_cursor": 0,
        "batch": 1,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(out_blob, args.out)
    print(f"\nwrote {args.out}: {len(conv)} layers {conv}  loop_count={args.loop_count}", flush=True)


if __name__ == "__main__":
    main()
