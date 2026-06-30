#!/usr/bin/env python
"""Streaming capacity screen: can looped / neg-eigval / both RWKV-7 fit a hard
GDN layer better than single-pass? NO DATA REUSE — a fresh window every step
(WindowStream over fwedu_train), held-out eval on the SEPARATE fwedu_val split.
All four variants train in lockstep on the same fresh windows for a fair compare.
"""
from __future__ import annotations

import sys
sys.modules.setdefault("torchvision", None)

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from rwkv8_deltanet import rwkv8_timemix_from_config
from looped_rwkv import LoopedRWKV
from build_memory_targets import load_token_stream
from data_stream import WindowStream


def _text(m):
    return getattr(m.model, "language_model", m.model)


def make_emit(run_name):
    rd = Path("runs") / run_name; rd.mkdir(parents=True, exist_ok=True)
    f = open(rd / "train.jsonl", "w")

    def emit(rec):
        f.write(json.dumps(rec) + "\n"); f.flush()

    return emit


class Capturer:
    """Runs the resident frozen base model and returns (h_in, y_out) for one GDN layer."""
    def __init__(self, tm, L):
        self.tm = tm; self.box = {}
        la = tm.layers[L].linear_attn
        la.register_forward_pre_hook(
            lambda m, a, kw: self.box.__setitem__("h", (a[0] if a else kw["hidden_states"]).detach()),
            with_kwargs=True)
        la.register_forward_hook(
            lambda m, a, o: self.box.__setitem__("y", (o[0] if isinstance(o, tuple) else o).detach()))

    @torch.no_grad()
    def __call__(self, ids):
        self.tm(input_ids=ids, use_cache=False)
        return self.box["h"].clone(), self.box["y"].clone()


@torch.no_grad()
def val_eval(mod, Hv, Yv, device, batch=4):
    s_d2 = s_y2 = s_oy = s_o2 = 0.0
    for i in range(0, Hv.shape[0], batch):
        h = Hv[i:i + batch].to(device); y = Yv[i:i + batch].to(device).float()
        o = mod(h); o = (o[0] if isinstance(o, tuple) else o).float()
        s_d2 += (o - y).pow(2).sum().item(); s_y2 += y.pow(2).sum().item()
        s_oy += (o * y).sum().item(); s_o2 += o.pow(2).sum().item()
    return s_d2 / (s_y2 + 1e-8), s_oy / ((s_o2 ** 0.5) * (s_y2 ** 0.5) + 1e-8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="Qwen3.5-9B-Base")
    ap.add_argument("--data", default="/thearray/git/babyllm/data/cache/qwen3.6_fwedu_train")
    ap.add_argument("--val-data", default="/thearray/git/babyllm/data/cache/qwen3.6_fwedu_val")
    ap.add_argument("--layer", type=int, default=4)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lr-sweep", default="", help="comma LRs -> sweep control@each LR instead of arch variants")
    ap.add_argument("--lr-floor-frac", type=float, default=0.1, help="cosine floor as fraction of peak")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--seqlen", type=int, default=1024)
    ap.add_argument("--n-loops", type=int, default=4)
    ap.add_argument("--n-val", type=int, default=24)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--decay-cap-delta", type=float, default=0.005)
    ap.add_argument("--run-prefix", default="")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    prefix = args.run_prefix or f"fitS_l{args.layer}"
    dev = args.device; L = args.layer

    from transformers import AutoModelForCausalLM
    print("loading base (resident for streaming capture) ...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(args.model_dir, dtype=torch.bfloat16,
                                                 low_cpu_mem_usage=True).to(dev).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    tm = _text(model)
    cfg = model.config.text_config if hasattr(model.config, "text_config") else model.config
    nL = cfg.num_hidden_layers; config = model.config
    gdn_sd = {k: v.detach().cpu() for k, v in tm.layers[L].linear_attn.state_dict().items()}
    cap = Capturer(tm, L)

    # held-out val: fixed windows from the SEPARATE val split (truly disjoint from train)
    vstream = WindowStream(load_token_stream(args.val_data), args.seqlen, seed=999)
    Hv, Yv = [], []
    for _ in range((args.n_val + args.batch - 1) // args.batch):
        h, y = cap(vstream.next_batch(args.batch, device=dev))
        Hv.append(h.to(torch.bfloat16).cpu()); Yv.append(y.to(torch.bfloat16).cpu())
    Hv = torch.cat(Hv)[:args.n_val]; Yv = torch.cat(Yv)[:args.n_val]
    print(f"held-out val {tuple(Hv.shape)} from {args.val_data}", flush=True)

    # train stream: fresh window every step, never repeats
    tstream = WindowStream(load_token_stream(args.data), args.seqlen, seed=0)

    dcap = args.decay_cap_delta
    def core(neg):
        return rwkv8_timemix_from_config(config, layer_idx=L, init_from_deltanet=gdn_sd,
                                         depth_n_layer=nL, decay_cap_delta=dcap, allow_neg_eigval=neg)
    if args.lr_sweep:
        pairs = [(s.strip(), float(s)) for s in args.lr_sweep.split(",")]
        builders = [(f"lr{lab}", (lambda: core(False)), v) for lab, v in pairs]
    else:
        builders = [("control", lambda: core(False), args.lr), ("neg_eigval", lambda: core(True), args.lr),
                    ("looped", lambda: LoopedRWKV(core(False), args.n_loops), args.lr),
                    ("both", lambda: LoopedRWKV(core(True), args.n_loops), args.lr)]
    variants = []
    for name, mk, vlr in builders:
        torch.manual_seed(0)
        mod = mk().to(device=dev, dtype=torch.bfloat16)
        ps = [p for p in mod.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(ps, lr=vlr, betas=(0.9, 0.95))
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.steps, vlr * args.lr_floor_frac)
        variants.append(dict(name=name, mod=mod, ps=ps, opt=opt, sch=sch,
                             emit=make_emit(f"{prefix}_{name}"), best=float("inf")))

    print(f"\n=== STREAMING fit L{L}: {args.steps} steps x batch {args.batch} = "
          f"{args.steps * args.batch} UNIQUE windows (no repeats); 4 variants lockstep ===", flush=True)
    t0 = time.time()
    for s in range(args.steps):
        ids = tstream.next_batch(args.batch, device=dev)        # FRESH, never reused
        h, y = cap(ids)
        for v in variants:
            out = v["mod"](h); out = out[0] if isinstance(out, tuple) else out
            loss = F.mse_loss(out.float(), y.float()) / (y.float().pow(2).mean() + 1e-8)
            v["opt"].zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(v["ps"], 1.0); v["opt"].step(); v["sch"].step()
            v["emit"]({"kind": "train", "step": s, "loss": float(loss), "lr": v["sch"].get_last_lr()[0]})
        if s % args.eval_every == 0 or s == args.steps - 1:
            parts = []
            for v in variants:
                vmse, vcos = val_eval(v["mod"], Hv, Yv, dev); v["best"] = min(v["best"], vmse)
                v["emit"]({"kind": "eval", "step": s, "loss": vmse, "ppl": vmse, "top1_acc": vcos})
                parts.append(f"{v['name']}={vmse:.3f}")
            print(f"  step {s:5d}  " + "  ".join(parts) +
                  f"  [{tstream.cursor} win, {time.time()-t0:.0f}s]", flush=True)
    for v in variants:
        v["emit"]({"kind": "checkpoint", "step": args.steps})
    print("\n====== STREAMING RESULT (held-out val rel-MSE, ZERO data reuse) ======", flush=True)
    base = variants[0]["best"]
    for v in variants:
        d = (v["best"] - base) / base * 100.0
        tag = "" if v["name"] == "control" else f"  ({d:+.1f}% vs control)"
        print(f"  {v['name']:10s}: {v['best']:.4f}{tag}", flush=True)


if __name__ == "__main__":
    main()
