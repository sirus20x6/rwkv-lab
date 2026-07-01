#!/usr/bin/env python
"""Recover full-RWKV ppl by distilling against the CLEAN original model.

Local block-MSE consolidation fails for the full stack: each RWKV layer can match
its GDN teacher to ~0.028 yet ppl is 456, because matching "GDN's response to the
already-drifted input" perpetuates drift. Fix: anchor to the clean trajectory.

Frozen teacher = the original all-GDN model. Student = the converted RWKV model.
Same input through both; train the student's RWKV layers so the student's residual
stream matches the TEACHER'S CLEAN residual stream at every layer (relative MSE,
which balances across depths and upweights the high-leverage early layers). The
student runs on its own drifted states (exposure bias) but is pulled back to clean.

Emits the dashboard schema to runs/<run-name>/ (train = distill loss, eval = ppl).
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
from convert_train import evaluate


def _text(m):
    return getattr(m.model, "language_model", m.model)


def dash_setup(run_name, converted, model_dir, steps):
    if not run_name:
        return (lambda r: None), (lambda s: None)
    rd = Path("runs") / run_name; rd.mkdir(parents=True, exist_ok=True)
    logf = open(rd / "train.jsonl", "a")

    def emit(r):
        logf.write(json.dumps(r) + "\n"); logf.flush()

    def sidecar(step):
        cfg = {"model_dir": str(Path(model_dir).resolve()), "patch_dir": "",
               "rwkv8_deltanet_layers": ",".join(map(str, sorted(converted))),
               "rwkv8_swap_mode": "timemix",
               "train_rwkv8_layers": ",".join(map(str, sorted(converted))),
               "install_mtp": 0, "engram_enabled": 0, "freeze_non_mla": 1}
        sd = rd / f"step_{step:06d}"; sd.mkdir(parents=True, exist_ok=True)
        (sd / "config.json").write_text(json.dumps({"step": step, "config": cfg}, indent=2))

    return emit, sidecar


def load_student(model_dir, rwkv_ckpt, decay_cap_delta, device, dtype):
    from transformers import AutoModelForCausalLM
    m = AutoModelForCausalLM.from_pretrained(model_dir, dtype=dtype, low_cpu_mem_usage=True).to(device).eval()
    tm = _text(m); layers = tm.layers
    cfg = m.config.text_config if hasattr(m.config, "text_config") else m.config
    ck = torch.load(rwkv_ckpt, map_location="cpu")
    conv = sorted(int(x) for x in ck["layers"])
    # Detect the LoopedRWKV format: an explicit loop_count, or per-layer state_dicts
    # carrying core.* keys (the wrapper) instead of bare-core keys. Bare checkpoints
    # (old convert_stack / distilled format) still load straight into the core.
    sample = ck["layers"][conv[0]]
    has_core = any(str(k).startswith("core.") for k in sample)   # authoritative: keys, not metadata
    meta_lc = int(ck.get("loop_count") or 0)
    # Detect by ACTUAL keys (a LoopedRWKV state dict carries core.* + residual_weight),
    # and fail loud on metadata/keys disagreement rather than silently half-loading a
    # mislabeled checkpoint (e.g. loop_count>1 over bare cores).
    if has_core:
        looped = True
        # n_loops is dim 0 of residual_weight; numel() would mis-read a non-scalar
        # gate ([4,64] head gates -> 256).
        rw_len = int(sample["residual_weight"].shape[0]) if "residual_weight" in sample else meta_lc
        if meta_lc > 1 and rw_len and rw_len != meta_lc:
            raise SystemExit(f"loop_count metadata {meta_lc} != residual_weight n_loops {rw_len}")
        loop_count = rw_len or meta_lc
    else:
        if meta_lc > 1:
            raise SystemExit(f"loop_count={meta_lc} but layer state dicts are bare (no core.*)")
        looped, loop_count = False, 0
    aneg = bool(ck.get("allow_neg_eigval", False))     # constructor state (_a_scale), NOT in state_dict
    gate_cap = float(ck.get("gate_cap", 0.0) or 0.0)   # constructor state too (assemble_looped records it)

    def _gate_mode_of(sd):
        """Infer the LoopedRWKV gate mode from a layer sd's actual tensors: the mode
        is constructor state, not state_dict content, so it must be reconstructed."""
        rw = sd.get("residual_weight")
        if rw is None or rw.ndim == 1:
            return "scalar"
        if "gate_chan" in sd:
            return "factored"
        return "channel" if rw.shape[1] == cfg.hidden_size else "head"

    for L in conv:
        core = rwkv8_timemix_from_config(m.config, layer_idx=L, depth_n_layer=cfg.num_hidden_layers,
                                         decay_cap_delta=decay_cap_delta, allow_neg_eigval=aneg)
        # LoopedRWKV wraps the core; its parameters() (incl. residual_weight) are the
        # ones we train jointly below — so the loop IS consolidated here.
        lsd = ck["layers"][L]
        r = (LoopedRWKV(core, n_loops=loop_count, hidden_size=cfg.hidden_size,
                        gate_mode=_gate_mode_of(lsd), gate_cap=gate_cap,
                        loop_index="loop_index_embed" in lsd)
             if looped else core)
        miss, unexp = r.load_state_dict(lsd, strict=False)
        if miss or unexp:                              # by construction empty; fail loud on any drift
            raise SystemExit(f"layer {L} load mismatch: missing={list(miss)[:6]} unexpected={list(unexp)[:6]}")
        r = r.to(device=device, dtype=dtype)
        if looped:
            r.float_gates()                            # gates fp32: bf16 ulp swallows tiny steps
        r._save_key = f"rwkv8_layer_{L}"
        setattr(layers[L], "linear_attn", r)
    print(f"loaded {len(conv)} layers "
          f"({'LoopedRWKV n_loops=' + str(loop_count) if looped else 'bare core'}, "
          f"gate_cap={gate_cap:g}, allow_neg_eigval={aneg})", flush=True)
    return m, tm, layers, conv, looped, loop_count, aneg, gate_cap


def _hook(store, i):
    def h(mod, a, out):
        store[i] = out[0] if isinstance(out, tuple) else out
    return h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="Qwen3.5-9B-Base")
    ap.add_argument("--rwkv-ckpt", default="Qwen3.5-9B-RWKV/rwkv_layers.pt")
    ap.add_argument("--data", default="/thearray/git/babyllm/data/cache/qwen3.6_fwedu_train")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seqlen", type=int, default=512)
    ap.add_argument("--decay-cap-delta", type=float, default=0.005)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--eval-windows", type=int, default=8)
    ap.add_argument("--eval-seqlen", type=int, default=1024)
    ap.add_argument("--run-name", default="distill_rwkv24")
    ap.add_argument("--out", default="Qwen3.5-9B-RWKV/rwkv_layers_distilled.pt")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16")
    args = ap.parse_args()
    dtype = getattr(torch, args.dtype)
    dev = args.device

    from transformers import AutoModelForCausalLM
    print("loading teacher (original all-GDN) ...", flush=True)
    teacher = AutoModelForCausalLM.from_pretrained(args.model_dir, dtype=dtype,
                                                   low_cpu_mem_usage=True).to(dev).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    t_tm = _text(teacher); t_layers = t_tm.layers
    print("loading student (RWKV) ...", flush=True)
    student, s_tm, s_layers, converted, looped, loop_count, aneg, gate_cap = load_student(
        args.model_dir, args.rwkv_ckpt, args.decay_cap_delta, dev, dtype)
    for p in student.parameters():
        p.requires_grad_(False)
    params = []
    for L in converted:
        for p in s_layers[L].linear_attn.parameters():
            p.requires_grad_(True); params.append(p)
    nL = len(t_layers)
    t_h, s_h = {}, {}
    for i in range(nL):
        t_layers[i].register_forward_hook(_hook(t_h, i))
        s_layers[i].register_forward_hook(_hook(s_h, i))
    opt = torch.optim.AdamW(params, lr=args.lr, betas=(0.9, 0.95))
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.steps, args.lr * 0.05)

    emit, sidecar = dash_setup(args.run_name, converted, args.model_dir, args.steps)
    toks = load_token_stream(args.data)
    tp = evaluate(t_tm, teacher.lm_head, toks, args.eval_windows, args.eval_seqlen, dev)
    sp = evaluate(s_tm, student.lm_head, toks, args.eval_windows, args.eval_seqlen, dev)
    print(f"teacher ppl={tp['ppl']:.3f}  |  student ppl(start)={sp['ppl']:.3f} "
          f"(distilling {len(converted)} RWKV layers)", flush=True)
    emit({"kind": "eval", "step": 0, "ppl": sp["ppl"], "top1_acc": sp["top1_acc"]})
    sidecar(0)

    N = len(toks); T = args.seqlen; maxs = N - (T + 1)
    rng = np.random.default_rng(0)
    t0 = time.time()
    best = float("inf")  # save on the first eval so args.out is always written (codex #7)
    for step in range(1, args.steps + 1):
        s0 = int(rng.integers(0, maxs + 1))
        x = torch.as_tensor(np.asarray(toks[s0:s0 + T], dtype=np.int64), device=dev).unsqueeze(0)
        with torch.no_grad():
            teacher(input_ids=x, use_cache=False)
        student(input_ids=x, use_cache=False)
        loss = x.new_zeros((), dtype=torch.float32)
        for i in range(nL):
            t = t_h[i].detach().float()
            loss = loss + F.mse_loss(s_h[i].float(), t) / (t.pow(2).mean() + 1e-6)
        loss = loss / nL
        opt.zero_grad(); loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step(); sch.step()
        emit({"kind": "train", "step": step, "loss": float(loss), "gnorm": float(gn),
              "lr": sch.get_last_lr()[0]})
        if step % max(1, args.steps // 40) == 0:
            print(f"  step {step} distill_rel_mse={float(loss):.4f} gnorm={float(gn):.2f} "
                  f"tok/s={step*T/(time.time()-t0):.0f}", flush=True)
        if step % args.eval_every == 0 or step == args.steps:
            ev = evaluate(s_tm, student.lm_head, toks, args.eval_windows, args.eval_seqlen, dev)
            emit({"kind": "eval", "step": step, "ppl": ev["ppl"], "top1_acc": ev["top1_acc"]})
            print(f"  [eval] step {step} ppl={ev['ppl']:.3f} top1={ev['top1_acc']:.3f} "
                  f"(teacher {tp['ppl']:.2f})", flush=True)
            if ev["ppl"] < best:
                best = ev["ppl"]
                blob = {"layers": {L: {k: v.detach().cpu() for k, v in s_layers[L].linear_attn.state_dict().items()}
                                   for L in converted}}
                if looped:  # round-trip as a LoopedRWKV artifact (assemble_looped format)
                    blob.update(converted=list(converted), loop_count=loop_count,
                                gate_cap=gate_cap, allow_neg_eigval=aneg,
                                stream_cursor=0, batch=1)
                torch.save(blob, args.out)
    emit({"kind": "checkpoint", "step": args.steps}); sidecar(args.steps)
    print(f"\nDONE: teacher {tp['ppl']:.2f} -> student start {sp['ppl']:.2f} -> best {best:.2f}", flush=True)
    print(f"saved best -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
