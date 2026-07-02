#!/usr/bin/env python
"""loop_probe.py — loop-iterate diagnostics + depth-usefulness sweep for LoopedRWKV.

Implements the diagnostics the recurrent-depth literature converged on:

  Trajectory (Pappone 2509.23314, per looped layer, per pass k over out^(k)):
    step_norm[k]   ‖Δ^(k)‖₂, Δ^(k)=out^(k+1)−out^(k)   — expect fast decay if loops refine
    step_cos[k]    cos(Δ^(k), Δ^(k−1))                  — plateau ~0.5-0.65 = complementary updates
    accel[k]       ‖out^(k+1)−2out^(k)+out^(k−1)‖₂      — THE halting signal (two-hit rule)
    accel_rel[k]   ‖Δ^(k)−Δ^(k−1)‖/(‖Δ^(k)‖+‖Δ^(k−1)‖)  — bounded [0,2] scale-free variant
    out_rms[k]     RMS of out^(k)                        — scale drift watch (readout blind spot,
                                                          2606.24898; block-MSE anchors it in
                                                          conversion training, CE-only phases don't)
    cka[k]         linear CKA(out^(k), out^(k+1))        — ~1.0 everywhere = stagnant loops (MeSH)
    dlr            mean cross-layer residual jump ÷ mean in-loop step (two-scale separation, ≫1)

  Depth usefulness (2606.24898): CE / ppl / top1 / mean logit margin with every looped
  layer clamped to K passes, K=1..n_loops. A flat curve = loops do nothing measurable —
  the honest metric (halting "speedup" on a flat curve is vanity).

Caveat: at n_loops 2-4 the trajectory curves are near their usable floor (the decay the
papers describe takes 5-10 steps); trends across checkpoints matter more than absolutes.

Usage:
  python loop_probe.py --rwkv-ckpt Qwen3.5-9B-RWKV/rwkv_layers.pt --run-name probe_base
"""
from __future__ import annotations

import sys
sys.modules.setdefault("torchvision", None)

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from build_memory_targets import load_token_stream
from distill_consolidate import load_student, _text


def _cka(a, b, eps=1e-6):
    """Linear CKA between [N,C] feature matrices (1.0 = identical structure)."""
    a = a - a.mean(0, keepdim=True)
    b = b - b.mean(0, keepdim=True)
    hsic_ab = (b.T @ a).pow(2).sum()
    hsic_aa = (a.T @ a).pow(2).sum()
    hsic_bb = (b.T @ b).pow(2).sum()
    return float(hsic_ab / (hsic_aa.sqrt() * hsic_bb.sqrt() + eps))


@torch.inference_mode()
def trace_windows(student, s_tm, s_layers, converted, toks, n_windows, T, dev):
    """Run fixed windows; collect per-layer loop traces + per-layer residual inputs."""
    rng = np.random.default_rng(4242)
    starts = rng.integers(0, len(toks) - (T + 1) + 1, size=n_windows, dtype=np.int64)
    layer_in = {}
    hooks = []
    for i, lay in enumerate(s_tm.layers):   # residual stream entering each layer (for DLR)
        def pre(mod, a, kw, i=i):
            h = a[0] if a else kw.get("hidden_states")
            if torch.is_tensor(h):
                layer_in[i] = h.detach()
        hooks.append(lay.register_forward_pre_hook(pre, with_kwargs=True))

    acc = {L: None for L in converted}   # accumulated per-layer stats
    dlr_num = {L: 0.0 for L in converted}
    dlr_den = {L: 0.0 for L in converted}
    n_layers = len(s_tm.layers)
    try:
        for s0 in starts:
            x = torch.as_tensor(np.asarray(toks[s0:s0 + T], dtype=np.int64),
                                device=dev).unsqueeze(0)
            for L in converted:
                s_layers[L].linear_attn._probe_trace = []
            s_tm(input_ids=x, use_cache=False)
            for L in converted:
                tr = s_layers[L].linear_attn._probe_trace
                del s_layers[L].linear_attn._probe_trace
                if len(tr) < 2:
                    continue
                xs = [t.float() for t in tr]                    # n_loops x [1,T,C]
                d = [xs[k + 1] - xs[k] for k in range(len(xs) - 1)]
                st = {
                    "step_norm": [float(dk.norm(dim=-1).mean()) for dk in d],
                    "step_cos": [float(F.cosine_similarity(d[k], d[k - 1], dim=-1).mean())
                                 for k in range(1, len(d))],
                    "accel": [float((d[k] - d[k - 1]).norm(dim=-1).mean())
                              for k in range(1, len(d))],
                    "accel_rel": [float(((d[k] - d[k - 1]).norm(dim=-1)
                                         / (d[k].norm(dim=-1) + d[k - 1].norm(dim=-1) + 1e-6)).mean())
                                  for k in range(1, len(d))],
                    "out_rms": [float(xk.pow(2).mean(-1).sqrt().mean()) for xk in xs],
                    "cka": [_cka(xs[k].reshape(-1, xs[k].shape[-1]),
                                 xs[k + 1].reshape(-1, xs[k].shape[-1]))
                            for k in range(len(xs) - 1)],
                }
                if acc[L] is None:
                    acc[L] = {k: [0.0] * len(v) for k, v in st.items()}
                for k, v in st.items():
                    acc[L][k] = [a + b for a, b in zip(acc[L][k], v)]
                # DLR: residual jump across this layer vs mean in-loop step
                if L + 1 < n_layers and L in layer_in and (L + 1) in layer_in:
                    jump = float((layer_in[L + 1].float() - layer_in[L].float())
                                 .norm(dim=-1).mean())
                    dlr_num[L] += jump
                    dlr_den[L] += sum(st["step_norm"]) / len(st["step_norm"])
    finally:
        for h in hooks:
            h.remove()
        for L in converted:   # a mid-forward exception must not leave traces armed
            if hasattr(s_layers[L].linear_attn, "_probe_trace"):
                del s_layers[L].linear_attn._probe_trace
    out = {}
    for L in converted:
        if acc[L] is None:
            continue
        out[L] = {k: [v / n_windows for v in vs] for k, vs in acc[L].items()}
        if dlr_den[L] > 0:
            out[L]["dlr"] = dlr_num[L] / dlr_den[L]
    return out


@torch.inference_mode()
def eval_at_k(s_tm, lm_head, toks, n_windows, T, dev, chunk=2048):
    """CE / ppl / top1 / mean top1-top2 logit margin on fixed windows (fp32 eval:
    bf16 CE sums quantize to a ~0.01 ppl grid — repo finding)."""
    rng = np.random.default_rng(12345)
    starts = rng.integers(0, len(toks) - (T + 1) + 1, size=n_windows, dtype=np.int64)
    Wt, bias = lm_head.weight, getattr(lm_head, "bias", None)
    tot_ce = torch.zeros((), dtype=torch.float64, device=dev)
    tot_ok = torch.zeros((), dtype=torch.int64, device=dev)
    tot_margin = torch.zeros((), dtype=torch.float64, device=dev)
    n_tok = 0
    for s0 in starts:
        ids = torch.as_tensor(np.asarray(toks[s0:s0 + T + 1], dtype=np.int64),
                              device=dev).unsqueeze(0)
        x, y = ids[:, :-1], ids[:, 1:]
        out = s_tm(input_ids=x, use_cache=False)
        hidden = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
        fh, fy = hidden.reshape(-1, hidden.shape[-1]), y.reshape(-1)
        for i in range(0, fh.shape[0], chunk):
            e = min(i + chunk, fh.shape[0])
            logits = F.linear(fh[i:e], Wt, bias).float()
            tot_ce += F.cross_entropy(logits, fy[i:e], reduction="sum").double()
            top2 = logits.topk(2, dim=-1).values
            tot_margin += (top2[:, 0] - top2[:, 1]).sum().double()
            tot_ok += (logits.argmax(-1) == fy[i:e]).sum()
            n_tok += e - i
    ce = tot_ce.item() / max(n_tok, 1)
    return {"loss": ce, "ppl": math.exp(min(ce, 20.0)),
            "top1": tot_ok.item() / max(n_tok, 1),
            "margin": tot_margin.item() / max(n_tok, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="Qwen3.5-9B-Base")
    ap.add_argument("--rwkv-ckpt", default="Qwen3.5-9B-RWKV/rwkv_layers.pt")
    ap.add_argument("--data", default="/thearray/git/babyllm/data/cache/qwen3.6_fwedu_train")
    ap.add_argument("--decay-cap-delta", type=float, default=0.005,
                    help="MUST match the value the checkpoint was trained with (constructor "
                         "state on the core — a mismatch silently changes the probed function)")
    ap.add_argument("--windows", type=int, default=4)
    ap.add_argument("--seqlen", type=int, default=1024)
    ap.add_argument("--eval-windows", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--out", default="", help="JSON output path; default runs/<run-name>/loop_probe.json")
    ap.add_argument("--run-name", default="loop_probe")
    ap.add_argument("--skip-sweep", type=int, default=0, help="1 = trajectory diagnostics only")
    args = ap.parse_args()
    dev, dtype = args.device, getattr(torch, args.dtype)

    student, s_tm, s_layers, converted, looped, loop_count, aneg, gate_cap = load_student(
        args.model_dir, args.rwkv_ckpt, args.decay_cap_delta, dev, dtype)
    if not looped or loop_count < 2:
        raise SystemExit("checkpoint carries bare cores / loop_count<2: nothing to probe")
    student.eval()
    toks = load_token_stream(args.data)
    T = args.seqlen

    print(f"tracing {len(converted)} looped layers (n_loops={loop_count}) on "
          f"{args.windows}x{T} windows ...", flush=True)
    traj = trace_windows(student, s_tm, s_layers, converted, toks, args.windows, T, dev)

    sweep = {}
    if not args.skip_sweep:
        wrappers = [s_layers[L].linear_attn for L in converted]
        try:
            for K in range(1, loop_count + 1):
                for r in wrappers:
                    r.n_loops = K
                sweep[K] = eval_at_k(s_tm, student.lm_head, toks, args.eval_windows, T, dev)
                print(f"  K={K}: ppl={sweep[K]['ppl']:.3f} top1={sweep[K]['top1']:.4f} "
                      f"margin={sweep[K]['margin']:.3f}", flush=True)
        finally:
            for r in wrappers:
                r.n_loops = loop_count

    # compact report
    print(f"\n{'layer':>5} {'dlr':>7} {'step_norm k=1..':<24} {'cos k=2..':<18} "
          f"{'accel_rel k=2..':<18} {'cka k=1..':<20}")
    for L in sorted(traj):
        t = traj[L]
        fmt = lambda v, n=3: "/".join(f"{x:.{n}f}" for x in v)
        print(f"{L:>5} {t.get('dlr', float('nan')):>7.2f} {fmt(t['step_norm']):<24} "
              f"{fmt(t['step_cos'], 2):<18} {fmt(t['accel_rel'], 2):<18} {fmt(t['cka'], 3):<20}")
    if sweep:
        base = sweep[loop_count]["ppl"]
        flat = all(abs(sweep[K]["ppl"] - base) / base < 0.001 for K in sweep)
        print(f"\nPPL vs K: " + "  ".join(f"K={K}:{v['ppl']:.3f}" for K, v in sweep.items()))
        if flat:
            print("  [!] depth-usefulness curve is FLAT (<0.1% ppl spread): the loops are not "
                  "changing the readout output — capacity claims from these loops are suspect "
                  "(readout-blind-spot paper's K-invariance signature).")

    out_path = Path(args.out) if args.out else Path("runs") / args.run_name / "loop_probe.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(
        {"ckpt": str(args.rwkv_ckpt), "loop_count": loop_count, "gate_cap": gate_cap,
         "trajectory": {str(k): v for k, v in traj.items()},
         "ppl_vs_k": {str(k): v for k, v in sweep.items()}}, indent=2))
    print(f"\nsaved -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
