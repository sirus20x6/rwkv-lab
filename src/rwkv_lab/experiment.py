"""Tier-1 experiment layer for RWKV-Lab — makes lever A/Bs CONCLUSIVE.

The from-scratch sweeps were inconclusive because (a) single-seed runs have a ~0.1-nat noise
floor, (b) web-text ppl doesn't measure what the levers are for, and (c) bad configs burned full
runs. This wraps the small-model harness with the three fixes:

  1. N-seed sweeps + mean±std aggregation + a significance call (|Δmean| vs pooled std).
  2. A preflight smoke gate — a few steps that reject diverging / NaN / non-learning configs in
     seconds before the full run.
  3. Synthetic-task training + eval (copy / associative-recall / induction) with ACCURACY, a
     low-noise capability-relevant signal, plus length-generalization (train at L, eval at 2L).

Levers are configured by name (baseline, loop2/3/4, hyper, cart, deq, ...); extend LEVERS below.

    python -m rwkv_lab.experiment --task recall:16 --configs baseline,loop3,loop3_hyper \
        --seeds 4 --steps 3000 --d-model 256 --n-layers 4
"""
from __future__ import annotations
import argparse, json, math, statistics, time
import numpy as np
import torch
import torch.nn.functional as F

from rwkv_lab.rwkv_pretrain import RWKV7Small
from rwkv_lab.synthetic_tasks import make_task, Task
from rwkv_lab.looped_rwkv import LoopedRWKV
from rwkv_lab import registry

# Lever configs -> LoopedRWKV kwargs ({} = bare core baseline). Add rows to grow the lever set.
LEVERS = {
    "baseline":    {},
    "loop2":       dict(n_loops=2),
    "loop3":       dict(n_loops=3),
    "loop4":       dict(n_loops=4),
    "loop3_hyper": dict(n_loops=3, hyper_lanes=2),
    "loop3_cart":  dict(n_loops=3, cart_anchor=True),
    "loop3_deq":   dict(n_loops=3, loop_deq=True),
    "loop3_factor": dict(n_loops=3, gate_mode="factored"),
}


def _norm_loopkw(kw: dict) -> dict:
    """Fill LoopedRWKV defaults so RWKV7Small's loop path constructs cleanly."""
    if not kw:
        return {}
    d = dict(n_loops=2, hyper_lanes=0, gate_mode="scalar", gate_cap=0.0, cart_anchor=False,
             loop_deq=False, deq_window=1, fixed_point_halt=False, adaptive_halt=False)
    d.update(kw)
    return d


def build(task: Task, d_model, n_layers, head_size, lever) -> RWKV7Small:
    return RWKV7Small(task.vocab, d_model, n_layers, head_size, _norm_loopkw(LEVERS[lever]))


def _masked_ce(logits, y, m):
    ce = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1), reduction="none")
    return (ce * m.reshape(-1)).sum() / m.sum().clamp_min(1)


def loop_gate_stats(model) -> float:
    """Mean |loop gate| (residual_weight) across LoopedRWKV blocks. ~0 => the loops never engaged
    (stayed at zero-init identity) — the direct test of whether recurrent depth did anything."""
    gs = [m.residual_weight.detach().float().abs().mean().item()
          for m in model.modules() if isinstance(m, LoopedRWKV) and hasattr(m, "residual_weight")]
    return sum(gs) / len(gs) if gs else 0.0


@torch.no_grad()
def _eval_acc(model, task, B, device, rng, iters=8):
    model.eval(); tot = 0.0
    for _ in range(iters):
        x, y, m = task.batch(B, device, rng)
        tot += Task.accuracy(model(x).float(), y, m)
    model.train()
    return tot / iters


def preflight(task, d_model, n_layers, head_size, lever, device, batch, steps=20):
    """Reject diverging / NaN / non-learning configs before a full run. Returns (ok, reason)."""
    torch.manual_seed(0); rng = np.random.default_rng(0)
    try:
        model = build(task, d_model, n_layers, head_size, lever).to(device, torch.bfloat16)
    except Exception as e:
        return False, f"build failed: {type(e).__name__}: {e}"
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    losses = []
    for i in range(steps):
        x, y, m = task.batch(batch, device, rng)
        loss = _masked_ce(model(x).float(), y, m)
        if not torch.isfinite(loss):
            return False, f"non-finite loss at step {i}"
        opt.zero_grad(set_to_none=True); loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not torch.isfinite(gn):
            return False, f"non-finite grad at step {i}"
        opt.step(); losses.append(float(loss))
    if losses[-1] > losses[0] + 0.5:
        return False, f"diverging (loss {losses[0]:.2f} -> {losses[-1]:.2f})"
    return True, f"ok (loss {losses[0]:.2f} -> {losses[-1]:.2f})"


def train_eval(task, d_model, n_layers, head_size, lever, seed, device, steps, batch, lr):
    """Train one model on the task; return metrics incl. length-generalization accuracy."""
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    model = build(task, d_model, n_layers, head_size, lever).to(device, torch.bfloat16)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.01)
    warm = max(1, steps // 20)
    for step in range(steps):
        w = min(1.0, (step + 1) / warm)                      # warmup
        cos = 0.5 * (1 + math.cos(math.pi * min(step, steps) / steps))   # 1 -> 0
        for g in opt.param_groups:
            g["lr"] = lr * w * (0.1 + 0.9 * cos)             # warmup then cosine decay to 0.1x
        x, y, m = task.batch(batch, device, rng)
        loss = _masked_ce(model(x).float(), y, m)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    acc = _eval_acc(model, task, batch, device, rng)
    out = {"loss": float(loss), "acc": acc, "gate": loop_gate_stats(model)}
    # length-generalization: eval on a 2x-longer task of the same family (train short, test long)
    arg = getattr(task, "L", None) or getattr(task, "n", None)
    if arg:
        long_spec = f"{task.name.rstrip('0123456789')}:{2 * arg}"
        try:
            out["acc_2x"] = _eval_acc(model, make_task(long_spec), batch, device, rng)
        except Exception:
            pass
    return out


def _agg(vals):
    m = statistics.mean(vals)
    s = statistics.stdev(vals) if len(vals) > 1 else 0.0
    return m, s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="recall:16")
    ap.add_argument("--configs", default="baseline,loop3")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--d-model", type=int, default=256); ap.add_argument("--n-layers", type=int, default=4)
    ap.add_argument("--head-size", type=int, default=64); ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    task = make_task(args.task)
    configs = args.configs.split(",")
    print(f"task={task.name} vocab={task.vocab}  configs={configs}  seeds={args.seeds}  steps={args.steps}  dev={dev}", flush=True)

    results = {}
    for cfg in configs:
        ok, why = preflight(task, args.d_model, args.n_layers, args.head_size, cfg, dev, args.batch)
        if not ok:
            print(f"  [{cfg}] PREFLIGHT REJECTED: {why}", flush=True); continue
        runs = []
        t0 = time.time()
        for s in range(args.seeds):
            runs.append(train_eval(task, args.d_model, args.n_layers, args.head_size, cfg, s,
                                   dev, args.steps, args.batch, args.lr))
        keys = runs[0].keys()
        results[cfg] = {k: _agg([r[k] for r in runs if k in r]) for k in keys}
        results[cfg]["_n"] = len(runs)
        acc_m, acc_s = results[cfg]["acc"]
        gate = results[cfg].get("gate", (0.0, 0.0))[0]
        print(f"  [{cfg}] preflight {why} | acc {acc_m:.3f}±{acc_s:.3f}"
              + (f" | acc_2x {results[cfg]['acc_2x'][0]:.3f}" if "acc_2x" in results[cfg] else "")
              + (f" | loop_gate {gate:.3f}{' (INERT)' if gate < 0.02 else ''}" if cfg != "baseline" else "")
              + f"  ({(time.time()-t0):.0f}s)", flush=True)
        registry.record(args.task, cfg, args.seeds, args.steps,
                        {k: list(v) for k, v in results[cfg].items() if isinstance(v, tuple)})

    # significance vs baseline: |Δmean| > (s_a + s_b) is a real effect above the noise
    if "baseline" in results:
        bm, bs = results["baseline"]["acc"]
        print(f"\n=== {task.name}: accuracy vs baseline ({bm:.3f}±{bs:.3f}), {args.seeds} seeds ===")
        for cfg, r in results.items():
            if cfg == "baseline":
                continue
            m, s = r["acc"]; d = m - bm; sig = abs(d) > (s + bs)
            print(f"  {cfg:16} {m:.3f}±{s:.3f}   Δ{d:+.3f}   {'SIGNIFICANT' if sig else 'within noise'}")
    if args.out:
        json.dump({c: {k: v for k, v in r.items()} for c, r in results.items()}, open(args.out, "w"))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
