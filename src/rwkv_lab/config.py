"""Declarative experiment configs — one YAML/JSON file fully specifies a run: data (synthetic task
OR a weighted corpus mixture), model, training budget, and the lever variants to compare. No Python
or shell edits to define an experiment; results land in the registry.

    python -m rwkv_lab.config run experiments/loops_on_recall.yaml

Schema:
    data:
      task: recall:16                 # (a) synthetic diagnostic — OR —
      sources:                        # (b) LM corpus (weighted mixture, tokenized+cached via ztok)
        - {kind: hf,    name: wikitext, weight: 0.7}
        - {kind: local, patterns: ["/thearray/git/moe-mla/**/*.py"], weight: 0.3}
      doc_boundary: true
      cap_mb: 50
    seeds: 4
    model: {d_model: 256, n_layers: 4, head_size: 64}
    train: {steps: 3000, lr: 3e-3, batch: 64, seq_len: 512, minutes: 10}
    configs:                          # name -> LoopedRWKV/aux kwargs
      baseline: {}
      loop3:    {n_loops: 3}
"""
from __future__ import annotations
import argparse, json, os, subprocess, sys
import yaml


def load(path: str) -> dict:
    txt = open(path).read()
    return yaml.safe_load(txt) if path.endswith((".yaml", ".yml")) else json.loads(txt)


def resolve_data(cfg: dict):
    """('synthetic', task_spec) or ('lm', (bin_path, off_path|None)) — building+caching the corpus."""
    d = cfg["data"]
    if "task" in d:
        return "synthetic", d["task"]
    from rwkv_lab.build_corpus import resolve_corpus
    return "lm", resolve_corpus(d)


# lever kwarg -> rwkv_pretrain CLI flag (for LM runs)
_LM_FLAG = {"n_loops": "--loop-count", "hyper_lanes": "--loop-hyper", "gate_mode": "--loop-gate",
            "cart_anchor": "--loop-cart-anchor", "loop_deq": "--loop-deq", "fixed_point_halt": "--loop-fp-halt",
            "adaptive_halt": "--loop-adaptive-halt", "nextlat_weight": "--nextlat-weight",
            "top_weight": "--top-weight", "lmtp_weight": "--lmtp-weight", "bst_weight": "--bst-weight",
            "jtp_weight": "--jtp-weight"}


def _run_synthetic(task_spec, cfg):
    import torch
    from rwkv_lab import experiment as E, registry
    E.LEVERS.update(cfg["configs"])                     # register the file's lever combos by name
    task = E.make_task(task_spec)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m, tr = cfg.get("model", {}), cfg.get("train", {})
    dm, nl, hs = int(m.get("d_model", 256)), int(m.get("n_layers", 4)), int(m.get("head_size", 64))
    steps, batch = int(tr.get("steps", 3000)), int(tr.get("batch", 64))
    lr = float(tr.get("lr", 3e-3))                      # YAML 1.1 parses '3e-3' as str -> coerce
    minutes = float(tr.get("minutes", 0.0))             # wall-clock budget (0 = use steps)
    seeds = int(cfg.get("seeds", 3))
    print(f"[config] synthetic task={task.name} configs={list(cfg['configs'])} seeds={seeds} dev={dev}", flush=True)
    results = {}
    for name in cfg["configs"]:
        ok, why = E.preflight(task, dm, nl, hs, name, dev, batch)
        if not ok:
            print(f"  [{name}] PREFLIGHT REJECTED: {why}", flush=True); continue
        runs = [E.train_eval(task, dm, nl, hs, name, s, dev, steps, batch, lr, minutes) for s in range(seeds)]
        agg = {k: E._agg([r[k] for r in runs if k in r]) for k in runs[0]}
        registry.record(task_spec, name, seeds, steps, {k: list(v) for k, v in agg.items()})
        results[name] = agg
        print(f"  [{name}] acc {agg['acc'][0]:.3f}±{agg['acc'][1]:.3f}"
              + (f" gate {agg['gate'][0]:.3f}" if "gate" in agg and name != "baseline" else ""), flush=True)
    if "baseline" in results:
        bm, bs = results["baseline"]["acc"]
        print(f"\n=== {task.name}: acc vs baseline ({bm:.3f}±{bs:.3f}) ===")
        for name, r in results.items():
            if name == "baseline":
                continue
            mm, ss = r["acc"]; d = mm - bm
            print(f"  {name:16} {mm:.3f}±{ss:.3f}  Δ{d:+.3f} {'SIGNIFICANT' if abs(d) > ss + bs else 'ns'}")


def _run_lm(data, cfg):
    bin_path, off_path = data
    m, tr = cfg.get("model", {}), cfg.get("train", {})
    for name, lever in cfg["configs"].items():
        cmd = [sys.executable, "-m", "rwkv_lab.rwkv_pretrain", "--data", bin_path, "--out", f"runs/cfg_{name}",
               "--d-model", str(m.get("d_model", 512)), "--n-layers", str(m.get("n_layers", 8)),
               "--head-size", str(m.get("head_size", 64)), "--batch", str(tr.get("batch", 16)),
               "--seq-len", str(tr.get("seq_len", 1024)), "--lr", str(tr.get("lr", 6e-4))]
        cmd += (["--steps", str(tr["steps"])] if "steps" in tr else ["--minutes", str(tr.get("minutes", 10))])
        if off_path:
            cmd += ["--doc-offsets", off_path]
        for k, v in lever.items():                     # translate lever kwargs -> CLI flags
            if k in _LM_FLAG:
                cmd += [_LM_FLAG[k], str(int(v) if isinstance(v, bool) else v)]
        print(f"[config] LM run '{name}': {' '.join(cmd[2:])}", flush=True)
        subprocess.run(cmd, env={**os.environ, "PYTHONPATH": "src"})


def run(cfg_path: str):
    cfg = load(cfg_path)
    kind, data = resolve_data(cfg)
    (_run_synthetic if kind == "synthetic" else _run_lm)(data, cfg)


# Default local corpus for the board's LM-mode launches (code + docs, doc-boundary, cached).
_LOCAL_LM_SPEC = {"sources": [{"kind": "local",
                               "patterns": ["/thearray/git/moe-mla/**/*.py", "/thearray/git/moe-mla/**/*.md"],
                               "weight": 1.0}], "cap_mb": 8.0, "doc_boundary": True}


def run_lm(levers, model, train):
    """Run a set of named levers on the local LM corpus via rwkv_pretrain — used by the board's LM
    mode so top/lmtp/bst/jtp (which need a token future) are launchable. Lever kwargs come from
    experiment.LEVERS (single source of truth); each run appears in the main trainboard leaderboard."""
    from rwkv_lab.experiment import LEVERS
    from rwkv_lab.build_corpus import resolve_corpus
    bin_path, off_path = resolve_corpus(_LOCAL_LM_SPEC)
    for name in levers:
        lever = LEVERS.get(name, {})
        cmd = [sys.executable, "-m", "rwkv_lab.rwkv_pretrain", "--data", bin_path, "--out", f"runs/lm_{name}",
               "--d-model", str(model.get("d_model", 256)), "--n-layers", str(model.get("n_layers", 4)),
               "--head-size", str(model.get("head_size", 64)), "--batch", str(train.get("batch", 16)),
               "--seq-len", str(train.get("seq_len", 512)), "--lr", str(train.get("lr", 6e-4))]
        cmd += (["--minutes", str(train["minutes"])] if train.get("minutes")     # wall-clock budget
                else ["--steps", str(train.get("steps", 2000))])
        if off_path:
            cmd += ["--doc-offsets", off_path]
        for k, v in lever.items():
            if k in _LM_FLAG:
                cmd += [_LM_FLAG[k], str(int(v) if isinstance(v, bool) else v)]
        print(f"[config] LM lever '{name}': {' '.join(cmd[2:])}", flush=True)
        subprocess.run(cmd, env={**os.environ, "PYTHONPATH": "src"})


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run"); r.add_argument("config")
    rl = sub.add_parser("run-lm")                            # board LM mode: named levers on local corpus
    rl.add_argument("--levers", required=True)
    rl.add_argument("--d-model", type=int, default=256)
    rl.add_argument("--n-layers", type=int, default=4)
    rl.add_argument("--head-size", type=int, default=64)
    rl.add_argument("--steps", type=int, default=2000)
    rl.add_argument("--minutes", type=float, default=0.0)
    rl.add_argument("--seq-len", type=int, default=512)
    rl.add_argument("--batch", type=int, default=16)
    rl.add_argument("--lr", type=float, default=6e-4)
    args = ap.parse_args()
    if args.cmd == "run-lm":
        run_lm(args.levers.split(","),
               {"d_model": args.d_model, "n_layers": args.n_layers, "head_size": args.head_size},
               {"steps": args.steps, "minutes": args.minutes, "seq_len": args.seq_len,
                "batch": args.batch, "lr": args.lr})
    else:
        run(args.config)


if __name__ == "__main__":
    main()
