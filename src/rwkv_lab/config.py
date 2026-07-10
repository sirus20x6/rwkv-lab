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
            "jtp_weight": "--jtp-weight", "seed_chain": "--seed-chain", "engram": "--engram",
            "deepembed": "--deepembed", "de_dim": "--de-dim", "de_mode": "--de-mode",
            "de_shift": "--de-shift", "de_emb_res": "--de-emb-res"}


def _run_synthetic(task_spec, cfg):
    import torch
    from rwkv_lab import experiment as E, registry
    E.enable_fast_matmul()                              # TF32 tensor cores (free ~1.1-1.3x on fp32)
    E.LEVERS.update(cfg["configs"])                     # register the file's lever combos by name
    fac = cfg.get("factorial", {})
    if fac.get("enabled"):
        factors = [n for n in cfg["configs"] if n != "baseline"]
        generated = E.factorial_configs(factors, int(fac.get("max_order", 2)))
        cfg = {**cfg, "configs": {"baseline": cfg["configs"].get("baseline", {}), **generated}}
        E.LEVERS.update(generated)
    task = E.make_task(task_spec)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m, tr = cfg.get("model", {}), cfg.get("train", {})
    dm, nl, hs = int(m.get("d_model", 256)), int(m.get("n_layers", 4)), int(m.get("head_size", 64))
    steps, batch = int(tr.get("steps", 3000)), int(tr.get("batch", 64))
    lr = float(tr.get("lr", 3e-3))                      # YAML 1.1 parses '3e-3' as str -> coerce
    minutes = float(tr.get("minutes", 0.0))             # wall-clock budget (0 = use steps)
    seeds = int(cfg.get("seeds", 3))
    print(f"[config] synthetic task={task.name} configs={list(cfg['configs'])} seeds={seeds} dev={dev}", flush=True)
    cid = registry.create_campaign(task_spec, cfg, name=str(cfg.get("name", "config campaign")),
                                   capsule=registry.capture_capsule({"declarative_config": cfg}),
                                   db=cfg.get("registry_db"))
    arm_ids, results, raw_runs = {}, {}, {}
    for name in cfg["configs"]:
        arm_ids[name] = registry.ensure_arm(cid, name, cfg["configs"][name], db=cfg.get("registry_db"))
        ok, why = E.preflight(task, dm, nl, hs, name, dev, batch)
        if not ok:
            registry.set_arm_status(arm_ids[name], "preflight_rejected", db=cfg.get("registry_db"))
            print(f"  [{name}] PREFLIGHT REJECTED: {why}", flush=True); continue
        runs = [E.train_eval(task, dm, nl, hs, name, s, dev, steps, batch, lr, minutes,
                             tr.get("optimizer", "adamw"), float(tr.get("weight_decay", 0.01)),
                             int(tr.get("warmup", 0)), tr.get("muon"), fp8=bool(tr.get("fp8", False)),
                             do_compile=bool(tr.get("compile", False)),
                             gen_block=int(tr.get("gen_block", 1)),
                             eval_factors=tuple(cfg.get("eval", {}).get("length_factors", (0.5,1,2,4,8))),
                             eval_noise=tuple(cfg.get("eval", {}).get("noise", (0.05,0.10))),
                             data_seed=500_000+s, profile=bool(tr.get("profile", True))) for s in range(seeds)]
        for s, run in enumerate(runs):
            series, prof, rng_state = run.pop("_series"), run.pop("_profile"), run.pop("_rng")
            registry.record_trial(cid, arm_ids[name], s, 0, minutes*60 or steps, run,
                                  series=series, profile=prof, rng=rng_state, db=cfg.get("registry_db"))
        raw_runs[name] = runs
        agg = E._aggregate_runs(runs)
        registry.record(task_spec, name, seeds, steps, {k: list(v) for k, v in agg.items()},
                        db=cfg.get("registry_db"))
        results[name] = agg
        print(f"  [{name}] acc {agg['acc'][0]:.3f}±{agg['acc'][1]:.3f}"
              + (f" gate {agg['gate'][0]:.3f}" if "gate" in agg and name != "baseline" else ""), flush=True)
    comparisons = E._paired_comparisons(raw_runs, seed=cid)
    for name, st in comparisons.items():
        registry.record_comparison(cid, arm_ids[name], arm_ids["baseline"], "acc", "explore", st,
                                   db=cfg.get("registry_db"))
        print(f"  {name:16} Δ{st['delta']:+.3f} CI[{st['ci_low']:+.3f},{st['ci_high']:+.3f}] "
              f"p_holm={st['p_adjusted']:.3g} next_n={st['recommended_n']}")
    registry.finish_campaign(cid, db=cfg.get("registry_db"))


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
        if tr.get("init_g1g"):                         # continued pretraining from pretrained g1g
            cmd += ["--init-g1g", tr["init_g1g"]]
        elif tr.get("resume"):                         # continue from a saved run checkpoint
            cmd += ["--resume", tr["resume"]]
        for k, v in lever.items():                     # translate lever kwargs -> CLI flags
            if k in _LM_FLAG:
                cmd += [_LM_FLAG[k], str(int(v) if isinstance(v, bool) else v)]
        print(f"[config] LM run '{name}': {' '.join(cmd[2:])}", flush=True)
        subprocess.run(cmd, env={**os.environ, "PYTHONPATH": "src"})


def run(cfg_path: str):
    cfg = load(cfg_path)
    kind, data = resolve_data(cfg)
    (_run_synthetic if kind == "synthetic" else _run_lm)(data, cfg)


# Corpora for the board's LM-mode launches (doc-boundary, cached by spec hash).
_LOCAL_LM_SPEC = {"sources": [{"kind": "local",
                               "patterns": ["/thearray/git/moe-mla/**/*.py", "/thearray/git/moe-mla/**/*.md"],
                               "weight": 1.0}], "cap_mb": 8.0, "doc_boundary": True}
# Open-PerfectBlend (Apache 2.0): ~788k chat/math/code/instruction conversations (~1.4GB text,
# 388M World tokens) flattened to role-tagged plain text — real headroom for LM lever A/Bs.
# doc_boundary False: median doc is 346 tok, so within-doc windows at seq 512 would drop most of
# the corpus — flat PACKED windows (docs joined by the sep token) are the right shape for chat.
_BLEND_LM_SPEC = {"sources": [{"kind": "hf", "name": "mlabonne/open-perfectblend", "weight": 1.0}],
                  "cap_mb": 1600.0, "doc_boundary": False}
# blend-mix: the same corpus semantically packed into standard context buckets (whole docs,
# best-fit-decreasing fill, pad-masked) for mixed context-length training with reciprocal batch
# scaling (rwkv_pretrain --ctx-buckets). Bucket sizes follow the doc-length distribution.
_BLEND_MIX_SPEC = {**_BLEND_LM_SPEC, "ctx_buckets": [512, 1024, 2048, 4096, 8192, 16384, 32768]}
CORPORA = {"local": _LOCAL_LM_SPEC, "blend": _BLEND_LM_SPEC, "blend-mix": _BLEND_MIX_SPEC}


def run_lm(levers, model, train, corpus="local"):
    """Run a set of named levers on an LM corpus via rwkv_pretrain — used by the board's LM
    mode so top/lmtp/bst/jtp (which need a token future) are launchable. Lever kwargs come from
    experiment.LEVERS (single source of truth); each run appears in the main trainboard leaderboard.
    corpus: 'local' (repo code+docs, 1.9M tok) or 'blend' (Open-PerfectBlend, ~450M tok)."""
    from rwkv_lab.experiment import LEVERS
    from rwkv_lab.build_corpus import resolve_corpus, resolve_buckets
    spec = CORPORA[corpus]
    if "ctx_buckets" in spec:                             # mixed context-length training
        data_args, off_path = ["--ctx-buckets", resolve_buckets(spec)], None
    else:
        bin_path, off_path = resolve_corpus(spec)
        data_args = ["--data", bin_path]
    for name in levers:
        lever = LEVERS.get(name, {})
        run_dir = f"runs/lm_{name}" if corpus == "local" else f"runs/lm_{corpus}_{name}"
        cmd = [sys.executable, "-m", "rwkv_lab.rwkv_pretrain", *data_args, "--out", run_dir,
               "--d-model", str(model.get("d_model", 256)), "--n-layers", str(model.get("n_layers", 4)),
               "--head-size", str(model.get("head_size", 64)), "--batch", str(train.get("batch", 16)),
               "--seq-len", str(train.get("seq_len", 512)), "--lr", str(train.get("lr", 6e-4)),
               "--optimizer", str(train.get("optimizer", "adamw")),
               "--weight-decay", str(train.get("weight_decay", 0.1))]
        if train.get("warmup"):
            cmd += ["--warmup", str(train["warmup"])]
        if train.get("fp8"):                               # fp8 compute (torchao Float8Linear)
            cmd += ["--fp8"]
        if train.get("compile"):                           # torch.compile the train forward
            cmd += ["--compile"]
        if int(train.get("grad_accum", 1) or 1) > 1:       # effective batch = batch * grad_accum
            cmd += ["--grad-accum", str(int(train["grad_accum"]))]
        if float(train.get("ema", 0.0) or 0.0) > 0:        # EMA shadow weights (eval + ckpt)
            cmd += ["--ema", str(train["ema"])]
        m = train.get("muon")                              # Muon-variant flags -> rwkv_pretrain
        if m and train.get("optimizer") == "muon":
            cmd += ["--sm-scale", str(m["scale"]), "--sm-spectral-power", str(m["spectral_power"]),
                    "--sm-ddc-strength", str(m["ddc_strength"]), "--sm-ns-steps", str(m["ns_steps"]),
                    "--sm-tile-size", str(m["tile_size"]), "--sm-plus-norm", str(m["plus_norm"])]
            for k in ["mona", "second_moment", "rsav", "da_muon", "aro"]:
                cmd += [f"--sm-{k.replace('_', '-')}", str(int(m[k]))]
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
    rl.add_argument("--init-g1g", default="")               # continued pretraining from g1g
    rl.add_argument("--resume", default="")                 # continue from a saved run checkpoint
    rl.add_argument("--optimizer", default="adamw")
    rl.add_argument("--weight-decay", type=float, default=0.1)
    rl.add_argument("--warmup", type=int, default=0)
    rl.add_argument("--fp8", action="store_true")
    rl.add_argument("--compile", action="store_true")
    rl.add_argument("--grad-accum", type=int, default=1)
    rl.add_argument("--ema", type=float, default=0.0)
    rl.add_argument("--corpus", default="local", choices=sorted(CORPORA))
    from rwkv_lab.rwkv_pretrain import add_muon_args, muon_opts_from
    add_muon_args(rl)                                        # --sm-* Muon variants
    args = ap.parse_args()
    if args.cmd == "run-lm":
        run_lm(args.levers.split(","),
               {"d_model": args.d_model, "n_layers": args.n_layers, "head_size": args.head_size},
               {"steps": args.steps, "minutes": args.minutes, "seq_len": args.seq_len,
                "batch": args.batch, "lr": args.lr, "init_g1g": args.init_g1g, "resume": args.resume,
                "optimizer": args.optimizer, "weight_decay": args.weight_decay, "warmup": args.warmup,
                "fp8": args.fp8, "compile": args.compile, "grad_accum": args.grad_accum,
                "ema": args.ema, "muon": muon_opts_from(args)}, corpus=args.corpus)
    else:
        run(args.config)


if __name__ == "__main__":
    main()
