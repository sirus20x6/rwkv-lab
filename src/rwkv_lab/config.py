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
import argparse, json, os, subprocess, sys, time
from pathlib import Path
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
            "de_shift": "--de-shift", "de_emb_res": "--de-emb-res",
            "u_mup_base_width": "--u-mup-base-width", "u_mup_base_depth": "--u-mup-base-depth",
            "online_memory": "--online-memory", "online_memory_mode": "--online-memory-mode",
            "online_memory_dim": "--online-memory-dim", "online_memory_lr": "--online-memory-lr",
            "online_memory_retention": "--online-memory-retention",
            "online_memory_window": "--online-memory-window", "nvfp4": "--nvfp4",
            "nvfp4_rht": "--nvfp4-rht"}


def _read_train_log(path: str, mode: str) -> tuple[dict, list, dict]:
    """Normalize either trainer's JSONL into campaign metrics/curve/profile."""
    rows = []
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                try: rows.append(json.loads(line))
                except (ValueError, TypeError): pass
    evals = [r for r in rows if r.get("kind") == "eval"]
    trains = [r for r in rows if r.get("kind") == "train"]
    if not evals:
        raise RuntimeError(f"trainer produced no eval record in {path}")
    final = evals[-1]
    if mode == "lm":
        val_loss = float(final.get("val_loss", final["loss"]))
        metrics = {"acc": -val_loss, "val_loss": val_loss,
                   "ppl": float(final.get("ppl", 0.0)), "step": float(final.get("step", 0))}
    else:
        metrics = {"acc": float(final.get("top1_acc", 0.0)),
                   "top1_acc": float(final.get("top1_acc", 0.0)),
                   "loss": float(final["loss"]), "ppl": float(final.get("ppl", 0.0)),
                   "step": float(final.get("step", 0))}
        if "block_val" in final: metrics["block_val"] = float(final["block_val"])
    curve_rows = trains or evals
    series = [{"step": int(r.get("step", i)), "loss": float(r["loss"])}
              for i, r in enumerate(curve_rows) if isinstance(r.get("loss"), (int, float))]
    profile = {}
    if trains and isinstance(trains[-1].get("tok_per_sec"), (int, float)):
        profile["tok_per_sec"] = float(trains[-1]["tok_per_sec"])
        metrics["tok_per_sec"] = profile["tok_per_sec"]
    return metrics, series, profile


def _execute_trial(cmd, out_dir, *, campaign_id, arm_id, seed, budget, mode,
                   rung=0, phase="explore", db=None, rng=None, artifacts=()):
    """Execute a trainer and atomically turn its outputs into one normalized trial."""
    from rwkv_lab import registry
    started = time.time()
    try:
        subprocess.run(cmd, env={**os.environ, "PYTHONPATH": "src"}, check=True)
        metrics, series, profile = _read_train_log(os.path.join(out_dir, "train.jsonl"), mode)
        profile["train_seconds"] = time.time() - started
        metrics["train_seconds"] = profile["train_seconds"]
        tid = registry.record_trial(campaign_id, arm_id, seed, rung, budget, metrics,
                                    series=series, profile=profile,
                                    rng={**(rng or {}), "command": cmd}, phase=phase,
                                    started_ts=started, db=db)
        for kind, path in (("train_log", os.path.join(out_dir, "train.jsonl")), *artifacts):
            if path and os.path.isfile(path):
                registry.record_artifact(campaign_id, path, kind, trial_id=tid,
                                         metadata={"seed": seed, "rung": rung}, db=db)
        return metrics
    except Exception as e:
        registry.record_trial(campaign_id, arm_id, seed, rung, budget, None, status="failed",
                              error=f"{type(e).__name__}: {e}", phase=phase,
                              rng={**(rng or {}), "command": cmd}, started_ts=started, db=db)
        print(f"  trial failed: {type(e).__name__}: {e}", flush=True)
        return None


def _record_campaign_comparisons(cid, arm_ids, runs, *, db=None, metric="acc", phase="explore"):
    from rwkv_lab import experiment as E, registry
    stats = E._paired_comparisons(runs, metric=metric, seed=cid)
    if "baseline" in arm_ids:
        for name, st in stats.items():
            registry.record_comparison(cid, arm_ids[name], arm_ids["baseline"], metric, phase, st, db=db)
            print(f"  {name:18} Δ{st['delta']:+.4f} CI[{st['ci_low']:+.4f},{st['ci_high']:+.4f}] "
                  f"p_holm={st['p_adjusted']:.3g}", flush=True)
    return stats


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


def _lm_command(data_args, off_path, out_dir, model, train, lever, seed, save_path):
    cmd = [sys.executable, "-m", "rwkv_lab.rwkv_pretrain", *data_args, "--out", out_dir,
           "--d-model", str(model.get("d_model", 512)), "--n-layers", str(model.get("n_layers", 8)),
           "--head-size", str(model.get("head_size", 64)), "--batch", str(train.get("batch", 16)),
           "--seq-len", str(train.get("seq_len", 1024)), "--lr", str(train.get("lr", 6e-4)),
           "--optimizer", str(train.get("optimizer", "adamw")),
           "--weight-decay", str(train.get("weight_decay", 0.1)), "--seed", str(seed),
           "--save", save_path]
    cmd += (["--minutes", str(train["minutes"])] if train.get("minutes")
            else ["--steps", str(train.get("steps", 2000))])
    if off_path: cmd += ["--doc-offsets", off_path]
    if train.get("init_g1g"): cmd += ["--init-g1g", str(train["init_g1g"])]
    elif train.get("resume"): cmd += ["--resume", str(train["resume"])]
    if train.get("warmup"): cmd += ["--warmup", str(train["warmup"])]
    if train.get("fp8"): cmd += ["--fp8"]
    if train.get("compile"): cmd += ["--compile"]
    if int(train.get("grad_accum", 1) or 1) > 1: cmd += ["--grad-accum", str(train["grad_accum"])]
    if float(train.get("ema", 0.0) or 0.0) > 0: cmd += ["--ema", str(train["ema"])]
    muon = train.get("muon")
    if muon and train.get("optimizer") == "muon":
        for key, value in muon.items():
            cmd += [f"--sm-{key.replace('_', '-')}", str(int(value) if isinstance(value, bool) else value)]
    for key, value in lever.items():
        if key in _LM_FLAG:
            if key in ("nvfp4", "nvfp4_rht"):
                if value: cmd += [_LM_FLAG[key]]
            else:
                cmd += [_LM_FLAG[key], str(int(value) if isinstance(value, bool) else value)]
    return cmd


def _run_lm_campaign(configs, model, train, data_args, off_path, *, task, seeds=1,
                     db=None, campaign_name=""):
    from rwkv_lab import experiment as E, registry
    campaign_cfg = {"mode": "lm", "model": model, "train": train, "configs": configs,
                    "seeds": seeds, "data_args": data_args, "doc_offsets": off_path,
                    "decision_policy": {"primary_metric": "acc", "display_metric": "-val_loss",
                                        "direction": "maximize", "alpha": 0.05,
                                        "multiple_testing": "holm"}}
    cid = registry.create_campaign(task, campaign_cfg, name=campaign_name or task,
                                   capsule=registry.capture_capsule({"data_args": data_args,
                                                                     "doc_offsets": off_path}), db=db)
    arm_ids = {n: registry.ensure_arm(cid, n, cfg, db=db) for n, cfg in configs.items()}
    runs = {n: [] for n in configs}
    budget = float(train.get("minutes", 0)) * 60 or float(train.get("steps", 2000))
    try:
        for name, lever in configs.items():
            for seed in range(int(seeds)):
                out_dir = os.path.join("runs", f"campaign_{cid}", name, f"seed_{seed:04d}")
                save_path = os.path.join(out_dir, "ckpt.pt")
                cmd = _lm_command(data_args, off_path, out_dir, model, train, lever, seed, save_path)
                print(f"[config] LM campaign={cid} arm='{name}' seed={seed}: {' '.join(cmd[2:])}", flush=True)
                metrics = _execute_trial(cmd, out_dir, campaign_id=cid, arm_id=arm_ids[name],
                                         seed=seed, budget=budget, mode="lm", db=db,
                                         rng={"model_seed": seed, "data_seed": seed},
                                         artifacts=(("checkpoint", save_path),))
                if metrics: runs[name].append(metrics)
            if not runs[name]: registry.set_arm_status(arm_ids[name], "failed", db=db)
        _record_campaign_comparisons(cid, arm_ids, runs, db=db)
        for name, arm_runs in runs.items():
            if arm_runs:
                agg = E._aggregate_runs(arm_runs)
                registry.record(task, name, len(arm_runs), int(train.get("steps", 0)),
                                {k: list(v) for k, v in agg.items()}, db=db)
        registry.finish_campaign(cid, db=db)
    except BaseException:
        registry.finish_campaign(cid, status="failed", db=db); raise
    return cid


def _run_lm(data, cfg):
    bin_path, off_path = data
    return _run_lm_campaign(cfg["configs"], cfg.get("model", {}), cfg.get("train", {}),
                            ["--data", bin_path], off_path, task=f"lm:{Path(bin_path).name}",
                            seeds=int(cfg.get("seeds", 1)), db=cfg.get("registry_db"),
                            campaign_name=str(cfg.get("name", "LM campaign")))


def _dict_cli_args(values: dict) -> list[str]:
    out = []
    for key, value in values.items():
        flag = "--" + str(key).replace("_", "-")
        if value is None: continue
        if isinstance(value, bool):
            out.append(flag if value else "--no-" + flag[2:])
        else:
            if isinstance(value, (list, tuple)): value = ",".join(str(x) for x in value)
            out += [flag, str(value)]
    return out


def _run_conversion(cfg):
    """Run paired per-layer conversion variants as one normalized campaign.

    Config shape: ``conversion: {model_dir, data, layers, args: {...}}``; common
    trainer options live in ``train`` and arm-specific CLI overrides in ``configs``.
    Layer/seed pairs are the paired experimental units.
    """
    from rwkv_lab import experiment as E, registry
    conv, train = cfg["conversion"], cfg.get("train", {})
    layers = conv.get("layers", [conv.get("layer", 0)])
    if isinstance(layers, str): layers = [int(x) for x in layers.split(",") if x.strip()]
    layers = [int(x) for x in layers]
    seeds, db = int(cfg.get("seeds", 1)), cfg.get("registry_db")
    configs = cfg.get("configs", {"baseline": {}})
    task = "conversion:" + ",".join(str(x) for x in layers)
    decision = {"primary_metric": "acc", "display_metric": "top1_acc", "direction": "maximize",
                "paired_unit": "layer_x_seed", "alpha": 0.05, "multiple_testing": "holm"}
    campaign_cfg = {**cfg, "mode": "conversion", "decision_policy": decision}
    cid = registry.create_campaign(task, campaign_cfg, name=str(cfg.get("name", "conversion campaign")),
                                   capsule=registry.capture_capsule({"model_dir": conv["model_dir"],
                                                                     "data": conv["data"]}), db=db)
    arm_ids = {n: registry.ensure_arm(cid, n, lever, db=db) for n, lever in configs.items()}
    runs = {n: [] for n in configs}
    common_keys = {"steps", "seq_len", "batch_size", "state_stride", "lr", "optimizer",
                   "weight_decay", "warmup_steps", "eval_every", "eval_windows",
                   "eval_batch_size", "save_every", "device", "dtype"}
    common = {k: v for k, v in train.items() if k in common_keys}
    common.update(conv.get("args", {}))
    steps = int(common.get("steps", 10_000))
    try:
        for name, lever in configs.items():
            for layer in layers:
                for seed in range(seeds):
                    pair_seed = layer * 1_000_000 + seed
                    out_dir = os.path.join("runs", f"campaign_{cid}", name,
                                           f"layer_{layer:02d}_seed_{seed:04d}")
                    values = {**common, **lever}
                    init = values.pop("init_rwkv_ckpt", conv.get("init_rwkv_ckpt", ""))
                    if init: values["init_rwkv_ckpt"] = str(init).format(layer=layer, layer02=f"{layer:02d}")
                    cmd = [sys.executable, "-m", "rwkv_lab.convert_train", "--model-dir", conv["model_dir"],
                           "--data", conv["data"], "--layer", str(layer), "--out", out_dir,
                           "--seed", str(seed), *_dict_cli_args(values)]
                    print(f"[config] conversion campaign={cid} arm='{name}' layer={layer} seed={seed}", flush=True)
                    final_ckpt = os.path.join(out_dir, f"step_{steps - 1:06d}", "ckpt.pt")
                    best_ckpt = os.path.join(out_dir, "best", "ckpt.pt")
                    metrics = _execute_trial(cmd, out_dir, campaign_id=cid, arm_id=arm_ids[name],
                                             seed=pair_seed, budget=steps, mode="conversion", db=db,
                                             rng={"model_seed": seed, "layer": layer,
                                                  "paired_unit": pair_seed},
                                             artifacts=(("checkpoint", final_ckpt),
                                                        ("best_checkpoint", best_ckpt)))
                    if metrics:
                        metrics["layer"] = float(layer); metrics["model_seed"] = float(seed)
                        runs[name].append(metrics)
            if not runs[name]: registry.set_arm_status(arm_ids[name], "failed", db=db)
        _record_campaign_comparisons(cid, arm_ids, runs, db=db)
        for name, arm_runs in runs.items():
            if arm_runs:
                agg = E._aggregate_runs(arm_runs)
                registry.record(task, name, len(arm_runs), steps,
                                {k: list(v) for k, v in agg.items()}, db=db)
        registry.finish_campaign(cid, db=db)
    except BaseException:
        registry.finish_campaign(cid, status="failed", db=db); raise
    return cid


def run(cfg_path: str):
    cfg = load(cfg_path)
    if "conversion" in cfg:
        return _run_conversion(cfg)
    kind, data = resolve_data(cfg)
    return (_run_synthetic if kind == "synthetic" else _run_lm)(data, cfg)


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


def run_lm(levers, model, train, corpus="local", *, seeds=1, db=None, campaign_name=""):
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
    configs = {name: LEVERS.get(name, {}) for name in levers}
    return _run_lm_campaign(configs, model, train, data_args, off_path,
                            task=f"lm:{corpus}", seeds=seeds, db=db,
                            campaign_name=campaign_name or f"{corpus} LM campaign")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run"); r.add_argument("config")
    rl = sub.add_parser("run-lm")                            # board LM mode: named levers on local corpus
    rl.add_argument("--levers", required=True)
    rl.add_argument("--seeds", type=int, default=1)
    rl.add_argument("--db", default=None)
    rl.add_argument("--campaign-name", default="")
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
                "ema": args.ema, "muon": muon_opts_from(args)}, corpus=args.corpus,
               seeds=args.seeds, db=args.db, campaign_name=args.campaign_name)
    else:
        run(args.config)


if __name__ == "__main__":
    main()
