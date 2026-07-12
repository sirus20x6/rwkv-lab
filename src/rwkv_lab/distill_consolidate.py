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

import os
import sys
sys.modules.setdefault("torchvision", None)

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .rwkv8_deltanet import rwkv8_timemix_from_config
from .looped_rwkv import LoopedRWKV, lora_config_from_sd, sample_loop_count
from .build_memory_targets import load_token_stream
from .convert_train import chunked_ce, evaluate
from .lookahead_module import add_lookahead_cli, lookahead_from_args


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


def load_student(model_dir, rwkv_ckpt, decay_cap_delta, device, dtype, force_hyper=0,
                 force_lora_rank=0, force_lora_targets="receptance,key,value,output"):
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
    if "allow_neg_eigval" not in ck:
        print("WARNING: artifact metadata has no 'allow_neg_eigval' key; ASSUMING True to match "
              "convert_train's training-side default (a-scale 2). If these layers were trained "
              "with allow_neg_eigval=False this silently changes the layer function.", flush=True)
    aneg = bool(ck.get("allow_neg_eigval", True))      # constructor state (_a_scale), NOT in state_dict
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
        # hyper lanes are constructor state, reconstructed from the sd like gate_mode;
        # --loop-hyper force-enables fresh identity-init lanes on a plain artifact
        # (loss-free upgrade: hyper params at init reproduce the plain loop exactly).
        hyper = int(lsd["hyper_read"].shape[0]) if "hyper_read" in lsd else int(force_hyper)
        lrank, ltargets = lora_config_from_sd(lsd)     # constructor state, like gate_mode
        if lrank == 0 and force_lora_rank:             # fresh no-op adapters on a plain artifact
            lrank = int(force_lora_rank)
            ltargets = tuple(t for t in str(force_lora_targets).split(",") if t.strip())
        if (hyper or lrank) and not looped:
            raise SystemExit("--loop-hyper/--loop-lora-rank need a LoopedRWKV checkpoint; "
                             "this artifact carries bare cores")
        r = (LoopedRWKV(core, n_loops=loop_count, hidden_size=cfg.hidden_size,
                        gate_mode=_gate_mode_of(lsd), gate_cap=gate_cap,
                        loop_index="loop_index_embed" in lsd, hyper_lanes=hyper,
                        lora_rank=lrank, lora_targets=ltargets or ("receptance",))
             if looped else core)
        miss, unexp = r.load_state_dict(lsd, strict=False)
        if hyper and "hyper_read" not in lsd:          # force-enabled: fresh no-op lanes are expected
            miss = [m for m in miss if not str(m).startswith("hyper_")]
        if lrank and not any(str(k).startswith("loop_lora_") for k in lsd):
            miss = [m for m in miss if not str(m).startswith("loop_lora_")]
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


def logit_kl(s_hpost, t_hpost, s_head, t_head, temp, chunk=256):
    """T^2 * KL(softmax(teacher/T) || softmax(student/T)) over the lm_head logits, meaned over
    tokens (OpenMOSE's consolidation adjustment: match the teacher OUTPUT distribution, not the
    internal hidden states). Chunked over the flattened token dim so the [tokens, vocab] logits
    never fully materialize (mirrors chunked_ce). Teacher side is no-grad; grad flows via student.
    F.kl_div(input=log q, target=p) computes sum p*(log p - log q) = KL(p||q), p=teacher here."""
    H = s_hpost.shape[-1]
    s_flat = s_hpost.reshape(-1, H)
    t_flat = t_hpost.reshape(-1, H)
    n = s_flat.shape[0]
    total = s_flat.new_zeros((), dtype=torch.float32)
    for i in range(0, n, chunk):
        s_lp = F.log_softmax(s_head(s_flat[i:i + chunk]).float() / temp, dim=-1)
        with torch.no_grad():
            t_p = F.softmax(t_head(t_flat[i:i + chunk]).float() / temp, dim=-1)
        total = total + F.kl_div(s_lp, t_p, reduction="sum")
    return (total / n) * (temp * temp)


# --consolidation preset: the recommended enhancement bundle for the consolidate
# stage, per the 2026-07 research review (loop levers are re-learned HERE — assembly
# strips isolation-trained loops — and the lookahead objectives have their gradient
# path here). Opt-in and override-respecting: any flag the user passes explicitly
# wins over the preset value, and baseline runs simply omit --consolidation.
# Deliberately NOT included: --loop-lora-rank / --nextlat-jump-* / --concept-*
# (unproven on this model; add as separate A/B arms), full-rank TOP (lm_head-sized
# optimizer memory; rank-256 is the safe default).
CONSOLIDATION_PRESET = {
    "loop_hyper": 2,           # largest measured loop-capacity lever (phi 0.45->0.65)
    "loop_sample": "poisson",  # dynamic depth beats fixed n (4 papers)
    "loop_iter_consist": 0.1,  # + sampling = LoopFormer shortcut-consistency recipe
    "nextlat_weight": 1.0,     # ppl-neutral belief-state pressure (kl/d ride defaults)
    "top_weight": 0.5,         # token-order ranking aux
    "top_rank": 256,           # factored head: avoids the 620M full-vocab matrix
    "ce_weight": 0.1,          # papers pair aux objectives with an NTP-style loss
    "kl_weight": 1.0,          # OpenMOSE: adjust accumulated per-layer residuals with logit-KL
}


def apply_consolidation_preset(args, argv=None):
    """Set preset values for every flag the user did NOT pass on the command line;
    explicit flags always win — including an explicit default like --loop-hyper 0
    (comparing against parser defaults could not see that; scanning argv can).
    Returns {flag: value} actually applied (for the launch banner)."""
    argv = sys.argv[1:] if argv is None else argv
    applied = {}
    for k, v in CONSOLIDATION_PRESET.items():
        opt = "--" + k.replace("_", "-")
        explicit = any(a == opt or a.startswith(opt + "=") for a in argv)
        if not explicit:
            setattr(args, k, v)
            applied[k] = v
    return applied


def build_parser():
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
    ap.add_argument("--ce-weight", type=float, default=0.0,
                    help="optional student LM CE loss weight (the lookahead papers pair "
                         "their aux objectives with an NTP-style loss; default off)")
    ap.add_argument("--kl-weight", type=float, default=0.0,
                    help="logit-KL consolidation weight (OpenMOSE): T^2 * KL(softmax(teacher/T) || "
                         "softmax(student/T)) on the final lm_head logits. Matches the teacher OUTPUT "
                         "distribution rather than internal hidden states, letting converted layers absorb "
                         "each other's residual error. 0=off; try 1.0. Pair with --w-resid 0 for pure-KL.")
    ap.add_argument("--kl-temp", type=float, default=1.0,
                    help="softmax temperature T for --kl-weight. 1.0=exact match; >1 softens (more "
                         "dark-knowledge). Loss is scaled by T^2 so its magnitude is ~T-independent.")
    ap.add_argument("--w-resid", type=float, default=1.0,
                    help="weight on the residual-stream relative-MSE (hidden-state matching, the legacy "
                         "consolidation loss). 1.0=unchanged; 0 = OpenMOSE pure-logit-KL (needs --kl-weight>0).")
    ap.add_argument("--loop-hyper", type=int, default=0,
                    help="force-enable K>=2 hyper-connection lanes at the loop boundary on a plain "
                         "LoopedRWKV artifact (fresh identity-init = loss-free upgrade); 0 = keep "
                         "whatever the checkpoint carries.")
    ap.add_argument("--loop-lora-rank", type=int, default=0,
                    help="force-enable per-pass LoRA (rank R, B zero-init = loss-free upgrade) on a "
                         "plain LoopedRWKV artifact; 0 = keep whatever the checkpoint carries.")
    ap.add_argument("--loop-lora-targets", default="receptance,key,value,output",
                    help="core linears the forced per-pass LoRA adapts (ignored when the ckpt "
                         "already carries adapters).")
    ap.add_argument("--loop-sample", default="off", choices=["off", "uniform", "poisson"],
                    help="sample the training loop count per step (dynamic beats fixed n); "
                         "evals always run at the checkpoint's full loop count.")
    ap.add_argument("--loop-iter-consist", type=float, default=0.0,
                    help="equilibrium-internalization weight: pull earlier loop iterates toward "
                         "sg(final iterate), per layer, averaged. Paper-analog weight 0.1. 0=off.")
    add_lookahead_cli(ap)
    ap.add_argument("--consolidation", action="store_true",
                    help="enable the recommended consolidation bundle: "
                         + ", ".join(f"{k}={v}" for k, v in CONSOLIDATION_PRESET.items())
                         + ". Explicit flags override preset values. Assumes a LoopedRWKV "
                         "artifact (loop levers fail loud on bare cores).")
    return ap


def main():
    ap = build_parser()
    args = ap.parse_args()
    if args.consolidation:
        applied = apply_consolidation_preset(args)
        kept = {k: getattr(args, k) for k in CONSOLIDATION_PRESET if k not in applied}
        print("--consolidation preset: " + ", ".join(f"{k}={v}" for k, v in applied.items())
              + (("  |  explicit overrides kept: "
                  + ", ".join(f"{k}={v}" for k, v in kept.items())) if kept else ""),
              flush=True)
    if args.loop_hyper == 1:
        raise SystemExit("--loop-hyper 1: K=1 hyper-connections are provably no better than a "
                         "plain residual (HC paper); use 2, or 0 to keep the checkpoint's lanes.")
    if args.kl_temp <= 0:
        raise SystemExit(f"--kl-temp must be > 0 (it divides the logits), got {args.kl_temp}")
    if args.kl_weight < 0 or args.w_resid < 0:
        raise SystemExit(f"--kl-weight/--w-resid must be >= 0, got {args.kl_weight}/{args.w_resid}")
    for _fl, _hz, _w in (("--nextlat-d", args.nextlat_d, args.nextlat_weight),
                         ("--nextlat-jump-k", args.nextlat_jump_k, args.nextlat_jump_weight),
                         ("--concept-chunk", args.concept_chunk, args.concept_weight)):
        if _w > 0 and _hz >= args.seqlen:    # parse-time, not step-1-after-model-load
            raise SystemExit(f"{_fl} {_hz} must be < --seqlen {args.seqlen}.")
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
        args.model_dir, args.rwkv_ckpt, args.decay_cap_delta, dev, dtype,
        force_hyper=args.loop_hyper, force_lora_rank=args.loop_lora_rank,
        force_lora_targets=args.loop_lora_targets)
    all_wrappers = [s_layers[L].linear_attn for L in converted] if looped else []
    loop_sampling = looped and loop_count > 1 and args.loop_sample != "off"
    wrappers = all_wrappers if loop_sampling else []
    rng_loop = np.random.default_rng(777)
    ic_on = args.loop_iter_consist > 0
    if ic_on and (not looped or loop_count < 2):
        raise SystemExit("--loop-iter-consist needs a LoopedRWKV checkpoint with loop_count > 1")
    for r in all_wrappers:
        r.iter_consist = ic_on
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
    # --- lookahead aux objectives (NextLat + TOP) + optional CE; all default OFF ---
    V_, D_ = student.lm_head.weight.shape
    la = lookahead_from_args(args, D_, V_, student.lm_head)
    la_params = []
    if la is not None:
        la.to(device=dev, dtype=dtype).train()
        la_params = list(la.parameters())
        print(f"lookahead: nextlat={'on' if la.nextlat else 'off'} "
              f"top={'on' if la.top else 'off'} "
              f"({sum(p.numel() for p in la_params)/1e6:.1f}M aux params, AdamW)", flush=True)
    aux_on = la is not None or args.ce_weight > 0
    kl_on = args.kl_weight > 0
    norm_box, t_norm_box = {}, {}
    if aux_on or kl_on:  # objectives consume the POST-final-norm hidden (what lm_head sees)
        norm = getattr(s_tm, "norm", None)
        if norm is None:
            raise RuntimeError("student text model has no .norm; --nextlat/--top/--ce/--kl-weight need it")
        norm.register_forward_hook(_hook(norm_box, "h"))
    if kl_on:            # teacher's post-final-norm hidden -> teacher logits (the KL target)
        t_norm = getattr(t_tm, "norm", None)
        if t_norm is None:
            raise RuntimeError("teacher text model has no .norm; --kl-weight needs it")
        t_norm.register_forward_hook(_hook(t_norm_box, "h"))
    groups = [{"params": params}]
    if la_params:
        groups.append({"params": la_params, "lr": args.lookahead_lr or args.lr})
    opt = torch.optim.AdamW(groups, lr=args.lr, betas=(0.9, 0.95))
    if la_params:
        # per-group-safe cosine: same curve as CosineAnnealingLR(eta_min=0.05*lr)
        # but MULTIPLICATIVE, so each group anneals to 5% of its own base lr
        # (a shared eta_min scalar would mis-floor --lookahead-lr; codex #1)
        lam = lambda s: 0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * min(s, args.steps) / args.steps))
        sch = torch.optim.lr_scheduler.LambdaLR(opt, lam)
    else:
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.steps, args.lr * 0.05)

    emit, sidecar = dash_setup(args.run_name, converted, args.model_dir, args.steps)
    toks = load_token_stream(args.data)
    tp = evaluate(t_tm, teacher.lm_head, toks, args.eval_windows, args.eval_seqlen, dev)
    sp = evaluate(s_tm, student.lm_head, toks, args.eval_windows, args.eval_seqlen, dev)
    print(f"teacher ppl={tp['ppl']:.3f}  |  student ppl(start)={sp['ppl']:.3f} "
          f"(distilling {len(converted)} RWKV layers)", flush=True)
    emit({"kind": "eval", "step": 0, "ppl": sp["ppl"], "top1_acc": sp["top1_acc"]})
    sidecar(0)

    extra = la.extra_tokens if la is not None else 0   # future tokens for TOP targets
    if args.ce_weight > 0:
        extra = max(extra, 1)                          # CE needs the +1 label
    N = len(toks); T = args.seqlen; maxs = N - (T + 1 + extra)
    rng = np.random.default_rng(0)
    t0 = time.time()
    best = float("inf")  # save on the first eval so args.out is always written (codex #7)
    for step in range(1, args.steps + 1):
        s0 = int(rng.integers(0, maxs + 1))
        ids_full = torch.as_tensor(np.asarray(toks[s0:s0 + T + extra], dtype=np.int64),
                                   device=dev).unsqueeze(0)
        x = ids_full[:, :T]
        loop_k = loop_count
        if loop_sampling:
            loop_k = sample_loop_count(args.loop_sample, loop_count, rng_loop)
            for r in wrappers:
                r.n_loops = loop_k
        try:
            with torch.no_grad():
                teacher(input_ids=x, use_cache=False)
            student(input_ids=x, use_cache=False)
        finally:
            if loop_sampling:              # exception-safe: evals/saves must see full depth
                for r in wrappers:
                    r.n_loops = loop_count
        rmse = x.new_zeros((), dtype=torch.float32)
        if args.w_resid > 0:               # residual-stream relative-MSE (hidden-state match)
            for i in range(nL):
                t = t_h[i].detach().float()
                rmse = rmse + F.mse_loss(s_h[i].float(), t) / (t.pow(2).mean() + 1e-6)
            rmse = rmse / nL
        distill = rmse
        loss = args.w_resid * distill
        parts = {}
        if kl_on:                          # OpenMOSE logit-KL: match teacher OUTPUT distribution
            kl = logit_kl(norm_box["h"], t_norm_box["h"], student.lm_head, teacher.lm_head, args.kl_temp)
            loss = loss + args.kl_weight * kl
            parts["kl"] = float(kl.detach())
        if ic_on:
            ics = [r.last_iter_consist for r in all_wrappers if r.last_iter_consist is not None]
            if ics:
                ic = torch.stack(ics).mean()
                loss = loss + args.loop_iter_consist * ic
                parts["iter_consist"] = float(ic)
        if aux_on:
            h_post = norm_box["h"]                     # student's post-norm final hidden
            if la is not None:
                aux = la.compute(h_post, ids_full, s_tm.embed_tokens, student.lm_head)
                loss = loss + aux.pop("aux_total")
                parts.update(aux)
            if args.ce_weight > 0:
                ce = chunked_ce(h_post, student.lm_head, ids_full[:, 1:T + 1])
                loss = loss + args.ce_weight * ce
                parts["ce"] = float(ce)
        opt.zero_grad(); loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(params + la_params, 1.0)
        opt.step(); sch.step()
        rec = {"kind": "train", "step": step, "loss": float(loss), "gnorm": float(gn),
               "lr": sch.get_last_lr()[0]}
        if aux_on or parts:
            rec["distill"] = float(distill); rec.update(parts)
        if loop_sampling:
            rec["loop_k"] = int(loop_k)
        emit(rec)
        if step % max(1, args.steps // 40) == 0:
            aux_str = "".join(f" {k}={v:.4f}" for k, v in parts.items())
            print(f"  step {step} distill_rel_mse={float(distill):.4f}{aux_str} "
                  f"gnorm={float(gn):.2f} tok/s={step*T/(time.time()-t0):.0f}", flush=True)
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
                out_path = Path(args.out)
                tmp_path = out_path.with_name(out_path.name + ".tmp")
                torch.save(blob, tmp_path)
                os.replace(tmp_path, out_path)
                if la is not None:  # training-only aux heads, sidecar for resume/inspection
                    # CPU tensors (codex #5); non-.pt name so layer-ckpt globs
                    # like assemble_looped's `*.pt` never sweep it up (codex #6)
                    torch.save({"state_dict": {k: v.detach().cpu()
                                               for k, v in la.state_dict().items()},
                                "config": {k: getattr(args, k) for k in
                                           ("nextlat_weight", "nextlat_kl_weight", "nextlat_d",
                                            "nextlat_hidden", "nextlat_jump_k", "nextlat_jump_weight",
                                            "top_weight", "top_window", "top_rank", "top_init",
                                            "concept_weight", "concept_chunk", "concept_segments",
                                            "concept_codes", "concept_vq_weight")}},
                               args.out + ".lookahead")
    emit({"kind": "checkpoint", "step": args.steps}); sidecar(args.steps)
    print(f"\nDONE: teacher {tp['ppl']:.2f} -> student start {sp['ppl']:.2f} -> best {best:.2f}", flush=True)
    print(f"saved best -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
