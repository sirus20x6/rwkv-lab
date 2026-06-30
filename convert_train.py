#!/usr/bin/env python
"""Stages 1-5 — single-layer GDN->RWKV-7 conversion trainer with SMT/DMT.

A dedicated, composable trainer for converting ONE Gated-DeltaNet layer to RWKV-7
with the full supervision stack, deliberately separate from the 2300-line MLA
train_mla.py so the working MLA pipeline is untouched. It reuses train_mla/
load_converted/layer_swap/smt_dmt helpers.

Loss stack (plan glistening-inventing-garden.md):
  * LM cross-entropy (student logits)                          [Stage baseline]
  * block-output MSE  (student RWKV out vs frozen teacher GDN) [Stage 1]
  * SMT one-step memory transition (vs P(teacher state))       [Stage 2]
  * DMT closed-loop rollout (student's own state)              [Stage 3]
  * (logit-KL to the fully-original teacher is optional/off by default — it
     needs a second full forward; SMT/DMT is the actual stability fix.)

Stability: hard decay cap on the student kernel, state-norm/update-ratio in the
SMT/DMT losses, NaN/Inf guard that skips the step. Param groups give the decay/
time params a low LR (the sensitive ones); MuonClip integration is flagged for
the real run (raw ChatGPT LR multipliers are wrong here — see the MuonClip units
note).

The teacher GDN layer is the SAME module object that was at layer L before the
swap (held by reference, frozen) — for backward conversion its input is identical
to the student's, so it is a faithful local teacher.

This is a research scaffold for the real run; smoke it with --steps 5 --seq-len
256 first. Codex-reviewed.
"""
from __future__ import annotations

import sys
# torch 2.11 vs torchvision::nms ABI mismatch; Qwen3.5 multimodal modeling pulls
# torchvision for the unused vision tower. Mark unavailable BEFORE transformers.
sys.modules.setdefault("torchvision", None)

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# trainboard run-control signal hooks: SIGUSR1 = checkpoint-now; SIGINT = graceful
# stop (save+exit). Harmless standalone — the dashboard sends these signals; absent
# the dashboard they simply never fire. (This IS the canonical trainer; the
# dashboard2/instrumented/ copy adds only the sys.path shim to run from its subdir.)
import signal
_SIG = {"ckpt": False, "stop": False}
def _on_sigusr1(_s, _f): _SIG["ckpt"] = True
def _on_sigint(_s, _f): _SIG["stop"] = True

from build_memory_targets import _GDNStateCapture, load_token_stream, find_gdn_layer
from smt_dmt import BilinearStateCodec, MemoryTargetCache, smt_transition_loss, dmt_rollout_loss, fit_codec

# --- performance: TF32 on the Blackwell tensor cores for fp32 matmuls. The bf16
# forward/backward and the bf16 teacher targets are UNAFFECTED (TF32 only touches fp32
# matmuls); what speeds up is the fp32 path — the MUON+/Muon^p Newton-Schulz
# orthogonalization on the 4096^2 mix matrices, the codec, and fp32 loss reductions
# (block-MSE/CKA). NS is already an approximation so TF32's 10-bit mantissa is harmless,
# and the distillation targets don't change. (Runs at import; a process that already
# imported this module is unaffected.)
if torch.cuda.is_available():
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def rollout_len_for_step(step, total, schedule, stride):
    """DMT rollout-horizon curriculum: grow the closed-loop horizon over training
    (short rollouts first, lengthen only as state stays stable). Returns a token
    length that is a multiple of stride and <= total window."""
    frac = step / max(total - 1, 1)
    rl = schedule[min(int(frac * len(schedule)), len(schedule) - 1)]
    return max(stride, (rl // stride) * stride)


def _set_decay_cap(module, delta: float):
    module.decay_cap_delta = float(delta)
    module._w_floor = (math.log(-math.log(1.0 - delta)) if delta > 0 else float("-inf"))


def _cur_lr(opt):
    """LR to log: the Muon group for MuonClip (its param_groups[0] is the Adam
    group), else the first group's lr."""
    for g in opt.param_groups:
        if g.get("use_muon"):
            return g["lr"]
    g = opt.param_groups[0]
    return g.get("scheduled_lr", g["lr"])  # schedulefree writes the warmup-adjusted lr here


def _zero_like_loss(ref):
    return ref.new_zeros((), dtype=torch.float32)


def _finite_float(x, default=0.0):
    if x is None:
        return default
    return float(x.detach() if torch.is_tensor(x) else x)


def _save_ckpt(out, step, student, codec, args, opt=None):
    """Write step_<step>/{config.json, ckpt.pt}. For --optimizer schedulefree the
    caller must opt.eval() first so `student` holds the averaged x weights. With
    --save-optimizer, also persist optimizer state so --init-rwkv-ckpt resumes warm."""
    sd = write_sidecar_config(out, step, args)
    blob = {"student": student.state_dict(), "codec": codec.state_dict(), "args": vars(args)}
    if opt is not None and getattr(args, "save_optimizer", True):
        blob["opt"], blob["opt_type"] = opt.state_dict(), args.optimizer
    torch.save(blob, sd / "ckpt.pt")
    return sd


def write_sidecar_config(out, step, args):
    """Write runs/<name>/step_<N>/config.json in train_mla's sidecar shape so the
    dashboard architecture panel renders convert runs: model arch (from
    model_dir), MLA layers (from patch_dir manifest), which layers are RWKV-8
    (rwkv8_deltanet_layers), the trained layer + freeze state, and param counts."""
    prior = [int(x) for x in str(args.prior_rwkv_layers).split(",") if str(x).strip()]
    rwkv_all = sorted(set(prior + [args.layer]))
    cfg = {
        "model_dir": args.model_dir,
        "patch_dir": args.patch_dir or "",
        "rwkv8_deltanet_layers": ",".join(str(i) for i in rwkv_all),
        "rwkv8_swap_mode": "timemix",
        "train_rwkv8_layers": str(args.layer),  # -> freeze_mode "rwkv8_layers"
        "install_mtp": 0,
        "engram_enabled": 0,
        "freeze_non_mla": 1,
    }
    sd = Path(out) / f"step_{step:06d}"
    sd.mkdir(parents=True, exist_ok=True)
    (sd / "config.json").write_text(json.dumps({"step": step, "config": cfg}, indent=2))
    return sd


def _save_best(out, step, ppl, student, codec, args, opt=None):
    """ATOMICALLY save the best-eval checkpoint to out/best/. Called on EVERY eval
    improvement, so the minimum can never slip between periodic (--save-every) saves.
    Writes to a temp file then os.replace -> a crash mid-save can't corrupt best/. With
    --save-optimizer, persists optimizer state so --init-rwkv-ckpt (-> best/) resumes warm."""
    bd = Path(out) / "best"
    bd.mkdir(parents=True, exist_ok=True)
    tmp = bd / "ckpt.pt.tmp"
    blob = {"student": student.state_dict(), "codec": codec.state_dict(),
            "args": vars(args), "step": int(step), "ppl": float(ppl)}
    if opt is not None and getattr(args, "save_optimizer", True):
        blob["opt"], blob["opt_type"] = opt.state_dict(), args.optimizer
    torch.save(blob, tmp)
    os.replace(tmp, bd / "ckpt.pt")
    (bd / "best.json").write_text(json.dumps({"step": int(step), "ppl": float(ppl)}))


def resolve_best_ckpt(path):
    """Resolve ANY run path to that run's BEST checkpoint. Restarting from a
    non-best checkpoint must be impossible: given a run dir, a step_*/ckpt.pt inside
    it, or a bare ckpt, return the lowest-eval-ppl checkpoint available. A direct
    non-run file (e.g. a converted_layers_lib/L##.pt) passes through unchanged."""
    p = Path(path)
    run_root = None
    for cand in [p, *p.parents]:
        if (cand / "train.jsonl").exists() or (cand / "best" / "ckpt.pt").exists():
            run_root = cand
            break
    if run_root is None:
        return str(p)  # not a run (library/raw ckpt) -> use as given
    best = run_root / "best" / "ckpt.pt"
    if best.exists():
        bj = run_root / "best" / "best.json"
        info = f" (ppl {json.loads(bj.read_text())['ppl']:.3f})" if bj.exists() else ""
        print(f"resolve_best_ckpt: {path} -> {best}{info}", flush=True)
        return str(best)
    # legacy run without best/: pick the lowest-ppl SAVED step from train.jsonl
    evals = {}
    tj = run_root / "train.jsonl"
    if tj.exists():
        for line in tj.read_text().splitlines():
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("kind") == "eval" and r.get("ppl") is not None:
                evals[r["step"]] = r["ppl"]
    saved = {}
    for d in run_root.glob("step_*"):
        if (d / "ckpt.pt").exists():
            try:
                s = int(d.name.split("_")[1])
            except Exception:
                continue
            if s in evals:
                saved[s] = evals[s]
    if saved:
        bs = min(saved, key=saved.get)
        print(f"resolve_best_ckpt: {path} -> step_{bs:06d} (best SAVED, ppl {saved[bs]:.3f}; "
              f"no best/ — legacy run)", flush=True)
        return str(run_root / f"step_{bs:06d}" / "ckpt.pt")
    return str(p)


def build(args):
    """Load the model, hold the teacher GDN layer L, swap in the RWKV student,
    freeze everything else. Returns a dict of handles."""
    import layer_swap
    from rwkv8_deltanet import RWKV8TimeMixDeltaNet  # noqa

    if args.patch_dir:
        from load_converted import load_converted_model
        prior = args.prior_rwkv_layers or None
        model, _ = load_converted_model(
            model_dir=args.model_dir, patch_dir=args.patch_dir,
            device_map={"": args.device}, dtype=getattr(torch, args.dtype),
            freeze_non_mla=True, install_mtp=False,
            rwkv8_deltanet_layers=prior, rwkv8_swap_mode="timemix",
        )
    else:
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            args.model_dir, dtype=getattr(torch, args.dtype), low_cpu_mem_usage=True,
        ).to(args.device)
    model.eval()

    # decoder layers + target (base ForCausalLM -> model.model.layers;
    # MLA conditional-gen wrapper -> model.model.language_model.layers)
    if hasattr(model.model, "language_model"):
        text_model = model.model.language_model
        layers_path = "model.language_model.layers"
    else:
        text_model = model.model
        layers_path = "model.layers"
    layers = text_model.layers
    decoder_layer = layers[args.layer]
    teacher_gdn = decoder_layer.linear_attn  # original GDN module (kept by ref)
    assert teacher_gdn.__class__.__name__.endswith("GatedDeltaNet"), teacher_gdn.__class__.__name__
    for p in teacher_gdn.parameters():
        p.requires_grad_(False)
    teacher_cap = _GDNStateCapture(teacher_gdn.chunk_gated_delta_rule, args.state_stride)
    teacher_gdn.chunk_gated_delta_rule = teacher_cap

    # swap in the RWKV-7 student (inherits out_proj/value from GDN; rest paper init)
    layer_swap.convert_deltanet_layer_to_rwkv8(
        model, args.layer, mode="timemix",
        layers_path=layers_path,
        timemix_num_heads=64, timemix_head_size=64,
        timemix_depth_n_layer=text_model.config.num_hidden_layers,
        timemix_decay_cap_delta=args.decay_cap_delta,  # set before init -> decay init clamps to cap
        timemix_allow_neg_eigval=args.allow_neg_eigval,  # match GDN's negative eigenvalues
    )
    core = decoder_layer.linear_attn
    _set_decay_cap(core, args.decay_cap_delta)  # idempotent safety

    # Train the student AS a LoopedRWKV so the loop (weight-tied refinement passes) is
    # trained DURING per-layer conversion, not bolted on afterward. residual_weight
    # zero-init => loop == single-pass at start, so the looped student matches the
    # teacher at least as well as a bare core, with loop_count-1 passes of extra
    # refinement capacity (the lever to push an isolated layer below its single-pass floor).
    if args.loop_count > 1:
        from looped_rwkv import LoopedRWKV
        _p0 = next(core.parameters())  # core is already on-device; move the new
        student = LoopedRWKV(core, n_loops=args.loop_count).to(device=_p0.device, dtype=_p0.dtype)
        decoder_layer.linear_attn = student  # iter_norm + residual_weight onto the same device/dtype
    else:
        student = core

    # optional warm-start from a previously-trained library/convert_train ckpt.
    init_opt_state = init_opt_type = None  # restored after make_optimizer (warm-resume momentum)
    if getattr(args, "init_rwkv_ckpt", ""):
        src = resolve_best_ckpt(args.init_rwkv_ckpt)  # restarting ALWAYS uses the run's BEST ckpt
        blob = torch.load(src, map_location="cpu", weights_only=False)
        if isinstance(blob, dict):
            init_opt_state, init_opt_type = blob.get("opt"), blob.get("opt_type")
        is_convert = isinstance(blob, dict) and "student" in blob
        if is_convert:                                        # a convert_train ckpt (isolation run)
            sd_src = blob["student"]
        elif isinstance(blob, dict):                          # library file ("state_dict") or a raw sd
            sd_src = blob.get("state_dict", blob)
        else:
            sd_src = blob
        src_looped = any(k.startswith("core.") for k in sd_src)  # looped sd vs bare core sd
        if args.loop_count > 1:
            # student is LoopedRWKV. KEEP the loop (residual_weight + iter_norm) ONLY when the
            # source is a prior ISOLATION convert_train ckpt — its loop was trained for THIS
            # layer and is correct (stripping it dropped 12.031 -> 12.197 single-pass). A LIBRARY
            # file's loop was trained in the cumulative stack and is WRONG here (it jumped step-0
            # 12.1 -> 15.4), so strip it to the single-pass core and re-learn the loop.
            if src_looped and is_convert:
                load_sd = dict(sd_src)                        # full: core.* + residual_weight + iter_norm.*
                strip_loop = False
            elif src_looped:
                load_sd = {k: v for k, v in sd_src.items() if k.startswith("core.")}
                strip_loop = True
            else:
                load_sd = {f"core.{k}": v for k, v in sd_src.items()}
                strip_loop = True
            miss, unexp = student.load_state_dict(load_sd, strict=False)
            if strip_loop:                                    # residual_weight/iter_norm legitimately absent
                miss = [m for m in miss if not (m == "residual_weight" or m.startswith("iter_norm."))]
            n_loaded = len(load_sd)
        else:
            load_sd = {k[len("core."):]: v for k, v in sd_src.items() if k.startswith("core.")}
            if not load_sd:  # already a bare core state_dict (no core. prefix)
                load_sd = {k: v for k, v in sd_src.items()
                           if k != "residual_weight" and not k.startswith("iter_norm.")}
            miss, unexp = student.load_state_dict(load_sd, strict=False)
            n_loaded = len(load_sd)
        if miss or unexp:  # same classes on both sides -> should be exact; fail loud, don't half-init
            raise SystemExit(f"warm-start key mismatch from {src}: "
                             f"missing={list(miss)[:6]} unexpected={list(unexp)[:6]}")
        print(f"warm-start from {src}: loaded {n_loaded} tensors "
              f"(loop_count={args.loop_count}) OK", flush=True)

    # freeze everything except the student (+ codec added later)
    for p in model.parameters():
        p.requires_grad_(False)
    for p in student.parameters():
        p.requires_grad_(True)

    # hooks: capture layer-L input h and the student block output
    box = {}
    decoder_layer.linear_attn.register_forward_pre_hook(
        lambda m, a, kw: box.__setitem__("h", (a[0] if a else kw["hidden_states"])),
        with_kwargs=True)
    decoder_layer.linear_attn.register_forward_hook(
        lambda m, a, o: box.__setitem__("y", o))

    # codec is a FIXED GDN->RWKV state-space map (pre-fit-then-frozen via
    # --codec-pretrain, else random-init frozen). SMT/DMT detach its output, so
    # it never receives gradient; freeze it explicitly to make that unambiguous.
    codec = BilinearStateCodec(
        gdn_heads=teacher_gdn.num_v_heads, gdn_dk=teacher_gdn.head_k_dim, gdn_dv=teacher_gdn.head_v_dim,
        rwkv_heads=64, rwkv_dk=64, rwkv_dv=64,
    ).to(args.device)
    for p in codec.parameters():
        p.requires_grad_(False)
    if getattr(args, "compile", 0):
        text_model = torch.compile(text_model)  # graph-breaks expected at fla/hook boundaries
        print("  torch.compile: text_model wrapped (experimental)", flush=True)
    return dict(model=model, text_model=text_model, student=student,
                teacher_gdn=teacher_gdn, teacher_cap=teacher_cap, box=box, codec=codec,
                lm_head=model.lm_head, init_opt_state=init_opt_state, init_opt_type=init_opt_type)


def make_optimizer(student, codec, args, text_cfg=None):
    """AdamW (default) or DeepSeek-V4-style MuonClip (--optimizer muonclip).
    AdamW path: decay/time params low LR, projections normal. MuonClip path routes
    2D matrices -> Muon and 1D/scalar/norm -> AdamW (see _make_muonclip). The codec
    is a frozen target map (SMT/DMT detach it) so it is never in the optimizer."""
    if args.optimizer == "muonclip":
        return _make_muonclip(student, text_cfg, args)
    if args.optimizer == "schedulefree":
        return _make_schedulefree(student, codec, args)
    if args.optimizer == "spectral_muon":
        return _make_spectral_muon(student, codec, args)
    decay_names = ("w0", "w1", "w2", "a0", "a1", "a2", "k_k", "k_a")
    decay_p, proj_p, readout_p = [], [], []
    for n, p in student.named_parameters():
        if not p.requires_grad:
            continue
        tail = n.rsplit(".", 1)[-1]  # match decay params even under LoopedRWKV's "core." prefix
        if "out_proj" in n:                       # readout: a faster readout closes the
            readout_p.append(p)                   # representation->behavior lag (2604.13082)
        elif tail in decay_names:
            decay_p.append(p)
        else:
            proj_p.append(p)
    if codec is not None:                         # codec joins the readout group once it
        readout_p += [p for _, p in codec.named_parameters() if p.requires_grad]  # unfreezes (consolidation)
    groups = [
        {"params": proj_p, "lr": args.lr, "name": "rwkv_proj"},
        {"params": decay_p, "lr": args.lr * args.decay_lr_mult, "name": "rwkv_decay"},
    ]
    if readout_p:
        groups.append({"params": readout_p, "lr": args.lr, "name": "rwkv_readout"})
    return torch.optim.AdamW(groups, betas=(0.9, 0.95), weight_decay=0.0)


def _make_spectral_muon(student, codec, args):
    """SpectralMuon (spectral_muon.py): 2D matrices -> configurable Muon update,
    everything else -> built-in AdamW. All --sm-* flags default to vanilla Muon, so
    `--optimizer spectral_muon` with no flags == plain Muon. Each lever maps to a
    2026 paper; stack them freely (e.g. --sm-plus-norm row --sm-second-moment 1)."""
    from spectral_muon import SpectralMuon
    muon_p, adam_p = [], []
    for n, p in student.named_parameters():
        if not p.requires_grad:
            continue
        (muon_p if (p.ndim == 2 and min(p.shape) > 1) else adam_p).append(p)
    if codec is not None:                              # codec joins the Adam side until it
        adam_p += [p for _, p in codec.named_parameters() if p.requires_grad]  # unfreezes
    groups = [
        {"params": muon_p, "lr": args.muon_lr, "use_muon": True, "name": "muon"},
        {"params": adam_p, "lr": args.muon_adam_lr, "use_muon": False, "name": "adam"},
    ]
    opt = SpectralMuon(groups, momentum=0.95, nesterov=bool(args.sm_nesterov),
                       ns_steps=args.sm_ns_steps, cubic=bool(args.sm_cheap_cubic),
                       spectral_power=args.sm_spectral_power, power_method=args.sm_power_method,
                       second_moment=bool(args.sm_second_moment),
                       equilibrate=args.sm_equilibrate, plus_norm=args.sm_plus_norm,
                       row_uniform=bool(args.sm_row_uniform), mona=bool(args.sm_mona),
                       mona_alpha=args.sm_mona_alpha, scale=args.sm_scale,
                       ddc_strength=args.sm_ddc_strength, ddc_mode=args.sm_ddc_mode)
    print(f"  SpectralMuon: {len(muon_p)} 2D mats @ muon_lr={args.muon_lr:.1e}, "
          f"{len(adam_p)} other @ adam_lr={args.muon_adam_lr:.1e} | levers: plus={args.sm_plus_norm} "
          f"eq={args.sm_equilibrate} 2nd={bool(args.sm_second_moment)} aurora={bool(args.sm_row_uniform)} "
          f"power={args.sm_spectral_power}({args.sm_power_method}) mona={bool(args.sm_mona)} cubic={bool(args.sm_cheap_cubic)} "
          f"ns={args.sm_ns_steps}", flush=True)
    return opt


def _make_muonclip(student, text_cfg, args):
    """DeepSeek-V4-style split via train_mla's GuardedMuonClip: 2D weight matrices ->
    Muon (Newton-Schulz orthogonalized), all 1D/scalar/norm params -> AdamW. Routing
    is automatic by p.ndim==2 (so the RWKV proj/decay-LoRA/gate matrices AND the 2D
    r_k [H,N] go to Muon; the non-2D x_*, w0, a0, k_k, k_a, ln_x go to Adam). QK-clip is OFF (no-op for RWKV's
    receptance/key, which are not GQA-shaped). lr_muon is CALIBRATED: the baked-in
    0.4*sqrt(max_dim) amplifier (~25.6x @4096) makes it ~vanilla-Muon-lr/25 —
    train_mla's validated 1e-4 (5e-4 blew up two runs)."""
    from muon_helpers import _make_guarded_muonclip_class, _ParamProxy
    from muon import MuonConfig
    if text_cfg is None:
        raise SystemExit("--optimizer muonclip needs the model text config (pass it from build()).")
    muon_named = [(n, p) for n, p in student.named_parameters() if p.requires_grad]
    muon_cfg = MuonConfig(
        unified_lr=False,
        lr_muon=args.muon_lr, lr_adam=args.muon_adam_lr,
        muon_beta=0.95, muon_decay=0.0,
        adam_betas=(0.9, 0.95), adam_decay=0.0, adam_eps=1e-10,
        enable_clipping=False, log_max_logits=False, log_dir="./logs",
    )
    opt = _make_guarded_muonclip_class()(
        _ParamProxy(muon_named), text_cfg, muon_cfg,
        max_muon_ratio=5e-4, max_adam_ratio=1e-4)
    # base MuonClip.flush_metrics has a latent AttributeError (writer never created
    # when log_dir is truthy); neutralize it (train_mla does the same).
    opt.flush_metrics = (lambda *a, **kw: None).__get__(opt, type(opt))
    n_muon = sum(p.numel() for g in opt.param_groups if g.get("use_muon") for p in g["params"])
    n_adam = sum(p.numel() for g in opt.param_groups if not g.get("use_muon") for p in g["params"])
    print(f"  GuardedMuonClip (V4 split): 2D->Muon {n_muon/1e6:.3f}M @ lr_muon={args.muon_lr:.1e}; "
          f"non-2D->Adam {n_adam/1e6:.4f}M @ lr_adam={args.muon_adam_lr:.1e}", flush=True)
    return opt


def _make_schedulefree(student, codec, args):
    """Schedule-Free AdamW (Defazio 2024 / ScheduleFree+ 2026). Replaces the LR
    schedule with iterate averaging: the averaged weights x are ALWAYS a usable
    model (anytime), so the step count need not be fixed up front — ideal when a
    layer needs an unknown (large) number of steps to reach its teacher PPL. Warmup
    is INTERNAL (warmup_steps); NO external scheduler. REQUIRES opt.train() before
    training and opt.eval() before every eval/save (handled in train())."""
    from schedulefree import AdamWScheduleFree
    decay_names = ("w0", "w1", "w2", "a0", "a1", "a2", "k_k", "k_a")
    decay_p, proj_p, loop_p, readout_p = [], [], [], []
    for n, p in student.named_parameters():
        if not p.requires_grad:
            continue
        tail = n.rsplit(".", 1)[-1]  # match decay params even under LoopedRWKV's "core." prefix
        if n == "residual_weight":
            loop_p.append(p)                       # loop gates: tiny gradient -> own high LR
        elif "out_proj" in n:
            readout_p.append(p)                    # readout: faster readout closes the lag (2604.13082)
        elif tail in decay_names:
            decay_p.append(p)
        else:
            proj_p.append(p)
    if codec is not None:
        readout_p += [p for _, p in codec.named_parameters() if p.requires_grad]
    groups = [
        {"params": proj_p, "lr": args.lr},
        {"params": decay_p, "lr": args.lr * args.decay_lr_mult},
    ]
    if loop_p:
        groups.append({"params": loop_p, "lr": args.lr * args.loop_lr_mult})
    if readout_p:
        groups.append({"params": readout_p, "lr": args.lr, "name": "rwkv_readout"})
    opt = AdamWScheduleFree(groups, lr=args.lr, betas=(0.9, 0.999), eps=1e-8,
                            weight_decay=args.weight_decay, warmup_steps=max(0, args.warmup_steps), r=args.sf_r)
    n = sum(p.numel() for g in groups for p in g["params"])
    loop_str = f"/{args.lr*args.loop_lr_mult:.1e}(loop)" if loop_p else ""
    print(f"  Schedule-Free AdamW: {n/1e6:.3f}M params, lr={args.lr:.1e}(proj)/"
          f"{args.lr*args.decay_lr_mult:.1e}(decay){loop_str}, warmup={args.warmup_steps}, r={args.sf_r} "
          f"— anytime averaging, no external LR schedule", flush=True)
    return opt


def train(args):
    h = build(args)
    model, text_model, student, teacher_gdn, teacher_cap, box, codec, lm_head = (
        h["model"], h["text_model"], h["student"], h["teacher_gdn"], h["teacher_cap"],
        h["box"], h["codec"], h["lm_head"])
    init_opt_state, init_opt_type = h["init_opt_state"], h["init_opt_type"]

    # Create the run dir + log + sidecar BEFORE the codec pre-fit so the dashboard
    # lists the run (and renders the architecture panel) during the ~2k-step pre-fit
    # instead of only once training starts. truncate ("w"): each run is fresh.
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    logf = open(out / "train.jsonl", "w")
    def emit(rec):
        logf.write(json.dumps(rec) + "\n"); logf.flush()
    def write_loop_rw():  # loop-usage card data for the dashboard (no-op for a bare core)
        rw = getattr(student, "residual_weight", None)
        if rw is None:
            return
        rwl = [float(x) for x in rw.detach().float().cpu().tolist()]
        mx = max((abs(x) for x in rwl), default=0.0)
        (out / "loop_rw.json").write_text(json.dumps({
            "loop_count": int(args.loop_count), "n_layers": 1, "n_pinned": int(mx >= 0.245),
            "mean_max_rw": mx, "layers": [{"layer": int(args.layer), "max_rw": mx, "rw": rwl}]}))
    write_sidecar_config(out, 0, args)
    write_loop_rw()  # initial (zeros) so the card appears immediately for looped runs

    # trainboard: install checkpoint/stop signal handlers (main thread).
    signal.signal(signal.SIGUSR1, _on_sigusr1)
    signal.signal(signal.SIGINT, _on_sigint)
    codec_rel = None  # trainboard: surfaced into the train record when a codec is pre-fit

    if args.codec_pretrain and args.codec_cache:
        print("pre-fitting codec ...", flush=True)
        codec, codec_rel = fit_codec(student, args.codec_cache, steps=args.codec_pretrain,
                                     device=args.device, train_readout=True)
        for p in codec.parameters():
            p.requires_grad_(False)
        # trainboard: honor a stop requested DURING the (uninterruptible) pre-fit,
        # before the underfit gate — nothing to checkpoint yet, so just exit.
        if _SIG["stop"]:
            print("  [interrupt] stop requested during codec pre-fit, exiting", flush=True)
            logf.close()
            sys.exit(0)
        # Abort gate: a codec that won't fit (the near-embedding L0/L1 collapse) makes the
        # SMT/DMT targets garbage and the layer trains to noise — the silent failure that
        # destroyed the cumulative run. Fail loud here instead of producing a broken layer.
        print(f"codec pre-fit rel_rmse={codec_rel:.4f} (max allowed {args.codec_rel_max})", flush=True)
        if codec_rel > args.codec_rel_max:
            raise SystemExit(
                f"codec underfit: rel_rmse {codec_rel:.4f} > --codec-rel-max {args.codec_rel_max}. "
                f"Near-embedding layers (L0/L1) need more --codec-pretrain steps, a larger codec, or "
                f"the predictive-z fallback (smt_dmt). Refusing to train on bad SMT/DMT targets.")
    elif args.codec_pretrain and not args.codec_cache:
        raise SystemExit("--codec-pretrain set but --codec-cache empty (need build_memory_targets.py output)")

    # PC-Layer (2606.06470): reparameterize the student's Linear weights with polynomial
    # spectral preconditioning (mergeable at inference). Applied BEFORE the optimizer so
    # it trains the raw weight; pc_strength live-blends g(W) with W.
    pc_strength = [0.0]
    if args.pc_layer > 0:
        from pc_layer import apply_pc_layer
        pc_strength, n_pc = apply_pc_layer(student, level=args.pc_layer)
        pc_strength[0] = args.pc_strength
        print(f"[pc-layer] level={args.pc_layer} on {n_pc} Linears, strength={args.pc_strength}", flush=True)

    opt = make_optimizer(student, codec, args, getattr(model.config, "text_config", model.config))
    if init_opt_state is not None:  # warm-resume: restore optimizer momentum (Adam/Muon) or
        if init_opt_type == args.optimizer:  # schedulefree averaging, else a restart cold-starts it
            try:
                opt.load_state_dict(init_opt_state)
                print(f"  resumed optimizer state ({args.optimizer}) from warm-start ckpt", flush=True)
            except Exception as e:
                print(f"  [warn] optimizer-state restore failed ({e}); cold-starting momentum", flush=True)
        else:
            print(f"  [warn] optimizer changed {init_opt_type}->{args.optimizer}; cold-starting momentum", flush=True)
    is_sf = args.optimizer == "schedulefree"
    # External LR schedule (warmup + optional cosine) for AdamW/MuonClip. Schedule-Free
    # owns its warmup internally and its averaging replaces decay, so it uses NO
    # external scheduler (an external LambdaLR would fight its internal lr handling).
    warmup = max(0, args.warmup_steps)
    def _lr_lambda(step):
        if warmup > 0 and step < warmup:
            return (step + 1) / warmup
        if args.lr_floor_frac >= 1.0 or args.steps <= warmup:
            return 1.0
        p = min(1.0, (step - warmup) / max(1, args.steps - warmup))
        return args.lr_floor_frac + 0.5 * (1.0 - args.lr_floor_frac) * (1.0 + math.cos(math.pi * p))
    sched = None if is_sf else torch.optim.lr_scheduler.LambdaLR(opt, _lr_lambda)
    if is_sf:
        opt.train()  # Schedule-Free: params held at y (the gradient-eval point) during training
    toks = load_token_stream(args.data)
    train_cache = MemoryTargetCache(args.train_cache) if args.train_cache else None
    if train_cache is not None:
        if train_cache.T != args.seq_len:
            raise SystemExit(f"--train-cache seq_len {train_cache.T} != --seq-len {args.seq_len}")
        if train_cache.stride != args.state_stride:
            raise SystemExit(f"--train-cache state_stride {train_cache.stride} != --state-stride {args.state_stride}")
        if args.w_lmce > 0.0:
            print("  [warn] --train-cache is local layer-target training; disabling LM CE for train steps", flush=True)
        print(f"[train-cache] local layer targets from {args.train_cache}: {train_cache.n_windows} windows", flush=True)
    N = len(toks); T = args.seq_len; max_start = N - (T + 1)
    rng = np.random.default_rng(args.seed)
    dev = args.device
    # grokking lever 1 — fixed reused trainset: cache N windows once and CYCLE them,
    # creating the small-fixed-set memorize→generalize regime where grokking occurs
    # (the default fresh-window sampling has no set to memorize, so no transition).
    fixed_starts = None
    eval_lo = 0  # disjoint-eval lower bound (set when a fixed trainset reserves the low half)
    if args.fixed_trainset > 0:
        # NOTE: --fixed-trainset MANUFACTURES the memorization regime (to study the
        # transition). It is the OPPOSITE of "best model fastest" — do NOT use it for a
        # production conversion; it is a diagnostic only.
        split = (max_start // 2) if args.disjoint_eval else max_start
        pool_rng = np.random.default_rng(args.seed + 7)
        fixed_starts = [int(pool_rng.integers(0, max(1, split) + 1)) for _ in range(args.fixed_trainset)]
        if args.disjoint_eval:
            eval_lo = min(max_start, split + T)    # eval windows start past the train pool
        print(f"[grokking] DIAGNOSTIC fixed trainset: cycling {args.fixed_trainset} windows "
              f"(eval {'disjoint' if args.disjoint_eval else 'OVERLAPPING'}) — NOT for production", flush=True)
    best_ppl = float("inf")  # save-on-improve: the eval minimum can never be missed
    t0 = time.time()
    # --- grokking diagnostics (memorization vs generalization) ---------------
    # block_ema tracks the *train* block-MSE so gen_gap = held-out block - train
    # block makes the "train fits while held-out regresses" (anti-grokking) creep
    # visible. Pure additions to train.jsonl; cost is one teacher_gdn forward per
    # eval window + a per-step EMA. See grokking_metrics.py.
    import grokking_metrics as gm
    block_ema = None
    def _largest_2d(mod):
        best, bn = None, -1
        for q in mod.parameters():
            if q.ndim == 2 and q.numel() > bn:
                best, bn = q, q.numel()
        return best
    grok_mat = _largest_2d(student) if args.log_grokking_metrics else None
    # grokking lever 3 — 2D student mix matrices for the nuclear-norm penalty
    nuc_mats = [p for p in student.parameters() if p.ndim == 2 and p.requires_grad] \
        if (args.nuc_weight > 0.0 or args.grok_autopilot) else []
    gf_ema = {}  # GrokFast: per-param slow-gradient EMA (lazy-init; amplifies the slow
    #              gradient so the model reaches the generalizing solution sooner)
    # live-tuning consumer (trainboard v2 run_controls table); fail-safe + optional.
    from live_controls import LiveControls
    ctl = LiveControls(args.control_db, Path(args.out).name, whitelist={
        "w_lmce", "w_block", "w_smt", "w_dmt", "grad_clip", "lr_scale",
        "eval_every", "save_every", "log_every",
        "weight_decay", "tail_weight_decay", "wd_tail_frac", "nuc_weight", "nuc_every",
        "grokfast_lamb", "grokfast_alpha", "readout_lr_mult",
        "sm_spectral_power", "sm_scale", "ddc_strength", "pc_strength", "llr_smax",
        "w_cos", "w_cka", "w_flow", "w_bridge", "agreement_gate"})
    if ctl.enabled:
        print(f"[live-tune] watching {args.control_db} for run '{Path(args.out).name}'", flush=True)
    # grokking autopilot — anti-collapse recovery (EMA-best + restore-best + reg
    # escalation). Off by default; the dashboard detector owns the in-place lr cool.
    from grok_autopilot import GrokAutopilot
    apilot = GrokAutopilot(student, codec, opt, out, is_sf, enabled=bool(args.grok_autopilot),
                           ema_decay=args.ap_ema_decay, collapse_thresh=args.ap_collapse_thresh,
                           patience=args.ap_patience, stall_patience=args.ap_stall_patience,
                           max_restarts=args.ap_max_restarts,
                           reg_mult=args.ap_reg_mult, restore_best=bool(args.ap_restore_best))
    if apilot.enabled:
        print(f"[autopilot] on: ema={apilot.use_ema} restore_best={apilot.restore_best} "
              f"max_restarts={apilot.max_restarts}", flush=True)
    # LLR (2605.22297): per-group LR multiplier from spectral heavy-tailedness (Hill-alpha).
    # True layerwise at consolidation (many layer-groups); within-layer here. Sets g["llr_mult"],
    # which the per-group LR block below applies alongside lr_scale.
    # relational / cross-arch distillation objectives (distill_objectives.py): extra
    # alignment-invariant terms beside block/SMT/DMT MSE. bridge fits frozen PCA projectors.
    import distill_objectives as do
    bridge = do.OPRDBridge(args.bridge_rank) if args.w_bridge > 0.0 else None
    llr = None
    if args.llr:
        from llr import LayerwiseLR
        llr = LayerwiseLR(opt, s_max=args.llr_smax, every=args.llr_every,
                          active_frac=args.llr_active_frac, total_steps=args.steps)
        print(f"[llr] heavy-tail layerwise LR: s_max={args.llr_smax} every={args.llr_every} "
              f"on {len(opt.param_groups)} groups", flush=True)
    # spectral loop levers: Hyperball Frobenius-sphere projection + river-valley switch.
    hb_R = ({p: float(p.detach().norm()) for g in opt.param_groups for p in g["params"]
             if p.ndim == 2} if args.hyperball else {})
    switched = False
    for step in range(args.steps):
        if (not switched and args.muon_to_adamw_frac > 0.0
                and args.optimizer in ("muonclip", "spectral_muon")
                and step >= args.muon_to_adamw_frac * args.steps):
            switched = True
            _prev = args.optimizer; args.optimizer = "adamw"   # river-valley: refine tail in AdamW
            opt = make_optimizer(student, codec, args, getattr(model.config, "text_config", model.config))
            # LambdaLR computes lr from ITS OWN internal counter (last_epoch), not the
            # closure's `step` — without last_epoch=step-1 here it resets to "step 0" on
            # every switch, re-triggering the warmup ramp from near-zero mid-run. Seed
            # initial_lr explicitly since last_epoch!=-1 skips LambdaLR's auto-setdefault.
            for g in opt.param_groups:
                g.setdefault("initial_lr", g["lr"])
            sched = torch.optim.lr_scheduler.LambdaLR(opt, _lr_lambda, last_epoch=step - 1)
            apilot.opt = opt  # restore-best clears momentum on apilot.opt — must track the live optimizer
            if llr is not None:
                llr.opt = opt
                for g in opt.param_groups:
                    g.setdefault("llr_mult", 1.0)
            print(f"  [river-valley] step {step}: switched {_prev} -> AdamW for the refinement tail", flush=True)
        # trainboard: act on dashboard signals at the loop top (NaN-`continue` below
        # can't swallow them). SIGUSR1 -> checkpoint without exit; SIGINT -> save+exit.
        if _SIG["ckpt"] or _SIG["stop"]:
            stopping = _SIG["stop"]
            if is_sf: opt.eval()
            _save_ckpt(out, step, student, codec, args, opt)
            reason = "interrupt" if stopping else "sigusr1"
            emit({"kind": "checkpoint", "step": step, "reason": reason})
            print(f"  [{reason}] checkpoint saved at step {step}", flush=True)
            if stopping:
                logf.close()
                sys.exit(0)
            _SIG["ckpt"] = False
            if is_sf: opt.train()
        # live-tuning: pull whitelisted overrides (no-op when --control-db missing).
        if step % max(1, args.control_poll_every) == 0:
            ctl.poll(step)
        w_lmce, w_block = ctl.get("w_lmce", args.w_lmce), ctl.get("w_block", args.w_block)
        w_cos = ctl.get("w_cos", args.w_cos); w_cka = ctl.get("w_cka", args.w_cka)
        w_flow = ctl.get("w_flow", args.w_flow); w_bridge = ctl.get("w_bridge", args.w_bridge)
        agree_gate = ctl.get("agreement_gate", args.agreement_gate)
        w_smt, w_dmt = ctl.get("w_smt", args.w_smt), ctl.get("w_dmt", args.w_dmt)
        grad_clip = ctl.get("grad_clip", args.grad_clip)
        nuc_w = ctl.get("nuc_weight", apilot.overrides.get("nuc_weight", args.nuc_weight))
        nuc_every = int(ctl.get("nuc_every", args.nuc_every))
        lr_scale = ctl.get("lr_scale", 1.0)
        readout_mult = ctl.get("readout_lr_mult", apilot.overrides.get("readout_lr_mult", args.readout_lr_mult))
        gf_lamb = ctl.get("grokfast_lamb", apilot.overrides.get("grokfast_lamb",
                          args.grokfast_lamb if args.grokfast else 0.0))
        gf_alpha = ctl.get("grokfast_alpha", args.grokfast_alpha)
        if args.optimizer == "spectral_muon" and not switched:   # live spectral_muon knobs
            sm_power = ctl.get("sm_spectral_power", args.sm_spectral_power)
            sm_scale = ctl.get("sm_scale", args.sm_scale)
            sm_ddc = ctl.get("ddc_strength", args.sm_ddc_strength)   # DDC live
            for g in opt.param_groups:
                if g.get("use_muon"):
                    g["spectral_power"], g["scale"], g["ddc_strength"] = sm_power, sm_scale, sm_ddc
        if args.pc_layer > 0:
            pc_strength[0] = ctl.get("pc_strength", args.pc_strength)    # PC-Layer live blend
        if llr is not None:
            llr.update(step, s_max=ctl.get("llr_smax", args.llr_smax))   # LLR live spread (periodic)
        eval_every = int(ctl.get("eval_every", args.eval_every))
        save_every = int(ctl.get("save_every", args.save_every))
        log_every = max(1, int(ctl.get("log_every", args.log_every)))
        # grokking lever 2 — decoupled weight decay = weight_decay (constant) + tail
        # ramp(tail_weight_decay over the last wd_tail_frac of training). 0 in the fit
        # phase unless weight_decay>0 (2605.04396 / 2606.05863). All live-tunable;
        # applied manually below so it works for adamw/muonclip (internal WD is 0).
        wd_const = ctl.get("weight_decay", apilot.overrides.get("weight_decay", args.weight_decay))
        tail_wd = ctl.get("tail_weight_decay", args.tail_weight_decay)
        wd_tfrac = ctl.get("wd_tail_frac", args.wd_tail_frac)
        t_start = args.steps * (1.0 - wd_tfrac)
        tail_phase = 0.0 if step < t_start else min(1.0, (step - t_start) / max(1.0, args.steps - t_start))
        decay_now = wd_const + tail_wd * tail_phase
        if is_sf:
            for g in opt.param_groups:
                g["weight_decay"] = decay_now  # schedulefree applies this at step time
        # periodic held-out eval -> dashboard ppl/top1 KPIs
        if eval_every > 0 and step % eval_every == 0:
            if is_sf:
                opt.eval()   # swap params to the averaged x so the eval reflects the real model
            student.eval()
            teacher_cap.want_states = False  # evaluate() only needs the block output, never states
            ev = evaluate(text_model, lm_head, toks, args.eval_windows, args.seq_len, dev,
                          teacher_gdn=teacher_gdn if args.log_grokking_metrics else None,
                          box=box if args.log_grokking_metrics else None, start_lo=eval_lo,
                          batch_size=args.eval_batch_size)
            if args.log_grokking_metrics and "block_val" in ev and block_ema is not None:
                ev["gen_gap"] = ev["block_val"] - block_ema
            emit({"kind": "eval", "step": step, **ev})
            print(f"  [eval] step {step} loss={ev['loss']:.4f} ppl={ev['ppl']:.3f} "
                  f"top1={ev['top1_acc']:.4f}", flush=True)
            write_loop_rw()  # refresh loop-usage card (residual_weight as the loop trains)
            # save the BEST eval NO MATTER WHAT -> the minimum is never lost between
            # periodic saves (which is how 12.023/12.030 slipped through before).
            if ev.get("ppl") is not None and ev["ppl"] < best_ppl:
                best_ppl = ev["ppl"]
                _save_best(out, step, ev["ppl"], student, codec, args, opt)
            # autopilot: also eval the weight-EMA (keep the lower ppl), then check for
            # anti-grokking collapse and escalate (reg up + optional restore-best).
            if args.grok_autopilot:
                ema_ev, ema_saved = apilot.eval_ema(
                    lambda: evaluate(text_model, lm_head, toks, args.eval_windows, args.seq_len, dev,
                                     start_lo=eval_lo, batch_size=args.eval_batch_size),
                    best_ppl, lambda pp: _save_best(out, step, pp, student, codec, args, opt))
                if ema_saved:
                    best_ppl = ema_ev["ppl"]
                    emit({"kind": "eval", "step": step, "ema": 1, **ema_ev})
                    print(f"  [autopilot] EMA best ppl={ema_ev['ppl']:.3f}", flush=True)
                act = apilot.on_eval(step, ev.get("ppl"), best_ppl)
                if act:
                    emit({"kind": "train", "step": step, **act})
                    print(f"  [autopilot] {act}", flush=True)
            # periodic ckpt (averaged x for schedulefree, since we're in eval mode here)
            if save_every > 0 and step > 0 and step % save_every == 0:
                _save_ckpt(out, step, student, codec, args, opt)
            student.train()
            if is_sf:
                opt.train()  # back to y for the training step

        start = None
        x = y = None
        if train_cache is None:
            start = (fixed_starts[step % len(fixed_starts)] if fixed_starts is not None
                     else int(rng.integers(0, max_start + 1)))
            ids = torch.as_tensor(np.asarray(toks[start:start + T + 1], dtype=np.int64), device=dev)
            x, y = ids[:-1].unsqueeze(0), ids[1:].unsqueeze(0)

        # student forward through the frozen backbone, or local cached layer-target training.
        hidden = None
        with torch.set_grad_enabled(True):
            if train_cache is None:
                outputs = text_model(input_ids=x, use_cache=False)
                hidden = outputs.last_hidden_state if hasattr(outputs, "last_hidden_state") else outputs[0]
                hL = box["h"]                      # [1,T,C] input to layer L (teacher-faithful)
                student_out = box["y"]             # [1,T,C] student RWKV block output

                # teacher GDN block output (+ boundary states only if SMT/DMT are active).
                # want_states gates _GDNStateCapture's expensive chunk-wise re-run — skip it
                # whenever SMT/DMT are off (e.g. an I/O-first phase) instead of paying for
                # boundary states that get thrown away below.
                need_state = w_smt > 0 or w_dmt > 0
                teacher_cap.want_states = need_state
                with torch.no_grad():
                    teacher_out = teacher_gdn(hL)
                    # teacher_cap.states: [n_bounds, B, Hv, Dk, Dv] -> [B, n_bounds, Hv, Dk, Dv]
                    S_gdn = teacher_cap.states.transpose(0, 1).contiguous() if need_state else None
            else:
                ci = int(rng.integers(0, train_cache.n_windows))
                hL = torch.as_tensor(np.asarray(train_cache.h[ci:ci + 1]), device=dev, dtype=getattr(torch, args.dtype))
                teacher_out = torch.as_tensor(np.asarray(train_cache.block_out[ci:ci + 1]), device=dev,
                                              dtype=getattr(torch, args.dtype))
                student_out = student(hL)
                need_state = w_smt > 0 or w_dmt > 0
                S_gdn = (torch.as_tensor(np.asarray(train_cache.state[ci:ci + 1]), device=dev,
                                         dtype=getattr(torch, args.dtype)) if need_state else None)

            lm_ce = chunked_ce(hidden, lm_head, y) if (w_lmce > 0.0 and hidden is not None) else _zero_like_loss(student_out)
            so, to = student_out.float(), teacher_out.float()
            if agree_gate > 0.0:                                  # trust-region per-token gating
                aw = do.agreement_weight(so, to)
                block = (aw * (so - to).pow(2)).sum() / (aw.sum() * so.shape[-1] + 1e-8)
            else:
                block = F.mse_loss(so, to)
            loss = w_lmce * lm_ce + w_block * block
            # relational / alignment-invariant distillation terms (cross-arch: match
            # direction / structure / dynamics, not coordinates). Default weights 0 = off.
            if w_cos > 0.0:
                loss = loss + w_cos * do.cosine_match(so, to)
            if w_cka > 0.0:
                loss = loss + w_cka * do.cka_loss(so, to)
            if w_flow > 0.0:
                loss = loss + w_flow * do.flow_loss(so, to)
            if w_bridge > 0.0 and bridge is not None:
                loss = loss + w_bridge * bridge.loss(so, to)
            fid_val = do.carry_fidelity(so, to) if args.distill_fidelity_log else None
            # SMT/DMT are the EXPENSIVE objectives (extra student rollouts) and pull
            # toward state fidelity, which can fight I/O fidelity (Codex review). Compute
            # them ONLY when weighted, so an I/O-only phase (w_smt=w_dmt=0) doesn't pay.
            smt = {"smt_memory": block.new_zeros(()), "smt_block": block.new_zeros(()), "smt_update_pen": 0.0}
            dmt = {"dmt_memory": block.new_zeros(()), "dmt_block": block.new_zeros(()), "dmt_state_rms": 0.0}
            target_states = None
            if need_state:
                with torch.no_grad():
                    target_states = codec(S_gdn.reshape(-1, *S_gdn.shape[2:])).reshape(
                        S_gdn.shape[0], S_gdn.shape[1], *codec.shape).detach()
            if w_smt > 0:
                smt = smt_transition_loss(student, codec, hL, stride=args.state_stride,
                                          block_out=teacher_out, target_states=target_states)
                loss = loss + w_smt * smt["smt_memory"] + args.w_smt_block * smt["smt_block"]
            if w_dmt > 0:
                # DMT rollout horizon grows over training (curriculum); re-align rl DOWN
                # to a stride multiple after clamping to T so the boundary count is exact.
                rl = rollout_len_for_step(step, args.steps, args.dmt_curriculum, args.state_stride)
                rl = max(args.state_stride, (min(rl, T) // args.state_stride) * args.state_stride)
                nb = rl // args.state_stride + 1
                dmt = dmt_rollout_loss(student, codec, hL[:, :rl],
                                       stride=args.state_stride, block_out=teacher_out[:, :rl],
                                       discount=args.dmt_discount, target_states=target_states[:, :nb])
                # dmt_block keeps I/O fidelity DURING the rollout (Codex): without it DMT
                # optimizes only latent-state trajectory, which drifts ppl up.
                loss = loss + w_dmt * dmt["dmt_memory"] + args.w_dmt_block * dmt["dmt_block"]

            # grokking lever 3 — spectral (nuclear-norm) penalty on the student's 2D
            # mix matrices: the tangential low-rank pressure L2 can't provide under
            # RMSNorm (2606.04405). SVD per matrix, so amortize with --nuc-every.
            nuc_val = 0.0
            if nuc_w > 0.0 and (step % max(1, nuc_every) == 0) and nuc_mats:
                nuc = sum(torch.linalg.matrix_norm(W.float(), ord="nuc") for W in nuc_mats)
                loss = loss + nuc_w * nuc
                nuc_val = nuc.detach()

        if not torch.isfinite(loss):
            print(f"step {step}: NON-FINITE loss -> skip", flush=True)
            opt.zero_grad(set_to_none=True)
            continue
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if gf_lamb > 0.0:                          # GrokFast: amplify the slow-varying
            for g in opt.param_groups:             # gradient component so the model reaches
                for p in g["params"]:              # the generalizing solution sooner (no delay)
                    if p.grad is None:
                        continue
                    e = gf_ema.get(p)
                    if e is None:
                        e = p.grad.detach().clone()
                    else:
                        e.mul_(gf_alpha).add_(p.grad.detach(), alpha=1.0 - gf_alpha)
                    gf_ema[p] = e
                    p.grad.add_(e, alpha=gf_lamb)
        gnorm = torch.nn.utils.clip_grad_norm_(
            [p for g in opt.param_groups for p in g["params"]], grad_clip)
        opt.step()
        if sched is not None:
            sched.step()
            for g in opt.param_groups:               # live LR multipliers (scheduled opts):
                m = lr_scale                          # global lr_scale, readout boost, LLR layerwise
                if readout_mult != 1.0 and g.get("name") == "rwkv_readout":
                    m = m * readout_mult              # boost to close the readout lag
                m = m * g.get("llr_mult", 1.0)        # LLR heavy-tail layerwise multiplier
                if m != 1.0:
                    g["lr"] = g["lr"] * m
        # grokking lever 2 (apply) — manual decoupled weight decay for non-schedulefree
        # optimizers (adamw/muonclip carry WD=0 internally, so this is the sole decay;
        # schedulefree already applied decay_now via group["weight_decay"] above).
        if not is_sf and decay_now > 0.0:
            with torch.no_grad():
                for g in opt.param_groups:
                    lr_g = g.get("scheduled_lr", g["lr"])
                    if lr_g <= 0:
                        continue
                    for p in g["params"]:
                        if p.requires_grad:
                            p.data.mul_(1.0 - lr_g * decay_now)
        # Hyperball: project each 2D weight back onto its initial Frobenius sphere.
        if hb_R:
            with torch.no_grad():
                for p, R in hb_R.items():
                    n = float(p.norm())
                    if n > 0.0:
                        p.mul_(R / n)
        if args.grok_autopilot:
            apilot.update_ema()

        if args.log_grokking_metrics:
            block_val_now = _finite_float(block)
            block_ema = block_val_now if block_ema is None else 0.98 * block_ema + 0.02 * block_val_now
        if step % log_every == 0 or step == args.steps - 1:
            rec = {"kind": "train", "step": step, "loss": _finite_float(loss),
                   "lr": _cur_lr(opt), "lm_ce": _finite_float(lm_ce),
                   "block": _finite_float(block), "smt_mem": _finite_float(smt["smt_memory"]),
                   "smt_update_pen": _finite_float(smt["smt_update_pen"]),
                   "dmt_mem": _finite_float(dmt["dmt_memory"]),
                   "dmt_state_rms": _finite_float(dmt.get("dmt_state_rms", 0.0)),
                   "gnorm": _finite_float(gnorm), "tok_per_sec": round((step + 1) * T / (time.time() - t0))}
            if nuc_w > 0.0:
                rec["nuc"] = _finite_float(nuc_val)
            if args.distill_fidelity_log and fid_val is not None:
                rec["carry_fidelity"] = _finite_float(fid_val)
            if args.log_grokking_metrics and args.grok_spec_every and (step % args.grok_spec_every == 0):
                rec["wnorm_rms"] = gm.weight_norm_rms(student.parameters())
                if grok_mat is not None:
                    rec["stable_rank"] = gm.stable_rank(grok_mat)
            if codec_rel is not None:
                rec["codec_rel"] = float(codec_rel)  # trainboard: constant after pre-fit (codec frozen)
            emit(rec)
            print(f"step {step}: loss={rec['loss']:.4f} lm_ce={rec['lm_ce']:.4f} "
                  f"block={rec['block']:.4f} smt={rec['smt_mem']:.4f} dmt={rec['dmt_mem']:.4f} "
                  f"state_rms={rec['dmt_state_rms']:.3f} gnorm={rec['gnorm']:.2f}", flush=True)

    if is_sf:
        opt.eval()  # averaged x for the final eval AND the saved checkpoint
    if args.eval_every > 0:
        student.eval()
        teacher_cap.want_states = False  # evaluate() only needs the block output, never states
        ev = evaluate(text_model, lm_head, toks, args.eval_windows, args.seq_len, dev,
                      teacher_gdn=teacher_gdn if args.log_grokking_metrics else None,
                      box=box if args.log_grokking_metrics else None, start_lo=eval_lo,
                      batch_size=args.eval_batch_size)
        if args.log_grokking_metrics and "block_val" in ev and block_ema is not None:
            ev["gen_gap"] = ev["block_val"] - block_ema
        emit({"kind": "eval", "step": args.steps - 1, **ev})
        print(f"  [eval-final] loss={ev['loss']:.4f} ppl={ev['ppl']:.3f} top1={ev['top1_acc']:.4f}", flush=True)
        if ev.get("ppl") is not None and ev["ppl"] < best_ppl:  # final eval may be the best
            best_ppl = ev["ppl"]
            _save_best(out, args.steps - 1, ev["ppl"], student, codec, args, opt)
    sd = _save_ckpt(out, args.steps - 1, student, codec, args, opt)  # x weights (schedulefree in eval mode)
    emit({"kind": "checkpoint", "step": args.steps - 1})
    print(f"saved -> {sd/'ckpt.pt'}", flush=True)


def chunked_ce(hidden, lm_head, labels, chunk=2048):
    """Inlined chunked lm_head+CE (matches train_mla.chunked_lmhead_ce) to avoid
    importing train_mla's heavy module chain."""
    B, T, H = hidden.shape
    flat_h = hidden.reshape(-1, H)
    flat_labels = labels.reshape(-1)
    Wt = lm_head.weight
    bias = getattr(lm_head, "bias", None)
    total = hidden.new_zeros((), dtype=torch.float32)
    Nn = flat_h.shape[0]
    for i in range(0, Nn, chunk):
        end = min(i + chunk, Nn)
        logits = F.linear(flat_h[i:end], Wt, bias).float()
        total = total + F.cross_entropy(logits, flat_labels[i:end], reduction="sum")
    return total / Nn


@torch.no_grad()
def evaluate(text_model, lm_head, toks, n_windows, T, device, seed=12345, chunk=2048,
             teacher_gdn=None, box=None, start_lo=0, batch_size=1):
    """Held-out LM eval (fixed windows) -> {loss, ppl, top1_acc}, in the dashboard's
    `kind:"eval"` schema. Uses the student model (text_model holds the RWKV layer).

    When `teacher_gdn` and the capture `box` are passed (grokking diagnostics on),
    also computes the HELD-OUT block-MSE (student vs teacher GDN block output) so the
    caller can form gen_gap = block_val - train_block_ema. The block fields ride in
    extra_json; the LM metrics are unchanged."""
    rng = np.random.default_rng(seed)
    N = len(toks); max_start = N - (T + 1)
    tot_ce = 0.0; tot_tok = 0; tot_correct = 0
    tot_block = 0.0; nb_block = 0
    starts = [int(rng.integers(min(start_lo, max_start), max_start + 1)) for _ in range(n_windows)]
    batch_size = max(1, int(batch_size))
    for off in range(0, n_windows, batch_size):
        ss = starts[off:off + batch_size]
        ids_np = np.stack([np.asarray(toks[s:s + T + 1], dtype=np.int64) for s in ss], axis=0)
        ids = torch.as_tensor(ids_np, device=device)
        x, y = ids[:, :-1], ids[:, 1:]
        out = text_model(input_ids=x, use_cache=False)
        hidden = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
        # held-out block-MSE: box["h"]/box["y"] were just refreshed by this forward
        if teacher_gdn is not None and box is not None:
            try:
                t_out = teacher_gdn(box["h"])
                tot_block += F.mse_loss(box["y"].float(), t_out.float(), reduction="sum").item()
                nb_block += box["y"].numel()
            except Exception:
                pass
        flat_h = hidden.reshape(-1, hidden.shape[-1]); flat_y = y.reshape(-1)
        Wt = lm_head.weight; bias = getattr(lm_head, "bias", None)
        for i in range(0, flat_h.shape[0], chunk):
            e = min(i + chunk, flat_h.shape[0])
            logits = F.linear(flat_h[i:e], Wt, bias).float()
            tot_ce += F.cross_entropy(logits, flat_y[i:e], reduction="sum").item()
            tot_correct += (logits.argmax(-1) == flat_y[i:e]).sum().item()
            tot_tok += e - i
    loss = tot_ce / max(tot_tok, 1)
    res = {"loss": loss, "ppl": math.exp(min(loss, 20.0)), "top1_acc": tot_correct / max(tot_tok, 1)}
    if nb_block > 0:
        res["block_val"] = tot_block / nb_block
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="/thearray/git/moe-mla/Qwen3.5-9B-Base")
    ap.add_argument("--patch-dir", default="", help="MLA patch dir (converted_9b_bkv_mtp); empty=base")
    ap.add_argument("--prior-rwkv-layers", default="", help="already-converted RWKV layers to load")
    ap.add_argument("--init-rwkv-ckpt", default="",
                    help="warm-start the bare RWKV core from a library layer file "
                         "(converted_layers_lib/L##.pt; uses its core.* subset). empty=GDN-init")
    ap.add_argument("--save-optimizer", action=argparse.BooleanOptionalAction, default=True,
                    help="persist optimizer state (Adam/Muon momentum, schedulefree averaging) in "
                         "best/ + step_ ckpts so --init-rwkv-ckpt resumes warm instead of cold-starting "
                         "momentum. Adds ~1-2x the trainable-param size per ckpt; --no-save-optimizer to skip.")
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=10000)
    ap.add_argument("--state-stride", type=int, default=64)
    ap.add_argument("--decay-cap-delta", type=float, default=0.005)
    ap.add_argument("--loop-count", type=int, default=4,
                    help="train the student AS a LoopedRWKV with this many weight-tied refinement "
                         "passes (residual_weight zero-init => loop==single-pass at start, so it can "
                         "only match the teacher at least as well as a bare core). 1 = bare core.")
    ap.add_argument("--loop-lr-mult", type=float, default=1.0,
                    help="LR multiplier for residual_weight (the loop gates). Default 1x: the refine/warm-start "
                         "case, where the loop is already trained and should move at the base rate. For a FRESH "
                         "conversion (residual_weight zero-init, tiny gradient) pass a higher mult, e.g. 30, so "
                         "the loop reaches useful magnitude.")
    ap.add_argument("--allow-neg-eigval", action=argparse.BooleanOptionalAction, default=True,
                    help="enable RWKV-7 negative eigenvalues (_a_scale 1->2, in-context gate a in (0,2)) to "
                         "match GDN's gated delta rule (beta in (0,2)). DEFAULT TRUE (the structural lever for "
                         "closing the base-ppl gap); use --no-allow-neg-eigval to disable. NOTE: changes a's "
                         "range, so warm-starting a False-trained core perturbs it (best tested fresh / re-adapted).")
    ap.add_argument("--w-lmce", type=float, default=0.1)
    ap.add_argument("--w-block", type=float, default=1.0)
    ap.add_argument("--w-smt", type=float, default=1.0)
    ap.add_argument("--w-dmt", type=float, default=1.0)
    ap.add_argument("--w-smt-block", type=float, default=0.0,
                    help="weight on the per-chunk block-MSE inside SMT (I/O fidelity during transitions)")
    ap.add_argument("--w-dmt-block", type=float, default=0.0,
                    help="weight on the per-chunk block-MSE inside the DMT rollout — keeps I/O fidelity "
                         "DURING closed-loop rollout (else DMT optimizes only latent state and ppl drifts up)")
    ap.add_argument("--dmt-discount", type=float, default=0.9)
    ap.add_argument("--dmt-curriculum", type=lambda s: [int(x) for x in s.split(",")],
                    default=[64, 128, 256, 512], help="DMT rollout-horizon schedule (tokens)")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=0.0,
                    help="decoupled weight decay. Regularizes the single-layer overfit that drives the "
                         "post-min eval-ppl creep (train block-MSE keeps dropping while held-out ppl rises). "
                         "Agent analysis flagged this as the missing anti-creep lever (was hard-coded 0).")
    ap.add_argument("--decay-lr-mult", type=float, default=0.1)
    ap.add_argument("--warmup-steps", type=int, default=200,
                    help="linear LR warmup ramp over the first N steps (0=off)")
    ap.add_argument("--lr-floor-frac", type=float, default=1.0,
                    help="after warmup, cosine-decay LR to this fraction of peak by the final "
                         "step (1.0 = hold at peak, no decay)")
    ap.add_argument("--codec-pretrain", type=int, default=0,
                    help="codec pre-fit steps (0 = random-init FROZEN map; pre-fit "
                         "strongly recommended so SMT/DMT targets are meaningful)")
    ap.add_argument("--codec-cache", default="", help="extraction cache for codec pre-fit (build_memory_targets.py)")
    ap.add_argument("--train-cache", default="",
                    help="optional build_memory_targets.py cache for local layer-target training. "
                         "Uses cached h/block_out/state instead of rerunning the frozen backbone and teacher "
                         "on train steps; LM CE is disabled for those train steps, while periodic eval still "
                         "uses --data and the full model.")
    ap.add_argument("--codec-rel-max", type=float, default=0.5,
                    help="abort if pre-fit codec rel_rmse exceeds this (catches the L0/L1 codec "
                         "collapse; healthy fits ~0.12-0.2). Only applies when --codec-pretrain>0.")
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--optimizer", default="adamw",
                    choices=["adamw", "muonclip", "schedulefree", "spectral_muon"])
    ap.add_argument("--sf-r", type=float, default=0.0,
                    help="Schedule-Free iterate-weighting power r (0=flat; ~1 better for runs >30B tokens)")
    ap.add_argument("--save-every", type=int, default=0,
                    help="also save a ckpt (averaged x, for schedulefree) every N steps (0=only at end); "
                         "should be a multiple of --eval-every")
    ap.add_argument("--muon-lr", type=float, default=1e-4,
                    help="Muon lr for 2D matrices (--optimizer muonclip). CALIBRATED: the baked-in "
                         "0.4*sqrt(max_dim) amplifier (~25.6x @4096) makes this ~vanilla-Muon-lr/25; "
                         "5e-4 blew up two train_mla runs.")
    ap.add_argument("--muon-adam-lr", type=float, default=3e-4,
                    help="AdamW lr for the 1D/scalar/norm params under --optimizer muonclip/spectral_muon")
    # --- spectral_muon levers (--optimizer spectral_muon; all default to vanilla Muon) ---
    ap.add_argument("--sm-ns-steps", type=int, default=5, help="Newton-Schulz iterations.")
    ap.add_argument("--sm-cheap-cubic", type=int, default=0,
                    help="Tier2: odd-cubic NS schedule, ~1/3 fewer matmuls (2606.00371); weaker orthogonalization.")
    ap.add_argument("--sm-plus-norm", default="none", choices=["none", "row", "col"],
                    help="Tier1 MUON+: row/col-normalize the orthogonalized update (2602.21545); up to 37%% faster.")
    ap.add_argument("--sm-equilibrate", default="none", choices=["none", "R", "C", "RC"],
                    help="Tier1 MuonEq: row/col equilibration BEFORE NS (2603.28254); R for hidden weights.")
    ap.add_argument("--sm-second-moment", type=int, default=0,
                    help="Tier2 Muon2: Adam-style 2nd-moment precondition before NS (2604.09967); ~40%% fewer NS iters.")
    ap.add_argument("--sm-row-uniform", type=int, default=0,
                    help="Tier2 Aurora: equal-row-norm for tall matrices; fixes dead neurons in wide MLPs (2606.27715).")
    ap.add_argument("--sm-spectral-power", type=float, default=0.0,
                    help="Tier3 Muon^p: U.Sigma^p.Vt (0=Muon UVt; ~1/3 good for FINETUNE, 2606.13867). p>0 uses SVD. Live.")
    ap.add_argument("--sm-power-method", default="svd", choices=["svd", "eigh"],
                    help="How Muon^p (p>0) computes U.Sigma^p.Vt: 'svd' (default, exact gesvd) or 'eigh' "
                         "(math-identical eigh-on-Gram, ~2-3x faster on the 4096^2 mix matrices, near-free on the "
                         "rank-8 factors). eigh squares the condition number (minor precision loss on the smallest "
                         "singular directions, clamped) — fine for p~1/3.")
    ap.add_argument("--sm-mona", type=int, default=0,
                    help="Tier3 MONA: Nesterov/curvature term in the gradient before NS (2605.26842).")
    ap.add_argument("--sm-mona-alpha", type=float, default=0.1, help="MONA curvature weight.")
    ap.add_argument("--sm-nesterov", type=int, default=0, help="Nesterov momentum on the Muon update.")
    ap.add_argument("--sm-scale", type=float, default=0.4,
                    help="Muon update amplifier scale*sqrt(max_dim) (0.4 matches this repo's MuonClip). Live.")
    # --- loop-level spectral levers ---
    ap.add_argument("--hyperball", type=int, default=0,
                    help="Tier2 Hyperball: project each 2D weight back to its initial Frobenius sphere each step "
                         "(2606.16899); removes WD tuning, ~20-30%% token speedup on normalized transformers.")
    ap.add_argument("--muon-to-adamw-frac", type=float, default=0.0,
                    help="Tier1 river-valley switch: at this fraction of training, switch a muon-family optimizer to "
                         "AdamW for the refinement tail (2606.21514) - fastest early AND lower final loss. 0=off.")
    # --- DDC (Dead-Direction Conditioner), PC-Layer, LLR ---
    ap.add_argument("--sm-ddc-strength", type=float, default=0.0,
                    help="DDC (2606.29176): fraction [0,1] of the per-channel rescale-gauge component removed from the "
                         "spectral_muon update; resists over-training collapse, gives cleaner minima. Live: ddc_strength.")
    ap.add_argument("--sm-ddc-mode", default="both", choices=["row", "col", "both"],
                    help="DDC gauge: row=output-channel scale, col=input-channel scale, both.")
    ap.add_argument("--pc-layer", type=int, default=0,
                    help="PC-Layer (2606.06470): polynomial spectral weight-preconditioning level (degree grows with it; "
                         "0=off, 2-4 typical). Reparam on student Linears, mergeable at inference. Costs extra forward VRAM.")
    ap.add_argument("--pc-strength", type=float, default=1.0,
                    help="PC-Layer blend s in [0,1]: W_eff=(1-s)W+s*g(W). Live: pc_strength.")
    ap.add_argument("--llr", type=int, default=0,
                    help="LLR (2605.22297): heavy-tail layerwise LR multiplier per param-group (Hill-alpha). Best with "
                         "many layer-groups (consolidation stage); within-layer here. 0=off.")
    ap.add_argument("--llr-smax", type=float, default=5.0, help="LLR max LR spread across groups. Live: llr_smax.")
    ap.add_argument("--llr-every", type=int, default=200, help="LLR: recompute per-group alpha every N steps.")
    ap.add_argument("--llr-active-frac", type=float, default=0.2,
                    help="LLR active only over the first this-fraction of training (then multipliers freeze).")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--compile", type=int, default=0,
                    help="EXPERIMENTAL: torch.compile the student backbone forward. fla Triton kernels "
                         "and the linear_attn activation-capture hooks (box['h']/box['y']) graph-break, so "
                         "the win is uncertain and must be validated (correct block loss + faster) before use.")
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--eval-every", type=int, default=100, help="held-out eval cadence (0=off)")
    ap.add_argument("--eval-windows", type=int, default=16,
                    help="held-out eval windows of --seq-len tokens each. Raise for final selection; "
                         "the default favors throughput during conversion.")
    ap.add_argument("--eval-batch-size", type=int, default=4,
                    help="number of eval windows per full-model forward. Increase until VRAM is full; "
                         "set 1 for old serial eval behavior.")
    ap.add_argument("--log-grokking-metrics", type=int, default=0,
                    help="emit memorization-vs-grokking diagnostics: held-out block_val + "
                         "gen_gap (block_val - train block EMA) at eval cadence. Pure additions "
                         "to train.jsonl; surfaces the post-min held-out creep the weight-decay "
                         "anti-creep lever targets. 0 to disable.")
    ap.add_argument("--grok-spec-every", type=int, default=0,
                    help="also emit weight-geometry diagnostics (wnorm_rms, stable_rank of the "
                         "largest student matrix) every N steps (0=off; stable_rank does an SVD, "
                         "so keep the cadence coarse, e.g. 200).")
    # --- grokking-encouraging levers (default-off; behavior unchanged at defaults) ---
    ap.add_argument("--fixed-trainset", type=int, default=0,
                    help="if >0, cache this many token windows once and CYCLE them every step "
                         "(reuse-data) instead of fresh windows — the small-fixed-set memorize→"
                         "generalize regime where grokking occurs. 0=fresh windows. Set at start; "
                         "NOT live-tunable (it defines the data regime).")
    ap.add_argument("--tail-weight-decay", type=float, default=0.0,
                    help="decoupled weight decay ramped in ONLY over the last --wd-tail-frac of "
                         "training (0 during the fit phase): the 'fit first, compress later' lever. "
                         "Applied manually so it works for adamw/muonclip (internal WD=0). Live-tunable.")
    ap.add_argument("--wd-tail-frac", type=float, default=0.3,
                    help="fraction of total steps over which --tail-weight-decay ramps 0->full. Live-tunable.")
    ap.add_argument("--nuc-weight", type=float, default=0.0,
                    help="spectral (nuclear-norm) penalty on the student's 2D mix matrices — the "
                         "tangential low-rank lever L2 can't provide under RMSNorm (2606.04405). SVD per "
                         "matrix; amortize with --nuc-every. 0=off (try ~1e-5). Live-tunable.")
    ap.add_argument("--nuc-every", type=int, default=1,
                    help="apply the --nuc-weight penalty every N steps (amortizes the SVD). Live-tunable.")
    # --- live tuning (trainboard v2 control table) ---
    ap.add_argument("--control-db",
                    default=str(Path(__file__).resolve().parent / "dashboard2" / "trainboard.db"),
                    help="trainboard SQLite DB polled for live overrides (run_controls table). Empty "
                         "string disables; a missing file is a silent no-op.")
    ap.add_argument("--control-poll-every", type=int, default=10,
                    help="poll the control table every N steps for live-tune overrides.")
    # --- grokking autopilot (anti-collapse recovery ladder; default off) ---
    ap.add_argument("--grok-autopilot", type=int, default=0,
                    help="enable anti-grokking-collapse recovery: weight-EMA best checkpoint + "
                         "on-collapse reg escalation and restore-best. 0=off. (The dashboard "
                         "detector separately writes an lr_scale cool on collapse.)")
    ap.add_argument("--ap-ema-decay", type=float, default=0.999,
                    help="autopilot weight-EMA decay; the averaged model resists post-grok wobble "
                         "(non-schedulefree only — schedulefree already averages).")
    ap.add_argument("--ap-collapse-thresh", type=float, default=0.02,
                    help="held-out ppl must exceed its own best by this fraction for --ap-patience "
                         "evals to count as a collapse (0.02 = 2%%).")
    ap.add_argument("--ap-patience", type=int, default=2,
                    help="consecutive regressing evals before the autopilot acts.")
    ap.add_argument("--ap-max-restarts", type=int, default=3,
                    help="cap on autopilot recovery escalations before it stops escalating.")
    ap.add_argument("--ap-reg-mult", type=float, default=2.0,
                    help="multiply nuc_weight/weight_decay by this on each recovery.")
    ap.add_argument("--ap-restore-best", type=int, default=1,
                    help="on collapse, roll the live model back to out/best/ckpt.pt (+clear "
                         "optimizer momentum) so it re-descends from the minimum. 1=on.")
    # --- best-model-fastest levers (reach generalization sooner; no memorization trap) ---
    ap.add_argument("--grokfast", type=int, default=0,
                    help="GrokFast: amplify the slow-varying gradient (EMA) so the model reaches "
                         "the generalizing solution sooner — collapses the grokking delay toward "
                         "immediate generalization. 0=off. Live: grokfast_lamb / grokfast_alpha.")
    ap.add_argument("--grokfast-alpha", type=float, default=0.98,
                    help="GrokFast gradient-EMA decay (slow-component memory). Live-tunable.")
    ap.add_argument("--grokfast-lamb", type=float, default=2.0,
                    help="GrokFast amplification of the slow gradient added back to the step. "
                         "Live-tunable; 0 disables even when --grokfast=1.")
    ap.add_argument("--readout-lr-mult", type=float, default=1.0,
                    help="LR multiplier on the readout group (student out_proj now; the codec too "
                         "once it unfreezes at consolidation) — a faster readout closes the "
                         "representation->behavior lag (2604.13082). Scheduled opts (adamw). Live-tunable.")
    ap.add_argument("--disjoint-eval", type=int, default=1,
                    help="when --fixed-trainset>0, draw eval windows from a region disjoint from the "
                         "cached train pool so the generalization gap is clean. 1=on.")
    ap.add_argument("--ap-stall-patience", type=int, default=3,
                    help="autopilot: held-out ppl not improving for this many evals = stuck in "
                         "memorization -> escalate reg + kick GrokFast to escape (distinct from the "
                         "post-grok collapse recovery).")
    # --- relational / cross-arch distillation objectives (distill_objectives.py) ---
    ap.add_argument("--w-cos", type=float, default=0.0,
                    help="cosine block/state match (2602.05262/2606.26488): direction-invariant, scale-robust. Live.")
    ap.add_argument("--w-cka", type=float, default=0.0,
                    help="CKA/Gram block match (2606.05682): orthogonal+scale invariant, dim-agnostic cross-arch. Live.")
    ap.add_argument("--w-flow", type=float, default=0.0,
                    help="PHF transition-flow match (2606.29340): match how features MOVE (direction+Gram), not where. Live.")
    ap.add_argument("--w-bridge", type=float, default=0.0,
                    help="OPRD-Bridge (2606.06021): match in a frozen low-rank PCA subspace (cross-arch). Live.")
    ap.add_argument("--bridge-rank", type=int, default=8, help="OPRD-Bridge subspace rank (small ~8 best).")
    ap.add_argument("--agreement-gate", type=float, default=0.0,
                    help="trust-region gating (2606.01249): >0 reweights block MSE by teacher-student cosine agreement, "
                         "down-weighting diverged tokens (stabilizes compounding cross-arch mismatch). Live.")
    ap.add_argument("--distill-fidelity-log", type=int, default=0,
                    help="emit carry_fidelity (2606.26488): label-free cosine drift monitor for the dashboard.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.save_every > 0 and args.eval_every > 0 and args.save_every % args.eval_every != 0:
        raise SystemExit(f"--save-every ({args.save_every}) must be a multiple of --eval-every "
                         f"({args.eval_every}): periodic saves only fire on eval steps.")
    if args.nuc_weight > 0.0 and args.nuc_every <= 1:
        print(f"  [warn] --nuc-weight={args.nuc_weight} with --nuc-every=1: a full SVD over every "
              f"2D student matrix runs EVERY step. Raise --nuc-every to amortize it unless you "
              f"specifically need per-step accuracy.", flush=True)
    train(args)


if __name__ == "__main__":
    main()
