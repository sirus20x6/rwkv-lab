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
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
# perf #1: expandable segments cuts the per-step CE-logit alloc/free fragmentation.
# Must be set before torch initializes the CUDA caching allocator.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
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


class _StopForward(Exception):
    """perf #3: raised in the layer-L forward hook to abort the backbone forward early
    when the upper stack + lm_head are unused (w_lmce==0). Gated by --truncate-forward."""

from build_memory_targets import _GDNStateCapture, load_token_stream, find_gdn_layer
from smt_dmt import BilinearStateCodec, MemoryTargetCache, smt_transition_loss, dmt_rollout_loss, fit_codec

# perf #2b: flash_attn's Triton CE — fused logsumexp+gather with in-place backward into the
# bf16 logit buffer (no fp32 logit copy, no separate grad alloc). Optional (--fused-ce).
try:
    from flash_attn.ops.triton.cross_entropy import cross_entropy_loss as _flash_ce
    _HAS_FLASH_CE = True
except Exception:
    _flash_ce, _HAS_FLASH_CE = None, False

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


def _optimizer_params(opt):
    return [p for g in opt.param_groups for p in g["params"]]


def _token_windows_tensor(toks, starts, length, device, out=None):
    starts = np.asarray(starts, dtype=np.int64)
    shape = (starts.shape[0], length)
    ids_np = out if out is not None and out.shape == shape else np.empty(shape, dtype=np.int64)
    for i, s in enumerate(starts):
        ids_np[i] = toks[int(s):int(s) + length]
    return torch.as_tensor(ids_np, device=device)


def _cache_batch_tensor(mm, indices, device, dtype):
    return torch.as_tensor(np.asarray(mm[np.asarray(indices, dtype=np.int64)]), device=device, dtype=dtype)


def _save_ckpt(out, step, student, codec, args, opt=None, rosa_soft=None, rosa_ctl=None,
               lookahead=None):
    """Write step_<step>/{config.json, ckpt.pt}. For --optimizer schedulefree the
    caller must opt.eval() first so `student` holds the averaged x weights. With
    --save-optimizer, also persist optimizer state so --init-rwkv-ckpt resumes warm."""
    sd = write_sidecar_config(out, step, args)
    blob = {"student": student.state_dict(), "codec": codec.state_dict(), "args": vars(args)}
    if opt is not None and getattr(args, "save_optimizer", True):
        blob["opt"], blob["opt_type"] = opt.state_dict(), args.optimizer
    if lookahead is not None:   # training-only aux heads (config rides in args)
        blob["lookahead"] = lookahead.state_dict()
    if rosa_soft is not None:
        blob["rosa_soft"] = rosa_soft.state_dict()
        # calibrated retrieval scale + controller EMA are NOT state_dict content
        # (plain attrs, kept out to preserve sd shape); persist them beside it so
        # a warm-restart resumes the calibrated temperature instead of re-estimating.
        blob["rosa_soft_scale"] = None if rosa_soft.scale is None else float(rosa_soft.scale)
        if rosa_ctl is not None:
            blob["rosa_ctl_ema"] = rosa_ctl.top_prob_ema
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


_SAVE_POOL = ThreadPoolExecutor(max_workers=1)  # best-ckpt writer: single worker => writes land in submit order
_LAST_SAVE = [None]  # most recent queued best-write future


def _flush_best_saves():
    """Block until any queued best-ckpt write has hit disk. Anything that torch.loads
    out/best/ckpt.pt MID-RUN (autopilot restore-best) must flush first, or it can read
    the previous minimum while the newest one is still in the write queue."""
    f = _LAST_SAVE[0]
    if f is not None:
        f.result()


def _snap_cpu(obj):
    """Deep-copy a (nested) state blob to CPU. A background torch.save must see an
    immutable snapshot: the live GPU tensors keep training while the write runs."""
    if torch.is_tensor(obj):
        return obj.detach().to("cpu", copy=True)
    if isinstance(obj, dict):
        return {k: _snap_cpu(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        vals = [_snap_cpu(v) for v in obj]
        return vals if isinstance(obj, list) else tuple(vals)
    return obj


def _save_best(out, step, ppl, student, codec, args, opt=None, rosa_soft=None, rosa_ctl=None,
               lookahead=None):
    """ATOMICALLY save the best-eval checkpoint to out/best/. Called on EVERY eval
    improvement, so the minimum can never slip between periodic (--save-every) saves.
    The CPU snapshot happens synchronously HERE — schedulefree's eval-mode x weights and
    the autopilot's EMA swap must be captured before the caller swaps them back — then
    the pickle+disk write runs on _SAVE_POOL so the improvement streaks early in training
    don't stall train steps. Writes to a temp file then os.replace -> a crash mid-save
    can't corrupt best/ (an in-flight write is lost, best/ keeps the previous minimum).
    With --save-optimizer, persists optimizer state so --init-rwkv-ckpt resumes warm."""
    bd = Path(out) / "best"
    bd.mkdir(parents=True, exist_ok=True)
    blob = {"student": student.state_dict(), "codec": codec.state_dict(),
            "args": vars(args), "step": int(step), "ppl": float(ppl)}
    if opt is not None and getattr(args, "save_optimizer", True):
        blob["opt"], blob["opt_type"] = opt.state_dict(), args.optimizer
    if lookahead is not None:   # training-only aux heads (config rides in args)
        blob["lookahead"] = lookahead.state_dict()
    if rosa_soft is not None:
        blob["rosa_soft"] = rosa_soft.state_dict()
        blob["rosa_soft_scale"] = None if rosa_soft.scale is None else float(rosa_soft.scale)
        if rosa_ctl is not None:  # controller EMA rides too (see _save_ckpt)
            blob["rosa_ctl_ema"] = rosa_ctl.top_prob_ema
    blob = _snap_cpu(blob)

    def _write():
        try:
            tmp = bd / "ckpt.pt.tmp"
            torch.save(blob, tmp)
            os.replace(tmp, bd / "ckpt.pt")
            (bd / "best.json").write_text(json.dumps({"step": int(step), "ppl": float(ppl)}))
        except Exception as e:
            print(f"  [warn] background best-ckpt write failed: {e}", flush=True)

    _LAST_SAVE[0] = _SAVE_POOL.submit(_write)


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


def loop_anneal_factor(step, trigger_step, anneal_steps):
    """Boost-decay factor in [0,1] for the loop-LR anneal: 1.0 until the gates escape
    (trigger_step is None before that), then cosine 1->0 over anneal_steps, then 0.
    effective mult = 1 + (boost-1)*factor, so factor 0 = boost fully released (1x).
    Deterministic and lag-free — round-1 gate A/B showed the DB-side reactive cooling
    fired at max|rw| 0.303 for one arm and 2.1 for the other (ingest/sampler lag),
    giving the arms materially different LR schedules. Gates only need the boost to
    ESCAPE zero (0 -> ~0.1); they grew 0.3 -> 3.3 under mult <= 15 regardless."""
    if trigger_step is None or step <= trigger_step:
        return 1.0
    t = step - trigger_step
    if t >= anneal_steps:
        return 0.0
    return 0.5 * (1.0 + math.cos(math.pi * t / anneal_steps))


def _loop_param_names(student):
    """The student's loop-gate param names (rwkv_loop group + loop_lr_mult), or an
    empty set for a bare core (loop_count==1). Centralizes what the three optimizer
    builders used to hardcode as `n in ("residual_weight","gate_chan")`, so adding a
    gate tensor (e.g. loop_index_embed) updates every path at once."""
    return student.loop_param_names() if hasattr(student, "loop_param_names") else set()


def _expand_loop_gates(load_sd, student):
    """Broadcast a coarser ckpt residual_weight into the student's gate shape:
    scalar [N] -> [N,G]/[N,C] (same value per group/channel) and head [N,G] -> [N,C]
    (repeat within group). The expanded gates reproduce the ckpt's loop output
    EXACTLY, so a granularity upgrade is loss-free. Finer -> coarser is refused
    (lossy) — relaunch with the ckpt's own --loop-gate instead."""
    rw_old = load_sd.get("residual_weight")
    rw_new = getattr(student, "residual_weight", None)
    if rw_old is None or rw_new is None or tuple(rw_old.shape) == tuple(rw_new.shape):
        return
    N = rw_new.shape[0]
    if rw_old.ndim == 1 and rw_new.ndim == 2 and rw_old.shape[0] == N:
        load_sd["residual_weight"] = rw_old.view(N, 1).expand(N, rw_new.shape[1]).clone()
    elif (rw_old.ndim == 2 and rw_new.ndim == 2 and rw_old.shape[0] == N
          and rw_new.shape[1] % rw_old.shape[1] == 0):
        load_sd["residual_weight"] = rw_old.repeat_interleave(
            rw_new.shape[1] // rw_old.shape[1], dim=1).clone()
    elif rw_old.shape[0] != N:
        raise SystemExit(f"ckpt residual_weight has n_loops={rw_old.shape[0]} but this run has "
                         f"--loop-count {N}: pass counts cannot be adapted; relaunch with "
                         f"--loop-count {rw_old.shape[0]}.")
    else:
        raise SystemExit(f"cannot adapt ckpt residual_weight {tuple(rw_old.shape)} to the "
                         f"--loop-gate shape {tuple(rw_new.shape)}: finer->coarser is lossy; "
                         f"relaunch with the ckpt's own --loop-gate.")
    print(f"  warm-start: residual_weight {tuple(rw_old.shape)} -> {tuple(rw_new.shape)} "
          f"(broadcast, loss-free)", flush=True)


def build(args):
    """Load the model, hold the teacher GDN layer L, swap in the RWKV student,
    freeze everything else. Returns a dict of handles."""
    import layer_swap
    import rwkv8_deltanet as _r8
    from rwkv8_deltanet import RWKV8TimeMixDeltaNet  # noqa
    if not _r8._HAS_FLA and os.environ.get("RWKV8_FORCE_PYREF") != "1":
        # A broken fla install silently routes every forward through the T-step
        # Python wkv7 reference (~100x slower); the only symptom would be awful
        # tok/s. Refuse to train in that state instead.
        raise SystemExit(f"fla (flash-linear-attention) failed to import: {_r8._FLA_IMPORT_ERROR!r}. "
                         f"Training would silently run the ~100x slower Python wkv7 reference. "
                         f"Fix the fla install, or set RWKV8_FORCE_PYREF=1 to run the reference "
                         f"intentionally.")

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
        student = LoopedRWKV(core, n_loops=args.loop_count, gate_mode=args.loop_gate,
                             gate_cap=args.loop_gate_cap, loop_index=bool(args.loop_index),
                             hyper_lanes=int(getattr(args, "loop_hyper", 0)),
                             lora_rank=int(getattr(args, "loop_lora_rank", 0)),
                             lora_targets=tuple(t for t in str(getattr(
                                 args, "loop_lora_targets", "")).split(",") if t.strip()),
                             ).to(device=_p0.device, dtype=_p0.dtype)
        student.float_gates()  # gates stay fp32: bf16 ulp swallows their tiny growth steps
        student.iter_consist = float(getattr(args, "loop_iter_consist", 0.0)) > 0
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
                # KEEPING the loop is only loss-free if the gate soft-cap matches: the
                # effective gate is cap*tanh(rw/cap) (or raw rw uncapped), so a cap change
                # silently reshapes every trained gate. Refuse, like finer->coarser gates.
                old_cap = float((blob.get("args") or {}).get("loop_gate_cap") or 0.0)
                if old_cap != float(args.loop_gate_cap):
                    raise SystemExit(
                        f"warm-start gate_cap mismatch: ckpt was trained with loop_gate_cap="
                        f"{old_cap:g}, this run has --loop-gate-cap {args.loop_gate_cap:g}. "
                        f"The kept loop would silently change function; relaunch with "
                        f"--loop-gate-cap {old_cap:g}.")
                load_sd = dict(sd_src)                        # full: core.* + residual_weight + iter_norm.*
                strip_loop = False
            elif src_looped:
                load_sd = {k: v for k, v in sd_src.items() if k.startswith("core.")}
                strip_loop = True
            else:
                load_sd = {f"core.{k}": v for k, v in sd_src.items()}
                strip_loop = True
            if not strip_loop:
                _expand_loop_gates(load_sd, student)          # coarser-gate ckpt -> this --loop-gate
            miss, unexp = student.load_state_dict(load_sd, strict=False)
            if strip_loop:                                    # loop params legitimately absent
                miss = [m for m in miss if not (m in _loop_param_names(student)
                                                or m.startswith("iter_norm."))]
            else:
                # gate tensors this student has but the ckpt lacks (e.g. a factored/loop-index/
                # hyper student warm-started from a plainer ckpt) init to their exact no-op:
                # gate_chan delta=0 -> channel factor 1; loop_index_embed=0 -> no offset;
                # hyper_alpha/mix/write/read at one-hot/identity/ones/e0 -> plain-loop function.
                new_gates = (_loop_param_names(student) - {"residual_weight"}) & set(miss)
                if new_gates:
                    miss = [m for m in miss if m not in new_gates]
                    print(f"  warm-start: ckpt lacks {sorted(new_gates)}; fresh no-op init", flush=True)
            n_loaded = len(load_sd)
        else:
            load_sd = {k[len("core."):]: v for k, v in sd_src.items() if k.startswith("core.")}
            if not load_sd:  # already a bare core state_dict (no core. prefix)
                load_sd = {k: v for k, v in sd_src.items()
                           if k != "residual_weight" and not k.startswith("iter_norm.")}
            miss, unexp = student.load_state_dict(load_sd, strict=False)
            n_loaded = len(load_sd)
        if miss or unexp:  # same classes on both sides -> should be exact; fail loud, don't half-init
            hint = ""
            if any(str(u).startswith("loop_lora_") for u in unexp):
                from looped_rwkv import lora_config_from_sd
                _r, _t = lora_config_from_sd(sd_src)
                hint += (f"\n  ckpt carries TRAINED per-pass LoRA adapters; relaunch with "
                         f"--loop-lora-rank {_r} --loop-lora-targets {','.join(_t)}")
            if any(str(u).startswith("hyper_") for u in unexp):
                _k = int(sd_src["hyper_read"].shape[0])
                hint += f"\n  ckpt carries TRAINED hyper-connection lanes; relaunch with --loop-hyper {_k}"
            raise SystemExit(f"warm-start key mismatch from {src}: "
                             f"missing={list(miss)[:6]} unexpected={list(unexp)[:6]}{hint}")
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
    def _capture_y(m, a, o):
        box["y"] = o
        if box.get("_truncate"):           # perf #3: stop the forward after layer L when w_lmce==0
            raise _StopForward
    decoder_layer.linear_attn.register_forward_hook(_capture_y)

    # ROSA-soft: additive retrieval injection on the layer's output. Registered AFTER
    # _capture_y so box["y"] (read by SMT/DMT) stays pre-injection -- the primary
    # conversion loss always measures pure RWKV-vs-GDN fidelity, unaffected by
    # --rosa-soft; only downstream layers + the LM-CE path see the augmented output.
    rosa_soft = None
    rosa_restore = None  # {"scale":..., "ema":...} from a warm-start ckpt (train() seeds the controller)
    if getattr(args, "rosa_soft", 0):
        from rosa_soft_layer import RosaAnchorLayer
        # dtype from args, NOT from next(student.parameters()): a LoopedRWKV student's
        # first param is the fp32-kept loop gate (float_gates), which would cast the
        # whole rosa layer to fp32 and dtype-crash against the bf16 stream at first eval.
        _p0 = next(student.parameters())
        rosa_soft = RosaAnchorLayer(
            hidden_size=text_model.config.hidden_size, M=args.rosa_soft_m,
            window_size=args.rosa_soft_window, scale=args.rosa_soft_scale,
            value_heads=args.rosa_soft_value_heads,
            logit_epsilon=args.rosa_soft_logit_epsilon,
            qk_damper_strength=args.rosa_soft_qk_damper,
            route_dim=(args.rosa_soft_dim or None),
        ).to(device=_p0.device, dtype=getattr(torch, args.dtype))
        rosa_soft.float_growth_params()  # e0/e1 fp32: zero-init growth params, same as loop gates
        for p in rosa_soft.parameters():
            p.requires_grad_(True)
        def _inject_rosa_soft(m, a, o):
            if not torch.is_tensor(o):
                # SMT/DMT call the same module directly with return_state=True to
                # probe pure RWKV-vs-GDN state/block fidelity -> (y, s1, shift_state)
                # tuple. Leave that untouched, same reasoning as box["y"] above.
                return None
            return o + rosa_soft.injection(box["h"])
        decoder_layer.linear_attn.register_forward_hook(_inject_rosa_soft)
        if getattr(args, "init_rwkv_ckpt", "") and isinstance(blob, dict) and "rosa_soft" in blob:
            rosa_soft.load_state_dict(blob["rosa_soft"])
            # the CALIBRATED retrieval scale is constructor/controller state, not
            # state_dict content: without restoring it, e0/e1/Wq/Wk trained against
            # one softmax temperature get re-run under the closed-form estimate and
            # the injection silently changes function at step 0.
            rosa_restore = {"scale": blob.get("rosa_soft_scale"), "ema": blob.get("rosa_ctl_ema")}
            if rosa_restore["scale"] is not None and args.rosa_soft_scale is None:
                rosa_soft.scale = float(rosa_restore["scale"])
            _sc = "none saved" if rosa_restore["scale"] is None else f"{float(rosa_restore['scale']):.4g}"
            print(f"  warm-start: loaded rosa_soft state from {src} (calibrated scale: {_sc})", flush=True)
    elif (getattr(args, "init_rwkv_ckpt", "") and isinstance(blob, dict) and "rosa_soft" in blob):
        print("  [warn] ckpt carries TRAINED rosa_soft weights (the injection was part of its "
              "LM function) but this run has --rosa-soft OFF — DISCARDING them changes step-0 "
              "behavior. Pass --rosa-soft 1 (with the ckpt's rosa dims) to keep them.", flush=True)

    # Lookahead (NextLat + TOP, lookahead_module.py): training-only future-prediction
    # heads fed by the FULL-forward post-norm hidden (same gradient path as LM-CE).
    # Both are dropped at inference — unlike rosa_soft they never alter the LM function,
    # so discarding them on a later run is always safe; ckpts carry them under
    # "lookahead" for warm restarts.
    lookahead = None
    if (getattr(args, "nextlat_weight", 0.0) > 0 or getattr(args, "top_weight", 0.0) > 0
            or getattr(args, "nextlat_jump_weight", 0.0) > 0
            or getattr(args, "concept_weight", 0.0) > 0):
        from lookahead_module import lookahead_from_args
        _p0 = next(student.parameters())
        _V, _D = model.lm_head.weight.shape
        lookahead = lookahead_from_args(args, _D, _V, model.lm_head).to(
            device=_p0.device, dtype=_p0.dtype)
        for p in lookahead.parameters():
            p.requires_grad_(True)
        print(f"  lookahead: nextlat={'on' if lookahead.nextlat else 'off'} "
              f"top={'on' if lookahead.top else 'off'} "
              f"({sum(p.numel() for p in lookahead.parameters())/1e6:.1f}M aux params -> AdamW group)",
              flush=True)
        if getattr(args, "init_rwkv_ckpt", "") and isinstance(blob, dict) and "lookahead" in blob:
            lookahead.load_state_dict(blob["lookahead"])
            print(f"  warm-start: loaded lookahead state from {src}", flush=True)

    # codec is a FIXED GDN->RWKV state-space map (pre-fit-then-frozen via
    # --codec-pretrain, else random-init frozen). SMT/DMT detach its output, so
    # it never receives gradient; freeze it explicitly to make that unambiguous.
    codec = BilinearStateCodec(
        gdn_heads=teacher_gdn.num_v_heads, gdn_dk=teacher_gdn.head_k_dim, gdn_dv=teacher_gdn.head_v_dim,
        rwkv_heads=64, rwkv_dk=64, rwkv_dv=64,
    ).to(args.device)
    for p in codec.parameters():
        p.requires_grad_(False)
    if getattr(args, "compile_core", 0):
        # IN-PLACE compile of the RWKV core only (nn.Module.compile): param identity,
        # state_dict keys, and the ckpt format are all unchanged, and the LoopedRWKV
        # wrapper / box+rosa hooks / losses stay eager. Fuses the eager elementwise
        # soup around chunk_rwkv7 (measured ~3x the kernel time uncompiled); dynamo
        # graph-breaks cleanly at the fla op. Expect a few recompiles: train/eval T,
        # SMT/DMT chunk shapes, state kwargs.
        _mode = getattr(args, "compile_core_mode", "default")
        core.compile(mode=(None if _mode == "default" else _mode))
        print(f"  [compile-core] RWKV core compiled in-place, mode={_mode} "
              f"(wrapper/hooks/losses stay eager)", flush=True)
    if getattr(args, "compile", 0):
        text_model = torch.compile(text_model)  # graph-breaks expected at fla/hook boundaries
        print("  torch.compile: text_model wrapped (experimental)", flush=True)
    return dict(model=model, text_model=text_model, student=student,
                teacher_gdn=teacher_gdn, teacher_cap=teacher_cap, box=box, codec=codec,
                lm_head=model.lm_head, init_opt_state=init_opt_state, init_opt_type=init_opt_type,
                rosa_soft=rosa_soft, rosa_restore=rosa_restore, lookahead=lookahead)


def make_optimizer(student, codec, args, text_cfg=None, rosa_soft=None, lookahead=None):
    """AdamW (default) or DeepSeek-V4-style MuonClip (--optimizer muonclip).
    AdamW path: decay/time params low LR, projections normal. MuonClip path routes
    2D matrices -> Muon and 1D/scalar/norm -> AdamW (see _make_muonclip). The codec
    is a frozen target map (SMT/DMT detach it) so it is never in the optimizer.
    rosa_soft and lookahead (AdamW/spectral_muon paths only; enforced at argparse
    time) each get their own group — lookahead MUST stay off Muon (the NextLat
    authors report Muon instability with the latent-prediction objective)."""
    if args.optimizer == "muonclip":
        return _make_muonclip(student, text_cfg, args)
    if args.optimizer == "schedulefree":
        return _make_schedulefree(student, codec, args)
    if args.optimizer == "spectral_muon":
        return _make_spectral_muon(student, codec, args, rosa_soft=rosa_soft, lookahead=lookahead)
    decay_names = ("w0", "w1", "w2", "a0", "a1", "a2", "k_k", "k_a")
    loop_names = _loop_param_names(student)
    decay_p, proj_p, readout_p, loop_p = [], [], [], []
    for n, p in student.named_parameters():
        if not p.requires_grad:
            continue
        parts = n.split(".")
        tail = parts[-1]  # match decay params even under LoopedRWKV's "core." prefix
        if n in loop_names:                        # loop gates (residual_weight/gate_chan/
            loop_p.append(p)                       # loop_index_embed): own group -> loop_lr_mult
        elif "out_proj" in n or "output" in parts[:-1]:  # readout: RWKV8 core names it `output`
            readout_p.append(p)                   # (converted FROM GDN's out_proj); a faster
        elif tail in decay_names:                 # readout closes the repr->behavior lag (2604.13082)
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
    if loop_p:  # appended after the core groups: leading-group momentum restore from
        # pre-loop-group ckpts stays possible. Base lr here; loop_lr_mult applies LIVE
        # per step (post-sched multiplier block), so it is steerable mid-run.
        groups.append({"params": loop_p, "lr": args.lr, "name": "rwkv_loop"})
    if rosa_soft is not None:
        groups.append({"params": [p for p in rosa_soft.parameters() if p.requires_grad],
                       "lr": args.rosa_soft_lr or args.lr, "name": "rosa_soft"})
    if lookahead is not None:  # appended LAST (after rosa_soft): leading-group momentum
        # restore from a no-lookahead ckpt stays possible, same reasoning as rosa_soft
        groups.append({"params": [p for p in lookahead.parameters() if p.requires_grad],
                       "lr": args.lookahead_lr or args.lr, "name": "lookahead"})
    return torch.optim.AdamW(groups, betas=(0.9, 0.95), weight_decay=0.0)


def _make_spectral_muon(student, codec, args, rosa_soft=None, lookahead=None):
    """SpectralMuon (spectral_muon.py): 2D matrices -> configurable Muon update,
    everything else -> built-in AdamW. The --sm-* argparse DEFAULTS are the validated
    sweep recipe (plus-norm row + eq R + DDC 0.5 both @ muon-lr 4e-6 — banked wins on
    all 24 GDN layers), so `--optimizer spectral_muon` with no flags == that recipe.
    Vanilla Muon: --sm-plus-norm none --sm-equilibrate none --sm-ddc-strength 0 (the
    SpectralMuon CLASS defaults stay vanilla; only the CLI defaults carry the recipe).
    Each lever maps to a 2026 paper; stack them freely (e.g. --sm-second-moment 1).
    rosa_soft (if attached) rides SpectralMuon's built-in AdamW fallback as its own
    group (use_muon=False, ddc off) so the core keeps Muon+ while rosa-soft is plain AdamW."""
    from spectral_muon import SpectralMuon
    loop_names = _loop_param_names(student)
    muon_p, adam_p, loop_p = [], [], []
    for n, p in student.named_parameters():
        if not p.requires_grad:
            continue
        if n in loop_names:                            # loop gates: own AdamW-fallback group
            loop_p.append(p)                           # so loop_lr_mult can steer them live
        elif p.ndim == 2 and min(p.shape) > 1:
            muon_p.append(p)
        else:
            adam_p.append(p)
    if codec is not None:                              # codec joins the Adam side until it
        adam_p += [p for _, p in codec.named_parameters() if p.requires_grad]  # unfreezes
    groups = [
        {"params": muon_p, "lr": args.muon_lr, "use_muon": True, "name": "muon"},
        {"params": adam_p, "lr": args.muon_adam_lr, "use_muon": False, "name": "adam"},
    ]
    if loop_p:  # after muon/adam (leading-group momentum restore), before rosa_soft.
        groups.append({"params": loop_p, "lr": args.muon_adam_lr, "use_muon": False,
                       "ddc_strength": 0.0, "name": "rwkv_loop"})
    if rosa_soft is not None:                          # ROSA-soft -> plain AdamW group, appended
        rosa_p = [p for p in rosa_soft.parameters() if p.requires_grad]  # LAST so a warm-resume from
        groups.append({"params": rosa_p, "lr": args.rosa_soft_lr or args.muon_adam_lr,  # a no-rosa ckpt
                       "use_muon": False, "ddc_strength": 0.0, "name": "rosa_soft"})     # still matches core groups
    if lookahead is not None:                          # lookahead heads -> AdamW fallback, NEVER
        la_p = [p for p in lookahead.parameters() if p.requires_grad]   # Muon (NextLat authors report
        groups.append({"params": la_p, "lr": args.lookahead_lr or args.muon_adam_lr,    # Muon instability); appended
                       "use_muon": False, "ddc_strength": 0.0, "name": "lookahead"})     # last, after rosa_soft
    opt = SpectralMuon(groups, momentum=0.95, nesterov=bool(args.sm_nesterov),
                       ns_steps=args.sm_ns_steps, cubic=bool(args.sm_cheap_cubic),
                       spectral_power=args.sm_spectral_power, power_method=args.sm_power_method,
                       second_moment=bool(args.sm_second_moment),
                       equilibrate=args.sm_equilibrate, plus_norm=args.sm_plus_norm,
                       row_uniform=bool(args.sm_row_uniform), mona=bool(args.sm_mona),
                       mona_alpha=args.sm_mona_alpha, scale=args.sm_scale,
                       ddc_strength=args.sm_ddc_strength, ddc_mode=args.sm_ddc_mode,
                       rsav=bool(args.sm_rsav), rsav_c=args.sm_rsav_c,
                       rsav_cap=args.sm_rsav_cap, rsav_relax=args.sm_rsav_relax)
    if loop_p:
        print(f"  + rwkv_loop: {sum(p.numel() for p in loop_p)} gate(s) on AdamW fallback "
              f"@ base lr={args.muon_adam_lr:.1e} x loop_lr_mult={args.loop_lr_mult:g} (live)", flush=True)
    if rosa_soft is not None:
        print(f"  + rosa_soft: {sum(p.numel() for p in rosa_p)/1e6:.3f}M params on AdamW fallback "
              f"@ lr={args.rosa_soft_lr or args.muon_adam_lr:.1e} (ddc off)", flush=True)
    if lookahead is not None:
        print(f"  + lookahead: {sum(p.numel() for p in la_p)/1e6:.1f}M params on AdamW fallback "
              f"@ lr={args.lookahead_lr or args.muon_adam_lr:.1e} (ddc off; Muon unstable for NextLat)",
              flush=True)
    print(f"  SpectralMuon: {len(muon_p)} 2D mats @ muon_lr={args.muon_lr:.1e}, "
          f"{len(adam_p)} other @ adam_lr={args.muon_adam_lr:.1e} | levers: plus={args.sm_plus_norm} "
          f"eq={args.sm_equilibrate} 2nd={bool(args.sm_second_moment)} aurora={bool(args.sm_row_uniform)} "
          f"power={args.sm_spectral_power}({args.sm_power_method}) mona={bool(args.sm_mona)} cubic={bool(args.sm_cheap_cubic)} "
          f"ns={args.sm_ns_steps} rsav={bool(args.sm_rsav)}"
          f"{f'(c={args.sm_rsav_c:g},cap={args.sm_rsav_cap:g})' if args.sm_rsav else ''}", flush=True)
    return opt


def _make_muonclip(student, text_cfg, args):
    """DeepSeek-V4-style split via train_mla's GuardedMuonClip: 2D weight matrices ->
    Muon (Newton-Schulz orthogonalized), all 1D/scalar/norm params -> AdamW. Routing
    is automatic by p.ndim==2 (so the RWKV proj/decay-LoRA/gate matrices AND the 2D
    r_k [H,N] go to Muon; the non-2D x_*, w0, a0, k_k, k_a, ln_x go to Adam). QK-clip is OFF (no-op for RWKV's
    receptance/key, which are not GQA-shaped). lr_muon is CALIBRATED: the baked-in
    0.4*sqrt(max_dim) amplifier (~25.6x @4096) makes it ~vanilla-Muon-lr/25 —
    train_mla's validated 1e-4 (5e-4 blew up two runs)."""
    if _loop_param_names(student):
        raise SystemExit(
            "--optimizer muonclip cannot train a LoopedRWKV student: GuardedMuonClip caps the "
            "step at ratio*RMS(p), so the zero-init loop gates are an absorbing state (RMS(p)=0 "
            "=> max step ~0 forever), and it has no rwkv_loop group, so --loop-lr-mult and the "
            "detector's live steering silently never apply. Use --optimizer spectral_muon or "
            "adamw, or --loop-count 1.")
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
    loop_names = _loop_param_names(student)
    decay_p, proj_p, loop_p, readout_p = [], [], [], []
    for n, p in student.named_parameters():
        if not p.requires_grad:
            continue
        parts = n.split(".")
        tail = parts[-1]  # match decay params even under LoopedRWKV's "core." prefix
        if n in loop_names:
            loop_p.append(p)                       # loop gates: tiny gradient -> own high LR
        elif "out_proj" in n or "output" in parts[:-1]:  # RWKV8 core names its readout `output`
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
    if loop_p:  # schedulefree has no live multipliers (no sched), so the mult is baked in
        groups.append({"params": loop_p, "lr": args.lr * args.loop_lr_mult, "name": "rwkv_loop"})
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


def _restore_matching_opt_state(opt, saved):
    """Partial optimizer-state restore for when the group COUNT changed since the ckpt
    (torch's load_state_dict is all-or-nothing and rejects that). Copies per-parameter
    momentum into the leading param groups whose param counts still match, in order, and
    leaves any group appended since (e.g. a fresh rosa_soft group) in its clean init
    state. Preserves the core Muon/Adam momentum across an architecture add-on. Returns
    the number of groups restored."""
    saved_groups = saved.get("param_groups", [])
    saved_state = saved.get("state", {})
    restored = 0
    for cg, sg in zip(opt.param_groups, saved_groups):
        if len(cg["params"]) != len(sg["params"]):
            break  # structure diverged here; this and later groups stay fresh
        if cg.get("name") and sg.get("name") and cg["name"] != sg["name"]:
            break  # group identity diverged (reordered) -> refuse to restore wrong state
        matched = False
        for cur_p, sidx in zip(cg["params"], sg["params"]):
            entry = saved_state.get(sidx)
            if entry is None:
                continue
            opt.state[cur_p] = {k: (v.to(device=cur_p.device) if torch.is_tensor(v) else v)
                                for k, v in entry.items()}
            matched = True
        restored += int(matched)
    return restored


def train(args):
    h = build(args)
    model, text_model, student, teacher_gdn, teacher_cap, box, codec, lm_head = (
        h["model"], h["text_model"], h["student"], h["teacher_gdn"], h["teacher_cap"],
        h["box"], h["codec"], h["lm_head"])
    init_opt_state, init_opt_type = h["init_opt_state"], h["init_opt_type"]
    rosa_soft = h["rosa_soft"]
    rosa_restore = h.get("rosa_restore") or {}  # warm-start calibrated scale + controller EMA
    lookahead = h.get("lookahead")               # NextLat/TOP aux heads (None when off)

    # Create the run dir + log + sidecar BEFORE the codec pre-fit so the dashboard
    # lists the run (and renders the architecture panel) during the ~2k-step pre-fit
    # instead of only once training starts. truncate ("w"): each run is fresh.
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    logf = open(out / "train.jsonl", "w")
    def emit(rec):
        logf.write(json.dumps(rec) + "\n"); logf.flush()
    # pin threshold: 0.245 matches the legacy UNBOUNDED regime (~the old 0.25 clamp).
    # With --loop-gate-cap the effective gate saturates at cap (cap*tanh), so a fixed
    # 0.245 either can never fire (cap<=0.245) or fires inside the healthy bounded
    # range — scale it to the cap instead. The threshold rides eval records
    # (loop_pin_thr) so the dashboard detector judges by the same number.
    # uncapped pin = "beyond the observed healthy regime", not the old 0.25 design rail:
    # round-1 gate A/B ran gates to 1.9-3.3 stably with ppl improving, so 0.245 flagged
    # every healthy looped run mid-escape. Capped runs keep the exact 0.98*cap rail.
    loop_pin_thr = 4.0 if args.loop_gate_cap <= 0.0 else 0.98 * args.loop_gate_cap
    # schedulefree bakes loop_lr_mult into the group lr (no live multipliers), so live
    # steering is a no-op there; loop_live=0 tells the detector not to write controls.
    loop_live = 0 if args.optimizer == "schedulefree" else 1
    # trainer-owned boost cooling (--loop-anneal-rw): anneal_t0 set when gates escape.
    loop_anneal_on = args.loop_anneal_rw > 0.0 and hasattr(student, "residual_weight")
    loop_anneal_t0 = None
    def write_loop_rw(cur_mult=None):  # loop-usage card data for the dashboard (no-op for a bare core).
        # Returns {loop_max_rw, loop_pinned, loop_pin_thr, loop_live, loop_anneal
        # [, loop_lr_mult]} so eval records carry the gate state into the DB
        # (extra_json) — that's what the detector's loop rules read. cur_mult is the
        # EFFECTIVE multiplier (launch arg x anneal factor, or live override): without
        # it the detector can't see an arg-launched boost and would "boost" a 30x run
        # down to 10x. loop_anneal=1 tells the detector the trainer owns cooling.
        rw = getattr(student, "residual_weight", None)
        if rw is None:
            return None
        eff = (student.effective_rw() if hasattr(student, "effective_rw")
               else rw.detach()).float().cpu()
        if eff.ndim == 1:  # scalar gates: signed per-pass values (legacy card format)
            rwl = [float(x) for x in eff.tolist()]
            split = None
        else:              # head/channel/factored: per-pass max|gate| across channels
            rwl = [float(x) for x in eff.abs().amax(dim=1).tolist()]
            C = int(eff.shape[1])
            G = int(getattr(student, "num_heads", 0) or 0)
            ch_per_group = int(getattr(student, "ch_per_group", 0) or 0)
            if G <= 0 and ch_per_group > 0 and C % ch_per_group == 0:
                G = C // ch_per_group
            if G <= 0:
                G = int(getattr(getattr(student, "core", None), "num_heads", 0) or 0)
            if G <= 0 or C % G:
                G = 1
            per_head = eff.reshape(eff.shape[0], G, C // G).abs().amax(dim=2)
            # Keep the live artifact compact: channel detail is max-pooled into at
            # most 64 contiguous channel buckets, enough to spot hot channel bands.
            bucket_n = min(64, C)
            per_chan = F.adaptive_max_pool1d(eff.abs().unsqueeze(1), bucket_n).squeeze(1)
            split = {
                "heads": G,
                "channels": C,
                "ch_per_head": C // G,
                "channel_buckets": bucket_n,
                "head_abs": [[float(v) for v in row] for row in per_head.tolist()],
                "channel_abs": [[float(v) for v in row] for row in per_chan.tolist()],
            }
        mx = max((abs(x) for x in rwl), default=0.0)
        pinned = int(mx >= loop_pin_thr)
        layer = {"layer": int(args.layer), "max_rw": mx, "rw": rwl}
        if split is not None:
            layer["split"] = split
        (out / "loop_rw.json").write_text(json.dumps({
            "loop_count": int(args.loop_count), "n_layers": 1, "n_pinned": pinned,
            "gate_mode": getattr(student, "gate_mode", "scalar"),
            "gate_cap": float(args.loop_gate_cap), "pin_thr": loop_pin_thr,
            "mean_max_rw": mx, "layers": [layer]}))
        lw = {"loop_max_rw": mx, "loop_pinned": pinned,
              "loop_pin_thr": loop_pin_thr, "loop_live": loop_live,
              "loop_anneal": 1 if loop_anneal_on else 0}
        if cur_mult is not None:
            lw["loop_lr_mult"] = float(cur_mult)
        return lw
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

    opt = make_optimizer(student, codec, args, getattr(model.config, "text_config", model.config),
                         rosa_soft=rosa_soft, lookahead=lookahead)
    opt_params = _optimizer_params(opt)
    if init_opt_state is not None and args.warm_optimizer:  # warm-resume: restore optimizer momentum (Adam/Muon) or
        if init_opt_type == args.optimizer:  # schedulefree averaging, else a restart cold-starts it
            try:
                opt.load_state_dict(init_opt_state)
                print(f"  resumed optimizer state ({args.optimizer}) from warm-start ckpt", flush=True)
            except Exception as e:
                n = _restore_matching_opt_state(opt, init_opt_state)
                print(f"  [warn] full opt-state restore failed ({e}); partially restored "
                      f"{n} matching param group(s), any new group starts fresh", flush=True)
        else:
            print(f"  [warn] optimizer changed {init_opt_type}->{args.optimizer}; cold-starting momentum", flush=True)
    elif init_opt_state is not None:  # --no-warm-optimizer: warm WEIGHTS but FRESH (cold) momentum, deliberate
        print("  [warm-optimizer OFF] warm-start weights loaded; optimizer momentum starts FRESH (cold) "
              "-- deliberate (e.g. loss objective changed, so stale momentum won't fight the new gradient)", flush=True)
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
    train_dtype = getattr(torch, args.dtype)
    batch_offsets = np.arange(args.batch_size, dtype=np.int64)
    token_batch_buf = None if args.train_cache else np.empty((args.batch_size, T + 1), dtype=np.int64)
    # --train-cache prefetch: the per-step memmap reads + H2D uploads (h/block_out/state,
    # ~100-200MB) are synchronous, so a single worker fetches ONE step ahead to hide them
    # behind the GPU step. The worker owns every train-time rng.integers call (train_cache
    # mode never touches rng on the main thread) and fetches are submitted in step order,
    # so the sampled window sequence is identical to drawing in-loop.
    cache_pool = cache_fut = None
    if train_cache is not None:
        cache_pool = ThreadPoolExecutor(max_workers=1)
        def _fetch_cache_batch(need_state):
            cis = rng.integers(0, train_cache.n_windows, size=args.batch_size, dtype=np.int64)
            return (cis,
                    _cache_batch_tensor(train_cache.h, cis, dev, train_dtype),
                    _cache_batch_tensor(train_cache.block_out, cis, dev, train_dtype),
                    _cache_batch_tensor(train_cache.state, cis, dev, train_dtype) if need_state else None)
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
        fixed_starts = pool_rng.integers(0, max(1, split) + 1,
                                         size=args.fixed_trainset, dtype=np.int64)
        if args.disjoint_eval:
            eval_lo = min(max_start, split + T)    # eval windows start past the train pool
        print(f"[grokking] DIAGNOSTIC fixed trainset: cycling {args.fixed_trainset} windows "
              f"(eval {'disjoint' if args.disjoint_eval else 'OVERLAPPING'}) — NOT for production", flush=True)
    best_ppl = float("inf")  # save-on-improve: the eval minimum can never be missed
    t0 = time.time()
    eval_time = 0.0  # tok_per_sec reports train throughput: eval/save pauses excluded
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
        "grokfast_lamb", "grokfast_alpha", "readout_lr_mult", "loop_lr_mult",
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
                           reg_mult=args.ap_reg_mult, restore_best=bool(args.ap_restore_best),
                           lookahead=lookahead)  # restore-best rolls aux heads back too
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
    # ROSA-soft scale calibration (reference recipe: rosa_soft.training). Every
    # --rosa-soft-scale-every steps one train forward also captures rosa_anchor_ops
    # telemetry; when --rosa-soft-scale is unset the RosaAnchorScaleController steers the
    # retrieval softmax scale toward --rosa-soft-target-top-prob. With a fixed scale the
    # probe still runs so top_prob/null_prob/truncated_fraction land in train.jsonl
    # (that's the signal --rosa-soft-logit-epsilon tuning needs).
    rosa_ctl, rosa_stats = None, {}
    if rosa_soft is not None and args.rosa_soft_scale_every > 0:
        from rosa_soft import RosaAnchorScaleConfig, RosaAnchorScaleController
        rosa_ctl = RosaAnchorScaleController(RosaAnchorScaleConfig(
            seq_len=args.seq_len, qk_bits=args.rosa_soft_m, window_size=args.rosa_soft_window,
            target_top_prob=args.rosa_soft_target_top_prob,
            update_interval=args.rosa_soft_scale_every,
            # explicit --rosa-soft-scale pins the temperature; else resume the warm-start
            # ckpt's CALIBRATED scale (e0/e1/Wq/Wk were trained against it) rather than
            # re-deriving the closed-form estimate and silently changing the injection.
            initial_scale=(args.rosa_soft_scale if args.rosa_soft_scale is not None
                           else rosa_restore.get("scale"))))
        if rosa_restore.get("ema") is not None:
            rosa_ctl.top_prob_ema = float(rosa_restore["ema"])  # resume calibration history too
        if args.rosa_soft_scale is None:
            rosa_soft.scale = rosa_ctl.scale      # controller owns the scale from step 0
        _src = ("fixed --rosa-soft-scale, log-only" if args.rosa_soft_scale is not None
                else "resumed from ckpt" if rosa_restore.get("scale") is not None else "adaptive")
        print(f"[rosa-soft] scale controller: init={rosa_ctl.scale:.4f} "
              f"target_top_prob={args.rosa_soft_target_top_prob} every={args.rosa_soft_scale_every} "
              f"({_src})", flush=True)
    # spectral loop levers: Hyperball Frobenius-sphere projection + river-valley switch.
    # Loop gates are EXCLUDED: they are zero-init 2D tensors whose whole job is to grow
    # off the zero sphere — projecting back to R=0 would re-zero them every step and
    # silently kill the loop. Same guard for any other zero-init 2D param (R=0 sphere
    # projection is degenerate).
    _hb_skip = {id(p) for n, p in student.named_parameters() if n in _loop_param_names(student)}
    if lookahead is not None:
        # aux heads are not student mix matrices: norm-pinning the lm_head-clone TOP
        # head (or the NextLat MLP) to its init sphere is never what --hyperball means
        _hb_skip |= {id(p) for p in lookahead.parameters()}
    hb_R = ({p: float(p.detach().norm()) for g in opt.param_groups for p in g["params"]
             if p.ndim == 2 and id(p) not in _hb_skip and float(p.detach().norm()) > 0.0}
            if args.hyperball else {})
    # hyper-connection params: anchored at exact 0/1/identity, exempt from manual decay
    hyper_skip = {id(p) for n, p in student.named_parameters()
                  if n.split(".")[-1].startswith("hyper_")}
    # per-step loop-count sampling (train forwards only; evals/saves see full depth)
    loop_sampling = args.loop_count > 1 and args.loop_sample != "off"
    if loop_sampling:
        from looped_rwkv import sample_loop_count
        rng_loop = np.random.default_rng(args.seed + 777)
    loop_k = args.loop_count
    switched = False
    loop_mult = args.loop_lr_mult  # base mult; live ctl override re-read each step
    loop_mult_eff = loop_mult      # x anneal factor once gates escape (--loop-anneal-rw)
    rosa_lmce_warned = False       # warn-once for the live w_lmce<=0 rosa guard
    dmt_graph = None               # --dmt-cuda-graph runner, built lazily at first DMT step
    for step in range(args.steps):
        if (not switched and args.muon_to_adamw_frac > 0.0
                and args.optimizer in ("muonclip", "spectral_muon")
                and step >= args.muon_to_adamw_frac * args.steps):
            switched = True
            _prev = args.optimizer; args.optimizer = "adamw"   # river-valley: refine tail in AdamW
            opt = make_optimizer(student, codec, args, getattr(model.config, "text_config", model.config),
                                 rosa_soft=rosa_soft, lookahead=lookahead)  # keep aux heads in the
            opt_params = _optimizer_params(opt)                             # rebuilt optimizer
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
            _save_ckpt(out, step, student, codec, args, opt, rosa_soft=rosa_soft, rosa_ctl=rosa_ctl, lookahead=lookahead)
            reason = "interrupt" if stopping else "sigusr1"
            emit({"kind": "checkpoint", "step": step, "reason": reason})
            print(f"  [{reason}] checkpoint saved at step {step}", flush=True)
            if stopping:
                if cache_pool is not None:      # drop the queued prefetch before exit
                    cache_pool.shutdown(wait=False, cancel_futures=True)
                logf.close()
                sys.exit(0)
            _SIG["ckpt"] = False
            if is_sf: opt.train()
        # live-tuning: pull whitelisted overrides (no-op when --control-db missing).
        if step % max(1, args.control_poll_every) == 0:
            ctl.poll(step)
        w_lmce, w_block = ctl.get("w_lmce", args.w_lmce), ctl.get("w_block", args.w_block)
        if rosa_soft is not None and w_lmce <= 0.0:
            # LM-CE is rosa's ONLY gradient path (enforced at launch); a live w_lmce=0
            # override would silently freeze rosa — and with --truncate-forward, stop
            # computing the injection in train forwards while evals still apply it.
            if not rosa_lmce_warned:
                print(f"  [warn] ignoring live w_lmce<=0 override: --rosa-soft trains only "
                      f"through LM-CE; keeping w_lmce={args.w_lmce:g}", flush=True)
                rosa_lmce_warned = True
            w_lmce = args.w_lmce
        w_cos = ctl.get("w_cos", args.w_cos); w_cka = ctl.get("w_cka", args.w_cka)
        w_flow = ctl.get("w_flow", args.w_flow); w_bridge = ctl.get("w_bridge", args.w_bridge)
        agree_gate = ctl.get("agreement_gate", args.agreement_gate)
        w_smt, w_dmt = ctl.get("w_smt", args.w_smt), ctl.get("w_dmt", args.w_dmt)
        grad_clip = ctl.get("grad_clip", args.grad_clip)
        nuc_w = ctl.get("nuc_weight", apilot.overrides.get("nuc_weight", args.nuc_weight))
        nuc_every = int(ctl.get("nuc_every", args.nuc_every))
        lr_scale = ctl.get("lr_scale", 1.0)
        readout_mult = ctl.get("readout_lr_mult", apilot.overrides.get("readout_lr_mult", args.readout_lr_mult))
        loop_mult = ctl.get("loop_lr_mult", args.loop_lr_mult)
        if loop_anneal_on:
            # escape check every 10 steps (tiny-tensor sync) until triggered, then fixed
            if loop_anneal_t0 is None and step % 10 == 0:
                if float(student.effective_rw().abs().max()) >= args.loop_anneal_rw:
                    loop_anneal_t0 = step
                    print(f"[loop-anneal] gates escaped (max|rw| >= {args.loop_anneal_rw:g}) at "
                          f"step {step}; boost {loop_mult:g}x -> 1x over {args.loop_anneal_steps} steps",
                          flush=True)
            loop_mult_eff = 1.0 + (loop_mult - 1.0) * loop_anneal_factor(
                step, loop_anneal_t0, args.loop_anneal_steps)
        else:
            loop_mult_eff = loop_mult
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
            _eval_t0 = time.time()
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
            # loop-usage card; gate state + EFFECTIVE loop_lr_mult ride the eval record too
            lw = write_loop_rw(args.loop_lr_mult if is_sf else loop_mult_eff)
            if lw:
                ev.update(lw)
            emit({"kind": "eval", "step": step, **ev})
            print(f"  [eval] step {step} loss={ev['loss']:.4f} ppl={ev['ppl']:.3f} "
                  f"top1={ev['top1_acc']:.4f}", flush=True)
            # save the BEST eval NO MATTER WHAT -> the minimum is never lost between
            # periodic saves (which is how 12.023/12.030 slipped through before).
            if ev.get("ppl") is not None and ev["ppl"] < best_ppl:
                best_ppl = ev["ppl"]
                _save_best(out, step, ev["ppl"], student, codec, args, opt, rosa_soft=rosa_soft, rosa_ctl=rosa_ctl, lookahead=lookahead)
            # autopilot: also eval the weight-EMA (keep the lower ppl), then check for
            # anti-grokking collapse and escalate (reg up + optional restore-best).
            if args.grok_autopilot:
                ema_ev, ema_saved = apilot.eval_ema(
                    lambda: evaluate(text_model, lm_head, toks, args.eval_windows, args.seq_len, dev,
                                     start_lo=eval_lo, batch_size=args.eval_batch_size),
                    best_ppl, lambda pp: _save_best(out, step, pp, student, codec, args, opt,
                                                    rosa_soft=rosa_soft, rosa_ctl=rosa_ctl,
                                                    lookahead=lookahead))
                if ema_saved:
                    best_ppl = ema_ev["ppl"]
                    emit({"kind": "eval", "step": step, "ema": 1, **ema_ev})
                    print(f"  [autopilot] EMA best ppl={ema_ev['ppl']:.3f}", flush=True)
                if apilot.restore_best:
                    _flush_best_saves()  # on_eval may torch.load best/ckpt.pt (restore-best)
                act = apilot.on_eval(step, ev.get("ppl"), best_ppl)
                if act:
                    emit({"kind": "train", "step": step, **act})
                    print(f"  [autopilot] {act}", flush=True)
            # periodic ckpt (averaged x for schedulefree, since we're in eval mode here)
            if save_every > 0 and step > 0 and step % save_every == 0:
                _save_ckpt(out, step, student, codec, args, opt, rosa_soft=rosa_soft, rosa_ctl=rosa_ctl, lookahead=lookahead)
            student.train()
            if is_sf:
                opt.train()  # back to y for the training step
            eval_time += time.time() - _eval_t0

        x = y = None
        B = args.batch_size
        if train_cache is None:
            if fixed_starts is not None:
                bstarts = fixed_starts[(step * B + batch_offsets) % len(fixed_starts)]
            else:
                bstarts = rng.integers(0, max_start + 1, size=B, dtype=np.int64)
            ids = _token_windows_tensor(toks, bstarts, T + 1, dev, out=token_batch_buf)  # [B,T+1]
            x, y = ids[:, :-1], ids[:, 1:]   # [B,T]

        if loop_sampling:   # applied tightly around the forwards below (exception-safe)
            loop_k = sample_loop_count(args.loop_sample, args.loop_count, rng_loop)
        # student forward through the frozen backbone, or local cached layer-target training.
        hidden = None
        with torch.set_grad_enabled(True):
            if train_cache is None:
                if rosa_ctl is not None and step % args.rosa_soft_scale_every == 0:
                    rosa_soft.want_telemetry = True  # this train forward also captures telemetry
                # lookahead needs the full forward's final hidden, exactly like LM-CE
                box["_truncate"] = bool(args.truncate_forward) and w_lmce == 0.0 and lookahead is None  # perf #3 (EXPERIMENTAL)
                if loop_sampling:
                    student.n_loops = loop_k
                try:
                    outputs = text_model(input_ids=x, use_cache=False)
                    hidden = outputs.last_hidden_state if hasattr(outputs, "last_hidden_state") else outputs[0]
                except _StopForward:
                    hidden = None              # truncated after layer L; box['h']/box['y'] already captured
                finally:
                    # the flag must not outlive THIS forward: _capture_y also fires on the
                    # SMT/DMT direct student calls below and on every evaluate() forward,
                    # where a stale True raises _StopForward into code that can't catch it.
                    box["_truncate"] = False
                    if loop_sampling:          # same reasoning: evals/saves/SMT-DMT need full depth
                        student.n_loops = args.loop_count
                hL = box["h"]                      # [B,T,C] input to layer L (teacher-faithful)
                student_out = box["y"]             # [B,T,C] student RWKV block output

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
                need_state = w_smt > 0 or w_dmt > 0
                if cache_fut is None:                        # prime the one-step-ahead pipeline
                    cache_fut = cache_pool.submit(_fetch_cache_batch, need_state)
                cis, hL, teacher_out, S_gdn = cache_fut.result()
                cache_fut = cache_pool.submit(_fetch_cache_batch, need_state)
                if need_state and S_gdn is None:             # SMT/DMT live-toggled on after prefetch
                    S_gdn = _cache_batch_tensor(train_cache.state, cis, dev, train_dtype)
                if loop_sampling:
                    student.n_loops = loop_k
                try:
                    student_out = student(hL)
                finally:
                    if loop_sampling:
                        student.n_loops = args.loop_count

            lm_ce = chunked_ce(hidden, lm_head, y, fused=bool(args.fused_ce)) if (w_lmce > 0.0 and hidden is not None) else _zero_like_loss(student_out)
            la_parts = None
            so, to = student_out.float(), teacher_out.float()
            if args.block_loss == "rel":              # OpenMOSE per-token relative L2 norm (validated):
                # mean_tok ||teacher-student|| / (||teacher||+eps) over the channel dim. Scale-invariant
                # PER TOKEN (each token normalized by its OWN teacher magnitude, not a global scale), and
                # uses the L2 norm (not squared) so large per-token errors stay ~linear, not quadratic.
                per_tok = (torch.linalg.vector_norm(to - so, dim=-1)
                           / (torch.linalg.vector_norm(to, dim=-1) + 1e-8))       # [B,T]
                if agree_gate > 0.0:                              # trust-region per-token gating
                    aw = do.agreement_weight(so, to).squeeze(-1)  # [B,T]
                    block = (aw * per_tok).sum() / (aw.sum() + 1e-8)
                else:
                    block = per_tok.mean()
            elif agree_gate > 0.0:                                # raw MSE, trust-region per-token gating
                aw = do.agreement_weight(so, to)
                block = (aw * (so - to).pow(2)).sum() / (aw.sum() * so.shape[-1] + 1e-8)
            else:
                block = F.mse_loss(so, to)
            loss = w_lmce * lm_ce + w_block * block
            if lookahead is not None and hidden is not None:
                # NextLat + TOP on the post-norm hidden. ids is [B,T+1] (x plus the
                # last label), so TOP covers the first T+1-W positions (see
                # LookaheadSystem.compute); NextLat covers all T. Weights are baked
                # into the system from the --nextlat-*/--top-* flags.
                la_out = lookahead.compute(hidden, ids, text_model.embed_tokens, lm_head)
                loss = loss + la_out.pop("aux_total")
                la_parts = la_out
            # iterate consistency (equilibrium internalization): captured NOW, before the
            # SMT/DMT direct calls below overwrite the wrapper's last_iter_consist attr
            ic_val = getattr(student, "last_iter_consist", None)
            if args.loop_iter_consist > 0 and ic_val is not None:
                loss = loss + args.loop_iter_consist * ic_val
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
            fid_val = (do.carry_fidelity(so, to)   # consumed only at log cadence — skip the rest
                       if args.distill_fidelity_log and (step % log_every == 0 or step == args.steps - 1)
                       else None)
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
                if dmt_graph is None and getattr(args, "dmt_cuda_graph", 0):
                    # lazy: capture must postdate ALL model surgery (loop wrap, pc-layer,
                    # warm-start) — first-use construction here makes that automatic.
                    from smt_dmt import DMTGraphedRollout
                    dmt_graph = DMTGraphedRollout(student)
                    print("  [dmt-cuda-graph] graphed rollout enabled (per-chunk-index CUDA "
                          "graphs, lazy capture as the curriculum grows)", flush=True)
                try:
                    dmt = dmt_rollout_loss(student, codec, hL[:, :rl],
                                           stride=args.state_stride, block_out=teacher_out[:, :rl],
                                           discount=args.dmt_discount, target_states=target_states[:, :nb],
                                           graphed=dmt_graph)
                except Exception as e:
                    if dmt_graph is None:
                        raise  # eager rollout failure is a real bug — fail loud
                    # FAIL-SOFT: graphs are an optional speedup (default-on); a capture or
                    # replay failure must never kill a long run. Recompute eager, disable
                    # graphs for the rest of the run (a genuine loss bug re-raises eagerly).
                    torch.cuda.synchronize()
                    print(f"  [warn] dmt-cuda-graph failed ({type(e).__name__}: {e}); falling "
                          f"back to the eager rollout for the rest of the run", flush=True)
                    dmt_graph = None
                    args.dmt_cuda_graph = 0
                    dmt = dmt_rollout_loss(student, codec, hL[:, :rl],
                                           stride=args.state_stride, block_out=teacher_out[:, :rl],
                                           discount=args.dmt_discount, target_states=target_states[:, :nb],
                                           graphed=None)
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

        opt.zero_grad(set_to_none=True)
        loss.backward()
        # NaN/Inf skip guard: must sit AFTER backward (a pre-backward isfinite() check
        # syncs and drains the kernel queue between fwd and bwd every step) and BEFORE
        # GrokFast (a non-finite gradient must never enter the gf_ema slow-grad average).
        if not torch.isfinite(loss):
            print(f"step {step}: NON-FINITE loss -> skip", flush=True)
            opt.zero_grad(set_to_none=True)
            continue
        # ROSA-soft probe -> scale calibration. Processed after backward so the telemetry
        # .cpu() syncs overlap the queued backward work instead of draining the fwd queue.
        if rosa_ctl is not None and rosa_soft.last_telemetry is not None:
            _telem, rosa_soft.last_telemetry = rosa_soft.last_telemetry, None
            if args.rosa_soft_scale is None:
                rosa_soft.scale = rosa_ctl.observe(_telem)  # steer toward target top_prob
            _tf = _telem.as_float_dict()
            rosa_stats = {"rosa_scale": float(rosa_soft.scale), "rosa_top_prob": _tf["top_prob"],
                          "rosa_null_prob": _tf["null_prob"], "rosa_entropy": _tf["entropy_norm"]}
            if "truncated_fraction" in _tf:
                rosa_stats["rosa_trunc"] = _tf["truncated_fraction"]
        if gf_lamb > 0.0:                          # GrokFast: amplify the slow-varying
            gf_ps, gf_old_e, gf_old_g = [], [], [] # gradient component so the model reaches
            for p in opt_params:                   # the generalizing solution sooner (no delay)
                if p.grad is None:
                    continue
                gf_ps.append(p)
                e = gf_ema.get(p)
                if e is None:
                    gf_ema[p] = p.grad.detach().clone()  # first sight: the EMA starts AT the grad
                else:
                    gf_old_e.append(e); gf_old_g.append(p.grad)
            if gf_old_e:                           # batched EMA update, one launch per op
                torch._foreach_mul_(gf_old_e, gf_alpha)
                torch._foreach_add_(gf_old_e, gf_old_g, alpha=1.0 - gf_alpha)
            if gf_ps:
                torch._foreach_add_([p.grad for p in gf_ps], [gf_ema[p] for p in gf_ps], alpha=gf_lamb)
        gnorm = torch.nn.utils.clip_grad_norm_(opt_params, grad_clip)
        opt.step()
        if sched is not None:
            sched.step()
            for g in opt.param_groups:               # live LR multipliers (scheduled opts):
                m = lr_scale                          # global lr_scale, readout boost, LLR layerwise
                if readout_mult != 1.0 and g.get("name") == "rwkv_readout":
                    m = m * readout_mult              # boost to close the readout lag
                if loop_mult_eff != 1.0 and g.get("name") == "rwkv_loop":
                    m = m * loop_mult_eff             # loop-gate steer (--loop-lr-mult / live x anneal)
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
                    # hyper-connection params are anchored at exact 0/1/identity values
                    # (codex: decay would drag them off the loss-free-upgrade identity);
                    # gate_chan et al. KEEP decay — the factored gauge fix relies on it.
                    ps = [p for p in g["params"] if p.requires_grad and id(p) not in hyper_skip]
                    if ps:
                        torch._foreach_mul_(ps, 1.0 - lr_g * decay_now)
        # Hyperball: project each 2D weight back onto its initial Frobenius sphere.
        if hb_R:
            with torch.no_grad():
                hb_ps = list(hb_R)
                norms = torch._foreach_norm(hb_ps)  # norms stay on-GPU: no host sync per matrix
                torch._foreach_mul_(hb_ps, [
                    torch.where(n > 0, R / n.float(), torch.ones_like(n, dtype=torch.float32))
                    for n, R in zip(norms, hb_R.values())])
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
                   "gnorm": _finite_float(gnorm),
                   "tok_per_sec": round((step + 1) * args.batch_size * T / max(time.time() - t0 - eval_time, 1e-9))}
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
            if rosa_stats:
                rec.update(rosa_stats)  # most recent probe (probe cadence may differ from log cadence)
            if la_parts:
                rec.update({f"la_{k}": v for k, v in la_parts.items()})
            if loop_sampling:
                rec["loop_k"] = int(loop_k)
            if args.loop_iter_consist > 0 and ic_val is not None:
                rec["iter_consist"] = _finite_float(ic_val)
            emit(rec)
            print(f"step {step}: loss={rec['loss']:.4f} lm_ce={rec['lm_ce']:.4f} "
                  f"block={rec['block']:.4f} smt={rec['smt_mem']:.4f} dmt={rec['dmt_mem']:.4f} "
                  f"state_rms={rec['dmt_state_rms']:.3f} gnorm={rec['gnorm']:.2f}", flush=True)

    if cache_pool is not None:
        cache_pool.shutdown(wait=False, cancel_futures=True)  # drop the dangling prefetch
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
        lw = write_loop_rw(args.loop_lr_mult if is_sf else loop_mult_eff)
        if lw:
            ev.update(lw)
        emit({"kind": "eval", "step": args.steps - 1, **ev})
        print(f"  [eval-final] loss={ev['loss']:.4f} ppl={ev['ppl']:.3f} top1={ev['top1_acc']:.4f}", flush=True)
        if ev.get("ppl") is not None and ev["ppl"] < best_ppl:  # final eval may be the best
            best_ppl = ev["ppl"]
            _save_best(out, args.steps - 1, ev["ppl"], student, codec, args, opt,
                       rosa_soft=rosa_soft, rosa_ctl=rosa_ctl, lookahead=lookahead)
    sd = _save_ckpt(out, args.steps - 1, student, codec, args, opt, rosa_soft=rosa_soft,
                    rosa_ctl=rosa_ctl, lookahead=lookahead)  # x weights (schedulefree in eval mode)
    emit({"kind": "checkpoint", "step": args.steps - 1})
    print(f"saved -> {sd/'ckpt.pt'}", flush=True)


def chunked_ce(hidden, lm_head, labels, chunk=2048, fused=False):
    """Inlined chunked lm_head+CE (matches train_mla.chunked_lmhead_ce) to avoid
    importing train_mla's heavy module chain. fused=True uses flash_attn's Triton CE:
    one-shot logits, in-place backward into the bf16 logit buffer (no fp32 copy, no
    separate grad alloc). Falls back to the chunked path if flash_attn is unavailable."""
    B, T, H = hidden.shape
    flat_h = hidden.reshape(-1, H)
    flat_labels = labels.reshape(-1)
    Wt = lm_head.weight
    bias = getattr(lm_head, "bias", None)
    if fused and _HAS_FLASH_CE:
        logits = F.linear(flat_h, Wt, bias)                              # [N, V] bf16, one shot
        losses, _ = _flash_ce(logits, flat_labels, inplace_backward=True)  # [N] per-token
        return losses.float().sum() / flat_h.shape[0]
    total = hidden.new_zeros((), dtype=torch.float32)
    Nn = flat_h.shape[0]
    for i in range(0, Nn, chunk):
        end = min(i + chunk, Nn)
        logits = F.linear(flat_h[i:end], Wt, bias)   # keep bf16; the fp32 logit copy was redundant memory
        # reduction="none" -> float -> sum: a bf16 "sum" scalar has ulp 32 at magnitude
        # ~5e3, quantizing the mean CE to a 2^-10-nat grid (different models collide on
        # identical loss values). Per-token bf16 rounding (~1e-4 after averaging) stays;
        # the catastrophic sum-then-round does not. Gradients were never affected.
        total = total + F.cross_entropy(logits, flat_labels[i:end], reduction="none").float().sum()
    return total / Nn


@torch.inference_mode()
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
    # accumulate on-GPU; one host sync at the end instead of several per chunk
    tot_ce = torch.zeros((), dtype=torch.float64, device=device)
    tot_correct = torch.zeros((), dtype=torch.int64, device=device)
    tot_block = torch.zeros((), dtype=torch.float64, device=device)
    tot_tok = 0; nb_block = 0
    starts = rng.integers(min(start_lo, max_start), max_start + 1,
                          size=n_windows, dtype=np.int64)
    batch_size = max(1, int(batch_size))
    ids_buf = np.empty((batch_size, T + 1), dtype=np.int64)
    Wt = lm_head.weight
    bias = getattr(lm_head, "bias", None)
    for off in range(0, n_windows, batch_size):
        ss = starts[off:off + batch_size]
        ids = _token_windows_tensor(toks, ss, T + 1, device, out=ids_buf)
        x, y = ids[:, :-1], ids[:, 1:]
        out = text_model(input_ids=x, use_cache=False)
        hidden = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
        # held-out block-MSE: box["h"]/box["y"] were just refreshed by this forward
        if teacher_gdn is not None and box is not None:
            try:
                t_out = teacher_gdn(box["h"])
                tot_block += F.mse_loss(box["y"].float(), t_out.float(), reduction="sum").double()
                nb_block += box["y"].numel()
            except Exception:
                pass
        flat_h = hidden.reshape(-1, hidden.shape[-1]); flat_y = y.reshape(-1)
        for i in range(0, flat_h.shape[0], chunk):
            e = min(i + chunk, flat_h.shape[0])
            logits = F.linear(flat_h[i:e], Wt, bias)
            # fp32 log-softmax for EVAL: a bf16 CE "sum" scalar has ulp 32 at ~5e3,
            # snapping mean CE to a 2^-10-nat grid — different models collided on
            # bit-identical eval ppl (gate A/B scalar==factored to 15 digits). The
            # transient fp32 logit chunk is free under inference_mode (no graph).
            tot_ce += F.cross_entropy(logits.float(), flat_y[i:e], reduction="sum").double()
            tot_correct += (logits.argmax(-1) == flat_y[i:e]).sum()
            tot_tok += e - i
    loss = tot_ce.item() / max(tot_tok, 1)
    res = {"loss": loss, "ppl": math.exp(min(loss, 20.0)), "top1_acc": tot_correct.item() / max(tot_tok, 1)}
    if nb_block > 0:
        res["block_val"] = tot_block.item() / nb_block
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
    ap.add_argument("--warm-optimizer", action=argparse.BooleanOptionalAction, default=True,
                    help="on warm-start (--init-rwkv-ckpt), also restore the saved optimizer momentum "
                         "(default). --no-warm-optimizer keeps the warm WEIGHTS but starts momentum FRESH "
                         "(cold) -- use when the loss objective changed since the ckpt (e.g. raw-MSE -> "
                         "normalized-MSE), so stale ppl-driven momentum doesn't fight the new gradient.")
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--batch-size", type=int, default=8,
                    help="train windows per step (mean-reduced loss). Bigger batch = less gradient "
                         "noise; scale LR up ~2-3x (sqrt(B)) vs the batch=1 recipes or it just trains "
                         "slower per token. Eval uses --eval-batch-size separately.")
    ap.add_argument("--steps", type=int, default=10000)
    ap.add_argument("--state-stride", type=int, default=64)
    ap.add_argument("--decay-cap-delta", type=float, default=0.005)
    ap.add_argument("--loop-count", type=int, default=4,
                    help="train the student AS a LoopedRWKV with this many weight-tied refinement "
                         "passes (residual_weight zero-init => loop==single-pass at start, so it can "
                         "only match the teacher at least as well as a bare core). 1 = bare core.")
    ap.add_argument("--loop-gate", default="scalar", choices=["scalar", "head", "channel", "factored"],
                    help="loop-gate granularity (LoopedRWKV residual_weight). scalar: one gate per pass "
                         "(legacy). head: one gate per head-group [n_loops,G] — per-group loop rate. "
                         "channel: per channel [n_loops,C]. factored: zero-init head factor x (1+gate_chan "
                         "channel delta) — pooled per-group gradient for faster escape from zero, plus an "
                         "automatic coarse-to-fine curriculum (gate_chan's grad is gated by the head factor "
                         "until it opens); weight decay pulls the channel factor toward 1, not 0. All modes "
                         "are exact no-ops at init; --init-rwkv-ckpt from a coarser gate mode broadcasts "
                         "losslessly (finer->coarser is refused).")
    ap.add_argument("--loop-gate-cap", type=float, default=0.0,
                    help="soft-cap the effective loop gate to (-cap, cap) via cap*tanh(g/cap). "
                         "residual_weight is otherwise UNBOUNDED (iter_norm bounds each pass's input, "
                         "not the accumulated output), so a hot loop LR can destabilize the block. "
                         "Bounds the loop contribution by construction (vs relying on grad-clip + the "
                         "dashboard's after-the-fact loop_pinned cool). tanh(0)=0 keeps the init no-op; "
                         "near 0 it is ~identity. 0=off (unbounded, legacy). Try ~0.5 for stability.")
    ap.add_argument("--loop-index", type=int, default=0,
                    help="add a per-pass, zero-init learned offset to each refinement pass's input so the "
                         "weight-tied core can specialize per pass (OpenMythos loop-index embedding). Zero-init "
                         "=> exact no-op at init; pass 1 stays the faithful single-pass. 0=off. Rides the "
                         "rwkv_loop group (loop_lr_mult steers it).")
    ap.add_argument("--loop-hyper", type=int, default=0,
                    help="hyper-connections at the loop boundary (2409.19606): K>=2 parallel residual lanes "
                         "with learned per-pass pool/mix/write and a learned output read. The iso-depth "
                         "scaling-law paper (2604.21106) measured this as the largest loop-capacity lever "
                         "(recurrence exponent 0.45->0.65). Composes with any --loop-gate; exact no-op at "
                         "init; ~n_loops*(K^2+2K)+K extra scalars riding the rwkv_loop group. 0=off, use 2.")
    ap.add_argument("--loop-lora-rank", type=int, default=0,
                    help="per-refinement-pass LoRA on the shared core's linears (unshared loop weights beat "
                         "weight-tying: CART +5-6%%, Dreamer, MoDr). B zero-init => exact no-op; pass 1 is "
                         "NEVER adapted (stays the faithful single-pass). Params ride the rwkv_loop group; "
                         "ckpt keys unchanged (hook-applied). 0=off; 8-16 typical.")
    ap.add_argument("--loop-lora-targets", default="receptance,key,value,output",
                    help="comma-separated core nn.Linear attrs the per-pass LoRA adapts.")
    ap.add_argument("--loop-sample", default="off", choices=["off", "uniform", "poisson"],
                    help="sample the training loop count per step (dynamic beats fixed n in 4 recurrent-"
                         "depth papers: depth extrapolation, less overthinking). uniform=U{1..n}; poisson="
                         "1+Pois(n-1) clamped (mass at full depth). Evals always run at full --loop-count.")
    ap.add_argument("--loop-iter-consist", type=float, default=0.0,
                    help="equilibrium-internalization weight (2605.12466): pull each earlier loop "
                         "iterate toward sg(final iterate) — internalizes refinement into fewer "
                         "passes and enables early exit. With --loop-sample this reproduces "
                         "LoopFormer's shortcut-consistency recipe. Paper-analog weight 0.1. 0=off.")
    ap.add_argument("--loop-lr-mult", type=float, default=1.0,
                    help="LR multiplier for residual_weight (the loop gates, their own 'rwkv_loop' group). "
                         "Default 1x: the refine/warm-start case, where the loop is already trained and should "
                         "move at the base rate. For a FRESH conversion (residual_weight zero-init, tiny "
                         "gradient) pass a higher mult, e.g. 30, so the loop reaches useful magnitude. Applies "
                         "to adamw/spectral_muon via the per-step multiplier (LIVE-tunable: loop_lr_mult; the "
                         "dashboard detector auto-boosts it on loop_stall, releases the boost back toward 1 "
                         "once gates clearly move (loop_release, max|rw|>=0.01), and cools it on loop_pinned) "
                         "and is baked into the group lr for schedulefree (no live multipliers there).")
    ap.add_argument("--loop-anneal-rw", type=float, default=0.0,
                    help="trainer-side deterministic loop-LR cooling: hold the full --loop-lr-mult boost "
                         "until max|effective gate| crosses this threshold (escape complete), then cosine-"
                         "decay the boost to 1x over --loop-anneal-steps. Replaces the dashboard detector's "
                         "reactive loop_pinned/loop_release cooling, which is ingest/sampler-laggy (round-1 "
                         "gate A/B: one arm cooled at max|rw| 0.303, the other at 2.1) and unfair across "
                         "A/B arms. The detector sees loop_anneal=1 on eval records and stops writing "
                         "loop_lr_mult controls (stall-boost still applies; pin becomes a watermark). "
                         "0=off (legacy detector-managed). Calibration: gates escape 0->0.1 in ~100-200 "
                         "steps at 30x; try 0.1.")
    ap.add_argument("--loop-anneal-steps", type=int, default=400,
                    help="steps over which the boost cosine-decays to 1x after --loop-anneal-rw triggers")
    ap.add_argument("--allow-neg-eigval", action=argparse.BooleanOptionalAction, default=True,
                    help="enable RWKV-7 negative eigenvalues (_a_scale 1->2, in-context gate a in (0,2)) to "
                         "match GDN's gated delta rule (beta in (0,2)). DEFAULT TRUE (the structural lever for "
                         "closing the base-ppl gap); use --no-allow-neg-eigval to disable. NOTE: changes a's "
                         "range, so warm-starting a False-trained core perturbs it (best tested fresh / re-adapted).")
    ap.add_argument("--w-lmce", type=float, default=0.1)
    ap.add_argument("--w-block", type=float, default=1.0)
    ap.add_argument("--block-loss", default="mse", choices=["mse", "rel"],
                    help="block-output distillation loss form. 'mse'=raw F.mse_loss (magnitude-weighted; "
                         "legacy default). 'rel'=OpenMOSE per-token relative L2 norm: mean over tokens of "
                         "||teacher-student|| / (||teacher||+1e-8) along the channel dim -- scale-invariant "
                         "per token (each token normalized by its own teacher magnitude), L2 norm NOT squared "
                         "so large per-token errors stay ~linear. rel is O(0.1-1) vs raw MSE's activation-scale "
                         "magnitude, so re-tune --w-block. Composes with --agreement-gate (per-token weighted).")
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
    ap.add_argument("--muon-lr", type=float, default=None,
                    help="Muon lr for 2D matrices. Default is PER-OPTIMIZER (units are NOT shared even though "
                         "the flag is): 4e-6 for spectral_muon (validated by the 24-layer gdn_sweep with the "
                         "recipe levers on), 1e-4 for muonclip (train_mla-validated; 5e-4 blew up two runs). "
                         "Both sit atop the baked-in 0.4*sqrt(max_dim) amplifier (~25.6x @4096, "
                         "~vanilla-Muon-lr/25).")
    ap.add_argument("--muon-adam-lr", type=float, default=3e-4,
                    help="AdamW lr for the 1D/scalar/norm params under --optimizer muonclip/spectral_muon")
    # --- spectral_muon levers (--optimizer spectral_muon). DEFAULTS = the validated sweep
    # recipe (plus-norm row + eq R + DDC 0.5 both @ muon-lr 4e-6; banked wins on all 24 GDN
    # layers). Vanilla Muon: --sm-plus-norm none --sm-equilibrate none --sm-ddc-strength 0. ---
    ap.add_argument("--sm-ns-steps", type=int, default=5, help="Newton-Schulz iterations.")
    ap.add_argument("--sm-cheap-cubic", type=int, default=0,
                    help="Tier2: odd-cubic NS schedule, ~1/3 fewer matmuls (2606.00371); weaker orthogonalization.")
    ap.add_argument("--sm-plus-norm", default="row", choices=["none", "row", "col"],
                    help="Tier1 MUON+: row/col-normalize the orthogonalized update (2602.21545); up to 37%% faster. "
                         "Default 'row' (validated recipe); 'none' for vanilla Muon.")
    ap.add_argument("--sm-equilibrate", default="R", choices=["none", "R", "C", "RC"],
                    help="Tier1 MuonEq: row/col equilibration BEFORE NS (2603.28254); R for hidden weights. "
                         "Default 'R' (validated recipe); 'none' for vanilla Muon.")
    ap.add_argument("--sm-second-moment", type=int, default=0,
                    help="Tier2 Muon2: Adam-style 2nd-moment precondition before NS (2604.09967); ~40%% fewer NS iters.")
    ap.add_argument("--sm-row-uniform", type=int, default=0,
                    help="Tier2 Aurora: equal-row-norm for tall matrices; fixes dead neurons in wide MLPs (2606.27715).")
    ap.add_argument("--sm-spectral-power", type=float, default=0.0,
                    help="Tier3 Muon^p: U.Sigma^p.Vt (0=Muon UVt; ~1/3 good for FINETUNE, 2606.13867). Live.")
    ap.add_argument("--sm-power-method", default="eigh", choices=["eigh", "svd"],
                    help="How Muon^p (p>0) computes U.Sigma^p.Vt: 'eigh' (default; math-identical eigh-on-Gram, "
                         "measured ~14x faster than gesvd on the 4096^2 mix matrices — 161ms vs 2.3s/matrix, the "
                         "iso_L0_muonp_sp 193-tok/s cliff — and near-free on the rank-8 factors) or 'svd' (exact "
                         "gesvd, debug/verification only). eigh squares the condition number (minor precision loss "
                         "on the smallest singular directions, clamped) — fine for p~1/3.")
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
    ap.add_argument("--sm-ddc-strength", type=float, default=0.5,
                    help="DDC (2606.29176): fraction [0,1] of the per-channel rescale-gauge component removed from the "
                         "spectral_muon update; resists over-training collapse, gives cleaner minima. Default 0.5 "
                         "(validated recipe); 0 for vanilla Muon. Live: ddc_strength.")
    ap.add_argument("--sm-ddc-mode", default="both", choices=["row", "col", "both"],
                    help="DDC gauge: row=output-channel scale, col=input-channel scale, both.")
    ap.add_argument("--sm-rsav", type=int, default=0,
                    help="RSAV (SpecMuon-inspired, 2602.16167 adapted): relaxed scalar-auxiliary-variable step gate on "
                         "gradient energy Σ||g||²; damps Muon steps when energy spikes. 0=off (vanilla). Unvalidated on LLMs.")
    ap.add_argument("--sm-rsav-c", type=float, default=1.0,
                    help="RSAV energy offset C (ξ=r/√(Σ||g||²+C)). Set ~ the run's typical Σ||g||² so C bites; sweep per model.")
    ap.add_argument("--sm-rsav-cap", type=float, default=0.2,
                    help="RSAV step-gate clamp: ξ ∈ [1-cap, 1+cap]. 0 forces ξ≡1 (inert = vanilla).")
    ap.add_argument("--sm-rsav-relax", type=float, default=0.0,
                    help="RSAV relaxation η∈[0,1] pulling r toward √(E+C) each step; 0=pure SAV lag, 1=strongest reset.")
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
    ap.add_argument("--compile-core", type=int, default=0,
                    help="torch.compile the RWKV CORE in-place (nn.Module.compile: param identity, "
                         "state_dict keys, ckpt format unchanged; LoopedRWKV wrapper + hooks + losses "
                         "stay eager). Fuses the token-shift/LoRA/sigmoid elementwise soup around "
                         "chunk_rwkv7 — measured ~3x the wkv kernel time uncompiled, and the loop pays "
                         "it n_loops times. A few recompiles expected (train/eval T, SMT/DMT chunk "
                         "shapes). Default OFF: fusion changes bf16 rounding, so never flip it mid-A/B; "
                         "validate with the perf test suite before adopting.")
    ap.add_argument("--compile-core-mode", default="default",
                    choices=["default", "reduce-overhead", "max-autotune"],
                    help="torch.compile mode for --compile-core. reduce-overhead adds inductor "
                         "CUDA-graph trees on the fused partitions (per-shape graphs, safe across "
                         "multiple calls per step) — the supported cudagraph path for the general "
                         "case; --dmt-cuda-graph captures whole DMT chunk steps instead.")
    ap.add_argument("--dmt-cuda-graph", type=int, default=None,
                    help="capture each DMT rollout chunk step as a CUDA graph pair (smt_dmt."
                         "DMTGraphedRollout): the rollout is sequential 64-token calls whose "
                         "tiny-kernel launch overhead dominates; graph replay collapses each "
                         "step's fwd+bwd to single graph launches. GPU-validated: losses bit-"
                         "exact vs eager (fp32 AND bf16), grads exact on all deterministic "
                         "params, 1.9x on a 16-chunk rollout fwd+bwd. One graph pair per chunk "
                         "index, captured lazily as the DMT curriculum grows; ragged tail chunks "
                         "run eager. COST: static buffers, ~3.5GB at B=8/T=1024/16 chunks. "
                         "Default AUTO: on whenever DMT runs (w_dmt>0) on CUDA, unless "
                         "--compile-core owns the core (dynamo inside manual capture is "
                         "unsupported). 1=force (refuses --compile-core/CPU), 0=off. FAIL-SOFT: "
                         "a capture/replay failure warns and falls back to the eager rollout — "
                         "it never kills a run.")
    ap.add_argument("--fused-ce", type=int, default=1,
                    help="use flash_attn's fused Triton cross-entropy (in-place backward into the bf16 "
                         "logit buffer: no fp32 logit copy, no separate grad alloc) for the LM-CE term. "
                         "Default on: GPU-validated exact loss (rel-err 6e-6), grads match within bf16 "
                         "noise (2.6e-3 hidden / 4.4e-4 lm_head), 15%% faster, -2GB peak. Falls back to the "
                         "chunked path automatically if flash_attn isn't importable. Set 0 to force chunked.")
    ap.add_argument("--truncate-forward", type=int, default=0,
                    help="EXPERIMENTAL: when --w-lmce 0 (I/O-first, non-cache), abort the backbone forward "
                         "right after the converted layer via a sentinel exception -> skips layers L+1..N + "
                         "the lm_head (~85-90%% of fwd FLOPs for an early layer). Needs GPU validation of the "
                         "autograd unwind before trusting; default off.")
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--eval-every", type=int, default=100, help="held-out eval cadence (0=off)")
    ap.add_argument("--eval-windows", type=int, default=16,
                    help="held-out eval windows of --seq-len tokens each. Raise for final selection; "
                         "the default favors throughput during conversion.")
    ap.add_argument("--eval-batch-size", type=int, default=8,
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
    # --- ROSA-soft (rosa_soft_layer.RosaAnchorLayer): additive retrieval injection on
    # the converted layer's output, via research/rosa_soft's differentiable CUDA
    # softmax-suffix-match op. Default off; e0=e1=0 init -> exact no-op until trained.
    ap.add_argument("--rosa-soft", type=int, default=0,
                    help="attach a RosaAnchorLayer to the converted layer's output "
                         "(rosa_soft_layer.py). Additive injection, no-op at init. "
                         "Requires --optimizer adamw or spectral_muon. 0=off.")
    ap.add_argument("--rosa-soft-m", type=int, default=4,
                    help="ROSA-soft route bit-width M (hidden_size must be divisible by it).")
    ap.add_argument("--rosa-soft-window", type=int, default=32,
                    help="ROSA-soft rosa_anchor_ops window_size (candidate lookback span).")
    ap.add_argument("--rosa-soft-value-heads", type=int, default=None,
                    help="ROSA-soft grouped value heads H_v. Default keeps the previous one-value-head-per-route "
                         "mapping; lower values reduce Wv params/activation while preserving output shape "
                         "when (hidden_size // M) %% H_v == 0.")
    ap.add_argument("--rosa-soft-dim", type=int, default=128,
                    help="ROSA-soft low-rank routing width d: route a d-dim Q/K subspace instead of the full "
                         "hidden. The CUDA kernel launches B*T*(d/M) blocks, so cost is LINEAR in d -- this is "
                         "the throughput lever. Default 128 (=32 routes at M=4, ~0.66 s/step/site post-2026-07-02 "
                         "perf-opts); full 4096-wide routing is ~30x slower (~21 s/step). Must be divisible by "
                         "--rosa-soft-m. 0=full hidden width (pass explicitly to opt into the slow path).")
    ap.add_argument("--rosa-soft-lr", type=float, default=None,
                    help="LR for the rosa_soft param group; default falls back to --lr (adamw) "
                         "or --muon-adam-lr (spectral_muon).")
    ap.add_argument("--rosa-soft-scale", type=float, default=None,
                    help="FIXED rosa_anchor_ops scale (disables auto-calibration; telemetry probing still "
                         "logs). Default: the RosaAnchorScaleController owns the scale when "
                         "--rosa-soft-scale-every>0, else per-call auto-resolve.")
    ap.add_argument("--rosa-soft-scale-every", type=int, default=100,
                    help="probe rosa_anchor_ops telemetry every N train steps (top_prob/null_prob/entropy/"
                         "truncated_fraction -> train.jsonl as rosa_*) and, when --rosa-soft-scale is unset, "
                         "auto-calibrate the retrieval softmax scale toward --rosa-soft-target-top-prob "
                         "(reference RosaAnchorScaleController, rosa_soft.training). 0=off (legacy static scale).")
    ap.add_argument("--rosa-soft-target-top-prob", type=float, default=0.8,
                    help="scale-controller target for the retrieval softmax top probability (reference "
                         "training default 0.8; higher = sharper retrieval).")
    ap.add_argument("--rosa-soft-logit-epsilon", type=float, default=0.0,
                    help="rosa_anchor_ops early-stop tolerance. 0.0 is exact; values like 1e-3 can speed "
                         "window>=64 runs after checking metrics/truncated_fraction.")
    ap.add_argument("--rosa-soft-qk-damper", type=float, default=0.0,
                    help="rosa_anchor_ops Q/K gradient damper in [0,1]. Default matches source; try 0.1..0.3 "
                         "only if Q/K saturation collapses route entropy.")
    # --- Lookahead (lookahead_module.py): NextLat + TOP future-prediction aux
    # objectives on the full-forward final hidden. Training-only, dropped at
    # inference; all default OFF. Requires --optimizer adamw or spectral_muon
    # (aux heads ride the AdamW fallback — NextLat is Muon-unstable per its authors).
    from lookahead_module import add_lookahead_cli
    add_lookahead_cli(ap)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.muon_lr is None:  # per-optimizer default: the two muon paths have different validated units
        args.muon_lr = 4e-6 if args.optimizer == "spectral_muon" else 1e-4
    if args.loop_hyper and args.loop_hyper < 2:
        raise SystemExit(f"--loop-hyper {args.loop_hyper}: K=1 hyper-connections are provably no better "
                         f"than a plain residual (HC paper Lambda-pattern needs n>1); use 2, or 0 to disable.")
    if args.loop_hyper and args.loop_count <= 1:
        raise SystemExit("--loop-hyper requires --loop-count > 1: the lanes live at the loop boundary.")
    if args.loop_hyper and args.optimizer == "schedulefree":
        raise SystemExit("--loop-hyper is incompatible with --optimizer schedulefree: it applies "
                         "weight decay via group['weight_decay'] to the whole rwkv_loop group, "
                         "which would drag the hyper 0/1/identity anchors off the loss-free upgrade.")
    if args.loop_lora_rank and args.loop_count <= 1:
        raise SystemExit("--loop-lora-rank requires --loop-count > 1: the adapters live on refinement passes.")
    if args.loop_lora_rank and args.compile_core:
        print("  [warn] --loop-lora-rank + --compile-core: the per-pass LoRA hooks on the core's "
              "linears will graph-break/recompile inside the compiled core — expect reduced speedup.",
              flush=True)
    if args.loop_sample != "off" and args.loop_count <= 1:
        raise SystemExit("--loop-sample needs --loop-count > 1: there is no loop to sample.")
    if args.loop_iter_consist > 0 and args.loop_count <= 1:
        raise SystemExit("--loop-iter-consist needs --loop-count > 1: there are no iterates to align.")
    if (args.loop_gate != "scalar" or args.loop_index or args.loop_hyper
            or args.loop_lora_rank) and args.optimizer == "muonclip":
        raise SystemExit(f"--loop-gate {args.loop_gate}/--loop-index requires adamw/spectral_muon/"
                         f"schedulefree: muonclip routes params by ndim and would send the 2D gate "
                         f"tensors to Muon (Newton-Schulz orthogonalization of loop gates is meaningless).")
    if args.loop_anneal_rw > 0.0 and args.optimizer == "schedulefree":
        raise SystemExit("--loop-anneal-rw requires a live loop multiplier (adamw/spectral_muon): "
                         "schedulefree bakes loop_lr_mult into the group lr at build time, so the "
                         "boost cannot be annealed during the run.")
    la_on = (args.nextlat_weight > 0 or args.top_weight > 0
             or args.nextlat_jump_weight > 0 or args.concept_weight > 0)
    if la_on and args.optimizer not in ("adamw", "spectral_muon"):
        raise SystemExit(f"--nextlat-weight/--top-weight require --optimizer adamw or spectral_muon "
                         f"(got {args.optimizer!r}): the lookahead group is wired into the AdamW and "
                         f"SpectralMuon(AdamW-fallback) paths only — NextLat is Muon-unstable per its authors.")
    if la_on and args.train_cache:
        raise SystemExit("--nextlat-weight/--top-weight are incompatible with --train-cache: cached "
                         "local training never runs the full forward, so there is no final hidden "
                         "for the lookahead objectives to supervise.")
    if la_on and args.truncate_forward and args.w_lmce <= 0.0:
        print("  [warn] --truncate-forward is ignored while lookahead objectives are on: "
              "NextLat/TOP need the full forward's final hidden every step.", flush=True)
    if args.top_weight > 0 and args.top_window >= args.seq_len:
        raise SystemExit(f"--top-window {args.top_window} must be < --seq-len {args.seq_len}: "
                         f"TOP covers the first seq_len+1-window positions of each window.")
    for _fl, _hz, _w in (("--nextlat-d", args.nextlat_d, args.nextlat_weight),
                         ("--nextlat-jump-k", args.nextlat_jump_k, args.nextlat_jump_weight),
                         ("--concept-chunk", args.concept_chunk, args.concept_weight)):
        if _w > 0 and _hz >= args.seq_len:   # parse-time, not step-1-after-model-load
            raise SystemExit(f"{_fl} {_hz} must be < --seq-len {args.seq_len}.")
    if args.rosa_soft and args.optimizer not in ("adamw", "spectral_muon"):
        raise SystemExit(f"--rosa-soft requires --optimizer adamw or spectral_muon (got {args.optimizer!r}); "
                         f"the rosa_soft group is wired into the AdamW and SpectralMuon paths only "
                         f"(spectral_muon runs rosa-soft on its built-in AdamW fallback).")
    if args.rosa_soft and args.w_lmce <= 0.0:
        raise SystemExit(f"--rosa-soft trains ONLY through the LM-CE path — block/SMT/DMT/eval-block all read "
                         f"the PRE-injection output (box['y']) — so --w-lmce must be > 0 (got {args.w_lmce}). "
                         f"The reference recipe (research/rosa_soft/rosa_soft/training.py) trains rosa purely "
                         f"on next-token CE; with w_lmce=0 the module is frozen dead weight.")
    if args.rosa_soft and args.train_cache:
        raise SystemExit("--rosa-soft is incompatible with --train-cache: cached local training disables LM CE "
                         "(rosa's only gradient path), and the cached path takes student(hL)'s return value, "
                         "which would fold the injection into the block loss (the non-cache path keeps the "
                         "block loss pure via box['y']).")
    if args.dmt_cuda_graph is None:
        # AUTO (default): graphed DMT whenever it is legal — the runner only constructs
        # when DMT is actually active (w_dmt>0), so this costs nothing otherwise.
        # compile-core owns the core (dynamo inside manual capture is unsupported) and
        # capture needs CUDA; both silently resolve auto to off rather than erroring.
        args.dmt_cuda_graph = int(not args.compile_core and str(args.device).startswith("cuda"))
    elif args.dmt_cuda_graph:  # EXPLICIT 1: fail loud on illegal combinations
        if args.compile_core:
            raise SystemExit("--dmt-cuda-graph is incompatible with --compile-core: a dynamo-compiled "
                             "forward inside torch.cuda.make_graphed_callables capture is unsupported "
                             "(inductor may allocate/recompile at replay time). Pick one; "
                             "--compile-core-mode reduce-overhead is the supported cudagraph path "
                             "for the compiled core.")
        if not str(args.device).startswith("cuda"):
            raise SystemExit(f"--dmt-cuda-graph requires --device cuda (CUDA graph capture), got {args.device!r}.")
        if args.w_dmt <= 0.0:
            print("  [warn] --dmt-cuda-graph set but --w-dmt is 0: the graphed rollout only "
                  "constructs when DMT is active (live w_dmt can still enable it).", flush=True)
    if args.rosa_soft:
        # pre-validate what rosa_anchor_ops would otherwise only reject at the FIRST
        # FORWARD — i.e. minutes later, after the model load.
        if not (1 <= args.rosa_soft_window <= 512):
            raise SystemExit(f"--rosa-soft-window {args.rosa_soft_window} out of range: rosa_anchor_ops "
                             f"CUDA backward supports window_size in [1, 512].")
        if not (1 <= args.rosa_soft_m <= 32):
            raise SystemExit(f"--rosa-soft-m {args.rosa_soft_m} out of range: per-route bit dim must be in [1, 32].")
        if args.rosa_soft_dim and args.rosa_soft_dim % args.rosa_soft_m != 0:
            raise SystemExit(f"--rosa-soft-dim {args.rosa_soft_dim} must be divisible by "
                             f"--rosa-soft-m {args.rosa_soft_m}.")
        if not str(args.device).startswith("cuda"):
            raise SystemExit(f"--rosa-soft requires --device cuda (rosa_anchor_ops is CUDA-only), got {args.device!r}.")
        # rosa_anchor_ops launches B*T*R CUDA blocks (R = route_dim/M query routes) — cost is
        # LINEAR in routes. Full width at C=4096 is R=1024 (~21.2s/step measured 2026-07-02);
        # default d=128 (R=32) is the throughput lever, orthogonal to window/value-heads.
        if not args.rosa_soft_dim:
            print("  [warn] --rosa-soft-dim 0 routes the FULL hidden width (R=C/M query routes; "
                  "R=1024 at C=4096 measured ~21.2s/step, 2026-07-02 post-perf-opts). "
                  "Strongly consider --rosa-soft-dim 128..512.",
                  flush=True)
        elif args.rosa_soft_dim // args.rosa_soft_m > 256:
            print(f"  [warn] --rosa-soft-dim {args.rosa_soft_dim} -> "
                  f"{args.rosa_soft_dim // args.rosa_soft_m} query routes; rosa_anchor_ops cost is "
                  f"linear in routes — expect slow steps.", flush=True)
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
