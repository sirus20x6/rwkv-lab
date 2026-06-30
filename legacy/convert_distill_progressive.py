#!/usr/bin/env python
"""Progressive GDN->RWKV-7 conversion with per-step distillation against the
clean original. Never lets the model break: convert one GDN layer (init+codec),
then distill ALL converted layers so far toward the frozen original model's
residual stream (relative MSE per layer) to absorb the small drift, then add the
next layer. Each distill starts from a near-clean model -> stable + recoverable,
unlike distilling the fully-broken 24-layer stack at the end (which plateaued ~38).

Order is top-down (backward) so a converting layer's input stays original-ish.
Emits the dashboard schema to runs/<run-name>/ (train = distill loss, eval = ppl).
"""
from __future__ import annotations

import sys
sys.modules.setdefault("torchvision", None)

import argparse
import json
import math
import os
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


@torch.no_grad()
def batched_eval(tm, lm_head, toks, n_windows, T, device, batch=8, seed=12345, chunk=4096):
    """Batched held-out LM eval -> {loss, ppl, top1_acc} (same windows as
    convert_train.evaluate for a given seed, since the rng draw order matches)."""
    rng = np.random.default_rng(seed); N = len(toks); maxs = N - (T + 1)
    ce = 0.0; ntok = 0; cor = 0; done = 0
    while done < n_windows:
        b = min(batch, n_windows - done)
        starts = [int(rng.integers(0, maxs + 1)) for _ in range(b)]
        ids = torch.as_tensor(np.stack([np.asarray(toks[s:s + T + 1], dtype=np.int64) for s in starts]),
                              device=device)
        x, y = ids[:, :-1], ids[:, 1:]
        out = tm(input_ids=x, use_cache=False)
        hidden = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
        fh = hidden.reshape(-1, hidden.shape[-1]); fy = y.reshape(-1)
        W = lm_head.weight; bias = getattr(lm_head, "bias", None)
        for i in range(0, fh.shape[0], chunk):
            e = min(i + chunk, fh.shape[0])
            lg = F.linear(fh[i:e], W, bias).float()
            ce += F.cross_entropy(lg, fy[i:e], reduction="sum").item()
            cor += (lg.argmax(-1) == fy[i:e]).sum().item(); ntok += e - i
        done += b
    loss = ce / max(ntok, 1)
    return {"loss": loss, "ppl": math.exp(min(loss, 20.0)), "top1_acc": cor / max(ntok, 1)}

from rwkv8_deltanet import rwkv8_timemix_from_config
from build_memory_targets import load_token_stream
from convert_train import evaluate
from convert_stack import capture_layer, fit_readout
from data_stream import WindowStream
from looped_rwkv import LoopedRWKV


def _text(m):
    return getattr(m.model, "language_model", m.model)


def _hook(store, i):
    def h(mod, a, out):
        store[i] = out[0] if isinstance(out, tuple) else out
    return h


def _loop_rw_mean(s_layers, converted):
    """Mean over converted layers of each LoopedRWKV's max |residual_weight| (loop
    passes 1..N). 0 == loops collapsed to single-pass. None if not looping."""
    vals = []
    for L in converted:
        rw = getattr(getattr(s_layers[L], "linear_attn", None), "residual_weight", None)
        if rw is not None and rw.numel() > 1:
            vals.append(float(rw.detach().abs()[1:].max()))
    return (sum(vals) / len(vals)) if vals else None


def logit_kl(s_final, t_final, lm_head, temp=1.0, chunk=4096):
    """KL(teacher || student) over the vocab, chunked over tokens so the
    248k-vocab logits never fully materialize. s_final/t_final: [B,T,H]."""
    H = s_final.shape[-1]
    fs = s_final.reshape(-1, H); ft = t_final.reshape(-1, H)
    W = lm_head.weight; bias = getattr(lm_head, "bias", None)
    Nn = fs.shape[0]; tot = fs.new_zeros((), dtype=torch.float32)
    for i in range(0, Nn, chunk):
        e = min(i + chunk, Nn)
        sl = F.linear(fs[i:e], W, bias).float() / temp
        tl = F.linear(ft[i:e], W, bias).float() / temp
        slp = F.log_softmax(sl, -1); tlp = F.log_softmax(tl, -1)
        tot = tot + (tlp.exp() * (tlp - slp)).sum()
    return tot / max(Nn, 1) * (temp * temp)


def lens_kl(s_hid, t_hid, norm, lm_head, stride, temp=1.0, chunk=4096):
    """A2 early-exit logit-KL: project an intermediate layer's hidden through the
    frozen final-norm + LM head (logit-lens) and KL-match student->teacher.
    Token-subsampled by `stride` to bound the 248k-vocab cost."""
    H = s_hid.shape[-1]
    fs = norm(s_hid.reshape(-1, H)[::stride])
    ft = norm(t_hid.reshape(-1, H)[::stride])
    return logit_kl(fs.unsqueeze(0), ft.unsqueeze(0), lm_head, temp, chunk)


def distill(teacher, t_tm, student, s_tm, lm_head, t_norm, named_params, t_h, s_h, t_blk, s_blk,
            converted, nL, toks, stream, steps, lr, seqlen, device, emit, gstep, batch=2,
            w_kl=1.0, w_hidden=1.0, w_block=0.5, w_exit=0.0, exit_stride=8,
            kl_temp=1.0, curr_start=0, skip_spike=8.0, lr_floor_frac=0.3,
            warmup=0, hold_frac=0.0, weight_decay=0.0, lr_decay_mult=0.2):
    # fp32 MASTER weights: the converted layers run in bf16, but bf16 AdamW moments
    # quantize away the tiny decay/gate/readout updates needed to shave the last ppl.
    # Keep fp32 masters + fp32 moments and round back into the bf16 forward params
    # each step. Split LR: sensitive RWKV-7 transition params (decay/gate/key-mod) at
    # lr*lr_decay_mult, everything else (projections/readout/loop) at the full lr.
    # weight_decay defaults OFF so the already-fitted readouts of early-converted layers
    # don't slowly shrink across the many later layers they stay trainable for.
    bf16 = [p for _, p in named_params]
    SLOW = (".w0", ".w1", ".w2", ".a0", ".a1", ".a2", ".k_k", ".k_a", ".out_correct_d")
    is_slow = [any(name.endswith(s) for s in SLOW) for name, _ in named_params]
    masters = [p.detach().clone().float().requires_grad_(True) for p in bf16]
    g_norm = [m for m, s in zip(masters, is_slow) if not s]
    g_slow = [m for m, s in zip(masters, is_slow) if s]
    groups = [{"params": g_norm, "lr": lr}]
    if g_slow:
        groups.append({"params": g_slow, "lr": lr * lr_decay_mult})
    opt = torch.optim.AdamW(groups, betas=(0.9, 0.95), weight_decay=weight_decay)
    # WSD schedule (warmup -> [hold] -> cosine-to-floor):
    #  - warmup: linear 0->peak over `warmup` steps. A fresh AdamW each layer starts
    #    with a ~0 second-moment estimate, so jumping straight to peak lr takes
    #    oversized first steps -> the early-in-layer gnorm spikes. Ramping guards that.
    #  - hold: optional plateau at peak (cosine is already near-flat at the top, so
    #    this is usually 0 -> off).
    #  - decay: cosine to lr*lr_floor_frac. The floor keeps hard layers moving (they
    #    ended blocks with gnorm ~0.6-0.8, still signal, when lr had hit ~2e-5).
    # warmup=0, hold_frac=0 reproduces the old pure CosineAnnealingLR exactly.
    warmup = max(0, min(int(warmup), max(steps - 1, 0)))
    hold = min(max(0, int(hold_frac * steps)), max(steps - warmup - 1, 0))
    decay_total = max(steps - warmup - hold, 1)

    def _wsd(epoch):
        if warmup and epoch < warmup:
            return (epoch + 1) / warmup
        e = epoch - warmup
        if e < hold:
            return 1.0
        p = min((e - hold) / decay_total, 1.0)
        return lr_floor_frac + 0.5 * (1.0 - lr_floor_frac) * (1.0 + math.cos(math.pi * p))

    sch = torch.optim.lr_scheduler.LambdaLR(opt, _wsd)
    last_kl = 0.0; gn_hist = []; skips = 0
    for i in range(steps):
        # Pillar B: local-first curriculum — ramp the rollout length short->full.
        T = int(curr_start + (i / max(steps - 1, 1)) * (seqlen - curr_start)) if (0 < curr_start < seqlen) else seqlen
        # FRESH windows every step from the shared stream — never reused across the
        # whole conversion (windows are seqlen-spaced, so a T<=seqlen prefix is non-overlapping).
        starts = stream.next_starts(batch)
        x = torch.as_tensor(np.stack([np.asarray(toks[int(s):int(s) + T], dtype=np.int64) for s in starts]),
                            device=device)
        with torch.no_grad():
            t_out = t_tm(input_ids=x, use_cache=False)
        t_final = (t_out.last_hidden_state if hasattr(t_out, "last_hidden_state") else t_out[0]).detach()
        s_out = s_tm(input_ids=x, use_cache=False)
        s_final = s_out.last_hidden_state if hasattr(s_out, "last_hidden_state") else s_out[0]

        kl = logit_kl(s_final, t_final, lm_head, kl_temp)            # global, targets ppl
        loss = w_kl * kl
        if w_hidden > 0:                                            # full-trajectory residual match
            h = x.new_zeros((), dtype=torch.float32)
            for li in range(nL):
                t = t_h[li].detach().float()
                h = h + F.mse_loss(s_h[li].float(), t) / (t.pow(2).mean() + 1e-6)
            loss = loss + w_hidden * (h / nL)
        if w_block > 0 and converted:                              # A1: per-layer readout (linear_attn) match
            bl = x.new_zeros((), dtype=torch.float32)
            for L in converted:
                t = t_blk[L].detach().float()
                bl = bl + F.mse_loss(s_blk[L].float(), t) / (t.pow(2).mean() + 1e-6)
            loss = loss + w_block * (bl / len(converted))
        if w_exit > 0 and converted and t_norm is not None:        # A2: per-layer early-exit logit-KL
            ek = x.new_zeros((), dtype=torch.float32)
            for L in converted:
                ek = ek + lens_kl(s_h[L], t_h[L].detach(), t_norm, lm_head, exit_stride, kl_temp)
            loss = loss + w_exit * (ek / len(converted))

        for p in bf16:                                             # fresh grads each step (no accumulation)
            p.grad = None
        if not torch.isfinite(loss):                               # C: reset on non-finite, don't propagate
            skips += 1; sch.step(); gstep += 1; continue           # don't emit junk loss to the chart
        loss.backward()
        for m, p in zip(masters, bf16):                            # mirror bf16 grads onto fp32 masters
            m.grad = p.grad.float() if p.grad is not None else None
        gn = torch.nn.utils.clip_grad_norm_(masters, 1.0)          # clip/step in fp32
        med = float(np.median(gn_hist)) if len(gn_hist) >= 10 else None
        spike = (not bool(torch.isfinite(gn))) or (med is not None and float(gn) > skip_spike * med)
        sch.step(); gstep += 1
        if spike:                                                  # C: trust-region — drop oversized/NaN steps
            skips += 1; continue                                   # skip update AND skip emit (junk loss)
        opt.step()
        with torch.no_grad():                                      # round fp32 masters back into bf16 forward params
            for m, p in zip(masters, bf16):
                p.copy_(m)
        gn_hist.append(float(gn))
        if len(gn_hist) > 50:
            gn_hist.pop(0)
        last_kl = float(kl)
        emit({"kind": "train", "step": gstep, "loss": float(loss), "kl": last_kl,
              "gnorm": float(gn), "lr": sch.get_last_lr()[0], "skipped": 0})
    if skips:
        print(f"      (skipped {skips}/{steps} unstable steps)", flush=True)
    return gstep, last_kl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="Qwen3.5-9B-Base")
    ap.add_argument("--data", default="/thearray/git/babyllm/data/cache/qwen3.6_fwedu_train")
    ap.add_argument("--val-data", default="/thearray/git/babyllm/data/cache/qwen3.6_fwedu_val",
                    help="held-out split for eval (disjoint from training data)")
    ap.add_argument("--max-layers", type=int, default=0)
    ap.add_argument("--cap-windows", type=int, default=32)
    ap.add_argument("--cap-seqlen", type=int, default=1024)
    ap.add_argument("--state-stride", type=int, default=64)
    ap.add_argument("--codec-steps", type=int, default=400)
    ap.add_argument("--codec-lr", type=float, default=1e-3)
    ap.add_argument("--decay-cap-delta", type=float, default=0.005)
    ap.add_argument("--loop-count", type=int, default=1, help="wrap each converted layer in N looped iterations (1=off)")
    ap.add_argument("--allow-neg-eigval", action="store_true", help="enable negative-eigenvalue RWKV-7 (a-gate x2)")
    ap.add_argument("--distill-steps", type=int, default=400)
    ap.add_argument("--distill-lr", type=float, default=2e-4)
    ap.add_argument("--distill-seqlen", type=int, default=512)
    ap.add_argument("--w-kl", type=float, default=1.0, help="logit-KL weight (primary, targets ppl)")
    ap.add_argument("--w-hidden", type=float, default=1.0, help="hidden-state MSE weight (auxiliary)")
    ap.add_argument("--w-block", type=float, default=0.5, help="A1: per-layer linear_attn readout MSE")
    ap.add_argument("--w-exit-kl", type=float, default=0.0, help="A2: per-layer early-exit logit-KL (expensive)")
    ap.add_argument("--exit-kl-stride", type=int, default=8, help="token subsample for early-exit KL")
    ap.add_argument("--curriculum-start", type=int, default=0, help="B: ramp rollout len from this to distill-seqlen (0=off)")
    ap.add_argument("--skip-spike", type=float, default=8.0, help="C: skip step if pre-clip gnorm > this x running median")
    ap.add_argument("--lr-floor-frac", type=float, default=0.3, help="cosine LR floor as fraction of peak (was 0.1; hard layers starved)")
    ap.add_argument("--warmup-steps", type=int, default=150, help="WSD: linear LR warmup 0->peak after each swap (cold-AdamW guard; longer = gentler layer-in)")
    ap.add_argument("--hold-frac", type=float, default=0.03, help="WSD: hold at peak for this fraction of steps after warmup before cosine decay (the 'stable' plateau / shelf)")
    ap.add_argument("--resume", action="store_true", help="reattach already-converted layers from --out and continue; skips them and advances the data stream (no reuse)")
    ap.add_argument("--snapshot-every", type=int, default=4, help="also write a numbered .L<id>.pt milestone snapshot every N layers (0=off)")
    ap.add_argument("--divergence-factor", type=float, default=1.5, help="halt and preserve last-good ckpt if a layer's post-distill ppl exceeds this x baseline")
    ap.add_argument("--weight-decay", type=float, default=0.0, help="AdamW weight decay (default 0: don't shrink already-fitted readouts of early layers)")
    ap.add_argument("--lr-decay-mult", type=float, default=0.2, help="LR multiplier for sensitive RWKV-7 transition params (w/a/k_k/k_a/out_correct_d)")
    ap.add_argument("--kl-temp", type=float, default=1.0)
    ap.add_argument("--batch", type=int, default=2, help="distill micro-batch (grad; memory grows with #layers)")
    ap.add_argument("--io-batch", type=int, default=8, help="batch for no-grad capture/eval forwards")
    ap.add_argument("--eval-windows", type=int, default=8)
    ap.add_argument("--eval-seqlen", type=int, default=1024)
    ap.add_argument("--run-name", default="progressive_rwkv")
    ap.add_argument("--out", default="Qwen3.5-9B-RWKV/rwkv_layers_progressive.pt")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16")
    args = ap.parse_args()
    dtype = getattr(torch, args.dtype); dev = args.device

    from transformers import AutoModelForCausalLM
    print("loading teacher (frozen original) ...", flush=True)
    teacher = AutoModelForCausalLM.from_pretrained(args.model_dir, dtype=dtype, low_cpu_mem_usage=True).to(dev).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    t_tm = _text(teacher); t_layers = t_tm.layers
    print("loading student ...", flush=True)
    student = AutoModelForCausalLM.from_pretrained(args.model_dir, dtype=dtype, low_cpu_mem_usage=True).to(dev).eval()
    for p in student.parameters():
        p.requires_grad_(False)
    s_tm = _text(student); s_layers = s_tm.layers
    cfg = student.config.text_config if hasattr(student.config, "text_config") else student.config
    nL = len(t_layers)
    t_norm = getattr(t_tm, "norm", None)                      # final RMSNorm for the early-exit lens
    t_h, s_h, t_blk, s_blk = {}, {}, {}, {}
    for i in range(nL):
        t_layers[i].register_forward_hook(_hook(t_h, i))
        s_layers[i].register_forward_hook(_hook(s_h, i))

    gdn = [i for i, t in enumerate(cfg.layer_types) if t != "full_attention"]
    for i in gdn:                                             # A1: capture teacher GDN readout per layer
        t_layers[i].linear_attn.register_forward_hook(_hook(t_blk, i))
    order = sorted(gdn, reverse=True)
    if args.max_layers:
        order = order[:args.max_layers]
    toks = load_token_stream(args.data)
    # one shared non-repeating stream for ALL layers' distill steps -> no window
    # is ever trained on twice across the whole conversion.
    stream = WindowStream(toks, args.distill_seqlen, seed=0)
    toks_eval = load_token_stream(args.val_data)   # held-out eval, disjoint from training

    # dashboard
    run_dir = Path("runs") / args.run_name; run_dir.mkdir(parents=True, exist_ok=True)
    logf = open(run_dir / "train.jsonl", "a")
    def emit(r): logf.write(json.dumps(r) + "\n"); logf.flush()
    def sidecar(step, conv):
        c = {"model_dir": str(Path(args.model_dir).resolve()), "patch_dir": "",
             "rwkv8_deltanet_layers": ",".join(map(str, sorted(conv))), "rwkv8_swap_mode": "timemix",
             "train_rwkv8_layers": ",".join(map(str, sorted(conv))), "install_mtp": 0,
             "engram_enabled": 0, "freeze_non_mla": 1}
        sd = run_dir / f"step_{step:06d}"; sd.mkdir(parents=True, exist_ok=True)
        (sd / "config.json").write_text(json.dumps({"step": step, "config": c}, indent=2))

    base = str(Path(args.out)); stem = base[:-3] if base.endswith(".pt") else base
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    def _blob(conv):
        return {"layers": {Lc: {k: v.detach().cpu() for k, v in s_layers[Lc].linear_attn.state_dict().items()}
                           for Lc in conv},
                "converted": sorted(conv),
                "loop_count": args.loop_count, "allow_neg_eigval": args.allow_neg_eigval,
                "stream_cursor": int(stream.cursor), "batch": args.batch}

    def save_ckpt(conv, layer_id=None, idx=None):
        # Write the new blob fully to a temp, then atomic-rename into place so a kill
        # mid-save can never truncate the only copy. Keep the prior file as .prev for a
        # one-layer rollback, and a numbered milestone every snapshot_every layers.
        torch.save(_blob(conv), base + ".tmp")
        if os.path.exists(base):
            # hardlink old base -> .prev (instant, same inode) BEFORE the replace, so
            # `base` always exists on disk; a kill can never leave only .prev/.tmp (#12).
            if os.path.exists(base + ".prev"):
                os.remove(base + ".prev")
            os.link(base, base + ".prev")
        os.replace(base + ".tmp", base)
        if layer_id is not None and idx is not None and args.snapshot_every > 0 and idx % args.snapshot_every == 0:
            shutil.copy2(base, f"{stem}.L{layer_id:02d}.pt")

    resuming = bool(args.resume and Path(args.out).exists())
    tp = batched_eval(t_tm, teacher.lm_head, toks_eval, args.eval_windows, args.eval_seqlen, dev, batch=args.io_batch)
    print(f"teacher ppl={tp['ppl']:.3f}", flush=True)
    if not resuming:
        emit({"kind": "eval", "step": 0, "ppl": tp["ppl"], "top1_acc": tp["top1_acc"]})

    converted, gstep = [], 0
    resume_loaded = []
    if resuming:
        blob = torch.load(args.out, map_location="cpu", weights_only=False)
        saved = blob.get("layers", {})
        if blob.get("loop_count", 1) != args.loop_count or bool(blob.get("allow_neg_eigval", False)) != bool(args.allow_neg_eigval):
            print(f"  !! resume mismatch: ckpt loop_count={blob.get('loop_count')} neg={blob.get('allow_neg_eigval')} "
                  f"vs args {args.loop_count}/{args.allow_neg_eigval}; aborting", flush=True)
            sys.exit(1)
        for L in order:                                      # reattach in conversion order, no distill
            if L not in saved:
                continue
            gdn_sd = {k: v.detach().cpu() for k, v in s_layers[L].linear_attn.state_dict().items()}
            core = rwkv8_timemix_from_config(student.config, layer_idx=L, init_from_deltanet=gdn_sd,
                                             depth_n_layer=cfg.num_hidden_layers,
                                             decay_cap_delta=args.decay_cap_delta,
                                             allow_neg_eigval=args.allow_neg_eigval).to(device=dev, dtype=dtype)
            rwkv = LoopedRWKV(core, args.loop_count).to(device=dev, dtype=dtype) if args.loop_count > 1 else core
            rwkv.load_state_dict({k: v.to(device=dev, dtype=dtype) for k, v in saved[L].items()})
            rwkv._save_key = f"rwkv8_layer_{L}"
            setattr(s_layers[L], "linear_attn", rwkv)
            rwkv.register_forward_hook(_hook(s_blk, L))
            converted.append(L); resume_loaded.append(L)
        # Derive prior progress from the log's last eval step, NOT len(resumed)*distill_steps,
        # so --distill-steps may CHANGE between runs without corrupting gstep or the stream
        # position. Each distill iteration advances gstep by 1 and consumes `batch` windows,
        # and the post-distill eval is emitted at that gstep -> last_eval_step == prior gstep
        # == windows_consumed/batch, regardless of how many steps each prior layer used.
        prior_gstep = len(resume_loaded) * args.distill_steps     # fallback if no log
        _logp = run_dir / "train.jsonl"
        if _logp.exists():
            _last = 0
            for _ln in _logp.read_text().splitlines():
                try:
                    _o = json.loads(_ln)
                    if _o.get("kind") == "eval":
                        _last = max(_last, int(_o.get("step", 0)))
                except Exception:
                    pass
            if _last > 0:
                prior_gstep = _last
        gstep = prior_gstep
        # +1 layer guard so a mid-layer kill never re-serves partially-consumed windows
        # (capture uses its own seed and never draws from `stream`); strict no-reuse.
        guard = args.distill_steps * args.batch if resume_loaded else 0
        if isinstance(blob.get("stream_cursor"), int):
            # exact windows-consumed stored in the ckpt -> batch-change safe (#8).
            stream.cursor = blob["stream_cursor"] + guard
        else:
            # legacy ckpt: assume batch unchanged (prior_gstep iterations * batch windows).
            stream.cursor += prior_gstep * args.batch + guard
        print(f"  resumed {len(resume_loaded)} layers {resume_loaded}; gstep->{gstep}, "
              f"stream.cursor->{stream.cursor}/{stream.n_windows}", flush=True)

    for n, L in enumerate(order, 1):
        if L in resume_loaded:                               # already converted in a prior run
            continue
        t0 = time.time()
        gdn_mod = s_layers[L].linear_attn
        gdn_sd = {k: v.detach().cpu() for k, v in gdn_mod.state_dict().items()}
        S, H, Y = capture_layer(s_tm, gdn_mod, toks, args.cap_windows, args.cap_seqlen,
                                args.state_stride, dev, seed=L, batch=args.io_batch)
        core = rwkv8_timemix_from_config(student.config, layer_idx=L, init_from_deltanet=gdn_sd,
                                         depth_n_layer=cfg.num_hidden_layers,
                                         decay_cap_delta=args.decay_cap_delta,
                                         allow_neg_eigval=args.allow_neg_eigval).to(device=dev, dtype=dtype)
        # codec-fit the CORE (single pass) first, then wrap in the N-loop so the
        # zero-init loop preserves the codec init at construction.
        rel = fit_readout(core, S, H, Y, gdn_mod.num_v_heads, gdn_mod.head_k_dim, gdn_mod.head_v_dim,
                          args.codec_steps, args.codec_lr, 512, dev)
        rwkv = LoopedRWKV(core, args.loop_count).to(device=dev, dtype=dtype) if args.loop_count > 1 else core
        rwkv._save_key = f"rwkv8_layer_{L}"
        setattr(s_layers[L], "linear_attn", rwkv)
        rwkv.register_forward_hook(_hook(s_blk, L))           # A1: capture student RWKV readout
        converted.append(L)
        # all converted layers trainable, everything else frozen
        named_params = []
        for p in student.parameters():
            p.requires_grad_(False)
        for Lc in converted:
            for name, p in s_layers[Lc].linear_attn.named_parameters():
                p.requires_grad_(True); named_params.append((f"L{Lc}.{name}", p))
        pre = batched_eval(s_tm, student.lm_head, toks_eval, args.eval_windows, args.eval_seqlen, dev, batch=args.io_batch)
        # marker at the swap step (vertical line); the eval series stays ONE ppl per
        # step (baseline at 0, then each layer's post-distill ppl) so the chart is clean.
        emit({"kind": "checkpoint", "step": gstep})
        gstep, dkl = distill(teacher, t_tm, student, s_tm, student.lm_head, t_norm, named_params, t_h, s_h,
                             t_blk, s_blk, converted, nL, toks, stream, args.distill_steps, args.distill_lr,
                             args.distill_seqlen, dev, emit, gstep, batch=args.batch,
                             w_kl=args.w_kl, w_hidden=args.w_hidden, w_block=args.w_block,
                             w_exit=args.w_exit_kl, exit_stride=args.exit_kl_stride, kl_temp=args.kl_temp,
                             curr_start=args.curriculum_start, skip_spike=args.skip_spike,
                             lr_floor_frac=args.lr_floor_frac,
                             warmup=args.warmup_steps, hold_frac=args.hold_frac,
                             weight_decay=args.weight_decay, lr_decay_mult=args.lr_decay_mult)
        post = batched_eval(s_tm, student.lm_head, toks_eval, args.eval_windows, args.eval_seqlen, dev, batch=args.io_batch)
        _ev = {"kind": "eval", "step": gstep, "ppl": post["ppl"], "top1_acc": post["top1_acc"]}
        _rwm = _loop_rw_mean(s_layers, converted)
        if _rwm is not None:
            _ev["loop_rw"] = _rwm
        emit(_ev)
        sidecar(gstep, converted)   # update arch panel: which layers are now RWKV
        print(f"  [{n:2d}/{len(order)}] L{L:2d}: codec_rel={rel:.3f} ppl {pre['ppl']:.2f}->{post['ppl']:.2f} "
              f"(distill_kl={dkl:.4f}, teacher {tp['ppl']:.2f}, {time.time()-t0:.0f}s)", flush=True)
        # Guarded save: a diverged layer must NOT overwrite the last-good cumulative
        # checkpoint. Publish only if post ppl is finite and within divergence_factor
        # of the baseline; else keep the prior file and halt (downstream layers would
        # just compound the garbage, and the .prev/.L snapshots remain for rollback).
        thresh = max(pre["ppl"], tp["ppl"]) * args.divergence_factor
        if math.isfinite(post["ppl"]) and post["ppl"] <= thresh:
            save_ckpt(converted, layer_id=L, idx=n)
        else:
            print(f"  !! L{L} DIVERGED: post ppl {post['ppl']:.2f} > {thresh:.2f} "
                  f"({args.divergence_factor}x baseline); keeping last-good {args.out}, halting.", flush=True)
            emit({"kind": "checkpoint", "step": gstep})
            break

    emit({"kind": "checkpoint", "step": gstep}); sidecar(gstep, converted)
    print(f"\nDONE: {len(converted)} layers converted; teacher {tp['ppl']:.2f}, final ppl in log.", flush=True)
    print(f"saved -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
