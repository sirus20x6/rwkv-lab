#!/usr/bin/env python
"""Stack GDN->RWKV-7 conversions and watch ppl as the stack grows.

Pure-RWKV track (no MLA). Converts the Gated-DeltaNet (linear_attn) layers of the
base model to RWKV-7 one at a time, BACKWARD (top-down): each layer's input then
depends only on still-original lower layers, so the base teacher stays valid per
layer with no re-extraction. Per layer:
  1. capture teacher (h, GDN boundary states, block_out) just-in-time,
  2. codec-fit the RWKV readout (the init + codec-prefit is what makes the swap
     near-lossless; per the single-layer result, training barely moves ppl),
  3. swap RWKV in, eval held-out ppl on the partially-converted model.

Prints the ppl-vs-#converted curve and saves the accumulated RWKV layer weights.
"""
from __future__ import annotations

import sys
sys.modules.setdefault("torchvision", None)  # torch2.11 vs torchvision::nms

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from rwkv8_deltanet import rwkv8_timemix_from_config
from smt_dmt import BilinearStateCodec, rwkv_readout
from build_memory_targets import _GDNStateCapture, load_token_stream
from convert_train import evaluate


def capture_layer(text_model, gdn_mod, toks, n_windows, seq_len, stride, device, seed, batch=1):
    """Run n_windows (in batches) through the (partially-converted) model and
    gather teacher (S_gdn, h, block_out) at stride boundaries for the GDN layer."""
    cap = _GDNStateCapture(gdn_mod.chunk_gated_delta_rule, stride)
    gdn_mod.chunk_gated_delta_rule = cap
    box = {}
    ph = gdn_mod.register_forward_pre_hook(
        lambda m, a, kw: box.__setitem__("h", (a[0] if a else kw["hidden_states"]).detach()),
        with_kwargs=True)
    hh = gdn_mod.register_forward_hook(lambda m, a, o: box.__setitem__("y", o.detach()))
    N = len(toks); maxs = N - (seq_len + 1)
    rng = np.random.default_rng(seed)
    Ss, Hs, Ys = [], [], []
    with torch.no_grad():
        done = 0
        while done < n_windows:
            b = min(batch, n_windows - done)
            starts = [int(rng.integers(0, maxs + 1)) for _ in range(b)]
            ids = torch.as_tensor(
                np.stack([np.asarray(toks[s:s + seq_len], dtype=np.int64) for s in starts]),
                device=device)
            text_model(input_ids=ids, use_cache=False)
            nb = cap.states.shape[0]
            rpos = [min(j * stride, seq_len) - 1 for j in range(1, nb)]
            for w in range(b):
                Ss.append(cap.states[1:nb, w].float().cpu())
                Hs.append(box["h"][w, rpos].float().cpu())
                Ys.append(box["y"][w, rpos].float().cpu())
            done += b
    ph.remove(); hh.remove()
    gdn_mod.chunk_gated_delta_rule = cap.real_fn
    return torch.cat(Ss), torch.cat(Hs), torch.cat(Ys)


def _capture_layer_old(text_model, gdn_mod, toks, n_windows, seq_len, stride, device, seed):
    cap = _GDNStateCapture(gdn_mod.chunk_gated_delta_rule, stride)
    gdn_mod.chunk_gated_delta_rule = cap
    box = {}
    ph = gdn_mod.register_forward_pre_hook(
        lambda m, a, kw: box.__setitem__("h", (a[0] if a else kw["hidden_states"]).detach()),
        with_kwargs=True)
    hh = gdn_mod.register_forward_hook(lambda m, a, o: box.__setitem__("y", o.detach()))
    N = len(toks); maxs = N - (seq_len + 1)
    rng = np.random.default_rng(seed)
    Ss, Hs, Ys = [], [], []
    with torch.no_grad():
        for _ in range(n_windows):
            s0 = int(rng.integers(0, maxs + 1))
            ids = torch.as_tensor(np.asarray(toks[s0:s0 + seq_len], dtype=np.int64),
                                  device=device).unsqueeze(0)
            text_model(input_ids=ids, use_cache=False)
            S = cap.states[:, 0]                                  # [n_bounds, Hg,Dk,Dv]
            nb = S.shape[0]
            rpos = [min(j * stride, seq_len) - 1 for j in range(1, nb)]
            Ss.append(S[1:nb].float().cpu())
            Hs.append(box["h"][0, rpos].float().cpu())
            Ys.append(box["y"][0, rpos].float().cpu())
    ph.remove(); hh.remove()
    gdn_mod.chunk_gated_delta_rule = cap.real_fn                  # restore
    return torch.cat(Ss), torch.cat(Hs), torch.cat(Ys)


def fit_readout(rwkv, S, H, Y, Hg, Dk, Dv, steps, lr, bs, device):
    """Codec + RWKV-readout fit (in float32), in place on rwkv. Returns final rel_rmse."""
    codec = BilinearStateCodec(gdn_heads=Hg, gdn_dk=Dk, gdn_dv=Dv,
                               rwkv_heads=rwkv.num_heads, rwkv_dk=rwkv.head_size,
                               rwkv_dv=rwkv.head_size).to(device)
    N = S.shape[0]; bs = min(bs, N)
    od = next(rwkv.parameters()).dtype
    rwkv.to(device=device, dtype=torch.float32)
    params = list(codec.parameters()) + [
        rwkv.receptance.weight, rwkv.key.weight, rwkv.value.weight, rwkv.output.weight,
        rwkv.g1, rwkv.g2, rwkv.r_k, rwkv.ln_x.weight, rwkv.ln_x.bias]
    for p in params:
        p.requires_grad_(True)
    opt = torch.optim.Adam(params, lr=lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps, lr * 0.05)
    gen = torch.Generator().manual_seed(0)
    rel = 1.0
    for _ in range(steps):
        idx = torch.randint(0, N, (bs,), generator=gen)
        s, h, y = S[idx].to(device).float(), H[idx].to(device).float(), Y[idx].to(device).float()
        opt.zero_grad()
        loss = F.mse_loss(rwkv_readout(rwkv, codec(s), h), y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step(); sch.step()
        rel = (loss.item() / y.pow(2).mean().item()) ** 0.5
    rwkv.to(dtype=od)
    return rel


def _dash_setup(args, order):
    """Return (emit, write_sidecar) that write the dashboard schema to
    runs/<run_name>/. emit appends train/eval/checkpoint records; sidecar lets
    the architecture panel render the stacked RWKV model. No-ops if no run-name."""
    if not args.run_name:
        return (lambda rec: None), (lambda step: None)
    run_dir = Path("runs") / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    logf = open(run_dir / "train.jsonl", "a")

    def emit(rec):
        logf.write(json.dumps(rec) + "\n"); logf.flush()

    def write_sidecar(step):
        cfg = {"model_dir": str(Path(args.model_dir).resolve()), "patch_dir": "",
               "rwkv8_deltanet_layers": ",".join(str(i) for i in sorted(order)),
               "rwkv8_swap_mode": "timemix",
               "train_rwkv8_layers": ",".join(str(i) for i in sorted(order)),
               "install_mtp": 0, "engram_enabled": 0, "freeze_non_mla": 1}
        sd = run_dir / f"step_{step:06d}"; sd.mkdir(parents=True, exist_ok=True)
        (sd / "config.json").write_text(json.dumps({"step": step, "config": cfg}, indent=2))

    return emit, write_sidecar


def block_mse_consolidate(model, text_model, layers, converted, gdn_teachers, toks, args, emit=None, base_step=0):
    """Dense, local consolidation: each step re-captures every converted layer's
    REAL input in the current stack (no-grad forward), then trains each RWKV layer
    to match its frozen GDN teacher on that input. Input is detached -> per-layer
    independent + stable; re-capturing each step lets downstream layers track
    upstream improvements, which is what cancels the compounding."""
    dev = args.device
    for p in model.parameters():
        p.requires_grad_(False)
    params = []
    for L in converted:
        gdn_teachers[L].eval()
        for p in gdn_teachers[L].parameters():
            p.requires_grad_(False)
        for p in layers[L].linear_attn.parameters():
            p.requires_grad_(True); params.append(p)
    opt = torch.optim.AdamW(params, lr=args.consolidate_lr, betas=(0.9, 0.95))
    # hooks capture each converted layer's input (residual entering linear_attn)
    box = {}
    handles = []
    for L in converted:
        def mk(Li):
            def pre(m, a, kw):
                box[Li] = (a[0] if a else kw["hidden_states"]).detach()
            return pre
        handles.append(layers[L].linear_attn.register_forward_pre_hook(mk(L), with_kwargs=True))
    N = len(toks); T = args.consolidate_seqlen; maxs = N - (T + 1)
    rng = np.random.default_rng(123)
    for step in range(args.consolidate_steps):
        s0 = int(rng.integers(0, maxs + 1))
        x = torch.as_tensor(np.asarray(toks[s0:s0 + T], dtype=np.int64), device=dev).unsqueeze(0)
        with torch.no_grad():
            text_model(input_ids=x, use_cache=False)            # fills box[L] for all L
        opt.zero_grad()
        tot = 0.0
        for L in converted:
            inp = box[L]
            rwkv_out = layers[L].linear_attn(inp)               # grad -> RWKV params
            with torch.no_grad():
                teach = gdn_teachers[L](inp)
            loss = F.mse_loss(rwkv_out.float(), teach.float())
            loss.backward()
            tot += loss.item()
        gn = torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        if emit is not None and step % max(1, args.consolidate_steps // 100) == 0:
            emit({"kind": "train", "step": base_step + step, "loss": tot / len(converted),
                  "gnorm": float(gn), "lr": args.consolidate_lr})
        if step % max(1, args.consolidate_steps // 12) == 0:
            print(f"    [block-mse] step {step} mean_block={tot/len(converted):.4f} gnorm={float(gn):.2f}", flush=True)
    for h in handles:
        h.remove()


def consolidate(model, text_model, layers, converted, toks, args):
    """Jointly LM-CE fine-tune all converted RWKV layers (rest frozen) so they
    co-adapt to each other's real (perturbed) outputs."""
    from convert_train import chunked_ce
    dev = args.device
    for p in model.parameters():
        p.requires_grad_(False)
    params = []
    for L in converted:
        for p in layers[L].linear_attn.parameters():
            p.requires_grad_(True); params.append(p)
    opt = torch.optim.AdamW(params, lr=args.consolidate_lr, betas=(0.9, 0.95))
    N = len(toks); T = args.consolidate_seqlen; maxs = N - (T + 1)
    rng = np.random.default_rng(123)
    for step in range(args.consolidate_steps):
        s0 = int(rng.integers(0, maxs + 1))
        ids = torch.as_tensor(np.asarray(toks[s0:s0 + T + 1], dtype=np.int64), device=dev)
        x, y = ids[:-1].unsqueeze(0), ids[1:].unsqueeze(0)
        out = text_model(input_ids=x, use_cache=False)
        hidden = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
        loss = chunked_ce(hidden, model.lm_head, y)
        opt.zero_grad(); loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        if step % max(1, args.consolidate_steps // 12) == 0:
            print(f"    [consol] step {step} lm_ce={float(loss):.4f} gnorm={float(gn):.2f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="Qwen3.5-9B-Base")
    ap.add_argument("--data", default="/thearray/git/babyllm/data/cache/qwen3.6_fwedu_train")
    ap.add_argument("--out", default="Qwen3.5-9B-RWKV/rwkv_layers.pt")
    ap.add_argument("--max-layers", type=int, default=0, help="convert only the first N (0=all)")
    ap.add_argument("--cap-windows", type=int, default=48)
    ap.add_argument("--cap-seqlen", type=int, default=1024)
    ap.add_argument("--state-stride", type=int, default=64)
    ap.add_argument("--decay-cap-delta", type=float, default=0.005)
    ap.add_argument("--codec-steps", type=int, default=500)
    ap.add_argument("--codec-lr", type=float, default=1e-3)
    ap.add_argument("--codec-bs", type=int, default=512)
    ap.add_argument("--eval-windows", type=int, default=8)
    ap.add_argument("--eval-seqlen", type=int, default=1024)
    ap.add_argument("--load-stack", default="", help="load a saved stack (.pt) and skip conversion")
    ap.add_argument("--consolidate-steps", type=int, default=0,
                    help="after stacking, jointly fine-tune all converted RWKV layers")
    ap.add_argument("--consolidate-method", default="block_mse", choices=["block_mse", "lm_ce"])
    ap.add_argument("--consolidate-lr", type=float, default=1e-3)
    ap.add_argument("--consolidate-seqlen", type=int, default=512)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--run-name", default="", help="if set, write runs/<name>/ in the dashboard schema")
    args = ap.parse_args()
    dtype = getattr(torch, args.dtype)

    from transformers import AutoModelForCausalLM
    print(f"loading base {args.model_dir} ...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(args.model_dir, dtype=dtype,
                                                 low_cpu_mem_usage=True).to(args.device).eval()
    text_model = getattr(model.model, "language_model", model.model)
    layers = text_model.layers
    cfg = model.config.text_config if hasattr(model.config, "text_config") else model.config
    lt = cfg.layer_types
    gdn = [i for i, t in enumerate(lt) if t != "full_attention"]
    order = sorted(gdn, reverse=True)                            # backward (top-down)
    if args.max_layers:
        order = order[:args.max_layers]
    toks = load_token_stream(args.data)

    base = evaluate(text_model, model.lm_head, toks, args.eval_windows, args.eval_seqlen, args.device)
    print(f"BASELINE (all GDN): ppl={base['ppl']:.3f} top1={base['top1_acc']:.3f}", flush=True)
    emit, write_sidecar = _dash_setup(args, order)
    emit({"kind": "eval", "step": 0, "ppl": base["ppl"], "top1_acc": base["top1_acc"]})

    accum, curve, converted, gdn_teachers = {}, [(0, base["ppl"], base["top1_acc"])], [], {}
    if args.load_stack:
        ck = torch.load(args.load_stack, map_location="cpu")
        for L, sd in ck["layers"].items():
            rwkv = rwkv8_timemix_from_config(model.config, layer_idx=L,
                                             depth_n_layer=cfg.num_hidden_layers,
                                             decay_cap_delta=args.decay_cap_delta)
            rwkv.load_state_dict(sd)
            rwkv = rwkv.to(device=args.device, dtype=dtype)
            rwkv._save_key = f"rwkv8_layer_{L}"
            gdn_teachers[L] = layers[L].linear_attn          # keep original GDN as teacher
            setattr(layers[L], "linear_attn", rwkv)
            converted.append(L); accum[L] = sd
        ev = evaluate(text_model, model.lm_head, toks, args.eval_windows, args.eval_seqlen, args.device)
        curve.append((len(converted), ev["ppl"], ev["top1_acc"]))
        emit({"kind": "eval", "step": len(converted), "ppl": ev["ppl"], "top1_acc": ev["top1_acc"]})
        print(f"loaded {len(converted)} layers -> ppl={ev['ppl']:.3f} top1={ev['top1_acc']:.3f}", flush=True)
    else:
        for n, L in enumerate(order, 1):
            t0 = time.time()
            dl = layers[L]; gdn_mod = dl.linear_attn
            gdn_sd = {k: v.detach().cpu() for k, v in gdn_mod.state_dict().items()}
            S, H, Y = capture_layer(text_model, gdn_mod, toks, args.cap_windows,
                                    args.cap_seqlen, args.state_stride, args.device, seed=L)
            rwkv = rwkv8_timemix_from_config(
                model.config, layer_idx=L, init_from_deltanet=gdn_sd,
                depth_n_layer=cfg.num_hidden_layers, decay_cap_delta=args.decay_cap_delta,
            ).to(device=args.device, dtype=dtype)
            rel = fit_readout(rwkv, S, H, Y, gdn_mod.num_v_heads, gdn_mod.head_k_dim,
                              gdn_mod.head_v_dim, args.codec_steps, args.codec_lr, args.codec_bs, args.device)
            rwkv._save_key = f"rwkv8_layer_{L}"
            gdn_teachers[L] = gdn_mod                         # keep original GDN as teacher
            setattr(dl, "linear_attn", rwkv)
            ev = evaluate(text_model, model.lm_head, toks, args.eval_windows, args.eval_seqlen, args.device)
            accum[L] = {k: v.detach().cpu() for k, v in rwkv.state_dict().items()}
            converted.append(L)
            curve.append((n, ev["ppl"], ev["top1_acc"]))
            emit({"kind": "eval", "step": n, "ppl": ev["ppl"], "top1_acc": ev["top1_acc"]})
            print(f"  [{n:2d}/{len(order)}] L{L:2d} -> ppl={ev['ppl']:.3f} top1={ev['top1_acc']:.3f} "
                  f"(codec_rel={rel:.3f}, {time.time()-t0:.0f}s)", flush=True)

    if args.consolidate_steps > 0 and converted:
        print(f"consolidating {len(converted)} layers via {args.consolidate_method}, "
              f"{args.consolidate_steps} steps ...", flush=True)
        bstep = len(converted)
        if args.consolidate_method == "block_mse":
            block_mse_consolidate(model, text_model, layers, converted, gdn_teachers, toks, args,
                                  emit=emit, base_step=bstep)
        else:
            consolidate(model, text_model, layers, converted, toks, args)
        ev = evaluate(text_model, model.lm_head, toks, args.eval_windows, args.eval_seqlen, args.device)
        curve.append((len(converted), ev["ppl"], ev["top1_acc"]))
        emit({"kind": "eval", "step": bstep + args.consolidate_steps, "ppl": ev["ppl"], "top1_acc": ev["top1_acc"]})
        emit({"kind": "checkpoint", "step": bstep + args.consolidate_steps})
        print(f"AFTER CONSOLIDATION ({len(converted)} layers) -> ppl={ev['ppl']:.3f} top1={ev['top1_acc']:.3f}", flush=True)
        for L in converted:
            accum[L] = {k: v.detach().cpu() for k, v in layers[L].linear_attn.state_dict().items()}

    write_sidecar(len(converted) + (args.consolidate_steps if args.consolidate_steps > 0 and converted else 0))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"layers": accum, "order": order, "curve": curve, "base_ppl": base["ppl"]}, args.out)
    print("\n=== ppl vs #converted ===", flush=True)
    for n, p, t in curve:
        print(f"  {n:2d} layers: ppl={p:.3f} top1={t:.3f}", flush=True)
    print(f"saved -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
