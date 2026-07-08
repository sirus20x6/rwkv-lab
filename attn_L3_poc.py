"""L3 attention -> RWKV-7 conversion proof-of-concept (RADLADS, freeze-most).

Two-stage recipe: Stage 1 = block-align (block-relative MSE, freeze-most) — how far the linear kernel
gets by matching the attention block's *output*. Stage 2 (--logit-kl) adds the RADLADS Step-2 this PoC
used to defer: top-k logit self-distillation with the converted core swapped into the full model, KL'd
against the base model's own top-k logits. The freeze-then-self-distill schedule and the top-k-logit
self-distillation are the portable ideas from arXiv:2605.16928 ("Full Attention Strikes Back") — note
that paper is about SPARSE-SOFTMAX attention, NOT linear/RWKV, so it offers no linearization recipe;
these two training-schedule pieces are the only transferable parts. Block-MSE alone floors ~0.234 (the
RoPE + per-head q/k-norm the linear kernel can't reproduce); the logit-KL stage is the lever to test
whether self-distillation on the model's own outputs recovers what block-averaged MSE leaves on the table.


Converts ONE full-attention layer (default index 3) of Qwen3.5-9B into an RWKV-7
core, RADLADS-style:
  * weight transfer: q_proj->receptance, k_proj->key (GQA 4->16 expand),
    v_proj->value (GQA expand), o_proj->output;
  * FREEZE-MOST: freeze the transferred projections, train ONLY the RWKV time-part
    (decay w-LoRA, ICLR a-LoRA, v/g-LoRA, k_k/k_a/r_k, GroupNorm, token-shift);
  * Step-1 hidden-state alignment: match the frozen attention layer's block output
    (input->output cached from the frozen base), per-token relative-L2 (OpenMOSE rel).

Not faithful yet: the RWKV core has NO RoPE (RAD-RWKV7 adds it for attention layers);
this PoC measures how far freeze-most gets WITHOUT it, so we know if RoPE / unfreezing
projections is the next lever. Emits runs/<out>/train.jsonl in the trainboard schema.

  python attn_L3_poc.py --out runs/attn_L3_poc --steps 3000
"""
from __future__ import annotations
import sys
sys.modules.setdefault("torchvision", None)
import os, math, json, time, argparse, torch, torch.nn as nn, torch.nn.functional as F, numpy as np
sys.path.insert(0, "/thearray/git/moe-mla")
from transformers import AutoModelForCausalLM
from build_memory_targets import load_token_stream
from rwkv8_deltanet import RWKV8TimeMixDeltaNet

MODEL = "/thearray/git/moe-mla/Qwen3.5-9B-Base"
DATA = "/thearray/git/babyllm/data/cache/qwen3.6_fwedu_train"
PROJ = ("receptance", "key", "value", "output")   # RADLADS-transferred (frozen in freeze-most)


def rel_loss(student, teacher):
    """OpenMOSE per-token relative L2 norm."""
    diff = torch.linalg.vector_norm(teacher.float() - student.float(), dim=-1)
    tgt = torch.linalg.vector_norm(teacher.float(), dim=-1) + 1e-8
    return (diff / tgt).mean()


def radlads_init(core, attn, n_q, n_kv, hd):
    """RADLADS transfer. Qwen3.5 attention is GATED: q_proj outputs per-head
    [query|gate] (view [...,n_q,hd*2] then chunk 2). Take the query half per head
    for receptance; k/v are GQA (n_kv heads) -> expand to n_q. q_norm/k_norm, RoPE
    and the sigmoid gate are NOT transferred (RWKV time-part learns/absorbs them)."""
    rep = n_q // n_kv
    with torch.no_grad():
        qw = attn.q_proj.weight                                      # [n_q*hd*2, C]
        C = qw.shape[1]
        qw = qw.view(n_q, hd * 2, C)[:, :hd, :].reshape(n_q * hd, C)  # per-head query slice -> [n_q*hd, C]
        core.receptance.weight.copy_(qw.to(core.receptance.weight.dtype))
        core.output.weight.copy_(attn.o_proj.weight.to(core.output.weight.dtype))
        for src_name, dst in (("k_proj", core.key), ("v_proj", core.value)):
            w = getattr(attn, src_name).weight                       # [n_kv*hd, C]
            w = w.view(n_kv, hd, C).repeat_interleave(rep, dim=0).reshape(n_q * hd, C)  # -> [n_q*hd, C]
            dst.weight.copy_(w.to(dst.weight.dtype))
    return {"receptance<-q_proj[query]", "output<-o_proj", "key<-k_proj(x%d)" % rep, "value<-v_proj(x%d)" % rep}


def set_freeze_most(core):
    """Freeze the RADLADS-transferred projections; train the RWKV time-part."""
    n_train, n_froze = 0, 0
    for name, p in core.named_parameters():
        proj = any(name == f"{g}.weight" or name.startswith(f"{g}.") for g in PROJ)
        p.requires_grad_(not proj)
        if proj:
            n_froze += p.numel()
        else:
            n_train += p.numel()
    return n_train, n_froze


class _RWKVAttnAdapter(nn.Module):
    """Wrap the RWKV core to match Qwen3Attention's call signature for ppl eval."""
    def __init__(self, core):
        super().__init__(); self.core = core

    def forward(self, hidden_states, position_embeddings=None, attention_mask=None,
                past_key_value=None, cache_position=None, **kw):
        out = self.core(hidden_states)
        if isinstance(out, tuple):
            out = out[0]
        return out, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=3)
    ap.add_argument("--out", default="runs/attn_L3_poc")
    ap.add_argument("--train-windows", type=int, default=256)
    ap.add_argument("--val-windows", type=int, default=32)
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lr-final", type=float, default=1e-5)     # RADLADS cosine 1e-3 -> 1e-5
    ap.add_argument("--freeze", choices=["most", "all"], default="most")
    ap.add_argument("--rope", type=int, default=0,
                    help="RAD-RWKV7: graft the teacher's RoPE onto r/k (the untransferred gap). 0=off.")
    ap.add_argument("--taylor", type=int, default=0,
                    help="Taylor-Calibrate (2606.16429, adapted): init per-head decay half-life from the "
                         "teacher's attention look-back distance. Needs eager attention. 0=off.")
    ap.add_argument("--comba", type=int, default=0,
                    help="Comba (2506.02475): decouple the delta-rule removal strength from the write "
                         "(removal weaker than write). Output-correction is already on via out_correct_d. 0=off.")
    # --- Two-stage self-distillation (the RADLADS Step-2 this PoC used to defer). Stage 1 = block-align
    #     (block-rel MSE, freeze-most); Stage 2 adds top-k logit-KL against the base model with the core
    #     swapped into the full stack. The freeze-then-self-distill schedule + top-k-logit self-distill
    #     are the portable pieces of 2605.16928 (a sparse-softmax paper, NOT a linearization recipe). ---
    ap.add_argument("--logit-kl", type=float, default=0.0,
                    help="weight for top-k logit-KL self-distillation vs the base model (0=off, block-MSE only).")
    ap.add_argument("--logit-topk", type=int, default=16,
                    help="top-k for the logit-KL (paper uses top-10 self-distill). Must be >0 when --logit-kl>0.")
    ap.add_argument("--stage1-frac", type=float, default=0.5,
                    help="fraction of steps as pure block-align warmup before logit-KL turns on (two-stage).")
    ap.add_argument("--logit-lr", type=float, default=3e-6,
                    help="LR during the logit-KL self-distill stage (paper Stage-2: 3e-6).")
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--ppl-every", type=int, default=500)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--ppl-windows", type=int, default=16)
    args = ap.parse_args()

    dev, dtype = "cuda", torch.bfloat16
    L = args.layer
    os.makedirs(args.out, exist_ok=True)
    jl = open(os.path.join(args.out, "train.jsonl"), "w", buffering=1)
    def emit(rec): jl.write(json.dumps(rec) + "\n")

    print(f"loading {MODEL} ...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=dtype, low_cpu_mem_usage=True).to(dev).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    layers = model.model.layers
    attn = layers[L].self_attn
    assert attn.__class__.__name__.endswith("Attention"), f"L{L} is {attn.__class__.__name__}, not attention"
    C = model.config.hidden_size
    n_q = model.config.num_attention_heads
    n_kv = model.config.num_key_value_heads
    hd = getattr(model.config, "head_dim", C // n_q)
    print(f"L{L} attention: C={C} n_q={n_q} n_kv={n_kv} head_dim={hd}", flush=True)

    toks = load_token_stream(DATA)
    T = args.seq_len
    rng = np.random.default_rng(0)
    n_all = args.train_windows + args.val_windows
    starts = rng.integers(0, len(toks) - (T + 1), size=n_all)

    # ---- Phase 1: cache the frozen attention layer's block I/O (teacher) ----
    print(f"caching teacher L{L} I/O over {n_all} windows ...", flush=True)
    cap = {}
    def pre_hook(mod, a, kw):
        cap["x"] = (a[0] if a else kw["hidden_states"]).detach()
    def post_hook(mod, a, kw, out):
        cap["y"] = (out[0] if isinstance(out, tuple) else out).detach()
    h1 = attn.register_forward_pre_hook(pre_hook, with_kwargs=True)
    h2 = attn.register_forward_hook(post_hook, with_kwargs=True)
    ntr = args.train_windows
    kl_on = args.logit_kl > 0.0
    if kl_on:
        assert args.logit_topk > 0, "--logit-topk must be >0 when --logit-kl>0 (full-vocab KL isn't cached)"
    xs, ys = [], []
    win_ids, tk_v, tk_i = [], [], []                            # for logit-KL: window token ids + teacher top-k
    for i, s in enumerate(starts):
        w = np.asarray(toks[s:s + T], dtype=np.int64)
        ids = torch.as_tensor(w, device=dev).unsqueeze(0)
        with torch.no_grad():
            out = model(input_ids=ids, use_cache=False)
            if kl_on and i < ntr:                              # cache teacher top-k logits over train windows
                v, idx = out.logits[0].float().topk(args.logit_topk, dim=-1)   # [T, k]
                win_ids.append(torch.as_tensor(w)); tk_v.append(v.cpu()); tk_i.append(idx.cpu())
            del out                                            # free the ~0.5GiB logits ModelOutput each window
        xs.append(cap["x"][0].to("cpu")); ys.append(cap["y"][0].to("cpu"))
    h1.remove(); h2.remove()
    X = torch.stack(xs); Y = torch.stack(ys)                    # [n_all, T, C] cpu bf16
    if kl_on:
        WIN = torch.stack(win_ids); TKv = torch.stack(tk_v); TKi = torch.stack(tk_i)   # [ntr,T] / [ntr,T,k]
        print(f"logit-KL: cached teacher top-{args.logit_topk} over {ntr} windows", flush=True)
    Xtr, Ytr, Xva, Yva = X[:ntr], Y[:ntr], X[ntr:], Y[ntr:]
    base_block = rel_loss(Xva.to(dev), Yva.to(dev)).item()      # identity baseline (how big is the attn residual)
    print(f"cached: train {tuple(Xtr.shape)} val {tuple(Xva.shape)}  (identity rel {base_block:.4f})", flush=True)

    # ---- Phase 2: build RWKV core, RADLADS init, freeze-most ----
    # RAD-RWKV7: mirror the teacher's RoPE (Qwen3.5 attention uses partial rotary + theta).
    tcfg = getattr(model.config, "text_config", model.config)
    rope_theta = float(getattr(tcfg, "rope_theta", 1e7))
    rope_frac = float(getattr(tcfg, "partial_rotary_factor", 0.25))
    core = RWKV8TimeMixDeltaNet(C, num_heads=n_q, head_size=hd, layer_idx=L,
                                depth_layer_id=L, depth_n_layer=model.config.num_hidden_layers,
                                use_rope=bool(args.rope), rope_theta=rope_theta,
                                rope_frac=rope_frac,
                                comba_decouple=bool(args.comba)).to(dev, dtype)
    if args.rope:
        print(f"RAD-RWKV7 RoPE: on (theta={rope_theta:g}, frac={rope_frac:g}, rope_dim={core.rope_dim})", flush=True)
    filled = radlads_init(core, attn, n_q, n_kv, hd)
    print("RADLADS init:", filled, flush=True)
    if args.taylor:
        # Taylor-Calibrate (2606.16429, adapted): set the per-head decay half-life from the
        # teacher head's average attention look-back distance. Needs teacher attention probs,
        # so require eager attention; graceful fallback if the impl doesn't expose them.
        import taylor_calibrate as tc
        nb = min(8, len(starts)); aps = []
        for s in starts[:nb]:
            ids = torch.as_tensor(np.asarray(toks[s:s + T], dtype=np.int64), device=dev).unsqueeze(0)
            with torch.no_grad():
                o = model(input_ids=ids, use_cache=False, output_attentions=True)
            at = getattr(o, "attentions", None)
            if at is not None and at[L] is not None:
                aps.append(at[L].detach().float().cpu())
        if aps:
            d_h = tc.teacher_lookback_distance(torch.cat(aps, dim=0))
            tc.set_halflife_decay(core, d_h)
            print(f"Taylor-Calibrate: half-life decay set from teacher look-back "
                  f"(d_h mean {float(d_h.mean()):.1f} / min {float(d_h.min()):.1f} / max {float(d_h.max()):.1f})",
                  flush=True)
        else:
            print("Taylor-Calibrate: teacher attn probs unavailable (SDPA/flash) — load the model with "
                  "attn_implementation='eager' to enable half-life init; skipping", flush=True)
    if args.freeze == "most":
        n_train, n_froze = set_freeze_most(core)
    else:
        n_train = sum(p.numel() for p in core.parameters()); n_froze = 0
        for p in core.parameters(): p.requires_grad_(True)
    print(f"freeze={args.freeze}: trainable={n_train/1e6:.3f}M frozen={n_froze/1e6:.3f}M", flush=True)

    json.dump({"loop_count": 1, "n_layers": 1, "layer": L, "mode": f"attn-radlads-{args.freeze}",
               "trainable_M": round(n_train / 1e6, 3), "identity_rel": round(base_block, 4)},
              open(os.path.join(args.out, "loop_rw.json"), "w"))

    opt = torch.optim.AdamW([p for p in core.parameters() if p.requires_grad],
                            lr=args.lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0)

    def val_block():
        core.eval()
        with torch.no_grad():
            tot, nb = 0.0, 0
            for i in range(0, Xva.shape[0], args.batch):
                xb = Xva[i:i + args.batch].to(dev); yb = Yva[i:i + args.batch].to(dev)
                o = core(xb); o = o[0] if isinstance(o, tuple) else o
                tot += rel_loss(o, yb).item(); nb += 1
        core.train()
        return tot / max(nb, 1)

    def full_ppl():
        """Swap the trained core into L{L}.self_attn and eval whole-model ppl."""
        core.eval()
        orig = layers[L].self_attn
        layers[L].self_attn = _RWKVAttnAdapter(core)
        try:
            r2 = np.random.default_rng(1234)
            tot_loss, tot_tok = 0.0, 0
            for _ in range(args.ppl_windows):
                s = int(r2.integers(0, len(toks) - (T + 1)))
                ids = torch.as_tensor(np.asarray(toks[s:s + T + 1], dtype=np.int64), device=dev).unsqueeze(0)
                with torch.no_grad():
                    logits = model(input_ids=ids[:, :T], use_cache=False).logits.float()
                tot_loss += F.cross_entropy(logits[0], ids[0, 1:T + 1], reduction="sum").item()
                tot_tok += T
        finally:
            layers[L].self_attn = orig
            core.train()
        return math.exp(tot_loss / tot_tok)

    # ---- Phase 3: two-stage train. Stage 1 = block-align (freeze-most block-rel MSE); Stage 2 adds
    #      top-k logit-KL self-distillation with the core swapped into the full stack. ----
    stage1_steps = int(args.steps * args.stage1_frac) if kl_on else args.steps

    def logit_kl_step():
        """Top-k logit-KL: swap the core into L{L}, forward cached windows, KL vs the base model's
        cached top-k teacher logits. Grad flows only to the trainable (unfrozen) core params."""
        bi = rng.choice(ntr, size=args.batch, replace=False)
        ids = WIN[bi].to(dev)                                    # [b, T]
        ti = TKi[bi].to(dev); tv = TKv[bi].to(dev)              # [b, T, k] teacher top-k idx / logits
        orig = layers[L].self_attn
        layers[L].self_attn = _RWKVAttnAdapter(core)             # grad-enabled: core is trainable
        try:
            slog = model(input_ids=ids, use_cache=False).logits.float()   # [b, T, V]
        finally:
            layers[L].self_attn = orig
        s_at = slog.gather(-1, ti)                               # student logits at teacher's top-k tokens
        tp = tv.softmax(-1)                                      # KL( teacher_topk || student_topk )
        return (tp * (tp.clamp_min(1e-9).log() - s_at.log_softmax(-1))).sum(-1).mean()

    core.train()
    t0 = time.time(); seen = 0
    idx = np.arange(ntr)
    print(f"start training: stage1(block-align)={stage1_steps} steps"
          + (f", stage2(logit-KL top-{args.logit_topk}, w={args.logit_kl})={args.steps - stage1_steps}"
             if kl_on else "") + " ...", flush=True)
    for step in range(args.steps + 1):
        frac = step / max(args.steps, 1)
        in_stage2 = kl_on and step >= stage1_steps
        lr = args.logit_lr if in_stage2 else \
            args.lr_final + 0.5 * (args.lr - args.lr_final) * (1 + math.cos(math.pi * frac))
        for g in opt.param_groups: g["lr"] = lr

        if step % args.eval_every == 0:
            vb = val_block()
            rec = {"kind": "eval", "step": step, "block_val": vb, "loss": vb}
            if kl_on: rec["stage"] = 2 if in_stage2 else 1    # extra key only when the two-stage path is active
            if step % args.ppl_every == 0:
                rec["ppl"] = full_ppl()
            emit(rec)
            msg = f"[{step}]" + (f" s{rec['stage']}" if kl_on else "") + f" val_block={vb:.4f}" \
                + (f" ppl={rec['ppl']:.4f}" if "ppl" in rec else "")
            print(msg, flush=True)

        if step == args.steps:
            break
        bi = rng.choice(idx, size=args.batch, replace=False)
        xb = Xtr[bi].to(dev); yb = Ytr[bi].to(dev)
        o = core(xb); o = o[0] if isinstance(o, tuple) else o
        block = rel_loss(o, yb)                                  # block-rel MSE anchors both stages
        kl = logit_kl_step() if in_stage2 else None
        loss = block + (args.logit_kl * kl if kl is not None else 0.0)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_([p for p in core.parameters() if p.requires_grad], 1.0)
        opt.step()
        seen += xb.shape[0] * T
        if step % args.log_every == 0:
            rec = {"kind": "train", "step": step, "loss": float(loss.detach()), "block": float(block.detach()),
                   "lr": lr, "gnorm": float(gn), "tok_per_sec": int(seen / max(time.time() - t0, 1e-6))}
            if kl is not None: rec["logit_kl"] = float(kl.detach())
            emit(rec)

    torch.save({"core": core.state_dict(), "layer": L, "freeze": args.freeze,
                "num_heads": n_q, "head_size": hd}, os.path.join(args.out, "core_final.pt"))
    emit({"kind": "checkpoint", "step": args.steps})
    print("done -> " + os.path.join(args.out, "core_final.pt"), flush=True)


if __name__ == "__main__":
    main()
