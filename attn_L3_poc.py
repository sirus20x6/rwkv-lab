"""L3 attention -> RWKV-7 conversion proof-of-concept (RADLADS Step-1, freeze-most).

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
    xs, ys = [], []
    for i, s in enumerate(starts):
        ids = torch.as_tensor(np.asarray(toks[s:s + T], dtype=np.int64), device=dev).unsqueeze(0)
        with torch.no_grad():
            model(input_ids=ids, use_cache=False)
        xs.append(cap["x"][0].to("cpu")); ys.append(cap["y"][0].to("cpu"))
    h1.remove(); h2.remove()
    X = torch.stack(xs); Y = torch.stack(ys)                    # [n_all, T, C] cpu bf16
    ntr = args.train_windows
    Xtr, Ytr, Xva, Yva = X[:ntr], Y[:ntr], X[ntr:], Y[ntr:]
    base_block = rel_loss(Xva.to(dev), Yva.to(dev)).item()      # identity baseline (how big is the attn residual)
    print(f"cached: train {tuple(Xtr.shape)} val {tuple(Xva.shape)}  (identity rel {base_block:.4f})", flush=True)

    # ---- Phase 2: build RWKV core, RADLADS init, freeze-most ----
    core = RWKV8TimeMixDeltaNet(C, num_heads=n_q, head_size=hd, layer_idx=L,
                                depth_layer_id=L, depth_n_layer=model.config.num_hidden_layers).to(dev, dtype)
    filled = radlads_init(core, attn, n_q, n_kv, hd)
    print("RADLADS init:", filled, flush=True)
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

    # ---- Phase 3: train (block-align), log to dashboard ----
    core.train()
    t0 = time.time(); seen = 0
    idx = np.arange(ntr)
    print("start training (block-align, freeze-most) ...", flush=True)
    for step in range(args.steps + 1):
        frac = step / max(args.steps, 1)
        lr = args.lr_final + 0.5 * (args.lr - args.lr_final) * (1 + math.cos(math.pi * frac))
        for g in opt.param_groups: g["lr"] = lr

        if step % args.eval_every == 0:
            vb = val_block()
            rec = {"kind": "eval", "step": step, "block_val": vb, "loss": vb}
            if step % args.ppl_every == 0:
                rec["ppl"] = full_ppl()
            emit(rec)
            msg = f"[{step}] val_block={vb:.4f}" + (f" ppl={rec['ppl']:.4f}" if "ppl" in rec else "")
            print(msg, flush=True)

        if step == args.steps:
            break
        bi = rng.choice(idx, size=args.batch, replace=False)
        xb = Xtr[bi].to(dev); yb = Ytr[bi].to(dev)
        o = core(xb); o = o[0] if isinstance(o, tuple) else o
        block = rel_loss(o, yb)
        blk = float(block.detach())
        opt.zero_grad(set_to_none=True)
        block.backward()
        gn = torch.nn.utils.clip_grad_norm_([p for p in core.parameters() if p.requires_grad], 1.0)
        opt.step()
        seen += xb.shape[0] * T
        if step % args.log_every == 0:
            emit({"kind": "train", "step": step, "loss": blk, "block": blk,
                  "lr": lr, "gnorm": float(gn), "tok_per_sec": int(seen / max(time.time() - t0, 1e-6))})

    torch.save({"core": core.state_dict(), "layer": L, "freeze": args.freeze,
                "num_heads": n_q, "head_size": hd}, os.path.join(args.out, "core_final.pt"))
    emit({"kind": "checkpoint", "step": args.steps})
    print("done -> " + os.path.join(args.out, "core_final.pt"), flush=True)


if __name__ == "__main__":
    main()
