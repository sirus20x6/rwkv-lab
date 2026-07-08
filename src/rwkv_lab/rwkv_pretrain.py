"""From-scratch small RWKV-7 pretraining with RWKV-Lab loop / recurrent-depth levers.

Builds a small RWKV-7 LM entirely from OUR modules (RWKV8TimeMixDeltaNet + RWKV8ChannelMix
+ emb/head), so the loop levers (LoopedRWKV) attach NATIVELY — no fla, no g1g remap, no
forward-reconciliation. Trains from random init on a World-tokenized stream (ztok), so there
is real loss headroom: recurrent depth / hyper-connections / CART / DEQ can actually help,
and a fixed-wall-clock A/B (~10 min) measures whether the extra compute-per-token pays off.

Each block's time-mix is optionally wrapped in LoopedRWKV(core, ...). is_first_rwkv_layer=True
on every layer (self-consistent from-scratch architecture; the cross-layer v_first residual is
a separate EXP-C concern and off here). Logs to a trainboard train.jsonl.

    python -m rwkv_lab.rwkv_pretrain --data models/g1g_tokens_big.bin --minutes 10 \
        --d-model 512 --n-layers 6 --loop-count 3 --loop-hyper 2 --out runs/loop_c3h2
"""
from __future__ import annotations
import argparse, json, math, os, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from rwkv_lab.rwkv8_deltanet import RWKV8TimeMixDeltaNet, RWKV8ChannelMixDeltaNet
from rwkv_lab.looped_rwkv import LoopedRWKV
from rwkv_lab.lookahead_module import LookaheadSystem


def _unwrap(o):
    return o[0] if isinstance(o, tuple) else o


class Block(nn.Module):
    def __init__(self, d, n_heads, head_size, i, n_layers, loop_kw):
        super().__init__()
        self.i = i
        if i == 0:
            self.ln0 = nn.LayerNorm(d)
        self.ln1 = nn.LayerNorm(d)
        self.ln2 = nn.LayerNorm(d)
        core = RWKV8TimeMixDeltaNet(d, num_heads=n_heads, head_size=head_size, layer_idx=i,
                                    depth_layer_id=i, depth_n_layer=max(n_layers, 2),
                                    is_first_rwkv_layer=(i == 0),   # native cross-layer v-residual
                                    out_correct=False)              # clean native g070
        self.att = LoopedRWKV(core, hidden_size=d, **loop_kw) if loop_kw else core
        self.ffn = RWKV8ChannelMixDeltaNet(d, layer_idx=i)

    def forward(self, x, v_first):
        if self.i == 0:
            x = self.ln0(x)
        a, v_first = self.att(self.ln1(x), v_first=v_first, return_v_first=True)
        x = x + a
        x = x + _unwrap(self.ffn(self.ln2(x)))
        return x, v_first


class RWKV7Small(nn.Module):
    def __init__(self, vocab, d, n_layers, head_size, loop_kw):
        super().__init__()
        assert d % head_size == 0
        self.emb = nn.Embedding(vocab, d)
        self.blocks = nn.ModuleList([Block(d, d // head_size, head_size, i, n_layers, loop_kw)
                                     for i in range(n_layers)])
        self.ln_out = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)
        self.apply(self._init)

    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, ids, return_hidden=False):
        x = self.emb(ids)
        v_first = None                           # layer 0 sets it; later layers lerp toward it
        for b in self.blocks:
            x, v_first = b(x, v_first)
        h = self.ln_out(x)                       # post-norm final hidden (what aux heads read)
        logits = self.head(h)
        return (logits, h) if return_hidden else logits


def loop_kwargs(a):
    """Map --loop-* flags to LoopedRWKV kwargs. Empty dict => bare core (no loop wrapper)."""
    any_on = a.loop_count > 1 or a.loop_hyper or a.loop_cart_anchor or a.loop_deq \
        or a.loop_fp_halt or a.loop_adaptive_halt or a.loop_iter_readout
    if not any_on:
        return {}
    return dict(n_loops=max(a.loop_count, 2), hyper_lanes=a.loop_hyper,
                gate_mode=a.loop_gate, gate_cap=a.loop_gate_cap,
                cart_anchor=bool(a.loop_cart_anchor), loop_deq=bool(a.loop_deq),
                deq_window=a.loop_deq_window, fixed_point_halt=bool(a.loop_fp_halt),
                adaptive_halt=bool(a.loop_adaptive_halt))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True); ap.add_argument("--out", default="runs/rwkv_scratch")
    ap.add_argument("--d-model", type=int, default=512); ap.add_argument("--n-layers", type=int, default=6)
    ap.add_argument("--head-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=6e-4); ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--batch", type=int, default=16); ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--minutes", type=float, default=10.0); ap.add_argument("--steps", type=int, default=0)
    ap.add_argument("--val-windows", type=int, default=40); ap.add_argument("--eval-every", type=int, default=50)
    ap.add_argument("--log-every", type=int, default=10); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--warmup", type=int, default=100)
    # loop levers
    ap.add_argument("--loop-count", type=int, default=1)
    ap.add_argument("--loop-hyper", type=int, default=0)
    ap.add_argument("--loop-gate", default="scalar", choices=["scalar", "head", "channel", "factored"])
    ap.add_argument("--loop-gate-cap", type=float, default=0.0)
    ap.add_argument("--loop-deq-window", type=int, default=1)
    for f in ["loop-cart-anchor", "loop-deq", "loop-fp-halt", "loop-adaptive-halt", "loop-iter-readout"]:
        ap.add_argument(f"--{f}", type=int, default=0)
    # latent-prediction / lookahead aux objectives (aux head on the final hidden; LM head unchanged)
    for f in ["nextlat-weight", "top-weight", "lmtp-weight", "bst-weight", "jtp-weight"]:
        ap.add_argument(f"--{f}", type=float, default=0.0)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    jl = open(os.path.join(args.out, "train.jsonl"), "w", buffering=1)
    emit = lambda r: jl.write(json.dumps(r) + "\n")
    dev = "cuda"; T = args.seq_len
    torch.manual_seed(args.seed); rng = np.random.default_rng(args.seed)

    lk = loop_kwargs(args)
    model = RWKV7Small(65536, args.d_model, args.n_layers, args.head_size, lk).to(dev, torch.bfloat16)
    nparam = sum(p.numel() for p in model.parameters())
    tag = f"scratch-L{args.n_layers}d{args.d_model}-loop{args.loop_count}" + \
          ("".join(k for k, v in [("H", args.loop_hyper), ("C", args.loop_cart_anchor),
           ("Q", args.loop_deq), ("F", args.loop_fp_halt), ("A", args.loop_adaptive_halt),
           ("R", args.loop_iter_readout)] if v) or "") + \
          (f"-{args.loop_gate}" if lk and args.loop_gate != "scalar" else "")
    print(f"model {tag}: {nparam/1e6:.1f}M params  loop_kw={lk}", flush=True)
    json.dump({"loop_count": args.loop_count, "n_layers": args.n_layers, "mode": tag,
               "params_m": round(nparam / 1e6, 2)}, open(os.path.join(args.out, "loop_rw.json"), "w"))

    toks = np.memmap(args.data, dtype=np.uint16, mode="r")
    n_val = args.val_windows * T
    val_toks, train_toks = toks[:n_val], toks[n_val:]
    print(f"tokens: {len(toks)/1e6:.1f}M (val {len(val_toks)}, train {len(train_toks)/1e6:.1f}M)", flush=True)

    def batch(src, n, width=T + 1):
        s = rng.integers(0, len(src) - width, size=n)
        x = np.stack([np.asarray(src[i:i + width], dtype=np.int64) for i in s])
        return torch.from_numpy(x).to(dev)

    def val_loss():
        model.eval()
        with torch.no_grad():
            tot = 0.0
            for i in range(0, args.val_windows, args.batch):
                x = batch(val_toks, min(args.batch, args.val_windows - i))
                lg = model(x[:, :T]).float()
                tot += F.cross_entropy(lg.reshape(-1, lg.size(-1)), x[:, 1:T + 1].reshape(-1)).item()
        model.train()
        return tot / math.ceil(args.val_windows / args.batch)

    heads = None
    if any(w > 0 for w in [args.nextlat_weight, args.top_weight, args.lmtp_weight,
                           args.bst_weight, args.jtp_weight]):
        heads = LookaheadSystem(args.d_model, 65536, nextlat_weight=args.nextlat_weight,
                                top_weight=args.top_weight, lmtp_weight=args.lmtp_weight,
                                bst_weight=args.bst_weight, jtp_weight=args.jtp_weight,
                                lm_head=model.head).to(dev, torch.bfloat16)
        print(f"aux heads enabled={heads.enabled} extra_tokens={heads.extra_tokens}", flush=True)
    params = list(model.parameters()) + (list(heads.parameters()) if heads else [])
    opt = torch.optim.AdamW(params, lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1)
    model.train(); t0 = time.time(); seen = 0; step = 0
    print(f"budget={'%.1f min' % args.minutes if not args.steps else str(args.steps)+' steps'}", flush=True)
    while True:
        if args.steps and step >= args.steps: break
        if not args.steps and (time.time() - t0) / 60.0 >= args.minutes: break
        if step % args.eval_every == 0:
            vl = val_loss(); emit({"kind": "eval", "step": step, "loss": vl, "val_loss": vl, "ppl": math.exp(vl)})
            print(f"[{step}] val {vl:.4f} (ppl {math.exp(vl):.2f})  {(time.time()-t0)/60:.1f}min", flush=True)
        lr = args.lr * min(1.0, (step + 1) / max(args.warmup, 1))       # linear warmup
        for g in opt.param_groups:
            g["lr"] = lr
        ex = heads.extra_tokens if heads else 0
        x = batch(train_toks, args.batch, width=T + 1 + ex)
        out = model(x[:, :T], return_hidden=bool(heads))
        lg = (out[0] if heads else out).float()
        loss = F.cross_entropy(lg.reshape(-1, lg.size(-1)), x[:, 1:T + 1].reshape(-1))
        if heads:                                            # + weighted aux (latent-prediction) loss
            loss = loss + heads.compute(out[1], x, model.emb, model.head)["aux_total"]
        opt.zero_grad(set_to_none=True); loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step(); seen += args.batch * T; step += 1
        if step % args.log_every == 0:
            emit({"kind": "train", "step": step, "loss": float(loss), "gnorm": float(gn),
                  "lr": lr, "tok_per_sec": int(seen / max(time.time() - t0, 1e-6))})
    vl = val_loss(); emit({"kind": "eval", "step": step, "loss": vl, "val_loss": vl, "ppl": math.exp(vl)})
    emit({"kind": "checkpoint", "step": step})
    print(f"DONE {tag}: {step} steps, final val {vl:.4f} (ppl {math.exp(vl):.2f})", flush=True)


if __name__ == "__main__":
    main()
