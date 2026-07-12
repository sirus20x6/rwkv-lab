"""Finetune a native BlinkDL RWKV-7 (g070, e.g. rwkv7-g1g-1.5b) with RWKV-Lab levers.

Loads a BlinkDL `.pth` into fla's (faithful, trainable) `RWKV7ForCausalLM` via a transpose-
correct remap (verified: 0 missing / 0 unexpected, baseline ~2.46 nats), then trains LM cross-
entropy on a World-tokenized stream. Optimizer is selectable: plain AdamW or our `spectral_muon`
with all `--sm-*` levers. Designed for QUICK (~10-min) A/B partitions to surface bugs fast and
compare levers on a real pretrained model. Logs to a trainboard `train.jsonl`.

Optimizer levers attach directly (model-agnostic). Loop/objective levers need our own RWKV core
(see EXP-C) and don't attach to fla's forward — this harness is the optimizer-lever track.

    python -m rwkv_lab.rwkv_finetune --model models/rwkv7-g1g-1.5b.pth --data models/g1g_tokens.bin \
        --optimizer spectral_muon --sm-rsav 1 --minutes 10 --out runs/g1g_rsav
"""
from __future__ import annotations
import argparse, json, math, os, time
import numpy as np
import torch
import torch.nn.functional as F


# ---- faithful BlinkDL g070 -> fla RWKV7 loader (transpose-correct remap) ----
def load_g1g_fla(path, n_layers=24, hidden=2048, n_heads=32, head_dim=64,
                 decay_r=96, a_r=96, v_r=64, gate_r=256, vocab=65536, inter=8192,
                 device="cuda", dtype=torch.bfloat16):
    from fla.models.rwkv7 import RWKV7Config, RWKV7ForCausalLM
    sd = torch.load(path, map_location="cpu", weights_only=True)
    cfg = RWKV7Config(hidden_size=hidden, num_hidden_layers=n_layers, num_heads=n_heads,
                      head_dim=head_dim, decay_low_rank_dim=decay_r, a_low_rank_dim=a_r,
                      v_low_rank_dim=v_r, gate_low_rank_dim=gate_r, vocab_size=vocab,
                      intermediate_size=inter, norm_bias=True, attn=None, fuse_norm=False)
    m = RWKV7ForCausalLM(cfg); tgt = m.state_dict()

    def put(k, v):
        assert k in tgt and tgt[k].numel() == v.numel(), f"{k}: {tuple(tgt[k].shape)} vs {tuple(v.shape)}"
        tgt[k] = v.reshape(tgt[k].shape).to(tgt[k].dtype)

    put("model.embeddings.weight", sd["emb.weight"])
    put("model.norm.weight", sd["ln_out.weight"]); put("model.norm.bias", sd["ln_out.bias"])
    put("lm_head.weight", sd["head.weight"])
    put("model.layers.0.pre_norm.weight", sd["blocks.0.ln0.weight"])
    put("model.layers.0.pre_norm.bias", sd["blocks.0.ln0.bias"])
    for i in range(n_layers):
        b, l, a, la = f"blocks.{i}", f"model.layers.{i}", f"blocks.{i}.att", f"model.layers.{i}.attn"
        put(f"{l}.attn_norm.weight", sd[f"{b}.ln1.weight"]); put(f"{l}.attn_norm.bias", sd[f"{b}.ln1.bias"])
        put(f"{l}.ffn_norm.weight", sd[f"{b}.ln2.weight"]); put(f"{l}.ffn_norm.bias", sd[f"{b}.ln2.bias"])
        for x in ("x_r", "x_w", "x_k", "x_v", "x_a", "x_g"):
            put(f"{la}.{x}", sd[f"{a}.{x}"])
        put(f"{la}.k_k", sd[f"{a}.k_k"]); put(f"{la}.k_a", sd[f"{a}.k_a"]); put(f"{la}.r_k", sd[f"{a}.r_k"])
        put(f"{la}.r_proj.weight", sd[f"{a}.receptance.weight"]); put(f"{la}.k_proj.weight", sd[f"{a}.key.weight"])
        put(f"{la}.v_proj.weight", sd[f"{a}.value.weight"]); put(f"{la}.o_proj.weight", sd[f"{a}.output.weight"])
        put(f"{la}.g_norm.weight", sd[f"{a}.ln_x.weight"]); put(f"{la}.g_norm.bias", sd[f"{a}.ln_x.bias"])
        def lora(n, bias=True):                                # BlinkDL X1(C,r)/X2(r,C)/X0(1,1,C) -> fla lora (transposed)
            put(f"{la}.{n}_lora.lora.0.weight", sd[f"{a}.{n}1"].t().contiguous())
            put(f"{la}.{n}_lora.lora.2.weight", sd[f"{a}.{n}2"].t().contiguous())
            if bias:
                put(f"{la}.{n}_lora.lora.2.bias", sd[f"{a}.{n}0"])
        lora("w"); lora("a"); lora("g", bias=False)
        if i > 0:                                              # v-residual (value gating) exists only for layers >=1
            lora("v")
        put(f"{l}.ffn.x_k", sd[f"{b}.ffn.x_k"]); put(f"{l}.ffn.key.weight", sd[f"{b}.ffn.key.weight"])
        put(f"{l}.ffn.value.weight", sd[f"{b}.ffn.value.weight"])
    info = m.load_state_dict(tgt, strict=False)
    assert not info.missing_keys and not info.unexpected_keys, (info.missing_keys[:4], info.unexpected_keys[:4])
    return m.to(device, dtype)


def build_optimizer(model, args):
    """AdamW, or spectral_muon (Muon on 2D weight matrices, AdamW on the rest) with our --sm-* levers."""
    if args.optimizer == "adamw":                              # finetune LR (adam_lr); args.lr is the Muon-scale LR
        return torch.optim.AdamW(model.parameters(), lr=args.adam_lr, betas=(0.9, 0.95), weight_decay=0.0)
    from rwkv_lab.spectral_muon import SpectralMuon
    muon, adam = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_mat = p.ndim == 2 and "embeddings" not in n and "lm_head" not in n and "norm" not in n
        (muon if is_mat else adam).append(p)
    groups = [
        {"params": muon, "use_muon": True, "lr": args.lr},
        {"params": adam, "use_muon": False, "lr": args.adam_lr},
    ]
    return SpectralMuon(groups, momentum=0.95, ns_steps=args.sm_ns_steps, scale=args.sm_scale,
                        spectral_power=args.sm_spectral_power, mona=bool(args.sm_mona),
                        second_moment=bool(args.sm_second_moment), plus_norm=args.sm_plus_norm,
                        ddc_strength=args.sm_ddc_strength, rsav=bool(args.sm_rsav),
                        tile_size=args.sm_tile_size, da_muon=bool(args.sm_da_muon),
                        aro=bool(args.sm_aro), weight_decay=0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="runs/g1g_ft")
    ap.add_argument("--optimizer", choices=["adamw", "spectral_muon"], default="adamw")
    ap.add_argument("--lr", type=float, default=1e-4)        # Muon matrix LR (finetune scale)
    ap.add_argument("--adam-lr", type=float, default=1e-5)   # AdamW LR for norms/embeds/1D + the adamw baseline
    ap.add_argument("--seq-len", type=int, default=1024); ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--minutes", type=float, default=10.0); ap.add_argument("--steps", type=int, default=0)
    ap.add_argument("--val-windows", type=int, default=32); ap.add_argument("--eval-every", type=int, default=50)
    ap.add_argument("--log-every", type=int, default=5); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    for f, d in [("sm-ns-steps", 5), ("sm-tile-size", 0)]:
        ap.add_argument(f"--{f}", type=int, default=d)
    for f in ["sm-mona", "sm-second-moment", "sm-rsav", "sm-da-muon", "sm-aro"]:
        ap.add_argument(f"--{f}", type=int, default=0)
    ap.add_argument("--sm-scale", type=float, default=0.4); ap.add_argument("--sm-spectral-power", type=float, default=0.0)
    ap.add_argument("--sm-ddc-strength", type=float, default=0.0); ap.add_argument("--sm-plus-norm", default="none")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    jl = open(os.path.join(args.out, "train.jsonl"), "w", buffering=1)
    emit = lambda r: jl.write(json.dumps(r) + "\n")
    dev = "cuda"; T = args.seq_len
    torch.manual_seed(args.seed); rng = np.random.default_rng(args.seed)

    print(f"loading {args.model} into fla RWKV7 ...", flush=True)
    model = load_g1g_fla(args.model, device=dev)
    # NOTE: token .bin is assumed uint16 (World vocab = 65536 ids fits exactly).
    # A file written with a wider dtype (e.g. u32) read as u16 shows up as every
    # 2nd halfword being zero — detect that instead of training on garbage.
    toks = np.memmap(args.data, dtype=np.uint16, mode="r")
    _sample = np.asarray(toks[: min(len(toks), 1 << 20)])
    if len(_sample) >= 4 and not _sample[1::2].any():
        raise ValueError(f"{args.data}: every 2nd uint16 is zero — file looks like "
                         "u32 tokens read as u16; re-export with --dtype u16")
    n_val = args.val_windows * T
    val_toks, train_toks = toks[:n_val], toks[n_val:]                 # fixed held-out val prefix
    # Fixed, evenly-spaced val windows: deterministic evals that never touch the training RNG.
    val_offsets = np.linspace(0, len(val_toks) - (T + 1), args.val_windows).astype(np.int64)
    print(f"tokens: {len(toks)/1e6:.1f}M  (val {len(val_toks)}, train {len(train_toks)/1e6:.1f}M)", flush=True)
    json.dump({"loop_count": 1, "n_layers": 24, "mode": f"g1g-ft-{args.optimizer}"},
              open(os.path.join(args.out, "loop_rw.json"), "w"))

    def batch(src, n):
        s = rng.integers(0, len(src) - (T + 1), size=n)
        x = np.stack([np.asarray(src[i:i + T + 1], dtype=np.int64) for i in s])
        return torch.from_numpy(x).to(dev)

    def val_loss():
        model.eval()
        with torch.no_grad():
            tot, n_tok = 0.0, 0
            for i in range(0, args.val_windows, args.batch):
                offs = val_offsets[i:i + args.batch]      # fixed windows: no training-RNG draw
                x = torch.from_numpy(np.stack(
                    [np.asarray(val_toks[o:o + T + 1], dtype=np.int64) for o in offs])).to(dev)
                lg = model(x[:, :T]).logits.float()
                tot += F.cross_entropy(lg.reshape(-1, lg.size(-1)), x[:, 1:T + 1].reshape(-1),
                                       reduction="sum").item()
                n_tok += x.shape[0] * T
        model.train()
        return tot / max(1, n_tok)

    opt = build_optimizer(model, args); model.train()
    print(f"optimizer={args.optimizer}  budget={'%.1f min' % args.minutes if not args.steps else str(args.steps)+' steps'}", flush=True)
    t0 = time.time(); seen = 0; step = 0
    while True:
        if args.steps and step >= args.steps: break
        if not args.steps and (time.time() - t0) / 60.0 >= args.minutes: break
        if step % args.eval_every == 0:
            vl = val_loss(); emit({"kind": "eval", "step": step, "loss": vl, "val_loss": vl, "ppl": math.exp(vl)})
            print(f"[{step}] val {vl:.4f} (ppl {math.exp(vl):.2f})  {(time.time()-t0)/60:.1f}min", flush=True)
        x = batch(train_toks, args.batch)
        lg = model(x[:, :T]).logits.float()
        loss = F.cross_entropy(lg.reshape(-1, lg.size(-1)), x[:, 1:T + 1].reshape(-1))
        opt.zero_grad(set_to_none=True); loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step(); seen += args.batch * T; step += 1
        if step % args.log_every == 0:
            emit({"kind": "train", "step": step, "loss": float(loss), "gnorm": float(gn),
                  "tok_per_sec": int(seen / max(time.time() - t0, 1e-6))})
    vl = val_loss(); emit({"kind": "eval", "step": step, "loss": vl, "val_loss": vl, "ppl": math.exp(vl)})
    emit({"kind": "checkpoint", "step": step})
    print(f"DONE {step} steps, final val {vl:.4f} (ppl {math.exp(vl):.2f})", flush=True)


if __name__ == "__main__":
    main()
