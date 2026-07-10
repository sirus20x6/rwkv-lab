"""From-scratch small RWKV-7 pretraining with RWKV-Lab loop / recurrent-depth levers.

Builds a small RWKV-7 LM entirely from OUR modules (RWKV8TimeMixDeltaNet + RWKV8ChannelMix
+ emb/head), so the loop levers (LoopedRWKV) attach NATIVELY — no fla, no g1g remap, no
forward-reconciliation. Trains from random init on a World-tokenized stream (ztok), so there
is real loss headroom: recurrent depth / hyper-connections / CART / DEQ can actually help,
and a fixed-wall-clock A/B (~10 min) measures whether the extra compute-per-token pays off.

Each block's time-mix is optionally wrapped in LoopedRWKV(core, ...). is_first_rwkv_layer=(i==0),
so the native RWKV-7 cross-layer value residual (v_first) is active — layer 0 defines the shared
value and later layers lerp toward it, threaded through the stack. Logs to a trainboard train.jsonl.

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
    def __init__(self, d, n_heads, head_size, i, n_layers, loop_kw, att_kw=None, ffn_hidden=None):
        super().__init__()
        self.i = i
        if i == 0:
            self.ln0 = nn.LayerNorm(d)
        self.ln1 = nn.LayerNorm(d)
        self.ln2 = nn.LayerNorm(d)
        core = RWKV8TimeMixDeltaNet(d, num_heads=n_heads, head_size=head_size, layer_idx=i,
                                    depth_layer_id=i, depth_n_layer=max(n_layers, 2),
                                    is_first_rwkv_layer=(i == 0),   # native cross-layer v-residual
                                    out_correct=False,              # clean native g070
                                    **(att_kw or {}))               # e.g. g1g LoRA dims
        self.att = LoopedRWKV(core, hidden_size=d, **loop_kw) if loop_kw else core
        self.ffn = RWKV8ChannelMixDeltaNet(d, ffn_hidden, layer_idx=i)

    def forward(self, x, v_first, seed=None, return_seed=False):
        if self.i == 0:
            x = self.ln0(x)
        if seed is not None or return_seed:                  # Future-Seed: seed this layer's wkv scan
            if isinstance(self.att, LoopedRWKV):
                # LoopedRWKV forwards state kwargs to pass 1 only; refinement passes would run
                # stateless — ambiguous semantics, so refuse rather than silently accept.
                raise ValueError("Future-Seed state on a LoopedRWKV att is unsupported")
            if return_seed:
                a, seed_out, _shift, v_first = self.att(self.ln1(x), v_first=v_first, return_v_first=True,
                                                        initial_state=seed, return_state=True)
            else:                                            # last chained layer: consume s_0, skip unused s_T
                a, v_first = self.att(self.ln1(x), v_first=v_first, return_v_first=True,
                                      initial_state=seed)
        else:
            a, v_first = self.att(self.ln1(x), v_first=v_first, return_v_first=True)
        x = x + a
        x = x + _unwrap(self.ffn(self.ln2(x)))
        if return_seed:
            return x, v_first, seed_out
        return x, v_first


class RWKV7Small(nn.Module):
    def __init__(self, vocab, d, n_layers, head_size, loop_kw, att_kw=None, ffn_hidden=None,
                 seed_chain=False):
        super().__init__()
        assert d % head_size == 0
        if seed_chain and loop_kw:
            raise ValueError("seed_chain (Future-Seed cross-layer state) is incompatible with loop "
                             "levers — run it without loops for a clean A/B")
        self.seed_chain = seed_chain          # Future-Seed: layer L starts from layer L-1's final wkv state
        self.emb = nn.Embedding(vocab, d)
        self.blocks = nn.ModuleList([Block(d, d // head_size, head_size, i, n_layers, loop_kw,
                                           att_kw, ffn_hidden)
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
        seed = None                              # Future-Seed: s_T of layer L-1 -> s_0 of layer L (None => 0)
        for j, b in enumerate(self.blocks):
            if self.seed_chain and j < len(self.blocks) - 1:
                x, v_first, seed = b(x, v_first, seed=seed, return_seed=True)
            else:                                # last layer consumes the seed but skips the unused s_T
                x, v_first = b(x, v_first, seed=seed)
        h = self.ln_out(x)                       # post-norm final hidden (what aux heads read)
        logits = self.head(h)
        return (logits, h) if return_hidden else logits


def _adamw8bit(params, lr, wd, paged):
    """bitsandbytes 8-bit AdamW (blockwise-quantized moment states, ~75% optimizer-memory cut at
    ~fp32 quality). paged=True routes state through CUDA unified memory to ride out OOM spikes on
    big models. CUDA-only in bnb; a missing/unusable bnb raises a clear error at construction."""
    try:
        import bitsandbytes as bnb
    except Exception as e:  # noqa: BLE001 — surface the real cause (no wheel, bad CUDA, etc.)
        raise RuntimeError("8-bit optimizer needs bitsandbytes: `uv pip install bitsandbytes`") from e
    Opt = bnb.optim.PagedAdamW8bit if paged else bnb.optim.AdamW8bit
    return Opt(params, lr=lr, betas=(0.9, 0.95), weight_decay=wd)


def build_optimizer(named_params, name, lr, wd, adam_lr=0.0, muon_opts=None):
    """AdamW, 8-bit AdamW (bitsandbytes), or spectral_muon (Muon on 2D weight matrices, AdamW on
    embeds/norms/1D). Shared by the LM and synthetic harnesses so the card's optimizer dropdown
    drives both. adam_lr (0 = use lr) is the fallback LR for non-matrix params under Muon — Muon
    matrix LRs run larger than AdamW's. muon_opts selects the Muon variant (spectral_power=Muon^p,
    ddc_strength=DDC, mona=Muon²/MONA, second_moment=Aurora, rsav, da_muon, aro, + scale/ns_steps)
    — passed straight to SpectralMuon. The 8-bit variants apply only to the AdamW path (the Muon
    fallback group is embeds/norms/1D — negligible state — so it stays fp32)."""
    named = [(n, p) for n, p in named_params if p.requires_grad]
    if name == "muon":
        from rwkv_lab.spectral_muon import SpectralMuon
        muon, adam = [], []
        for n, p in named:
            is_mat = p.ndim == 2 and not any(k in n for k in ("emb", "head", "norm"))
            (muon if is_mat else adam).append(p)
        groups = [{"params": muon, "use_muon": True, "lr": lr},
                  {"params": adam, "use_muon": False, "lr": adam_lr or lr}]
        return SpectralMuon(groups, weight_decay=wd, **(muon_opts or {}))
    params = [p for _, p in named]
    if name in ("adamw8bit", "paged-adamw8bit"):
        return _adamw8bit(params, lr, wd, paged=(name == "paged-adamw8bit"))
    import torch as _t
    fused = bool(params) and params[0].is_cuda      # fused AdamW = one fused CUDA kernel (CUDA-only)
    return _t.optim.AdamW(params, lr=lr, betas=(0.9, 0.95), weight_decay=wd, fused=fused)


def apply_fp8(module):
    """Swap eligible nn.Linear layers to torchao Float8Linear so their GEMMs run on the fp8
    tensor cores (Blackwell sm_120 / Hopper). This is orthogonal to the optimizer: bf16/fp32
    MASTER weights are kept and dynamically cast to fp8 per forward, so build_optimizer, the
    training loop, and checkpointing are all unchanged. Only converts linears whose in/out
    features are both multiples of 16 (the fp8 GEMM constraint) and skips the vocab head +
    embeddings (fp8 there costs quality for little FLOP). Returns #layers converted.

    Note: eager fp8 trains correctly but the throughput win needs torch.compile to fuse the
    cast+GEMM; without it fp8 can be net-neutral on small models. Clear error if torchao missing."""
    try:
        from torchao.float8 import convert_to_float8_training
        from torchao.float8.float8_linear import Float8Linear
    except Exception as e:  # noqa: BLE001 — surface the real cause (no wheel, bad CUDA, etc.)
        raise RuntimeError("fp8 training needs torchao: `uv pip install torchao`") from e
    import torch.nn as _nn

    def keep(m, fqn):
        return (isinstance(m, _nn.Linear) and "head" not in fqn.lower()
                and m.in_features % 16 == 0 and m.out_features % 16 == 0)

    convert_to_float8_training(module, module_filter_fn=keep)
    return sum(isinstance(x, Float8Linear) for x in module.modules())


def enable_fast_matmul():
    """Turn on TF32 tensor cores for fp32 matmuls (the full-vocab CE, Newton-Schulz, any fp32 op)
    + cuDNN TF32. Free ~1.1-1.3x on Ampere+; no effect on the bf16/fp8 paths. Idempotent — the
    careful-zone trainers already set this; the research harnesses (this file, experiment.py) did
    not. Call once at entrypoint startup."""
    import torch as _t
    _t.set_float32_matmul_precision("high")
    _t.backends.cuda.matmul.allow_tf32 = True
    _t.backends.cudnn.allow_tf32 = True


# --sm-* CLI flags -> SpectralMuon kwargs (the Muon variants exposed by the card).
def add_muon_args(ap):
    ap.add_argument("--sm-scale", type=float, default=0.4)
    ap.add_argument("--sm-spectral-power", type=float, default=0.0)   # Muon^p
    ap.add_argument("--sm-ddc-strength", type=float, default=0.0)     # DDC
    ap.add_argument("--sm-ns-steps", type=int, default=5)
    ap.add_argument("--sm-tile-size", type=int, default=0)
    ap.add_argument("--sm-plus-norm", default="none")
    for f in ["mona", "second-moment", "rsav", "da-muon", "aro"]:
        ap.add_argument(f"--sm-{f}", type=int, default=0)


def muon_opts_from(a):
    return dict(scale=a.sm_scale, spectral_power=a.sm_spectral_power, ddc_strength=a.sm_ddc_strength,
                ns_steps=a.sm_ns_steps, tile_size=a.sm_tile_size, plus_norm=a.sm_plus_norm,
                mona=bool(a.sm_mona), second_moment=bool(a.sm_second_moment), rsav=bool(a.sm_rsav),
                da_muon=bool(a.sm_da_muon), aro=bool(a.sm_aro))


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
    enable_fast_matmul()
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True); ap.add_argument("--out", default="runs/rwkv_scratch")
    ap.add_argument("--doc-offsets", default="", help="build_corpus .off.npy => within-doc windows")
    ap.add_argument("--gpu-data", default="auto", choices=["auto", "on", "off"],
                    help="hold the token corpus on GPU for CPU-free window sampling (auto = if it fits the cap)")
    ap.add_argument("--gpu-data-cap-gb", type=float, default=24.0,
                    help="max int32 corpus size to place on GPU under --gpu-data auto")
    ap.add_argument("--d-model", type=int, default=512); ap.add_argument("--n-layers", type=int, default=6)
    ap.add_argument("--head-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=6e-4); ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--batch", type=int, default=16); ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--minutes", type=float, default=10.0); ap.add_argument("--steps", type=int, default=0)
    ap.add_argument("--val-windows", type=int, default=40); ap.add_argument("--eval-every", type=int, default=50)
    ap.add_argument("--log-every", type=int, default=10); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--optimizer", default="adamw",
                    choices=["adamw", "adamw8bit", "paged-adamw8bit", "muon"])
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--fp8", action="store_true",
                    help="run eligible Linear GEMMs in fp8 (torchao Float8Linear; Blackwell/Hopper)")
    ap.add_argument("--compile", action="store_true",
                    help="torch.compile the training forward (fuses fp8 cast+GEMM; ~2x on Blackwell)")
    add_muon_args(ap)
    ap.add_argument("--lr-schedule", default="cosine", choices=["constant", "cosine"])
    ap.add_argument("--decay-steps", type=int, default=0)   # cosine horizon; 0 => use --steps
    ap.add_argument("--save", default=""); ap.add_argument("--resume", default="")
    ap.add_argument("--init-g1g", default="", help="continue-train from a pretrained g1g .pth (dims forced to g1g)")
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
    ap.add_argument("--seed-chain", type=int, default=0,   # int like the loop flags (lever-translatable)
                    help="Future-Seed: seed layer L's wkv scan with layer L-1's final state (from-scratch, no loops)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    jl = open(os.path.join(args.out, "train.jsonl"), "w", buffering=1)
    emit = lambda r: jl.write(json.dumps(r) + "\n")
    dev = "cuda"; T = args.seq_len
    torch.manual_seed(args.seed); rng = np.random.default_rng(args.seed)

    lk = loop_kwargs(args)
    if args.init_g1g:                                        # continued pretraining from pretrained g1g
        from rwkv_lab.native_g1g import load_g1g_native, add_loops
        model, ginfo = load_g1g_native(args.init_g1g, device=dev)
        if lk:
            add_loops(model, lk)                             # levers attach identity-at-init
        model = model.to(dev, torch.bfloat16)
        print(f"init from g1g {args.init_g1g}: loaded {ginfo['loaded']}/{ginfo['n_ckpt']} tensors "
              f"(dims forced to g1g 24L/d2048/h64)", flush=True)
        if args.seed_chain:
            print("warn: --seed-chain ignored for g1g init (from-scratch only)", flush=True)
    else:
        model = RWKV7Small(65536, args.d_model, args.n_layers, args.head_size, lk,
                           seed_chain=bool(args.seed_chain)).to(dev, torch.bfloat16)
        if args.seed_chain:
            print("Future-Seed: cross-layer state chaining ON (s_0^L = s_T^{L-1})", flush=True)
    if args.fp8:
        n8 = apply_fp8(model)
        print(f"fp8: {n8} Linear layers -> Float8Linear (torchao)", flush=True)
    nparam = sum(p.numel() for p in model.parameters())
    seed_chain = bool(args.seed_chain) and not args.init_g1g  # g1g branch ignores the flag
    tag = f"scratch-L{args.n_layers}d{args.d_model}-loop{args.loop_count}" + \
          ("".join(k for k, v in [("H", args.loop_hyper), ("C", args.loop_cart_anchor),
           ("Q", args.loop_deq), ("F", args.loop_fp_halt), ("A", args.loop_adaptive_halt),
           ("R", args.loop_iter_readout)] if v) or "") + \
          (f"-{args.loop_gate}" if lk and args.loop_gate != "scalar" else "") + \
          ("-seedchain" if seed_chain else "")
    print(f"model {tag}: {nparam/1e6:.1f}M params  loop_kw={lk}", flush=True)
    json.dump({"loop_count": args.loop_count, "n_layers": args.n_layers, "mode": tag,
               "seed_chain": seed_chain,
               "params_m": round(nparam / 1e6, 2)}, open(os.path.join(args.out, "loop_rw.json"), "w"))

    toks = np.memmap(args.data, dtype=np.uint16, mode="r")
    n_val = args.val_windows * T
    val_toks, train_toks = toks[:n_val], toks[n_val:]
    print(f"tokens: {len(toks)/1e6:.1f}M (val {len(val_toks)}, train {len(train_toks)/1e6:.1f}M)", flush=True)

    train_docs = None
    if args.doc_offsets:                                       # within-doc windows (no mid-doc cuts)
        allo = np.load(args.doc_offsets).astype(np.int64)
        ends = np.append(allo[1:], len(toks))
        train_docs = [(int(s), int(e)) for s, e in zip(allo, ends) if s >= n_val and e - s >= T + 1]
        print(f"doc-boundary batching: {len(train_docs)} train docs >= {T+1} tok", flush=True)

    def batch(src, n, width=T + 1):
        s = rng.integers(0, len(src) - width, size=n)
        x = np.stack([np.asarray(src[i:i + width], dtype=np.int64) for i in s])
        return torch.from_numpy(x).to(dev)

    def train_batch(n, width=T + 1):
        if not train_docs:                                    # flat fallback
            return batch(train_toks, n, width)
        rows = []
        for _ in range(n):
            s, e = train_docs[int(rng.integers(0, len(train_docs)))]
            i = int(rng.integers(s, e - width + 1))
            rows.append(np.asarray(toks[i:i + width], dtype=np.int64))
        return torch.from_numpy(np.stack(rows)).to(dev)

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
    named = list(model.named_parameters()) + (list(heads.named_parameters()) if heads else [])
    opt = build_optimizer(named, args.optimizer, args.lr, args.weight_decay, muon_opts=muon_opts_from(args))
    print(f"optimizer={args.optimizer} lr={args.lr} wd={args.weight_decay}", flush=True)
    step = 0
    if args.resume and os.path.exists(args.resume):
        ck = torch.load(args.resume, map_location=dev)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); step = ck.get("step", 0)
        print(f"resumed from {args.resume} @ step {step}", flush=True)
    # Compiled handle for the TRAIN forward only: eager `model` still owns state_dict/params, so
    # checkpoints stay uncompiled (no `_orig_mod.` prefix) and the eval path (below) never toggles
    # the compiled graph's train/eval mode (which would force costly recompiles).
    fwd = torch.compile(model) if args.compile else model
    if args.compile:
        print("torch.compile: enabled (step 0 compiles; forward only, checkpoints uncompiled)", flush=True)
    # Training-batch sampler. Hold the corpus on GPU (int32) when it fits, so each step's window
    # sampling is a pure GPU gather — no per-step CPU gather, no H2D. This lets tiny models run
    # data-unbound at very high step rates (the memmap CPU path serializes the GPU behind Python).
    # Falls back to the CPU memmap sampler for corpora too large for VRAM.
    width = T + 1 + (heads.extra_tokens if heads else 0)
    gpu_gb = len(train_toks) * 4 / 1e9
    use_gpu_data = args.gpu_data == "on" or (args.gpu_data == "auto" and gpu_gb <= args.gpu_data_cap_gb
                                             and len(train_toks) > width)
    if use_gpu_data:
        tg = torch.from_numpy(np.ascontiguousarray(train_toks, dtype=np.int32)).to(dev)
        ar = torch.arange(width, device=dev)
        if train_docs:                                          # doc-boundary: sample doc, then offset
            ds = torch.tensor([s - n_val for s, e in train_docs], device=dev)
            dl = torch.tensor([e - s for s, e in train_docs], device=dev)
            def sample_train():
                di = torch.randint(0, ds.numel(), (args.batch,), device=dev)
                maxoff = (dl[di] - width).clamp(min=0)
                off = (torch.rand(args.batch, device=dev) * (maxoff + 1).float()).long().minimum(maxoff)
                return tg[(ds[di] + off)[:, None] + ar[None, :]].long()
        else:                                                  # flat: uniform window over the corpus
            hi = tg.numel() - width
            def sample_train():
                idx = torch.randint(0, hi, (args.batch,), device=dev)
                return tg[idx[:, None] + ar[None, :]].long()
        print(f"gpu-data: corpus on GPU ({gpu_gb:.2f} GB int32) — window sampling is GPU-side", flush=True)
    else:
        def sample_train():
            return train_batch(args.batch, width=width)
        print(f"gpu-data: OFF ({gpu_gb:.2f} GB corpus) — CPU memmap sampler", flush=True)
    model.train(); t0 = time.time(); seen = 0
    print(f"budget={'%.1f min' % args.minutes if not args.steps else str(args.steps)+' steps'}", flush=True)
    while True:
        if args.steps and step >= args.steps: break
        if not args.steps and (time.time() - t0) / 60.0 >= args.minutes: break
        if step % args.eval_every == 0:
            vl = val_loss(); emit({"kind": "eval", "step": step, "loss": vl, "val_loss": vl, "ppl": math.exp(vl)})
            print(f"[{step}] val {vl:.4f} (ppl {math.exp(vl):.2f})  {(time.time()-t0)/60:.1f}min", flush=True)
        lr = args.lr * min(1.0, (step + 1) / max(args.warmup, 1))       # linear warmup
        horizon = args.decay_steps or args.steps
        if args.lr_schedule == "cosine" and horizon:                    # then cosine decay to 0.1x
            lr *= 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(step, horizon) / horizon))
        for g in opt.param_groups:
            g["lr"] = lr
        x = sample_train()
        out = fwd(x[:, :T], return_hidden=bool(heads))
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
    if args.save:
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step, "config": tag},
                   args.save)
        print(f"saved -> {args.save}", flush=True)
    print(f"DONE {tag}: {step} steps, final val {vl:.4f} (ppl {math.exp(vl):.2f})", flush=True)


if __name__ == "__main__":
    main()
