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
    def __init__(self, d, n_heads, head_size, i, n_layers, loop_kw, att_kw=None, ffn_hidden=None,
                 de_vocab=0, de_dim=0, de_mode="out", de_shift=False, de_emb_res=False):
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
        # DeepEmbed (BlinkDL, RWKV-8): per-layer per-token multiplicative FFN gate — sparse capacity
        # whose lookup is ~free at inference. Two forms, both parametrized additively around 1 so a
        # zero output end = exact identity (and zero-init trains in bf16, where updates to a literal
        # 1.0 would round away):
        #   out    (v1): ffn_out * (1 + de(ids)) — purely token-dependent, full-width or low-rank
        #   hidden (BlinkDL rwkv_v7a exact): gate the FFN HIDDEN k, input-dependent through s1:
        #          ss = s1(xs) @ E_tok[r,r];  k *= 1 + s0 + s2(ss)
        #          de_shift gives the gate input its own token-shift mix (BlinkDL: "very large");
        #          de_emb_res folds the global token embedding into E_tok via a learned projection.
        self.de_mode = de_mode
        self.de_emb = self.de_proj = self.de_s1 = self.de_s2 = self.de_er = None
        self.de_s0 = self.de_xs = None
        if de_vocab and de_mode == "hidden":
            self.de_r = r = de_dim if de_dim > 0 else 32
            fh = self.ffn.ffn_hidden_size
            self.de_emb = nn.Embedding(de_vocab, r * r)     # per-token r x r matrix
            self.de_s1 = nn.Linear(d, r, bias=False)
            self.de_s2 = nn.Linear(r, fh, bias=False)       # zero-init output end (identity gate)
            self.de_s0 = nn.Parameter(torch.zeros(fh))
            if de_shift:
                self.de_xs = nn.Parameter(torch.zeros(d))   # xs = x + (x_prev - x) * x_s
            if de_emb_res:
                self.de_er = nn.Linear(d, r * r, bias=False)  # zero-init residual fold of emb(ids)
        elif de_vocab:
            if de_dim and de_dim < d:            # low-rank: table [V, r] + zero-init proj r -> d
                self.de_emb = nn.Embedding(de_vocab, de_dim)
                self.de_proj = nn.Linear(de_dim, d, bias=False)
            else:                                # full width: zero-init table [V, d]
                self.de_emb = nn.Embedding(de_vocab, d)

    def forward(self, x, v_first, seed=None, return_seed=False, ids=None, e0=None):
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
        xin = self.ln2(x)
        if self.de_mode == "hidden" and self.de_emb is not None and ids is not None:
            h = xin
            if self.de_xs is not None:                       # separate DE token-shift
                hp = torch.zeros_like(h); hp[:, 1:] = h[:, :-1]
                h = h + (hp - h) * self.de_xs.to(h.dtype)
            E = self.de_emb(ids)
            if self.de_er is not None and e0 is not None:    # emb-residual fold
                E = E + self.de_er(e0)
            B, T = ids.shape
            ss = torch.einsum("btr,btrs->bts", self.de_s1(h),
                              E.view(B, T, self.de_r, self.de_r).to(h.dtype))
            gate = 1.0 + self.de_s0.to(h.dtype) + self.de_s2(ss)
            f = _unwrap(self.ffn(xin, hidden_gate=gate))
        else:
            f = _unwrap(self.ffn(xin))
            if self.de_emb is not None and ids is not None:  # v1: gate the FFN output
                g = self.de_emb(ids)
                if self.de_proj is not None:
                    g = self.de_proj(g)
                f = f * (1.0 + g)
        x = x + f
        if return_seed:
            return x, v_first, seed_out
        return x, v_first


class RWKV7Small(nn.Module):
    def __init__(self, vocab, d, n_layers, head_size, loop_kw, att_kw=None, ffn_hidden=None,
                 seed_chain=False, deepembed=False, de_dim=0, de_mode="out", de_shift=False,
                 de_emb_res=False):
        super().__init__()
        assert d % head_size == 0
        if seed_chain and loop_kw:
            raise ValueError("seed_chain (Future-Seed cross-layer state) is incompatible with loop "
                             "levers — run it without loops for a clean A/B")
        if de_mode not in ("out", "hidden"):
            raise ValueError(f"de_mode must be 'out' or 'hidden', got {de_mode!r}")
        self.seed_chain = seed_chain          # Future-Seed: layer L starts from layer L-1's final wkv state
        self.deepembed = deepembed            # DeepEmbed: per-layer per-token FFN gate (needs ids)
        self.de_emb_res = de_emb_res          # hidden-mode: blocks also need the raw token embedding
        self.emb = nn.Embedding(vocab, d)
        self.blocks = nn.ModuleList([Block(d, d // head_size, head_size, i, n_layers, loop_kw,
                                           att_kw, ffn_hidden,
                                           de_vocab=vocab if deepembed else 0, de_dim=de_dim,
                                           de_mode=de_mode, de_shift=de_shift, de_emb_res=de_emb_res)
                                     for i in range(n_layers)])
        self.ln_out = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)
        self.apply(self._init)
        for b in self.blocks:                 # DeepEmbed identity-at-init: the global _init above
            if b.de_emb is None:              # re-randomized the tables — re-zero each gate's OUTPUT
                continue                      # end (bare Parameters like de_s0/de_xs are untouched)
            if b.de_mode == "hidden":
                b.de_s2.weight.data.zero_()
                if b.de_er is not None:
                    b.de_er.weight.data.zero_()
            else:
                (b.de_proj if b.de_proj is not None else b.de_emb).weight.data.zero_()

    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, ids, return_hidden=False, hidden_only=False):
        x = self.emb(ids)
        de_ids = ids if self.deepembed else None  # DeepEmbed blocks gate their FFN by token id
        e0 = x if (self.deepembed and self.de_emb_res) else None  # raw emb for the residual fold
        v_first = None                           # layer 0 sets it; later layers lerp toward it
        seed = None                              # Future-Seed: s_T of layer L-1 -> s_0 of layer L (None => 0)
        for j, b in enumerate(self.blocks):
            if self.seed_chain and j < len(self.blocks) - 1:
                x, v_first, seed = b(x, v_first, seed=seed, return_seed=True, ids=de_ids, e0=e0)
            else:                                # last layer consumes the seed but skips the unused s_T
                x, v_first = b(x, v_first, seed=seed, ids=de_ids, e0=e0)
        h = self.ln_out(x)                       # post-norm final hidden (what aux heads read)
        if hidden_only:
            return h
        logits = self.head(h)
        eng = getattr(self, "engram", None)      # copy head: gated bonus on the recalled token
        if eng is not None:                      # (exact no-op at init; see enable_engram)
            logits = eng.logit_bias(logits)
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
            # engram tables/projections are embedding-like: always AdamW (Muon LR-unit trap)
            is_mat = p.ndim == 2 and not any(k in n for k in ("emb", "head", "norm", "engram"))
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
        # engram/DeepEmbed excluded: zero-init growth projections would fight fp8 dynamic scaling
        return (isinstance(m, _nn.Linear) and "head" not in fqn.lower() and "engram" not in fqn.lower()
                and ".de_" not in fqn and m.in_features % 16 == 0 and m.out_features % 16 == 0)

    convert_to_float8_training(module, module_filter_fn=keep)
    return sum(isinstance(x, Float8Linear) for x in module.modules())


def enable_engram(model, vocab, d_model, head_size, n_layers, loop_count=1,
                  d_row=64, rows=4096, sites="auto", boundary_id=None):
    """Attach an Engram Lexical Memory Bank (engram_lmb Path C: parameter-free token-SAM recall
    over the raw ids reading a learned table; gated v-stream + inter-layer residual injection;
    copy-head logit bias applied in RWKV7Small.forward) to a model ALREADY on its final
    device/dtype. The bank is registered as `model.engram`, so parameters(), state_dict(),
    grad clipping and checkpoints all include it — resume needs --engram, like --seed-chain.
    Exact no-op at init (zero output projections). Returns (lmb, site_list)."""
    from rwkv_lab.engram_lmb import (LexicalMemoryBank, attach_engram, float_growth_params,
                                     install_input_ids_hook)
    if sites == "auto":   # depth-scaled shallow+mid placement (the 9B {3,15}/32 profile)
        site_list = sorted({min(max(1, n_layers // 8), n_layers - 1),
                            min(max(1, n_layers // 2), n_layers - 1)})
    else:
        site_list = sorted({int(s) for s in str(sites).split(",")})
        bad = [s for s in site_list if not 0 <= s < n_layers]
        if bad:
            raise ValueError(f"engram sites {bad} out of range for {n_layers} layers")
    p0 = next(model.parameters())
    lmb = LexicalMemoryBank(hidden_size=d_model, vocab_size=vocab, layer_sites=site_list,
                            d_row=d_row, table_rows=min(rows, vocab),
                            num_heads=d_model // head_size, max_loops=max(loop_count, 1),
                            boundary_id=boundary_id)
    lmb.to(device=p0.device, dtype=p0.dtype)
    float_growth_params(lmb)              # 1-D gates/scales stay fp32 (bf16 ULP swallows growth)
    attach_engram(model, lmb, resolve="blocks")
    install_input_ids_hook(model, lmb)    # model pre-hook stashes ids for the recall
    model.engram = lmb                    # registered submodule: optimizer/ckpt/clip see it
    return lmb, site_list


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
    ap.add_argument("--data", default=""); ap.add_argument("--out", default="runs/rwkv_scratch")
    ap.add_argument("--ctx-buckets", default="",
                    help="packed-buckets meta json (build_corpus.pack_context_buckets) — mixed "
                         "context-length training with reciprocal batch scaling; replaces --data")
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
    # Engram Lexical Memory Bank (engram_lmb Path C) — from-scratch lever
    ap.add_argument("--engram", type=int, default=0,
                    help="attach an Engram LMB: token-SAM recall + learned table, gated injection + copy head")
    ap.add_argument("--engram-sites", default="auto", help="comma layer indices, or auto (shallow+mid)")
    ap.add_argument("--engram-drow", type=int, default=64, help="learned-table row width")
    ap.add_argument("--engram-rows", type=int, default=4096, help="table rows (hashed; capped at vocab)")
    ap.add_argument("--engram-warmup", type=int, default=1000, help="steps to ramp injection 0 -> 1")
    ap.add_argument("--engram-boundary-id", type=int, default=-1,
                    help="EOD token id segmenting recall (-1 = none)")
    ap.add_argument("--deepembed", type=int, default=0,
                    help="DeepEmbed (RWKV-8): per-layer per-token FFN-output gate, 1 + emb(ids)")
    ap.add_argument("--de-dim", type=int, default=0,
                    help="DeepEmbed width: out-mode low-rank r (0 = full d_model); hidden-mode rank r (0 = 32)")
    ap.add_argument("--de-mode", default="out", choices=["out", "hidden"],
                    help="out = gate FFN output (v1); hidden = BlinkDL rwkv_v7a exact (bilinear gate on FFN hidden)")
    ap.add_argument("--de-shift", type=int, default=0,
                    help="hidden-mode: separate token-shift mix for the gate input (BlinkDL: 'very large')")
    ap.add_argument("--de-emb-res", type=int, default=0,
                    help="hidden-mode: fold the global token embedding into the per-token gate matrix")
    ap.add_argument("--grad-accum", type=int, default=1,
                    help="micro-batches accumulated per optimizer step (effective batch = batch * N)")
    ap.add_argument("--ema", type=float, default=0.0,
                    help="EMA decay for a shadow weight copy (e.g. 0.999); eval + checkpoint carry it. 0 = off")
    args = ap.parse_args()
    if not args.data and not args.ctx_buckets:
        ap.error("one of --data or --ctx-buckets is required")
    if args.ctx_buckets:                       # mixed-context mode: fixed-width features rejected
        if any(w > 0 for w in [args.nextlat_weight, args.top_weight, args.lmtp_weight,
                               args.bst_weight, args.jtp_weight]):
            ap.error("--ctx-buckets: aux lookahead heads are fixed-width — unsupported")
        if args.doc_offsets:
            print("warn: --doc-offsets ignored with --ctx-buckets (rows are already packed)", flush=True)
            args.doc_offsets = ""

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
        if args.deepembed:
            print("warn: --deepembed ignored for g1g init (from-scratch only)", flush=True)
    else:
        model = RWKV7Small(65536, args.d_model, args.n_layers, args.head_size, lk,
                           seed_chain=bool(args.seed_chain), deepembed=bool(args.deepembed),
                           de_dim=args.de_dim, de_mode=args.de_mode, de_shift=bool(args.de_shift),
                           de_emb_res=bool(args.de_emb_res)).to(dev, torch.bfloat16)
        if args.seed_chain:
            print("Future-Seed: cross-layer state chaining ON (s_0^L = s_T^{L-1})", flush=True)
        if args.deepembed:
            if args.de_mode == "hidden":
                r = args.de_dim if args.de_dim > 0 else 32
                w, what = r * r, f"hidden gate rank {r}"
            else:
                w = r = args.de_dim if 0 < args.de_dim < args.d_model else args.d_model
                what = f"output gate width {r}"
            print(f"DeepEmbed: {what}"
                  + (" + de-shift" if args.de_shift and args.de_mode == "hidden" else "")
                  + (" + emb-residual" if args.de_emb_res and args.de_mode == "hidden" else "")
                  + f" — {args.n_layers} tables of 65536x{w}"
                  f" ({args.n_layers * 65536 * w / 1e6:.0f}M sparse params)", flush=True)
    if args.fp8:
        n8 = apply_fp8(model)
        print(f"fp8: {n8} Linear layers -> Float8Linear (torchao)", flush=True)
    lmb = None
    if args.engram:
        if args.init_g1g:
            print("warn: --engram ignored for g1g init (from-scratch only; dims come from g1g)", flush=True)
        else:                                 # after fp8 so the engram Linears stay bf16
            # packed bucket rows join docs with the sep token: default the recall boundary to it
            # so recall never crosses packed-document boundaries.
            bid = args.engram_boundary_id if args.engram_boundary_id >= 0 else \
                (1 if args.ctx_buckets else None)
            lmb, esites = enable_engram(model, 65536, args.d_model, args.head_size, args.n_layers,
                                        loop_count=args.loop_count, d_row=args.engram_drow,
                                        rows=args.engram_rows, sites=args.engram_sites,
                                        boundary_id=bid)
            print(f"engram: LMB sites={esites} d_row={args.engram_drow} "
                  f"rows={min(args.engram_rows, 65536)} boundary_id={bid} "
                  f"(+{sum(p.numel() for p in lmb.parameters())/1e6:.2f}M params)", flush=True)
    nparam = sum(p.numel() for p in model.parameters())
    seed_chain = bool(args.seed_chain) and not args.init_g1g  # g1g branch ignores the flag
    tag = f"scratch-L{args.n_layers}d{args.d_model}-loop{args.loop_count}" + \
          ("".join(k for k, v in [("H", args.loop_hyper), ("C", args.loop_cart_anchor),
           ("Q", args.loop_deq), ("F", args.loop_fp_halt), ("A", args.loop_adaptive_halt),
           ("R", args.loop_iter_readout)] if v) or "") + \
          (f"-{args.loop_gate}" if lk and args.loop_gate != "scalar" else "") + \
          ("-seedchain" if seed_chain else "") + ("-engram" if lmb is not None else "") + \
          ("-de" + ("h" if args.de_mode == "hidden" else "") + (str(args.de_dim) if args.de_dim else "")
           + ("s" if args.de_shift and args.de_mode == "hidden" else "")
           + ("r" if args.de_emb_res and args.de_mode == "hidden" else "")
           if args.deepembed and not args.init_g1g else "") + \
          ("-mixctx" if args.ctx_buckets else "")
    print(f"model {tag}: {nparam/1e6:.1f}M params  loop_kw={lk}", flush=True)
    json.dump({"loop_count": args.loop_count, "n_layers": args.n_layers, "mode": tag,
               "seed_chain": seed_chain, "engram": lmb is not None,
               "deepembed": bool(args.deepembed) and not args.init_g1g,
               "mixed_ctx": bool(args.ctx_buckets),
               "params_m": round(nparam / 1e6, 2)}, open(os.path.join(args.out, "loop_rw.json"), "w"))

    # Mixed context-length training: packed [rows, T] buckets at standard context sizes with
    # RECIPROCAL batch scaling — B_bucket = (batch * seq_len) / T_bucket — so a short-context step
    # runs high-batch and a long-context step low-batch, holding tokens/step and activation VRAM
    # (~B*T) roughly constant. Buckets are sampled ∝ their real (non-pad) train tokens; pad (0)
    # is masked out of the loss. Per-bucket val loss doubles as a length-generalization readout.
    buckets = None
    if args.ctx_buckets:
        meta = json.load(open(args.ctx_buckets))
        buckets = []
        ref_tok = args.batch * args.seq_len              # token budget per micro-step
        vb = max(1, args.val_windows // max(len(meta["buckets"]), 1))
        tot_gb = sum(b["rows"] * b["T"] for b in meta["buckets"]) * 4 / 1e9
        # engram's recall runs on CPU: keep rows CPU-side so the prefetch thread reads them free
        bkt_gpu = args.gpu_data != "off" and tot_gb <= args.gpu_data_cap_gb and not args.engram
        for b in meta["buckets"]:
            arr = np.fromfile(b["bin"], dtype=np.uint16).astype(np.int32).reshape(b["rows"], b["T"])
            t = torch.from_numpy(arr)
            if bkt_gpu:
                t = t.to(dev)
            n_val_rows = min(vb, max(1, b["rows"] // 10))
            buckets.append({"T": b["T"], "rows": b["rows"], "data": t, "n_val": n_val_rows,
                            "B": max(1, round(ref_tok / (b["T"] - 1))),
                            "w": b["real_tokens"] * (b["rows"] - n_val_rows) / b["rows"]})
        bprobs = np.array([b["w"] for b in buckets]); bprobs = bprobs / bprobs.sum()
        print("mixed-ctx: " + "  ".join(f"ctx{b['T']}xB{b['B']}({p*100:.0f}%)"
                                        for b, p in zip(buckets, bprobs))
              + f"  [{'GPU' if bkt_gpu else 'CPU'} {tot_gb:.2f} GB, budget {ref_tok} tok/step]", flush=True)
        toks = train_toks = np.zeros(0, dtype=np.uint16)  # flat structures unused in bucket mode
        val_toks, train_docs = toks, None
    else:
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

    def batch_cpu(src, n, width=T + 1, sampler_rng=None):
        rgen = rng if sampler_rng is None else sampler_rng
        s = rgen.integers(0, len(src) - width, size=n)
        x = np.stack([np.asarray(src[i:i + width], dtype=np.int64) for i in s])
        return torch.from_numpy(x)

    def batch(src, n, width=T + 1):
        return batch_cpu(src, n, width).to(dev)

    def train_batch_cpu(n, width=T + 1, sampler_rng=None):
        rgen = rng if sampler_rng is None else sampler_rng
        if not train_docs:                                    # flat fallback
            return batch_cpu(train_toks, n, width, sampler_rng=rgen)
        rows = []
        for _ in range(n):
            s, e = train_docs[int(rgen.integers(0, len(train_docs)))]
            i = int(rgen.integers(s, e - width + 1))
            rows.append(np.asarray(toks[i:i + width], dtype=np.int64))
        return torch.from_numpy(np.stack(rows))

    def train_batch(n, width=T + 1):
        return train_batch_cpu(n, width).to(dev)

    # EMA shadow weights, fp32 (bf16 ULP would swallow (1-decay)-sized updates at high decay).
    # Eval and checkpoints carry the EMA copy; the live weights keep training unperturbed.
    ema = {n: p.detach().float().clone() for n, p in model.named_parameters()} if args.ema > 0 else None
    if ema is not None:
        if not args.ema < 1.0:
            raise ValueError(f"--ema must be in (0, 1), got {args.ema}")
        print(f"ema: decay {args.ema} — eval + checkpoint use the EMA weights", flush=True)

    def ema_update():
        with torch.no_grad():
            for n, p in model.named_parameters():
                ema[n].mul_(args.ema).add_(p.float(), alpha=1.0 - args.ema)

    def ema_swap():
        """Swap EMA weights in for eval; returns the live backup for ema_restore."""
        backup = {n: p.detach().clone() for n, p in model.named_parameters()}
        with torch.no_grad():
            for n, p in model.named_parameters():
                p.copy_(ema[n].to(p.dtype))
        return backup

    def ema_restore(backup):
        with torch.no_grad():
            for n, p in model.named_parameters():
                p.copy_(backup[n])

    def val_loss():
        model.eval()
        bak = ema_swap() if ema is not None else None
        with torch.no_grad():
            tot = 0.0
            for i in range(0, args.val_windows, args.batch):
                xc = batch_cpu(val_toks, min(args.batch, args.val_windows - i))
                recall = None
                if lmb is not None:
                    from rwkv_lab.engram_lmb import token_rosa_recall, RecallResult
                    recall = token_rosa_recall(xc[:, :T], 65536, lmb.boundary_id)
                    recall = RecallResult(*(v.to(dev) for v in recall))
                x = xc.to(dev)
                lg = (model(x[:, :T], precomputed_recall=recall)
                      if recall is not None else model(x[:, :T])).float()
                tot += F.cross_entropy(lg.reshape(-1, lg.size(-1)), x[:, 1:T + 1].reshape(-1)).item()
        if bak is not None:
            ema_restore(bak)
        model.train()
        return tot / math.ceil(args.val_windows / args.batch)

    if buckets is not None:
        def val_loss():  # noqa: F811 — bucket mode: token-weighted CE over each bucket's held-out rows
            model.eval()
            bak = ema_swap() if ema is not None else None
            tot = 0.0; cnt = 0; per = []
            with torch.no_grad():
                for b in buckets:
                    nll = 0.0; n = 0
                    for i in range(0, b["n_val"], b["B"]):
                        xc = b["data"][i:i + b["B"]].long()
                        recall = None
                        if lmb is not None:   # rows are CPU-side when engram is on
                            from rwkv_lab.engram_lmb import token_rosa_recall, RecallResult
                            xi = xc[:, :-1].cpu()
                            rr = token_rosa_recall(xi, 65536, lmb.boundary_id)
                            recall = RecallResult(rr.recalled, rr.valid & (xi != 0), rr.mlen, rr.dist)
                            recall = RecallResult(*(v.to(dev) for v in recall))
                        x = xc if xc.is_cuda else xc.to(dev)
                        tgt = x[:, 1:]
                        lg = (model(x[:, :-1], precomputed_recall=recall)
                              if recall is not None else model(x[:, :-1])).float()
                        nll += float(F.cross_entropy(lg.reshape(-1, lg.size(-1)), tgt.reshape(-1),
                                                     ignore_index=0, reduction="sum"))
                        n += int((tgt != 0).sum())
                    per.append((b["T"], nll / max(n, 1))); tot += nll; cnt += n
            print("  val/ctx  " + "  ".join(f"{T}: {v:.4f}" for T, v in per), flush=True)
            if bak is not None:
                ema_restore(bak)
            model.train()
            return tot / max(cnt, 1)

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
    step = 0; resume_recall_rng = None
    if args.resume and os.path.exists(args.resume):
        ck = torch.load(args.resume, map_location=dev, weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); step = ck.get("step", 0)
        if heads is not None and ck.get("heads") is not None:
            heads.load_state_dict(ck["heads"])
        if ema is not None:                  # saved EMA if present, else re-seed from loaded weights
            src = ck.get("ema") or {}
            for n, p in model.named_parameters():
                ema[n] = src[n].float().to(dev) if n in src else p.detach().float().clone()
        if ck.get("numpy_rng") is not None: rng.bit_generator.state = ck["numpy_rng"]
        if ck.get("torch_rng") is not None: torch.set_rng_state(ck["torch_rng"].cpu())
        if ck.get("cuda_rng") is not None: torch.cuda.set_rng_state(ck["cuda_rng"].cpu(), device=dev)
        resume_recall_rng = ck.get("recall_numpy_rng")
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
    use_gpu_data = buckets is None and (args.gpu_data == "on"
                                        or (args.gpu_data == "auto" and gpu_gb <= args.gpu_data_cap_gb
                                            and len(train_toks) > width))
    if use_gpu_data and lmb is not None:
        # Engram's exact suffix automaton is CPU-side. Sampling the ids on GPU
        # would immediately copy them back and serialize the stream, so let the
        # recall worker sample both ids and recall together from the memmap.
        use_gpu_data = False
        print("gpu-data: disabled for Engram; CPU recall prefetch owns window sampling", flush=True)
    if buckets is not None:
        def sample_train():   # pick a bucket ∝ real tokens; reciprocal batch keeps tok/step flat
            b = buckets[int(rng.choice(len(buckets), p=bprobs))]
            rows = torch.randint(b["n_val"], b["rows"], (b["B"],), device=b["data"].device)
            x = b["data"][rows]
            return (x if x.is_cuda else x.to(dev)).long()
    elif use_gpu_data:
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
    recall_pool = None
    if lmb is not None and not use_gpu_data:
        # Build token-SAM recall from the original CPU window one step ahead.
        # This removes the GPU->CPU ids copy and overlaps Numba with the current
        # GPU step; pinned tensors make both ids and recall uploads asynchronous.
        from concurrent.futures import ThreadPoolExecutor
        from rwkv_lab.engram_lmb import token_rosa_recall, RecallResult
        recall_pool = ThreadPoolExecutor(max_workers=1)
        recall_rng = np.random.default_rng(args.seed + 1009)
        if resume_recall_rng is not None:
            recall_rng.bit_generator.state = resume_recall_rng

        if buckets is not None:
            def _prefetch_engram():
                # bucket-aware: sample a bucket ∝ real tokens, then rows from its CPU tensor.
                # Recall is width-agnostic; pad tails (0) never recall — mask their validity.
                b = buckets[int(recall_rng.choice(len(buckets), p=bprobs))]
                ridx = torch.from_numpy(recall_rng.integers(b["n_val"], b["rows"], size=b["B"]))
                ids = b["data"][ridx].long()
                xin = ids[:, :-1]
                rr = token_rosa_recall(xin, 65536, lmb.boundary_id)
                rr = RecallResult(rr.recalled, rr.valid & (xin != 0), rr.mlen, rr.dist)
                return ids.pin_memory(), RecallResult(*(v.pin_memory() for v in rr))
        else:
            def _prefetch_engram():
                ids = train_batch_cpu(args.batch, width=width, sampler_rng=recall_rng)
                rr = token_rosa_recall(ids[:, :T], 65536, lmb.boundary_id)
                return ids.pin_memory(), RecallResult(*(v.pin_memory() for v in rr))

        recall_future = recall_pool.submit(_prefetch_engram)

        def sample_train():
            nonlocal recall_future
            ids, rr = recall_future.result()
            recall_future = recall_pool.submit(_prefetch_engram)
            return (ids.to(dev, non_blocking=True),
                    RecallResult(*(v.to(dev, non_blocking=True) for v in rr)))
        print("engram recall: CPU-prefetched one step ahead (pinned async H2D)", flush=True)
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
        if lmb is not None:                  # ramp Engram injection in (gates learn on live recall)
            lmb.set_warmup(min(1.0, (step + 1) / max(args.engram_warmup, 1)))
        opt.zero_grad(set_to_none=True)
        ga = max(args.grad_accum, 1)
        for _ in range(ga):                  # grad accumulation: effective batch = batch * ga
            sample = sample_train()
            x, precomputed_recall = sample if isinstance(sample, tuple) else (sample, None)
            # Mixed-ctx rows are exactly T_bucket wide (pad-masked); flat windows are T+1(+extra).
            xin, tgt = (x[:, :-1], x[:, 1:]) if buckets is not None else (x[:, :T], x[:, 1:T + 1])
            # The ordinary path skips RWKV7Small's full vocabulary output and lets
            # fused CE reuse its bf16 logit allocation during backward. Engram's
            # sparse copy-head mutates logits, so it retains the compatible path.
            if lmb is None:
                hidden = fwd(xin, hidden_only=True)
                from rwkv_lab.fused_ce import lmhead_cross_entropy
                loss = lmhead_cross_entropy(hidden, model.head, tgt, fused=True,
                                            ignore_index=(0 if buckets is not None else None))
                out = (None, hidden) if heads else None
            else:
                out = fwd(xin, return_hidden=bool(heads),
                          precomputed_recall=precomputed_recall)
                lg = (out[0] if heads else out).float()
                loss = F.cross_entropy(lg.reshape(-1, lg.size(-1)), tgt.reshape(-1),
                                       ignore_index=(0 if buckets is not None else -100))
            if heads:                                        # + weighted aux (latent-prediction) loss
                loss = loss + heads.compute(out[1], x, model.emb, model.head)["aux_total"]
            (loss / ga if ga > 1 else loss).backward()
            seen += xin.shape[0] * xin.shape[1]
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step(); step += 1
        if ema is not None:
            ema_update()
        if step % args.log_every == 0:
            emit({"kind": "train", "step": step, "loss": float(loss), "gnorm": float(gn),
                  "lr": lr, "tok_per_sec": int(seen / max(time.time() - t0, 1e-6))})
    vl = val_loss(); emit({"kind": "eval", "step": step, "loss": vl, "val_loss": vl, "ppl": math.exp(vl)})
    if recall_pool is not None:
        recall_pool.shutdown(wait=False, cancel_futures=True)
    emit({"kind": "checkpoint", "step": step})
    if args.save:
        blob = {"model": model.state_dict(), "opt": opt.state_dict(), "step": step, "config": tag,
                "heads": heads.state_dict() if heads is not None else None,
                "numpy_rng": rng.bit_generator.state, "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state(dev).cpu(),
                "recall_numpy_rng": (recall_rng.bit_generator.state if lmb is not None and not use_gpu_data
                                      else None)}
        if ema is not None:
            blob["ema"] = ema
        torch.save(blob, args.save)
        print(f"saved -> {args.save}", flush=True)
    print(f"DONE {tag}: {step} steps, final val {vl:.4f} (ppl {math.exp(vl):.2f})", flush=True)


if __name__ == "__main__":
    main()
