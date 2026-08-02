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

import argparse
import json
import math
import os
import time
from collections.abc import Sequence
from contextlib import nullcontext
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from rwkv_lab.lookahead_module import LookaheadSystem
from rwkv_lab.looped_rwkv import LoopedRWKV
from rwkv_lab.optimizer_speedups import (
    TailEMA,
    split_tied_embedding_head,
    tail_linear_multiplier,
    tie_embedding_head,
)
from rwkv_lab.rwkv8_deltanet import RWKV8ChannelMixDeltaNet, RWKV8TimeMixDeltaNet
from rwkv_lab.training_components import (
    AdamWConfiguration,
    OptimizerImplementation,
    PowerCoolConfiguration,
    ScheduleImplementation,
    build_registered_optimizer,
    powercool_multiplier,
)

if TYPE_CHECKING:
    from rwkv_lab.trainvm_adapters import WorkerTrainingComponents
    from rwkv_lab.trainvm_worker import WorkerObservability, WorkerStepProfiler
from rwkv_lab.training_speedups import (
    AsyncCPUBatchPrefetcher,
    context_batch_for_step,
    parse_context_curriculum,
)


def _unwrap(o):
    return o[0] if isinstance(o, tuple) else o


class Block(nn.Module):
    def __init__(self, d, n_heads, head_size, i, n_layers, loop_kw, att_kw=None, ffn_hidden=None,
                 de_vocab=0, de_dim=0, de_mode="out", de_shift=False, de_emb_res=False,
                 routing_free_kw=None):
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
        if routing_free_kw:
            # Liu et al. (2026), https://arxiv.org/abs/2604.00801 and official code:
            # https://github.com/liuyilun2000/RoutingFreeMoE/tree/release
            from rwkv_lab.routing_free_moe import RoutingFreeMoE
            self.ffn = RoutingFreeMoE(d, hidden_dim=ffn_hidden, **routing_free_kw)
        else:
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

    def forward(self, x, v_first, seed=None, return_seed=False, ids=None, e0=None,
                reset_mask=None):
        if (self.i == 0 and not getattr(self, "_megakernel_skip_ln0", False)
                and not getattr(self, "_megakernel_skip_ln0_permanent", False)):
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
            a, v_first = self.att(self.ln1(x), v_first=v_first, return_v_first=True,
                                  reset_mask=reset_mask)
        x = x + a
        xin = self.ln2(x)
        if self.de_mode == "hidden" and self.de_emb is not None and ids is not None:
            h = xin
            if self.de_xs is not None:                       # separate DE token-shift
                hp = torch.zeros_like(h)
                hp[:, 1:] = h[:, :-1]
                if reset_mask is not None:
                    hp = hp.masked_fill(reset_mask[..., None], 0.0)
                h = h + (hp - h) * self.de_xs.to(h.dtype)
            E = self.de_emb(ids)
            if self.de_er is not None and e0 is not None:    # emb-residual fold
                E = E + self.de_er(e0)
            B, T = ids.shape
            ss = torch.einsum("btr,btrs->bts", self.de_s1(h),
                              E.view(B, T, self.de_r, self.de_r).to(h.dtype))
            gate = 1.0 + self.de_s0.to(h.dtype) + self.de_s2(ss)
            f = _unwrap(self.ffn(xin, hidden_gate=gate, reset_mask=reset_mask))
        else:
            f = _unwrap(self.ffn(xin, reset_mask=reset_mask))
            if self.de_emb is not None and ids is not None:  # v1: gate the FFN output
                g = self.de_emb(ids)
                if self.de_proj is not None:
                    g = self.de_proj(g)
                f = f * (1.0 + g)
        x = x + f
        if return_seed:
            return x, v_first, seed_out
        return x, v_first

    def forward_recurrent(self, x, v_first, state=None, *, ids=None, e0=None,
                          return_ffn_parts=False):
        """Process one causal chunk and return exact time/shift carries.

        RWKV-7's matrix state is constant-size (Peng et al., 2025,
        https://arxiv.org/abs/2503.14456). Keeping both that state and the
        TimeMix/ChannelMix token shifts makes chunked decoding equivalent to a
        full-prefix forward for native, causal blocks.
        """
        if isinstance(self.att, LoopedRWKV):
            raise ValueError("recurrent decoding does not support looped attention")
        state = state or {}
        if (self.i == 0 and not getattr(self, "_megakernel_skip_ln0", False)
                and not getattr(self, "_megakernel_skip_ln0_permanent", False)):
            x = self.ln0(x)
        fused_boundaries = (
            getattr(self, "_megakernel_boundaries", False)
            and x.shape[1] == 1 and x.is_cuda and not torch.is_grad_enabled()
            and isinstance(self.att, RWKV8TimeMixDeltaNet)
            and isinstance(self.ffn, RWKV8ChannelMixDeltaNet)
        )
        if fused_boundaries:
            from rwkv_lab.megakernel_ops import layer_norm_six_mix
            previous = state.get("att_shift")
            if previous is None:
                previous = torch.zeros_like(x)
            coefficients = torch.stack((
                self.att.x_r, self.att.x_w, self.att.x_k,
                self.att.x_v, self.att.x_a, self.att.x_g,
            )).reshape(6, x.shape[-1])
            time_mixed, normalized = layer_norm_six_mix(
                x, previous.to(x.dtype), coefficients,
                self.ln1.weight, self.ln1.bias, eps=self.ln1.eps)
            a, wkv_state, att_shift, v_first = self.att(
                normalized, v_first=v_first, return_v_first=True,
                initial_state=state.get("wkv"), return_state=True,
                _megakernel_time_mix=time_mixed,
                _megakernel_time_shift=normalized)
        else:
            a, wkv_state, att_shift, v_first = self.att(
                self.ln1(x), v_first=v_first, return_v_first=True,
                initial_state=state.get("wkv"), shift_state=state.get("att_shift"),
                return_state=True)
        if fused_boundaries:
            from rwkv_lab.megakernel_ops import residual_add_layer_norm_channel_mix
            previous = state.get("ffn_shift")
            if previous is None:
                previous = torch.zeros_like(x)
            x, channel_mixed, xin = residual_add_layer_norm_channel_mix(
                x, a, previous.to(x.dtype), self.ffn.x_k,
                self.ln2.weight, self.ln2.bias, eps=self.ln2.eps)
        else:
            x = x + a
            xin = self.ln2(x)
        de_shift = None
        if self.de_mode == "hidden" and self.de_emb is not None and ids is not None:
            h = xin
            if self.de_xs is not None:
                previous = torch.zeros_like(h)
                if state.get("de_shift") is not None:
                    previous[:, :1] = state["de_shift"].to(h.dtype).reshape(
                        h.shape[0], 1, h.shape[-1])
                if h.shape[1] > 1:
                    previous[:, 1:] = h[:, :-1]
                de_shift = h[:, -1:]
                h = h + (previous - h) * self.de_xs.to(h.dtype)
            E = self.de_emb(ids)
            if self.de_er is not None and e0 is not None:
                E = E + self.de_er(e0)
            B, T = ids.shape
            ss = torch.einsum("btr,btrs->bts", self.de_s1(h),
                              E.view(B, T, self.de_r, self.de_r).to(h.dtype))
            gate = 1.0 + self.de_s0.to(h.dtype) + self.de_s2(ss)
            f, ffn_shift = self.ffn(
                xin, hidden_gate=gate, shift_state=state.get("ffn_shift"),
                return_state=True,
                _megakernel_channel_mix=(channel_mixed if fused_boundaries else None))
        else:
            ffn_result = self.ffn(
                xin, shift_state=state.get("ffn_shift"), return_state=True,
                _megakernel_channel_mix=(channel_mixed if fused_boundaries else None))
            if isinstance(ffn_result, tuple):
                f, ffn_shift = ffn_result
            else:  # Routing-Free MoE is token-wise and has no shift carry.
                f, ffn_shift = ffn_result, None
            if self.de_emb is not None and ids is not None:
                g = self.de_emb(ids)
                if self.de_proj is not None:
                    g = self.de_proj(g)
                f = f * (1.0 + g)
        output = (x, f) if return_ffn_parts else x + f
        next_state = {"wkv": wkv_state, "att_shift": att_shift, "ffn_shift": ffn_shift}
        if de_shift is not None:
            next_state["de_shift"] = de_shift
        return output, v_first, next_state


class RWKV7Small(nn.Module):
    def __init__(self, vocab, d, n_layers, head_size, loop_kw, att_kw=None, ffn_hidden=None,
                 seed_chain=False, deepembed=False, de_dim=0, de_mode="out", de_shift=False,
                 de_emb_res=False, routing_free_kw=None):
        super().__init__()
        assert d % head_size == 0
        if seed_chain and loop_kw:
            raise ValueError("seed_chain (Future-Seed cross-layer state) is incompatible with loop "
                             "levers — run it without loops for a clean A/B")
        if de_mode not in ("out", "hidden"):
            raise ValueError(f"de_mode must be 'out' or 'hidden', got {de_mode!r}")
        if deepembed and routing_free_kw:
            raise ValueError("DeepEmbed and Routing-Free MoE are alternative FFN designs")
        self.seed_chain = seed_chain          # Future-Seed: layer L starts from layer L-1's final wkv state
        self.deepembed = deepembed            # DeepEmbed: per-layer per-token FFN gate (needs ids)
        self.de_emb_res = de_emb_res          # hidden-mode: blocks also need the raw token embedding
        self.emb = nn.Embedding(vocab, d)
        self.blocks = nn.ModuleList([Block(d, d // head_size, head_size, i, n_layers, loop_kw,
                                           att_kw, ffn_hidden,
                                           de_vocab=vocab if deepembed else 0, de_dim=de_dim,
                                           de_mode=de_mode, de_shift=de_shift, de_emb_res=de_emb_res,
                                           routing_free_kw=routing_free_kw)
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

    def forward(self, ids, return_hidden=False, hidden_only=False, reset_mask=None):
        if getattr(self, "state_offset_adapter", None) is not None:
            if reset_mask is not None or return_hidden or hidden_only:
                raise ValueError("state-offset tuning uses the recurrent LM-loss path without "
                                 "packing or auxiliary hidden-state objectives")
            return self.forward_recurrent(ids)[0]
        if reset_mask is not None:
            reason = self.packing_incompatibility()
            if reason:
                raise ValueError(f"checkpoint is not reset-mask packing compatible: {reason}")
            if reset_mask.shape != ids.shape or not torch.all(reset_mask[:, 0]):
                raise ValueError("reset_mask must match ids and reset the first token")
        folded_embedding = getattr(self, "_megakernel_folded_embedding", None)
        use_folded_embedding = (getattr(self, "_megakernel_use_folded_embedding", False)
                                and folded_embedding is not None)
        if use_folded_embedding:
            x = F.embedding(ids, folded_embedding)
        else:
            x = self.emb(ids)
        de_ids = ids if self.deepembed else None  # DeepEmbed blocks gate their FFN by token id
        e0 = None
        if self.deepembed and self.de_emb_res:     # DeepEmbed residual requires the raw embedding
            e0 = self.emb(ids) if use_folded_embedding else x
        v_first = None                           # layer 0 sets it; later layers lerp toward it
        seed = None                              # Future-Seed: s_T of layer L-1 -> s_0 of layer L (None => 0)
        for j, b in enumerate(self.blocks):
            if self.seed_chain and j < len(self.blocks) - 1:
                x, v_first, seed = b(x, v_first, seed=seed, return_seed=True, ids=de_ids, e0=e0)
            else:                                # last layer consumes the seed but skips the unused s_T
                x, v_first = b(x, v_first, seed=seed, ids=de_ids, e0=e0,
                               reset_mask=reset_mask)
        memory = getattr(self, "online_memory", None)
        if memory is not None:
            kernel = getattr(self, "_online_memory_kernel", None)
            if kernel is None:
                x = memory(x, record_stats=False)
            else:
                state = memory.initial_state(x.shape[0], device=x.device)
                x, _, _ = kernel(x, state.memory, state.momentum)
        h = self.ln_out(x)                       # post-norm final hidden (what aux heads read)
        if hidden_only:
            return h
        logits = self.head(h)
        eng = getattr(self, "engram", None)      # copy head: gated bonus on the recalled token
        if eng is not None:                      # (exact no-op at init; see enable_engram)
            logits = eng.logit_bias(logits)
        return (logits, h) if return_hidden else logits

    def packing_incompatibility(self) -> str | None:
        """Return why exact reset-mask sequence packing is unavailable."""
        if getattr(self, "state_offset_adapter", None) is not None:
            return "state offsets require explicit per-sequence recurrent state"
        if self.seed_chain:
            return "Future-Seed depends on a whole-layer final state"
        if any(isinstance(block.att, LoopedRWKV) for block in self.blocks):
            return "looped attention has no per-segment state reset contract"
        if getattr(self, "online_memory", None) is not None:
            return "online memory has no per-segment reset contract"
        if getattr(self, "engram", None) is not None:
            return "Engram recall indexes complete token history"
        return None

    def recurrent_incompatibility(self) -> str | None:
        """Return why exact state-carrying generation is unavailable, if any."""
        if self.seed_chain:
            return "Future-Seed depends on a whole-layer final state"
        if any(isinstance(block.att, LoopedRWKV) for block in self.blocks):
            return "looped attention has no unambiguous cross-chunk state contract"
        if getattr(self, "engram", None) is not None:
            return "Engram copy recall requires the complete token history"
        return None

    def forward_recurrent(self, ids, state=None):
        """Exact causal chunk forward for native RWKV checkpoints.

        Unsupported experimental levers are rejected explicitly so rollout
        code can choose its batched full-prefix fallback without silently
        changing semantics.
        """
        reason = self.recurrent_incompatibility()
        if reason:
            raise ValueError(f"checkpoint is not recurrent-generation compatible: {reason}")
        if ids.ndim != 2 or ids.shape[1] < 1:
            raise ValueError("recurrent input must have shape [batch,time] with time >= 1")
        offset_adapter = getattr(self, "state_offset_adapter", None)
        if offset_adapter is not None and ids.shape[1] > 1:
            # State-offset Tuning modifies state at every timestep (Kang et al.,
            # ACL 2025, https://arxiv.org/abs/2503.03499). Splitting a chunk is
            # the slow, exact oracle; a fused scan must qualify against it.
            outputs = []
            for position in range(ids.shape[1]):
                logits, state = self.forward_recurrent(ids[:, position:position + 1], state)
                outputs.append(logits)
            return torch.cat(outputs, dim=1), state
        memory = getattr(self, "online_memory", None)
        if isinstance(state, dict):
            states = state.get("blocks")
            memory_state = state.get("online_memory")
        else:
            states = state
            memory_state = None
        state_adapter = getattr(self, "state_adapter", None)
        if states is None and state_adapter is not None:
            # RWKV-native state tuning; implementation and community sources:
            # src/rwkv_lab/state_tuning.py and https://github.com/wc2395082443-del/rwkv-rlhf
            states = state_adapter.expanded(ids.shape[0])
        if offset_adapter is not None:
            step = int(state.get("offset_step", 0)) if isinstance(state, dict) else 0
            states = offset_adapter.apply(states, ids.shape[0], step=step)
        states = states if states is not None else [None] * len(self.blocks)
        if len(states) != len(self.blocks):
            raise ValueError("recurrent state does not match model depth")
        folded_embedding = getattr(self, "_megakernel_folded_embedding", None)
        use_folded_embedding = (getattr(self, "_megakernel_use_folded_embedding", False)
                                and folded_embedding is not None)
        if use_folded_embedding:
            x = F.embedding(ids, folded_embedding)
        else:
            x = self.emb(ids)
        de_ids = ids if self.deepembed else None
        e0 = None
        if self.deepembed and self.de_emb_res:
            e0 = self.emb(ids) if use_folded_embedding else x
        v_first = None
        next_states = []
        fused_final_norm = (
            getattr(self, "_megakernel_final_norm", False)
            and ids.shape[1] == 1 and ids.is_cuda and not torch.is_grad_enabled()
        )
        prefused_hidden = None
        for index, (block, block_state) in enumerate(zip(self.blocks, states)):
            last = fused_final_norm and index == len(self.blocks) - 1
            x, v_first, block_state = block.forward_recurrent(
                x, v_first, block_state, ids=de_ids, e0=e0,
                return_ffn_parts=last)
            if last:
                from rwkv_lab.megakernel_ops import residual_add_layer_norm
                residual, update = x
                x, prefused_hidden = residual_add_layer_norm(
                    residual, update, self.ln_out.weight, self.ln_out.bias,
                    eps=self.ln_out.eps)
            next_states.append(block_state)
        if memory is not None:
            # Titans/MIRAS/ATLAS memory is itself a causal recurrent state.  Carrying
            # its matrix and momentum across chunks is exactly equivalent to the
            # full-prefix scan; see online_memory.py for the cited update rules.
            x, next_memory_state = memory(
                x, memory_state, return_state=True, record_stats=False)
            return self.head(self.ln_out(x)), {
                "blocks": next_states, "online_memory": next_memory_state,
                **({"offset_step": step + ids.shape[1]} if offset_adapter is not None else {}),
            }
        if offset_adapter is not None:
            return self.head(self.ln_out(x)), {"blocks": next_states,
                                               "offset_step": step + ids.shape[1]}
        hidden = prefused_hidden if prefused_hidden is not None else self.ln_out(x)
        if (getattr(self, "_megakernel_row_one", False) and ids.shape[1] == 1
                and not torch.is_grad_enabled()):
            from rwkv_lab.megakernel_linear import row_one_linear
            logits = row_one_linear(hidden, self.head.weight, use_candidate=True)
        else:
            logits = self.head(hidden)
        return logits, next_states


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


def build_optimizer(named_params, name, lr, wd, adam_lr=0.0, muon_opts=None,
                    u_mup_config=None, replicated_params=None,
                    worker_components: WorkerTrainingComponents | None = None):
    """AdamW, 8-bit AdamW (bitsandbytes), or spectral_muon (Muon on 2D weight matrices, AdamW on
    embeds/norms/1D). Shared by the LM and synthetic harnesses so the card's optimizer dropdown
    drives both. adam_lr (0 = use lr) is the fallback LR for non-matrix params under Muon — Muon
    matrix LRs run larger than AdamW's. muon_opts selects the Muon variant (spectral_power=Muon^p,
    ddc_strength=DDC, mona=Muon²/MONA, second_moment=Aurora, rsav, da_muon, aro, + scale/ns_steps)
    — passed straight to SpectralMuon. The 8-bit variants apply only to the AdamW path (the Muon
    fallback group is embeds/norms/1D — negligible state — so it stays fp32)."""
    named = [(n, p) for n, p in named_params if p.requires_grad]
    replicated_ids = {id(param) for param in (replicated_params or ())}
    if name == "muon":
        if u_mup_config is not None:
            raise ValueError("u-muP optimizer scaling currently supports AdamW-family optimizers")
        from rwkv_lab.spectral_muon import SpectralMuon
        muon, adam, adam_replicated = [], [], []
        for n, p in named:
            # engram tables/projections are embedding-like: always AdamW (Muon LR-unit trap)
            is_mat = p.ndim == 2 and not any(k in n for k in ("emb", "head", "norm", "engram"))
            if is_mat:
                muon.append(p)
            elif id(p) in replicated_ids:
                adam_replicated.append(p)
            else:
                adam.append(p)
        groups = [{"params": muon, "use_muon": True, "lr": lr},
                  {"params": adam, "use_muon": False, "lr": adam_lr or lr}]
        if adam_replicated:
            groups.append({"params": adam_replicated, "use_muon": False,
                           "lr": adam_lr or lr})
        return SpectralMuon(groups, weight_decay=wd, **(muon_opts or {}))
    if u_mup_config is not None:
        from rwkv_lab.u_mup import parameter_groups
        params = parameter_groups(named, lr=lr, weight_decay=wd, config=u_mup_config)
    else:
        ordinary = [p for _, p in named if id(p) not in replicated_ids]
        replicated = [p for _, p in named if id(p) in replicated_ids]
        params = ([{"params": ordinary}, {"params": replicated}]
                  if replicated else ordinary)
    if name in ("adamw8bit", "paged-adamw8bit"):
        return _adamw8bit(params, lr, wd, paged=(name == "paged-adamw8bit"))
    if worker_components is not None:
        return worker_components.optimizer(params)
    first = params[0]["params"][0] if params and isinstance(params[0], dict) else (params[0] if params else None)
    fused = first is not None and first.is_cuda     # fused AdamW = one fused CUDA kernel (CUDA-only)
    return build_registered_optimizer(
        OptimizerImplementation.TORCH_ADAMW_V1,
        params,
        AdamWConfiguration(
            learning_rate=lr,
            beta1=0.9,
            beta2=0.95,
            weight_decay=wd,
            foreach=False,
            fused=fused,
        ),
    )


def apply_fp8(module, *, include_head=False, skip_channelmix=False):
    """Swap eligible nn.Linear layers to torchao Float8Linear so their GEMMs run on the fp8
    tensor cores (Blackwell sm_120 / Hopper). This is orthogonal to the optimizer: bf16/fp32
    MASTER weights are kept and dynamically cast to fp8 per forward, so build_optimizer, the
    training loop, and checkpointing are all unchanged. Only converts linears whose in/out
    features are both multiples of 16 (the fp8 GEMM constraint) and skips the vocab head +
    embeddings (fp8 there costs quality for little FLOP). Returns #layers converted.

    Note: eager fp8 trains correctly but the throughput win needs torch.compile to fuse the
    cast+GEMM; without it fp8 can be net-neutral on small models. Clear error if torchao missing."""
    try:
        from torchao.float8 import Float8LinearConfig, convert_to_float8_training
        from torchao.float8.float8_linear import Float8Linear
    except Exception as e:  # noqa: BLE001 — surface the real cause (no wheel, bad CUDA, etc.)
        raise RuntimeError("fp8 training needs torchao: `uv pip install torchao`") from e
    import torch.nn as _nn

    def keep(m, fqn):
        # engram/DeepEmbed excluded: zero-init growth projections would fight fp8 dynamic scaling
        lower = fqn.lower()
        if not isinstance(m, _nn.Linear) or "engram" in lower or ".de_" in lower:
            return False
        if "head" in lower and not include_head:
            return False
        if skip_channelmix and (lower.endswith(".ffn.key") or lower.endswith(".ffn.value")):
            return False
        return m.in_features % 16 == 0 and m.out_features % 16 == 0

    # Rowwise scaling with high-precision weight-gradient accumulation is the
    # safer recipe for the large vocabulary projection. Older TorchAO releases
    # lack recipe construction, so retain their tensorwise default.
    try:
        config = Float8LinearConfig.from_recipe_name(
            "rowwise_with_gw_hp" if include_head else "tensorwise"
        )
        convert_to_float8_training(module, config=config, module_filter_fn=keep)
    except (AttributeError, TypeError):
        convert_to_float8_training(module, module_filter_fn=keep)
    return sum(isinstance(x, Float8Linear) for x in module.modules())


def enable_fused_channelmix(model, *, cached_fp8_up=False):
    """Enable the fused training block on every native ChannelMix module."""
    count = 0
    for module in model.modules():
        if isinstance(module, RWKV8ChannelMixDeltaNet):
            module.enable_fused_training(cached_fp8_up=cached_fp8_up)
            count += 1
    return count


def refresh_channelmix_fp8(model):
    """Refresh cached W1 copies after optimizer updates."""
    for module in model.modules():
        if isinstance(module, RWKV8ChannelMixDeltaNet):
            module.refresh_fp8_cache()


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
    ap.add_argument("--sm-row-update-floor", type=float, default=0.0,
                    help="minimum per-output-row ||update||/||weight|| after Muon")
    ap.add_argument("--sm-radial-brake", type=float, default=0.0,
                    help="outward radial-update multiplier in [0,1]; 0 disables")
    ap.add_argument("--sm-radius-pin", type=int, default=0,
                    help="remove finite tangential-step norm drift after Muon")
    ap.add_argument("--sm-cautious-wd", type=int, default=0,
                    help="apply WD only where the Muon update already shrinks a coordinate")
    ap.add_argument("--muon-adam-interval", type=int, default=1,
                    help="average fallback-Adam gradients and update every N Muon steps")
    for f in ["mona", "second-moment", "rsav", "da-muon", "aro", "batched", "compile-ns"]:
        ap.add_argument(f"--sm-{f}", type=int, default=0)
    ap.add_argument("--sm-aro-compile", type=int, default=0)


def muon_opts_from(a):
    return dict(scale=a.sm_scale, spectral_power=a.sm_spectral_power, ddc_strength=a.sm_ddc_strength,
                ns_steps=a.sm_ns_steps, tile_size=a.sm_tile_size, plus_norm=a.sm_plus_norm,
                mona=bool(a.sm_mona), second_moment=bool(a.sm_second_moment), rsav=bool(a.sm_rsav),
                da_muon=bool(a.sm_da_muon), aro=bool(a.sm_aro),
                aro_compile=bool(a.sm_aro_compile), batched=bool(a.sm_batched),
                compile_ns=bool(a.sm_compile_ns),
                row_update_floor=a.sm_row_update_floor,
                radial_brake=a.sm_radial_brake,
                radius_pin=bool(a.sm_radius_pin),
                cautious_weight_decay=bool(a.sm_cautious_wd),
                adam_update_interval=a.muon_adam_interval)


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


def resolved_worker_component_contract(
    args: argparse.Namespace,
    powercool_configuration: PowerCoolConfiguration | None,
    worker_components: WorkerTrainingComponents | None,
) -> tuple[
    PowerCoolConfiguration | None,
    dict[str, dict[str, str]] | None,
    str | None,
]:
    if worker_components is None:
        return powercool_configuration, None, None
    if args.optimizer != "adamw":
        raise ValueError("RWKV worker composition currently requires AdamW")
    if args.lr_schedule != "powercool" or powercool_configuration is None:
        raise ValueError("RWKV worker composition currently requires PowerCool")
    if args.u_mup_base_width:
        raise ValueError("RWKV worker composition does not yet encode u-muP routing")
    if args.distributed != "none":
        raise ValueError(
            "RWKV worker composition does not yet encode distributed gradient clipping"
        )
    optimizer_configuration = dict(
        worker_components.configuration("optimizer", category="optimizer")
    )
    expected_optimizer = {
        "learning_rate": args.lr,
        "beta1": 0.9,
        "beta2": 0.95,
        "epsilon": 1.0e-8,
    }
    if any(
        optimizer_configuration.get(name) != value
        for name, value in expected_optimizer.items()
    ):
        raise ValueError(
            "authority optimizer composition disagrees with RWKV configuration"
        )
    if dict(
        worker_components.configuration(
            "weight_decay", category="weight_decay_schedule"
        )
    ) != {"weight_decay": args.weight_decay}:
        raise ValueError(
            "authority weight-decay composition disagrees with RWKV configuration"
        )
    clipping_configuration = dict(
        worker_components.configuration(
            "gradient_clipping", category="gradient_clipping"
        )
    )
    if clipping_configuration != {
        "max_norm": args.grad_clip,
        "norm_type": 2.0,
        "error_if_nonfinite": False,
    }:
        raise ValueError(
            "authority gradient-clipping composition disagrees with RWKV configuration"
        )
    if dict(
        worker_components.configuration(
            "gradient_accumulation", category="gradient_accumulation"
        )
    ) != {"microbatches_per_optimizer_step": args.grad_accum}:
        raise ValueError(
            "authority gradient-accumulation composition disagrees with RWKV configuration"
        )
    if dict(
        worker_components.configuration("objective", category="objective")
    ) != {"chunk_size": 2048, "prefer_fused": True}:
        raise ValueError(
            "authority objective composition disagrees with RWKV configuration"
        )
    if dict(
        worker_components.configuration("precision", category="precision")
    ) != {
        "parameter_dtype": "bfloat16",
        "compute_dtype": "bfloat16",
        "reduction_dtype": "float32",
        "gradient_scaling": False,
    }:
        raise ValueError(
            "authority precision composition disagrees with RWKV configuration"
        )
    implementation, resolved_schedule = (
        worker_components.learning_rate_configuration()
    )
    if (
        implementation is not ScheduleImplementation.POWERCOOL_V1
        or resolved_schedule != powercool_configuration
    ):
        raise ValueError(
            "authority LR-schedule composition disagrees with RWKV configuration"
        )
    return (
        resolved_schedule,
        dict(worker_components.evidence()),
        worker_components.composition.composition_digest,
    )


def main(
    argv: Sequence[str] | None = None,
    *,
    worker_components: WorkerTrainingComponents | None = None,
    worker_step_profiler: WorkerStepProfiler | None = None,
    worker_observability: WorkerObservability | None = None,
):
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
    ap.add_argument("--cpu-prefetch", action=argparse.BooleanOptionalAction, default=True,
                    help="build and pin one ordinary memmap batch ahead (exact-resume safe)")
    ap.add_argument("--ctx-curriculum", default="",
                    help="opt-in fraction:seq_len stages, e.g. 0:256,0.33:512,0.67:1024; "
                         "batch scales reciprocally to keep tokens/update constant")
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
    ap.add_argument("--fp8-head", action="store_true",
                    help="include the vocabulary projection in TorchAO FP8 training")
    ap.add_argument("--fused-channelmix", action="store_true",
                    help="custom-autograd fused Linear->ReLU²->Linear ChannelMix training path")
    ap.add_argument("--cached-fp8-up", action="store_true",
                    help="quantize ChannelMix W1 once per optimizer step; BF16 backward")
    ap.add_argument("--nvfp4", action="store_true",
                    help="NVFP4 QAT; backend defaults to the portable fake-quant oracle")
    ap.add_argument("--nvfp4-backend", default="fake",
                    choices=("fake", "transformer_engine"),
                    help="native Transformer Engine is fail-closed on parity and throughput")
    ap.add_argument("--nvfp4-rht", action="store_true",
                    help="apply randomized Hadamard transforms before eligible NVFP4 GEMMs")
    ap.add_argument("--compile", action="store_true",
                    help="torch.compile the training forward (fuses fp8 cast+GEMM; ~2x on Blackwell)")
    ap.add_argument("--compile-fullgraph", action="store_true",
                    help="require a static full training-forward graph")
    ap.add_argument("--compile-prewarm", action=argparse.BooleanOptionalAction, default=True,
                    help="compile every context-curriculum shape before the training clock starts")
    ap.add_argument("--distributed", default="none", choices=["none", "fsdp2"],
                    help="torchrun backend; fsdp2 shards RWKV blocks, root params, optimizer, and checkpoints")
    ap.add_argument("--activation-checkpointing", action="store_true",
                    help="non-reentrant per-block activation checkpointing (works with FSDP2)")
    ap.add_argument("--cpu-offload", action="store_true",
                    help="FSDP2 parameter/gradient CPU offload (requires --distributed fsdp2)")
    ap.add_argument("--fsdp-prefetch-depth", type=int, default=1,
                    help="explicit adjacent-block FSDP2 forward/backward prefetch depth (0 disables)")
    ap.add_argument("--fsdp-sparse-embeddings", action="store_true",
                    help="replicate embedding tables and synchronize only rows touched this update")
    add_muon_args(ap)
    ap.add_argument("--lr-schedule", default="cosine", choices=["constant", "cosine", "powercool"])
    ap.add_argument("--powercool-cooldown-fraction", type=float, default=0.20)
    ap.add_argument("--powercool-power", type=float, default=2.0)
    ap.add_argument("--powercool-min-lr", type=float, default=0.0)
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
    ap.add_argument("--u-mup-base-width", type=int, default=0,
                    help="enable u-muP scale transfer from this tuned base width (0 = off)")
    ap.add_argument("--u-mup-base-depth", type=int, default=1)
    ap.add_argument("--online-memory", type=int, default=0,
                    help="attach in-forward associative memory (0 = off, 1 = on)")
    ap.add_argument("--online-memory-mode", default="titans",
                    choices=["titans", "miras", "atlas", "nested"])
    ap.add_argument("--online-memory-dim", type=int, default=0)
    ap.add_argument("--online-memory-lr", type=float, default=0.05)
    ap.add_argument("--online-memory-retention", type=float, default=0.99)
    ap.add_argument("--online-memory-window", type=int, default=4)
    ap.add_argument("--online-memory-kernel", default="auto",
                    choices=("auto", "eager", "compile"),
                    help="parity/speed-gated compiled associative-memory scan")
    ap.add_argument("--balance-state", action="store_true",
                    help="QRWKV7 balanced write/forget recurrence for large-scale conversion stability")
    ap.add_argument("--state-offset", type=int, default=0,
                    help="ACL'25 State-offset Tuning oracle; freeze base and learn FP32 recurrent offsets")
    ap.add_argument("--state-offset-interval", type=int, default=1,
                    help="inject state offsets every N recurrent tokens (paper-faithful default: 1)")
    ap.add_argument("--routing-free-moe", type=int, default=0,
                    help="replace ChannelMix with router/softmax/top-k-free self-activating experts")
    ap.add_argument("--routing-free-experts", type=int, default=4)
    ap.add_argument("--routing-free-rank", type=int, default=32)
    ap.add_argument("--routing-free-threshold", type=float, default=0.2)
    ap.add_argument("--routing-free-balance", type=float, default=0.5,
                    help="0=token balancing, 1=expert balancing")
    ap.add_argument("--routing-free-aux-weight", type=float, default=0.01)
    ap.add_argument("--byte-aware-vocab", default="",
                    help="world-vocab file enabling zero-init token-length + UTF-8 byte-position embeddings")
    ap.add_argument("--byte-aware-max-bytes", type=int, default=16)
    ap.add_argument("--byte-aware-dim", type=int, default=0)
    ap.add_argument("--grad-accum", type=int, default=1,
                    help="micro-batches accumulated per optimizer step (effective batch = batch * N)")
    ap.add_argument("--ema", type=float, default=0.0,
                    help="EMA decay for a shadow weight copy (e.g. 0.999); eval + checkpoint carry it. 0 = off")
    ap.add_argument("--tail-ema-start", type=float, default=0.0,
                    help="fraction at which late partial-eval EMA starts; 0 disables")
    ap.add_argument("--tail-ema-horizon", type=int, default=150)
    ap.add_argument("--tail-ema-blend", type=float, default=0.6)
    ap.add_argument("--tie-head-until", type=float, default=0.0,
                    help="tie embedding/head until this fraction of training; 0 disables")
    ap.add_argument("--lmtp-cooldown-fraction", type=float, default=0.0,
                    help="linearly decay LMTP weight from this fraction to zero at the horizon")
    args = ap.parse_args(argv)
    if args.loop_iter_readout:
        ap.error("--loop-iter-readout is only supported in convert_train.py "
                 "(this harness implements no per-iteration readout loss)")
    if args.routing_free_moe and args.deepembed:
        ap.error("--routing-free-moe and --deepembed are alternative FFN designs")
    if not 0 <= args.routing_free_balance <= 1:
        ap.error("--routing-free-balance must be in [0,1]")
    if not args.nvfp4 and (args.nvfp4_rht or args.nvfp4_backend != "fake"):
        ap.error("--nvfp4-rht/--nvfp4-backend require --nvfp4")
    if args.fp8_head and not args.fp8:
        ap.error("--fp8-head requires --fp8")
    if args.cached_fp8_up and not args.fused_channelmix:
        ap.error("--cached-fp8-up requires --fused-channelmix")
    if args.compile_fullgraph and not args.compile:
        ap.error("--compile-fullgraph requires --compile")
    if args.fsdp_sparse_embeddings and args.distributed != "fsdp2":
        ap.error("--fsdp-sparse-embeddings requires --distributed fsdp2")
    if args.tie_head_until and not 0.0 < args.tie_head_until < 1.0:
        ap.error("--tie-head-until must be 0 (off) or a fraction in (0,1)")
    if args.tie_head_until and (args.distributed == "fsdp2" or args.fp8_head):
        ap.error("--tie-head-until is not compatible with FSDP2 or --fp8-head")
    if args.tie_head_until and args.init_g1g:
        ap.error("--tie-head-until currently requires the native from-scratch model")
    if args.tail_ema_start and not 0.0 < args.tail_ema_start < 1.0:
        ap.error("--tail-ema-start must be 0 (off) or a fraction in (0,1)")
    if not 0.0 <= args.tail_ema_blend <= 1.0 or args.tail_ema_horizon < 1:
        ap.error("Tail-EMA requires blend in [0,1] and a positive horizon")
    if args.ema and args.tail_ema_start:
        ap.error("--ema and --tail-ema-start are alternative readouts")
    if args.tail_ema_start and args.distributed == "fsdp2":
        ap.error("Tail-EMA is not yet compatible with sharded parameters")
    if args.lmtp_cooldown_fraction and not 0.0 < args.lmtp_cooldown_fraction < 1.0:
        ap.error("--lmtp-cooldown-fraction must be 0 (off) or a fraction in (0,1)")
    if args.lmtp_cooldown_fraction and args.lmtp_weight <= 0:
        ap.error("--lmtp-cooldown-fraction requires --lmtp-weight > 0")
    if args.muon_adam_interval < 1:
        ap.error("--muon-adam-interval must be >= 1")
    if not 0.0 <= args.sm_radial_brake <= 1.0:
        ap.error("--sm-radial-brake must be in [0,1]")
    if args.sm_row_update_floor < 0:
        ap.error("--sm-row-update-floor must be non-negative")
    if args.optimizer != "muon" and (
        args.muon_adam_interval != 1
        or args.sm_row_update_floor
        or args.sm_radial_brake
        or args.sm_radius_pin
        or args.sm_cautious_wd
    ):
        ap.error("Muon postconditioning and Adam cadence flags require --optimizer muon")
    if args.lr <= 0:
        ap.error("--lr must be positive")
    powercool_schedule = None
    if args.lr_schedule == "powercool":
        powercool_horizon = args.decay_steps or args.steps
        if powercool_horizon <= 0:
            ap.error("--lr-schedule powercool requires --steps or --decay-steps")
        try:
            powercool_schedule = PowerCoolConfiguration(
                warmup_steps=args.warmup,
                max_steps=powercool_horizon,
                minimum_ratio=args.powercool_min_lr / args.lr,
                cooldown_fraction=args.powercool_cooldown_fraction,
                power=args.powercool_power,
            )
        except (TypeError, ValueError) as error:
            ap.error(f"invalid PowerCool schedule: {error}")
    try:
        powercool_schedule, component_evidence, component_digest = (
            resolved_worker_component_contract(
                args, powercool_schedule, worker_components
            )
        )
    except ValueError as error:
        ap.error(str(error))
    precision_policy = (
        worker_components.precision()
        if worker_components is not None
        else None
    )
    parameter_dtype = (
        precision_policy.parameter_dtype
        if precision_policy is not None
        else torch.bfloat16
    )
    if not args.data and not args.ctx_buckets:
        ap.error("one of --data or --ctx-buckets is required")
    if args.ctx_buckets:                       # mixed-context mode: fixed-width features rejected
        if any(w > 0 for w in [args.nextlat_weight, args.top_weight, args.lmtp_weight,
                               args.bst_weight, args.jtp_weight]):
            ap.error("--ctx-buckets: aux lookahead heads are fixed-width — unsupported")
        if args.doc_offsets:
            print("warn: --doc-offsets ignored with --ctx-buckets (rows are already packed)", flush=True)
            args.doc_offsets = ""
    if args.ctx_curriculum and args.ctx_buckets:
        ap.error("--ctx-curriculum and --ctx-buckets are alternative context schedules")
    if args.ctx_curriculum and args.engram:
        ap.error("--ctx-curriculum is not yet compatible with fixed-width Engram recall prefetch")
    if args.ctx_curriculum and not (args.steps or args.decay_steps):
        ap.error("--ctx-curriculum requires --steps or --decay-steps as its horizon")
    if (args.tie_head_until or args.tail_ema_start or args.lmtp_cooldown_fraction) and not (
        args.steps or args.decay_steps
    ):
        ap.error("scheduled optimizer/readout features require --steps or --decay-steps")
    if args.fsdp_prefetch_depth < 0:
        ap.error("--fsdp-prefetch-depth must be non-negative")

    if args.cpu_offload and args.distributed != "fsdp2":
        ap.error("--cpu-offload requires --distributed fsdp2")
    if args.distributed == "fsdp2" and args.compile:
        ap.error("--compile + FSDP2 is not enabled until graph/checkpoint parity is validated")
    from rwkv_lab.distributed import DistributedContext, initialize
    dist = initialize() if args.distributed != "none" else DistributedContext(
        0, 1, 0, "cuda" if torch.cuda.is_available() else "cpu")
    if args.distributed == "fsdp2" and dist.world_size == 1:
        ap.error("--distributed fsdp2 must be launched with torchrun and WORLD_SIZE > 1")
    os.makedirs(args.out, exist_ok=True)
    jl = open(os.path.join(args.out, "train.jsonl"), "w", buffering=1) if dist.is_primary \
        else open(os.devnull, "w")
    emit = lambda r: jl.write(json.dumps(r) + "\n")

    def publish_metric(
        name: str,
        value: int | float,
        *,
        metric_step: int,
        sample_weight: float = 1.0,
    ) -> None:
        if worker_observability is not None:
            worker_observability.publish_if_declared(
                name,
                value,
                step=metric_step,
                sample_weight=sample_weight,
            )

    def publish_eval(metric_step: int, value: float) -> None:
        publish_metric("validation_loss", value, metric_step=metric_step)
        publish_metric("perplexity", math.exp(value), metric_step=metric_step)

    dev = dist.device; T = args.seq_len
    context_stages = parse_context_curriculum(
        args.ctx_curriculum, max_seq_len=T)
    context_horizon = args.decay_steps or args.steps
    resume_blob = None
    if args.resume and os.path.exists(args.resume) and args.distributed != "fsdp2":
        # The tie/split topology must be known before constructing the optimizer.
        # Reuse this CPU load later so large checkpoints are read only once.
        resume_blob = torch.load(args.resume, map_location="cpu", weights_only=False)
    resume_step_hint = int((resume_blob or {}).get("step", 0))
    split_step = (
        round(args.tie_head_until * context_horizon)
        if args.tie_head_until else 0
    )
    resume_head_tied = bool(
        (resume_blob or {}).get(
            "head_tied",
            bool(args.tie_head_until and resume_step_hint < split_step),
        )
    )
    if context_stages:
        schedule_text = ", ".join(
            f"{stage.start_fraction:g}:{stage.seq_len}" for stage in context_stages)
        print(f"context curriculum: {schedule_text}; "
              f"token budget={args.batch * T}/micro-step", flush=True)
    # Every rank must construct identical parameters before FSDP shards them. Data/dropout streams
    # diverge by rank only after construction (and are checkpointed per rank for exact resume).
    torch.manual_seed(args.seed); rng = np.random.default_rng(args.seed + dist.rank)

    lk = loop_kwargs(args)
    u_mup_cfg = None
    if args.init_g1g:                                        # continued pretraining from pretrained g1g
        from rwkv_lab.native_g1g import load_g1g_native, add_loops
        model, ginfo = load_g1g_native(args.init_g1g, device=dev)
        if lk:
            add_loops(model, lk)                             # levers attach identity-at-init
        model = (
            precision_policy.convert_module(model, dev)
            if precision_policy is not None
            else model.to(dev, parameter_dtype)
        )
        print(f"init from g1g {args.init_g1g}: loaded {ginfo['loaded']}/{ginfo['n_ckpt']} tensors "
              f"(dims forced to g1g 24L/d2048/h64)", flush=True)
        if args.seed_chain:
            print("warn: --seed-chain ignored for g1g init (from-scratch only)", flush=True)
        if args.deepembed:
            print("warn: --deepembed ignored for g1g init (from-scratch only)", flush=True)
        if args.online_memory or args.u_mup_base_width:
            print("warn: --online-memory/--u-mup ignored for g1g init (from-scratch only)", flush=True)
    else:
        model = RWKV7Small(65536, args.d_model, args.n_layers, args.head_size, lk,
                           att_kw={"balance_state": args.balance_state},
                           seed_chain=bool(args.seed_chain), deepembed=bool(args.deepembed),
                           de_dim=args.de_dim, de_mode=args.de_mode, de_shift=bool(args.de_shift),
                           de_emb_res=bool(args.de_emb_res),
                           routing_free_kw=({"n_experts": args.routing_free_experts,
                                             "rank": args.routing_free_rank,
                                             "threshold": args.routing_free_threshold,
                                             "balance_interpolation": args.routing_free_balance}
                                            if args.routing_free_moe else None))
        if args.byte_aware_vocab:
            from rwkv_lab.tokenizer_experiments import (install_byte_aware_embedding,
                                                         load_world_token_bytes)
            install_byte_aware_embedding(model, load_world_token_bytes(args.byte_aware_vocab),
                                         max_bytes=args.byte_aware_max_bytes,
                                         byte_dim=args.byte_aware_dim)
            print(f"byte-aware embeddings: max_bytes={args.byte_aware_max_bytes} "
                  f"dim={args.byte_aware_dim or args.d_model}", flush=True)
        if args.state_offset:
            if any(w > 0 for w in [args.nextlat_weight, args.top_weight, args.lmtp_weight,
                                   args.bst_weight, args.jtp_weight]):
                ap.error("--state-offset does not support auxiliary hidden-state objectives")
            from rwkv_lab.state_tuning import install_state_offset_adapter
            install_state_offset_adapter(model, interval=args.state_offset_interval)
            print(f"state-offset tuning: interval={args.state_offset_interval}; base frozen", flush=True)
        if args.online_memory:
            from rwkv_lab.online_memory import install_online_memory
            mem = install_online_memory(model, d_memory=(args.online_memory_dim or None),
                                        mode=args.online_memory_mode,
                                        learning_rate=args.online_memory_lr,
                                        retention=args.online_memory_retention,
                                        atlas_window=args.online_memory_window)
            print(f"online-memory: {mem.mode} d_memory={mem.d_memory}", flush=True)
        if args.u_mup_base_width:
            from rwkv_lab.u_mup import UMuPConfig, initialize_u_mup
            u_mup_cfg = UMuPConfig(args.u_mup_base_width, args.d_model, args.n_layers,
                                   args.u_mup_base_depth)
            initialize_u_mup(model, u_mup_cfg)
            print(f"u-muP: base d{args.u_mup_base_width}/L{args.u_mup_base_depth} -> "
                  f"d{args.d_model}/L{args.n_layers}", flush=True)
        model = (
            precision_policy.convert_module(model, dev)
            if precision_policy is not None
            else model.to(dev, parameter_dtype)
        )
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
    head_tied = False
    if args.tie_head_until and resume_head_tied:
        tie_embedding_head(model)
        head_tied = True
        print(f"embedding/head tied until step {split_step}", flush=True)
    if args.fused_channelmix:
        nfused = enable_fused_channelmix(
            model, cached_fp8_up=args.cached_fp8_up
        )
        print(
            f"fused ChannelMix: {nfused} blocks"
            + (" + cached FP8 up-projection" if args.cached_fp8_up else ""),
            flush=True,
        )
    if args.fp8:
        n8 = apply_fp8(
            model,
            include_head=args.fp8_head,
            skip_channelmix=args.fused_channelmix,
        )
        print(f"fp8: {n8} Linear layers -> Float8Linear (torchao)"
              + (" including vocabulary head" if args.fp8_head else ""), flush=True)
    if args.fused_channelmix:
        native_ffns = [
            module for module in model.modules()
            if isinstance(module, RWKV8ChannelMixDeltaNet)
        ]
        if native_ffns:
            from rwkv_lab.fused_channelmix import qualify_channelmix_training

            rng_before = torch.get_rng_state()
            cuda_before = (
                torch.cuda.get_rng_state(dev) if torch.cuda.is_available() else None
            )
            probe_tokens = max(128, min(2048, (args.batch * T // 128) * 128))
            probe = torch.randn(
                1, probe_tokens, args.d_model,
                device=dev, dtype=next(model.parameters()).dtype,
            )
            report = qualify_channelmix_training(
                native_ffns[0],
                probe,
                allow_cached_fp8=args.cached_fp8_up,
            )
            selected = report["adopted"]
            settings = {
                "eager": (False, False, False),
                "fused_bf16": (True, False, False),
                "triton_bf16": (True, False, True),
                "cached_fp8": (True, True, False),
                "triton_cached_fp8": (True, True, True),
            }[selected]
            for module in native_ffns:
                if settings[0]:
                    module.enable_fused_training(
                        cached_fp8_up=settings[1],
                        triton_fused=settings[2],
                    )
                else:
                    module._fused_training = False
                    module._cached_fp8_up = False
                    module._triton_fused_training = False
            torch.set_rng_state(rng_before)
            if cuda_before is not None:
                torch.cuda.set_rng_state(cuda_before, device=dev)
            emit({"kind": "kernel_qualification", "step": 0,
                  "report": {"kernel": "channelmix_training", **report}})
            print(
                f"ChannelMix qualification: {selected} "
                f"({report.get('speedup', 1.0):.2f}x; {report.get('times_ms', {})})",
                flush=True,
            )
    if args.nvfp4:
        if args.fp8:
            ap.error("--fp8 and --nvfp4 are mutually exclusive operand formats")
        from rwkv_lab.nvfp4 import convert_to_nvfp4_training, qualify_native_nvfp4
        if args.nvfp4_backend == "transformer_engine":
            representative = next((layer for layer in model.modules()
                                   if isinstance(layer, nn.Linear) and
                                   layer.in_features % 16 == 0 and
                                   layer.out_features % 16 == 0), None)
            if representative is None:
                ap.error("native NVFP4 found no aligned Linear to qualify")
            sample = torch.randn(8, representative.in_features, device=dev,
                                 dtype=representative.weight.dtype)
            report = qualify_native_nvfp4(
                representative, sample, rht=args.nvfp4_rht)
            if not report.get("adopted"):
                ap.error(f"native NVFP4 failed parity/performance qualification: {report}")
            emit({"kind": "kernel_qualification", "step": 0, "report": report})
        n4 = convert_to_nvfp4_training(
            model, rht=args.nvfp4_rht, backend=args.nvfp4_backend)
        print(f"nvfp4-{args.nvfp4_backend}: {n4} Linear layers"
              + (" + RHT" if args.nvfp4_rht else ""), flush=True)
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
    if args.online_memory and not args.init_g1g and args.online_memory_kernel != "eager":
        if args.compile or args.distributed == "fsdp2":
            message = ("owned by whole-model torch.compile" if args.compile
                       else "kept eager until FSDP2 graph parity is qualified")
            print(f"online-memory kernel: {message}", flush=True)
        else:
            from rwkv_lab.online_memory import (install_compiled_online_memory,
                                                 qualify_compiled_online_memory)
            probe = torch.randn(min(args.batch, 2), min(args.seq_len, 128), args.d_model,
                                device=dev, dtype=parameter_dtype)
            report = qualify_compiled_online_memory(
                model.online_memory, probe, tolerance=2e-2, repeats=3)
            emit({"kind": "kernel_qualification", "step": 0, "report": report})
            if report.get("adopted"):
                install_compiled_online_memory(model)
                print(f"online-memory kernel: compiled ({report['speedup']:.2f}x probe speedup)",
                      flush=True)
            elif args.online_memory_kernel == "compile":
                ap.error(f"compiled online memory failed parity/performance qualification: {report}")
            else:
                print(f"online-memory kernel: eager (compiled candidate rejected: {report})", flush=True)
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
          (f"-umup{args.u_mup_base_width}" if u_mup_cfg is not None else "") + \
          (f"-mem-{args.online_memory_mode}" if args.online_memory and not args.init_g1g else "") + \
          ("-stateoffset" if args.state_offset and not args.init_g1g else "") + \
          ("-rfmoe" if args.routing_free_moe and not args.init_g1g else "") + \
          ("-byteaware" if args.byte_aware_vocab and not args.init_g1g else "") + \
          ("-nvfp4" + ("rht" if args.nvfp4_rht else "")
           + ("-te" if args.nvfp4_backend == "transformer_engine" else "")
           if args.nvfp4 else "") + \
          ("-mixctx" if args.ctx_buckets else "")
    print(f"model {tag}: {nparam/1e6:.1f}M params  loop_kw={lk}", flush=True)
    json.dump({"loop_count": args.loop_count, "n_layers": args.n_layers, "mode": tag,
               "seed_chain": seed_chain, "engram": lmb is not None,
               "deepembed": bool(args.deepembed) and not args.init_g1g,
               "u_mup_base_width": args.u_mup_base_width if u_mup_cfg is not None else 0,
               "online_memory": args.online_memory_mode if args.online_memory and not args.init_g1g else "",
               "state_offset": bool(args.state_offset) and not args.init_g1g,
               "routing_free_moe": bool(args.routing_free_moe) and not args.init_g1g,
               "byte_aware": bool(args.byte_aware_vocab) and not args.init_g1g,
               "nvfp4": bool(args.nvfp4),
               "mixed_ctx": bool(args.ctx_buckets),
               "training_components": component_evidence,
               "component_composition_digest": component_digest,
               "params_m": round(nparam / 1e6, 2)}, open(os.path.join(args.out, "loop_rw.json"), "w"))

    heads = None
    if any(w > 0 for w in [args.nextlat_weight, args.top_weight, args.lmtp_weight,
                           args.bst_weight, args.jtp_weight]):
        heads = LookaheadSystem(args.d_model, 65536, nextlat_weight=args.nextlat_weight,
                                top_weight=args.top_weight, lmtp_weight=args.lmtp_weight,
                                bst_weight=args.bst_weight, jtp_weight=args.jtp_weight,
                                lm_head=model.head)
        heads = (
            precision_policy.convert_module(heads, dev)
            if precision_policy is not None
            else heads.to(dev, parameter_dtype)
        )
        print(f"aux heads enabled={heads.enabled} extra_tokens={heads.extra_tokens}", flush=True)
    # Widest window any train batch samples: T+1 targets plus the aux heads' future-token fetch.
    width = T + 1 + (heads.extra_tokens if heads else 0)

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
        n_val = args.val_windows * (T + 1)                 # eval windows are T+1 wide
        val_toks, train_toks = toks[:n_val], toks[n_val:]
        # FIXED val windows: deterministic evenly-spaced offsets, computed once — evals never
        # consume the training RNG and always score the same windows.
        val_offsets = np.linspace(0, len(val_toks) - (T + 1), args.val_windows).astype(np.int64)
        print(f"tokens: {len(toks)/1e6:.1f}M (val {len(val_toks)}, train {len(train_toks)/1e6:.1f}M)", flush=True)

    train_docs = None
    if args.doc_offsets:                                       # within-doc windows (no mid-doc cuts)
        allo = np.load(args.doc_offsets).astype(np.int64)
        ends = np.append(allo[1:], len(toks))
        train_docs = [(int(s), int(e)) for s, e in zip(allo, ends) if s >= n_val and e - s >= width]
        print(f"doc-boundary batching: {len(train_docs)} train docs >= {width} tok", flush=True)

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
            if e - s < width:                             # train_docs is pre-filtered; never silent
                raise ValueError(f"doc [{s},{e}) shorter than sample width {width} — "
                                 f"train_docs filter out of sync with batch width")
            i = int(rgen.integers(s, e - width + 1))
            rows.append(np.asarray(toks[i:i + width], dtype=np.int64))
        return torch.from_numpy(np.stack(rows))

    def train_batch(n, width=T + 1):
        return train_batch_cpu(n, width).to(dev)

    # EMA shadow weights, fp32 (bf16 ULP would swallow (1-decay)-sized updates at high decay).
    # Eval and checkpoints carry the EMA copy; the live weights keep training unperturbed.
    ema = {n: p.detach().float().clone() for n, p in model.named_parameters()} if args.ema > 0 else None
    ema_named = list(model.named_parameters()) if ema is not None else []
    ema_values = [ema[name] for name, _ in ema_named]
    if ema is not None:
        if not args.ema < 1.0:
            raise ValueError(f"--ema must be in (0, 1), got {args.ema}")
        print(f"ema: decay {args.ema} — eval + checkpoint use the EMA weights", flush=True)
    tail_ema = (
        TailEMA(
            model.named_parameters(),
            start_step=round(args.tail_ema_start * context_horizon),
            horizon=args.tail_ema_horizon,
            blend=args.tail_ema_blend,
        )
        if args.tail_ema_start else None
    )
    if tail_ema is not None:
        print(
            f"tail EMA: start={tail_ema.start_step} horizon={args.tail_ema_horizon} "
            f"eval_blend={args.tail_ema_blend} (embedding excluded)",
            flush=True,
        )

    def ema_update():
        with torch.no_grad():
            live_fp32 = [p.detach().float() for _, p in ema_named]
            torch._foreach_lerp_(ema_values, live_fp32, 1.0 - args.ema)

    def ema_swap():
        """Swap EMA weights in for eval; returns the live backup for ema_restore."""
        backup = {n: p.detach().clone() for n, p in ema_named}
        with torch.no_grad():
            for n, p in ema_named:
                p.copy_(ema[n].to(p.dtype))
        return backup

    def ema_restore(backup):
        with torch.no_grad():
            for n, p in ema_named:
                p.copy_(backup[n])

    def readout_swap():
        if ema is not None:
            return ("ema", ema_swap())
        if tail_ema is not None:
            return ("tail", tail_ema.swap_for_eval())
        return (None, None)

    def readout_restore(readout):
        kind, backup = readout
        if kind == "ema":
            ema_restore(backup)
        elif kind == "tail":
            tail_ema.restore(backup)

    def val_loss():
        model.eval()
        readout = readout_swap()
        with torch.no_grad():
            tot = 0.0
            for i in range(0, args.val_windows, args.batch):
                offs = val_offsets[i:i + args.batch]      # fixed windows: no training-RNG draw
                xc = torch.from_numpy(np.stack(
                    [np.asarray(val_toks[o:o + T + 1], dtype=np.int64) for o in offs]))
                recall = None
                if lmb is not None:
                    from rwkv_lab.engram_lmb import token_rosa_recall, RecallResult
                    recall = token_rosa_recall(xc[:, :T], 65536, lmb.boundary_id)
                    recall = RecallResult(*(v.to(dev) for v in recall))
                x = xc.to(dev)
                lg = (model(x[:, :T], precomputed_recall=recall)
                      if recall is not None else model(x[:, :T])).float()
                tot += F.cross_entropy(lg.reshape(-1, lg.size(-1)), x[:, 1:T + 1].reshape(-1)).item()
        readout_restore(readout)
        model.train()
        return tot / math.ceil(args.val_windows / args.batch)

    if buckets is not None:
        def val_loss():  # noqa: F811 — bucket mode: token-weighted CE over each bucket's held-out rows
            model.eval()
            readout = readout_swap()
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
            readout_restore(readout)
            model.train()
            return tot / max(cnt, 1)

    sparse_vocab_params = []
    sparse_engram_params = []
    sparse_ignored_params: set[nn.Parameter] = set()
    if args.fsdp_sparse_embeddings:
        embedding_weight = getattr(getattr(model, "emb", None), "weight", None)
        if isinstance(embedding_weight, nn.Parameter):
            sparse_vocab_params.append(embedding_weight)
        for block in getattr(model, "blocks", ()):
            de_weight = getattr(getattr(block, "de_emb", None), "weight", None)
            if isinstance(de_weight, nn.Parameter):
                sparse_vocab_params.append(de_weight)
        if lmb is not None:
            sparse_engram_params.extend(lmb.table.tables)
            sparse_engram_params.extend(lmb.table.row_scale)
        sparse_ignored_params.update(sparse_vocab_params)
        sparse_ignored_params.update(sparse_engram_params)
        print(
            f"sparse embedding comms: {len(sparse_vocab_params)} token tables + "
            f"{len(sparse_engram_params)} Engram row tensors kept replicated",
            flush=True,
        )

    if args.distributed == "fsdp2":
        if heads is not None or ema is not None:
            ap.error("FSDP2 currently requires auxiliary heads and EMA to be disabled")
        from rwkv_lab.distributed import checkpoint_rwkv_blocks, fully_shard_rwkv
        if args.activation_checkpointing:
            checkpoint_rwkv_blocks(model)
        fully_shard_rwkv(model, cpu_offload=args.cpu_offload,
                         prefetch_depth=args.fsdp_prefetch_depth,
                         ignored_params=sparse_ignored_params)
        print(f"fsdp2: world={dist.world_size} rank={dist.rank} local_rank={dist.local_rank} "
              f"cpu_offload={args.cpu_offload} activation_ckpt={args.activation_checkpointing} "
              f"prefetch_depth={args.fsdp_prefetch_depth}", flush=True)
    elif args.activation_checkpointing:
        from rwkv_lab.distributed import checkpoint_rwkv_blocks
        checkpoint_rwkv_blocks(model)
        print("activation checkpointing: per RWKV block", flush=True)
    named = list(model.named_parameters()) + (list(heads.named_parameters()) if heads else [])
    opt = build_optimizer(named, args.optimizer, args.lr, args.weight_decay,
                          muon_opts=muon_opts_from(args), u_mup_config=u_mup_cfg,
                          replicated_params=sparse_ignored_params,
                          worker_components=worker_components)
    weight_decay_schedule = (
        worker_components.weight_decay_schedule(opt)
        if worker_components is not None
        else None
    )
    gradient_accumulation = (
        worker_components.gradient_accumulation()
        if worker_components is not None
        else None
    )
    training_objective = (
        worker_components.objective()
        if worker_components is not None
        else None
    )
    print(f"optimizer={args.optimizer} lr={args.lr} wd={args.weight_decay}", flush=True)
    step = 0; resume_recall_rng = None; did_resume = False
    if args.resume and os.path.exists(args.resume):
        if args.distributed == "fsdp2":
            from rwkv_lab.distributed import load_checkpoint
            ck = load_checkpoint(args.resume, model, opt)
            step = int(ck.get("step", 0))
        else:
            ck = resume_blob
            model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); step = ck.get("step", 0)
        if worker_components is not None and ck.get(
            "component_composition_digest"
        ) != component_digest:
            raise ValueError("resume training-component composition mismatch")
        if heads is not None and ck.get("heads") is not None:
            heads.load_state_dict(ck["heads"])
        if ema is not None:                  # saved EMA if present, else re-seed from loaded weights
            src = ck.get("ema") or {}
            for n, p in model.named_parameters():
                ema[n].copy_(src[n].float().to(dev) if n in src else p.detach().float())
        if tail_ema is not None and ck.get("tail_ema") is not None:
            tail_ema.load_state_dict(ck["tail_ema"])
        if ck.get("numpy_rng") is not None: rng.bit_generator.state = ck["numpy_rng"]
        if ck.get("torch_rng") is not None: torch.set_rng_state(ck["torch_rng"].cpu())
        if torch.cuda.is_available() and ck.get("cuda_rng") is not None:
            torch.cuda.set_rng_state(ck["cuda_rng"].cpu(), device=dev)
        resume_recall_rng = ck.get("recall_numpy_rng")
        did_resume = True
        print(f"resumed from {args.resume} @ step {step}", flush=True)
        # Release the host copy now that every tensor has been copied into the
        # model, optimizer, and RNG state. Holding it (and `resume_blob`) for the
        # whole run pins a full checkpoint in RAM for no further benefit.
        ck = None
        resume_blob = None
    if args.cached_fp8_up:
        refresh_channelmix_fp8(model)
    if args.distributed == "fsdp2" and not did_resume:
        torch.manual_seed(args.seed + dist.rank)
    # Compiled handle for the TRAIN forward only: eager `model` still owns state_dict/params, so
    # checkpoints stay uncompiled (no `_orig_mod.` prefix) and the eval path (below) never toggles
    # the compiled graph's train/eval mode (which would force costly recompiles).
    fwd = (
        torch.compile(model, dynamic=False, fullgraph=args.compile_fullgraph)
        if args.compile else model
    )
    if args.compile:
        print("torch.compile: enabled (step 0 compiles; forward only, checkpoints uncompiled)", flush=True)
    # Training-batch sampler. Hold the corpus on GPU (int32) when it fits, so each step's window
    # sampling is a pure GPU gather — no per-step CPU gather, no H2D. This lets tiny models run
    # data-unbound at very high step rates (the memmap CPU path serializes the GPU behind Python).
    # Falls back to the CPU memmap sampler for corpora too large for VRAM.
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
    cpu_prefetcher = None
    cpu_prefetch_shape = None
    if buckets is not None:
        def sample_train(_n=None, _width=None):  # reciprocal batch keeps tok/step flat
            b = buckets[int(rng.choice(len(buckets), p=bprobs))]
            rows = torch.randint(b["n_val"], b["rows"], (b["B"],), device=b["data"].device)
            x = b["data"][rows]
            return (x if x.is_cuda else x.to(dev)).long()
    elif use_gpu_data:
        tg = torch.from_numpy(np.ascontiguousarray(train_toks, dtype=np.int32)).to(dev)
        ar_cache = {}

        def offsets(sample_width):
            if sample_width not in ar_cache:
                ar_cache[sample_width] = torch.arange(sample_width, device=dev)
            return ar_cache[sample_width]

        if train_docs:                                          # doc-boundary: sample doc, then offset
            ds = torch.tensor([s - n_val for s, e in train_docs], device=dev)
            dl = torch.tensor([e - s for s, e in train_docs], device=dev)
            if int(dl.min()) < width:                     # filter guarantees this; never read past doc end
                raise ValueError(f"doc-boundary GPU sampler: doc shorter than width {width}")
            def sample_train(n=args.batch, sample_width=width):
                di = torch.randint(0, ds.numel(), (n,), device=dev)
                maxoff = dl[di] - sample_width             # >= 0: docs pre-filtered to >= max width
                off = (torch.rand(n, device=dev) * (maxoff + 1).float()).long().minimum(maxoff)
                return tg[(ds[di] + off)[:, None] + offsets(sample_width)[None, :]].long()
        else:                                                  # flat: uniform window over the corpus
            def sample_train(n=args.batch, sample_width=width):
                hi = tg.numel() - sample_width
                idx = torch.randint(0, hi, (n,), device=dev)
                return tg[idx[:, None] + offsets(sample_width)[None, :]].long()
        print(f"gpu-data: corpus on GPU ({gpu_gb:.2f} GB int32) — window sampling is GPU-side", flush=True)
    else:
        def sample_train(n=args.batch, sample_width=width):
            nonlocal cpu_prefetcher, cpu_prefetch_shape
            shape = (int(n), int(sample_width))
            if args.cpu_prefetch and "cuda" in str(dev):
                if shape != cpu_prefetch_shape:
                    if cpu_prefetcher is not None:
                        cpu_prefetcher.close()
                    cpu_prefetcher = AsyncCPUBatchPrefetcher(
                        lambda local_rng: train_batch_cpu(
                            shape[0], width=shape[1], sampler_rng=local_rng),
                        rng, pin_memory=True)
                    cpu_prefetch_shape = shape
                return cpu_prefetcher.next().to(dev, non_blocking=True)
            return train_batch(n, width=sample_width)

        mode = "prefetch" if args.cpu_prefetch and "cuda" in str(dev) else "synchronous"
        print(f"gpu-data: OFF ({gpu_gb:.2f} GB corpus) — CPU memmap sampler ({mode})", flush=True)
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

        def sample_train(_n=None, _width=None):
            nonlocal recall_future
            ids, rr = recall_future.result()
            recall_future = recall_pool.submit(_prefetch_engram)
            return (ids.to(dev, non_blocking=True),
                    RecallResult(*(v.to(dev, non_blocking=True) for v in rr)))
        print("engram recall: CPU-prefetched one step ahead (pinned async H2D)", flush=True)
    if args.compile and args.compile_prewarm and lmb is None:
        shapes = {
            (
                stage.seq_len,
                max(1, round(T * args.batch / stage.seq_len)),
            )
            for stage in context_stages
        } if context_stages else {(T, args.batch)}
        cpu_rng = torch.get_rng_state()
        cuda_rng = (
            torch.cuda.get_rng_state(dev) if torch.cuda.is_available() else None
        )
        model.train()
        print(f"compile prewarm: {sorted(shapes)}", flush=True)
        for warm_seq, warm_batch in sorted(shapes):
            opt.zero_grad(set_to_none=True)
            warm_ids = torch.zeros(
                warm_batch, warm_seq, dtype=torch.long, device=dev
            )
            warm_hidden = fwd(warm_ids, hidden_only=True)
            if training_objective is not None:
                warm_loss = training_objective(
                    warm_hidden, model.head, warm_ids
                )
            else:
                from rwkv_lab.fused_ce import lmhead_cross_entropy

                warm_loss = lmhead_cross_entropy(
                    warm_hidden, model.head, warm_ids, fused=True
                )
            warm_loss.backward()
        opt.zero_grad(set_to_none=True)
        torch.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state(cuda_rng, device=dev)
        print("compile prewarm: complete; parameters and optimizer were not stepped", flush=True)

    model.train(); t0 = time.time(); seen = 0; last_context_shape = None
    print(f"budget={'%.1f min' % args.minutes if not args.steps else str(args.steps)+' steps'}", flush=True)
    while True:
        if args.steps and step >= args.steps: break
        if not args.steps and (time.time() - t0) / 60.0 >= args.minutes: break
        if step % args.eval_every == 0:
            vl = val_loss(); emit({"kind": "eval", "step": step, "loss": vl, "val_loss": vl, "ppl": math.exp(vl)})
            publish_eval(step, vl)
            print(f"[{step}] val {vl:.4f} (ppl {math.exp(vl):.2f})  {(time.time()-t0)/60:.1f}min", flush=True)
        train_seq_len, train_batch_size = context_batch_for_step(
            context_stages, step=step, total_steps=context_horizon or 1,
            max_seq_len=T, base_batch=args.batch)
        train_width = train_seq_len + 1 + (heads.extra_tokens if heads else 0)
        context_shape = (train_seq_len, train_batch_size)
        if context_shape != last_context_shape:
            if context_stages:
                print(f"context stage @ step {step}: seq={train_seq_len} "
                      f"batch={train_batch_size}", flush=True)
                emit({"kind": "context_stage", "step": step, "seq_len": train_seq_len,
                      "batch": train_batch_size})
            last_context_shape = context_shape
        if head_tied and step >= split_step:
            new_head = split_tied_embedding_head(model, opt)
            if ema is not None:
                ema["head.weight"] = new_head.detach().float().clone()
                ema_named.append(("head.weight", new_head))
                ema_values.append(ema["head.weight"])
            if tail_ema is not None:
                tail_ema.add_parameter("head.weight", new_head)
            head_tied = False
            named = list(model.named_parameters()) + (
                list(heads.named_parameters()) if heads else []
            )
            print(f"embedding/head untied at step {step}; optimizer moments cloned", flush=True)
        if heads is not None and args.lmtp_cooldown_fraction:
            heads.lmtp_weight = args.lmtp_weight * tail_linear_multiplier(
                step,
                context_horizon,
                args.lmtp_cooldown_fraction,
            )
        lr = args.lr * min(1.0, (step + 1) / max(args.warmup, 1))       # linear warmup
        horizon = args.decay_steps or args.steps
        if args.lr_schedule == "powercool" and horizon:
            if powercool_schedule is None:
                raise RuntimeError("PowerCool schedule was not initialized")
            lr = args.lr * powercool_multiplier(step, powercool_schedule)
        elif args.lr_schedule == "cosine" and horizon:                 # then cosine decay to 0.1x
            lr *= 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(step, horizon) / horizon))
        for g in opt.param_groups:
            g["lr"] = lr * g.get("u_mup_lr_mult", 1.0)
        if lmb is not None:                  # ramp Engram injection in (gates learn on live recall)
            lmb.set_warmup(min(1.0, (step + 1) / max(args.engram_warmup, 1)))
        opt.zero_grad(set_to_none=True)
        sparse_vocab_rows = []
        sparse_recalled_rows = []
        ga = (
            gradient_accumulation.microbatches_per_optimizer_step
            if gradient_accumulation is not None
            else max(args.grad_accum, 1)
        )
        microbatch_indices = (
            gradient_accumulation.microbatch_indices()
            if gradient_accumulation is not None
            else range(ga)
        )
        for micro_step in microbatch_indices:  # effective batch = batch * ga
            input_wait = (
                worker_step_profiler.input_wait()
                if worker_step_profiler is not None
                else nullcontext()
            )
            with input_wait:
                sample = sample_train(train_batch_size, train_width)
            x, precomputed_recall = sample if isinstance(sample, tuple) else (sample, None)
            # Mixed-ctx rows are exactly T_bucket wide (pad-masked); flat windows are T+1(+extra).
            xin, tgt = ((x[:, :-1], x[:, 1:]) if buckets is not None
                        else (x[:, :train_seq_len], x[:, 1:train_seq_len + 1]))
            if args.fsdp_sparse_embeddings:
                sparse_vocab_rows.append(torch.unique(xin))
                if precomputed_recall is not None:
                    valid_recalled = precomputed_recall.recalled[
                        precomputed_recall.valid
                    ]
                    if valid_recalled.numel():
                        sparse_recalled_rows.append(torch.unique(valid_recalled))
            # The ordinary path skips RWKV7Small's full vocabulary output and lets
            # fused CE reuse its bf16 logit allocation during backward. Engram's
            # sparse copy-head mutates logits, so it retains the compatible path.
            if lmb is None:
                hidden = fwd(xin, hidden_only=True)
                if training_objective is not None:
                    loss = training_objective(
                        hidden,
                        model.head,
                        tgt,
                        ignore_index=(0 if buckets is not None else None),
                    )
                else:
                    from rwkv_lab.fused_ce import lmhead_cross_entropy

                    loss = lmhead_cross_entropy(
                        hidden,
                        model.head,
                        tgt,
                        fused=True,
                        ignore_index=(0 if buckets is not None else None),
                    )
                out = (None, hidden) if heads else None
            else:
                out = fwd(xin, return_hidden=bool(heads),
                          precomputed_recall=precomputed_recall)
                lg = (out[0] if heads else out).float()
                loss = F.cross_entropy(lg.reshape(-1, lg.size(-1)), tgt.reshape(-1),
                                       ignore_index=(0 if buckets is not None else -100))
            if heads:                                        # + weighted aux (latent-prediction) loss
                loss = loss + heads.compute(out[1], x, model.emb, model.head)["aux_total"]
            if args.routing_free_moe and not args.init_g1g:
                root = model.module if hasattr(model, "module") else model
                loss = loss + args.routing_free_aux_weight * sum(
                    (block.ffn.aux_loss.to(loss.device) for block in root.blocks),
                    loss.new_zeros(()))
            if args.distributed == "fsdp2" and ga > 1:
                from rwkv_lab.distributed import set_requires_gradient_sync
                set_requires_gradient_sync(model, micro_step == ga - 1)
            scaled_loss = (
                gradient_accumulation.scale_loss(loss)
                if gradient_accumulation is not None
                else (loss / ga if ga > 1 else loss)
            )
            scaled_loss.backward()
            seen += xin.shape[0] * xin.shape[1]
        if args.fsdp_sparse_embeddings:
            from rwkv_lab.distributed import sparse_sync_parameter_rows

            vocab_rows = (
                torch.unique(torch.cat(sparse_vocab_rows))
                if sparse_vocab_rows else torch.empty(0, device=dev, dtype=torch.long)
            )
            for parameter in sparse_vocab_params:
                sparse_sync_parameter_rows(parameter, vocab_rows)
            if sparse_engram_params:
                recalled = (
                    torch.unique(torch.cat(sparse_recalled_rows))
                    if sparse_recalled_rows
                    else torch.empty(0, device=dev, dtype=torch.long)
                )
                physical = (
                    torch.unique(lmb.table.access_idx[recalled].long())
                    if recalled.numel()
                    else recalled
                )
                for parameter in sparse_engram_params:
                    sparse_sync_parameter_rows(parameter, physical)
        if args.distributed == "fsdp2":
            from rwkv_lab.distributed import clip_grad_norm
            gn = clip_grad_norm(model, args.grad_clip)
        else:
            clip_params = list(model.parameters()) + (
                list(heads.parameters()) if heads else []
            )
            gn = (
                worker_components.gradient_clipping(clip_params)
                if worker_components is not None
                else torch.nn.utils.clip_grad_norm_(clip_params, args.grad_clip)
            )
        if weight_decay_schedule is not None:
            weight_decay_schedule.step(step)
        opt.step(); step += 1
        if worker_step_profiler is not None:
            worker_step_profiler.step(step)
        if worker_observability is not None:
            worker_observability.optimizer_step(step)
        if args.cached_fp8_up:
            refresh_channelmix_fp8(model)
        if ema is not None:
            ema_update()
        if tail_ema is not None:
            tail_ema.update(step)
        if step % args.log_every == 0:
            emit({"kind": "train", "step": step, "loss": float(loss.detach()),
                  "gnorm": float(gn.detach()),
                  "lr": lr, "tok_per_sec": int(seen / max(time.time() - t0, 1e-6))})
            publish_metric("training_loss", float(loss.detach()), metric_step=step)
            publish_metric("gradient_norm", float(gn.detach()), metric_step=step)
            publish_metric("learning_rate", lr, metric_step=step)
            publish_metric(
                "tokens_per_second",
                int(seen / max(time.time() - t0, 1e-6)),
                metric_step=step,
            )
    vl = val_loss(); emit({"kind": "eval", "step": step, "loss": vl, "val_loss": vl, "ppl": math.exp(vl)})
    publish_eval(step, vl)
    if recall_pool is not None:
        recall_pool.shutdown(wait=False, cancel_futures=True)
    if cpu_prefetcher is not None:
        cpu_prefetcher.close()
    emit({"kind": "checkpoint", "step": step})
    if args.save:
        # Self-describing architecture is shared by ordinary .pt and FSDP2/DCP checkpoints.
        arch = {"d_model": args.d_model, "n_layers": args.n_layers,
                         "head_size": args.head_size, "seed_chain": seed_chain,
                         "deepembed": bool(args.deepembed) and not args.init_g1g,
                         "de_dim": args.de_dim, "de_mode": args.de_mode,
                         "de_shift": bool(args.de_shift), "de_emb_res": bool(args.de_emb_res),
                         "u_mup_base_width": args.u_mup_base_width,
                         "u_mup_base_depth": args.u_mup_base_depth,
                         "online_memory": bool(args.online_memory) and not args.init_g1g,
                         "online_memory_mode": args.online_memory_mode,
                         "online_memory_dim": args.online_memory_dim,
                         "online_memory_lr": args.online_memory_lr,
                         "online_memory_retention": args.online_memory_retention,
                         "online_memory_window": args.online_memory_window,
                         "balance_state": bool(args.balance_state),
                         "state_offset": bool(args.state_offset) and not args.init_g1g,
                         "state_offset_interval": args.state_offset_interval,
                         "routing_free_moe": bool(args.routing_free_moe) and not args.init_g1g,
                         "routing_free_experts": args.routing_free_experts,
                         "routing_free_rank": args.routing_free_rank,
                         "routing_free_threshold": args.routing_free_threshold,
                         "routing_free_balance": args.routing_free_balance,
                         "byte_aware": bool(args.byte_aware_vocab) and not args.init_g1g,
                         "byte_aware_max_bytes": args.byte_aware_max_bytes,
                         "byte_aware_dim": args.byte_aware_dim,
                         "nvfp4": bool(args.nvfp4), "nvfp4_rht": bool(args.nvfp4_rht),
                         "nvfp4_backend": args.nvfp4_backend,
                         "engram": lmb is not None, "engram_sites": args.engram_sites,
                         "engram_drow": args.engram_drow, "engram_rows": args.engram_rows,
                         "engram_boundary_id": (lmb.boundary_id if lmb is not None else None),
                         "fused_channelmix": bool(args.fused_channelmix),
                         "cached_fp8_up": bool(args.cached_fp8_up),
                         "fp8_head": bool(args.fp8_head),
                         "tail_ema_start": args.tail_ema_start,
                         "tail_ema_horizon": args.tail_ema_horizon,
                         "tail_ema_blend": args.tail_ema_blend,
                         "tie_head_until": args.tie_head_until,
                         "lmtp_cooldown_fraction": args.lmtp_cooldown_fraction,
                         "loop_kw": lk}
        rng_extra = {"step": step, "config": tag, "arch": arch,
                     "head_tied": head_tied,
                     "numpy_rng": rng.bit_generator.state,
                     "recall_numpy_rng": (recall_rng.bit_generator.state
                                           if lmb is not None and not use_gpu_data else None),
                     "torch_rng": torch.get_rng_state()}
        if component_digest is not None:
            rng_extra["component_composition_digest"] = component_digest
        if torch.cuda.is_available():
            rng_extra["cuda_rng"] = torch.cuda.get_rng_state(dev).cpu()
        if args.distributed == "fsdp2":
            from rwkv_lab.distributed import save_checkpoint
            save_checkpoint(args.save, model, opt, extra=rng_extra)
        else:
            blob = {"model": model.state_dict(), "opt": opt.state_dict(),
                    "heads": heads.state_dict() if heads is not None else None, **rng_extra}
            if ema is not None:
                blob["ema"] = ema
            if tail_ema is not None:
                blob["tail_ema"] = tail_ema.state_dict()
            torch.save(blob, args.save)
        print(f"saved -> {args.save}", flush=True)
    print(f"DONE {tag}: {step} steps, final val {vl:.4f} (ppl {math.exp(vl):.2f})", flush=True)
    return {
        "checkpoint": args.save or None,
        "final_validation_loss": vl,
        "step": step,
    }


if __name__ == "__main__":
    main()
