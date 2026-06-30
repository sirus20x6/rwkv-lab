"""
MLA-only finetune for the converted Qwen3.6-35B-A3B.

All parameters frozen except the 10 swapped MLAAttention modules (≈1.1B params
out of ~36B). Streams random windows from engram_tokens.bin (Qwen-tokenized).

Design choices:
- Single GPU, bf16 mixed precision, activation checkpointing on the frozen
  decoder layers so forward memory stays ~flat across seq_len.
- AdamW on fp32 master copies of the MLA params only. For 1.1B trainable
  params that's ~9GB of optimizer state.
- Random-window sampling: each microbatch picks uniform random offsets into
  the training range (reserving the last --eval-tokens for held-out PPL).
  Good enough mixing for a short finetune; proper per-document boundaries
  and EOS packing aren't needed at this budget.
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import signal
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
import torch.nn.functional as F

# Global flag set by SIGINT/SIGTERM handler — checked between optimizer steps.
# The first signal triggers a graceful save; the second kills hard (so the user
# isn't trapped if a save stalls).
_interrupt_flag = {"count": 0}

def _sig_handler(signum, frame):  # pragma: no cover — wired at runtime
    _interrupt_flag["count"] += 1
    if _interrupt_flag["count"] == 1:
        print(f"\n[interrupt] received signal {signum}; will save + exit after "
              "current accumulation step. Send another Ctrl-C / TERM to force.")
    else:
        print("\n[interrupt] second signal received, exiting immediately")
        sys.exit(130)

# trainboard: SIGUSR1 requests a checkpoint WITHOUT exiting (the dashboard's
# "checkpoint now"). The handler only sets a flag; the loop saves at the next
# safe point between optimizer steps.
_ckpt_flag = {"count": 0}

def _sigusr1_handler(signum, frame):  # pragma: no cover — wired at runtime
    _ckpt_flag["count"] += 1  # only set the flag; the loop prints + saves at a safe point


def chunked_ce(logits: torch.Tensor, labels: torch.Tensor, chunk: int = 2048) -> torch.Tensor:
    """Cross-entropy with per-chunk fp32 materialization. `chunk` is the
    NUMBER OF TOKENS in each chunk (not positions-per-batch), so the fp32
    materialized size is chunk*V*4 bytes regardless of batch size."""
    B, T, V = logits.shape
    flat_logits = logits.reshape(-1, V)
    flat_labels = labels.reshape(-1)
    N = flat_logits.shape[0]
    total = logits.new_zeros((), dtype=torch.float32)
    for i in range(0, N, chunk):
        end = min(i + chunk, N)
        lc = flat_logits[i:end].float()
        yc = flat_labels[i:end]
        total = total + F.cross_entropy(lc, yc, reduction="sum")
    return total / N


def chunked_lmhead_ce(hidden: torch.Tensor, lm_head: torch.nn.Linear,
                      labels: torch.Tensor, chunk: int = 2048) -> torch.Tensor:
    """Combined lm_head + cross-entropy without materializing the full
    [B, T, V] logits tensor. Caps peak bf16 tensor to chunk*V bytes instead
    of B*T*V bytes (the full logits), saving ~4 GB per MTP forward at
    batch=4, seq=2048, vocab=248K."""
    B, T, H = hidden.shape
    flat_h = hidden.reshape(-1, H)
    flat_labels = labels.reshape(-1)
    N = flat_h.shape[0]
    total = hidden.new_zeros((), dtype=torch.float32)
    W = lm_head.weight
    bias = getattr(lm_head, "bias", None)
    for i in range(0, N, chunk):
        end = min(i + chunk, N)
        # Materialize chunk logits only
        chunk_logits = F.linear(flat_h[i:end], W, bias).float()  # [chunk, V]
        total = total + F.cross_entropy(chunk_logits, flat_labels[i:end], reduction="sum")
    return total / N


@torch.no_grad()
def chunked_lmhead_metrics(hidden: torch.Tensor, lm_head: torch.nn.Linear,
                           labels: torch.Tensor, chunk: int = 2048,
                           topk: tuple[int, ...] = (1, 5)) -> dict:
    """lm_head + sum CE + top-k counts without materializing [B,T,V] logits."""
    B, T, H = hidden.shape
    flat_h = hidden.reshape(-1, H)
    flat_labels = labels.reshape(-1)
    N = flat_h.shape[0]
    total_loss_t = hidden.new_zeros((), dtype=torch.float32)
    top_counts_t = {k: hidden.new_zeros((), dtype=torch.long) for k in topk}
    max_k = max(topk)
    W = lm_head.weight
    bias = getattr(lm_head, "bias", None)
    for i in range(0, N, chunk):
        end = min(i + chunk, N)
        logits = F.linear(flat_h[i:end].to(W.dtype), W, bias).float()
        yc = flat_labels[i:end]
        total_loss_t = total_loss_t + F.cross_entropy(logits, yc, reduction="sum")
        _, top_idx = logits.topk(max_k, dim=-1)
        matches = (top_idx == yc.unsqueeze(-1))
        cum = matches.cumsum(dim=-1).clamp(max=1)
        for k in topk:
            top_counts_t[k] = top_counts_t[k] + cum[:, k - 1].sum()
    return {
        "loss_sum": total_loss_t.item(),
        "top_counts": {k: int(v.item()) for k, v in top_counts_t.items()},
        "n_tokens": N,
    }


@torch.no_grad()
def chunked_metrics(logits: torch.Tensor, labels: torch.Tensor,
                    chunk: int = 2048, topk: tuple[int, ...] = (1, 5)) -> dict:
    """Sum-reduction CE loss plus top-k match counts, chunked by absolute
    token count so memory is bounded regardless of batch size. Accumulates on
    GPU and calls .item() only at the end to avoid per-chunk GPU sync."""
    B, T, V = logits.shape
    flat_logits = logits.reshape(-1, V)
    flat_labels = labels.reshape(-1)
    N = flat_logits.shape[0]
    total_loss_t = flat_logits.new_zeros((), dtype=torch.float32)
    top_counts_t = {k: flat_logits.new_zeros((), dtype=torch.long) for k in topk}
    max_k = max(topk)
    for i in range(0, N, chunk):
        end = min(i + chunk, N)
        lc = flat_logits[i:end].float()
        yc = flat_labels[i:end]
        total_loss_t = total_loss_t + F.cross_entropy(lc, yc, reduction="sum")
        _, top_idx = lc.topk(max_k, dim=-1)
        matches = (top_idx == yc.unsqueeze(-1))
        cum = matches.cumsum(dim=-1).clamp(max=1)
        for k in topk:
            top_counts_t[k] = top_counts_t[k] + cum[:, k - 1].sum()
    # Single sync at the end
    return {
        "loss_sum": total_loss_t.item(),
        "top_counts": {k: int(v.item()) for k, v in top_counts_t.items()},
        "n_tokens": N,
    }

# trainboard: this is a DUPLICATE of ../../train_mla.py (original never modified).
# It lives two dirs below the repo root, so prepend the root to sys.path before
# the repo-local imports below resolve when run as
# `python dashboard2/instrumented/train_mla.py` from the repo.
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from load_converted import load_converted_model
from mla_module import MLAAttention
from mutor_module import MuToRHead, mutor_loss
from fsp_module import FSPHead, fsp_loss
from parallel_heads_module import ParallelHeads, parallel_heads_loss
from safe_torch import safe_torch_load
# Engram imports are deferred to main() because engram_ext lives outside this
# package (under /thearray/git/engram/python). Import at call site only when
# engram is actually enabled — keeps import of this module cheap for callers
# that never touch Engram (eval_only.py, etc.).


def _text_model(model):
    """Return the backbone module that produces hidden states without lm_head."""
    return getattr(model.model, "language_model", model.model)


def _last_hidden(outputs):
    if hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
        return outputs.last_hidden_state
    return outputs[0]


@dataclass
class TrainConfig:
    # Model / data
    model_dir: str = "/thearray/git/moe-mla/Qwen3.6-35B-A3B"
    patch_dir: str = "/thearray/git/moe-mla/converted"
    tokens_bin: str = "/thearray/data/engram_tokens.bin"
    total_tokens_in_bin: int = 75_306_005_724   # from the manifest
    eval_tokens: int = 100_000_000              # held out from the tail
    out_dir: str = "/thearray/git/moe-mla/runs/mla_ft_v1"
    resume: str = ""                            # path to a previous ckpt.pt, or "" to start fresh from SVD init
    resume_warmup_steps: int = 100              # re-warmup lr linearly over this many steps after resume
                                                # (Adam's v_t adapted to small updates at the end of the prior run;
                                                # jumping back to peak lr causes big overshoots until v_t catches up)
    install_mtp: int = 0                        # 1 = install Qwen3.6's MTP module, train it with aux loss
                                                # (requires patch built with --include-mtp=1)
    mtp_loss_weight: float = 0.3                # λ for loss = lm_loss + λ * mtp_loss (DeepSeek V3 default)
    train_mtp_only: int = 0                     # 1 = freeze backbone MLA, only MTP trainable.
                                                # Backbone forward runs in no_grad → ~3x faster per step.
                                                # Requires install_mtp=1.
    train_aux_only: int = 0                     # 1 = freeze EVERYTHING except aux heads
                                                # (mutor_head, fsp_head, parallel_heads). Use this
                                                # to warm up aux heads from random init before
                                                # Phase 2 unfreezes backbone. Implies backbone
                                                # forward is run under no_grad (like train_mtp_only).
    mtp_chain_horizons: str = "2"               # comma-sep list of horizons to supervise.
                                                # e.g. "2,3,4" applies loss at h=2, h=3, h=4 via chained
                                                # MTP calls (teacher-forced). Each additional horizon
                                                # adds one MTP forward pass + its activations to the
                                                # backward graph.
    mtp_chain_weights: str = "1.0"              # matching weights; summed loss = sum(w_k * loss_h_k).
                                                # e.g. "1.0,0.5,0.25" de-emphasizes higher horizons.

    # MuToR — register-style multi-horizon auxiliary loss (Gerontopoulos et al. 2025).
    # Samples K positions per batch, predicts token[p+d] from hidden[p]+offset_emb(d),
    # with d ~ Uniform({d_min..d_max}). Enabled only when backbone has grad (Phase 2).
    mutor_enabled: int = 0                      # 1 = add MuToR aux loss
    mutor_weight: float = 0.3                   # coefficient in total loss
    mutor_num_registers: int = 32               # registers per sequence
    mutor_d_max: int = 4                        # largest horizon d
    mutor_d_min: int = 2                        # smallest horizon d (skip NTP-equivalent d=1)

    # FSP-BCE — handcrafted future-summary multi-hot BCE (Mahajan et al. 2025).
    # At K sampled positions, predict a multi-hot bag of tokens in the next tau
    # positions via BCE on sigmoid(lm_head(hidden + fsp_bias)).
    fsp_enabled: int = 0
    fsp_weight: float = 0.1
    fsp_num_positions: int = 64
    fsp_tau: int = 12
    fsp_idf_path: str = ""                      # optional path to precomputed [V] weight tensor

    # Parallel MTP heads (Gloeckle et al. 2024) — K independent heads on top
    # of backbone hidden, each predicting a specific future horizon. Can stack
    # alongside chained MTP (both train backbone through different pathways).
    # Memory: K * ~50M per head (SwiGLU MLP w/ expansion=4 at H=2048).
    parallel_enabled: int = 0
    parallel_horizons: str = "2,3,4"            # comma-sep horizons >= 2
    parallel_weights: str = "0.3,0.2,0.1"       # matching weights per head
    parallel_head_expansion: int = 4            # MLP expansion factor per head

    # Engram — DeepSeek's conditional-memory module (Cheng et al. 2026,
    # arXiv:2601.07372). Installs at decoder-layer indices specified in the
    # patch manifest. Uses hashed N-gram embeddings as a static O(1) memory
    # lookup fused via context-aware gating. Applies to the already-loaded
    # MLA/MTP/aux model as an additive residual at select layers — nothing
    # else changes; existing losses still drive backbone/MTP/aux.
    engram_enabled: int = 0
    engram_patch_dir: str = ""                  # path to engram_converted/ or engram_converted_v2/
    engram_lr_mult: float = 5.0                 # paper: Engram params train at 5x base lr, wd=0
    engram_zero_value_proj: int = -1            # -1=auto: zero only for non-prefilled patches.
                                                # 0=keep patch value_proj, 1=force zero identity start.
                                                # Prefill patches mark manifest["prefilled"]=true.
    train_engram_only: int = 0                  # 1 = freeze MLA / MTP / aux; only Engram trainable.
                                                # Unlike train_mtp_only / train_aux_only, backbone is
                                                # NOT wrapped in no_grad — Engram's gate & value_proj
                                                # read backbone hidden, so we need autograd to reach
                                                # Engram params through it. Backbone params stay
                                                # frozen (requires_grad=False) so they don't update.
    train_mla_only: int = 0                     # 1 = freeze MTP / aux / Engram; only MLA trainable.
                                                # For Phase 1.5 — distill a fresh BKV MLA back to
                                                # producing hidden states that match what the frozen
                                                # downstream expects, without discarding Phase 2/3
                                                # training work.
    mla_from_patch: int = 0                     # 1 = on resume, load MTP / aux / Engram state from
                                                # ckpt BUT skip the ckpt's mla_state_dicts and use
                                                # the fresh patch weights instead. Use when you
                                                # rebuilt the MLA patch (e.g. with --bkv-stats) and
                                                # want to keep the Phase 2/3 downstream but retrain
                                                # MLA from the new init.
    xsa_enabled: int = 0                        # 1 = enable Exclusive Self Attention on all MLA
                                                # modules (backbone + MTP). Subtracts projection on
                                                # per-head value vector pre-o_proj.
                                                # arxiv.org/abs/2603.09078
    rwkv8_deltanet_layers: str = ""             # comma-sep linear-attention layer indices to replace
                                                # with an RWKV-8 module. Example: "30" or "26,30".
                                                # Leave empty to keep all DeltaNet layers.
    rwkv8_swap_mode: str = "timemix"            # "timemix" = RWKV_Tmix_x070 port (Stage 1+ default);
                                                # "channelmix" = legacy small FFN-style stand-in.
    rwkv8_ffn_hidden_size: int = 0              # 0 = 4 * hidden_size. Use smaller values for cheap
                                                # one-layer probes, e.g. 4096 or 8192. (channelmix only)
    rwkv8_init_output_scale: float = 1e-3       # small output init for stable one-layer replacement
                                                # (channelmix only; timemix zero-inits output).
    train_rwkv8_only: int = 0                   # 1 = freeze everything except RWKV-8 modules.
                                                # Differs from train_mla_only (which trains MLA *and*
                                                # RWKV-8 together). For Stage 1 single-layer probe.
                                                # Mutually exclusive with train_mla_only.
    rwkv8_inherit_from_loaded: int = 0          # 1 = on resume, broadcast a TRAINED RWKV-8 module's
                                                # state_dict into every newly-installed RWKV-8 module
                                                # that didn't load state from ckpt. Use when adding
                                                # more RWKV-8 layers to a run that already trained
                                                # one — far better init than paper/zero.
    train_rwkv8_layers: str = ""                # comma-sep RWKV-8 layer indices to train.
                                                # Freezes everything else, INCLUDING other RWKV-8
                                                # layers. Use to isolate one fresh layer at a time
                                                # while previously-trained RWKV-8 layers stay
                                                # locked. Mutually exclusive with train_rwkv8_only,
                                                # train_mla_only.

    # Training schedule
    seq_len: int = 2048
    micro_batch_size: int = 1
    grad_accum_steps: int = 16
    max_steps: int = 4600                       # ≈150M tokens at seq=2048, batch=16
    lr: float = 1e-4
    min_lr: float = 1e-5
    warmup_steps: int = 200
    weight_decay: float = 0.0                   # MLA inits are already near the optimum; WD=0 avoids pulling away
    grad_clip: float = 1.0                      # <= 0 disables grad clipping / norm reduction
    grad_clip_every: int = 1                    # clip every N optimizer steps; 1 = every step, 0 = disabled

    # Optimizer choice.
    # "adamw8bit" : bitsandbytes' 8-bit AdamW (default; works with Engram SparseAdam).
    # "muon"      : SingleDeviceMuonWithAuxAdam from the muon-optimizer pkg.
    #               NOTE: as of muon-clip install this package is overwritten;
    #               new runs should use "muonclip" instead.
    # "muonclip"  : MuonClip from the muon-clip pkg (Muon on 2D + Adam on 1D,
    #               with corrected RMS so a single peak LR scales both groups
    #               coherently). enable_clipping is OFF by default — QK-clipping's
    #               weight-rescale math assumes standard q_proj/k_proj layers
    #               sharing input, which MLA breaks. The qk_norm + output_gate
    #               built into our MLA are doing the stability work clipping
    #               would otherwise do.
    optimizer: str = "adamw8bit"                # "adamw8bit" | "muon" | "muonclip"
    muon_aux_lr: float = 3e-4                   # lr for the AuxAdam side (1D params) when optimizer=muon

    # Phase C v4 — GuardedMuonClip + Prodigy. The muon-clip pkg's muon_update
    # applies a `0.4*sqrt(max_dim)` corrected-RMS amplifier (utils_muon/utils.py:409)
    # on top of the orthogonalized direction. For our 4096-dim hidden matrices this
    # is ~25.6x; for 151936×4096 embeddings ~155x. Vanilla Muon has no such factor,
    # so reusing Phase B's lr=5e-4 from vanilla Muon under MuonClip blew up two
    # runs. The guard caps per-param `lr * RMS(update) / RMS(param)`; route
    # embed_tokens/lm_head out of Muon entirely (see muon_exclude_embed_lmhead);
    # use Prodigy on non-matrix params (real meta-LR adaptation).
    guarded_muonclip: int = 0                   # 1 = subclass MuonClip with per-param update-RMS cap
    guard_max_muon_ratio: float = 5e-4          # cap on lr*RMS(update)/RMS(param) for Muon side
    guard_max_adam_ratio: float = 1e-4          # cap for Adam side (only meaningful if not using Prodigy)
    muon_exclude_embed_lmhead: int = 0          # 1 = route embed_tokens / lm_head out of Muon
    prodigy_aux: int = 0                        # 1 = run Prodigy as a second optimizer over non-matrix params
                                                #     (and embed/lm_head if muon_exclude_embed_lmhead=1).
                                                #     Implies muon_exclude_embed_lmhead=1.
    prodigy_d0: float = 1e-6                    # initial D estimate (Prodigy auto-adapts upward)
    prodigy_lr_mult: float = 1.0                # multiplier; Prodigy's effective step is d * lr_mult
    prodigy_safeguard_warmup: int = 1           # Prodigy's built-in early-step guard
    prodigy_decouple: int = 1                   # decoupled weight decay

    # Reduce-on-Plateau LR controller. Reactive replacement for the cosine/wsd
    # tail: when the eval h1_ppl stops improving for `rop_patience` consecutive
    # evals (with `rop_rel_threshold` minimum relative improvement), multiply
    # all optimizer lrs by `rop_factor` (down to a `rop_min_mult` floor).
    # Composes with the base lr_schedule: schedule sets the upper envelope,
    # ROP can only reduce. Use with --lr-schedule constant to let ROP fully
    # own the decay decision.
    rop_enable: int = 0                         # 1 = enable ROP wrapper on top of base schedule
    rop_patience: int = 4                       # consecutive non-improving evals before drop
    rop_factor: float = 0.5                     # multiply lr by this on each trigger
    rop_rel_threshold: float = 0.005            # require >0.5% improvement to count as progress
    rop_min_mult: float = 0.01                  # floor: lr never goes below base * 0.01
    rop_warmup_grace_steps: int = 100           # don't trigger during initial warmup
    rop_cooldown: int = 2                       # evals to wait after a drop before re-counting
    gradient_checkpointing: int = 1             # 1 = re-compute activations during backward (saves VRAM, ~30% slower).
                                                # 0 = keep all activations (faster, needs headroom). Only has effect
                                                # when backbone has grad (not in train_mtp_only / train_aux_only).
    lr_schedule: str = "cosine"                 # "cosine" | "wsd" | "constant".
                                                # cosine: linear warmup → cosine decay to min_lr (legacy default).
                                                # wsd:    linear warmup → constant at peak → cosine tail to min_lr.
                                                # constant: linear warmup → hold at peak (no decay).
    wsd_stable_fraction: float = 0.7            # for lr_schedule=wsd, fraction of post-warmup steps at peak.
                                                # remaining (1-stable_fraction) is cosine decay to min_lr.
    freeze_non_mla: int = 1                     # 1 = load_converted_model freezes every non-MLA param (default).
                                                # 0 = leave the whole model trainable so Phase-B-style runs can
                                                #     unfreeze the backbone after an MLA primer phase.

    # Logging / eval
    log_every: int = 10
    eval_every: int = 500
    eval_batches: int = 32                      # 32 × seq_len for each eval
    save_every: int = 1000                      # step-based checkpoint cadence (0 disables)
    save_every_seconds: int = 0                 # additional wall-clock cadence (0 disables).
                                                # Useful for long runs where steps are slow —
                                                # e.g., 900 = save at least every 15 min.
    max_saved_checkpoints: int = 0              # keep only last N step_* dirs (0 = keep all).
                                                # Applied after each save, only to auto-saves
                                                # (not the final post-loop save).
    seed: int = 0


# ---------------------------------------------------------------------------
# Data: mmap'd uint32 token stream, random-window sampler.
# ---------------------------------------------------------------------------
def open_tokens(cfg: TrainConfig) -> tuple[np.memmap, int, int]:
    """Return (arr, train_end, eval_end). Training range is [0, train_end);
    eval range is [train_end, eval_end)."""
    arr = np.memmap(cfg.tokens_bin, dtype=np.uint32, mode="r")
    N = cfg.total_tokens_in_bin
    actual = int(arr.shape[0])
    if N <= 0:
        raise ValueError(f"total_tokens_in_bin must be positive, got {N}")
    if N > actual:
        raise ValueError(
            f"total_tokens_in_bin={N} exceeds actual token file length "
            f"{actual} for {cfg.tokens_bin}"
        )
    if cfg.eval_tokens <= 0:
        raise ValueError(f"eval_tokens must be positive, got {cfg.eval_tokens}")
    eval_end = N
    train_end = N - cfg.eval_tokens
    if train_end <= 0:
        raise ValueError(
            f"eval_tokens={cfg.eval_tokens} leaves no training range for N={N}"
        )
    return arr, train_end, eval_end


def validate_train_config(cfg: TrainConfig) -> None:
    """Validate knobs that otherwise fail late with modulo/division errors."""
    positive = {
        "seq_len": cfg.seq_len,
        "micro_batch_size": cfg.micro_batch_size,
        "grad_accum_steps": cfg.grad_accum_steps,
        "max_steps": cfg.max_steps,
        "warmup_steps": cfg.warmup_steps,
        "resume_warmup_steps": cfg.resume_warmup_steps,
        "log_every": cfg.log_every,
        "eval_every": cfg.eval_every,
        "eval_batches": cfg.eval_batches,
    }
    for name, value in positive.items():
        if value <= 0:
            raise ValueError(f"{name} must be > 0, got {value}")
    nonnegative = {
        "save_every": cfg.save_every,
        "save_every_seconds": cfg.save_every_seconds,
        "max_saved_checkpoints": cfg.max_saved_checkpoints,
        "grad_clip_every": cfg.grad_clip_every,
    }
    for name, value in nonnegative.items():
        if value < 0:
            raise ValueError(f"{name} must be >= 0, got {value}")
    if cfg.grad_clip < 0:
        raise ValueError(f"grad_clip must be >= 0, got {cfg.grad_clip}")
    if cfg.min_lr < 0 or cfg.lr <= 0:
        raise ValueError(f"lr must be > 0 and min_lr >= 0, got lr={cfg.lr}, min_lr={cfg.min_lr}")
    if cfg.min_lr > cfg.lr:
        raise ValueError(f"min_lr ({cfg.min_lr}) must be <= lr ({cfg.lr})")
    if cfg.engram_zero_value_proj not in (-1, 0, 1):
        raise ValueError(
            "engram_zero_value_proj must be -1 (auto), 0 (keep patch), or 1 (force zero)"
        )


def sample_windows(arr: np.memmap, lo: int, hi: int, seq_len: int,
                   batch_size: int, rng: np.random.Generator) -> torch.Tensor:
    """Draw `batch_size` random non-overlapping-ish windows from arr[lo:hi].
    seq_len+1 so we can shift for LM loss.

    Vectorized: one fancy-index into memmap, then a single dtype cast to int64.
    Previously did batch_size separate slices + casts — dominated by Python
    overhead at larger batch sizes.
    """
    max_start = hi - (seq_len + 1)
    if max_start < lo:
        raise ValueError(
            f"range [{lo}, {hi}) is too short for seq_len={seq_len} "
            f"(need at least {seq_len + 1} tokens)"
        )
    if batch_size == 1:
        start = int(rng.integers(low=lo, high=max_start + 1))
        out = arr[start:start + seq_len + 1].astype(np.int64, copy=True)[None, :]
        return torch.from_numpy(out)
    starts = rng.integers(low=lo, high=max_start + 1, size=batch_size)
    # Build [batch, seq_len+1] index grid and gather in one call.
    offsets = np.arange(seq_len + 1, dtype=np.int64)
    idx = starts[:, None].astype(np.int64) + offsets[None, :]
    out = arr[idx.reshape(-1)].astype(np.int64).reshape(batch_size, seq_len + 1)
    return torch.from_numpy(out)


class _TrainWindowIter(torch.utils.data.IterableDataset):
    """Infinite stream of (seq_len+1,) int64 windows sampled randomly from
    arr[lo:hi]. Designed to run in a DataLoader worker so the memmap reads
    overlap with GPU compute instead of stalling the training thread.

    Determinism: each worker seeds its own rng from `seed + worker_id`. We
    typically use num_workers=1 so the effective seed is `seed` exactly. Not
    bit-identical to the in-thread rng since we now create the Generator per
    process, but reproducible within a run.
    """
    def __init__(self, arr_path: str, lo: int, hi: int, seq_len: int, seed: int):
        super().__init__()
        self.arr_path = arr_path
        self.lo = lo
        self.hi = hi
        self.seq_len = seq_len
        self.seed = seed

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        worker_seed = int(self.seed) + (info.id if info is not None else 0)
        rng = np.random.default_rng(worker_seed)
        arr = np.memmap(self.arr_path, dtype=np.uint32, mode="r")
        max_start = self.hi - (self.seq_len + 1)
        while True:
            start = int(rng.integers(low=self.lo, high=max_start + 1))
            sample = arr[start:start + self.seq_len + 1].astype(np.int64, copy=True)
            yield torch.from_numpy(sample)


# ---------------------------------------------------------------------------
# Schedule. Three options selectable via cfg.lr_schedule:
#   cosine  : linear warmup → cosine decay to min_lr (legacy default)
#   wsd     : linear warmup → constant at peak for `wsd_stable_fraction` of
#             post-warmup steps → cosine decay tail to min_lr
#   constant: linear warmup → hold at peak (no decay)
# ---------------------------------------------------------------------------
def _post_warmup_lr(step: int, cfg: TrainConfig, schedule_start: int) -> float:
    """LR for steps after warmup, dispatched by cfg.lr_schedule. `schedule_start`
    is the step at which warmup ended (so the schedule's "0" is at this step)."""
    warm = cfg.warmup_steps if schedule_start == 0 else 0
    total = max(1, cfg.max_steps - schedule_start - warm)
    rel = max(0.0, (step - schedule_start - warm) / total)
    rel = min(1.0, rel)

    schedule = getattr(cfg, "lr_schedule", "cosine")
    if schedule == "constant":
        return cfg.lr
    if schedule == "wsd":
        stable = max(0.0, min(1.0, getattr(cfg, "wsd_stable_fraction", 0.7)))
        if rel <= stable:
            return cfg.lr
        # Cosine tail over the final (1 - stable) fraction.
        tail_progress = (rel - stable) / max(1e-9, 1.0 - stable)
        coeff = 0.5 * (1 + math.cos(math.pi * tail_progress))
        return cfg.min_lr + (cfg.lr - cfg.min_lr) * coeff
    # cosine (default / legacy)
    coeff = 0.5 * (1 + math.cos(math.pi * rel))
    return cfg.min_lr + (cfg.lr - cfg.min_lr) * coeff


def lr_at(step: int, cfg: TrainConfig, start_step: int = 0) -> float:
    # Fresh training: initial warmup, then schedule.
    if start_step == 0:
        if step < cfg.warmup_steps:
            return cfg.lr * (step + 1) / cfg.warmup_steps
        return _post_warmup_lr(step, cfg, schedule_start=0)

    # Resume: linear re-warmup from min_lr to peak over resume_warmup_steps,
    # then the chosen schedule over the remaining steps.
    rewarmup_end = start_step + cfg.resume_warmup_steps
    if step < rewarmup_end:
        frac = (step - start_step + 1) / cfg.resume_warmup_steps
        return cfg.min_lr + (cfg.lr - cfg.min_lr) * frac
    return _post_warmup_lr(step, cfg, schedule_start=rewarmup_end)


# ---------------------------------------------------------------------------
# Reduce-on-Plateau LR controller. Wraps the base schedule with a reactive
# multiplier driven by eval h1_ppl: when ppl stops improving by `rel_threshold`
# for `patience` consecutive evals, multiply lr by `factor` (floor at
# `min_mult`). The base schedule still sets the upper envelope; ROP only
# reduces. Designed to compose with --lr-schedule constant or wsd.
# ---------------------------------------------------------------------------
class _PlateauLRController:
    def __init__(self, cfg: TrainConfig):
        self.enabled = bool(cfg.rop_enable)
        self.patience = int(cfg.rop_patience)
        self.factor = float(cfg.rop_factor)
        self.rel_threshold = float(cfg.rop_rel_threshold)
        self.min_mult = float(cfg.rop_min_mult)
        self.warmup_grace = int(cfg.rop_warmup_grace_steps)
        self.cooldown_total = int(cfg.rop_cooldown)
        self.best = float("inf")
        self.bad_count = 0
        self.cooldown = 0
        self.mult = 1.0
        self.n_drops = 0

    def step(self, eval_ppl: float, train_step: int) -> tuple[float, bool]:
        """Return (current_mult, dropped_this_eval)."""
        if not self.enabled:
            return 1.0, False
        dropped = False
        if train_step < self.warmup_grace:
            return self.mult, False
        if self.cooldown > 0:
            self.cooldown -= 1
            self.best = min(self.best, eval_ppl)
            return self.mult, False
        improvement = (self.best - eval_ppl) / max(self.best, 1e-9) if self.best != float("inf") else 1.0
        if improvement >= self.rel_threshold:
            self.best = eval_ppl
            self.bad_count = 0
        else:
            self.bad_count += 1
            if self.bad_count >= self.patience:
                new_mult = max(self.min_mult, self.mult * self.factor)
                if new_mult < self.mult:
                    self.mult = new_mult
                    self.n_drops += 1
                    dropped = True
                self.bad_count = 0
                self.cooldown = self.cooldown_total
        return self.mult, dropped


# ---------------------------------------------------------------------------
# Phase C v4: GuardedMuonClip + per-param routing helpers.
#
# muon-clip's `muon_update` (utils_muon/utils.py:409) ends with
#     grad = (0.4 * sqrt(max(rows, cols))) * grad
# i.e. a corrected-RMS amplifier on top of the orthogonalized update. For our
# 4096-dim hidden matrices that is ~25.6×; for 151936×4096 embeddings ~155×.
# Vanilla Muon (Keller Jordan) has no such factor, so reusing Phase B's vanilla
# `lr=5e-4` under MuonClip put effective step magnitudes 25-60× too high. Two
# Phase C runs diverged catastrophically on this.
#
# GuardedMuonClip overrides single_muon_step to cap each param's effective
# update so that
#     lr * RMS(update) / RMS(p) <= max_<muon|adam>_ratio
# without changing the orthogonalized direction. Diagnostic counters record
# how many params saturated the cap in the most recent step.
# ---------------------------------------------------------------------------
def _is_muon_matrix(name: str, p: torch.Tensor) -> bool:
    """Codex-recommended routing: 2D weight tensors that are NOT embeddings or
    output heads. Embeddings (151936×4096) under MuonClip's 0.4*sqrt(155936)
    ≈ 155× amplifier are structurally unsafe; route them out and let an Adam
    family optimizer handle them."""
    if p.ndim != 2:
        return False
    if "embed_tokens" in name or "lm_head" in name:
        return False
    return True


class _ParamProxy(torch.nn.Module):
    """Fake nn.Module that yields a fixed (name, param) list from
    `named_parameters()`. Used to feed MuonClip a curated subset of the model's
    params (e.g. excluding embed_tokens / lm_head) without modifying the model
    itself or MuonClip's internal routing."""
    def __init__(self, named_params):
        super().__init__()
        self._named_params = list(named_params)

    def named_parameters(self, prefix: str = "", recurse: bool = True,
                         remove_duplicate: bool = True):
        for n, p in self._named_params:
            yield (prefix + n if prefix else n), p


def _make_guarded_muonclip_class():
    """Lazy class factory: imports the muon-clip pkg only if the user actually
    enables `guarded_muonclip=1`. Returns a subclass of MuonClip with a guarded
    `single_muon_step`."""
    from muon import MuonClip
    from utils_muon import muon_update, adam_update

    class GuardedMuonClip(MuonClip):
        """MuonClip subclass whose per-param effective step is capped at
        `lr * RMS(update) / RMS(p) <= max_<muon|adam>_ratio`.

        Direction is unchanged (Newton-Schulz output is preserved); only the
        per-param scalar `alpha` is reduced. Set max_*_ratio big enough and the
        guard is a no-op; small enough and it acts as a hard stability cap that
        prevents the corrected-RMS amplifier from blowing up early-step updates.
        """
        def __init__(self, *args, max_muon_ratio: float = 5e-4,
                     max_adam_ratio: float = 1e-4, **kwargs):
            super().__init__(*args, **kwargs)
            assert not self.enable_clipping, (
                "GuardedMuonClip assumes enable_clipping=False — QK-clipping is "
                "incompatible with MLA's q_a_proj/q_b_proj naming."
            )
            self.max_muon_ratio = float(max_muon_ratio)
            self.max_adam_ratio = float(max_adam_ratio)
            # Per-step diagnostics — read by the train loop and logged.
            self.last_guard_saturation = {
                "muon_sat": 0, "muon_total": 0,
                "adam_sat": 0, "adam_total": 0,
            }

        @torch.no_grad()
        def single_muon_step(self, closure=None):
            loss = None
            if closure is not None:
                with torch.enable_grad():
                    loss = closure()

            # GPU accumulators: incremented in-loop without syncing, read once at end.
            dev = next((p.device for g in self.param_groups for p in g["params"]
                        if p.requires_grad), torch.device("cpu"))
            muon_sat_t = torch.zeros((), device=dev, dtype=torch.long)
            adam_sat_t = torch.zeros((), device=dev, dtype=torch.long)
            muon_total = 0
            adam_total = 0

            for group in self.param_groups:
                lr = float(group["lr"])
                wd = float(group.get("weight_decay", 0.0))
                if group["use_muon"]:
                    cap = self.max_muon_ratio
                    for p in group["params"]:
                        if p.grad is None:
                            p.grad = torch.zeros_like(p)
                        state = self.state[p]
                        if len(state) == 0:
                            state["momentum_buffer"] = torch.zeros_like(p)
                            state["velocity_buffer"] = torch.zeros(
                                (p.size(-2), 1), device=p.device,
                            )
                            state["step"] = 0
                        state["step"] += 1
                        update = muon_update(
                            p.grad, state["momentum_buffer"], state["velocity_buffer"],
                            step=state["step"], beta=group["beta"], eps=group["eps"],
                            ortho_polynomials=self.ortho_polynomials,
                            ns_steps=self.ns_steps, cans_ortho=self.cans_ortho,
                        )
                        # GPU-resident scaling: lr * RMS(update) / RMS(p) <= cap.
                        p_rms = p.detach().float().square().mean().sqrt().clamp_min(1e-12)
                        u_rms = update.float().square().mean().sqrt().clamp_min(1e-12)
                        rho = lr * u_rms / p_rms
                        scale = torch.clamp(cap / rho.clamp_min(1e-12), max=1.0)
                        update.mul_(scale)
                        muon_sat_t += (scale < 0.99).long()
                        muon_total += 1
                        if wd:
                            p.mul_(1 - lr * wd)
                        p.add_(update.reshape(p.shape), alpha=-lr)
                else:  # Adam side
                    cap = self.max_adam_ratio
                    for p in group["params"]:
                        if p.grad is None:
                            p.grad = torch.zeros_like(p)
                        state = self.state[p]
                        if len(state) == 0:
                            state["exp_avg"] = torch.zeros_like(p)
                            state["exp_avg_sq"] = torch.zeros_like(p)
                            state["step"] = 0
                        state["step"] += 1
                        update = adam_update(
                            p.grad, state["exp_avg"], state["exp_avg_sq"],
                            state["step"], group["betas"], group["eps"],
                        )
                        p_rms = p.detach().float().square().mean().sqrt().clamp_min(1e-12)
                        u_rms = update.float().square().mean().sqrt().clamp_min(1e-12)
                        rho = lr * u_rms / p_rms
                        scale = torch.clamp(cap / rho.clamp_min(1e-12), max=1.0)
                        update.mul_(scale)
                        adam_sat_t += (scale < 0.99).long()
                        adam_total += 1
                        if wd:
                            p.mul_(1 - lr * wd)
                        p.add_(update, alpha=-lr)

            # Single sync per step for diagnostics.
            self.last_guard_saturation = {
                "muon_sat": int(muon_sat_t.item()),
                "muon_total": muon_total,
                "adam_sat": int(adam_sat_t.item()),
                "adam_total": adam_total,
            }
            self._step += 1
            return loss

    return GuardedMuonClip


# ---------------------------------------------------------------------------
# Eval: cross-entropy over fixed windows from the held-out range.
# ---------------------------------------------------------------------------
@torch.no_grad()
def eval_loss(model, arr: np.memmap, eval_start: int, eval_end: int,
              cfg: TrainConfig, device: torch.device) -> tuple[float, float]:
    model.eval()
    text_model = _text_model(model)
    rng = np.random.default_rng(12345)  # deterministic eval
    total_loss = 0.0
    total_tokens = 0
    for _ in range(cfg.eval_batches):
        ids = sample_windows(arr, eval_start, eval_end, cfg.seq_len,
                             cfg.micro_batch_size, rng).to(device)
        x, y = ids[:, :-1], ids[:, 1:]
        outputs = text_model(input_ids=x, use_cache=False)
        hidden = _last_hidden(outputs)
        loss = chunked_lmhead_ce(hidden, model.lm_head, y) * y.numel()
        total_loss += loss.item()
        total_tokens += y.numel()
        del ids, x, y, outputs, hidden, loss
    model.train()
    # Defragment the allocator pool so the next training step's activation tensors
    # can find contiguous space. Without this we OOM a few steps after eval.
    torch.cuda.empty_cache()
    mean_loss = total_loss / total_tokens
    return mean_loss, math.exp(mean_loss)


@torch.no_grad()
def multi_horizon_eval(model, arr: np.memmap, eval_start: int, eval_end: int,
                       cfg: TrainConfig, device: torch.device,
                       horizons: tuple[int, ...] = (1, 2, 3, 4)) -> dict:
    """Return {horizon k: {ppl, loss, top1, top5, tokens}} for each horizon.
    h=1 uses backbone; h>=2 uses k-1 chained MTP calls teacher-forced with
    ground-truth intermediate tokens."""
    model.eval()
    mtp = getattr(model, "mtp_trainer", None)
    max_k = max(horizons)
    if mtp is None:
        max_k = 1

    text_model = _text_model(model)

    totals = {k: {"loss_sum": 0.0, "top1": 0, "top5": 0, "n": 0} for k in horizons}

    # Force eval batch to 1 regardless of training batch, to keep peak memory
    # bounded when we materialize per-chunk logits for multiple horizons.
    eval_mb = 1

    # Cache the eval batches across calls. Eval uses a fixed-seed RNG so the
    # same cfg.eval_batches windows are drawn every time; keeping the tensors
    # alive gives them stable data_ptr, which lets EngramModule's eval hash
    # cache hit across eval calls (saves one full hash compute per engram
    # layer per batch per eval).
    cache_key = (cfg.seq_len, cfg.eval_batches, eval_start, eval_end)
    if getattr(model, "_eval_batches_cache_key", None) != cache_key:
        rng_cache = np.random.default_rng(12345)
        model._eval_batches_cache = [
            sample_windows(arr, eval_start, eval_end, cfg.seq_len, eval_mb, rng_cache).to(device)
            for _ in range(cfg.eval_batches)
        ]
        model._eval_batches_cache_key = cache_key
    _eval_batches = model._eval_batches_cache

    for _b_idx in range(cfg.eval_batches):
        w_full = _eval_batches[_b_idx]
        B, L = w_full.shape
        T = L - 1
        x = w_full[:, :-1]

        outputs = text_model(input_ids=x, use_cache=False)
        backbone_hidden = _last_hidden(outputs)
        if 1 in horizons:
            m = chunked_lmhead_metrics(backbone_hidden, model.lm_head, w_full[:, 1:])
            totals[1]["loss_sum"] += m["loss_sum"]
            totals[1]["top1"] += m["top_counts"][1]
            totals[1]["top5"] += m["top_counts"][5]
            totals[1]["n"] += m["n_tokens"]

        if mtp is not None and max_k >= 2:
            h_prev = backbone_hidden.detach()
            # Free hidden_states tuple — not needed for h>=2 after h_prev.
            del outputs, backbone_hidden
            # Precompute RoPE once for the full position range up to max_k-1..L-1;
            # each iteration slices instead of recalling text_model.rotary_emb.
            full_pos = torch.arange(0, L, device=device).unsqueeze(0).expand(B, -1)
            cos_all, sin_all = text_model.rotary_emb(h_prev, full_pos)
            for k in range(2, max_k + 1):
                seg = L - k
                if seg <= 0:
                    break
                h_in = h_prev[:, :seg]
                ids_in = w_full[:, k - 1:L - 1]
                pos_in = full_pos[:, k - 1:L - 1]
                cos_in = cos_all[:, k - 1:L - 1]
                sin_in = sin_all[:, k - 1:L - 1]

                mtp_out = mtp(
                    input_ids=ids_in,
                    hidden_states=h_in,
                    position_embeddings=(cos_in, sin_in),
                    position_ids=pos_in,
                )
                if k in horizons:
                    labels_k = w_full[:, k:L]
                    m = chunked_lmhead_metrics(mtp_out, model.lm_head, labels_k)
                    totals[k]["loss_sum"] += m["loss_sum"]
                    totals[k]["top1"] += m["top_counts"][1]
                    totals[k]["top5"] += m["top_counts"][5]
                    totals[k]["n"] += m["n_tokens"]
                    del labels_k
                # Next chain step uses mtp_out as hidden input; release h_prev.
                h_prev_old = h_prev
                h_prev = mtp_out
                del h_prev_old, cos_in, sin_in, pos_in, h_in, ids_in
        else:
            del outputs, backbone_hidden
        del w_full, x

    model.train()
    torch.cuda.empty_cache()
    out = {}
    for k in horizons:
        t = totals[k]
        if t["n"] > 0:
            mean_loss = t["loss_sum"] / t["n"]
            out[k] = {
                "loss": mean_loss,
                "ppl": math.exp(mean_loss),
                "top1_acc": t["top1"] / t["n"],
                "top5_acc": t["top5"] / t["n"],
                "tokens": t["n"],
            }
    return out


# ---------------------------------------------------------------------------
# Checkpoint: save only the MLA params + optimizer state + step.
# ---------------------------------------------------------------------------
def _prune_old_checkpoints(out_dir: Path, keep: int) -> None:
    """Delete all but the most-recent `keep` step_*/ directories. No-op if
    keep <= 0. The final post-loop save is never pruned (caller saves after
    max_steps so it's always the most recent)."""
    if keep <= 0:
        return
    ckpts = sorted(
        [d for d in out_dir.iterdir() if d.is_dir() and d.name.startswith("step_")],
        key=lambda d: d.name,  # lexicographic == step order with zero-padded names
    )
    for d in ckpts[:-keep]:
        try:
            shutil.rmtree(d)
            print(f"[ckpt prune] removed {d.name}")
        except OSError as e:
            print(f"[ckpt prune] failed to remove {d.name}: {e}")


def save_checkpoint(step: int, mla_modules: list[MLAAttention], optimizer, cfg: TrainConfig,
                    model=None, engram_modules: list | None = None,
                    optimizer_host=None, optimizer_aux=None) -> None:
    # Write to a dot-prefixed temp dir first, then rename. Makes the save
    # atomic: a kill mid-write leaves a .step_XXXXXX.tmp/ that the resume
    # code ignores (only step_XXXXXX/ dirs match the glob).
    out_final = Path(cfg.out_dir) / f"step_{step:06d}"
    out = Path(cfg.out_dir) / f".step_{step:06d}.tmp"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    # Collect MLA state_dicts. Keys are each module's `_save_key` attr
    # (set by load_converted_model): "layer_{i}" for backbone MLA modules
    # and "mtp" for the MTP's MLA module if present.
    mla_sds: dict = {}
    for m in mla_modules:
        key = getattr(m, "_save_key", None)
        if key is None:
            raise RuntimeError(
                "MLA module missing _save_key attribute — older load path? "
                "Re-load via load_converted_model to get the tagged modules."
            )
        mla_sds[key] = {k: v.detach().cpu() for k, v in m.state_dict().items()}

    # MTP's non-attn trainable params (fc, norms, decoder MLP, etc.) — save
    # everything on the MTP module EXCEPT the shared embed_tokens and the
    # self_attn (already saved via MLA list as key="mtp").
    mtp_extra_sd = None
    if model is not None and getattr(model, "mtp_trainer", None) is not None:
        mtp = model.mtp_trainer
        shared_embed_id = id(mtp.embed_tokens.weight) if mtp.embed_tokens is not None else None
        # collect by named_parameters so we can skip embed_tokens and the self_attn
        # (MLA'd self_attn is saved separately under key "mtp" above).
        mtp_extra_sd = {}
        for name, p in mtp.named_parameters():
            if id(p) == shared_embed_id:
                continue
            if ".self_attn." in name or name.startswith("layers.0.self_attn"):
                continue
            mtp_extra_sd[name] = p.detach().cpu()
    payload = {
        "step": step,
        "mla_state_dicts": mla_sds,
        "optimizer_state": optimizer.state_dict(),
        "config": asdict(cfg),
    }
    if mtp_extra_sd is not None:
        payload["mtp_extra_state_dict"] = mtp_extra_sd

    # Phase 2 auxiliary heads. Small, always worth saving when present.
    if model is not None and getattr(model, "mutor_head", None) is not None:
        payload["mutor_state_dict"] = {
            k: v.detach().cpu() for k, v in model.mutor_head.state_dict().items()
        }
    if model is not None and getattr(model, "fsp_head", None) is not None:
        # Skip non-persistent buffers (vocab_w if empty). state_dict() already
        # handles persistent flags correctly, so we can pass as-is.
        payload["fsp_state_dict"] = {
            k: v.detach().cpu() for k, v in model.fsp_head.state_dict().items()
        }
    if model is not None and getattr(model, "parallel_heads", None) is not None:
        payload["parallel_heads_state_dict"] = {
            k: v.detach().cpu() for k, v in model.parallel_heads.state_dict().items()
        }
        payload["parallel_horizons"] = list(model.parallel_heads.horizons)
    # Phase 3 — Engram. Save per-layer state_dicts keyed by layer_id. The big
    # embedding tables (host-offloaded, sometimes 40GB+) live on pinned CPU
    # memory already, so .detach().cpu() is essentially a no-op for them.
    if engram_modules:
        payload["engram_state_dicts"] = {
            f"layer_{m.layer_id}": {k: v.detach().cpu() for k, v in m.state_dict().items()}
            for m in engram_modules
        }
    if optimizer_host is not None:
        payload["optimizer_state_host"] = optimizer_host.state_dict()
    if optimizer_aux is not None:
        payload["optimizer_state_aux"] = optimizer_aux.state_dict()

    # When freeze_non_mla=0 (full unfreeze), the backbone (token embeddings,
    # decoder MLPs, norms, lm_head, MTP/MoE FFN, etc.) is being trained too.
    # Without this block those weights are NEVER persisted — they'd be reset
    # to the base safetensors on resume, silently throwing away every step
    # of full-unfreeze training. We did exactly that mistake on Phase B
    # qwen9b_phaseB_unfrozen and lost ~600 steps of backbone learning;
    # this guard prevents that from ever recurring.
    if model is not None and getattr(cfg, "freeze_non_mla", 1) == 0:
        # Set of param-id()s already covered by dedicated saves (MLA, MTP-extras,
        # aux heads, engram). Everything else is "backbone" and goes here.
        covered_ids = set()
        for m in mla_modules:
            for p in m.parameters():
                covered_ids.add(id(p))
        if model.mtp_trainer is not None if hasattr(model, "mtp_trainer") else False:
            mtp = model.mtp_trainer
            shared_id = id(mtp.embed_tokens.weight) if mtp.embed_tokens is not None else None
            for p in mtp.parameters():
                # mtp_extra_sd already saved everything except shared embed and self_attn;
                # self_attn params are also in mla_modules' covered_ids above.
                if shared_id is not None and id(p) == shared_id:
                    continue
                covered_ids.add(id(p))
        if hasattr(model, "mutor_head") and model.mutor_head is not None:
            for p in model.mutor_head.parameters():
                covered_ids.add(id(p))
        if hasattr(model, "fsp_head") and model.fsp_head is not None:
            for p in model.fsp_head.parameters():
                covered_ids.add(id(p))
        if hasattr(model, "parallel_heads") and model.parallel_heads is not None:
            for p in model.parallel_heads.parameters():
                covered_ids.add(id(p))
        if engram_modules:
            for em in engram_modules:
                for p in em.parameters():
                    covered_ids.add(id(p))
        backbone_sd = {n: p.detach().cpu() for n, p in model.named_parameters()
                       if id(p) not in covered_ids}
        payload["backbone_state_dict"] = backbone_sd
        n_backbone = sum(t.numel() for t in backbone_sd.values())
        print(f"  [ckpt] saved backbone_state_dict: {len(backbone_sd)} tensors, "
              f"{n_backbone/1e9:.2f}B params")

    torch.save(payload, out / "ckpt.pt")
    # Sidecar JSON of cfg + step so the dashboard can read run config cheaply
    # (no torch.load on a multi-GB checkpoint).
    (out / "config.json").write_text(json.dumps({
        "step": step,
        "config": asdict(cfg),
    }, indent=2))
    # Atomic commit: rename temp dir to final. If final exists (re-save of the
    # same step), replace it.
    if out_final.exists():
        shutil.rmtree(out_final)
    out.rename(out_final)


def _full_attn_indices(cfg: TrainConfig) -> list[int]:
    manifest = json.loads((Path(cfg.patch_dir) / "manifest.json").read_text())
    return manifest["full_attn_layer_indices"]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    for f in TrainConfig.__dataclass_fields__.values():
        ap.add_argument(f"--{f.name.replace('_','-')}", type=type(f.default), default=f.default)
    args = ap.parse_args()
    cfg = TrainConfig(**{k.replace("-","_"): v for k, v in vars(args).items()})
    validate_train_config(cfg)

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_f = (out_dir / "train.jsonl").open("a")
    def log(row: dict) -> None:
        log_f.write(json.dumps(row) + "\n")
        log_f.flush()

    print("Loading model (this can take ~1 minute)...")
    model, mla_modules = load_converted_model(
        model_dir=cfg.model_dir, patch_dir=cfg.patch_dir,
        device_map="cuda:0", dtype=torch.bfloat16,
        freeze_non_mla=bool(cfg.freeze_non_mla),
        install_mtp=bool(cfg.install_mtp),
        rwkv8_deltanet_layers=cfg.rwkv8_deltanet_layers,
        rwkv8_ffn_hidden_size=cfg.rwkv8_ffn_hidden_size or None,
        rwkv8_init_output_scale=cfg.rwkv8_init_output_scale,
        rwkv8_swap_mode=cfg.rwkv8_swap_mode,
    )
    if not cfg.freeze_non_mla:
        print("freeze_non_mla=0: backbone + MTP + lm_head + embeddings all trainable")
    mtp_installed = getattr(model, "mtp_trainer", None) is not None

    if cfg.xsa_enabled:
        for m in mla_modules:
            if isinstance(m, MLAAttention):
                m.xsa_enabled = True
        print("xsa_enabled: toggled on MLA modules")

    # MTP-only mode: freeze backbone MLA, only MTP params remain trainable.
    # Speeds up backward ~3x and lets us wrap backbone forward in no_grad.
    if cfg.train_mtp_only:
        if not mtp_installed:
            raise ValueError("--train-mtp-only=1 requires --install-mtp=1")
        for m in mla_modules:
            if getattr(m, "_save_key", "") != "mtp":
                for p in m.parameters():
                    p.requires_grad_(False)
        print("train_mtp_only: backbone MLA frozen; only MTP trainable.")

    # Aux-only mode: freeze EVERYTHING (backbone MLA + MTP + lm_head) except
    # the aux heads. Use this to warm up MuToR / FSP / parallel heads from
    # random init without perturbing the already-converged MTP.
    if cfg.train_aux_only:
        if not mtp_installed:
            raise ValueError("--train-aux-only=1 requires --install-mtp=1")
        # Freeze backbone MLA
        for m in mla_modules:
            for p in m.parameters():
                p.requires_grad_(False)
        # Freeze all MTP params (module itself, including the MLA attention swapped in)
        if getattr(model, "mtp_trainer", None) is not None:
            for p in model.mtp_trainer.parameters():
                p.requires_grad_(False)
        print("train_aux_only: everything frozen except aux heads "
              "(mutor_head, fsp_head, parallel_heads).")

    # MuToR / FSP-BCE auxiliary heads (Phase 2 enhancements). Attach to the
    # model so they're auto-discovered by model.parameters() and saved in
    # checkpoints. Kept trainable regardless of train_mtp_only freezing — they
    # are new params that only matter when turned on via flags.
    text_cfg = getattr(model.config, "text_config", model.config)
    _hidden_size = text_cfg.hidden_size
    _vocab_size = text_cfg.vocab_size
    _model_device = next(model.parameters()).device
    _model_dtype = next(model.parameters()).dtype
    if cfg.mutor_enabled:
        mh = MuToRHead(hidden_size=_hidden_size, d_max=cfg.mutor_d_max).to(
            device=_model_device, dtype=_model_dtype)
        for p in mh.parameters():
            p.requires_grad_(True)
        model.mutor_head = mh
        print(f"mutor enabled: d=[{cfg.mutor_d_min},{cfg.mutor_d_max}] "
              f"K={cfg.mutor_num_registers} w={cfg.mutor_weight}")
    if cfg.fsp_enabled:
        fh = FSPHead(hidden_size=_hidden_size,
                     idf_path=cfg.fsp_idf_path or None,
                     vocab_size=_vocab_size).to(
            device=_model_device, dtype=_model_dtype)
        for p in fh.parameters():
            p.requires_grad_(True)
        model.fsp_head = fh
        print(f"fsp enabled: tau={cfg.fsp_tau} K={cfg.fsp_num_positions} "
              f"w={cfg.fsp_weight} idf={'yes' if fh.has_vocab_weights() else 'no'}")
    if cfg.parallel_enabled:
        _ph_horizons = [int(h.strip()) for h in cfg.parallel_horizons.split(",")]
        _ph_weights = [float(w.strip()) for w in cfg.parallel_weights.split(",")]
        if len(_ph_weights) != len(_ph_horizons):
            raise ValueError("parallel_horizons and parallel_weights must have same length")
        ph = ParallelHeads(
            hidden_size=_hidden_size, horizons=_ph_horizons,
            expansion=cfg.parallel_head_expansion,
        ).to(device=_model_device, dtype=_model_dtype)
        for p in ph.parameters():
            p.requires_grad_(True)
        model.parallel_heads = ph
        n_ph = sum(p.numel() for p in ph.parameters())
        print(f"parallel_heads enabled: horizons={_ph_horizons} weights={_ph_weights} "
              f"expansion={cfg.parallel_head_expansion}  params={n_ph/1e6:.1f}M")
    # Cache parsed lists once for the train loop (falsy when disabled).
    _ph_horizons_cache = [int(h.strip()) for h in cfg.parallel_horizons.split(",")] if cfg.parallel_enabled else []
    _ph_weights_cache = [float(w.strip()) for w in cfg.parallel_weights.split(",")] if cfg.parallel_enabled else []

    # Engram install. Must come AFTER aux heads are attached (so freeze logic
    # can distinguish engram params from aux/MTP/MLA) and BEFORE the
    # train_engram_only freeze pass (which freezes everything non-engram).
    engram_modules: list = []
    optimizer_host = None  # SparseAdam for host-offloaded embedding tables
    optimizer_aux = None   # Phase C v4: Prodigy for non-matrix params + embed/lm_head
    if cfg.engram_enabled:
        if not cfg.engram_patch_dir:
            raise ValueError("--engram-enabled=1 requires --engram-patch-dir=<dir>")
        # Deferred imports — engram_ext is under /thearray/git/engram/python
        import sys as _sys
        _ENGRAM_PY = "/thearray/git/engram/python"
        if _ENGRAM_PY not in _sys.path:
            _sys.path.insert(0, _ENGRAM_PY)
        from engram_ext.engram_module import EngramConfig as _EngramConfig
        from engram_integration import install_engram
        from load_mla_engram import _apply_engram_patch, _read_patch

        eng_manifest = json.loads((Path(cfg.engram_patch_dir) / "manifest.json").read_text())
        eng_cfg = _EngramConfig(**eng_manifest["engram_config"])
        _host_offload = bool(eng_manifest.get("host_offload", False))
        engram_modules = install_engram(
            model,
            layer_indices=eng_manifest["layer_indices"],
            engram_cfg=eng_cfg,
            hidden_size=eng_manifest["hidden_size"],
            host_offload_embedding=_host_offload,
        )
        _patch = _read_patch(Path(cfg.engram_patch_dir) / "patch.safetensors")
        _apply_engram_patch(engram_modules, _patch, eng_manifest["layer_indices"])
        del _patch
        _prefilled = bool(eng_manifest.get("prefilled", False))
        _zero_value_proj = (
            not _prefilled
            if cfg.engram_zero_value_proj == -1
            else bool(cfg.engram_zero_value_proj)
        )
        if _zero_value_proj:
            # Identity-init override for fresh/random Engram patches: zero
            # value_proj.weight so the Engram residual output is exactly zero
            # at step 0. Do NOT do this for prefilled patches, whose
            # value_proj is part of the learned standalone N-gram signal.
            with torch.no_grad():
                for _em in engram_modules:
                    _em.value_proj.weight.zero_()
            _value_proj_note = "value_proj zero-inited for identity start"
        else:
            _value_proj_note = "value_proj kept from patch"
        _n_eng = sum(p.numel() for m in engram_modules for p in m.parameters())
        print(f"engram installed: {len(engram_modules)} modules at layers "
              f"{eng_manifest['layer_indices']}  params={_n_eng/1e9:.2f}B  "
              f"host_offload={_host_offload}  prefilled={_prefilled}  "
              f"({_value_proj_note})")

    if cfg.train_engram_only:
        if not engram_modules:
            raise ValueError("--train-engram-only=1 requires --engram-enabled=1")
        # Freeze everything that is not an Engram param. This includes MLA,
        # MTP (if installed), lm_head, aux heads (mutor/fsp/parallel), and
        # the rest of the backbone (which was already frozen by
        # load_converted_model).
        _engram_pids = {id(p) for m in engram_modules for p in m.parameters()}
        for p in model.parameters():
            if id(p) not in _engram_pids:
                p.requires_grad_(False)
        print("train_engram_only: only Engram modules trainable (MLA/MTP/aux frozen).")

    if cfg.train_mla_only:
        # Freeze everything except the MLA modules. Use for Phase 1.5 BKV retrain:
        # fresh BKV-init MLA needs to learn to produce hidden states that work
        # with frozen Phase 2/3 downstream (MTP + aux + Engram). Distillation-style.
        _mla_pids = {id(p) for m in mla_modules for p in m.parameters()}
        frozen, trainable = 0, 0
        for p in model.parameters():
            if id(p) in _mla_pids:
                p.requires_grad_(True)
                trainable += 1
            else:
                p.requires_grad_(False)
                frozen += 1
        print(f"train_mla_only: {trainable} MLA params trainable, {frozen} others frozen "
              f"(MTP/aux/Engram/backbone all frozen).")

    if cfg.train_rwkv8_only:
        # Single-layer RWKV-8 probe: freeze everything except the RWKV-8
        # modules. Distinct from train_mla_only, which would also train the
        # MLA backbone + MTP MLA — for Stage 1 we want only the new module
        # to move while everything around it (including converged MLA) stays
        # put. Selects modules by ``_save_key`` prefix so it works for both
        # timemix and channelmix swap modes.
        if cfg.train_mla_only:
            raise ValueError("--train-rwkv8-only and --train-mla-only are mutually exclusive")
        if cfg.train_rwkv8_layers:
            raise ValueError("--train-rwkv8-only and --train-rwkv8-layers are mutually exclusive")
        _rwkv_modules = [m for m in mla_modules
                         if str(getattr(m, "_save_key", "")).startswith("rwkv8_layer_")]
        if not _rwkv_modules:
            raise ValueError("--train-rwkv8-only=1 requires --rwkv8-deltanet-layers set")
        _rwkv_pids = {id(p) for m in _rwkv_modules for p in m.parameters()}
        frozen, trainable = 0, 0
        for p in model.parameters():
            tr = id(p) in _rwkv_pids
            p.requires_grad_(tr)
            trainable += int(tr)
            frozen += int(not tr)
        _modes = sorted({str(getattr(m, "_swap_mode", "?")) for m in _rwkv_modules})
        print(f"train_rwkv8_only: {trainable} RWKV-8 params trainable, "
              f"{frozen} others frozen (modes={_modes}).")

    if cfg.train_rwkv8_layers:
        # Per-layer RWKV-8 freeze: train ONLY the listed RWKV-8 layer indices,
        # freeze everything else (including other RWKV-8 layers). Use this
        # to add a fresh layer at a time without disturbing already-converged
        # ones — avoids the multi-layer cascade overfit we saw at Stage 2b.
        if cfg.train_mla_only:
            raise ValueError("--train-rwkv8-layers and --train-mla-only are mutually exclusive")
        _wanted_idx: set[int] = set()
        for tok in cfg.train_rwkv8_layers.split(","):
            tok = tok.strip()
            if tok:
                _wanted_idx.add(int(tok))
        _wanted_keys = {f"rwkv8_layer_{i}" for i in _wanted_idx}
        _selected = [m for m in mla_modules
                     if str(getattr(m, "_save_key", "")) in _wanted_keys]
        _present_keys = {getattr(m, "_save_key", "") for m in _selected}
        _missing = _wanted_keys - _present_keys
        if _missing:
            raise ValueError(
                "--train-rwkv8-layers references modules not installed "
                f"(set --rwkv8-deltanet-layers to include them): {sorted(_missing)}"
            )
        if not _selected:
            raise ValueError("--train-rwkv8-layers parsed to an empty selection")
        _sel_pids = {id(p) for m in _selected for p in m.parameters()}
        frozen, trainable = 0, 0
        for p in model.parameters():
            tr = id(p) in _sel_pids
            p.requires_grad_(tr)
            trainable += int(tr)
            frozen += int(not tr)
        _sel_list = sorted(_present_keys)
        print(f"train_rwkv8_layers: {trainable} params trainable across "
              f"{_sel_list}; {frozen} others frozen.")

    # Parse chained-horizon schedule once (hoist out of the training loop).
    _chain_horizons = [int(h.strip()) for h in cfg.mtp_chain_horizons.split(",")]
    _chain_weights = [float(w.strip()) for w in cfg.mtp_chain_weights.split(",")]
    if len(_chain_weights) != len(_chain_horizons):
        raise ValueError("mtp_chain_horizons and mtp_chain_weights must have same length")
    for i, h in enumerate(_chain_horizons):
        if h != i + 2:
            raise ValueError(f"mtp_chain_horizons must be '2', '2,3', or '2,3,4'; got {cfg.mtp_chain_horizons}")
    if cfg.train_mtp_only and len(_chain_horizons) > 1:
        print(f"chained MTP loss: horizons={_chain_horizons}  weights={_chain_weights}")
    if cfg.train_mtp_only and (cfg.mutor_enabled or cfg.fsp_enabled or cfg.parallel_enabled):
        print("note: aux heads active in train_mtp_only mode — they'll train "
              "their own params (offset_emb, fsp_bias, parallel MLPs) against "
              "the frozen backbone hidden. Use this for aux-head warmup before "
              "unfreezing.")
    # Activation checkpointing across all decoder layers to cut activation memory.
    # Skip when backbone runs under no_grad (aux-only / mtp-only modes) — there
    # is no backward through backbone, so checkpointing only adds recompute cost
    # with zero memory benefit.
    _backbone_has_grad = not (cfg.train_mtp_only or cfg.train_aux_only)
    if _backbone_has_grad and cfg.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        # Selective: layers before the first Engram-enabled layer have no
        # trainable params behind them, so their recompute during backward
        # is pure waste. Their activations only need to survive to the
        # first Engram layer's backward, which is 1 forward step away.
        # Keep them in full-activation mode.
        if engram_modules:
            _min_engram_layer = min(m.layer_id for m in engram_modules)
            _text = _text_model(model)
            _disabled = 0
            for _i, _layer in enumerate(_text.layers):
                if _i < _min_engram_layer and hasattr(_layer, "gradient_checkpointing"):
                    _layer.gradient_checkpointing = False
                    _disabled += 1
            print(f"gradient checkpointing: enabled (disabled on {_disabled} "
                  f"pre-Engram layers [0, {_min_engram_layer - 1}])")
        else:
            print("gradient checkpointing: enabled (backbone has grad)")
    elif _backbone_has_grad and not cfg.gradient_checkpointing:
        print("gradient checkpointing: disabled (--gradient-checkpointing=0, uses more VRAM, ~30% faster backward)")
    else:
        print("gradient checkpointing: skipped (backbone runs in no_grad)")
    device = next(model.parameters()).device
    train_pos_base = torch.arange(0, cfg.seq_len, device=device).unsqueeze(0)

    trainable = [p for p in model.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in trainable)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"trainable params: {n_train/1e9:.3f}B / {n_total/1e9:.3f}B")

    # 8-bit AdamW via bitsandbytes: saves ~8GB of optimizer state vs fp32 AdamW
    # for a 1.3B trainable param count. Required to fit on 96GB with the 35B base.
    import bitsandbytes as bnb
    if engram_modules:
        # Split trainable params into three bins:
        #   - main (MLA / MTP / aux / anything else trainable): base LR, standard WD
        #   - engram GPU (projections, norms, conv): 5x LR, WD=0 (paper recipe)
        #   - engram host (the big embedding tables on pinned CPU RAM): 5x LR
        #     via torch.optim.SparseAdam, because the sparse=True backward path
        #     is incompatible with bnb's 8-bit dense AdamW.
        from engram_integration import collect_host_embedding_params, collect_gpu_engram_params
        eng_host_params = collect_host_embedding_params(engram_modules)
        eng_gpu_params = collect_gpu_engram_params(engram_modules)
        eng_ids = {id(p) for p in eng_host_params} | {id(p) for p in eng_gpu_params}
        main_params = [p for p in trainable if id(p) not in eng_ids]

        param_groups = [
            {"params": main_params, "lr": cfg.lr, "weight_decay": cfg.weight_decay},
        ]
        if eng_gpu_params:
            param_groups.append(
                {"params": eng_gpu_params, "lr": cfg.lr * cfg.engram_lr_mult,
                 "weight_decay": 0.0}
            )
        optimizer = bnb.optim.AdamW8bit(param_groups, betas=(0.9, 0.95))
        if eng_host_params:
            optimizer_host = torch.optim.SparseAdam(
                eng_host_params, lr=cfg.lr * cfg.engram_lr_mult, betas=(0.9, 0.95)
            )
        print(f"  main (MLA/MTP/aux/...): {sum(p.numel() for p in main_params)/1e6:.1f}M @ lr={cfg.lr:.1e}")
        if eng_gpu_params:
            print(f"  engram GPU (proj/norm/conv): {sum(p.numel() for p in eng_gpu_params)/1e6:.1f}M "
                  f"@ lr={cfg.lr*cfg.engram_lr_mult:.1e} wd=0")
        if eng_host_params:
            print(f"  engram host (SparseAdam): {sum(p.numel() for p in eng_host_params)/1e9:.2f}B "
                  f"@ lr={cfg.lr*cfg.engram_lr_mult:.1e}")
    else:
        if cfg.optimizer == "muonclip":
            # MuonClip from the muon-clip pkg. Constructor takes (model, model_config, MuonConfig)
            # and auto-routes 2D → Muon and other → Adam internally based on p.ndim.
            # We disable QK-clipping because its weight-rescale math hardcodes
            # standard q_proj/k_proj GQA layout that MLA breaks (see TrainConfig
            # comment). Corrected RMS still applies, so a single `lr` scales both
            # groups coherently.
            from muon import MuonClip, MuonConfig
            text_cfg = getattr(model.config, "text_config", model.config)

            # Phase C v4 routing: prodigy_aux=1 implies muon_exclude_embed_lmhead=1.
            # Build a curated `_ParamProxy` view of the model so MuonClip only sees
            # 2D matrices that aren't embeddings / output heads. Everything else
            # (1D norms+biases, embed_tokens, lm_head, MTP fc when 1D, etc.) goes
            # to Prodigy as a second optimizer.
            #
            # NOTE: `muon_exclude_embed_lmhead=1` is only meaningful WITH
            # `prodigy_aux=1`, because MuonClip's auto-routing puts every 2D
            # tensor in its Muon group; there is no way to keep embed_tokens
            # in MuonClip's Adam branch without forking the package. We enforce
            # this coupling rather than silently producing a no-op exclusion.
            use_prodigy = bool(cfg.prodigy_aux)
            if cfg.muon_exclude_embed_lmhead and not use_prodigy:
                raise ValueError(
                    "muon_exclude_embed_lmhead=1 requires prodigy_aux=1 to provide "
                    "an external optimizer for the excluded params. MuonClip's "
                    "auto-routing has no way to keep 2D tensors out of Muon "
                    "while still handling them internally."
                )
            if use_prodigy:
                named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
                muon_named = [(n, p) for n, p in named if _is_muon_matrix(n, p)]
                aux_params = [p for n, p in named if not _is_muon_matrix(n, p)]
                muon_target = _ParamProxy(muon_named)
            else:
                muon_target = model
                aux_params = []
                named = []  # only used for the diagnostic print below

            muon_cfg = MuonConfig(
                unified_lr=False,
                lr_muon=cfg.lr,
                lr_adam=cfg.muon_aux_lr,
                muon_beta=0.95,
                muon_decay=cfg.weight_decay,
                adam_betas=(0.9, 0.95),
                adam_decay=cfg.weight_decay,
                adam_eps=1e-10,
                enable_clipping=False,    # MLA-incompatible by default; revisit with custom hook later
                log_max_logits=False,
                log_dir="./logs",         # truthy → MuonClip skips SummaryWriter (its bug, but works for us)
            )
            if cfg.guarded_muonclip:
                _GuardedCls = _make_guarded_muonclip_class()
                optimizer = _GuardedCls(
                    muon_target, text_cfg, muon_cfg,
                    max_muon_ratio=cfg.guard_max_muon_ratio,
                    max_adam_ratio=cfg.guard_max_adam_ratio,
                )
                _opt_label = "GuardedMuonClip"
            else:
                optimizer = MuonClip(muon_target, text_cfg, muon_cfg)
                _opt_label = "MuonClip"
            # MuonClip's flush_metrics has a latent AttributeError: it references
            # self.writer unconditionally but only creates it when log_dir is
            # falsy (their `if not log_dir` is backwards). With our log_dir set
            # the writer is never created, then crashes on first optimizer.step().
            # Monkey-patch to a no-op since we don't use TB/wandb here anyway.
            # (GuardedMuonClip's overridden step doesn't call flush_metrics, but
            # base MuonClip's does — keep the patch in both branches for safety.)
            optimizer.flush_metrics = (lambda *a, **kw: None).__get__(optimizer, type(optimizer))
            n_muon = sum(p.numel() for g in optimizer.param_groups if g.get("use_muon")
                         for p in g["params"])
            n_adam = sum(p.numel() for g in optimizer.param_groups if not g.get("use_muon")
                         for p in g["params"])
            print(f"  {_opt_label} 2D (Muon-managed): {n_muon/1e6:.1f}M @ lr_muon={cfg.lr:.1e} momentum=0.95")
            print(f"  {_opt_label} non-2D (Adam-managed): {n_adam/1e6:.1f}M @ lr_adam={cfg.muon_aux_lr:.1e}")
            if cfg.guarded_muonclip:
                print(f"  guard caps: muon RMS-ratio<={cfg.guard_max_muon_ratio:.0e}, "
                      f"adam RMS-ratio<={cfg.guard_max_adam_ratio:.0e}")
            if use_prodigy:
                _excluded = [n for n, p in named
                             if p.ndim == 2 and ("embed_tokens" in n or "lm_head" in n)]
                print(f"  excluded from Muon (routed to Prodigy): {_excluded}")
            print(f"  enable_clipping=False (MLA-incompatible default; qk_norm + output_gate handle stability)")
            if use_prodigy and aux_params:
                from prodigyopt import Prodigy
                optimizer_aux = Prodigy(
                    aux_params,
                    lr=cfg.prodigy_lr_mult,
                    betas=(0.9, 0.95),
                    weight_decay=cfg.weight_decay,
                    decouple=bool(cfg.prodigy_decouple),
                    safeguard_warmup=bool(cfg.prodigy_safeguard_warmup),
                    d0=cfg.prodigy_d0,
                )
                for g in optimizer_aux.param_groups:
                    g.setdefault("peak_lr", g.get("lr", cfg.prodigy_lr_mult))
                print(f"  Prodigy aux (non-matrix + embed + lm_head): "
                      f"{sum(p.numel() for p in aux_params)/1e6:.1f}M  "
                      f"d0={cfg.prodigy_d0:.0e}  decouple={bool(cfg.prodigy_decouple)}  "
                      f"safeguard_warmup={bool(cfg.prodigy_safeguard_warmup)}")
        elif cfg.optimizer == "muon":
            # Muon on 2D matrix weights + Adam on everything else (norms, biases,
            # 1D). SingleDeviceMuonWithAuxAdam owns both updates in one
            # optimizer; param_groups are tagged with use_muon True/False.
            # NOTE: muon-optimizer pkg is overwritten by muon-clip in our env.
            # This path is kept for legacy-config repro; new runs use "muonclip".
            from muon import SingleDeviceMuonWithAuxAdam
            muon_params = [p for p in trainable if p.ndim >= 2]
            aux_params = [p for p in trainable if p.ndim < 2]
            param_groups = [
                {"params": muon_params, "lr": cfg.lr, "momentum": 0.95,
                 "weight_decay": cfg.weight_decay, "use_muon": True},
                {"params": aux_params, "lr": cfg.muon_aux_lr, "betas": (0.9, 0.95),
                 "eps": 1e-10, "weight_decay": cfg.weight_decay, "use_muon": False},
            ]
            optimizer = SingleDeviceMuonWithAuxAdam(param_groups)
            print(f"  muon (2D matrix weights): {sum(p.numel() for p in muon_params)/1e6:.1f}M "
                  f"@ lr={cfg.lr:.1e} momentum=0.95")
            print(f"  aux-adam (norms/1D): {sum(p.numel() for p in aux_params)/1e6:.1f}M "
                  f"@ lr={cfg.muon_aux_lr:.1e}")
        else:
            optimizer = bnb.optim.AdamW8bit(trainable, lr=cfg.lr, betas=(0.9, 0.95),
                                             weight_decay=cfg.weight_decay)

    # Capture each param group's "peak" lr right after construction so the LR
    # scheduler can scale per-group multiplicatively (preserves muon-vs-adam
    # ratios when ROP triggers, etc.).
    for g in optimizer.param_groups:
        g.setdefault("peak_lr", g.get("lr", cfg.lr))

    # Resume from a prior ckpt.pt if requested. Loads the per-layer MLA state
    # dicts on top of the patch-init, then the optimizer state, then starts the
    # step counter where we left off so the LR cosine schedule continues cleanly.
    start_step = 0
    if cfg.resume:
        print(f"resuming from: {cfg.resume}")
        ckpt = safe_torch_load(cfg.resume, map_location="cpu")
        saved_mla = ckpt["mla_state_dicts"]
        if cfg.mla_from_patch:
            print(f"mla_from_patch=1: IGNORING ckpt's mla_state_dicts, keeping "
                  f"patch-init MLA weights (use when you rebuilt the patch and "
                  f"want to retrain MLA from the new init)")
        loaded_mla, skipped_mla = 0, []
        for m in mla_modules:
            key = getattr(m, "_save_key", None)
            if key is None:
                skipped_mla.append(f"<no _save_key attr on {type(m).__name__}>")
                continue
            if cfg.mla_from_patch:
                skipped_mla.append(f"{key} (mla_from_patch)")
                continue
            if key not in saved_mla:
                skipped_mla.append(key)
                continue
            sd = {k: v.to(device=next(m.parameters()).device, dtype=next(m.parameters()).dtype)
                  for k, v in saved_mla[key].items()}
            missing, unexpected = m.load_state_dict(sd, strict=False)
            if unexpected:
                raise RuntimeError(f"resume: unexpected keys in {key}: {unexpected}")
            loaded_mla += 1
        print(f"resume: loaded {loaded_mla}/{len(mla_modules)} MLA modules "
              f"({'all' if not skipped_mla else 'missing keys: ' + ','.join(skipped_mla)})")
        if skipped_mla and not cfg.mla_from_patch:
            print(f"[warn] skipped MLA modules will use patch-init weights "
                  f"(SVD from convert.py), not trained state")
        # When adding new RWKV-8 layers alongside an already-trained one,
        # paper-init starts those new layers from scratch — which produces
        # catastrophic cold-start ppl when many layers swap at once. With
        # rwkv8_inherit_from_loaded=1, copy the state_dict of an
        # already-trained RWKV-8 module into each newly-installed module
        # that DID NOT load state from the ckpt. Donor selection is
        # nearest-neighbor: each fresh layer inherits from the loaded layer
        # with the smallest |layer_idx_diff|, so sequential conversions stay
        # close to their freshly-trained predecessor (e.g. L28 inherits
        # from L29, not L30).
        if cfg.rwkv8_inherit_from_loaded:
            def _layer_idx_of(m):
                key = str(getattr(m, "_save_key", ""))
                if not key.startswith("rwkv8_layer_"):
                    return None
                try:
                    return int(key.rsplit("_", 1)[-1])
                except ValueError:
                    return None
            loaded_rwkv = [
                (_layer_idx_of(m), m) for m in mla_modules
                if _layer_idx_of(m) is not None
                and getattr(m, "_save_key", None) in saved_mla
            ]
            fresh_rwkv = [
                (_layer_idx_of(m), m) for m in mla_modules
                if _layer_idx_of(m) is not None
                and getattr(m, "_save_key", None) not in saved_mla
            ]
            if not loaded_rwkv:
                raise ValueError(
                    "rwkv8_inherit_from_loaded=1 but the resume ckpt has no "
                    "trained RWKV-8 module to copy from. Set =0 or resume from "
                    "a ckpt with at least one rwkv8_layer_* state."
                )
            if fresh_rwkv:
                # Cache donor state_dicts on CPU once so multiple fresh layers
                # picking the same donor don't redo the clone.
                donor_sd_cache: dict[int, dict[str, torch.Tensor]] = {}
                broadcast_pairs: list[tuple[str, str]] = []
                for fresh_idx, fresh_m in fresh_rwkv:
                    donor_idx, donor_m = min(
                        loaded_rwkv, key=lambda lr: abs(lr[0] - fresh_idx)
                    )
                    if donor_idx not in donor_sd_cache:
                        donor_sd_cache[donor_idx] = {
                            k: v.detach().clone()
                            for k, v in donor_m.state_dict().items()
                        }
                    target_dev = next(fresh_m.parameters()).device
                    target_dt = next(fresh_m.parameters()).dtype
                    fresh_m.load_state_dict(
                        {k: v.to(device=target_dev, dtype=target_dt)
                         for k, v in donor_sd_cache[donor_idx].items()},
                        strict=True,
                    )
                    broadcast_pairs.append((
                        f"rwkv8_layer_{donor_idx}", f"rwkv8_layer_{fresh_idx}"
                    ))
                print(f"rwkv8_inherit_from_loaded: nearest-neighbor broadcast "
                      f"({len(broadcast_pairs)} pairs):")
                for donor_key, fresh_key in broadcast_pairs:
                    print(f"  {donor_key} -> {fresh_key}")
        # Phase 2 auxiliary heads — load if present and enabled this run.
        if "mutor_state_dict" in ckpt and getattr(model, "mutor_head", None) is not None:
            try:
                model.mutor_head.load_state_dict(ckpt["mutor_state_dict"])
                print("mutor_head state restored from checkpoint")
            except RuntimeError as e:
                print(f"mutor_head load failed, using fresh init: {e}")
        if "fsp_state_dict" in ckpt and getattr(model, "fsp_head", None) is not None:
            try:
                model.fsp_head.load_state_dict(ckpt["fsp_state_dict"], strict=False)
                print("fsp_head state restored from checkpoint")
            except RuntimeError as e:
                print(f"fsp_head load failed, using fresh init: {e}")
        if "parallel_heads_state_dict" in ckpt and getattr(model, "parallel_heads", None) is not None:
            # Only load if saved horizons match current config — otherwise
            # the architectures differ and state_dict shapes won't align.
            saved_horizons = ckpt.get("parallel_horizons", [])
            if list(saved_horizons) == list(model.parallel_heads.horizons):
                try:
                    model.parallel_heads.load_state_dict(ckpt["parallel_heads_state_dict"])
                    print("parallel_heads state restored from checkpoint")
                except RuntimeError as e:
                    print(f"parallel_heads load failed, using fresh init: {e}")
            else:
                print(f"parallel_heads horizons changed "
                      f"({saved_horizons} -> {model.parallel_heads.horizons}), using fresh init")
        # Restore MTP's non-attn state (fc, norms, MLP, etc.) if present in ckpt
        if "mtp_extra_state_dict" in ckpt and getattr(model, "mtp_trainer", None) is not None:
            mtp = model.mtp_trainer
            named_params = dict(mtp.named_parameters())
            loaded, skipped = 0, 0
            for name, v in ckpt["mtp_extra_state_dict"].items():
                p = named_params.get(name)
                if p is None:
                    skipped += 1
                    continue
                with torch.no_grad():
                    p.data.copy_(v.to(device=p.device, dtype=p.dtype))
                loaded += 1
            print(f"mtp_extra_state_dict: loaded {loaded} tensors, skipped {skipped}")
        # Engram state (per-layer). Keyed by "layer_<idx>"; each value is the
        # EngramModule's state_dict. Tensors are cast to each param's current
        # dtype/device (host-offloaded embedding tables stay on CPU).
        if "engram_state_dicts" in ckpt and engram_modules:
            saved_eng = ckpt["engram_state_dicts"]
            restored = 0
            for m in engram_modules:
                key = f"layer_{m.layer_id}"
                if key not in saved_eng:
                    print(f"engram {key}: no state in ckpt; keeping patch init")
                    continue
                em_params = dict(m.named_parameters())
                em_buffers = dict(m.named_buffers())
                sd = {}
                for k, v in saved_eng[key].items():
                    target = em_params.get(k)
                    if target is None:
                        target = em_buffers.get(k)
                    if target is None:
                        sd[k] = v  # unknown key — let load_state_dict report
                    else:
                        sd[k] = v.to(device=target.device, dtype=target.dtype)
                m.load_state_dict(sd, strict=False)
                restored += 1
            print(f"engram state restored for {restored}/{len(engram_modules)} modules")
        # Backbone state — saved only when the ckpt was produced under
        # freeze_non_mla=0 (full unfreeze). Without this, full-unfreeze runs
        # silently revert backbone to base safetensors on resume.
        if "backbone_state_dict" in ckpt:
            sd = ckpt["backbone_state_dict"]
            named = dict(model.named_parameters())
            loaded, skipped = 0, 0
            for name, v in sd.items():
                p = named.get(name)
                if p is None:
                    skipped += 1
                    continue
                with torch.no_grad():
                    p.data.copy_(v.to(device=p.device, dtype=p.dtype))
                loaded += 1
            n_params = sum(t.numel() for t in sd.values())
            print(f"backbone_state_dict: restored {loaded} tensors "
                  f"({n_params/1e9:.2f}B params), skipped {skipped}")
        # Optimizer state may not be compatible when switching training mode
        # (e.g. backbone-MLA-only ckpt -> MTP-only resume has different param
        # counts). Try to load; on mismatch, start fresh.
        # Also skip when mla_from_patch=1: the old optimizer stats (m, v) are
        # attuned to the OLD MLA weights; applying them to fresh BKV-init
        # weights would mis-step early training. Fresh optimizer is safer.
        if cfg.mla_from_patch:
            print("optimizer state skipped (mla_from_patch=1: old Adam m/v "
                  "stats are for the prior MLA weights, not the fresh patch)")
        else:
            try:
                optimizer.load_state_dict(ckpt["optimizer_state"])
                print("optimizer state restored from checkpoint")
            except (ValueError, KeyError, RuntimeError) as e:
                print(f"optimizer state skipped (incompatible with current training "
                      f"mode: {e}). Starting with fresh optimizer state.")
        if "optimizer_state_host" in ckpt and optimizer_host is not None:
            try:
                optimizer_host.load_state_dict(ckpt["optimizer_state_host"])
                print("optimizer_host (SparseAdam) state restored from checkpoint")
            except (ValueError, KeyError, RuntimeError) as e:
                print(f"optimizer_host state skipped: {e}")
        if "optimizer_state_aux" in ckpt and optimizer_aux is not None:
            try:
                optimizer_aux.load_state_dict(ckpt["optimizer_state_aux"])
                print("optimizer_aux (Prodigy) state restored from checkpoint")
            except (ValueError, KeyError, RuntimeError) as e:
                print(f"optimizer_aux state skipped: {e}")
        start_step = int(ckpt["step"])
        print(f"resumed at step {start_step}")
        del ckpt

    arr, train_end, eval_end = open_tokens(cfg)
    eval_start = train_end
    print(f"data: {train_end/1e9:.2f}B train tokens, "
          f"{(eval_end-train_end)/1e6:.1f}M eval tokens")

    # Reduce-on-Plateau LR controller (no-op when --rop-enable=0).
    plateau_ctrl = _PlateauLRController(cfg)
    if plateau_ctrl.enabled:
        print(f"ROP: patience={plateau_ctrl.patience} factor={plateau_ctrl.factor} "
              f"rel_threshold={plateau_ctrl.rel_threshold} min_mult={plateau_ctrl.min_mult} "
              f"warmup_grace={plateau_ctrl.warmup_grace}")

    # Helper: run eval, optionally including horizon-2/3/4 metrics when MTP is installed.
    def do_eval(step_now: int, prefix: str = "") -> None:
        t0 = time.time()
        if mtp_installed:
            horizons = (1, 2, 3, 4)
            results = multi_horizon_eval(model, arr, eval_start, eval_end, cfg, device,
                                         horizons=horizons)
            # Print compact summary + log
            lines = [f"  {prefix}step {step_now}  ({time.time()-t0:.0f}s)"]
            for k in horizons:
                if k in results:
                    r = results[k]
                    lines.append(
                        f"    h={k}  ppl={r['ppl']:7.3f}  "
                        f"top1={r['top1_acc']*100:5.1f}%  top5={r['top5_acc']*100:5.1f}%"
                    )
            print("\n".join(lines))
            row = {"step": step_now, "kind": "eval",
                   "loss": results[1]["loss"], "ppl": results[1]["ppl"],
                   "top1_acc": results[1]["top1_acc"], "top5_acc": results[1]["top5_acc"]}
            for k in horizons:
                if k in results:
                    row[f"h{k}_loss"] = results[k]["loss"]
                    row[f"h{k}_ppl"] = results[k]["ppl"]
                    row[f"h{k}_top1"] = results[k]["top1_acc"]
                    row[f"h{k}_top5"] = results[k]["top5_acc"]
            h1_for_rop = results[1]["ppl"]
        else:
            el, ep = eval_loss(model, arr, eval_start, eval_end, cfg, device)
            print(f"  {prefix}step {step_now} | eval_loss={el:.4f}  ppl={ep:.2f}  ({time.time()-t0:.0f}s)")
            row = {"step": step_now, "kind": "eval", "loss": el, "ppl": ep}
            h1_for_rop = ep
        # Feed the eval h1_ppl into the plateau controller. Its current mult
        # is read by the LR scheduling block below and applied multiplicatively
        # to all param-group lrs.
        if plateau_ctrl.enabled:
            mult, dropped = plateau_ctrl.step(h1_for_rop, step_now)
            row["rop_mult"] = mult
            if dropped:
                print(f"  [ROP] eval plateau hit; dropping lr mult to {mult:.4f} "
                      f"(drop #{plateau_ctrl.n_drops})")
                row["rop_dropped"] = True
        log(row)

    # Baseline eval (at whatever step we resumed from) before continuing training
    print("baseline eval...")
    do_eval(start_step)

    # Install signal handlers for graceful save-and-exit. The first Ctrl-C /
    # TERM will finish the current accum step, save, and exit cleanly; the
    # second forces exit if the save itself hangs.
    signal.signal(signal.SIGINT, _sig_handler)
    signal.signal(signal.SIGTERM, _sig_handler)
    signal.signal(signal.SIGUSR1, _sigusr1_handler)  # trainboard: checkpoint-now

    # DataLoader prefetch: overlap memmap reads with GPU compute. 1 worker is
    # enough at micro_batch_size=1 because each sample is only ~16 KB; the
    # bottleneck was page-fault latency on random-access reads, not throughput.
    # Python 3.14's default forkserver has a handshake bug with torch DataLoader,
    # so we force fork. The worker doesn't touch CUDA, so fork is safe.
    _train_loader = torch.utils.data.DataLoader(
        _TrainWindowIter(cfg.tokens_bin, 0, train_end, cfg.seq_len, seed=cfg.seed),
        batch_size=cfg.micro_batch_size,
        num_workers=1,
        pin_memory=True,
        prefetch_factor=4,
        persistent_workers=True,
        multiprocessing_context="fork",
    )
    _train_iter = iter(_train_loader)

    # Grad clipping target set is stable after setup/resume. Build it once so
    # host-offloaded Engram runs do not rediscover host params every step.
    if optimizer_host is not None:
        from engram_integration import collect_host_embedding_params as _chp
        _host_ids = {id(p) for p in _chp(engram_modules)}
        clip_targets = [p for p in trainable if id(p) not in _host_ids]
    else:
        clip_targets = trainable

    # Training loop
    model.train()
    step = start_step
    running_loss_t = None
    running_tokens = 0
    last_gnorm_t = None
    t_win = time.time()
    last_save_time = time.time()
    if cfg.save_every_seconds > 0:
        print(f"time-based checkpoint: every {cfg.save_every_seconds}s")
    if cfg.max_saved_checkpoints > 0:
        print(f"checkpoint retention: keep last {cfg.max_saved_checkpoints}")
    while step < cfg.max_steps:
        optimizer.zero_grad(set_to_none=True)
        if optimizer_host is not None:
            optimizer_host.zero_grad(set_to_none=True)
        if optimizer_aux is not None:
            optimizer_aux.zero_grad(set_to_none=True)
        accum_loss_t = None
        for _ in range(cfg.grad_accum_steps):
            ids = next(_train_iter).to(device, non_blocking=True)
            x, y = ids[:, :-1], ids[:, 1:]
            if cfg.train_aux_only:
                # Aux-only: backbone forward (no_grad), compute aux losses only.
                # No chain MTP — MTP is frozen, computing through it wastes
                # compute since there's no gradient to absorb.
                text_model = _text_model(model)
                with torch.no_grad():
                    text_out = text_model(input_ids=x, use_cache=False)
                    backbone_hidden = _last_hidden(text_out).detach()
                    del text_out

                loss = backbone_hidden.new_zeros((), dtype=torch.float32)
                if cfg.mutor_enabled:
                    mut_l, n_mut = mutor_loss(
                        backbone_hidden, y, model.lm_head, model.mutor_head,
                        num_registers=cfg.mutor_num_registers,
                        d_min=cfg.mutor_d_min, d_max=cfg.mutor_d_max,
                    )
                    if n_mut > 0:
                        loss = loss + cfg.mutor_weight * mut_l
                if cfg.fsp_enabled:
                    fsp_l, n_fsp = fsp_loss(
                        backbone_hidden, y, model.lm_head, model.fsp_head,
                        num_positions=cfg.fsp_num_positions, tau=cfg.fsp_tau,
                    )
                    if n_fsp > 0:
                        loss = loss + cfg.fsp_weight * fsp_l
                if cfg.parallel_enabled:
                    ph_l, n_ph = parallel_heads_loss(
                        backbone_hidden, y, model.lm_head, model.parallel_heads,
                        weights=_ph_weights_cache,
                    )
                    if n_ph > 0:
                        loss = loss + ph_l
                logits = loss  # placeholder for downstream del
                del backbone_hidden
            elif cfg.train_mtp_only:
                text_model = _text_model(model)
                with torch.no_grad():
                    text_out = text_model(input_ids=x, use_cache=False)
                    backbone_hidden = _last_hidden(text_out).detach()
                    del text_out

                T = x.shape[1]
                # Precompute full-range RoPE once. Each horizon slices into this
                # instead of making len(_chain_horizons) separate rotary_emb calls.
                full_pos = train_pos_base[:, :T].expand(x.shape[0], -1)
                cos_all, sin_all = text_model.rotary_emb(backbone_hidden, full_pos)

                losses_by_horizon: list[torch.Tensor] = []
                prev_out = backbone_hidden            # [B, T, H], k=0 init
                for k, (horizon, weight) in enumerate(zip(_chain_horizons, _chain_weights)):
                    # At chain step k we predict horizon = k + 2.
                    # Current sequence length: L_k = T - k. Input is y[:, k:T].
                    L_k = T - k
                    hidden_k = prev_out[:, :L_k]          # [B, L_k, H]
                    ids_k = y[:, k:k + L_k]               # [B, L_k], tokens at abs pos k+1..T
                    # Absolute-position RoPE (tokens live at abs positions k..T-1)
                    pos_k = full_pos[:, k:k + L_k]
                    cos_k = cos_all[:, k:k + L_k]
                    sin_k = sin_all[:, k:k + L_k]

                    mtp_out_k = model.mtp_trainer(
                        input_ids=ids_k,
                        hidden_states=hidden_k,
                        position_embeddings=(cos_k, sin_k),
                        position_ids=pos_k,
                    )
                    # Valid prediction range: first L_k - 1 positions (last predicts
                    # beyond label range). Labels for this horizon: y[k+1 : T].
                    valid_logits_hidden = mtp_out_k[:, :L_k - 1]
                    valid_labels = y[:, k + 1:]  # length T - k - 1 = L_k - 1
                    loss_h = chunked_lmhead_ce(valid_logits_hidden, model.lm_head, valid_labels)
                    losses_by_horizon.append(weight * loss_h)

                    prev_out = mtp_out_k
                    del hidden_k, ids_k, pos_k, cos_k, sin_k

                loss = sum(losses_by_horizon)

                # Aux losses against the frozen backbone_hidden. Gradient flows
                # only to the aux heads' own trainable params (offset_emb,
                # fsp_bias, parallel MLPs) because backbone_hidden is detached
                # (produced under torch.no_grad above). This is the clean way
                # to warm up aux heads before unfreezing the backbone.
                if cfg.mutor_enabled:
                    mut_l, n_mut = mutor_loss(
                        backbone_hidden, y, model.lm_head, model.mutor_head,
                        num_registers=cfg.mutor_num_registers,
                        d_min=cfg.mutor_d_min, d_max=cfg.mutor_d_max,
                    )
                    if n_mut > 0:
                        loss = loss + cfg.mutor_weight * mut_l
                if cfg.fsp_enabled:
                    fsp_l, n_fsp = fsp_loss(
                        backbone_hidden, y, model.lm_head, model.fsp_head,
                        num_positions=cfg.fsp_num_positions, tau=cfg.fsp_tau,
                    )
                    if n_fsp > 0:
                        loss = loss + cfg.fsp_weight * fsp_l
                if cfg.parallel_enabled:
                    ph_l, n_ph = parallel_heads_loss(
                        backbone_hidden, y, model.lm_head, model.parallel_heads,
                        weights=_ph_weights_cache,
                    )
                    if n_ph > 0:
                        loss = loss + ph_l

                logits = loss  # placeholder so downstream `del logits` doesn't error
                del backbone_hidden, prev_out, losses_by_horizon
            elif mtp_installed:
                text_model = _text_model(model)
                outputs = text_model(input_ids=x, use_cache=False)
                backbone_hidden = _last_hidden(outputs)
                lm_loss = chunked_lmhead_ce(backbone_hidden, model.lm_head, y)

                T = x.shape[1]
                # Precompute full-range RoPE once; each horizon slices.
                full_pos = train_pos_base[:, :T].expand(x.shape[0], -1)
                cos_all, sin_all = text_model.rotary_emb(backbone_hidden, full_pos)

                # Chained MTP loss over _chain_horizons (h=2,3,4 by default).
                # Same cascade as train_mtp_only branch, but backbone_hidden has
                # requires_grad here, so the full chain's gradient flows into it.
                chain_losses: list[torch.Tensor] = []
                prev_out = backbone_hidden
                for k, (horizon, weight) in enumerate(zip(_chain_horizons, _chain_weights)):
                    L_k = T - k
                    hidden_k = prev_out[:, :L_k]
                    ids_k = y[:, k:k + L_k]
                    pos_k = full_pos[:, k:k + L_k]
                    cos_k = cos_all[:, k:k + L_k]
                    sin_k = sin_all[:, k:k + L_k]
                    mtp_out_k = model.mtp_trainer(
                        input_ids=ids_k,
                        hidden_states=hidden_k,
                        position_embeddings=(cos_k, sin_k),
                        position_ids=pos_k,
                    )
                    valid_logits_hidden = mtp_out_k[:, :L_k - 1]
                    valid_labels = y[:, k + 1:]
                    loss_h = chunked_lmhead_ce(valid_logits_hidden, model.lm_head, valid_labels)
                    chain_losses.append(weight * loss_h)
                    prev_out = mtp_out_k
                    del hidden_k, ids_k, pos_k, cos_k, sin_k

                mtp_loss = sum(chain_losses)
                loss = lm_loss + cfg.mtp_loss_weight * mtp_loss

                # MuToR aux loss: multi-horizon register-style prediction at
                # randomly-sampled positions in backbone_hidden. In combined
                # mode (here), gradient flows into both offset_emb AND backbone.
                if cfg.mutor_enabled:
                    mut_l, n_mut = mutor_loss(
                        backbone_hidden, y, model.lm_head, model.mutor_head,
                        num_registers=cfg.mutor_num_registers,
                        d_min=cfg.mutor_d_min, d_max=cfg.mutor_d_max,
                    )
                    if n_mut > 0:
                        loss = loss + cfg.mutor_weight * mut_l
                # FSP-BCE aux loss: future-bag multi-hot BCE at sampled positions.
                if cfg.fsp_enabled:
                    fsp_l, n_fsp = fsp_loss(
                        backbone_hidden, y, model.lm_head, model.fsp_head,
                        num_positions=cfg.fsp_num_positions, tau=cfg.fsp_tau,
                    )
                    if n_fsp > 0:
                        loss = loss + cfg.fsp_weight * fsp_l
                # Parallel heads (Gloeckle) — each head's weight is already
                # applied internally, so we add the summed parallel loss with
                # unit coefficient.
                if cfg.parallel_enabled:
                    ph_l, n_ph = parallel_heads_loss(
                        backbone_hidden, y, model.lm_head, model.parallel_heads,
                        weights=_ph_weights_cache,
                    )
                    if n_ph > 0:
                        loss = loss + ph_l
                logits = loss  # placeholder for downstream del
                del outputs, backbone_hidden, prev_out, chain_losses
            else:
                if cfg.mutor_enabled or cfg.fsp_enabled or cfg.parallel_enabled:
                    text_model = _text_model(model)
                    outputs = text_model(input_ids=x, use_cache=False)
                    backbone_hidden = _last_hidden(outputs)
                    loss = chunked_lmhead_ce(backbone_hidden, model.lm_head, y)
                    if cfg.mutor_enabled:
                        mut_l, n_mut = mutor_loss(
                            backbone_hidden, y, model.lm_head, model.mutor_head,
                            num_registers=cfg.mutor_num_registers,
                            d_min=cfg.mutor_d_min, d_max=cfg.mutor_d_max,
                        )
                        if n_mut > 0:
                            loss = loss + cfg.mutor_weight * mut_l
                    if cfg.fsp_enabled:
                        fsp_l, n_fsp = fsp_loss(
                            backbone_hidden, y, model.lm_head, model.fsp_head,
                            num_positions=cfg.fsp_num_positions, tau=cfg.fsp_tau,
                        )
                        if n_fsp > 0:
                            loss = loss + cfg.fsp_weight * fsp_l
                    if cfg.parallel_enabled:
                        ph_l, n_ph = parallel_heads_loss(
                            backbone_hidden, y, model.lm_head, model.parallel_heads,
                            weights=_ph_weights_cache,
                        )
                        if n_ph > 0:
                            loss = loss + ph_l
                    logits = loss  # placeholder for downstream del
                    del outputs, backbone_hidden
                else:
                    text_model = _text_model(model)
                    outputs = text_model(input_ids=x, use_cache=False)
                    hidden = _last_hidden(outputs)
                    loss = chunked_lmhead_ce(hidden, model.lm_head, y)
                    logits = loss  # placeholder for downstream del
                    del outputs, hidden

            (loss / cfg.grad_accum_steps).backward()
            loss_for_log = loss.detach() / cfg.grad_accum_steps
            accum_loss_t = loss_for_log if accum_loss_t is None else accum_loss_t + loss_for_log
            running_tokens += y.numel()
            del logits, loss

        if accum_loss_t is not None:
            running_loss_t = (
                accum_loss_t if running_loss_t is None else running_loss_t + accum_loss_t
            )

        # LR schedule. `lr` is computed relative to cfg.lr (the main-group peak).
        # We derive a schedule fraction and apply it per-group relative to each
        # group's stored peak, then multiply by the ROP plateau-multiplier (if
        # enabled). The plateau mult is set inside do_eval() and persists between
        # evals; it can only reduce, never raise.
        lr = lr_at(step, cfg, start_step=start_step)
        sched_frac = lr / cfg.lr if cfg.lr > 0 else 1.0
        rop_mult = plateau_ctrl.mult if plateau_ctrl is not None else 1.0
        if engram_modules and len(optimizer.param_groups) >= 2:
            optimizer.param_groups[0]["lr"] = lr * rop_mult
            optimizer.param_groups[1]["lr"] = lr * cfg.engram_lr_mult * rop_mult
        elif cfg.optimizer in ("muon", "muonclip"):
            # Each group's stored "peak_lr" was captured at construction; scale
            # all groups by sched_frac × rop_mult, preserving cross-group ratios.
            for g in optimizer.param_groups:
                g["lr"] = g["peak_lr"] * sched_frac * rop_mult
        else:
            for g in optimizer.param_groups:
                g["lr"] = lr * rop_mult
        if optimizer_host is not None:
            for g in optimizer_host.param_groups:
                g["lr"] = lr * cfg.engram_lr_mult * rop_mult
        if optimizer_aux is not None:
            # Prodigy's `lr` is a multiplier; the actual step is d_t * lr.
            # We still apply the warmup/schedule fraction so Prodigy's effective
            # step ramps up alongside Muon's during the resume rewarmup.
            for g in optimizer_aux.param_groups:
                g["lr"] = g["peak_lr"] * sched_frac * rop_mult
        # Grad clip GPU params only. Sparse CPU grads can't be clip-normed
        # jointly (mixed-device + sparse). Set --grad-clip <= 0 or
        # --grad-clip-every=0 to skip the reduction entirely.
        clip_due = (
            cfg.grad_clip > 0
            and cfg.grad_clip_every > 0
            and step % cfg.grad_clip_every == 0
        )
        if clip_due:
            last_gnorm_t = torch.nn.utils.clip_grad_norm_(clip_targets, cfg.grad_clip).detach()
        optimizer.step()
        if optimizer_host is not None:
            optimizer_host.step()
        if optimizer_aux is not None:
            optimizer_aux.step()
        step += 1

        if step % cfg.log_every == 0:
            dt = time.time() - t_win
            tps = running_tokens / dt
            avg = (
                float((running_loss_t / cfg.log_every).item())
                if running_loss_t is not None else 0.0
            )
            gnorm = float(last_gnorm_t.item()) if last_gnorm_t is not None else None
            gnorm_s = f"{gnorm:.2f}" if gnorm is not None else "n/a"
            # Guard saturation diagnostic (only present on GuardedMuonClip).
            guard_s = ""
            guard_extra = {}
            if hasattr(optimizer, "last_guard_saturation"):
                gs = optimizer.last_guard_saturation
                guard_s = (f"  guard_sat=m{gs['muon_sat']}/{gs['muon_total']}"
                           f"|a{gs['adam_sat']}/{gs['adam_total']}")
                guard_extra = {f"guard_{k}": v for k, v in gs.items()}
            # Prodigy's effective step size = d (auto-adapted) — log if available.
            if optimizer_aux is not None and len(optimizer_aux.param_groups) > 0:
                d_est = optimizer_aux.param_groups[0].get("d", None)
                if d_est is not None:
                    guard_s += f"  d_prodigy={d_est:.2e}"
                    guard_extra["d_prodigy"] = float(d_est)
            print(f"  step {step:5d} | loss={avg:.4f}  lr={lr:.2e}  "
                  f"gnorm={gnorm_s}  tok/s={tps:.0f}{guard_s}")
            log({"step": step, "kind": "train", "loss": avg, "lr": lr, **guard_extra,
                 "gnorm": gnorm, "tok_per_sec": tps})
            running_loss_t = None
            running_tokens = 0
            t_win = time.time()

        if step % cfg.eval_every == 0:
            do_eval(step)
            t_win = time.time()  # don't count eval time in tok/s

        # Step-based save
        step_save_due = cfg.save_every > 0 and step % cfg.save_every == 0
        # Time-based save (wall clock, independent of step count)
        time_save_due = (cfg.save_every_seconds > 0
                        and time.time() - last_save_time >= cfg.save_every_seconds)
        if step_save_due or time_save_due:
            reason = "step" if step_save_due else f"{int(time.time()-last_save_time)}s elapsed"
            print(f"  [ckpt] saving at step {step} ({reason})...")
            save_checkpoint(step, mla_modules, optimizer, cfg, model=model,
                            engram_modules=engram_modules, optimizer_host=optimizer_host,
                            optimizer_aux=optimizer_aux)
            _prune_old_checkpoints(Path(cfg.out_dir), cfg.max_saved_checkpoints)
            log({"step": step, "kind": "checkpoint"})
            last_save_time = time.time()
            t_win = time.time()

        # trainboard: SIGUSR1 checkpoint-without-exit ("checkpoint now"). Mirrors
        # the step-based save above but never exits; the loop continues training.
        if _ckpt_flag["count"] > 0:
            _ckpt_flag["count"] = 0
            print(f"  [checkpoint] SIGUSR1 save at step {step}...")
            save_checkpoint(step, mla_modules, optimizer, cfg, model=model,
                            engram_modules=engram_modules, optimizer_host=optimizer_host,
                            optimizer_aux=optimizer_aux)
            _prune_old_checkpoints(Path(cfg.out_dir), cfg.max_saved_checkpoints)
            log({"step": step, "kind": "checkpoint", "reason": "sigusr1"})
            last_save_time = time.time()
            t_win = time.time()

        # Signal-based save-and-exit (Ctrl-C / systemd stop)
        if _interrupt_flag["count"] > 0:
            print(f"[interrupt] saving at step {step} before exit...")
            save_checkpoint(step, mla_modules, optimizer, cfg, model=model,
                            engram_modules=engram_modules, optimizer_host=optimizer_host,
                            optimizer_aux=optimizer_aux)
            log({"step": step, "kind": "checkpoint", "reason": "interrupt"})
            log_f.close()
            print(f"[interrupt] saved -> {cfg.out_dir}/step_{step:06d}. bye.")
            sys.exit(0)

    # Final eval + save (never pruned — always the latest)
    do_eval(step, prefix="final ")
    save_checkpoint(step, mla_modules, optimizer, cfg, model=model,
                    engram_modules=engram_modules, optimizer_host=optimizer_host,
                    optimizer_aux=optimizer_aux)
    log_f.close()
    print(f"done. step {step} / {cfg.max_steps}  ckpt -> {cfg.out_dir}/step_{step:06d}")


if __name__ == "__main__":
    main()
