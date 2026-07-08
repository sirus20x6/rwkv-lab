#!/usr/bin/env python
"""Stage 0.7 / 2-3 — Supervised + Dynamical Memory Training for GDN->RWKV-7.

Pieces (plan glistening-inventing-garden.md):

  BilinearStateCodec  P : GDN state [B,32,128,128] -> RWKV-7 state [B,64,64,64].
      Structured per-head bilinear (A,B,M), ~18K params (a dense Linear would be
      ~137B). This is the GDN->RWKV state-space target map. P(S_gdn) is the SMT
      target; the student's own recurrent state is supervised toward it.

  rwkv_readout(layer,S,h)  faithful RWKV-7 readout that reuses the student's
      params, so the codec-grounding fit (reconstruct teacher block_out from
      P(S_gdn)) doubles as a principled init for the student's read-side params.

  smt_transition_loss / dmt_rollout_loss  one-step + closed-loop state
      supervision using the Stage-0 state-threaded RWKV8TimeMixDeltaNet.forward
      (initial_state / shift_state / return_state).

CODEC GROUNDING (the flagged risk): P is grounded by reconstructing the teacher's
block output through the student readout (an external, fixed target), which keeps
it from collapsing onto whatever the student state happens to be. P may be pre-fit
and frozen (fit_codec) or co-trained; we default to co-train with a low LR plus
the block-output anchor. If direct state MSE underfits, swap to a learned
predictive canonical-z target (documented fallback, not yet wired).

Reuses recurrent_stability_metrics from looped_rwkv_rosa_engram_v3.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .looped_rwkv_rosa_engram_v3 import recurrent_stability_metrics
except Exception:  # pragma: no cover
    recurrent_stability_metrics = None


# ---------------------------------------------------------------------------
class BilinearStateCodec(nn.Module):
    """GDN state [B,Hg,Dkg,Dvg] -> RWKV state [B,Hr,Dkr,Dvr] via S' = M (A S Bᵀ)."""

    def __init__(self, gdn_heads=32, gdn_dk=128, gdn_dv=128,
                 rwkv_heads=64, rwkv_dk=64, rwkv_dv=64):
        super().__init__()
        self.A = nn.Parameter(torch.empty(rwkv_dk, gdn_dk))
        self.B = nn.Parameter(torch.empty(rwkv_dv, gdn_dv))
        nn.init.orthogonal_(self.A)
        nn.init.orthogonal_(self.B)
        # head-mix init: map each rwkv head to a gdn head (2 rwkv heads per gdn head)
        M0 = torch.zeros(rwkv_heads, gdn_heads)
        for hr in range(rwkv_heads):
            M0[hr, min(hr * gdn_heads // rwkv_heads, gdn_heads - 1)] = 1.0
        self.M = nn.Parameter(M0)
        self.shape = (rwkv_heads, rwkv_dk, rwkv_dv)

    def forward(self, S: torch.Tensor) -> torch.Tensor:
        S = S.float()
        proj = torch.einsum("ik,bhkl,jl->bhij", self.A, S, self.B)  # [B,Hg,Dkr,Dvr]
        return torch.einsum("rh,bhij->brij", self.M, proj)          # [B,Hr,Dkr,Dvr]


def rwkv_readout(layer, S: torch.Tensor, h: torch.Tensor,
                 h_prev: torch.Tensor | None = None) -> torch.Tensor:
    """RWKV-7 read-side applied to an EXTERNAL state S [B,H,N,N] and input h [B,C].

    Mirrors RWKV8TimeMixDeltaNet.forward's read tail INCLUDING the token-shift mixing
    (xr/xk/xv/xg = h + (h_prev-h)*x_*) and the Comba r-=d*k correction, so the codec
    grounding matches the real rollout's readout. h_prev = the previous token's hidden
    (fit_codec passes cache.h[:, pos-1]); if None, falls back to the no-shift static
    probe (legacy). Reuses the layer's own params."""
    B, C = h.shape
    H, N = layer.num_heads, layer.head_size
    if h_prev is not None:
        xx = h_prev - h                                    # == forward's time_shift(x) - x
        xr = h + xx * layer.x_r.view(1, C)
        xk = h + xx * layer.x_k.view(1, C)
        xv = h + xx * layer.x_v.view(1, C)
        xg = h + xx * layer.x_g.view(1, C)
    else:
        xr = xk = xv = xg = h
    r = layer.receptance(xr).view(B, H, N)
    k = layer.key(xk).view(B, H, N)
    v = layer.value(xv).view(B, H, N)
    g = torch.sigmoid(xg @ layer.g1) @ layer.g2
    # Comba output-correction (out_correct_d init 0 => no-op), matches forward.
    r = r - layer.out_correct_d.repeat_interleave(N).view(1, H, N) * k
    out = torch.einsum("bhk,bhkv->bhv", r, S.to(r.dtype))           # read: sum_k S[k,v] r[k]
    out = layer.ln_x(out.reshape(B, C)).view(B, C)
    bonus = (r * k * layer.r_k.view(1, H, N)).sum(-1, keepdim=True) * v
    return layer.output((out + bonus.reshape(B, C)) * g)


# ---------------------------------------------------------------------------
# Calibration cache (build_memory_targets.py output)
# ---------------------------------------------------------------------------
class MemoryTargetCache:
    """Lazy reader over the memmaps produced by build_memory_targets.py."""

    def __init__(self, cache_dir: str):
        import numpy as np
        d = Path(cache_dir)
        self.m = json.loads((d / "manifest.json").read_text())
        sh = self.m["shapes"]
        self.h = np.memmap(d / self.m["files"]["h"], dtype="float16", mode="r", shape=tuple(sh["h"]))
        self.block_out = np.memmap(d / self.m["files"]["block_out"], dtype="float16", mode="r", shape=tuple(sh["block_out"]))
        self.state = np.memmap(d / self.m["files"]["state"], dtype="float16", mode="r", shape=tuple(sh["state"]))
        self.stride = self.m["state_stride"]
        self.T = self.m["seq_len"]
        self.n_windows = self.m["n_windows"]
        self.n_bounds = self.m["n_bounds"]

    def boundary_positions(self):
        """Token position read out by boundary state j (j>=1): min(j*stride,T)-1."""
        return [min(j * self.stride, self.T) - 1 for j in range(1, self.n_bounds)]


def fit_codec(layer, cache_dir: str, *, steps=400, lr=1e-3, weight_decay=1e-4,
              batch_size=512, device="cuda", train_readout=True, verbose=True):
    """Ground P (and optionally the student readout) by reconstructing the teacher
    block output from P(S_gdn) at boundary positions. Returns (codec, final_rel_rmse)
    where rel_rmse = sqrt(sum SE / sum Y^2) over the full calibration set (LOWER is
    better; healthy fits ~0.12-0.2, a collapse approaches ~1.0). The caller gates on it.

    Anchors P in an external target (block_out) so it can't collapse onto the
    student state. If train_readout, the student's receptance/key/value/g/ln_x/
    output/r_k are updated too -> principled read-side init (Stage 0+ functional).
    """
    import numpy as np
    cache = MemoryTargetCache(cache_dir)
    codec = BilinearStateCodec(
        gdn_heads=cache.m["num_v_heads"], gdn_dk=cache.m["head_k_dim"], gdn_dv=cache.m["head_v_dim"],
        rwkv_heads=layer.num_heads, rwkv_dk=layer.head_size, rwkv_dv=layer.head_size,
    ).to(device)

    pos = cache.boundary_positions()                      # token positions
    # Keep targets on CPU and minibatch to GPU each step: the state cache can be
    # ~9GB+ (e.g. 256 windows), too large to hold on GPU alongside the 9B model.
    S_cpu = torch.from_numpy(np.asarray(cache.state[:, 1:cache.n_bounds]))  # [W,nb1,Hg,Dk,Dv] f16
    H_cpu = torch.from_numpy(np.asarray(cache.h[:, pos]))                   # [W,nb1,C] f16
    # previous token's hidden at each boundary, for the readout token-shift mix
    # (all boundary positions are >= stride-1, so pos-1 is always in range).
    Hp_cpu = torch.from_numpy(np.asarray(cache.h[:, np.asarray(pos) - 1]))  # [W,nb1,C] f16
    Y_cpu = torch.from_numpy(np.asarray(cache.block_out[:, pos]))           # [W,nb1,C] f16
    W, nb1, C = H_cpu.shape
    S_cpu = S_cpu.reshape(W * nb1, *S_cpu.shape[2:])
    H_cpu = H_cpu.reshape(W * nb1, C)
    Hp_cpu = Hp_cpu.reshape(W * nb1, C)
    Y_cpu = Y_cpu.reshape(W * nb1, C)
    N = W * nb1
    bs = min(batch_size, N)

    # fit in float32 (cache is float16; model layer may be bf16)
    orig_dtype = next(layer.parameters()).dtype
    layer.to(device=device, dtype=torch.float32)

    params = list(codec.parameters())
    if train_readout:
        for p in [layer.receptance.weight, layer.key.weight, layer.value.weight,
                  layer.output.weight, layer.g1, layer.g2, layer.r_k, layer.ln_x.weight,
                  layer.ln_x.bias, layer.x_r, layer.x_k, layer.x_v, layer.x_g]:
            p.requires_grad_(True); params.append(p)
    opt = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=lr * 0.05)
    gen = torch.Generator().manual_seed(0)
    for it in range(steps):
        idx = torch.randint(0, N, (bs,), generator=gen)
        S = S_cpu[idx].to(device).float()
        H = H_cpu[idx].to(device).float()
        Hp = Hp_cpu[idx].to(device).float()
        Y = Y_cpu[idx].to(device).float()
        opt.zero_grad()
        pred = rwkv_readout(layer, codec(S), H, Hp)
        loss = F.mse_loss(pred, Y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        if verbose and (it % max(1, steps // 8) == 0 or it == steps - 1):
            rel = (loss.item() / Y.pow(2).mean().item()) ** 0.5
            print(f"  [fit_codec] step {it} mse={loss.item():.5f} rel_rmse={rel:.4f}", flush=True)
    # Final fit quality over the FULL calibration set (the per-minibatch rel above is
    # noisy) — this is the stable number the caller gates on.
    with torch.no_grad():
        tot_se = 0.0; tot_y2 = 0.0
        for i in range(0, N, bs):
            S = S_cpu[i:i + bs].to(device).float()
            H = H_cpu[i:i + bs].to(device).float()
            Hp = Hp_cpu[i:i + bs].to(device).float()
            Y = Y_cpu[i:i + bs].to(device).float()
            pred = rwkv_readout(layer, codec(S), H, Hp)
            tot_se += F.mse_loss(pred, Y, reduction="sum").item()
            tot_y2 += Y.pow(2).sum().item()
    final_rel = (tot_se / max(tot_y2, 1e-12)) ** 0.5
    if verbose:
        print(f"  [fit_codec] FINAL rel_rmse={final_rel:.4f} (over {N} samples)", flush=True)
    for p in codec.parameters():
        p.requires_grad_(False)
    layer.to(dtype=orig_dtype)  # restore model layer dtype for the main forward
    return codec, final_rel


# ---------------------------------------------------------------------------
# SMT / DMT losses (consumed by train_mla.py Stage 2-3)
# ---------------------------------------------------------------------------
def _chunks(T, stride):
    return [(lo, min(lo + stride, T)) for lo in range(0, T, stride)]


def _project_targets(codec, S_gdn, target_states=None):
    if target_states is not None:
        return target_states
    B = S_gdn.shape[0]
    return codec(S_gdn.reshape(-1, *S_gdn.shape[2:])).reshape(B, S_gdn.shape[1], *codec.shape)


def smt_transition_loss(layer, codec, h_seq, S_gdn=None, *, stride,
                        block_out=None, max_update_ratio=0.25, target_states=None):
    """One-step (per-chunk) supervised memory transition.

    For each chunk, start from the teacher state P(S_gdn[j]) and run the student
    one chunk; the predicted final state must match P(S_gdn[j+1]).

      h_seq:   [B,T,C]   layer inputs
      S_gdn:   [B,nb,Hg,Dk,Dv] teacher boundary states (nb = n_bounds)
    Independent transitions -> no BPTT through the whole prefix (SMT).
    """
    B, T, C = h_seq.shape
    chunks = _chunks(T, stride)
    tgt = _project_targets(codec, S_gdn, target_states)
    mem_loss = h_seq.new_zeros(())
    blk_loss = h_seq.new_zeros(())
    upen = h_seq.new_zeros(())
    n = 0
    n_upen = 0  # chunks with a non-zero prior state (exclude the zero-init chunk)

    by_len = {}
    for j, (lo, hi) in enumerate(chunks):
        by_len.setdefault(hi - lo, []).append((j, lo, hi))

    for clen, group in by_len.items():
        G = len(group)
        x = torch.cat([h_seq[:, lo:hi] for _j, lo, hi in group], dim=0)
        s0 = torch.cat([tgt[:, j] for j, _lo, _hi in group], dim=0).detach()
        s1_tgt = torch.cat([tgt[:, j + 1] for j, _lo, _hi in group], dim=0).detach()
        shifts = []
        for _j, lo, _hi in group:
            shifts.append(h_seq[:, lo - 1:lo] if lo > 0 else h_seq[:, :1].new_zeros(B, 1, C))
        shift = torch.cat(shifts, dim=0)
        # skip_refine: SMT supervises the CORE's state transition + within-chunk readout.
        # LoopedRWKV refinement passes are stateless re-reads of the window, so on a
        # 64-token chunk they compute a different function than the full-window block
        # loss trains — and only pass-1 states are supervised anyway. Pass-1 only:
        # consistent semantics, no n_loops x chunk cost. Bare cores ignore the kwarg.
        y, s1, _sh = layer(x, initial_state=s0, shift_state=shift, return_state=True,
                           skip_refine=True)
        mem_loss = mem_loss + F.mse_loss(s1.float(), s1_tgt) * G
        if block_out is not None:
            y_tgt = torch.cat([block_out[:, lo:hi] for _j, lo, hi in group], dim=0)
            blk_loss = blk_loss + F.mse_loss(y.float(), y_tgt.float()) * G

        # update ratio ||s1-s0||/||s0||, but the first chunk starts from the zero
        # initial state (||s0||~0) -> skip those to avoid a div-by-zero blow-up,
        # and don't let fully-masked chunks dilute the logged mean.
        prior_norm = s0.flatten(1).norm(dim=-1).view(G, B)
        valid = (prior_norm > 1e-3).float()
        if valid.sum() > 0:
            ur = ((s1.float() - s0).flatten(1).norm(dim=-1).view(G, B)
                  / prior_norm.clamp_min(1e-3))
            per_chunk_valid = valid.sum(dim=1)
            mask = per_chunk_valid > 0
            if mask.any():
                per_chunk = (F.relu(ur - max_update_ratio).pow(2) * valid).sum(dim=1) \
                    / per_chunk_valid.clamp_min(1.0)
                upen = upen + per_chunk[mask].sum()
                n_upen += int(mask.sum().item())
        n += G
    return {"smt_memory": mem_loss / n, "smt_block": blk_loss / n,
            "smt_update_pen": upen / max(n_upen, 1)}


class DMTGraphedRollout:
    """CUDA-graph replay for the DMT rollout's launch-bound chunk steps.

    The rollout is inherently sequential (state feedback), so each of the nb chunk
    steps launches ~dozens of tiny elementwise kernels + the wkv kernel + eager
    autograd for a 64-token slice — CPU launch overhead dominates GPU work. Each
    step's forward AND backward are captured once and replayed as single graph
    launches.

    HAND-ROLLED capture (not torch.cuda.make_graphed_callables). Root cause, torch
    2.11: each leaf's AccumulateGrad node is created ONCE (weakly cached on the
    tensor) and bound to the stream current at first use. The trainer's main
    forward/backward runs on the DEFAULT stream, so the layer's params carry
    default-bound accumulators; any backward capture that references them makes
    the engine add a legacy-stream dependency -> cudaErrorStreamCaptureImplicit
    -> capture invalidated (this is also why stock make_graphed_callables fails
    here). Fix: the captured graphs NEVER touch the real params. The runner owns
    STATIC PARAM COPIES, created and first-touched on the capture stream (their
    accumulators bind there), and the captured forward runs through
    torch.func.functional_call on those copies. Real param VALUES are synced into
    the copies once per iteration (chunk 0, one foreach D2D copy); the autograd
    Function takes the real params as inputs and returns the copies' captured
    grads for them, so .grad accumulation on the real params stays eager and
    exact. Warmup + both captures share ONE stream ("global" mode, bit-exact
    replay validated; "relaxed" fallback kept); replays launch from whatever
    stream is current at call time.

    One graph pair PER CHUNK INDEX: graphs own static input/output/workspace
    buffers, so one instance may only replay once per iteration, and the rollout
    calls the step nb times. Pairs are created lazily as the DMT curriculum grows
    nb (monotone), never re-captured or freed. Cost: static buffers per index
    (tens of MB each at B=8/stride=64) + one static copy of the layer params.

    Baked at capture: B, stride, C, dtypes, and shift as a TENSOR (zeros == the
    shift_state=None zero-pad semantics, so every index shares one signature).
    j=0 differs only in requires_grad (teacher state/zero shift are leaves; later
    states are autograd outputs). Param identity must not change after the first
    capture (lazy first-call capture during training makes construction-after-
    surgery automatic). Unused params (a LoopedRWKV's gates under skip_refine)
    get no captured grad path — exactly matching eager DMT, where the gates get
    no DMT gradient either. The shift OUTPUT is a slice of the (no-grad) chunk
    input, so — as in eager — no gradient flows through it into the previous
    chunk; incoming grads for it are dropped.
    """

    def __init__(self, layer, num_warmup_iters=3):
        self.layer = layer
        self.num_warmup_iters = int(num_warmup_iters)
        self.capture_mode = None  # "global" or "relaxed", set at first capture
        self._stream = None   # ONE stream shared by every warmup + capture (see docstring)
        self._pnames = None   # trainable param names (fixed order)
        self._preal = None    # real params, same order
        self._pstatic = None  # static copies, first-touched on the capture stream
        self._steps = []  # graphed step per chunk index (single-use-per-iteration buffers)

    @torch.no_grad()
    def _sync_params(self):
        """Copy real param values into the static copies (once per iteration)."""
        torch._foreach_copy_(self._pstatic, [p.detach() for p in self._preal])

    def _make_step(self, x, s0, shift, first):
        layer = self.layer
        assert not x.requires_grad, "DMT chunk inputs come from the frozen backbone (no grad)"
        if self._stream is None:
            self._stream = torch.cuda.Stream()
        S = self._stream
        if self._pstatic is None:
            named = [(n, p) for n, p in layer.named_parameters() if p.requires_grad]
            self._pnames = [n for n, _ in named]
            self._preal = [p for _, p in named]
            with torch.cuda.stream(S):  # first touch ON S: accumulators bind to S
                self._pstatic = [p.detach().clone().requires_grad_(True) for p in self._preal]
            torch.cuda.current_stream().wait_stream(S)
        pstatic = self._pstatic
        pdict = dict(zip(self._pnames, pstatic))
        static_in = (x.detach().clone(),
                     s0.detach().clone().requires_grad_(not first),  # j>=1: autograd outputs
                     shift.detach().clone().requires_grad_(not first))
        diff_in = tuple(t for t in static_in if t.requires_grad)

        def run():
            # functional_call: the graph binds to the STATIC param copies, never the
            # real (default-stream-bound) params — see class docstring.
            return torch.func.functional_call(
                layer, pdict, (static_in[0],),
                dict(initial_state=static_in[1], shift_state=static_in[2],
                     return_state=True, skip_refine=True))

        # Park .grad across warmup/capture: capture must be grad-neutral no matter
        # what a hook or warmup pass does.
        saved = [(p, p.grad) for p in self._preal + pstatic]
        for p, _ in saved:
            p.grad = None
        try:
            S.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(S):
                for _ in range(max(1, self.num_warmup_iters)):
                    outs = run()
                    douts = [o for o in outs if o.requires_grad]
                    torch.autograd.grad(douts, diff_in + tuple(pstatic),
                                        grad_outputs=[torch.ones_like(o) for o in douts],
                                        allow_unused=True)
                del outs, douts  # drop the warmup autograd graph before capture
            torch.cuda.current_stream().wait_stream(S)
            torch.cuda.synchronize()

            def capture(mode):
                fwd_graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(fwd_graph, stream=S, capture_error_mode=mode):
                    static_out = run()
                douts = tuple(o for o in static_out if o.requires_grad)
                static_gout = tuple(torch.empty_like(o) for o in douts)
                bwd_graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(bwd_graph, pool=fwd_graph.pool(), stream=S,
                                      capture_error_mode=mode):
                    static_gin = torch.autograd.grad(
                        douts, diff_in + tuple(pstatic), grad_outputs=static_gout,
                        allow_unused=True, retain_graph=False)
                return fwd_graph, bwd_graph, static_out, static_gout, static_gin

            if self.capture_mode is None:
                try:
                    cap = capture("global")
                    self.capture_mode = "global"
                except Exception:
                    torch.cuda.synchronize()
                    cap = capture("relaxed")
                    self.capture_mode = "relaxed"
                print(f"  [dmt-cuda-graph] capture_error_mode={self.capture_mode}", flush=True)
            else:
                cap = capture(self.capture_mode)
        finally:
            for p, g in saved:
                p.grad = g
        fwd_graph, bwd_graph, static_out, static_gout, static_gin = cap

        grad_pos = [i for i, o in enumerate(static_out) if o.requires_grad]
        n_in, n_diff = len(static_in), len(diff_in)

        class _GraphedStep(torch.autograd.Function):
            @staticmethod
            def forward(ctx, *flat):
                for dst, src in zip(static_in, flat[:n_in]):
                    dst.copy_(src)
                fwd_graph.replay()
                return tuple(o.detach() for o in static_out)

            @staticmethod
            def backward(ctx, *gout):
                for sg, i in zip(static_gout, grad_pos):
                    if gout[i] is None:
                        sg.zero_()
                    else:
                        sg.copy_(gout[i])
                bwd_graph.replay()
                res, k = [], 0
                for t in static_in:                    # grads for (x, s0, shift)
                    if t.requires_grad:
                        g = static_gin[k]; k += 1
                        res.append(None if g is None else g.detach())
                    else:
                        res.append(None)
                # grads for the REAL params (values identical: pstatic == preal at
                # replay time); the engine accumulates them into .grad eagerly.
                for g in static_gin[n_diff:]:
                    res.append(None if g is None else g.detach())
                return tuple(res)

        def graphed(x, s0, shift):
            return _GraphedStep.apply(x, s0, shift, *self._preal)

        return graphed

    def step(self, j, x, s0, shift):
        if j >= len(self._steps):
            assert j == len(self._steps), "chunk indices must be visited in order"
            self._steps.append(self._make_step(x, s0, shift, first=(j == 0)))
        if j == 0 and self._pstatic is not None:
            self._sync_params()  # once per iteration: real -> static param values
        return self._steps[j](x, s0, shift)


def dmt_rollout_loss(layer, codec, h_seq, S_gdn=None, *, stride, discount=1.0,
                     block_out=None, max_update_ratio=0.25, target_states=None,
                     graphed=None):
    """Closed-loop rollout: start from teacher state, then consume the student's
    OWN states across chunks; pull the trajectory toward the projected teacher
    states. Trains against exposure-bias drift (DMT). graphed (DMTGraphedRollout)
    replays full-stride chunk steps as captured CUDA graphs; ragged tail chunks
    fall back to the eager call."""
    B, T, C = h_seq.shape
    chunks = _chunks(T, stride)
    tgt = _project_targets(codec, S_gdn, target_states)
    st = tgt[:, 0].detach()
    # graphs need one call signature for all chunk indices: a zeros shift tensor
    # is bit-identical to shift_state=None (the core zero-pads the first token).
    shift = h_seq.new_zeros(B, 1, C) if graphed is not None else None
    mem_loss = h_seq.new_zeros(())
    blk_loss = h_seq.new_zeros(())
    wsum = 0.0
    states = [st]
    for j, (lo, hi) in enumerate(chunks):
        if graphed is not None and hi - lo == stride:
            y, st, shift = graphed.step(j, h_seq[:, lo:hi], st, shift)
        else:  # no graphs, or the ragged tail chunk (shape differs -> eager)
            y, st, shift = layer(h_seq[:, lo:hi], initial_state=st,
                                 shift_state=shift, return_state=True,
                                 skip_refine=True)  # pass-1 (core) semantics — see smt note
        w = discount ** j
        mem_loss = mem_loss + w * F.mse_loss(st.float(), tgt[:, j + 1].detach())
        if block_out is not None:
            blk_loss = blk_loss + w * F.mse_loss(y.float(), block_out[:, lo:hi].float())
        wsum += w
        states.append(st)
    out = {"dmt_memory": mem_loss / wsum, "dmt_block": blk_loss / max(wsum, 1e-9)}
    if recurrent_stability_metrics is not None:
        traj = torch.stack([s.float() for s in states], dim=1)  # [B, nb, H,K,V]
        met = recurrent_stability_metrics(traj)
        out["dmt_state_rms"] = met.state_rms.mean().detach()
        out["dmt_finite"] = met.finite_fraction.detach()
    return out


if __name__ == "__main__":
    # mechanical self-test on synthetic shapes (no model load)
    torch.manual_seed(0)
    codec = BilinearStateCodec()
    S = torch.randn(4, 32, 128, 128)
    out = codec(S)
    assert out.shape == (4, 64, 64, 64), out.shape
    print("codec shape OK:", tuple(out.shape))
    from .rwkv8_deltanet import RWKV8TimeMixDeltaNet
    layer = RWKV8TimeMixDeltaNet(hidden_size=4096, num_heads=64, head_size=64,
                                 depth_layer_id=29, depth_n_layer=32)
    h = torch.randn(2, 4096)
    y = rwkv_readout(layer, codec(torch.randn(2, 32, 128, 128)), h)
    assert y.shape == (2, 4096), y.shape
    print("readout shape OK:", tuple(y.shape))
    # loss plumbing
    hseq = torch.randn(2, 128, 4096)
    Sg = torch.randn(2, 3, 32, 128, 128)
    import os
    os.environ["RWKV8_FORCE_PYREF"] = "1"
    smt = smt_transition_loss(layer, codec, hseq, Sg, stride=64, block_out=torch.randn(2, 128, 4096))
    dmt = dmt_rollout_loss(layer, codec, hseq, Sg, stride=64, block_out=torch.randn(2, 128, 4096))
    print("smt:", {k: round(float(v), 4) for k, v in smt.items()})
    print("dmt:", {k: round(float(v), 4) for k, v in dmt.items()})
    print("SMT_DMT SELF-TEST PASSED")
