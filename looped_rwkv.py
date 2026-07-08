#!/usr/bin/env python
"""Weight-tied N-loop refinement wrapper around RWKV8TimeMixDeltaNet.

LT2-style (research/LT2-RWKV apps/LT2/transformer.py): each looped iteration runs
the core on a *normalized* hidden and adds it as a residual with a zero-init
weight. The pre-norm is the key stabilizer — it bounds the core's input no matter
how large the running output gets, breaking the positive-feedback gain. (An
earlier version fed the raw, un-normalized output back as input and accumulated
it; that has unbounded gain and diverged to Inf/NaN within a few layers.)

Init-preserving: residual_weight=0 => loop 1 == single-pass, so a codec-initialized
core (and the lossless top layers) are untouched; loops 2..N only add once trained.

Gate granularity (gate_mode): how many independent gates each refinement pass has.
  scalar   — one gate per pass (legacy; all channels absorb the pass equally)
  head     — one gate per head-group per pass [n_loops, G]; per-group "loop rate"
  channel  — one gate per channel per pass [n_loops, C]
  factored — head factor x channel factor: rw[i,g] * (1 + gate_chan[i,c]).
             residual_weight (head factor) is zero-init and receives the POOLED
             gradient of its group's channels (better escape SNR); gate_chan stores
             a DELTA around 1 (zero-init -> factor 1), so (a) the product is 0 at
             init (exact no-op preserved), (b) gate_chan's gradient is gated by the
             head factor -> automatic coarse-to-fine curriculum, and (c) weight
             decay pulls the channel factor toward 1, not 0 (the gauge fix).

gate_cap (>0): soft-cap the effective gate to (-cap, cap) via cap*tanh(g/cap).
  residual_weight is otherwise unbounded (iter_norm bounds each pass's INPUT but
  not the accumulated output), so a hot loop LR could destabilize the block. The
  cap bounds the loop's contribution BY CONSTRUCTION (OpenMythos's spectral-radius
  argument, adapted), instead of relying on grad-clip + the dashboard's after-the-
  fact loop_pinned cool. tanh(0)=0 so the init no-op is preserved; near 0 it is
  ~identity, so small gates behave exactly as uncapped.

loop_index (bool): add a per-pass, zero-init learned offset to the pass input so
  the weight-tied core can specialize each refinement pass (OpenMythos loop-index
  embedding). Zero-init => adds nothing at init, so pass 1 stays the faithful
  single-pass and the whole loop is still an exact no-op until trained.

hyper_lanes (K>=2): hyper-connections at the loop boundary (arXiv 2409.19606; the
  iso-depth scaling-law paper 2604.21106 measured this as the largest loop-capacity
  lever, recurrence-equivalence exponent 0.45 -> 0.65). The single running output
  is replaced by K parallel residual lanes; each refinement pass reads a learned
  lane pool as its input (hyper_alpha, one-hot init rotating by pass), writes its
  gated increment to lanes with per-lane shares (hyper_write, ones init), and mixes
  lanes (hyper_mix, identity init); the block output is a learned lane read
  (hyper_read, one-hot-on-lane-0 init — HC's uniform sum-pool is float-inexact for
  non-power-of-2 K; the pool can be learned). Composition with the ZERO-INIT _gate keeps the
  exact no-op invariant that the HC paper's ones-init write alone would break:
  at init all lanes equal the pass-1 output, every pool/mix/read reproduces it, and
  the write adds gate*inc = 0. K=1 is refused (provably no better than a plain
  residual — the HC paper's Lambda-pattern needs n>1). Static HC only: with K=2 the
  paper finds static ~= dynamic, and static keeps the loop cheap. NOTE: like the
  other gates these are exact-value-anchored tiny tensors -> fp32 via float_gates();
  and weight decay would pull hyper_mix/hyper_alpha away from their identity/one-hot
  anchors — keep decay off the rwkv_loop group (repo default decay_now=0).

lora_rank (R>0): per-refinement-pass LoRA on the shared core's linears (CART found
  unsharing loop weights beat weight-tying by 5-6%; Dreamer found tied projections
  hurt; MoDr's LoRA-branch recipe). Pass i (i>=2) computes each target linear as
  W x + B_i A_i x with A_i kaiming-init, B_i ZERO-init -> exact no-op until trained.
  PASS 1 IS NEVER ADAPTED: it must stay the faithful single-pass function the codec
  initialized. Applied via forward hooks keyed on _lora_pass, so the core's module
  tree and every existing checkpoint key are unchanged; the A/B tensors live on the
  wrapper as loop_lora_A/B ParameterDicts ("{pass}_{target}") and ride the rwkv_loop
  optimizer group. Direct core calls (SMT/DMT, skip_refine) see _lora_pass=0 -> the
  bare shared core, always.

sample_loop_count(): per-step training loop-count sampling (module-level helper for
  the trainers). Dynamic beats fixed n in four recurrent-depth papers (depth
  extrapolation, less overthinking); "uniform" = U{1..n}; "poisson" = 1+Pois(n-1)
  clamped to [1,n] (mass at full depth, occasional shallow pass).

iter_consist (attr, default False): equilibrium internalization (Solve-the-Loop
  2605.12466: the backbone proposal drifts toward the loop's own fixed point, so
  fewer iterations are needed over training; a finite-horizon cousin of NextLat on
  loop iterates). When True and grads are enabled, the forward also computes
  self.last_iter_consist = mean_i MSE(out_i, sg(out_final)) over passes i<n —
  pulling every earlier iterate toward the (stop-grad) final one. The trainer adds
  --loop-iter-consist W * last_iter_consist to the loss. Combined with
  --loop-sample this reproduces LoopFormer's shortcut-consistency recipe (short
  trajectories trained on the task loss + latent consistency to the full route).
  TENSION to be aware of: "silent thinking" (2603.21676) found per-iterate ANSWER
  supervision harmful for OOD compositional generalization; this is output-space
  self-consistency, not answer supervision, but default it off and A/B it.

Scale note (readout blind-spot, arXiv 2606.24898): conversion training pins the
  block output's scale to the GDN teacher via block/consolidate MSE, so the
  CE-invisible hidden-norm drift that pure-CE looped LMs suffer is anchored here;
  iter_norm bounds each pass's input and gate_cap bounds each pass's contribution.
  In pure-CE phases (w_lmce-only), watch per-pass output RMS via loop_probe.py.

skip_refine (forward kwarg): return the pass-1 output only. Refinement passes
  re-run the core WITHOUT initial_state/shift_state (each pass re-reads the given
  window from a zero state), so in chunked state-supervised calls (SMT/DMT) the
  refined output is NOT the function the full-window block loss trains. Chunked
  callers pass skip_refine=True to get pure pass-1 (core) semantics — consistent
  with the pass-1-only state supervision, and no n_loops x chunk cost. A bare
  core swallows the kwarg via **kwargs, so call sites need no isinstance checks.

The gate params (residual_weight/gate_chan/loop_index_embed) are kept in fp32 via
float_gates(): they grow from zero by tiny optimizer steps that bf16's ~3
significant digits can quantize away. _gate()/loop_index cast back to the stream
dtype at the use site, so the residual stream never gets promoted.

All modes/options are exact no-ops at init. A coarser checkpoint broadcasts
losslessly into a finer mode (convert_train._expand_loop_gates).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def sample_loop_count(mode: str, n_loops: int, rng) -> int:
    """Per-step training loop count. rng is a numpy Generator."""
    if mode == "uniform":
        return int(rng.integers(1, n_loops + 1))
    if mode == "poisson":
        return int(min(n_loops, 1 + rng.poisson(max(0, n_loops - 1))))
    return n_loops


def lora_config_from_sd(sd):
    """Infer (rank, targets) of a LoopedRWKV per-pass LoRA from its state dict —
    like gate_mode, the config is constructor state and must be reconstructed."""
    keys = [str(k) for k in sd if str(k).startswith("loop_lora_A.")]
    if not keys:
        return 0, ()
    rank = int(sd[keys[0]].shape[0])
    targets = tuple(sorted({k.split(".", 1)[1].split("_", 1)[1] for k in keys}))
    return rank, targets


class _RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        dt = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * self.weight.float()).to(dt)


class LoopedRWKV(nn.Module):
    def __init__(self, core, n_loops: int = 4, hidden_size: int | None = None,
                 gate_mode: str = "scalar", gate_cap: float = 0.0,
                 loop_index: bool = False, hyper_lanes: int = 0,
                 lora_rank: int = 0, lora_targets=("receptance", "key", "value", "output")):
        super().__init__()
        self.core = core
        self.n_loops = int(n_loops)
        H = hidden_size if hidden_size is not None else core.hidden_size
        G = int(getattr(core, "num_heads", 0)) or 64
        assert H % G == 0, f"hidden {H} not divisible by head-groups {G}"
        self.gate_mode = gate_mode
        self.gate_cap = float(gate_cap)
        self.loop_index = bool(loop_index)
        self.hyper_lanes = int(hyper_lanes)
        if self.hyper_lanes:
            if self.hyper_lanes < 2:
                raise ValueError(f"hyper_lanes={self.hyper_lanes}: K=1 hyper-connections are "
                                 f"provably no better than a plain residual (HC paper); use K>=2")
            K = self.hyper_lanes
            alpha = torch.zeros(self.n_loops, K)      # pass-input lane pool, one-hot
            for i in range(self.n_loops):             # rotating by pass (HC's e_{k mod n})
                alpha[i, i % K] = 1.0
            self.hyper_alpha = nn.Parameter(alpha)
            self.hyper_mix = nn.Parameter(             # lane->lane mixing, identity init
                torch.eye(K).unsqueeze(0).repeat(self.n_loops, 1, 1))
            self.hyper_write = nn.Parameter(torch.ones(self.n_loops, K))  # per-lane write share
            read = torch.zeros(K)          # one-hot (not HC's uniform sum-pool): 1/K
            read[0] = 1.0                  # weights round for non-power-of-2 K, breaking
            self.hyper_read = nn.Parameter(read)  # the exact-no-op invariant; e_0 is exact for any K
        self.ch_per_group = H // G
        self.iter_norm = _RMSNorm(H)                       # pre-norm => bounded iteration input
        if gate_mode == "scalar":
            self.residual_weight = nn.Parameter(torch.zeros(self.n_loops))  # zero-init
        elif gate_mode == "head":
            self.residual_weight = nn.Parameter(torch.zeros(self.n_loops, G))
        elif gate_mode == "channel":
            self.residual_weight = nn.Parameter(torch.zeros(self.n_loops, H))
        elif gate_mode == "factored":
            self.residual_weight = nn.Parameter(torch.zeros(self.n_loops, G))   # head factor (0)
            self.gate_chan = nn.Parameter(torch.zeros(self.n_loops, H))         # channel DELTA (0 -> factor 1)
        else:
            raise ValueError(f"unknown gate_mode {gate_mode!r}")
        if self.loop_index:                                # per-pass input offset (zero-init no-op)
            self.loop_index_embed = nn.Parameter(torch.zeros(self.n_loops, H))
        self.lora_rank = int(lora_rank)
        self._lora_pass = 0                                # 0 = bare shared core (pass 1 / direct calls)
        self.iter_consist = False                          # trainer-set; see module docstring
        self.last_iter_consist = None                      # fp32 scalar after a consist forward
        if self.lora_rank > 0:
            if self.n_loops < 2:
                raise ValueError("lora_rank needs n_loops > 1: the adapters live on refinement passes")
            targets = tuple(t for t in lora_targets
                            if isinstance(getattr(core, t, None), nn.Linear))
            rejected = tuple(t for t in lora_targets if t not in targets)
            if rejected:   # loud, not silent: a typo'd target would otherwise just train less
                raise ValueError(f"lora_targets {rejected} are not nn.Linear attrs on the core "
                                 f"(valid here: {tuple(n for n, m in core.named_children() if isinstance(m, nn.Linear))})")
            self.lora_targets = targets
            A, Bd = {}, {}
            for i in range(1, self.n_loops):
                for t in targets:
                    lin = getattr(core, t)
                    A[f"{i}_{t}"] = nn.Parameter(
                        torch.randn(self.lora_rank, lin.in_features) / math.sqrt(lin.in_features))
                    Bd[f"{i}_{t}"] = nn.Parameter(          # zero-init -> exact no-op
                        torch.zeros(lin.out_features, self.lora_rank))
            self.loop_lora_A = nn.ParameterDict(A)
            self.loop_lora_B = nn.ParameterDict(Bd)
            for t in targets:                               # hooks keep module tree + ckpt keys intact
                getattr(core, t).register_forward_hook(self._make_lora_hook(t))
        self._save_key = getattr(core, "_save_key", None)

    def _make_lora_hook(self, t):
        def hook(mod, args, output):
            i = self._lora_pass
            if not i:                                       # pass 1 / direct core calls: bare core
                return None
            A = self.loop_lora_A[f"{i}_{t}"]
            B = self.loop_lora_B[f"{i}_{t}"]
            x = args[0]
            return output + F.linear(F.linear(x, A.to(x.dtype)), B.to(x.dtype))
        return hook

    def float_gates(self):
        """Re-cast the loop-gate params to fp32 (call after a module-wide
        .to(dtype=bf16)). Zero-init gates grow by tiny optimizer steps; bf16
        quantizes those away once the gate has magnitude (the repo's
        fp32-master-weights finding). Tiny tensors, off the matmul hot path —
        forward casts back to the stream dtype at the use site."""
        self.residual_weight.data = self.residual_weight.data.float()
        if self.gate_mode == "factored":
            self.gate_chan.data = self.gate_chan.data.float()
        if self.loop_index:
            self.loop_index_embed.data = self.loop_index_embed.data.float()
        if self.hyper_lanes:
            # anchored at exact 0/1 values; bf16 ulp at 1.0 is ~0.004 — far coarser
            # than the optimizer steps that move these
            for n in ("hyper_alpha", "hyper_mix", "hyper_write", "hyper_read"):
                getattr(self, n).data = getattr(self, n).data.float()
        if self.lora_rank > 0:
            # B grows from zero by tiny steps (the fp32-master-weights finding);
            # the hook casts to the stream dtype at the matmul
            for pd in (self.loop_lora_A, self.loop_lora_B):
                for k in pd:
                    pd[k].data = pd[k].data.float()
        return self

    def loop_param_names(self) -> set[str]:
        """Names of the loop-GATE params: the zero-init, tiny-gradient tensors that
        want the dedicated rwkv_loop optimizer group + loop_lr_mult steering.
        iter_norm/core params are NOT gates and stay in their normal groups."""
        names = {"residual_weight"}
        if self.gate_mode == "factored":
            names.add("gate_chan")
        if self.loop_index:
            names.add("loop_index_embed")
        if self.hyper_lanes:
            names |= {"hyper_alpha", "hyper_mix", "hyper_write", "hyper_read"}
        if self.lora_rank > 0:   # full dotted names: optimizer routing matches named_parameters()
            names |= {f"loop_lora_A.{k}" for k in self.loop_lora_A}
            names |= {f"loop_lora_B.{k}" for k in self.loop_lora_B}
        return names

    @staticmethod
    def _t(y):
        return y[0] if isinstance(y, tuple) else y

    def _gate(self, i):
        """Effective gate for pass i: 0-dim (scalar) or [C] (head/channel/factored),
        soft-capped to (-gate_cap, gate_cap) when gate_cap>0."""
        rw = self.residual_weight[i]
        if self.gate_mode in ("scalar", "channel"):
            g = rw
        else:
            g = rw.repeat_interleave(self.ch_per_group)    # [G] -> [C]
            if self.gate_mode == "factored":
                g = g * (1.0 + self.gate_chan[i])
        if self.gate_cap > 0.0:
            g = self.gate_cap * torch.tanh(g / self.gate_cap)  # tanh(0)=0 keeps the init no-op
        return g

    @torch.no_grad()
    def effective_rw(self):
        """Per-pass effective gates for telemetry: [n_loops] (scalar) or [n_loops, C].
        Reflects gate_cap, so the dashboard/detector see the true bounded gate.
        _gate(i) is 0-dim for scalar and [C] otherwise, so a single stack covers both."""
        return torch.stack([self._gate(i) for i in range(self.n_loops)])

    def forward(self, hidden_states, *args, **kwargs):
        skip_refine = bool(kwargs.pop("skip_refine", False))
        loop_trace = kwargs.pop("loop_trace", None)   # list -> append each pass's out (loop_probe.py)
        if loop_trace is None:                        # HF layer call sites can't thread kwargs;
            loop_trace = getattr(self, "_probe_trace", None)  # the probe sets this attr instead
        return_state = bool(kwargs.get("return_state", False))
        first = self.core(hidden_states, *args, **kwargs)
        if return_state:
            out, final_state, new_shift_state = first
        else:
            out = self._t(first)                          # pass 1 == single-pass output
        # collect NON-detached per-pass outputs when either the equilibrium-consistency loss
        # (iter_consist) OR external per-iterate readout supervision (keep_iterates, the
        # Readout-Blind-Spot fix 2606.24898 — the trainer supervises each iterate vs the
        # teacher target) is active.
        keep_iters = getattr(self, "keep_iterates", False)
        collect = ((self.iter_consist or keep_iters) and torch.is_grad_enabled()
                   and not skip_refine and self.n_loops > 1)
        consist = collect and self.iter_consist       # consistency loss only when requested
        iters = [out] if collect else None            # NON-detached (unlike loop_trace)
        try:
            if self.hyper_lanes and not skip_refine and self.n_loops > 1:
                # Hyper-connection lanes: K copies of the running output, mixed per pass.
                # At init (one-hot alpha, identity mix, zero gates, uniform read) this is
                # numerically identical to the plain loop below — see module docstring.
                K = self.hyper_lanes
                lanes = out.unsqueeze(0).expand(K, *out.shape)         # [K,B,T,C]
                bshape = (K,) + (1,) * out.dim()
                r = self.hyper_read.to(out.dtype)
                if loop_trace is not None:
                    loop_trace.append(out.detach())
                for i in range(1, self.n_loops):
                    a = self.hyper_alpha[i].to(out.dtype)
                    inp = hidden_states + (a.view(bshape) * lanes).sum(0)   # lane pool
                    if self.loop_index:
                        inp = inp + self.loop_index_embed[i].to(inp.dtype)
                    if self.lora_rank:
                        self._lora_pass = i               # per-pass adapters on the shared core
                    inc = self._t(self.core(self.iter_norm(inp)))
                    ginc = self._gate(i).to(inc.dtype) * inc
                    M = self.hyper_mix[i].to(out.dtype)
                    w = self.hyper_write[i].to(out.dtype)
                    lanes = torch.einsum("kj,j...->k...", M, lanes) + w.view(bshape) * ginc
                    if loop_trace is not None:                         # per-pass lane read
                        loop_trace.append((r.view(bshape) * lanes).sum(0).detach())
                    if collect:
                        iters.append((r.view(bshape) * lanes).sum(0))
                out = (r.view(bshape) * lanes).sum(0)                  # output lane read
            else:
                if loop_trace is not None:
                    loop_trace.append(out.detach())
                for i in range(1, self.n_loops):
                    if skip_refine:                           # pass-1 (core) semantics only
                        break
                    # refine on a NORMALIZED hidden (input + running output); zero-init gates.
                    inp = hidden_states + out
                    if self.loop_index:                       # per-pass specialization offset
                        inp = inp + self.loop_index_embed[i].to(inp.dtype)
                    if self.lora_rank:
                        self._lora_pass = i
                    inc = self._t(self.core(self.iter_norm(inp)))
                    out = out + self._gate(i).to(inc.dtype) * inc
                    if loop_trace is not None:
                        loop_trace.append(out.detach())
                    if collect:
                        iters.append(out)
        finally:
            # a mid-refinement exception must never leave adapters armed: direct core
            # calls (SMT/DMT, evaluate) rely on _lora_pass==0 meaning the bare core
            self._lora_pass = 0
        if consist and len(iters) > 1:
            fin = iters[-1].detach().float()          # early iterates chase the final one,
            self.last_iter_consist = torch.stack(     # never the other way (stop-grad)
                [F.mse_loss(o.float(), fin) for o in iters[:-1]]).mean()
        else:
            self.last_iter_consist = None
        # Readout-Blind-Spot hook: non-detached per-pass outputs for the trainer to supervise
        # each iterate against the EXTERNAL teacher target (distinct from iter_consist, which
        # is self-supervised toward the student's own final). None unless keep_iterates is set.
        self.last_iterates = iters if collect else None
        if return_state:
            # SMT/DMT supervise the underlying RWKV recurrent memory. The refinement
            # passes are output refinements, not separate target state spaces.
            return out, final_state, new_shift_state
        return out
