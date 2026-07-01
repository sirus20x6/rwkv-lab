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

import torch
import torch.nn as nn


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
                 loop_index: bool = False):
        super().__init__()
        self.core = core
        self.n_loops = int(n_loops)
        H = hidden_size if hidden_size is not None else core.hidden_size
        G = int(getattr(core, "num_heads", 0)) or 64
        assert H % G == 0, f"hidden {H} not divisible by head-groups {G}"
        self.gate_mode = gate_mode
        self.gate_cap = float(gate_cap)
        self.loop_index = bool(loop_index)
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
        self._save_key = getattr(core, "_save_key", None)

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
        return_state = bool(kwargs.get("return_state", False))
        first = self.core(hidden_states, *args, **kwargs)
        if return_state:
            out, final_state, new_shift_state = first
        else:
            out = self._t(first)                          # pass 1 == single-pass output
        for i in range(1, self.n_loops):
            if skip_refine:                               # pass-1 (core) semantics only
                break
            # refine on a NORMALIZED hidden (input + running output); zero-init gates.
            inp = hidden_states + out
            if self.loop_index:                           # per-pass specialization offset
                inp = inp + self.loop_index_embed[i].to(inp.dtype)
            inc = self._t(self.core(self.iter_norm(inp)))
            out = out + self._gate(i).to(inc.dtype) * inc
        if return_state:
            # SMT/DMT supervise the underlying RWKV recurrent memory. The refinement
            # passes are output refinements, not separate target state spaces.
            return out, final_state, new_shift_state
        return out
