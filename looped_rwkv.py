#!/usr/bin/env python
"""Weight-tied N-loop refinement wrapper around RWKV8TimeMixDeltaNet.

LT2-style (apps/LT2/transformer.py): each looped iteration runs the core on a
*normalized* hidden and adds it as a residual with a zero-init weight. The
pre-norm is the key stabilizer — it bounds the core's input no matter how large
the running output gets, breaking the positive-feedback gain. (An earlier version
fed the raw, un-normalized output back as input and accumulated it; that has
unbounded gain and diverged to Inf/NaN within a few layers.)

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
All modes are exact no-ops at init. A coarser checkpoint broadcasts losslessly
into a finer mode (convert_train._expand_loop_gates).
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
                 gate_mode: str = "scalar"):
        super().__init__()
        self.core = core
        self.n_loops = int(n_loops)
        H = hidden_size if hidden_size is not None else core.hidden_size
        G = int(getattr(core, "num_heads", 0)) or 64
        assert H % G == 0, f"hidden {H} not divisible by head-groups {G}"
        self.gate_mode = gate_mode
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
        self._save_key = getattr(core, "_save_key", None)

    @staticmethod
    def _t(y):
        return y[0] if isinstance(y, tuple) else y

    def _gate(self, i):
        """Effective gate for pass i: 0-dim (scalar) or [C] (head/channel/factored)."""
        rw = self.residual_weight[i]
        if self.gate_mode in ("scalar", "channel"):
            return rw
        rw = rw.repeat_interleave(self.ch_per_group)       # [G] -> [C]
        if self.gate_mode == "factored":
            rw = rw * (1.0 + self.gate_chan[i])
        return rw

    @torch.no_grad()
    def effective_rw(self):
        """Per-pass effective gates for telemetry: [n_loops] (scalar) or [n_loops, C]."""
        if self.gate_mode == "scalar":
            return self.residual_weight.detach().clone()
        return torch.stack([self._gate(i) for i in range(self.n_loops)])

    def forward(self, hidden_states, *args, **kwargs):
        return_state = bool(kwargs.get("return_state", False))
        first = self.core(hidden_states, *args, **kwargs)
        if return_state:
            out, final_state, new_shift_state = first
        else:
            out = self._t(first)                          # pass 1 == single-pass output
        for i in range(1, self.n_loops):
            # refine on a NORMALIZED hidden (input + running output); zero-init gates.
            inc = self._t(self.core(self.iter_norm(hidden_states + out)))
            out = out + self._gate(i) * inc
        if return_state:
            # SMT/DMT supervise the underlying RWKV recurrent memory. The refinement
            # passes are output refinements, not separate target state spaces.
            return out, final_state, new_shift_state
        return out
