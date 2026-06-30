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
    def __init__(self, core, n_loops: int = 4, hidden_size: int | None = None):
        super().__init__()
        self.core = core
        self.n_loops = int(n_loops)
        H = hidden_size if hidden_size is not None else core.hidden_size
        self.iter_norm = _RMSNorm(H)                       # pre-norm => bounded iteration input
        self.residual_weight = nn.Parameter(torch.zeros(self.n_loops))  # zero-init
        self._save_key = getattr(core, "_save_key", None)

    @staticmethod
    def _t(y):
        return y[0] if isinstance(y, tuple) else y

    def forward(self, hidden_states, *args, **kwargs):
        return_state = bool(kwargs.get("return_state", False))
        first = self.core(hidden_states, *args, **kwargs)
        if return_state:
            out, final_state, new_shift_state = first
        else:
            out = self._t(first)                          # pass 1 == single-pass output
        for i in range(1, self.n_loops):
            # refine on a NORMALIZED hidden (input + running output); zero-init weight.
            inc = self._t(self.core(self.iter_norm(hidden_states + out)))
            out = out + self.residual_weight[i] * inc
        if return_state:
            # SMT/DMT supervise the underlying RWKV recurrent memory. The refinement
            # passes are output refinements, not separate target state spaces.
            return out, final_state, new_shift_state
        return out
