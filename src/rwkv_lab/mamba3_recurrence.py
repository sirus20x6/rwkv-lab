"""Isolated Mamba-3 recurrence ablations: exponential discretization, complex state, and MIMO.

Lahoti et al. (2026), https://arxiv.org/abs/2603.15569. Community lead:
https://discord.com/channels/992359628979568762/992362722035507270/1483679154481266729
"""
from __future__ import annotations
import torch
import torch.nn as nn


class Mamba3Recurrence(nn.Module):
    def __init__(self, d_model: int, state_dim: int, *, complex_state: bool = False,
                 n_inputs: int = 1, n_outputs: int = 1):
        super().__init__(); self.state_dim = state_dim; self.complex_state = complex_state
        width = state_dim * (2 if complex_state else 1)
        self.decay = nn.Parameter(torch.zeros(state_dim))
        self.in_proj = nn.Linear(d_model, width * n_inputs, bias=False)
        self.out_proj = nn.Linear(width * n_outputs, d_model, bias=False)
        self.output_mix = nn.Parameter(torch.randn(n_outputs, n_inputs) / max(n_inputs, 1) ** 0.5)
        self.n_inputs, self.n_outputs = n_inputs, n_outputs

    def forward(self, x: torch.Tensor, state: torch.Tensor | None = None):
        b, t, _ = x.shape; width = self.state_dim * (2 if self.complex_state else 1)
        state = x.new_zeros(b, self.n_inputs, width) if state is None else state
        decay = torch.exp(-torch.exp(self.decay)).repeat_interleave(2 if self.complex_state else 1)
        outputs = []
        for i in range(t):
            drive = self.in_proj(x[:, i]).view(b, self.n_inputs, width)
            if self.complex_state:
                pair = state.view(b, self.n_inputs, self.state_dim, 2)
                angle = torch.tanh(drive.view_as(pair)[..., 1])
                c, s = torch.cos(angle), torch.sin(angle)
                real = pair[..., 0] * c - pair[..., 1] * s
                imag = pair[..., 0] * s + pair[..., 1] * c
                state = torch.stack((real, imag), -1).reshape_as(state)
            state = state * decay + drive
            read = torch.einsum("oi,biw->bow", self.output_mix, state).flatten(1)
            outputs.append(self.out_proj(read))
        return torch.stack(outputs, 1), state
