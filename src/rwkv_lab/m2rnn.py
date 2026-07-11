"""Matrix-to-Matrix nonlinear recurrent layer oracle.

Mishra et al. (2026), https://arxiv.org/abs/2603.14360. Community lead:
https://discord.com/channels/992359628979568762/992362722035507270/1484506344588574881
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class M2RNN(nn.Module):
    def __init__(self, d_model: int, state_size: int, rank: int = 8):
        super().__init__(); self.state_size = state_size
        self.q = nn.Linear(d_model, state_size, bias=False); self.k = nn.Linear(d_model, state_size, bias=False)
        self.v = nn.Linear(d_model, state_size, bias=False); self.read = nn.Linear(state_size, d_model, bias=False)
        self.left = nn.Parameter(torch.randn(state_size, rank) * 0.02)
        self.right = nn.Parameter(torch.randn(state_size, rank) * 0.02)

    def forward(self, x: torch.Tensor, state: torch.Tensor | None = None):
        b, t, _ = x.shape; state = x.new_zeros(b, self.state_size, self.state_size) if state is None else state
        out = []
        for i in range(t):
            k, v, q = self.k(x[:, i]), self.v(x[:, i]), self.q(x[:, i])
            transition = torch.tanh(state + self.left @ self.right.T)
            state = 0.9 * transition + torch.einsum("bi,bj->bij", k, v)
            out.append(self.read(torch.einsum("bi,bij->bj", F.normalize(q, dim=-1), state)))
        return torch.stack(out, 1), state
