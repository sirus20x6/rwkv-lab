"""Routing-Free MoE: decentralized expert self-activation.

Implements the router/softmax/top-k-free architecture of Routing-Free
Mixture-of-Experts (Liu et al., 2026), https://arxiv.org/abs/2604.00801 and
https://github.com/liuyilun2000/RoutingFreeMoE/tree/release. Community lead:
https://discord.com/channels/992359628979568762/992359629419991142/1525186839227400242
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class RoutingFreeExpert(nn.Module):
    def __init__(self, d_model: int, hidden_dim: int, rank: int):
        super().__init__()
        self.down = nn.Linear(d_model, rank, bias=False)
        self.up = nn.Linear(rank, hidden_dim, bias=False)
        self.gate = nn.Linear(d_model, hidden_dim, bias=False)
        self.out = nn.Linear(hidden_dim, d_model, bias=False)
        self.bias = nn.Parameter(torch.zeros(()))

    def forward(self, x: torch.Tensor, threshold: float = 0.0):
        internal = self.down(x)
        # The preference originates inside the expert (AoE norm), then ReLU
        # supplies natural sparsity without centralized comparison (§3.1).
        proxy = internal.float().norm(dim=-1) / max(internal.shape[-1] ** 0.5, 1.0) - self.bias
        weight = F.relu(proxy - threshold).to(x.dtype)
        value = self.out(F.silu(self.up(internal)) * self.gate(x))
        active = proxy > threshold
        return value * weight.unsqueeze(-1), weight, active


class RoutingFreeMoE(nn.Module):
    def __init__(self, d_model: int, *, n_experts: int = 4, hidden_dim: int | None = None,
                 rank: int = 32, threshold: float = 0.2, balance_interpolation: float = 0.5):
        super().__init__()
        if n_experts < 1 or rank < 1 or not 0 <= balance_interpolation <= 1:
            raise ValueError("invalid Routing-Free MoE configuration")
        hidden_dim = hidden_dim or 4 * d_model
        self.experts = nn.ModuleList(RoutingFreeExpert(d_model, hidden_dim, rank)
                                     for _ in range(n_experts))
        self.threshold, self.balance_interpolation = threshold, balance_interpolation
        self.last_stats: dict[str, float] = {}
        self.aux_loss = torch.tensor(0.0)

    def forward(self, x: torch.Tensor, **_kwargs) -> torch.Tensor:
        triples = [expert(x, self.threshold) for expert in self.experts]
        values = torch.stack([v for v, _, _ in triples], dim=-2)
        proxy = torch.stack([w for _, w, _ in triples], dim=-1)
        active = torch.stack([a for _, _, a in triples], dim=-1)
        # Dot products between detached binary load and differentiable proxy,
        # interpolating expert- and token-balancing (§3.2, eqs. 13–15).
        target = active.float().mean().detach()
        expert_load = active.float().flatten(0, -2).mean(0).detach()
        expert_proxy = proxy.float().flatten(0, -2).mean(0)
        token_load = active.float().mean(-1).detach()
        token_proxy = proxy.float().mean(-1)
        eb = ((expert_load - target) * (expert_proxy - target)).mean()
        tb = ((token_load - target) * (token_proxy - target)).mean()
        alpha = self.balance_interpolation
        self.aux_loss = alpha * eb + (1 - alpha) * tb
        self.last_stats = {"activation_density": float(active.float().mean().detach()),
                           "experts_per_token": float(active.float().sum(-1).mean().detach()),
                           "expert_balance_loss": float(eb.detach()),
                           "token_balance_loss": float(tb.detach())}
        return values.sum(dim=-2)
