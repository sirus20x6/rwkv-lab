"""Energy-based compatibility learning and bounded latent refinement.

Gladstone et al. (2025), https://arxiv.org/abs/2507.02092. Community lead:
https://discord.com/channels/992359628979568762/992359629419991142/1521434262778286160
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class CompatibilityEnergy(nn.Module):
    def __init__(self, d_model: int, hidden: int | None = None):
        super().__init__(); hidden = hidden or d_model
        self.net = nn.Sequential(nn.Linear(3*d_model, hidden), nn.SiLU(), nn.Linear(hidden, 1))
    def forward(self, context, candidate):
        return self.net(torch.cat((context, candidate, context*candidate), -1)).squeeze(-1)


def contrastive_energy_loss(model, context, positive, negative, margin: float = 1.0):
    return F.relu(margin + model(context, positive) - model(context, negative)).mean()


def refine_latent(model, context, initial, *, steps=4, step_size=0.05, max_delta=1.0):
    """Optimize only the candidate latent; model weights remain unchanged."""
    initial = initial.detach()
    value = initial.clone()
    for _ in range(steps):
        # Detach each iteration so the projected (non-leaf) value from the
        # previous step becomes a fresh leaf; otherwise requires_grad_ raises.
        value = value.detach().requires_grad_(True); energy = model(context, value).sum()
        grad, = torch.autograd.grad(energy, value)
        value = (value - step_size*grad).detach()
        delta = value - initial
        value = initial + delta * (max_delta / delta.norm(dim=-1, keepdim=True).clamp_min(max_delta))
    return value.detach()
