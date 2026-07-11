"""Supervised Memory Training (SMT) objectives for recurrent models.

Kumar et al., *Pretraining Recurrent Networks without Recurrence* (2026),
https://arxiv.org/abs/2606.06479, learns predictive memory labels with a
time-parallel encoder and supervises ``(m_t, x_{t+1}) -> m_{t+1}``.  Community
lead: https://discord.com/channels/992359628979568762/992362722035507270/1516721026589786122
"""
from __future__ import annotations

from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class SMTLoss:
    total: torch.Tensor
    dynamics: torch.Tensor
    future: torch.Tensor
    uniformity: torch.Tensor


class SupervisedMemoryTransition(nn.Module):
    """Time-parallel one-step updater trained from teacher memory labels."""
    def __init__(self, memory_dim: int, input_dim: int, hidden_dim: int | None = None):
        super().__init__()
        hidden_dim = hidden_dim or 2 * memory_dim
        self.net = nn.Sequential(nn.Linear(memory_dim + input_dim, hidden_dim), nn.SiLU(),
                                 nn.Linear(hidden_dim, memory_dim))

    def forward(self, memory: torch.Tensor, next_input: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat((memory, next_input), dim=-1))


def uniformity_loss(memory: torch.Tensor, temperature: float = 2.0) -> torch.Tensor:
    """Wang-Isola pairwise uniformity regularizer used by SMT to resist collapse."""
    flat = memory.reshape(-1, memory.shape[-1]).float()
    if flat.shape[0] < 2:
        return flat.sum() * 0
    flat = F.normalize(flat, dim=-1)
    return torch.pdist(flat).square().mul(-temperature).exp().mean().clamp_min(1e-12).log()


def supervised_memory_loss(updater: nn.Module, teacher_memory: torch.Tensor,
                           next_inputs: torch.Tensor, *, future_loss: torch.Tensor | None = None,
                           dynamics_weight: float = 1.0, future_weight: float = 1.0,
                           uniformity_weight: float = 0.01,
                           detach_labels: bool = False) -> SMTLoss:
    """Compute SMT's parallel transition, predictive-future, and anti-collapse terms.

    ``teacher_memory`` is [B,T+1,M], while ``next_inputs`` is [B,T,X].
    Labels may remain attached for joint encoder/updater training (paper default)
    or be detached when consuming an offline teacher-label artifact.
    """
    if teacher_memory.ndim != 3 or next_inputs.shape[:2] != (teacher_memory.shape[0],
                                                             teacher_memory.shape[1] - 1):
        raise ValueError("expected teacher_memory [B,T+1,M] and next_inputs [B,T,X]")
    current, target = teacher_memory[:, :-1], teacher_memory[:, 1:]
    if detach_labels:
        current, target = current.detach(), target.detach()
    predicted = updater(current, next_inputs)
    dynamics = F.mse_loss(predicted.float(), target.float())
    future = dynamics.new_zeros(()) if future_loss is None else future_loss
    uniformity = uniformity_loss(teacher_memory)
    total = dynamics_weight * dynamics + future_weight * future + uniformity_weight * uniformity
    return SMTLoss(total, dynamics, future, uniformity)


@torch.no_grad()
def rollout_drift(updater: nn.Module, initial_memory: torch.Tensor, inputs: torch.Tensor,
                  teacher_memory: torch.Tensor) -> torch.Tensor:
    """DMT readiness diagnostic: per-step MSE under the updater's own state distribution."""
    memory, errors = initial_memory, []
    for step in range(inputs.shape[1]):
        memory = updater(memory, inputs[:, step])
        errors.append(F.mse_loss(memory.float(), teacher_memory[:, step + 1].float()))
    return torch.stack(errors)
