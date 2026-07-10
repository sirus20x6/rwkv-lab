"""Decoupled DiLoCo outer-update primitives.

References:

* Douillard et al., "DiLoCo: Distributed Low-Communication Training of
  Language Models", arXiv:2311.08105, https://arxiv.org/abs/2311.08105.
* "Decoupled DiLoCo: Asynchronous Distributed Training for Foundation
  Models", arXiv:2604.21428, https://arxiv.org/abs/2604.21428.

DiLoCo treats each learner's local parameter displacement as a pseudo-gradient.
The decoupled variant admits independently arriving, token-weighted updates.
This module owns that merge and outer optimizer state; process launch, learner
leases, and failure recovery remain in Adamaton.
"""
from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping
import torch


@dataclass
class LearnerUpdate:
    learner: str
    base_version: int
    tokens: int
    delta: dict[str, torch.Tensor]


def pseudo_gradient(reference: Mapping[str, torch.Tensor],
                    local: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Return reference-local, the DiLoCo outer pseudo-gradient."""

    if reference.keys() != local.keys():
        raise ValueError("reference and local parameter names differ")
    return {k: reference[k].detach().float() - local[k].detach().float() for k in reference}


def token_weighted_merge(updates: list[LearnerUpdate]) -> dict[str, torch.Tensor]:
    if not updates or sum(max(0, u.tokens) for u in updates) <= 0:
        raise ValueError("at least one positive-token learner update is required")
    keys = updates[0].delta.keys()
    if any(u.delta.keys() != keys for u in updates):
        raise ValueError("learner update parameter names differ")
    total = float(sum(max(0, u.tokens) for u in updates))
    return {k: sum(u.delta[k].float() * (max(0, u.tokens) / total) for u in updates)
            for k in keys}


class DecoupledDiLoCo:
    """Asynchronous token-weighted outer SGD with momentum and staleness guard."""

    def __init__(self, *, outer_lr: float = 0.7, momentum: float = 0.9,
                 max_staleness: int = 2):
        self.outer_lr, self.momentum = float(outer_lr), float(momentum)
        self.max_staleness = int(max_staleness)
        self.version = 0
        self.velocity: dict[str, torch.Tensor] = {}

    def apply(self, params: Mapping[str, torch.Tensor], updates: list[LearnerUpdate]) -> int:
        accepted = [u for u in updates if 0 <= self.version - u.base_version <= self.max_staleness]
        if not accepted:
            return 0
        grad = token_weighted_merge(accepted)
        with torch.no_grad():
            for name, p in params.items():
                g = grad[name].to(device=p.device, dtype=torch.float32)
                v = self.velocity.get(name)
                v = torch.zeros_like(g) if v is None else v.to(device=g.device, dtype=g.dtype)
                v = self.momentum * v + g
                self.velocity[name] = v.detach().cpu()
                p.add_(v.to(device=p.device, dtype=p.dtype), alpha=-self.outer_lr)
        self.version += 1
        return len(accepted)

    def state_dict(self) -> dict:
        return {"version": self.version, "velocity": self.velocity,
                "outer_lr": self.outer_lr, "momentum": self.momentum,
                "max_staleness": self.max_staleness}

    def load_state_dict(self, state: dict) -> None:
        self.version = int(state["version"])
        self.velocity = {k: v.clone() for k, v in state["velocity"].items()}
        self.outer_lr = float(state.get("outer_lr", self.outer_lr))
        self.momentum = float(state.get("momentum", self.momentum))
        self.max_staleness = int(state.get("max_staleness", self.max_staleness))
