"""Test-time learned associative memory for recurrent language models.

Primary references:

* Behrouz et al., "Titans: Learning to Memorize at Test Time",
  arXiv:2501.00663, https://arxiv.org/abs/2501.00663.
* Behrouz et al., "It's All Connected" (the MIRAS design framework),
  arXiv:2504.13173, https://arxiv.org/abs/2504.13173.
* Behrouz et al., "ATLAS: Learning to Optimally Memorize the Context at Test
  Time", arXiv:2505.23735, https://arxiv.org/abs/2505.23735.
* Behrouz et al., "Nested Learning: The Illusion of Deep Learning
  Architectures", arXiv:2512.24695, https://arxiv.org/abs/2512.24695.

The module implements their shared model-side mechanism: an associative-memory
matrix is optimized *inside the forward pass*.  MIRAS choices are explicit:
the memory model, attentional-bias loss, retention rule, and learning rule.
``atlas`` mode adds a short window of past key/value pairs to the update rather
than optimizing only against the current token. ``nested`` mode adds a learned
controller that adjusts update rate and retention from the current surprise.

This CPU-readable path uses an exact sequential scan. It is intentionally a
reference implementation for A/B validation before a chunk-parallel kernel.
"""
from __future__ import annotations

from dataclasses import dataclass
from collections import deque
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class OnlineMemoryState:
    memory: torch.Tensor
    momentum: torch.Tensor
    steps: int = 0

    def detach(self) -> "OnlineMemoryState":
        return OnlineMemoryState(self.memory.detach(), self.momentum.detach(), self.steps)


class OnlineAssociativeMemory(nn.Module):
    """Differentiable in-forward memory update with Titans/MIRAS/ATLAS modes."""

    MODES = ("titans", "miras", "atlas", "nested")

    def __init__(self, d_model: int, *, d_memory: int | None = None, mode: str = "titans",
                 learning_rate: float = 0.05, retention: float = 0.99,
                 momentum: float = 0.9, atlas_window: int = 4, detach_updates: bool = False):
        super().__init__()
        if mode not in self.MODES:
            raise ValueError(f"mode must be one of {self.MODES}, got {mode!r}")
        self.d_model = int(d_model)
        self.d_memory = int(d_memory or min(d_model, 128))
        self.mode = mode
        self.learning_rate = float(learning_rate)
        self.retention = float(retention)
        self.momentum_beta = float(momentum)
        self.atlas_window = max(1, int(atlas_window))
        self.detach_updates = bool(detach_updates)

        self.norm = nn.RMSNorm(d_model)
        self.q_proj = nn.Linear(d_model, self.d_memory, bias=False)
        self.k_proj = nn.Linear(d_model, self.d_memory, bias=False)
        self.v_proj = nn.Linear(d_model, self.d_memory, bias=False)
        self.out_proj = nn.Linear(self.d_memory, d_model, bias=False)
        self.gate = nn.Linear(d_model, 1)
        # Identity-at-init makes this a clean off-by-default lever.
        nn.init.zeros_(self.out_proj.weight)
        nn.init.constant_(self.gate.bias, -4.0)
        self.controller = nn.Sequential(nn.Linear(1, 8), nn.SiLU(), nn.Linear(8, 2)) \
            if mode == "nested" else None
        if self.controller is not None:
            nn.init.zeros_(self.controller[-1].weight)
            nn.init.zeros_(self.controller[-1].bias)
        self.last_stats: dict[str, float] = {}

    def initial_state(self, batch: int, *, device=None, dtype=torch.float32) -> OnlineMemoryState:
        z = torch.zeros(batch, self.d_memory, self.d_memory, device=device, dtype=dtype)
        return OnlineMemoryState(z, torch.zeros_like(z), 0)

    def _bias_error(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # MIRAS treats the internal objective as an architectural choice.
        if self.mode == "miras":
            pn, tn = F.normalize(pred, dim=-1), F.normalize(target, dim=-1)
            return tn - pn
        return target - pred

    def forward(self, hidden: torch.Tensor, state: OnlineMemoryState | None = None,
                *, return_state: bool = False):
        if hidden.ndim != 3:
            raise ValueError("hidden must have shape [batch,time,channels]")
        B, T, _ = hidden.shape
        x = self.norm(hidden)
        q = F.normalize(self.q_proj(x).float(), dim=-1)
        k = F.normalize(self.k_proj(x).float(), dim=-1)
        v = self.v_proj(x).float()
        st = state or self.initial_state(B, device=hidden.device)
        memory, mom = st.memory.float(), st.momentum.float()
        if memory.shape != (B, self.d_memory, self.d_memory):
            raise ValueError("online-memory state geometry does not match this module")

        outputs, surprises = [], []
        history: deque[tuple[torch.Tensor, torch.Tensor]] = deque(maxlen=self.atlas_window)
        for t in range(T):
            qt, kt, vt = q[:, t], k[:, t], v[:, t]
            pred = torch.einsum("bi,bij->bj", qt, memory)
            err = self._bias_error(pred, vt)
            surprise = err.square().mean(-1, keepdim=True).sqrt()
            outputs.append(pred)
            surprises.append(surprise)

            history.append((kt, vt))
            pairs = list(history) if self.mode == "atlas" else [(kt, vt)]
            grad = torch.zeros_like(memory)
            for kh, vh in pairs:
                ph = torch.einsum("bi,bij->bj", kh, memory)
                eh = self._bias_error(ph, vh)
                grad = grad + torch.einsum("bi,bj->bij", kh, eh)
            grad = grad / len(pairs)

            eta = hidden.new_full((B, 1, 1), self.learning_rate, dtype=torch.float32)
            retain = hidden.new_full((B, 1, 1), self.retention, dtype=torch.float32)
            if self.controller is not None:
                ctrl = self.controller(surprise.to(hidden.dtype)).float()
                eta = eta * (2.0 * torch.sigmoid(ctrl[:, :1, None]))
                retain = 1.0 - (1.0 - retain) * (2.0 * torch.sigmoid(ctrl[:, 1:, None]))
            mom = self.momentum_beta * mom + (1.0 - self.momentum_beta) * grad
            memory = retain * memory + eta * mom
            if self.detach_updates:
                memory, mom = memory.detach(), mom.detach()

        recalled = torch.stack(outputs, dim=1).to(hidden.dtype)
        gate = torch.sigmoid(self.gate(x))
        out = hidden + gate * self.out_proj(recalled)
        ss = torch.cat(surprises, dim=1)
        self.last_stats = {
            "surprise_mean": float(ss.detach().mean()),
            "memory_norm": float(memory.detach().norm(dim=(-2, -1)).mean()),
            "gate_mean": float(gate.detach().mean()),
        }
        next_state = OnlineMemoryState(memory, mom, st.steps + T)
        return (out, next_state) if return_state else out


def install_online_memory(model: nn.Module, **kwargs) -> OnlineAssociativeMemory:
    """Register a final-hidden online memory on an RWKV-Lab model."""

    d_model = int(getattr(getattr(model, "head", None), "in_features", 0))
    if not d_model:
        raise ValueError("model must expose head.in_features")
    mem = OnlineAssociativeMemory(d_model, **kwargs)
    p = next(model.parameters())
    mem.to(device=p.device, dtype=p.dtype)
    model.online_memory = mem
    return mem
