"""Key-Value Means compressed block-recurrent memory oracle.

Goldstein & Cheah (2026), https://arxiv.org/abs/2605.09877 and official code
https://github.com/recursal/KVM-paper. Community lead:
https://discord.com/channels/992359628979568762/992362722035507270/1503948385940672553
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import torch
import torch.nn.functional as F


@dataclass
class KVMState:
    keys: torch.Tensor
    values: torch.Tensor
    radii: torch.Tensor


def state_budget(step: int, initial: int, mode: str = "fixed", maximum: int | None = None) -> int:
    if mode == "fixed": value = initial
    elif mode == "sqrt": value = initial + int(math.sqrt(max(step, 0)))
    elif mode == "saturating": value = initial + int(math.sqrt(max(step, 0)))
    else: raise ValueError("mode must be fixed, sqrt, or saturating")
    return min(value, maximum) if maximum is not None else value


def initialize_state(keys: torch.Tensor, values: torch.Tensor) -> KVMState:
    radii = values.float().norm(dim=-1).clamp_min(1e-6)
    return KVMState(keys.clone(), values.clone(), radii)


def normalized_state(state: KVMState) -> tuple[torch.Tensor, torch.Tensor]:
    keys = F.normalize(state.keys.float(), dim=-1).to(state.keys.dtype)
    values = F.normalize(state.values.float(), dim=-1).to(state.values.dtype)
    return keys, values * state.radii.to(values.dtype).unsqueeze(-1)


def update_state(state: KVMState, keys: torch.Tensor, values: torch.Tensor, *,
                 budget: int, sink_rows: int = 0, merge_weights: torch.Tensor | None = None) -> KVMState:
    """Append least-redundant overflow rows, then winner-take-all merge the rest."""
    sk, _ = normalized_state(state)
    similarity = F.normalize(keys.float(), dim=-1) @ sk.float().T
    novelty = similarity.max(-1).values
    append_n = max(0, min(budget - state.keys.shape[0], keys.shape[0]))
    order = novelty.argsort()
    append_idx, merge_idx = order[:append_n], order[append_n:]
    out_k = torch.cat((state.keys, keys[append_idx]), 0)
    out_v = torch.cat((state.values, values[append_idx]), 0)
    out_r = torch.cat((state.radii, values[append_idx].float().norm(dim=-1).clamp_min(1e-6)), 0)
    if merge_idx.numel():
        target_keys = F.normalize(out_k.float(), dim=-1)
        scores = F.normalize(keys[merge_idx].float(), dim=-1) @ target_keys.T
        if sink_rows:
            scores[:, :sink_rows] = -torch.inf
        targets = scores.argmax(-1)
        weights = (torch.ones(keys.shape[0], device=keys.device, dtype=keys.dtype)
                   if merge_weights is None else merge_weights)
        out_k = out_k.index_add(0, targets, keys[merge_idx] * weights[merge_idx, None])
        out_v = out_v.index_add(0, targets, values[merge_idx] * weights[merge_idx, None])
    return KVMState(out_k, out_v, out_r)


def read_memory(query: torch.Tensor, state: KVMState, window_k: torch.Tensor | None = None,
                window_v: torch.Tensor | None = None, temperature: float = 1.0) -> torch.Tensor:
    sk, sv = normalized_state(state)
    keys = sk if window_k is None else torch.cat((sk, window_k), -2)
    values = sv if window_v is None else torch.cat((sv, window_v), -2)
    weights = F.softmax((query.float() @ keys.float().T) / temperature, -1).to(values.dtype)
    return weights @ values
