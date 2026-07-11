"""Learned routing over reusable RWKV recurrent states.

This is a correctness-first implementation of two RWKV community proposals:
dynamic selection among tuned initial states
https://discord.com/channels/992359628979568762/992359629419991142/1458334147058733106
and retrieval/training of document-derived initial states
https://discord.com/channels/992359628979568762/992372861924823080/1103114894439628871.
These are community proposals, not paper-backed performance claims.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn


@dataclass(frozen=True)
class StateLeaf:
    path: tuple[Any, ...]
    shape: tuple[int, ...]
    start: int
    stop: int


def _leaves(tree: Any, path=()):
    if torch.is_tensor(tree):
        yield path, tree
    elif isinstance(tree, dict):
        for key in sorted(tree):
            yield from _leaves(tree[key], path + (key,))
    elif isinstance(tree, (list, tuple)):
        for index, value in enumerate(tree):
            yield from _leaves(value, path + (index,))
    else:
        raise TypeError(f"state bank supports tensor/dict/list trees, got {type(tree).__name__}")


def flatten_state(tree: Any) -> tuple[torch.Tensor, tuple[StateLeaf, ...]]:
    flat = []; schema = []; offset = 0
    for path, tensor in _leaves(tree):
        # Stored states represent one request; strip its singleton batch axis.
        value = tensor[0] if tensor.ndim and tensor.shape[0] == 1 else tensor
        row = value.detach().float().reshape(-1)
        flat.append(row); schema.append(StateLeaf(path, tuple(value.shape), offset, offset + row.numel()))
        offset += row.numel()
    return torch.cat(flat), tuple(schema)


def _assign(root, path, value):
    target = root
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value


def unflatten_state(flat: torch.Tensor, schema: tuple[StateLeaf, ...]) -> Any:
    """Rebuild a batched recurrent-state tree from ``[batch,width]`` rows."""
    if flat.ndim != 2:
        raise ValueError("flat routed state must have shape [batch,width]")
    paths = [leaf.path for leaf in schema]
    if not paths:
        return []
    root = [] if isinstance(paths[0][0], int) else {}
    for leaf in schema:
        target = root
        for depth, key in enumerate(leaf.path):
            last = depth == len(leaf.path) - 1
            if last:
                value = flat[:, leaf.start:leaf.stop].reshape(flat.shape[0], *leaf.shape)
                if isinstance(target, list):
                    while len(target) <= key: target.append(None)
                    target[key] = value
                else: target[key] = value
                continue
            next_key = leaf.path[depth + 1]
            child = [] if isinstance(next_key, int) else {}
            if isinstance(target, list):
                while len(target) <= key: target.append(None)
                if target[key] is None: target[key] = child
                target = target[key]
            else:
                target = target.setdefault(key, child)
    return root


class RoutedStateBank(nn.Module):
    """Soft or hard query routing over trainable, constant-size state slots."""
    def __init__(self, example_state: Any, *, query_dim: int, slots: int = 8,
                 trainable_states: bool = True, hyper_rank: int = 0):
        super().__init__()
        if query_dim < 1 or slots < 1:
            raise ValueError("query_dim and slots must be positive")
        initial, self.schema = flatten_state(example_state)
        states = initial.repeat(slots, 1)
        self.states = nn.Parameter(states, requires_grad=trainable_states)
        self.router = nn.Linear(query_dim, slots)
        nn.init.zeros_(self.router.weight); nn.init.zeros_(self.router.bias)
        self.hyper = (nn.Sequential(nn.Linear(query_dim, hyper_rank), nn.Tanh(),
                                    nn.Linear(hyper_rank, initial.numel())) if hyper_rank else None)
        if self.hyper is not None:
            nn.init.zeros_(self.hyper[-1].weight); nn.init.zeros_(self.hyper[-1].bias)

    def route(self, query: torch.Tensor, *, hard: bool = False,
              temperature: float = 1.0) -> tuple[Any, dict[str, torch.Tensor]]:
        if query.ndim != 2 or temperature <= 0:
            raise ValueError("query must be [batch,dim] and temperature positive")
        weights = torch.softmax(self.router(query.float()) / temperature, -1)
        if hard:
            index = weights.argmax(-1)
            selected = self.states[index]
        else:
            index = weights.argmax(-1)
            selected = weights @ self.states
        if self.hyper is not None:
            selected = selected + self.hyper(query.float())
        entropy = -(weights * weights.clamp_min(1e-12).log()).sum(-1)
        stats = {"weights": weights, "selected": index,
                 "entropy": entropy, "collapse": weights.max(-1).values}
        return unflatten_state(selected, self.schema), stats

    @torch.no_grad()
    def seed_slot(self, slot: int, state: Any) -> None:
        value, schema = flatten_state(state)
        if schema != self.schema or value.numel() != self.states.shape[1]:
            raise ValueError("state slot geometry mismatch")
        self.states[slot].copy_(value.to(self.states))
