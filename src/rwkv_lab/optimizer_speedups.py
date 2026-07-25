"""Opt-in optimizer/readout schedules adapted from modded-nanogpt.

The helpers in this module are deliberately trainer-agnostic and checkpointable:

* :class:`TailEMA` starts late and blends (rather than replaces) weights for eval.
* embedding/head tying can be split in-place without losing optimizer moments.
* tail multipliers let expensive auxiliary objectives turn off late in training.
"""
from __future__ import annotations

import copy
from collections.abc import Iterable

import torch
from torch import nn


def tail_linear_multiplier(step: int, total_steps: int, start_fraction: float) -> float:
    """Return 1 before ``start_fraction`` and linearly decay to zero afterwards."""
    if not 0.0 <= start_fraction < 1.0:
        raise ValueError("tail schedule start fraction must be in [0, 1)")
    if total_steps <= 0:
        raise ValueError("tail schedule requires a positive step horizon")
    start = round(start_fraction * total_steps)
    if step <= start:
        return 1.0
    return max(0.0, (total_steps - step) / max(total_steps - start, 1))


def _excluded(name: str, exclude: tuple[str, ...]) -> bool:
    """Match a fragment against whole dotted path components, not substrings.

    A raw ``"emb" in name`` test also swallows ``loop_index_embed``, ``de_emb``,
    and anything else that merely contains the letters, silently dropping those
    parameters from the average.
    """
    components = name.split(".")
    return any(fragment in components for fragment in exclude)


class TailEMA:
    """Late FP32 EMA with partial eval-time interpolation.

    ``start_step`` is inclusive. Parameters whose dotted name contains an
    excluded component are left live during evaluation (the token embedding is
    normally excluded).
    """

    def __init__(
        self,
        named_parameters: Iterable[tuple[str, nn.Parameter]],
        *,
        start_step: int,
        horizon: int,
        blend: float,
        exclude: tuple[str, ...] = ("emb",),
    ) -> None:
        if start_step < 0:
            raise ValueError("TailEMA start_step must be non-negative")
        if horizon < 1:
            raise ValueError("TailEMA horizon must be positive")
        if not 0.0 <= blend <= 1.0:
            raise ValueError("TailEMA blend must be in [0, 1]")
        self.start_step = int(start_step)
        self.decay = 1.0 - 1.0 / float(horizon)
        self.blend = float(blend)
        self.exclude = tuple(exclude)
        self.named = [
            (name, param)
            for name, param in named_parameters
            if param.requires_grad and not _excluded(name, self.exclude)
        ]
        self.shadow: dict[str, torch.Tensor] = {}
        self.updates = 0

    def add_parameter(self, name: str, param: nn.Parameter) -> None:
        """Begin tracking a parameter created by scheduled model surgery."""
        if _excluded(name, self.exclude):
            return
        if any(existing == name for existing, _ in self.named):
            return
        self.named.append((name, param))
        if self.shadow:
            self.shadow[name] = param.detach().float().clone()

    @property
    def active(self) -> bool:
        return bool(self.shadow)

    @torch.no_grad()
    def update(self, step: int) -> None:
        if step < self.start_step:
            return
        if not self.shadow:
            self.shadow = {
                name: param.detach().float().clone() for name, param in self.named
            }
        else:
            # Seed any tracked parameter the shadow does not cover yet. A resumed
            # state_dict keeps only names it recognizes, so a checkpoint written
            # before a parameter existed would otherwise KeyError here instead of
            # simply starting that parameter's average now.
            for name, param in self.named:
                if name not in self.shadow:
                    self.shadow[name] = param.detach().float().clone()
            shadows = [self.shadow[name] for name, _ in self.named]
            live = [param.detach().float() for _, param in self.named]
            torch._foreach_lerp_(shadows, live, 1.0 - self.decay)
        self.updates += 1

    @torch.no_grad()
    def swap_for_eval(self) -> dict[str, torch.Tensor] | None:
        if not self.shadow:
            return None
        backup = {}
        for name, param in self.named:
            shadow = self.shadow.get(name)
            if shadow is None:
                continue  # not averaged yet; evaluate this parameter live
            backup[name] = param.detach().clone()
            blended = torch.lerp(param.detach().float(), shadow, self.blend)
            param.copy_(blended.to(param.dtype))
        return backup

    @torch.no_grad()
    def restore(self, backup: dict[str, torch.Tensor] | None) -> None:
        if backup is None:
            return
        for name, param in self.named:
            if name in backup:
                param.copy_(backup[name])

    def state_dict(self) -> dict:
        return {
            "start_step": self.start_step,
            "decay": self.decay,
            "blend": self.blend,
            "updates": self.updates,
            "shadow": self.shadow,
        }

    def load_state_dict(self, state: dict) -> None:
        self.updates = int(state.get("updates", 0))
        source = state.get("shadow") or {}
        devices = {name: param.device for name, param in self.named}
        self.shadow = {
            name: value.float().to(devices[name])
            for name, value in source.items()
            if name in devices
        }


def tie_embedding_head(model: nn.Module) -> None:
    """Make ``model.head.weight`` and ``model.emb.weight`` one Parameter."""
    embedding = getattr(getattr(model, "emb", None), "weight", None)
    head = getattr(model, "head", None)
    if embedding is None or not isinstance(head, nn.Linear):
        raise ValueError("head tying requires model.emb.weight and an nn.Linear model.head")
    if head.weight.shape != embedding.shape:
        raise ValueError("embedding and head weights must have identical shapes")
    head.weight = embedding


def split_tied_embedding_head(model: nn.Module, optimizer: torch.optim.Optimizer) -> nn.Parameter:
    """Untie the output head and clone the embedding optimizer state.

    The new head is added to the same optimizer group as the shared embedding.
    Tensor moments are cloned, so the two parameters begin with identical
    optimizer trajectories and then diverge normally.
    """
    embedding = getattr(getattr(model, "emb", None), "weight", None)
    head = getattr(model, "head", None)
    if embedding is None or not isinstance(head, nn.Linear):
        raise ValueError("head splitting requires model.emb.weight and an nn.Linear model.head")
    if head.weight is not embedding:
        raise ValueError("embedding and head are not tied")

    source_group = next(
        (
            group
            for group in optimizer.param_groups
            if any(param is embedding for param in group["params"])
        ),
        None,
    )
    if source_group is None:
        raise ValueError("tied embedding is absent from the optimizer")
    new_head = nn.Parameter(embedding.detach().clone(), requires_grad=True)
    head.weight = new_head
    # Append instead of creating a new group. This keeps checkpoint group
    # topology identical to a trainer resumed after the scheduled split.
    source_group["params"].append(new_head)
    if embedding in optimizer.state:
        optimizer.state[new_head] = {
            key: (value.detach().clone() if torch.is_tensor(value) else copy.deepcopy(value))
            for key, value in optimizer.state[embedding].items()
        }
    return new_head


__all__ = [
    "TailEMA",
    "split_tied_embedding_head",
    "tail_linear_multiplier",
    "tie_embedding_head",
]
