"""Expert merging and tolerance-corrected conversion evaluation.

Hauzenberger et al. (2026), https://arxiv.org/abs/2603.15590, adds a merge
stage after independently linearizing experts. Community lead:
https://discord.com/channels/992359628979568762/992362722035507270/1483557410512699473
"""
from __future__ import annotations
import torch
import torch.nn.functional as F


def merge_expert_states(states: list[dict[str, torch.Tensor]], weights: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
    if not states: raise ValueError("at least one expert state is required")
    if any(set(s) != set(states[0]) for s in states): raise ValueError("expert state keys differ")
    weights = torch.ones(len(states)) / len(states) if weights is None else weights.float() / weights.sum()
    if weights.numel() != len(states): raise ValueError("one weight per expert is required")
    return {key: sum((weights[i].to(value.device) * states[i][key].float()
                      for i, value in enumerate([s[key] for s in states])), torch.zeros_like(states[0][key].float()))
            for key in states[0]}


def merge_logit_loss(merged: torch.Tensor, experts: list[torch.Tensor], temperature: float = 1.0) -> torch.Tensor:
    target = torch.stack([F.softmax(x.float() / temperature, -1) for x in experts]).mean(0)
    return F.kl_div(F.log_softmax(merged.float() / temperature, -1), target,
                    reduction="batchmean") * temperature**2


def tolerance_win_tie(student: torch.Tensor, teacher: torch.Tensor, tolerance: float = 0.01) -> dict:
    """Task scores: count student wins plus ties within an absolute tolerance."""
    if student.shape != teacher.shape: raise ValueError("score vectors must match")
    delta = student.float() - teacher.float()
    wins, ties = delta.gt(tolerance), delta.abs().le(tolerance)
    accepted = int((wins | ties).sum())
    return {"wins": int(wins.sum()), "ties": int(ties.sum()), "losses": int((delta < -tolerance).sum()),
            "win_and_tie_rate": accepted / delta.numel(), "tolerance": tolerance}
