"""PowerCool learning-rate schedule.

PowerCool is the name used for a power-law learning-rate *cooldown* reported
by OpenAI in its NanoGPT speedrun write-up.  The public report does not give
the private run's exponent or exact hyperparameters, so this module exposes
those choices instead of pretending there is a canonical recipe.

The schedule is warmup, optional constant/plateau phase, then a power-law
cooldown over the final ``cooldown_fraction`` of training::

    lr = min_lr + (peak_lr - min_lr) * (1 - p) ** power

where ``p`` runs from 0 to 1 inside the cooldown. ``power=1`` is linear;
larger values decay more aggressively, while values below one retain more
learning rate until late in the cooldown.
"""
from __future__ import annotations

import math


def powercool_lr(
    step: int,
    *,
    peak_lr: float,
    min_lr: float = 0.0,
    total_steps: int,
    warmup_steps: int = 0,
    cooldown_fraction: float = 0.20,
    power: float = 2.0,
) -> float:
    """Return the learning rate at zero-based optimizer ``step``.

    Validation is deliberately strict because a malformed cooldown silently
    changes the effective training budget and is difficult to diagnose later.
    """
    if total_steps <= 0:
        raise ValueError("total_steps must be positive")
    if step < 0:
        raise ValueError("step must be non-negative")
    if peak_lr < 0 or min_lr < 0 or min_lr > peak_lr:
        raise ValueError("require 0 <= min_lr <= peak_lr")
    if warmup_steps < 0 or warmup_steps > total_steps:
        raise ValueError("warmup_steps must be in [0, total_steps]")
    if not 0.0 < cooldown_fraction <= 1.0:
        raise ValueError("cooldown_fraction must be in (0, 1]")
    if power <= 0 or not math.isfinite(power):
        raise ValueError("power must be finite and positive")

    if warmup_steps:
        # Match the usual trainer convention: the first update gets 1/N of the
        # peak rate and the last warmup update reaches the peak rate.
        warmup = min(step + 1, warmup_steps) / warmup_steps
    else:
        warmup = 1.0

    cooldown_start = total_steps * (1.0 - cooldown_fraction)
    if step < cooldown_start:
        return peak_lr * warmup

    cooldown_span = max(total_steps * cooldown_fraction, 1.0)
    progress = min(max((step - cooldown_start) / cooldown_span, 0.0), 1.0)
    cooled = min_lr + (peak_lr - min_lr) * (1.0 - progress) ** power
    return cooled * warmup


__all__ = ["powercool_lr"]
