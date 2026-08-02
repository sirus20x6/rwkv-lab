"""Closed configuration for deterministic scalar-metric campaign decisions."""

from __future__ import annotations

import math
from dataclasses import dataclass


def _label(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or not value.isascii()
        or not 0 < len(value) <= 128
        or not value[0].isalnum()
        or any(not (character.isalnum() or character in "._-") for character in value)
    ):
        raise ValueError(f"{name} must be a bounded ASCII label")
    return value


@dataclass(frozen=True, slots=True)
class ScalarMetricDecisionConfig:
    metric: str
    direction: str
    left_subject: str
    right_subject: str
    absolute_tolerance: float = 0.0

    def __post_init__(self) -> None:
        for name in ("metric", "left_subject", "right_subject"):
            _label(getattr(self, name), name)
        if self.left_subject == self.right_subject:
            raise ValueError("decision subjects must be distinct")
        if self.direction not in {"minimize", "maximize"}:
            raise ValueError("direction must be minimize or maximize")
        if (
            isinstance(self.absolute_tolerance, bool)
            or not isinstance(self.absolute_tolerance, (int, float))
            or not math.isfinite(float(self.absolute_tolerance))
            or float(self.absolute_tolerance) < 0.0
        ):
            raise ValueError("absolute_tolerance must be finite and nonnegative")
        object.__setattr__(
            self, "absolute_tolerance", float(self.absolute_tolerance)
        )


__all__ = ["ScalarMetricDecisionConfig"]
