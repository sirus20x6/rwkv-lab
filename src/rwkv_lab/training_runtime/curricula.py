from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from itertools import pairwise
from typing import Any

from .resolved import resolved_component_parts


class CurriculumImplementation(str, Enum):
    CONTEXT_LENGTH_V1 = "rwkv_lab.curriculum.context_length.v1"


@dataclass(frozen=True, slots=True)
class ContextStage:
    start_fraction: float
    sequence_length: int

    @property
    def seq_len(self) -> int:
        """Compatibility spelling used by the legacy RWKV loop."""
        return self.sequence_length


def parse_context_stages(
    specification: str, *, maximum_sequence_length: int
) -> tuple[ContextStage, ...]:
    if not isinstance(specification, str):
        raise TypeError("context curriculum stages must be a string")
    if not specification:
        return ()
    stages: list[ContextStage] = []
    for item in specification.split(","):
        try:
            fraction_text, length_text = item.strip().split(":", 1)
            stage = ContextStage(float(fraction_text), int(length_text))
        except (TypeError, ValueError) as error:
            raise ValueError(
                "context curriculum must be comma-separated fraction:length stages"
            ) from error
        if not math.isfinite(stage.start_fraction) or not (
            0.0 <= stage.start_fraction < 1.0
        ):
            raise ValueError("context-curriculum fractions must be finite in [0, 1)")
        if not 0 < stage.sequence_length <= maximum_sequence_length:
            raise ValueError(
                "context-curriculum lengths must be in "
                f"[1, {maximum_sequence_length}]"
            )
        stages.append(stage)
    if stages[0].start_fraction != 0.0:
        raise ValueError("context curriculum must start at fraction 0")
    if any(
        left.start_fraction >= right.start_fraction
        for left, right in pairwise(stages)
    ):
        raise ValueError("context-curriculum fractions must be strictly increasing")
    if any(
        left.sequence_length > right.sequence_length
        for left, right in pairwise(stages)
    ):
        raise ValueError(
            "context-curriculum sequence lengths must be non-decreasing"
        )
    return tuple(stages)


@dataclass(frozen=True, slots=True)
class ContextLengthCurriculumConfiguration:
    maximum_sequence_length: int
    base_batch_size: int
    stages: str = ""

    def __post_init__(self) -> None:
        for value, label in (
            (self.maximum_sequence_length, "maximum_sequence_length"),
            (self.base_batch_size, "base_batch_size"),
        ):
            if (
                not isinstance(value, int)
                or isinstance(value, bool)
                or not 1 <= value <= 1_048_576
            ):
                raise ValueError(f"{label} must be an integer in [1, 1048576]")
        parse_context_stages(
            self.stages, maximum_sequence_length=self.maximum_sequence_length
        )

    @classmethod
    def from_resolved(
        cls, configuration: Mapping[str, Any]
    ) -> ContextLengthCurriculumConfiguration:
        if set(configuration) != {
            "maximum_sequence_length",
            "base_batch_size",
            "stages",
        }:
            raise ValueError(
                "resolved context curriculum has missing or unknown fields"
            )
        return cls(**configuration)


class ContextLengthCurriculum:
    """Derive context and batch from the optimizer-step cursor and token budget."""

    def __init__(self, configuration: ContextLengthCurriculumConfiguration) -> None:
        self.configuration = configuration
        self.stages = parse_context_stages(
            configuration.stages,
            maximum_sequence_length=configuration.maximum_sequence_length,
        )

    def for_step(self, step: int, total_steps: int) -> tuple[int, int]:
        return context_batch_for_stages(
            self.stages,
            step=step,
            total_steps=total_steps,
            maximum_sequence_length=self.configuration.maximum_sequence_length,
            base_batch_size=self.configuration.base_batch_size,
        )

    def state_dict(self) -> dict[str, object]:
        return {}

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        if state:
            raise ValueError("optimizer-step-derived curriculum state must be empty")


def context_batch_for_stages(
    stages: tuple[ContextStage, ...],
    *,
    step: int,
    total_steps: int,
    maximum_sequence_length: int,
    base_batch_size: int,
) -> tuple[int, int]:
    """Evaluate a context schedule while holding the base token budget constant."""
    configuration = ContextLengthCurriculumConfiguration(
        maximum_sequence_length=maximum_sequence_length,
        base_batch_size=base_batch_size,
    )
    if not isinstance(stages, tuple) or not all(
        isinstance(stage, ContextStage) for stage in stages
    ):
        raise TypeError("context curriculum stages must be a tuple of ContextStage")
    if not isinstance(step, int) or isinstance(step, bool) or step < 0:
        raise ValueError("curriculum step must be a non-negative integer")
    if (
        not isinstance(total_steps, int)
        or isinstance(total_steps, bool)
        or total_steps <= 0
    ):
        raise ValueError("context curriculum requires a positive step horizon")
    if not stages:
        return (
            configuration.maximum_sequence_length,
            configuration.base_batch_size,
        )
    progress = min(step / total_steps, 1.0)
    active = stages[0]
    for stage in stages[1:]:
        if progress < stage.start_fraction:
            break
        active = stage
    token_budget = (
        configuration.maximum_sequence_length * configuration.base_batch_size
    )
    return active.sequence_length, max(1, round(token_budget / active.sequence_length))


def build_registered_curriculum(
    implementation: CurriculumImplementation,
    configuration: ContextLengthCurriculumConfiguration,
) -> ContextLengthCurriculum:
    if implementation is not CurriculumImplementation.CONTEXT_LENGTH_V1:
        raise ValueError(f"unsupported curriculum implementation: {implementation!r}")
    if not isinstance(configuration, ContextLengthCurriculumConfiguration):
        raise TypeError("context-length curriculum requires its typed configuration")
    return ContextLengthCurriculum(configuration)


def curriculum_from_resolved_component(
    component: Mapping[str, Any],
) -> ContextLengthCurriculum:
    implementation, configuration = resolved_component_parts(component, "curriculum")
    try:
        selected = CurriculumImplementation(implementation)
    except ValueError as error:
        raise ValueError(
            "resolved curriculum implementation is not allowlisted"
        ) from error
    return build_registered_curriculum(
        selected, ContextLengthCurriculumConfiguration.from_resolved(configuration)
    )


__all__ = [
    "ContextLengthCurriculum",
    "ContextLengthCurriculumConfiguration",
    "ContextStage",
    "CurriculumImplementation",
    "build_registered_curriculum",
    "context_batch_for_stages",
    "curriculum_from_resolved_component",
    "parse_context_stages",
]
