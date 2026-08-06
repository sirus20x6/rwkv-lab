from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any

from .resolved import resolved_component_parts


class EvaluationScheduleImplementation(str, Enum):
    LAUNCH_GATE_PERIODIC_V1 = "rwkv_lab.evaluation_schedule.launch_gate_periodic.v1"
    LAUNCH_GATE_PERIODIC_V2 = "rwkv_lab.evaluation_schedule.launch_gate_periodic.v2"


@dataclass(frozen=True, slots=True)
class EvaluationScheduleConfiguration:
    full_step_zero: bool = True
    qualitative_every_steps: int = 0
    full_every_steps: int = 0
    defer_full_scalar: bool = True
    final: bool = True
    launch_gate_examples: int | None = None

    def __post_init__(self) -> None:
        if self.launch_gate_examples is not None and (
            not isinstance(self.launch_gate_examples, int)
            or isinstance(self.launch_gate_examples, bool)
            or not 1 <= self.launch_gate_examples <= 1_000_000
        ):
            raise ValueError("launch_gate_examples must be a positive bounded integer")
        for value, label in (
            (self.qualitative_every_steps, "qualitative_every_steps"),
            (self.full_every_steps, "full_every_steps"),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{label} must be a nonnegative integer")
        if not self.full_step_zero:
            raise ValueError("the true full scalar step-zero baseline is mandatory")

    @classmethod
    def from_resolved_v1(
        cls, configuration: Mapping[str, Any]
    ) -> EvaluationScheduleConfiguration:
        expected = {
            "launch_gate_examples",
            "full_step_zero",
            "qualitative_every_steps",
            "full_every_steps",
            "defer_full_scalar",
            "final",
        }
        if set(configuration) != expected:
            raise ValueError("resolved evaluation schedule configuration is inexact")
        return cls(**configuration)

    @classmethod
    def from_resolved_v2(
        cls, configuration: Mapping[str, Any]
    ) -> EvaluationScheduleConfiguration:
        expected = {
            "full_step_zero",
            "qualitative_every_steps",
            "full_every_steps",
            "final",
        }
        if set(configuration) != expected:
            raise ValueError("resolved evaluation schedule v2 configuration is inexact")
        return cls(defer_full_scalar=False, **configuration)


@dataclass(frozen=True, slots=True)
class EvaluationDecision:
    qualitative: bool
    full_scalar: bool
    launch_gate: bool
    defer_full_scalar: bool


@dataclass(frozen=True, slots=True)
class EvaluationSchedule:
    configuration: EvaluationScheduleConfiguration

    def for_step(self, step: int, *, final: bool = False) -> EvaluationDecision:
        if not isinstance(step, int) or isinstance(step, bool) or step < 0:
            raise ValueError("evaluation step must be a nonnegative integer")
        if step == 0:
            return EvaluationDecision(
                True, True, True, self.configuration.defer_full_scalar
            )
        qualitative = self.configuration.qualitative_every_steps > 0 and (
            step % self.configuration.qualitative_every_steps == 0
        )
        full_scalar = self.configuration.full_every_steps > 0 and (
            step % self.configuration.full_every_steps == 0
        )
        if final and self.configuration.final:
            qualitative = True
            full_scalar = True
        return EvaluationDecision(
            qualitative, full_scalar, False, self.configuration.defer_full_scalar
        )


def evaluation_schedule_from_resolved_component(
    component: Mapping[str, Any],
) -> EvaluationSchedule:
    implementation, configuration = resolved_component_parts(
        component, "evaluation_schedule"
    )
    if implementation == EvaluationScheduleImplementation.LAUNCH_GATE_PERIODIC_V1:
        configuration = EvaluationScheduleConfiguration.from_resolved_v1(
            configuration
        )
    elif implementation == EvaluationScheduleImplementation.LAUNCH_GATE_PERIODIC_V2:
        configuration = EvaluationScheduleConfiguration.from_resolved_v2(
            configuration
        )
    else:
        raise ValueError(
            "resolved evaluation schedule implementation is not allowlisted"
        )
    return EvaluationSchedule(configuration)


__all__ = [
    "EvaluationDecision",
    "EvaluationSchedule",
    "EvaluationScheduleConfiguration",
    "EvaluationScheduleImplementation",
    "evaluation_schedule_from_resolved_component",
]
