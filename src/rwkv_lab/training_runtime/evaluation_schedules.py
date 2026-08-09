from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from enum import Enum
from functools import lru_cache
from typing import Any

from .resolved import resolved_component_parts


class EvaluationScheduleImplementation(str, Enum):
    LAUNCH_GATE_PERIODIC_V1 = "rwkv_lab.evaluation_schedule.launch_gate_periodic.v1"
    LAUNCH_GATE_PERIODIC_V2 = "rwkv_lab.evaluation_schedule.launch_gate_periodic.v2"
    MILESTONE_CADENCE_V3 = "rwkv_lab.evaluation_schedule.milestone_cadence.v3"


class EvaluationKind(str, Enum):
    """The distinct evaluation milestones a run can publish.

    They are separate kinds rather than one "evaluation" because they cost
    wildly different amounts and answer different questions: a probe is a
    bounded scalar read taken often, a full scalar read covers the whole frozen
    split, a qualitative milestone runs autoregressive generation, and the
    final audit is the one-shot terminal evidence bundle.
    """

    SCALAR_PROBE = "scalar_probe"
    SCALAR_FULL = "scalar_full"
    QUALITATIVE = "qualitative"
    FINAL_AUDIT = "final_audit"


# Only these control keys may be patched live. Everything that decides *which*
# examples an evaluation reads is deliberately absent: a cadence patch changes
# when evidence is taken, never what the evidence is taken over.
EVALUATION_CADENCE_CONTROLS: tuple[str, ...] = (
    "evaluation.full_every_steps",
    "evaluation.probe_every_steps",
    "evaluation.qualitative_every_steps",
)

_MAXIMUM_MILESTONES = 256
_MAXIMUM_STEP = 1_000_000_000
_MAXIMUM_PLANNED_MILESTONES = 100_000


def _milestone_steps(values: Iterable[Any], label: str) -> tuple[int, ...]:
    steps: list[int] = []
    for value in values:
        if not isinstance(value, str) or not value.isascii() or not value.isdigit():
            raise ValueError(f"{label} entries must be decimal step literals")
        if value != str(int(value)):
            raise ValueError(f"{label} entries must be canonical decimal literals")
        step = int(value)
        if not 1 <= step <= _MAXIMUM_STEP:
            raise ValueError(f"{label} entries must be positive bounded steps")
        steps.append(step)
    if len(steps) > _MAXIMUM_MILESTONES:
        raise ValueError(f"{label} declares too many milestones")
    if len(set(steps)) != len(steps):
        raise ValueError(f"{label} entries must be unique")
    return tuple(sorted(steps))


def _milestone_fractions(values: Iterable[Any], label: str) -> tuple[str, ...]:
    fractions: list[Decimal] = []
    for value in values:
        if not isinstance(value, str) or not value.isascii():
            raise ValueError(f"{label} entries must be decimal fraction literals")
        try:
            fraction = Decimal(value)
        except InvalidOperation as error:
            raise ValueError(
                f"{label} entries must be decimal fraction literals"
            ) from error
        if str(fraction) != value or not fraction.is_finite():
            raise ValueError(f"{label} entries must be canonical decimal literals")
        if not Decimal(0) < fraction <= Decimal(1):
            raise ValueError(f"{label} entries must lie in (0, 1]")
        if -fraction.as_tuple().exponent > 6:
            raise ValueError(f"{label} entries must declare at most six decimals")
        fractions.append(fraction)
    if len(fractions) > _MAXIMUM_MILESTONES:
        raise ValueError(f"{label} declares too many milestones")
    if len(set(fractions)) != len(fractions):
        raise ValueError(f"{label} entries must be unique")
    return tuple(str(fraction) for fraction in sorted(fractions))


def _resolved_steps(
    fractions: Sequence[str], total_steps: int
) -> tuple[int, ...]:
    resolved: set[int] = set()
    for literal in fractions:
        scaled = (Decimal(literal) * Decimal(total_steps)).to_integral_value(
            rounding=ROUND_HALF_UP
        )
        resolved.add(min(max(int(scaled), 1), total_steps))
    return tuple(sorted(resolved))


@dataclass(frozen=True, slots=True)
class EvaluationScheduleConfiguration:
    full_step_zero: bool = True
    qualitative_every_steps: int = 0
    full_every_steps: int = 0
    defer_full_scalar: bool = True
    final: bool = True
    launch_gate_examples: int | None = None
    probe_every_steps: int = 0
    full_milestone_steps: tuple[int, ...] = ()
    probe_milestone_steps: tuple[int, ...] = ()
    qualitative_milestone_steps: tuple[int, ...] = ()
    full_milestone_fractions: tuple[str, ...] = ()
    probe_milestone_fractions: tuple[str, ...] = ()
    qualitative_milestone_fractions: tuple[str, ...] = ()
    probe_examples: int = 0
    mutable_cadence: bool = False

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
            (self.probe_every_steps, "probe_every_steps"),
            (self.probe_examples, "probe_examples"),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{label} must be a nonnegative integer")
        if self.probe_examples > 1_000_000_000:
            raise ValueError("probe_examples must be a nonnegative integer")
        for values, label in (
            (self.full_milestone_steps, "full_milestone_steps"),
            (self.probe_milestone_steps, "probe_milestone_steps"),
            (self.qualitative_milestone_steps, "qualitative_milestone_steps"),
        ):
            if tuple(values) != _milestone_steps(
                [str(value) for value in values], label
            ):
                raise ValueError(f"{label} must be a canonical ascending step set")
        for values, label in (
            (self.full_milestone_fractions, "full_milestone_fractions"),
            (self.probe_milestone_fractions, "probe_milestone_fractions"),
            (
                self.qualitative_milestone_fractions,
                "qualitative_milestone_fractions",
            ),
        ):
            if tuple(values) != _milestone_fractions(values, label):
                raise ValueError(f"{label} must be a canonical ascending fraction set")
        if not self.full_step_zero:
            raise ValueError("the true full scalar step-zero baseline is mandatory")

    @property
    def example_identity(self) -> tuple[Any, ...]:
        """Everything in this component that decides which examples are read.

        `with_cadence_assignments` refuses to move any of it, so a live cadence
        patch is structurally incapable of changing the evaluated subset. Tests
        assert this identity is invariant across cadence mutation rather than
        taking the claim on trust.
        """

        return (self.full_step_zero, self.final, self.probe_examples)

    def with_cadence_assignments(
        self, assignments: Mapping[str, Any]
    ) -> EvaluationScheduleConfiguration:
        if not assignments:
            return self
        if not self.mutable_cadence:
            raise ValueError("this evaluation schedule declares no live cadence")
        updates: dict[str, int] = {}
        for key, value in assignments.items():
            if key not in EVALUATION_CADENCE_CONTROLS:
                raise ValueError("only declared evaluation cadence keys are live")
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError("evaluation cadence must be a nonnegative integer")
            updates[key.removeprefix("evaluation.")] = value
        patched = replace(self, **updates)
        if patched.example_identity != self.example_identity:
            raise ValueError("a cadence patch may not move evaluation example identity")
        return patched

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

    @classmethod
    def from_resolved_v3(
        cls, configuration: Mapping[str, Any]
    ) -> EvaluationScheduleConfiguration:
        expected = {
            "full_step_zero",
            "final",
            "full_every_steps",
            "probe_every_steps",
            "qualitative_every_steps",
            "full_milestone_steps",
            "probe_milestone_steps",
            "qualitative_milestone_steps",
            "full_milestone_fractions",
            "probe_milestone_fractions",
            "qualitative_milestone_fractions",
            "probe_examples",
            "mutable_cadence",
        }
        if set(configuration) != expected:
            raise ValueError("resolved evaluation schedule v3 configuration is inexact")
        return cls(
            defer_full_scalar=False,
            full_step_zero=configuration["full_step_zero"],
            final=configuration["final"],
            full_every_steps=configuration["full_every_steps"],
            probe_every_steps=configuration["probe_every_steps"],
            qualitative_every_steps=configuration["qualitative_every_steps"],
            full_milestone_steps=_milestone_steps(
                configuration["full_milestone_steps"], "full_milestone_steps"
            ),
            probe_milestone_steps=_milestone_steps(
                configuration["probe_milestone_steps"], "probe_milestone_steps"
            ),
            qualitative_milestone_steps=_milestone_steps(
                configuration["qualitative_milestone_steps"],
                "qualitative_milestone_steps",
            ),
            full_milestone_fractions=_milestone_fractions(
                configuration["full_milestone_fractions"], "full_milestone_fractions"
            ),
            probe_milestone_fractions=_milestone_fractions(
                configuration["probe_milestone_fractions"], "probe_milestone_fractions"
            ),
            qualitative_milestone_fractions=_milestone_fractions(
                configuration["qualitative_milestone_fractions"],
                "qualitative_milestone_fractions",
            ),
            probe_examples=configuration["probe_examples"],
            mutable_cadence=configuration["mutable_cadence"],
        )


@dataclass(frozen=True, slots=True)
class EvaluationDecision:
    qualitative: bool
    full_scalar: bool
    launch_gate: bool
    defer_full_scalar: bool
    probe: bool = False
    final_audit: bool = False

    @property
    def kinds(self) -> tuple[EvaluationKind, ...]:
        selected: list[EvaluationKind] = []
        if self.probe:
            selected.append(EvaluationKind.SCALAR_PROBE)
        if self.full_scalar:
            selected.append(EvaluationKind.SCALAR_FULL)
        if self.qualitative:
            selected.append(EvaluationKind.QUALITATIVE)
        if self.final_audit:
            selected.append(EvaluationKind.FINAL_AUDIT)
        return tuple(selected)


@dataclass(frozen=True, slots=True)
class EvaluationMilestone:
    step: int
    kinds: tuple[EvaluationKind, ...]


@dataclass(frozen=True, slots=True)
class EvaluationPlan:
    """The whole declared milestone timeline, resolved against a run length.

    Producing it up front is what makes cadence cost visible before a run
    starts burning GPU hours on it, and it is what the engine publishes so the
    dashboard can draw milestones that have not happened yet.
    """

    total_steps: int
    milestones: tuple[EvaluationMilestone, ...]
    probe_examples: int

    def kinds_at(self, step: int) -> tuple[EvaluationKind, ...]:
        for milestone in self.milestones:
            if milestone.step == step:
                return milestone.kinds
        return ()

    def steps_for(self, kind: EvaluationKind) -> tuple[int, ...]:
        return tuple(
            milestone.step
            for milestone in self.milestones
            if kind in milestone.kinds
        )

    @property
    def counts(self) -> Mapping[str, int]:
        return {kind.value: len(self.steps_for(kind)) for kind in EvaluationKind}


@dataclass(frozen=True, slots=True)
class EvaluationSchedule:
    configuration: EvaluationScheduleConfiguration

    def _periodic(self, step: int, interval: int) -> bool:
        return interval > 0 and step % interval == 0

    def for_step(
        self, step: int, *, final: bool = False, total_steps: int = 0
    ) -> EvaluationDecision:
        if not isinstance(step, int) or isinstance(step, bool) or step < 0:
            raise ValueError("evaluation step must be a nonnegative integer")
        configuration = self.configuration
        if step == 0:
            return EvaluationDecision(
                True, True, True, configuration.defer_full_scalar, probe=False
            )
        if total_steps <= 0 and (
            configuration.full_milestone_fractions
            or configuration.probe_milestone_fractions
            or configuration.qualitative_milestone_fractions
        ):
            raise ValueError(
                "fractional evaluation milestones need a declared run length"
            )
        resolved = _resolved_milestones(configuration, max(total_steps, step))
        qualitative = (
            self._periodic(step, configuration.qualitative_every_steps)
            or step in resolved[EvaluationKind.QUALITATIVE]
        )
        full_scalar = (
            self._periodic(step, configuration.full_every_steps)
            or step in resolved[EvaluationKind.SCALAR_FULL]
        )
        probe = (
            self._periodic(step, configuration.probe_every_steps)
            or step in resolved[EvaluationKind.SCALAR_PROBE]
        )
        final_audit = False
        if final and configuration.final:
            qualitative = True
            full_scalar = True
            final_audit = True
        # A full scalar read already covers the split the probe samples from,
        # so a step carrying both publishes the full one only.
        probe = probe and not full_scalar
        return EvaluationDecision(
            qualitative,
            full_scalar,
            False,
            configuration.defer_full_scalar,
            probe=probe,
            final_audit=final_audit,
        )

    def plan(self, total_steps: int) -> EvaluationPlan:
        if (
            not isinstance(total_steps, int)
            or isinstance(total_steps, bool)
            or total_steps < 1
        ):
            raise ValueError("an evaluation plan needs a positive run length")
        configuration = self.configuration
        # Enumerate the declared milestones directly rather than walking every
        # optimizer step: a long run has far more steps than milestones, and
        # the plan is rebuilt from scratch on every applied cadence patch.
        candidates = {0, total_steps}
        for interval in (
            configuration.full_every_steps,
            configuration.probe_every_steps,
            configuration.qualitative_every_steps,
        ):
            if interval > 0:
                if total_steps // interval > _MAXIMUM_PLANNED_MILESTONES:
                    raise ValueError(
                        "the declared evaluation cadence plans more milestones "
                        "than a run may carry"
                    )
                candidates.update(range(interval, total_steps + 1, interval))
        for steps in _resolved_milestones(configuration, total_steps).values():
            candidates.update(steps)
        milestones: list[EvaluationMilestone] = []
        for step in sorted(candidates):
            decision = self.for_step(
                step, final=step == total_steps, total_steps=total_steps
            )
            kinds = decision.kinds
            if step == 0:
                kinds = tuple(
                    kind for kind in kinds if kind is not EvaluationKind.FINAL_AUDIT
                )
            if kinds:
                milestones.append(EvaluationMilestone(step, kinds))
        return EvaluationPlan(
            total_steps=total_steps,
            milestones=tuple(milestones),
            probe_examples=configuration.probe_examples,
        )

    def with_cadence_assignments(
        self, assignments: Mapping[str, Any]
    ) -> EvaluationSchedule:
        return EvaluationSchedule(
            self.configuration.with_cadence_assignments(assignments)
        )


@lru_cache(maxsize=256)
def _resolved_milestones(
    configuration: EvaluationScheduleConfiguration, total_steps: int
) -> Mapping[EvaluationKind, frozenset[int]]:
    """Explicit and fractional milestones resolved against one run length.

    Cached because `for_step` is called once per candidate step while a plan is
    being built, and the resolution depends only on the frozen configuration
    and the run length.
    """

    declared = {
        EvaluationKind.SCALAR_FULL: (
            configuration.full_milestone_steps,
            configuration.full_milestone_fractions,
        ),
        EvaluationKind.SCALAR_PROBE: (
            configuration.probe_milestone_steps,
            configuration.probe_milestone_fractions,
        ),
        EvaluationKind.QUALITATIVE: (
            configuration.qualitative_milestone_steps,
            configuration.qualitative_milestone_fractions,
        ),
    }
    resolved: dict[EvaluationKind, frozenset[int]] = {}
    for kind, (steps, fractions) in declared.items():
        selected = {step for step in steps if step <= total_steps}
        if fractions:
            if total_steps <= 0:
                raise ValueError(
                    "fractional evaluation milestones need a declared run length"
                )
            selected.update(_resolved_steps(fractions, total_steps))
        resolved[kind] = frozenset(selected)
    return resolved


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
    elif implementation == EvaluationScheduleImplementation.MILESTONE_CADENCE_V3:
        configuration = EvaluationScheduleConfiguration.from_resolved_v3(
            configuration
        )
    else:
        raise ValueError(
            "resolved evaluation schedule implementation is not allowlisted"
        )
    return EvaluationSchedule(configuration)


__all__ = [
    "EVALUATION_CADENCE_CONTROLS",
    "EvaluationDecision",
    "EvaluationKind",
    "EvaluationMilestone",
    "EvaluationPlan",
    "EvaluationSchedule",
    "EvaluationScheduleConfiguration",
    "EvaluationScheduleImplementation",
    "evaluation_schedule_from_resolved_component",
]
