from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import IntEnum
from types import MappingProxyType
from typing import Protocol

from .session import (
    CommandKind,
    ControlDisposition,
    WorkerCommand,
    wire,
)

Scalar = bool | int | float | str
ControlApplier = Callable[[Mapping[str, Scalar], Mapping[str, Scalar]], None]


class WorkerControlError(RuntimeError):
    pass


def _valid_scalar(value: object) -> bool:
    if isinstance(value, (bool, str)):
        return True
    if isinstance(value, int):
        return -(1 << 63) <= value <= (1 << 63) - 1
    return isinstance(value, float) and math.isfinite(value)


class SafePoint(IntEnum):
    NEXT_MICROBATCH = wire.APPLY_POINT_NEXT_MICROBATCH
    NEXT_OPTIMIZER_STEP = wire.APPLY_POINT_NEXT_OPTIMIZER_STEP
    NEXT_EVAL = wire.APPLY_POINT_NEXT_EVAL
    NEXT_CHECKPOINT = wire.APPLY_POINT_NEXT_CHECKPOINT


class _Session(Protocol):
    def poll_commands(self, maximum: int | None = None) -> tuple[WorkerCommand, ...]: ...

    def acknowledge_controls(
        self,
        command: WorkerCommand,
        disposition: ControlDisposition,
        *,
        effective_values: Mapping[str, Scalar] | None = None,
        effective_step: int = 0,
        diagnostics: tuple[tuple[int, str, str, str, str], ...] = (),
        wait: bool = True,
    ) -> int: ...


@dataclass(frozen=True, slots=True)
class AppliedControlPatch:
    command_id: str
    control_revision: int
    safe_point: SafePoint
    effective_step: int
    assignments: Mapping[str, Scalar]


def _diagnostic(code: str, message: str) -> tuple[tuple[int, str, str, str, str], ...]:
    return (
        (
            wire.Diagnostic.SEVERITY_ERROR,
            code,
            "/spec/controls/catalog",
            message,
            "",
        ),
    )


class WorkerControlRuntime:
    """Apply controller patches in revision order at trainer-owned safe points."""

    def __init__(
        self,
        session: _Session,
        initial_values: Mapping[str, Scalar],
        initial_revision: int,
    ) -> None:
        if (
            not isinstance(initial_revision, int)
            or isinstance(initial_revision, bool)
            or initial_revision < 0
        ):
            raise WorkerControlError("effective control revision must be uint64")
        if not isinstance(initial_values, Mapping):
            raise WorkerControlError("effective controls must be a mapping")
        if any(
            not isinstance(key, str)
            or not key
            or len(key.encode("utf-8")) > 1024
            or not _valid_scalar(value)
            for key, value in initial_values.items()
        ):
            raise WorkerControlError("effective controls contain an invalid scalar")
        self._session = session
        self._effective = dict(initial_values)
        self._revision = initial_revision
        self._pending: list[WorkerCommand] = []

    @property
    def effective_values(self) -> Mapping[str, Scalar]:
        return MappingProxyType(dict(self._effective))

    @property
    def effective_revision(self) -> int:
        return self._revision

    def _collect(self) -> None:
        commands = self._session.poll_commands()
        if any(command.kind is not CommandKind.CONTROLS for command in commands):
            raise WorkerControlError(
                "worker received a lifecycle command unsupported by this protocol revision"
            )
        self._pending.extend(commands)

    def apply(
        self,
        safe_point: SafePoint,
        *,
        effective_step: int,
        applier: ControlApplier,
    ) -> tuple[AppliedControlPatch, ...]:
        if not isinstance(safe_point, SafePoint):
            raise TypeError("safe point must be allowlisted")
        if (
            not isinstance(effective_step, int)
            or isinstance(effective_step, bool)
            or effective_step <= 0
        ):
            raise WorkerControlError("safe-point effective step must be positive")
        self._collect()
        applied: list[AppliedControlPatch] = []
        while self._pending:
            command = self._pending[0]
            if command.expected_control_revision != self._revision:
                self._session.acknowledge_controls(
                    command,
                    ControlDisposition.REJECTED,
                    diagnostics=_diagnostic(
                        "control.worker_revision_conflict",
                        "worker effective control revision differs from the command",
                    ),
                )
                self._pending.pop(0)
                continue
            if command.requires_pause or command.apply_point == wire.APPLY_POINT_RESTART:
                self._session.acknowledge_controls(
                    command,
                    ControlDisposition.RESTART_REQUIRED,
                    diagnostics=_diagnostic(
                        "control.restart_required",
                        "control requires a paused replacement worker",
                    ),
                )
                self._pending.pop(0)
                continue
            if command.apply_point not in {
                wire.APPLY_POINT_IMMEDIATE,
                int(safe_point),
            }:
                break
            assignments = {item.key: item.value for item in command.assignments}
            if any(
                not item.key
                or len(item.key.encode("utf-8")) > 1024
                or not _valid_scalar(item.value)
                for item in command.assignments
            ):
                self._session.acknowledge_controls(
                    command,
                    ControlDisposition.REJECTED,
                    diagnostics=_diagnostic(
                        "control.invalid_assignment",
                        "control patch contains an invalid scalar assignment",
                    ),
                )
                self._pending.pop(0)
                continue
            if len(assignments) != len(command.assignments):
                self._session.acknowledge_controls(
                    command,
                    ControlDisposition.REJECTED,
                    diagnostics=_diagnostic(
                        "control.duplicate_assignment",
                        "control patch contains duplicate assignment keys",
                    ),
                )
                self._pending.pop(0)
                continue
            candidate = {**self._effective, **assignments}
            try:
                applier(
                    MappingProxyType(candidate),
                    MappingProxyType(assignments),
                )
            except Exception:  # noqa: BLE001 - diagnostics must not disclose trainer state
                self._session.acknowledge_controls(
                    command,
                    ControlDisposition.REJECTED,
                    diagnostics=_diagnostic(
                        "control.adapter_rejected",
                        "adapter rejected the atomic control patch",
                    ),
                )
                self._pending.pop(0)
                continue
            acknowledgement_step = (
                0
                if command.apply_point == wire.APPLY_POINT_IMMEDIATE
                else effective_step
            )
            self._session.acknowledge_controls(
                command,
                ControlDisposition.APPLIED,
                effective_values=assignments,
                effective_step=acknowledgement_step,
            )
            self._effective = candidate
            self._revision = command.control_revision
            self._pending.pop(0)
            applied.append(
                AppliedControlPatch(
                    command_id=command.command_id,
                    control_revision=command.control_revision,
                    safe_point=safe_point,
                    effective_step=acknowledgement_step,
                    assignments=MappingProxyType(assignments),
                )
            )
        return tuple(applied)

    def microbatch(
        self, step: int, applier: ControlApplier
    ) -> tuple[AppliedControlPatch, ...]:
        return self.apply(SafePoint.NEXT_MICROBATCH, effective_step=step, applier=applier)

    def optimizer_step(
        self, step: int, applier: ControlApplier
    ) -> tuple[AppliedControlPatch, ...]:
        return self.apply(
            SafePoint.NEXT_OPTIMIZER_STEP, effective_step=step, applier=applier
        )

    def evaluation(
        self, step: int, applier: ControlApplier
    ) -> tuple[AppliedControlPatch, ...]:
        return self.apply(SafePoint.NEXT_EVAL, effective_step=step, applier=applier)

    def checkpoint(
        self, step: int, applier: ControlApplier
    ) -> tuple[AppliedControlPatch, ...]:
        return self.apply(SafePoint.NEXT_CHECKPOINT, effective_step=step, applier=applier)


def controls_from_invocation(
    session: _Session, invocation: object
) -> WorkerControlRuntime:
    controls = getattr(invocation, "controls", None)
    revision = getattr(invocation, "effective_control_revision", None)
    if not isinstance(controls, Mapping):
        raise WorkerControlError("worker invocation controls are missing")
    return WorkerControlRuntime(session, controls, revision)


__all__ = [
    "AppliedControlPatch",
    "ControlApplier",
    "SafePoint",
    "WorkerControlError",
    "WorkerControlRuntime",
    "controls_from_invocation",
]
