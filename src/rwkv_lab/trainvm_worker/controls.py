from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import IntEnum
from types import MappingProxyType
from typing import Protocol

from .session import (
    CheckpointDisposition,
    CommandKind,
    ControlDisposition,
    LifecycleDisposition,
    WorkerCommand,
    wire,
)

Scalar = bool | int | float | str
ControlApplier = Callable[[Mapping[str, Scalar], Mapping[str, Scalar]], None]


class WorkerControlError(RuntimeError):
    pass


class WorkerCancellationRequested(RuntimeError):
    """The authority accepted a graceful cancel at a trainer safe point."""


class WorkerResourcesReleasedPause(RuntimeError):
    """The authority accepted a pause that retires this worker process."""


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

    def acknowledge_checkpoint(
        self,
        command: WorkerCommand,
        disposition: CheckpointDisposition,
        *,
        optimizer_step: int = 0,
        artifact_id: str = "",
        diagnostics: tuple[tuple[int, str, str, str, str], ...] = (),
        wait: bool = True,
    ) -> int: ...

    def acknowledge_lifecycle(
        self,
        command: WorkerCommand,
        disposition: LifecycleDisposition,
        *,
        optimizer_step: int = 0,
        artifact_id: str = "",
        diagnostics: tuple[tuple[int, str, str, str, str], ...] = (),
        wait: bool = True,
    ) -> int: ...

    def heartbeat(
        self, optimizer_step: int, phase: str, *, wait: bool = False
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

    def checkpoint_state(self) -> dict[str, object]:
        return {
            "effective_control_revision": self._revision,
            "effective_controls": dict(self._effective),
        }

    def verify_checkpoint_state(self, state: Mapping[str, object]) -> None:
        if (
            state.get("effective_control_revision") != self._revision
            or state.get("effective_controls") != self._effective
        ):
            raise WorkerControlError("checkpoint worker-control state mismatch")

    def _collect(self) -> None:
        commands = self._session.poll_commands()
        if any(
            command.kind
            not in {
                CommandKind.CONTROLS,
                CommandKind.CHECKPOINT,
                CommandKind.PAUSE,
                CommandKind.RESUME,
                CommandKind.CANCEL,
            }
            for command in commands
        ):
            raise WorkerControlError("worker received an unsupported command")
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
            if command.kind is CommandKind.CANCEL:
                self._pending.pop(0)
                self._session.acknowledge_lifecycle(
                    command, LifecycleDisposition.APPLIED
                )
                raise WorkerCancellationRequested(command.reason)
            if command.kind is not CommandKind.CONTROLS:
                break
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

    @property
    def checkpoint_requested(self) -> bool:
        self._collect()
        return bool(
            self._pending and self._pending[0].kind is CommandKind.CHECKPOINT
        )

    @property
    def checkpoint_boundary_requested(self) -> bool:
        self._collect()
        if not self._pending:
            return False
        command = self._pending[0]
        return command.kind in {CommandKind.CHECKPOINT, CommandKind.PAUSE} or (
            command.kind is CommandKind.CONTROLS
            and command.apply_point == wire.APPLY_POINT_NEXT_CHECKPOINT
        )

    @property
    def checkpoint_completion_requested(self) -> bool:
        self._collect()
        return bool(
            self._pending
            and self._pending[0].kind in {CommandKind.CHECKPOINT, CommandKind.PAUSE}
        )

    def publish_requested_checkpoint(
        self,
        request: object,
        *,
        progress: Callable[[int], None] | None = None,
    ) -> object | None:
        """Publish one saved safe-point and acknowledge consecutive requests.

        The request is deliberately duck-typed here to keep the scalar-control
        module independent of checkpoint filesystem mechanics.
        """

        self._collect()
        commands: list[WorkerCommand] = []
        for command in self._pending:
            if command.kind not in {CommandKind.CHECKPOINT, CommandKind.PAUSE}:
                break
            commands.append(command)
            if command.kind is CommandKind.PAUSE:
                break
        if not commands:
            return None
        from .checkpoint import (  # local import avoids a module cycle
            CheckpointPublicationRequest,
            CheckpointPublisher,
        )

        requires_publication = any(
            command.kind is CommandKind.CHECKPOINT
            or (command.kind is CommandKind.PAUSE and command.checkpoint_first)
            for command in commands
        )
        if requires_publication and not isinstance(request, CheckpointPublicationRequest):
            raise WorkerControlError(
                "checkpoint publication requires a typed immutable request"
            )
        published = None
        try:
            if requires_publication:
                assert isinstance(request, CheckpointPublicationRequest)
                published = CheckpointPublisher(
                    self._session, output_name=request.output_name
                ).publish(
                    request.source_directory,
                    optimizer_step=request.optimizer_step,
                    resume_grade=request.resume_grade,
                    state_components=request.state_components,
                    parent_artifact_ids=request.parent_artifact_ids,
                    progress=progress,
                )
        except Exception:
            diagnostics = _diagnostic(
                "checkpoint.publication_failed",
                "worker could not publish the requested immutable checkpoint",
            )
            for command in commands:
                if command.kind is CommandKind.CHECKPOINT:
                    self._session.acknowledge_checkpoint(
                        command,
                        CheckpointDisposition.REJECTED,
                        diagnostics=diagnostics,
                    )
                else:
                    self._session.acknowledge_lifecycle(
                        command,
                        LifecycleDisposition.REJECTED,
                        diagnostics=diagnostics,
                    )
                self._pending.pop(0)
            raise
        for command in commands:
            if command.kind is CommandKind.CHECKPOINT:
                assert published is not None
                self._session.acknowledge_checkpoint(
                    command,
                    CheckpointDisposition.APPLIED,
                    optimizer_step=request.optimizer_step,
                    artifact_id=published.artifact_id,
                )
            else:
                self._session.acknowledge_lifecycle(
                    command,
                    LifecycleDisposition.APPLIED,
                    optimizer_step=(request.optimizer_step if published else 0),
                    artifact_id=(published.artifact_id if published else ""),
                )
            self._pending.pop(0)
            if command.kind is CommandKind.PAUSE:
                if command.release_resources:
                    raise WorkerResourcesReleasedPause(
                        "worker resources will be released while paused"
                    )
                self._wait_for_resume(
                    request.optimizer_step
                    if isinstance(request, CheckpointPublicationRequest)
                    else 0
                )
        return published

    def publish_policy_checkpoint(
        self,
        request: object,
        *,
        progress: Callable[[int], None] | None = None,
    ) -> object:
        """Immediately publish one adapter-policy checkpoint.

        Periodic and launch-gate checkpoints are operation outputs rather than
        controller commands.  They still use the same immutable publisher and
        session authority as an on-demand checkpoint; this method deliberately
        does not consume or acknowledge a pending command.
        """

        from .checkpoint import CheckpointPublicationRequest, CheckpointPublisher

        if not isinstance(request, CheckpointPublicationRequest):
            raise WorkerControlError(
                "policy checkpoint publication requires a typed request"
            )
        return CheckpointPublisher(
            self._session, output_name=request.output_name
        ).publish(
            request.source_directory,
            optimizer_step=request.optimizer_step,
            resume_grade=request.resume_grade,
            state_components=request.state_components,
            parent_artifact_ids=request.parent_artifact_ids,
            progress=progress,
        )

    def publish_evaluation_gallery(
        self,
        request: object,
        *,
        checkpoint: object,
    ) -> object:
        """Publish checkpoint-bound evaluation evidence before training mutates."""

        from .checkpoint import PublishedCheckpoint
        from .eval_gallery import EvalGalleryPublicationRequest, EvalGalleryPublisher

        if not isinstance(request, EvalGalleryPublicationRequest) or not isinstance(
            checkpoint, PublishedCheckpoint
        ):
            raise WorkerControlError(
                "evaluation publication requires typed gallery/checkpoint evidence"
            )
        if (
            request.checkpoint_manifest_digest is not None
            or request.checkpoint_request_index is not None
        ):
            raise WorkerControlError(
                "immediate evaluation request must leave checkpoint binding to the runtime"
            )
        return EvalGalleryPublisher(
            self._session, output_name=request.output_name
        ).publish(
            step=request.step,
            step_domain=request.step_domain,
            checkpoint_manifest_digest=checkpoint.manifest_sha256,
            evaluator_profile_digest=request.evaluator_profile_digest,
            use_policy_digest=request.use_policy_digest,
            items=request.items,
            parent_artifact_ids=(
                *request.parent_artifact_ids,
                checkpoint.artifact_id,
            ),
        )

    def _wait_for_resume(self, optimizer_step: int) -> None:
        import time

        while True:
            self._collect()
            if self._pending and self._pending[0].kind is CommandKind.RESUME:
                command = self._pending.pop(0)
                self._session.acknowledge_lifecycle(
                    command, LifecycleDisposition.APPLIED
                )
                return
            if self._pending and self._pending[0].kind is CommandKind.CONTROLS:
                command = self._pending.pop(0)
                if command.expected_control_revision != self._revision:
                    disposition = ControlDisposition.REJECTED
                    diagnostics = _diagnostic(
                        "control.worker_revision_conflict",
                        "worker effective control revision differs from the command",
                    )
                elif (
                    command.requires_pause
                    or command.apply_point == wire.APPLY_POINT_RESTART
                ):
                    disposition = ControlDisposition.RESTART_REQUIRED
                    diagnostics = _diagnostic(
                        "control.restart_required",
                        "control requires a paused replacement worker",
                    )
                else:
                    disposition = ControlDisposition.REJECTED
                    diagnostics = _diagnostic(
                        "control.paused_barrier",
                        "live control cannot be applied while the worker is paused",
                    )
                self._session.acknowledge_controls(
                    command,
                    disposition,
                    diagnostics=diagnostics,
                )
                continue
            if self._pending and self._pending[0].kind is CommandKind.CANCEL:
                command = self._pending.pop(0)
                self._session.acknowledge_lifecycle(
                    command, LifecycleDisposition.APPLIED
                )
                raise WorkerCancellationRequested(command.reason)
            if self._pending:
                raise WorkerControlError(
                    "a non-resume command blocks the retained-resource pause barrier"
                )
            self._session.heartbeat(optimizer_step, "paused", wait=True)
            time.sleep(2.0)

    def publish_requested_checkpoint_directory(
        self,
        source_directory: str,
        *,
        optimizer_step: int,
        resume_grade: str,
        state_components: tuple[str, ...],
        output_name: str = "checkpoint",
        parent_artifact_ids: tuple[str, ...] = (),
        progress: Callable[[int], None] | None = None,
    ) -> object | None:
        from .checkpoint import CheckpointPublicationRequest

        return self.publish_requested_checkpoint(
            CheckpointPublicationRequest(
                source_directory=source_directory,
                optimizer_step=optimizer_step,
                resume_grade=resume_grade,
                state_components=state_components,
                output_name=output_name,
                parent_artifact_ids=parent_artifact_ids,
            ),
            progress=progress,
        )

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
    "Scalar",
    "WorkerControlError",
    "WorkerControlRuntime",
    "controls_from_invocation",
]
