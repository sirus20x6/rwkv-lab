"""Typed compile/warmup requests delivered by the TrainVM authority."""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol

from trainvm.v1 import trainvm_pb2 as wire

from ._canonical import canonical_dumps, is_digest, sha256_digest
from .invocation import WorkerInvocation


class WorkerExecutionPhaseError(ValueError):
    pass


class ExecutionPhase(str, Enum):
    COMPILE = "compile"
    WARMUP = "warmup"


class ExecutionPhaseDisposition(str, Enum):
    COMPLETED = "completed"
    SKIPPED = "skipped"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class ExecutionPhaseRequest:
    phase: ExecutionPhase
    enabled: bool
    steps: int | None
    request_digest: str


class _PhaseReceiptChannel(Protocol):
    def execution_phase_receipt(
        self,
        request: ExecutionPhaseRequest,
        disposition: ExecutionPhaseDisposition,
        *,
        steps_executed: int,
        state_fingerprint_before: str,
        state_fingerprint_after: str,
        started_at_ns: int,
        completed_at_ns: int,
        diagnostics: Iterable[tuple[int, str, str, str, str]] = (),
        wait: bool = True,
    ) -> int: ...


_WIRE_PHASE = {
    ExecutionPhase.COMPILE: wire.WorkerExecutionPhaseRequest.PHASE_COMPILE,
    ExecutionPhase.WARMUP: wire.WorkerExecutionPhaseRequest.PHASE_WARMUP,
}


def _expected_request(
    invocation: WorkerInvocation, phase: ExecutionPhase
) -> ExecutionPhaseRequest | None:
    execution = invocation.execution
    if execution is None:
        return None
    declaration = execution.get(phase.value)
    if declaration is None:
        return None
    if not isinstance(declaration, Mapping) or set(declaration) not in (
        {"enabled"},
        {"enabled", "steps"},
    ):
        raise WorkerExecutionPhaseError(
            f"worker invocation {phase.value} declaration is not closed"
        )
    enabled = declaration.get("enabled")
    if not isinstance(enabled, bool):
        raise WorkerExecutionPhaseError(
            f"worker invocation {phase.value} enabled flag is invalid"
        )
    steps: int | None = None
    if phase is ExecutionPhase.COMPILE:
        if "steps" in declaration:
            raise WorkerExecutionPhaseError("compile phase cannot declare steps")
    elif "steps" in declaration:
        candidate = declaration["steps"]
        if (
            isinstance(candidate, bool)
            or not isinstance(candidate, int)
            or candidate < 1
            or candidate > 10_000
        ):
            raise WorkerExecutionPhaseError("warmup phase steps are invalid")
        steps = candidate
    if enabled and phase is ExecutionPhase.WARMUP and steps is None:
        raise WorkerExecutionPhaseError("enabled warmup phase requires steps")
    if not enabled and steps is not None:
        raise WorkerExecutionPhaseError("disabled warmup phase cannot declare steps")
    body: dict[str, Any] = {
        "api_version": "trainvm.worker-execution-phase-request/v1",
        "enabled": enabled,
        "invocation_digest": invocation.invocation_digest,
        "phase": phase.value,
    }
    if steps is not None:
        body["steps"] = steps
    return ExecutionPhaseRequest(
        phase=phase,
        enabled=enabled,
        steps=steps,
        request_digest=sha256_digest(canonical_dumps(body)),
    )


def decode_execution_phase_requests(
    values: Sequence[wire.WorkerExecutionPhaseRequest],
    invocation: WorkerInvocation,
) -> tuple[ExecutionPhaseRequest, ...]:
    expected = tuple(
        request
        for phase in ExecutionPhase
        if (request := _expected_request(invocation, phase)) is not None
    )
    if len(values) != len(expected):
        raise WorkerExecutionPhaseError(
            "WorkerWelcome execution-phase requests disagree with the invocation"
        )
    decoded: list[ExecutionPhaseRequest] = []
    seen: set[ExecutionPhase] = set()
    by_phase = {request.phase: request for request in expected}
    for value in values:
        matching = [
            phase
            for phase, wire_phase in _WIRE_PHASE.items()
            if value.phase == wire_phase
        ]
        if len(matching) != 1:
            raise WorkerExecutionPhaseError("execution-phase request phase is invalid")
        phase = matching[0]
        request = by_phase.get(phase)
        steps = value.steps if value.HasField("steps") else None
        if (
            request is None
            or phase in seen
            or value.enabled != request.enabled
            or steps != request.steps
            or not is_digest(value.request_digest)
            or value.request_digest != request.request_digest
        ):
            raise WorkerExecutionPhaseError(
                "execution-phase request binding disagrees with the invocation"
            )
        seen.add(phase)
        decoded.append(request)
    if seen != set(by_phase):
        raise WorkerExecutionPhaseError("execution-phase request set is incomplete")
    return tuple(decoded)


def state_fingerprint(state: Mapping[str, Any]) -> str:
    """Fingerprint a caller's complete JSON-safe training-state identity."""

    if not isinstance(state, Mapping) or not state:
        raise WorkerExecutionPhaseError(
            "phase state identity must be a nonempty mapping"
        )
    try:
        return sha256_digest(canonical_dumps(dict(state)))
    except (TypeError, ValueError) as error:
        raise WorkerExecutionPhaseError(
            "phase state identity is not finite canonical JSON"
        ) from error


class WorkerExecutionPhaseRuntime:
    """Execute one requested phase and atomically receipt its state proof.

    ``snapshot`` must cover the complete trajectory-affecting state named by
    the protocol.  The runtime deliberately fails completion when the before
    and after fingerprints differ; warmup/compile work must be disposable.
    """

    def __init__(self, channel: _PhaseReceiptChannel) -> None:
        self._channel = channel

    def run(
        self,
        request: ExecutionPhaseRequest,
        *,
        snapshot: Callable[[], Mapping[str, Any]],
        execute: Callable[[int, Callable[[], None]], None],
    ) -> int:
        before = state_fingerprint(snapshot())
        started_at_ns = time.time_ns()
        steps = request.steps or 0
        steps_executed = 0

        def mark_step() -> None:
            nonlocal steps_executed
            if steps_executed >= steps:
                raise WorkerExecutionPhaseError(
                    "execution phase completed more steps than requested"
                )
            steps_executed += 1

        if not request.enabled:
            return self._channel.execution_phase_receipt(
                request,
                ExecutionPhaseDisposition.SKIPPED,
                steps_executed=0,
                state_fingerprint_before=before,
                state_fingerprint_after=state_fingerprint(snapshot()),
                started_at_ns=started_at_ns,
                completed_at_ns=time.time_ns(),
            )
        try:
            execute(steps, mark_step)
            if steps_executed != steps:
                raise WorkerExecutionPhaseError(
                    "execution phase did not complete every requested step"
                )
        except Exception as error:
            after = state_fingerprint(snapshot())
            self._channel.execution_phase_receipt(
                request,
                ExecutionPhaseDisposition.FAILED,
                steps_executed=steps_executed,
                state_fingerprint_before=before,
                state_fingerprint_after=after,
                started_at_ns=started_at_ns,
                completed_at_ns=time.time_ns(),
                diagnostics=(
                    (
                        wire.Diagnostic.SEVERITY_ERROR,
                        "execution.phase_failed",
                        f"/spec/execution/{request.phase.value}",
                        str(error) or type(error).__name__,
                        "inspect the immutable phase receipt before retrying",
                    ),
                ),
            )
            raise
        return self._channel.execution_phase_receipt(
            request,
            ExecutionPhaseDisposition.COMPLETED,
            steps_executed=steps_executed,
            state_fingerprint_before=before,
            state_fingerprint_after=state_fingerprint(snapshot()),
            started_at_ns=started_at_ns,
            completed_at_ns=time.time_ns(),
        )


__all__ = [
    "ExecutionPhase",
    "ExecutionPhaseDisposition",
    "ExecutionPhaseRequest",
    "WorkerExecutionPhaseError",
    "WorkerExecutionPhaseRuntime",
    "decode_execution_phase_requests",
    "state_fingerprint",
]
