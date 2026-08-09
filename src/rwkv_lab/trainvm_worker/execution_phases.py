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


class WorkerExecutionPhaseCancelled(Exception):
    """A controller cancellation stopped an enabled phase at a step boundary.

    Deliberately not a ``WorkerExecutionPhaseError``: nothing was violated, so
    an adapter that catches phase *errors* to report a defect must not swallow
    this one.  The phase is still receipted before it is raised.
    """


class ExecutionPhase(str, Enum):
    COMPILE = "compile"
    WARMUP = "warmup"


class ExecutionPhaseDisposition(str, Enum):
    COMPLETED = "completed"
    SKIPPED = "skipped"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(frozen=True, slots=True)
class ExecutionPhaseRequest:
    phase: ExecutionPhase
    enabled: bool
    steps: int | None
    request_digest: str


class _PhaseReceiptChannel(Protocol):
    @property
    def execution_phase_cancellation(self) -> str | None:
        """The controller's cancellation reason, or None while it has not sent one.

        Read-only and sticky.  The phase runtime consults this between bounded
        steps instead of draining the command queue, because the adapter still
        has to receive that command to acknowledge the lifecycle transition.
        """

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
            # Checked after the step rather than before, so a cancellation
            # always lands on a completed step boundary. `execute` must call
            # mark_step once per step for this to bound anything; a callback
            # that ignores the raise is caught by the step-count check below.
            reason = self._channel.execution_phase_cancellation
            if reason is not None:
                raise WorkerExecutionPhaseCancelled(reason)

        after: str | None = None
        try:
            if request.enabled:
                reason = self._channel.execution_phase_cancellation
                if reason is not None:
                    raise WorkerExecutionPhaseCancelled(reason)
                execute(steps, mark_step)
                if steps_executed != steps:
                    raise WorkerExecutionPhaseError(
                        "execution phase did not complete every requested step"
                    )
            after = state_fingerprint(snapshot())
            if after != before:
                raise WorkerExecutionPhaseError(
                    "execution phase did not restore the training trajectory"
                )
        except WorkerExecutionPhaseCancelled as cancellation:
            # The trajectory still has to be intact: a cancelled phase is as
            # disposable as a completed one. If the adapter left it moved, the
            # authority refuses the cancelled receipt and the session raises,
            # which is the correct outcome — that is a real defect.
            self._channel.execution_phase_receipt(
                request,
                ExecutionPhaseDisposition.CANCELLED,
                steps_executed=steps_executed,
                state_fingerprint_before=before,
                state_fingerprint_after=state_fingerprint(snapshot()),
                started_at_ns=started_at_ns,
                completed_at_ns=time.time_ns(),
                diagnostics=(
                    (
                        wire.Diagnostic.SEVERITY_WARNING,
                        "execution.phase_cancelled",
                        f"/spec/execution/{request.phase.value}",
                        str(cancellation)
                        or "the controller cancelled this attempt",
                        "the phase stopped at a step boundary; its receipt "
                        "records how far it got",
                    ),
                ),
            )
            raise
        except Exception as error:
            if after is None:
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
        assert after is not None
        return self._channel.execution_phase_receipt(
            request,
            (
                ExecutionPhaseDisposition.COMPLETED
                if request.enabled
                else ExecutionPhaseDisposition.SKIPPED
            ),
            steps_executed=steps_executed,
            state_fingerprint_before=before,
            state_fingerprint_after=after,
            started_at_ns=started_at_ns,
            completed_at_ns=time.time_ns(),
        )


class WorkerExecutionPhases:
    """One-shot coordinator for every phase declared by an invocation.

    Adapters receive this object before they initialize their trainer.  They
    must explicitly execute (or skip, when disabled) every request.  This
    prevents a declarative compile/warmup request from being silently ignored.
    """

    def __init__(
        self,
        channel: _PhaseReceiptChannel,
        requests: Sequence[ExecutionPhaseRequest],
    ) -> None:
        self._runtime = WorkerExecutionPhaseRuntime(channel)
        self._requests: dict[ExecutionPhase, ExecutionPhaseRequest] = {}
        for request in requests:
            if request.phase in self._requests:
                raise WorkerExecutionPhaseError(
                    "execution phase request is duplicated"
                )
            self._requests[request.phase] = request
        self._receipted: set[ExecutionPhase] = set()

    @property
    def phases(self) -> frozenset[ExecutionPhase]:
        return frozenset(self._requests)

    def request(self, phase: ExecutionPhase) -> ExecutionPhaseRequest | None:
        return self._requests.get(phase)

    def run(
        self,
        phase: ExecutionPhase,
        *,
        snapshot: Callable[[], Mapping[str, Any]],
        execute: Callable[[int, Callable[[], None]], None],
    ) -> int | None:
        request = self._requests.get(phase)
        if request is None:
            return None
        if phase in self._receipted:
            raise WorkerExecutionPhaseError(
                f"execution phase {phase.value} was already receipted"
            )
        try:
            sequence = self._runtime.run(
                request,
                snapshot=snapshot,
                execute=execute,
            )
        except Exception:
            # A failed phase has a durable receipt too.  The adapter still
            # re-raises and terminates, so no later phase may run accidentally.
            self._receipted.add(phase)
            raise
        self._receipted.add(phase)
        return sequence

    def require_complete(self) -> None:
        missing = set(self._requests).difference(self._receipted)
        if missing:
            names = ", ".join(sorted(phase.value for phase in missing))
            raise WorkerExecutionPhaseError(
                f"adapter omitted declared execution phases: {names}"
            )


__all__ = [
    "ExecutionPhase",
    "ExecutionPhaseDisposition",
    "ExecutionPhaseRequest",
    "WorkerExecutionPhaseCancelled",
    "WorkerExecutionPhaseError",
    "WorkerExecutionPhaseRuntime",
    "WorkerExecutionPhases",
    "decode_execution_phase_requests",
    "state_fingerprint",
]
