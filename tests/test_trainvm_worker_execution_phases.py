from __future__ import annotations

from dataclasses import replace

import pytest
from test_trainvm_worker_documents import invocation_document

from rwkv_lab.trainvm_worker import (
    ExecutionPhase,
    ExecutionPhaseDisposition,
    WorkerExecutionPhaseError,
    WorkerExecutionPhaseRuntime,
    decode_execution_phase_requests,
    load_worker_invocation,
    state_fingerprint,
)
from rwkv_lab.trainvm_worker._canonical import canonical_dumps, sha256_digest
from rwkv_lab.trainvm_worker.execution_phases import wire


def _fixture() -> tuple:
    invocation = load_worker_invocation(
        invocation_document(
            execution={
                "component": "trainer",
                "operation": "train",
                "compile": {"enabled": True},
                "warmup": {"enabled": True, "steps": 2},
            }
        )
    )
    values = []
    for phase, enum_value, declaration in (
        (
            "compile",
            wire.WorkerExecutionPhaseRequest.PHASE_COMPILE,
            {"enabled": True},
        ),
        (
            "warmup",
            wire.WorkerExecutionPhaseRequest.PHASE_WARMUP,
            {"enabled": True, "steps": 2},
        ),
    ):
        body = {
            "api_version": "trainvm.worker-execution-phase-request/v1",
            "enabled": declaration["enabled"],
            "invocation_digest": invocation.invocation_digest,
            "phase": phase,
        }
        value = wire.WorkerExecutionPhaseRequest(
            phase=enum_value, enabled=declaration["enabled"]
        )
        if "steps" in declaration:
            value.steps = declaration["steps"]
            body["steps"] = declaration["steps"]
        value.request_digest = sha256_digest(canonical_dumps(body))
        values.append(value)
    return invocation, values


def test_phase_requests_are_exactly_bound_to_the_invocation() -> None:
    invocation, values = _fixture()
    requests = decode_execution_phase_requests(values, invocation)
    assert requests[0].phase is ExecutionPhase.COMPILE
    assert requests[0].steps is None
    assert requests[1].phase is ExecutionPhase.WARMUP
    assert requests[1].steps == 2

    values[1].steps = 3
    with pytest.raises(WorkerExecutionPhaseError):
        decode_execution_phase_requests(values, invocation)


def test_phase_state_fingerprint_is_canonical_and_finite() -> None:
    assert state_fingerprint({"b": 2, "a": [1, True]}) == state_fingerprint(
        {"a": [1, True], "b": 2}
    )
    with pytest.raises(WorkerExecutionPhaseError):
        state_fingerprint({"loss": float("nan")})


class _Channel:
    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def execution_phase_receipt(self, request, disposition, **values):
        self.calls.append((request, disposition, values))
        return len(self.calls)


def test_phase_runtime_counts_steps_and_proves_disposable_state() -> None:
    invocation, values = _fixture()
    warmup = decode_execution_phase_requests(values, invocation)[1]
    channel = _Channel()
    runtime = WorkerExecutionPhaseRuntime(channel)
    state = {"model": "m1", "optimizer": "o1", "rng": 9, "cursor": 4}
    visited = 0

    def execute(steps, mark_step):
        nonlocal visited
        for _ in range(steps):
            visited += 1
            mark_step()

    assert runtime.run(warmup, snapshot=lambda: state, execute=execute) == 1
    assert visited == 2
    _, disposition, receipt = channel.calls[0]
    assert disposition is ExecutionPhaseDisposition.COMPLETED
    assert receipt["steps_executed"] == 2
    assert receipt["state_fingerprint_before"] == receipt["state_fingerprint_after"]


def test_phase_runtime_receipts_partial_failure_before_reraising() -> None:
    invocation, values = _fixture()
    warmup = decode_execution_phase_requests(values, invocation)[1]
    channel = _Channel()
    runtime = WorkerExecutionPhaseRuntime(channel)

    def fail(_steps, mark_step):
        mark_step()
        raise RuntimeError("synthetic warmup failure")

    with pytest.raises(RuntimeError, match="synthetic warmup failure"):
        runtime.run(warmup, snapshot=lambda: {"state": 1}, execute=fail)
    _, disposition, receipt = channel.calls[0]
    assert disposition is ExecutionPhaseDisposition.FAILED
    assert receipt["steps_executed"] == 1
    assert receipt["diagnostics"][0][1] == "execution.phase_failed"


def test_runtime_rejects_an_underexecuted_completed_phase() -> None:
    invocation, values = _fixture()
    warmup = decode_execution_phase_requests(values, invocation)[1]
    channel = _Channel()
    runtime = WorkerExecutionPhaseRuntime(channel)
    with pytest.raises(WorkerExecutionPhaseError):
        runtime.run(
            replace(warmup, steps=2),
            snapshot=lambda: {"state": 1},
            execute=lambda _steps, _mark_step: None,
        )
    assert channel.calls[0][1] is ExecutionPhaseDisposition.FAILED
