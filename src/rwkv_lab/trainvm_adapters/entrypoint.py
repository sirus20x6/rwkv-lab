from __future__ import annotations

import sys
from collections.abc import Callable, Sequence

from rwkv_lab.trainvm_worker import (
    WorkerBootstrap,
    WorkerInvocation,
    WorkerSession,
    WorkerStepProfiler,
    apply_worker_runtime_policy,
    read_worker_bootstrap_fd,
    step_profiler_from_invocation,
)

from .handlers import HandlerResult, execute_invocation

WORKER_BOOTSTRAP_DESCRIPTOR = 4


class WorkerEntrypointError(RuntimeError):
    pass


BootstrapReader = Callable[[int], WorkerBootstrap]
InvocationExecutor = Callable[[WorkerInvocation, WorkerStepProfiler], HandlerResult]
SessionFactory = Callable[[WorkerBootstrap], WorkerSession]


def _bootstrap_descriptor(arguments: Sequence[str]) -> int:
    expected = f"--trainvm-bootstrap-fd={WORKER_BOOTSTRAP_DESCRIPTOR}"
    if tuple(arguments) != (expected,):
        raise WorkerEntrypointError(
            "authority worker accepts only its fixed bootstrap descriptor"
        )
    return WORKER_BOOTSTRAP_DESCRIPTOR


def run_worker(
    bootstrap_descriptor: int = WORKER_BOOTSTRAP_DESCRIPTOR,
    *,
    bootstrap_reader: BootstrapReader = read_worker_bootstrap_fd,
    session_factory: SessionFactory = WorkerSession,
    executor: InvocationExecutor = execute_invocation,
) -> int:
    """Execute one authority-bound invocation and durably report its result."""

    bootstrap = bootstrap_reader(bootstrap_descriptor)
    with session_factory(bootstrap) as session:
        if session.completed_before_connect:
            return 0
        try:
            apply_worker_runtime_policy(
                getattr(session.invocation, "resources", {})
            )
            with step_profiler_from_invocation(
                session, session.invocation
            ) as step_profiler:
                result = executor(session.invocation, step_profiler)
        except Exception as error:  # noqa: BLE001 - trainer failures are terminal events
            # The durable event is intentionally bounded and contains neither the
            # exception message nor invocation values, which may disclose paths.
            session.finish(
                "operation.failed",
                {
                    "code": "adapter_execution_failed",
                    "error_type": type(error).__name__[:128],
                },
            )
            return 1
        session.finish(
            result.event_type,
            result.payload,
            optimizer_step=result.optimizer_step,
        )
    return 0


def main(arguments: Sequence[str] | None = None) -> int:
    return run_worker(
        _bootstrap_descriptor(sys.argv[1:] if arguments is None else arguments)
    )


__all__ = [
    "WORKER_BOOTSTRAP_DESCRIPTOR",
    "WorkerEntrypointError",
    "main",
    "run_worker",
]
