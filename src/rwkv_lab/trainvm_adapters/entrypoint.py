from __future__ import annotations

import sys
import traceback
from collections.abc import Callable, Sequence
from dataclasses import replace

from rwkv_lab.trainvm_runtime_guard import verified_runtime_closure_fingerprint
from rwkv_lab.trainvm_worker import (
    WorkerBootstrap,
    WorkerCancellationRequested,
    WorkerControlRuntime,
    WorkerExecutionPhases,
    WorkerInvocation,
    WorkerObservability,
    WorkerPublicationRuntime,
    WorkerResourcesReleasedPause,
    WorkerSession,
    WorkerStepProfiler,
    accelerator_fence_count,
    apply_worker_runtime_policy,
    bind_eval_gallery_checkpoints,
    controls_from_invocation,
    measure_worker_runtime_evidence,
    observability_from_invocation,
    publish_artifact_requests,
    publish_checkpoint_requests,
    publish_eval_gallery_requests,
    read_worker_bootstrap_fd,
    step_profiler_from_invocation,
)

from .handlers import HandlerResult, execute_invocation
from .io import bind_worker_process_environment

WORKER_BOOTSTRAP_DESCRIPTOR = 4


class WorkerEntrypointError(RuntimeError):
    pass


BootstrapReader = Callable[[int], WorkerBootstrap]
InvocationExecutor = Callable[
    [
        WorkerInvocation,
        WorkerStepProfiler,
        WorkerObservability,
        WorkerControlRuntime,
        WorkerExecutionPhases,
        WorkerPublicationRuntime,
    ],
    HandlerResult,
]
SessionFactory = Callable[[WorkerBootstrap], WorkerSession]
ClosureFingerprintReader = Callable[[], str | None]


def publish_worker_runtime_evidence(
    session: WorkerSession,
    *,
    closure_fingerprint: ClosureFingerprintReader | None = None,
) -> bool:
    """Measure this worker's runtime and send it over the connection it holds.

    Returns whether a report was sent, which is the only thing a caller can
    observe: acceptance is the immutable receipt the authority publishes, and
    refusal is the stream's terminal status, neither of which arrives here.

    Two conditions decide whether a report exists to send at all, and both are
    read from something the authority sealed rather than from this process's
    argv or environment -- a cache decision must never rest on either:

    * **the verified runtime closure.** `measure_worker_runtime_evidence`
      binds the report to the digest the pre-import guard verified, and the
      authority refuses a report whose fingerprint is not the one its sealed
      launch pinned. Outside a sealed deployment the guard never ran, there is
      no such digest, and re-deriving one here would be a second answer to the
      question the guard exists to ask. Nothing is sent.
    * **the accelerator fence.** The portable report this measures names the
      CPU vendor and carries no placement identity, and the authority admits
      that shape only from a launch it fenced to no accelerator -- it derives
      placement specificity from the device selection itself and refuses a
      disagreeing probe. So a launch holding accelerators is left unpublished
      rather than sent a report that would be refused: refusal is a terminal
      stream status, and a worker that killed its own training run to say
      something about a cache namespace would be worse than one that says
      nothing. Delivering the fenced device identity to the worker is what
      that case needs, and no sealed document carries it today.
    """
    # Resolved through the module namespace rather than bound as a default, so
    # the guard reading stays one substitutable seam rather than a value frozen
    # at import time.
    reader = closure_fingerprint or verified_runtime_closure_fingerprint
    fingerprint = reader()
    if fingerprint is None:
        return False
    if accelerator_fence_count(getattr(session.invocation, "resources", {})) != 0:
        return False
    session.publish_runtime_evidence(
        measure_worker_runtime_evidence(
            session.bootstrap,
            runtime_closure_fingerprint=fingerprint,
            selected_devices=(),
            placement_specific=False,
        )
    )
    return True


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
            # Before any adapter import: the report describes the runtime that
            # is about to compile, and the authority's cache-namespace
            # admission is the reader. Measuring after the trainer has loaded
            # would describe a process the compile decision was not made in.
            publish_worker_runtime_evidence(session)
            # Before any adapter import pulls in a library that resolves HOME.
            bind_worker_process_environment(session.invocation.workspace)
            apply_worker_runtime_policy(
                getattr(session.invocation, "resources", {})
            )
            with step_profiler_from_invocation(
                session, session.invocation
            ) as step_profiler:
                observability = observability_from_invocation(
                    session, session.invocation
                )
                controls = controls_from_invocation(session, session.invocation)
                publications = WorkerPublicationRuntime(
                    session,
                    checkpoint_progress=lambda step: observability.optimizer_step(
                        step, "publishing_checkpoint"
                    ),
                    gallery_progress=lambda step: observability.optimizer_step(
                        step, "publishing_eval_gallery"
                    ),
                )
                execution_phases = WorkerExecutionPhases(
                    session, session.execution_phase_requests
                )
                observability.optimizer_step(0, "initializing")
                result = executor(
                    session.invocation,
                    step_profiler,
                    observability,
                    controls,
                    execution_phases,
                    publications,
                )
                execution_phases.require_complete()
            terminal_checkpoints = publish_checkpoint_requests(
                session,
                result.checkpoint_requests,
                progress=lambda step: observability.optimizer_step(
                    step, "publishing_checkpoint"
                ),
            )
            # A live revision is already durable by the time the trainer
            # returns, so the terminal result must report it alongside the
            # terminal checkpoint. Gallery binding below deliberately uses only
            # `terminal_checkpoints`: a live gallery was bound to its own
            # same-step checkpoint when it was frozen, and re-binding by index
            # here would attach it to whatever happened to land first.
            published_checkpoints = (
                *publications.published_checkpoints,
                *terminal_checkpoints,
            )
            if published_checkpoints:
                result = replace(
                    result,
                    payload={
                        **result.payload,
                        "checkpoint_artifact_ids": [
                            checkpoint.artifact_id
                            for checkpoint in published_checkpoints
                        ],
                    },
                )
            published_artifacts = publish_artifact_requests(
                session,
                result.artifact_requests,
                progress=lambda: observability.optimizer_step(
                    result.optimizer_step or 0, "publishing_artifact"
                ),
            )
            if published_artifacts:
                result = replace(
                    result,
                    payload={
                        **result.payload,
                        "artifact_ids": [
                            artifact.artifact_id for artifact in published_artifacts
                        ],
                    },
                )
            terminal_galleries = publish_eval_gallery_requests(
                session,
                bind_eval_gallery_checkpoints(
                    result.eval_gallery_requests, terminal_checkpoints
                ),
                progress=lambda step: observability.optimizer_step(
                    step, "publishing_eval_gallery"
                ),
            )
            published_galleries = (
                *publications.published_galleries,
                *terminal_galleries,
            )
            if published_galleries:
                result = replace(
                    result,
                    payload={
                        **result.payload,
                        "eval_gallery_artifact_ids": [
                            gallery.artifact_id for gallery in published_galleries
                        ],
                    },
                )
        except (WorkerCancellationRequested, WorkerResourcesReleasedPause):
            # The lifecycle acknowledgement is the durable terminal intent.
            # Host authority observes/reaps the process and releases resources;
            # a normal operation result would race the cancellation state.
            return 0
        except Exception as error:  # noqa: BLE001 - trainer failures are terminal events
            # The traceback goes to this process's stderr, which the host
            # authority's operator can read and which is not retained anywhere.
            # It is deliberately not part of the durable event below: the
            # bound on that event is what keeps paths and invocation values out
            # of a shared, replicated artifact. Without this, a trainer failure
            # reached the operator as the exception's type name and nothing
            # else — "ValueError", with no message, file, or line anywhere on
            # the machine.
            traceback.print_exc()
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
    "publish_worker_runtime_evidence",
    "run_worker",
]
