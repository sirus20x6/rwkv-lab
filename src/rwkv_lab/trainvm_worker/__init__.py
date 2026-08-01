"""Typed worker-side boundary for the TrainVM authority."""

from .bootstrap import (
    MAXIMUM_BOOTSTRAP_BYTES,
    BootstrapError,
    WorkerBootstrap,
    load_worker_bootstrap,
    read_worker_bootstrap_fd,
)
from .invocation import (
    MAXIMUM_INVOCATION_BYTES,
    InvocationError,
    WorkerInvocation,
    load_worker_invocation,
)
from .session import (
    CommandKind,
    ControlAssignment,
    ControlDisposition,
    WorkerCommand,
    WorkerReceipt,
    WorkerSession,
    WorkerSessionError,
)

__all__ = [
    "MAXIMUM_BOOTSTRAP_BYTES",
    "MAXIMUM_INVOCATION_BYTES",
    "BootstrapError",
    "CommandKind",
    "ControlAssignment",
    "ControlDisposition",
    "InvocationError",
    "WorkerBootstrap",
    "WorkerCommand",
    "WorkerInvocation",
    "WorkerReceipt",
    "WorkerSession",
    "WorkerSessionError",
    "load_worker_bootstrap",
    "load_worker_invocation",
    "read_worker_bootstrap_fd",
]
