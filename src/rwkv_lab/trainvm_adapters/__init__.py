"""Family adapter integration above the generic worker and tensor runtimes."""

from .components import AdapterComponentError, WorkerTrainingComponents
from .entrypoint import (
    WORKER_BOOTSTRAP_DESCRIPTOR,
    WorkerEntrypointError,
    run_worker,
)
from .handlers import (
    AdapterDispatchError,
    HandlerResult,
    execute_invocation,
    supported_adapter_keys,
)
from .io import AdapterInputError, read_inline_config, require_run_directory

__all__ = [
    "WORKER_BOOTSTRAP_DESCRIPTOR",
    "AdapterComponentError",
    "AdapterDispatchError",
    "AdapterInputError",
    "HandlerResult",
    "WorkerEntrypointError",
    "WorkerTrainingComponents",
    "execute_invocation",
    "read_inline_config",
    "require_run_directory",
    "run_worker",
    "supported_adapter_keys",
]
