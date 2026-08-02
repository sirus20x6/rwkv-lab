"""Family adapter integration above the generic worker and tensor runtimes."""

from .components import AdapterComponentError, WorkerTrainingComponents
from .content_authority import (
    ContentAuthorityError,
    InputContentRootIdentity,
    measure_input_content_root,
    verify_input_content_roots,
)
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
from .io import (
    AdapterInputError,
    WorkspacePathAuthority,
    read_inline_config,
    require_run_directory,
)
from .rwkv_scratch import RWKVScratchTrainConfig

__all__ = [
    "WORKER_BOOTSTRAP_DESCRIPTOR",
    "AdapterComponentError",
    "AdapterDispatchError",
    "AdapterInputError",
    "ContentAuthorityError",
    "HandlerResult",
    "InputContentRootIdentity",
    "RWKVScratchTrainConfig",
    "WorkerEntrypointError",
    "WorkerTrainingComponents",
    "WorkspacePathAuthority",
    "execute_invocation",
    "measure_input_content_root",
    "read_inline_config",
    "require_run_directory",
    "run_worker",
    "supported_adapter_keys",
    "verify_input_content_roots",
]
