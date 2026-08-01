"""Typed worker-side boundary for the TrainVM authority."""

from .bootstrap import (
    MAXIMUM_BOOTSTRAP_BYTES,
    BootstrapError,
    WorkerBootstrap,
    load_worker_bootstrap,
    read_worker_bootstrap_fd,
)
from .eval_gallery import (
    EVAL_GALLERY_SCHEMA,
    EvalGalleryError,
    EvalGalleryItem,
    EvalGalleryPublisher,
    GalleryImage,
    PublishedEvalGallery,
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
from .training import (
    MAXIMUM_COMPONENT_SLOTS,
    RESOLVED_TRAINING_API_VERSION,
    ResolvedTrainingComponent,
    ResolvedTrainingComposition,
    TrainingCompositionError,
    load_resolved_training_composition,
)

__all__ = [
    "EVAL_GALLERY_SCHEMA",
    "MAXIMUM_BOOTSTRAP_BYTES",
    "MAXIMUM_COMPONENT_SLOTS",
    "MAXIMUM_INVOCATION_BYTES",
    "RESOLVED_TRAINING_API_VERSION",
    "BootstrapError",
    "CommandKind",
    "ControlAssignment",
    "ControlDisposition",
    "EvalGalleryError",
    "EvalGalleryItem",
    "EvalGalleryPublisher",
    "GalleryImage",
    "InvocationError",
    "PublishedEvalGallery",
    "ResolvedTrainingComponent",
    "ResolvedTrainingComposition",
    "TrainingCompositionError",
    "WorkerBootstrap",
    "WorkerCommand",
    "WorkerInvocation",
    "WorkerReceipt",
    "WorkerSession",
    "WorkerSessionError",
    "load_resolved_training_composition",
    "load_worker_bootstrap",
    "load_worker_invocation",
    "read_worker_bootstrap_fd",
]
