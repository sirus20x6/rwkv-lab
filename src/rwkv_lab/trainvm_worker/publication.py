"""Narrow live-publication authority for checkpoint-bound eval revisions."""

from __future__ import annotations

from collections.abc import Callable

from .checkpoint import (
    CheckpointPublicationRequest,
    PublishedCheckpoint,
    publish_checkpoint_requests,
)
from .eval_gallery import (
    EvalGalleryPublicationRequest,
    PublishedEvalGallery,
    bind_eval_gallery_checkpoints,
    publish_eval_gallery_requests,
)


class WorkerPublicationError(RuntimeError):
    pass


class WorkerPublicationRuntime:
    """Publish append-only eval revisions without exposing the worker session."""

    def __init__(
        self,
        session: object,
        *,
        checkpoint_progress: Callable[[int], None] | None = None,
        gallery_progress: Callable[[int], None] | None = None,
    ) -> None:
        self._session = session
        self._checkpoint_progress = checkpoint_progress
        self._gallery_progress = gallery_progress
        self._checkpoints: list[PublishedCheckpoint] = []
        self._galleries: list[PublishedEvalGallery] = []
        self._last_step: int | None = None

    @property
    def published_checkpoints(self) -> tuple[PublishedCheckpoint, ...]:
        return tuple(self._checkpoints)

    @property
    def published_galleries(self) -> tuple[PublishedEvalGallery, ...]:
        return tuple(self._galleries)

    def publish_eval_revision(
        self,
        checkpoint: CheckpointPublicationRequest,
        gallery: EvalGalleryPublicationRequest,
    ) -> tuple[PublishedCheckpoint, PublishedEvalGallery]:
        """Freeze one same-step checkpoint/gallery pair in lineage order."""

        if not isinstance(checkpoint, CheckpointPublicationRequest):
            raise WorkerPublicationError(
                "live eval publication requires a typed checkpoint request"
            )
        if not isinstance(gallery, EvalGalleryPublicationRequest):
            raise WorkerPublicationError(
                "live eval publication requires a typed gallery request"
            )
        if checkpoint.optimizer_step != gallery.step:
            raise WorkerPublicationError(
                "live eval checkpoint and gallery steps disagree"
            )
        if gallery.checkpoint_request_index != 0:
            raise WorkerPublicationError(
                "live eval gallery must select its paired checkpoint"
            )
        if gallery.checkpoint_manifest_digest is not None:
            raise WorkerPublicationError(
                "live eval gallery cannot provide a checkpoint digest"
            )
        if self._last_step is not None and gallery.step <= self._last_step:
            raise WorkerPublicationError(
                "live eval gallery steps must be strictly increasing"
            )

        (published_checkpoint,) = publish_checkpoint_requests(
            self._session,
            (checkpoint,),
            progress=self._checkpoint_progress,
        )
        # A checkpoint announcement may already be durable even if gallery image
        # validation fails. Keep that fact visible to terminal result assembly.
        self._checkpoints.append(published_checkpoint)
        (bound_gallery,) = bind_eval_gallery_checkpoints(
            (gallery,), (published_checkpoint,)
        )
        (published_gallery,) = publish_eval_gallery_requests(
            self._session,
            (bound_gallery,),
            progress=self._gallery_progress,
        )
        self._galleries.append(published_gallery)
        self._last_step = gallery.step
        return published_checkpoint, published_gallery


__all__ = ["WorkerPublicationError", "WorkerPublicationRuntime"]
