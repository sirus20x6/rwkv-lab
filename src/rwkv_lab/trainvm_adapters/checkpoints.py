"""Family-neutral extraction of completed trainer checkpoint results."""

from __future__ import annotations

import json
import os
import stat
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from rwkv_lab.trainvm_worker import CheckpointPublicationRequest, WorkerInvocation


class CheckpointResultError(ValueError):
    pass


def declares_checkpoint(invocation: WorkerInvocation) -> bool:
    publishes = getattr(invocation, "publishes", {})
    return isinstance(publishes, Mapping) and "checkpoint" in publishes


def completion_reason(status: str) -> str:
    return {
        "cache_span_complete": "cache_span_complete",
        "interrupted": "checkpointed_interruption",
    }.get(status, "training_complete")


def _read_trainer_document(path: Path) -> Mapping[str, Any]:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise CheckpointResultError("trainer result document is unavailable") from error
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or not 0 < before.st_size <= 1024 * 1024:
            raise CheckpointResultError("trainer result document is invalid")
        chunks: list[bytes] = []
        total = 0
        while chunk := os.read(descriptor, 64 * 1024):
            total += len(chunk)
            if total > 1024 * 1024:
                raise CheckpointResultError("trainer result document is oversized")
            chunks.append(chunk)
        after = os.fstat(descriptor)
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ):
            raise CheckpointResultError("trainer result document changed while read")
    finally:
        os.close(descriptor)
    try:
        value = json.loads(b"".join(chunks))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CheckpointResultError("trainer result document is malformed") from error
    if not isinstance(value, dict):
        raise CheckpointResultError("trainer result document is not an object")
    return value


def checkpoint_request(
    invocation: WorkerInvocation,
    run_directory: Path,
    checkpoint: object,
    step: object,
    *,
    state_components: tuple[str, ...],
) -> CheckpointPublicationRequest:
    if (
        not isinstance(checkpoint, str)
        or not isinstance(step, int)
        or isinstance(step, bool)
        or step < 0
    ):
        raise CheckpointResultError("trainer checkpoint result is incomplete")
    try:
        run_root = run_directory.resolve(strict=True)
        checkpoint_path = Path(checkpoint)
        if not checkpoint_path.is_absolute():
            raise CheckpointResultError("trainer checkpoint path is not absolute")
        source = checkpoint_path.resolve(strict=True)
    except OSError as error:
        raise CheckpointResultError("trainer checkpoint path is unavailable") from error
    if not source.is_dir() or (source != run_root and run_root not in source.parents):
        raise CheckpointResultError("trainer checkpoint escaped its run directory")
    parents: list[str] = []
    inputs = getattr(invocation, "inputs", {})
    if isinstance(inputs, Mapping):
        for value in inputs.values():
            if isinstance(value, Mapping):
                artifact_id = value.get("artifact_id")
                if isinstance(artifact_id, str) and artifact_id not in parents:
                    parents.append(artifact_id)
    return CheckpointPublicationRequest(
        source_directory=source,
        optimizer_step=step,
        resume_grade="compatible",
        state_components=state_components,
        parent_artifact_ids=tuple(parents),
    )


def completed_checkpoint_request(
    invocation: WorkerInvocation,
    run_directory: Path,
    *,
    document_names: tuple[str, ...],
    step_fields: tuple[str, ...],
    state_components: tuple[str, ...],
) -> tuple[CheckpointPublicationRequest | None, int | None, str]:
    if not declares_checkpoint(invocation):
        return None, None, "complete"
    document = None
    for name in document_names:
        path = run_directory / name
        if path.is_file() and not path.is_symlink():
            document = _read_trainer_document(path)
            break
    if document is None:
        raise CheckpointResultError(
            "trainer did not publish a terminal result document"
        )
    step = next((document.get(name) for name in step_fields if name in document), None)
    request = checkpoint_request(
        invocation,
        run_directory,
        document.get("checkpoint"),
        step,
        state_components=state_components,
    )
    status = document.get("state", "complete")
    if not isinstance(status, str):
        raise CheckpointResultError("trainer result state is invalid")
    return request, request.optimizer_step, status


__all__ = [
    "CheckpointResultError",
    "checkpoint_request",
    "completed_checkpoint_request",
    "completion_reason",
    "declares_checkpoint",
]
