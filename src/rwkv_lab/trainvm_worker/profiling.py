"""Bounded, authority-declared GPU profiling for TrainVM workers.

The VM owns the declaration and artifact identity.  Tensor runtimes own the
actual profiler backend and call :meth:`WorkerStepProfiler.step` exactly once
after each optimizer update.  Keeping this protocol separate from trainers
prevents profile-window and publication logic from becoming another family-
specific collection of command-line switches.
"""

from __future__ import annotations

import errno
import hashlib
import math
import os
import shutil
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any, Protocol, Self

from ._canonical import canonical_dumps, is_bounded_text, sha256_digest

try:
    from trainvm.v1 import trainvm_pb2 as wire
except ImportError as error:  # pragma: no cover - installation contract
    raise RuntimeError(
        "TrainVM GPU profiling requires the 'trainvm-worker' project extra"
    ) from error


GPU_TRACE_SCHEMA = "trainvm.gpu-trace.v1"
MAXIMUM_TRACE_BYTES = 2 * 1024 * 1024 * 1024
MAXIMUM_KERNEL_ROWS = 256
_TRACE_FIELDS = frozenset(
    {
        "enabled",
        "backend",
        "warmup_steps",
        "skip_steps",
        "capture_steps",
        "output_artifact",
        "activities",
        "record_shapes",
        "profile_memory",
        "with_stack",
    }
)


class GpuProfileError(RuntimeError):
    pass


class _ArtifactSession(Protocol):
    bootstrap: object
    invocation: object

    def artifact(self, **values: object) -> int: ...


@dataclass(frozen=True, slots=True)
class GpuTraceRequest:
    backend: str
    skip_steps: int
    warmup_steps: int
    capture_steps: int
    output_name: str
    activities: tuple[str, ...]
    record_shapes: bool
    profile_memory: bool
    with_stack: bool

    @property
    def total_steps(self) -> int:
        return self.skip_steps + self.warmup_steps + self.capture_steps


@dataclass(frozen=True, slots=True)
class PublishedGpuTrace:
    artifact_id: str
    manifest_path: Path
    manifest_sha256: str
    trace_sha256: str
    size_bytes: int
    worker_sequence: int


def _integer(value: object, label: str, minimum: int, maximum: int) -> int:
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or not minimum <= value <= maximum
    ):
        raise GpuProfileError(f"GPU trace {label} is outside its bound")
    return value


def _text(value: object, label: str, maximum: int = 256) -> str:
    if not is_bounded_text(value, maximum):
        raise GpuProfileError(f"GPU trace {label} is invalid")
    assert isinstance(value, str)
    return value


def trace_request_from_invocation(invocation: object) -> GpuTraceRequest | None:
    execution = getattr(invocation, "execution", None)
    if execution is None:
        return None
    if not isinstance(execution, Mapping):
        raise GpuProfileError("worker execution declaration is not an object")
    trace = execution.get("gpu_trace")
    if trace is None:
        return None
    if not isinstance(trace, Mapping) or not set(trace).issubset(_TRACE_FIELDS):
        raise GpuProfileError("GPU trace declaration fields are inexact")
    if trace.get("enabled") is False:
        if set(trace) != {"enabled"}:
            raise GpuProfileError("disabled GPU trace retains configuration")
        return None
    required = {
        "enabled",
        "backend",
        "warmup_steps",
        "skip_steps",
        "capture_steps",
        "output_artifact",
        "activities",
    }
    if trace.get("enabled") is not True or not required.issubset(trace):
        raise GpuProfileError("enabled GPU trace declaration is incomplete")
    backend = _text(trace["backend"], "backend")
    if backend not in {"torch", "nsys", "ncu"}:
        raise GpuProfileError("GPU trace backend is unsupported")
    activities_value = trace["activities"]
    if (
        not isinstance(activities_value, (tuple, list))
        or not activities_value
        or len(activities_value) > 2
        or any(value not in {"cpu", "accelerator"} for value in activities_value)
        or len(set(activities_value)) != len(activities_value)
        or "accelerator" not in activities_value
    ):
        raise GpuProfileError("GPU trace activities are invalid")
    options: dict[str, bool] = {}
    for name in ("record_shapes", "profile_memory", "with_stack"):
        value = trace.get(name, False)
        if not isinstance(value, bool):
            raise GpuProfileError(f"GPU trace {name} must be boolean")
        options[name] = value
    if backend != "torch" and any(options.values()):
        raise GpuProfileError("external GPU profiler has torch-only options")
    request = GpuTraceRequest(
        backend=backend,
        skip_steps=_integer(trace["skip_steps"], "skip_steps", 0, 256),
        warmup_steps=_integer(trace["warmup_steps"], "warmup_steps", 0, 256),
        capture_steps=_integer(trace["capture_steps"], "capture_steps", 1, 128),
        output_name=_text(trace["output_artifact"], "output artifact"),
        activities=tuple(activities_value),
        record_shapes=options["record_shapes"],
        profile_memory=options["profile_memory"],
        with_stack=options["with_stack"],
    )
    if request.total_steps > 512:
        raise GpuProfileError("GPU trace total window exceeds 512 optimizer steps")
    return request


def _roots(values: object, label: str) -> tuple[Path, ...]:
    if not isinstance(values, (tuple, list)) or not values:
        raise GpuProfileError(f"workspace {label} is missing")
    result: list[Path] = []
    for value in values:
        if not isinstance(value, str) or not Path(value).is_absolute():
            raise GpuProfileError(f"workspace {label} contains an invalid root")
        try:
            path = Path(value).resolve(strict=True)
        except OSError as error:
            raise GpuProfileError(f"workspace {label} root is unavailable") from error
        if not path.is_dir():
            raise GpuProfileError(f"workspace {label} root is not a directory")
        result.append(path)
    return tuple(result)


def _within(path: Path, roots: tuple[Path, ...]) -> bool:
    return any(path == root or root in path.parents for root in roots)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _sha256_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    total = 0
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            total += len(chunk)
            if total > MAXIMUM_TRACE_BYTES:
                raise GpuProfileError("GPU trace exceeds its publication bound")
            digest.update(chunk)
    if total == 0:
        raise GpuProfileError("GPU profiler produced an empty trace")
    return "sha256:" + digest.hexdigest(), total


class GpuTracePublisher:
    """Freeze a raw trace and bounded summary as one immutable revision."""

    def __init__(self, session: _ArtifactSession, request: GpuTraceRequest) -> None:
        self._session = session
        self._request = request
        invocation = session.invocation
        workspace = getattr(invocation, "workspace", None)
        publishes = getattr(invocation, "publishes", None)
        if not isinstance(workspace, Mapping) or not isinstance(publishes, Mapping):
            raise GpuProfileError("worker profiling publication contract is invalid")
        output = publishes.get(request.output_name)
        if not isinstance(output, Mapping):
            raise GpuProfileError("GPU trace output is absent from worker invocation")
        declaration = output.get("declaration")
        if (
            not isinstance(declaration, Mapping)
            or declaration.get("type") != "opaque"
            or declaration.get("schema") != GPU_TRACE_SCHEMA
            or declaration.get("immutability") != "mutable_until_publish"
            or declaration.get("fingerprint") != "adapter"
        ):
            raise GpuProfileError("GPU trace artifact declaration is incompatible")
        self._logical_name = _text(output.get("logical_name"), "logical name")
        write_roots = _roots(
            workspace.get("allowed_write_roots"), "allowed_write_roots"
        )
        run_directory = workspace.get("run_directory")
        if not isinstance(run_directory, str) or not Path(run_directory).is_absolute():
            raise GpuProfileError("GPU trace run directory is invalid")
        try:
            run_root = Path(run_directory).resolve(strict=True)
        except OSError as error:
            raise GpuProfileError("GPU trace run directory is unavailable") from error
        if not _within(run_root, write_roots):
            raise GpuProfileError("GPU trace run directory is outside write authority")
        self._root = run_root / "trainvm_artifacts" / "gpu_traces"
        self._root.mkdir(mode=0o750, parents=True, exist_ok=True)
        self._root = self._root.resolve(strict=True)
        if not _within(self._root, write_roots):
            raise GpuProfileError("GPU trace publication root escaped write authority")

    @property
    def staging_root(self) -> Path:
        root = self._root / ".staging"
        root.mkdir(mode=0o750, exist_ok=True)
        return root

    def publish(
        self,
        *,
        raw_trace: Path,
        optimizer_steps: tuple[int, ...],
        summary: Mapping[str, object],
    ) -> PublishedGpuTrace:
        if (
            len(optimizer_steps) != self._request.capture_steps
            or any(step < 0 for step in optimizer_steps)
            or any(right <= left for left, right in pairwise(optimizer_steps))
        ):
            raise GpuProfileError("GPU trace optimizer-step window is incomplete")
        raw_trace = raw_trace.resolve(strict=True)
        if not _within(raw_trace, (self.staging_root.resolve(strict=True),)):
            raise GpuProfileError("raw GPU trace is outside private staging")
        trace_sha256, trace_size = _sha256_file(raw_trace)
        if not isinstance(summary, Mapping):
            raise GpuProfileError("GPU trace summary is not an object")
        bootstrap = self._session.bootstrap
        invocation = self._session.invocation
        body: dict[str, object] = {
            "activities": list(self._request.activities),
            "api_version": GPU_TRACE_SCHEMA,
            "attempt_id": _text(bootstrap.attempt_id, "attempt ID", 1024),
            "backend": self._request.backend,
            "capture_steps": self._request.capture_steps,
            "first_optimizer_step": optimizer_steps[0],
            "instrumented_timing": True,
            "invocation_digest": _text(
                invocation.invocation_digest, "invocation digest", 128
            ),
            "last_optimizer_step": optimizer_steps[-1],
            "node_id": _text(bootstrap.node_id, "node ID", 1024),
            "options": {
                "profile_memory": self._request.profile_memory,
                "record_shapes": self._request.record_shapes,
                "with_stack": self._request.with_stack,
            },
            "run_id": _text(bootstrap.run_id, "run ID", 1024),
            "sensitivity": "restricted",
            "skip_steps": self._request.skip_steps,
            "summary": dict(summary),
            "trace_sha256": trace_sha256,
            "trace_size_bytes": trace_size,
            "warmup_steps": self._request.warmup_steps,
        }
        manifest_digest = sha256_digest(canonical_dumps(body))
        artifact_id = (
            "gpu-trace-"
            + hashlib.sha256(
                canonical_dumps(
                    [
                        bootstrap.run_id,
                        bootstrap.node_id,
                        bootstrap.attempt_id,
                        optimizer_steps[0],
                        optimizer_steps[-1],
                        manifest_digest,
                    ]
                )
            ).hexdigest()
        )
        revision = self._root / artifact_id
        temporary = Path(tempfile.mkdtemp(prefix=".revision-", dir=self._root))
        try:
            trace_destination = temporary / "trace.json"
            with raw_trace.open("rb") as source, trace_destination.open("xb") as target:
                shutil.copyfileobj(source, target, 1024 * 1024)
                target.flush()
                os.fsync(target.fileno())
            manifest = {**body, "canonical_manifest_digest": manifest_digest}
            manifest_bytes = canonical_dumps(manifest)
            manifest_path = temporary / "manifest.json"
            with manifest_path.open("xb") as target:
                target.write(manifest_bytes)
                target.flush()
                os.fsync(target.fileno())
            for path in (trace_destination, manifest_path):
                path.chmod(0o440)
            _fsync_directory(temporary)
            temporary.chmod(0o550)
            try:
                os.rename(temporary, revision)
                _fsync_directory(self._root)
            except OSError as error:
                if error.errno not in {errno.EEXIST, errno.ENOTEMPTY}:
                    raise GpuProfileError(
                        "GPU trace revision could not be atomically published"
                    ) from error
                existing = revision / "manifest.json"
                if not existing.is_file() or existing.read_bytes() != manifest_bytes:
                    raise GpuProfileError(
                        "GPU trace revision identity already exists with different bytes"
                    )
            manifest_path = revision / "manifest.json"
        finally:
            if temporary.exists():
                temporary.chmod(0o750)
                shutil.rmtree(temporary)
            raw_trace.unlink(missing_ok=True)
        manifest_sha256 = sha256_digest(manifest_path.read_bytes())
        sequence = self._session.artifact(
            artifact_id=artifact_id,
            logical_name=self._logical_name,
            kind=wire.ARTIFACT_KIND_OPAQUE,
            schema=GPU_TRACE_SCHEMA,
            uri=manifest_path.resolve(strict=True).as_uri(),
            size_bytes=trace_size + manifest_path.stat().st_size,
            fingerprint_algorithm="adapter",
            fingerprint=manifest_sha256,
            parent_artifact_ids=(),
            wait=True,
        )
        return PublishedGpuTrace(
            artifact_id=artifact_id,
            manifest_path=manifest_path,
            manifest_sha256=manifest_sha256,
            trace_sha256=trace_sha256,
            size_bytes=trace_size + manifest_path.stat().st_size,
            worker_sequence=sequence,
        )


class WorkerStepProfiler(Protocol):
    def __enter__(self) -> Self: ...

    def __exit__(self, *exception: object) -> None: ...

    def step(self, optimizer_step: int) -> None: ...


class NullStepProfiler:
    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exception: object) -> None:
        return None

    def step(self, optimizer_step: int) -> None:
        if not isinstance(optimizer_step, int) or isinstance(optimizer_step, bool):
            raise GpuProfileError("optimizer step must be an integer")


def _event_value(event: object, *names: str) -> float:
    for name in names:
        value = getattr(event, name, None)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            return float(value)
    return 0.0


class TorchStepProfiler:
    def __init__(self, session: _ArtifactSession, request: GpuTraceRequest) -> None:
        if request.backend != "torch":
            raise GpuProfileError(
                "Nsight profiling requires a separately qualified host launch profile"
            )
        self._request = request
        self._publisher = GpuTracePublisher(session, request)
        self._profile: Any = None
        self._trace_path: Path | None = None
        self._summary: Mapping[str, object] | None = None
        self._steps_seen = 0
        self._captured_steps: list[int] = []
        self._last_step: int | None = None

    def _trace_ready(self, profile: object) -> None:
        if self._trace_path is not None:
            raise GpuProfileError("GPU profiler emitted more than one trace window")
        descriptor, name = tempfile.mkstemp(
            prefix="torch-trace-", suffix=".json", dir=self._publisher.staging_root
        )
        os.close(descriptor)
        path = Path(name)
        path.unlink()
        profile.export_chrome_trace(str(path))
        rows: list[dict[str, object]] = []
        cpu_total = 0.0
        accelerator_total = 0.0
        for event in profile.key_averages():
            cpu = _event_value(event, "self_cpu_time_total")
            accelerator = _event_value(
                event,
                "self_device_time_total",
                "self_cuda_time_total",
            )
            calls = int(getattr(event, "count", 0) or 0)
            cpu_total += cpu
            accelerator_total += accelerator
            rows.append(
                {
                    "accelerator_time_us": accelerator,
                    "calls": calls,
                    "cpu_time_us": cpu,
                    "name": str(getattr(event, "key", ""))[:512],
                }
            )
        rows.sort(
            key=lambda row: (
                -float(row["accelerator_time_us"]),
                -float(row["cpu_time_us"]),
                str(row["name"]),
            )
        )
        self._summary = {
            "accelerator_time_us": accelerator_total,
            "cpu_time_us": cpu_total,
            "kernel_or_operator_count": len(rows),
            "top_operators": rows[:MAXIMUM_KERNEL_ROWS],
        }
        self._trace_path = path

    def __enter__(self) -> Self:
        try:
            import torch
        except ImportError as error:  # pragma: no cover - dependency contract
            raise GpuProfileError("torch profiler backend is unavailable") from error
        if not torch.cuda.is_available():
            raise GpuProfileError("torch GPU profiler requires CUDA")
        activities = []
        if "cpu" in self._request.activities:
            activities.append(torch.profiler.ProfilerActivity.CPU)
        activities.append(torch.profiler.ProfilerActivity.CUDA)
        self._profile = torch.profiler.profile(
            activities=activities,
            schedule=torch.profiler.schedule(
                wait=self._request.skip_steps,
                warmup=self._request.warmup_steps,
                active=self._request.capture_steps,
                repeat=1,
            ),
            on_trace_ready=self._trace_ready,
            record_shapes=self._request.record_shapes,
            profile_memory=self._request.profile_memory,
            with_stack=self._request.with_stack,
        )
        self._profile.__enter__()
        return self

    def step(self, optimizer_step: int) -> None:
        if (
            not isinstance(optimizer_step, int)
            or isinstance(optimizer_step, bool)
            or optimizer_step < 0
            or (self._last_step is not None and optimizer_step <= self._last_step)
        ):
            raise GpuProfileError("GPU trace optimizer steps must increase")
        self._last_step = optimizer_step
        index = self._steps_seen
        capture_start = self._request.skip_steps + self._request.warmup_steps
        if capture_start <= index < self._request.total_steps:
            self._captured_steps.append(optimizer_step)
        self._steps_seen += 1
        assert self._profile is not None
        self._profile.step()

    def __exit__(self, *exception: object) -> None:
        if self._profile is None:
            return
        try:
            self._profile.__exit__(*exception)
            if exception and exception[0] is not None:
                return
            if self._steps_seen < self._request.total_steps:
                raise GpuProfileError(
                    "training ended before the declared GPU trace window completed"
                )
            if self._trace_path is None or self._summary is None:
                raise GpuProfileError("torch profiler did not emit its trace window")
            self._publisher.publish(
                raw_trace=self._trace_path,
                optimizer_steps=tuple(self._captured_steps),
                summary=self._summary,
            )
        finally:
            self._profile = None


def step_profiler_from_invocation(
    session: _ArtifactSession, invocation: object
) -> WorkerStepProfiler:
    request = trace_request_from_invocation(invocation)
    if request is None:
        return NullStepProfiler()
    return TorchStepProfiler(session, request)


__all__ = [
    "GPU_TRACE_SCHEMA",
    "GpuProfileError",
    "GpuTracePublisher",
    "GpuTraceRequest",
    "NullStepProfiler",
    "PublishedGpuTrace",
    "TorchStepProfiler",
    "WorkerStepProfiler",
    "step_profiler_from_invocation",
    "trace_request_from_invocation",
]
