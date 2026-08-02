"""Bounded, authority-declared GPU profiling for TrainVM workers.

The VM owns the declaration and artifact identity.  Tensor runtimes own the
actual profiler backend and call :meth:`WorkerStepProfiler.step` exactly once
after each optimizer update.  Keeping this protocol separate from trainers
prevents profile-window and publication logic from becoming another family-
specific collection of command-line switches.
"""

from __future__ import annotations

import errno
import fcntl
import hashlib
import math
import os
import shutil
import stat
import tempfile
import time
from collections.abc import Iterable, Iterator, Mapping
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any, Protocol, Self, TypeVar

from ._canonical import (
    CanonicalJsonError,
    canonical_dumps,
    canonical_loads,
    exact_fields,
    is_bounded_text,
    is_digest,
    sha256_digest,
)

try:
    from trainvm.v1 import trainvm_pb2 as wire
except ImportError as error:  # pragma: no cover - installation contract
    raise RuntimeError(
        "TrainVM GPU profiling requires the 'trainvm-worker' project extra"
    ) from error


GPU_TRACE_SCHEMA = "trainvm.gpu-trace.v1"
EXTERNAL_PROFILER_AUTHORITY_SCHEMA = "trainvm.external-profiler-authority/v1"
EXTERNAL_PROFILER_AUTHORITY_DESCRIPTOR = 5
EXTERNAL_PROFILER_NVTX_RANGE = "trainvm.profile.capture"
MAXIMUM_TRACE_BYTES = 2 * 1024 * 1024 * 1024
MAXIMUM_KERNEL_ROWS = 256
MAXIMUM_EXTERNAL_PROFILER_AUTHORITY_BYTES = 16 * 1024
_F_GET_SEALS = getattr(fcntl, "F_GET_SEALS", 1034)
_F_SEAL_SEAL = getattr(fcntl, "F_SEAL_SEAL", 0x0001)
_F_SEAL_SHRINK = getattr(fcntl, "F_SEAL_SHRINK", 0x0002)
_F_SEAL_GROW = getattr(fcntl, "F_SEAL_GROW", 0x0004)
_F_SEAL_WRITE = getattr(fcntl, "F_SEAL_WRITE", 0x0008)
_InputItem = TypeVar("_InputItem")
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


@dataclass(frozen=True, slots=True)
class ExternalProfilerAuthority:
    backend: str
    run_id: str
    node_id: str
    attempt_id: str
    launch_profile_digest: str
    profiler_executable_digest: str
    authority_digest: str


_EXTERNAL_PROFILER_AUTHORITY_FIELDS = frozenset(
    {
        "api_version",
        "attempt_id",
        "authority_digest",
        "backend",
        "launch_profile_digest",
        "node_id",
        "profiler_executable_digest",
        "run_id",
    }
)


def load_external_profiler_authority(raw: bytes) -> ExternalProfilerAuthority:
    """Decode hostd's sealed proof that this worker is profiler-wrapped."""

    try:
        document = canonical_loads(
            raw, maximum_bytes=MAXIMUM_EXTERNAL_PROFILER_AUTHORITY_BYTES
        )
        exact_fields(document, _EXTERNAL_PROFILER_AUTHORITY_FIELDS)
        authority_digest = document["authority_digest"]
        body = dict(document)
        del body["authority_digest"]
        valid = (
            body["api_version"] == EXTERNAL_PROFILER_AUTHORITY_SCHEMA
            and body["backend"] in {"nsys", "ncu"}
            and is_bounded_text(body["run_id"], 1024)
            and is_bounded_text(body["node_id"], 1024)
            and is_bounded_text(body["attempt_id"], 1024)
            and is_digest(body["launch_profile_digest"])
            and is_digest(body["profiler_executable_digest"])
            and is_digest(authority_digest)
            and sha256_digest(canonical_dumps(body)) == authority_digest
        )
        if not valid:
            raise GpuProfileError("external profiler authority is invalid")
        return ExternalProfilerAuthority(
            backend=body["backend"],
            run_id=body["run_id"],
            node_id=body["node_id"],
            attempt_id=body["attempt_id"],
            launch_profile_digest=body["launch_profile_digest"],
            profiler_executable_digest=body["profiler_executable_digest"],
            authority_digest=authority_digest,
        )
    except GpuProfileError:
        raise
    except (CanonicalJsonError, KeyError, TypeError, ValueError) as error:
        raise GpuProfileError(
            "external profiler authority decoding failed closed"
        ) from error


def read_external_profiler_authority_fd(
    descriptor: int = EXTERNAL_PROFILER_AUTHORITY_DESCRIPTOR,
) -> ExternalProfilerAuthority:
    """Read the fixed, sealed hostd authority descriptor without seeking it."""

    try:
        metadata = os.fstat(descriptor)
        required_seals = _F_SEAL_WRITE | _F_SEAL_GROW | _F_SEAL_SHRINK | _F_SEAL_SEAL
        seals = fcntl.fcntl(descriptor, _F_GET_SEALS)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_size <= 0
            or metadata.st_size > MAXIMUM_EXTERNAL_PROFILER_AUTHORITY_BYTES
            or seals & required_seals != required_seals
        ):
            raise GpuProfileError(
                "external profiler descriptor is not sealed authority"
            )
        raw = os.pread(descriptor, metadata.st_size, 0)
        if len(raw) != metadata.st_size:
            raise GpuProfileError("external profiler authority was read incompletely")
        return load_external_profiler_authority(raw)
    except GpuProfileError:
        raise
    except OSError as error:
        raise GpuProfileError(
            "external profiler authority descriptor is unavailable"
        ) from error


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

    @property
    def external_staging_root(self) -> Path:
        """Create the private parent used by host-derived Nsight outputs."""

        root = self._root / ".external"
        root.mkdir(mode=0o750, exist_ok=True)
        root = root.resolve(strict=True)
        if root.parent != self._root:
            raise GpuProfileError("external GPU trace staging escaped authority")
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

    def input_wait(self) -> AbstractContextManager[None]: ...

    def track_input(self, values: Iterable[_InputItem]) -> Iterator[_InputItem]: ...

    def step(self, optimizer_step: int) -> None: ...


class NullStepProfiler:
    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exception: object) -> None:
        return None

    @contextmanager
    def input_wait(self) -> Iterator[None]:
        yield

    def track_input(self, values: Iterable[_InputItem]) -> Iterator[_InputItem]:
        return iter(values)

    def step(self, optimizer_step: int) -> None:
        if not isinstance(optimizer_step, int) or isinstance(optimizer_step, bool):
            raise GpuProfileError("optimizer step must be an integer")


class _ExternalProfilerRuntime(Protocol):
    def start_capture(self) -> None: ...

    def stop_capture(self) -> None: ...


class _TorchExternalProfilerRuntime:
    """CUDA/NVTX controls used by a host-owned Nsight wrapper.

    This class does not start a process and accepts no command-line or
    environment input. It only opens and closes the fixed range already bound
    by hostd's sealed launch authority.
    """

    def __init__(self) -> None:
        try:
            import torch
        except ImportError as error:  # pragma: no cover - dependency contract
            raise GpuProfileError(
                "external GPU profiling requires the CUDA Torch runtime"
            ) from error
        if not torch.cuda.is_available():
            raise GpuProfileError("external GPU profiler requires CUDA")
        self._torch = torch

    @staticmethod
    def _require_cuda_success(status: object, operation: str) -> None:
        if status is not None and status != 0:
            raise GpuProfileError(
                f"external GPU profiler {operation} returned {status!r}"
            )

    def start_capture(self) -> None:
        self._require_cuda_success(
            self._torch.cuda.cudart().cudaProfilerStart(), "start"
        )
        try:
            self._torch.cuda.nvtx.range_push(EXTERNAL_PROFILER_NVTX_RANGE)
        except Exception:
            # Do not leave nsys collecting an unbounded run if NVTX setup
            # fails after cudaProfilerStart succeeded.
            self._torch.cuda.cudart().cudaProfilerStop()
            raise

    def stop_capture(self) -> None:
        self._torch.cuda.nvtx.range_pop()
        self._require_cuda_success(self._torch.cuda.cudart().cudaProfilerStop(), "stop")


class ExternalStepProfiler:
    """Optimizer-step authority for an already host-wrapped Nsight process."""

    def __init__(
        self,
        session: _ArtifactSession,
        request: GpuTraceRequest,
        *,
        authority: ExternalProfilerAuthority | None = None,
        runtime: _ExternalProfilerRuntime | None = None,
    ) -> None:
        if request.backend not in {"nsys", "ncu"}:
            raise GpuProfileError("external step profiler requires an Nsight backend")
        self._session = session
        self._request = request
        self._authority = authority
        self._runtime = runtime
        self._publisher = GpuTracePublisher(session, request)
        # The host owns the basename and passes it only through a sealed launch
        # profile. The worker owns write authority for the run and prepares the
        # fixed private parent before Nsight opens that basename.
        _ = self._publisher.external_staging_root
        self._steps_seen = 0
        self._last_step: int | None = None
        self._captured_steps: list[int] = []
        self._active = False
        self._entered = False
        self._finished = False

    @property
    def captured_steps(self) -> tuple[int, ...]:
        return tuple(self._captured_steps)

    @property
    def complete(self) -> bool:
        return (
            self._entered
            and not self._active
            and self._steps_seen >= self._request.total_steps
            and len(self._captured_steps) == self._request.capture_steps
        )

    def _validate_authority(self) -> None:
        if self._authority is None:
            self._authority = read_external_profiler_authority_fd()
        bootstrap = self._session.bootstrap
        if (
            self._authority.backend != self._request.backend
            or self._authority.run_id != getattr(bootstrap, "run_id", None)
            or self._authority.node_id != getattr(bootstrap, "node_id", None)
            or self._authority.attempt_id != getattr(bootstrap, "attempt_id", None)
        ):
            raise GpuProfileError(
                "external profiler authority disagrees with this worker attempt"
            )

    def _start(self) -> None:
        if self._active:
            raise GpuProfileError("external GPU capture is already active")
        assert self._runtime is not None
        self._runtime.start_capture()
        self._active = True

    def _stop(self) -> None:
        if not self._active:
            raise GpuProfileError("external GPU capture is not active")
        assert self._runtime is not None
        try:
            self._runtime.stop_capture()
        finally:
            self._active = False

    def __enter__(self) -> Self:
        if self._entered or self._finished:
            raise GpuProfileError("external profiler entered more than once")
        self._validate_authority()
        if self._runtime is None:
            self._runtime = _TorchExternalProfilerRuntime()
        self._entered = True
        if self._request.skip_steps + self._request.warmup_steps == 0:
            self._start()
        return self

    @contextmanager
    def input_wait(self) -> Iterator[None]:
        yield

    def track_input(self, values: Iterable[_InputItem]) -> Iterator[_InputItem]:
        return iter(values)

    def step(self, optimizer_step: int) -> None:
        if (
            not self._entered
            or self._finished
            or not isinstance(optimizer_step, int)
            or isinstance(optimizer_step, bool)
            or optimizer_step < 0
            or (self._last_step is not None and optimizer_step <= self._last_step)
        ):
            raise GpuProfileError("external GPU trace optimizer steps must increase")
        self._last_step = optimizer_step
        capture_start = self._request.skip_steps + self._request.warmup_steps
        if capture_start <= self._steps_seen < self._request.total_steps:
            if not self._active:
                raise GpuProfileError("external GPU trace window is not active")
            self._captured_steps.append(optimizer_step)
        self._steps_seen += 1
        if self._steps_seen == self._request.total_steps:
            self._stop()
        elif self._steps_seen == capture_start:
            self._start()

    def __exit__(self, *exception: object) -> None:
        if not self._entered or self._finished:
            return
        self._finished = True
        if exception and exception[0] is not None:
            if self._active:
                self._stop()
            return
        if not self.complete:
            if self._active:
                self._stop()
            raise GpuProfileError(
                "training ended before the external GPU trace window completed"
            )


def _event_value(event: object, *names: str) -> float:
    for name in names:
        value = getattr(event, name, None)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            return float(value)
    return 0.0


def _event_interval(event: object) -> tuple[float, float] | None:
    interval = getattr(event, "time_range", None)
    start = getattr(interval, "start", None)
    end = getattr(interval, "end", None)
    if not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
        return None
    start_value = float(start)
    end_value = float(end)
    if (
        not math.isfinite(start_value)
        or not math.isfinite(end_value)
        or end_value < start_value
    ):
        return None
    return start_value, end_value


def _interval_union_duration(intervals: list[tuple[float, float]]) -> float:
    if not intervals:
        return 0.0
    ordered = sorted(intervals)
    total = 0.0
    left, right = ordered[0]
    for next_left, next_right in ordered[1:]:
        if next_left <= right:
            right = max(right, next_right)
            continue
        total += right - left
        left, right = next_left, next_right
    return total + right - left


def _profile_activity_summary(profile: object) -> dict[str, object]:
    events_method = getattr(profile, "events", None)
    if not callable(events_method):
        raise GpuProfileError("torch profiler does not expose activity events")
    accelerator_intervals: list[tuple[float, float]] = []
    step_intervals: list[tuple[float, float]] = []
    for event in events_method():
        interval = _event_interval(event)
        if interval is None:
            continue
        device_type = getattr(event, "device_type", None)
        if getattr(device_type, "name", "") == "CUDA":
            accelerator_intervals.append(interval)
        if str(getattr(event, "key", "")) == "ProfilerStep*":
            step_intervals.append(interval)
    if not step_intervals:
        raise GpuProfileError(
            "torch profiler omitted captured optimizer-step intervals"
        )
    window_start = min(interval[0] for interval in step_intervals)
    window_end = max(interval[1] for interval in step_intervals)
    wall_time = window_end - window_start
    if not math.isfinite(wall_time) or wall_time <= 0:
        raise GpuProfileError("torch profiler produced an invalid captured-step window")
    clipped = [
        (max(start, window_start), min(end, window_end))
        for start, end in accelerator_intervals
        if end > window_start and start < window_end
    ]
    active_time = _interval_union_duration(clipped)
    active_ratio = min(1.0, max(0.0, active_time / wall_time))
    return {
        "accelerator_launch_count": len(clipped),
        "captured_step_wall_time_us": wall_time,
        "gpu_active_ratio": active_ratio,
        "gpu_active_time_us": active_time,
    }


def _input_stall_summary(
    captured: list[float | None], wall_time_us: float
) -> dict[str, object]:
    if not captured or all(value is None for value in captured):
        return {}
    if any(value is None for value in captured):
        raise GpuProfileError("input-stall observation is incomplete across capture")
    measured = [float(value) for value in captured if value is not None]
    if any(not math.isfinite(value) or value < 0 for value in measured):
        raise GpuProfileError("input-stall observation is invalid")
    total = sum(measured)
    if total > wall_time_us * 1.01:
        raise GpuProfileError("input-stall time exceeds the captured step window")
    total = min(total, wall_time_us)
    return {
        "input_stall_ratio": total / wall_time_us,
        "input_stall_time_us": total,
    }


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
        self._torch: Any = None
        self._allocator_baseline: tuple[int, int] | None = None
        self._pending_input_stall_us = 0.0
        self._pending_input_observations = 0
        self._captured_input_stall_us: list[float | None] = []

    @contextmanager
    def input_wait(self) -> Iterator[None]:
        started = time.perf_counter_ns()
        try:
            yield
        finally:
            elapsed = (time.perf_counter_ns() - started) / 1000.0
            if elapsed < 0 or not math.isfinite(elapsed):
                raise GpuProfileError("input-stall clock observation is invalid")
            self._pending_input_stall_us += elapsed
            self._pending_input_observations += 1

    def track_input(self, values: Iterable[_InputItem]) -> Iterator[_InputItem]:
        iterator = iter(values)
        while True:
            started = time.perf_counter_ns()
            try:
                value = next(iterator)
            except StopIteration:
                return
            elapsed = (time.perf_counter_ns() - started) / 1000.0
            if elapsed < 0 or not math.isfinite(elapsed):
                raise GpuProfileError("input-stall clock observation is invalid")
            self._pending_input_stall_us += elapsed
            self._pending_input_observations += 1
            yield value

    def _reset_allocator_window(self) -> None:
        if self._torch is None:
            raise GpuProfileError("torch profiler allocator state is unavailable")
        self._torch.cuda.reset_peak_memory_stats()
        self._allocator_baseline = (
            int(self._torch.cuda.memory_allocated()),
            int(self._torch.cuda.memory_reserved()),
        )

    def _allocator_summary(self) -> dict[str, int]:
        if self._torch is None or self._allocator_baseline is None:
            raise GpuProfileError("torch profiler allocator window was not initialized")
        baseline_allocated, baseline_reserved = self._allocator_baseline
        peak_allocated = int(self._torch.cuda.max_memory_allocated())
        peak_reserved = int(self._torch.cuda.max_memory_reserved())
        if (
            min(
                baseline_allocated,
                baseline_reserved,
                peak_allocated,
                peak_reserved,
            )
            < 0
            or baseline_allocated > baseline_reserved
            or peak_allocated > peak_reserved
            or peak_allocated < baseline_allocated
            or peak_reserved < baseline_reserved
        ):
            raise GpuProfileError("torch allocator metrics are internally inconsistent")
        return {
            "allocator_baseline_allocated_bytes": baseline_allocated,
            "allocator_baseline_reserved_bytes": baseline_reserved,
            "allocator_peak_allocated_bytes": peak_allocated,
            "allocator_peak_reserved_bytes": peak_reserved,
        }

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
        activity_summary = _profile_activity_summary(profile)
        self._summary = {
            "accelerator_time_us": accelerator_total,
            "cpu_time_us": cpu_total,
            "kernel_or_operator_count": len(rows),
            "top_operators": rows[:MAXIMUM_KERNEL_ROWS],
            **activity_summary,
            **_input_stall_summary(
                self._captured_input_stall_us,
                float(activity_summary["captured_step_wall_time_us"]),
            ),
            **self._allocator_summary(),
        }
        self._trace_path = path

    def __enter__(self) -> Self:
        try:
            import torch
        except ImportError as error:  # pragma: no cover - dependency contract
            raise GpuProfileError("torch profiler backend is unavailable") from error
        if not torch.cuda.is_available():
            raise GpuProfileError("torch GPU profiler requires CUDA")
        self._torch = torch
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
        if self._request.skip_steps + self._request.warmup_steps == 0:
            self._reset_allocator_window()
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
            self._captured_input_stall_us.append(
                self._pending_input_stall_us
                if self._pending_input_observations > 0
                else None
            )
        self._pending_input_stall_us = 0.0
        self._pending_input_observations = 0
        self._steps_seen += 1
        assert self._profile is not None
        self._profile.step()
        if self._steps_seen == self._request.skip_steps + self._request.warmup_steps:
            self._reset_allocator_window()

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
            self._torch = None


def step_profiler_from_invocation(
    session: _ArtifactSession, invocation: object
) -> WorkerStepProfiler:
    request = trace_request_from_invocation(invocation)
    if request is None:
        return NullStepProfiler()
    if request.backend == "torch":
        return TorchStepProfiler(session, request)
    return ExternalStepProfiler(session, request)


__all__ = [
    "EXTERNAL_PROFILER_AUTHORITY_DESCRIPTOR",
    "EXTERNAL_PROFILER_AUTHORITY_SCHEMA",
    "EXTERNAL_PROFILER_NVTX_RANGE",
    "GPU_TRACE_SCHEMA",
    "ExternalProfilerAuthority",
    "ExternalStepProfiler",
    "GpuProfileError",
    "GpuTracePublisher",
    "GpuTraceRequest",
    "NullStepProfiler",
    "PublishedGpuTrace",
    "TorchStepProfiler",
    "WorkerStepProfiler",
    "load_external_profiler_authority",
    "read_external_profiler_authority_fd",
    "step_profiler_from_invocation",
    "trace_request_from_invocation",
]
