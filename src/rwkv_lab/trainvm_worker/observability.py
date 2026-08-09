"""Authority-declared heartbeat and scalar-metric publication.

This module keeps observability policy out of model-family trainers. Trainers
offer measurements by stable name; the immutable invocation decides which
ones are published and owns their unit and step-domain semantics.
"""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Callable, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from types import MappingProxyType
from typing import Protocol

from ._canonical import is_bounded_text
from .execution_phases import ExecutionPhase

MAXIMUM_METRICS = 256
MAXIMUM_LABELS = 16
_UINT64_MAX = (1 << 64) - 1
_OBSERVABILITY_REQUIRED = frozenset(
    {"heartbeat_seconds", "metrics", "retain_raw_metrics_days"}
)
_OBSERVABILITY_OPTIONAL = frozenset(
    {"eval_gallery_artifact", "log_artifact"}
)
_METRIC_REQUIRED = frozenset(
    {"name", "type", "unit", "step_domain", "aggregation"}
)
_METRIC_OPTIONAL = frozenset({"description"})
_METRIC_TYPES = frozenset({"counter", "gauge", "histogram"})
_STEP_DOMAINS = frozenset(
    {"microbatch", "optimizer_step", "sample", "token", "epoch", "wall_time"}
)
_AGGREGATIONS = frozenset(
    {"last", "sum", "mean", "weighted_mean", "min", "max", "histogram"}
)


class WorkerObservabilityError(ValueError):
    pass


class _Session(Protocol):
    def heartbeat(
        self,
        optimizer_step: int,
        phase: str,
        *,
        execution_phase: ExecutionPhase | None = None,
        wait: bool = False,
    ) -> int: ...

    def metric(
        self,
        name: str,
        value: bool | int | float | str,  # noqa: PYI041 - protobuf preserves ints
        *,
        unit: str,
        step_domain: str,
        step: int,
        sample_weight: float = 1.0,
        labels: Mapping[str, str] | None = None,
        wait: bool = False,
    ) -> int: ...


@dataclass(frozen=True, slots=True)
class MetricDeclaration:
    name: str
    metric_type: str
    unit: str
    step_domain: str
    aggregation: str
    description: str | None = None


@dataclass(frozen=True, slots=True)
class ObservabilityDeclaration:
    heartbeat_seconds: int
    metrics: Mapping[str, MetricDeclaration]
    retain_raw_metrics_days: int


def _uint64(value: object, label: str) -> int:
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or value < 0
        or value > _UINT64_MAX
    ):
        raise WorkerObservabilityError(f"{label} must be uint64")
    return value


def load_observability_declaration(value: object) -> ObservabilityDeclaration:
    if not isinstance(value, Mapping):
        raise WorkerObservabilityError("worker observability declaration is not an object")
    fields = set(value)
    if not _OBSERVABILITY_REQUIRED.issubset(fields) or not fields.issubset(
        _OBSERVABILITY_REQUIRED | _OBSERVABILITY_OPTIONAL
    ):
        raise WorkerObservabilityError("worker observability declaration is inexact")
    heartbeat = value["heartbeat_seconds"]
    retention = value["retain_raw_metrics_days"]
    raw_metrics = value["metrics"]
    if (
        not isinstance(heartbeat, int)
        or isinstance(heartbeat, bool)
        or not 1 <= heartbeat <= 300
        or not isinstance(retention, int)
        or isinstance(retention, bool)
        or retention < 1
        or not isinstance(raw_metrics, (tuple, list))
        or len(raw_metrics) > MAXIMUM_METRICS
    ):
        raise WorkerObservabilityError("worker observability bounds are invalid")
    for optional in _OBSERVABILITY_OPTIONAL:
        if optional in value and not is_bounded_text(value[optional], 256):
            raise WorkerObservabilityError(
                "worker observability artifact reference is invalid"
            )
    metrics: dict[str, MetricDeclaration] = {}
    for raw in raw_metrics:
        if not isinstance(raw, Mapping):
            raise WorkerObservabilityError("worker metric declaration is not an object")
        metric_fields = set(raw)
        if not _METRIC_REQUIRED.issubset(metric_fields) or not metric_fields.issubset(
            _METRIC_REQUIRED | _METRIC_OPTIONAL
        ):
            raise WorkerObservabilityError("worker metric declaration is inexact")
        name = raw["name"]
        description = raw.get("description")
        if (
            not is_bounded_text(name, 256)
            or name in metrics
            or not isinstance(raw["type"], str)
            or raw["type"] not in _METRIC_TYPES
            or not is_bounded_text(raw["unit"], 128)
            or not isinstance(raw["step_domain"], str)
            or raw["step_domain"] not in _STEP_DOMAINS
            or not isinstance(raw["aggregation"], str)
            or raw["aggregation"] not in _AGGREGATIONS
            or (
                description is not None
                and (
                    not isinstance(description, str)
                    or len(description.encode("utf-8")) > 2048
                )
            )
        ):
            raise WorkerObservabilityError("worker metric semantics are invalid")
        assert isinstance(name, str)
        metrics[name] = MetricDeclaration(
            name=name,
            metric_type=raw["type"],
            unit=raw["unit"],
            step_domain=raw["step_domain"],
            aggregation=raw["aggregation"],
            description=description,
        )
    return ObservabilityDeclaration(heartbeat, MappingProxyType(metrics), retention)


class WorkerObservability:
    """Publish only metrics selected by the sealed experiment declaration."""

    def __init__(
        self,
        session: _Session,
        declaration: ObservabilityDeclaration,
        *,
        monotonic_ns: Callable[[], int] = time.monotonic_ns,
    ) -> None:
        self._session = session
        self.declaration = declaration
        self._monotonic_ns = monotonic_ns
        self._last_heartbeat_ns: int | None = None
        self._heartbeat_lock = threading.Lock()

    def optimizer_step(
        self,
        step: int,
        phase: str = "train",
        *,
        execution_phase: ExecutionPhase | None = None,
    ) -> int | None:
        """Publish a rate-limited heartbeat.

        ``execution_phase`` binds the heartbeat to an authority-requested
        compile or warmup.  It is the routed field; ``phase`` remains the
        free-text operator label and may not name a phase on its own.
        """

        step = _uint64(step, "optimizer step")
        if not is_bounded_text(phase, 128):
            raise WorkerObservabilityError("worker heartbeat phase is invalid")
        with self._heartbeat_lock:
            now = self._monotonic_ns()
            if not isinstance(now, int) or isinstance(now, bool) or now < 0:
                raise WorkerObservabilityError("worker monotonic clock is invalid")
            if self._last_heartbeat_ns is not None and now < self._last_heartbeat_ns:
                raise WorkerObservabilityError("worker monotonic clock regressed")
            interval = self.declaration.heartbeat_seconds * 1_000_000_000
            if (
                self._last_heartbeat_ns is not None
                and now - self._last_heartbeat_ns < interval
            ):
                return None
            sequence = self._session.heartbeat(
                step, phase, execution_phase=execution_phase, wait=False
            )
            self._last_heartbeat_ns = now
            return sequence

    @contextmanager
    def keepalive(
        self,
        step: int,
        phase: str,
        *,
        execution_phase: ExecutionPhase | None = None,
    ):
        """Keep a declared heartbeat alive around one blocking worker phase."""

        step = _uint64(step, "optimizer step")
        if not is_bounded_text(phase, 128):
            raise WorkerObservabilityError("worker heartbeat phase is invalid")
        self.optimizer_step(step, phase, execution_phase=execution_phase)
        stopped = threading.Event()
        failures: list[BaseException] = []

        def pulse() -> None:
            interval = max(0.05, self.declaration.heartbeat_seconds / 2.0)
            while not stopped.wait(interval):
                try:
                    self.optimizer_step(
                        step, phase, execution_phase=execution_phase
                    )
                except BaseException as error:  # noqa: BLE001 - relay after join
                    failures.append(error)
                    stopped.set()

        thread = threading.Thread(
            target=pulse,
            name=f"trainvm-heartbeat-{phase}",
            daemon=True,
        )
        thread.start()
        body_failed = True
        try:
            yield
            body_failed = False
        finally:
            stopped.set()
            thread.join()
            if failures and not body_failed:
                raise failures[0]

    def publish_if_declared(
        self,
        name: str,
        value: bool | int | float | str,  # noqa: PYI041 - protobuf preserves ints
        *,
        step: int,
        sample_weight: float = 1.0,
        labels: Mapping[str, str] | None = None,
    ) -> int | None:
        declaration = self.declaration.metrics.get(name)
        if declaration is None:
            return None
        step = _uint64(step, "metric step")
        if isinstance(value, float) and not math.isfinite(value):
            raise WorkerObservabilityError("worker metric value must be finite")
        if (
            not isinstance(value, (bool, int, float, str))
            or (
                isinstance(value, int)
                and not isinstance(value, bool)
                and not -(1 << 63) <= value < (1 << 63)
            )
            or (isinstance(value, str) and len(value.encode("utf-8")) > 4096)
        ):
            raise WorkerObservabilityError("worker metric value has unsupported type")
        if (
            isinstance(sample_weight, bool)
            or not isinstance(sample_weight, (int, float))
            or not math.isfinite(sample_weight)
            or sample_weight <= 0
        ):
            raise WorkerObservabilityError("worker metric weight must be positive and finite")
        metric_labels = dict(labels or {})
        if len(metric_labels) > MAXIMUM_LABELS or any(
            not is_bounded_text(key, 128) or not is_bounded_text(item, 256)
            for key, item in metric_labels.items()
        ):
            raise WorkerObservabilityError("worker metric labels are invalid")
        return self._session.metric(
            name,
            value,
            unit=declaration.unit,
            step_domain=declaration.step_domain,
            step=step,
            sample_weight=sample_weight,
            labels=metric_labels,
            wait=False,
        )


def observability_from_invocation(
    session: _Session, invocation: object
) -> WorkerObservability:
    return WorkerObservability(
        session,
        load_observability_declaration(getattr(invocation, "observability", None)),
    )


__all__ = [
    "MAXIMUM_METRICS",
    "MetricDeclaration",
    "ObservabilityDeclaration",
    "WorkerObservability",
    "WorkerObservabilityError",
    "load_observability_declaration",
    "observability_from_invocation",
]
