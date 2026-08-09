from __future__ import annotations

import time
from types import SimpleNamespace

import pytest

from rwkv_lab.trainvm_worker.observability import (
    WorkerObservability,
    WorkerObservabilityError,
    load_observability_declaration,
    observability_from_invocation,
)


def declaration() -> dict[str, object]:
    return {
        "heartbeat_seconds": 5,
        "metrics": [
            {
                "name": "train.loss",
                "type": "gauge",
                "unit": "dimensionless",
                "step_domain": "optimizer_step",
                "aggregation": "weighted_mean",
            }
        ],
        "retain_raw_metrics_days": 30,
    }


class FakeSession:
    def __init__(self) -> None:
        self.heartbeats: list[tuple[object, ...]] = []
        self.metrics: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def heartbeat(self, *values: object, **options: object) -> int:
        self.heartbeats.append((*values, options))
        return len(self.heartbeats)

    def metric(self, *values: object, **options: object) -> int:
        self.metrics.append((values, options))
        return len(self.metrics)


def test_observability_is_closed_and_rejects_duplicate_metrics() -> None:
    loaded = load_observability_declaration(declaration())
    assert loaded.heartbeat_seconds == 5
    assert loaded.metrics["train.loss"].step_domain == "optimizer_step"
    with pytest.raises(TypeError):
        loaded.metrics["other"] = loaded.metrics["train.loss"]  # type: ignore[index]
    invalid = declaration()
    invalid["extra"] = True
    with pytest.raises(WorkerObservabilityError, match="inexact"):
        load_observability_declaration(invalid)
    invalid = declaration()
    invalid["metrics"] = [*invalid["metrics"], *invalid["metrics"]]  # type: ignore[index]
    with pytest.raises(WorkerObservabilityError, match="semantics"):
        load_observability_declaration(invalid)
    invalid = declaration()
    invalid["metrics"][0]["type"] = []  # type: ignore[index]
    with pytest.raises(WorkerObservabilityError, match="semantics"):
        load_observability_declaration(invalid)


def test_heartbeat_cadence_and_metric_declaration_own_wire_semantics() -> None:
    session = FakeSession()
    clock = iter((1_000, 2_000, 5_000_001_001)).__next__
    observer = WorkerObservability(
        session, load_observability_declaration(declaration()), monotonic_ns=clock
    )
    assert observer.optimizer_step(1) == 1
    assert observer.optimizer_step(2) is None
    assert observer.optimizer_step(3) == 2
    assert observer.publish_if_declared("not.selected", 3.0, step=3) is None
    assert observer.publish_if_declared(
        "train.loss", 1.25, step=3, sample_weight=8, labels={"route": "photo"}
    ) == 1
    assert session.heartbeats == [
        (1, "train", {"execution_phase": None, "wait": False}),
        (3, "train", {"execution_phase": None, "wait": False}),
    ]
    assert session.metrics == [
        (
            ("train.loss", 1.25),
            {
                "unit": "dimensionless",
                "step_domain": "optimizer_step",
                "step": 3,
                "sample_weight": 8,
                "labels": {"route": "photo"},
                "wait": False,
            },
        )
    ]


def test_observability_rejects_invalid_runtime_samples() -> None:
    observer = observability_from_invocation(
        FakeSession(), SimpleNamespace(observability=declaration())
    )
    with pytest.raises(WorkerObservabilityError, match="finite"):
        observer.publish_if_declared("train.loss", float("nan"), step=1)
    with pytest.raises(WorkerObservabilityError, match="uint64"):
        observer.optimizer_step(-1)


def test_blocking_phase_keepalive_publishes_until_the_phase_returns() -> None:
    selected = declaration()
    selected["heartbeat_seconds"] = 1
    session = FakeSession()
    observer = WorkerObservability(
        session, load_observability_declaration(selected)
    )

    with observer.keepalive(7, "loading"):
        time.sleep(1.1)

    assert len(session.heartbeats) >= 2
    assert {heartbeat[:2] for heartbeat in session.heartbeats} == {(7, "loading")}
