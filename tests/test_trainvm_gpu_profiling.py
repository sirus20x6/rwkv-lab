from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from rwkv_lab.trainvm_worker.profiling import (
    GPU_TRACE_SCHEMA,
    GpuProfileError,
    GpuTracePublisher,
    NullStepProfiler,
    trace_request_from_invocation,
)


def digest(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def execution(**updates: object) -> dict[str, object]:
    trace: dict[str, object] = {
        "enabled": True,
        "backend": "torch",
        "warmup_steps": 2,
        "skip_steps": 1,
        "capture_steps": 3,
        "output_artifact": "gpu_trace",
        "activities": ["cpu", "accelerator"],
        "record_shapes": True,
        "profile_memory": True,
        "with_stack": False,
    }
    trace.update(updates)
    return {
        "component": "trainer",
        "operation": "train",
        "gpu_trace": trace,
    }


class FakeSession:
    def __init__(self, root: Path) -> None:
        self.bootstrap = SimpleNamespace(
            run_id="run-1", node_id="train", attempt_id="train@1"
        )
        self.invocation = SimpleNamespace(
            invocation_digest=digest(b"invocation"),
            execution=execution(),
            workspace={
                "run_directory": str(root),
                "allowed_write_roots": [str(root.parent)],
            },
            publishes={
                "gpu_trace": {
                    "logical_name": "gpu_trace",
                    "declaration": {
                        "type": "opaque",
                        "schema": GPU_TRACE_SCHEMA,
                        "immutability": "mutable_until_publish",
                        "fingerprint": "adapter",
                    },
                }
            },
        )
        self.artifacts: list[dict[str, object]] = []

    def artifact(self, **values: object) -> int:
        self.artifacts.append(values)
        return len(self.artifacts)


def test_gpu_trace_request_is_closed_bounded_and_step_domain_explicit() -> None:
    request = trace_request_from_invocation(SimpleNamespace(execution=execution()))
    assert request is not None
    assert request.total_steps == 6
    assert request.activities == ("cpu", "accelerator")

    assert (
        trace_request_from_invocation(
            SimpleNamespace(execution={"gpu_trace": {"enabled": False}})
        )
        is None
    )
    for update, diagnostic in (
        ({"backend": "shell"}, "backend"),
        ({"capture_steps": 129}, "bound"),
        ({"activities": ["cpu"]}, "activities"),
        ({"unknown": True}, "inexact"),
    ):
        with pytest.raises(GpuProfileError, match=diagnostic):
            trace_request_from_invocation(
                SimpleNamespace(execution=execution(**update))
            )


def test_trace_publisher_freezes_manifest_trace_and_receipt(tmp_path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    session = FakeSession(run)
    request = trace_request_from_invocation(session.invocation)
    assert request is not None
    publisher = GpuTracePublisher(session, request)
    raw = publisher.staging_root / "raw.json"
    raw.write_text('{"traceEvents":[]}', encoding="utf-8")

    published = publisher.publish(
        raw_trace=raw,
        optimizer_steps=(4, 5, 6),
        summary={
            "accelerator_time_us": 12.5,
            "cpu_time_us": 8.0,
            "kernel_or_operator_count": 1,
            "top_operators": [
                {
                    "accelerator_time_us": 12.5,
                    "calls": 1,
                    "cpu_time_us": 8.0,
                    "name": "train_step",
                }
            ],
        },
    )
    manifest = json.loads(published.manifest_path.read_text(encoding="utf-8"))
    trace = published.manifest_path.with_name("trace.json")
    assert manifest["api_version"] == GPU_TRACE_SCHEMA
    assert manifest["first_optimizer_step"] == 4
    assert manifest["last_optimizer_step"] == 6
    assert manifest["sensitivity"] == "restricted"
    assert manifest["instrumented_timing"] is True
    assert digest(trace.read_bytes()) == manifest["trace_sha256"]
    assert published.manifest_sha256 == digest(published.manifest_path.read_bytes())
    assert not raw.exists()
    assert session.artifacts == [
        {
            "artifact_id": published.artifact_id,
            "logical_name": "gpu_trace",
            "kind": 7,
            "schema": GPU_TRACE_SCHEMA,
            "uri": published.manifest_path.resolve().as_uri(),
            "size_bytes": published.size_bytes,
            "fingerprint_algorithm": "adapter",
            "fingerprint": published.manifest_sha256,
            "parent_artifact_ids": (),
            "wait": True,
        }
    ]


def test_trace_publication_rejects_partial_windows_and_path_escape(tmp_path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    session = FakeSession(run)
    request = trace_request_from_invocation(session.invocation)
    assert request is not None
    publisher = GpuTracePublisher(session, request)
    outside = tmp_path / "outside.json"
    outside.write_text("{}", encoding="utf-8")
    with pytest.raises(GpuProfileError, match="outside private staging"):
        publisher.publish(
            raw_trace=outside,
            optimizer_steps=(4, 5, 6),
            summary={},
        )
    raw = publisher.staging_root / "partial.json"
    raw.write_text("{}", encoding="utf-8")
    with pytest.raises(GpuProfileError, match="window is incomplete"):
        publisher.publish(raw_trace=raw, optimizer_steps=(4, 5), summary={})


def test_null_profiler_keeps_nonprofiled_workers_free_of_runtime_dependencies() -> None:
    with NullStepProfiler() as profiler:
        profiler.step(1)
        profiler.step(2)
    with pytest.raises(GpuProfileError, match="integer"):
        profiler.step(True)
