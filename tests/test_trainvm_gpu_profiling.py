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
    _input_stall_summary,
    _interval_union_duration,
    _profile_activity_summary,
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


def test_profile_activity_summary_counts_launches_and_unions_gpu_time() -> None:
    class Device:
        def __init__(self, name: str) -> None:
            self.name = name

    def event(key: str, device: str, start: float, end: float) -> SimpleNamespace:
        return SimpleNamespace(
            key=key,
            device_type=Device(device),
            time_range=SimpleNamespace(start=start, end=end),
        )

    profile = SimpleNamespace(
        events=lambda: [
            event("ProfilerStep*", "CPU", 100.0, 200.0),
            event("ProfilerStep*", "CPU", 210.0, 300.0),
            event("kernel-a", "CUDA", 90.0, 130.0),
            event("kernel-b", "CUDA", 120.0, 180.0),
            event("kernel-c", "CUDA", 250.0, 320.0),
            event("aten::add", "CPU", 120.0, 140.0),
        ]
    )
    assert _interval_union_duration([(2.0, 4.0), (1.0, 3.0), (8.0, 9.0)]) == 4.0
    assert _profile_activity_summary(profile) == {
        "accelerator_launch_count": 3,
        "captured_step_wall_time_us": 200.0,
        "gpu_active_ratio": 0.65,
        "gpu_active_time_us": 130.0,
    }


def test_profile_activity_summary_requires_optimizer_step_window() -> None:
    profile = SimpleNamespace(events=list)
    with pytest.raises(GpuProfileError, match="optimizer-step intervals"):
        _profile_activity_summary(profile)


def test_input_stall_summary_requires_complete_explicit_boundaries() -> None:
    assert _input_stall_summary([10.0, 20.0], 100.0) == {
        "input_stall_ratio": 0.3,
        "input_stall_time_us": 30.0,
    }
    assert _input_stall_summary([None, None], 100.0) == {}
    with pytest.raises(GpuProfileError, match="incomplete"):
        _input_stall_summary([10.0, None], 100.0)
    with pytest.raises(GpuProfileError, match="exceeds"):
        _input_stall_summary([101.1], 100.0)


def test_null_profiler_input_boundaries_are_transparent() -> None:
    profiler = NullStepProfiler()
    with profiler.input_wait():
        value = 3
    assert value == 3
    assert list(profiler.track_input([1, 2])) == [1, 2]
