from __future__ import annotations

import json
from dataclasses import asdict
from types import MappingProxyType, SimpleNamespace
from typing import Self

import pytest

from rwkv_lab.trainvm_adapters.content_authority import measure_input_content_root
from rwkv_lab.trainvm_adapters.entrypoint import (
    WORKER_BOOTSTRAP_DESCRIPTOR,
    WorkerEntrypointError,
    main,
    run_worker,
)
from rwkv_lab.trainvm_adapters.handlers import (
    AdapterDispatchError,
    HandlerResult,
    _appearance_expert,
    _rwkv_scratch,
    execute_invocation,
    supported_adapter_keys,
)
from rwkv_lab.trainvm_adapters.io import (
    AdapterInputError,
    WorkspacePathAuthority,
    read_inline_config,
    require_run_directory,
)
from rwkv_lab.trainvm_worker import (
    WorkerCancellationRequested,
    WorkerResourcesReleasedPause,
)


class FakeSession:
    def __init__(self, bootstrap: object, *, completed: bool = False) -> None:
        self.bootstrap = bootstrap
        self.completed_before_connect = completed
        self.invocation = SimpleNamespace(
            run_id="run-1",
            controls={},
            effective_control_revision=0,
            observability={
                "heartbeat_seconds": 5,
                "metrics": [],
                "retain_raw_metrics_days": 1,
            },
        )
        self.finished: list[tuple[str, object, int | None]] = []
        self.heartbeats: list[tuple[int, str]] = []
        self.closed = False

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        self.closed = True

    def finish(
        self, event_type: str, payload: object, *, optimizer_step: int | None = None
    ) -> None:
        self.finished.append((event_type, payload, optimizer_step))

    def heartbeat(self, optimizer_step: int, phase: str, *, wait: bool = False) -> int:
        assert wait is False
        self.heartbeats.append((optimizer_step, phase))
        return len(self.heartbeats)


def _sealed_workspace(read_root, run_directory):
    return {
        "run_directory": str(run_directory),
        "allowed_read_roots": [str(read_root)],
        "input_content_roots": [asdict(measure_input_content_root(read_root))],
        "allowed_write_roots": [str(run_directory.parent)],
    }


def test_inline_config_is_frozen_content_not_a_mutable_path(tmp_path) -> None:
    frozen = MappingProxyType(
        {"config": MappingProxyType({"steps": 12, "buckets": (512, 768)})}
    )
    assert read_inline_config(frozen) == {"steps": 12, "buckets": [512, 768]}
    with pytest.raises(AdapterInputError, match="inline object"):
        read_inline_config({"config": str(tmp_path / "mutable.json")})
    with pytest.raises(AdapterInputError, match="exactly one"):
        read_inline_config({"config": {}, "argv": ["--import", "anything"]})


def test_run_directory_must_equal_the_authority_workspace(tmp_path) -> None:
    run_directory = tmp_path / "run"
    run_directory.mkdir()
    workspace = {
        "run_directory": str(run_directory),
        "allowed_read_roots": [str(tmp_path)],
        "allowed_write_roots": [str(tmp_path)],
    }
    assert require_run_directory(str(run_directory), workspace) == run_directory
    with pytest.raises(AdapterInputError, match="workspace authority"):
        require_run_directory(str(tmp_path / "other"), workspace)
    with pytest.raises(AdapterInputError, match="absolute normalized"):
        require_run_directory("relative/run", workspace)


def test_workspace_path_authority_confines_reads_writes_and_symlinks(tmp_path) -> None:
    read_root = tmp_path / "read"
    write_root = tmp_path / "write"
    outside = tmp_path / "outside"
    for directory in (read_root, write_root, outside):
        directory.mkdir()
    source = read_root / "manifest.jsonl"
    source.write_text("{}\n", encoding="utf-8")
    (read_root / "escape").symlink_to(outside, target_is_directory=True)
    authority = WorkspacePathAuthority.from_workspace(
        {
            "run_directory": str(write_root / "run"),
            "allowed_read_roots": [str(read_root)],
            "allowed_write_roots": [str(write_root)],
        }
    )

    assert authority.read_path(str(source), label="manifest", kind="file") == source
    assert (
        authority.write_directory(str(write_root / "cache" / "new"), label="cache")
        == write_root / "cache" / "new"
    )
    with pytest.raises(AdapterInputError, match="outside declared read roots"):
        authority.read_path(
            str(read_root / "escape"), label="dataset", kind="directory"
        )
    with pytest.raises(AdapterInputError, match="outside declared write roots"):
        authority.write_directory(str(outside / "cache"), label="cache")


def test_workspace_reads_require_declared_content_coverage(tmp_path) -> None:
    read_root = tmp_path / "read"
    write_root = tmp_path / "write"
    read_root.mkdir()
    write_root.mkdir()
    sealed = read_root / "sealed.bin"
    uncovered = read_root / "uncovered.bin"
    sealed.write_bytes(b"sealed")
    uncovered.write_bytes(b"uncovered")
    authority = WorkspacePathAuthority.from_workspace(
        {
            "run_directory": str(write_root / "run"),
            "allowed_read_roots": [str(read_root)],
            "input_content_roots": (
                MappingProxyType(asdict(measure_input_content_root(sealed))),
            ),
            "allowed_write_roots": [str(write_root)],
        },
        require_content=True,
    )
    assert authority.read_path(str(sealed), label="sealed", kind="file") == sealed
    with pytest.raises(AdapterInputError, match="not covered"):
        authority.read_path(str(uncovered), label="uncovered", kind="file")


def test_packed_manifest_cannot_escape_its_verified_directory(tmp_path) -> None:
    read_root = tmp_path / "read"
    pack = read_root / "pack"
    write_root = tmp_path / "write"
    pack.mkdir(parents=True)
    write_root.mkdir()
    (read_root / "outside.bin").write_bytes(b"packed")
    (pack / "manifest.json").write_text(
        json.dumps({"packed_file": "../outside.bin"}), encoding="utf-8"
    )
    authority = WorkspacePathAuthority.from_workspace(
        {
            "run_directory": str(write_root / "run"),
            "allowed_read_roots": [str(read_root)],
            "input_content_roots": [asdict(measure_input_content_root(read_root))],
            "allowed_write_roots": [str(write_root)],
        },
        require_content=True,
    )
    with pytest.raises(AdapterInputError, match="escapes"):
        authority.verify_json_relative_file_reference(
            pack,
            manifest_name="manifest.json",
            field="packed_file",
            label="train pack",
        )


def test_jsonl_reference_scan_rejects_an_oversized_line(tmp_path) -> None:
    read_root = tmp_path / "read"
    write_root = tmp_path / "write"
    read_root.mkdir()
    write_root.mkdir()
    manifest = read_root / "oversized.jsonl"
    manifest.write_bytes(b"x" * (4 * 1024 * 1024 + 1))
    authority = WorkspacePathAuthority.from_workspace(
        {
            "run_directory": str(write_root / "run"),
            "allowed_read_roots": [str(read_root)],
            "input_content_roots": [asdict(measure_input_content_root(read_root))],
            "allowed_write_roots": [str(write_root)],
        },
        require_content=True,
    )

    with pytest.raises(AdapterInputError, match="line 1 exceeds its size bound"):
        authority.verify_jsonl_file_references(
            manifest,
            fields=("image", "image_path"),
            label="training manifest",
        )


def test_appearance_handler_passes_only_canonical_authorized_paths(
    tmp_path, monkeypatch
) -> None:
    from rwkv_lab import mage_flow_expert_train

    read_root = tmp_path / "read"
    run_directory = tmp_path / "write" / "run"
    read_root.mkdir()
    run_directory.mkdir(parents=True)
    manifest = read_root / "train.jsonl"
    image = read_root / "image.png"
    image.write_bytes(b"image")
    manifest.write_text(json.dumps({"image": str(image)}) + "\n", encoding="utf-8")
    observed = []
    monkeypatch.setattr(
        mage_flow_expert_train,
        "train",
        lambda config, *, worker_components, worker_step_profiler, worker_observability, worker_controls: (
            observed.append(
                (
                    config,
                    worker_components,
                    worker_step_profiler,
                    worker_observability,
                    worker_controls,
                )
            )
        ),
    )
    invocation = SimpleNamespace(
        inputs={
            "config": {
                "train_manifest": str(manifest),
                "output_dir": str(run_directory),
            }
        },
        workspace=_sealed_workspace(read_root, run_directory),
    )
    components = SimpleNamespace()
    observability = SimpleNamespace()
    controls = SimpleNamespace(
        effective_values={
            "learning_rate": 2.0e-5,
            "eval_every": 25,
            "caption_dropout": 0.2,
            "mixed_precision": "bf16",
        }
    )

    assert _appearance_expert(
        invocation, components, observability=observability, controls=controls
    ) == HandlerResult("worker.completed", {"reason": "training_complete"})
    assert observed[0][0].train_manifest == str(manifest.resolve())
    assert observed[0][0].output_dir == str(run_directory.resolve())
    assert observed[0][1] is components
    assert observed[0][3] is observability
    assert observed[0][4] is controls
    assert observed[0][0].learning_rate == pytest.approx(2.0e-5)
    assert observed[0][0].eval_every == 25
    assert observed[0][0].caption_dropout == pytest.approx(0.2)


def test_family_handler_rejects_same_path_mutation_before_trainer_call(
    tmp_path, monkeypatch
) -> None:
    from rwkv_lab import mage_flow_expert_train

    read_root = tmp_path / "read"
    run_directory = tmp_path / "write" / "run"
    read_root.mkdir()
    run_directory.mkdir(parents=True)
    manifest = read_root / "train.jsonl"
    image = read_root / "image.png"
    image.write_bytes(b"image")
    manifest.write_text(json.dumps({"image": str(image)}) + "\n", encoding="utf-8")
    workspace = _sealed_workspace(read_root, run_directory)
    manifest.write_text('{"changed":true}\n', encoding="utf-8")
    called = False

    def train(*_args, **_kwargs):
        nonlocal called
        called = True

    monkeypatch.setattr(mage_flow_expert_train, "train", train)
    invocation = SimpleNamespace(
        inputs={
            "config": {
                "train_manifest": str(manifest),
                "output_dir": str(run_directory),
            }
        },
        workspace=workspace,
    )
    with pytest.raises(AdapterInputError, match="content identity verification"):
        _appearance_expert(invocation, SimpleNamespace())
    assert called is False


def test_mageflow_manifest_cannot_reference_unsealed_payload(
    tmp_path, monkeypatch
) -> None:
    from rwkv_lab import mage_flow_expert_train

    read_root = tmp_path / "read"
    run_directory = tmp_path / "write" / "run"
    read_root.mkdir()
    run_directory.mkdir(parents=True)
    image = read_root / "outside-root.png"
    image.write_bytes(b"image")
    manifest = read_root / "train.jsonl"
    manifest.write_text(json.dumps({"image": str(image)}) + "\n", encoding="utf-8")
    called = False

    def train(*_args, **_kwargs):
        nonlocal called
        called = True

    monkeypatch.setattr(mage_flow_expert_train, "train", train)
    invocation = SimpleNamespace(
        inputs={
            "config": {
                "train_manifest": str(manifest),
                "output_dir": str(run_directory),
            }
        },
        workspace={
            "run_directory": str(run_directory),
            "allowed_read_roots": [str(read_root)],
            "input_content_roots": [asdict(measure_input_content_root(manifest))],
            "allowed_write_roots": [str(run_directory.parent)],
        },
    )
    with pytest.raises(AdapterInputError, match="not covered"):
        _appearance_expert(invocation, SimpleNamespace())
    assert called is False


def test_appearance_handler_returns_declared_immutable_checkpoint_request(
    tmp_path, monkeypatch
) -> None:
    from rwkv_lab import mage_flow_expert_train

    read_root = tmp_path / "read"
    run_directory = tmp_path / "write" / "run"
    read_root.mkdir()
    run_directory.mkdir(parents=True)
    manifest = read_root / "train.jsonl"
    image = read_root / "image.png"
    image.write_bytes(b"image")
    manifest.write_text(json.dumps({"image": str(image)}) + "\n", encoding="utf-8")

    def train(config, **_kwargs):
        checkpoint = run_directory / "checkpoint-00000023"
        checkpoint.mkdir()
        (checkpoint / "state.pt").write_bytes(b"state")
        (run_directory / "complete.json").write_text(
            json.dumps(
                {
                    "global_step": 23,
                    "checkpoint": str(checkpoint),
                }
            ),
            encoding="utf-8",
        )

    monkeypatch.setattr(mage_flow_expert_train, "train", train)
    invocation = SimpleNamespace(
        inputs={
            "config": {
                "train_manifest": str(manifest),
                "output_dir": str(run_directory),
            }
        },
        workspace=_sealed_workspace(read_root, run_directory),
        publishes={"checkpoint": {}},
    )

    result = _appearance_expert(
        invocation, SimpleNamespace(), observability=SimpleNamespace()
    )
    assert result.optimizer_step == 23
    assert result.payload == {"reason": "training_complete"}
    assert len(result.checkpoint_requests) == 1
    request = result.checkpoint_requests[0]
    assert request.source_directory == run_directory / "checkpoint-00000023"
    assert request.optimizer_step == 23
    assert request.resume_grade == "compatible"
    assert request.state_components[0] == "component_composition"


def test_rwkv_scratch_handler_lowers_only_typed_arguments_and_terminal_checkpoint(
    tmp_path, monkeypatch
) -> None:
    from rwkv_lab import rwkv_pretrain

    read_root = tmp_path / "read"
    run_directory = tmp_path / "write" / "run"
    read_root.mkdir()
    run_directory.parent.mkdir()
    corpus = read_root / "tokens.bin"
    corpus.write_bytes(b"\x00\x00" * 1024)
    observed = []

    def train(arguments, **kwargs):
        observed.append((arguments, kwargs))
        checkpoint = run_directory / "checkpoint-final" / "state.pt"
        checkpoint.write_bytes(b"checkpoint")
        return {"checkpoint": str(checkpoint), "step": 120}

    monkeypatch.setattr(rwkv_pretrain, "main", train)
    invocation = SimpleNamespace(
        inputs={
            "config": {
                "data": str(corpus),
                "output_dir": str(run_directory),
                "steps": 120,
            }
        },
        workspace=_sealed_workspace(read_root, run_directory),
        publishes={"checkpoint": {}},
    )
    components = SimpleNamespace()
    profiler = SimpleNamespace()
    observability = SimpleNamespace()
    controls = SimpleNamespace()

    result = _rwkv_scratch(
        invocation,
        components,
        step_profiler=profiler,
        observability=observability,
        controls=controls,
    )
    arguments, keyword_arguments = observed[0]
    assert arguments[arguments.index("--data") + 1] == str(corpus.resolve())
    assert arguments[arguments.index("--out") + 1] == str(run_directory.resolve())
    assert arguments[arguments.index("--optimizer") + 1] == "adamw"
    assert arguments[arguments.index("--lr-schedule") + 1] == "powercool"
    assert "--compile" not in arguments
    assert keyword_arguments == {
        "worker_components": components,
        "worker_step_profiler": profiler,
        "worker_observability": observability,
        "worker_controls": controls,
    }
    assert result.optimizer_step == 120
    assert len(result.checkpoint_requests) == 1
    request = result.checkpoint_requests[0]
    assert request.source_directory == run_directory / "checkpoint-final"
    assert request.resume_grade == "terminal_checkpoint"


def test_dispatch_table_is_closed_and_training_composition_is_required() -> None:
    assert supported_adapter_keys() == {
        (
            "rwkv-lab.mageflow-appearance-expert",
            "1.0.0",
            "train",
            "rwkv_lab.mageflow_appearance_expert.v1.Train",
        ),
        (
            "rwkv-lab.mageflow-terminal-expert",
            "1.0.0",
            "train",
            "rwkv_lab.mageflow_terminal_expert.v1.Train",
        ),
        (
            "rwkv-lab.qwen-ao3",
            "1.0.0",
            "train",
            "rwkv_lab.qwen_ao3.v1.Train",
        ),
        (
            "rwkv-lab.rwkv-scratch",
            "1.0.0",
            "train",
            "rwkv_lab.rwkv_scratch.v1.Train",
        ),
    }
    unknown = SimpleNamespace(
        adapter={
            "adapter": "user.supplied.module",
            "version": "1.0.0",
            "operation": "train",
            "contract": "arbitrary.Import",
        },
        training=None,
    )
    with pytest.raises(AdapterDispatchError, match="closed adapter handler"):
        execute_invocation(unknown)

    supported = SimpleNamespace(
        adapter={
            "adapter": "rwkv-lab.qwen-ao3",
            "version": "1.0.0",
            "operation": "train",
            "contract": "rwkv_lab.qwen_ao3.v1.Train",
        },
        training=None,
    )
    with pytest.raises(AdapterDispatchError, match="no resolved composition"):
        execute_invocation(supported)


def test_runner_reports_success_with_optimizer_step() -> None:
    bootstrap = SimpleNamespace(run_id="run-1")
    sessions: list[FakeSession] = []

    def factory(value: object) -> FakeSession:
        assert value is bootstrap
        session = FakeSession(value)
        sessions.append(session)
        return session

    status = run_worker(
        bootstrap_reader=lambda descriptor: (
            bootstrap
            if descriptor == WORKER_BOOTSTRAP_DESCRIPTOR
            else pytest.fail("wrong descriptor")
        ),
        session_factory=factory,
        executor=lambda invocation, _profiler, _observability, _controls: HandlerResult(
            "worker.completed", {"reason": "training_complete"}, 41
        ),
    )
    assert status == 0
    assert sessions[0].finished == [
        ("worker.completed", {"reason": "training_complete"}, 41)
    ]
    assert sessions[0].heartbeats == [(0, "initializing")]
    assert sessions[0].closed


def test_runner_publishes_handler_checkpoints_before_terminal_event(
    monkeypatch, tmp_path
) -> None:
    from rwkv_lab.trainvm_worker import CheckpointPublicationRequest

    bootstrap = SimpleNamespace(run_id="run-1")
    session = FakeSession(bootstrap)
    request = CheckpointPublicationRequest(
        tmp_path,
        optimizer_step=19,
        resume_grade="compatible",
        state_components=("model",),
    )
    observed = []

    def publish(actual_session, requests, *, progress):
        assert actual_session is session
        observed.extend(requests)
        progress(19)
        return (SimpleNamespace(artifact_id="checkpoint-artifact-19"),)

    monkeypatch.setattr(
        "rwkv_lab.trainvm_adapters.entrypoint.publish_checkpoint_requests", publish
    )
    status = run_worker(
        bootstrap_reader=lambda _descriptor: bootstrap,
        session_factory=lambda _bootstrap: session,
        executor=lambda _invocation, _profiler, _observability, _controls: (
            HandlerResult(
                "worker.completed",
                {"reason": "training_complete"},
                19,
                (request,),
            )
        ),
    )
    assert status == 0
    assert observed == [request]
    assert session.finished == [
        (
            "worker.completed",
            {
                "reason": "training_complete",
                "checkpoint_artifact_ids": ["checkpoint-artifact-19"],
            },
            19,
        )
    ]


def test_runner_reports_sanitized_failure_and_skips_completed_replay() -> None:
    bootstrap = SimpleNamespace(run_id="run-1")
    failed = FakeSession(bootstrap)

    def raise_secret(
        _invocation: object,
        _profiler: object,
        _observability: object,
        _controls: object,
    ) -> HandlerResult:
        raise RuntimeError("secret dataset path")

    assert (
        run_worker(
            bootstrap_reader=lambda _descriptor: bootstrap,
            session_factory=lambda _bootstrap: failed,
            executor=raise_secret,
        )
        == 1
    )
    assert failed.finished == [
        (
            "operation.failed",
            {"code": "adapter_execution_failed", "error_type": "RuntimeError"},
            None,
        )
    ]
    assert "secret" not in repr(failed.finished)

    completed = FakeSession(bootstrap, completed=True)
    assert (
        run_worker(
            bootstrap_reader=lambda _descriptor: bootstrap,
            session_factory=lambda _bootstrap: completed,
            executor=lambda _invocation, _profiler, _observability, _controls: (
                pytest.fail("replayed completed work")
            ),
        )
        == 0
    )
    assert completed.finished == []


@pytest.mark.parametrize(
    "lifecycle_exit",
    [WorkerCancellationRequested, WorkerResourcesReleasedPause],
)
def test_runner_treats_acknowledged_lifecycle_exit_as_clean_process_exit(
    lifecycle_exit: type[RuntimeError],
) -> None:
    bootstrap = SimpleNamespace(run_id="run-1")
    session = FakeSession(bootstrap)

    def cancel_at_safe_point(*_args: object) -> HandlerResult:
        raise lifecycle_exit("operator requested")

    assert (
        run_worker(
            bootstrap_reader=lambda _descriptor: bootstrap,
            session_factory=lambda _bootstrap: session,
            executor=cancel_at_safe_point,
        )
        == 0
    )
    assert session.finished == []
    assert session.closed


def test_cli_accepts_only_the_authority_descriptor(monkeypatch) -> None:
    observed: list[int] = []
    monkeypatch.setattr(
        "rwkv_lab.trainvm_adapters.entrypoint.run_worker",
        lambda descriptor: observed.append(descriptor) or 0,
    )
    assert main(["--trainvm-bootstrap-fd=4"]) == 0
    assert observed == [4]
    for arguments in ([], ["--trainvm-bootstrap-fd=5"], ["--help"]):
        with pytest.raises(WorkerEntrypointError, match="only its fixed"):
            main(arguments)
