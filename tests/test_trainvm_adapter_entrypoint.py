from __future__ import annotations

from types import MappingProxyType, SimpleNamespace
from typing import Self

import pytest

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
    execute_invocation,
    supported_adapter_keys,
)
from rwkv_lab.trainvm_adapters.io import (
    AdapterInputError,
    WorkspacePathAuthority,
    read_inline_config,
    require_run_directory,
)


class FakeSession:
    def __init__(self, bootstrap: object, *, completed: bool = False) -> None:
        self.bootstrap = bootstrap
        self.completed_before_connect = completed
        self.invocation = SimpleNamespace(
            run_id="run-1",
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


def test_appearance_handler_passes_only_canonical_authorized_paths(
    tmp_path, monkeypatch
) -> None:
    from rwkv_lab import mage_flow_expert_train

    read_root = tmp_path / "read"
    run_directory = tmp_path / "write" / "run"
    read_root.mkdir()
    run_directory.mkdir(parents=True)
    manifest = read_root / "train.jsonl"
    manifest.write_text("{}\n", encoding="utf-8")
    observed = []
    monkeypatch.setattr(
        mage_flow_expert_train,
        "train",
        lambda config, *, worker_components, worker_step_profiler, worker_observability: observed.append(
            (config, worker_components, worker_step_profiler, worker_observability)
        ),
    )
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
            "allowed_write_roots": [str(run_directory.parent)],
        },
    )
    components = SimpleNamespace()
    observability = SimpleNamespace()

    assert _appearance_expert(
        invocation, components, observability=observability
    ) == HandlerResult(
        "worker.completed", {"reason": "training_complete"}
    )
    assert observed[0][0].train_manifest == str(manifest.resolve())
    assert observed[0][0].output_dir == str(run_directory.resolve())
    assert observed[0][1] is components
    assert observed[0][3] is observability


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
        executor=lambda invocation, _profiler, _observability: HandlerResult(
            "worker.completed", {"reason": "training_complete"}, 41
        ),
    )
    assert status == 0
    assert sessions[0].finished == [
        ("worker.completed", {"reason": "training_complete"}, 41)
    ]
    assert sessions[0].heartbeats == [(0, "initializing")]
    assert sessions[0].closed


def test_runner_reports_sanitized_failure_and_skips_completed_replay() -> None:
    bootstrap = SimpleNamespace(run_id="run-1")
    failed = FakeSession(bootstrap)

    def raise_secret(
        _invocation: object, _profiler: object, _observability: object
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
            executor=lambda _invocation, _profiler, _observability: pytest.fail(
                "replayed completed work"
            ),
        )
        == 0
    )
    assert completed.finished == []


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
