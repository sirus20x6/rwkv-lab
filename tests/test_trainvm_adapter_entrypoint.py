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
    execute_invocation,
    supported_adapter_keys,
)
from rwkv_lab.trainvm_adapters.io import (
    AdapterInputError,
    read_inline_config,
    require_run_directory,
)


class FakeSession:
    def __init__(self, bootstrap: object, *, completed: bool = False) -> None:
        self.bootstrap = bootstrap
        self.completed_before_connect = completed
        self.invocation = SimpleNamespace(run_id="run-1")
        self.finished: list[tuple[str, object, int | None]] = []
        self.closed = False

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        self.closed = True

    def finish(
        self, event_type: str, payload: object, *, optimizer_step: int | None = None
    ) -> None:
        self.finished.append((event_type, payload, optimizer_step))


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
    assert require_run_directory(
        str(run_directory), {"run_directory": str(run_directory)}
    ) == run_directory
    with pytest.raises(AdapterInputError, match="workspace authority"):
        require_run_directory(
            str(tmp_path / "other"), {"run_directory": str(run_directory)}
        )
    with pytest.raises(AdapterInputError, match="workspace authority"):
        require_run_directory("relative/run", {"run_directory": "relative/run"})


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
        executor=lambda invocation: HandlerResult(
            "worker.completed", {"reason": "training_complete"}, 41
        ),
    )
    assert status == 0
    assert sessions[0].finished == [
        ("worker.completed", {"reason": "training_complete"}, 41)
    ]
    assert sessions[0].closed


def test_runner_reports_sanitized_failure_and_skips_completed_replay() -> None:
    bootstrap = SimpleNamespace(run_id="run-1")
    failed = FakeSession(bootstrap)

    def raise_secret(_invocation: object) -> HandlerResult:
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
            executor=lambda _invocation: pytest.fail("replayed completed work"),
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
