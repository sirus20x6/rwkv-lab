from __future__ import annotations

import random

import pytest

torch = pytest.importorskip("torch")

from rwkv_lab.trainvm_adapters.mageflow_phases import (
    run_mageflow_execution_phases,
)
from rwkv_lab.trainvm_worker import (
    ExecutionPhase,
    ExecutionPhaseDisposition,
    ExecutionPhaseRequest,
    WorkerExecutionPhaseError,
    WorkerExecutionPhases,
)


class _Channel:
    def __init__(self) -> None:
        self.receipts = []
        self.execution_phase_cancellation: str | None = None

    def execution_phase_receipt(self, request, disposition, **values):
        self.receipts.append((request, disposition, values))
        return len(self.receipts)


def _phases(channel: _Channel) -> WorkerExecutionPhases:
    return WorkerExecutionPhases(
        channel,
        (
            ExecutionPhaseRequest(
                ExecutionPhase.COMPILE,
                True,
                None,
                "sha256:" + "1" * 64,
            ),
            ExecutionPhaseRequest(
                ExecutionPhase.WARMUP,
                True,
                2,
                "sha256:" + "2" * 64,
            ),
        ),
    )


def test_mageflow_bridge_runs_cold_compile_and_exact_disposable_warmup() -> None:
    torch.manual_seed(9)
    random.seed(10)
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3)
    channel = _Channel()
    phases = _phases(channel)
    original = {name: value.detach().clone() for name, value in model.state_dict().items()}
    torch_rng = torch.get_rng_state().clone()
    python_rng = random.getstate()
    calls = {"compile": 0, "training": 0, "synchronize": 0}

    def compile_workload():
        calls["compile"] += 1
        return {"enabled": True, "compiled_blocks": 1}

    def training_workload():
        calls["training"] += 1
        random.random()
        loss = model(torch.rand(2, 3)).sum()
        loss.backward()

    report = run_mageflow_execution_phases(
        phases,
        trajectory_model=model,
        optimizer=optimizer,
        optimizer_step=4,
        disabled_compile_report={"enabled": False},
        compile_workload=compile_workload,
        training_workload=training_workload,
        extra_state=lambda: {"cursor": 4, "scheduler": {"step": 4}},
        synchronize=lambda: calls.__setitem__(
            "synchronize", calls["synchronize"] + 1
        ),
    )

    phases.require_complete()
    assert report == {"enabled": True, "compiled_blocks": 1}
    assert calls == {"compile": 1, "training": 3, "synchronize": 3}
    assert torch.equal(torch.get_rng_state(), torch_rng)
    assert random.getstate() == python_rng
    assert all(parameter.grad is None for parameter in model.parameters())
    for name, value in model.state_dict().items():
        assert torch.equal(value, original[name])
    assert [receipt[1] for receipt in channel.receipts] == [
        ExecutionPhaseDisposition.COMPLETED,
        ExecutionPhaseDisposition.COMPLETED,
    ]
    assert channel.receipts[1][2]["steps_executed"] == 2


def test_mageflow_bridge_fails_if_disposable_work_changes_weights() -> None:
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3)
    channel = _Channel()
    phases = WorkerExecutionPhases(
        channel,
        (
            ExecutionPhaseRequest(
                ExecutionPhase.COMPILE,
                True,
                None,
                "sha256:" + "3" * 64,
            ),
        ),
    )

    def corrupt_trajectory():
        model.weight.data.add_(1)

    with pytest.raises(WorkerExecutionPhaseError, match="restore"):
        run_mageflow_execution_phases(
            phases,
            trajectory_model=model,
            optimizer=optimizer,
            optimizer_step=0,
            disabled_compile_report={"enabled": False},
            compile_workload=lambda: {"enabled": True},
            training_workload=corrupt_trajectory,
            extra_state=lambda: {"cursor": 0},
        )
    assert channel.receipts[0][1] is ExecutionPhaseDisposition.FAILED
