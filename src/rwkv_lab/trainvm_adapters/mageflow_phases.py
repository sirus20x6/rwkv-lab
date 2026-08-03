"""Family bridge from TrainVM phases to disposable MageFlow workloads."""

from __future__ import annotations

import random
from collections.abc import Callable, Mapping
from typing import Any

import torch

from rwkv_lab.trainvm_worker import (
    ExecutionPhase,
    WorkerExecutionPhases,
    torch_trajectory_state,
)


def run_mageflow_execution_phases(
    phases: WorkerExecutionPhases,
    *,
    trajectory_model: Any,
    optimizer: Any,
    optimizer_step: int,
    disabled_compile_report: Mapping[str, Any],
    compile_workload: Callable[[], Mapping[str, Any]],
    training_workload: Callable[[], None],
    extra_state: Callable[[], Mapping[str, Any]],
    synchronize: Callable[[], None] = lambda: None,
) -> Mapping[str, Any]:
    """Run cold compile and exact-count warmup without advancing training.

    MageFlow owns construction of one real-shaped forward/backward workload.
    This bridge owns the reusable invariants: RNG preservation, gradient
    disposal, exact step accounting, complete Torch state proofs, and the lazy
    compiler trigger performed inside the compile receipt.
    """

    compile_report = disabled_compile_report

    def snapshot() -> Mapping[str, Any]:
        return torch_trajectory_state(
            trajectory_model,
            optimizer,
            optimizer_step=optimizer_step,
            extra=extra_state(),
        )

    def disposable_workloads(
        count: int,
        mark_step: Callable[[], None],
        *,
        count_steps: bool,
    ) -> None:
        python_rng = random.getstate()
        torch_rng = torch.get_rng_state()
        cuda_rng = (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        )
        try:
            for _ in range(count):
                optimizer.zero_grad(set_to_none=True)
                training_workload()
                synchronize()
                optimizer.zero_grad(set_to_none=True)
                if count_steps:
                    mark_step()
        finally:
            optimizer.zero_grad(set_to_none=True)
            torch.set_rng_state(torch_rng)
            if cuda_rng is not None:
                torch.cuda.set_rng_state_all(cuda_rng)
            random.setstate(python_rng)

    def compile_phase(_steps: int, mark_step: Callable[[], None]) -> None:
        nonlocal compile_report
        compile_report = compile_workload()
        # Torch regional compilation is lazy. One real-shaped disposable
        # forward/backward forces the cold graphs before this receipt closes.
        disposable_workloads(1, mark_step, count_steps=False)

    phases.run(
        ExecutionPhase.COMPILE,
        snapshot=snapshot,
        execute=compile_phase,
    )
    phases.run(
        ExecutionPhase.WARMUP,
        snapshot=snapshot,
        execute=lambda count, mark_step: disposable_workloads(
            count, mark_step, count_steps=True
        ),
    )
    return compile_report


__all__ = ["run_mageflow_execution_phases"]
