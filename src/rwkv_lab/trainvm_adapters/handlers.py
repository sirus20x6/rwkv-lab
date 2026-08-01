from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from rwkv_lab.trainvm_worker import WorkerInvocation

from .components import WorkerTrainingComponents
from .io import read_inline_config, require_run_directory


class AdapterDispatchError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class HandlerResult:
    event_type: str
    payload: Mapping[str, Any]
    optimizer_step: int | None = None


AdapterKey = tuple[str, str, str, str]
Handler = Callable[[WorkerInvocation, WorkerTrainingComponents], HandlerResult]


def _appearance_expert(
    invocation: WorkerInvocation, components: WorkerTrainingComponents
) -> HandlerResult:
    from rwkv_lab.mage_flow_expert_train import MageFlowExpertTrainConfig, train

    config = MageFlowExpertTrainConfig(**read_inline_config(invocation.inputs))
    require_run_directory(config.output_dir, invocation.workspace)
    train(config, worker_components=components)
    return HandlerResult("worker.completed", {"reason": "training_complete"})


def _terminal_expert(
    invocation: WorkerInvocation, components: WorkerTrainingComponents
) -> HandlerResult:
    from rwkv_lab.mage_flow_terminal_train import TerminalExpertTrainConfig, train

    config = TerminalExpertTrainConfig(**read_inline_config(invocation.inputs))
    require_run_directory(config.output_dir, invocation.workspace)
    train(config, worker_components=components)
    return HandlerResult("worker.completed", {"reason": "training_complete"})


def _qwen_ao3(
    invocation: WorkerInvocation, components: WorkerTrainingComponents
) -> HandlerResult:
    from rwkv_lab.qwen_ao3_cpt import QwenAO3Config, train

    config = QwenAO3Config(**read_inline_config(invocation.inputs))
    require_run_directory(config.run_dir, invocation.workspace)
    result = train(config, worker_components=components)
    step = result.get("step")
    return HandlerResult(
        "worker.completed",
        {
            "reason": "training_complete",
            "status": str(result.get("status", "complete")),
        },
        optimizer_step=(step if isinstance(step, int) and step >= 0 else None),
    )


_HANDLERS: Mapping[AdapterKey, Handler] = {
    (
        "rwkv-lab.mageflow-appearance-expert",
        "1.0.0",
        "train",
        "rwkv_lab.mageflow_appearance_expert.v1.Train",
    ): _appearance_expert,
    (
        "rwkv-lab.mageflow-terminal-expert",
        "1.0.0",
        "train",
        "rwkv_lab.mageflow_terminal_expert.v1.Train",
    ): _terminal_expert,
    (
        "rwkv-lab.qwen-ao3",
        "1.0.0",
        "train",
        "rwkv_lab.qwen_ao3.v1.Train",
    ): _qwen_ao3,
}


def supported_adapter_keys() -> frozenset[AdapterKey]:
    return frozenset(_HANDLERS)


def execute_invocation(invocation: WorkerInvocation) -> HandlerResult:
    adapter = invocation.adapter
    key = (
        adapter["adapter"],
        adapter["version"],
        adapter["operation"],
        adapter["contract"],
    )
    try:
        handler = _HANDLERS[key]
    except KeyError as error:
        raise AdapterDispatchError(
            "worker invocation has no closed adapter handler"
        ) from error
    if invocation.training is None:
        raise AdapterDispatchError("training adapter has no resolved composition")
    components = WorkerTrainingComponents(
        invocation.training, invocation.training.model_family
    )
    return handler(invocation, components)
