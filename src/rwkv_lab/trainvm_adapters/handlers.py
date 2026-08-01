from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from typing import Any

from rwkv_lab.trainvm_worker import (
    NullStepProfiler,
    WorkerInvocation,
    WorkerStepProfiler,
)

from .components import WorkerTrainingComponents
from .io import WorkspacePathAuthority, read_inline_config


class AdapterDispatchError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class HandlerResult:
    event_type: str
    payload: Mapping[str, Any]
    optimizer_step: int | None = None


AdapterKey = tuple[str, str, str, str]
Handler = Callable[
    [WorkerInvocation, WorkerTrainingComponents, WorkerStepProfiler], HandlerResult
]


def _appearance_expert(
    invocation: WorkerInvocation,
    components: WorkerTrainingComponents,
    step_profiler: WorkerStepProfiler | None = None,
) -> HandlerResult:
    from rwkv_lab.mage_flow_expert_train import MageFlowExpertTrainConfig, train

    config = MageFlowExpertTrainConfig(**read_inline_config(invocation.inputs))
    paths = WorkspacePathAuthority.from_workspace(invocation.workspace)
    config = replace(
        config,
        train_manifest=str(
            paths.read_path(config.train_manifest, label="train_manifest", kind="file")
        ),
        output_dir=str(paths.exact_run_directory(config.output_dir)),
        eval_manifest=(
            str(
                paths.read_path(
                    config.eval_manifest, label="eval_manifest", kind="file"
                )
            )
            if config.eval_manifest
            else None
        ),
        resume_from=(
            str(
                paths.read_path(
                    config.resume_from, label="resume_from", kind="directory"
                )
            )
            if config.resume_from
            else None
        ),
        continuation_from=(
            str(
                paths.read_path(
                    config.continuation_from,
                    label="continuation_from",
                    kind="directory",
                )
            )
            if config.continuation_from
            else None
        ),
        encoder_cache_dir=_cache_directory(
            paths, config.encoder_cache_dir, config.encoder_cache_mode
        ),
    )
    train(
        config,
        worker_components=components,
        worker_step_profiler=step_profiler or NullStepProfiler(),
    )
    return HandlerResult("worker.completed", {"reason": "training_complete"})


def _terminal_expert(
    invocation: WorkerInvocation,
    components: WorkerTrainingComponents,
    step_profiler: WorkerStepProfiler | None = None,
) -> HandlerResult:
    from rwkv_lab.mage_flow_terminal_train import TerminalExpertTrainConfig, train

    config = TerminalExpertTrainConfig(**read_inline_config(invocation.inputs))
    paths = WorkspacePathAuthority.from_workspace(invocation.workspace)
    expert_checkpoints = (
        {
            domain: str(
                paths.read_path(
                    checkpoint,
                    label=f"expert_checkpoints.{domain}",
                    kind="file",
                )
            )
            for domain, checkpoint in config.expert_checkpoints.items()
        }
        if config.expert_checkpoints
        else None
    )
    config = replace(
        config,
        train_manifest=str(
            paths.read_path(config.train_manifest, label="train_manifest", kind="file")
        ),
        expert_checkpoint=str(
            paths.read_path(
                config.expert_checkpoint, label="expert_checkpoint", kind="file"
            )
        ),
        output_dir=str(paths.exact_run_directory(config.output_dir)),
        eval_manifest=_optional_read_path(
            paths, config.eval_manifest, label="eval_manifest", kind="file"
        ),
        shared_backbone_checkpoint=_optional_read_path(
            paths,
            config.shared_backbone_checkpoint,
            label="shared_backbone_checkpoint",
            kind="file",
        ),
        resume_from=_optional_read_path(
            paths, config.resume_from, label="resume_from", kind="directory"
        ),
        model_path=_optional_read_path(
            paths, config.model_path, label="model_path", kind="directory"
        ),
        encoder_cache_dir=_cache_directory(
            paths, config.encoder_cache_dir, config.encoder_cache_mode
        ),
        encoder_cache_coverage_manifest=_optional_read_path(
            paths,
            config.encoder_cache_coverage_manifest,
            label="encoder_cache_coverage_manifest",
            kind="file",
        ),
        tread_loop_checkpoint=_optional_read_path(
            paths,
            config.tread_loop_checkpoint,
            label="tread_loop_checkpoint",
            kind="file",
        ),
        domain_window_schedule=_optional_read_path(
            paths,
            config.domain_window_schedule,
            label="domain_window_schedule",
            kind="file",
        ),
        expert_checkpoints=expert_checkpoints,
    )
    train(
        config,
        worker_components=components,
        worker_step_profiler=step_profiler or NullStepProfiler(),
    )
    return HandlerResult("worker.completed", {"reason": "training_complete"})


def _qwen_ao3(
    invocation: WorkerInvocation,
    components: WorkerTrainingComponents,
    step_profiler: WorkerStepProfiler | None = None,
) -> HandlerResult:
    from rwkv_lab.qwen_ao3_cpt import QwenAO3Config, train

    config = QwenAO3Config(**read_inline_config(invocation.inputs))
    paths = WorkspacePathAuthority.from_workspace(invocation.workspace)
    config = replace(
        config,
        model_dir=str(
            paths.read_path(config.model_dir, label="model_dir", kind="directory")
        ),
        train_pack_dir=str(
            paths.read_path(
                config.train_pack_dir, label="train_pack_dir", kind="directory"
            )
        ),
        eval_pack_dir=str(
            paths.read_path(
                config.eval_pack_dir, label="eval_pack_dir", kind="directory"
            )
        ),
        run_dir=str(paths.exact_run_directory(config.run_dir)),
        resume=_optional_read_path(
            paths, config.resume, label="resume", kind="directory"
        )
        or "",
    )
    result = train(
        config,
        worker_components=components,
        worker_step_profiler=step_profiler or NullStepProfiler(),
    )
    step = result.get("step")
    return HandlerResult(
        "worker.completed",
        {
            "reason": "training_complete",
            "status": str(result.get("status", "complete")),
        },
        optimizer_step=(step if isinstance(step, int) and step >= 0 else None),
    )


def _optional_read_path(
    paths: WorkspacePathAuthority,
    value: str | None,
    *,
    label: str,
    kind: str,
) -> str | None:
    return str(paths.read_path(value, label=label, kind=kind)) if value else None


def _cache_directory(
    paths: WorkspacePathAuthority, value: str | None, mode: str
) -> str | None:
    if not value or mode == "off":
        return value
    if mode == "read_only":
        return str(paths.read_path(value, label="encoder_cache_dir", kind="directory"))
    if mode == "read_write":
        return str(paths.write_directory(value, label="encoder_cache_dir"))
    # Trainer validation owns the closed mode enum; do not make an unknown mode
    # look like a read- or write-authorized path before that diagnostic runs.
    return value


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


def execute_invocation(
    invocation: WorkerInvocation,
    step_profiler: WorkerStepProfiler | None = None,
) -> HandlerResult:
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
    return handler(invocation, components, step_profiler or NullStepProfiler())
