from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from rwkv_lab.trainvm_worker import (
    CheckpointPublicationRequest,
    NullStepProfiler,
    WorkerControlRuntime,
    WorkerInvocation,
    WorkerObservability,
    WorkerStepProfiler,
)

from .checkpoints import (
    checkpoint_request,
    completed_checkpoint_request,
    completion_reason,
    declares_checkpoint,
)
from .components import WorkerTrainingComponents
from .io import WorkspacePathAuthority, read_inline_config
from .mageflow_controls import lower_initial_mageflow_controls
from .rwkv_scratch import RWKVScratchTrainConfig


class AdapterDispatchError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class HandlerResult:
    event_type: str
    payload: Mapping[str, Any]
    optimizer_step: int | None = None
    checkpoint_requests: tuple[CheckpointPublicationRequest, ...] = ()


AdapterKey = tuple[str, str, str, str]
Handler = Callable[
    [
        WorkerInvocation,
        WorkerTrainingComponents,
        WorkerStepProfiler,
        WorkerObservability,
        WorkerControlRuntime,
    ],
    HandlerResult,
]


def _appearance_expert(
    invocation: WorkerInvocation,
    components: WorkerTrainingComponents,
    step_profiler: WorkerStepProfiler | None = None,
    observability: WorkerObservability | None = None,
    controls: WorkerControlRuntime | None = None,
) -> HandlerResult:
    from rwkv_lab.mage_flow_expert_train import MageFlowExpertTrainConfig, train

    config = MageFlowExpertTrainConfig(**read_inline_config(invocation.inputs))
    if controls is not None:
        lower_initial_mageflow_controls(config, controls)
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
        worker_observability=observability,
        worker_controls=controls,
    )
    request, step, status = completed_checkpoint_request(
        invocation,
        Path(config.output_dir),
        document_names=("complete.json", "status.json"),
        step_fields=("global_step", "step"),
        state_components=(
            "component_composition",
            "control_state",
            "data_cursor",
            "lr_schedule",
            "model",
            "optimizer",
            "parameter_routing",
            "rng_accelerator",
            "rng_numpy",
            "rng_python",
            "rng_torch",
        ),
    )
    return HandlerResult(
        "worker.completed",
        {"reason": completion_reason(status)},
        optimizer_step=step,
        checkpoint_requests=((request,) if request is not None else ()),
    )


def _terminal_expert(
    invocation: WorkerInvocation,
    components: WorkerTrainingComponents,
    step_profiler: WorkerStepProfiler | None = None,
    observability: WorkerObservability | None = None,
    controls: WorkerControlRuntime | None = None,
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
        worker_observability=observability,
    )
    request, step, status = completed_checkpoint_request(
        invocation,
        Path(config.output_dir),
        document_names=("status.json",),
        step_fields=("step",),
        state_components=(
            "component_composition",
            "data_cursor",
            "expert_routing",
            "lr_schedule",
            "model",
            "optimizer",
            "parameter_routing",
            "rng_accelerator",
            "rng_python",
            "rng_torch",
        ),
    )
    return HandlerResult(
        "worker.completed",
        {"reason": completion_reason(status)},
        optimizer_step=step,
        checkpoint_requests=((request,) if request is not None else ()),
    )


def _qwen_ao3(
    invocation: WorkerInvocation,
    components: WorkerTrainingComponents,
    step_profiler: WorkerStepProfiler | None = None,
    observability: WorkerObservability | None = None,
    controls: WorkerControlRuntime | None = None,
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
        worker_observability=observability,
    )
    step = result.get("step")
    checkpoint_requests: tuple[CheckpointPublicationRequest, ...] = ()
    checkpoint = result.get("checkpoint")
    if declares_checkpoint(invocation):
        if not isinstance(checkpoint, str):
            raise AdapterDispatchError("trainer omitted its declared checkpoint")
        checkpoint_requests = (
            checkpoint_request(
                invocation,
                Path(config.run_dir),
                checkpoint,
                step,
                state_components=(
                    "component_composition",
                    "data_cursor",
                    "expert_routing",
                    "model",
                    "optimizer",
                    "rng_accelerator",
                    "rng_numpy",
                    "rng_python",
                    "rng_torch",
                ),
            ),
        )
    return HandlerResult(
        "worker.completed",
        {
            "reason": "training_complete",
            "status": str(result.get("status", "complete")),
        },
        optimizer_step=(step if isinstance(step, int) and step >= 0 else None),
        checkpoint_requests=checkpoint_requests,
    )


def _rwkv_scratch(
    invocation: WorkerInvocation,
    components: WorkerTrainingComponents,
    step_profiler: WorkerStepProfiler | None = None,
    observability: WorkerObservability | None = None,
    controls: WorkerControlRuntime | None = None,
) -> HandlerResult:
    from rwkv_lab.rwkv_pretrain import main as train

    config = RWKVScratchTrainConfig(**read_inline_config(invocation.inputs))
    paths = WorkspacePathAuthority.from_workspace(invocation.workspace)
    data = paths.read_path(config.data, label="data", kind="file")
    run_directory = paths.exact_run_directory(config.output_dir)
    resume = (
        paths.read_path(config.resume, label="resume", kind="file")
        if config.resume
        else None
    )
    checkpoint_directory = run_directory / "checkpoint-final"
    try:
        checkpoint_directory.mkdir(mode=0o750, parents=True, exist_ok=False)
    except FileExistsError as error:
        raise AdapterDispatchError(
            "scratch-RWKV checkpoint staging already exists"
        ) from error
    checkpoint = checkpoint_directory / "state.pt"
    try:
        result = train(
            config.trainer_arguments(
                data=str(data),
                output_dir=str(run_directory),
                checkpoint=str(checkpoint),
                resume=(str(resume) if resume is not None else None),
            ),
            worker_components=components,
            worker_step_profiler=step_profiler or NullStepProfiler(),
            worker_observability=observability,
            worker_controls=controls,
        )
    except SystemExit as error:
        raise AdapterDispatchError(
            "scratch-RWKV rejected its typed adapter configuration"
        ) from error
    if not isinstance(result, Mapping):
        raise AdapterDispatchError("scratch-RWKV omitted its terminal result")
    step = result.get("step")
    if not isinstance(step, int) or isinstance(step, bool) or step < 0:
        raise AdapterDispatchError("scratch-RWKV returned an invalid optimizer step")
    if result.get("checkpoint") != str(checkpoint) or not checkpoint.is_file():
        raise AdapterDispatchError(
            "scratch-RWKV omitted its authority-selected terminal checkpoint"
        )
    requests: tuple[CheckpointPublicationRequest, ...] = ()
    if declares_checkpoint(invocation):
        requests = (
            checkpoint_request(
                invocation,
                run_directory,
                str(checkpoint_directory),
                step,
                resume_grade="terminal_checkpoint",
                state_components=(
                    "component_composition",
                    "control_state",
                    "model",
                    "optimizer",
                    "rng_accelerator",
                    "rng_numpy",
                    "rng_torch",
                ),
            ),
        )
    return HandlerResult(
        "worker.completed",
        {"reason": "training_complete"},
        optimizer_step=step,
        checkpoint_requests=requests,
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
    (
        "rwkv-lab.rwkv-scratch",
        "1.0.0",
        "train",
        "rwkv_lab.rwkv_scratch.v1.Train",
    ): _rwkv_scratch,
}


def supported_adapter_keys() -> frozenset[AdapterKey]:
    return frozenset(_HANDLERS)


def execute_invocation(
    invocation: WorkerInvocation,
    step_profiler: WorkerStepProfiler | None = None,
    observability: WorkerObservability | None = None,
    controls: WorkerControlRuntime | None = None,
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
    if observability is None:
        raise AdapterDispatchError("training adapter has no worker observability authority")
    if controls is None:
        raise AdapterDispatchError("training adapter has no worker control authority")
    return handler(
        invocation,
        components,
        step_profiler or NullStepProfiler(),
        observability,
        controls,
    )
