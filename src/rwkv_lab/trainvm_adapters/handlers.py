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
    resolve_resume_checkpoint,
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
from .qwen_controls import lower_initial_qwen_controls
from .rwkv_scratch import RWKVScratchTrainConfig
from .transformer_mla import PROFILE_ADAPTERS, TransformerMLATrainConfig


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


def _raw_config_path(
    config: Mapping[str, Any], name: str, *, required: bool
) -> str | None:
    value = config.get(name)
    if value is None and not required:
        return None
    if not isinstance(value, str) or not value:
        raise AdapterDispatchError(f"adapter config {name} is not a path string")
    return value


def _resume_payload(
    invocation: WorkerInvocation,
    paths: WorkspacePathAuthority,
    *,
    required_state: frozenset[str],
) -> Path | None:
    resolved = resolve_resume_checkpoint(invocation)
    if resolved is None:
        return None
    missing = required_state.difference(resolved.state_components)
    if missing:
        raise AdapterDispatchError(
            "resume checkpoint omits required state: " + ", ".join(sorted(missing))
        )
    return paths.read_path(
        str(resolved.payload_directory),
        label="controller resume checkpoint payload",
        kind="directory",
        require_content_identity=False,
    )


def _appearance_expert(
    invocation: WorkerInvocation,
    components: WorkerTrainingComponents,
    step_profiler: WorkerStepProfiler | None = None,
    observability: WorkerObservability | None = None,
    controls: WorkerControlRuntime | None = None,
) -> HandlerResult:
    paths = WorkspacePathAuthority.from_workspace(
        invocation.workspace, require_content=True
    )
    raw_config = read_inline_config(invocation.inputs)
    train_manifest = paths.read_path(
        _raw_config_path(raw_config, "train_manifest", required=True) or "",
        label="train_manifest",
        kind="file",
    )
    eval_value = _raw_config_path(raw_config, "eval_manifest", required=False)
    eval_manifest = (
        paths.read_path(eval_value, label="eval_manifest", kind="file")
        if eval_value
        else None
    )
    paths.verify_jsonl_file_references(
        train_manifest,
        fields=("image", "image_path"),
        label="train_manifest",
    )
    if eval_manifest is not None:
        paths.verify_jsonl_file_references(
            eval_manifest,
            fields=("image", "image_path"),
            label="eval_manifest",
        )
    from rwkv_lab.mage_flow_expert_train import MageFlowExpertTrainConfig, train

    config = MageFlowExpertTrainConfig(**raw_config)
    if controls is not None:
        lower_initial_mageflow_controls(config, controls)
    resume_payload = _resume_payload(
        invocation,
        paths,
        required_state=frozenset({"data_cursor", "model", "optimizer", "rng_torch"}),
    )
    config = replace(
        config,
        train_manifest=str(train_manifest),
        output_dir=str(paths.exact_run_directory(config.output_dir)),
        eval_manifest=(str(eval_manifest) if eval_manifest is not None else None),
        resume_from=(
            str(resume_payload)
            if resume_payload is not None
            else str(
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
            "control_revision",
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
    paths = WorkspacePathAuthority.from_workspace(
        invocation.workspace, require_content=True
    )
    raw_config = read_inline_config(invocation.inputs)
    train_manifest = paths.read_path(
        _raw_config_path(raw_config, "train_manifest", required=True) or "",
        label="train_manifest",
        kind="file",
    )
    eval_value = _raw_config_path(raw_config, "eval_manifest", required=False)
    eval_manifest = (
        paths.read_path(eval_value, label="eval_manifest", kind="file")
        if eval_value
        else None
    )
    paths.verify_jsonl_file_references(
        train_manifest,
        fields=("image", "image_path"),
        label="train_manifest",
    )
    if eval_manifest is not None:
        paths.verify_jsonl_file_references(
            eval_manifest,
            fields=("image", "image_path"),
            label="eval_manifest",
        )
    from rwkv_lab.mage_flow_terminal_train import TerminalExpertTrainConfig, train

    config = TerminalExpertTrainConfig(**raw_config)
    if controls is not None:
        lower_initial_mageflow_controls(config, controls)
    resume_payload = _resume_payload(
        invocation,
        paths,
        required_state=frozenset(
            {"data_cursor", "expert_routing", "model", "optimizer", "rng_torch"}
        ),
    )
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
        train_manifest=str(train_manifest),
        expert_checkpoint=str(
            paths.read_path(
                config.expert_checkpoint, label="expert_checkpoint", kind="file"
            )
        ),
        output_dir=str(paths.exact_run_directory(config.output_dir)),
        eval_manifest=(str(eval_manifest) if eval_manifest is not None else None),
        shared_backbone_checkpoint=_optional_read_path(
            paths,
            config.shared_backbone_checkpoint,
            label="shared_backbone_checkpoint",
            kind="file",
        ),
        resume_from=(
            str(resume_payload)
            if resume_payload is not None
            else _optional_read_path(
                paths,
                config.resume_from,
                label="resume_from",
                kind="directory",
            )
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
        worker_controls=controls,
    )
    request, step, status = completed_checkpoint_request(
        invocation,
        Path(config.output_dir),
        document_names=("status.json",),
        step_fields=("step",),
        state_components=(
            "component_composition",
            "control_revision",
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
    paths = WorkspacePathAuthority.from_workspace(
        invocation.workspace, require_content=True
    )
    raw_config = read_inline_config(invocation.inputs)
    model_dir = paths.read_path(
        _raw_config_path(raw_config, "model_dir", required=True) or "",
        label="model_dir",
        kind="directory",
    )
    train_pack_dir = paths.read_path(
        _raw_config_path(raw_config, "train_pack_dir", required=True) or "",
        label="train_pack_dir",
        kind="directory",
    )
    eval_pack_dir = paths.read_path(
        _raw_config_path(raw_config, "eval_pack_dir", required=True) or "",
        label="eval_pack_dir",
        kind="directory",
    )
    paths.verify_json_relative_file_reference(
        train_pack_dir,
        manifest_name="manifest.json",
        field="packed_file",
        label="train_pack_dir",
    )
    paths.verify_json_relative_file_reference(
        eval_pack_dir,
        manifest_name="manifest.json",
        field="packed_file",
        label="eval_pack_dir",
    )
    from rwkv_lab.qwen_ao3_cpt import QwenAO3Config, train

    config = QwenAO3Config(**raw_config)
    if controls is not None:
        config = lower_initial_qwen_controls(config, controls)
    resume_payload = _resume_payload(
        invocation,
        paths,
        required_state=frozenset({"data_cursor", "model", "optimizer", "rng_torch"}),
    )
    config = replace(
        config,
        model_dir=str(model_dir),
        train_pack_dir=str(train_pack_dir),
        eval_pack_dir=str(eval_pack_dir),
        run_dir=str(paths.exact_run_directory(config.run_dir)),
        resume=(
            str(resume_payload)
            if resume_payload is not None
            else _optional_read_path(
                paths,
                config.resume,
                label="resume",
                kind="directory",
            )
        )
        or "",
    )
    result = train(
        config,
        worker_components=components,
        worker_step_profiler=step_profiler or NullStepProfiler(),
        worker_observability=observability,
        worker_controls=controls,
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
                    "control_revision",
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
    paths = WorkspacePathAuthority.from_workspace(
        invocation.workspace, require_content=True
    )
    raw_config = read_inline_config(invocation.inputs)
    data = paths.read_path(
        _raw_config_path(raw_config, "data", required=True) or "",
        label="data",
        kind="file",
    )
    from rwkv_lab.rwkv_pretrain import main as train

    config = RWKVScratchTrainConfig(**raw_config)
    resume_payload = _resume_payload(
        invocation,
        paths,
        required_state=frozenset({"model", "optimizer", "rng_torch"}),
    )
    run_directory = paths.exact_run_directory(config.output_dir)
    resume = None
    if resume_payload is not None:
        resume = paths.read_path(
            str(resume_payload / "state.pt"),
            label="controller resume checkpoint state",
            kind="file",
            require_content_identity=False,
        )
    elif config.resume:
        resume = paths.read_path(config.resume, label="resume", kind="file")
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
                    "control_revision",
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


def _transformer_mla(
    invocation: WorkerInvocation,
    components: WorkerTrainingComponents,
    step_profiler: WorkerStepProfiler | None = None,
    observability: WorkerObservability | None = None,
    controls: WorkerControlRuntime | None = None,
) -> HandlerResult:
    paths = WorkspacePathAuthority.from_workspace(
        invocation.workspace, require_content=True
    )
    raw_config = read_inline_config(invocation.inputs)
    config = TransformerMLATrainConfig(**raw_config)
    adapter_name = invocation.adapter["adapter"]
    if config.adapter != adapter_name:
        raise AdapterDispatchError(
            "Transformer MLA profile does not match the sealed adapter key"
        )
    components.require_implementation(
        "optimizer",
        category="optimizer",
        allowed=frozenset(
            {
                "rwkv_lab.optimizer.torch_adamw.v1",
                "rwkv_lab.optimizer.torch_adamw_no_decay.v2",
            }
        ),
    )
    if config.profile == "engram":
        components.require_implementation(
            "host_optimizer",
            category="optimizer",
            allowed=frozenset({"rwkv_lab.optimizer.torch_sparse_adam.v1"}),
        )
    if controls is not None and dict(getattr(controls, "effective_values", {})):
        raise AdapterDispatchError(
            "Transformer MLA v1 profiles do not declare initial controls"
        )
    declaration = getattr(observability, "declaration", None)
    metrics = getattr(declaration, "metrics", {})
    emitted_metrics = {
        "train.loss",
        "train.learning_rate",
        "train.gradient_norm",
        "train.tokens_per_second",
        "eval.loss",
        "eval.perplexity",
        "eval.top1_accuracy",
        "eval.top5_accuracy",
    }
    for name in emitted_metrics.intersection(metrics):
        if metrics[name].step_domain != "optimizer_step":
            raise AdapterDispatchError(
                "Transformer MLA metrics require optimizer_step declarations"
            )

    model_dir = paths.read_path(
        config.model_dir, label="model_dir", kind="directory"
    )
    patch_dir = paths.read_path(
        config.patch_dir, label="patch_dir", kind="directory"
    )
    tokens_bin = paths.read_path(
        config.tokens_bin, label="tokens_bin", kind="file"
    )
    fsp_idf_path = (
        str(paths.read_path(config.fsp_idf_path, label="fsp_idf_path", kind="file"))
        if config.fsp_idf_path
        else ""
    )
    engram_patch_dir = (
        str(
            paths.read_path(
                config.engram_patch_dir,
                label="engram_patch_dir",
                kind="directory",
            )
        )
        if config.engram_patch_dir
        else ""
    )
    resume_payload = _resume_payload(
        invocation,
        paths,
        required_state=frozenset({"model", "optimizer", "topology"}),
    )
    resume = None
    if resume_payload is not None:
        resume = paths.read_path(
            str(resume_payload / "ckpt.pt"),
            label="controller resume checkpoint state",
            kind="file",
            require_content_identity=False,
        )

    config = replace(
        config,
        model_dir=str(model_dir),
        patch_dir=str(patch_dir),
        tokens_bin=str(tokens_bin),
        output_dir=str(paths.exact_run_directory(config.output_dir)),
        fsp_idf_path=fsp_idf_path,
        engram_patch_dir=engram_patch_dir,
    )
    from rwkv_lab.train_mla import train

    trainer_config = replace(
        config.trainer_configuration(),
        resume=(str(resume) if resume is not None else ""),
    )
    result = train(
        trainer_config,
        worker_components=components,
        worker_step_profiler=step_profiler or NullStepProfiler(),
        worker_observability=observability,
        worker_controls=controls,
    )
    if not isinstance(result, Mapping):
        raise AdapterDispatchError("Transformer MLA trainer omitted its terminal result")
    step = result.get("step")
    if not isinstance(step, int) or isinstance(step, bool) or step < 0:
        raise AdapterDispatchError("Transformer MLA trainer returned an invalid step")
    requests: tuple[CheckpointPublicationRequest, ...] = ()
    if declares_checkpoint(invocation):
        requests = (
            checkpoint_request(
                invocation,
                Path(config.output_dir),
                result.get("checkpoint"),
                step,
                resume_grade="compatible",
                state_components=(
                    "component_composition",
                    "control_revision",
                    "lr_schedule",
                    "model",
                    "optimizer",
                    "optimizer_groups",
                    "plateau_state",
                    "topology",
                ),
            ),
        )
    interrupted = result.get("status") == "interrupted"
    return HandlerResult(
        "operation.failed" if interrupted else "worker.completed",
        {
            "reason": (
                "checkpointed_interruption"
                if interrupted
                else "training_complete"
            ),
            "status": str(result.get("status", "complete")),
        },
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
    **{
        (
            adapter,
            "1.0.0",
            "train",
            {
                "mla": "rwkv_lab.transformer_mla.v1.Train",
                "mtp": "rwkv_lab.transformer_mla_mtp.v1.Train",
                "mutor": "rwkv_lab.transformer_mla_mutor.v1.Train",
                "fsp": "rwkv_lab.transformer_mla_fsp.v1.Train",
                "parallel": "rwkv_lab.transformer_mla_parallel.v1.Train",
                "rwkv8": "rwkv_lab.transformer_mla_rwkv8.v1.Train",
                "engram": "rwkv_lab.transformer_mla_engram.v1.Train",
                "full_backbone": "rwkv_lab.transformer_mla_full_backbone.v1.Train",
            }[profile],
        ): _transformer_mla
        for profile, adapter in PROFILE_ADAPTERS.items()
    },
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
        raise AdapterDispatchError(
            "training adapter has no worker observability authority"
        )
    if controls is None:
        raise AdapterDispatchError("training adapter has no worker control authority")
    return handler(
        invocation,
        components,
        step_profiler or NullStepProfiler(),
        observability,
        controls,
    )
