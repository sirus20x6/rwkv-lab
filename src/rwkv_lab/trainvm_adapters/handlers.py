from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
from argparse import Namespace
from collections.abc import Callable, Mapping
from contextlib import nullcontext
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from rwkv_lab.rwkv_optimizer_finetune import RWKVOptimizerFinetuneConfig
from rwkv_lab.trainvm_worker import (
    ArtifactPublicationRequest,
    CheckpointPublicationRequest,
    EvalGalleryPublicationRequest,
    ExecutionPhase,
    NullStepProfiler,
    WorkerControlRuntime,
    WorkerExecutionPhases,
    WorkerInvocation,
    WorkerObservability,
    WorkerPublicationRuntime,
    WorkerStepProfiler,
    load_input_artifact_json,
    resolve_input_artifact,
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
from .mageflow_gallery import completed_mageflow_gallery_request
from .metric_decision import ScalarMetricDecisionConfig
from .posttraining import RWKVPostTrainConfig
from .qwen_controls import lower_initial_qwen_controls
from .rlvr import RLVRTrainConfig
from .rwkv_scratch import RWKVScratchTrainConfig
from .transformer_mla import PROFILE_ADAPTERS, TransformerMLATrainConfig
from .vision_compressor import VisionTeacherCompressorConfig
from .vision_frozen import VisionFrozenAdapterConfig
from .vision_native import VisionNativeHeadConfig
from .vision_student import VisionRWKVStudentConfig


class AdapterDispatchError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class HandlerResult:
    event_type: str
    payload: Mapping[str, Any]
    optimizer_step: int | None = None
    checkpoint_requests: tuple[CheckpointPublicationRequest, ...] = ()
    artifact_requests: tuple[ArtifactPublicationRequest, ...] = ()
    eval_gallery_requests: tuple[EvalGalleryPublicationRequest, ...] = ()


def _declares_artifact_output(invocation: object, name: str) -> bool:
    publishes = getattr(invocation, "publishes", None)
    return isinstance(publishes, Mapping) and name in publishes


def _stage_canonical_json_artifact(
    run_directory: Path,
    *,
    attempt_id: object,
    stem: str,
    filename: str,
    document: Mapping[str, object],
) -> Path:
    suffix = hashlib.sha256(str(attempt_id).encode("utf-8")).hexdigest()[:16]
    staging = run_directory / f"{stem}-{suffix}"
    try:
        encoded = json.dumps(
            document,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise AdapterDispatchError("artifact result is not finite JSON") from error
    if not encoded or len(encoded) > 64 * 1024:
        raise AdapterDispatchError("artifact result exceeds its byte bound")
    try:
        staging.mkdir(mode=0o750, parents=True, exist_ok=False)
        descriptor = os.open(
            staging / filename,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o440,
        )
        try:
            with os.fdopen(descriptor, "wb") as output:
                output.write(encoded)
                output.flush()
                os.fsync(output.fileno())
        except BaseException:
            try:
                os.close(descriptor)
            except OSError:
                pass
            raise
        directory_descriptor = os.open(staging, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    except BaseException:
        if staging.exists() and staging.is_dir() and not staging.is_symlink():
            shutil.rmtree(staging)
        raise
    return staging


AdapterKey = tuple[str, str, str, str]
Handler = Callable[
    [
        WorkerInvocation,
        WorkerTrainingComponents | None,
        WorkerStepProfiler,
        WorkerObservability,
        WorkerControlRuntime,
        WorkerExecutionPhases | None,
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


def _mageflow_live_eval_publisher(
    invocation: WorkerInvocation,
    runtime: WorkerPublicationRuntime | None,
    run_directory: Path,
    eval_manifest: Path | None,
    *,
    state_components: tuple[str, ...],
) -> Callable[[Path, int], None] | None:
    """Give a MageFlow trainer only a same-step publication callback."""

    if (
        runtime is None
        or eval_manifest is None
        or not _declares_artifact_output(invocation, "eval_gallery")
    ):
        return None

    def publish(checkpoint_directory: Path, step: int) -> None:
        checkpoint = checkpoint_request(
            invocation,
            run_directory,
            str(checkpoint_directory),
            step,
            resume_grade="compatible",
            state_components=state_components,
        )
        gallery = completed_mageflow_gallery_request(
            invocation,
            run_directory,
            eval_manifest,
            step=step,
            checkpoint_request_index=0,
        )
        if gallery is None:
            raise AdapterDispatchError(
                "MageFlow live eval publication omitted its gallery request"
            )
        runtime.publish_eval_revision(checkpoint, gallery)

    return publish


def _appearance_expert(
    invocation: WorkerInvocation,
    components: WorkerTrainingComponents,
    step_profiler: WorkerStepProfiler | None = None,
    observability: WorkerObservability | None = None,
    controls: WorkerControlRuntime | None = None,
    execution_phases: WorkerExecutionPhases | None = None,
    publications: WorkerPublicationRuntime | None = None,
) -> HandlerResult:
    paths = WorkspacePathAuthority.from_workspace(
        invocation.workspace, require_content=True
    )
    raw_config = read_inline_config(invocation.inputs)
    if execution_phases is not None:
        compile_request = execution_phases.request(ExecutionPhase.COMPILE)
        compile_enabled = (
            compile_request.enabled if compile_request is not None else False
        )
        raw_config = {
            **raw_config,
            "compile_transformer_blocks": compile_enabled,
            "compile_vae_encoder": compile_enabled,
        }
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
    checkpoint_state = (
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
    )
    train(
        config,
        worker_components=components,
        worker_step_profiler=step_profiler or NullStepProfiler(),
        worker_observability=observability,
        worker_controls=controls,
        worker_execution_phases=execution_phases,
        worker_eval_publication=_mageflow_live_eval_publisher(
            invocation,
            publications,
            Path(config.output_dir),
            eval_manifest,
            state_components=checkpoint_state,
        ),
    )
    request, step, status = completed_checkpoint_request(
        invocation,
        Path(config.output_dir),
        document_names=("complete.json", "status.json"),
        step_fields=("global_step", "step"),
        state_components=checkpoint_state,
    )
    gallery = None
    if (
        status == "complete"
        and eval_manifest is not None
        and _declares_artifact_output(invocation, "eval_gallery")
    ):
        if request is None or step is None:
            raise AdapterDispatchError(
                "MageFlow gallery publication requires a terminal checkpoint"
            )
        gallery = completed_mageflow_gallery_request(
            invocation,
            Path(config.output_dir),
            eval_manifest,
            step=step,
            checkpoint_request_index=0,
        )
    return HandlerResult(
        "worker.completed",
        {"reason": completion_reason(status)},
        optimizer_step=step,
        checkpoint_requests=((request,) if request is not None else ()),
        eval_gallery_requests=((gallery,) if gallery is not None else ()),
    )


def _mageflow_full_backbone(
    invocation: WorkerInvocation,
    components: WorkerTrainingComponents,
    step_profiler: WorkerStepProfiler | None = None,
    observability: WorkerObservability | None = None,
    controls: WorkerControlRuntime | None = None,
    execution_phases: WorkerExecutionPhases | None = None,
    publications: WorkerPublicationRuntime | None = None,
) -> HandlerResult:
    """Run the sealed, full NR-MMDiT continued-pretraining profile."""

    paths = WorkspacePathAuthority.from_workspace(
        invocation.workspace, require_content=True
    )
    raw_config = read_inline_config(invocation.inputs)
    if execution_phases is not None:
        compile_request = execution_phases.request(ExecutionPhase.COMPILE)
        compile_enabled = (
            compile_request.enabled if compile_request is not None else False
        )
        raw_config = {
            **raw_config,
            "compile_transformer_blocks": compile_enabled,
            "compile_vae_encoder": compile_enabled,
        }
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
    model_path_value = _raw_config_path(raw_config, "model_path", required=True)
    model_path = paths.read_path(
        model_path_value or "", label="model_path", kind="directory"
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
    from rwkv_lab.mage_flow_pretrain import MageFlowTrainConfig, train

    config = MageFlowTrainConfig(**raw_config)
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
        eval_manifest=(str(eval_manifest) if eval_manifest is not None else None),
        model_path=str(model_path),
        output_dir=str(paths.exact_run_directory(config.output_dir)),
        resume_from=(
            str(resume_payload)
            if resume_payload is not None
            else (
                str(
                    paths.read_path(
                        config.resume_from, label="resume_from", kind="directory"
                    )
                )
                if config.resume_from
                else None
            )
        ),
    )
    checkpoint_state = (
        "component_composition",
        "control_revision",
        "data_cursor",
        "lr_schedule",
        "model",
        "optimizer",
        "parameter_routing",
        "rng_accelerator",
        "rng_python",
        "rng_torch",
    )
    train(
        config,
        worker_components=components,
        worker_step_profiler=step_profiler or NullStepProfiler(),
        worker_observability=observability,
        worker_controls=controls,
        worker_execution_phases=execution_phases,
        worker_eval_publication=_mageflow_live_eval_publisher(
            invocation,
            publications,
            Path(config.output_dir),
            eval_manifest,
            state_components=checkpoint_state,
        ),
    )
    request, step, status = completed_checkpoint_request(
        invocation,
        Path(config.output_dir),
        document_names=("complete.json", "interrupted.json", "status.json"),
        step_fields=("global_step", "step"),
        state_components=checkpoint_state,
    )
    gallery = None
    if (
        status == "complete"
        and eval_manifest is not None
        and _declares_artifact_output(invocation, "eval_gallery")
    ):
        if request is None or step is None:
            raise AdapterDispatchError(
                "MageFlow gallery publication requires a terminal checkpoint"
            )
        gallery = completed_mageflow_gallery_request(
            invocation,
            Path(config.output_dir),
            eval_manifest,
            step=step,
            checkpoint_request_index=0,
        )
    return HandlerResult(
        "worker.completed",
        {"reason": completion_reason(status)},
        optimizer_step=step,
        checkpoint_requests=((request,) if request is not None else ()),
        eval_gallery_requests=((gallery,) if gallery is not None else ()),
    )


def _terminal_expert(
    invocation: WorkerInvocation,
    components: WorkerTrainingComponents,
    step_profiler: WorkerStepProfiler | None = None,
    observability: WorkerObservability | None = None,
    controls: WorkerControlRuntime | None = None,
    execution_phases: WorkerExecutionPhases | None = None,
    publications: WorkerPublicationRuntime | None = None,
) -> HandlerResult:
    paths = WorkspacePathAuthority.from_workspace(
        invocation.workspace, require_content=True
    )
    raw_config = read_inline_config(invocation.inputs)
    if execution_phases is not None:
        compile_request = execution_phases.request(ExecutionPhase.COMPILE)
        compile_enabled = (
            compile_request.enabled if compile_request is not None else False
        )
        raw_config = {
            **raw_config,
            "compile_transformer_blocks": compile_enabled,
            "compile_vae_encoder": compile_enabled,
        }
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
    checkpoint_state = (
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
    )
    train(
        config,
        worker_components=components,
        worker_step_profiler=step_profiler or NullStepProfiler(),
        worker_observability=observability,
        worker_controls=controls,
        worker_execution_phases=execution_phases,
        worker_eval_publication=_mageflow_live_eval_publisher(
            invocation,
            publications,
            Path(config.output_dir),
            eval_manifest,
            state_components=checkpoint_state,
        ),
    )
    request, step, status = completed_checkpoint_request(
        invocation,
        Path(config.output_dir),
        document_names=("status.json",),
        step_fields=("step",),
        state_components=checkpoint_state,
    )
    gallery = None
    if (
        status == "complete"
        and eval_manifest is not None
        and _declares_artifact_output(invocation, "eval_gallery")
    ):
        if request is None or step is None:
            raise AdapterDispatchError(
                "MageFlow gallery publication requires a terminal checkpoint"
            )
        gallery = completed_mageflow_gallery_request(
            invocation,
            Path(config.output_dir),
            eval_manifest,
            step=step,
            checkpoint_request_index=0,
        )
    return HandlerResult(
        "worker.completed",
        {"reason": completion_reason(status)},
        optimizer_step=step,
        checkpoint_requests=((request,) if request is not None else ()),
        eval_gallery_requests=((gallery,) if gallery is not None else ()),
    )


def _qwen_ao3(
    invocation: WorkerInvocation,
    components: WorkerTrainingComponents,
    step_profiler: WorkerStepProfiler | None = None,
    observability: WorkerObservability | None = None,
    controls: WorkerControlRuntime | None = None,
    execution_phases: WorkerExecutionPhases | None = None,
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
    execution_phases: WorkerExecutionPhases | None = None,
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
            worker_execution_phases=execution_phases,
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


def _rwkv_optimizer_finetune(
    invocation: WorkerInvocation,
    components: WorkerTrainingComponents,
    step_profiler: WorkerStepProfiler | None = None,
    observability: WorkerObservability | None = None,
    controls: WorkerControlRuntime | None = None,
    execution_phases: WorkerExecutionPhases | None = None,
) -> HandlerResult:
    """Run a content-bound pretrained RWKV optimizer experiment."""

    del execution_phases
    if not declares_checkpoint(invocation) or not _declares_artifact_output(
        invocation, "result"
    ):
        raise AdapterDispatchError(
            "RWKV optimizer finetuning requires checkpoint and result outputs"
        )
    paths = WorkspacePathAuthority.from_workspace(
        invocation.workspace, require_content=True
    )
    raw_config = read_inline_config(invocation.inputs)
    model_path = paths.read_path(
        _raw_config_path(raw_config, "model_path", required=True) or "",
        label="model_path",
        kind="file",
    )
    data_path = paths.read_path(
        _raw_config_path(raw_config, "data_path", required=True) or "",
        label="data_path",
        kind="file",
    )
    if "output_dir" in raw_config:
        raise AdapterDispatchError(
            "RWKV optimizer finetuning output is node-authority selected"
        )
    run_directory = paths.node_run_directory(invocation.node_id)
    content_identity = [
        {
            "path": identity.path,
            "kind": identity.kind,
            "file_count": identity.file_count,
            "total_bytes": identity.total_bytes,
            "tree_sha256": identity.tree_sha256,
        }
        for identity in sorted(paths.input_content_roots, key=lambda value: value.path)
    ]
    input_identity_digest = "sha256:" + hashlib.sha256(
        json.dumps(
            content_identity,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    resume_payload = _resume_payload(
        invocation,
        paths,
        required_state=frozenset(
            {
                "component_composition",
                "control_revision",
                "data_cursor",
                "input_content_identity",
                "learning_rate_schedule",
                "model",
                "optimizer",
                "parameter_routing",
                "rng_accelerator",
                "rng_numpy",
                "rng_python",
                "rng_torch",
            }
        ),
    )
    config_values = dict(raw_config)
    config_values.update(
        {
            "model_path": str(model_path),
            "data_path": str(data_path),
            "output_dir": str(run_directory),
            "input_identity_digest": input_identity_digest,
            "resume_from": str(resume_payload) if resume_payload else None,
        }
    )
    config = RWKVOptimizerFinetuneConfig(**config_values)
    from rwkv_lab.rwkv_optimizer_finetune import train

    train_result = train(
        config,
        worker_components=components,
        worker_step_profiler=step_profiler or NullStepProfiler(),
        worker_observability=observability,
        worker_controls=controls,
    )
    if not isinstance(train_result, Mapping):
        raise AdapterDispatchError("RWKV optimizer finetuning omitted its result")
    request, step, status = completed_checkpoint_request(
        invocation,
        run_directory,
        document_names=("complete.json", "interrupted.json"),
        step_fields=("step",),
        state_components=(
            "component_composition",
            "control_revision",
            "data_cursor",
            "input_content_identity",
            "learning_rate_schedule",
            "model",
            "optimizer",
            "parameter_routing",
            "rng_accelerator",
            "rng_numpy",
            "rng_python",
            "rng_torch",
        ),
    )
    if train_result.get("state") != status or train_result.get("step") != step:
        raise AdapterDispatchError(
            "RWKV optimizer finetuning result disagrees with its terminal document"
        )
    artifact_requests: tuple[ArtifactPublicationRequest, ...] = ()
    if status == "complete":
        eval_loss = train_result.get("eval_loss")
        if (
            isinstance(eval_loss, bool)
            or not isinstance(eval_loss, (int, float))
            or not math.isfinite(float(eval_loss))
        ):
            raise AdapterDispatchError(
                "RWKV optimizer finetuning completed without finite eval loss"
            )
        result_directory = _stage_canonical_json_artifact(
            run_directory,
            attempt_id=getattr(invocation, "attempt_id", "attempt"),
            stem="scalar-result",
            filename="result.json",
            document={
                "api_version": "rwkv-lab.scalar-metric-result/v1",
                "direction": "minimize",
                "metric": "eval.loss",
                "optimizer_step": step,
                "subject": config.subject,
                "value": float(eval_loss),
            },
        )
        artifact_requests = (
            ArtifactPublicationRequest(
                source_directory=result_directory,
                output_name="result",
            ),
        )
    return HandlerResult(
        "operation.failed" if status == "interrupted" else "worker.completed",
        {"reason": completion_reason(status), "status": status},
        optimizer_step=step,
        checkpoint_requests=((request,) if request is not None else ()),
        artifact_requests=artifact_requests,
    )


def _rwkv_posttraining(
    invocation: WorkerInvocation,
    components: WorkerTrainingComponents,
    step_profiler: WorkerStepProfiler | None = None,
    observability: WorkerObservability | None = None,
    controls: WorkerControlRuntime | None = None,
    execution_phases: WorkerExecutionPhases | None = None,
) -> HandlerResult:
    """Run one immutable-parent, restart-only RWKV post-training attempt."""

    if getattr(invocation, "resume", None) is not None:
        raise AdapterDispatchError(
            "RWKV post-training v1 is restart-only and rejects resume state"
        )
    publishes = getattr(invocation, "publishes", {})
    if not isinstance(publishes, Mapping) or "adapter" not in publishes:
        raise AdapterDispatchError(
            "RWKV post-training invocation omits its required adapter artifact"
        )
    effective_controls = (
        getattr(controls, "effective_values", {}) if controls is not None else {}
    )
    if not isinstance(effective_controls, Mapping):
        raise AdapterDispatchError(
            "RWKV post-training received an invalid control snapshot"
        )
    if effective_controls:
        raise AdapterDispatchError(
            "RWKV post-training v1 does not declare initial controls"
        )
    verification = (
        observability.keepalive(0, "verifying_inputs")
        if observability is not None
        else nullcontext()
    )
    with verification:
        paths = WorkspacePathAuthority.from_workspace(
            invocation.workspace, require_content=True
        )
        config = RWKVPostTrainConfig(**read_inline_config(invocation.inputs))
        checkpoint = paths.read_path(config.checkpoint, label="checkpoint", kind="file")
        data = paths.read_path(config.data, label="data", kind="file")
        eval_data = (
            str(paths.read_path(config.eval_data, label="eval_data", kind="file"))
            if config.eval_data
            else ""
        )
        template = (
            str(paths.read_path(config.template, label="template", kind="file"))
            if config.template
            else ""
        )
        token_cache = (
            str(paths.write_directory(config.token_cache, label="token_cache"))
            if config.token_cache
            else ""
        )
        run_directory = paths.exact_run_directory(config.output_dir)
    attempt_id = str(getattr(invocation, "attempt_id", "attempt"))
    attempt_suffix = hashlib.sha256(attempt_id.encode("utf-8")).hexdigest()[:16]
    staging = run_directory / f"posttraining-output-{attempt_suffix}"
    try:
        staging.mkdir(mode=0o750, parents=True, exist_ok=False)
    except FileExistsError as error:
        raise AdapterDispatchError(
            "RWKV post-training output staging already exists"
        ) from error
    from rwkv_lab.posttrain_train import train

    result = train(
        checkpoint=str(checkpoint),
        data=str(data),
        output=str(staging),
        objective=config.objective,
        adapter_name=config.adapter_name,
        rank=config.rank,
        alpha=config.alpha,
        targets=config.targets,
        steps=config.steps,
        batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        minimum_learning_rate_ratio=config.minimum_learning_rate_ratio,
        warmup_steps=config.warmup_steps,
        weight_decay=config.weight_decay,
        max_gradient_norm=config.max_gradient_norm,
        beta=config.beta,
        gamma=config.gamma,
        max_length=config.max_length,
        seed=config.seed,
        device=config.device,
        template=template,
        eval_data=eval_data,
        token_cache=token_cache,
        max_train_tokens=config.max_train_tokens,
        packing=config.packing,
        base_quantization=config.base_quantization,
        quant_block_size=config.quant_block_size,
        quant_backend=config.quant_backend,
        activation_offload=config.activation_offload,
        log_every=config.log_every,
        worker_components=components,
        worker_step_profiler=step_profiler or NullStepProfiler(),
        worker_observability=observability,
        worker_controls=controls,
    )
    if not isinstance(result, Mapping):
        raise AdapterDispatchError(
            "RWKV post-training trainer omitted its terminal result"
        )
    step = result.get("steps")
    if not isinstance(step, int) or isinstance(step, bool) or step < 1:
        raise AdapterDispatchError(
            "RWKV post-training trainer returned an invalid optimizer step"
        )
    result_path = staging / "posttrain-result.json"
    adapter_directory = staging / "adapter"
    if (
        not result_path.is_file()
        or result_path.is_symlink()
        or not adapter_directory.is_dir()
        or adapter_directory.is_symlink()
    ):
        raise AdapterDispatchError(
            "RWKV post-training trainer omitted its result or adapter payload"
        )
    return HandlerResult(
        "worker.completed",
        {"reason": "training_complete", "objective": config.objective},
        optimizer_step=step,
        artifact_requests=(
            ArtifactPublicationRequest(
                source_directory=staging,
                output_name="adapter",
            ),
        ),
    )


def _rlvr(
    invocation: WorkerInvocation,
    components: WorkerTrainingComponents,
    step_profiler: WorkerStepProfiler | None = None,
    observability: WorkerObservability | None = None,
    controls: WorkerControlRuntime | None = None,
    execution_phases: WorkerExecutionPhases | None = None,
) -> HandlerResult:
    """Run one bounded RLVR candidate with a terminal immutable checkpoint."""

    if getattr(invocation, "resume", None) is not None:
        raise AdapterDispatchError(
            "RLVR v1 is terminal-checkpoint only and rejects controller resume"
        )
    if not declares_checkpoint(invocation):
        raise AdapterDispatchError(
            "RLVR invocation omits its required checkpoint artifact"
        )
    effective_controls = (
        getattr(controls, "effective_values", {}) if controls is not None else {}
    )
    if not isinstance(effective_controls, Mapping):
        raise AdapterDispatchError("RLVR received an invalid control snapshot")
    if effective_controls:
        raise AdapterDispatchError("RLVR v1 does not declare initial controls")

    verification = (
        observability.keepalive(0, "verifying_inputs")
        if observability is not None
        else nullcontext()
    )
    with verification:
        paths = WorkspacePathAuthority.from_workspace(
            invocation.workspace, require_content=True
        )
        config = RLVRTrainConfig(**read_inline_config(invocation.inputs))
        checkpoint = paths.read_path(config.checkpoint, label="checkpoint", kind="file")
        vocab = paths.read_path(config.vocab, label="vocab", kind="file")
        tasks = (
            paths.read_path(config.tasks, label="tasks", kind="file")
            if config.tasks
            else None
        )
        heldout_tasks = (
            paths.read_path(config.heldout_tasks, label="heldout_tasks", kind="file")
            if config.heldout_tasks
            else None
        )
        reference_checkpoint = (
            paths.read_path(
                config.reference_checkpoint,
                label="reference_checkpoint",
                kind="file",
            )
            if config.reference_checkpoint
            else None
        )
        verifier_executable = (
            paths.read_path(
                config.verifier_executable,
                label="verifier_executable",
                kind="file",
            )
            if config.verifier_executable
            else None
        )
        run_directory = paths.exact_run_directory(config.output_dir)

    from rwkv_lab.rlvr_train import run

    result = run(
        Namespace(
            ckpt=str(checkpoint),
            resume="",
            out=str(run_directory),
            tasks=str(tasks) if tasks is not None else "",
            heldout_tasks=str(heldout_tasks) if heldout_tasks is not None else "",
            algorithm=config.algorithm,
            steps=config.steps,
            prompts_per_step=config.prompts_per_step,
            group_size=config.group_size,
            epochs=config.epochs,
            max_new=config.max_new_tokens,
            rollout_engine=config.rollout_engine,
            rollout_devices=",".join(config.rollout_devices),
            temperature=config.temperature,
            eval_temperature=config.eval_temperature,
            top_p=config.top_p,
            top_k=config.top_k,
            stop_token=config.stop_token,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            optimizer=config.optimizer,
            warmup=config.warmup_steps,
            grad_clip=config.max_gradient_norm,
            clip_low=config.clip_low,
            clip_high=config.clip_high,
            kl_coef=config.kl_coefficient,
            reference=config.reference,
            reference_ckpt=(
                str(reference_checkpoint) if reference_checkpoint is not None else ""
            ),
            train_tasks=config.train_tasks,
            eval_tasks=config.eval_tasks,
            difficulty=config.difficulty,
            curriculum_stages=",".join(
                str(value) for value in config.curriculum_stages
            ),
            sft_steps=config.sft_steps,
            sft_batch_size=config.sft_batch_size,
            sft_lr=config.sft_learning_rate,
            preflight_prompts=config.preflight_prompts,
            min_preflight_reward=config.minimum_preflight_reward,
            max_preflight_reward=config.maximum_preflight_reward,
            min_preflight_active_groups=config.minimum_preflight_active_groups,
            eval_every=config.eval_every,
            eval_prompts=config.eval_prompts,
            eval_group_size=config.eval_group_size,
            min_heldout_delta=config.minimum_heldout_delta,
            confidence=config.confidence,
            bootstrap_samples=config.bootstrap_samples,
            require_confidence=config.require_confidence,
            max_family_regression=config.maximum_family_regression,
            max_rollout_tokens=config.maximum_rollout_tokens,
            max_train_seconds=config.maximum_train_seconds,
            save_every=config.save_every,
            verifier_command=(
                (str(verifier_executable), *config.verifier_arguments)
                if verifier_executable is not None
                else ()
            ),
            verifier_timeout=config.verifier_timeout,
            log_samples=config.log_samples,
            seed=config.seed,
            device=config.device,
            use_ema=config.use_ema,
            vocab=str(vocab),
        ),
        worker_components=components,
        worker_step_profiler=step_profiler or NullStepProfiler(),
        worker_observability=observability,
        worker_controls=controls,
    )
    if not isinstance(result, Mapping) or result.get("status") != "complete":
        raise AdapterDispatchError("RLVR trainer omitted its terminal result")
    step = result.get("steps_completed")
    source_checkpoint = result.get("checkpoint")
    if not isinstance(source_checkpoint, str):
        raise AdapterDispatchError("RLVR trainer omitted its candidate checkpoint")
    resolved_checkpoint = Path(source_checkpoint).resolve(strict=True)
    if (
        not resolved_checkpoint.is_file()
        or resolved_checkpoint.is_symlink()
        or run_directory.resolve(strict=True) not in resolved_checkpoint.parents
    ):
        raise AdapterDispatchError("RLVR candidate checkpoint escaped run authority")
    checkpoint_directory = run_directory / "checkpoint-final"
    try:
        checkpoint_directory.mkdir(mode=0o750, exist_ok=False)
    except FileExistsError as error:
        raise AdapterDispatchError("RLVR checkpoint staging already exists") from error
    state_path = checkpoint_directory / "state.pt"
    try:
        os.link(resolved_checkpoint, state_path)
    except OSError:
        shutil.copy2(resolved_checkpoint, state_path, follow_symlinks=False)
    request = checkpoint_request(
        invocation,
        run_directory,
        str(checkpoint_directory),
        step,
        resume_grade="terminal_checkpoint",
        state_components=(
            "component_composition",
            "model",
            "optimizer",
            "rng_python",
            "rng_torch",
        ),
    )
    return HandlerResult(
        "worker.completed",
        {
            "reason": "training_complete",
            "training_status": result.get("training_status"),
            "promotion_eligible": bool(
                isinstance(result.get("promotion"), Mapping)
                and result["promotion"].get("eligible") is True
            ),
        },
        optimizer_step=request.optimizer_step,
        checkpoint_requests=(request,),
    )


def _vision_teacher_compressor(
    invocation: WorkerInvocation,
    components: WorkerTrainingComponents,
    step_profiler: WorkerStepProfiler | None = None,
    observability: WorkerObservability | None = None,
    controls: WorkerControlRuntime | None = None,
    execution_phases: WorkerExecutionPhases | None = None,
) -> HandlerResult:
    """Run the canonical multi-teacher vision compressor under TrainVM authority."""

    if not declares_checkpoint(invocation):
        raise AdapterDispatchError(
            "vision compressor invocation omits its required checkpoint artifact"
        )
    effective_controls = (
        getattr(controls, "effective_values", {}) if controls is not None else {}
    )
    if not isinstance(effective_controls, Mapping):
        raise AdapterDispatchError(
            "vision compressor received an invalid control snapshot"
        )
    if effective_controls:
        raise AdapterDispatchError(
            "vision compressor v1 does not declare initial controls"
        )

    verification = (
        observability.keepalive(0, "verifying_inputs")
        if observability is not None
        else nullcontext()
    )
    with verification:
        paths = WorkspacePathAuthority.from_workspace(
            invocation.workspace, require_content=True
        )
        config = VisionTeacherCompressorConfig(**read_inline_config(invocation.inputs))
        train_manifest = paths.read_path(
            config.train_manifest, label="train_manifest", kind="file"
        )
        eval_manifest = paths.read_path(
            config.eval_manifest, label="eval_manifest", kind="file"
        )
        for label, manifest in (
            ("train_manifest", train_manifest),
            ("eval_manifest", eval_manifest),
        ):
            paths.verify_jsonl_file_references(
                manifest,
                fields=("image", "image_path"),
                label=label,
            )
        moon_cache = paths.read_path(
            config.moon_cache, label="moon_cache", kind="directory"
        )
        fusion_cache = paths.read_path(
            config.fusion_cache, label="fusion_cache", kind="directory"
        )
        moonvit = paths.read_path(
            config.moonvit_checkpoint, label="moonvit_checkpoint", kind="file"
        )
        siglip2 = paths.read_path(
            config.siglip2_model, label="siglip2_model", kind="directory"
        )
        dinov2 = paths.read_path(
            config.dinov2_model, label="dinov2_model", kind="directory"
        )
        sam = paths.read_path(config.sam_model, label="sam_model", kind="directory")
        init_from = (
            paths.read_path(config.init_from, label="init_from", kind="file")
            if config.init_from
            else None
        )
        run_directory = paths.exact_run_directory(config.output_dir)
        resume_payload = _resume_payload(
            invocation,
            paths,
            required_state=frozenset(
                {
                    "component_composition",
                    "control_revision",
                    "data_cursor",
                    "model",
                    "optimizer",
                    "rng_accelerator",
                    "rng_python",
                    "rng_torch",
                }
            ),
        )
        resume_file = (
            paths.read_path(
                str(resume_payload / "state.pt"),
                label="resume checkpoint state",
                kind="file",
                require_content_identity=False,
            )
            if resume_payload is not None
            else None
        )

    from rwkv_lab.vision_teacher_compressor import train

    result = train(
        Namespace(
            data=train_manifest,
            eval_data=eval_manifest,
            moon_cache=moon_cache,
            fusion_cache=fusion_cache,
            out=run_directory,
            moonvit=moonvit,
            siglip2=str(siglip2),
            dinov2=str(dinov2),
            sam=str(sam),
            steps=config.steps,
            batch=config.batch_size,
            workers=config.workers,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            teacher_dropout=config.teacher_dropout,
            relational_weight=config.relational_weight,
            variance_weight=config.variance_weight,
            covariance_weight=config.covariance_weight,
            diversity_weight=config.diversity_weight,
            max_gradient_norm=config.max_gradient_norm,
            eval_every=config.eval_every,
            checkpoint_every=config.checkpoint_every,
            log_every=config.log_every,
            seed=config.seed,
            resume="none",
            resume_from=str(resume_file) if resume_file is not None else None,
            init_from=init_from,
            preflight_only=False,
            device=config.device,
        ),
        worker_components=components,
        worker_step_profiler=step_profiler or NullStepProfiler(),
        worker_observability=observability,
        worker_controls=controls,
    )
    if not isinstance(result, Mapping):
        raise AdapterDispatchError("vision compressor omitted its terminal result")
    status = result.get("status")
    if status not in {"complete", "interrupted"}:
        raise AdapterDispatchError("vision compressor returned an invalid status")
    step = result.get("step")
    request = checkpoint_request(
        invocation,
        run_directory,
        result.get("checkpoint"),
        step,
        resume_grade="compatible",
        state_components=(
            "component_composition",
            "control_revision",
            "data_cursor",
            "model",
            "optimizer",
            "rng_accelerator",
            "rng_python",
            "rng_torch",
        ),
    )
    return HandlerResult(
        "operation.failed" if status == "interrupted" else "worker.completed",
        {"reason": completion_reason(status)},
        optimizer_step=request.optimizer_step,
        checkpoint_requests=(request,),
    )


def _vision_frozen_adapter(
    invocation: WorkerInvocation,
    components: WorkerTrainingComponents,
    step_profiler: WorkerStepProfiler | None = None,
    observability: WorkerObservability | None = None,
    controls: WorkerControlRuntime | None = None,
    execution_phases: WorkerExecutionPhases | None = None,
) -> HandlerResult:
    """Run the canonical cached MoonViT/compressor caption trainer."""

    if not declares_checkpoint(invocation):
        raise AdapterDispatchError(
            "frozen vision invocation omits its required checkpoint artifact"
        )
    if not _declares_artifact_output(invocation, "result"):
        raise AdapterDispatchError(
            "frozen vision invocation omits its required scalar result artifact"
        )
    effective_controls = (
        getattr(controls, "effective_values", {}) if controls is not None else {}
    )
    if not isinstance(effective_controls, Mapping):
        raise AdapterDispatchError("frozen vision received an invalid control snapshot")
    if effective_controls:
        raise AdapterDispatchError("frozen vision v1 does not declare initial controls")

    verification = (
        observability.keepalive(0, "verifying_inputs")
        if observability is not None
        else nullcontext()
    )
    with verification:
        paths = WorkspacePathAuthority.from_workspace(
            invocation.workspace, require_content=True
        )
        config = VisionFrozenAdapterConfig(**read_inline_config(invocation.inputs))
        train_manifests = tuple(
            paths.read_path(value, label=f"train_manifests[{index}]", kind="file")
            for index, value in enumerate(config.train_manifests)
        )
        eval_manifests = tuple(
            paths.read_path(value, label=f"eval_manifests[{index}]", kind="file")
            for index, value in enumerate(config.eval_manifests)
        )
        for label, manifests in (
            ("train_manifests", train_manifests),
            ("eval_manifests", eval_manifests),
        ):
            for index, manifest in enumerate(manifests):
                paths.verify_jsonl_file_references(
                    manifest,
                    fields=("image", "image_path"),
                    label=f"{label}[{index}]",
                )
        rwkv = paths.read_path(
            config.rwkv_checkpoint, label="rwkv_checkpoint", kind="file"
        )
        moonvit = paths.read_path(
            config.moonvit_checkpoint, label="moonvit_checkpoint", kind="file"
        )
        vocab = paths.read_path(config.vocab, label="vocab", kind="file")
        moon_cache = paths.read_path(
            config.moon_cache, label="moon_cache", kind="directory"
        )
        compressor = fusion_cache = siglip2 = dinov2 = sam = None
        if config.arm == "compressor":
            compressor = paths.read_path(
                config.compressor_checkpoint,
                label="compressor_checkpoint",
                kind="file",
            )
            fusion_cache = paths.read_path(
                config.fusion_cache, label="fusion_cache", kind="directory"
            )
            siglip2 = paths.read_path(
                config.siglip2_model, label="siglip2_model", kind="directory"
            )
            dinov2 = paths.read_path(
                config.dinov2_model, label="dinov2_model", kind="directory"
            )
            sam = paths.read_path(config.sam_model, label="sam_model", kind="directory")
        run_directory = paths.node_run_directory(invocation.node_id)
        resume_payload = _resume_payload(
            invocation,
            paths,
            required_state=frozenset(
                {
                    "component_composition",
                    "control_revision",
                    "data_cursor",
                    "model",
                    "optimizer",
                    "rng_accelerator",
                    "rng_python",
                    "rng_torch",
                }
            ),
        )
        resume_file = (
            paths.read_path(
                str(resume_payload / "state.pt"),
                label="resume checkpoint state",
                kind="file",
                require_content_identity=False,
            )
            if resume_payload is not None
            else None
        )

    def csv(values: tuple[int, ...]) -> str:
        return ",".join(str(value) for value in values)

    arguments = [
        "--data",
        *(str(path) for path in train_manifests),
        "--eval-data",
        *(str(path) for path in eval_manifests),
        "--rwkv",
        str(rwkv),
        "--moonvit",
        str(moonvit),
        "--vocab",
        str(vocab),
        "--vision-backend",
        "moonvit",
        "--feature-cache",
        str(moon_cache),
        "--feature-cache-only",
        "--out",
        str(run_directory),
        "--steps",
        str(config.steps),
        "--batch",
        str(config.batch_size),
        "--min-batch",
        str(config.min_batch_size),
        "--max-batch",
        str(config.max_batch_size),
        "--target-batch-tokens",
        str(config.target_batch_tokens),
        "--max-text-tokens",
        str(config.max_text_tokens),
        "--prefix-tokens",
        str(config.prefix_tokens),
        "--feature-cache-max-bytes",
        str(config.feature_cache_max_bytes),
        "--max-input-patches",
        str(config.max_input_patches),
        "--moonvit-tap-layers",
        csv(config.moonvit_tap_layers),
        "--vision-view-mode",
        config.vision_view_mode,
        "--vision-resampler-layers",
        "0",
        "--deep-vision-layers",
        csv(config.deep_vision_layers),
        "--deep-vision-rank",
        str(config.deep_vision_rank),
        "--grounding-early-tokens",
        str(config.grounding_early_tokens),
        "--grounding-early-weight",
        str(config.grounding_early_weight),
        "--grounding-contrastive-weight",
        str(config.grounding_contrastive_weight),
        "--grounding-contrastive-dim",
        str(config.grounding_contrastive_dim),
        "--grounding-temperature",
        str(config.grounding_temperature),
        "--lr",
        str(config.learning_rate),
        "--loop-lr",
        str(config.loop_learning_rate),
        "--weight-decay",
        str(config.weight_decay),
        "--grad-clip",
        str(config.max_gradient_norm),
        "--loop-count",
        str(config.loop_count),
        "--loop-start-step",
        str(config.loop_start_step),
        "--loop-ramp-steps",
        str(config.loop_ramp_steps),
        "--loop-gate-cap",
        str(config.loop_gate_cap),
        "--engram-sites",
        csv(config.engram_sites),
        "--engram-drow",
        str(config.engram_drow),
        "--engram-rows",
        str(config.engram_rows),
        "--engram-lr",
        str(config.engram_learning_rate),
        "--engram-warmup-steps",
        str(config.engram_warmup_steps),
        "--engram-boundary-id",
        str(config.engram_boundary_id),
        "--nextlat-weight",
        str(config.nextlat_weight),
        "--nextlat-hidden",
        str(config.nextlat_hidden),
        "--manifest-stat-workers",
        str(config.manifest_stat_workers),
        "--checkpoint-every",
        str(config.checkpoint_every),
        "--eval-every",
        str(config.eval_every),
        "--eval-examples",
        str(config.eval_examples),
        "--eval-samples",
        str(config.eval_samples),
        "--eval-ocr-samples",
        str(config.eval_ocr_samples),
        "--eval-structured-samples",
        str(config.eval_structured_samples),
        "--eval-sample-max-new",
        str(config.eval_sample_max_new),
        "--eval-sample-exclude-sources",
        config.eval_sample_exclude_sources,
        "--log-every",
        str(config.log_every),
        "--profile-steps",
        str(config.profile_steps),
        "--operator-profile-steps",
        "0",
        "--seed",
        str(config.seed),
        "--resume",
        str(resume_file) if resume_file is not None else "none",
        "--sandwich-prompt" if config.sandwich_prompt else "--no-sandwich-prompt",
        "--loop-index" if config.loop_index else "--no-loop-index",
        (
            "--prefetch-next-batch"
            if config.prefetch_next_batch
            else "--no-prefetch-next-batch"
        ),
        "--require-fused-ce" if config.require_fused_ce else "--no-require-fused-ce",
    ]
    if config.engram:
        arguments.append("--engram")
    if config.arm == "compressor":
        arguments.extend(
            (
                "--vision-compressor-checkpoint",
                str(compressor),
                "--fusion-feature-cache",
                str(fusion_cache),
                "--fusion-cache-only",
                "--siglip2-model",
                str(siglip2),
                "--dinov2-model",
                str(dinov2),
                "--sam-model",
                str(sam),
            )
        )

    from rwkv_lab.vision_train import train

    result = train(
        arguments,
        worker_components=components,
        worker_step_profiler=step_profiler or NullStepProfiler(),
        worker_observability=observability,
        worker_controls=controls,
    )
    if not isinstance(result, Mapping):
        raise AdapterDispatchError("frozen vision omitted its terminal result")
    status = result.get("status")
    if status not in {"complete", "interrupted"}:
        raise AdapterDispatchError("frozen vision returned an invalid status")
    best_eval_loss = result.get("best_eval_loss")
    if status == "complete" and (
        isinstance(best_eval_loss, bool)
        or not isinstance(best_eval_loss, (int, float))
        or not math.isfinite(float(best_eval_loss))
    ):
        raise AdapterDispatchError(
            "frozen vision completed without a finite best evaluation loss"
        )
    request = checkpoint_request(
        invocation,
        run_directory,
        result.get("checkpoint"),
        result.get("step"),
        resume_grade="compatible",
        state_components=(
            "component_composition",
            "control_revision",
            "data_cursor",
            "model",
            "optimizer",
            "rng_accelerator",
            "rng_python",
            "rng_torch",
        ),
    )
    result_requests: tuple[ArtifactPublicationRequest, ...] = ()
    if status == "complete":
        result_directory = _stage_canonical_json_artifact(
            run_directory,
            attempt_id=getattr(invocation, "attempt_id", "attempt"),
            stem="scalar-result",
            filename="result.json",
            document={
                "api_version": "rwkv-lab.scalar-metric-result/v1",
                "direction": "minimize",
                "metric": "eval.loss",
                "optimizer_step": request.optimizer_step,
                "subject": config.arm,
                "value": float(best_eval_loss),
            },
        )
        result_requests = (
            ArtifactPublicationRequest(
                source_directory=result_directory,
                output_name="result",
            ),
        )
    return HandlerResult(
        "operation.failed" if status == "interrupted" else "worker.completed",
        {"reason": completion_reason(status), "arm": config.arm},
        optimizer_step=request.optimizer_step,
        checkpoint_requests=(request,),
        artifact_requests=result_requests,
    )


def _vision_native_head(
    invocation: WorkerInvocation,
    components: WorkerTrainingComponents,
    step_profiler: WorkerStepProfiler | None = None,
    observability: WorkerObservability | None = None,
    controls: WorkerControlRuntime | None = None,
    execution_phases: WorkerExecutionPhases | None = None,
) -> HandlerResult:
    """Calibrate the compressor-owned native RWKV output head."""

    if not declares_checkpoint(invocation):
        raise AdapterDispatchError(
            "vision native-head invocation omits its required checkpoint artifact"
        )
    effective_controls = (
        getattr(controls, "effective_values", {}) if controls is not None else {}
    )
    if not isinstance(effective_controls, Mapping):
        raise AdapterDispatchError(
            "vision native-head received an invalid control snapshot"
        )
    if effective_controls:
        raise AdapterDispatchError(
            "vision native-head v1 does not declare initial controls"
        )
    verification = (
        observability.keepalive(0, "verifying_inputs")
        if observability is not None
        else nullcontext()
    )
    with verification:
        paths = WorkspacePathAuthority.from_workspace(
            invocation.workspace, require_content=True
        )
        config = VisionNativeHeadConfig(**read_inline_config(invocation.inputs))
        baseline = paths.read_path(
            config.baseline_checkpoint, label="baseline_checkpoint", kind="file"
        )
        compressor = paths.read_path(
            config.compressor_checkpoint,
            label="compressor_checkpoint",
            kind="file",
        )
        train_manifest = paths.read_path(
            config.train_manifest, label="train_manifest", kind="file"
        )
        eval_manifest = paths.read_path(
            config.eval_manifest, label="eval_manifest", kind="file"
        )
        for label, manifest in (
            ("train_manifest", train_manifest),
            ("eval_manifest", eval_manifest),
        ):
            paths.verify_jsonl_file_references(
                manifest, fields=("image", "image_path"), label=label
            )
        moon_cache = paths.read_path(
            config.moon_cache, label="moon_cache", kind="directory"
        )
        fusion_cache = paths.read_path(
            config.fusion_cache, label="fusion_cache", kind="directory"
        )
        moonvit = paths.read_path(
            config.moonvit_checkpoint, label="moonvit_checkpoint", kind="file"
        )
        siglip2 = paths.read_path(
            config.siglip2_model, label="siglip2_model", kind="directory"
        )
        dinov2 = paths.read_path(
            config.dinov2_model, label="dinov2_model", kind="directory"
        )
        sam = paths.read_path(config.sam_model, label="sam_model", kind="directory")
        vocab = paths.read_path(config.vocab, label="vocab", kind="file")
        run_directory = paths.exact_run_directory(config.output_dir)
        resume_payload = _resume_payload(
            invocation,
            paths,
            required_state=frozenset(
                {
                    "component_composition",
                    "control_revision",
                    "model",
                    "optimizer",
                    "rng_accelerator",
                    "rng_python",
                    "rng_torch",
                }
            ),
        )
        resume_file = (
            paths.read_path(
                str(resume_payload / "state.pt"),
                label="resume checkpoint state",
                kind="file",
                require_content_identity=False,
            )
            if resume_payload is not None
            else None
        )

    from rwkv_lab.vision_native_train import train

    result = train(
        Namespace(
            baseline=str(baseline),
            compressor=str(compressor),
            data=str(train_manifest),
            eval_data=str(eval_manifest),
            moon_cache=str(moon_cache),
            fusion_cache=str(fusion_cache),
            moonvit=str(moonvit),
            siglip2=str(siglip2),
            dinov2=str(dinov2),
            sam=str(sam),
            vocab=str(vocab),
            out=str(run_directory),
            steps=config.steps,
            batch=config.batch_size,
            workers=config.workers,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            grad_clip=config.max_gradient_norm,
            eval_every=config.eval_every,
            eval_examples=config.eval_examples,
            checkpoint_every=config.checkpoint_every,
            seed=config.seed,
            resume="none",
            resume_from=str(resume_file) if resume_file is not None else "",
            device=config.device,
        ),
        worker_components=components,
        worker_step_profiler=step_profiler or NullStepProfiler(),
        worker_observability=observability,
        worker_controls=controls,
    )
    if not isinstance(result, Mapping):
        raise AdapterDispatchError("vision native-head omitted its terminal result")
    status = result.get("status")
    if status not in {"complete", "interrupted"}:
        raise AdapterDispatchError("vision native-head returned an invalid status")
    request = checkpoint_request(
        invocation,
        run_directory,
        result.get("checkpoint"),
        result.get("step"),
        resume_grade="compatible",
        state_components=(
            "component_composition",
            "control_revision",
            "model",
            "optimizer",
            "rng_accelerator",
            "rng_python",
            "rng_torch",
        ),
    )
    return HandlerResult(
        "operation.failed" if status == "interrupted" else "worker.completed",
        {"reason": completion_reason(status)},
        optimizer_step=request.optimizer_step,
        checkpoint_requests=(request,),
    )


def _scalar_metric_decision(
    invocation: WorkerInvocation,
    _components: WorkerTrainingComponents | None,
    _step_profiler: WorkerStepProfiler | None = None,
    observability: WorkerObservability | None = None,
    controls: WorkerControlRuntime | None = None,
    execution_phases: WorkerExecutionPhases | None = None,
) -> HandlerResult:
    """Compare two immutable scalar results and publish one lineage-bound receipt."""

    if getattr(invocation, "resume", None) is not None:
        raise AdapterDispatchError("scalar metric decision is stateless")
    if not _declares_artifact_output(invocation, "decision"):
        raise AdapterDispatchError(
            "scalar metric decision omits its required decision artifact"
        )
    effective_controls = (
        getattr(controls, "effective_values", {}) if controls is not None else {}
    )
    if not isinstance(effective_controls, Mapping) or effective_controls:
        raise AdapterDispatchError("scalar metric decision rejects controls")
    inputs = getattr(invocation, "inputs", None)
    if not isinstance(inputs, Mapping) or set(inputs) != {"config", "left", "right"}:
        raise AdapterDispatchError(
            "scalar metric decision requires exactly config, left, and right inputs"
        )
    verification = (
        observability.keepalive(0, "verifying_inputs")
        if observability is not None
        else nullcontext()
    )
    with verification:
        config = ScalarMetricDecisionConfig(
            **read_inline_config({"config": inputs["config"]})
        )
        left = resolve_input_artifact(
            invocation,
            "left",
            required_kind="report",
            required_schema="rwkv-lab.scalar-metric-result.v1",
        )
        right = resolve_input_artifact(
            invocation,
            "right",
            required_kind="report",
            required_schema="rwkv-lab.scalar-metric-result.v1",
        )
        paths = WorkspacePathAuthority.from_workspace(
            invocation.workspace, require_content=False
        )
        run_directory = paths.node_run_directory(invocation.node_id)

    def load_result(artifact, expected_subject: str) -> tuple[float, int]:
        document = load_input_artifact_json(
            artifact, "result.json", maximum_bytes=64 * 1024
        )
        if set(document) != {
            "api_version",
            "direction",
            "metric",
            "optimizer_step",
            "subject",
            "value",
        }:
            raise AdapterDispatchError("scalar metric result fields are inexact")
        value = document.get("value")
        step = document.get("optimizer_step")
        if (
            document.get("api_version") != "rwkv-lab.scalar-metric-result/v1"
            or document.get("metric") != config.metric
            or document.get("direction") != config.direction
            or document.get("subject") != expected_subject
            or isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or isinstance(step, bool)
            or not isinstance(step, int)
            or step < 0
        ):
            raise AdapterDispatchError("scalar metric result semantics are invalid")
        return float(value), step

    left_value, left_step = load_result(left, config.left_subject)
    right_value, right_step = load_result(right, config.right_subject)
    signed_delta = right_value - left_value
    if abs(signed_delta) <= config.absolute_tolerance:
        outcome = "tie"
        selected_subject = None
        selected_artifact_id = None
    else:
        left_wins = (
            left_value < right_value
            if config.direction == "minimize"
            else left_value > right_value
        )
        outcome = "selected"
        selected_subject = config.left_subject if left_wins else config.right_subject
        selected_artifact_id = left.artifact_id if left_wins else right.artifact_id
    decision_directory = _stage_canonical_json_artifact(
        run_directory,
        attempt_id=getattr(invocation, "attempt_id", "attempt"),
        stem="scalar-decision",
        filename="decision.json",
        document={
            "absolute_tolerance": config.absolute_tolerance,
            "api_version": "rwkv-lab.scalar-metric-decision/v1",
            "candidates": [
                {
                    "artifact_id": left.artifact_id,
                    "optimizer_step": left_step,
                    "subject": config.left_subject,
                    "value": left_value,
                },
                {
                    "artifact_id": right.artifact_id,
                    "optimizer_step": right_step,
                    "subject": config.right_subject,
                    "value": right_value,
                },
            ],
            "direction": config.direction,
            "metric": config.metric,
            "outcome": outcome,
            "selected_artifact_id": selected_artifact_id,
            "selected_subject": selected_subject,
            "signed_right_minus_left": signed_delta,
        },
    )
    return HandlerResult(
        "operation.completed",
        {"outcome": outcome, "selected_subject": selected_subject},
        artifact_requests=(
            ArtifactPublicationRequest(
                source_directory=decision_directory,
                output_name="decision",
                parent_artifact_ids=(left.artifact_id, right.artifact_id),
            ),
        ),
    )


def _vision_rwkv_student(
    invocation: WorkerInvocation,
    components: WorkerTrainingComponents,
    step_profiler: WorkerStepProfiler | None = None,
    observability: WorkerObservability | None = None,
    controls: WorkerControlRuntime | None = None,
    execution_phases: WorkerExecutionPhases | None = None,
) -> HandlerResult:
    """Distill the frozen vision stack into the raw-pixel RWKV student."""

    if not declares_checkpoint(invocation):
        raise AdapterDispatchError(
            "vision student invocation omits its required checkpoint artifact"
        )
    effective_controls = (
        getattr(controls, "effective_values", {}) if controls is not None else {}
    )
    if not isinstance(effective_controls, Mapping):
        raise AdapterDispatchError(
            "vision student received an invalid control snapshot"
        )
    if effective_controls:
        raise AdapterDispatchError(
            "vision student v1 does not declare initial controls"
        )

    verification = (
        observability.keepalive(0, "verifying_inputs")
        if observability is not None
        else nullcontext()
    )
    with verification:
        paths = WorkspacePathAuthority.from_workspace(
            invocation.workspace, require_content=True
        )
        config = VisionRWKVStudentConfig(**read_inline_config(invocation.inputs))
        baseline = paths.read_path(
            config.baseline_checkpoint, label="baseline_checkpoint", kind="file"
        )
        compressor = paths.read_path(
            config.compressor_checkpoint,
            label="compressor_checkpoint",
            kind="file",
        )
        native_head = paths.read_path(
            config.native_head_checkpoint,
            label="native_head_checkpoint",
            kind="file",
        )
        train_manifest = paths.read_path(
            config.train_manifest, label="train_manifest", kind="file"
        )
        eval_manifest = paths.read_path(
            config.eval_manifest, label="eval_manifest", kind="file"
        )
        for label, manifest in (
            ("train_manifest", train_manifest),
            ("eval_manifest", eval_manifest),
        ):
            paths.verify_jsonl_file_references(
                manifest, fields=("image", "image_path"), label=label
            )
        moon_cache = paths.read_path(
            config.moon_cache, label="moon_cache", kind="directory"
        )
        fusion_cache = paths.read_path(
            config.fusion_cache, label="fusion_cache", kind="directory"
        )
        moonvit = paths.read_path(
            config.moonvit_checkpoint, label="moonvit_checkpoint", kind="file"
        )
        siglip2 = paths.read_path(
            config.siglip2_model, label="siglip2_model", kind="directory"
        )
        dinov2 = paths.read_path(
            config.dinov2_model, label="dinov2_model", kind="directory"
        )
        sam = paths.read_path(config.sam_model, label="sam_model", kind="directory")
        vocab = paths.read_path(config.vocab, label="vocab", kind="file")
        run_directory = paths.exact_run_directory(config.output_dir)
        resume_payload = _resume_payload(
            invocation,
            paths,
            required_state=frozenset(
                {
                    "component_composition",
                    "control_revision",
                    "data_cursor",
                    "model",
                    "optimizer",
                    "rng_accelerator",
                    "rng_python",
                    "rng_torch",
                }
            ),
        )
        resume_file = (
            paths.read_path(
                str(resume_payload / "state.pt"),
                label="resume checkpoint state",
                kind="file",
                require_content_identity=False,
            )
            if resume_payload is not None
            else None
        )

    from rwkv_lab.vision_rwkv_student_train import train

    result = train(
        Namespace(
            baseline=str(baseline),
            compressor=str(compressor),
            native_head=str(native_head),
            data=str(train_manifest),
            eval_data=str(eval_manifest),
            moon_cache=str(moon_cache),
            fusion_cache=str(fusion_cache),
            moonvit=str(moonvit),
            siglip2=str(siglip2),
            dinov2=str(dinov2),
            sam=str(sam),
            vocab=str(vocab),
            out=str(run_directory),
            steps=config.steps,
            batch=config.batch_size,
            workers=config.workers,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            grad_clip=config.max_gradient_norm,
            caption_weight=config.caption_weight,
            teacher_reconstruction_weight=config.teacher_reconstruction_weight,
            latent_relational_weight=config.latent_relational_weight,
            teacher_relational_weight=config.teacher_relational_weight,
            variance_weight=config.variance_weight,
            covariance_weight=config.covariance_weight,
            diversity_weight=config.diversity_weight,
            eval_every=config.eval_every,
            eval_examples=config.eval_examples,
            checkpoint_every=config.checkpoint_every,
            seed=config.seed,
            resume="none",
            resume_from=str(resume_file) if resume_file is not None else "",
            device=config.device,
            image_size=config.image_size,
            grid_size=config.grid_size,
            hidden_size=config.hidden_size,
            layers=config.layers,
            head_size=config.head_size,
            ffn_hidden=config.ffn_hidden,
            no_checkpoint_blocks=not config.checkpoint_blocks,
            preflight_only=False,
        ),
        worker_components=components,
        worker_step_profiler=step_profiler or NullStepProfiler(),
        worker_observability=observability,
        worker_controls=controls,
    )
    if not isinstance(result, Mapping):
        raise AdapterDispatchError("vision student omitted its terminal result")
    status = result.get("status")
    if status not in {"complete", "interrupted"}:
        raise AdapterDispatchError("vision student returned an invalid status")
    request = checkpoint_request(
        invocation,
        run_directory,
        result.get("checkpoint"),
        result.get("step"),
        resume_grade="compatible",
        state_components=(
            "component_composition",
            "control_revision",
            "data_cursor",
            "model",
            "optimizer",
            "rng_accelerator",
            "rng_python",
            "rng_torch",
        ),
    )
    return HandlerResult(
        "operation.failed" if status == "interrupted" else "worker.completed",
        {"reason": completion_reason(status)},
        optimizer_step=request.optimizer_step,
        checkpoint_requests=(request,),
    )


def _transformer_mla(
    invocation: WorkerInvocation,
    components: WorkerTrainingComponents,
    step_profiler: WorkerStepProfiler | None = None,
    observability: WorkerObservability | None = None,
    controls: WorkerControlRuntime | None = None,
    execution_phases: WorkerExecutionPhases | None = None,
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

    model_dir = paths.read_path(config.model_dir, label="model_dir", kind="directory")
    patch_dir = paths.read_path(config.patch_dir, label="patch_dir", kind="directory")
    tokens_bin = paths.read_path(config.tokens_bin, label="tokens_bin", kind="file")
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
        raise AdapterDispatchError(
            "Transformer MLA trainer omitted its terminal result"
        )
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
                "checkpointed_interruption" if interrupted else "training_complete"
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
        "rwkv-lab.mageflow-full-backbone",
        "1.0.0",
        "train",
        "rwkv_lab.mageflow_full_backbone.v1.Train",
    ): _mageflow_full_backbone,
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
        "rwkv-lab.scalar-metric-decision",
        "1.0.0",
        "decide",
        "rwkv_lab.scalar_metric_decision.v1.Decide",
    ): _scalar_metric_decision,
    (
        "rwkv-lab.rwkv-optimizer-finetune",
        "1.0.0",
        "train",
        "rwkv_lab.rwkv_optimizer_finetune.v1.Train",
    ): _rwkv_optimizer_finetune,
    (
        "rwkv-lab.rwkv-posttraining",
        "1.0.0",
        "train",
        "rwkv_lab.rwkv_posttraining.v1.Train",
    ): _rwkv_posttraining,
    (
        "rwkv-lab.rwkv-rlvr",
        "1.0.0",
        "train",
        "rwkv_lab.rwkv_rlvr.v1.Train",
    ): _rlvr,
    (
        "rwkv-lab.rwkv-scratch",
        "1.0.0",
        "train",
        "rwkv_lab.rwkv_scratch.v1.Train",
    ): _rwkv_scratch,
    (
        "rwkv-lab.vision-teacher-compressor",
        "1.0.0",
        "train",
        "rwkv_lab.vision_teacher_compressor.v1.Train",
    ): _vision_teacher_compressor,
    (
        "rwkv-lab.vision-frozen-adapter",
        "1.0.0",
        "train",
        "rwkv_lab.vision_frozen_adapter.v1.Train",
    ): _vision_frozen_adapter,
    (
        "rwkv-lab.vision-native-head",
        "1.0.0",
        "train",
        "rwkv_lab.vision_native_head.v1.Train",
    ): _vision_native_head,
    (
        "rwkv-lab.vision-rwkv-student",
        "1.0.0",
        "train",
        "rwkv_lab.vision_rwkv_student.v1.Train",
    ): _vision_rwkv_student,
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
    execution_phases: WorkerExecutionPhases | None = None,
    publications: WorkerPublicationRuntime | None = None,
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
    if execution_phases is not None and handler not in {
        _appearance_expert,
        _mageflow_full_backbone,
        _rwkv_scratch,
        _terminal_expert,
    }:
        preinitialization_state = {
            "lifecycle": "pre_initialization",
            "invocation_digest": invocation.invocation_digest,
        }
        for phase in sorted(execution_phases.phases, key=lambda item: item.value):
            request = execution_phases.request(phase)
            if request is None:  # pragma: no cover - coordinator owns this set
                raise AdapterDispatchError("execution phase request disappeared")

            def unsupported(_steps, _mark_step, *, phase_name=phase.value):
                raise AdapterDispatchError(
                    f"adapter does not implement enabled {phase_name} phase"
                )

            execution_phases.run(
                phase,
                snapshot=lambda: preinitialization_state,
                execute=unsupported,
            )
    if invocation.training is None:
        if handler is not _scalar_metric_decision:
            raise AdapterDispatchError("training adapter has no resolved composition")
        components = None
    else:
        components = WorkerTrainingComponents(
            invocation.training, invocation.training.model_family
        )
    if observability is None:
        raise AdapterDispatchError(
            "training adapter has no worker observability authority"
        )
    if controls is None:
        raise AdapterDispatchError("training adapter has no worker control authority")
    arguments = (
        invocation,
        components,
        step_profiler or NullStepProfiler(),
        observability,
        controls,
        execution_phases,
    )
    if handler in {_appearance_expert, _mageflow_full_backbone, _terminal_expert}:
        return handler(*arguments, publications)
    return handler(*arguments)
