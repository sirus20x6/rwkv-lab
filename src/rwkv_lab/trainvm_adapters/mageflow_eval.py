"""Immutable-checkpoint standalone evaluation for terminal MageFlow routes.

The evaluator reconstructs the exact architecture declared by the run contract
embedded in a published checkpoint.  It deliberately has no optimizer,
training loop, checkpoint writer, or mutable-resume path.
"""

from __future__ import annotations

import json
import math
import os
import random
import stat
from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

from rwkv_lab.mage_flow_adaptation import (
    EXPERT_DOMAINS,
    MAGE_FLOW_BASE_ID,
    MAGE_FLOW_BASE_REVISION,
)


class MageFlowStandaloneEvalError(ValueError):
    pass


_MAXIMUM_CONTRACT_BYTES = 4 * 1024 * 1024
_CONFIG_FIELDS = frozenset(
    {
        "domain",
        "eval_manifest",
        "model_path",
        "eval_microbatch_size",
        "eval_examples",
        "attention_backend",
    }
)


def _bounded_json_object(path: Path) -> dict[str, Any]:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise MageFlowStandaloneEvalError(
            "checkpoint contract file is unavailable"
        ) from error
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_size <= 0
            or before.st_size > _MAXIMUM_CONTRACT_BYTES
        ):
            raise MageFlowStandaloneEvalError("checkpoint contract file is invalid")
        chunks: list[bytes] = []
        total = 0
        while chunk := os.read(descriptor, 64 * 1024):
            total += len(chunk)
            if total > _MAXIMUM_CONTRACT_BYTES:
                raise MageFlowStandaloneEvalError(
                    "checkpoint contract file exceeds its byte bound"
                )
            chunks.append(chunk)
        after = os.fstat(descriptor)
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ):
            raise MageFlowStandaloneEvalError(
                "checkpoint contract file changed while read"
            )
    finally:
        os.close(descriptor)
    try:
        value = json.loads(b"".join(chunks))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise MageFlowStandaloneEvalError(
            "checkpoint contract file is malformed"
        ) from error
    if not isinstance(value, dict):
        raise MageFlowStandaloneEvalError("checkpoint contract is not an object")
    return value


def _checkpoint_file(
    checkpoint_directory: Path,
    value: object,
    *,
    label: str,
    required: bool,
) -> Path | None:
    if value is None and not required:
        return None
    if (
        not isinstance(value, str)
        or not value
        or Path(value).is_absolute()
        or Path(value).name != value
        or value in {".", ".."}
    ):
        raise MageFlowStandaloneEvalError(f"checkpoint {label} path is invalid")
    path = checkpoint_directory / value
    try:
        info = path.stat(follow_symlinks=False)
    except OSError as error:
        raise MageFlowStandaloneEvalError(
            f"checkpoint {label} is unavailable"
        ) from error
    if not stat.S_ISREG(info.st_mode) or info.st_size <= 0:
        raise MageFlowStandaloneEvalError(
            f"checkpoint {label} is not a nonempty regular file"
        )
    return path


@dataclass(frozen=True, slots=True)
class MageFlowStandaloneEvalConfig:
    domain: str
    eval_manifest: str
    model_path: str
    eval_microbatch_size: int = 1
    eval_examples: int | None = None
    attention_backend: str | None = None

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> MageFlowStandaloneEvalConfig:
        if set(value) - _CONFIG_FIELDS:
            raise MageFlowStandaloneEvalError(
                "standalone evaluation configuration fields are inexact"
            )
        try:
            result = cls(**dict(value))
        except TypeError as error:
            raise MageFlowStandaloneEvalError(
                "standalone evaluation configuration is incomplete"
            ) from error
        result.validate()
        return result

    def validate(self) -> None:
        if self.domain not in EXPERT_DOMAINS:
            raise MageFlowStandaloneEvalError("domain must be photo or animation")
        if not isinstance(self.eval_manifest, str) or not self.eval_manifest:
            raise MageFlowStandaloneEvalError("eval_manifest must be a path")
        if not isinstance(self.model_path, str) or not self.model_path:
            raise MageFlowStandaloneEvalError("model_path must be a path")
        if (
            not isinstance(self.eval_microbatch_size, int)
            or isinstance(self.eval_microbatch_size, bool)
            or self.eval_microbatch_size < 1
        ):
            raise MageFlowStandaloneEvalError(
                "eval_microbatch_size must be a positive integer"
            )
        if self.eval_examples is not None and (
            not isinstance(self.eval_examples, int)
            or isinstance(self.eval_examples, bool)
            or self.eval_examples < 1
        ):
            raise MageFlowStandaloneEvalError(
                "eval_examples must be a positive integer or null"
            )
        if self.attention_backend not in {None, "flash2", "flash4"}:
            raise MageFlowStandaloneEvalError(
                "attention_backend must be flash2, flash4, or null"
            )


@dataclass(frozen=True, slots=True)
class ResolvedMageFlowEvaluation:
    trainer_config: Any
    optimizer_step: int
    expert_checkpoint: Path
    shared_backbone_checkpoint: Path | None
    repa_projection_checkpoint: Path | None
    tread_loop_checkpoint: Path | None


def resolve_mageflow_evaluation(
    request: MageFlowStandaloneEvalConfig,
    checkpoint_directory: Path,
    output_directory: Path,
    *,
    optimizer_step: int,
) -> ResolvedMageFlowEvaluation:
    """Resolve an eval-only configuration from an embedded training contract."""

    from rwkv_lab.mage_flow_terminal_train import (
        RUN_SCHEMA,
        TerminalExpertTrainConfig,
    )
    from rwkv_lab.mage_flow_tread_looping import TreadLoopConfig

    checkpoint_directory = checkpoint_directory.expanduser().resolve(strict=True)
    if not checkpoint_directory.is_dir():
        raise MageFlowStandaloneEvalError("checkpoint payload is not a directory")
    checkpoint = _bounded_json_object(checkpoint_directory / "checkpoint.json")
    checkpoint_step = checkpoint.get("step")
    fingerprint = checkpoint.get("fingerprint")
    if (
        checkpoint.get("schema") != RUN_SCHEMA
        or not isinstance(checkpoint_step, int)
        or isinstance(checkpoint_step, bool)
        or checkpoint_step < 0
        or checkpoint_step != optimizer_step
        or not isinstance(fingerprint, str)
        or not fingerprint
    ):
        raise MageFlowStandaloneEvalError(
            "checkpoint metadata disagrees with checkpoint authority"
        )

    run_contract_path = _checkpoint_file(
        checkpoint_directory,
        checkpoint.get("run_contract"),
        label="run contract",
        required=True,
    )
    assert run_contract_path is not None
    run_contract = _bounded_json_object(run_contract_path)
    raw_training_config = run_contract.get("config")
    if (
        run_contract.get("schema") != RUN_SCHEMA
        or run_contract.get("fingerprint") != fingerprint
        or not isinstance(raw_training_config, Mapping)
    ):
        raise MageFlowStandaloneEvalError(
            "embedded run contract disagrees with checkpoint metadata"
        )
    allowed_training_fields = {item.name for item in fields(TerminalExpertTrainConfig)}
    if set(raw_training_config) - allowed_training_fields:
        raise MageFlowStandaloneEvalError(
            "embedded run contract contains unknown training fields"
        )
    try:
        original = TerminalExpertTrainConfig(**dict(raw_training_config))
    except (TypeError, ValueError) as error:
        raise MageFlowStandaloneEvalError(
            "embedded training configuration is invalid"
        ) from error
    if (
        original.model_id != MAGE_FLOW_BASE_ID
        or original.model_revision != MAGE_FLOW_BASE_REVISION
    ):
        raise MageFlowStandaloneEvalError(
            "embedded training contract is not pinned to Mage-Flow-Base"
        )

    experts = checkpoint.get("experts")
    if isinstance(experts, Mapping):
        expert_name = experts.get(request.domain)
    elif checkpoint.get("domain") == request.domain:
        expert_name = checkpoint.get("expert")
    else:
        expert_name = None
    expert = _checkpoint_file(
        checkpoint_directory,
        expert_name,
        label=f"{request.domain} expert",
        required=True,
    )
    assert expert is not None
    shared = _checkpoint_file(
        checkpoint_directory,
        checkpoint.get("shared_backbone"),
        label="shared backbone",
        required=False,
    )
    repa = _checkpoint_file(
        checkpoint_directory,
        checkpoint.get("repa_projection"),
        label="REPA projection",
        required=bool(original.repa_enabled),
    )
    if not original.repa_enabled and repa is not None:
        raise MageFlowStandaloneEvalError(
            "checkpoint has REPA state but its run contract disables REPA"
        )
    loop_configuration = TreadLoopConfig.from_dict(original.tread_factored_looping)
    tread = _checkpoint_file(
        checkpoint_directory,
        checkpoint.get("tread_loop_controller"),
        label="TREAD controller",
        required=bool(loop_configuration.combined.enabled),
    )
    if not loop_configuration.combined.enabled and tread is not None:
        raise MageFlowStandaloneEvalError(
            "checkpoint has TREAD state but its run contract disables TREAD"
        )

    values = asdict(original)
    values.update(
        {
            "domain": request.domain,
            "train_manifest": request.eval_manifest,
            "eval_manifest": request.eval_manifest,
            "expert_checkpoint": str(expert),
            "shared_backbone_checkpoint": str(shared) if shared is not None else None,
            "resume_from": None,
            "output_dir": str(output_directory),
            "model_path": request.model_path,
            "max_steps": max(1, optimizer_step),
            "eval_microbatch_size": request.eval_microbatch_size,
            "eval_examples": request.eval_examples,
            "eval_on_resume": False,
            "gradient_checkpointing": False,
            "activation_checkpointing_mode": "none",
            "compile_vae_encoder": False,
            "encoder_cache_dir": None,
            "encoder_cache_mode": "off",
            "offload_cached_encoders": False,
            "encoder_cache_coverage_manifest": None,
            "encoder_cache_covered_until_step": None,
            "repa_reset_projection_on_resume": False,
            "allow_objective_migration_on_resume": False,
            "reset_optimizer_on_architecture_migration": False,
            "tread_loop_checkpoint": None,
            "domain_window_schedule": None,
            "expert_checkpoints": None,
            "rapid_expert_alternation": False,
            "expert_optimizer_state_device": "cpu",
        }
    )
    if request.attention_backend is not None:
        values["attention_backend"] = request.attention_backend
    try:
        trainer_config = TerminalExpertTrainConfig(**values)
        trainer_config.validate()
    except (TypeError, ValueError) as error:
        raise MageFlowStandaloneEvalError(
            "checkpoint-derived evaluation configuration is invalid"
        ) from error
    return ResolvedMageFlowEvaluation(
        trainer_config=trainer_config,
        optimizer_step=optimizer_step,
        expert_checkpoint=expert,
        shared_backbone_checkpoint=shared,
        repa_projection_checkpoint=repa,
        tread_loop_checkpoint=tread,
    )


def evaluate_mageflow_checkpoint(
    evaluation: ResolvedMageFlowEvaluation,
    output_directory: Path,
) -> dict[str, float | int]:
    """Evaluate one immutable checkpoint without constructing training state."""

    import torch

    from rwkv_lab.mage_flow_expert_train import annotate_domain_token_lengths
    from rwkv_lab.mage_flow_optimizations import (
        configure_activation_checkpointing,
        convert_trainable_image_ffns_to_float8,
    )
    from rwkv_lab.mage_flow_terminal_experts import (
        configure_terminal_training_scope,
        convert_terminal_path_to_lightning_blocks,
        install_terminal_expert,
        load_terminal_expert,
        load_terminal_shared_backbone,
        terminal_architecture_report,
    )
    from rwkv_lab.mage_flow_terminal_train import _domain_rows, _run_evaluation
    from rwkv_lab.mage_flow_training_objectives import (
        VAERepresentationAlignment,
        load_repa_projection,
    )
    from rwkv_lab.mage_flow_tread_looping import (
        TreadLoopConfig,
        install_tread_factored_looping,
        load_tread_loop_controller,
    )

    try:
        from mage_flow import MageFlowPipeline
    except ImportError as error:
        raise RuntimeError("use the isolated Mage-Flow environment") from error
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("standalone MageFlow evaluation requires BF16 CUDA")

    config = evaluation.trainer_config
    device = torch.device("cuda", torch.cuda.current_device())
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    output_directory = output_directory.expanduser().resolve()
    output_directory.mkdir(mode=0o750, parents=True, exist_ok=True)

    pipeline = MageFlowPipeline.from_pretrained(
        str(Path(config.model_path).expanduser().resolve()),
        device=str(device),
        attn_type=config.attention_backend,
    )
    model = pipeline.model
    model.vae.sample_posterior = config.vae_sample_posterior
    model.config.compile_vae_encoder = False
    model.vae.eval().requires_grad_(False)
    model.txt_enc.eval().requires_grad_(False)
    transformer = model.transformer
    controller = install_terminal_expert(
        transformer,
        config.domain,
        dtype=next(transformer.parameters()).dtype,
        device=device,
    )
    loop_config = TreadLoopConfig.from_dict(config.tread_factored_looping)
    loop_controller = (
        install_tread_factored_looping(
            transformer,
            loop_config,
            offload_replaced_source_blocks=config.offload_replaced_backbone_core,
        )
        if loop_config.combined.enabled
        else None
    )
    convert_terminal_path_to_lightning_blocks(
        transformer,
        controller,
        use_swiglu=config.lightning_swiglu,
        use_rmsnorm=config.lightning_rmsnorm,
    )
    configure_terminal_training_scope(
        transformer,
        controller,
        train_backbone_final_fraction=config.train_backbone_final_fraction,
    )
    repa = None
    if config.repa_enabled:
        repa = VAERepresentationAlignment(
            int(transformer.inner_dim),
            int(transformer.img_in.in_features),
            hidden_dim=config.repa_projection_hidden_dim,
            smooth_l1_beta=config.repa_smooth_l1_beta,
            student_normalization=config.repa_student_normalization,
            per_example_loss_cap=config.repa_per_example_loss_cap,
        ).to(device=device, dtype=next(transformer.parameters()).dtype)
    convert_trainable_image_ffns_to_float8(
        transformer,
        enabled=config.float8_training,
        recipe=config.float8_recipe,
    )
    load_terminal_expert(
        controller,
        config.domain,
        evaluation.expert_checkpoint,
    )
    if evaluation.shared_backbone_checkpoint is not None:
        load_terminal_shared_backbone(
            transformer, evaluation.shared_backbone_checkpoint
        )
    if evaluation.tread_loop_checkpoint is not None:
        if loop_controller is None:
            raise MageFlowStandaloneEvalError(
                "checkpoint TREAD state has no installed controller"
            )
        load_tread_loop_controller(
            loop_controller, evaluation.tread_loop_checkpoint
        )
        loop_controller.refresh_inference_skip_refinements()
    if evaluation.repa_projection_checkpoint is not None:
        if repa is None:
            raise MageFlowStandaloneEvalError(
                "checkpoint REPA state has no installed projection"
            )
        load_repa_projection(repa, evaluation.repa_projection_checkpoint)
    architecture = terminal_architecture_report(controller)
    if architecture.get("passed") is not True:
        raise MageFlowStandaloneEvalError(
            "checkpoint-derived terminal architecture failed preflight"
        )
    configure_activation_checkpointing(transformer, "none")
    transformer.eval().requires_grad_(False)
    if repa is not None:
        repa.eval().requires_grad_(False)

    rows = _domain_rows(config.eval_manifest, config.domain)
    annotate_domain_token_lengths(model, rows)
    observed = _run_evaluation(
        pipeline=pipeline,
        transformer=transformer,
        controller=controller,
        repa=repa,
        model=model,
        rows=rows,
        config=config,
        device=device,
        output_dir=output_directory,
        step=evaluation.optimizer_step,
    )
    metrics: dict[str, float | int] = {}
    for name, value in observed.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        if not math.isfinite(float(value)):
            raise MageFlowStandaloneEvalError(
                f"standalone evaluation metric {name!r} is not finite"
            )
        metrics[str(name)] = value
    if "eval/primary_loss" not in metrics:
        raise MageFlowStandaloneEvalError(
            "standalone evaluation omitted its primary loss"
        )
    return metrics


__all__ = [
    "MageFlowStandaloneEvalConfig",
    "MageFlowStandaloneEvalError",
    "ResolvedMageFlowEvaluation",
    "evaluate_mageflow_checkpoint",
    "resolve_mageflow_evaluation",
]
