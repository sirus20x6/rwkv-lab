"""Single-resident terminal-expert training for Mage-Flow.

Each run trains one domain window.  The complete 12-block Mage-Flow backbone
and exactly one three-block terminal expert are resident on the GPU.  The
inactive expert remains an independent checkpoint and is never added to the
module tree or optimizer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shlex
import shutil
import signal
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from rwkv_lab.mage_flow_adaptation import (
    EXPERT_DOMAINS,
    MAGE_FLOW_BASE_ID,
    MAGE_FLOW_BASE_REVISION,
    MAGE_SOURCE_REVISION,
    load_domain_manifest,
)
from rwkv_lab.mage_flow_expert_train import (
    EVAL_GALLERY_SAMPLES_PER_DOMAIN,
    _drop_vae_decoder,
    _forward_transformer,
    annotate_domain_token_lengths,
    cache_frozen_encoders,
    effective_conditioning_prompts,
    encode_domain_batch,
    generate_eval_gallery,
)
from rwkv_lab.mage_flow_optimizations import (
    ACTIVATION_CHECKPOINT_MODES,
    ENCODER_CACHE_MODES,
    FLOAT8_RECIPES,
    FrozenEncoderCache,
    cache_coverage,
    compile_transformer_blocks,
    configure_activation_checkpointing,
    convert_trainable_image_ffns_to_float8,
)
from rwkv_lab.mage_flow_pretrain import (
    _cosine_multiplier,
    _load_image_tensor,
    rectified_flow_loss,
)
from rwkv_lab.mage_flow_terminal_experts import (
    configure_terminal_training_scope,
    install_terminal_expert,
    load_terminal_expert,
    load_terminal_shared_backbone,
    save_terminal_expert,
    save_terminal_shared_backbone,
    terminal_architecture_report,
    terminal_optimizer_parameter_groups,
)

RUN_SCHEMA = "rwkv-lab.mage-flow-terminal-train.v1"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, payload: Any) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass
class TerminalExpertTrainConfig:
    domain: str
    train_manifest: str
    expert_checkpoint: str
    output_dir: str
    eval_manifest: str | None = None
    shared_backbone_checkpoint: str | None = None
    resume_from: str | None = None
    model_id: str = MAGE_FLOW_BASE_ID
    model_revision: str = MAGE_FLOW_BASE_REVISION
    official_source_revision: str = MAGE_SOURCE_REVISION
    max_steps: int = 2_000
    microbatch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 1e-5
    backbone_learning_rate_multiplier: float = 0.5
    train_backbone_final_fraction: float = 1 / 3
    warmup_steps: int = 100
    min_learning_rate_ratio: float = 0.1
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0
    caption_dropout: float = 0.1
    timestep_sampling: str = "uniform"
    timestep_shift: float = 1.0
    checkpoint_every: int = 250
    eval_every: int = 250
    eval_microbatch_size: int = 1
    eval_examples: int | None = None
    keep_last_n: int = 3
    seed: int = 42
    mixed_precision: str = "bf16"
    gradient_checkpointing: bool = True
    activation_checkpointing_mode: str = "full"
    vae_sample_posterior: bool = True
    compile_vae_encoder: bool = True
    attention_backend: str = "flash2"
    compile_transformer_blocks: bool = False
    compile_transformer_mode: str = "default"
    compile_transformer_dynamic: bool = True
    float8_training: bool = False
    float8_recipe: str = "tensorwise"
    encoder_cache_dir: str | None = None
    encoder_cache_mode: str = "off"
    offload_cached_encoders: bool = False

    @classmethod
    def from_path(cls, path: Path) -> TerminalExpertTrainConfig:
        return cls(**json.loads(path.read_text(encoding="utf-8")))

    def validate(self, *, inspect_manifests: bool = True) -> None:
        if self.domain not in EXPERT_DOMAINS:
            raise ValueError("domain must be photo or animation")
        if self.model_id != MAGE_FLOW_BASE_ID:
            raise ValueError("terminal training is pinned to Mage-Flow-Base")
        if self.model_revision != MAGE_FLOW_BASE_REVISION:
            raise ValueError("unqualified Mage-Flow revision")
        if self.official_source_revision != MAGE_SOURCE_REVISION:
            raise ValueError("unqualified Mage source revision")
        if not Path(self.train_manifest).expanduser().is_file():
            raise ValueError("training manifest does not exist")
        if not Path(self.expert_checkpoint).expanduser().is_file():
            raise ValueError("terminal expert checkpoint does not exist")
        if self.eval_manifest and not Path(self.eval_manifest).expanduser().is_file():
            raise ValueError("evaluation manifest does not exist")
        if self.shared_backbone_checkpoint and not Path(
            self.shared_backbone_checkpoint
        ).expanduser().is_file():
            raise ValueError("shared-backbone checkpoint does not exist")
        if self.resume_from and not Path(self.resume_from).expanduser().is_dir():
            raise ValueError("resume checkpoint directory does not exist")
        if self.max_steps < 1 or self.microbatch_size < 1:
            raise ValueError("step and microbatch counts must be positive")
        if self.gradient_accumulation_steps < 1:
            raise ValueError("gradient accumulation must be positive")
        if self.learning_rate <= 0 or self.backbone_learning_rate_multiplier <= 0:
            raise ValueError("learning rates must be positive")
        if not 0 <= self.train_backbone_final_fraction <= 1:
            raise ValueError("backbone fraction must be in [0, 1]")
        if not 0 <= self.caption_dropout < 1:
            raise ValueError("caption dropout must be in [0, 1)")
        if self.mixed_precision != "bf16":
            raise ValueError("terminal training is qualified only for BF16")
        if self.activation_checkpointing_mode not in ACTIVATION_CHECKPOINT_MODES:
            raise ValueError("unsupported activation_checkpointing_mode")
        if self.attention_backend not in {"flash2", "flash4"}:
            raise ValueError("attention_backend must be flash2 or flash4")
        if self.float8_recipe not in FLOAT8_RECIPES:
            raise ValueError("unsupported float8_recipe")
        if self.float8_training and not self.compile_transformer_blocks:
            raise ValueError(
                "float8_training requires compile_transformer_blocks"
            )
        if self.encoder_cache_mode not in ENCODER_CACHE_MODES:
            raise ValueError("unsupported encoder_cache_mode")
        if self.encoder_cache_mode != "off" and not self.encoder_cache_dir:
            raise ValueError("encoder_cache_mode requires encoder_cache_dir")
        if self.offload_cached_encoders and self.encoder_cache_mode != "read_only":
            raise ValueError(
                "offload_cached_encoders requires a complete read_only cache"
            )
        if self.checkpoint_every < 1 or self.eval_every < 1:
            raise ValueError("checkpoint and evaluation intervals must be positive")
        if self.eval_microbatch_size < 1 or self.keep_last_n < 1:
            raise ValueError("invalid evaluation/checkpoint configuration")
        if inspect_manifests:
            rows = load_domain_manifest(Path(self.train_manifest))
            if not any(row["domain"] == self.domain for row in rows):
                raise ValueError(f"manifest has no {self.domain} training rows")
            if self.eval_manifest:
                eval_rows = load_domain_manifest(Path(self.eval_manifest))
                if not any(row["domain"] == self.domain for row in eval_rows):
                    raise ValueError(f"manifest has no {self.domain} evaluation rows")


def _contract_fingerprint(config: TerminalExpertTrainConfig) -> str:
    values = asdict(config)
    values.pop("output_dir", None)
    values.pop("resume_from", None)
    payload = {
        "schema": RUN_SCHEMA,
        "config": values,
        "train_manifest_sha256": _sha256(Path(config.train_manifest)),
        "eval_manifest_sha256": (
            _sha256(Path(config.eval_manifest)) if config.eval_manifest else None
        ),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _domain_rows(path: str, domain: str) -> list[dict[str, Any]]:
    rows = [row for row in load_domain_manifest(Path(path)) if row["domain"] == domain]
    if not rows:
        raise ValueError(f"{path} contains no {domain} rows")
    return rows


def _epoch_batches(
    row_count: int,
    *,
    batch_size: int,
    seed: int,
    epoch: int,
) -> list[list[int]]:
    indices = list(range(row_count))
    random.Random(seed + epoch * 1_000_003).shuffle(indices)
    return [
        indices[start : start + batch_size]
        for start in range(0, len(indices), batch_size)
    ]


def _save_checkpoint(
    *,
    controller,
    transformer,
    optimizer,
    scheduler,
    output_dir: Path,
    domain: str,
    step: int,
    epoch: int,
    batch_index: int,
    fingerprint: str,
    keep: int,
) -> Path:
    final = output_dir / f"checkpoint-{step:08d}"
    if final.is_dir():
        return final
    temporary = output_dir / f".checkpoint-{step:08d}.incomplete"
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    expert_name = f"mageflow-{domain}-terminal-expert.safetensors"
    save_terminal_expert(controller, temporary / expert_name)
    shared = save_terminal_shared_backbone(
        transformer,
        controller,
        temporary / "mageflow-shared-final-third.safetensors",
    )
    torch.save(
        {
            "schema": RUN_SCHEMA,
            "step": step,
            "epoch": epoch,
            "batch_index": batch_index,
            "fingerprint": fingerprint,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "python_rng": random.getstate(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all(),
        },
        temporary / "trainer_state.pt",
    )
    _atomic_json(
        temporary / "checkpoint.json",
        {
            "schema": RUN_SCHEMA,
            "domain": domain,
            "step": step,
            "epoch": epoch,
            "batch_index": batch_index,
            "fingerprint": fingerprint,
            "expert": expert_name,
            "shared_backbone": shared.name if shared else None,
            "created_at": _utc_now(),
        },
    )
    os.replace(temporary, final)
    checkpoints = sorted(output_dir.glob("checkpoint-*"))
    for stale in checkpoints[:-keep]:
        shutil.rmtree(stale)
    return final


def _load_resume(
    checkpoint: Path,
    *,
    controller,
    transformer,
    optimizer,
    scheduler,
    domain: str,
    fingerprint: str,
) -> tuple[int, int, int]:
    contract = json.loads((checkpoint / "checkpoint.json").read_text())
    if contract.get("schema") != RUN_SCHEMA or contract.get("domain") != domain:
        raise ValueError("resume checkpoint has the wrong schema or domain")
    if contract.get("fingerprint") != fingerprint:
        raise ValueError("resume checkpoint training contract changed")
    load_terminal_expert(
        controller,
        domain,
        checkpoint / str(contract["expert"]),
    )
    shared = contract.get("shared_backbone")
    if shared:
        load_terminal_shared_backbone(transformer, checkpoint / str(shared))
    state = torch.load(
        checkpoint / "trainer_state.pt",
        map_location="cpu",
        weights_only=True,
    )
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    random.setstate(state["python_rng"])
    torch.set_rng_state(state["torch_rng"])
    torch.cuda.set_rng_state_all(state["cuda_rng"])
    return int(state["step"]), int(state["epoch"]), int(state["batch_index"])


def _evaluate(
    *,
    transformer,
    controller,
    model,
    rows: list[dict[str, Any]],
    config: TerminalExpertTrainConfig,
    device: torch.device,
) -> dict[str, float | int]:
    was_training = transformer.training
    transformer.eval()
    selected = rows[: config.eval_examples] if config.eval_examples else rows
    total_loss = 0.0
    total_examples = 0
    cuda_devices = [
        device.index if device.index is not None else torch.cuda.current_device()
    ]
    with torch.no_grad(), torch.random.fork_rng(devices=cuda_devices):
        torch.manual_seed(config.seed + 77_777)
        torch.cuda.manual_seed_all(config.seed + 77_777)
        for start in range(0, len(selected), config.eval_microbatch_size):
            batch_rows = selected[start : start + config.eval_microbatch_size]
            encoder_cache = getattr(model, "_training_encoder_cache", None)
            images = [
                (
                    None
                    if encoder_cache is not None
                    and encoder_cache.has_moments(row)
                    else _load_image_tensor(row)
                )
                for row in batch_rows
            ]
            flow = encode_domain_batch(
                model,
                batch_rows,
                images,
                config,
                device,
                caption_dropout=0.0,
            )
            with (
                controller.route(config.domain),
                torch.autocast(device_type="cuda", dtype=torch.bfloat16),
            ):
                prediction = _forward_transformer(transformer, flow)
                _loss, observed = rectified_flow_loss(
                    prediction,
                    flow["velocity"],
                )
            total_loss += float(observed.item()) * len(batch_rows)
            total_examples += len(batch_rows)
    transformer.train(was_training)
    return {
        "loss": total_loss / total_examples,
        "eval/primary_loss": total_loss / total_examples,
        f"eval/{config.domain}_examples": total_examples,
        f"eval/{config.domain}_via_{config.domain}_loss": total_loss / total_examples,
        "eval/routes_per_example": 1,
    }


def _run_evaluation(
    *,
    pipeline,
    transformer,
    controller,
    model,
    rows: list[dict[str, Any]],
    config: TerminalExpertTrainConfig,
    device: torch.device,
    output_dir: Path,
    step: int,
) -> dict[str, float | int | str]:
    """Run scalar metrics and original/generated side-by-side gallery together."""
    _atomic_json(
        output_dir / "status.json",
        {
            "schema": RUN_SCHEMA,
            "state": "evaluating",
            "step": step,
            "domain": config.domain,
            "updated_at": _utc_now(),
        },
    )
    metrics: dict[str, float | int | str] = _evaluate(
        transformer=transformer,
        controller=controller,
        model=model,
        rows=rows,
        config=config,
        device=device,
    )
    artifact = generate_eval_gallery(
        pipeline,
        transformer,
        controller,
        rows,
        config,
        device,
        output_dir,
        step=step,
        domains=(config.domain,),
    )
    metrics["eval/gallery_artifact"] = str(artifact)
    metrics["eval/gallery_samples"] = EVAL_GALLERY_SAMPLES_PER_DOMAIN
    with (output_dir / "train.jsonl").open("a") as handle:
        handle.write(json.dumps({"kind": "eval", "step": step, **metrics}) + "\n")
    _atomic_json(
        output_dir / "status.json",
        {
            "schema": RUN_SCHEMA,
            "state": "training",
            "step": step,
            "domain": config.domain,
            "updated_at": _utc_now(),
        },
    )
    return metrics


def train(config: TerminalExpertTrainConfig) -> None:
    config.validate()
    try:
        from huggingface_hub import snapshot_download
        from mage_flow import MageFlowPipeline
    except ImportError as error:
        raise RuntimeError("use the isolated Mage-Flow environment") from error
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("terminal expert training requires BF16 CUDA")

    device = torch.device("cuda", torch.cuda.current_device())
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    output_dir = Path(config.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_json(
        output_dir / "status.json",
        {"schema": RUN_SCHEMA, "state": "initializing", "step": 0},
    )

    model_dir = snapshot_download(
        repo_id=config.model_id,
        revision=config.model_revision,
        local_files_only=True,
    )
    pipeline = MageFlowPipeline.from_pretrained(
        model_dir,
        device=str(device),
        attn_type=config.attention_backend,
    )
    model = pipeline.model
    model.vae.sample_posterior = config.vae_sample_posterior
    model.vae.eval().requires_grad_(False)
    model.txt_enc.eval().requires_grad_(False)
    transformer = model.transformer
    controller = install_terminal_expert(
        transformer,
        config.domain,
        dtype=torch.float32,
        device=device,
    )
    load_terminal_expert(
        controller,
        config.domain,
        Path(config.expert_checkpoint),
    )
    if config.shared_backbone_checkpoint:
        load_terminal_shared_backbone(
            transformer,
            Path(config.shared_backbone_checkpoint),
        )
    scope = configure_terminal_training_scope(
        transformer,
        controller,
        train_backbone_final_fraction=config.train_backbone_final_fraction,
    )
    float8_report = convert_trainable_image_ffns_to_float8(
        transformer,
        enabled=config.float8_training,
        recipe=config.float8_recipe,
    )
    if config.float8_training:
        scope = configure_terminal_training_scope(
            transformer,
            controller,
            train_backbone_final_fraction=config.train_backbone_final_fraction,
        )
    checkpoint_mode = (
        config.activation_checkpointing_mode
        if config.gradient_checkpointing
        else "none"
    )
    activation_checkpoint_report = configure_activation_checkpointing(
        transformer,
        checkpoint_mode,
    )
    groups = terminal_optimizer_parameter_groups(
        transformer,
        controller,
        expert_learning_rate=config.learning_rate,
        backbone_learning_rate_multiplier=config.backbone_learning_rate_multiplier,
    )
    optimizer = torch.optim.AdamW(
        groups,
        betas=(config.adam_beta1, config.adam_beta2),
        eps=config.adam_epsilon,
        weight_decay=config.weight_decay,
        foreach=True,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: _cosine_multiplier(
            step,
            warmup_steps=config.warmup_steps,
            max_steps=config.max_steps,
            min_ratio=config.min_learning_rate_ratio,
        ),
    )
    train_rows = _domain_rows(config.train_manifest, config.domain)
    eval_rows = (
        _domain_rows(config.eval_manifest, config.domain)
        if config.eval_manifest
        else []
    )
    annotate_domain_token_lengths(model, train_rows)
    if eval_rows:
        annotate_domain_token_lengths(model, eval_rows)
    encoder_cache = None
    encoder_cache_report: dict[str, Any] = {
        "mode": "off",
        "complete": False,
    }
    if config.encoder_cache_mode != "off":
        from mage_flow.models.utils import PROMPT_TEMPLATE

        encoder_cache = FrozenEncoderCache(
            config.encoder_cache_dir,
            mode=config.encoder_cache_mode,
            model_id=config.model_id,
            model_revision=config.model_revision,
        )
        model._training_encoder_cache = encoder_cache
        cache_rows = [*train_rows, *eval_rows]
        cache_prompts, _cache_kinds = effective_conditioning_prompts(
            cache_rows,
            caption_dropout=0.0,
            rng=random.Random(config.seed),
        )
        info = PROMPT_TEMPLATE["mage-flow"]
        encoder_cache_report = {
            "mode": config.encoder_cache_mode,
            "root": str(encoder_cache.root),
            **cache_coverage(
                encoder_cache,
                cache_rows,
                prompts=cache_prompts,
                template=info["template"],
                drop_idx=int(info["start_idx"]),
                require_null_prompt=config.caption_dropout > 0,
            ),
        }
        if (
            config.encoder_cache_mode == "read_only"
            and not encoder_cache_report["complete"]
        ):
            raise RuntimeError(
                f"read-only encoder cache is incomplete: {encoder_cache_report}"
            )
        if config.offload_cached_encoders:
            model.txt_enc.to("cpu")
            if not hasattr(model.vae, "dconv_encoder"):
                raise RuntimeError(
                    "offload_cached_encoders requires MageVAE.dconv_encoder"
                )
            model.vae.dconv_encoder.to("cpu")
            model._training_text_encoder_offloaded = True
            model._training_vae_encoder_offloaded = True
    if not eval_rows:
        _drop_vae_decoder(model)
    fingerprint = _contract_fingerprint(config)

    step = epoch = batch_start = 0
    if config.resume_from:
        step, epoch, batch_start = _load_resume(
            Path(config.resume_from),
            controller=controller,
            transformer=transformer,
            optimizer=optimizer,
            scheduler=scheduler,
            domain=config.domain,
            fingerprint=fingerprint,
        )
    compile_report = compile_transformer_blocks(
        transformer,
        enabled=config.compile_transformer_blocks,
        mode=config.compile_transformer_mode,
        dynamic=config.compile_transformer_dynamic,
    )
    architecture = terminal_architecture_report(controller)
    if not architecture["passed"]:
        raise RuntimeError(f"architecture preflight failed: {architecture}")
    _atomic_json(
        output_dir / "run_contract.json",
        {
            "schema": RUN_SCHEMA,
            "created_at": _utc_now(),
            "config": asdict(config),
            "architecture": architecture,
            "training_scope": scope,
            "train_examples": len(train_rows),
            "eval_examples": len(eval_rows),
            "resident_experts": 1,
            "inactive_expert_on_gpu": False,
            "fingerprint": fingerprint,
            "runtime_optimizations": {
                "attention_backend": config.attention_backend,
                "activation_checkpointing": activation_checkpoint_report,
                "regional_compile": compile_report,
                "float8": float8_report,
                "frozen_encoder_cache": encoder_cache_report,
                "cached_encoder_offload": config.offload_cached_encoders,
            },
        },
    )

    stop = {"requested": False}

    def handle_stop(_signum, _frame):
        stop["requested"] = True

    signal.signal(signal.SIGINT, handle_stop)
    signal.signal(signal.SIGTERM, handle_stop)
    transformer.train()
    trainable = [
        parameter for parameter in transformer.parameters() if parameter.requires_grad
    ]
    optimizer.zero_grad(set_to_none=True)
    accumulation = 0
    loss_sum = 0.0
    _atomic_json(
        output_dir / "status.json",
        {"schema": RUN_SCHEMA, "state": "training", "step": step},
    )
    if eval_rows:
        _run_evaluation(
            pipeline=pipeline,
            transformer=transformer,
            controller=controller,
            model=model,
            rows=eval_rows,
            config=config,
            device=device,
            output_dir=output_dir,
            step=step,
        )

    while step < config.max_steps:
        batches = _epoch_batches(
            len(train_rows),
            batch_size=config.microbatch_size,
            seed=config.seed,
            epoch=epoch,
        )
        for batch_index, indices in enumerate(batches[batch_start:], start=batch_start):
            batch_rows = [train_rows[index] for index in indices]
            images = [
                (
                    None
                    if encoder_cache is not None
                    and encoder_cache.has_moments(row)
                    else _load_image_tensor(row)
                )
                for row in batch_rows
            ]
            flow = encode_domain_batch(model, batch_rows, images, config, device)
            with (
                controller.route(config.domain),
                torch.autocast(device_type="cuda", dtype=torch.bfloat16),
            ):
                prediction = _forward_transformer(transformer, flow)
                loss, observed = rectified_flow_loss(prediction, flow["velocity"])
                scaled = loss / config.gradient_accumulation_steps
            scaled.backward()
            accumulation += 1
            loss_sum += float(observed.item())
            if accumulation < config.gradient_accumulation_steps:
                continue

            grad_norm = torch.nn.utils.clip_grad_norm_(
                trainable,
                config.max_grad_norm,
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            metrics = {
                "kind": "train",
                "step": step,
                "loss": loss_sum / accumulation,
                "gnorm": float(grad_norm),
                "expert_lr": float(optimizer.param_groups[0]["lr"]),
                "shared_backbone_lr": (
                    float(optimizer.param_groups[1]["lr"])
                    if len(optimizer.param_groups) > 1
                    else None
                ),
                "domain": config.domain,
                "samples": accumulation * config.microbatch_size,
                "epoch": epoch,
            }
            accumulation = 0
            loss_sum = 0.0
            with (output_dir / "train.jsonl").open("a") as handle:
                handle.write(json.dumps(metrics) + "\n")
            _atomic_json(
                output_dir / "status.json",
                {"schema": RUN_SCHEMA, "state": "training", "step": step},
            )

            if eval_rows and step % config.eval_every == 0:
                _run_evaluation(
                    pipeline=pipeline,
                    transformer=transformer,
                    controller=controller,
                    model=model,
                    rows=eval_rows,
                    config=config,
                    device=device,
                    output_dir=output_dir,
                    step=step,
                )
            if step % config.checkpoint_every == 0 or stop["requested"]:
                checkpoint = _save_checkpoint(
                    controller=controller,
                    transformer=transformer,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    output_dir=output_dir,
                    domain=config.domain,
                    step=step,
                    epoch=epoch,
                    batch_index=batch_index + 1,
                    fingerprint=fingerprint,
                    keep=config.keep_last_n,
                )
                if stop["requested"]:
                    _atomic_json(
                        output_dir / "status.json",
                        {
                            "schema": RUN_SCHEMA,
                            "state": "interrupted",
                            "step": step,
                            "checkpoint": str(checkpoint),
                        },
                    )
                    return
            if step >= config.max_steps:
                break
        epoch += 1
        batch_start = 0

    checkpoint = _save_checkpoint(
        controller=controller,
        transformer=transformer,
        optimizer=optimizer,
        scheduler=scheduler,
        output_dir=output_dir,
        domain=config.domain,
        step=step,
        epoch=epoch,
        batch_index=0,
        fingerprint=fingerprint,
        keep=config.keep_last_n,
    )
    save_terminal_expert(
        controller,
        output_dir / f"mageflow-{config.domain}-terminal-expert.safetensors",
        dtype=torch.bfloat16,
    )
    save_terminal_shared_backbone(
        transformer,
        controller,
        output_dir / "mageflow-shared-final-third.safetensors",
        dtype=torch.bfloat16,
    )
    _atomic_json(
        output_dir / "status.json",
        {
            "schema": RUN_SCHEMA,
            "state": "complete",
            "step": step,
            "checkpoint": str(checkpoint),
        },
    )


def prepare_run(config: TerminalExpertTrainConfig, run_dir: Path) -> dict[str, Any]:
    config.validate()
    run_dir = run_dir.expanduser().resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    config_path = run_dir / "train_config.json"
    _atomic_json(config_path, asdict(config))
    repo = Path(__file__).resolve().parents[2]
    launcher = run_dir / "launch.sh"
    default_venv = (
        ".venv-mage-flow-fa4"
        if config.attention_backend == "flash4" or config.float8_training
        else ".venv-mage-flow"
    )
    launcher.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        'RUN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"\n'
        f"REPO_ROOT={shlex.quote(str(repo))}\n"
        'export HF_HOME="${MAGE_FLOW_HF_HOME:-$REPO_ROOT/.hf_cache}"\n'
        f'VENV="${{MAGE_FLOW_VENV:-$REPO_ROOT/{default_venv}}}"\n'
        'exec "$VENV/bin/python" -m rwkv_lab.mage_flow_terminal_train '
        'train --config "$RUN_DIR/train_config.json"\n',
        encoding="utf-8",
    )
    launcher.chmod(0o755)
    receipt = {
        "schema": RUN_SCHEMA,
        "config": str(config_path),
        "launcher": str(launcher),
        "domain": config.domain,
        "train_examples": len(_domain_rows(config.train_manifest, config.domain)),
        "eval_examples": (
            len(_domain_rows(config.eval_manifest, config.domain))
            if config.eval_manifest
            else 0
        ),
        "resident_experts": 1,
    }
    _atomic_json(run_dir / "plan.json", receipt)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--config", type=Path, required=True)
    prepare.add_argument("--run-dir", type=Path, required=True)
    run = commands.add_parser("train")
    run.add_argument("--config", type=Path, required=True)
    cache = commands.add_parser("cache-encoders")
    cache.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = TerminalExpertTrainConfig.from_path(args.config)
    if args.command == "prepare":
        print(json.dumps(prepare_run(config, args.run_dir), indent=2))
    elif args.command == "cache-encoders":
        print(json.dumps(cache_frozen_encoders(config), indent=2, sort_keys=True))
    else:
        train(config)


if __name__ == "__main__":
    main()
