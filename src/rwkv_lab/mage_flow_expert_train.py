"""Routed appearance-expert trainer for Microsoft Mage-Flow-Base.

The trainer supports both the original expert-only stage and staged shared
backbone adaptation around fixed, already-trained appearance experts.  The
Mage-VAE and Qwen3-VL encoder always remain frozen.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import shlex
import shutil
import signal
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from rwkv_lab.mage_flow_adaptation import (
    CFG_NULL_CONDITION,
    EXPERT_DOMAINS,
    MAGE_FLOW_BASE_ID,
    MAGE_FLOW_BASE_REVISION,
    MAGE_SOURCE_REVISION,
    UNCAPTIONED_IMAGE_CONDITION,
    AppearanceExpertController,
    assert_homogeneous_batch,
    audit_domain_rows,
    homogeneous_domain_batches,
    inject_appearance_experts,
    load_appearance_expert,
    load_domain_manifest,
    save_appearance_expert,
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
    _lens_to_cu,
    _load_image_tensor,
    _prefetched,
    _sample_timesteps,
    rectified_flow_loss,
    rectified_flow_path,
)
from rwkv_lab.training_components import (
    AdamWConfiguration,
    AppearanceExpertRoutingConfiguration,
    LinearWarmupCosineConfiguration,
    OptimizerImplementation,
    ParameterRouterImplementation,
    ScheduleImplementation,
    build_registered_optimizer,
    build_registered_parameter_routing,
    build_registered_schedule,
)
from rwkv_lab.training_parameter_routing import ParameterRoutingResult

if TYPE_CHECKING:
    from rwkv_lab.trainvm_adapters import WorkerTrainingComponents
    from rwkv_lab.trainvm_worker import WorkerObservability, WorkerStepProfiler

RUN_SCHEMA = "rwkv-lab.mage-flow-expert-train.v1"
OFFICIAL_REPOSITORY = "https://github.com/microsoft/Mage"
EVAL_GALLERY_SAMPLES_PER_DOMAIN = 4
EVAL_GALLERY_STEPS = 30
EVAL_GALLERY_CFG = 5.0


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


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.expanduser().resolve().open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass
class MageFlowExpertTrainConfig:
    train_manifest: str
    output_dir: str
    eval_manifest: str | None = None
    model_id: str = MAGE_FLOW_BASE_ID
    model_revision: str = MAGE_FLOW_BASE_REVISION
    official_source_revision: str = MAGE_SOURCE_REVISION
    max_steps: int = 16_000
    microbatch_size: int = 2
    gradient_accumulation_steps: int = 4
    photo_weight: float = 0.5
    animation_weight: float = 0.5
    learning_rate: float = 1.0e-4
    shared_learning_rate_multiplier: float = 1.0
    learning_rate_schedule_steps: int | None = None
    min_learning_rate_ratio: float = 0.1
    warmup_steps: int = 500
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_epsilon: float = 1.0e-8
    max_grad_norm: float = 1.0
    caption_dropout: float = 0.1
    final_block_fraction: float = 2 / 3
    expert_parameter_fraction: float = 0.15
    expert_width_fraction: float | None = None
    shared_ffn_multiplier: float = 4.0
    expert_hidden_alignment: int = 128
    timestep_sampling: str = "uniform"
    timestep_shift: float = 1.0
    checkpoint_every: int = 500
    eval_every: int = 500
    # None means every row in every available evaluation domain is consumed
    # during the one unified evaluation phase.
    eval_batches_per_domain: int | None = None
    eval_microbatch_size: int | None = None
    keep_last_n: int = 3
    prefetch_batches: int = 3
    seed: int = 42
    resume_from: str | None = None
    continuation_from: str | None = None
    continuation_reset_scheduler: bool = False
    continuation_reset_optimizer: bool = False
    continuation_preserve_data_position: bool = False
    train_experts: bool = True
    train_shared_final_fraction: float = 0.0
    mandatory_expert_routing: bool = False
    auto_resume_latest: bool = False
    mixed_precision: str = "bf16"
    gradient_checkpointing: bool = True
    activation_checkpointing_mode: str = "full"
    vae_sample_posterior: bool = True
    compile_vae_encoder: bool = True
    attention_backend: str = "flash2"
    compile_transformer_blocks: bool = False
    compile_transformer_mode: str = "default"
    compile_transformer_dynamic: bool = False
    float8_training: bool = False
    float8_recipe: str = "tensorwise"
    encoder_cache_dir: str | None = None
    encoder_cache_mode: str = "off"
    offload_cached_encoders: bool = False
    max_uncaptioned_fraction: float = 0.15

    @classmethod
    def from_path(cls, path: Path) -> MageFlowExpertTrainConfig:
        return cls(**json.loads(path.read_text(encoding="utf-8")))

    @property
    def domain_weights(self) -> dict[str, float]:
        return {
            "general": 0.0,
            "photo": self.photo_weight,
            "animation": self.animation_weight,
        }

    def validate(self, *, inspect_manifests: bool = True) -> None:
        train_path = Path(self.train_manifest).expanduser()
        if not train_path.is_file():
            raise ValueError(f"train manifest not found: {train_path}")
        if self.eval_manifest and not Path(self.eval_manifest).expanduser().is_file():
            raise ValueError(f"eval manifest not found: {self.eval_manifest}")
        if self.model_id != MAGE_FLOW_BASE_ID:
            raise ValueError(
                f"expert training is pinned to {MAGE_FLOW_BASE_ID}, got {self.model_id}"
            )
        if self.model_revision != MAGE_FLOW_BASE_REVISION:
            raise ValueError("unqualified Mage-Flow-Base revision")
        if self.official_source_revision != MAGE_SOURCE_REVISION:
            raise ValueError("unqualified Microsoft Mage source revision")
        if self.max_steps < 1 or self.microbatch_size < 1:
            raise ValueError("max_steps and microbatch_size must be positive")
        if self.gradient_accumulation_steps < 1:
            raise ValueError("gradient_accumulation_steps must be positive")
        if self.photo_weight < 0 or self.animation_weight < 0:
            raise ValueError("domain weights cannot be negative")
        if self.photo_weight + self.animation_weight <= 0:
            raise ValueError("at least one expert-domain weight must be positive")
        if self.resume_from and self.continuation_from:
            raise ValueError("resume_from and continuation_from are mutually exclusive")
        if self.continuation_reset_scheduler and not self.continuation_from:
            raise ValueError("continuation_reset_scheduler requires continuation_from")
        if self.continuation_reset_optimizer and not self.continuation_from:
            raise ValueError("continuation_reset_optimizer requires continuation_from")
        if self.continuation_preserve_data_position and not self.continuation_from:
            raise ValueError(
                "continuation_preserve_data_position requires continuation_from"
            )
        if self.continuation_reset_optimizer and not self.continuation_reset_scheduler:
            raise ValueError(
                "resetting the optimizer also requires resetting the scheduler"
            )
        if not self.train_experts and self.train_shared_final_fraction <= 0:
            raise ValueError("at least one expert or shared parameter must train")
        if not 0 <= self.train_shared_final_fraction <= 1:
            raise ValueError("train_shared_final_fraction must be in [0, 1]")
        if self.learning_rate <= 0 or not 0 <= self.min_learning_rate_ratio <= 1:
            raise ValueError("invalid learning-rate configuration")
        if self.shared_learning_rate_multiplier <= 0:
            raise ValueError("shared_learning_rate_multiplier must be positive")
        if (
            self.learning_rate_schedule_steps is not None
            and self.learning_rate_schedule_steps < 1
        ):
            raise ValueError("learning_rate_schedule_steps must be positive")
        if not 0 <= self.caption_dropout < 1:
            raise ValueError("caption_dropout must be in [0, 1)")
        if not 0 < self.final_block_fraction <= 1:
            raise ValueError("final_block_fraction must be in (0, 1]")
        if not 0 < self.expert_parameter_fraction <= 1:
            raise ValueError("expert_parameter_fraction must be in (0, 1]")
        if self.expert_width_fraction is not None:
            raise ValueError(
                "final training targets expert_parameter_fraction; "
                "expert_width_fraction must be null"
            )
        if self.shared_ffn_multiplier <= 0:
            raise ValueError("shared_ffn_multiplier must be positive")
        if self.expert_hidden_alignment < 1:
            raise ValueError("expert_hidden_alignment must be positive")
        if self.timestep_sampling not in {
            "uniform",
            "shifted_uniform",
            "logit_normal",
        }:
            raise ValueError("unsupported timestep_sampling")
        if self.timestep_shift <= 0:
            raise ValueError("timestep_shift must be positive")
        if (
            self.checkpoint_every < 1
            or self.eval_every < 1
            or (self.eval_microbatch_size is not None and self.eval_microbatch_size < 1)
            or (
                self.eval_batches_per_domain is not None
                and self.eval_batches_per_domain < 1
            )
        ):
            raise ValueError("checkpoint/evaluation intervals must be positive")
        if self.keep_last_n < 1 or self.prefetch_batches < 0:
            raise ValueError("invalid checkpoint retention or prefetch depth")
        if self.mixed_precision != "bf16":
            raise ValueError("Mage-Flow expert training is qualified only for bf16")
        if self.activation_checkpointing_mode not in ACTIVATION_CHECKPOINT_MODES:
            raise ValueError("unsupported activation_checkpointing_mode")
        if self.attention_backend not in {"flash2", "flash4"}:
            raise ValueError("attention_backend must be flash2 or flash4")
        if self.float8_recipe not in FLOAT8_RECIPES:
            raise ValueError("unsupported float8_recipe")
        if self.float8_training and not self.compile_transformer_blocks:
            raise ValueError(
                "float8_training requires compile_transformer_blocks for a "
                "qualified performant kernel path"
            )
        if self.encoder_cache_mode not in ENCODER_CACHE_MODES:
            raise ValueError("unsupported encoder_cache_mode")
        if self.encoder_cache_mode != "off" and not self.encoder_cache_dir:
            raise ValueError("encoder_cache_mode requires encoder_cache_dir")
        if self.offload_cached_encoders and self.encoder_cache_mode != "read_only":
            raise ValueError(
                "offload_cached_encoders requires a complete read_only cache"
            )
        if not 0 <= self.max_uncaptioned_fraction <= 1:
            raise ValueError("max_uncaptioned_fraction must be in [0, 1]")
        if inspect_manifests:
            rows = load_domain_manifest(train_path)
            preparation_report = train_path.with_suffix(
                train_path.suffix + ".report.json"
            )
            if preparation_report.is_file():
                prepared = json.loads(preparation_report.read_text(encoding="utf-8"))
                if not prepared.get("audit", {}).get("passed", False):
                    raise ValueError(
                        "training manifest preparation report failed its input audit"
                    )
            audit = audit_domain_rows(
                rows, max_uncaptioned_fraction=self.max_uncaptioned_fraction
            )
            if not audit["passed"]:
                raise ValueError(f"training manifest failed audit: {audit}")
            expert_audit = audit_domain_rows(
                [row for row in rows if row["domain"] in EXPERT_DOMAINS],
                max_uncaptioned_fraction=self.max_uncaptioned_fraction,
            )
            if not expert_audit["passed"]:
                raise ValueError(
                    f"expert-domain training rows failed audit: {expert_audit}"
                )
            counts = audit["domain_counts"]
            for domain, weight in (
                ("photo", self.photo_weight),
                ("animation", self.animation_weight),
            ):
                if weight > 0 and not counts.get(domain):
                    raise ValueError(f"no training rows for weighted domain {domain}")
            if self.eval_manifest:
                eval_path = Path(self.eval_manifest).expanduser()
                preparation_report = eval_path.with_suffix(
                    eval_path.suffix + ".report.json"
                )
                if preparation_report.is_file():
                    prepared = json.loads(
                        preparation_report.read_text(encoding="utf-8")
                    )
                    if not prepared.get("audit", {}).get("passed", False):
                        raise ValueError(
                            "evaluation manifest preparation report failed its "
                            "input audit"
                        )
                eval_rows = load_domain_manifest(eval_path)
                eval_audit = audit_domain_rows(
                    eval_rows,
                    max_uncaptioned_fraction=self.max_uncaptioned_fraction,
                )
                if not eval_audit["passed"]:
                    raise ValueError(f"evaluation manifest failed audit: {eval_audit}")
                eval_counts = eval_audit["domain_counts"]
                for domain, weight in (
                    ("photo", self.photo_weight),
                    ("animation", self.animation_weight),
                ):
                    if weight > 0 and not eval_counts.get(domain):
                        raise ValueError(
                            f"evaluation manifest has no rows for weighted domain "
                            f"{domain}"
                        )
                train_ids = {str(row["image_id"]) for row in rows}
                eval_ids = {str(row["image_id"]) for row in eval_rows}
                overlap = train_ids & eval_ids
                if overlap:
                    raise ValueError(
                        f"train/evaluation image leakage: {len(overlap)} duplicate "
                        "image IDs"
                    )


def training_contract_fingerprint(
    config: MageFlowExpertTrainConfig,
    *,
    component_composition_digest: str | None = None,
) -> str:
    """Fingerprint every setting and input that can change optimization."""
    values = asdict(config)
    values.pop("resume_from", None)
    values.pop("continuation_from", None)
    values.pop("continuation_reset_scheduler", None)
    values.pop("continuation_reset_optimizer", None)
    values.pop("continuation_preserve_data_position", None)
    values.pop("eval_microbatch_size", None)
    values.pop("auto_resume_latest", None)
    values.pop("output_dir", None)
    payload = {
        "schema": RUN_SCHEMA,
        "config": values,
        "train_manifest_sha256": _file_sha256(Path(config.train_manifest)),
        "eval_manifest_sha256": (
            _file_sha256(Path(config.eval_manifest)) if config.eval_manifest else None
        ),
    }
    if component_composition_digest is not None:
        payload["component_composition_digest"] = component_composition_digest
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def resolved_worker_component_contract(
    config: MageFlowExpertTrainConfig,
    worker_components: WorkerTrainingComponents | None,
) -> tuple[float, Mapping[str, Mapping[str, str]] | None, str | None]:
    """Bind duplicated legacy config fields until the worker adapter removes them."""

    if worker_components is None:
        return config.learning_rate, None, None
    optimizer_configuration = dict(
        worker_components.configuration("optimizer", category="optimizer")
    )
    router_configuration = dict(
        worker_components.configuration(
            "parameter_router", category="parameter_router"
        )
    )
    schedule_configuration = dict(
        worker_components.configuration(
            "learning_rate", category="learning_rate_schedule"
        )
    )
    clipping_configuration = dict(
        worker_components.configuration(
            "gradient_clipping", category="gradient_clipping"
        )
    )
    expected_optimizer = {
        "learning_rate": config.learning_rate,
        "beta1": config.adam_beta1,
        "beta2": config.adam_beta2,
        "epsilon": config.adam_epsilon,
        "weight_decay": config.weight_decay,
        "foreach": True,
        "fused": False,
    }
    expected_router = {
        "shared_backbone_multiplier": config.shared_learning_rate_multiplier
    }
    expected_schedule = {
        "warmup_steps": config.warmup_steps,
        "max_steps": (
            config.learning_rate_schedule_steps
            if config.learning_rate_schedule_steps is not None
            else config.max_steps
        ),
        "minimum_ratio": config.min_learning_rate_ratio,
    }
    expected_clipping = {
        "max_norm": config.max_grad_norm,
        "norm_type": 2.0,
        "error_if_nonfinite": False,
    }
    if optimizer_configuration != expected_optimizer:
        raise ValueError(
            "authority optimizer composition disagrees with MageFlow configuration"
        )
    if router_configuration != expected_router:
        raise ValueError(
            "authority parameter-router composition disagrees with MageFlow configuration"
        )
    if schedule_configuration != expected_schedule:
        raise ValueError(
            "authority LR-schedule composition disagrees with MageFlow configuration"
        )
    if clipping_configuration != expected_clipping:
        raise ValueError(
            "authority gradient-clipping composition disagrees with MageFlow configuration"
        )
    return (
        float(optimizer_configuration["learning_rate"]),
        worker_components.evidence(),
        worker_components.composition.composition_digest,
    )


def effective_conditioning_prompts(
    rows: Sequence[Mapping[str, Any]],
    *,
    caption_dropout: float,
    rng: random.Random | Any = random,
) -> tuple[list[str], list[str]]:
    """Resolve semantic conditions into prompts accepted by official Mage.

    A single space is the released pipeline's true null prompt.  It is used
    only for explicit CFG cases or stochastic dropout of captioned samples.
    Missing captions retain the dedicated ``<uncaptioned-image>`` marker.
    """
    if not 0 <= caption_dropout < 1:
        raise ValueError("caption_dropout must be in [0, 1)")
    prompts, kinds = [], []
    for row in rows:
        kind = str(row.get("conditioning_kind", ""))
        is_captioned = bool(row.get("is_captioned", False))
        text = str(row.get("conditioning_text", ""))
        if kind == "cfg_null" or text == CFG_NULL_CONDITION:
            prompts.append(" ")
            kinds.append("cfg_null")
        elif not is_captioned:
            if kind != "uncaptioned_image" or text != UNCAPTIONED_IMAGE_CONDITION:
                raise ValueError("uncaptioned row does not use the dedicated condition")
            prompts.append(UNCAPTIONED_IMAGE_CONDITION)
            kinds.append("uncaptioned_image")
        elif rng.random() < caption_dropout:
            prompts.append(" ")
            kinds.append("cfg_dropout")
        else:
            if not text.strip():
                raise ValueError("captioned row has empty conditioning text")
            prompts.append(text)
            kinds.append(kind or "caption")
    return prompts, kinds


@dataclass(frozen=True)
class PrefetchedFrozenEncoderBatch:
    """CPU-resident frozen-encoder tensors loaded ahead of GPU execution."""

    row_digests: tuple[str, ...]
    text_by_prompt: dict[str, torch.Tensor]
    moments: tuple[tuple[torch.Tensor, torch.Tensor] | None, ...]
    load_seconds: float
    pinned: bool


def prefetch_frozen_encoder_batch(
    model,
    rows: Sequence[Mapping[str, Any]],
    config: MageFlowExpertTrainConfig,
) -> PrefetchedFrozenEncoderBatch | None:
    """Read one cached Qwen/VAE batch without touching CUDA or global RNG state."""
    from mage_flow.models.utils import PROMPT_TEMPLATE

    encoder_cache: FrozenEncoderCache | None = getattr(
        model, "_training_encoder_cache", None
    )
    if encoder_cache is None:
        return None
    started_at = time.perf_counter()
    info = PROMPT_TEMPLATE["mage-flow"]
    template, drop_idx = info["template"], int(info["start_idx"])
    # Resolve the non-dropped alternatives with a private RNG. The actual
    # dropout choice remains on the training thread so async I/O cannot perturb
    # the checkpointed Python RNG stream.
    prompts, _kinds = effective_conditioning_prompts(
        rows,
        caption_dropout=0.0,
        rng=random.Random(0),
    )
    candidates = set(prompts)
    if config.caption_dropout > 0:
        candidates.add(" ")
    text_by_prompt = {
        prompt: encoder_cache.load_text(prompt, template, drop_idx)
        for prompt in candidates
    }
    missing_prompts = [
        prompt for prompt, value in text_by_prompt.items() if value is None
    ]
    if missing_prompts and encoder_cache.mode == "read_only":
        raise RuntimeError(
            f"read-only encoder cache is missing {len(missing_prompts)} "
            "prefetched Qwen conditioning entries"
        )
    moments = tuple(encoder_cache.load_moments(row) for row in rows)
    if encoder_cache.mode == "read_only" and any(value is None for value in moments):
        raise RuntimeError("read-only encoder cache is missing prefetched VAE moments")

    pinned = torch.cuda.is_available()
    if pinned:
        text_by_prompt = {
            prompt: value.pin_memory() if value is not None else value
            for prompt, value in text_by_prompt.items()
        }
        moments = tuple(
            (
                value[0].pin_memory(),
                value[1].pin_memory(),
            )
            if value is not None
            else None
            for value in moments
        )
    return PrefetchedFrozenEncoderBatch(
        row_digests=tuple(encoder_cache.moments_digest(row) for row in rows),
        text_by_prompt={
            prompt: value
            for prompt, value in text_by_prompt.items()
            if value is not None
        },
        moments=moments,
        load_seconds=time.perf_counter() - started_at,
        pinned=pinned,
    )


def annotate_domain_token_lengths(model, rows: Sequence[dict[str, Any]]) -> None:
    """Measure packed Qwen lengths using the non-dropped conditioning."""
    from mage_flow.models.utils import PROMPT_TEMPLATE

    tokenizer = model.txt_enc.tokenizer
    info = PROMPT_TEMPLATE["mage-flow"]
    template, drop_idx = info["template"], int(info["start_idx"])
    max_length = int(model.txt_enc.tokenizer_max_length) + drop_idx
    for start in range(0, len(rows), 512):
        batch = rows[start : start + 512]
        encoded = tokenizer(
            [template.format(str(row["conditioning_text"])) for row in batch],
            max_length=max_length,
            truncation=True,
            padding=False,
        )["input_ids"]
        for row, token_ids in zip(batch, encoded, strict=True):
            text_tokens = max(1, len(token_ids) - drop_idx)
            row["text_tokens"] = text_tokens
            row["packed_tokens"] = int(row["latent_tokens"]) + text_tokens


def epoch_training_batches(
    rows: Sequence[Mapping[str, Any]],
    config: MageFlowExpertTrainConfig,
    *,
    epoch: int,
) -> list[list[int]]:
    """Return deterministic expert-only homogeneous microbatches."""
    expert_rows = [
        index for index, row in enumerate(rows) if row.get("domain") in EXPERT_DOMAINS
    ]
    if not expert_rows:
        raise ValueError("training manifest contains no expert-domain rows")
    subset = [rows[index] for index in expert_rows]
    batch_count = math.ceil(len(subset) / config.microbatch_size)
    local_batches = homogeneous_domain_batches(
        subset,
        batch_size=config.microbatch_size,
        batch_count=batch_count,
        weights=config.domain_weights,
        seed=config.seed,
        epoch=epoch,
    )
    return [[expert_rows[index] for index in batch] for batch in local_batches]


def encode_domain_batch(
    model,
    rows: Sequence[Mapping[str, Any]],
    images: Sequence[torch.Tensor | None],
    config: MageFlowExpertTrainConfig,
    device: torch.device,
    *,
    caption_dropout: float | None = None,
    prefetched_cache: PrefetchedFrozenEncoderBatch | None = None,
) -> dict[str, Any]:
    """Build official packed Qwen/VAE conditioning and flow targets."""
    from mage_flow.models.utils import PROMPT_TEMPLATE
    from mage_flow.pipeline import _encode_texts_packed

    domain = assert_homogeneous_batch(rows, range(len(rows)))
    dropout = config.caption_dropout if caption_dropout is None else caption_dropout
    prompts, conditioning_kinds = effective_conditioning_prompts(
        rows, caption_dropout=dropout
    )
    info = PROMPT_TEMPLATE["mage-flow"]
    template, drop_idx = info["template"], int(info["start_idx"])
    encoder_cache: FrozenEncoderCache | None = getattr(
        model, "_training_encoder_cache", None
    )
    if prefetched_cache is not None:
        if encoder_cache is None:
            raise ValueError("prefetched encoder tensors require an active cache")
        expected = tuple(encoder_cache.moments_digest(row) for row in rows)
        if prefetched_cache.row_digests != expected:
            raise ValueError("prefetched encoder batch does not match its rows")
    if encoder_cache is None:
        txt_flat, _vec, text_lens = _encode_texts_packed(
            model,
            prompts,
            template,
            drop_idx,
            device,
        )
    else:
        cached_text: dict[str, torch.Tensor] = (
            dict(prefetched_cache.text_by_prompt)
            if prefetched_cache is not None
            else {}
        )
        missing_prompts = []
        for prompt in prompts:
            if prompt in cached_text:
                continue
            value = encoder_cache.load_text(prompt, template, drop_idx)
            if value is None:
                missing_prompts.append(prompt)
            else:
                cached_text[prompt] = value
        if missing_prompts:
            if encoder_cache.mode == "read_only":
                raise RuntimeError(
                    f"read-only encoder cache is missing {len(missing_prompts)} "
                    "Qwen conditioning entries"
                )
            missing_flat, _missing_vec, missing_lens = _encode_texts_packed(
                model,
                missing_prompts,
                template,
                drop_idx,
                device,
            )
            offset = 0
            for prompt, length in zip(
                missing_prompts, missing_lens, strict=True
            ):
                value = missing_flat[offset : offset + length]
                encoder_cache.save_text(prompt, template, drop_idx, value)
                cached_text[prompt] = value.detach().cpu()
                offset += length
        ordered_text = [cached_text[prompt] for prompt in prompts]
        text_lens = [int(value.shape[0]) for value in ordered_text]
        txt_flat = torch.cat(ordered_text, dim=0).to(
            device=device,
            dtype=torch.bfloat16,
            non_blocking=bool(prefetched_cache and prefetched_cache.pinned),
        )
    txt = txt_flat.reshape(1, -1, txt_flat.shape[-1]).to(
        device=device, dtype=torch.bfloat16
    )
    txt_cu = _lens_to_cu(text_lens, device)
    repa_enabled = bool(getattr(config, "repa_enabled", False))
    use_repa_posterior_mean = bool(
        getattr(config, "repa_use_posterior_mean", True)
    )
    repa_target: torch.Tensor | None = None

    if encoder_cache is None:
        if any(image is None for image in images):
            raise RuntimeError("uncached VAE batch contains a missing image tensor")
        # Pinned host tensors make these copies asynchronous. A dedicated stream
        # lets PCIe transfer overlap the frozen Qwen forward on the default stream.
        transfer_stream = getattr(model, "_training_transfer_stream", None)
        if transfer_stream is None:
            transfer_stream = torch.cuda.Stream(device=device)
            model._training_transfer_stream = transfer_stream
        with torch.cuda.stream(transfer_stream):
            gpu_images = [
                image.to(device=device, non_blocking=True)
                for image in images
                if image is not None
            ]
        torch.cuda.current_stream(device).wait_stream(transfer_stream)
        if repa_enabled:
            if not hasattr(model.vae, "_encode_moments"):
                raise RuntimeError(
                    "deterministic REPA targets require MageVAE._encode_moments"
                )
            groups: dict[tuple[int, int], list[int]] = {}
            for index, image in enumerate(gpu_images):
                shape = (int(image.shape[-2]), int(image.shape[-1]))
                groups.setdefault(shape, []).append(index)
            moments_by_index: list[
                tuple[torch.Tensor, torch.Tensor] | None
            ] = [None] * len(gpu_images)
            for indices in groups.values():
                batch = torch.stack(
                    [gpu_images[index] for index in indices], dim=0
                )
                batch = batch.to(memory_format=torch.contiguous_format).float()
                batch = batch.to(model.vae.device, dtype=model.vae.dtype)
                mean_batch, logvar_batch = model.vae._encode_moments(batch)
                for local_index, image_index in enumerate(indices):
                    moments_by_index[image_index] = (
                        mean_batch[local_index : local_index + 1],
                        logvar_batch[local_index : local_index + 1],
                    )
            packed_latents = []
            packed_means = []
            image_shapes = []
            for value in moments_by_index:
                if value is None:
                    raise RuntimeError("internal VAE moment encoding failure")
                mean, logvar = value
                latent = FrozenEncoderCache.sample_moments(
                    mean,
                    logvar,
                    sample_posterior=config.vae_sample_posterior,
                )
                _, _, latent_height, latent_width = latent.shape
                image_shapes.append([(1, latent_height, latent_width)])
                packed_latents.append(
                    latent.permute(0, 2, 3, 1).reshape(-1, latent.shape[1])
                )
                packed_means.append(
                    mean.permute(0, 2, 3, 1).reshape(-1, mean.shape[1])
                )
            clean = torch.cat(packed_latents, dim=0).unsqueeze(0)
            posterior_mean = torch.cat(packed_means, dim=0).unsqueeze(0)
            repa_target = posterior_mean if use_repa_posterior_mean else clean
        else:
            clean, image_shapes = model.compute_vae_encodings(
                gpu_images, with_ids=False
            )
        clean = clean.to(device=device, dtype=torch.bfloat16)
        if repa_target is not None:
            repa_target = repa_target.to(device=device, dtype=torch.bfloat16)
    else:
        moments: list[tuple[torch.Tensor, torch.Tensor] | None] = (
            list(prefetched_cache.moments)
            if prefetched_cache is not None
            else [encoder_cache.load_moments(row) for row in rows]
        )
        missing_indices = [
            index for index, value in enumerate(moments) if value is None
        ]
        if missing_indices:
            if encoder_cache.mode == "read_only":
                raise RuntimeError(
                    f"read-only encoder cache is missing {len(missing_indices)} "
                    "VAE posterior-moment entries"
                )
            if not hasattr(model.vae, "_encode_moments"):
                raise RuntimeError(
                    "posterior-moment caching requires MageVAE._encode_moments"
                )
            for index in missing_indices:
                image = images[index]
                if image is None:
                    raise RuntimeError("VAE cache miss has no decoded image tensor")
                batch = image.unsqueeze(0).to(
                    device=device,
                    dtype=model.vae.dtype,
                    non_blocking=True,
                )
                mean, logvar = model.vae._encode_moments(batch)
                encoder_cache.save_moments(rows[index], mean, logvar)
                moments[index] = (mean, logvar)
        packed_latents = []
        packed_means = []
        image_shapes = []
        for value in moments:
            if value is None:
                raise RuntimeError("internal VAE cache population failure")
            mean, logvar = (
                tensor.to(
                    device=device,
                    dtype=torch.bfloat16,
                    non_blocking=bool(prefetched_cache and prefetched_cache.pinned),
                )
                for tensor in value
            )
            latent = encoder_cache.sample_moments(
                mean,
                logvar,
                sample_posterior=config.vae_sample_posterior,
            )
            _, _, latent_height, latent_width = latent.shape
            image_shapes.append([(1, latent_height, latent_width)])
            packed_latents.append(
                latent.permute(0, 2, 3, 1).reshape(
                    -1, latent.shape[1]
                )
            )
            if repa_enabled:
                packed_means.append(
                    mean.permute(0, 2, 3, 1).reshape(
                        -1, mean.shape[1]
                    )
                )
        clean = torch.cat(packed_latents, dim=0).unsqueeze(0)
        if repa_enabled:
            posterior_mean = torch.cat(packed_means, dim=0).unsqueeze(0)
            repa_target = (
                posterior_mean if use_repa_posterior_mean else clean
            )
    image_lens = [int(row["latent_tokens"]) for row in rows]
    if clean.shape[1] != sum(image_lens):
        raise RuntimeError(
            f"VAE token mismatch: got {clean.shape[1]}, expected {sum(image_lens)}"
        )
    noise = torch.randn_like(clean)
    timesteps = _sample_timesteps(config, len(rows), device)
    token_timesteps = torch.repeat_interleave(
        timesteps, torch.tensor(image_lens, device=device)
    ).view(1, -1, 1)
    noised, velocity = rectified_flow_path(clean, noise, token_timesteps)
    return {
        "domain": domain,
        "conditioning_kinds": conditioning_kinds,
        # Training-only objectives may reuse the frozen VAE representation and
        # may permute the sampled Gaussian across an accumulation window. Keep
        # both exact tensors instead of reconstructing either from the path.
        "clean": clean.detach(),
        "repa_target": (
            repa_target.detach() if repa_target is not None else None
        ),
        "noise": noise,
        "img": noised.to(dtype=clean.dtype),
        "txt": txt,
        "timesteps": timesteps,
        "img_shapes": [[shape[0] for shape in image_shapes]],
        "img_cu_seqlens": _lens_to_cu(image_lens, device),
        "txt_cu_seqlens": txt_cu,
        "velocity": velocity,
        "image_lens": image_lens,
        "text_lens": text_lens,
    }


def _forward_transformer(
    transformer,
    flow: Mapping[str, Any],
    *,
    return_hidden_layer: int | None = None,
):
    extra = (
        {}
        if return_hidden_layer is None
        else {"return_hidden_layer": return_hidden_layer}
    )
    return transformer(
        img=flow["img"],
        txt=flow["txt"],
        timesteps=flow["timesteps"],
        img_shapes=flow["img_shapes"],
        img_cu_seqlens=flow["img_cu_seqlens"],
        txt_cu_seqlens=flow["txt_cu_seqlens"],
        **extra,
    )


def _trainable_parameters(
    controller: AppearanceExpertController,
) -> list[torch.nn.Parameter]:
    parameters = list(controller.parameters())
    if not parameters or any(not parameter.requires_grad for parameter in parameters):
        raise RuntimeError("appearance-expert trainable parameter contract is broken")
    return parameters


def configure_training_scope(
    transformer,
    controller: AppearanceExpertController,
    *,
    train_experts: bool,
    shared_final_fraction: float,
) -> dict[str, Any]:
    """Freeze the model, then enable only the requested experts/shared tail."""
    transformer.requires_grad_(False)
    expert_parameters = list(controller.parameters())
    expert_ids = {id(parameter) for parameter in expert_parameters}
    if train_experts:
        for parameter in expert_parameters:
            parameter.requires_grad_(True)

    blocks = transformer.transformer_blocks
    shared_block_indices: tuple[int, ...] = ()
    if shared_final_fraction > 0:
        selected_count = max(1, round(len(blocks) * shared_final_fraction))
        shared_block_indices = tuple(range(len(blocks) - selected_count, len(blocks)))
        for index in shared_block_indices:
            for parameter in blocks[index].parameters():
                if id(parameter) not in expert_ids:
                    parameter.requires_grad_(True)

    trainable = [
        (name, parameter)
        for name, parameter in transformer.named_parameters()
        if parameter.requires_grad
    ]
    shared_trainable = [
        (name, parameter)
        for name, parameter in trainable
        if id(parameter) not in expert_ids
    ]
    expert_trainable = [
        (name, parameter)
        for name, parameter in trainable
        if id(parameter) in expert_ids
    ]
    return {
        "train_experts": train_experts,
        "shared_final_fraction": shared_final_fraction,
        "shared_block_indices": list(shared_block_indices),
        "trainable_parameter_count": sum(
            parameter.numel() for _, parameter in trainable
        ),
        "shared_trainable_parameter_count": sum(
            parameter.numel() for _, parameter in shared_trainable
        ),
        "expert_trainable_parameter_count": sum(
            parameter.numel() for _, parameter in expert_trainable
        ),
        "shared_trainable_parameter_names": [name for name, _ in shared_trainable],
        "expert_trainable_parameter_names": [name for name, _ in expert_trainable],
    }


def _all_trainable_parameters(transformer) -> list[torch.nn.Parameter]:
    parameters = [
        parameter for parameter in transformer.parameters() if parameter.requires_grad
    ]
    if not parameters:
        raise RuntimeError("training scope contains no trainable parameters")
    return parameters


def optimizer_parameter_groups(
    transformer,
    controller: AppearanceExpertController,
    *,
    learning_rate: float,
    shared_learning_rate_multiplier: float,
) -> list[dict[str, Any]]:
    """Separate expert and original-backbone parameters for differential LRs."""
    return list(
        optimizer_parameter_routing(
            transformer,
            controller,
            learning_rate=learning_rate,
            shared_learning_rate_multiplier=shared_learning_rate_multiplier,
        ).groups
    )


def optimizer_parameter_routing(
    transformer,
    controller: AppearanceExpertController,
    *,
    learning_rate: float,
    shared_learning_rate_multiplier: float,
    worker_components: WorkerTrainingComponents | None = None,
) -> ParameterRoutingResult:
    """Prove exclusive appearance-expert/backbone ownership before grouping."""

    expert_ids = frozenset(id(parameter) for parameter in controller.parameters())
    if worker_components is not None:
        return worker_components.parameter_routing(
            transformer.named_parameters(remove_duplicate=False),
            {"expert": expert_ids},
            base_learning_rate=learning_rate,
        )
    return build_registered_parameter_routing(
        ParameterRouterImplementation.MAGEFLOW_APPEARANCE_EXPERT_V1,
        transformer.named_parameters(remove_duplicate=False),
        {"expert": expert_ids},
        base_learning_rate=learning_rate,
        configuration=AppearanceExpertRoutingConfiguration(
            shared_backbone_multiplier=shared_learning_rate_multiplier
        ),
    )


def training_scope_preflight_report(
    transformer,
    controller: AppearanceExpertController,
    scope: Mapping[str, Any],
) -> dict[str, Any]:
    """Verify that the live trainable set exactly matches the staged scope."""
    expected = set(scope["shared_trainable_parameter_names"]) | set(
        scope["expert_trainable_parameter_names"]
    )
    actual = {
        name
        for name, parameter in transformer.named_parameters()
        if parameter.requires_grad
    }
    expert_ids = {id(parameter) for parameter in controller.parameters()}
    frozen_experts = [
        name
        for name, parameter in transformer.named_parameters()
        if id(parameter) in expert_ids and not parameter.requires_grad
    ]
    unexpectedly_trainable = sorted(actual - expected)
    unexpectedly_frozen = sorted(expected - actual)
    return {
        "passed": not unexpectedly_trainable and not unexpectedly_frozen,
        "unexpectedly_trainable_parameters": unexpectedly_trainable,
        "unexpectedly_frozen_parameters": unexpectedly_frozen,
        "frozen_expert_parameter_count": len(frozen_experts),
        "scope": dict(scope),
    }


def _shared_trainable_tensors(transformer) -> dict[str, torch.Tensor]:
    expert_markers = (".img_mlp.experts.", ".img_mlp.scales.")
    return {
        name: parameter.detach().cpu().contiguous()
        for name, parameter in transformer.named_parameters()
        if parameter.requires_grad
        and not any(marker in f".{name}" for marker in expert_markers)
    }


def save_shared_trainable_backbone(
    transformer, path: Path, *, dtype: torch.dtype | None = None
) -> Path | None:
    """Save full replacement tensors for the selectively trainable backbone."""
    from safetensors.torch import save_file

    tensors = _shared_trainable_tensors(transformer)
    if not tensors:
        return None
    if dtype is not None:
        tensors = {name: value.to(dtype=dtype) for name, value in tensors.items()}
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(path))
    return path


def load_shared_trainable_backbone(transformer, path: Path) -> int:
    """Load a selectively trained shared-backbone replacement if present."""
    from safetensors.torch import load_file

    path = path.expanduser().resolve()
    if not path.is_file():
        return 0
    tensors = load_file(str(path), device="cpu")
    parameters = dict(transformer.named_parameters())
    unexpected = sorted(set(tensors) - set(parameters))
    if unexpected:
        raise ValueError(
            f"shared-backbone checkpoint has unexpected keys: {unexpected}"
        )
    with torch.no_grad():
        for name, value in tensors.items():
            target = parameters[name]
            if target.shape != value.shape:
                raise ValueError(
                    f"shared-backbone tensor shape mismatch for {name}: "
                    f"{tuple(value.shape)} != {tuple(target.shape)}"
                )
            target.copy_(value.to(device=target.device, dtype=target.dtype))
    return len(tensors)


def expert_preflight_report(
    transformer,
    controller: AppearanceExpertController,
    *,
    require_zero_output: bool,
) -> dict[str, Any]:
    """Verify frozen-backbone isolation and fresh-checkpoint equivalence."""
    expert_parameters = list(controller.parameters())
    expert_ids = {id(parameter) for parameter in expert_parameters}
    foreign_trainable = [
        name
        for name, parameter in transformer.named_parameters()
        if parameter.requires_grad and id(parameter) not in expert_ids
    ]
    frozen_experts = [
        name
        for name, parameter in transformer.named_parameters()
        if id(parameter) in expert_ids and not parameter.requires_grad
    ]
    nonzero_outputs = []
    if require_zero_output:
        for index, wrapper in controller.wrappers.items():
            for domain in EXPERT_DOMAINS:
                output = wrapper.experts[domain].fc2
                if (
                    torch.count_nonzero(output.weight).item()
                    or torch.count_nonzero(output.bias).item()
                ):
                    nonzero_outputs.append(f"{index}:{domain}")
    routes = {
        index: wrapper.active_domain for index, wrapper in controller.wrappers.items()
    }
    requested_fraction = controller.config.requested_parameter_fraction
    parameter_fraction_error = (
        abs(controller.config.actual_parameter_fraction - requested_fraction)
        if requested_fraction is not None
        else 0.0
    )
    passed = (
        not foreign_trainable
        and not frozen_experts
        and not nonzero_outputs
        and set(routes.values()) == {"general"}
        and parameter_fraction_error <= 0.0025
    )
    return {
        "passed": passed,
        "require_zero_output": require_zero_output,
        "foreign_trainable_parameters": foreign_trainable,
        "frozen_expert_parameters": frozen_experts,
        "nonzero_output_experts": nonzero_outputs,
        "active_routes": routes,
        "requested_parameter_fraction": requested_fraction,
        "actual_parameter_fraction": controller.config.actual_parameter_fraction,
        "parameter_fraction_error": parameter_fraction_error,
        "combined_shared_fraction": controller.config.combined_shared_fraction,
        "trainable_parameters": sum(
            parameter.numel() for parameter in expert_parameters
        ),
    }


def _capture_rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def _restore_rng_state(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state.get("cuda"):
        torch.cuda.set_rng_state_all(state["cuda"])


def _checkpoint_sort_key(path: Path) -> int:
    try:
        return int(path.name.rsplit("-", 1)[-1])
    except ValueError:
        return -1


def _prune_checkpoints(output_dir: Path, keep: int) -> None:
    checkpoints = sorted(
        (path for path in output_dir.glob("checkpoint-*") if path.is_dir()),
        key=_checkpoint_sort_key,
    )
    for path in checkpoints[:-keep]:
        shutil.rmtree(path)


def _link_or_copy(source: Path, destination: Path) -> None:
    """Hardlink an immutable checkpoint tensor, falling back across filesystems."""
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def prepare_fixed_expert_cache(source_checkpoint: Path, output_dir: Path) -> Path:
    """Keep one non-pruned canonical copy of experts frozen for this stage."""
    source_checkpoint = source_checkpoint.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    final = output_dir / "fixed_experts"
    required = [f"mageflow-{domain}-expert.safetensors" for domain in EXPERT_DOMAINS]
    if final.is_dir() and all((final / name).is_file() for name in required):
        return final
    temporary = output_dir / ".fixed_experts.incomplete"
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    for name in required:
        source = source_checkpoint / name
        if not source.is_file():
            raise ValueError(f"fixed expert source is missing {source}")
        _link_or_copy(source, temporary / name)
        sidecar = source.with_suffix(source.suffix + ".json")
        if sidecar.is_file():
            _link_or_copy(sidecar, temporary / sidecar.name)
    os.replace(temporary, final)
    return final


def _copy_fixed_experts_into_checkpoint(source_dir: Path, destination: Path) -> None:
    for domain in EXPERT_DOMAINS:
        name = f"mageflow-{domain}-expert.safetensors"
        source = source_dir / name
        if not source.is_file():
            raise ValueError(f"fixed expert cache is missing {source}")
        _link_or_copy(source, destination / name)
        sidecar = source.with_suffix(source.suffix + ".json")
        if sidecar.is_file():
            _link_or_copy(sidecar, destination / sidecar.name)


def latest_compatible_checkpoint(
    output_dir: Path,
    configured_checkpoint: Path | None,
    *,
    contract_fingerprint: str,
) -> Path:
    """Resolve the newest complete exact-resume checkpoint for crash recovery."""
    candidates = []
    if configured_checkpoint is not None:
        configured_checkpoint = configured_checkpoint.expanduser().resolve()
        if not configured_checkpoint.is_dir():
            raise ValueError(f"resume checkpoint not found: {configured_checkpoint}")
        candidates.append(configured_checkpoint)
    for candidate in output_dir.expanduser().resolve().glob("checkpoint-*"):
        if candidate.is_dir():
            candidates.append(candidate)
    compatible = []
    for candidate in candidates:
        metadata_path = candidate / "checkpoint.json"
        state_path = candidate / "trainer_state.pt"
        if not metadata_path.is_file() or not state_path.is_file():
            continue
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if metadata.get("contract_fingerprint") == contract_fingerprint:
            compatible.append(candidate)
    if not compatible:
        raise ValueError("no complete checkpoint matches the training contract")
    return max(compatible, key=_checkpoint_sort_key)


def save_training_checkpoint(
    controller: AppearanceExpertController,
    optimizer,
    scheduler,
    output_dir: Path,
    *,
    global_step: int,
    epoch: int,
    batch_index: int,
    keep_last_n: int,
    contract_fingerprint: str | None = None,
    transformer=None,
    fixed_expert_source_dir: Path | None = None,
) -> Path:
    """Atomically save routed weights, optimizer/scheduler, RNG, and position."""
    output_dir = output_dir.expanduser().resolve()
    final = output_dir / f"checkpoint-{global_step:08d}"
    if final.is_dir():
        return final
    temporary = output_dir / f".checkpoint-{global_step:08d}.incomplete"
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    if fixed_expert_source_dir is not None:
        _copy_fixed_experts_into_checkpoint(fixed_expert_source_dir, temporary)
    else:
        save_appearance_expert(
            controller, "photo", temporary / "mageflow-photo-expert.safetensors"
        )
        save_appearance_expert(
            controller,
            "animation",
            temporary / "mageflow-animation-expert.safetensors",
        )
    shared_path = None
    if transformer is not None:
        shared_path = save_shared_trainable_backbone(
            transformer,
            temporary / "mageflow-shared-final-third.safetensors",
        )
    state = {
        "schema": RUN_SCHEMA,
        "global_step": global_step,
        "epoch": epoch,
        "batch_index": batch_index,
        "contract_fingerprint": contract_fingerprint,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "rng": _capture_rng_state(),
    }
    torch.save(state, temporary / "trainer_state.pt")
    _atomic_json(
        temporary / "checkpoint.json",
        {
            "schema": RUN_SCHEMA,
            "created_at": _utc_now(),
            "global_step": global_step,
            "epoch": epoch,
            "batch_index": batch_index,
            "contract_fingerprint": contract_fingerprint,
            "base_model": controller.base_model,
            "base_revision": controller.base_revision,
            "shared_backbone": shared_path.name if shared_path else None,
            "fixed_experts_reused": fixed_expert_source_dir is not None,
        },
    )
    os.replace(temporary, final)
    _prune_checkpoints(output_dir, keep_last_n)
    return final


def load_training_checkpoint(
    controller: AppearanceExpertController,
    optimizer,
    scheduler,
    checkpoint: Path,
    *,
    expected_contract_fingerprint: str | None = None,
    restore_optimizer: bool = True,
    restore_scheduler: bool = True,
    transformer=None,
) -> dict[str, int]:
    """Restore an exact local checkpoint created by this trainer."""
    checkpoint = checkpoint.expanduser().resolve()
    contract = json.loads((checkpoint / "checkpoint.json").read_text(encoding="utf-8"))
    if contract.get("schema") != RUN_SCHEMA:
        raise ValueError("unsupported expert-training checkpoint schema")
    if (
        expected_contract_fingerprint is not None
        and contract.get("contract_fingerprint") != expected_contract_fingerprint
    ):
        raise ValueError("resume checkpoint training contract does not match")
    load_appearance_expert(
        controller,
        "photo",
        checkpoint / "mageflow-photo-expert.safetensors",
    )
    load_appearance_expert(
        controller,
        "animation",
        checkpoint / "mageflow-animation-expert.safetensors",
    )
    shared_path = checkpoint / "mageflow-shared-final-third.safetensors"
    if shared_path.is_file():
        if transformer is None:
            raise ValueError(
                "checkpoint contains shared-backbone weights but no transformer "
                "was provided"
            )
        load_shared_trainable_backbone(transformer, shared_path)
    state = torch.load(
        checkpoint / "trainer_state.pt", map_location="cpu", weights_only=True
    )
    if state.get("schema") != RUN_SCHEMA:
        raise ValueError("trainer state has an unsupported schema")
    if state.get("contract_fingerprint") != contract.get("contract_fingerprint"):
        raise ValueError("checkpoint contract and trainer state disagree")
    if restore_optimizer:
        optimizer.load_state_dict(state["optimizer"])
    if restore_scheduler:
        scheduler.load_state_dict(state["scheduler"])
    _restore_rng_state(state["rng"])
    return {
        "global_step": int(state["global_step"]),
        "epoch": int(state["epoch"]),
        "batch_index": int(state["batch_index"]),
    }


def export_final_weights(
    controller: AppearanceExpertController, transformer, output_dir: Path
) -> dict[str, str]:
    output_dir = output_dir.expanduser().resolve()
    photo = output_dir / "mageflow-photo-expert.safetensors"
    animation = output_dir / "mageflow-animation-expert.safetensors"
    save_appearance_expert(controller, "photo", photo, dtype=torch.bfloat16)
    save_appearance_expert(controller, "animation", animation, dtype=torch.bfloat16)
    exports = {"photo": str(photo), "animation": str(animation)}
    shared = save_shared_trainable_backbone(
        transformer,
        output_dir / "mageflow-shared-final-third.safetensors",
        dtype=torch.bfloat16,
    )
    if shared is not None:
        exports["shared_final_third"] = str(shared)
    return exports


def export_final_experts(
    controller: AppearanceExpertController, output_dir: Path
) -> dict[str, str]:
    """Backward-compatible expert-only export helper."""
    photo = output_dir / "mageflow-photo-expert.safetensors"
    animation = output_dir / "mageflow-animation-expert.safetensors"
    save_appearance_expert(controller, "photo", photo, dtype=torch.bfloat16)
    save_appearance_expert(controller, "animation", animation, dtype=torch.bfloat16)
    return {"photo": str(photo), "animation": str(animation)}


def _evaluation_batches(
    rows: Sequence[Mapping[str, Any]],
    *,
    domain: str,
    batch_size: int,
    count: int | None,
) -> list[list[int]]:
    indices = [index for index, row in enumerate(rows) if row["domain"] == domain]
    if not indices:
        return []
    batches = []
    limit = len(indices) if count is None else min(len(indices), batch_size * count)
    for start in range(0, limit, batch_size):
        batches.append(indices[start : start + batch_size])
    return batches


def evaluate_routes(
    transformer,
    controller: AppearanceExpertController,
    model,
    rows: Sequence[dict[str, Any]],
    config: MageFlowExpertTrainConfig,
    device: torch.device,
) -> dict[str, float | int]:
    """Evaluate the complete data-domain by appearance-route loss matrix."""
    was_training = transformer.training
    transformer.eval()
    results = {}
    routes = (
        EXPERT_DOMAINS
        if config.mandatory_expert_routing
        else ("general", *EXPERT_DOMAINS)
    )
    cuda_devices = (
        [device.index if device.index is not None else torch.cuda.current_device()]
        if device.type == "cuda"
        else []
    )
    with torch.no_grad(), torch.random.fork_rng(devices=cuda_devices):
        torch.manual_seed(config.seed + 77_777)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(config.seed + 77_777)
        total_examples = 0
        primary_loss_sum = 0.0
        primary_examples = 0
        for data_domain in ("general", "photo", "animation"):
            values: dict[str, list[float]] = {route: [] for route in routes}
            batches = _evaluation_batches(
                rows,
                domain=data_domain,
                batch_size=(config.eval_microbatch_size or config.microbatch_size),
                count=config.eval_batches_per_domain,
            )
            domain_examples = sum(len(indices) for indices in batches)
            if domain_examples:
                results[f"eval/{data_domain}_examples"] = domain_examples
                total_examples += domain_examples
            for indices in batches:
                batch_rows = [rows[index] for index in indices]
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
                for route in routes:
                    with (
                        controller.route(route),
                        torch.autocast(
                            device_type=device.type,
                            dtype=torch.bfloat16,
                            enabled=device.type == "cuda",
                        ),
                    ):
                        prediction = _forward_transformer(transformer, flow)
                    _loss, observed = rectified_flow_loss(prediction, flow["velocity"])
                    observed_value = float(observed.item())
                    values[route].append(observed_value)
                    if data_domain in EXPERT_DOMAINS and route == data_domain:
                        primary_loss_sum += observed_value * len(indices)
                        primary_examples += len(indices)
            for route, route_values in values.items():
                if route_values:
                    results[f"eval/{data_domain}_via_{route}_loss"] = sum(
                        route_values
                    ) / len(route_values)
        results["eval/examples"] = total_examples
        results["eval/routes_per_example"] = len(routes)
        if primary_examples:
            primary_loss = primary_loss_sum / primary_examples
            results["loss"] = primary_loss
            results["eval/primary_loss"] = primary_loss
    transformer.train(was_training)
    return results


def _eval_gallery_rows(
    rows: Sequence[dict[str, Any]],
    *,
    count_per_domain: int = EVAL_GALLERY_SAMPLES_PER_DOMAIN,
    domains: Sequence[str] = EXPERT_DOMAINS,
) -> list[dict[str, Any]]:
    """Select a deterministic balanced photo/animation qualitative gallery."""
    selected: list[dict[str, Any]] = []
    for domain in domains:
        if domain not in EXPERT_DOMAINS:
            raise ValueError(f"unsupported evaluation gallery domain {domain!r}")
        domain_rows = [row for row in rows if row["domain"] == domain]
        if len(domain_rows) < count_per_domain:
            raise ValueError(
                f"evaluation gallery needs {count_per_domain} {domain} rows, "
                f"found {len(domain_rows)}"
            )
        selected.extend(domain_rows[:count_per_domain])
    return selected


def generate_eval_gallery(
    pipeline,
    transformer,
    controller: AppearanceExpertController,
    rows: Sequence[dict[str, Any]],
    config: MageFlowExpertTrainConfig,
    device: torch.device,
    output_dir: Path,
    *,
    step: int,
    domains: Sequence[str] = EXPERT_DOMAINS,
) -> Path:
    """Generate a balanced expert-routed gallery consumable by trainboard."""
    selected = _eval_gallery_rows(rows, domains=domains)
    selection_payload = [
        {
            "image_id": str(row["image_id"]),
            "domain": str(row["domain"]),
            "caption_sha256": hashlib.sha256(
                str(row["caption"]).encode("utf-8")
            ).hexdigest(),
        }
        for row in selected
    ]
    selection_sha256 = hashlib.sha256(
        json.dumps(
            selection_payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    artifact = output_dir / "eval_samples" / f"step_{step:08d}.json"

    image_dir = output_dir / "eval_generations" / f"step_{step:08d}"
    image_dir.mkdir(parents=True, exist_ok=True)
    text_encoder = pipeline.model.txt_enc
    text_encoder_was_offloaded = bool(
        getattr(pipeline.model, "_training_text_encoder_offloaded", False)
    )
    if text_encoder_was_offloaded:
        text_encoder.to(device)
    had_instance_screen = "screen_text" in vars(text_encoder)
    previous_instance_screen = vars(text_encoder).get("screen_text")
    was_training = bool(transformer.training)
    cuda_devices = (
        [device.index if device.index is not None else torch.cuda.current_device()]
        if device.type == "cuda"
        else []
    )

    class AllowedVerdict:
        violates = False

    generated: list[tuple[dict[str, Any], int, Any]] = []
    try:
        # These are private held-out training-domain prompts. Bypassing the
        # public inference refusal image is required for an honest expert eval.
        text_encoder.screen_text = lambda _prompt: AllowedVerdict()
        transformer.eval()
        with torch.random.fork_rng(devices=cuda_devices):
            for domain_index, domain in enumerate(domains):
                domain_rows = [row for row in selected if row["domain"] == domain]
                prompts = [str(row["caption"]) for row in domain_rows]
                seeds = [
                    config.seed + 200_000 + domain_index * 10_000 + index
                    for index in range(len(domain_rows))
                ]
                heights = [int(row["train_height"]) for row in domain_rows]
                widths = [int(row["train_width"]) for row in domain_rows]
                with (
                    controller.route(domain),
                    torch.autocast(
                        device_type=device.type,
                        dtype=torch.bfloat16,
                        enabled=device.type == "cuda",
                    ),
                ):
                    images = pipeline.generate(
                        prompts,
                        seeds=seeds,
                        steps=EVAL_GALLERY_STEPS,
                        cfg=EVAL_GALLERY_CFG,
                        heights=heights,
                        widths=widths,
                        device=str(device),
                    )
                generated.extend(zip(domain_rows, seeds, images, strict=True))
    finally:
        if had_instance_screen:
            text_encoder.screen_text = previous_instance_screen
        else:
            del text_encoder.screen_text
        if text_encoder_was_offloaded:
            text_encoder.to("cpu")
        transformer.train(was_training)

    items = []
    for index, (row, seed, image) in enumerate(generated):
        domain = str(row["domain"])
        image_path = (image_dir / f"{domain}_{index:02d}.png").resolve()
        temporary = image_path.with_name(image_path.name + ".tmp")
        image.save(temporary, format="PNG")
        os.replace(temporary, image_path)
        items.append(
            {
                "image": str(image_path),
                "target_image": str(row["image"]),
                "prompt": str(row["caption"]),
                "reference": f"held-out {domain} target",
                "caption": (
                    f"{domain} expert · step {step} · seed {seed} · "
                    f"{row['train_width']}×{row['train_height']} · "
                    f"{EVAL_GALLERY_STEPS} steps · CFG {EVAL_GALLERY_CFG:g}"
                ),
                "tokens": 0,
                "stopped_at_eod": True,
                "source": str(row.get("source", "mage_flow_expert_eval")),
                "domain": domain,
                "route": domain,
            }
        )
    _atomic_json(
        artifact,
        {
            "schema": RUN_SCHEMA,
            "eval_kind": "image_generation",
            "step": step,
            "ppl": 0.0,
            "decoding": "mage_flow_rectified_flow",
            "max_new": 0,
            "complete": True,
            "generation_steps": EVAL_GALLERY_STEPS,
            "cfg": EVAL_GALLERY_CFG,
            "samples_per_domain": EVAL_GALLERY_SAMPLES_PER_DOMAIN,
            "domains": list(domains),
            "selection_sha256": selection_sha256,
            "selection": selection_payload,
            "items": items,
        },
    )
    return artifact


def run_unified_evaluation(
    pipeline,
    transformer,
    controller: AppearanceExpertController,
    model,
    rows: Sequence[dict[str, Any]],
    config: MageFlowExpertTrainConfig,
    device: torch.device,
    output_dir: Path,
    *,
    step: int,
) -> dict[str, float | int | str]:
    """Run scalar route evaluation and balanced generation at one global step."""
    _atomic_json(
        output_dir / "status.json",
        {
            "schema": RUN_SCHEMA,
            "state": "evaluating",
            "step": step,
            "eval_examples": len(rows),
            "updated_at": _utc_now(),
        },
    )
    metrics: dict[str, float | int | str] = evaluate_routes(
        transformer, controller, model, rows, config, device
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
    )
    metrics["eval/gallery_samples"] = EVAL_GALLERY_SAMPLES_PER_DOMAIN * len(
        EXPERT_DOMAINS
    )
    metrics["eval/gallery_artifact"] = str(artifact)
    with (output_dir / "train.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"kind": "eval", "step": step, **metrics}) + "\n")
    _atomic_json(
        output_dir / "status.json",
        {
            "schema": RUN_SCHEMA,
            "state": "training",
            "step": step,
            "updated_at": _utc_now(),
        },
    )
    return metrics


def unified_evaluation_is_complete(output_dir: Path, step: int) -> bool:
    """Return whether both scalar and gallery evaluation already exist."""
    artifact = output_dir / "eval_samples" / f"step_{step:08d}.json"
    if not artifact.is_file():
        return False
    try:
        payload = json.loads(artifact.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if payload.get("complete") is not True:
        return False
    log_path = output_dir / "train.jsonl"
    if not log_path.is_file():
        return False
    with log_path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("kind") == "eval" and int(row.get("step", -1)) == step:
                return True
    return False


def _drop_vae_decoder(model) -> None:
    """Remove the unused decoder after the pipeline has loaded its weights."""
    vae = model.vae
    if hasattr(vae, "decoder_model"):
        vae.decoder_model = None
    elif hasattr(vae, "decoder"):
        vae.decoder = None


def cache_frozen_encoders(config: MageFlowExpertTrainConfig) -> dict[str, Any]:
    """Materialize reusable Qwen states and VAE posterior moments."""
    config.validate()
    if config.encoder_cache_mode != "read_write" or not config.encoder_cache_dir:
        raise ValueError(
            "cache-encoders requires encoder_cache_mode='read_write' and "
            "encoder_cache_dir"
        )
    try:
        from huggingface_hub import snapshot_download
        from mage_flow import MageFlowPipeline
        from mage_flow.models.utils import PROMPT_TEMPLATE
        from mage_flow.pipeline import _encode_texts_packed
    except ImportError as error:
        raise RuntimeError(
            "Mage-Flow dependencies are missing; use the isolated Mage environment"
        ) from error
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Mage-Flow encoder caching requires a BF16 CUDA GPU")

    device = torch.device("cuda", torch.cuda.current_device())
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    random.seed(config.seed)
    model_path = getattr(config, "model_path", None)
    model_dir = (
        str(Path(model_path).expanduser().resolve())
        if model_path
        else snapshot_download(
            repo_id=config.model_id,
            revision=config.model_revision,
            local_files_only=True,
        )
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
    cache = FrozenEncoderCache(
        config.encoder_cache_dir,
        mode="read_write",
        model_id=config.model_id,
        model_revision=config.model_revision,
    )
    model._training_encoder_cache = cache
    train_rows = load_domain_manifest(Path(config.train_manifest))
    eval_rows = (
        load_domain_manifest(Path(config.eval_manifest)) if config.eval_manifest else []
    )
    rows = [*train_rows, *eval_rows]
    info = PROMPT_TEMPLATE["mage-flow"]
    template, drop_idx = info["template"], int(info["start_idx"])
    prompts, _kinds = effective_conditioning_prompts(
        rows,
        caption_dropout=0.0,
        rng=random.Random(config.seed),
    )
    status_path = cache.root / "cache_build_status.json"
    started = time.perf_counter()
    encoded_rows = 0
    reused_rows = 0
    with torch.inference_mode():
        if config.caption_dropout > 0 and not cache.has_text(
            " ", template, drop_idx
        ):
            txt_flat, _vec, lens = _encode_texts_packed(
                model, [" "], template, drop_idx, device
            )
            cache.save_text(" ", template, drop_idx, txt_flat[: lens[0]])
        for index, (row, prompt) in enumerate(zip(rows, prompts, strict=True), 1):
            has_text = cache.has_text(prompt, template, drop_idx)
            has_moments = cache.has_moments(row)
            if has_text and has_moments:
                reused_rows += 1
            else:
                image = (
                    None
                    if has_moments
                    else _load_image_tensor(row).pin_memory()
                )
                encode_domain_batch(
                    model,
                    [row],
                    [image],
                    config,
                    device,
                    caption_dropout=0.0,
                )
                encoded_rows += 1
            if index % 100 == 0 or index == len(rows):
                _atomic_json(
                    status_path,
                    {
                        "schema": FrozenEncoderCache.schema,
                        "state": "building",
                        "processed": index,
                        "total": len(rows),
                        "encoded_rows": encoded_rows,
                        "reused_rows": reused_rows,
                        "elapsed_seconds": time.perf_counter() - started,
                        "updated_at": _utc_now(),
                    },
                )
    coverage = cache_coverage(
        cache,
        rows,
        prompts=prompts,
        template=template,
        drop_idx=drop_idx,
        require_null_prompt=config.caption_dropout > 0,
    )
    if not coverage["complete"]:
        raise RuntimeError(f"encoder cache build is incomplete: {coverage}")
    receipt = {
        "schema": FrozenEncoderCache.schema,
        "state": "complete",
        "completed_at": _utc_now(),
        "elapsed_seconds": time.perf_counter() - started,
        "train_rows": len(train_rows),
        "eval_rows": len(eval_rows),
        "encoded_rows": encoded_rows,
        "reused_rows": reused_rows,
        "coverage": coverage,
    }
    _atomic_json(cache.root / "cache_build_receipt.json", receipt)
    _atomic_json(status_path, receipt)
    return receipt


def train(
    config: MageFlowExpertTrainConfig,
    *,
    worker_components: WorkerTrainingComponents | None = None,
    worker_step_profiler: WorkerStepProfiler | None = None,
    worker_observability: WorkerObservability | None = None,
) -> None:
    """Run single-GPU routed expert/shared-backbone optimization."""
    config.validate()
    try:
        from huggingface_hub import snapshot_download
        from mage_flow import MageFlowPipeline
    except ImportError as error:
        raise RuntimeError(
            "Mage-Flow dependencies are missing; use the isolated Mage environment"
        ) from error
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Mage-Flow expert training requires a BF16 CUDA GPU")

    device = torch.device("cuda", torch.cuda.current_device())
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    random.seed(config.seed)
    torch.set_float32_matmul_precision("high")

    output_dir = Path(config.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_json(
        output_dir / "status.json",
        {
            "schema": RUN_SCHEMA,
            "state": "initializing",
            "step": 0,
            "updated_at": _utc_now(),
        },
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
    model.config.compile_vae_encoder = config.compile_vae_encoder
    if config.compile_vae_encoder:
        model.maybe_compile_vae_encoder()
    model.vae.eval().requires_grad_(False)
    model.txt_enc.eval().requires_grad_(False)
    transformer = model.transformer
    transformer.train()

    controller = inject_appearance_experts(
        transformer,
        final_block_fraction=config.final_block_fraction,
        expert_parameter_fraction=config.expert_parameter_fraction,
        expert_width_fraction=config.expert_width_fraction,
        shared_ffn_multiplier=config.shared_ffn_multiplier,
        expert_hidden_alignment=config.expert_hidden_alignment,
        base_model=config.model_id,
        base_revision=config.model_revision,
        # Retain FP32 master expert weights and AdamW moments. BF16 autocast
        # below keeps the actual matmuls on tensor cores.
        expert_dtype=torch.float32,
    )
    scope = configure_training_scope(
        transformer,
        controller,
        train_experts=config.train_experts,
        shared_final_fraction=config.train_shared_final_fraction,
    )
    float8_report = convert_trainable_image_ffns_to_float8(
        transformer,
        enabled=config.float8_training,
        recipe=config.float8_recipe,
    )
    if config.float8_training:
        # TorchAO swaps eligible Linear modules in-place. Reassert the exact
        # trainable boundary after conversion before optimizer construction.
        scope = configure_training_scope(
            transformer,
            controller,
            train_experts=config.train_experts,
            shared_final_fraction=config.train_shared_final_fraction,
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
    trainable = _all_trainable_parameters(transformer)
    preflight = training_scope_preflight_report(transformer, controller, scope)
    if not preflight["passed"]:
        raise RuntimeError(f"training-scope preflight failed: {preflight}")
    optimizer_learning_rate, component_evidence, component_digest = (
        resolved_worker_component_contract(config, worker_components)
    )
    optimizer_routing = optimizer_parameter_routing(
        transformer,
        controller,
        learning_rate=optimizer_learning_rate,
        shared_learning_rate_multiplier=config.shared_learning_rate_multiplier,
        worker_components=worker_components,
    )
    optimizer_groups = list(optimizer_routing.groups)
    if worker_components is not None:
        optimizer = worker_components.optimizer(optimizer_groups)
        scheduler = worker_components.learning_rate_schedule(optimizer)
    else:
        optimizer = build_registered_optimizer(
            OptimizerImplementation.FP32_MASTER_ADAMW_V1,
            optimizer_groups,
            AdamWConfiguration(
                learning_rate=config.learning_rate,
                beta1=config.adam_beta1,
                beta2=config.adam_beta2,
                epsilon=config.adam_epsilon,
                weight_decay=config.weight_decay,
            ),
        )
        scheduler = build_registered_schedule(
            ScheduleImplementation.LINEAR_WARMUP_COSINE_V1,
            optimizer,
            LinearWarmupCosineConfiguration(
                warmup_steps=config.warmup_steps,
                max_steps=(
                    config.learning_rate_schedule_steps
                    if config.learning_rate_schedule_steps is not None
                    else config.max_steps
                ),
                minimum_ratio=config.min_learning_rate_ratio,
            ),
        )

    train_rows = load_domain_manifest(Path(config.train_manifest))
    eval_rows = (
        load_domain_manifest(Path(config.eval_manifest)) if config.eval_manifest else []
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
        encoder_cache = FrozenEncoderCache(
            config.encoder_cache_dir,
            mode=config.encoder_cache_mode,
            model_id=config.model_id,
            model_revision=config.model_revision,
        )
        model._training_encoder_cache = encoder_cache
        from mage_flow.models.utils import PROMPT_TEMPLATE

        info = PROMPT_TEMPLATE["mage-flow"]
        cache_rows = [*train_rows, *eval_rows]
        cache_prompts, _cache_kinds = effective_conditioning_prompts(
            cache_rows,
            caption_dropout=0.0,
            rng=random.Random(config.seed),
        )
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
            if hasattr(model.vae, "dconv_encoder"):
                model.vae.dconv_encoder.to("cpu")
            else:
                raise RuntimeError(
                    "offload_cached_encoders requires MageVAE.dconv_encoder"
                )
            model._training_text_encoder_offloaded = True
            model._training_vae_encoder_offloaded = True
    if not eval_rows:
        _drop_vae_decoder(model)
    contract_fingerprint = training_contract_fingerprint(
        config, component_composition_digest=component_digest
    )

    global_step, start_epoch, start_batch = 0, 0, 0
    loaded_checkpoint: Path | None = None
    exact_crash_recovery = False
    if config.resume_from:
        loaded_checkpoint = Path(config.resume_from)
        if config.auto_resume_latest:
            loaded_checkpoint = latest_compatible_checkpoint(
                output_dir,
                loaded_checkpoint,
                contract_fingerprint=contract_fingerprint,
            )
        restored = load_training_checkpoint(
            controller,
            optimizer,
            scheduler,
            loaded_checkpoint,
            expected_contract_fingerprint=contract_fingerprint,
            transformer=transformer,
        )
        global_step = restored["global_step"]
        start_epoch = restored["epoch"]
        start_batch = restored["batch_index"]
    elif config.continuation_from:
        if config.auto_resume_latest:
            try:
                loaded_checkpoint = latest_compatible_checkpoint(
                    output_dir,
                    None,
                    contract_fingerprint=contract_fingerprint,
                )
                exact_crash_recovery = True
            except ValueError:
                loaded_checkpoint = None
        if exact_crash_recovery:
            restored = load_training_checkpoint(
                controller,
                optimizer,
                scheduler,
                loaded_checkpoint,
                expected_contract_fingerprint=contract_fingerprint,
                transformer=transformer,
            )
        else:
            loaded_checkpoint = Path(config.continuation_from).expanduser().resolve()
            restored = load_training_checkpoint(
                controller,
                optimizer,
                scheduler,
                loaded_checkpoint,
                restore_optimizer=not config.continuation_reset_optimizer,
                restore_scheduler=not config.continuation_reset_scheduler,
                transformer=transformer,
            )
            if config.continuation_reset_scheduler:
                for parameter_group in optimizer.param_groups:
                    group_learning_rate = (
                        config.learning_rate
                        if parameter_group.get("group_name") == "experts"
                        else config.learning_rate
                        * config.shared_learning_rate_multiplier
                    )
                    parameter_group["lr"] = group_learning_rate
                    parameter_group["initial_lr"] = group_learning_rate
        global_step = restored["global_step"]
        if exact_crash_recovery or config.continuation_preserve_data_position:
            start_epoch = restored["epoch"]
            start_batch = restored["batch_index"]
        else:
            # A data continuation begins its new frozen manifest at the first
            # deterministic batch unless explicitly preserving membership.
            start_epoch = 0
            start_batch = 0
    fixed_expert_source_dir = None
    if not config.train_experts:
        if loaded_checkpoint is None:
            raise ValueError("frozen experts require a checkpoint source")
        fixed_expert_source_dir = prepare_fixed_expert_cache(
            loaded_checkpoint, output_dir
        )
        for parameter in controller.parameters():
            parameter.data = parameter.data.to(dtype=torch.bfloat16)
    if global_step >= config.max_steps:
        raise ValueError(
            f"checkpoint step {global_step} is not below max_steps {config.max_steps}"
        )
    compile_report = compile_transformer_blocks(
        transformer,
        enabled=config.compile_transformer_blocks,
        mode=config.compile_transformer_mode,
        dynamic=config.compile_transformer_dynamic,
    )

    _atomic_json(
        output_dir / "run_contract.json",
        {
            "schema": RUN_SCHEMA,
            "created_at": _utc_now(),
            "config": asdict(config),
            "official_repository": OFFICIAL_REPOSITORY,
            "official_source_revision": config.official_source_revision,
            "base_model": config.model_id,
            "base_revision": config.model_revision,
            "torch": torch.__version__,
            "gpu": torch.cuda.get_device_name(device),
            "runtime_optimizations": {
                "attention_backend": config.attention_backend,
                "activation_checkpointing": activation_checkpoint_report,
                "regional_compile": compile_report,
                "float8": float8_report,
                "optimizer_precision": optimizer.precision_report(),
                "frozen_encoder_cache": encoder_cache_report,
                "cached_encoder_offload": config.offload_cached_encoders,
            },
            "train_examples": len(train_rows),
            "eval_examples": len(eval_rows),
            "eval_domain_counts": dict(
                sorted(Counter(str(row["domain"]) for row in eval_rows).items())
            ),
            "evaluation_contract": {
                "unified_phase": True,
                "all_manifest_examples": config.eval_batches_per_domain is None,
                "routes": (
                    list(EXPERT_DOMAINS)
                    if config.mandatory_expert_routing
                    else ["general", *EXPERT_DOMAINS]
                ),
                "routing_contract": (
                    "exactly_one_expert"
                    if config.mandatory_expert_routing
                    else "general_or_one_expert"
                ),
                "gallery_samples_per_domain": EVAL_GALLERY_SAMPLES_PER_DOMAIN,
                "gallery_generation_steps": EVAL_GALLERY_STEPS,
                "gallery_cfg": EVAL_GALLERY_CFG,
            },
            "expert_config": asdict(controller.config),
            "runtime_expert_dtype": (
                "bfloat16" if not config.train_experts else "float32"
            ),
            "training_scope": scope,
            "optimizer_groups": [
                {
                    "name": str(group.get("group_name", "unnamed")),
                    "parameters": sum(
                        parameter.numel() for parameter in group["params"]
                    ),
                    "initial_learning_rate": float(group["initial_lr"]),
                }
                for group in optimizer_groups
            ],
            "parameter_routing": optimizer_routing.report,
            "training_components": component_evidence,
            "trainable_parameters": sum(parameter.numel() for parameter in trainable),
            "trainable_parameters_per_domain": {
                domain: (
                    controller.parameter_count(domain) if config.train_experts else 0
                )
                for domain in EXPERT_DOMAINS
            },
            "installed_expert_parameters_per_domain": {
                domain: controller.parameter_count(domain) for domain in EXPERT_DOMAINS
            },
            "general_training_weight": 0.0,
            "domain_dropout": 0.0,
            "training_contract_sha256": contract_fingerprint,
            "lineage": {
                "mode": (
                    "exact_resume"
                    if config.resume_from or exact_crash_recovery
                    else "data_route_continuation"
                    if config.continuation_from
                    else "fresh"
                ),
                "parent_checkpoint": (
                    str(loaded_checkpoint) if loaded_checkpoint is not None else None
                ),
                "data_position_reset": bool(config.continuation_from)
                and not exact_crash_recovery
                and not config.continuation_preserve_data_position,
                "optimizer_restored": bool(
                    config.resume_from
                    or exact_crash_recovery
                    or config.continuation_from
                )
                and (exact_crash_recovery or not config.continuation_reset_optimizer),
                "scheduler_restored": bool(
                    config.resume_from
                    or exact_crash_recovery
                    or config.continuation_from
                )
                and (exact_crash_recovery or not config.continuation_reset_scheduler),
                "rng_restored": bool(config.resume_from or config.continuation_from),
            },
            "training_scope_preflight": preflight,
        },
    )

    stop_requested = {"value": False}

    def handle_stop(_signum, _frame):
        stop_requested["value"] = True

    signal.signal(signal.SIGINT, handle_stop)
    signal.signal(signal.SIGTERM, handle_stop)
    _atomic_json(
        output_dir / "status.json",
        {
            "schema": RUN_SCHEMA,
            "state": "training",
            "step": global_step,
            "updated_at": _utc_now(),
        },
    )
    if eval_rows and not unified_evaluation_is_complete(output_dir, global_step):
        run_unified_evaluation(
            pipeline,
            transformer,
            controller,
            model,
            eval_rows,
            config,
            device,
            output_dir,
            step=global_step,
        )

    optimizer.zero_grad(set_to_none=True)
    epoch = start_epoch
    accumulation_index = 0
    accumulation_loss: torch.Tensor | None = None
    accumulation_domains: Counter[str] = Counter()
    accumulation_samples = 0
    accumulation_target_tokens = 0
    accumulation_text_tokens = 0
    accumulation_uncaptioned = 0
    accumulation_cfg_dropout = 0
    accumulation_input_wait = 0.0
    encode_events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
    train_events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
    update_window_started = time.perf_counter()
    input_wait_started = update_window_started
    last_epoch, next_batch = start_epoch, start_batch
    while global_step < config.max_steps:
        batches = epoch_training_batches(train_rows, config, epoch=epoch)
        first_batch = start_batch if epoch == start_epoch else 0
        batch_stream = (
            (batch_index, [train_rows[index] for index in indices])
            for batch_index, indices in enumerate(
                batches[first_batch:], start=first_batch
            )
        )

        def load_batch(item):
            batch_index, batch_rows = item
            image_tensors = []
            for row in batch_rows:
                if encoder_cache is not None and encoder_cache.has_moments(row):
                    image_tensors.append(None)
                else:
                    image = _load_image_tensor(row)
                    image_tensors.append(image.pin_memory())
            return (
                batch_index,
                batch_rows,
                image_tensors,
            )

        loaded_batches = _prefetched(
            batch_stream, load_batch, depth=config.prefetch_batches
        )
        if worker_step_profiler is not None:
            loaded_batches = worker_step_profiler.track_input(loaded_batches)
        for batch_index, batch_rows, images in loaded_batches:
            accumulation_input_wait += time.perf_counter() - input_wait_started
            encode_start = torch.cuda.Event(enable_timing=True)
            encode_end = torch.cuda.Event(enable_timing=True)
            encode_start.record()
            flow = encode_domain_batch(model, batch_rows, images, config, device)
            encode_end.record()
            encode_events.append((encode_start, encode_end))
            domain = str(flow["domain"])
            # Keep the route active through backward: activation checkpointing
            # recomputes block forwards during this call.
            train_start = torch.cuda.Event(enable_timing=True)
            train_end = torch.cuda.Event(enable_timing=True)
            train_start.record()
            with controller.route(domain):
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    prediction = _forward_transformer(transformer, flow)
                    loss, observed_mse = rectified_flow_loss(
                        prediction, flow["velocity"]
                    )
                    scaled_loss = loss / config.gradient_accumulation_steps
                scaled_loss.backward()
            train_end.record()
            train_events.append((train_start, train_end))
            accumulation_index += 1
            accumulation_loss = (
                observed_mse
                if accumulation_loss is None
                else accumulation_loss + observed_mse
            )
            accumulation_domains[domain] += 1
            accumulation_samples += len(batch_rows)
            accumulation_target_tokens += sum(flow["image_lens"])
            accumulation_text_tokens += sum(flow["text_lens"])
            accumulation_uncaptioned += flow["conditioning_kinds"].count(
                "uncaptioned_image"
            )
            accumulation_cfg_dropout += flow["conditioning_kinds"].count("cfg_dropout")
            last_epoch, next_batch = epoch, batch_index + 1
            input_wait_started = time.perf_counter()
            if accumulation_index < config.gradient_accumulation_steps:
                continue

            optimizer_start = torch.cuda.Event(enable_timing=True)
            optimizer_end = torch.cuda.Event(enable_timing=True)
            optimizer_start.record()
            grad_norm = (
                worker_components.gradient_clipping(trainable)
                if worker_components is not None
                else torch.nn.utils.clip_grad_norm_(
                    trainable, config.max_grad_norm
                )
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_end.record()
            torch.cuda.synchronize(device)
            step_seconds = time.perf_counter() - update_window_started
            encode_ms = sum(start.elapsed_time(end) for start, end in encode_events)
            train_ms = sum(start.elapsed_time(end) for start, end in train_events)
            optimizer_ms = optimizer_start.elapsed_time(optimizer_end)
            loss_value = (
                float(accumulation_loss.item()) / config.gradient_accumulation_steps
            )
            accumulation_index = 0
            global_step += 1
            if worker_step_profiler is not None:
                worker_step_profiler.step(global_step)

            metrics = {
                "kind": "train",
                "step": global_step,
                "loss": loss_value,
                "lr": float(scheduler.get_last_lr()[0]),
                "expert_lr": next(
                    (
                        float(group["lr"])
                        for group in optimizer.param_groups
                        if group.get("group_name") == "experts"
                    ),
                    None,
                ),
                "shared_backbone_lr": next(
                    (
                        float(group["lr"])
                        for group in optimizer.param_groups
                        if group.get("group_name") == "shared_backbone"
                    ),
                    None,
                ),
                "gnorm": float(grad_norm),
                "step_seconds": step_seconds,
                "samples_per_second": accumulation_samples / step_seconds,
                "target_tokens_per_second": (accumulation_target_tokens / step_seconds),
                "phase/encode_ms": encode_ms,
                "phase/forward_backward_ms": train_ms,
                "phase/optimizer_ms": optimizer_ms,
                "phase/input_wait_ms": accumulation_input_wait * 1000.0,
                "phase/gpu_fraction": min(
                    1.0,
                    (encode_ms + train_ms + optimizer_ms) / (step_seconds * 1000.0),
                ),
                "domain_microbatches": dict(sorted(accumulation_domains.items())),
                "samples": accumulation_samples,
                "target_tokens": accumulation_target_tokens,
                "text_tokens": accumulation_text_tokens,
                "uncaptioned_samples": accumulation_uncaptioned,
                "cfg_dropout_samples": accumulation_cfg_dropout,
                "epoch": epoch,
            }
            if worker_observability is not None:
                worker_observability.optimizer_step(global_step)
                worker_observability.publish_if_declared(
                    "train.loss",
                    loss_value,
                    step=global_step,
                    sample_weight=accumulation_samples,
                )
                worker_observability.publish_if_declared(
                    "train.images_per_second",
                    metrics["samples_per_second"],
                    step=global_step,
                )
                worker_observability.publish_if_declared(
                    "system.gpu_memory_used",
                    int(torch.cuda.memory_allocated(device)),
                    step=global_step,
                )
            accumulation_loss = None
            accumulation_domains.clear()
            accumulation_samples = 0
            accumulation_target_tokens = 0
            accumulation_text_tokens = 0
            accumulation_uncaptioned = 0
            accumulation_cfg_dropout = 0
            accumulation_input_wait = 0.0
            encode_events.clear()
            train_events.clear()
            with (output_dir / "train.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(metrics) + "\n")
            _atomic_json(
                output_dir / "status.json",
                {
                    "schema": RUN_SCHEMA,
                    "state": "training",
                    "step": global_step,
                    "updated_at": _utc_now(),
                },
            )
            update_window_started = time.perf_counter()
            input_wait_started = update_window_started

            if eval_rows and global_step % config.eval_every == 0:
                run_unified_evaluation(
                    pipeline,
                    transformer,
                    controller,
                    model,
                    eval_rows,
                    config,
                    device,
                    output_dir,
                    step=global_step,
                )
            if global_step % config.checkpoint_every == 0:
                save_training_checkpoint(
                    controller,
                    optimizer,
                    scheduler,
                    output_dir,
                    global_step=global_step,
                    epoch=epoch,
                    batch_index=batch_index + 1,
                    keep_last_n=config.keep_last_n,
                    contract_fingerprint=contract_fingerprint,
                    transformer=transformer,
                    fixed_expert_source_dir=fixed_expert_source_dir,
                )
            if stop_requested["value"]:
                checkpoint = save_training_checkpoint(
                    controller,
                    optimizer,
                    scheduler,
                    output_dir,
                    global_step=global_step,
                    epoch=epoch,
                    batch_index=batch_index + 1,
                    keep_last_n=config.keep_last_n,
                    contract_fingerprint=contract_fingerprint,
                    transformer=transformer,
                    fixed_expert_source_dir=fixed_expert_source_dir,
                )
                _atomic_json(
                    output_dir / "status.json",
                    {
                        "schema": RUN_SCHEMA,
                        "state": "interrupted",
                        "step": global_step,
                        "checkpoint": str(checkpoint),
                        "updated_at": _utc_now(),
                    },
                )
                return
            if global_step >= config.max_steps:
                break

        if global_step >= config.max_steps:
            break
        epoch += 1
        start_batch = 0
        last_epoch, next_batch = epoch, 0

    final_checkpoint = save_training_checkpoint(
        controller,
        optimizer,
        scheduler,
        output_dir,
        global_step=global_step,
        epoch=last_epoch,
        batch_index=next_batch,
        keep_last_n=config.keep_last_n,
        contract_fingerprint=contract_fingerprint,
        transformer=transformer,
        fixed_expert_source_dir=fixed_expert_source_dir,
    )
    exports = export_final_weights(controller, transformer, output_dir)
    _atomic_json(
        output_dir / "complete.json",
        {
            "schema": RUN_SCHEMA,
            "completed_at": _utc_now(),
            "global_step": global_step,
            "checkpoint": str(final_checkpoint),
            "weights": exports,
            "experts": {domain: exports[domain] for domain in EXPERT_DOMAINS},
            "shared_backbone": exports.get("shared_final_third"),
        },
    )
    _atomic_json(
        output_dir / "status.json",
        {
            "schema": RUN_SCHEMA,
            "state": "complete",
            "step": global_step,
            "updated_at": _utc_now(),
        },
    )


def prepare_run(config: MageFlowExpertTrainConfig, run_dir: Path) -> dict[str, Any]:
    """Write a pinned single-GPU launch directory."""
    config.validate()
    run_dir = run_dir.expanduser().resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    config_path = run_dir / "train_config.json"
    _atomic_json(config_path, asdict(config))
    repo_root = Path(__file__).resolve().parents[2]
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
        f"REPO_ROOT={shlex.quote(str(repo_root))}\n"
        'export HF_HOME="${MAGE_FLOW_HF_HOME:-$REPO_ROOT/.hf_cache}"\n'
        f'VENV="${{MAGE_FLOW_VENV:-$REPO_ROOT/{default_venv}}}"\n'
        'exec "$VENV/bin/python" -m rwkv_lab.mage_flow_expert_train '
        'train --config "$RUN_DIR/train_config.json"\n',
        encoding="utf-8",
    )
    launcher.chmod(0o755)
    rows = load_domain_manifest(Path(config.train_manifest))
    audit = audit_domain_rows(
        rows, max_uncaptioned_fraction=config.max_uncaptioned_fraction
    )
    expert_audit = audit_domain_rows(
        [row for row in rows if row["domain"] in EXPERT_DOMAINS],
        max_uncaptioned_fraction=config.max_uncaptioned_fraction,
    )
    eval_rows = (
        load_domain_manifest(Path(config.eval_manifest)) if config.eval_manifest else []
    )
    eval_audit = (
        audit_domain_rows(
            eval_rows,
            max_uncaptioned_fraction=config.max_uncaptioned_fraction,
        )
        if eval_rows
        else None
    )
    receipt = {
        "schema": RUN_SCHEMA,
        "created_at": _utc_now(),
        "model_id": config.model_id,
        "model_revision": config.model_revision,
        "official_source_revision": config.official_source_revision,
        "config": str(config_path),
        "launcher": str(launcher),
        "train_audit": audit,
        "expert_train_audit": expert_audit,
        "eval_audit": eval_audit,
        "evaluation_contract": {
            "unified_phase": True,
            "all_manifest_examples": config.eval_batches_per_domain is None,
            "eval_examples": len(eval_rows),
            "eval_domain_counts": dict(
                sorted(Counter(str(row["domain"]) for row in eval_rows).items())
            ),
            "routes": (
                list(EXPERT_DOMAINS)
                if config.mandatory_expert_routing
                else ["general", *EXPERT_DOMAINS]
            ),
            "routing_contract": (
                "exactly_one_expert"
                if config.mandatory_expert_routing
                else "general_or_one_expert"
            ),
            "gallery_samples_per_domain": EVAL_GALLERY_SAMPLES_PER_DOMAIN,
            "gallery_generation_steps": EVAL_GALLERY_STEPS,
            "gallery_cfg": EVAL_GALLERY_CFG,
        },
        "expert_only": config.train_experts and config.train_shared_final_fraction == 0,
        "training_scope": {
            "train_experts": config.train_experts,
            "train_shared_final_fraction": config.train_shared_final_fraction,
        },
        "general_training_weight": 0.0,
        "domain_dropout": 0.0,
        "training_contract_sha256": training_contract_fingerprint(config),
    }
    _atomic_json(run_dir / "preparation_receipt.json", receipt)
    return receipt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan = subparsers.add_parser("plan")
    plan.add_argument("--train-manifest", type=Path, required=True)
    plan.add_argument("--eval-manifest", type=Path)
    plan.add_argument("--run-dir", type=Path, required=True)
    plan.add_argument("--output-dir", type=Path, required=True)
    plan.add_argument("--max-steps", type=int, default=16_000)
    plan.add_argument("--microbatch-size", type=int, default=2)
    plan.add_argument("--gradient-accumulation-steps", type=int, default=4)
    plan.add_argument("--photo-weight", type=float, default=0.5)
    plan.add_argument("--animation-weight", type=float, default=0.5)
    plan.add_argument("--learning-rate", type=float, default=1.0e-4)
    plan.add_argument("--shared-learning-rate-multiplier", type=float, default=1.0)
    plan.add_argument("--learning-rate-schedule-steps", type=int)
    plan.add_argument("--warmup-steps", type=int, default=500)
    plan.add_argument("--caption-dropout", type=float, default=0.1)
    plan.add_argument("--final-block-fraction", type=float, default=2 / 3)
    plan.add_argument("--expert-parameter-fraction", type=float, default=0.15)
    plan.add_argument("--checkpoint-every", type=int, default=500)
    plan.add_argument("--eval-every", type=int, default=500)
    plan.add_argument("--eval-microbatch-size", type=int)
    plan.add_argument("--seed", type=int, default=42)
    plan.add_argument("--continuation-from", type=Path)
    plan.add_argument("--continuation-reset-scheduler", action="store_true")
    plan.add_argument("--continuation-reset-optimizer", action="store_true")
    plan.add_argument("--continuation-preserve-data-position", action="store_true")
    plan.add_argument(
        "--freeze-experts",
        action="store_false",
        dest="train_experts",
        default=True,
    )
    plan.add_argument("--train-shared-final-fraction", type=float, default=0.0)
    plan.add_argument("--mandatory-expert-routing", action="store_true")
    plan.add_argument("--auto-resume-latest", action="store_true")
    plan.add_argument(
        "--attention-backend",
        choices=("flash2", "flash4"),
        default="flash2",
    )
    plan.add_argument(
        "--activation-checkpointing-mode",
        choices=sorted(ACTIVATION_CHECKPOINT_MODES),
        default="full",
    )
    plan.add_argument("--compile-transformer-blocks", action="store_true")
    plan.add_argument("--compile-transformer-mode", default="default")
    plan.add_argument("--float8-training", action="store_true")
    plan.add_argument(
        "--float8-recipe",
        choices=sorted(FLOAT8_RECIPES),
        default="tensorwise",
    )
    plan.add_argument("--encoder-cache-dir", type=Path)
    plan.add_argument(
        "--encoder-cache-mode",
        choices=sorted(ENCODER_CACHE_MODES),
        default="off",
    )
    plan.add_argument("--offload-cached-encoders", action="store_true")

    run = subparsers.add_parser("train")
    run.add_argument("--config", type=Path, required=True)
    cache = subparsers.add_parser("cache-encoders")
    cache.add_argument("--config", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "cache-encoders":
        config = MageFlowExpertTrainConfig.from_path(args.config)
        print(json.dumps(cache_frozen_encoders(config), indent=2, sort_keys=True))
        return
    if args.command == "plan":
        config = MageFlowExpertTrainConfig(
            train_manifest=str(args.train_manifest.expanduser().resolve()),
            eval_manifest=(
                str(args.eval_manifest.expanduser().resolve())
                if args.eval_manifest
                else None
            ),
            output_dir=str(args.output_dir.expanduser().resolve()),
            max_steps=args.max_steps,
            microbatch_size=args.microbatch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            photo_weight=args.photo_weight,
            animation_weight=args.animation_weight,
            learning_rate=args.learning_rate,
            shared_learning_rate_multiplier=args.shared_learning_rate_multiplier,
            learning_rate_schedule_steps=args.learning_rate_schedule_steps,
            warmup_steps=args.warmup_steps,
            caption_dropout=args.caption_dropout,
            final_block_fraction=args.final_block_fraction,
            expert_parameter_fraction=args.expert_parameter_fraction,
            checkpoint_every=args.checkpoint_every,
            eval_every=args.eval_every,
            eval_microbatch_size=args.eval_microbatch_size,
            seed=args.seed,
            continuation_from=(
                str(args.continuation_from.expanduser().resolve())
                if args.continuation_from
                else None
            ),
            continuation_reset_scheduler=args.continuation_reset_scheduler,
            continuation_reset_optimizer=args.continuation_reset_optimizer,
            continuation_preserve_data_position=(
                args.continuation_preserve_data_position
            ),
            train_experts=args.train_experts,
            train_shared_final_fraction=args.train_shared_final_fraction,
            mandatory_expert_routing=args.mandatory_expert_routing,
            auto_resume_latest=args.auto_resume_latest,
            attention_backend=args.attention_backend,
            activation_checkpointing_mode=args.activation_checkpointing_mode,
            compile_transformer_blocks=args.compile_transformer_blocks,
            compile_transformer_mode=args.compile_transformer_mode,
            float8_training=args.float8_training,
            float8_recipe=args.float8_recipe,
            encoder_cache_dir=(
                str(args.encoder_cache_dir.expanduser().resolve())
                if args.encoder_cache_dir
                else None
            ),
            encoder_cache_mode=args.encoder_cache_mode,
            offload_cached_encoders=args.offload_cached_encoders,
        )
        print(
            json.dumps(
                prepare_run(config, args.run_dir),
                indent=2,
                sort_keys=True,
            )
        )
        return
    config = MageFlowExpertTrainConfig.from_path(args.config.expanduser().resolve())
    try:
        train(config)
    except Exception as error:
        output_dir = Path(config.output_dir).expanduser().resolve()
        prior_step = 0
        status_path = output_dir / "status.json"
        if status_path.is_file():
            try:
                prior_step = int(
                    json.loads(status_path.read_text(encoding="utf-8")).get("step", 0)
                )
            except (OSError, ValueError, json.JSONDecodeError):
                pass
        _atomic_json(
            status_path,
            {
                "schema": RUN_SCHEMA,
                "state": "failed",
                "step": prior_step,
                "error_type": type(error).__name__,
                "error": str(error),
                "updated_at": _utc_now(),
            },
        )
        raise


if __name__ == "__main__":
    main()
