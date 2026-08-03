"""Single-resident terminal-expert training for Mage-Flow.

Each run trains one domain window. Exactly one three-block terminal expert is
resident. TREAD only removes selected image tokens from a contiguous span of
the otherwise-complete original backbone; learned factored loops reuse the
existing backbone and expert blocks without replacing them. The inactive
expert remains an independent checkpoint and is never added to the module
tree.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import random
import shlex
import shutil
import signal
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

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
    prefetch_frozen_encoder_batch,
    unified_evaluation_is_complete,
)
from rwkv_lab.mage_flow_optimizations import (
    ACTIVATION_CHECKPOINT_MODES,
    ENCODER_CACHE_MODES,
    FLOAT8_RECIPES,
    FP32MasterAdamW,
    FrozenEncoderCache,
    cache_coverage,
    compile_transformer_blocks,
    configure_activation_checkpointing,
    convert_trainable_image_ffns_to_float8,
)
from rwkv_lab.mage_flow_pretrain import (
    _load_image_tensor,
    _prefetched,
    rectified_flow_loss,
)
from rwkv_lab.mage_flow_terminal_experts import (
    configure_terminal_training_scope,
    convert_terminal_path_to_lightning_blocks,
    install_terminal_expert,
    load_terminal_expert,
    load_terminal_shared_backbone,
    save_terminal_expert,
    save_terminal_shared_backbone,
    terminal_architecture_report,
    terminal_optimizer_parameter_routing,
)
from rwkv_lab.mage_flow_training_objectives import (
    FLOW_LOSS_WEIGHTINGS,
    VAERepresentationAlignment,
    apply_immiscible_noise_assignment,
    effective_flow_loss_weights,
    flow_min_snr_weights,
    load_repa_projection,
    rectified_flow_loss_per_example,
    save_repa_projection,
    velocity_direction_loss_per_example,
    weighted_rectified_flow_loss,
    weighted_velocity_direction_loss,
)
from rwkv_lab.mage_flow_tread_looping import (
    TreadLoopConfig,
    auxiliary_loop_flow_loss,
    install_tread_factored_looping,
    learned_loop_ponder_loss,
    load_tread_loop_controller,
    write_mageflow_loop_telemetry,
)
from rwkv_lab.training_components import (
    AdamWConfiguration,
    LinearWarmupCosineConfiguration,
    OptimizerImplementation,
    ScheduleImplementation,
    build_registered_optimizer,
    build_registered_schedule,
)
from rwkv_lab.trainvm_adapters.mageflow_controls import MageFlowMutableControls

if TYPE_CHECKING:
    from rwkv_lab.trainvm_adapters import WorkerTrainingComponents
    from rwkv_lab.trainvm_worker import (
        WorkerControlRuntime,
        WorkerExecutionPhases,
        WorkerObservability,
        WorkerStepProfiler,
    )

RUN_SCHEMA = "rwkv-lab.mage-flow-terminal-train.v3"
CACHE_SPAN_SCHEMA = "rwkv-lab.mage-flow-cache-span.v1"
_RUNTIME_FIELDS = frozenset(
    {
        "encoder_cache_dir",
        "encoder_cache_mode",
        "offload_cached_encoders",
        "encoder_cache_coverage_manifest",
        "encoder_cache_covered_until_step",
        "prefetch_batches",
        "compile_transformer_blocks",
        "compile_transformer_mode",
        "compile_transformer_dynamic",
        "eval_on_resume",
        "expert_optimizer_state_device",
    }
)


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


# Local Mage-Flow-Base weights. This was a bare absolute path that existed only
# on the maintainer's host, so every TerminalExpertTrainConfig built elsewhere
# inherited it and failed validation on a path the user never chose.
#
# Resolution order, chosen so the maintainer's runs behave exactly as before:
#   1. MAGE_FLOW_BASE_PATH, for hosts that cache the weights somewhere else;
#   2. the historical path, but only when it actually exists;
#   3. None, meaning "not cached locally" — the run resolves by model_id.
#
# Step 2 is deliberately existence-checked rather than unconditional: an
# absolute default that is merely absent should not be an error, because the
# caller never asked for it. A path the caller DID supply is still validated.
MAGE_FLOW_BASE_LOCAL_PATH_ENV = "MAGE_FLOW_BASE_PATH"
MAGE_FLOW_BASE_HISTORICAL_PATH = (
    "/thearray/git/ob/text-generation-webui/models/Mage-Flow-Base"
)


def default_model_path() -> str | None:
    """Best-effort local Mage-Flow-Base directory, or None when uncached."""
    configured = os.environ.get(MAGE_FLOW_BASE_LOCAL_PATH_ENV)
    if configured:
        return configured
    if Path(MAGE_FLOW_BASE_HISTORICAL_PATH).expanduser().is_dir():
        return MAGE_FLOW_BASE_HISTORICAL_PATH
    return None


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
    model_path: str | None = field(default_factory=lambda: default_model_path())
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
    loop_gate_update_multiplier: float = 1.0
    loop_gate_max_grad_norm: float = 1.0
    caption_dropout: float = 0.1
    timestep_sampling: str = "uniform"
    timestep_shift: float = 1.0
    checkpoint_every: int = 250
    eval_every: int = 250
    eval_microbatch_size: int = 1
    eval_examples: int | None = None
    eval_on_resume: bool = True
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
    compile_transformer_dynamic: bool = False
    prefetch_batches: int = 4
    balance_accumulation_window: bool = True
    float8_training: bool = False
    float8_recipe: str = "tensorwise"
    encoder_cache_dir: str | None = None
    encoder_cache_mode: str = "off"
    offload_cached_encoders: bool = False
    encoder_cache_coverage_manifest: str | None = None
    encoder_cache_covered_until_step: int | None = None
    repa_enabled: bool = True
    repa_loss_weight: float = 0.5
    repa_hidden_layer: int = 1
    repa_projection_hidden_dim: int | None = None
    repa_smooth_l1_beta: float = 1.0
    repa_learning_rate_multiplier: float = 1.0
    repa_student_normalization: str = "token_rms"
    repa_per_example_loss_cap: float | None = 5.0
    repa_use_posterior_mean: bool = True
    repa_exclude_loop_gate_gradients: bool = True
    repa_reset_projection_on_resume: bool = False
    immiscible_enabled: bool = True
    flow_loss_weighting: str = "soft_min_snr"
    flow_min_snr_gamma: float = 5.0
    normalize_flow_loss_weights: bool = True
    velocity_direction_loss_weight: float = 0.0
    velocity_direction_loss_epsilon: float = 1.0e-6
    allow_objective_migration_on_resume: bool = False
    lightning_swiglu: bool = False
    lightning_rmsnorm: bool = False
    reset_optimizer_on_architecture_migration: bool = False
    tread_factored_looping: dict[str, Any] = field(
        default_factory=lambda: TreadLoopConfig.combined_training_preset().to_dict()
    )
    offload_replaced_backbone_core: bool = False
    tread_loop_checkpoint: str | None = None
    domain_window_schedule: str | None = None
    expert_checkpoints: dict[str, str] | None = None
    rapid_expert_alternation: bool = False
    expert_optimizer_state_device: str = "cpu"

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
        if self.model_path and not Path(self.model_path).expanduser().is_dir():
            raise ValueError(
                "local Mage-Flow model path does not exist: "
                f"{self.model_path!r}. Set model_path explicitly, set "
                f"${MAGE_FLOW_BASE_LOCAL_PATH_ENV} to where Mage-Flow-Base is "
                "cached on this host, or leave it unset to resolve by model_id."
            )
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
        if self.tread_loop_checkpoint and not Path(
            self.tread_loop_checkpoint
        ).expanduser().is_file():
            raise ValueError("TREAD/factored-loop checkpoint does not exist")
        if bool(self.domain_window_schedule) != bool(self.expert_checkpoints):
            raise ValueError(
                "domain_window_schedule and expert_checkpoints must be set together"
            )
        if self.expert_optimizer_state_device not in {"cpu", "cuda"}:
            raise ValueError("expert_optimizer_state_device must be cpu or cuda")
        if self.rapid_expert_alternation:
            if not self.domain_window_schedule:
                raise ValueError(
                    "rapid expert alternation requires both expert checkpoints"
                )
            if self.expert_optimizer_state_device != "cuda":
                raise ValueError(
                    "rapid expert alternation requires CUDA-resident optimizer state"
                )
        if self.domain_window_schedule:
            schedule_path = Path(self.domain_window_schedule).expanduser()
            if not schedule_path.is_file():
                raise ValueError("domain-window schedule does not exist")
            schedule = json.loads(schedule_path.read_text(encoding="utf-8"))
            if (
                schedule.get("schema")
                != "rwkv-lab.mage-flow-domain-window-schedule.v1"
            ):
                raise ValueError("unsupported domain-window schedule schema")
            if Path(schedule["train_manifest"]).resolve() != Path(
                self.train_manifest
            ).expanduser().resolve():
                raise ValueError("domain-window schedule has the wrong manifest")
            if int(schedule["max_steps"]) != self.max_steps:
                raise ValueError("domain-window schedule has the wrong max_steps")
            windows = schedule.get("windows")
            if not isinstance(windows, list) or not windows:
                raise ValueError("domain-window schedule contains no windows")
            if not self.resume_from and windows[0].get("domain") != self.domain:
                raise ValueError(
                    "initial domain must match the first residency window"
                )
            if set(self.expert_checkpoints or {}) != set(EXPERT_DOMAINS):
                raise ValueError(
                    f"expert_checkpoints must contain exactly {list(EXPERT_DOMAINS)}"
                )
            missing_experts = [
                str(path)
                for path in (self.expert_checkpoints or {}).values()
                if not Path(path).expanduser().is_file()
            ]
            if missing_experts:
                raise ValueError(
                    f"alternating expert checkpoints do not exist: {missing_experts}"
                )
            if Path((self.expert_checkpoints or {})[self.domain]).resolve() != Path(
                self.expert_checkpoint
            ).expanduser().resolve():
                raise ValueError(
                    "expert_checkpoint must match the initial domain checkpoint"
                )
        if self.resume_from and not Path(self.resume_from).expanduser().is_dir():
            raise ValueError("resume checkpoint directory does not exist")
        if self.resume_from and self.domain_window_schedule:
            checkpoint_contract = json.loads(
                (
                    Path(self.resume_from).expanduser() / "checkpoint.json"
                ).read_text(encoding="utf-8")
            )
            checkpoint_step = int(checkpoint_contract.get("step", -1))
            active_window = next(
                (
                    window
                    for window in schedule["windows"]
                    if int(window["start_step"])
                    <= checkpoint_step
                    < int(window["end_step"])
                ),
                None,
            )
            if (
                checkpoint_contract.get("schema") != RUN_SCHEMA
                or checkpoint_contract.get("domain") != self.domain
                or (
                    not self.rapid_expert_alternation
                    and (
                        active_window is None
                        or active_window.get("domain") != self.domain
                    )
                )
            ):
                raise ValueError(
                    "alternating resume domain does not match the active "
                    "residency window"
                )
        if self.max_steps < 1 or self.microbatch_size < 1:
            raise ValueError("step and microbatch counts must be positive")
        if self.gradient_accumulation_steps < 1:
            raise ValueError("gradient accumulation must be positive")
        if self.prefetch_batches < 0:
            raise ValueError("prefetch_batches must be nonnegative")
        if self.learning_rate <= 0 or self.backbone_learning_rate_multiplier <= 0:
            raise ValueError("learning rates must be positive")
        if self.loop_gate_update_multiplier <= 0:
            raise ValueError("loop_gate_update_multiplier must be positive")
        if self.loop_gate_max_grad_norm <= 0:
            raise ValueError("loop_gate_max_grad_norm must be positive")
        if not 0 <= self.train_backbone_final_fraction <= 1:
            raise ValueError("backbone fraction must be in [0, 1]")
        if not 0 <= self.caption_dropout < 1:
            raise ValueError("caption dropout must be in [0, 1)")
        if self.timestep_sampling not in {
            "uniform",
            "shifted_uniform",
            "logit_normal",
        }:
            raise ValueError("unsupported timestep_sampling")
        if self.timestep_shift <= 0:
            raise ValueError("timestep_shift must be positive")
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
        if self.encoder_cache_coverage_manifest:
            coverage_path = Path(
                self.encoder_cache_coverage_manifest
            ).expanduser()
            if not coverage_path.is_file():
                raise ValueError("encoder-cache coverage manifest does not exist")
            if self.encoder_cache_mode != "read_only":
                raise ValueError(
                    "bounded encoder-cache coverage requires read_only mode"
                )
            if not self.resume_from:
                raise ValueError(
                    "bounded encoder-cache coverage requires a resume checkpoint"
                )
            if self.encoder_cache_covered_until_step is None:
                raise ValueError(
                    "bounded encoder-cache coverage requires an ending step"
                )
        if self.encoder_cache_covered_until_step is not None:
            if not self.encoder_cache_coverage_manifest:
                raise ValueError(
                    "encoder-cache ending step requires a coverage manifest"
                )
            if not 1 <= self.encoder_cache_covered_until_step <= self.max_steps:
                raise ValueError("encoder-cache ending step is outside training")
            if self.resume_from:
                resume_contract = json.loads(
                    (
                        Path(self.resume_from).expanduser() / "checkpoint.json"
                    ).read_text(encoding="utf-8")
                )
                if self.encoder_cache_covered_until_step <= int(
                    resume_contract["step"]
                ):
                    raise ValueError(
                        "encoder-cache ending step must follow the resume step"
                    )
        if self.repa_loss_weight < 0:
            raise ValueError("repa_loss_weight must be nonnegative")
        if self.repa_enabled and self.repa_loss_weight <= 0:
            raise ValueError("enabled REPA requires a positive loss weight")
        if not 0 <= self.repa_hidden_layer < 15:
            raise ValueError("repa_hidden_layer must select one of 15 path blocks")
        if (
            self.repa_projection_hidden_dim is not None
            and self.repa_projection_hidden_dim < 1
        ):
            raise ValueError("repa_projection_hidden_dim must be positive")
        if self.repa_smooth_l1_beta <= 0:
            raise ValueError("repa_smooth_l1_beta must be positive")
        if self.repa_learning_rate_multiplier <= 0:
            raise ValueError("repa_learning_rate_multiplier must be positive")
        if self.repa_student_normalization not in {"none", "token_rms"}:
            raise ValueError("unsupported repa_student_normalization")
        if (
            self.repa_per_example_loss_cap is not None
            and self.repa_per_example_loss_cap <= 0
        ):
            raise ValueError("repa_per_example_loss_cap must be positive")
        if self.repa_reset_projection_on_resume and not self.resume_from:
            raise ValueError(
                "repa_reset_projection_on_resume requires a resume checkpoint"
            )
        if self.flow_loss_weighting not in FLOW_LOSS_WEIGHTINGS:
            raise ValueError("unsupported flow_loss_weighting")
        if self.flow_min_snr_gamma <= 0:
            raise ValueError("flow_min_snr_gamma must be positive")
        if self.velocity_direction_loss_weight < 0:
            raise ValueError("velocity_direction_loss_weight must be nonnegative")
        if self.velocity_direction_loss_epsilon <= 0:
            raise ValueError("velocity_direction_loss_epsilon must be positive")
        if self.allow_objective_migration_on_resume and not self.resume_from:
            raise ValueError(
                "allow_objective_migration_on_resume requires a resume checkpoint"
            )
        if self.lightning_swiglu != self.lightning_rmsnorm:
            raise ValueError(
                "the qualified Lightning block migration enables SwiGLU and "
                "RMSNorm together"
            )
        if (
            self.reset_optimizer_on_architecture_migration
            and not self.resume_from
        ):
            raise ValueError(
                "optimizer reset for architecture migration requires resume"
            )
        loop_config = TreadLoopConfig.from_dict(self.tread_factored_looping)
        route_start, _ = loop_config.resolve_and_validate(
            path_depth=15,
            hidden_width=3072,
            attention_heads=24,
        )
        if (
            self.repa_enabled
            and self.repa_exclude_loop_gate_gradients
            and loop_config.combined.enabled
            and self.repa_hidden_layer >= route_start
        ):
            raise ValueError(
                "REPA loop-gate isolation requires a hidden tap before "
                "the TREAD/looping route"
            )
        if self.checkpoint_every < 1 or self.eval_every < 1:
            raise ValueError("checkpoint and evaluation intervals must be positive")
        if self.eval_microbatch_size < 1 or self.keep_last_n < 1:
            raise ValueError("invalid evaluation/checkpoint configuration")
        if inspect_manifests:
            rows = load_domain_manifest(Path(self.train_manifest))
            required_domains = (
                EXPERT_DOMAINS if self.domain_window_schedule else (self.domain,)
            )
            for domain in required_domains:
                if not any(row["domain"] == domain for row in rows):
                    raise ValueError(f"manifest has no {domain} training rows")
            if self.eval_manifest:
                eval_rows = load_domain_manifest(Path(self.eval_manifest))
                for domain in required_domains:
                    if not any(row["domain"] == domain for row in eval_rows):
                        raise ValueError(
                            f"manifest has no {domain} evaluation rows"
                        )


def _contract_fingerprint(
    config: TerminalExpertTrainConfig,
    *,
    component_composition_digest: str | None = None,
    mutable_control_keys: Sequence[str] = (),
) -> str:
    values = asdict(config)
    values.pop("output_dir", None)
    values.pop("resume_from", None)
    mutable = tuple(sorted(set(mutable_control_keys)))
    unknown_mutable = set(mutable) - {
        "learning_rate",
        "eval_every",
        "caption_dropout",
    }
    if unknown_mutable:
        raise ValueError("training contract has an unknown mutable control")
    for key in mutable:
        values.pop(key, None)
    payload = {
        "schema": RUN_SCHEMA,
        "config": values,
        "mutable_control_keys": mutable,
        "train_manifest_sha256": _sha256(Path(config.train_manifest)),
        "eval_manifest_sha256": (
            _sha256(Path(config.eval_manifest)) if config.eval_manifest else None
        ),
        "domain_window_schedule_sha256": (
            _sha256(Path(config.domain_window_schedule))
            if config.domain_window_schedule
            else None
        ),
    }
    if component_composition_digest is not None:
        payload["component_composition_digest"] = component_composition_digest
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def resolved_worker_component_contract(
    config: TerminalExpertTrainConfig,
    worker_components: WorkerTrainingComponents | None,
    worker_controls: WorkerControlRuntime | None = None,
) -> tuple[float, Mapping[str, Mapping[str, str]] | None, str | None]:
    if worker_components is None:
        return config.learning_rate, None, None
    expected = {
        "optimizer": {
            "learning_rate": (
                worker_components.configuration(
                    "optimizer", category="optimizer"
                )["learning_rate"]
                if worker_controls is not None
                and "learning_rate" in worker_controls.effective_values
                else config.learning_rate
            ),
            "beta1": config.adam_beta1,
            "beta2": config.adam_beta2,
            "epsilon": config.adam_epsilon,
            "foreach": True,
            "fused": False,
        },
        "weight_decay": {"weight_decay": config.weight_decay},
        "parameter_router": {
            "shared_backbone_multiplier": config.backbone_learning_rate_multiplier,
            "repa_projection_multiplier": config.repa_learning_rate_multiplier,
        },
        "learning_rate": {
            "warmup_steps": config.warmup_steps,
            "max_steps": config.max_steps,
            "minimum_ratio": config.min_learning_rate_ratio,
        },
        "gradient_clipping": {
            "max_norm": config.max_grad_norm,
            "norm_type": 2.0,
            "error_if_nonfinite": False,
        },
        "loop_gate_gradient_clipping": {
            "max_norm": config.loop_gate_max_grad_norm,
            "norm_type": 2.0,
            "error_if_nonfinite": False,
        },
    }
    categories = {
        "optimizer": "optimizer",
        "weight_decay": "weight_decay_schedule",
        "parameter_router": "parameter_router",
        "learning_rate": "learning_rate_schedule",
        "gradient_clipping": "gradient_clipping",
        "loop_gate_gradient_clipping": "gradient_clipping",
    }
    for slot, configuration in expected.items():
        actual = dict(
            worker_components.configuration(slot, category=categories[slot])
        )
        if actual != configuration:
            raise ValueError(
                f"authority {categories[slot]} composition disagrees with "
                "terminal MageFlow configuration"
            )
    return (
        float(config.learning_rate),
        worker_components.evidence(),
        worker_components.composition.composition_digest,
    )


def _runtime_only_resume_change(
    checkpoint: Path,
    config: TerminalExpertTrainConfig,
) -> bool:
    """Accept old checkpoints when only frozen-cache runtime knobs changed."""

    run_contract = _checkpoint_run_contract(checkpoint, config)
    if run_contract is None:
        return False
    previous = dict(run_contract.get("config") or {})
    current = asdict(config)
    defaults = asdict(
        TerminalExpertTrainConfig(
            domain=config.domain,
            train_manifest=config.train_manifest,
            expert_checkpoint=config.expert_checkpoint,
            output_dir=config.output_dir,
        )
    )
    for key, value in defaults.items():
        previous.setdefault(key, value)
    previous_effective_batch = int(previous["microbatch_size"]) * int(
        previous["gradient_accumulation_steps"]
    )
    current_effective_batch = (
        config.microbatch_size * config.gradient_accumulation_steps
    )
    if previous_effective_batch != current_effective_batch:
        return False
    ignored = {
        "output_dir",
        "resume_from",
        "microbatch_size",
        "gradient_accumulation_steps",
        *_RUNTIME_FIELDS,
    }
    if config.domain_window_schedule:
        # Residency is checkpoint state, not a changed training contract.
        ignored.update({"domain", "expert_checkpoint", "expert_checkpoints"})
    for key in ignored:
        previous.pop(key, None)
        current.pop(key, None)
    return previous == current


def _repa_reset_resume_change(
    checkpoint: Path,
    config: TerminalExpertTrainConfig,
) -> bool:
    """Allow an explicit one-time migration away from a legacy REPA objective."""

    if not config.repa_reset_projection_on_resume:
        return False
    run_contract = _checkpoint_run_contract(checkpoint, config)
    if run_contract is None:
        return False
    previous = dict(run_contract.get("config") or {})
    current = asdict(config)
    defaults = asdict(
        TerminalExpertTrainConfig(
            domain=config.domain,
            train_manifest=config.train_manifest,
            expert_checkpoint=config.expert_checkpoint,
            output_dir=config.output_dir,
        )
    )
    for key, value in defaults.items():
        previous.setdefault(key, value)
    previous_effective_batch = int(previous["microbatch_size"]) * int(
        previous["gradient_accumulation_steps"]
    )
    current_effective_batch = (
        config.microbatch_size * config.gradient_accumulation_steps
    )
    if previous_effective_batch != current_effective_batch:
        return False
    ignored = {
        "output_dir",
        "resume_from",
        "microbatch_size",
        "gradient_accumulation_steps",
        "repa_loss_weight",
        "repa_hidden_layer",
        "repa_projection_hidden_dim",
        "repa_smooth_l1_beta",
        "repa_learning_rate_multiplier",
        "repa_student_normalization",
        "repa_per_example_loss_cap",
        "repa_use_posterior_mean",
        "repa_exclude_loop_gate_gradients",
        "repa_reset_projection_on_resume",
        *_RUNTIME_FIELDS,
    }
    if config.domain_window_schedule:
        ignored.update({"domain", "expert_checkpoint", "expert_checkpoints"})
    for key in ignored:
        previous.pop(key, None)
        current.pop(key, None)
    return previous == current


def _objective_resume_change(
    checkpoint: Path,
    config: TerminalExpertTrainConfig,
) -> bool:
    """Allow an explicit LightningDiT-style objective/sampling migration."""

    if not config.allow_objective_migration_on_resume:
        return False
    run_contract = _checkpoint_run_contract(checkpoint, config)
    if run_contract is None:
        return False
    previous = dict(run_contract.get("config") or {})
    current = asdict(config)
    defaults = asdict(
        TerminalExpertTrainConfig(
            domain=config.domain,
            train_manifest=config.train_manifest,
            expert_checkpoint=config.expert_checkpoint,
            output_dir=config.output_dir,
        )
    )
    for key, value in defaults.items():
        previous.setdefault(key, value)
    previous_effective_batch = int(previous["microbatch_size"]) * int(
        previous["gradient_accumulation_steps"]
    )
    current_effective_batch = (
        config.microbatch_size * config.gradient_accumulation_steps
    )
    if previous_effective_batch != current_effective_batch:
        return False
    ignored = {
        "output_dir",
        "resume_from",
        "microbatch_size",
        "gradient_accumulation_steps",
        "timestep_sampling",
        "velocity_direction_loss_weight",
        "velocity_direction_loss_epsilon",
        "allow_objective_migration_on_resume",
        # The corrected REPA projector is now part of the resumed checkpoint;
        # do not reset it a second time.
        "repa_reset_projection_on_resume",
        *_RUNTIME_FIELDS,
    }
    if config.domain_window_schedule:
        ignored.update({"domain", "expert_checkpoint", "expert_checkpoints"})
    for key in ignored:
        previous.pop(key, None)
        current.pop(key, None)
    return previous == current


def _architecture_resume_change(
    checkpoint: Path,
    config: TerminalExpertTrainConfig,
) -> bool:
    """Allow the explicit one-way GELU/LayerNorm to Lightning-block migration."""

    if not (
        _architecture_migration_required(checkpoint, config)
        and config.reset_optimizer_on_architecture_migration
    ):
        return False
    run_contract = _checkpoint_run_contract(checkpoint, config)
    if run_contract is None:
        return False
    previous = dict(run_contract.get("config") or {})
    current = asdict(config)
    defaults = asdict(
        TerminalExpertTrainConfig(
            domain=config.domain,
            train_manifest=config.train_manifest,
            expert_checkpoint=config.expert_checkpoint,
            output_dir=config.output_dir,
        )
    )
    for key, value in defaults.items():
        previous.setdefault(key, value)
    previous_effective_batch = int(previous["microbatch_size"]) * int(
        previous["gradient_accumulation_steps"]
    )
    current_effective_batch = (
        config.microbatch_size * config.gradient_accumulation_steps
    )
    if previous_effective_batch != current_effective_batch:
        return False
    ignored = {
        "output_dir",
        "resume_from",
        "microbatch_size",
        "gradient_accumulation_steps",
        "lightning_swiglu",
        "lightning_rmsnorm",
        "reset_optimizer_on_architecture_migration",
        *_RUNTIME_FIELDS,
    }
    if config.domain_window_schedule:
        ignored.update({"domain", "expert_checkpoint", "expert_checkpoints"})
    for key in ignored:
        previous.pop(key, None)
        current.pop(key, None)
    return previous == current


def _architecture_migration_required(
    checkpoint: Path,
    config: TerminalExpertTrainConfig,
) -> bool:
    """Return whether this resume crosses the legacy-to-Lightning boundary.

    The reset flag authorizes a one-time optimizer reset; it must not discard
    optimizer momentum on every later resume from an already migrated
    checkpoint.
    """

    if not (config.lightning_swiglu and config.lightning_rmsnorm):
        return False
    run_contract = _checkpoint_run_contract(checkpoint, config)
    if run_contract is None:
        return False
    previous = dict(run_contract.get("config") or {})
    return not (
        bool(previous.get("lightning_swiglu", False))
        and bool(previous.get("lightning_rmsnorm", False))
    )


def _checkpoint_run_contract(
    checkpoint: Path,
    config: TerminalExpertTrainConfig,
) -> dict[str, Any] | None:
    checkpoint_contract = json.loads(
        (checkpoint / "checkpoint.json").read_text(encoding="utf-8")
    )
    fingerprint = str(checkpoint_contract.get("fingerprint") or "")
    output_dir = Path(config.output_dir).expanduser().resolve()
    candidates = [
        output_dir / "run_contract.json",
        output_dir / "run_contracts" / f"{fingerprint}.json",
        checkpoint.parent / "run_contract.json",
        checkpoint.parent / "run_contracts" / f"{fingerprint}.json",
        checkpoint.parent.parent / "run_contract.json",
        checkpoint.parent.parent / "run_contracts" / f"{fingerprint}.json",
    ]
    for path in candidates:
        if not path.is_file():
            continue
        contract = json.loads(path.read_text(encoding="utf-8"))
        if contract.get("fingerprint") == fingerprint:
            return contract
    return None


def _resume_batch_index(
    checkpoint: Path,
    config: TerminalExpertTrainConfig,
    saved_batch_index: int,
) -> int:
    """Translate a checkpoint data position across equal-size batch layouts."""
    run_contract = _checkpoint_run_contract(checkpoint, config)
    if run_contract is None:
        return saved_batch_index
    previous = dict(run_contract.get("config") or {})
    previous_microbatch = int(previous.get("microbatch_size", config.microbatch_size))
    consumed_examples = saved_batch_index * previous_microbatch
    if consumed_examples % config.microbatch_size:
        raise ValueError(
            "resume data position is not divisible by the new microbatch size"
        )
    return consumed_examples // config.microbatch_size


def _resume_domain_positions(
    checkpoint: Path,
    config: TerminalExpertTrainConfig,
    saved_positions: Mapping[str, Mapping[str, Any]] | None,
) -> dict[str, dict[str, int]] | None:
    """Translate every domain cursor when an equal-size batch layout changes."""

    if not isinstance(saved_positions, Mapping):
        return None
    return {
        str(saved_domain): {
            "epoch": int(position["epoch"]),
            "batch_start": _resume_batch_index(
                checkpoint,
                config,
                int(position["batch_start"]),
            ),
        }
        for saved_domain, position in saved_positions.items()
    }


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
    accumulation_steps: int = 1,
    token_counts: list[int] | None = None,
    balance_accumulation_window: bool = False,
) -> list[list[int]]:
    indices = list(range(row_count))
    random.Random(seed + epoch * 1_000_003).shuffle(indices)
    if not balance_accumulation_window or batch_size == 1:
        return [
            indices[start : start + batch_size]
            for start in range(0, len(indices), batch_size)
        ]
    if accumulation_steps < 1:
        raise ValueError("accumulation_steps must be positive")
    if token_counts is None or len(token_counts) != row_count:
        raise ValueError("balanced batching requires one token count per row")

    result: list[list[int]] = []
    window_size = batch_size * accumulation_steps
    for start in range(0, len(indices), window_size):
        window = indices[start : start + window_size]
        microbatch_count = math.ceil(len(window) / batch_size)
        bins: list[list[int]] = [[] for _ in range(microbatch_count)]
        loads = [0] * microbatch_count
        for index in sorted(
            window,
            key=lambda value: (-token_counts[value], value),
        ):
            candidates = [
                bin_index
                for bin_index, values in enumerate(bins)
                if len(values) < batch_size
            ]
            selected = min(candidates, key=lambda value: (loads[value], value))
            bins[selected].append(index)
            loads[selected] += token_counts[index]
        result.extend(bins)
    return result


def _rapid_alternating_batches(
    rows_by_domain: Mapping[str, list[dict[str, Any]]],
    domain_positions: Mapping[str, Mapping[str, int]],
    *,
    starting_domain: str,
    batch_size: int,
    accumulation_steps: int,
    seed: int,
    balance_accumulation_window: bool,
    optimizer_steps: int | None = None,
):
    """Yield an endless deterministic photo/animation batch stream.

    Planning positions are private to this iterator, so asynchronous prefetch
    never advances the authoritative checkpoint cursors.
    """

    planned = {
        domain: {
            "epoch": int(domain_positions[domain]["epoch"]),
            "batch_start": int(domain_positions[domain]["batch_start"]),
        }
        for domain in EXPERT_DOMAINS
    }
    batch_cache: dict[tuple[str, int], list[list[int]]] = {}

    def batches_for(domain: str, epoch: int) -> list[list[int]]:
        key = (domain, epoch)
        if key not in batch_cache:
            rows = rows_by_domain[domain]
            batch_cache[key] = _epoch_batches(
                len(rows),
                batch_size=batch_size,
                seed=seed,
                epoch=epoch,
                accumulation_steps=accumulation_steps,
                token_counts=[int(row["latent_tokens"]) for row in rows],
                balance_accumulation_window=balance_accumulation_window,
            )
        return batch_cache[key]

    domain = starting_domain
    microbatch_in_step = 0
    yielded = 0
    maximum_microbatches = (
        optimizer_steps * accumulation_steps
        if optimizer_steps is not None
        else None
    )
    while True:
        if maximum_microbatches is not None and yielded >= maximum_microbatches:
            return
        position = planned[domain]
        epoch = int(position["epoch"])
        batch_index = int(position["batch_start"])
        batches = batches_for(domain, epoch)
        if batch_index >= len(batches):
            epoch += 1
            batch_index = 0
            position["epoch"] = epoch
            position["batch_start"] = batch_index
            batches = batches_for(domain, epoch)
        indices = batches[batch_index]
        position["batch_start"] = batch_index + 1
        yield domain, epoch, batch_index, indices
        yielded += 1
        microbatch_in_step += 1
        if microbatch_in_step == accumulation_steps:
            microbatch_in_step = 0
            domain = next(
                candidate for candidate in EXPERT_DOMAINS if candidate != domain
            )


def _cache_span_should_stop(
    *,
    step: int,
    max_steps: int,
    covered_until_step: int | None,
) -> bool:
    """Stop only at an intermediate cache edge; final-step cleanup still runs."""

    return (
        covered_until_step is not None
        and step >= covered_until_step
        and step < max_steps
    )


def plan_cache_span(
    config: TerminalExpertTrainConfig,
    checkpoint: Path,
    *,
    optimizer_steps: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Resolve the exact examples consumed after an early-window checkpoint."""

    if optimizer_steps < 1:
        raise ValueError("cache span must contain at least one optimizer step")
    checkpoint = checkpoint.expanduser().resolve()
    contract = json.loads(
        (checkpoint / "checkpoint.json").read_text(encoding="utf-8")
    )
    if contract.get("schema") != RUN_SCHEMA:
        raise ValueError("cache-span checkpoint has the wrong schema")
    start_step = int(contract["step"])
    end_step = start_step + optimizer_steps
    if end_step > config.max_steps:
        raise ValueError("cache span extends beyond configured training")

    rows_by_domain = {
        domain: _domain_rows(config.train_manifest, domain)
        for domain in (
            EXPERT_DOMAINS if config.domain_window_schedule else (config.domain,)
        )
    }
    positions = {domain: {"epoch": 0, "batch": 0} for domain in rows_by_domain}
    current_domain = str(contract["domain"])
    trainer_state: Mapping[str, Any] | None = None
    saved_positions = contract.get("domain_positions")
    trainer_state_path = checkpoint / "trainer_state.pt"
    if not isinstance(saved_positions, Mapping) and trainer_state_path.is_file():
        trainer_state = torch.load(
            trainer_state_path,
            map_location="cpu",
            weights_only=True,
        )
        saved_positions = trainer_state.get("domain_positions")
    if isinstance(saved_positions, Mapping):
        for domain, position in saved_positions.items():
            if domain not in positions:
                continue
            positions[str(domain)] = {
                "epoch": int(position["epoch"]),
                "batch": int(position["batch_start"]),
            }
    positions[current_domain] = {
        "epoch": int(contract["epoch"]),
        "batch": int(contract["batch_index"]),
    }
    schedule = (
        json.loads(Path(config.domain_window_schedule).read_text(encoding="utf-8"))
        if config.domain_window_schedule
        else None
    )
    saved_window_index = contract.get("window_index")
    if saved_window_index is None and trainer_state is not None:
        saved_window_index = trainer_state.get("window_index")
    window_index = int(saved_window_index or 0)
    if config.rapid_expert_alternation:
        current_domain = next(
            domain for domain in EXPERT_DOMAINS if domain != current_domain
        )
    elif schedule is not None:
        windows = schedule["windows"]
        matching = [
            index
            for index, window in enumerate(windows)
            if int(window["start_step"]) <= start_step < int(window["end_step"])
        ]
        if matching != [window_index] or current_domain != str(
            windows[window_index]["domain"]
        ):
            raise ValueError(
                "cache-span checkpoint does not match its saved residency window"
            )

    occurrences: list[dict[str, Any]] = []
    domain_occurrences = {domain: 0 for domain in rows_by_domain}
    batch_cache: dict[tuple[str, int], list[list[int]]] = {}

    def batches_for(domain: str, epoch: int) -> list[list[int]]:
        key = (domain, epoch)
        if key not in batch_cache:
            batch_cache[key] = _epoch_batches(
                len(rows_by_domain[domain]),
                batch_size=config.microbatch_size,
                seed=config.seed,
                epoch=epoch,
                accumulation_steps=config.gradient_accumulation_steps,
                token_counts=[
                    int(row["latent_tokens"]) for row in rows_by_domain[domain]
                ],
                balance_accumulation_window=config.balance_accumulation_window,
            )
        return batch_cache[key]

    for completed_step in range(start_step + 1, end_step + 1):
        position = positions[current_domain]
        for _ in range(config.gradient_accumulation_steps):
            batches = batches_for(current_domain, int(position["epoch"]))
            if int(position["batch"]) >= len(batches):
                position["epoch"] = int(position["epoch"]) + 1
                position["batch"] = 0
                batches = batches_for(current_domain, int(position["epoch"]))
            indices = batches[int(position["batch"])]
            position["batch"] = int(position["batch"]) + 1
            selected = [rows_by_domain[current_domain][index] for index in indices]
            occurrences.extend(selected)
            domain_occurrences[current_domain] += len(selected)

        if config.rapid_expert_alternation:
            current_domain = next(
                domain for domain in EXPERT_DOMAINS if domain != current_domain
            )
        elif schedule is not None:
            window = schedule["windows"][window_index]
            if completed_step == int(window["end_step"]):
                window_index += 1
                if window_index >= len(schedule["windows"]):
                    if completed_step != config.max_steps:
                        raise RuntimeError("domain schedule ended before cache span")
                    continue
                current_domain = str(
                    schedule["windows"][window_index]["domain"]
                )

    unique_rows: list[dict[str, Any]] = []
    seen = set()
    for row in occurrences:
        identity = str(Path(row["image"]).expanduser().resolve())
        if identity not in seen:
            seen.add(identity)
            unique_rows.append(row)
    report = {
        "schema": CACHE_SPAN_SCHEMA,
        "checkpoint": str(checkpoint),
        "checkpoint_fingerprint": contract["fingerprint"],
        "train_manifest": str(Path(config.train_manifest).expanduser().resolve()),
        "train_manifest_sha256": _sha256(Path(config.train_manifest)),
        "domain_window_schedule": (
            str(Path(config.domain_window_schedule).expanduser().resolve())
            if config.domain_window_schedule
            else None
        ),
        "start_step": start_step,
        "end_step": end_step,
        "optimizer_steps": optimizer_steps,
        "microbatch_size": config.microbatch_size,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "example_occurrences": len(occurrences),
        "unique_examples": len(unique_rows),
        "domain_occurrences": domain_occurrences,
        "ending_domain": current_domain,
        "ending_positions": positions,
    }
    return unique_rows, report


def write_cache_span_plan(
    config: TerminalExpertTrainConfig,
    checkpoint: Path,
    *,
    optimizer_steps: int,
    output: Path,
) -> dict[str, Any]:
    rows, report = plan_cache_span(
        config,
        checkpoint,
        optimizer_steps=optimizer_steps,
    )
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(temporary, output)
    report = {
        **report,
        "coverage_manifest": str(output),
        "coverage_manifest_sha256": _sha256(output),
    }
    _atomic_json(output.with_suffix(output.suffix + ".plan.json"), report)
    return report


def prepare_cache_span(
    config: TerminalExpertTrainConfig,
    checkpoint: Path,
    *,
    optimizer_steps: int,
    output_dir: Path,
) -> dict[str, Any]:
    """Create the exact coverage manifest plus build and resume configs."""

    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    coverage_manifest = output_dir / "coverage.jsonl"
    plan = write_cache_span_plan(
        config,
        checkpoint,
        optimizer_steps=optimizer_steps,
        output=coverage_manifest,
    )
    checkpoint = checkpoint.expanduser().resolve()
    checkpoint_contract = json.loads(
        (checkpoint / "checkpoint.json").read_text(encoding="utf-8")
    )
    checkpoint_domain = str(checkpoint_contract["domain"])
    saved_experts = checkpoint_contract.get("experts")
    checkpoint_experts = (
        {
            domain: str(checkpoint / str(saved_experts[domain]))
            for domain in EXPERT_DOMAINS
        }
        if isinstance(saved_experts, Mapping)
        else {
            domain: str(Path((config.expert_checkpoints or {})[domain]).resolve())
            for domain in EXPERT_DOMAINS
        }
    )
    entries = output_dir / "entries"
    build_values = asdict(config)
    build_values.update(
        {
            "domain": checkpoint_domain,
            "train_manifest": str(coverage_manifest),
            "expert_checkpoint": checkpoint_experts[checkpoint_domain],
            "output_dir": str(output_dir / "build"),
            "resume_from": None,
            "encoder_cache_dir": str(entries),
            "encoder_cache_mode": "read_write",
            "offload_cached_encoders": False,
            "encoder_cache_coverage_manifest": None,
            "encoder_cache_covered_until_step": None,
            "domain_window_schedule": None,
            "expert_checkpoints": None,
            "rapid_expert_alternation": False,
            "expert_optimizer_state_device": "cpu",
            "allow_objective_migration_on_resume": False,
            "reset_optimizer_on_architecture_migration": False,
        }
    )
    build_config = output_dir / "cache_build_config.json"
    _atomic_json(build_config, build_values)

    resume_values = asdict(config)
    resume_values.update(
        {
            "domain": checkpoint_domain,
            "expert_checkpoint": checkpoint_experts[checkpoint_domain],
            "expert_checkpoints": checkpoint_experts,
            "resume_from": str(checkpoint),
            "encoder_cache_dir": str(entries),
            "encoder_cache_mode": "read_only",
            "offload_cached_encoders": True,
            "encoder_cache_coverage_manifest": str(coverage_manifest),
            "encoder_cache_covered_until_step": int(plan["end_step"]),
            "eval_on_resume": False,
        }
    )
    resume_config = output_dir / "resume_config.json"
    _atomic_json(resume_config, resume_values)
    receipt = {
        **plan,
        "cache_root": str(output_dir),
        "cache_entries": str(entries),
        "cache_build_config": str(build_config),
        "resume_config": str(resume_config),
    }
    _atomic_json(output_dir / "preparation_receipt.json", receipt)
    return receipt


@dataclass(frozen=True)
class DomainWindow:
    """One interval during which exactly one expert is GPU resident."""

    index: int
    domain: str
    start_step: int
    end_step: int

    @property
    def steps(self) -> int:
        return self.end_step - self.start_step


def _optimizer_value_to_cpu(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {
            key: _optimizer_value_to_cpu(item) for key, item in value.items()
        }
    if isinstance(value, list):
        return [_optimizer_value_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_optimizer_value_to_cpu(item) for item in value)
    return value


def _optimizer_value_to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device=device)
    if isinstance(value, dict):
        return {
            key: _optimizer_value_to_device(item, device)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_optimizer_value_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_optimizer_value_to_device(item, device) for item in value)
    return value


class ResidentExpertOptimizerBank:
    """Park one inactive expert's FP32 masters and Adam state on the CPU.

    The optimizer keeps its shared-backbone parameters and state untouched.
    Expert parameter objects remain stable while their model tensors, master
    weights, and moments are replaced at a domain boundary.
    """

    def __init__(
        self,
        optimizer: FP32MasterAdamW,
        expert_model_parameters: list[torch.nn.Parameter],
        *,
        storage_device: torch.device | str = "cpu",
    ) -> None:
        expert_ids = {id(parameter) for parameter in expert_model_parameters}
        if not expert_ids:
            raise ValueError("expert optimizer bank requires expert parameters")
        self.optimizer = optimizer
        self.pairs = [
            (model, master, independent)
            for model, master, independent in optimizer._model_master_pairs
            if id(model) in expert_ids
        ]
        if len(self.pairs) != len(expert_ids):
            raise ValueError(
                "expert optimizer bank could not resolve every model parameter"
            )
        self.storage_device = torch.device(storage_device)
        if self.storage_device.type not in {"cpu", "cuda"}:
            raise ValueError("expert optimizer storage must be CPU or CUDA")
        self._inactive: dict[str, list[dict[str, Any]]] = {}

    @property
    def inactive_domains(self) -> tuple[str, ...]:
        return tuple(sorted(self._inactive))

    @torch.no_grad()
    def park(self, domain: str) -> None:
        if domain not in EXPERT_DOMAINS:
            raise ValueError(f"unsupported expert domain {domain!r}")
        if domain in self._inactive:
            raise ValueError(f"{domain} expert optimizer state is already parked")
        records = []
        for model, master, _independent in self.pairs:
            records.append(
                {
                    "master": master.detach()
                    .to(device=self.storage_device)
                    .clone(),
                    "optimizer": _optimizer_value_to_device(
                        self.optimizer.state.pop(master, {}),
                        self.storage_device,
                    ),
                }
            )
            model.grad = None
            master.grad = None
        self._inactive[domain] = records

    @torch.no_grad()
    def exchange(self, outgoing: str, incoming: str) -> None:
        """Exchange two same-device expert states without tensor copies."""

        if outgoing in self._inactive:
            raise ValueError(f"{outgoing} expert state is already inactive")
        records = self._inactive.pop(incoming, None)
        if records is None:
            raise ValueError(f"{incoming} expert state is not parked")
        if len(records) != len(self.pairs):
            raise ValueError("parked expert optimizer state has the wrong length")

        outgoing_records = []
        for (model, master, _independent), record in zip(
            self.pairs, records, strict=True
        ):
            incoming_master = record["master"]
            if incoming_master.shape != master.shape:
                raise ValueError("parked expert master weight has the wrong shape")
            if incoming_master.device != master.device:
                raise ValueError("parked expert master is not optimizer resident")

            outgoing_master = master.data
            outgoing_optimizer = self.optimizer.state.pop(master, {})
            master.data = incoming_master
            incoming_optimizer = record["optimizer"]
            if not isinstance(incoming_optimizer, Mapping):
                raise TypeError("parked expert optimizer entry must be a mapping")
            self.optimizer.state[master] = dict(incoming_optimizer)
            model.copy_(master.to(device=model.device, dtype=model.dtype))
            model.grad = None
            master.grad = None
            outgoing_records.append(
                {
                    "master": outgoing_master,
                    "optimizer": outgoing_optimizer,
                }
            )
        self._inactive[outgoing] = outgoing_records

    @torch.no_grad()
    def activate(self, domain: str, *, initialize_from_model: bool = False) -> bool:
        """Restore a parked domain or initialize clean state from loaded weights."""

        if domain not in EXPERT_DOMAINS:
            raise ValueError(f"unsupported expert domain {domain!r}")
        records = self._inactive.pop(domain, None)
        if records is None:
            if not initialize_from_model:
                raise ValueError(f"{domain} expert optimizer state is not parked")
            for model, master, _independent in self.pairs:
                master.copy_(model.detach().to(device=master.device, dtype=master.dtype))
                self.optimizer.state.pop(master, None)
            return False
        if len(records) != len(self.pairs):
            raise ValueError("parked expert optimizer state has the wrong length")
        for (model, master, _independent), record in zip(
            self.pairs, records, strict=True
        ):
            saved_master = record["master"]
            if saved_master.shape != master.shape:
                raise ValueError("parked expert master weight has the wrong shape")
            master.copy_(
                saved_master.to(device=master.device, dtype=master.dtype)
            )
            model.copy_(master.to(device=model.device, dtype=model.dtype))
            restored = _optimizer_value_to_device(
                record["optimizer"], master.device
            )
            if not isinstance(restored, Mapping):
                raise TypeError("parked expert optimizer entry must be a mapping")
            self.optimizer.state[master] = dict(restored)
        return True

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema": "rwkv-lab.resident-expert-optimizer-bank.v1",
            "inactive": _optimizer_value_to_cpu(self._inactive),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if state.get("schema") != "rwkv-lab.resident-expert-optimizer-bank.v1":
            raise ValueError("unsupported resident expert optimizer bank schema")
        inactive = state.get("inactive")
        if not isinstance(inactive, Mapping):
            raise TypeError("resident expert optimizer bank is missing state")
        domains = set(inactive)
        if not domains.issubset(EXPERT_DOMAINS):
            raise ValueError("resident expert optimizer bank has invalid domains")
        restored: dict[str, list[dict[str, Any]]] = {}
        for domain, records in inactive.items():
            if not isinstance(records, list) or len(records) != len(self.pairs):
                raise ValueError("resident expert optimizer bank length mismatch")
            restored[str(domain)] = _optimizer_value_to_device(
                records,
                self.storage_device,
            )
        self._inactive = restored


class ResidentExpertSwitcher:
    """Atomically exchange the resident expert and its optimizer state."""

    def __init__(
        self,
        controller,
        optimizer_bank: ResidentExpertOptimizerBank,
        checkpoints: Mapping[str, str | Path],
        work_dir: Path,
    ) -> None:
        if set(checkpoints) != set(EXPERT_DOMAINS):
            raise ValueError(
                f"expert switcher requires exactly {list(EXPERT_DOMAINS)}"
            )
        self.controller = controller
        self.optimizer_bank = optimizer_bank
        self.latest = {
            domain: Path(path).expanduser().resolve()
            for domain, path in checkpoints.items()
        }
        missing = [str(path) for path in self.latest.values() if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"expert switch checkpoints missing: {missing}")
        self.work_dir = work_dir.expanduser().resolve()
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.switch_count = 0

    @property
    def resident_domain(self) -> str:
        return self.controller.resident_domain

    def switch(self, domain: str) -> bool:
        if domain not in EXPERT_DOMAINS:
            raise ValueError(f"unsupported expert domain {domain!r}")
        outgoing = self.resident_domain
        if domain == outgoing:
            return False
        if (
            self.optimizer_bank.storage_device.type == "cuda"
            and domain in self.optimizer_bank.inactive_domains
        ):
            self.optimizer_bank.exchange(outgoing, domain)
            self.controller.transformer._terminal_expert_domain = domain
            self.switch_count += 1
            return True
        self.optimizer_bank.park(outgoing)
        if self.optimizer_bank.storage_device.type == "cpu":
            outgoing_path = (
                self.work_dir / f"mageflow-{outgoing}-terminal-expert.safetensors"
            )
            save_terminal_expert(self.controller, outgoing_path)
            self.latest[outgoing] = outgoing_path
        load_terminal_expert(self.controller, domain, self.latest[domain])
        self.optimizer_bank.activate(
            domain,
            initialize_from_model=domain not in self.optimizer_bank.inactive_domains,
        )
        self.switch_count += 1
        return True

    def save_domain(self, domain: str, path: Path) -> None:
        """Save either resident expert without changing the final residency."""

        if domain == self.resident_domain:
            save_terminal_expert(self.controller, path)
            return
        if self.optimizer_bank.storage_device.type != "cuda":
            shutil.copy2(self.latest[domain], path)
            metadata = self.latest[domain].with_suffix(
                self.latest[domain].suffix + ".json"
            )
            if metadata.is_file():
                shutil.copy2(
                    metadata,
                    path.with_suffix(path.suffix + ".json"),
                )
            return
        outgoing = self.resident_domain
        self.optimizer_bank.exchange(outgoing, domain)
        self.controller.transformer._terminal_expert_domain = domain
        try:
            save_terminal_expert(self.controller, path)
        finally:
            self.optimizer_bank.exchange(domain, outgoing)
            self.controller.transformer._terminal_expert_domain = outgoing


def weighted_domain_windows(
    domain_counts: dict[str, int],
    *,
    max_steps: int,
    minimum_steps: int = 500,
    maximum_steps: int = 1_000,
    seed: int = 42,
) -> list[DomainWindow]:
    """Build deterministic, strictly alternating, data-weighted windows.

    A photo/animation pair is assigned a random total span whose proportional
    split keeps both windows inside the configured bounds. Encoding the data
    ratio in window length guarantees a real expert swap at each complete
    boundary; random domain selection could repeatedly select the resident
    expert and silently create overlong windows.

    The final window may be shortened when ``max_steps`` truncates a pair.
    """

    if set(domain_counts) != set(EXPERT_DOMAINS):
        raise ValueError(
            f"domain counts must contain exactly {list(EXPERT_DOMAINS)}"
        )
    if any(not isinstance(count, int) or count <= 0 for count in domain_counts.values()):
        raise ValueError("domain counts must be positive integers")
    if max_steps < 1:
        raise ValueError("max_steps must be positive")
    if minimum_steps < 1 or maximum_steps < minimum_steps:
        raise ValueError("invalid domain-window bounds")

    total_examples = sum(domain_counts.values())
    weights = {
        domain: domain_counts[domain] / total_examples for domain in EXPERT_DOMAINS
    }
    minimum_pair_steps = math.ceil(
        max(minimum_steps / weight for weight in weights.values())
    )
    maximum_pair_steps = math.floor(
        min(maximum_steps / weight for weight in weights.values())
    )
    if minimum_pair_steps > maximum_pair_steps:
        raise ValueError(
            "the requested domain ratio cannot be represented by strictly "
            "alternating windows inside the configured bounds"
        )

    rng = random.Random(seed)
    # Start with the larger domain, then alternate unconditionally.
    pair_order = sorted(
        EXPERT_DOMAINS,
        key=lambda domain: (-domain_counts[domain], domain),
    )
    windows: list[DomainWindow] = []
    step = 0
    while step < max_steps:
        remaining = max_steps - step
        if remaining < minimum_pair_steps:
            # A deliberately short run cannot contain another complete pair.
            # Preserve strict alternation and let only its final window be
            # shorter than the configured minimum.
            pair_steps = remaining
        elif remaining <= maximum_pair_steps:
            pair_steps = remaining
        else:
            # Leave enough room for a valid final pair instead of truncating
            # one domain at the end and biasing the epoch ratio.
            pair_steps = rng.randint(
                minimum_pair_steps,
                min(maximum_pair_steps, remaining - minimum_pair_steps),
            )
        first_steps = round(pair_steps * weights[pair_order[0]])
        allocations = {
            pair_order[0]: first_steps,
            pair_order[1]: pair_steps - first_steps,
        }
        shortened_final_pair = pair_steps < minimum_pair_steps
        if not shortened_final_pair and any(
            allocation < minimum_steps or allocation > maximum_steps
            for allocation in allocations.values()
        ):
            # Rounding can only miss by one, but fail closed rather than emit a
            # schedule that violates the residency contract.
            raise RuntimeError("weighted domain-window rounding violated bounds")
        for domain in pair_order:
            if step >= max_steps:
                break
            end_step = min(max_steps, step + allocations[domain])
            windows.append(
                DomainWindow(
                    index=len(windows),
                    domain=domain,
                    start_step=step,
                    end_step=end_step,
                )
            )
            step = end_step
    return windows


def domain_window_schedule_report(
    train_manifest: str | Path,
    *,
    max_steps: int,
    minimum_steps: int = 500,
    maximum_steps: int = 1_000,
    seed: int = 42,
) -> dict[str, Any]:
    """Describe a fresh-epoch alternating schedule for a frozen manifest."""

    rows = load_domain_manifest(Path(train_manifest))
    counts = {
        domain: sum(row["domain"] == domain for row in rows)
        for domain in EXPERT_DOMAINS
    }
    windows = weighted_domain_windows(
        counts,
        max_steps=max_steps,
        minimum_steps=minimum_steps,
        maximum_steps=maximum_steps,
        seed=seed,
    )
    scheduled_steps = {
        domain: sum(window.steps for window in windows if window.domain == domain)
        for domain in EXPERT_DOMAINS
    }
    return {
        "schema": "rwkv-lab.mage-flow-domain-window-schedule.v1",
        "train_manifest": str(Path(train_manifest).expanduser().resolve()),
        "fresh_data_epoch": True,
        "seed": seed,
        "window_minimum_steps": minimum_steps,
        "window_maximum_steps": maximum_steps,
        "max_steps": max_steps,
        "domain_counts": counts,
        "domain_weights": {
            domain: counts[domain] / sum(counts.values())
            for domain in EXPERT_DOMAINS
        },
        "scheduled_steps": scheduled_steps,
        "scheduled_step_weights": {
            domain: scheduled_steps[domain] / max_steps
            for domain in EXPERT_DOMAINS
        },
        "windows": [asdict(window) | {"steps": window.steps} for window in windows],
    }


def _save_checkpoint(
    *,
    controller,
    transformer,
    repa: VAERepresentationAlignment | None,
    optimizer,
    scheduler,
    output_dir: Path,
    domain: str,
    step: int,
    epoch: int,
    batch_index: int,
    fingerprint: str,
    keep: int,
    switcher: ResidentExpertSwitcher | None = None,
    domain_positions: Mapping[str, Mapping[str, int]] | None = None,
    window_index: int | None = None,
    control_state: Mapping[str, object] | None = None,
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
    expert_names = {domain: expert_name}
    if switcher is not None:
        for inactive_domain in EXPERT_DOMAINS:
            if inactive_domain == domain:
                continue
            inactive_name = (
                f"mageflow-{inactive_domain}-terminal-expert.safetensors"
            )
            switcher.save_domain(
                inactive_domain,
                temporary / inactive_name,
            )
            expert_names[inactive_domain] = inactive_name
    repa_name = None
    if repa is not None:
        repa_name = "mageflow-vae-repa-projection.safetensors"
        save_repa_projection(repa, temporary / repa_name)
    shared = save_terminal_shared_backbone(
        transformer,
        controller,
        temporary / "mageflow-shared-final-third.safetensors",
    )
    trainer_state = {
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
        "resident_expert_optimizer_bank": (
            switcher.optimizer_bank.state_dict()
            if switcher is not None
            else None
        ),
        "domain_positions": (
            {key: dict(value) for key, value in domain_positions.items()}
            if domain_positions is not None
            else None
        ),
        "window_index": window_index,
    }
    if control_state is not None:
        trainer_state.update(control_state)
    torch.save(trainer_state, temporary / "trainer_state.pt")
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
            "experts": expert_names,
            "resident_expert": domain,
            "repa_projection": repa_name,
            "shared_backbone": shared.name if shared else None,
            "domain_positions": (
                {key: dict(value) for key, value in domain_positions.items()}
                if domain_positions is not None
                else None
            ),
            "window_index": window_index,
            "effective_control_revision": (
                control_state.get("effective_control_revision")
                if control_state is not None
                else None
            ),
            "created_at": _utc_now(),
        },
    )
    os.replace(temporary, final)
    checkpoints = sorted(output_dir.glob("checkpoint-*"))
    for stale in checkpoints[:-keep]:
        shutil.rmtree(stale)
    return final


def _best_evaluation_improved(
    output_dir: Path,
    metrics: Mapping[str, Any],
) -> bool:
    loss = float(metrics["eval/primary_loss"])
    if not math.isfinite(loss):
        return False
    manifest = output_dir / "best" / "best.json"
    if not manifest.is_file():
        return True
    previous = json.loads(manifest.read_text(encoding="utf-8"))
    return loss < float(previous["loss"])


def _promote_best_checkpoint(
    source: Path,
    output_dir: Path,
    *,
    step: int,
    loss: float,
) -> Path:
    """Publish an immutable hardlinked best checkpoint outside retention."""

    source = source.expanduser().resolve()
    if not source.is_dir() or not (source / "checkpoint.json").is_file():
        raise FileNotFoundError(f"best checkpoint source is invalid: {source}")
    best_dir = output_dir / "best"
    best_dir.mkdir(parents=True, exist_ok=True)
    final = best_dir / f"checkpoint-{step:08d}"
    if not final.is_dir():
        temporary = best_dir / f".checkpoint-{step:08d}.{os.getpid()}.tmp"
        if temporary.exists():
            shutil.rmtree(temporary)

        def hardlink_or_copy(src: str, dst: str) -> str:
            try:
                os.link(src, dst)
                return dst
            except OSError:
                return shutil.copy2(src, dst)

        shutil.copytree(source, temporary, copy_function=hardlink_or_copy)
        os.replace(temporary, final)
    previous_manifest = best_dir / "best.json"
    previous_checkpoint = None
    if previous_manifest.is_file():
        previous = json.loads(previous_manifest.read_text(encoding="utf-8"))
        previous_checkpoint = best_dir / str(previous.get("checkpoint", ""))
    _atomic_json(
        previous_manifest,
        {
            "schema": "rwkv-lab.mage-flow-best-checkpoint.v1",
            "checkpoint": final.name,
            "step": int(step),
            "loss": float(loss),
            "metric": "eval/primary_loss",
            "promoted_at": _utc_now(),
        },
    )
    if (
        previous_checkpoint is not None
        and previous_checkpoint != final
        and previous_checkpoint.is_dir()
        and previous_checkpoint.parent == best_dir
    ):
        shutil.rmtree(previous_checkpoint)
    return final


def _export_checkpoint_experts(checkpoint: Path, output_dir: Path) -> dict[str, Path]:
    """Publish every final expert from the just-written atomic checkpoint."""

    contract = json.loads(
        (checkpoint / "checkpoint.json").read_text(encoding="utf-8")
    )
    checkpoint_experts = contract.get("experts")
    if not isinstance(checkpoint_experts, Mapping):
        checkpoint_experts = {str(contract["domain"]): contract["expert"]}
    exported: dict[str, Path] = {}
    for domain, source_name in checkpoint_experts.items():
        source = checkpoint / str(source_name)
        if not source.is_file():
            raise FileNotFoundError(source)
        destination = (
            output_dir / f"mageflow-{domain}-terminal-expert.safetensors"
        )
        shutil.copy2(source, destination)
        metadata = source.with_suffix(source.suffix + ".json")
        if metadata.is_file():
            shutil.copy2(
                metadata,
                destination.with_suffix(destination.suffix + ".json"),
            )
        exported[str(domain)] = destination
    return exported


def _load_resume(
    checkpoint: Path,
    *,
    config: TerminalExpertTrainConfig,
    controller,
    transformer,
    repa: VAERepresentationAlignment | None,
    optimizer,
    scheduler,
    switcher: ResidentExpertSwitcher | None,
    domain: str,
    fingerprint: str,
    worker_controls: WorkerControlRuntime | None = None,
) -> tuple[int, int, int, dict[str, dict[str, int]] | None, int | None]:
    reset_architecture_optimizer = _architecture_migration_required(
        checkpoint,
        config,
    )
    if reset_architecture_optimizer and (
        not config.reset_optimizer_on_architecture_migration
    ):
        raise ValueError(
            "legacy-to-Lightning migration requires an explicit fresh optimizer"
        )
    contract = json.loads((checkpoint / "checkpoint.json").read_text())
    if contract.get("schema") != RUN_SCHEMA or contract.get("domain") != domain:
        raise ValueError("resume checkpoint has the wrong schema or domain")
    if (
        contract.get("fingerprint") != fingerprint
        and not _runtime_only_resume_change(checkpoint, config)
        and not _repa_reset_resume_change(checkpoint, config)
        and not _objective_resume_change(checkpoint, config)
        and not _architecture_resume_change(checkpoint, config)
    ):
        raise ValueError("resume checkpoint training contract changed")
    state = torch.load(
        checkpoint / "trainer_state.pt",
        map_location="cpu",
        weights_only=True,
    )
    if state.get("schema") != RUN_SCHEMA:
        raise ValueError("resume trainer state has the wrong schema")
    if state.get("fingerprint") != contract.get("fingerprint"):
        raise ValueError("resume checkpoint metadata and trainer state disagree")
    if worker_controls is not None:
        if contract.get("effective_control_revision") != state.get(
            "effective_control_revision"
        ):
            raise ValueError("checkpoint control metadata and state disagree")
        worker_controls.verify_checkpoint_state(state)
    load_terminal_expert(
        controller,
        domain,
        checkpoint / str(contract["expert"]),
    )
    repa_name = contract.get("repa_projection")
    fresh_repa_state = None
    if repa is not None:
        if not repa_name:
            raise ValueError("resume checkpoint is missing its REPA projection")
        if config.repa_reset_projection_on_resume:
            fresh_repa_state = {
                name: value.detach().clone()
                for name, value in repa.state_dict().items()
            }
        else:
            load_repa_projection(repa, checkpoint / str(repa_name))
    elif repa_name:
        raise ValueError("resume checkpoint has REPA but the run disabled it")
    shared = contract.get("shared_backbone")
    if shared:
        load_terminal_shared_backbone(transformer, checkpoint / str(shared))
    if not reset_architecture_optimizer:
        optimizer.load_state_dict(state["optimizer"])
    if fresh_repa_state is not None:
        repa.load_state_dict(fresh_repa_state, strict=True)
        optimizer.reset_parameter_state(list(repa.parameters()))
    scheduler.load_state_dict(state["scheduler"])
    if reset_architecture_optimizer:
        restored_lrs = scheduler.get_last_lr()
        if len(restored_lrs) != len(optimizer.param_groups):
            raise ValueError(
                "architecture migration scheduler groups do not match optimizer"
            )
        for group, learning_rate in zip(
            optimizer.param_groups, restored_lrs, strict=True
        ):
            group["lr"] = float(learning_rate)
    if switcher is not None:
        bank_state = state.get("resident_expert_optimizer_bank")
        if bank_state is None:
            raise ValueError(
                "alternating resume checkpoint is missing its expert optimizer bank"
            )
        if not reset_architecture_optimizer:
            if switcher.optimizer_bank.storage_device.type == "cuda":
                torch.cuda.empty_cache()
            switcher.optimizer_bank.load_state_dict(bank_state)
        expert_names = contract.get("experts")
        if not isinstance(expert_names, Mapping):
            raise ValueError(
                "alternating resume checkpoint is missing expert checkpoints"
            )
        for expert_domain in EXPERT_DOMAINS:
            expert_path = checkpoint / str(expert_names[expert_domain])
            if not expert_path.is_file():
                raise FileNotFoundError(expert_path)
            switcher.latest[expert_domain] = expert_path.resolve()
        saved_window_index = state.get("window_index")
        switcher.switch_count = (
            int(saved_window_index) if saved_window_index is not None else 0
        )
    random.setstate(state["python_rng"])
    torch.set_rng_state(state["torch_rng"])
    torch.cuda.set_rng_state_all(state["cuda_rng"])
    batch_index = _resume_batch_index(
        checkpoint,
        config,
        int(state["batch_index"]),
    )
    domain_positions = _resume_domain_positions(
        checkpoint,
        config,
        state.get("domain_positions"),
    )
    window_index = state.get("window_index")
    return (
        int(state["step"]),
        int(state["epoch"]),
        batch_index,
        domain_positions,
        int(window_index) if window_index is not None else None,
    )


def _evaluation_optimization_loss(
    flow_loss: float,
    *,
    velocity_direction_loss: float = 0.0,
    velocity_direction_weight: float = 0.0,
    loop_auxiliary_loss: float = 0.0,
    loop_auxiliary_weight: float = 0.0,
    loop_ponder_loss: float = 0.0,
    loop_ponder_weight: float = 0.0,
    repa_loss: float = 0.0,
    repa_weight: float = 0.0,
) -> float:
    return (
        flow_loss
        + velocity_direction_weight * velocity_direction_loss
        + loop_auxiliary_weight * loop_auxiliary_loss
        + loop_ponder_weight * loop_ponder_loss
        + repa_weight * repa_loss
    )


def _backward_training_objective(
    base_loss: torch.Tensor,
    repa_contribution: torch.Tensor | None,
) -> torch.Tensor:
    """Backpropagate the complete objective in one memory-bounded traversal.

    Loop-gate isolation is structural: validation requires the REPA tap to
    precede the routed recurrent region, so the REPA graph cannot reach those
    downstream controls.  Splitting this into two backward traversals retains
    the activation-checkpoint graph and can nearly double peak VRAM.
    """

    if repa_contribution is None:
        base_loss.backward()
        return base_loss
    total = base_loss + repa_contribution
    total.backward()
    return total


def _evaluate(
    *,
    transformer,
    controller,
    repa: VAERepresentationAlignment | None,
    model,
    rows: list[dict[str, Any]],
    config: TerminalExpertTrainConfig,
    device: torch.device,
) -> dict[str, float | int]:
    was_training = transformer.training
    transformer.eval()
    active_loop_controller = getattr(
        transformer, "tread_loop_controller", None
    )
    if active_loop_controller is not None:
        active_loop_controller.refresh_inference_skip_refinements()
    selected = rows[: config.eval_examples] if config.eval_examples else rows
    total_loss = 0.0
    total_repa_loss = 0.0
    total_loop_aux_loss = 0.0
    total_loop_ponder_loss = 0.0
    total_flow_weighted_loss = 0.0
    total_direction_loss = 0.0
    total_direction_weighted_loss = 0.0
    total_flow_weight = 0.0
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
                forwarded = _forward_transformer(
                    transformer,
                    flow,
                    return_hidden_layer=(
                        config.repa_hidden_layer if repa is not None else None
                    ),
                )
                if repa is None:
                    prediction = forwarded
                    repa_loss = None
                else:
                    prediction, hidden = forwarded
                    repa_loss = repa(
                        hidden,
                        flow["repa_target"],
                        flow["image_lens"],
                    )
                _loss, observed = rectified_flow_loss(
                    prediction,
                    flow["velocity"],
                )
                per_example = rectified_flow_loss_per_example(
                    prediction,
                    flow["velocity"],
                    flow["image_lens"],
                )
                direction_per_example = velocity_direction_loss_per_example(
                    prediction,
                    flow["velocity"],
                    flow["image_lens"],
                    epsilon=config.velocity_direction_loss_epsilon,
                )
                loop_aux_loss = (
                    auxiliary_loop_flow_loss(
                        active_loop_controller,
                        flow["velocity"],
                    )
                    if active_loop_controller is not None
                    else flow["velocity"].new_zeros((), dtype=torch.float32)
                )
                loop_ponder_loss = (
                    learned_loop_ponder_loss(active_loop_controller)
                    if active_loop_controller is not None
                    else flow["velocity"].new_zeros((), dtype=torch.float32)
                )
            raw_flow_weights = flow_min_snr_weights(
                flow["timesteps"],
                weighting=config.flow_loss_weighting,
                gamma=config.flow_min_snr_gamma,
            )
            total_loss += float(observed.item()) * len(batch_rows)
            total_flow_weighted_loss += float(
                (per_example.detach() * raw_flow_weights).sum().item()
            )
            total_direction_loss += float(
                direction_per_example.detach().sum().item()
            )
            total_direction_weighted_loss += float(
                (
                    direction_per_example.detach() * raw_flow_weights
                ).sum().item()
            )
            total_flow_weight += float(raw_flow_weights.sum().item())
            if repa_loss is not None:
                total_repa_loss += float(repa_loss.item()) * len(batch_rows)
            total_loop_aux_loss += float(loop_aux_loss.item()) * len(batch_rows)
            total_loop_ponder_loss += float(loop_ponder_loss.item()) * len(
                batch_rows
            )
            total_examples += len(batch_rows)
    transformer.train(was_training)
    metrics: dict[str, float | int] = {
        "loss": total_loss / total_examples,
        "eval/primary_loss": total_loss / total_examples,
        f"eval/{config.domain}_examples": total_examples,
        f"eval/{config.domain}_via_{config.domain}_loss": total_loss / total_examples,
        "eval/routes_per_example": 1,
        "eval/flow_weighted_loss": (
            total_flow_weighted_loss
            / (
                total_flow_weight
                if config.normalize_flow_loss_weights
                else total_examples
            )
        ),
        "eval/velocity_direction_loss": total_direction_loss / total_examples,
        "eval/velocity_direction_weighted_loss": (
            total_direction_weighted_loss
            / (
                total_flow_weight
                if config.normalize_flow_loss_weights
                else total_examples
            )
        ),
    }
    loop_auxiliary_weight = 0.0
    loop_ponder_weight = 0.0
    if active_loop_controller is not None:
        metrics["eval/loop_auxiliary_loss"] = total_loop_aux_loss / total_examples
        loop_auxiliary_weight = (
            active_loop_controller.config.looping.auxiliary_weight
        )
        metrics["eval/loop_ponder_loss"] = total_loop_ponder_loss / total_examples
        loop_ponder_weight = (
            active_loop_controller.config.looping.factorization.ponder_weight
        )
    mean_repa = 0.0
    if repa is not None:
        mean_repa = total_repa_loss / total_examples
        metrics["eval/repa_loss"] = mean_repa
    metrics["eval/optimization_loss"] = _evaluation_optimization_loss(
        float(metrics["eval/flow_weighted_loss"]),
        velocity_direction_loss=float(
            metrics["eval/velocity_direction_weighted_loss"]
        ),
        velocity_direction_weight=config.velocity_direction_loss_weight,
        loop_auxiliary_loss=float(metrics.get("eval/loop_auxiliary_loss", 0.0)),
        loop_auxiliary_weight=loop_auxiliary_weight,
        loop_ponder_loss=float(metrics.get("eval/loop_ponder_loss", 0.0)),
        loop_ponder_weight=loop_ponder_weight,
        repa_loss=mean_repa,
        repa_weight=config.repa_loss_weight if repa is not None else 0.0,
    )
    return metrics


def _run_evaluation(
    *,
    pipeline,
    transformer,
    controller,
    repa: VAERepresentationAlignment | None,
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
        repa=repa,
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


class _EvaluationSwitchingController:
    def __init__(self, controller, switcher: ResidentExpertSwitcher) -> None:
        self.controller = controller
        self.switcher = switcher

    def route(self, domain: str):
        self.switcher.switch(domain)
        loop_controller = getattr(
            self.controller.transformer,
            "tread_loop_controller",
            None,
        )
        if loop_controller is not None:
            loop_controller.refresh_inference_skip_refinements()
        return self.controller.route(domain)


def _run_alternating_evaluation(
    *,
    pipeline,
    transformer,
    controller,
    switcher: ResidentExpertSwitcher,
    repa: VAERepresentationAlignment | None,
    model,
    rows_by_domain: Mapping[str, list[dict[str, Any]]],
    config: TerminalExpertTrainConfig,
    device: torch.device,
    output_dir: Path,
    step: int,
) -> dict[str, float | int | str]:
    """Evaluate both routes, then restore the scheduled training resident."""

    training_domain = config.domain
    _atomic_json(
        output_dir / "status.json",
        {
            "schema": RUN_SCHEMA,
            "state": "evaluating",
            "step": step,
            "domains": list(EXPERT_DOMAINS),
            "updated_at": _utc_now(),
        },
    )
    combined: dict[str, float | int | str] = {}
    primary_weighted_sum = 0.0
    primary_examples = 0
    all_rows: list[dict[str, Any]] = []
    try:
        for domain in EXPERT_DOMAINS:
            switcher.switch(domain)
            config.domain = domain
            rows = rows_by_domain[domain]
            all_rows.extend(rows)
            metrics = _evaluate(
                transformer=transformer,
                controller=controller,
                repa=repa,
                model=model,
                rows=rows,
                config=config,
                device=device,
            )
            examples = int(metrics[f"eval/{domain}_examples"])
            primary_weighted_sum += float(metrics["eval/primary_loss"]) * examples
            primary_examples += examples
            for key, value in metrics.items():
                if key.startswith(f"eval/{domain}_"):
                    combined[key] = value
                elif key != "loss":
                    normalized = key.removeprefix("eval/")
                    combined[f"eval/{domain}/{normalized}"] = value

        proxy = _EvaluationSwitchingController(controller, switcher)
        artifact = generate_eval_gallery(
            pipeline,
            transformer,
            proxy,
            all_rows,
            config,
            device,
            output_dir,
            step=step,
            domains=EXPERT_DOMAINS,
        )
    finally:
        switcher.switch(training_domain)
        config.domain = training_domain

    combined["loss"] = primary_weighted_sum / primary_examples
    combined["eval/primary_loss"] = combined["loss"]
    combined["eval/routes_per_example"] = 1
    combined["eval/gallery_artifact"] = str(artifact)
    combined["eval/gallery_samples"] = (
        EVAL_GALLERY_SAMPLES_PER_DOMAIN * len(EXPERT_DOMAINS)
    )
    with (output_dir / "train.jsonl").open("a") as handle:
        handle.write(json.dumps({"kind": "eval", "step": step, **combined}) + "\n")
    _atomic_json(
        output_dir / "status.json",
        {
            "schema": RUN_SCHEMA,
            "state": "training",
            "step": step,
            "domain": training_domain,
            "updated_at": _utc_now(),
        },
    )
    return combined


def train(
    config: TerminalExpertTrainConfig,
    *,
    worker_components: WorkerTrainingComponents | None = None,
    worker_step_profiler: WorkerStepProfiler | None = None,
    worker_observability: WorkerObservability | None = None,
    worker_controls: WorkerControlRuntime | None = None,
    worker_execution_phases: WorkerExecutionPhases | None = None,
) -> None:
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
    train_log = output_dir / "train.jsonl"
    if not train_log.is_file() or train_log.stat().st_size == 0:
        with train_log.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "kind": "checkpoint",
                        "step": 0,
                        "reason": "run_initialized_pre_step_eval",
                        "domain": config.domain,
                    }
                )
                + "\n"
            )
    _atomic_json(
        output_dir / "status.json",
        {"schema": RUN_SCHEMA, "state": "initializing", "step": 0},
    )

    model_dir = (
        str(Path(config.model_path).expanduser().resolve())
        if config.model_path
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
    model.config.compile_vae_encoder = config.compile_vae_encoder
    model.vae.eval().requires_grad_(False)
    model.txt_enc.eval().requires_grad_(False)
    transformer = model.transformer
    controller = install_terminal_expert(
        transformer,
        config.domain,
        # FlashAttention requires FP16/BF16 activations. Keep the copied
        # terminal blocks in the released backbone's BF16 training dtype.
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
    # Construct the configured architecture before loading weights.  Legacy
    # GELU/LayerNorm checkpoints are translated by the Lightning modules'
    # strict-load hooks, while already migrated checkpoints load directly.
    # Loading first only works for legacy checkpoints and makes every later
    # resume fail on SwiGLU/RMSNorm tensor names.
    lightning_block_report = convert_terminal_path_to_lightning_blocks(
        transformer,
        controller,
        use_swiglu=config.lightning_swiglu,
        use_rmsnorm=config.lightning_rmsnorm,
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
    if config.tread_loop_checkpoint:
        if loop_controller is None:
            raise ValueError("a loop checkpoint requires combined mode")
        load_tread_loop_controller(
            loop_controller,
            Path(config.tread_loop_checkpoint),
        )
    scope = configure_terminal_training_scope(
        transformer,
        controller,
        train_backbone_final_fraction=config.train_backbone_final_fraction,
    )
    repa: VAERepresentationAlignment | None = None
    if config.repa_enabled:
        repa = VAERepresentationAlignment(
            int(transformer.inner_dim),
            int(transformer.img_in.in_features),
            hidden_dim=config.repa_projection_hidden_dim,
            smooth_l1_beta=config.repa_smooth_l1_beta,
            student_normalization=config.repa_student_normalization,
            per_example_loss_cap=config.repa_per_example_loss_cap,
        ).to(
            device=device,
            dtype=next(transformer.parameters()).dtype,
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
    optimizer_learning_rate, component_evidence, component_digest = (
        resolved_worker_component_contract(
            config, worker_components, worker_controls
        )
    )
    optimizer_routing = terminal_optimizer_parameter_routing(
        transformer,
        controller,
        expert_learning_rate=optimizer_learning_rate,
        backbone_learning_rate_multiplier=config.backbone_learning_rate_multiplier,
        repa_projection=repa,
        repa_learning_rate_multiplier=config.repa_learning_rate_multiplier,
        worker_components=worker_components,
    )
    groups = list(optimizer_routing.groups)
    if worker_components is not None:
        optimizer = worker_components.optimizer(groups)
        weight_decay_schedule = worker_components.weight_decay_schedule(optimizer)
    else:
        optimizer = build_registered_optimizer(
            OptimizerImplementation.FP32_MASTER_ADAMW_V1,
            groups,
            AdamWConfiguration(
                learning_rate=config.learning_rate,
                beta1=config.adam_beta1,
                beta2=config.adam_beta2,
                epsilon=config.adam_epsilon,
                weight_decay=config.weight_decay,
            ),
        )
        weight_decay_schedule = None
    domain_schedule = (
        json.loads(
            Path(config.domain_window_schedule).read_text(encoding="utf-8")
        )
        if config.domain_window_schedule
        else None
    )
    switcher = None
    optimizer_bank: ResidentExpertOptimizerBank | None = None
    if domain_schedule is not None:
        expert_group = next(
            group for group in groups if group.get("group_name") == "terminal_expert"
        )
        optimizer_bank = ResidentExpertOptimizerBank(
            optimizer,
            list(expert_group["params"]),
            storage_device=config.expert_optimizer_state_device,
        )
        switcher = ResidentExpertSwitcher(
            controller,
            optimizer_bank,
            config.expert_checkpoints or {},
            output_dir / "expert_residency",
        )
    if worker_components is not None:
        scheduler = worker_components.learning_rate_schedule(optimizer)
    else:
        scheduler = build_registered_schedule(
            ScheduleImplementation.LINEAR_WARMUP_COSINE_V1,
            optimizer,
            LinearWarmupCosineConfiguration(
                warmup_steps=config.warmup_steps,
                max_steps=config.max_steps,
                minimum_ratio=config.min_learning_rate_ratio,
            ),
        )
    mutable_controls = (
        MageFlowMutableControls(config, scheduler, worker_controls)
        if worker_controls is not None
        else None
    )
    required_domains = EXPERT_DOMAINS if domain_schedule is not None else (config.domain,)
    train_rows_by_domain = {
        domain: _domain_rows(config.train_manifest, domain)
        for domain in required_domains
    }
    eval_rows_by_domain = (
        {
            domain: _domain_rows(config.eval_manifest, domain)
            for domain in required_domains
        }
        if config.eval_manifest
        else {domain: [] for domain in required_domains}
    )
    train_rows = train_rows_by_domain[config.domain]
    eval_rows = eval_rows_by_domain[config.domain]
    all_train_rows = [
        row for domain in required_domains for row in train_rows_by_domain[domain]
    ]
    all_eval_rows = [
        row for domain in required_domains for row in eval_rows_by_domain[domain]
    ]
    annotate_domain_token_lengths(model, all_train_rows)
    if all_eval_rows:
        annotate_domain_token_lengths(model, all_eval_rows)
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
        cache_train_rows = (
            load_domain_manifest(Path(config.encoder_cache_coverage_manifest))
            if config.encoder_cache_coverage_manifest
            else all_train_rows
        )
        cache_rows = [*cache_train_rows, *all_eval_rows]
        cache_prompts, _cache_kinds = effective_conditioning_prompts(
            cache_rows,
            caption_dropout=0.0,
            rng=random.Random(config.seed),
        )
        info = PROMPT_TEMPLATE["mage-flow"]
        encoder_cache_report = {
            "mode": config.encoder_cache_mode,
            "root": str(encoder_cache.root),
            "bounded_coverage_manifest": (
                str(
                    Path(config.encoder_cache_coverage_manifest)
                    .expanduser()
                    .resolve()
                )
                if config.encoder_cache_coverage_manifest
                else None
            ),
            "covered_until_step": config.encoder_cache_covered_until_step,
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
    if not all_eval_rows:
        _drop_vae_decoder(model)
    fingerprint = _contract_fingerprint(
        config,
        component_composition_digest=component_digest,
        mutable_control_keys=(
            tuple(
                key
                for key in worker_controls.effective_values
                if key in {"learning_rate", "eval_every", "caption_dropout"}
            )
            if worker_controls is not None
            else ()
        ),
    )

    step = epoch = batch_start = 0
    resumed_domain_positions: dict[str, dict[str, int]] | None = None
    resumed_window_index: int | None = None
    if config.resume_from:
        (
            step,
            epoch,
            batch_start,
            resumed_domain_positions,
            resumed_window_index,
        ) = _load_resume(
            Path(config.resume_from),
            config=config,
            controller=controller,
            transformer=transformer,
            repa=repa,
            optimizer=optimizer,
            scheduler=scheduler,
            switcher=switcher,
            domain=config.domain,
            fingerprint=fingerprint,
            worker_controls=worker_controls,
        )
    if loop_controller is not None:
        loop_controller.refresh_inference_skip_refinements()
    if worker_execution_phases is None:
        if config.compile_vae_encoder:
            model.maybe_compile_vae_encoder()
        compile_report = compile_transformer_blocks(
            transformer,
            enabled=config.compile_transformer_blocks,
            mode=config.compile_transformer_mode,
            dynamic=config.compile_transformer_dynamic,
        )
    else:
        from rwkv_lab.trainvm_adapters.mageflow_phases import (
            run_mageflow_execution_phases,
        )
        from rwkv_lab.trainvm_worker import ExecutionPhase

        compile_report = compile_transformer_blocks(
            transformer,
            enabled=False,
            mode=config.compile_transformer_mode,
            dynamic=config.compile_transformer_dynamic,
        )
        compile_request = worker_execution_phases.request(ExecutionPhase.COMPILE)
        warmup_request = worker_execution_phases.request(ExecutionPhase.WARMUP)
        needs_disposable_workload = any(
            request is not None and request.enabled
            for request in (compile_request, warmup_request)
        )
        phase_batch_rows: list[dict[str, Any]] = []
        phase_images: list[torch.Tensor | None] = []
        phase_prefetched_cache = None
        if needs_disposable_workload:
            phase_epoch = epoch
            phase_batch_start = batch_start
            while not phase_batch_rows:
                phase_batches = _epoch_batches(
                    len(train_rows),
                    batch_size=config.microbatch_size,
                    seed=config.seed,
                    epoch=phase_epoch,
                    accumulation_steps=config.gradient_accumulation_steps,
                    token_counts=[int(row["latent_tokens"]) for row in train_rows],
                    balance_accumulation_window=config.balance_accumulation_window,
                )
                if phase_batch_start < len(phase_batches):
                    phase_batch_rows = [
                        train_rows[index]
                        for index in phase_batches[phase_batch_start]
                    ]
                    break
                phase_epoch += 1
                phase_batch_start = 0
                if phase_epoch > epoch + 1:
                    raise RuntimeError(
                        "MageFlow terminal phase could not resolve a training batch"
                    )
            phase_images = [
                (
                    None
                    if encoder_cache is not None and encoder_cache.has_moments(row)
                    else _load_image_tensor(row).pin_memory()
                )
                for row in phase_batch_rows
            ]
            phase_prefetched_cache = prefetch_frozen_encoder_batch(
                model,
                phase_batch_rows,
                config,
            )
            if phase_prefetched_cache is not None and (
                any(
                    value is None
                    for value in phase_prefetched_cache.text_by_prompt.values()
                )
                or any(
                    value is None for value in phase_prefetched_cache.moments
                )
            ):
                raise RuntimeError(
                    "MageFlow terminal phases require complete frozen-encoder cache coverage"
                )

        def phase_extra_state() -> Mapping[str, Any]:
            return {
                "contract_fingerprint": fingerprint,
                "controls": (
                    worker_controls.checkpoint_state()
                    if worker_controls is not None
                    else {}
                ),
                "data_cursor": {
                    "batch_index": batch_start,
                    "domain": config.domain,
                    "domain_positions": resumed_domain_positions or {},
                    "epoch": epoch,
                    "optimizer_step": step,
                    "window_index": resumed_window_index,
                },
                "frozen_encoders": {
                    "model_id": config.model_id,
                    "model_revision": config.model_revision,
                },
                "resident_expert_optimizer_bank": (
                    optimizer_bank.state_dict()
                    if optimizer_bank is not None
                    else {}
                ),
                "scheduler": scheduler.state_dict(),
            }

        def phase_training_workload() -> None:
            flow = encode_domain_batch(
                model,
                phase_batch_rows,
                phase_images,
                config,
                device,
                prefetched_cache=phase_prefetched_cache,
            )
            if config.immiscible_enabled:
                apply_immiscible_noise_assignment([flow])
            flow_weight_batches, _metrics = effective_flow_loss_weights(
                [flow["timesteps"]],
                weighting=config.flow_loss_weighting,
                gamma=config.flow_min_snr_gamma,
                normalize=config.normalize_flow_loss_weights,
            )
            flow_weights = flow_weight_batches[0]
            with (
                controller.route(config.domain),
                torch.autocast(device_type="cuda", dtype=torch.bfloat16),
            ):
                forwarded = _forward_transformer(
                    transformer,
                    flow,
                    return_hidden_layer=(
                        config.repa_hidden_layer if repa is not None else None
                    ),
                )
                if repa is None:
                    prediction = forwarded
                    repa_loss = None
                else:
                    prediction, hidden = forwarded
                    repa_loss = repa(
                        hidden,
                        flow["repa_target"],
                        flow["image_lens"],
                    )
                loss, _observed = rectified_flow_loss(
                    prediction, flow["velocity"]
                )
                if (
                    config.immiscible_enabled
                    or config.flow_loss_weighting != "uniform"
                ):
                    weighted_loss, _ = weighted_rectified_flow_loss(
                        prediction,
                        flow["velocity"],
                        flow["image_lens"],
                        flow_weights,
                        effective_example_count=int(flow_weights.numel()),
                    )
                else:
                    weighted_loss = loss / config.gradient_accumulation_steps
                if config.velocity_direction_loss_weight > 0:
                    if (
                        config.immiscible_enabled
                        or config.flow_loss_weighting != "uniform"
                    ):
                        weighted_direction_loss, _direction_loss = (
                            weighted_velocity_direction_loss(
                                prediction,
                                flow["velocity"],
                                flow["image_lens"],
                                flow_weights,
                                effective_example_count=int(
                                    flow_weights.numel()
                                ),
                                epsilon=config.velocity_direction_loss_epsilon,
                            )
                        )
                    else:
                        direction_loss = velocity_direction_loss_per_example(
                            prediction,
                            flow["velocity"],
                            flow["image_lens"],
                            epsilon=config.velocity_direction_loss_epsilon,
                        ).mean()
                        weighted_direction_loss = (
                            direction_loss / config.gradient_accumulation_steps
                        )
                else:
                    weighted_direction_loss = loss.new_zeros(())
                optimization_loss = (
                    weighted_loss
                    + config.velocity_direction_loss_weight
                    * weighted_direction_loss
                )
                if loop_controller is not None:
                    optimization_loss = (
                        optimization_loss
                        + loop_controller.config.looping.auxiliary_weight
                        * auxiliary_loop_flow_loss(
                            loop_controller, flow["velocity"]
                        )
                        / config.gradient_accumulation_steps
                        + loop_controller.config.looping.factorization.ponder_weight
                        * learned_loop_ponder_loss(loop_controller)
                        / config.gradient_accumulation_steps
                    )
                repa_contribution = (
                    config.repa_loss_weight
                    * repa_loss
                    / config.gradient_accumulation_steps
                    if repa_loss is not None
                    else None
                )
            _backward_training_objective(
                optimization_loss,
                repa_contribution,
            )

        def compile_phase() -> Mapping[str, Any]:
            if config.compile_vae_encoder:
                model.maybe_compile_vae_encoder()
            return compile_transformer_blocks(
                transformer,
                enabled=True,
                mode=config.compile_transformer_mode,
                dynamic=config.compile_transformer_dynamic,
            )

        trajectory_models = {"transformer": transformer}
        if repa is not None:
            trajectory_models["repa"] = repa
        compile_report = run_mageflow_execution_phases(
            worker_execution_phases,
            trajectory_model=trajectory_models,
            optimizer=optimizer,
            optimizer_step=step,
            disabled_compile_report=compile_report,
            compile_workload=compile_phase,
            training_workload=phase_training_workload,
            extra_state=phase_extra_state,
            synchronize=lambda: torch.cuda.synchronize(device),
        )
    architecture = terminal_architecture_report(controller)
    if not architecture["passed"]:
        raise RuntimeError(f"architecture preflight failed: {architecture}")
    existing_contract_path = output_dir / "run_contract.json"
    if existing_contract_path.is_file():
        existing_contract = json.loads(
            existing_contract_path.read_text(encoding="utf-8")
        )
        existing_fingerprint = existing_contract.get("fingerprint")
        if existing_fingerprint:
            _atomic_json(
                output_dir
                / "run_contracts"
                / f"{existing_fingerprint}.json",
                existing_contract,
            )
    _atomic_json(
        existing_contract_path,
        {
            "schema": RUN_SCHEMA,
            "created_at": _utc_now(),
            "config": asdict(config),
            "architecture": architecture,
            "architecture_migration": lightning_block_report,
            "training_scope": scope,
            "parameter_routing": optimizer_routing.report,
            "training_components": component_evidence,
            "train_examples": len(all_train_rows),
            "eval_examples": len(all_eval_rows),
            "train_examples_by_domain": {
                domain: len(rows) for domain, rows in train_rows_by_domain.items()
            },
            "eval_examples_by_domain": {
                domain: len(rows) for domain, rows in eval_rows_by_domain.items()
            },
            "domain_window_schedule": domain_schedule,
            "resident_experts": 1,
            "inactive_expert_on_gpu": False,
            "fingerprint": fingerprint,
            "runtime_optimizations": {
                "attention_backend": config.attention_backend,
                "activation_checkpointing": activation_checkpoint_report,
                "regional_compile": compile_report,
                "float8": float8_report,
                "optimizer_precision": optimizer.precision_report(),
                "frozen_encoder_cache": encoder_cache_report,
                "cached_encoder_offload": config.offload_cached_encoders,
                "cache_prefetch_batches": config.prefetch_batches,
                "expert_residency": {
                    "policy": (
                        "strict_per_optimizer_step"
                        if config.rapid_expert_alternation
                        else "scheduled_windows"
                    ),
                    "optimizer_state_device": (
                        config.expert_optimizer_state_device
                    ),
                    "host_device_transfer_at_switch": (
                        config.expert_optimizer_state_device != "cuda"
                    ),
                },
                "balanced_accumulation_window": (
                    config.balance_accumulation_window
                ),
                "deferred_training_metric_sync": True,
                "loop_gate_optimization": {
                    "update_multiplier": config.loop_gate_update_multiplier,
                    "max_grad_norm": config.loop_gate_max_grad_norm,
                    "separate_from_main_gradient_clip": (
                        config.loop_gate_update_multiplier != 1.0
                        or config.loop_gate_max_grad_norm
                        != config.max_grad_norm
                    ),
                },
            },
            "training_objectives": {
                "vae_repa": (
                    {
                        **repa.report(),
                        "enabled": True,
                        "loss_weight": config.repa_loss_weight,
                        "hidden_layer": config.repa_hidden_layer,
                        "learning_rate_multiplier": (
                            config.repa_learning_rate_multiplier
                        ),
                    }
                    if repa is not None
                    else {"enabled": False}
                ),
                "immiscible_diffusion": {
                    "enabled": config.immiscible_enabled,
                    "assignment_scope": "gradient_accumulation_window",
                    "assignment_solver": "hungarian",
                    "noise_marginal": "permutation_preserved",
                },
                "flow_loss_weighting": {
                    "method": config.flow_loss_weighting,
                    "gamma": config.flow_min_snr_gamma,
                    "normalize_over_accumulation_window": (
                        config.normalize_flow_loss_weights
                    ),
                    "parameterization": "rectified_flow_velocity",
                },
                "timestep_sampling": {
                    "method": config.timestep_sampling,
                    "shift": config.timestep_shift,
                },
                "velocity_direction_loss": {
                    "enabled": config.velocity_direction_loss_weight > 0,
                    "weight": config.velocity_direction_loss_weight,
                    "epsilon": config.velocity_direction_loss_epsilon,
                    "reduction": "channel_cosine_then_per_image_token_mean",
                    "inherits_flow_weighting": True,
                },
            },
        },
    )

    stop = {"requested": False}

    def handle_stop(_signum, _frame):
        stop["requested"] = True

    signal.signal(signal.SIGINT, handle_stop)
    signal.signal(signal.SIGTERM, handle_stop)
    transformer.train()
    if repa is not None:
        repa.train()
    trainable = [
        parameter for parameter in transformer.parameters() if parameter.requires_grad
    ]
    if repa is not None:
        trainable.extend(
            parameter for parameter in repa.parameters() if parameter.requires_grad
        )
    loop_gate_parameters = (
        [
            parameter
            for parameter in loop_controller.gate_parameters()
            if parameter.requires_grad
        ]
        if loop_controller is not None
        else []
    )
    loop_gate_ids = {id(parameter) for parameter in loop_gate_parameters}
    main_trainable = [
        parameter for parameter in trainable if id(parameter) not in loop_gate_ids
    ]
    optimizer.zero_grad(set_to_none=True)
    accumulation = 0
    loss_values: list[torch.Tensor] = []
    repa_loss_values: list[torch.Tensor] = []
    repa_metric_values: dict[str, list[torch.Tensor]] = {}
    direction_loss_values: list[torch.Tensor] = []
    weighted_direction_loss_values: list[torch.Tensor] = []
    loop_aux_loss_values: list[torch.Tensor] = []
    loop_ponder_loss_values: list[torch.Tensor] = []
    weighted_flow_loss_values: list[torch.Tensor] = []
    optimization_loss_values: list[torch.Tensor] = []
    sample_sum = 0
    pending_flows: list[dict[str, Any]] = []
    cache_prefetch_load_seconds = 0.0
    use_effective_batch_window = (
        config.immiscible_enabled or config.flow_loss_weighting != "uniform"
    )
    _atomic_json(
        output_dir / "status.json",
        {"schema": RUN_SCHEMA, "state": "training", "step": step},
    )
    if all_eval_rows and (not config.resume_from or config.eval_on_resume):
        if mutable_controls is not None and step > 0:
            worker_controls.evaluation(step, mutable_controls.apply)
        if switcher is None:
            initial_evaluation = _run_evaluation(
                pipeline=pipeline,
                transformer=transformer,
                controller=controller,
                repa=repa,
                model=model,
                rows=eval_rows,
                config=config,
                device=device,
                output_dir=output_dir,
                step=step,
            )
        else:
            initial_evaluation = _run_alternating_evaluation(
                pipeline=pipeline,
                transformer=transformer,
                controller=controller,
                switcher=switcher,
                repa=repa,
                model=model,
                rows_by_domain=eval_rows_by_domain,
                config=config,
                device=device,
                output_dir=output_dir,
                step=step,
            )
        if config.resume_from and _best_evaluation_improved(
            output_dir,
            initial_evaluation,
        ):
            _promote_best_checkpoint(
                Path(config.resume_from),
                output_dir,
                step=step,
                loss=float(initial_evaluation["eval/primary_loss"]),
            )
    optimizer_window_started_at = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(device)
    domain_positions = resumed_domain_positions or {
        domain: {"epoch": 0, "batch_start": 0} for domain in required_domains
    }
    domain_positions[config.domain] = {
        "epoch": epoch,
        "batch_start": batch_start,
    }
    window_index = resumed_window_index or 0
    if config.rapid_expert_alternation and config.resume_from:
        next_domain = next(
            domain for domain in EXPERT_DOMAINS if domain != config.domain
        )
        if switcher is None or not switcher.switch(next_domain):
            raise RuntimeError("rapid resume did not exchange the resident expert")
        config.domain = next_domain
        train_rows = train_rows_by_domain[next_domain]
        eval_rows = eval_rows_by_domain[next_domain]
        epoch = int(domain_positions[next_domain]["epoch"])
        batch_start = int(domain_positions[next_domain]["batch_start"])

    rapid_batch_stream = None
    if config.rapid_expert_alternation:
        cache_or_training_end = min(
            config.max_steps,
            (
                config.encoder_cache_covered_until_step
                if config.encoder_cache_covered_until_step is not None
                else config.max_steps
            ),
        )
        rapid_items = _rapid_alternating_batches(
            train_rows_by_domain,
            domain_positions,
            starting_domain=config.domain,
            batch_size=config.microbatch_size,
            accumulation_steps=config.gradient_accumulation_steps,
            seed=config.seed,
            balance_accumulation_window=config.balance_accumulation_window,
            optimizer_steps=cache_or_training_end - step,
        )

        def load_rapid_batch(item):
            domain, item_epoch, batch_index, indices = item
            rows = train_rows_by_domain[domain]
            batch_rows = [rows[index] for index in indices]
            images = [
                (
                    None
                    if encoder_cache is not None
                    and encoder_cache.has_moments(row)
                    else _load_image_tensor(row).pin_memory()
                )
                for row in batch_rows
            ]
            prefetched_cache = prefetch_frozen_encoder_batch(
                model,
                batch_rows,
                config,
            )
            return (
                domain,
                item_epoch,
                batch_index,
                batch_rows,
                images,
                prefetched_cache,
            )

        rapid_batch_stream = iter(
            _prefetched(
                rapid_items,
                load_rapid_batch,
                depth=config.prefetch_batches,
            )
        )

    while step < config.max_steps:
        switched_domain = False
        if rapid_batch_stream is not None:
            if worker_step_profiler is None:
                loaded_batches = (next(rapid_batch_stream),)
            else:
                with worker_step_profiler.input_wait():
                    loaded_batches = (next(rapid_batch_stream),)
        else:
            batches = _epoch_batches(
                len(train_rows),
                batch_size=config.microbatch_size,
                seed=config.seed,
                epoch=epoch,
                accumulation_steps=config.gradient_accumulation_steps,
                token_counts=[int(row["latent_tokens"]) for row in train_rows],
                balance_accumulation_window=config.balance_accumulation_window,
            )
            stream_stop_step = min(
                config.max_steps,
                (
                    config.encoder_cache_covered_until_step
                    if config.encoder_cache_covered_until_step is not None
                    else config.max_steps
                ),
            )
            if domain_schedule is not None:
                stream_stop_step = min(
                    stream_stop_step,
                    int(domain_schedule["windows"][window_index]["end_step"]),
                )
            remaining_microbatches = (
                (stream_stop_step - step) * config.gradient_accumulation_steps
                - accumulation
            )
            batch_stream = itertools.islice(
                enumerate(batches[batch_start:], start=batch_start),
                max(0, remaining_microbatches),
            )
            active_train_rows = train_rows

            def load_batch(
                item,
                rows=active_train_rows,
                item_epoch=epoch,
                domain=config.domain,
            ):
                batch_index, indices = item
                batch_rows = [rows[index] for index in indices]
                images = [
                    (
                        None
                        if encoder_cache is not None
                        and encoder_cache.has_moments(row)
                        else _load_image_tensor(row).pin_memory()
                    )
                    for row in batch_rows
                ]
                prefetched_cache = prefetch_frozen_encoder_batch(
                    model,
                    batch_rows,
                    config,
                )
                return (
                    domain,
                    item_epoch,
                    batch_index,
                    batch_rows,
                    images,
                    prefetched_cache,
                )

            loaded_batches = _prefetched(
                batch_stream,
                load_batch,
                depth=config.prefetch_batches,
            )

        if worker_step_profiler is not None and rapid_batch_stream is None:
            loaded_batches = worker_step_profiler.track_input(loaded_batches)
        for (
            batch_domain,
            batch_epoch,
            batch_index,
            batch_rows,
            images,
            prefetched_cache,
        ) in loaded_batches:
            if batch_domain != config.domain:
                raise RuntimeError(
                    "rapid prefetch domain diverged from resident expert"
                )
            if mutable_controls is not None:
                worker_controls.microbatch(step + 1, mutable_controls.apply)
            epoch = int(batch_epoch)
            flow = encode_domain_batch(
                model,
                batch_rows,
                images,
                config,
                device,
                prefetched_cache=prefetched_cache,
            )
            if prefetched_cache is not None:
                cache_prefetch_load_seconds += prefetched_cache.load_seconds
            immiscible_metrics: dict[str, int | float] | None = None
            if use_effective_batch_window:
                pending_flows.append(flow)
                if len(pending_flows) < config.gradient_accumulation_steps:
                    continue
                active_flows = pending_flows
                pending_flows = []
                if config.immiscible_enabled:
                    immiscible_metrics = apply_immiscible_noise_assignment(
                        active_flows
                    )
            else:
                active_flows = [flow]
            flow_weight_batches, flow_weight_metrics = effective_flow_loss_weights(
                [active_flow["timesteps"] for active_flow in active_flows],
                weighting=config.flow_loss_weighting,
                gamma=config.flow_min_snr_gamma,
                normalize=config.normalize_flow_loss_weights,
            )
            effective_example_count = sum(
                int(weights.numel()) for weights in flow_weight_batches
            )
            timestep_window = torch.cat(
                [active_flow["timesteps"].detach().float() for active_flow in active_flows]
            )

            for active_flow, flow_weights in zip(
                active_flows, flow_weight_batches, strict=True
            ):
                with (
                    controller.route(config.domain),
                    torch.autocast(device_type="cuda", dtype=torch.bfloat16),
                ):
                    forwarded = _forward_transformer(
                        transformer,
                        active_flow,
                        return_hidden_layer=(
                            config.repa_hidden_layer if repa is not None else None
                        ),
                    )
                    if repa is None:
                        prediction = forwarded
                        repa_loss = None
                    else:
                        prediction, hidden = forwarded
                        repa_loss = repa(
                            hidden,
                            active_flow["repa_target"],
                            active_flow["image_lens"],
                        )
                    loss, observed = rectified_flow_loss(
                        prediction, active_flow["velocity"]
                    )
                    if use_effective_batch_window:
                        weighted_loss, _ = (
                            weighted_rectified_flow_loss(
                                prediction,
                                active_flow["velocity"],
                                active_flow["image_lens"],
                                flow_weights,
                                effective_example_count=effective_example_count,
                            )
                        )
                    else:
                        weighted_loss = (
                            loss / config.gradient_accumulation_steps
                        )
                    if config.velocity_direction_loss_weight > 0:
                        if use_effective_batch_window:
                            (
                                weighted_direction_loss,
                                direction_loss,
                            ) = weighted_velocity_direction_loss(
                                prediction,
                                active_flow["velocity"],
                                active_flow["image_lens"],
                                flow_weights,
                                effective_example_count=effective_example_count,
                                epsilon=config.velocity_direction_loss_epsilon,
                            )
                        else:
                            direction_per_example = (
                                velocity_direction_loss_per_example(
                                    prediction,
                                    active_flow["velocity"],
                                    active_flow["image_lens"],
                                    epsilon=(
                                        config.velocity_direction_loss_epsilon
                                    ),
                                )
                            )
                            direction_loss = direction_per_example.mean()
                            weighted_direction_loss = (
                                direction_loss
                                / config.gradient_accumulation_steps
                            )
                    else:
                        direction_loss = active_flow["velocity"].new_zeros(
                            (), dtype=torch.float32
                        )
                        weighted_direction_loss = direction_loss
                    optimization_loss = (
                        weighted_loss
                        + config.velocity_direction_loss_weight
                        * weighted_direction_loss
                    )
                    if loop_controller is None:
                        loop_aux_loss = active_flow["velocity"].new_zeros(
                            (), dtype=torch.float32
                        )
                    else:
                        loop_aux_loss = auxiliary_loop_flow_loss(
                            loop_controller,
                            active_flow["velocity"],
                        )
                        optimization_loss = (
                            optimization_loss
                            + loop_controller.config.looping.auxiliary_weight
                            * loop_aux_loss
                            / config.gradient_accumulation_steps
                        )
                    if loop_controller is None:
                        loop_ponder_loss = active_flow["velocity"].new_zeros(
                            (), dtype=torch.float32
                        )
                    else:
                        loop_ponder_loss = learned_loop_ponder_loss(
                            loop_controller
                        )
                        optimization_loss = (
                            optimization_loss
                            + loop_controller.config.looping.factorization.ponder_weight
                            * loop_ponder_loss
                            / config.gradient_accumulation_steps
                        )
                    if repa_loss is not None:
                        repa_contribution = (
                            config.repa_loss_weight
                            * repa_loss
                            / config.gradient_accumulation_steps
                        )
                    else:
                        repa_contribution = None
                optimization_loss = _backward_training_objective(
                    optimization_loss,
                    repa_contribution,
                )
                accumulation += 1
                loss_values.append(observed.detach())
                direction_loss_values.append(direction_loss.detach())
                weighted_direction_loss_values.append(
                    weighted_direction_loss.detach()
                )
                if repa_loss is not None:
                    repa_loss_values.append(repa_loss.detach())
                    for name, value in repa.last_metrics.items():
                        repa_metric_values.setdefault(name, []).append(value)
                loop_aux_loss_values.append(loop_aux_loss.detach())
                loop_ponder_loss_values.append(loop_ponder_loss.detach())
                weighted_flow_loss_values.append(weighted_loss.detach())
                optimization_loss_values.append(optimization_loss.detach())
                sample_sum += len(active_flow["image_lens"])
            if accumulation < config.gradient_accumulation_steps:
                continue

            if mutable_controls is not None:
                worker_controls.optimizer_step(step + 1, mutable_controls.apply)

            isolate_loop_gates = bool(loop_gate_parameters) and (
                config.loop_gate_update_multiplier != 1.0
                or config.loop_gate_max_grad_norm != config.max_grad_norm
            )
            if isolate_loop_gates:
                if worker_components is not None:
                    grad_norm = worker_components.gradient_clipping(
                        main_trainable
                    )
                    loop_gate_grad_norm = worker_components.gradient_clipping(
                        loop_gate_parameters,
                        slot="loop_gate_gradient_clipping",
                    )
                else:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        main_trainable,
                        config.max_grad_norm,
                    )
                    loop_gate_grad_norm = torch.nn.utils.clip_grad_norm_(
                        loop_gate_parameters,
                        config.loop_gate_max_grad_norm,
                    )
            else:
                grad_norm = (
                    worker_components.gradient_clipping(trainable)
                    if worker_components is not None
                    else torch.nn.utils.clip_grad_norm_(
                        trainable,
                        config.max_grad_norm,
                    )
                )
                loop_gate_grad_norm = grad_norm.new_zeros(())
            optimizer_step_started_at = time.perf_counter()
            if weight_decay_schedule is not None:
                weight_decay_schedule.step(step)
            optimizer.step(
                parameter_update_scales=(
                    {
                        parameter: config.loop_gate_update_multiplier
                        for parameter in loop_gate_parameters
                    }
                    if config.loop_gate_update_multiplier != 1.0
                    else None
                )
            )
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_step_seconds = time.perf_counter() - optimizer_step_started_at
            step += 1
            if worker_step_profiler is not None:
                worker_step_profiler.step(step)
            learning_rates = {
                str(group.get("group_name")): float(group["lr"])
                for group in optimizer.param_groups
            }
            deferred_totals = torch.stack(
                [
                    torch.stack(values).float().sum()
                    for values in (
                        loss_values,
                        (
                            repa_loss_values
                            if repa_loss_values
                            else [loss_values[0].new_zeros(())]
                        ),
                        loop_aux_loss_values,
                        loop_ponder_loss_values,
                        weighted_flow_loss_values,
                        direction_loss_values,
                        weighted_direction_loss_values,
                        optimization_loss_values,
                    )
                ]
            ).cpu()
            (
                loss_sum,
                repa_loss_sum,
                loop_aux_loss_sum,
                loop_ponder_loss_sum,
                weighted_flow_loss_sum,
                direction_loss_sum,
                weighted_direction_loss_sum,
                optimization_loss_sum,
            ) = [float(value) for value in deferred_totals]
            repa_metrics = {
                f"repa/{name}": float(torch.stack(values).float().mean().cpu())
                for name, values in repa_metric_values.items()
                if values
            }
            metrics = {
                "kind": "train",
                "step": step,
                "loss": loss_sum / accumulation,
                "flow_weighted_loss": weighted_flow_loss_sum,
                "velocity_direction_loss": direction_loss_sum / accumulation,
                "velocity_direction_weighted_loss": (
                    weighted_direction_loss_sum
                ),
                "velocity_direction_loss_weight": (
                    config.velocity_direction_loss_weight
                ),
                "optimization_loss": optimization_loss_sum,
                "repa_loss": (
                    repa_loss_sum / accumulation if repa is not None else None
                ),
                "loop_auxiliary_loss": loop_aux_loss_sum / accumulation,
                "loop_ponder_loss": loop_ponder_loss_sum / accumulation,
                **(
                    {
                        f"loop/{key}": value
                        for key, value in loop_controller.performance_metrics().items()
                    }
                    if loop_controller is not None
                    else {}
                ),
                "gnorm": float(grad_norm),
                "loop_gate_gnorm": float(loop_gate_grad_norm),
                "loop_gate_update_multiplier": (
                    config.loop_gate_update_multiplier
                ),
                "expert_lr": learning_rates.get("terminal_expert"),
                "shared_backbone_lr": learning_rates.get("shared_backbone"),
                "repa_lr": learning_rates.get("vae_repa_projection"),
                "domain": config.domain,
                "samples": sample_sum,
                "optimizer_window_seconds": (
                    time.perf_counter() - optimizer_window_started_at
                ),
                "optimizer_step_seconds": optimizer_step_seconds,
                "cache_prefetch_load_seconds": cache_prefetch_load_seconds,
                "examples_per_second": sample_sum
                / max(time.perf_counter() - optimizer_window_started_at, 1e-9),
                "peak_allocated_vram_bytes": int(
                    torch.cuda.max_memory_allocated(device)
                ),
                "epoch": epoch,
                "timestep/sampling": config.timestep_sampling,
                "timestep/mean": float(timestep_window.mean().cpu()),
                "timestep/min": float(timestep_window.min().cpu()),
                "timestep/max": float(timestep_window.max().cpu()),
                **repa_metrics,
            }
            metrics.update(
                {
                    f"flow_weight/{key}": value
                    for key, value in flow_weight_metrics.items()
                }
            )
            if immiscible_metrics is not None:
                metrics.update(
                    {
                        f"immiscible/{key}": value
                        for key, value in immiscible_metrics.items()
                    }
                )
            if worker_observability is not None:
                worker_observability.optimizer_step(step)
                worker_observability.publish_if_declared(
                    "train.loss",
                    loss_sum / accumulation,
                    step=step,
                    sample_weight=sample_sum,
                    labels={"route": config.domain},
                )
                worker_observability.publish_if_declared(
                    "train.images_per_second",
                    metrics["examples_per_second"],
                    step=step,
                    labels={"route": config.domain},
                )
                worker_observability.publish_if_declared(
                    "system.gpu_memory_used",
                    metrics["peak_allocated_vram_bytes"],
                    step=step,
                    labels={"route": config.domain},
                )
            if config.rapid_expert_alternation:
                domain_positions[config.domain] = {
                    "epoch": epoch,
                    "batch_start": batch_index + 1,
                }
            accumulation = 0
            loss_values = []
            repa_loss_values = []
            repa_metric_values = {}
            direction_loss_values = []
            weighted_direction_loss_values = []
            loop_aux_loss_values = []
            loop_ponder_loss_values = []
            weighted_flow_loss_values = []
            optimization_loss_values = []
            sample_sum = 0
            cache_prefetch_load_seconds = 0.0
            optimizer_window_started_at = time.perf_counter()
            torch.cuda.reset_peak_memory_stats(device)
            with (output_dir / "train.jsonl").open("a") as handle:
                handle.write(json.dumps(metrics) + "\n")
            if loop_controller is not None:
                write_mageflow_loop_telemetry(
                    output_dir / "loop_rw.json",
                    loop_controller,
                    step=step,
                    resident_domain=config.domain,
                )
            _atomic_json(
                output_dir / "status.json",
                {
                    "schema": RUN_SCHEMA,
                    "state": "training",
                    "step": step,
                    "domain": config.domain,
                    "domain_window": (
                        None
                        if config.rapid_expert_alternation
                        else window_index if domain_schedule else None
                    ),
                    "expert_switches": switcher.switch_count if switcher else 0,
                },
            )

            evaluation_metrics = None
            if all_eval_rows and step % config.eval_every == 0:
                if mutable_controls is not None:
                    worker_controls.evaluation(step, mutable_controls.apply)
                if switcher is None:
                    evaluation_metrics = _run_evaluation(
                        pipeline=pipeline,
                        transformer=transformer,
                        controller=controller,
                        repa=repa,
                        model=model,
                        rows=eval_rows,
                        config=config,
                        device=device,
                        output_dir=output_dir,
                        step=step,
                    )
                else:
                    evaluation_metrics = _run_alternating_evaluation(
                        pipeline=pipeline,
                        transformer=transformer,
                        controller=controller,
                        switcher=switcher,
                        repa=repa,
                        model=model,
                        rows_by_domain=eval_rows_by_domain,
                        config=config,
                        device=device,
                        output_dir=output_dir,
                        step=step,
                    )
            best_evaluation = (
                evaluation_metrics is not None
                and _best_evaluation_improved(output_dir, evaluation_metrics)
            )
            checkpoint_requested = bool(
                worker_controls is not None
                and worker_controls.checkpoint_boundary_requested
            )
            if (
                step % config.checkpoint_every == 0
                or stop["requested"]
                or best_evaluation
                or checkpoint_requested
            ):
                if mutable_controls is not None:
                    worker_controls.checkpoint(step, mutable_controls.apply)
                checkpoint = _save_checkpoint(
                    controller=controller,
                    transformer=transformer,
                    repa=repa,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    output_dir=output_dir,
                    domain=config.domain,
                    step=step,
                    epoch=epoch,
                    batch_index=batch_index + 1,
                    fingerprint=fingerprint,
                    keep=config.keep_last_n,
                    switcher=switcher,
                    domain_positions=domain_positions,
                    window_index=(
                        None
                        if config.rapid_expert_alternation
                        else window_index if domain_schedule else None
                    ),
                    control_state=(
                        worker_controls.checkpoint_state()
                        if worker_controls is not None
                        else None
                    ),
                )
                if (
                    worker_controls is not None
                    and worker_controls.checkpoint_completion_requested
                ):
                    worker_controls.publish_requested_checkpoint_directory(
                        str(checkpoint),
                        optimizer_step=step,
                        resume_grade="exact",
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
                            "rng_numpy",
                            "rng_python",
                            "rng_torch",
                        ),
                    )
                if best_evaluation:
                    _promote_best_checkpoint(
                        checkpoint,
                        output_dir,
                        step=step,
                        loss=float(evaluation_metrics["eval/primary_loss"]),
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
            if _cache_span_should_stop(
                step=step,
                max_steps=config.max_steps,
                covered_until_step=config.encoder_cache_covered_until_step,
            ):
                if mutable_controls is not None:
                    worker_controls.checkpoint(step, mutable_controls.apply)
                checkpoint = _save_checkpoint(
                    controller=controller,
                    transformer=transformer,
                    repa=repa,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    output_dir=output_dir,
                    domain=config.domain,
                    step=step,
                    epoch=epoch,
                    batch_index=batch_index + 1,
                    fingerprint=fingerprint,
                    keep=config.keep_last_n,
                    switcher=switcher,
                    domain_positions=domain_positions,
                    window_index=(
                        None
                        if config.rapid_expert_alternation
                        else window_index if domain_schedule else None
                    ),
                    control_state=(
                        worker_controls.checkpoint_state()
                        if worker_controls is not None
                        else None
                    ),
                )
                _atomic_json(
                    output_dir / "status.json",
                    {
                        "schema": RUN_SCHEMA,
                        "state": "cache_span_complete",
                        "step": step,
                        "checkpoint": str(checkpoint),
                        "encoder_cache_coverage_manifest": (
                            config.encoder_cache_coverage_manifest
                        ),
                    },
                )
                return
            if step >= config.max_steps:
                break
            if config.rapid_expert_alternation:
                next_domain = next(
                    domain for domain in EXPERT_DOMAINS if domain != config.domain
                )
                if switcher is None or not switcher.switch(next_domain):
                    raise RuntimeError(
                        "rapid alternation did not exchange the resident expert"
                    )
                config.domain = next_domain
                train_rows = train_rows_by_domain[next_domain]
                eval_rows = eval_rows_by_domain[next_domain]
                epoch = int(domain_positions[next_domain]["epoch"])
                batch_start = int(domain_positions[next_domain]["batch_start"])
                with (output_dir / "train.jsonl").open("a") as handle:
                    handle.write(
                        json.dumps(
                            {
                                "kind": "domain_switch",
                                "step": step,
                                "policy": "strict_per_optimizer_step",
                                "domain": next_domain,
                                "expert_switches": switcher.switch_count,
                            }
                        )
                        + "\n"
                    )
                switched_domain = True
                break
            if domain_schedule is not None:
                windows = domain_schedule["windows"]
                active_window = windows[window_index]
                if step == int(active_window["end_step"]):
                    domain_positions[config.domain] = {
                        "epoch": epoch,
                        "batch_start": batch_index + 1,
                    }
                    window_index += 1
                    next_domain = str(windows[window_index]["domain"])
                    if switcher is None or not switcher.switch(next_domain):
                        raise RuntimeError(
                            "domain boundary did not exchange the resident expert"
                        )
                    config.domain = next_domain
                    train_rows = train_rows_by_domain[next_domain]
                    eval_rows = eval_rows_by_domain[next_domain]
                    epoch = int(domain_positions[next_domain]["epoch"])
                    batch_start = int(
                        domain_positions[next_domain]["batch_start"]
                    )
                    switch_metrics = {
                        "kind": "domain_switch",
                        "step": step,
                        "window": window_index,
                        "domain": next_domain,
                        "window_end_step": int(
                            windows[window_index]["end_step"]
                        ),
                        "expert_switches": switcher.switch_count,
                    }
                    with (output_dir / "train.jsonl").open("a") as handle:
                        handle.write(json.dumps(switch_metrics) + "\n")
                    _atomic_json(
                        output_dir / "status.json",
                        {
                            "schema": RUN_SCHEMA,
                            "state": "training",
                            "step": step,
                            "domain": next_domain,
                            "domain_window": window_index,
                            "expert_switches": switcher.switch_count,
                        },
                    )
                    switched_domain = True
                    break
        if switched_domain:
            continue
        epoch += 1
        batch_start = 0
        domain_positions[config.domain] = {
            "epoch": epoch,
            "batch_start": batch_start,
        }

    if all_eval_rows and not unified_evaluation_is_complete(output_dir, step):
        if mutable_controls is not None:
            worker_controls.evaluation(step, mutable_controls.apply)
        if switcher is None:
            _run_evaluation(
                pipeline=pipeline,
                transformer=transformer,
                controller=controller,
                repa=repa,
                model=model,
                rows=eval_rows,
                config=config,
                device=device,
                output_dir=output_dir,
                step=step,
            )
        else:
            _run_alternating_evaluation(
                pipeline=pipeline,
                transformer=transformer,
                controller=controller,
                switcher=switcher,
                repa=repa,
                model=model,
                rows_by_domain=eval_rows_by_domain,
                config=config,
                device=device,
                output_dir=output_dir,
                step=step,
            )
    if mutable_controls is not None:
        worker_controls.checkpoint(step, mutable_controls.apply)
    checkpoint = _save_checkpoint(
        controller=controller,
        transformer=transformer,
        repa=repa,
        optimizer=optimizer,
        scheduler=scheduler,
        output_dir=output_dir,
        domain=config.domain,
        step=step,
        epoch=epoch,
        batch_index=0,
        fingerprint=fingerprint,
        keep=config.keep_last_n,
        switcher=switcher,
        domain_positions=domain_positions,
        window_index=(
            None
            if config.rapid_expert_alternation
            else window_index if domain_schedule else None
        ),
        control_state=(
            worker_controls.checkpoint_state()
            if worker_controls is not None
            else None
        ),
    )
    _export_checkpoint_experts(checkpoint, output_dir)
    save_terminal_shared_backbone(
        transformer,
        controller,
        output_dir / "mageflow-shared-final-third.safetensors",
        dtype=torch.bfloat16,
    )
    if repa is not None:
        save_repa_projection(
            repa,
            output_dir / "mageflow-vae-repa-projection.safetensors",
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
        "export PYTHONUNBUFFERED=1\n"
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
        "train_examples": sum(
            len(_domain_rows(config.train_manifest, domain))
            for domain in (
                EXPERT_DOMAINS
                if config.domain_window_schedule
                else (config.domain,)
            )
        ),
        "eval_examples": (
            sum(
                len(_domain_rows(config.eval_manifest, domain))
                for domain in (
                    EXPERT_DOMAINS
                    if config.domain_window_schedule
                    else (config.domain,)
                )
            )
            if config.eval_manifest
            else 0
        ),
        "resident_experts": 1,
        "alternating_domain_windows": bool(config.domain_window_schedule),
        "training_objectives": {
            "vae_repa": config.repa_enabled,
            "immiscible_diffusion": config.immiscible_enabled,
            "flow_loss_weighting": config.flow_loss_weighting,
            "timestep_sampling": config.timestep_sampling,
            "velocity_direction_loss_weight": (
                config.velocity_direction_loss_weight
            ),
        },
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
    cache_span = commands.add_parser("plan-cache-span")
    cache_span.add_argument("--config", type=Path, required=True)
    cache_span.add_argument("--checkpoint", type=Path, required=True)
    cache_span.add_argument("--optimizer-steps", type=int, required=True)
    cache_span.add_argument("--output", type=Path, required=True)
    prepare_cache = commands.add_parser("prepare-cache-span")
    prepare_cache.add_argument("--config", type=Path, required=True)
    prepare_cache.add_argument("--checkpoint", type=Path, required=True)
    prepare_cache.add_argument("--optimizer-steps", type=int, required=True)
    prepare_cache.add_argument("--output-dir", type=Path, required=True)
    schedule = commands.add_parser("plan-domain-windows")
    schedule.add_argument("--train-manifest", type=Path, required=True)
    schedule.add_argument("--max-steps", type=int, required=True)
    schedule.add_argument("--minimum-steps", type=int, default=500)
    schedule.add_argument("--maximum-steps", type=int, default=1_000)
    schedule.add_argument("--seed", type=int, default=42)
    schedule.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.command == "plan-domain-windows":
        report = domain_window_schedule_report(
            args.train_manifest,
            max_steps=args.max_steps,
            minimum_steps=args.minimum_steps,
            maximum_steps=args.maximum_steps,
            seed=args.seed,
        )
        if args.output:
            _atomic_json(args.output, report)
        print(json.dumps(report, indent=2, sort_keys=True))
        return
    config = TerminalExpertTrainConfig.from_path(args.config)
    if args.command == "prepare":
        print(json.dumps(prepare_run(config, args.run_dir), indent=2))
    elif args.command == "cache-encoders":
        print(json.dumps(cache_frozen_encoders(config), indent=2, sort_keys=True))
    elif args.command == "plan-cache-span":
        print(
            json.dumps(
                write_cache_span_plan(
                    config,
                    args.checkpoint,
                    optimizer_steps=args.optimizer_steps,
                    output=args.output,
                ),
                indent=2,
                sort_keys=True,
            )
        )
    elif args.command == "prepare-cache-span":
        print(
            json.dumps(
                prepare_cache_span(
                    config,
                    args.checkpoint,
                    optimizer_steps=args.optimizer_steps,
                    output_dir=args.output_dir,
                ),
                indent=2,
                sort_keys=True,
            )
        )
    else:
        train(config)


if __name__ == "__main__":
    main()
