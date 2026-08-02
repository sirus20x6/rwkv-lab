"""Closed TrainVM configuration for the canonical frozen-vision RWKV trainer."""

from __future__ import annotations

import math
from dataclasses import dataclass


def _path(value: object, label: str, *, optional: bool = False) -> str:
    if optional and value == "":
        return ""
    if not isinstance(value, str) or not value or "\x00" in value:
        raise ValueError(f"{label} must be a nonempty path string")
    return value


def _paths(value: object, label: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(f"{label} must be a nonempty path list")
    return tuple(_path(item, f"{label}[{index}]") for index, item in enumerate(value))


def _integer(value: object, label: str, minimum: int, maximum: int) -> int:
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or not minimum <= value <= maximum
    ):
        raise ValueError(f"{label} must be an integer in [{minimum}, {maximum}]")
    return value


def _finite(value: object, label: str, minimum: float, maximum: float) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or not minimum <= float(value) <= maximum
    ):
        raise ValueError(f"{label} must be finite in [{minimum}, {maximum}]")
    return float(value)


def _integer_tuple(value: object, label: str, *, minimum: int = 0) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(f"{label} must be a nonempty integer list")
    result = tuple(
        _integer(item, f"{label}[{index}]", minimum, 1_000_000)
        for index, item in enumerate(value)
    )
    if len(set(result)) != len(result):
        raise ValueError(f"{label} must not contain duplicate values")
    return result


@dataclass(frozen=True, slots=True)
class VisionFrozenAdapterConfig:
    """The two established MoonViT/compressor A/B arms as declarative data."""

    arm: str
    train_manifests: tuple[str, ...]
    eval_manifests: tuple[str, ...]
    rwkv_checkpoint: str
    moonvit_checkpoint: str
    vocab: str
    moon_cache: str
    compressor_checkpoint: str = ""
    fusion_cache: str = ""
    siglip2_model: str = ""
    dinov2_model: str = ""
    sam_model: str = ""
    steps: int = 2000
    batch_size: int = 4
    min_batch_size: int = 2
    max_batch_size: int = 16
    target_batch_tokens: int = 4096
    max_text_tokens: int = 768
    prefix_tokens: int = 128
    feature_cache_max_bytes: int = 32 * 2**30
    max_input_patches: int = 1024
    moonvit_tap_layers: tuple[int, ...] = (8, 17, 26)
    vision_view_mode: str = "full-quadrants"
    deep_vision_layers: tuple[int, ...] = (8, 16, 24)
    deep_vision_rank: int = 256
    grounding_early_tokens: int = 24
    grounding_early_weight: float = 3.0
    grounding_contrastive_weight: float = 0.1
    grounding_contrastive_dim: int = 512
    grounding_temperature: float = 0.07
    learning_rate: float = 2.0e-4
    loop_learning_rate: float = 1.0e-5
    weight_decay: float = 0.01
    max_gradient_norm: float = 1.0
    loop_count: int = 2
    loop_start_step: int = 1
    loop_ramp_steps: int = 0
    loop_gate_cap: float = 0.25
    loop_index: bool = True
    engram: bool = True
    engram_sites: tuple[int, ...] = (3, 15)
    engram_drow: int = 128
    engram_rows: int = 65_536
    engram_learning_rate: float = 1.0e-3
    engram_warmup_steps: int = 0
    engram_boundary_id: int = 0
    nextlat_weight: float = 0.1
    nextlat_hidden: int = 1024
    sandwich_prompt: bool = True
    manifest_stat_workers: int = 64
    prefetch_next_batch: bool = True
    checkpoint_every: int = 100
    eval_every: int = 250
    eval_examples: int = 64
    eval_samples: int = 8
    eval_ocr_samples: int = 2
    eval_structured_samples: int = 2
    eval_sample_max_new: int = 256
    eval_sample_exclude_sources: str = "joy,civitai,nsfw,porn,manga,pose_vr,grid"
    log_every: int = 1
    profile_steps: int = 10
    require_fused_ce: bool = True
    seed: int = 20260714

    def __post_init__(self) -> None:
        if self.arm not in {"moonvit", "compressor"}:
            raise ValueError("arm must be moonvit or compressor")
        object.__setattr__(
            self, "train_manifests", _paths(self.train_manifests, "train_manifests")
        )
        object.__setattr__(
            self, "eval_manifests", _paths(self.eval_manifests, "eval_manifests")
        )
        for label in (
            "rwkv_checkpoint",
            "moonvit_checkpoint",
            "vocab",
            "moon_cache",
        ):
            _path(getattr(self, label), label)
        compressor_paths = (
            "compressor_checkpoint",
            "fusion_cache",
            "siglip2_model",
            "dinov2_model",
            "sam_model",
        )
        for label in compressor_paths:
            _path(
                getattr(self, label),
                label,
                optional=self.arm == "moonvit",
            )
        if self.arm == "moonvit" and any(
            getattr(self, name) for name in compressor_paths
        ):
            raise ValueError("moonvit arm must not declare compressor-only inputs")

        for label, minimum, maximum in (
            ("steps", 1, 1_000_000_000),
            ("batch_size", 1, 1_048_576),
            ("min_batch_size", 1, 1_048_576),
            ("max_batch_size", 1, 1_048_576),
            ("target_batch_tokens", 1, 1 << 40),
            ("max_text_tokens", 1, 1_000_000),
            ("prefix_tokens", 1, 1_000_000),
            ("feature_cache_max_bytes", 0, 1 << 63),
            ("max_input_patches", 1, 1_000_000),
            ("deep_vision_rank", 1, 1_000_000),
            ("grounding_early_tokens", 0, 1_000_000),
            ("grounding_contrastive_dim", 1, 1_000_000),
            ("loop_count", 1, 1_000_000),
            ("loop_start_step", 0, 1_000_000_000),
            ("loop_ramp_steps", 0, 1_000_000_000),
            ("engram_drow", 1, 1_000_000),
            ("engram_rows", 1, 1 << 40),
            ("engram_warmup_steps", 0, 1_000_000_000),
            ("engram_boundary_id", 0, 1_000_000_000),
            ("nextlat_hidden", 1, 1_000_000),
            ("manifest_stat_workers", 1, 4096),
            ("checkpoint_every", 1, self.steps),
            ("eval_every", 1, self.steps),
            ("eval_examples", 1, 1_000_000),
            ("eval_samples", 0, 1_000_000),
            ("eval_ocr_samples", 0, 1_000_000),
            ("eval_structured_samples", 0, 1_000_000),
            ("eval_sample_max_new", 0, 1_000_000),
            ("log_every", 1, self.steps),
            ("profile_steps", 0, 1_000_000),
            ("seed", 0, (1 << 63) - 1),
        ):
            _integer(getattr(self, label), label, minimum, maximum)
        if not self.min_batch_size <= self.batch_size <= self.max_batch_size:
            raise ValueError(
                "min_batch_size <= batch_size <= max_batch_size is required"
            )
        if self.eval_ocr_samples + self.eval_structured_samples > self.eval_samples:
            raise ValueError("reserved eval categories exceed eval_samples")
        for label, minimum, maximum in (
            ("grounding_early_weight", 0.0, 1.0e6),
            ("grounding_contrastive_weight", 0.0, 1.0e6),
            ("grounding_temperature", 1.0e-12, 1.0e6),
            ("learning_rate", 1.0e-12, 1.0),
            ("loop_learning_rate", 0.0, 1.0),
            ("weight_decay", 0.0, 10.0),
            ("max_gradient_norm", 1.0e-12, 1.0e12),
            ("loop_gate_cap", 0.0, 1.0),
            ("engram_learning_rate", 0.0, 1.0),
            ("nextlat_weight", 0.0, 1.0e6),
        ):
            _finite(getattr(self, label), label, minimum, maximum)
        for label in (
            "loop_index",
            "engram",
            "sandwich_prompt",
            "prefetch_next_batch",
            "require_fused_ce",
        ):
            if not isinstance(getattr(self, label), bool):
                raise TypeError(f"{label} must be boolean")
        for label in ("moonvit_tap_layers", "deep_vision_layers", "engram_sites"):
            object.__setattr__(
                self,
                label,
                _integer_tuple(getattr(self, label), label),
            )
        if self.vision_view_mode not in {"full", "full-quadrants"}:
            raise ValueError("vision_view_mode must be full or full-quadrants")
        if (
            not isinstance(self.eval_sample_exclude_sources, str)
            or "\x00" in self.eval_sample_exclude_sources
        ):
            raise ValueError("eval_sample_exclude_sources must be a string")


__all__ = ["VisionFrozenAdapterConfig"]
