"""Typed configuration for declarative MageFlow encoder-cache operations."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path


class MageFlowCacheConfigError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class MageFlowCachePlanConfig:
    trainer: Mapping[str, object]
    optimizer_steps: int

    def __post_init__(self) -> None:
        if not isinstance(self.trainer, Mapping) or not self.trainer:
            raise MageFlowCacheConfigError(
                "cache plan requires a nonempty terminal trainer configuration"
            )
        if (
            isinstance(self.optimizer_steps, bool)
            or not isinstance(self.optimizer_steps, int)
            or self.optimizer_steps < 1
        ):
            raise MageFlowCacheConfigError(
                "cache plan optimizer_steps must be a positive integer"
            )
        try:
            encoded = json.dumps(
                dict(self.trainer),
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
        except (TypeError, ValueError) as error:
            raise MageFlowCacheConfigError(
                "cache plan trainer configuration is not finite JSON"
            ) from error
        if not encoded or len(encoded) > 36 * 1024:
            raise MageFlowCacheConfigError(
                "cache plan trainer configuration exceeds its byte bound"
            )


@dataclass(frozen=True, slots=True)
class MageFlowEncoderCacheConfig:
    train_manifest: str
    eval_manifest: str | None
    encoder_cache_dir: str
    model_id: str
    model_revision: str
    model_path: str | None
    attention_backend: str
    vae_sample_posterior: bool
    caption_dropout: float
    seed: int
    timestep_sampling: str
    timestep_shift: float
    repa_enabled: bool
    repa_use_posterior_mean: bool
    encoder_cache_mode: str = "read_write"

    def validate(self) -> None:
        if self.encoder_cache_mode != "read_write":
            raise MageFlowCacheConfigError(
                "declarative cache build requires read_write staging"
            )
        for label, value in (
            ("train_manifest", self.train_manifest),
            ("encoder_cache_dir", self.encoder_cache_dir),
        ):
            if not isinstance(value, str) or not Path(value).is_absolute():
                raise MageFlowCacheConfigError(f"{label} must be absolute")
        if (
            self.eval_manifest is not None
            and not Path(self.eval_manifest).is_absolute()
        ):
            raise MageFlowCacheConfigError("eval_manifest must be absolute")
        if self.model_path is not None and not Path(self.model_path).is_absolute():
            raise MageFlowCacheConfigError("model_path must be absolute")
        if not self.model_id or not self.model_revision:
            raise MageFlowCacheConfigError("cache model identity is incomplete")
        if self.attention_backend not in {"flash2", "flash4"}:
            raise MageFlowCacheConfigError("cache attention backend is unsupported")
        if self.timestep_sampling not in {
            "uniform",
            "logit_normal",
            "shifted_uniform",
        }:
            raise MageFlowCacheConfigError("cache timestep sampling is unsupported")
        if (
            isinstance(self.timestep_shift, bool)
            or not isinstance(self.timestep_shift, (int, float))
            or not math.isfinite(float(self.timestep_shift))
            or self.timestep_shift <= 0
        ):
            raise MageFlowCacheConfigError("cache timestep shift is invalid")
        if (
            isinstance(self.seed, bool)
            or not isinstance(self.seed, int)
            or not 0 <= self.seed < 1 << 63
        ):
            raise MageFlowCacheConfigError("cache seed is outside int64")
        if (
            isinstance(self.caption_dropout, bool)
            or not isinstance(self.caption_dropout, (int, float))
            or not 0 <= float(self.caption_dropout) < 1
        ):
            raise MageFlowCacheConfigError("cache caption_dropout is invalid")


def finite_cache_receipt(value: object) -> Mapping[str, object]:
    """Validate the bounded result returned by the tensor cache builder."""

    if not isinstance(value, Mapping):
        raise MageFlowCacheConfigError("cache builder omitted its result")
    try:
        encoded = json.dumps(
            dict(value),
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise MageFlowCacheConfigError(
            "cache builder result is not finite JSON"
        ) from error
    if not encoded or len(encoded) > 64 * 1024:
        raise MageFlowCacheConfigError("cache builder result exceeds its byte bound")
    for item in value.values():
        if isinstance(item, float) and not math.isfinite(item):
            raise MageFlowCacheConfigError("cache builder result is not finite")
    return value


__all__ = [
    "MageFlowCacheConfigError",
    "MageFlowCachePlanConfig",
    "MageFlowEncoderCacheConfig",
    "finite_cache_receipt",
]
