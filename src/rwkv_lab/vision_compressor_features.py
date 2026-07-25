"""Frozen canonical multi-teacher features for RWKV caption training."""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Sequence

import torch
from torch import Tensor, nn

from rwkv_lab.vision_teacher_compressor import (
    SCHEMA as TRAINING_SCHEMA,
    CanonicalTeacherCompressor,
    CompressorConfig,
    split_cached_features,
)
from rwkv_lab.moonvit import pool_features


FROZEN_SCHEMA = 1


class CanonicalLatentPrefixProjector(nn.Module):
    """Map a fixed 128x1024 canonical latent into the frozen RWKV width."""

    def __init__(self, rwkv_hidden: int, prefix_tokens: int = 128,
                 latent_width: int = 1024):
        super().__init__()
        self.prefix_tokens = int(prefix_tokens)
        self.latent_width = int(latent_width)
        self.resampler = None
        self.norm = nn.LayerNorm(latent_width)
        # Match the parameter budget of MoonViT's 4x1152 -> RWKV bridge. The
        # canonical input is narrower, so a 2x intermediate keeps adapter
        # capacity from becoming a hidden disadvantage in the A/B test.
        intermediate = rwkv_hidden * 2
        self.project = nn.Sequential(
            nn.Linear(latent_width, intermediate), nn.GELU(),
            nn.Linear(intermediate, rwkv_hidden))
        self.position = nn.Parameter(
            torch.empty(1, prefix_tokens, rwkv_hidden))
        nn.init.normal_(self.position, std=0.02)

    def pool_features(self, item: Tensor) -> Tensor:
        """Retain the ordinary Moon cache-fill contract before compression."""
        return pool_features(item, self.prefix_tokens)

    def forward(self, features: Sequence[Tensor]) -> Tensor:
        if not features:
            raise ValueError("canonical feature batch is empty")
        value = torch.stack(list(features))
        if tuple(value.shape[1:]) != (self.prefix_tokens, self.latent_width):
            raise ValueError(f"invalid canonical feature shape {tuple(value.shape)}")
        return self.project(self.norm(value)) + self.position


class FrozenTeacherCompressor(nn.Module):
    """Inference-only six-teacher compressor with a strict freeze contract."""

    def __init__(self, model: CanonicalTeacherCompressor, source: Path):
        super().__init__()
        self.model = model
        self.source = source.resolve()
        self.requires_grad_(False)
        self.eval()

    @classmethod
    def from_checkpoint(cls, path: str | Path, *, device: str = "cuda",
                        dtype: torch.dtype = torch.bfloat16
                        ) -> "FrozenTeacherCompressor":
        source = Path(path)
        blob = torch.load(source, map_location="cpu", weights_only=False)
        schema = int(blob.get("schema", -1))
        if schema not in (TRAINING_SCHEMA, FROZEN_SCHEMA):
            raise ValueError(f"unsupported compressor checkpoint schema {schema}")
        config = CompressorConfig(**blob["config"])
        if config.tokens != 128 or config.latent_width != 1024:
            raise ValueError(f"unsupported canonical contract {config}")
        model = CanonicalTeacherCompressor(config)
        model.load_state_dict(blob["model"], strict=True)
        model.to(device=device, dtype=dtype).eval().requires_grad_(False)
        result = cls(model, source)
        if any(parameter.requires_grad for parameter in result.parameters()):
            raise RuntimeError("compressor freeze contract failed")
        return result

    @torch.no_grad()
    def forward(self, moon: Sequence[Tensor], fusion: Sequence[Tensor]) -> Tensor:
        if len(moon) != len(fusion) or not moon:
            raise ValueError("paired Moon/fusion feature batch is required")
        moon_batch = torch.stack(list(moon))
        fusion_batch = torch.stack(list(fusion))
        streams = split_cached_features(moon_batch, fusion_batch)
        latent, _ = self.model(streams)
        return latent


class RWKVNativeTeacherCompressor(nn.Module):
    """Frozen teacher compressor with an RWKV-native output contract.

    The canonical 128x1024 representation remains a useful internal bottleneck,
    but it is not the deployment boundary.  This module owns the final reduction
    and width conversion and emits ``[batch, prefix_tokens, rwkv_hidden]``
    directly.  Consequently a caption model can consume its output as ordinary
    RWKV input embeddings without owning or training a vision bridge.
    """

    def __init__(self, compressor: FrozenTeacherCompressor, rwkv_hidden: int,
                 prefix_tokens: int = 64):
        super().__init__()
        latent_tokens = int(compressor.model.config.tokens)
        latent_width = int(compressor.model.config.latent_width)
        if prefix_tokens < 1 or latent_tokens % prefix_tokens:
            raise ValueError(
                "native prefix_tokens must evenly divide compressor tokens")
        self.compressor = compressor
        self.prefix_tokens = int(prefix_tokens)
        self.latent_tokens = latent_tokens
        self.latent_width = latent_width
        self.output_norm = nn.LayerNorm(latent_width)
        self.output_projection = nn.Linear(latent_width, int(rwkv_hidden))
        self.output_position = nn.Parameter(
            torch.empty(1, self.prefix_tokens, int(rwkv_hidden)))
        nn.init.normal_(self.output_position, std=0.02)
        self.compressor.requires_grad_(False).eval()

    def train(self, mode: bool = True) -> "RWKVNativeTeacherCompressor":
        super().train(mode)
        # ``Module.train`` recursively changes children, so restore the frozen
        # compressor's deterministic inference contract after every call.
        self.compressor.eval()
        return self

    def native_parameters(self):
        yield from self.output_norm.parameters()
        yield from self.output_projection.parameters()
        yield self.output_position

    def forward(self, moon: Sequence[Tensor], fusion: Sequence[Tensor]) -> Tensor:
        canonical = self.compressor(moon, fusion)
        batch = canonical.shape[0]
        group = self.latent_tokens // self.prefix_tokens
        pooled = canonical.reshape(
            batch, self.prefix_tokens, group, self.latent_width).mean(dim=2)
        return (self.output_projection(self.output_norm(pooled))
                + self.output_position)


class NativePrefixIdentity(nn.Module):
    """Parameter-free compatibility shim for generic caption-loss plumbing."""

    def __init__(self, prefix_tokens: int, hidden_size: int):
        super().__init__()
        self.prefix_tokens = int(prefix_tokens)
        self.hidden_size = int(hidden_size)

    def forward(self, features: Sequence[Tensor]) -> Tensor:
        if not features:
            raise ValueError("native prefix batch is empty")
        value = torch.stack(list(features))
        if tuple(value.shape[1:]) != (self.prefix_tokens, self.hidden_size):
            raise ValueError(f"invalid native prefix shape {tuple(value.shape)}")
        return value


def frozen_payload(training_checkpoint: str | Path) -> dict:
    """Extract only deployment weights and provenance from a training file."""
    source = Path(training_checkpoint)
    blob = torch.load(source, map_location="cpu", weights_only=False)
    config = CompressorConfig(**blob["config"])
    if int(blob.get("schema", -1)) != TRAINING_SCHEMA:
        raise ValueError("not a compressor training checkpoint")
    model = CanonicalTeacherCompressor(config)
    model.load_state_dict(blob["model"], strict=True)
    return {
        "schema": FROZEN_SCHEMA,
        "config": asdict(config),
        "model": model.state_dict(),
        "source": str(source.resolve()),
        "source_step": int(blob["step"]),
        "frozen": True,
    }
