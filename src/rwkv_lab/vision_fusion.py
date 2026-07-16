"""Frozen multi-tower image prefix encoder for native RWKV-7 checkpoints.

The default towers deliberately mirror the VisualRWKV direction while using the
newer SigLIP2 semantic encoder:

* SigLIP2: image--text semantics and captions;
* DINOv2: dense, non-language visual features;
* SAM ViT-B: high-resolution object/edge structure.

Each tower is pooled *before* fusion.  Feeding every native patch token to a
10k-context RWKV would make training needlessly expensive (SAM alone produces
4096 image tokens), so the default prefix is a fixed 256 tokens.  The towers
are frozen by default; initially train only the projections and the downstream
RWKV adapters/LM.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class VisionTowerConfig:
    siglip2: str = "google/siglip2-base-patch16-512"
    dinov2: str = "facebook/dinov2-base"
    sam: str = "facebook/sam-vit-base"
    siglip_width: int = 768
    siglip_tokens: int = 64
    dinov2_tokens: int = 96
    sam_tokens: int = 96

    @property
    def token_budget(self) -> int:
        return self.siglip_tokens + self.dinov2_tokens + self.sam_tokens

    def fingerprint(self) -> str:
        """Fingerprint local tower identities without hashing multi-GB weights."""
        parts = []
        for value in (self.siglip2, self.dinov2, self.sam):
            path = Path(value).expanduser()
            if path.exists():
                path = path.resolve()
                files = sorted(candidate for candidate in path.rglob("*")
                               if candidate.is_file()) if path.is_dir() else [path]
                identity = []
                for candidate in files:
                    stat = candidate.stat()
                    identity.append((str(candidate.relative_to(path) if path.is_dir()
                                         else candidate.name), stat.st_size,
                                     stat.st_mtime_ns))
                parts.append((str(path), identity))
            else:
                parts.append((str(value), "remote"))
        parts.append(("siglip_width", int(self.siglip_width)))
        return hashlib.sha256(repr(parts).encode()).hexdigest()


def pool_tokens(tokens: Tensor, count: int) -> Tensor:
    """Average-pool ``[batch, sequence, channels]`` tokens to ``count``."""
    if tokens.ndim != 3:
        raise ValueError(f"expected [batch, sequence, channels], got {tuple(tokens.shape)}")
    if count < 1:
        raise ValueError("pooled token count must be positive")
    return F.adaptive_avg_pool1d(tokens.transpose(1, 2), count).transpose(1, 2)


class FusedVisionPrefix(nn.Module):
    """Produce a fixed-length RWKV input prefix from the three frozen towers.

    ``load_pretrained`` is intentionally explicit, keeping importing this module
    cheap for CPU-only tests and avoiding a network request at import time.
    """

    def __init__(self, rwkv_hidden_size: int = 2048, config: VisionTowerConfig | None = None):
        super().__init__()
        self.config = config or VisionTowerConfig()
        if self.config.siglip_width < 1:
            raise ValueError("SigLIP2 feature width must be positive")
        self.rwkv_hidden_size = rwkv_hidden_size
        # DINOv2-B emits 768 channels and SAM ViT-B's image encoder neck emits
        # 256. SigLIP2 is configurable: Base emits 768 while So400m emits 1152.
        tower_widths = (self.config.siglip_width, 768, 256)
        self.projections = nn.ModuleList(nn.Linear(width, rwkv_hidden_size, bias=False)
                                         for width in tower_widths)
        self.tower_type = nn.Parameter(torch.empty(3, 1, rwkv_hidden_size))
        nn.init.normal_(self.tower_type, std=0.02)
        self.siglip_processor = self.dinov2_processor = self.sam_processor = None
        self.siglip = self.dinov2 = self.sam = None

    def load_pretrained(self, *, device: torch.device | str = "cpu", dtype: torch.dtype | None = None) -> "FusedVisionPrefix":
        """Load and freeze official Hugging Face tower weights."""
        from transformers import (AutoImageProcessor, AutoModel, AutoProcessor,
                                  SamModel, SamProcessor, SiglipVisionModel)

        kwargs = {"torch_dtype": dtype} if dtype is not None else {}
        self.siglip_processor = AutoProcessor.from_pretrained(self.config.siglip2)
        # Load only the vision submodule. AutoModel materializes the unused
        # several-hundred-million-parameter text tower before we can discard it,
        # causing avoidable host-memory peaks for So400m/g-opt checkpoints.
        self.siglip = SiglipVisionModel.from_pretrained(
            self.config.siglip2, **kwargs)
        self.dinov2_processor = AutoImageProcessor.from_pretrained(self.config.dinov2)
        self.dinov2 = AutoModel.from_pretrained(self.config.dinov2, **kwargs)
        self.sam_processor = SamProcessor.from_pretrained(self.config.sam)
        self.sam = SamModel.from_pretrained(self.config.sam, **kwargs)
        for tower in (self.siglip, self.dinov2, self.sam):
            tower.requires_grad_(False).eval().to(device)
        return self.to(device)

    def _require_loaded(self) -> None:
        if any(tower is None for tower in (self.siglip, self.dinov2, self.sam)):
            raise RuntimeError("call load_pretrained() before encoding images")

    @torch.no_grad()
    def extract_tower_tokens(self, images: Sequence[object], *, device: torch.device | str) -> tuple[Tensor, Tensor, Tensor]:
        """Extract unprojected tokens; images are PIL images accepted by HF processors."""
        self._require_loaded()
        siglip_inputs = self.siglip_processor(images=images, return_tensors="pt").to(device)
        dino_inputs = self.dinov2_processor(images=images, return_tensors="pt").to(device)
        sam_inputs = self.sam_processor(images=images, return_tensors="pt").to(device)
        siglip_dtype = next(self.siglip.parameters()).dtype
        dino_dtype = next(self.dinov2.parameters()).dtype
        sam_dtype = next(self.sam.parameters()).dtype
        siglip = self.siglip(
            pixel_values=siglip_inputs.pixel_values.to(siglip_dtype)).last_hidden_state
        dino = self.dinov2(
            pixel_values=dino_inputs.pixel_values.to(dino_dtype)).last_hidden_state
        # [B, 256, H, W] -> spatial-token sequence; no prompts/mask decoder needed.
        sam = self.sam.get_image_embeddings(
            sam_inputs.pixel_values.to(sam_dtype)).flatten(2).transpose(1, 2)
        return siglip, dino, sam

    def forward(self, images: Sequence[object], *, device: torch.device | str | None = None) -> Tensor:
        self._require_loaded()
        device = device or next(self.projections.parameters()).device
        raw = self.extract_tower_tokens(images, device=device)
        counts = (self.config.siglip_tokens, self.config.dinov2_tokens, self.config.sam_tokens)
        # Frozen towers may be loaded in bf16 while the trainable projections
        # deliberately retain fp32 master weights. Do not rely on an ambient
        # autocast context merely to make the public forward method type-correct.
        prefixes = []
        for tokens, count, projection, type_embedding in zip(
                raw, counts, self.projections, self.tower_type):
            projected = projection(
                pool_tokens(tokens, count).to(projection.weight.dtype))
            prefixes.append(projected + type_embedding.to(projected.dtype))
        return torch.cat(prefixes, dim=1)


class AlignedFrozenVisionFeatures(nn.Module):
    """Frozen SigLIP2/DINOv2/SAM extractor aligned to one token grid.

    Unlike :class:`FusedVisionPrefix`, this returns unprojected features so the
    result can be cached independently of every trainable adapter update.
    """

    def __init__(self, config: VisionTowerConfig | None = None):
        super().__init__()
        self.config = config or VisionTowerConfig()
        if self.config.siglip_width < 1:
            raise ValueError("SigLIP2 feature width must be positive")
        self.extractor = FusedVisionPrefix(1, self.config)
        # The projections/type embeddings belong to the older prefix API and
        # are deliberately not part of this frozen raw-feature tower.
        del self.extractor.projections
        del self.extractor.tower_type
        self.cache_fingerprint = self.config.fingerprint()

    @property
    def width(self) -> int:
        return int(self.config.siglip_width) + 768 + 256

    @property
    def loaded(self) -> bool:
        return all(getattr(self.extractor, name) is not None
                   for name in ("siglip", "dinov2", "sam"))

    def load_pretrained(self, *, device: torch.device | str = "cpu",
                        dtype: torch.dtype | None = None
                        ) -> "AlignedFrozenVisionFeatures":
        # Reuse the carefully matched processors/model classes, temporarily
        # restoring the two attributes expected by FusedVisionPrefix.to().
        self.extractor.projections = nn.ModuleList()
        self.extractor.tower_type = nn.Parameter(
            torch.empty(0), requires_grad=False)
        self.extractor.load_pretrained(device=device, dtype=dtype)
        del self.extractor.projections
        del self.extractor.tower_type
        self.requires_grad_(False).eval()
        return self

    @torch.no_grad()
    def forward(self, images: Sequence[object], *, tokens: int,
                device: torch.device | str | None = None) -> Tensor:
        if tokens < 1:
            raise ValueError("aligned vision token count must be positive")
        if not self.loaded:
            raise RuntimeError("call load_pretrained() before extracting fusion features")
        device = device or next(self.extractor.siglip.parameters()).device
        raw = self.extractor.extract_tower_tokens(images, device=device)
        pooled = [pool_tokens(value, tokens) for value in raw]
        output = torch.cat(pooled, dim=-1)
        if output.shape[-1] != self.width:
            raise RuntimeError(
                f"configured fusion width {self.width} does not match tower output "
                f"{output.shape[-1]}")
        return output


class VisionFusionResidual(nn.Module):
    """Zero-init trainable residual from aligned frozen features into RWKV."""

    def __init__(self, rwkv_hidden_size: int, *, rank: int = 512,
                 source_width: int = 768 + 768 + 256):
        super().__init__()
        if rank < 1:
            raise ValueError("vision fusion rank must be positive")
        self.norm = nn.LayerNorm(source_width)
        self.down = nn.Linear(source_width, rank, bias=False)
        self.act = nn.GELU()
        self.up = nn.Linear(rank, rwkv_hidden_size, bias=False)
        nn.init.zeros_(self.up.weight)

    def forward(self, features: Tensor | Sequence[Tensor]) -> Tensor:
        value = (torch.stack(list(features))
                 if not torch.is_tensor(features) else features)
        if value.ndim != 3 or value.shape[-1] != self.norm.normalized_shape[0]:
            raise ValueError(f"invalid aligned fusion features: {tuple(value.shape)}")
        return self.up(self.act(self.down(self.norm(value))))


def valid_aligned_feature(item: object, tokens: int,
                          width: int = 768 + 768 + 256) -> bool:
    return (torch.is_tensor(item)
            and tuple(item.shape) == (int(tokens), int(width))
            and item.dtype in (torch.float16, torch.bfloat16, torch.float32)
            and bool(torch.isfinite(item).all()))


def valid_sam_dense_feature(item: object) -> bool:
    """Validate SAM ViT-B's native dense image-encoder feature grid."""
    return (torch.is_tensor(item)
            and tuple(item.shape) == (256, 64, 64)
            and item.dtype in (torch.float16, torch.bfloat16, torch.float32)
            and bool(torch.isfinite(item).all()))


def sam_tower_fingerprint(model: str | Path) -> str:
    """Fingerprint one local/remote SAM tower without hashing its weights."""
    path = Path(model).expanduser()
    if not path.exists():
        identity: object = (str(model), "remote")
    else:
        path = path.resolve()
        files = (sorted(candidate for candidate in path.rglob("*")
                        if candidate.is_file())
                 if path.is_dir() else [path])
        identity = (str(path), [
            (str(candidate.relative_to(path) if path.is_dir() else candidate.name),
             candidate.stat().st_size, candidate.stat().st_mtime_ns)
            for candidate in files
        ])
    return hashlib.sha256(repr(identity).encode()).hexdigest()


def sam_dense_cache_key(image_path: str | Path, *, tower_fingerprint: str,
                        source_size: int | None = None,
                        source_mtime_ns: int | None = None) -> str:
    image = Path(image_path).resolve()
    if source_size is None or source_mtime_ns is None:
        stat = image.stat()
        source_size, source_mtime_ns = stat.st_size, stat.st_mtime_ns
    value = (f"sam-dense-grid-v1|{image}|size={source_size}"
             f"|mtime={source_mtime_ns}|shape=256x64x64"
             f"|tower={tower_fingerprint}")
    return hashlib.sha256(value.encode()).hexdigest() + ".pt"


def aligned_feature_cache_key(image_path: str | Path, *, tokens: int,
                              tower_fingerprint: str,
                              source_size: int | None = None,
                              source_mtime_ns: int | None = None) -> str:
    image = Path(image_path).resolve()
    if source_size is None or source_mtime_ns is None:
        stat = image.stat()
        source_size, source_mtime_ns = stat.st_size, stat.st_mtime_ns
    value = (f"aligned-siglip2-dinov2-sam-v1|{image}|size={source_size}"
             f"|mtime={source_mtime_ns}|tokens={int(tokens)}"
             f"|towers={tower_fingerprint}")
    return hashlib.sha256(value.encode()).hexdigest() + ".pt"
