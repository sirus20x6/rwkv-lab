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
import math
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
    # Crop SAM's zero-padded canvas and pool it on a 2D lattice instead of the
    # flattened row-major sequence. Off by default only because the shipped
    # frozen teacher compressor was trained against the uncropped features;
    # enable it for any new cache. See `sam_cropped_tokens`.
    sam_crop_padding: bool = False

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
        if self.sam_crop_padding:
            # Only extend the fingerprint when enabled, so existing caches built
            # with the legacy geometry keep their keys.
            parts.append(("sam_crop_padding", True))
        return hashlib.sha256(repr(parts).encode()).hexdigest()


def pool_tokens(tokens: Tensor, count: int) -> Tensor:
    """Average-pool ``[batch, sequence, channels]`` tokens to ``count``.

    This pools the *flattened* row-major sequence, so each output token averages
    a run of consecutive grid cells — a horizontal sliver, not a region. That is
    acceptable for towers whose value is global semantics, and wrong for a tower
    whose value is spatial structure; see :func:`pool_grid_tokens`.
    """
    if tokens.ndim != 3:
        raise ValueError(f"expected [batch, sequence, channels], got {tuple(tokens.shape)}")
    if count < 1:
        raise ValueError("pooled token count must be positive")
    return F.adaptive_avg_pool1d(tokens.transpose(1, 2), count).transpose(1, 2)


def factor_grid(count: int) -> tuple[int, int]:
    """Most-square ``(rows, cols)`` with ``rows * cols == count`` (cols >= rows)."""
    if count < 1:
        raise ValueError("pooled token count must be positive")
    rows = max((value for value in range(1, int(count ** 0.5) + 1)
                if count % value == 0), default=1)
    return rows, count // rows


def pool_grid_tokens(grid: Tensor, count: int) -> Tensor:
    """Average-pool a ``[batch, channels, height, width]`` map to ``count`` tokens.

    Unlike :func:`pool_tokens`, each output token is a genuine rectangular region
    of the source map, so the two spatial axes survive pooling. Output order is
    row-major over the pooled lattice.
    """
    if grid.ndim != 4:
        raise ValueError(
            f"expected [batch, channels, height, width], got {tuple(grid.shape)}")
    rows, cols = factor_grid(count)
    pooled = F.adaptive_avg_pool2d(grid.float(), (rows, cols))
    return pooled.flatten(2).transpose(1, 2).to(grid.dtype)


def sam_live_cells(reshaped_height: int, reshaped_width: int, *,
                   patch: int = 16, grid: int = 64) -> tuple[int, int]:
    """Grid cells of a SAM embedding that cover real pixels rather than padding.

    ``SamImageProcessor`` resizes the longest edge to 1024 and then zero-pads to
    a 1024x1024 canvas, so a non-square image leaves a large block of the 64x64
    embedding describing padding only. ``reshaped_input_sizes`` from the
    processor gives the live extent on that canvas.
    """
    rows = max(1, min(grid, math.ceil(int(reshaped_height) / patch)))
    cols = max(1, min(grid, math.ceil(int(reshaped_width) / patch)))
    return rows, cols


def _reshaped_input_sizes(inputs: object) -> Tensor | None:
    """Read ``reshaped_input_sizes`` from a processor result of either shape."""
    value = getattr(inputs, "reshaped_input_sizes", None)
    if value is None:
        try:
            value = inputs["reshaped_input_sizes"]  # type: ignore[index]
        except (TypeError, KeyError, IndexError):
            return None
    return value


def sam_cropped_tokens(dense: Tensor, reshaped_sizes: Tensor, count: int) -> Tensor:
    """Pool SAM embeddings over live pixels only, preserving both spatial axes.

    ``dense`` is the ``[batch, 256, 64, 64]`` image-encoder output. Each image is
    cropped to its own live extent before pooling, so no output token averages
    padding and every token maps to a fixed fraction of the real image.
    """
    if dense.ndim != 4:
        raise ValueError(
            f"expected [batch, 256, height, width], got {tuple(dense.shape)}")
    sizes = reshaped_sizes.detach().cpu().reshape(-1, 2).tolist()
    if len(sizes) != dense.shape[0]:
        raise ValueError("one reshaped input size is required per SAM image")
    pooled = []
    for index, (height, width) in enumerate(sizes):
        rows, cols = sam_live_cells(height, width, grid=dense.shape[-2])
        crop = dense[index : index + 1, :, :rows, :cols]
        pooled.append(pool_grid_tokens(crop, count))
    return torch.cat(pooled, dim=0)


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
    def extract_tower_features(self, images: Sequence[object], *,
                               device: torch.device | str
                               ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Extract unprojected tower features plus SAM's live canvas extent.

        SAM is returned as its native ``[B, 256, 64, 64]`` map so a caller can
        crop the padded canvas before pooling; ``reshaped_input_sizes`` gives the
        live extent per image.
        """
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
        # [B, 256, H, W]; no prompts/mask decoder needed.
        sam = self.sam.get_image_embeddings(sam_inputs.pixel_values.to(sam_dtype))
        sizes = _reshaped_input_sizes(sam_inputs)
        if sizes is None:
            raise ValueError(
                "SAM processor output is missing 'reshaped_input_sizes'; it is "
                "required to distinguish live pixels from the padded canvas")
        return siglip, dino, sam, sizes

    @torch.no_grad()
    def extract_tower_tokens(self, images: Sequence[object], *, device: torch.device | str) -> tuple[Tensor, Tensor, Tensor]:
        """Extract unprojected tokens; images are PIL images accepted by HF processors."""
        siglip, dino, sam, _ = self.extract_tower_features(images, device=device)
        return siglip, dino, sam.flatten(2).transpose(1, 2)

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
    """Frozen SigLIP2/DINOv2/SAM extractor pooled to one common token count.

    Unlike :class:`FusedVisionPrefix`, this returns unprojected features so the
    result can be cached independently of every trainable adapter update.

    "Aligned" here means a shared token *count*, not a shared spatial frame. The
    three processors disagree about geometry and this class does not reconcile
    them: SigLIP2 resizes to a square (squashing aspect), DINOv2 resizes the
    short edge then center-crops 224 (discarding the edges of a non-square
    image), and SAM resizes the long edge then pads to a square. Token ``i`` of
    each tower therefore describes a different region of the source image, so
    the channel-wise concatenation below is a bag of three views rather than a
    registered feature stack. Downstream adapters have to learn around that.
    ``VisionTowerConfig.sam_crop_padding`` fixes the SAM half of the problem;
    reconciling SigLIP2 and DINOv2 would change towers that already train, so it
    is deliberately left as a separate decision.
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
        siglip, dino, sam, sam_sizes = self.extractor.extract_tower_features(
            images, device=device)
        pooled = [pool_tokens(siglip, tokens), pool_tokens(dino, tokens)]
        if self.config.sam_crop_padding:
            pooled.append(sam_cropped_tokens(sam, sam_sizes, tokens))
        else:
            pooled.append(pool_tokens(sam.flatten(2).transpose(1, 2), tokens))
        output = torch.cat(pooled, dim=-1)
        if output.shape[-1] != self.width:
            raise RuntimeError(
                f"configured fusion width {self.width} does not match tower output "
                f"{output.shape[-1]}")
        return output


class SamAlignedFrozenFeatures(nn.Module):
    """Frozen SAM image-encoder features for a fixed global RADIO residual.

    RADIO's tiled prefix is variable-length.  SAM is therefore pooled to one
    fixed global span (128 tokens by default) and fused into the leading RADIO
    tokens; detail-tile tokens remain untouched.  This keeps cache size bounded
    and avoids pretending that independently pooled SAM tokens are aligned to
    RADIO's per-tile sequence.

    Redundancy warning: the only backend this is wired to, C-RADIOv4-1D-H, is
    itself an agglomerative distillation of SigLIP2-g, DINOv3-7B and **SAM3**,
    with SAM3 supplying dense structure (``use_summary: false``). Attaching SAM
    ViT-B here therefore layers a 2023 91M-parameter encoder onto a backbone that
    already absorbed a much stronger successor. Keep it as an ablation.
    """

    width = 256

    def __init__(self, model: str | Path = "facebook/sam-vit-base", *,
                 tokens: int = 128):
        super().__init__()
        if tokens < 1:
            raise ValueError("SAM fusion token count must be positive")
        self.model_name = str(model)
        self.fusion_tokens = int(tokens)
        # v2: crop the padded canvas and pool on a 2D lattice. v1 features are
        # not comparable, so the fingerprint must reject any v1 cache entry.
        self.cache_fingerprint = hashlib.sha256(
            f"sam-global-v2|{self.fusion_tokens}|{sam_tower_fingerprint(model)}".encode()
        ).hexdigest()
        self.processor = None
        self.sam = None

    @property
    def loaded(self) -> bool:
        return self.processor is not None and self.sam is not None

    def load_pretrained(self, *, device: torch.device | str = "cpu",
                        dtype: torch.dtype | None = None
                        ) -> "SamAlignedFrozenFeatures":
        from transformers import SamModel, SamProcessor

        kwargs = {"torch_dtype": dtype} if dtype is not None else {}
        self.processor = SamProcessor.from_pretrained(self.model_name)
        self.sam = SamModel.from_pretrained(self.model_name, **kwargs)
        self.sam.requires_grad_(False).eval().to(device)
        self.requires_grad_(False).eval()
        return self

    @torch.no_grad()
    def forward(self, images: Sequence[object], *, tokens: int,
                device: torch.device | str | None = None) -> Tensor:
        if not self.loaded:
            raise RuntimeError("call load_pretrained() before extracting SAM features")
        if int(tokens) != self.fusion_tokens:
            raise ValueError(
                f"SAM fusion is fixed at {self.fusion_tokens} tokens, got {tokens}")
        device = device or next(self.sam.parameters()).device
        inputs = self.processor(images=images, return_tensors="pt").to(device)
        dtype = next(self.sam.parameters()).dtype
        dense = self.sam.get_image_embeddings(inputs.pixel_values.to(dtype))
        # Crop the zero-padded canvas before pooling. Pooling the full 64x64 grid
        # spends 25% of a 4:3 feature and 44% of a 16:9 feature describing
        # padding, and those tokens are identical for every image of that aspect
        # ratio — the tower contributes constants instead of structure.
        return sam_cropped_tokens(
            dense, inputs["reshaped_input_sizes"], self.fusion_tokens)


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
