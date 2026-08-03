"""Detail-preserving RADIO1D-H tiling and width-preserving RWKV alignment.

Every 512px tile remains present and keeps all 256 ordered RADIO1D global
tokens, so total visual tokens scale with image size. Every token stays
2560-wide. A compact 128-token tier still exists behind an explicit
``adaptive_token_threshold`` but is off by default: because tokens-per-tile is
flat inside a tier, any step down makes total tokens fall as an image gains
tiles (see ``token_budget_is_monotone``).

Tile geometry is two boxes, not one. ``source_box`` says which image region a
tile covers; the letterbox content box says where real pixels sit inside that
tile's padded square canvas. Both are needed to relate a feature position to
the image-frame coordinates used by ``box=``/``mask16=`` targets.
"""
from __future__ import annotations

import math
import importlib.util
import sys
import types
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
from PIL import Image, ImageOps
from torch import Tensor, nn

from rwkv_lab.deep_vision import DeepVisionInjector


RADIO_TILE_SIZE = 512
RADIO_TOKENS_PER_TILE = 256
RADIO_COMPACT_TOKENS_PER_TILE = 128
RADIO_HIDDEN_SIZE = 2560
# Largest prefix a native C-RADIOv4-H image can produce: the 2048-pixel cap over
# a 16-pixel patch is 128 cells per axis, and column pairing halves that once.
# Not a tile quantum -- a native prefix has no tile structure -- but it is a real
# invariant: any longer prefix is a shape error, not an image.
RADIO_NATIVE_MAX_TOKENS = (2048 // 16) ** 2
RWKV_REFERENCE_CONTEXT = 10_240
DEFAULT_MAX_DETAIL_TILES = 48
DEFAULT_COMPLEXITY_BUDGET_RATIO = 0.75
DEFAULT_COMPLEXITY_TOKEN_QUANTUM = 16
# Above the largest reachable tile count, so every image keeps the full native
# budget. The previous default of 12 halved tokens-per-tile the moment an image
# crossed 12 tiles, which made TOTAL visual tokens fall as images got larger:
# 12 tiles -> 3072 tokens, 13 tiles -> 1664. Measured on the OCR shard, 97% of
# rows sit in that dead zone (185/200 are exactly 13 tiles), and ocr_ppl stayed
# flat near 10.0 for 33k steps while caption_ppl converged to 3.84.
DEFAULT_ADAPTIVE_TOKEN_THRESHOLD = DEFAULT_MAX_DETAIL_TILES + 1
TILE_SCHEMA = "radio1d-rwkv-tiles-v1"


@dataclass(frozen=True)
class RadioTile:
    """One conditioned RADIO input and its location in the source image."""

    image: Image.Image
    source_box: tuple[float, float, float, float]
    role: str
    index: int
    row: int
    column: int
    grid_rows: int
    grid_columns: int

    @property
    def role_id(self) -> int:
        if self.role == "thumbnail":
            return 0
        if self.role == "detail":
            return 1
        raise ValueError(f"unknown tile role {self.role!r}")


@dataclass(frozen=True)
class ContextEstimate:
    visual_tokens: int
    text_tokens: int
    total_tokens: int
    reference_context: int

    @property
    def above_reference(self) -> bool:
        return self.total_tokens > self.reference_context


@dataclass
class RadioVisualPrefix:
    embeddings: Tensor
    mask: Tensor
    tile_count: Tensor
    token_counts: Tensor | None = None


def adaptive_tokens_per_tile(
        maximum: int, *, ratio: float = DEFAULT_COMPLEXITY_BUDGET_RATIO,
        quantum: int = DEFAULT_COMPLEXITY_TOKEN_QUANTUM) -> int:
    """Return a cache-compatible average budget for content routing.

    RADIO1D's tokens are prefix ordered, so routing never invents or pools a
    representation: it chooses how much of each tile's native prefix survives.
    Keeping the average fixed within an exact tile-count bucket lets batched
    RWKV training remain completely unpadded while individual tiles receive
    different amounts of detail.
    """
    if maximum < 1 or quantum < 1 or maximum % quantum:
        raise ValueError("maximum must be a positive multiple of quantum")
    if not 0 < ratio <= 1:
        raise ValueError("complexity budget ratio must be in (0,1]")
    return max(quantum, min(
        maximum, int(round(maximum * ratio / quantum)) * quantum))


def radio_tile_complexity(features: Tensor) -> Tensor:
    """Estimate semantic density cheaply from native RADIO token diversity.

    Only a sparse channel probe and at most 64 ordered tokens are inspected on
    the CPU cache tensor.  Dispersion detects multiple distinct concepts while
    early/tail drift detects detail that was not already captured by the first
    prefix tokens.  This is deliberately scale invariant: raw feature norm is
    not a reliable proxy for visual importance.
    """
    if features.ndim != 3 or features.shape[0] < 1 or features.shape[1] < 1:
        raise ValueError("features must be [tiles,tokens,channels]")
    token_probe = min(int(features.shape[1]), 64)
    channel_stride = max(1, int(features.shape[2]) // 64)
    probe = features[:, :token_probe, ::channel_stride].float()
    probe = torch.nn.functional.normalize(probe, dim=-1, eps=1e-6)
    centroid = torch.nn.functional.normalize(
        probe.mean(dim=1), dim=-1, eps=1e-6)
    dispersion = 1 - (probe * centroid.unsqueeze(1)).sum(dim=-1).mean(dim=1)
    span = max(1, min(16, token_probe // 2))
    early = torch.nn.functional.normalize(
        probe[:, :span].mean(dim=1), dim=-1, eps=1e-6)
    tail = torch.nn.functional.normalize(
        probe[:, -span:].mean(dim=1), dim=-1, eps=1e-6)
    drift = 1 - (early * tail).sum(dim=-1)
    return (dispersion + 0.5 * drift).clamp_min(1e-6)


def allocate_radio_token_counts(
        features: Tensor, roles: Tensor | None = None, *,
        ratio: float = DEFAULT_COMPLEXITY_BUDGET_RATIO,
        quantum: int = DEFAULT_COMPLEXITY_TOKEN_QUANTUM) -> Tensor:
    """Allocate one fixed total across tiles according to content complexity."""
    tiles, maximum = int(features.shape[0]), int(features.shape[1])
    average = adaptive_tokens_per_tile(maximum, ratio=ratio, quantum=quantum)
    minimum = min(average, max(quantum, maximum // 4))
    minimum = max(quantum, minimum // quantum * quantum)
    counts = torch.full((tiles,), minimum, dtype=torch.long)
    target = tiles * average
    units = (target - int(counts.sum())) // quantum
    scores = radio_tile_complexity(features).cpu()
    if roles is not None:
        if roles.shape != (tiles,):
            raise ValueError("roles must have one value per tile")
        # Never starve the global thumbnail: it carries cross-tile context even
        # when its letterboxed pixels look less locally diverse.
        scores = scores + roles.cpu().eq(0).float() * scores.mean() * 0.10
    for _ in range(units):
        available = counts < maximum
        if not bool(available.any()):
            raise RuntimeError("adaptive RADIO allocator exhausted its capacity")
        # Diminishing returns distribute tokens approximately in proportion to
        # semantic density without allowing one noisy tile to consume the run.
        utility = scores / (counts.float() + quantum)
        utility = utility.masked_fill(~available, -torch.inf)
        counts[int(utility.argmax())] += quantum
    if int(counts.sum()) != target:
        raise RuntimeError("adaptive RADIO allocation did not preserve its total")
    return counts


def tokens_per_tile_for_tile_count(
        tile_count: int, *, threshold: int = DEFAULT_ADAPTIVE_TOKEN_THRESHOLD
        ) -> int:
    """Select the deterministic native RADIO token budget for a tile grid."""
    if tile_count < 1:
        raise ValueError("tile_count must be positive")
    if threshold < 1:
        raise ValueError("adaptive token threshold must be positive")
    return (RADIO_TOKENS_PER_TILE if tile_count <= threshold
            else RADIO_COMPACT_TOKENS_PER_TILE)


def token_budget_is_monotone(
        *, threshold: int = DEFAULT_ADAPTIVE_TOKEN_THRESHOLD,
        max_tiles: int = DEFAULT_MAX_DETAIL_TILES + 1) -> bool:
    """Whether total visual tokens never fall as an image gains tiles.

    Tokens-per-tile is flat inside a tier, so ``tiles * tokens`` already grows
    across the tier; a step down at boundary T must therefore still clear
    ``256 * (T-1)``. A 256->128 halving never can — at T=13 the largest legal
    value is 236 — so any threshold below ``max_tiles`` makes a larger image
    receive strictly less representation than a smaller one.
    """
    totals = [tile * tokens_per_tile_for_tile_count(tile, threshold=threshold)
              for tile in range(1, int(max_tiles) + 1)]
    return all(a <= b for a, b in zip(totals, totals[1:]))


def estimate_context(tile_count: int, text_tokens: int,
                     *, tokens_per_tile: int | None = None,
                     adaptive_token_threshold: int = DEFAULT_ADAPTIVE_TOKEN_THRESHOLD,
                     reference_context: int = RWKV_REFERENCE_CONTEXT,
                     structural_tokens: int = 0) -> ContextEstimate:
    """Report context use without rejecting or truncating long examples."""
    if tile_count < 1 or text_tokens < 0 or structural_tokens < 0:
        raise ValueError("tile_count must be positive and token counts nonnegative")
    if tokens_per_tile is None:
        tokens_per_tile = tokens_per_tile_for_tile_count(
            tile_count, threshold=adaptive_token_threshold)
    if tokens_per_tile < 1:
        raise ValueError("tokens_per_tile must be positive")
    visual = tile_count * tokens_per_tile
    total = visual + text_tokens + structural_tokens
    return ContextEstimate(visual, text_tokens, total, reference_context)


def choose_detail_grid(width: int, height: int, *, tile_size: int = RADIO_TILE_SIZE,
                       max_detail_tiles: int = DEFAULT_MAX_DETAIL_TILES
                       ) -> tuple[int, int]:
    """Choose an aspect-aware grid with roughly one tile per 512^2 pixels.

    The count is a source-detail target, not a context-derived budget.  Aspect
    accuracy is weighted slightly more than hitting the exact target because it
    prevents detail cells from requiring severe letterboxing.
    """
    if width < 1 or height < 1 or tile_size < 1 or max_detail_tiles < 1:
        raise ValueError("image, tile, and grid dimensions must be positive")
    target = min(max_detail_tiles,
                 max(1, math.ceil((width * height) / float(tile_size * tile_size))))
    if target == 1:
        return 1, 1
    aspect = width / height
    best: tuple[float, int, int, int] | None = None
    for rows in range(1, max_detail_tiles + 1):
        for columns in range(1, max_detail_tiles // rows + 1):
            count = rows * columns
            ratio_error = abs(math.log((columns / rows) / aspect))
            count_error = abs(count - target) / target
            # Prefer additional coverage on exact ties: excess detail is safer
            # than insufficient detail for the high-fidelity configuration.
            under_target = max(0, target - count) / target
            score = ratio_error + 0.20 * count_error + 0.08 * under_target
            candidate = (score, -count, rows, columns)
            if best is None or candidate < best:
                best = candidate
    assert best is not None
    return best[2], best[3]


def letterbox_content_box(crop_width: float, crop_height: float
                          ) -> tuple[float, float, float, float]:
    """Where real pixels sit inside a letterboxed square canvas, normalized.

    ``_letterbox`` contains a crop inside a square and centers it, so a
    non-square crop leaves bars. Nothing downstream recorded that transform,
    while ``box=``/``mask16=`` targets are normalized to the ORIGINAL image
    (see ``_normalized_box`` in build_captioning_first_mix.py, which divides by
    the COCO image width/height). Predicting an image-frame coordinate from a
    canvas-frame feature therefore required an offset of up to 37% of the
    canvas that varies per aspect ratio and was never supplied — unidentifiable
    for single-tile rows, whose source box is always (0,0,1,1).
    """
    if crop_width <= 0 or crop_height <= 0:
        raise ValueError("letterbox crop dimensions must be positive")
    aspect = crop_width / crop_height
    width, height = (1.0, 1.0 / aspect) if aspect >= 1.0 else (aspect, 1.0)
    return ((1.0 - width) / 2, (1.0 - height) / 2,
            (1.0 + width) / 2, (1.0 + height) / 2)


def content_boxes_from_source(source_boxes: Tensor, image_aspect: Tensor) -> Tensor:
    """Per-tile content extent inside each canvas, from source boxes + aspect.

    Derived rather than cached: a tile's crop aspect is fully determined by its
    normalized source box and the source image's own aspect, so existing feature
    caches stay valid.
    """
    if source_boxes.ndim != 3 or source_boxes.shape[-1] != 4:
        raise ValueError("source boxes must be [batch, tiles, 4]")
    aspect = image_aspect.reshape(-1, 1).to(
        device=source_boxes.device, dtype=torch.float32)
    if aspect.shape[0] != source_boxes.shape[0]:
        raise ValueError("one image aspect is required per batch row")
    if not bool((aspect > 0).all()):
        raise ValueError("image aspect ratios must be positive")
    boxes = source_boxes.float()
    span_x = (boxes[..., 2] - boxes[..., 0]).clamp_min(1e-6) * aspect
    span_y = (boxes[..., 3] - boxes[..., 1]).clamp_min(1e-6)
    crop_aspect = span_x / span_y
    width = torch.where(crop_aspect >= 1, torch.ones_like(crop_aspect),
                        crop_aspect)
    height = torch.where(crop_aspect >= 1, 1.0 / crop_aspect,
                         torch.ones_like(crop_aspect))
    return torch.stack(((1 - width) / 2, (1 - height) / 2,
                        (1 + width) / 2, (1 + height) / 2), dim=-1)


def _letterbox(image: Image.Image, size: int,
               fill: tuple[int, int, int] = (0, 0, 0)) -> Image.Image:
    image = image.convert("RGB")
    contained = ImageOps.contain(image, (size, size), Image.Resampling.BICUBIC)
    canvas = Image.new("RGB", (size, size), fill)
    canvas.paste(contained, ((size - contained.width) // 2,
                             (size - contained.height) // 2))
    return canvas


def native_aspect_size(width: int, height: int, *, budget: int = RADIO_TILE_SIZE,
                       step: int = 16) -> tuple[int, int]:
    """Nearest RADIO-supported non-square size preserving the crop's aspect.

    RADIO accepts any multiple of ``min_resolution_step`` independently per axis
    (verified: 512x288 -> 576 tokens, 288x512 -> 576), so letterboxing to a
    square is pure waste — it spends tokens on grey bars and forces the model to
    undo an offset that is never recorded. ``budget`` bounds the longer edge, so
    a square crop still lands on the model's preferred 512x512.
    """
    if width < 1 or height < 1 or budget < step or step < 1:
        raise ValueError("tile geometry must be positive and at least one step")
    scale = budget / max(width, height)
    sized = []
    for value in (height, width):
        snapped = int(round(value * scale / step)) * step
        sized.append(max(step, min(budget, snapped)))
    return sized[0], sized[1]


def _resize_native(image: Image.Image, *, budget: int = RADIO_TILE_SIZE,
                   step: int = 16) -> Image.Image:
    """Resize a crop to its own supported resolution, with no padding."""
    image = image.convert("RGB")
    height, width = native_aspect_size(
        image.width, image.height, budget=budget, step=step)
    return image.resize((width, height), Image.Resampling.BICUBIC)


def _detail_boxes(width: int, height: int, rows: int, columns: int,
                  overlap: float) -> Iterable[tuple[int, int, int, int]]:
    if not 0 <= overlap < 1:
        raise ValueError("overlap must be in [0, 1)")
    cell_width, cell_height = width / columns, height / rows
    expand_x, expand_y = cell_width * overlap / 2, cell_height * overlap / 2
    for row in range(rows):
        for column in range(columns):
            x0 = max(0, math.floor(column * cell_width - expand_x))
            y0 = max(0, math.floor(row * cell_height - expand_y))
            x1 = min(width, math.ceil((column + 1) * cell_width + expand_x))
            y1 = min(height, math.ceil((row + 1) * cell_height + expand_y))
            yield x0, y0, x1, y1


def build_radio_tiles(image: Image.Image, *, tile_size: int = RADIO_TILE_SIZE,
                      max_detail_tiles: int = DEFAULT_MAX_DETAIL_TILES,
                      overlap: float = 0.125,
                      letterbox: bool = True) -> list[RadioTile]:
    """Create a global thumbnail plus generous aspect-aware detail crops.

    A one-cell source is represented once rather than duplicated as identical
    thumbnail and detail tiles. EXIF orientation is applied before geometry is
    computed.

    ``letterbox=False`` sizes every tile to its own supported non-square
    resolution instead of padding it into a square. Detail cells differ by only
    a few pixels of edge rounding, so they are snapped to one shared size and
    stay batchable; the thumbnail keeps its own aspect. This removes the padding
    entirely — 25-75% of the tokens on a non-square crop described grey bars.
    """
    source = ImageOps.exif_transpose(image).convert("RGB")
    width, height = source.size
    rows, columns = choose_detail_grid(
        width, height, tile_size=tile_size, max_detail_tiles=max_detail_tiles)
    detail_count = rows * columns

    def shape(crop: Image.Image) -> Image.Image:
        return (_letterbox(crop, tile_size) if letterbox
                else _resize_native(crop, budget=tile_size))

    if detail_count == 1:
        return [RadioTile(shape(source), (0.0, 0.0, 1.0, 1.0),
                          "thumbnail", 0, 0, 0, 1, 1)]

    boxes = list(_detail_boxes(width, height, rows, columns, overlap))
    if letterbox:
        detail_size = None
    else:
        # One shared size for every detail cell, from the mean cell geometry, so
        # a single batched forward covers them all.
        mean_w = sum(x1 - x0 for x0, _, x1, _ in boxes) / len(boxes)
        mean_h = sum(y1 - y0 for _, y0, _, y1 in boxes) / len(boxes)
        cell_h, cell_w = native_aspect_size(
            max(1, round(mean_w)), max(1, round(mean_h)), budget=tile_size)
        detail_size = (cell_w, cell_h)

    tiles = [RadioTile(shape(source), (0.0, 0.0, 1.0, 1.0),
                       "thumbnail", 0, -1, -1, rows, columns)]
    for index, box in enumerate(boxes, start=1):
        x0, y0, x1, y1 = box
        row, column = divmod(index - 1, columns)
        crop = source.crop(box)
        sized = (shape(crop) if detail_size is None
                 else crop.resize(detail_size, Image.Resampling.BICUBIC))
        tiles.append(RadioTile(
            sized, (x0 / width, y0 / height, x1 / width, y1 / height),
            "detail", index, row, column, rows, columns))
    return tiles


def tile_metadata(tiles: Sequence[RadioTile]) -> tuple[Tensor, Tensor]:
    if not tiles:
        raise ValueError("at least one tile is required")
    boxes = torch.tensor([tile.source_box for tile in tiles], dtype=torch.float32)
    roles = torch.tensor([tile.role_id for tile in tiles], dtype=torch.long)
    return boxes, roles


def pad_tile_metadata(samples: Sequence[Sequence[RadioTile]],
                      maximum: int | None = None
                      ) -> tuple[Tensor, Tensor, Tensor]:
    """Pad geometry in exactly the same tile dimension as cached features."""
    if not samples or any(not sample for sample in samples):
        raise ValueError("every sample must contain at least one tile")
    maximum = maximum or max(len(sample) for sample in samples)
    if maximum < max(len(sample) for sample in samples):
        raise ValueError("maximum cannot truncate tile metadata")
    boxes = torch.zeros((len(samples), maximum, 4), dtype=torch.float32)
    roles = torch.zeros((len(samples), maximum), dtype=torch.long)
    mask = torch.zeros((len(samples), maximum), dtype=torch.bool)
    for row, sample in enumerate(samples):
        sample_boxes, sample_roles = tile_metadata(sample)
        boxes[row, :len(sample)] = sample_boxes
        roles[row, :len(sample)] = sample_roles
        mask[row, :len(sample)] = True
    return boxes, roles, mask


def tiles_to_tensor(tiles: Sequence[RadioTile]) -> Tensor:
    """Convert RGB PIL tiles to RADIO's documented [0,1] NCHW input."""
    if not tiles:
        raise ValueError("at least one tile is required")
    arrays = [np.asarray(tile.image, dtype=np.float32) / 255.0 for tile in tiles]
    value = torch.from_numpy(np.stack(arrays)).permute(0, 3, 1, 2)
    return value.contiguous()


def load_radio1d_h(model_path: str | Path, *, device: str | torch.device = "cuda",
                   dtype: torch.dtype = torch.bfloat16,
                   keep_decoder: bool = False) -> nn.Module:
    """Load and freeze the pinned Hugging Face RADIO1D-H implementation.

    ``keep_decoder`` retains the reconstruction decoder so a caller can ask for
    the teacher-aligned spatial grid instead of only the compressed 1D tokens.
    """
    # The installed flash-attn binary can lag the host PyTorch ABI after a
    # system update. RADIO uses its own timm attention path, so prevent
    # Transformers from eagerly importing that unrelated optional extension.
    # Scoped and restored: patching only `transformers.utils` misses modules that
    # already imported the symbol by value, and leaving it installed changes
    # behavior for unrelated code for the rest of the process.
    from rwkv_lab.rwkv_finetune import _flash_attn_2_reported_unavailable

    with _flash_attn_2_reported_unavailable():
        return _load_radio1d_h(model_path, device=device, dtype=dtype,
                               keep_decoder=keep_decoder)


def _load_radio1d_h(model_path: str | Path, *, device: str | torch.device,
                    dtype: torch.dtype, keep_decoder: bool) -> nn.Module:
    # Transformers 4.52's local trust_remote_code copier omits a transitive
    # NVIDIA relative import (`utils.py`). Load the pinned directory as a real
    # Python package instead, then use the standard sharded-checkpoint loader.
    path = Path(model_path).resolve()
    if not (path / "hf_model.py").is_file():
        raise FileNotFoundError(f"RADIO1D hf_model.py is missing from {path}")
    package_name = f"_rwkv_radio1d_{abs(hash(str(path))):x}"
    module_name = f"{package_name}.hf_model"
    if module_name in sys.modules:
        remote = sys.modules[module_name]
    else:
        package = types.ModuleType(package_name)
        package.__path__ = [str(path)]
        package.__package__ = package_name
        sys.modules[package_name] = package
        specification = importlib.util.spec_from_file_location(
            module_name, path / "hf_model.py")
        if specification is None or specification.loader is None:
            raise ImportError(f"cannot import RADIO1D implementation from {path}")
        remote = importlib.util.module_from_spec(specification)
        sys.modules[module_name] = remote
        specification.loader.exec_module(remote)

    config = remote.RADIOConfig.from_pretrained(str(path), local_files_only=True)
    dtype_name = str(dtype).removeprefix("torch.")
    config.args["dtype"] = dtype_name
    config.args["amp_dtype"] = dtype_name
    model = remote.RADIOModel(config)
    from transformers.modeling_utils import load_sharded_checkpoint
    result = load_sharded_checkpoint(
        model, str(path), strict=True, prefer_safe=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(
            f"RADIO checkpoint mismatch missing={result.missing_keys} "
            f"unexpected={result.unexpected_keys}")
    # The ~314M reconstruction decoder is dead weight for encoder-only feature
    # caching, so it is dropped by default. It is NOT, however, merely a
    # "training-only head": RADIO1D's teachers (SigLIP2-g, DINOv3-7B, SAM3)
    # supervise the *decoder* output, not the 1D tokens. The 1D tokens are a
    # compressed code and this decoder is the learned unpacker back into the
    # teacher-aligned spatial basis. Keep it when a consumer needs that basis
    # rather than the code — see `extract_radio_dense_features`.
    encoder = model.radio_model.model
    if not keep_decoder and hasattr(encoder, "decoder"):
        encoder.decoder = None
    model.eval().requires_grad_(False)
    return model.to(device=device)


def extract_radio_global_tokens(model: nn.Module, pixels: Tensor,
                                *, num_tokens: int = RADIO_TOKENS_PER_TILE
                                ) -> tuple[Tensor, Tensor]:
    """Extract only RADIO1D's ordered globals, excluding class/register tokens."""
    wrapper = getattr(model, "radio_model", None)
    if wrapper is None or not hasattr(wrapper, "input_conditioner"):
        raise TypeError("model does not expose RADIO's input conditioner")
    encoder = getattr(wrapper, "model", None)
    if encoder is None or not hasattr(encoder, "forward_encoder"):
        raise TypeError("model does not expose RADIO1D.forward_encoder")
    conditioned = wrapper.input_conditioner(pixels)
    output = encoder.forward_encoder(conditioned, num_tokens=num_tokens)
    if not isinstance(output, dict) or not {
            "global_tokens", "global_token_mask"} <= output.keys():
        raise RuntimeError("RADIO1D encoder did not return global-token outputs")
    tokens, mask = output["global_tokens"], output["global_token_mask"]
    expected = (pixels.shape[0], num_tokens, RADIO_HIDDEN_SIZE)
    if tuple(tokens.shape) != expected:
        raise RuntimeError(f"RADIO global tokens {tuple(tokens.shape)} != {expected}")
    if tuple(mask.shape) != expected[:2] or not bool(mask.all()):
        raise RuntimeError("RADIO did not produce all requested global tokens")
    if not bool(torch.isfinite(tokens).all()):
        raise RuntimeError("RADIO global tokens contain non-finite values")
    return tokens, mask


@torch.inference_mode()
def extract_radio_dense_features(
        model: nn.Module, pixels: Tensor, *,
        num_tokens: int = RADIO_TOKENS_PER_TILE,
        pooled_tokens: int | None = None) -> Tensor:
    """Return RADIO1D's teacher-aligned spatial grid rather than its 1D code.

    This is the counterpart to :func:`extract_radio_global_tokens`. RADIO1D's
    teachers — SigLIP2-g, DINOv3-7B and SAM3 — supervise the *decoder's*
    reconstructed 2D grid, not the encoder's 1D tokens. Those tokens are a
    compressed code for the grid, and the decoder is the learned unpacker; a
    downstream model handed only the code would have to invert a 6-block, ~314M
    parameter decoder to recover the basis the teachers actually shaped. SAM3's
    contribution is dense-only (``use_summary: false`` in the RADIO config), so
    it is precisely the part that does not survive that omission.

    The decoder adds no information beyond the 1D tokens — by construction it
    reconstructs from them alone, using its own learnable prefix tokens. What it
    adds is the basis. Because the result is a deterministic function of a tile,
    it caches exactly like the encoder output.

    Requires ``load_radio1d_h(..., keep_decoder=True)``. Returns
    ``[batch, tokens, channels]`` when ``pooled_tokens`` is given, otherwise the
    full ``[batch, height, width, channels]`` grid.
    """
    wrapper = getattr(model, "radio_model", None)
    encoder = getattr(wrapper, "model", None) if wrapper is not None else None
    if encoder is None or not hasattr(encoder, "forward_features"):
        raise TypeError("model does not expose RADIO1D.forward_features")
    if getattr(encoder, "decoder", None) is None:
        raise RuntimeError(
            "RADIO1D was loaded without its reconstruction decoder; pass "
            "keep_decoder=True to load_radio1d_h to use the teacher-aligned grid")
    conditioned = wrapper.input_conditioner(pixels)
    output = encoder.forward_features(
        conditioned, num_tokens=num_tokens, neck_name="decoder")
    if not isinstance(output, dict) or "decoder" not in output:
        raise RuntimeError("RADIO1D did not return a decoder reconstruction")
    decoded = output["decoder"]
    prefix = int(getattr(encoder, "num_prefix_tokens", 0))
    patches = decoded[:, prefix:]
    side = int(round(patches.shape[1] ** 0.5))
    if side * side != patches.shape[1]:
        raise RuntimeError(
            f"RADIO decoder returned {patches.shape[1]} patches, which is not square")
    grid = patches.reshape(patches.shape[0], side, side, patches.shape[-1])
    if not bool(torch.isfinite(grid).all()):
        raise RuntimeError("RADIO decoder features contain non-finite values")
    if pooled_tokens is None:
        return grid
    from rwkv_lab.vision_fusion import pool_grid_tokens

    # 2D pooling: each token stays a rectangular region of the reconstruction.
    return pool_grid_tokens(grid.permute(0, 3, 1, 2), pooled_tokens)


@torch.inference_mode()
def encode_radio_tiles(model: nn.Module, tiles: Sequence[RadioTile], *,
                       device: str | torch.device = "cuda", batch_size: int = 8,
                       num_tokens: int = RADIO_TOKENS_PER_TILE,
                       cache_dtype: torch.dtype = torch.bfloat16) -> Tensor:
    """Encode tiles in bounded batches and return CPU `[tiles,tokens,2560]`."""
    if batch_size < 1 or num_tokens not in (
            RADIO_COMPACT_TOKENS_PER_TILE, RADIO_TOKENS_PER_TILE):
        raise ValueError("batch_size must be positive and num_tokens 128 or 256")
    pixels = tiles_to_tensor(tiles)
    chunks: list[Tensor] = []
    for start in range(0, len(tiles), batch_size):
        batch = pixels[start:start + batch_size].to(device=device, non_blocking=True)
        tokens, _ = extract_radio_global_tokens(
            model, batch, num_tokens=num_tokens)
        chunks.append(tokens.to(device="cpu", dtype=cache_dtype))
    return torch.cat(chunks, dim=0).contiguous()


def pad_radio_features(samples: Sequence[Tensor]) -> tuple[Tensor, Tensor]:
    """Pad variable tile counts, retaining a per-tile validity mask."""
    if not samples:
        raise ValueError("at least one feature tensor is required")
    token_counts = {int(value.shape[1]) for value in samples if value.ndim == 3}
    for value in samples:
        if (value.ndim != 3 or value.shape[-1] != RADIO_HIDDEN_SIZE
                or value.shape[1] not in (
                    RADIO_COMPACT_TOKENS_PER_TILE, RADIO_TOKENS_PER_TILE)):
            raise ValueError("RADIO features must be [tiles,128|256,2560]")
    if len(token_counts) != 1:
        raise ValueError("RADIO feature batch must use one tokens-per-tile budget")
    tokens_per_tile = token_counts.pop()
    maximum = max(value.shape[0] for value in samples)
    result = samples[0].new_zeros((len(samples), maximum,
                                  tokens_per_tile, RADIO_HIDDEN_SIZE))
    mask = torch.zeros((len(samples), maximum), dtype=torch.bool,
                       device=samples[0].device)
    for row, value in enumerate(samples):
        result[row, :value.shape[0]].copy_(value)
        mask[row, :value.shape[0]] = True
    return result, mask


def fourier_box_features(
        boxes: Tensor, frequency_count: int = 4) -> Tensor:
    """Parameter-free Fourier basis for normalized source boxes."""
    center = (boxes[..., :2] + boxes[..., 2:]) / 2
    size = boxes[..., 2:] - boxes[..., :2]
    geometry = torch.cat((boxes, center, size), dim=-1).float()
    frequencies = 2.0 ** torch.arange(
        frequency_count, device=boxes.device, dtype=torch.float32)
    phase = geometry.unsqueeze(-1) * frequencies * math.pi
    return torch.cat((phase.sin(), phase.cos()), dim=-1).flatten(-2)


class FourierBoxEmbedding(nn.Module):
    """Embed normalized source boxes without discretizing image geometry."""

    def __init__(self, output_size: int, frequency_count: int = 4):
        super().__init__()
        self.frequency_count = frequency_count
        # x0,y0,x1,y1,cx,cy,width,height, each with sin and cos.
        self.projection = nn.Linear(8 * frequency_count * 2, output_size)
        nn.init.zeros_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)

    def encode(self, boxes: Tensor) -> Tensor:
        return fourier_box_features(boxes, self.frequency_count)

    def project(self, features: Tensor) -> Tensor:
        expected = 8 * self.frequency_count * 2
        if features.shape[-1] != expected:
            raise ValueError(
                f"box Fourier features need width {expected}, got "
                f"{features.shape[-1]}")
        return self.projection(features.to(self.projection.weight.dtype))

    def forward(self, boxes: Tensor) -> Tensor:
        return self.project(self.encode(boxes))


class RadioRWKVBridge(nn.Module):
    """Token- and width-preserving semantic alignment for frozen foundations."""

    def __init__(self, hidden_size: int = RADIO_HIDDEN_SIZE, rank: int = 256,
                 tokens_per_tile: int = RADIO_TOKENS_PER_TILE,
                 max_tiles: int = DEFAULT_MAX_DETAIL_TILES + 1,
                 input_size: int | None = None,
                 letterbox_geometry: bool = True):
        super().__init__()
        # This module is width-PRESERVING: its output goes straight into the LM
        # token stream, so hidden_size must equal the LM width. RADIO1D happens
        # to emit 2560, matching RWKV-2.9B exactly. Encoders of other widths
        # (C-RADIOv4-H is 1280) need an explicit input projection first.
        self.input_size = int(input_size or hidden_size)
        self.input_projection = (
            nn.Linear(self.input_size, hidden_size, bias=False)
            if self.input_size != hidden_size else None)
        if self.input_projection is not None:
            nn.init.normal_(self.input_projection.weight,
                            std=hidden_size ** -0.5)
        self.hidden_size = hidden_size
        self.tokens_per_tile = tokens_per_tile
        self.max_tiles = max_tiles
        self.norm = nn.LayerNorm(hidden_size)
        self.down = nn.Linear(hidden_size, rank, bias=False)
        self.up = nn.Linear(rank, hidden_size, bias=False)
        self.gate = nn.Parameter(torch.zeros(()))
        self.role_embedding = nn.Embedding(2, hidden_size)
        self.tile_embedding = nn.Embedding(max_tiles, hidden_size)
        self.token_embedding = nn.Embedding(tokens_per_tile, hidden_size)
        self.box_embedding = FourierBoxEmbedding(hidden_size)
        # Where real pixels sit inside each letterboxed canvas. Zero-init like
        # every other metadata embedding, so supplying it is an exact no-op at
        # step 0. Skipped entirely when nothing letterboxes (native whole-image
        # encoding), where it would sit in the checkpoint never receiving a
        # gradient.
        self.content_embedding = (
            FourierBoxEmbedding(hidden_size) if letterbox_geometry else None)
        # The gated residual begins as an exact no-op, but its gate must still
        # receive a gradient. Zeroing both gate and Up would deadlock the branch.
        nn.init.normal_(self.up.weight, std=1e-3)
        nn.init.zeros_(self.role_embedding.weight)
        nn.init.zeros_(self.tile_embedding.weight)
        nn.init.zeros_(self.token_embedding.weight)

    def _align_packed(self, features: Tensor, boxes: Tensor, roles: Tensor,
                      tile_indices: Tensor, token_indices: Tensor,
                      content: Tensor | None = None,
                      box_fourier: Tensor | None = None) -> Tensor:
        """Align an already packed, unpadded RADIO sequence."""
        if features.ndim != 3 or features.shape[-1] != self.input_size:
            raise ValueError(
                f"packed features must be [batch,tokens,{self.input_size}]")
        if self.input_projection is not None:
            features = self.input_projection(features)
        batch, length = features.shape[:2]
        if (boxes.shape != (batch, length, 4)
                or roles.shape != (batch, length)
                or tile_indices.shape != (batch, length)
                or token_indices.shape != (batch, length)):
            raise ValueError("packed RADIO metadata does not match its sequence")
        if content is not None and content.shape != boxes.shape:
            raise ValueError("packed content boxes must match the source boxes")
        value = self.norm(features)
        residual = self.up(torch.nn.functional.gelu(self.down(value)))
        value = value + self.gate.tanh() * residual
        metadata = (
            self.role_embedding(roles)
            + self.tile_embedding(tile_indices)
            + self.token_embedding(token_indices)
            + (self.box_embedding(boxes) if box_fourier is None
               else self.box_embedding.project(box_fourier)).to(value.dtype)
        )
        if content is not None:
            if self.content_embedding is None:
                raise ValueError(
                    "this bridge was built without letterbox geometry")
            # source box says WHICH image region a tile covers; content box says
            # where that region actually lives inside the padded canvas. Both are
            # needed to map a feature position back to an image coordinate.
            metadata = metadata + self.content_embedding(content).to(value.dtype)
        return value + metadata

    def forward_variable(
            self, samples: Sequence[tuple[Tensor, Tensor, Tensor]], *,
            budget_ratio: float = DEFAULT_COMPLEXITY_BUDGET_RATIO,
            token_quantum: int = DEFAULT_COMPLEXITY_TOKEN_QUANTUM,
            image_aspect: Tensor | None = None,
            ) -> RadioVisualPrefix:
        """Pack content-routed tile prefixes with no recurrent padding steps."""
        if not samples:
            raise ValueError("at least one RADIO sample is required")
        tile_counts = {int(tokens.shape[0]) for tokens, _, _ in samples}
        maxima = {int(tokens.shape[1]) for tokens, _, _ in samples}
        if len(tile_counts) != 1 or len(maxima) != 1:
            raise ValueError("adaptive RADIO batches need one tile/max-token bucket")
        tile_count, maximum = tile_counts.pop(), maxima.pop()
        if tile_count > self.max_tiles or maximum > self.tokens_per_tile:
            raise ValueError("adaptive RADIO sample exceeds bridge capacity")
        device = self.gate.device
        packed_features, packed_boxes, packed_roles = [], [], []
        packed_tiles, packed_tokens, all_counts = [], [], []
        lengths: set[int] = set()
        for tokens, boxes, roles in samples:
            if (tokens.ndim != 3 or tokens.shape[-1] != self.input_size
                    or boxes.shape != (tile_count, 4)
                    or roles.shape != (tile_count,)):
                raise ValueError("malformed adaptive RADIO cache tuple")
            counts = allocate_radio_token_counts(
                tokens, roles, ratio=budget_ratio, quantum=token_quantum)
            all_counts.append(counts)
            row_features, row_boxes, row_roles = [], [], []
            row_tiles, row_tokens = [], []
            for tile, count in enumerate(counts.tolist()):
                row_features.append(tokens[tile, :count])
                row_boxes.append(boxes[tile].expand(count, 4))
                row_roles.append(roles[tile].expand(count))
                row_tiles.append(torch.full((count,), tile, dtype=torch.long))
                row_tokens.append(torch.arange(count, dtype=torch.long))
            packed_features.append(torch.cat(row_features))
            packed_boxes.append(torch.cat(row_boxes))
            packed_roles.append(torch.cat(row_roles))
            packed_tiles.append(torch.cat(row_tiles))
            packed_tokens.append(torch.cat(row_tokens))
            lengths.add(sum(counts.tolist()))
        if len(lengths) != 1:
            raise RuntimeError("adaptive RADIO routing introduced batch padding")
        features = torch.stack(packed_features).to(device=device, non_blocking=True)
        box_tokens = torch.stack(packed_boxes).to(device=device, non_blocking=True)
        role_tokens = torch.stack(packed_roles).to(device=device, non_blocking=True)
        tile_indices = torch.stack(packed_tiles).to(device=device, non_blocking=True)
        token_indices = torch.stack(packed_tokens).to(device=device, non_blocking=True)
        content_tokens = None
        if image_aspect is not None:
            # Recompute per packed token: routing gives each tile a different
            # count, so the per-tile content box has to be expanded the same way
            # the source box already was.
            content_rows = content_boxes_from_source(
                torch.stack([boxes for _, boxes, _ in samples]).to(device),
                image_aspect)
            content_tokens = torch.stack([
                torch.cat([content_rows[row, tile].expand(count, 4)
                           for tile, count in enumerate(counts.tolist())])
                for row, counts in enumerate(all_counts)
            ]).to(device=device, non_blocking=True)
        embeddings = self._align_packed(
            features, box_tokens, role_tokens, tile_indices, token_indices,
            content_tokens)
        mask = torch.ones(embeddings.shape[:2], dtype=torch.bool, device=device)
        return RadioVisualPrefix(
            embeddings=embeddings, mask=mask,
            tile_count=torch.full((len(samples),), tile_count, dtype=torch.long,
                                  device=device),
            token_counts=torch.stack(all_counts).to(device=device))

    def forward_native(
            self, features: Tensor, boxes: Tensor,
            box_fourier: Tensor | None = None) -> RadioVisualPrefix:
        """Align whole-image native features carrying per-token geometry.

        There is no tile axis and no per-index position table: every token
        already knows where it sits via its own box, so tile/token embeddings
        are held at index 0 (both are zero-init and contribute nothing).
        """
        if features.ndim != 3 or features.shape[-1] != self.input_size:
            raise ValueError(
                f"native features must be [batch,tokens,{self.input_size}]")
        if boxes.shape != (*features.shape[:2], 4):
            raise ValueError("native boxes must be [batch,tokens,4]")
        if (box_fourier is not None
                and box_fourier.shape[:2] != features.shape[:2]):
            raise ValueError("native box Fourier features do not match tokens")
        batch, length = features.shape[:2]
        zeros = torch.zeros((batch, length), dtype=torch.long,
                            device=features.device)
        value = self._align_packed(
            features, boxes, zeros, zeros, zeros,
            box_fourier=box_fourier)
        return RadioVisualPrefix(
            embeddings=value,
            mask=torch.ones((batch, length), dtype=torch.bool,
                            device=features.device),
            tile_count=torch.ones(batch, dtype=torch.long, device=features.device),
            token_counts=torch.full((batch, 1), length, dtype=torch.long,
                                    device=features.device))

    def forward(self, features: Tensor, boxes: Tensor, roles: Tensor,
                tile_mask: Tensor | None = None,
                image_aspect: Tensor | None = None) -> RadioVisualPrefix:
        active_tokens = int(features.shape[-2]) if features.ndim == 4 else 0
        if (features.ndim != 4 or features.shape[-1] != self.input_size
                or active_tokens not in (
                    RADIO_COMPACT_TOKENS_PER_TILE, self.tokens_per_tile)):
            raise ValueError(
                f"features must be [batch,tiles,128|{self.tokens_per_tile},"
                f"{self.input_size}]")
        batch, tiles = features.shape[:2]
        if tiles > self.max_tiles:
            raise ValueError(f"{tiles} tiles exceeds configured maximum {self.max_tiles}")
        if boxes.shape != (batch, tiles, 4) or roles.shape != (batch, tiles):
            raise ValueError("boxes must be [B,T,4] and roles must be [B,T]")
        if tile_mask is None:
            tile_mask = torch.ones((batch, tiles), dtype=torch.bool,
                                   device=features.device)
        if tile_mask.shape != (batch, tiles):
            raise ValueError("tile_mask must be [B,T]")

        tile_index = torch.arange(tiles, device=features.device)
        token_index = torch.arange(active_tokens, device=features.device)
        token_mask = tile_mask.unsqueeze(-1).expand(
            batch, tiles, active_tokens).reshape(batch, -1)
        packed_boxes = boxes.unsqueeze(2).expand(
            batch, tiles, active_tokens, 4).reshape(batch, -1, 4)
        packed_roles = roles.unsqueeze(2).expand(
            batch, tiles, active_tokens).reshape(batch, -1)
        packed_tiles = tile_index[None, :, None].expand(
            batch, tiles, active_tokens).reshape(batch, -1)
        packed_tokens = token_index[None, None, :].expand(
            batch, tiles, active_tokens).reshape(batch, -1)
        packed_content = None
        if image_aspect is not None:
            packed_content = content_boxes_from_source(
                boxes, image_aspect).unsqueeze(2).expand(
                    batch, tiles, active_tokens, 4).reshape(batch, -1, 4)
        value = self._align_packed(
            features.reshape(batch, -1, self.input_size), packed_boxes,
            packed_roles, packed_tiles, packed_tokens, packed_content)
        value = value.masked_fill(~token_mask.unsqueeze(-1), 0)
        return RadioVisualPrefix(
            embeddings=value, mask=token_mask,
            tile_count=tile_mask.sum(dim=1),
            token_counts=tile_mask.long() * active_tokens)


class RadioFeatureProjector(nn.Module):
    """Adapt exact-tile-count cached feature tuples into one RWKV prefix."""

    def __init__(self, bridge: RadioRWKVBridge | None = None, *,
                 adaptive_complexity: bool = False,
                 complexity_budget_ratio: float = DEFAULT_COMPLEXITY_BUDGET_RATIO,
                 complexity_token_quantum: int = DEFAULT_COMPLEXITY_TOKEN_QUANTUM):
        super().__init__()
        self.bridge = bridge or RadioRWKVBridge()
        self.adaptive_complexity = bool(adaptive_complexity)
        self.complexity_budget_ratio = float(complexity_budget_ratio)
        self.complexity_token_quantum = int(complexity_token_quantum)
        self.prefix_tokens = 0  # Variable; callers use ``visual_width`` below.
        self.last_token_counts: Tensor | None = None

    @property
    def visual_width(self) -> int:
        return int(self.prefix_tokens)

    def forward_native(
            self, features: Sequence[tuple[Tensor, Tensor, Tensor]]
            | tuple[Tensor, Tensor, Tensor]
            | tuple[Tensor, Tensor, Tensor, Tensor]) -> Tensor:
        """Whole-image native path: one variable-length prefix per row."""
        if not features:
            raise ValueError("RADIO projector needs cached features")
        device = self.bridge.gate.device
        box_fourier = None
        if (isinstance(features, tuple) and len(features) in (3, 4)
                and all(torch.is_tensor(value) for value in features)
                and features[0].ndim == 3):
            tokens, boxes, _roles = features[:3]
            if len(features) == 4:
                box_fourier = features[3].to(
                    device=device, non_blocking=True)
            if boxes.shape != (*tokens.shape[:2], 4):
                raise ValueError("batched native boxes do not match tokens")
            tokens = tokens.to(device=device, non_blocking=True)
            boxes = boxes.to(device=device, non_blocking=True)
        else:
            lengths = {int(item[0].shape[0]) for item in features}
            if len(lengths) != 1:
                raise ValueError(
                    f"native batches must share one token count, got {sorted(lengths)}")
            tokens = torch.stack([item[0] for item in features]).to(
                device=device, non_blocking=True)
            boxes = torch.stack([item[1] for item in features]).to(
                device=device, non_blocking=True)
        result = self.bridge.forward_native(
            tokens, boxes, box_fourier=box_fourier)
        self.prefix_tokens = result.embeddings.shape[1]
        self.last_token_counts = result.token_counts
        return result.embeddings

    def forward(self, features: Sequence[tuple[Tensor, Tensor, Tensor]],
                image_aspect: Tensor | None = None) -> Tensor:
        """``image_aspect`` (width/height per row) unlocks letterbox geometry.

        Without it the bridge cannot express where real pixels sit inside each
        padded canvas, which makes image-frame box regression unidentifiable.
        """
        if not features:
            raise ValueError("RADIO projector needs cached features")
        tile_counts = {int(item[0].shape[0]) for item in features}
        if len(tile_counts) != 1:
            raise ValueError("RADIO batches must have one exact tile count")
        token_counts = {int(item[0].shape[1]) for item in features}
        if len(token_counts) != 1:
            raise ValueError("RADIO batches must have one token budget")
        if self.adaptive_complexity:
            result = self.bridge.forward_variable(
                features, budget_ratio=self.complexity_budget_ratio,
                token_quantum=self.complexity_token_quantum,
                image_aspect=image_aspect)
            self.last_token_counts = result.token_counts
            self.prefix_tokens = result.embeddings.shape[1]
            return result.embeddings
        device = self.bridge.gate.device
        tokens = torch.stack([item[0] for item in features]).to(
            device=device, non_blocking=True)
        boxes = torch.stack([item[1] for item in features]).to(
            device=device, non_blocking=True)
        roles = torch.stack([item[2] for item in features]).to(
            device=device, non_blocking=True)
        result = self.bridge(tokens, boxes, roles,
                             image_aspect=image_aspect)
        if not bool(result.mask.all()):
            raise RuntimeError("exact RADIO batch unexpectedly contains padding")
        self.prefix_tokens = result.embeddings.shape[1]
        self.last_token_counts = result.token_counts
        return result.embeddings


class RadioPrefixInjector(DeepVisionInjector):
    """Re-inject the aligned, unpadded RADIO prefix at frozen RWKV depths.

    Each site gets an independent zero-initialized low-rank adapter. Reinjection
    modifies only the visual span; subsequent text consumes the refreshed
    recurrent state causally. Padded visual prefixes are rejected because a
    masked interior gap has not been proven to be an RWKV state no-op.
    """

    def __init__(self, hidden_size: int = RADIO_HIDDEN_SIZE,
                 layer_indices: Sequence[int] = (8, 16, 24), *, rank: int = 256,
                 tokens_per_tile: int = RADIO_TOKENS_PER_TILE,
                 token_quantum: int | None = RADIO_COMPACT_TOKENS_PER_TILE,
                 max_native_tokens: int = RADIO_NATIVE_MAX_TOKENS,
                 grouped_precompute: bool = False):
        """``token_quantum=None`` accepts native-resolution prefixes.

        A tiled RADIO prefix is always a whole number of tiles, so its length
        is necessarily a multiple of the smallest tile budget. A native
        C-RADIOv4-H prefix is ``grid_h * (grid_w // 2)`` for one image and
        carries no tile structure at all, so that multiple is not a property
        it can satisfy; ``max_native_tokens`` bounds it instead. The padding and
        finiteness checks still apply.
        """
        super().__init__(
            hidden_size, layer_indices, rank=rank,
            grouped_precompute=grouped_precompute)
        self.hidden_size = hidden_size
        self.tokens_per_tile = tokens_per_tile
        self.token_quantum = None if token_quantum is None else int(token_quantum)
        self.max_native_tokens = int(max_native_tokens)
        if self.max_native_tokens < 1:
            raise ValueError("max_native_tokens must be positive")

    @contextmanager
    def use_aligned_prefix(
            self, prefix: RadioVisualPrefix | Tensor,
            starts: int | Sequence[int] = 0):
        if isinstance(prefix, RadioVisualPrefix):
            if self.token_quantum is None:
                raise ValueError(
                    "native-resolution injector received a tiled RADIO prefix")
            if not bool(prefix.mask.all()):
                raise ValueError(
                    "RADIO reinjection requires exact tile-count buckets without padding")
            embeddings = prefix.embeddings
            tile_count = int(prefix.tile_count[0].item())
            if tile_count < 1 or not bool(prefix.tile_count.eq(tile_count).all()):
                raise ValueError("RADIO prefix requires one exact tile count")
            if embeddings.shape[1] % tile_count:
                raise ValueError("RADIO prefix token span does not match its tiles")
            expected_tiles = tile_count
            if not bool(prefix.tile_count.eq(expected_tiles).all()):
                raise ValueError("RADIO prefix tile count does not match its token span")
        else:
            embeddings = prefix
        if embeddings.ndim != 3 or embeddings.shape[-1] != self.hidden_size:
            raise ValueError(
                f"aligned RADIO prefix must be [B,T,{self.hidden_size}]")
        if self.token_quantum is None:
            # A native prefix satisfies no tile multiple, but it is still one
            # image's grid_h * (grid_w // 2) cells, so it is bounded above by
            # the encoder's own resolution cap. Without this the native path
            # had no length invariant at all, and a mis-shaped batch (a flat
            # [B*T, C] view, or tokens and channels transposed) reached the
            # injector as a plausible-looking prefix.
            if not 1 <= embeddings.shape[1] <= self.max_native_tokens:
                raise ValueError(
                    f"native RADIO prefix length {embeddings.shape[1]} is "
                    f"outside 1..{self.max_native_tokens}, the cell count of "
                    f"the largest image C-RADIOv4-H can encode")
        else:
            token_quantum = min(self.token_quantum, self.tokens_per_tile)
            if (embeddings.shape[1] < token_quantum or
                    embeddings.shape[1] % token_quantum):
                raise ValueError(
                    f"RADIO prefix length must be a positive multiple of {token_quantum}")
        if not bool(torch.isfinite(embeddings).all()):
            raise ValueError("aligned RADIO prefix contains non-finite values")
        with super().use_prefix(embeddings, starts):
            yield

    @contextmanager
    def use_prefix(self, prefix: RadioVisualPrefix | Tensor,
                   starts: int | Sequence[int] = 0):
        """Compatibility entry point used by the shared vision trainer."""
        with self.use_aligned_prefix(prefix, starts):
            yield
