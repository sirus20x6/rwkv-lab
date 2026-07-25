"""C-RADIOv4-H spatial-feature backend, the non-1D sibling of ``radio1d_rwkv``.

The 1D variant compresses an image into nested global tokens and needs a 311M
decoder to recover a spatial grid. C-RADIOv4-H emits ``(H/16, W/16)`` patch
features directly at 1280 channels from the same teachers (SigLIP2-g,
DINOv3-7B, SAM3), so the spatial basis is native rather than reconstructed.

Two consequences shape this module:

* Tokens are positional, not importance-ordered, so prefix truncation is
  meaningless here. Budget is set by pooling to a fixed lattice instead.
* Any per-axis multiple of 16 is supported, so tiles keep their own aspect and
  no token is ever spent on letterbox padding.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from PIL import Image
from torch import Tensor, nn

from rwkv_lab.radio1d_rwkv import RadioTile, build_radio_tiles

V4H_HIDDEN_SIZE = 1280
V4H_PATCH = 16
V4H_TILE_SIZE = 512          # this checkpoint's own preferred_resolution
V4H_LATTICE = 16             # pooled tokens per tile axis -> 256 per tile
CACHE_SCHEMA = "radio-v4h-spatial-v1"


@dataclass(frozen=True)
class V4HCacheMetadata:
    schema: str
    checkpoint_revision: str
    source_sha256: str
    source_size: int
    source_mtime_ns: int
    tile_size: int
    hidden_size: int
    source_boxes: tuple[tuple[float, float, float, float], ...]
    roles: tuple[int, ...]
    thumbnail_grid: tuple[int, int]
    detail_grid: tuple[int, int] | None

    def to_json(self) -> str:
        return json.dumps({
            "schema": self.schema,
            "checkpoint_revision": self.checkpoint_revision,
            "source_sha256": self.source_sha256,
            "source_size": self.source_size,
            "source_mtime_ns": self.source_mtime_ns,
            "tile_size": self.tile_size,
            "hidden_size": self.hidden_size,
            "source_boxes": [list(b) for b in self.source_boxes],
            "roles": list(self.roles),
            "thumbnail_grid": list(self.thumbnail_grid),
            "detail_grid": list(self.detail_grid) if self.detail_grid else None,
        }, sort_keys=True)

    @classmethod
    def from_json(cls, value: str) -> "V4HCacheMetadata":
        raw = json.loads(value)
        return cls(
            schema=raw["schema"],
            checkpoint_revision=raw["checkpoint_revision"],
            source_sha256=raw["source_sha256"],
            source_size=raw["source_size"],
            source_mtime_ns=raw["source_mtime_ns"],
            tile_size=raw["tile_size"],
            hidden_size=raw["hidden_size"],
            source_boxes=tuple(tuple(b) for b in raw["source_boxes"]),
            roles=tuple(raw["roles"]),
            thumbnail_grid=tuple(raw["thumbnail_grid"]),
            detail_grid=tuple(raw["detail_grid"]) if raw["detail_grid"] else None,
        )

    @property
    def tile_count(self) -> int:
        return len(self.roles)


def sha256_file(path: Path, block: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(block):
            digest.update(chunk)
    return digest.hexdigest()


def cache_path(cache_dir: Path, source: Path) -> Path:
    """Same identity scheme as the 1D cache: sha256 of the real source path."""
    resolved = os.path.realpath(os.fsencode(source))
    name = hashlib.sha256(resolved).hexdigest()
    return Path(cache_dir) / name[:2] / f"{name}.safetensors"


def load_radio_v4h(model_path: str | Path, *, device: str | torch.device = "cuda",
                   dtype: torch.dtype = torch.bfloat16) -> nn.Module:
    """Load and freeze C-RADIOv4-H from a pinned local directory.

    Loaded as a real package rather than through ``trust_remote_code`` for the
    same reason as the 1D loader: the copier drops a transitive relative import.
    Weights ship as one unsharded ``model.safetensors``, so the sharded loader
    does not apply.
    """
    from safetensors.torch import load_file

    from rwkv_lab.rwkv_finetune import _flash_attn_2_reported_unavailable

    path = Path(model_path).resolve()
    if not (path / "hf_model.py").is_file():
        raise FileNotFoundError(f"C-RADIOv4-H hf_model.py is missing from {path}")
    weights = path / "model.safetensors"
    if not weights.is_file():
        raise FileNotFoundError(f"C-RADIOv4-H model.safetensors is missing from {path}")

    with _flash_attn_2_reported_unavailable():
        package = f"_rwkv_radio_v4h_{abs(hash(str(path))):x}"
        module_name = f"{package}.hf_model"
        if module_name in sys.modules:
            remote = sys.modules[module_name]
        else:
            container = types.ModuleType(package)
            container.__path__ = [str(path)]
            container.__package__ = package
            sys.modules[package] = container
            spec = importlib.util.spec_from_file_location(module_name, path / "hf_model.py")
            if spec is None or spec.loader is None:
                raise ImportError(f"cannot import C-RADIOv4-H from {path}")
            remote = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = remote
            spec.loader.exec_module(remote)

        config = remote.RADIOConfig.from_pretrained(str(path), local_files_only=True)
        model = remote.RADIOModel(config)
        missing, unexpected = model.load_state_dict(
            load_file(str(weights)), strict=False)
        if missing or unexpected:
            raise RuntimeError(
                f"C-RADIOv4-H checkpoint mismatch missing={missing[:5]} "
                f"unexpected={unexpected[:5]}")
    model.eval().requires_grad_(False)
    return model.to(device=device, dtype=dtype)


def tiles_to_tensor(tiles: Sequence[RadioTile]) -> Tensor:
    """Stack same-shaped tiles as RADIO's documented [0,1] NCHW input."""
    if not tiles:
        raise ValueError("at least one tile is required")
    sizes = {tile.image.size for tile in tiles}
    if len(sizes) != 1:
        raise ValueError(f"tiles must share one size to stack, got {sizes}")
    arrays = [np.asarray(tile.image, dtype=np.float32) / 255.0 for tile in tiles]
    return torch.from_numpy(np.stack(arrays)).permute(0, 3, 1, 2).contiguous()


@torch.inference_mode()
def encode_v4h_group(model: nn.Module, tiles: Sequence[RadioTile], *,
                     device: str | torch.device = "cuda",
                     batch_size: int = 8,
                     cache_dtype: torch.dtype = torch.bfloat16) -> Tensor:
    """Encode same-shaped tiles to ``[tiles, gh, gw, 1280]`` spatial grids."""
    pixels = tiles_to_tensor(tiles)
    height, width = pixels.shape[-2:]
    grid_h, grid_w = height // V4H_PATCH, width // V4H_PATCH
    chunks = []
    for start in range(0, len(tiles), batch_size):
        batch = pixels[start:start + batch_size].to(device=device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _summary, features = model(batch)
        if features.ndim != 3 or features.shape[-1] != V4H_HIDDEN_SIZE:
            raise RuntimeError(
                f"expected [batch, tokens, {V4H_HIDDEN_SIZE}], got {tuple(features.shape)}")
        if features.shape[1] != grid_h * grid_w:
            raise RuntimeError(
                f"{features.shape[1]} tokens does not match the {grid_h}x{grid_w} grid")
        chunks.append(features.reshape(-1, grid_h, grid_w, V4H_HIDDEN_SIZE)
                      .to(device="cpu", dtype=cache_dtype))
    result = torch.cat(chunks, dim=0)
    if not bool(torch.isfinite(result).all()):
        raise RuntimeError("C-RADIOv4-H produced non-finite features")
    return result


def encode_v4h_tiles(model: nn.Module, tiles: Sequence[RadioTile], *,
                     device: str | torch.device = "cuda",
                     batch_size: int = 8) -> tuple[Tensor, Tensor | None]:
    """Encode one image's tiles, grouped into (thumbnail, details).

    Non-square tiling yields at most two shapes per image, so two batched
    forwards cover everything without padding a single token.
    """
    thumbnails = [t for t in tiles if t.role == "thumbnail"]
    details = [t for t in tiles if t.role != "thumbnail"]
    if len(thumbnails) != 1:
        raise ValueError("expected exactly one thumbnail tile")
    thumb = encode_v4h_group(model, thumbnails, device=device, batch_size=batch_size)
    if not details:
        return thumb, None
    return thumb, encode_v4h_group(model, details, device=device,
                                   batch_size=batch_size)


def pool_to_lattice(grids: Tensor, lattice: int = V4H_LATTICE) -> Tensor:
    """Average-pool ``[tiles, gh, gw, C]`` grids to ``[tiles, lattice^2, C]``.

    Token ``(r, c)`` always covers the same fractional region of its tile, at
    any input aspect, so the lattice is a consistent spatial basis even though
    the underlying grids differ in shape.
    """
    if grids.ndim != 4:
        raise ValueError(f"expected [tiles, gh, gw, channels], got {tuple(grids.shape)}")
    if lattice < 1:
        raise ValueError("lattice must be positive")
    pooled = torch.nn.functional.adaptive_avg_pool2d(
        grids.permute(0, 3, 1, 2).float(), (lattice, lattice))
    return pooled.flatten(2).transpose(1, 2).to(grids.dtype)


V4H_MAX_EDGE = 2048          # this checkpoint's max_resolution
V4H_NATIVE_STEP = 32         # /16 patches, and an even column count for pairing
NATIVE_CACHE_SCHEMA = "radio-v4h-native-v1"


def native_image_size(width: int, height: int, *, max_edge: int = V4H_MAX_EDGE,
                      step: int = V4H_NATIVE_STEP) -> tuple[int, int]:
    """Return ``(height, width)`` at native scale, downscaling only past the cap.

    Tiling existed because RADIO1D emitted a fixed 256 nested tokens per tile,
    so splitting the image was the only way to buy detail. C-RADIOv4-H's token
    count is ``(H/16)*(W/16)``, so resolution buys detail directly and tiling
    only adds a redundant thumbnail, 12.5% overlapping pixels, and an upscale of
    every detail crop. Measured on this corpus, 99%+ of images already sit under
    the 2048 cap, so this never invents pixels: it is a pure identity for all but
    ~6% of OCR pages.

    ``step`` is 32 rather than 16 so the patch-grid width stays even and
    ``pool_and_pair`` can pair columns without a ragged edge.
    """
    if width < 1 or height < 1 or max_edge < step or step < 1:
        raise ValueError("image geometry must be positive")
    scale = min(1.0, max_edge / max(width, height))
    sized = []
    for value in (height, width):
        # FLOOR, not round: rounding to the nearest step can round UP (1650 ->
        # 1664) and invent pixels, which is exactly what this path exists to
        # avoid. Flooring costs at most `step`-1 pixels of true resolution.
        snapped = int(value * scale // step) * step
        sized.append(max(step, min(max_edge, snapped)))
    return sized[0], sized[1]


def resize_array(array: "np.ndarray", height: int, width: int, *,
                 downscale: str = "area", upscale: str = "lanczos",
                 threads: int | None = None) -> "np.ndarray":
    """cv2 resize, ~3.5x PIL on this pipeline, with direction-aware filters.

    INTER_AREA down / INTER_LANCZOS4 up. Area is cv2's own recommendation for
    decimation: bicubic and Lanczos both ring, and ringing costs glyph edges
    exactly where OCR needs them. Most reductions here are mild snap-downs
    (median 0.968x) where the choice is immaterial, but the 2048 cap reaches
    0.711x on 6.4% of OCR pages, where it is not. Bicubic remains selectable.
    """
    import cv2

    if threads is not None:
        cv2.setNumThreads(int(threads))
    source_h, source_w = array.shape[:2]
    if (height, width) == (source_h, source_w):
        return array
    enlarging = height > source_h or width > source_w
    name = upscale if enlarging else downscale
    flags = {"bicubic": cv2.INTER_CUBIC, "lanczos": cv2.INTER_LANCZOS4,
             "area": cv2.INTER_AREA, "linear": cv2.INTER_LINEAR}
    if name not in flags:
        raise ValueError(f"unknown interpolation {name!r}")
    return cv2.resize(array, (width, height), interpolation=flags[name])


def build_native_image(image: Image.Image, *, max_edge: int = V4H_MAX_EDGE,
                       step: int = V4H_NATIVE_STEP,
                       downscale: str = "area", upscale: str = "lanczos",
                       threads: int | None = None
                       ) -> tuple["np.ndarray", tuple[int, int]]:
    """One whole-image input at native scale as an HWC uint8 array.

    Returns numpy rather than PIL so the encoder consumes it directly; the
    resize itself runs through cv2, which measured ~3.5x PIL on this pipeline.
    """
    from PIL import ImageOps

    source = ImageOps.exif_transpose(image).convert("RGB")
    height, width = native_image_size(*source.size, max_edge=max_edge, step=step)
    array = np.asarray(source)
    array = resize_array(array, height, width, downscale=downscale,
                         upscale=upscale, threads=threads)
    return array, (height // V4H_PATCH, width // V4H_PATCH)


def pair_columns(grid: Tensor) -> Tensor:
    """Concatenate horizontally adjacent cells: ``[n,gh,gw,C] -> [n,gh*gw/2,2C]``.

    The parameter-free route from 1280 channels to the LM's 2560, applied at the
    grid's own resolution rather than after pooling, so no spatial detail is
    averaged away first.
    """
    if grid.ndim != 4:
        raise ValueError(f"expected [n, gh, gw, channels], got {tuple(grid.shape)}")
    n, gh, gw, channels = grid.shape
    if gw % 2:
        raise ValueError(f"grid width {gw} must be even to pair columns")
    return grid.reshape(n, gh, gw // 2, 2 * channels).reshape(
        n, gh * (gw // 2), 2 * channels)


def pool_and_pair(grids: Tensor, lattice: int = V4H_LATTICE,
                  axis: str = "columns") -> Tensor:
    """Pool to a 2:1 lattice and concatenate adjacent pairs, doubling width.

    C-RADIOv4-H emits 1280 channels but the bridge is width-preserving into a
    2560-wide LM. Concatenating two neighbouring cells reaches 2560 with no
    parameters and no information loss, so the bridge stays byte-identical to
    the RADIO1D arm and the encoder becomes the only variable under test.

    ``axis="columns"`` pools to ``(lattice, 2*lattice)`` and pairs horizontally,
    which RETAINS more horizontal resolution before pairing — the scarce axis
    for text. ``axis="rows"`` is the mirror image. Either way each output token
    covers a square ``1/lattice`` x ``1/lattice`` region of the tile.
    """
    if grids.ndim != 4:
        raise ValueError(f"expected [tiles, gh, gw, channels], got {tuple(grids.shape)}")
    if lattice < 1:
        raise ValueError("lattice must be positive")
    if axis not in ("columns", "rows"):
        raise ValueError("axis must be 'columns' or 'rows'")
    tiles, _, _, channels = grids.shape
    target = (lattice, 2 * lattice) if axis == "columns" else (2 * lattice, lattice)
    pooled = torch.nn.functional.adaptive_avg_pool2d(
        grids.permute(0, 3, 1, 2).float(), target)          # [t, C, H, W]
    if axis == "columns":
        split = pooled.reshape(tiles, channels, lattice, lattice, 2)
        ordered = split.permute(0, 2, 3, 4, 1)              # [t, r, c, pair, C]
    else:
        split = pooled.reshape(tiles, channels, lattice, 2, lattice)
        ordered = split.permute(0, 2, 4, 3, 1)              # [t, r, c, pair, C]
    return ordered.reshape(tiles, lattice * lattice, 2 * channels).to(grids.dtype)


@torch.inference_mode()
def encode_native(model: nn.Module, image: Image.Image, *,
                  device: str | torch.device = "cuda",
                  max_edge: int = V4H_MAX_EDGE,
                  downscale: str = "area", upscale: str = "lanczos",
                  threads: int | None = None,
                  cache_dtype: torch.dtype = torch.bfloat16
                  ) -> tuple[Tensor, tuple[int, int]]:
    """Encode one whole image at native scale to ``[1, gh, gw, 1280]``."""
    sized, (grid_h, grid_w) = build_native_image(
        image, max_edge=max_edge, downscale=downscale, upscale=upscale,
        threads=threads)
    pixels = torch.from_numpy(
        np.ascontiguousarray(sized, dtype=np.float32) / 255.0).permute(2, 0, 1)[None]
    with torch.autocast("cuda", dtype=torch.bfloat16):
        _summary, features = model(pixels.to(device=device, non_blocking=True))
    if features.ndim != 3 or features.shape[-1] != V4H_HIDDEN_SIZE:
        raise RuntimeError(
            f"expected [1, tokens, {V4H_HIDDEN_SIZE}], got {tuple(features.shape)}")
    if features.shape[1] != grid_h * grid_w:
        raise RuntimeError(
            f"{features.shape[1]} tokens does not match the {grid_h}x{grid_w} grid")
    grid = features.reshape(1, grid_h, grid_w, V4H_HIDDEN_SIZE).to(
        device="cpu", dtype=cache_dtype)
    if not bool(torch.isfinite(grid).all()):
        raise RuntimeError("C-RADIOv4-H produced non-finite features")
    return grid, (grid_h, grid_w)


def save_native_cache(path: Path, grid: Tensor, *, revision: str,
                      source: Path, source_sha256: str | None = None,
                      max_edge: int = V4H_MAX_EDGE) -> None:
    from safetensors.torch import save_file

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    stat = Path(source).stat()
    meta = json.dumps({
        "schema": NATIVE_CACHE_SCHEMA,
        "checkpoint_revision": revision,
        "source_sha256": source_sha256 or sha256_file(Path(source)),
        "source_size": stat.st_size,
        "source_mtime_ns": stat.st_mtime_ns,
        "max_edge": max_edge,
        "hidden_size": V4H_HIDDEN_SIZE,
        "grid": [int(grid.shape[1]), int(grid.shape[2])],
    }, sort_keys=True)
    temporary = path.parent / f".{path.name}.{os.getpid()}.tmp"
    save_file({"grid": grid.contiguous()}, str(temporary),
              metadata={"schema": NATIVE_CACHE_SCHEMA, "radio_metadata": meta})
    os.replace(temporary, path)


def native_cache_is_current(path: Path, source: Path, revision: str, *,
                            max_edge: int = V4H_MAX_EDGE,
                            source_sha256: str | None = None) -> bool:
    from safetensors import safe_open

    path = Path(path)
    if not path.is_file():
        return False
    try:
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            header = handle.metadata() or {}
        if header.get("schema") != NATIVE_CACHE_SCHEMA:
            return False
        meta = json.loads(header["radio_metadata"])
        stat = Path(source).stat()
        identity = (meta["source_mtime_ns"] == stat.st_mtime_ns
                    or (source_sha256 is not None and len(source_sha256) == 64
                        and meta["source_sha256"] == source_sha256))
        return (meta["checkpoint_revision"] == revision
                and meta["max_edge"] == max_edge
                and meta["hidden_size"] == V4H_HIDDEN_SIZE
                and meta["source_size"] == stat.st_size and identity)
    except (OSError, ValueError, KeyError, RuntimeError):
        return False


def load_native_grid(path: Path) -> tuple[Tensor, tuple[int, int]]:
    from safetensors import safe_open

    with safe_open(str(path), framework="pt", device="cpu") as handle:
        header = handle.metadata() or {}
        if header.get("schema") != NATIVE_CACHE_SCHEMA:
            raise ValueError(f"unsupported native cache schema {header.get('schema')}")
        meta = json.loads(header["radio_metadata"])
        grid = handle.get_tensor("grid")
    shape = (int(grid.shape[1]), int(grid.shape[2]))
    if list(shape) != list(meta["grid"]):
        raise ValueError("cached grid shape disagrees with its metadata")
    return grid, shape


def native_token_boxes(grid_h: int, grid_w: int) -> Tensor:
    """Normalized image-frame extent of every paired token: ``[gh*gw/2, 4]``.

    Native grids vary in shape, so a flat token index is not a position: index
    100 is row 5 in a 40-wide grid and row 1 in a 104-wide one. Geometry has to
    be carried continuously instead, which is what FourierBoxEmbedding already
    does for tile boxes -- here applied per token, at 164k parameters rather
    than the ~15.4M a learned per-index table would cost.
    """
    if grid_h < 1 or grid_w < 2 or grid_w % 2:
        raise ValueError("grid must be at least 1x2 with an even width")
    pairs = grid_w // 2
    rows = torch.arange(grid_h, dtype=torch.float32).repeat_interleave(pairs)
    cols = torch.arange(pairs, dtype=torch.float32).repeat(grid_h)
    return torch.stack((
        cols * 2 / grid_w, rows / grid_h,
        (cols * 2 + 2) / grid_w, (rows + 1) / grid_h), dim=-1)


def load_native_features(rows: Sequence[dict], cache_dir: Path, *, revision: str,
                         root: Path | None = None, max_edge: int = V4H_MAX_EDGE
                         ) -> list[tuple[Tensor, Tensor, Tensor]]:
    """Load native grids as one whole-image "tile" per row, paired to 2560.

    Read-only by design: a miss raises rather than encoding inline, so a
    prefetch worker never touches CUDA from a foreign thread.
    """
    root = Path(root) if root is not None else Path.cwd()
    output = []
    for row in rows:
        source = Path(row["image"])
        source = source if source.is_absolute() else root / source
        target = cache_path(cache_dir, source)
        if not native_cache_is_current(target, source, revision, max_edge=max_edge,
                                       source_sha256=row.get("image_sha256")):
            raise FileNotFoundError(
                f"no current native C-RADIOv4-H cache entry for {source}")
        grid, (grid_h, grid_w) = load_native_grid(target)
        tokens = pair_columns(grid)[0]                    # [gh*gw/2, 2560]
        boxes = native_token_boxes(grid_h, grid_w)        # [gh*gw/2, 4]
        roles = torch.zeros(tokens.shape[0], dtype=torch.long)
        output.append((tokens, boxes, roles))
    return output


def native_grid_for(row: dict, *, root: Path | None = None,
                    max_edge: int = V4H_MAX_EDGE) -> tuple[int, int]:
    """Patch grid a row will produce, for sampler bucketing without decoding."""
    root = Path(root) if root is not None else Path.cwd()
    width, height = row.get("width"), row.get("height")
    if not width or not height:
        source = Path(row["image"])
        with Image.open(source if source.is_absolute() else root / source) as image:
            width, height = image.size
    h, w = native_image_size(int(width), int(height), max_edge=max_edge)
    return h // V4H_PATCH, w // V4H_PATCH


def save_v4h_cache(path: Path, metadata: V4HCacheMetadata,
                   thumbnail: Tensor, details: Tensor | None) -> None:
    """Atomically write one image's raw spatial grids."""
    from safetensors.torch import save_file

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tensors = {"thumbnail": thumbnail.contiguous()}
    if details is not None:
        tensors["details"] = details.contiguous()
    temporary = path.parent / f".{path.name}.{os.getpid()}.tmp"
    save_file(tensors, str(temporary),
              metadata={"schema": CACHE_SCHEMA,
                        "radio_metadata": metadata.to_json()})
    os.replace(temporary, path)


def load_v4h_cache(path: Path) -> tuple[V4HCacheMetadata, Tensor, Tensor | None]:
    from safetensors import safe_open

    with safe_open(str(path), framework="pt", device="cpu") as handle:
        header = handle.metadata() or {}
        if header.get("schema") != CACHE_SCHEMA:
            raise ValueError(f"unsupported C-RADIOv4-H cache schema {header.get('schema')}")
        metadata = V4HCacheMetadata.from_json(header["radio_metadata"])
        keys = set(handle.keys())
        thumbnail = handle.get_tensor("thumbnail")
        details = handle.get_tensor("details") if "details" in keys else None
    expected = 1 + (0 if details is None else details.shape[0])
    if expected != metadata.tile_count:
        raise ValueError("cached tile count does not match its metadata")
    return metadata, thumbnail, details


def v4h_cache_is_current(path: Path, source: Path, revision: str, *,
                         tile_size: int = V4H_TILE_SIZE,
                         source_sha256: str | None = None) -> bool:
    from safetensors import safe_open

    path = Path(path)
    if not path.is_file():
        return False
    try:
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            header = handle.metadata() or {}
        if header.get("schema") != CACHE_SCHEMA:
            return False
        metadata = V4HCacheMetadata.from_json(header["radio_metadata"])
        stat = Path(source).stat()
        identity = (metadata.source_mtime_ns == stat.st_mtime_ns
                    or (source_sha256 is not None and len(source_sha256) == 64
                        and metadata.source_sha256 == source_sha256))
        return (metadata.checkpoint_revision == revision
                and metadata.tile_size == tile_size
                and metadata.hidden_size == V4H_HIDDEN_SIZE
                and metadata.source_size == stat.st_size
                and identity)
    except (OSError, ValueError, KeyError, RuntimeError):
        return False


def make_v4h_metadata(source: Path, revision: str, tiles: Sequence[RadioTile], *,
                      tile_size: int = V4H_TILE_SIZE,
                      source_sha256: str | None = None) -> V4HCacheMetadata:
    stat = Path(source).stat()
    thumbnails = [t for t in tiles if t.role == "thumbnail"]
    details = [t for t in tiles if t.role != "thumbnail"]
    if len(thumbnails) != 1:
        raise ValueError("expected exactly one thumbnail tile")
    thumb = thumbnails[0].image
    detail_grid = None
    if details:
        first = details[0].image
        if any(t.image.size != first.size for t in details):
            raise ValueError("detail tiles must share one size")
        detail_grid = (first.height // V4H_PATCH, first.width // V4H_PATCH)
    return V4HCacheMetadata(
        schema=CACHE_SCHEMA,
        checkpoint_revision=revision,
        source_sha256=source_sha256 or sha256_file(Path(source)),
        source_size=stat.st_size,
        source_mtime_ns=stat.st_mtime_ns,
        tile_size=tile_size,
        hidden_size=V4H_HIDDEN_SIZE,
        source_boxes=tuple(t.source_box for t in tiles),
        roles=tuple(t.role_id for t in tiles),
        thumbnail_grid=(thumb.height // V4H_PATCH, thumb.width // V4H_PATCH),
        detail_grid=detail_grid,
    )


def load_v4h_features(rows: Sequence[dict], cache_dir: Path, *, revision: str,
                      lattice: int = V4H_LATTICE, root: Path | None = None,
                      pair_axis: str = "columns"
                      ) -> list[tuple[Tensor, Tensor, Tensor]]:
    """Load cached grids and pool them into the bridge's ``(tokens, boxes, roles)``.

    Deliberately read-only: a miss raises rather than encoding inline, because
    CUDA's context is thread-local and the prefetch worker must never touch it
    (the 1D pipeline learned this the hard way — see prefetch_cached_radio_rows).
    """
    root = Path(root) if root is not None else Path.cwd()
    output: list[tuple[Tensor, Tensor, Tensor]] = []
    for row in rows:
        source = Path(row["image"])
        source = source if source.is_absolute() else root / source
        target = cache_path(cache_dir, source)
        if not v4h_cache_is_current(target, source, revision,
                                    source_sha256=row.get("image_sha256")):
            raise FileNotFoundError(
                f"no current C-RADIOv4-H cache entry for {source}; run "
                f"scripts/cache_v4h_features.py first")
        metadata, thumbnail, details = load_v4h_cache(target)
        pooled = [pool_and_pair(thumbnail, lattice, pair_axis)]
        if details is not None:
            pooled.append(pool_and_pair(details, lattice, pair_axis))
        tokens = torch.cat(pooled, dim=0)
        boxes = torch.tensor(metadata.source_boxes, dtype=torch.float32)
        roles = torch.tensor(metadata.roles, dtype=torch.long)
        if tokens.shape[0] != boxes.shape[0] or tokens.shape[0] != roles.shape[0]:
            raise RuntimeError(f"tile/geometry mismatch in {target}")
        output.append((tokens, boxes, roles))
    return output


def v4h_tile_counts(rows: Sequence[dict], cache_dir: Path, *,
                    root: Path | None = None) -> list[int]:
    """Tile count per row, read from cache metadata without loading features."""
    from safetensors import safe_open

    root = Path(root) if root is not None else Path.cwd()
    counts = []
    for row in rows:
        source = Path(row["image"])
        source = source if source.is_absolute() else root / source
        with safe_open(str(cache_path(cache_dir, source)),
                       framework="pt", device="cpu") as handle:
            header = handle.metadata() or {}
        counts.append(V4HCacheMetadata.from_json(header["radio_metadata"]).tile_count)
    return counts


def cache_one_image(model: nn.Module, source: Path, cache_dir: Path, *,
                    revision: str, tile_size: int = V4H_TILE_SIZE,
                    max_detail_tiles: int = 48, batch_size: int = 8,
                    source_sha256: str | None = None) -> tuple[Path, int]:
    """Encode and cache one image; returns its cache path and tile count."""
    with Image.open(source) as image:
        tiles = build_radio_tiles(image, tile_size=tile_size,
                                  max_detail_tiles=max_detail_tiles,
                                  letterbox=False)
    thumbnail, details = encode_v4h_tiles(model, tiles, batch_size=batch_size)
    metadata = make_v4h_metadata(source, revision, tiles, tile_size=tile_size,
                                 source_sha256=source_sha256)
    target = cache_path(cache_dir, source)
    save_v4h_cache(target, metadata, thumbnail, details)
    return target, len(tiles)


__all__ = [
    "CACHE_SCHEMA", "V4HCacheMetadata", "V4H_HIDDEN_SIZE", "V4H_LATTICE",
    "V4H_PATCH", "V4H_TILE_SIZE", "cache_one_image", "cache_path",
    "encode_v4h_group", "encode_v4h_tiles", "load_radio_v4h", "load_v4h_cache",
    "load_v4h_features", "make_v4h_metadata", "pool_and_pair",
    "pool_to_lattice",
    "save_v4h_cache", "sha256_file", "tiles_to_tensor",
    "v4h_cache_is_current", "v4h_tile_counts", "build_native_image",
    "encode_native", "load_native_features", "load_native_grid",
    "native_cache_is_current", "native_grid_for", "native_image_size",
    "native_token_boxes", "pair_columns", "resize_array",
    "save_native_cache", "V4H_MAX_EDGE",
    "NATIVE_CACHE_SCHEMA",
]
