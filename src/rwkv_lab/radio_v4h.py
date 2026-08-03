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


FINGERPRINT_SAMPLE_BYTES = 1 << 20   # head and tail probe per artifact


def _artifact_signature(label: str, candidate: Path) -> dict:
    """Cheap change evidence for one artifact: size, mtime and a byte sample.

    ``(size, mtime_ns)`` alone is exactly the signal this module's docstring
    calls unreliable: ``cp -p``, ``rsync -t`` and ``tar -p`` all restore both
    from the *replacement's* source, so a same-size swap of a weight file is
    invisible to stat. Hashing the first and last megabyte costs microseconds
    even on a multi-GB safetensors file and no swap that changes the encoder
    leaves both ends byte-identical (safetensors carries its own header and
    tensor tail, both of which move under any real change).
    """
    stat = candidate.stat()
    digest = hashlib.sha256()
    with candidate.open("rb") as handle:
        digest.update(handle.read(FINGERPRINT_SAMPLE_BYTES))
        if stat.st_size > FINGERPRINT_SAMPLE_BYTES:
            handle.seek(stat.st_size - FINGERPRINT_SAMPLE_BYTES)
            digest.update(handle.read(FINGERPRINT_SAMPLE_BYTES))
    return {
        "path": label,
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sample_sha256": digest.hexdigest(),
    }


def v4h_artifact_fingerprint(
        model_path: str | Path, *, include_adaptors: bool = False,
        fingerprint_cache: str | Path | None = None) -> str:
    """Content fingerprint every artifact that determines cached V4H vectors.

    Directory mtimes do not change when an existing weight file is replaced,
    and a human-readable cache revision can accidentally be reused. Hashing the
    actual model, remote source, and (for fused caches) teacher adaptors makes
    the checkpoint contract independent of both failure modes.

    Cost: the full digest streams several GB, so it is computed once and stored
    in ``fingerprint_cache``. The stored short-circuit is keyed on size, mtime
    AND a bounded content sample (see ``_artifact_signature``) rather than stat
    alone, so a mtime-preserving replacement re-hashes instead of returning the
    previous fingerprint. ``include_adaptors=True`` additionally requires the
    fused-cache artifacts; every required file is named individually when one is
    absent, because "which file, and why is it required" is the only thing the
    operator needs from this failure.
    """
    path = Path(model_path).resolve()
    artifacts = [
        ("model/config.json", path / "config.json",
         "encoder geometry and feature width"),
        ("model/model.safetensors", path / "model.safetensors",
         "the encoder weights themselves"),
        ("producer/radio_v4h.py", Path(__file__).resolve(),
         "the snapping, packing and cache conventions of this producer"),
    ]
    if include_adaptors:
        import rwkv_lab.radio_v4h_adaptors as adaptor_module
        artifacts.extend((
            ("model/" + adaptor_module.DEFAULT_ADAPTOR_CHECKPOINT,
             path / adaptor_module.DEFAULT_ADAPTOR_CHECKPOINT,
             "teacher adaptor weights for the fused 4096-wide cache "
             "(include_adaptors=True)"),
            ("model/" + adaptor_module.DEFAULT_COMPACTOR,
             path / adaptor_module.DEFAULT_COMPACTOR,
             "the pinned DINO QR compactor for the fused cache; build it with "
             "scripts/build_v4h_dino_compactor.py"),
            ("producer/radio_v4h_adaptors.py",
             Path(adaptor_module.__file__).resolve(),
             "the fusion order and calibration scales"),
        ))
    artifacts.extend(
        (f"model/{candidate.name}", candidate,
         "pinned remote model source")
        for candidate in sorted(path.glob("*.py")))
    missing = [(candidate, reason) for _, candidate, reason in artifacts
               if not candidate.is_file()]
    if missing:
        detail = "; ".join(f"{candidate} ({reason})" for candidate, reason in missing)
        raise FileNotFoundError(
            f"C-RADIOv4-H fingerprint needs {len(missing)} absent "
            f"artifact(s) under {path}: {detail}")
    signatures = [_artifact_signature(label, candidate)
                  for label, candidate, _ in artifacts]
    cache_file = (
        Path(fingerprint_cache) if fingerprint_cache is not None else None)
    if cache_file is not None and cache_file.is_file():
        try:
            cached = json.loads(cache_file.read_text())
            fingerprint = str(cached["fingerprint"])
            if (cached.get("schema") == "radio-v4h-artifact-fingerprint-v2"
                    and cached.get("model_path") == str(path)
                    and cached.get("artifacts") == signatures
                    and len(fingerprint) == 64):
                return fingerprint
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            pass
    digest = hashlib.sha256()
    digest.update(b"radio-v4h-artifacts-v1\0")
    for (label, candidate, _reason), signature in zip(artifacts, signatures):
        encoded_label = label.encode()
        digest.update(len(encoded_label).to_bytes(4, "big"))
        digest.update(encoded_label)
        digest.update(int(signature["size"]).to_bytes(8, "big"))
        with candidate.open("rb") as handle:
            while chunk := handle.read(8 * 1024 * 1024):
                digest.update(chunk)
    fingerprint = digest.hexdigest()
    if cache_file is not None:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        temporary = cache_file.parent / (
            f".{cache_file.name}.{os.getpid()}.tmp")
        temporary.write_text(json.dumps({
            "schema": "radio-v4h-artifact-fingerprint-v2",
            "model_path": str(path),
            "artifacts": signatures,
            "fingerprint": fingerprint,
        }, sort_keys=True))
        os.replace(temporary, cache_file)
    return fingerprint


def cache_path(cache_dir: Path, source: Path) -> Path:
    """Same identity scheme as the 1D cache: sha256 of the real source path.

    The path deliberately encodes only the SOURCE, not the producing
    configuration, so that an existing 117k-entry cache stays addressable when
    the reader's revision or feature width changes; re-pathing every entry would
    orphan a cache that is otherwise perfectly valid. The configuration
    discriminator lives in the entry's own metadata instead, and
    ``save_native_cache`` refuses to replace an entry written under a different
    ``(hidden_size, revision, snapping steps)`` configuration -- so pointing a
    4096-wide writer at a 1280-wide cache directory now fails loudly per image
    rather than silently destroying it.
    """
    resolved = os.path.realpath(os.fsencode(source))
    name = hashlib.sha256(resolved).hexdigest()
    return Path(cache_dir) / name[:2] / f"{name}.safetensors"


def _pinned_package_name(path: Path, kind: str) -> str:
    """Deterministic synthetic package name for one pinned model directory.

    ``hash(str)`` is salted per interpreter (PYTHONHASHSEED), so a name built
    from it differs between processes and between a worker and its parent. A
    digest keeps one directory mapped to one package, which is what lets the
    encoder and the adaptor loader share a single copy of the remote classes.
    """
    digest = hashlib.sha256(os.fsencode(str(Path(path).resolve()))).hexdigest()
    return f"_rwkv_{kind}_{digest[:16]}"


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
        package = _pinned_package_name(path, "radio_v4h")
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
V4H_NATIVE_HEIGHT_STEP = 16  # one native patch; rows are never channel-paired
V4H_NATIVE_WIDTH_STEP = 32   # two patches so column-packing has no ragged edge
V4H_NATIVE_STEP = V4H_NATIVE_WIDTH_STEP  # compatibility alias
NATIVE_CACHE_SCHEMA = "radio-v4h-native-v1"
# Single source of truth. The writer and the reader MUST agree: a mismatch does
# not look like a mismatch, it looks like a totally absent cache, because
# native_cache_is_current folds "wrong revision" into the same False.
#
# The suffix names the snapping convention, and MUST be bumped whenever the
# steps change. It was bumped from the unqualified "c-radiov4-h-native" when
# height stopped sharing width's 32-pixel step: a 1500px page snapped to 92
# rows under the shared step and to 93 under the split one, both entries claim
# the same revision, and native_cache_is_current has no way to tell them apart
# -- so one cache directory silently held two incompatible geometries.
DEFAULT_NATIVE_REVISION = "c-radiov4-h-native-h16w32"


def native_image_size(width: int, height: int, *, max_edge: int = V4H_MAX_EDGE,
                      step: int | None = None,
                      height_step: int = V4H_NATIVE_HEIGHT_STEP,
                      width_step: int = V4H_NATIVE_WIDTH_STEP) -> tuple[int, int]:
    """Return ``(height, width)`` at native scale, downscaling only past the cap.

    Tiling existed because RADIO1D emitted a fixed 256 nested tokens per tile,
    so splitting the image was the only way to buy detail. C-RADIOv4-H's token
    count is ``(H/16)*(W/16)``, so resolution buys detail directly and tiling
    only adds a redundant thumbnail, 12.5% overlapping pixels, and an upscale of
    every detail crop. Measured on this corpus, 99%+ of images already sit under
    the 2048 cap, so this never invents pixels: it is a pure identity for all but
    ~6% of OCR pages.

    Only width needs a 32-pixel step so the patch-grid width stays even for
    legacy column pairing. Height uses the native 16-pixel patch step. The old
    shared 32-pixel step introduced avoidable aspect distortion.
    """
    if step is not None:
        # Compatibility for callers that intentionally request one shared
        # quantum. New code should use the per-axis defaults.
        height_step = width_step = int(step)
    if (width < 1 or height < 1 or max_edge < max(height_step, width_step)
            or height_step < 1 or width_step < 1):
        raise ValueError("image geometry must be positive")
    scale = min(1.0, max_edge / max(width, height))
    sized = []
    for value, axis_step in ((height, height_step), (width, width_step)):
        # FLOOR, not round: rounding to the nearest step can round UP (1650 ->
        # 1664) and invent pixels, which is exactly what this path exists to
        # avoid. Flooring costs at most `step`-1 pixels of true resolution.
        snapped = int(value * scale // axis_step) * axis_step
        sized.append(max(axis_step, min(max_edge, snapped)))
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
                       step: int | None = None,
                       height_step: int = V4H_NATIVE_HEIGHT_STEP,
                       width_step: int = V4H_NATIVE_WIDTH_STEP,
                       downscale: str = "area", upscale: str = "lanczos",
                       threads: int | None = None
                       ) -> tuple["np.ndarray", tuple[int, int]]:
    """One whole-image input at native scale as an HWC uint8 array.

    Returns numpy rather than PIL so the encoder consumes it directly; the
    resize itself runs through cv2, which measured ~3.5x PIL on this pipeline.
    """
    from PIL import ImageOps

    source = ImageOps.exif_transpose(image).convert("RGB")
    height, width = native_image_size(
        *source.size, max_edge=max_edge, step=step,
        height_step=height_step, width_step=width_step)
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


def native_cache_config(path: Path) -> dict | None:
    """The producing configuration of one cache entry, or None if unreadable."""
    from safetensors import safe_open

    try:
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            header = handle.metadata() or {}
        meta = json.loads(header["radio_metadata"])
        return {
            "hidden_size": int(meta["hidden_size"]),
            "checkpoint_revision": str(meta["checkpoint_revision"]),
            "max_edge": int(meta["max_edge"]),
            "height_step": meta.get("height_step"),
            "width_step": meta.get("width_step"),
        }
    except (OSError, ValueError, KeyError, TypeError, RuntimeError):
        return None


def save_native_cache(path: Path, grid: Tensor, *, revision: str,
                      source: Path, source_sha256: str | None = None,
                      max_edge: int = V4H_MAX_EDGE,
                      height_step: int = V4H_NATIVE_HEIGHT_STEP,
                      width_step: int = V4H_NATIVE_WIDTH_STEP,
                      allow_reconfigure: bool = False) -> None:
    """Atomically write one native grid, refusing cross-configuration clobbers.

    ``cache_path`` hashes only the source, so a 4096-wide fused writer aimed at
    a 1280-wide native cache directory addresses exactly the same files. Every
    entry would fail its currency check, get re-encoded, and ``os.replace`` the
    old one away: a silent one-way destruction of a 117k-image cache. An entry
    written under a different feature width, revision or snapping convention is
    therefore an error here, not an overwrite; pass ``allow_reconfigure=True``
    to deliberately rewrite a directory in place.

    The snapping steps are recorded so a future convention change is visible in
    the entry itself rather than only in a revision string somebody remembered
    to bump.
    """
    from safetensors.torch import save_file

    path = Path(path)
    hidden_size = int(grid.shape[-1])
    if not allow_reconfigure and path.is_file():
        existing = native_cache_config(path)
        incoming = {
            "hidden_size": hidden_size,
            "checkpoint_revision": revision,
            "max_edge": int(max_edge),
            "height_step": int(height_step),
            "width_step": int(width_step),
        }
        if existing is not None and any(
                existing[key] is not None and existing[key] != value
                for key, value in incoming.items()):
            raise ValueError(
                f"refusing to overwrite {path}: it was written as {existing} "
                f"but this writer produces {incoming}. Cache entries are keyed "
                f"on the source path alone, so writing here would destroy a "
                f"cache built for a different configuration. Use a separate "
                f"--cache-dir, or pass allow_reconfigure=True to rewrite it.")
    path.parent.mkdir(parents=True, exist_ok=True)
    stat = Path(source).stat()
    meta = json.dumps({
        "schema": NATIVE_CACHE_SCHEMA,
        "checkpoint_revision": revision,
        "source_sha256": source_sha256 or sha256_file(Path(source)),
        "source_size": stat.st_size,
        "source_mtime_ns": stat.st_mtime_ns,
        "max_edge": max_edge,
        "height_step": int(height_step),
        "width_step": int(width_step),
        "hidden_size": hidden_size,
        "grid": [int(grid.shape[1]), int(grid.shape[2])],
    }, sort_keys=True)
    temporary = path.parent / f".{path.name}.{os.getpid()}.tmp"
    save_file({"grid": grid.contiguous()}, str(temporary),
              metadata={"schema": NATIVE_CACHE_SCHEMA, "radio_metadata": meta})
    os.replace(temporary, path)


def native_cache_is_current(path: Path, source: Path, revision: str, *,
                            max_edge: int = V4H_MAX_EDGE,
                            hidden_size: int = V4H_HIDDEN_SIZE,
                            height_step: int = V4H_NATIVE_HEIGHT_STEP,
                            width_step: int = V4H_NATIVE_WIDTH_STEP,
                            source_sha256: str | None = None) -> bool:
    """Whether one entry was produced by exactly this configuration.

    Entries written before the snapping steps were recorded carry no step
    fields; those are read as "the caller's convention" rather than rejected,
    because the revision string is what separates them (it was bumped when the
    height step split off from the width step) and re-encoding an otherwise
    valid 117k-entry cache for a metadata addition would be gratuitous. Once a
    step field is present it is compared exactly, so the next convention change
    cannot pass silently.
    """
    from safetensors import safe_open

    path = Path(path)
    if not path.is_file():
        return False
    try:
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            header = handle.metadata() or {}
            keys = list(handle.keys())
            tensor_shape = (
                list(handle.get_slice("grid").get_shape())
                if keys == ["grid"] else None)
        if header.get("schema") != NATIVE_CACHE_SCHEMA:
            return False
        meta = json.loads(header["radio_metadata"])
        stat = Path(source).stat()
        identity = (meta["source_mtime_ns"] == stat.st_mtime_ns
                    or (source_sha256 is not None and len(source_sha256) == 64
                        and meta["source_sha256"] == source_sha256))
        steps = (meta.get("height_step", height_step) == height_step
                 and meta.get("width_step", width_step) == width_step)
        return (meta["checkpoint_revision"] == revision
                and meta["max_edge"] == max_edge
                and meta["hidden_size"] == hidden_size
                and steps
                and tensor_shape == [1, *meta["grid"], hidden_size]
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


def native_cache_token_count(path: Path, *, hidden_size: int,
                             exact_header: bool = False) -> int:
    """Return the cached cell count from file geometry without mmaping the tensor.

    Native caches contain exactly one contiguous bf16 ``grid`` tensor, so the
    cell count is ``(file size - header) / (hidden_size * 2)``.

    ``exact_header`` selects how the header length is obtained, and the default
    is load-bearing. Reading safetensors' little-endian u64 prefix is only 8
    bytes of *syscall*, but opening the file and touching one byte faults in a
    filesystem readahead window -- measured at ~1.6 MB of real block I/O per
    entry on the array holding these caches. Planning a sampler over ~110k rows
    therefore turned a pure-metadata pass into a multi-gigabyte read and added
    hours to every trainer startup, on a run that restarts every 500 steps.
    ``stat()`` alone touches no data blocks, so the default infers the header
    from the size remainder and stays metadata-only.

    The inferred form is exact whenever the header fits in one cell, which the
    bounds below assert. It can only be wrong if a very long free-form
    ``--revision`` string pushes the header past ``hidden_size * 2`` bytes --
    and since the revision is uniform across a cache directory, that is a
    property of the cache, not of an entry. Pass ``exact_header=True`` where a
    single authoritative answer is wanted (validation, audits, one probe entry)
    rather than in a per-row loop.

    The inferred form also cannot validate ``hidden_size``: an incorrect width
    is absorbed into the inferred header and yields a plausible but wrong count
    (1279 reads a 1280-wide entry as 1201 cells rather than raising). Only
    ``exact_header=True`` divides by a known payload and rejects it. This is
    tolerable for planning because ``load_native_features`` checks the width
    against the real tensor when the row loads, so a mismatch fails loudly
    there instead of training on mis-bucketed geometry -- but do not treat a
    default-form count as evidence that a cache has the width you asked for.
    """
    if hidden_size < 1:
        raise ValueError("native cache hidden size must be positive")
    path = Path(path)
    size = path.stat().st_size
    cell_bytes = int(hidden_size) * torch.bfloat16.itemsize
    if exact_header:
        with path.open("rb") as handle:
            prefix = handle.read(8)
        if len(prefix) != 8:
            raise ValueError(f"{path} is too short to be a safetensors file")
        header_bytes = 8 + int.from_bytes(prefix, "little")
        payload = size - header_bytes
        cells, remainder = divmod(max(payload, 0), cell_bytes)
        if payload < cell_bytes or remainder:
            raise ValueError(
                f"{path} does not have valid native-cache file geometry "
                f"for hidden size {hidden_size}: {payload} payload bytes after "
                f"a {header_bytes}-byte header is not a positive multiple of "
                f"{cell_bytes}")
        return cells
    cells, remainder = divmod(max(size, 0), cell_bytes)
    if cells < 1 or not 8 <= remainder < cell_bytes:
        raise ValueError(
            f"{path} does not have valid native-cache file geometry "
            f"for hidden size {hidden_size}: {size} bytes is not one bf16 "
            f"grid plus a header smaller than one {cell_bytes}-byte cell")
    return cells


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


def native_cell_boxes(grid_h: int, grid_w: int) -> Tensor:
    """Normalized image-frame extent of each unpaired native patch cell."""
    if grid_h < 1 or grid_w < 1:
        raise ValueError("grid must be positive")
    rows = torch.arange(grid_h, dtype=torch.float32).repeat_interleave(grid_w)
    cols = torch.arange(grid_w, dtype=torch.float32).repeat(grid_h)
    return torch.stack((
        cols / grid_w, rows / grid_h,
        (cols + 1) / grid_w, (rows + 1) / grid_h), dim=-1)


def load_native_features(rows: Sequence[dict], cache_dir: Path, *, revision: str,
                         root: Path | None = None, max_edge: int = V4H_MAX_EDGE,
                         hidden_size: int = V4H_HIDDEN_SIZE,
                         packing: str = "pair_columns",
                         ) -> list[tuple[Tensor, Tensor, Tensor]]:
    """Load native grids as one whole-image "tile" per row.

    Read-only by design: a miss raises rather than encoding inline, so a
    prefetch worker never touches CUDA from a foreign thread. ``pair_columns``
    concatenates adjacent cells and doubles feature width; ``cells`` preserves
    every patch as an independent token at the cached feature width.
    """
    root = Path(root) if root is not None else Path.cwd()
    output = []
    for row in rows:
        source = Path(row["image"])
        source = source if source.is_absolute() else root / source
        target = cache_path(cache_dir, source)
        if not native_cache_is_current(
                target, source, revision, max_edge=max_edge,
                hidden_size=hidden_size,
                source_sha256=row.get("image_sha256")):
            if not target.is_file():
                raise FileNotFoundError(
                    f"no native C-RADIOv4-H cache entry for {source}")
            raise FileNotFoundError(
                f"stale native cache entry for {source}: it exists but does not "
                f"validate. Check revision (asked {revision!r}) and max_edge "
                f"(asked {max_edge}) against what wrote it.")
        grid, (grid_h, grid_w) = load_native_grid(target)
        if grid.shape[-1] != hidden_size:
            raise ValueError(
                f"cached feature width {grid.shape[-1]} != {hidden_size}")
        if packing == "pair_columns":
            tokens = pair_columns(grid)[0]
            boxes = native_token_boxes(grid_h, grid_w)
        elif packing == "cells":
            tokens = grid.flatten(1, 2)[0]
            boxes = native_cell_boxes(grid_h, grid_w)
        else:
            raise ValueError(f"unknown native packing {packing!r}")
        roles = torch.zeros(tokens.shape[0], dtype=torch.long)
        output.append((tokens, boxes, roles))
    return output


EXIF_ORIENTATION_TAG = 0x0112
# The four orientations ImageOps.exif_transpose implements with a transpose.
EXIF_TRANSPOSED_ORIENTATIONS = frozenset({5, 6, 7, 8})


def exif_swaps_axes(image: Image.Image) -> bool:
    """Whether EXIF orientation makes the stored size the transposed one."""
    try:
        orientation = image.getexif().get(EXIF_ORIENTATION_TAG)
    except (OSError, ValueError, AttributeError):
        return False
    return orientation in EXIF_TRANSPOSED_ORIENTATIONS


def native_grid_for(row: dict, *, root: Path | None = None,
                    max_edge: int = V4H_MAX_EDGE,
                    exif: bool = True) -> tuple[int, int]:
    """Patch grid a row will produce, for sampler bucketing without decoding.

    ``build_native_image`` runs ``ImageOps.exif_transpose`` first, so a stored
    1500x1008 page with an EXIF 90-degree flag is encoded as 1008x1500. Under
    the old shared 32-pixel step the transposed prediction still gave the same
    cell count; with 16 rows / 32 columns it does not (63x93 = 5859 predicted
    against 93x62 = 5766 written), and the sampler then batches a row against a
    grid the encoder never produced. Orientation is read from the file header --
    a row may supply ``exif_orientation`` to skip even that -- because it is not
    recoverable from manifest width/height, which are the pre-transpose values.
    """
    root = Path(root) if root is not None else Path.cwd()
    source = Path(row["image"])
    source = source if source.is_absolute() else root / source
    width, height = row.get("width"), row.get("height")
    orientation = row.get("exif_orientation")
    swap = exif and orientation in EXIF_TRANSPOSED_ORIENTATIONS
    if not width or not height or (exif and orientation is None):
        with Image.open(source) as image:      # header only, never decoded
            if not width or not height:
                width, height = image.size
            if exif and orientation is None:
                swap = exif_swaps_axes(image)
    if swap:
        width, height = height, width
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
    "v4h_artifact_fingerprint",
    "v4h_cache_is_current", "v4h_tile_counts", "build_native_image",
    "encode_native", "load_native_features", "load_native_grid",
    "DEFAULT_NATIVE_REVISION", "exif_swaps_axes", "native_cache_config",
    "native_cache_is_current",
    "native_cache_token_count", "native_grid_for", "native_image_size",
    "native_cell_boxes", "native_token_boxes", "pair_columns", "resize_array",
    "save_native_cache", "V4H_MAX_EDGE",
    "NATIVE_CACHE_SCHEMA", "V4H_NATIVE_HEIGHT_STEP", "V4H_NATIVE_WIDTH_STEP",
]
