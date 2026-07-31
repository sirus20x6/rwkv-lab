#!/usr/bin/env python3
"""Cache fused SigLIP2-g + SAM3 + compact-DINO C-RADIOv4-H grids.

Each native patch becomes one 4096-wide token suitable for the 4096-wide RWKV
7.2B/13.3B family. Images are encoded directly so the corrected 16-pixel
height / 32-pixel width snapping policy is reflected in the cache.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rwkv_lab.radio_v4h import (  # noqa: E402
    V4H_MAX_EDGE, build_native_image, cache_path, load_radio_v4h,
    native_cache_is_current, save_native_cache)
from rwkv_lab.radio_v4h_adaptors import (  # noqa: E402
    FUSED_ADAPTOR_WIDTH, load_v4h_adaptor_fusion)

DEFAULT_REVISION = "c-radiov4-h-all-adaptors-4096-native-v2-calibrated"


def manifest_rows(paths: list[Path], shard_index: int, num_shards: int):
    # Deduplicate BEFORE assigning a shard. Sharding on the manifest line number
    # first would hand the same repeated image to two shards, which then encode
    # it concurrently: safe, because each writer is atomic, but wasted GPU time.
    unique_index = 0
    seen: set[Path] = set()
    for manifest in paths:
        with manifest.open() as handle:
            for line_number, line in enumerate(handle, 1):
                line = line.strip()
                if not line:
                    continue
                value = json.loads(line)
                source = value.get("image") or value.get("image_path")
                if not source:
                    raise ValueError(
                        f"{manifest}:{line_number} has no image path")
                source = Path(source)
                source = source if source.is_absolute() else ROOT / source
                source = source.resolve()
                if source in seen:
                    continue
                seen.add(source)
                selected = unique_index % num_shards == shard_index
                unique_index += 1
                if not selected:
                    continue
                yield source, value.get("image_sha256")


@torch.inference_mode()
def encode_fused(model, fusion, image: Image.Image, *, max_edge: int,
                 device: torch.device):
    sized, (grid_h, grid_w) = build_native_image(image, max_edge=max_edge)
    pixels = torch.from_numpy(
        np.ascontiguousarray(sized, dtype=np.float32) / 255.0
    ).permute(2, 0, 1)[None].to(device, non_blocking=True)
    with torch.autocast(device.type, dtype=torch.bfloat16):
        output = model(pixels)
        features = output[1] if not isinstance(output, dict) else output["backbone"].features
        fused = fusion(features)
    if fused.shape != (1, grid_h * grid_w, FUSED_ADAPTOR_WIDTH):
        raise RuntimeError(
            f"fused shape {tuple(fused.shape)} does not match "
            f"(1,{grid_h * grid_w},{FUSED_ADAPTOR_WIDTH})")
    grid = fused.reshape(
        1, grid_h, grid_w, FUSED_ADAPTOR_WIDTH
    ).to(device="cpu", dtype=torch.bfloat16)
    if not bool(torch.isfinite(grid).all()):
        raise RuntimeError("C-RADIO adaptor fusion produced non-finite features")
    return grid, (grid_h, grid_w)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, nargs="+", required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument(
        "--model", type=Path, default=ROOT / "models/vision/C-RADIOv4-H")
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--max-edge", type=int, default=V4H_MAX_EDGE)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--device", default="cuda",
                        help="encoder device for this shard, e.g. cuda:1; "
                             "parallel shards must be given distinct devices "
                             "or they all land on cuda:0")
    parser.add_argument("--max-failure-rate", type=float, default=0.01,
                        help="share of attempted images allowed to fail before "
                             "this shard reports a hard failure (exit 1) "
                             "instead of a tolerated one (exit 3)")
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--allow-reconfigure", action="store_true",
        help="replace entries written by a different revision/width/snapping "
             "contract; without this, cross-configuration overwrites fail")
    args = parser.parse_args()
    if not 0 <= args.shard_index < args.num_shards:
        parser.error("shard-index must be in [0, num-shards)")
    if args.max_edge < 32 or args.num_shards < 1:
        parser.error("max-edge and num-shards are invalid")
    if not 0 <= args.max_failure_rate <= 1:
        parser.error("--max-failure-rate must be in [0, 1]")

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = True
    model = load_radio_v4h(args.model, device=device)
    fusion, _ = load_v4h_adaptor_fusion(args.model, device=device)

    completed = skipped = failed = tokens = written = 0
    started = time.monotonic()
    for source, source_sha256 in manifest_rows(
            args.manifest, args.shard_index, args.num_shards):
        if args.limit is not None and completed >= args.limit:
            break
        target = cache_path(args.cache_dir, source)
        if native_cache_is_current(
                target, source, args.revision, max_edge=args.max_edge,
                hidden_size=FUSED_ADAPTOR_WIDTH,
                source_sha256=source_sha256):
            skipped += 1
            continue
        try:
            with Image.open(source) as image:
                grid, (grid_h, grid_w) = encode_fused(
                    model, fusion, image, max_edge=args.max_edge,
                    device=device)
            save_native_cache(
                target, grid, revision=args.revision, source=source,
                source_sha256=source_sha256, max_edge=args.max_edge,
                allow_reconfigure=args.allow_reconfigure)
        except Exception as error:  # one corrupt source must not kill a shard
            failed += 1
            if isinstance(error, torch.cuda.OutOfMemoryError):
                torch.cuda.empty_cache()
            print(json.dumps({
                "phase": "error", "shard": args.shard_index,
                "source": str(source),
                "error": f"{type(error).__name__}: {error}"[:500],
            }), flush=True)
            continue
        completed += 1
        tokens += grid_h * grid_w
        written += target.stat().st_size
        if completed == 1 or completed % 100 == 0:
            elapsed = time.monotonic() - started
            print(json.dumps({
                "phase": "progress", "shard": args.shard_index,
                "completed": completed, "skipped": skipped, "failed": failed,
                "gb": round(written / 1e9, 2),
                "mean_tokens": round(tokens / completed),
                "images_per_second": round(completed / max(elapsed, 1e-6), 3),
            }, sort_keys=True), flush=True)
    elapsed = time.monotonic() - started
    attempted = completed + failed
    failure_rate = failed / max(attempted, 1)
    print(json.dumps({
        "phase": "complete", "shard": args.shard_index,
        "device": str(device),
        "completed": completed, "skipped": skipped, "failed": failed,
        "failure_rate": round(failure_rate, 6),
        "max_failure_rate": args.max_failure_rate,
        "gb": round(written / 1e9, 2),
        "mean_tokens": round(tokens / max(completed, 1)),
        "elapsed_seconds": round(elapsed, 1),
    }, sort_keys=True), flush=True)
    # A single corrupt JPEG and an OOM storm are different events. Exit 3 says
    # "some sources were skipped, the rest of the shard is sound"; exit 1 says
    # the cache this shard produced cannot be trusted at all.
    if not failed:
        return 0
    return 3 if failure_rate <= args.max_failure_rate else 1


if __name__ == "__main__":
    raise SystemExit(main())
