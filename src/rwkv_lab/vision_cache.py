"""Prefill frozen MoonViT features before RWKV caption training.

Images are sorted by their post-resize patch count so each MoonViT batch has
similar spatial geometry. Existing atomic cache entries are skipped, making the
command safely restartable.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
from PIL import Image

from rwkv_lab.moonvit import (MoonViT, _resize_geometry, checkpoint_fingerprint,
                              feature_cache_key, pool_features,
                              valid_pooled_feature_archive,
                              valid_pooled_feature_payload)

ROOT = Path(__file__).resolve().parents[2]


def manifest_images(paths: list[str]) -> list[Path]:
    unique: dict[str, Path] = {}
    for source in paths:
        with Path(source).open() as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                if not row.get("image"):
                    continue
                image = Path(row["image"])
                image = image if image.is_absolute() else ROOT / image
                if image.is_file():
                    unique[str(image.resolve())] = image.resolve()
    return list(unique.values())


def decode(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB")


def cache_entry_valid(path: Path, prefix_tokens: int, stages: int = 0) -> bool:
    """Verify the payload, not merely its filename, before skipping an image."""
    try:
        item = torch.load(path, map_location="cpu", weights_only=True)
    except (OSError, EOFError, RuntimeError, pickle.UnpicklingError,
            zipfile.BadZipFile):
        return False
    return valid_pooled_feature_archive(path, item, prefix_tokens, stages)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--moonvit", default="models/kimi-k2.6-moonvit/model-00064-of-000064.safetensors")
    ap.add_argument("--cache", default="caches/moonvit_features_stage1_v3")
    ap.add_argument("--prefix-tokens", type=int, default=64)
    ap.add_argument("--max-input-patches", type=int, default=1024)
    ap.add_argument("--tap-layers", default="")
    ap.add_argument("--view-mode", choices=("full", "full-quadrants"),
                    default="full")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--sort-window", type=int, default=64,
                    help="decode this many images once, sort them in RAM, then encode sub-batches")
    ap.add_argument("--num-shards", type=int, default=1,
                    help="split cache keys deterministically across this many processes")
    ap.add_argument("--shard-index", type=int, default=0,
                    help="zero-based shard handled by this process")
    args = ap.parse_args()
    try:
        tap_layers = tuple(sorted({int(value.strip()) for value in
                                   args.tap_layers.split(",") if value.strip()}))
    except ValueError as exc:
        ap.error(f"invalid --tap-layers: {exc}")
    if any(index < 0 or index >= 27 for index in tap_layers):
        ap.error("--tap-layers must contain block indices from 0 to 26")
    # Zero stages means the unstaged 3-dim pooled layout; any tap count,
    # including one, produces the staged 4-dim layout.
    stages = len(tap_layers)
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        ap.error("--shard-index must be in [0, --num-shards)")
    if args.batch < 1 or args.workers < 1 or args.sort_window < 1:
        ap.error("--batch, --workers, and --sort-window must be positive")

    cache = Path(args.cache)
    cache.mkdir(parents=True, exist_ok=True)
    fingerprint = checkpoint_fingerprint(args.moonvit)
    images = manifest_images(args.data)
    pending = []
    assigned = 0
    existing = 0
    for image in images:
        key = feature_cache_key(image, max_input_patches=args.max_input_patches,
                                prefix_tokens=args.prefix_tokens,
                                vision_fingerprint=fingerprint,
                                tap_layers=tap_layers,
                                view_mode=args.view_mode)
        if int(key[:16], 16) % args.num_shards != args.shard_index:
            continue
        assigned += 1
        target = cache / key
        if not target.is_file() or not cache_entry_valid(
                target, args.prefix_tokens, stages):
            target.unlink(missing_ok=True)
            pending.append((image, target))
        else:
            existing += 1
    label = f"shard {args.shard_index + 1}/{args.num_shards}"
    print(f"cache {label}: {existing} existing / {len(pending)} missing ({assigned} assigned)", flush=True)
    if not pending:
        return

    vision = MoonViT.from_checkpoint(args.moonvit, device="cuda",
                                     max_input_patches=args.max_input_patches,
                                     tap_layers=tap_layers,
                                     view_mode=args.view_mode)
    vision.requires_grad_(False)
    vision.eval()
    started = time.time()
    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        def submit_window(window_start: int):
            window = pending[window_start:window_start + args.sort_window]
            futures = [pool.submit(decode, item[0]) for item in window]
            return window, futures

        starts = list(range(0, len(pending), args.sort_window))
        current = submit_window(starts[0])
        for window_number, window_start in enumerate(starts):
            window, futures = current
            decoded = [future.result() for future in futures]
            # Keep CPU image I/O and conversion one window ahead of MoonViT.
            # This uses only host RAM and removes the decode bubble between
            # otherwise fully occupied GPU windows.
            if window_number + 1 < len(starts):
                current = submit_window(starts[window_number + 1])
            prepared = []
            for item, image in zip(window, decoded):
                new_w, new_h, pad_w, pad_h = _resize_geometry(
                    *image.size, max_input_patches=args.max_input_patches)
                count = ((new_w + pad_w) // 14) * ((new_h + pad_h) // 14)
                prepared.append((count, item, image))
            prepared.sort(key=lambda row: row[0])
            for start in range(0, len(prepared), args.batch):
                batch = prepared[start:start + args.batch]
                encoded = vision.encode_many([row[2] for row in batch])
                for (_, (_, target), _), feature in zip(batch, encoded):
                    pooled = pool_features(feature, args.prefix_tokens).squeeze(0).cpu()
                    if not valid_pooled_feature_payload(
                            pooled, args.prefix_tokens, stages):
                        raise FloatingPointError(
                            f"MoonViT produced an invalid pooled feature for {target}")
                    temporary = target.with_name(
                        f".{target.name}.{os.getpid()}.tmp")
                    try:
                        torch.save(pooled, temporary)
                        os.replace(temporary, target)
                    finally:
                        temporary.unlink(missing_ok=True)
                done += len(batch)
                elapsed = time.time() - started
                print({"shard": args.shard_index, "done": done, "total": len(pending),
                       "images_per_s": round(done / max(elapsed, 1e-6), 2),
                       "elapsed_s": round(elapsed, 1)}, flush=True)


if __name__ == "__main__":
    main()
