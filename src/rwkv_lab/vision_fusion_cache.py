"""Prefill aligned frozen SigLIP2+DINOv2+SAM features for RWKV training."""
from __future__ import annotations

import argparse
import json
import os
import pickle
import zipfile
from pathlib import Path

import torch
from PIL import Image

from rwkv_lab.moonvit import valid_torch_archive_storages
from rwkv_lab.vision_fusion import (
    AlignedFrozenVisionFeatures, VisionTowerConfig,
    aligned_feature_cache_key, valid_aligned_feature)

ROOT = Path(__file__).resolve().parents[2]


def manifest_images(paths: list[str]) -> list[tuple[Path, int, int]]:
    unique: dict[str, tuple[Path, int, int]] = {}
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
                try:
                    image = image.resolve()
                    stat = image.stat()
                except OSError:
                    continue
                unique[str(image)] = (image, stat.st_size, stat.st_mtime_ns)
    return list(unique.values())


def valid_entry(path: Path, tokens: int, width: int) -> bool:
    try:
        item = torch.load(path, map_location="cpu", weights_only=True)
        return (valid_aligned_feature(item, tokens, width)
                and valid_torch_archive_storages(path, item))
    except (OSError, EOFError, RuntimeError, pickle.UnpicklingError,
            zipfile.BadZipFile):
        return False


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--prefix-tokens", type=int, default=128)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--siglip2", default="models/vision/siglip2-so400m-patch16-512")
    ap.add_argument("--siglip2-width", type=int, default=1152)
    ap.add_argument("--dinov2", default="models/vision/dinov2-base")
    ap.add_argument("--sam", default="models/vision/sam-vit-base")
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--shard-index", type=int, default=0)
    args = ap.parse_args()
    if args.prefix_tokens < 1 or args.batch < 1 or args.siglip2_width < 1:
        ap.error("--prefix-tokens, --batch, and --siglip2-width must be positive")
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        ap.error("--shard-index must be in [0, --num-shards)")

    config = VisionTowerConfig(
        siglip2=args.siglip2, dinov2=args.dinov2, sam=args.sam,
        siglip_width=args.siglip2_width)
    feature_width = args.siglip2_width + 768 + 256
    fingerprint = config.fingerprint()
    cache = Path(args.cache)
    cache.mkdir(parents=True, exist_ok=True)
    pending = []
    existing = 0
    for image, size, mtime in manifest_images(args.data):
        key = aligned_feature_cache_key(
            image, tokens=args.prefix_tokens, tower_fingerprint=fingerprint,
            source_size=size, source_mtime_ns=mtime)
        if int(key[:16], 16) % args.num_shards != args.shard_index:
            continue
        target = cache / key
        if target.is_file() and valid_entry(
                target, args.prefix_tokens, feature_width):
            existing += 1
        else:
            target.unlink(missing_ok=True)
            pending.append((image, target))
    print({"kind": "fusion_cache", "existing": existing,
           "missing": len(pending), "shard": args.shard_index}, flush=True)
    if not pending:
        return

    tower = AlignedFrozenVisionFeatures(config).load_pretrained(
        device="cuda", dtype=torch.bfloat16)
    done = 0
    skipped = 0
    for start in range(0, len(pending), args.batch):
        candidates = pending[start:start + args.batch]
        batch = []
        images = []
        for image_path, target in candidates:
            # One corrupt/truncated file must not kill an hours-long prefill;
            # skip it and let the remaining batch members proceed.
            try:
                with Image.open(image_path) as image:
                    images.append(image.convert("RGB"))
            except (OSError, ValueError, Image.DecompressionBombError) as error:
                skipped += 1
                print({"kind": "fusion_cache", "skipped_image": str(image_path),
                       "error": repr(error), "shard": args.shard_index}, flush=True)
                continue
            batch.append((image_path, target))
        if not batch:
            continue
        features = tower(images, tokens=args.prefix_tokens, device="cuda")
        for (_, target), item in zip(batch, features.unbind(0)):
            item = item.detach().to(device="cpu", dtype=torch.bfloat16)
            if not valid_aligned_feature(item, args.prefix_tokens, tower.width):
                raise FloatingPointError(f"invalid fusion feature for {target}")
            temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
            try:
                torch.save(item, temporary)
                os.replace(temporary, target)
            finally:
                temporary.unlink(missing_ok=True)
        done += len(batch)
        print({"kind": "fusion_cache", "done": done,
               "total": len(pending), "shard": args.shard_index}, flush=True)
    if skipped:
        print({"kind": "fusion_cache", "skipped_total": skipped,
               "total": len(pending), "shard": args.shard_index}, flush=True)


if __name__ == "__main__":
    main()
