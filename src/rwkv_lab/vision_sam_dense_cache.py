"""Cache SAM's native 256x64x64 dense image features without 1-D pooling."""
from __future__ import annotations

import argparse
import json
import os
import pickle
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
from PIL import Image

from rwkv_lab.moonvit import valid_torch_archive_storages
from rwkv_lab.vision_fusion import (
    sam_dense_cache_key, sam_tower_fingerprint, valid_sam_dense_feature)

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


def valid_entry(path: Path) -> bool:
    try:
        item = torch.load(path, map_location="cpu", weights_only=True)
        return (valid_sam_dense_feature(item)
                and valid_torch_archive_storages(path, item))
    except (OSError, EOFError, RuntimeError, pickle.UnpicklingError,
            zipfile.BadZipFile):
        return False


def decode(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--sam", default="models/vision/sam-vit-base")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    if args.batch < 1 or args.workers < 1:
        ap.error("--batch and --workers must be positive")

    fingerprint = sam_tower_fingerprint(args.sam)
    cache = Path(args.cache)
    cache.mkdir(parents=True, exist_ok=True)
    pending: list[tuple[Path, Path]] = []
    existing = 0
    for image, size, mtime in manifest_images(args.data):
        key = sam_dense_cache_key(
            image, tower_fingerprint=fingerprint,
            source_size=size, source_mtime_ns=mtime)
        target = cache / key
        if target.is_file() and valid_entry(target):
            existing += 1
        else:
            target.unlink(missing_ok=True)
            pending.append((image, target))
    print({"kind": "sam_dense_cache", "existing": existing,
           "missing": len(pending), "shape": [256, 64, 64]}, flush=True)
    if not pending:
        return

    from transformers import SamModel, SamProcessor

    processor = SamProcessor.from_pretrained(args.sam)
    model = SamModel.from_pretrained(
        args.sam, torch_dtype=torch.bfloat16).requires_grad_(False).eval().cuda()
    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for start in range(0, len(pending), args.batch):
            batch = pending[start:start + args.batch]
            images = list(pool.map(decode, (row[0] for row in batch)))
            pixels = processor(images=images, return_tensors="pt").pixel_values
            with torch.no_grad():
                features = model.get_image_embeddings(
                    pixels.to(device="cuda", dtype=torch.bfloat16))
            for (_, target), item in zip(batch, features.unbind(0)):
                item = item.detach().to(device="cpu", dtype=torch.bfloat16)
                if not valid_sam_dense_feature(item):
                    raise FloatingPointError(f"invalid dense SAM feature for {target}")
                temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
                try:
                    torch.save(item, temporary)
                    os.replace(temporary, target)
                finally:
                    temporary.unlink(missing_ok=True)
            done += len(batch)
            print({"kind": "sam_dense_cache", "done": done,
                   "total": len(pending)}, flush=True)


if __name__ == "__main__":
    main()
