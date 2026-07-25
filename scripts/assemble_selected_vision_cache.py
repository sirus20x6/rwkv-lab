#!/usr/bin/env python3
"""Build a hard-linked cache view containing exactly the requested manifests."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rwkv_lab.moonvit import checkpoint_fingerprint, feature_cache_key
from rwkv_lab.vision_fusion import VisionTowerConfig, aligned_feature_cache_key


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_index(sources: list[Path]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for source in sources:
        if not source.is_dir():
            raise FileNotFoundError(source)
        for entry in source.iterdir():
            if not entry.is_file() or entry.suffix != ".pt":
                continue
            previous = result.get(entry.name)
            if previous is not None and not previous.samefile(entry):
                raise RuntimeError(f"cache-key collision: {previous} vs {entry}")
            result[entry.name] = entry
    return result


def link_exact(expected: set[str], sources: list[Path], output: Path) -> None:
    indexed = source_index(sources)
    missing = expected - indexed.keys()
    if missing:
        raise FileNotFoundError(f"{len(missing)} cache entries missing; first={next(iter(missing))}")
    output.mkdir(parents=True, exist_ok=True)
    for name in expected:
        target = output / name
        source = indexed[name]
        if target.exists():
            if not target.samefile(source):
                raise RuntimeError(f"nonmatching existing cache entry: {target}")
        else:
            os.link(source, target)
    actual = {item.name for item in output.iterdir()
              if item.is_file() and item.suffix == ".pt"}
    if actual != expected:
        raise RuntimeError(
            f"cache view differs: missing={len(expected-actual)} extra={len(actual-expected)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--eval", type=Path, required=True)
    parser.add_argument("--moon-source", type=Path, action="append", required=True)
    parser.add_argument("--fusion-source", type=Path, action="append", required=True)
    parser.add_argument("--moon-output", type=Path, required=True)
    parser.add_argument("--fusion-output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--moonvit", type=Path, default=ROOT / "models/kimi-k2.6-moonvit/model-00064-of-000064.safetensors")
    parser.add_argument("--siglip2", default=str(ROOT / "models/vision/siglip2-so400m-patch16-512"))
    parser.add_argument("--dinov2", default=str(ROOT / "models/vision/dinov2-base"))
    parser.add_argument("--sam", default=str(ROOT / "models/vision/sam-vit-base"))
    args = parser.parse_args()
    moon_fp = checkpoint_fingerprint(args.moonvit)
    fusion_fp = VisionTowerConfig(siglip2=args.siglip2, dinov2=args.dinov2,
                                  sam=args.sam, siglip_width=1152).fingerprint()
    moon_names, fusion_names, images = set(), set(), set()
    for manifest in (args.train, args.eval):
        for line in manifest.open():
            if not line.strip():
                continue
            row = json.loads(line)
            image = Path(row["image"])
            image = (image if image.is_absolute() else ROOT / image).resolve()
            stat = image.stat()
            images.add(str(image))
            moon_names.add(feature_cache_key(
                image, max_input_patches=1024, prefix_tokens=128,
                vision_fingerprint=moon_fp, source_size=stat.st_size,
                source_mtime_ns=stat.st_mtime_ns, tap_layers=(8, 17, 26),
                view_mode="full-quadrants"))
            fusion_names.add(aligned_feature_cache_key(
                image, tokens=128, tower_fingerprint=fusion_fp,
                source_size=stat.st_size, source_mtime_ns=stat.st_mtime_ns))
    if len(moon_names) != len(images) or len(fusion_names) != len(images):
        raise RuntimeError("content-addressed cache keys collided")
    link_exact(moon_names, args.moon_source, args.moon_output)
    link_exact(fusion_names, args.fusion_source, args.fusion_output)
    receipt = {
        "schema": 1, "train_sha256": sha256(args.train),
        "eval_sha256": sha256(args.eval), "expected_entries": len(images),
        "prefix_tokens": 128, "moonvit_taps": "8,17,26",
        "view_mode": "full-quadrants", "siglip2_width": 1152,
        "moon_cache": str(args.moon_output.resolve()),
        "fusion_cache": str(args.fusion_output.resolve()),
        "moon_sources": [str(path.resolve()) for path in args.moon_source],
        "fusion_sources": [str(path.resolve()) for path in args.fusion_source],
    }
    temporary = args.receipt.with_suffix(args.receipt.suffix + ".tmp")
    temporary.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, args.receipt)
    print({"kind": "selected_cache_view", "ready": True,
           "entries": len(images), "receipt": str(args.receipt)}, flush=True)


if __name__ == "__main__":
    main()
