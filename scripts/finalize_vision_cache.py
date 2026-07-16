#!/usr/bin/env python3
"""Verify a complete pooled-feature cache and publish a fast-start receipt."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from rwkv_lab.moonvit import checkpoint_fingerprint, feature_cache_key
from rwkv_lab.vision_cache import cache_entry_valid


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train", type=Path, required=True)
    ap.add_argument("--eval", type=Path, required=True)
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--moonvit", type=Path, default=ROOT /
                    "models/kimi-k2.6-moonvit/model-00064-of-000064.safetensors")
    ap.add_argument("--prefix-tokens", type=int, default=64)
    ap.add_argument("--max-input-patches", type=int, default=1024)
    ap.add_argument("--tap-layers", default="")
    ap.add_argument("--view-mode", choices=("full", "full-quadrants"),
                    default="full")
    ap.add_argument("--workers", type=int, default=32)
    return ap.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(16 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def manifest_images(paths: list[Path]) -> list[Path]:
    unique = {}
    for manifest in paths:
        with manifest.open() as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                image = Path(row["image"])
                image = image if image.is_absolute() else ROOT / image
                unique[str(image.resolve())] = image.resolve()
    return list(unique.values())


def main() -> None:
    args = parse_args()
    if args.workers < 1:
        raise SystemExit("--workers must be positive")
    cache = args.cache.resolve()
    tap_layers = tuple(sorted({int(value.strip()) for value in
                               args.tap_layers.split(",") if value.strip()}))
    stages = len(tap_layers) if tap_layers else 1
    fingerprint = checkpoint_fingerprint(args.moonvit)
    images = manifest_images([args.train, args.eval])
    expected = {
        feature_cache_key(
            image, max_input_patches=args.max_input_patches,
            prefix_tokens=args.prefix_tokens, vision_fingerprint=fingerprint,
            tap_layers=tap_layers, view_mode=args.view_mode)
        for image in images
    }
    actual_paths = {
        entry.name: entry for entry in cache.iterdir()
        if entry.is_file() and entry.suffix == ".pt"
    }
    missing = expected - actual_paths.keys()
    extra = actual_paths.keys() - expected
    if missing or extra:
        raise SystemExit(
            f"cache key set differs: missing={len(missing)} extra={len(extra)}"
        )

    started_mtime = cache.stat().st_mtime_ns
    paths = [actual_paths[name] for name in sorted(expected)]
    started = time.time()
    invalid = []

    def verify(path: Path) -> tuple[Path, bool]:
        return path, cache_entry_valid(path, args.prefix_tokens, stages)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = pool.map(verify, paths, buffersize=args.workers * 4)
        for done, (path, valid) in enumerate(results, 1):
            if not valid:
                invalid.append(path)
            if done % 10_000 == 0 or done == len(paths):
                print({"kind": "cache_verify", "done": done, "total": len(paths),
                       "invalid": len(invalid),
                       "entries_per_s": round(done / max(time.time() - started, 1e-6), 2)},
                      flush=True)
    if invalid:
        raise SystemExit(f"cache contains {len(invalid)} invalid payloads")
    finished_mtime = cache.stat().st_mtime_ns
    if finished_mtime != started_mtime:
        raise SystemExit("cache directory changed during payload verification")

    receipt = {
        "schema": 1,
        "cache": str(cache),
        "train_sha256": file_sha256(args.train),
        "eval_sha256": file_sha256(args.eval),
        "vision_fingerprint": fingerprint,
        "prefix_tokens": args.prefix_tokens,
        "max_input_patches": args.max_input_patches,
        "tap_layers": list(tap_layers),
        "view_mode": args.view_mode,
        "expected_entries": len(images),
        "entries": len(paths),
        "total_bytes": sum(path.stat().st_size for path in paths),
        "directory_mtime_ns": finished_mtime,
        "payloads_verified": True,
        "verification_seconds": time.time() - started,
        "verified_at": time.time(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    with temporary.open("w") as handle:
        json.dump(receipt, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(args.output)
    print(json.dumps(receipt, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
