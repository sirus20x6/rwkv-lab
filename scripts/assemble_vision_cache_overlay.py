#!/usr/bin/env python3
"""Assemble immutable teacher-cache shards into one hard-linked training view."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train", type=Path, required=True)
    ap.add_argument("--eval", type=Path, required=True)
    ap.add_argument("--moon-source", type=Path, action="append", required=True)
    ap.add_argument("--fusion-source", type=Path, action="append", required=True)
    ap.add_argument("--moon-output", type=Path, required=True)
    ap.add_argument("--fusion-output", type=Path, required=True)
    ap.add_argument("--receipt", type=Path, required=True)
    ap.add_argument("--prefix-tokens", type=int, default=128)
    ap.add_argument("--moonvit-taps", default="8,17,26")
    ap.add_argument("--view-mode", default="full-quadrants")
    ap.add_argument("--siglip2-model", type=Path, default=ROOT /
                    "models/vision/siglip2-so400m-patch16-512")
    ap.add_argument("--siglip2-width", type=int, default=1152)
    args = ap.parse_args()
    if args.prefix_tokens < 1 or args.siglip2_width < 1:
        ap.error("token and width values must be positive")
    return args


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(16 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def manifest_images(paths: Sequence[Path]) -> set[Path]:
    images = set()
    for manifest in paths:
        with manifest.open() as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                image = Path(str(row["image"]))
                image = image if image.is_absolute() else ROOT / image
                image = image.resolve()
                if not image.is_file():
                    raise FileNotFoundError(image)
                images.add(image)
    return images


def cache_entries(path: Path) -> list[Path]:
    if not path.is_dir():
        raise FileNotFoundError(path)
    temporary = [item for item in path.iterdir() if item.name.endswith(".tmp")]
    if temporary:
        raise RuntimeError(f"cache has {len(temporary)} temporary entries: {path}")
    return [item for item in path.iterdir()
            if item.is_file() and item.suffix == ".pt"]


def link_cache_sources(sources: Sequence[Path], output: Path) -> int:
    """Create an idempotent same-filesystem hard-link union."""
    output.mkdir(parents=True, exist_ok=True)
    source_entries = [(source, cache_entries(source)) for source in sources]
    for source, entries in source_entries:
        if source.resolve() == output.resolve():
            raise RuntimeError("cache output cannot also be a source")
        if source.stat().st_dev != output.stat().st_dev:
            raise RuntimeError(
                f"hard-link cache overlay crosses filesystems: {source} -> {output}")
        for entry in entries:
            target = output / entry.name
            if target.exists():
                if not target.samefile(entry):
                    raise RuntimeError(f"cache-key collision is not a hard link: {target}")
                continue
            os.link(entry, target)
    output_entries = cache_entries(output)
    expected_names = {entry.name for _, entries in source_entries for entry in entries}
    actual_names = {entry.name for entry in output_entries}
    if actual_names != expected_names:
        raise RuntimeError(
            f"cache overlay differs: missing={len(expected_names - actual_names)} "
            f"extra={len(actual_names - expected_names)}")
    return len(output_entries)


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    expected = len(manifest_images([args.train, args.eval]))
    moon_count = link_cache_sources(args.moon_source, args.moon_output)
    fusion_count = link_cache_sources(args.fusion_source, args.fusion_output)
    if moon_count != expected or fusion_count != expected:
        raise SystemExit(
            f"overlay count mismatch: expected={expected} "
            f"moon={moon_count} fusion={fusion_count}")
    receipt = {
        "schema": 1,
        "train_sha256": file_sha256(args.train),
        "eval_sha256": file_sha256(args.eval),
        "prefix_tokens": args.prefix_tokens,
        "moonvit_taps": args.moonvit_taps,
        "view_mode": args.view_mode,
        "siglip2_model": str(args.siglip2_model.resolve()),
        "siglip2_width": args.siglip2_width,
        "expected_entries": expected,
        "moon_cache": str(args.moon_output.resolve()),
        "fusion_cache": str(args.fusion_output.resolve()),
        "moon_cache_mtime_ns": args.moon_output.stat().st_mtime_ns,
        "fusion_cache_mtime_ns": args.fusion_output.stat().st_mtime_ns,
        "moon_sources": [str(path.resolve()) for path in args.moon_source],
        "fusion_sources": [str(path.resolve()) for path in args.fusion_source],
        "storage": "same-filesystem hard-link overlay",
    }
    atomic_json(args.receipt, receipt)
    print(json.dumps({"kind": "vision_cache_overlay", "state": "ready",
                      "receipt": str(args.receipt), **receipt},
                     indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
