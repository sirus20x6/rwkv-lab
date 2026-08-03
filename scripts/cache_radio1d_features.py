#!/usr/bin/env python3
"""Build resumable full-budget tiled RADIO1D-H caches from a JSONL manifest."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rwkv_lab.radio1d_cache import (  # noqa: E402
    cache_is_current, cache_path, make_metadata, save_cache,
)
from rwkv_lab.radio1d_rwkv import (  # noqa: E402
    DEFAULT_ADAPTIVE_TOKEN_THRESHOLD, DEFAULT_MAX_DETAIL_TILES,
    build_radio_tiles, encode_radio_tiles, estimate_context, load_radio1d_h,
    tokens_per_tile_for_tile_count,
)


DEFAULT_REVISION = "e18692120c7a3203496e1a99056a4149ede135fc"


def lexical_cache_path(cache_dir: Path, source: Path) -> Path:
    """Compute the common non-symlink cache identity without disk traversal."""
    absolute = os.path.abspath(os.fsencode(source))
    name = hashlib.sha256(absolute).hexdigest()
    return cache_dir / name[:2] / f"{name}.safetensors"


def rows(path: Path, start: int, limit: int | None):
    emitted = 0
    with path.open() as handle:
        for index, line in enumerate(handle):
            if index < start:
                continue
            if limit is not None and emitted >= limit:
                return
            if not line.strip():
                continue
            value = json.loads(line)
            source = value.get("image") or value.get("image_path")
            if not source:
                raise ValueError(f"manifest row {index} has no image path")
            yield index, Path(source), value.get("image_sha256")
            emitted += 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path,
                        default=ROOT / "models/vision/C-RADIOv4-1D-H")
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--batch-size", type=int, default=8,
                        help="tiles per RADIO GPU forward")
    parser.add_argument("--max-detail-tiles", type=int,
                        default=DEFAULT_MAX_DETAIL_TILES)
    parser.add_argument("--adaptive-token-threshold", type=int,
                        default=DEFAULT_ADAPTIVE_TOKEN_THRESHOLD)
    parser.add_argument("--overlap", type=float, default=0.125)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--text-token-estimate", type=int, default=2048)
    parser.add_argument(
        "--trust-existing", action="store_true",
        help=("fast-resume atomic entries already present in this dedicated "
              "cache; symlink-path misses still receive full validation"))
    args = parser.parse_args()
    if args.batch_size < 1 or args.max_detail_tiles < 1 or args.start < 0:
        parser.error("batch size and tile count must be positive; start nonnegative")

    model = None
    started = time.monotonic()
    completed = skipped = visual_tokens = above_reference = 0
    trusted_names: set[str] = set()
    if args.trust_existing and args.cache_dir.is_dir():
        trusted_names = {
            path.name
            for shard in args.cache_dir.iterdir() if shard.is_dir()
            for path in shard.glob("*.safetensors")
        }
        print(json.dumps({
            "phase": "fast_resume_inventory", "entries": len(trusted_names),
        }, sort_keys=True), flush=True)
    for row_index, source, source_sha256 in rows(
            args.manifest, args.start, args.limit):
        lexical_target = lexical_cache_path(args.cache_dir, source)
        if lexical_target.name in trusted_names:
            skipped += 1
            continue
        target = cache_path(args.cache_dir, source)
        if cache_is_current(
                target, source, args.revision,
                adaptive_token_threshold=args.adaptive_token_threshold,
                source_sha256=source_sha256):
            skipped += 1
            continue
        if model is None:
            model = load_radio1d_h(args.model)
        with Image.open(source) as image:
            tiles = build_radio_tiles(
                image, max_detail_tiles=args.max_detail_tiles,
                overlap=args.overlap)
        tokens_per_tile = tokens_per_tile_for_tile_count(
            len(tiles), threshold=args.adaptive_token_threshold)
        features = encode_radio_tiles(
            model, tiles, batch_size=args.batch_size,
            num_tokens=tokens_per_tile)
        metadata = make_metadata(
            source, args.revision, tiles,
            adaptive_token_threshold=args.adaptive_token_threshold)
        save_cache(target, metadata, features)
        context = estimate_context(
            len(tiles), args.text_token_estimate,
            adaptive_token_threshold=args.adaptive_token_threshold)
        completed += 1
        visual_tokens += context.visual_tokens
        above_reference += int(context.above_reference)
        if completed == 1 or completed % 10 == 0:
            elapsed = time.monotonic() - started
            print(json.dumps({
                "row": row_index, "completed": completed, "resumed": skipped,
                "tiles": len(tiles), "visual_tokens": context.visual_tokens,
                "estimated_total_tokens": context.total_tokens,
                "above_10240": context.above_reference,
                "images_per_second": round(completed / elapsed, 4),
                "visual_tokens_per_second": round(visual_tokens / elapsed, 2),
                "cache": str(target),
            }, sort_keys=True), flush=True)
    elapsed = time.monotonic() - started
    print(json.dumps({
        "status": "complete", "completed": completed, "resumed": skipped,
        "visual_tokens": visual_tokens,
        "examples_above_10240": above_reference,
        "elapsed_seconds": round(elapsed, 3),
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
