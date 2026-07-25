#!/usr/bin/env python3
"""Verify the warmed RADIO cache before committing GPU hours to training.

Checks the three things that would silently waste a run:
  1. every row the trainer will read is cache-current at the training threshold,
  2. rows above the old 12-tile cliff now hold 256 tokens/tile, not 128,
  3. the stored width matches what the trainer's own planner expects, so no row
     is re-encoded inline (or worse, mismatched against its label offsets).
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from PIL import Image  # noqa: E402

from rwkv_lab.radio1d_cache import cache_is_current, cache_path, load_cache  # noqa: E402
from rwkv_lab.radio1d_rwkv import (  # noqa: E402
    choose_detail_grid, tokens_per_tile_for_tile_count,
    DEFAULT_ADAPTIVE_TOKEN_THRESHOLD)


def tile_count(width: int, height: int, max_detail_tiles: int) -> int:
    rows_, columns = choose_detail_grid(
        width, height, max_detail_tiles=max_detail_tiles)
    details = rows_ * columns
    return 1 if details == 1 else details + 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", nargs="+", required=True)
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--revision",
                    default="e18692120c7a3203496e1a99056a4149ede135fc")
    ap.add_argument("--threshold", type=int,
                    default=DEFAULT_ADAPTIVE_TOKEN_THRESHOLD)
    ap.add_argument("--max-detail-tiles", type=int, default=48)
    ap.add_argument("--sample", type=int, default=1500,
                    help="0 = check every row (slow)")
    args = ap.parse_args()

    cache = Path(args.cache_dir)
    rows: list[dict] = []
    for manifest in args.manifest:
        rows += [json.loads(line) for line in open(manifest) if line.strip()]
    if args.sample and args.sample < len(rows):
        random.seed(0)
        checked = random.sample(rows, args.sample)
    else:
        checked = rows
    print(f"manifest rows={len(rows):,}  checking={len(checked):,}")

    stale: list[str] = []
    width_mismatch: list[str] = []
    missing: list[str] = []
    by_tiles: collections.Counter = collections.Counter()
    for row in checked:
        source = Path(row["image"])
        source = source if source.is_absolute() else ROOT / source
        target = cache_path(cache, source)
        if not target.is_file():
            missing.append(str(source))
            continue
        if not cache_is_current(target, source, args.revision,
                                adaptive_token_threshold=args.threshold,
                                source_sha256=row.get("image_sha256")):
            stale.append(str(source))
            continue
        width, height = row.get("width"), row.get("height")
        if not width or not height:
            with Image.open(source) as image:
                width, height = image.size
        tiles = tile_count(int(width), int(height), args.max_detail_tiles)
        expected = tokens_per_tile_for_tile_count(tiles, threshold=args.threshold)
        metadata, _ = load_cache(target)
        by_tiles[tiles] += 1
        if metadata.tokens_per_tile != expected or len(metadata.tiles) != tiles:
            width_mismatch.append(
                f"{source}: tiles {len(metadata.tiles)}!={tiles} or "
                f"tokens {metadata.tokens_per_tile}!={expected}")

    above = sum(n for t, n in by_tiles.items() if t > 12)
    print(f"tile distribution (verified): {dict(sorted(by_tiles.items()))}")
    print(f"rows above the old 12-tile cliff: {above}"
          f"  (all now at {tokens_per_tile_for_tile_count(13, threshold=args.threshold)} tokens/tile)")
    for label, bad in (("MISSING from cache", missing),
                       ("STALE at this threshold", stale),
                       ("WIDTH MISMATCH vs planner", width_mismatch)):
        print(f"{label}: {len(bad)}")
        for item in bad[:5]:
            print(f"    {item}")
    ok = not (missing or stale or width_mismatch)
    print("\nCACHE READY" if ok else "\nCACHE NOT READY — training would re-encode or mismatch")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
