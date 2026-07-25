#!/usr/bin/env python3
"""Cache C-RADIOv4-H raw spatial grids for a manifest.

Stores the native ``(H/16, W/16)`` grid per tile rather than a pooled lattice,
so the token budget can be changed later without re-encoding. Tiles keep their
own aspect ratio, so no token is spent on letterbox padding.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rwkv_lab.radio_v4h import (  # noqa: E402
    V4H_TILE_SIZE, cache_one_image, cache_path, v4h_cache_is_current)

DEFAULT_REVISION = "c-radiov4-h"


def rows(manifest: Path, start: int, limit: int | None):
    emitted = 0
    with manifest.open() as handle:
        for index, line in enumerate(handle):
            line = line.strip()
            if not line or index < start:
                continue
            if limit is not None and emitted >= limit:
                return
            value = json.loads(line)
            source = value.get("image") or value.get("image_path")
            if not source:
                raise ValueError(f"manifest row {index} has no image path")
            yield index, Path(source), value.get("image_sha256")
            emitted += 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, nargs="+", required=True)
    ap.add_argument("--cache-dir", type=Path, required=True)
    ap.add_argument("--model", type=Path, default=ROOT / "models/vision/C-RADIOv4-H")
    ap.add_argument("--revision", default=DEFAULT_REVISION)
    ap.add_argument("--tile-size", type=int, default=V4H_TILE_SIZE)
    ap.add_argument("--max-detail-tiles", type=int, default=48)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()
    if args.batch_size < 1 or args.tile_size < 16 or args.start < 0:
        ap.error("batch size and tile size must be positive; start non-negative")

    model = None
    completed = skipped = failed = 0
    tiles_done = 0
    bytes_written = 0
    started = time.monotonic()
    for manifest in args.manifest:
        print(json.dumps({"phase": "manifest", "path": str(manifest)}), flush=True)
        for _index, source, sha in rows(manifest, args.start, args.limit):
            source = source if source.is_absolute() else ROOT / source
            target = cache_path(args.cache_dir, source)
            if v4h_cache_is_current(target, source, args.revision,
                                    tile_size=args.tile_size, source_sha256=sha):
                skipped += 1
                continue
            if model is None:
                from rwkv_lab.radio_v4h import load_radio_v4h
                model = load_radio_v4h(args.model)
            try:
                written, count = cache_one_image(
                    model, source, args.cache_dir, revision=args.revision,
                    tile_size=args.tile_size,
                    max_detail_tiles=args.max_detail_tiles,
                    batch_size=args.batch_size, source_sha256=sha)
            except Exception as error:  # noqa: BLE001 - one bad image must not end the pass
                failed += 1
                print(json.dumps({"phase": "error", "source": str(source),
                                  "error": f"{type(error).__name__}: {error}"[:200]}),
                      flush=True)
                continue
            completed += 1
            tiles_done += count
            bytes_written += written.stat().st_size
            if completed == 1 or completed % 25 == 0:
                elapsed = time.monotonic() - started
                print(json.dumps({
                    "phase": "progress", "completed": completed,
                    "skipped": skipped, "failed": failed, "tiles": tiles_done,
                    "gb": round(bytes_written / 1e9, 2),
                    "images_per_second": round(completed / max(elapsed, 1e-6), 3),
                }, sort_keys=True), flush=True)
    elapsed = time.monotonic() - started
    print(json.dumps({
        "phase": "complete", "completed": completed, "skipped": skipped,
        "failed": failed, "tiles": tiles_done,
        "gb": round(bytes_written / 1e9, 2),
        "elapsed_seconds": round(elapsed, 1),
    }, sort_keys=True), flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
