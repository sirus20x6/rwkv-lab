#!/usr/bin/env python3
"""Cache C-RADIOv4-H features at native resolution, one grid per image.

No tiling: the image is encoded whole at its own resolution (snapped down to a
multiple of 32, capped at RADIO's 2048). Tiling existed because RADIO1D emitted
a fixed 256 nested tokens per tile, so splitting was the only way to buy detail.
v4h's token count is (H/16)*(W/16), so resolution buys it directly -- and the
tiled path was spending tokens on a redundant thumbnail, 12.5% overlap, and an
upscale of every detail crop.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from PIL import Image  # noqa: E402

from rwkv_lab.radio_v4h import (  # noqa: E402
    V4H_MAX_EDGE, cache_path, encode_native, native_cache_is_current,
    save_native_cache)

DEFAULT_REVISION = "c-radiov4-h-native"


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
            yield Path(source), value.get("image_sha256")
            emitted += 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, nargs="+", required=True)
    ap.add_argument("--cache-dir", type=Path, required=True)
    ap.add_argument("--model", type=Path, default=ROOT / "models/vision/C-RADIOv4-H")
    ap.add_argument("--revision", default=DEFAULT_REVISION)
    ap.add_argument("--max-edge", type=int, default=V4H_MAX_EDGE)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()
    if args.max_edge < 32 or args.start < 0:
        ap.error("max-edge must be >= 32 and start non-negative")

    model = None
    completed = skipped = failed = 0
    tokens = 0
    written = 0
    started = time.monotonic()
    for manifest in args.manifest:
        print(json.dumps({"phase": "manifest", "path": str(manifest)}), flush=True)
        for source, sha in rows(manifest, args.start, args.limit):
            source = source if source.is_absolute() else ROOT / source
            target = cache_path(args.cache_dir, source)
            if native_cache_is_current(target, source, args.revision,
                                       max_edge=args.max_edge, source_sha256=sha):
                skipped += 1
                continue
            if model is None:
                from rwkv_lab.radio_v4h import load_radio_v4h
                model = load_radio_v4h(args.model)
            try:
                with Image.open(source) as image:
                    grid, (grid_h, grid_w) = encode_native(
                        model, image, max_edge=args.max_edge)
                save_native_cache(target, grid, revision=args.revision,
                                  source=source, source_sha256=sha,
                                  max_edge=args.max_edge)
            except Exception as error:  # noqa: BLE001 - one bad image must not end the pass
                failed += 1
                print(json.dumps({"phase": "error", "source": str(source),
                                  "error": f"{type(error).__name__}: {error}"[:200]}),
                      flush=True)
                continue
            completed += 1
            tokens += grid_h * grid_w // 2
            written += target.stat().st_size
            if completed == 1 or completed % 250 == 0:
                elapsed = time.monotonic() - started
                print(json.dumps({
                    "phase": "progress", "completed": completed, "skipped": skipped,
                    "failed": failed, "gb": round(written / 1e9, 2),
                    "mean_paired_tokens": round(tokens / completed),
                    "images_per_second": round(completed / max(elapsed, 1e-6), 3),
                }, sort_keys=True), flush=True)
    elapsed = time.monotonic() - started
    print(json.dumps({
        "phase": "complete", "completed": completed, "skipped": skipped,
        "failed": failed, "gb": round(written / 1e9, 2),
        "mean_paired_tokens": round(tokens / max(completed, 1)),
        "elapsed_seconds": round(elapsed, 1),
    }, sort_keys=True), flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
