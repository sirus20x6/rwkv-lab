#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
PYTHON_BIN="${VISION_PYTHON:-$PWD/.venv/bin/python}"
MANIFEST="curated_vision/vision_next_shard_001_train.jsonl"
MOON_CACHE="/workspace/downloads/cache/moe-mla/moonvit_next_128_shard_001"
MOON_PARTIAL="/workspace/downloads/cache/moe-mla/moonvit_next_128_shard_001_partial"
FUSION_CACHE="/workspace/downloads/cache/moe-mla/fusion_so400m_next_128_shard_001"
SIGLIP="models/vision/siglip2-so400m-patch16-512"
RECEIPT="curated_vision/vision_next_so400m_128_shard_001.cache.json"

if [[ ! -f "$MANIFEST" || ! -x "$PYTHON_BIN" || ! -e "$SIGLIP" ]]; then
  echo "shard 001 cache inputs are incomplete" >&2
  exit 1
fi
if [[ -d "$MOON_PARTIAL" && ! -e "$MOON_CACHE" ]]; then
  mv "$MOON_PARTIAL" "$MOON_CACHE"
elif [[ -e "$MOON_PARTIAL" && -e "$MOON_CACHE" ]]; then
  echo "both partial and stable MoonViT shard directories exist" >&2
  exit 1
fi

PYTHONPATH=src "$PYTHON_BIN" -m rwkv_lab.vision_cache \
  --data "$MANIFEST" --cache "$MOON_CACHE" \
  --prefix-tokens 128 --tap-layers 8,17,26 \
  --view-mode full-quadrants --batch "${VISION_SHARD_MOON_BATCH:-2}" \
  --workers "${VISION_SHARD_CACHE_WORKERS:-16}" --sort-window 64

PYTHONPATH=src "$PYTHON_BIN" -m rwkv_lab.vision_fusion_cache \
  --data "$MANIFEST" --cache "$FUSION_CACHE" \
  --prefix-tokens 128 --siglip2 "$SIGLIP" --siglip2-width 1152 \
  --batch "${VISION_SHARD_FUSION_BATCH:-8}"

export MANIFEST MOON_CACHE FUSION_CACHE SIGLIP RECEIPT
PYTHONPATH=src "$PYTHON_BIN" - <<'PY'
import hashlib
import json
import os
from pathlib import Path

manifest = Path(os.environ["MANIFEST"])
moon = Path(os.environ["MOON_CACHE"]).resolve()
fusion = Path(os.environ["FUSION_CACHE"]).resolve()
expected = sum(bool(line.strip()) for line in manifest.open())
counts = {"moon": sum(path.suffix == ".pt" for path in moon.iterdir()),
          "fusion": sum(path.suffix == ".pt" for path in fusion.iterdir())}
if counts != {"moon": expected, "fusion": expected}:
    raise SystemExit(f"cache count mismatch: expected={expected} actual={counts}")
receipt = {
    "schema": 1,
    "shard_index": 1,
    "train_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
    "expected_entries": expected,
    "prefix_tokens": 128,
    "moonvit_taps": "8,17,26",
    "view_mode": "full-quadrants",
    "siglip2_model": str(Path(os.environ["SIGLIP"]).resolve()),
    "siglip2_width": 1152,
    "moon_cache": str(moon),
    "fusion_cache": str(fusion),
    "moon_cache_mtime_ns": moon.stat().st_mtime_ns,
    "fusion_cache_mtime_ns": fusion.stat().st_mtime_ns,
}
target = Path(os.environ["RECEIPT"])
temporary = target.with_suffix(target.suffix + ".tmp")
temporary.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
os.replace(temporary, target)
print(json.dumps(receipt, indent=2, sort_keys=True))
PY
