#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
PYTHON_BIN="${VISION_PYTHON:-$PWD/.venv/bin/python}"
TRAIN="${VISION_NEXT_TRAIN:-curated_vision/vision_eight_hour.jsonl}"
EVAL="${VISION_NEXT_EVAL:-curated_vision/vision_eight_hour_eval.jsonl}"
PREFIX="${VISION_NEXT_PREFIX_TOKENS:-128}"
TAPS="${VISION_NEXT_MOONVIT_TAPS:-8,17,26}"
VIEW_MODE="${VISION_NEXT_VIEW_MODE:-full-quadrants}"
MOON_CACHE="${VISION_NEXT_MOON_CACHE:-/thearray/downloads/cache/moe-mla/moonvit_next_${PREFIX}}"
FUSION_CACHE="${VISION_NEXT_FUSION_CACHE:-/thearray/downloads/cache/moe-mla/fusion_so400m_next_${PREFIX}}"
RECEIPT="${VISION_NEXT_CACHE_RECEIPT:-curated_vision/vision_next_so400m_${PREFIX}.cache.json}"
SIGLIP_MODEL="${VISION_NEXT_SIGLIP2_MODEL:-models/vision/siglip2-so400m-patch16-512}"
SIGLIP_WIDTH="${VISION_NEXT_SIGLIP2_WIDTH:-1152}"

for required in "$PYTHON_BIN" "$TRAIN" "$EVAL" "$SIGLIP_MODEL"; do
  if [[ ! -e "$required" ]]; then
    echo "missing next-lever input: $required" >&2
    exit 1
  fi
done

export TRAIN EVAL PREFIX MOON_CACHE FUSION_CACHE SIGLIP_WIDTH
PYTHONPATH=src "$PYTHON_BIN" - <<'PY'
import json
import os
import shutil
from pathlib import Path

root = Path.cwd()
images = set()
for name in ("TRAIN", "EVAL"):
    for line in Path(os.environ[name]).open():
        if not line.strip():
            continue
        image = Path(json.loads(line)["image"])
        images.add(str((image if image.is_absolute() else root / image).resolve()))
tokens = int(os.environ["PREFIX"])
# Three MoonViT stages [T,4,1152] plus aligned [T,1792], stored in bf16.
fusion_width = int(os.environ["SIGLIP_WIDTH"]) + 768 + 256
payload = len(images) * tokens * (3 * 4 * 1152 + fusion_width) * 2
parent = Path(os.environ["MOON_CACHE"]).parent
parent.mkdir(parents=True, exist_ok=True)
free = shutil.disk_usage(parent).free
print({"kind": "next_cache_capacity", "images": len(images),
       "payload_tib": round(payload / 2**40, 3),
       "free_tib": round(free / 2**40, 3)})
if free < payload * 1.10:
    raise SystemExit("insufficient free space for next-lever caches plus 10% safety margin")
PY

PYTHONPATH=src "$PYTHON_BIN" -m rwkv_lab.vision_cache \
  --data "$TRAIN" "$EVAL" \
  --cache "$MOON_CACHE" \
  --prefix-tokens "$PREFIX" \
  --tap-layers "$TAPS" \
  --view-mode "$VIEW_MODE" \
  --batch "${VISION_NEXT_MOON_CACHE_BATCH:-2}" \
  --workers "${VISION_NEXT_CACHE_WORKERS:-8}" \
  --sort-window "${VISION_NEXT_CACHE_SORT_WINDOW:-16}"

PYTHONPATH=src "$PYTHON_BIN" -m rwkv_lab.vision_fusion_cache \
  --data "$TRAIN" "$EVAL" \
  --cache "$FUSION_CACHE" \
  --prefix-tokens "$PREFIX" \
  --siglip2 "$SIGLIP_MODEL" \
  --siglip2-width "$SIGLIP_WIDTH" \
  --batch "${VISION_NEXT_FUSION_CACHE_BATCH:-8}"

export TRAIN EVAL PREFIX TAPS VIEW_MODE MOON_CACHE FUSION_CACHE RECEIPT SIGLIP_MODEL SIGLIP_WIDTH
PYTHONPATH=src "$PYTHON_BIN" - <<'PY'
import hashlib
import json
import os
from pathlib import Path

root = Path.cwd()
manifests = [Path(os.environ["TRAIN"]), Path(os.environ["EVAL"])]
images = set()
for manifest in manifests:
    for line in manifest.open():
        if not line.strip():
            continue
        row = json.loads(line)
        image = Path(row["image"])
        image = image if image.is_absolute() else root / image
        if image.is_file():
            images.add(str(image.resolve()))
expected = len(images)
moon = Path(os.environ["MOON_CACHE"]).resolve()
fusion = Path(os.environ["FUSION_CACHE"]).resolve()
counts = {
    "moon": sum(path.suffix == ".pt" for path in moon.iterdir()),
    "fusion": sum(path.suffix == ".pt" for path in fusion.iterdir()),
}
if counts != {"moon": expected, "fusion": expected}:
    raise SystemExit(f"next cache count mismatch: expected={expected} actual={counts}")
receipt = {
    "schema": 1,
    "train_sha256": hashlib.sha256(manifests[0].read_bytes()).hexdigest(),
    "eval_sha256": hashlib.sha256(manifests[1].read_bytes()).hexdigest(),
    "prefix_tokens": int(os.environ["PREFIX"]),
    "moonvit_taps": os.environ["TAPS"],
    "view_mode": os.environ["VIEW_MODE"],
    "siglip2_model": str(Path(os.environ["SIGLIP_MODEL"]).resolve()),
    "siglip2_width": int(os.environ["SIGLIP_WIDTH"]),
    "expected_entries": expected,
    "moon_cache": str(moon),
    "fusion_cache": str(fusion),
    "moon_cache_mtime_ns": moon.stat().st_mtime_ns,
    "fusion_cache_mtime_ns": fusion.stat().st_mtime_ns,
}
target = Path(os.environ["RECEIPT"])
target.parent.mkdir(parents=True, exist_ok=True)
temporary = target.with_suffix(target.suffix + ".tmp")
temporary.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
temporary.replace(target)
print(json.dumps(receipt, indent=2, sort_keys=True))
PY
