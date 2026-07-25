#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 5 ]]; then
  echo "usage: $0 MANIFEST MOON_CACHE FUSION_CACHE RECEIPT LABEL" >&2
  exit 2
fi
cd "$(dirname "$0")/.."
PYTHON_BIN="${VISION_PYTHON:-$PWD/.venv/bin/python}"
MANIFEST="$1"
MOON_CACHE="$2"
FUSION_CACHE="$3"
RECEIPT="$4"
LABEL="$5"
SIGLIP="models/vision/siglip2-so400m-patch16-512"

for required in "$PYTHON_BIN" "$MANIFEST" "$SIGLIP"; do
  [[ -e "$required" ]] || { echo "cache input missing: $required" >&2; exit 1; }
done

PYTHONPATH=src "$PYTHON_BIN" -m rwkv_lab.vision_cache \
  --data "$MANIFEST" --cache "$MOON_CACHE" \
  --prefix-tokens 128 --tap-layers 8,17,26 --view-mode full-quadrants \
  --batch "${VISION_SHARD_MOON_BATCH:-2}" \
  --workers "${VISION_SHARD_CACHE_WORKERS:-16}" --sort-window 64

PYTHONPATH=src "$PYTHON_BIN" -m rwkv_lab.vision_fusion_cache \
  --data "$MANIFEST" --cache "$FUSION_CACHE" \
  --prefix-tokens 128 --siglip2 "$SIGLIP" --siglip2-width 1152 \
  --batch "${VISION_SHARD_FUSION_BATCH:-8}"

export MANIFEST MOON_CACHE FUSION_CACHE SIGLIP RECEIPT LABEL
"$PYTHON_BIN" - <<'PY'
import hashlib, json, os
from pathlib import Path

manifest = Path(os.environ["MANIFEST"])
moon = Path(os.environ["MOON_CACHE"]).resolve()
fusion = Path(os.environ["FUSION_CACHE"]).resolve()
expected = sum(bool(line.strip()) for line in manifest.open())
counts = {"moon": sum(p.suffix == ".pt" for p in moon.iterdir()),
          "fusion": sum(p.suffix == ".pt" for p in fusion.iterdir())}
if counts != {"moon": expected, "fusion": expected}:
    raise SystemExit(f"cache count mismatch: expected={expected} actual={counts}")
receipt = {
    "schema": 1, "label": os.environ["LABEL"],
    "train_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
    "expected_entries": expected, "prefix_tokens": 128,
    "moonvit_taps": "8,17,26", "view_mode": "full-quadrants",
    "siglip2_model": str(Path(os.environ["SIGLIP"]).resolve()),
    "siglip2_width": 1152, "moon_cache": str(moon),
    "fusion_cache": str(fusion),
}
target = Path(os.environ["RECEIPT"])
target.parent.mkdir(parents=True, exist_ok=True)
temporary = target.with_suffix(target.suffix + ".tmp")
temporary.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
os.replace(temporary, target)
print(json.dumps(receipt, indent=2, sort_keys=True), flush=True)
PY
