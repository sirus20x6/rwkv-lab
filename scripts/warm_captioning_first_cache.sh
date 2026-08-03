#!/usr/bin/env bash
# Pre-warm the RADIO feature cache for the flat-256 token budget.
#
# Raising --radio-adaptive-token-threshold from 12 to 49 invalidates cached
# entries only where the resulting token width actually changes -- images above
# 12 tiles, encoded at 128 tokens/tile and now needing 256. Measured on this
# corpus that is ~4% of rows (~4,100 of 110,000), essentially the OCR subset,
# which is exactly the data the fix targets. The other ~138k entries and all
# 533 GB stay valid.
#
# DO NOT ADD --trust-existing. It skips by FILENAME before cache_is_current is
# ever consulted, so every stale 128-token entry -- which exists on disk under
# the right name -- would be skipped rather than re-encoded. Measured: it skips
# 100% of rows, finishes in seconds, and regenerates nothing while reporting
# success. That flag is for resuming an interrupted build under an UNCHANGED
# policy; it is wrong precisely when the policy has moved.
#
# The full validation pass costs ~1 hour of CPU/IO across 110k rows (it opens
# each safetensors header) plus GPU only for the ~4% that are genuinely stale.
#
# Running this is OPTIONAL: cached_radio_features re-encodes misses on demand
# during training. Pre-warming just moves that cost out of the training loop.
#
# Must run from the repo root: the cache key is sha256(abspath(image)) and the
# manifests store paths relative to it. The `cd` below guarantees that (set -e
# aborts if it fails). The preflight then checks the two things `cd` cannot:
# that the images are actually reachable from here, and that none are symlinks
# (cache_path uses realpath, the cache tool uses abspath -- they diverge only
# for links, which would send warmed features to a key the trainer never reads).
set -euo pipefail
cd /thearray/git/moe-mla

CACHE=/thearray/downloads/cache/moe-mla/radio1d_captioning_first
THRESHOLD=49          # must match --radio-adaptive-token-threshold in the resume script

# Preflight: confirm this process derives the same cache key the trainer will
# read, before spending any GPU.
PYTHONPATH=src python - "$CACHE" <<'PREFLIGHT'
import json, sys
from pathlib import Path
sys.path.insert(0, "scripts")
from rwkv_lab.radio1d_cache import cache_path
from cache_radio1d_features import lexical_cache_path
cache, root = Path(sys.argv[1]), Path.cwd()
rows = [json.loads(l) for _, l in zip(range(50),
        open("curated_vision/captioning_first_train.jsonl"))]
missing = [r["image"] for r in rows if not (root / r["image"]).exists()]
if missing:
    raise SystemExit(f"images not reachable from {root}: {missing[:3]}")
bad = [r["image"] for r in rows
       if lexical_cache_path(cache, Path(r["image"]))
       != cache_path(cache, root / r["image"])]
if bad:
    raise SystemExit(f"cache key mismatch (symlinked source?): {bad[:3]}")
print(f"preflight ok: {len(rows)} images reachable, cache keys agree")
PREFLIGHT

for MANIFEST in curated_vision/captioning_first_train.jsonl \
                curated_vision/captioning_first_eval.jsonl; do
  echo "=== warming ${MANIFEST} ==="
  python scripts/cache_radio1d_features.py \
    --manifest "${MANIFEST}" \
    --cache-dir "${CACHE}" \
    --model models/vision/C-RADIOv4-1D-H \
    --adaptive-token-threshold "${THRESHOLD}" \
    --max-detail-tiles 48 \
    --overlap 0.125 \
    --batch-size 8
done
