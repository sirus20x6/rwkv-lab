#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
COMMON=(--root /thearray/git/datasets/porn \
  --db /thearray/downloads/cache/moe-mla/local_porn_image_dedup.sqlite \
  --manifest curated_vision/local_porn_unlabeled_dedup.jsonl)
POLICY=curated_vision/local_porn_dedupe_policy.json
if [[ ! -f "$POLICY" ]]; then
  echo "dedupe cutoff policy is required; use http://127.0.0.1:9126" >&2
  exit 75
fi
DISTANCE=$(jq -er '.phash_distance | select(type=="number" and .>=0 and .<=7)' "$POLICY")
MIN_SIDE=$(jq -er '.min_side | select(type=="number" and .>=1)' "$POLICY")
.venv/bin/python scripts/build_unlabeled_image_manifest.py "${COMMON[@]}" \
  --phase cluster --phash-distance "$DISTANCE" --min-side "$MIN_SIDE"
.venv/bin/python scripts/build_unlabeled_image_manifest.py "${COMMON[@]}" --phase export
