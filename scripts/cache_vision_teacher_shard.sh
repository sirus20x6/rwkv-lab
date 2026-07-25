#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 || ! "$1" =~ ^[0-9]+$ ]]; then
  echo "usage: $0 SHARD_INDEX" >&2
  exit 2
fi
cd "$(dirname "$0")/.."
INDEX=$((10#$1))
TAG=$(printf '%03d' "$INDEX")
MANIFEST="curated_vision/vision_next_shard_${TAG}_train.jsonl"
if [[ ! -f "$MANIFEST" ]]; then
  .venv/bin/python scripts/build_vision_teacher_shard.py \
    --source curated_vision/vision_eight_hour.jsonl \
    --output "$MANIFEST" --index "$INDEX" --size 40000
fi
exec scripts/cache_vision_teacher_manifest.sh "$MANIFEST" \
  "/workspace/downloads/cache/moe-mla/moonvit_next_128_shard_${TAG}" \
  "/workspace/downloads/cache/moe-mla/fusion_so400m_next_128_shard_${TAG}" \
  "curated_vision/vision_next_so400m_128_shard_${TAG}.cache.json" \
  "i1_shard_${TAG}"
