#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
SOURCE="curated_vision/local_porn_unlabeled_dedup.jsonl"
MANIFEST="curated_vision/local_porn_teacher_shard_000.jsonl"
if [[ ! -f "$MANIFEST" ]]; then
  .venv/bin/python scripts/build_vision_teacher_shard.py \
    --source "$SOURCE" --output "$MANIFEST" --index 0 --size 40000
fi
exec scripts/cache_vision_teacher_manifest.sh "$MANIFEST" \
  /thearray/downloads/cache/moe-mla/moonvit_next_128_local_porn_000 \
  /thearray/downloads/cache/moe-mla/fusion_so400m_next_128_local_porn_000 \
  curated_vision/local_porn_teacher_shard_000.cache.json local_porn_000
