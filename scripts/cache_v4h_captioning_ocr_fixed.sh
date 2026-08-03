#!/usr/bin/env bash
# Warm misses and deliberately replace entries from the retired shared-32px
# snapping contract. The revision is passed explicitly so this writer and the
# trainer cannot silently drift to different defaults.
set -euo pipefail
cd /thearray/git/moe-mla

exec python scripts/cache_v4h_native.py \
  --manifest curated_vision/captioning_ocr_fixed_train.jsonl \
             curated_vision/captioning_ocr_fixed_eval.jsonl \
  --cache-dir "${V4H_CACHE:-/thearray/downloads/cache/moe-mla/radio_v4h_native_captioning_first}" \
  --model models/vision/C-RADIOv4-H \
  --revision "${V4H_REVISION:-c-radiov4-h-native-h16w32}" \
  --allow-reconfigure \
  --max-edge 2048
