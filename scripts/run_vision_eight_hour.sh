#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export VISION_FINISH_RUN=runs/moonvit_rwkv_eight_hour
export VISION_FINISH_TRAIN=curated_vision/vision_eight_hour.jsonl
export VISION_FINISH_EVAL=curated_vision/vision_eight_hour_eval.jsonl
export VISION_FINISH_SUMMARY=curated_vision/vision_eight_hour.summary.json
export VISION_FINISH_SAMPLER=curated_vision/vision_eight_hour.sampler.json
export VISION_FINISH_CACHE=/home/sirus/.cache/moe-mla/moonvit_features_eight_hour
export VISION_FINISH_CACHE_RECEIPT=curated_vision/vision_eight_hour.cache.json
export VISION_FINISH_TARGET_STEPS="${VISION_FINISH_TARGET_STEPS:-30392}"

# The cache receipt is generated only after every expected feature has been
# payload-validated. The parent launcher independently checks its immutable
# dataset contract, entry count, and directory generation before honoring this
# fast-start flag.
export SKIP_CACHE_VERIFY=1

exec scripts/run_vision_finish_grounded.sh
