#!/usr/bin/env bash
set -euo pipefail

# Checkpoint-compatible continuation of the eight-hour caption run with:
# learned-query visual resampling, deep visual reinjection, early-token
# grounding weight, and in-batch image/text contrastive negatives.
cd "$(dirname "$0")/.."

export VISION_FINISH_RUN="${VISION_FINISH_RUN:-runs/moonvit_rwkv_eight_hour_grounded}"
export VISION_FINISH_TRAIN="${VISION_FINISH_TRAIN:-curated_vision/vision_eight_hour.jsonl}"
export VISION_FINISH_EVAL="${VISION_FINISH_EVAL:-curated_vision/vision_eight_hour_eval.jsonl}"
export VISION_FINISH_SUMMARY="${VISION_FINISH_SUMMARY:-curated_vision/vision_eight_hour.summary.json}"
export VISION_FINISH_SAMPLER="${VISION_FINISH_SAMPLER:-curated_vision/vision_eight_hour.sampler.json}"
export VISION_FINISH_CACHE="${VISION_FINISH_CACHE:-/home/sirus/.cache/moe-mla/moonvit_features_eight_hour}"
export VISION_FINISH_CACHE_RECEIPT="${VISION_FINISH_CACHE_RECEIPT:-curated_vision/vision_eight_hour.cache.json}"
export VISION_FINISH_TARGET_STEPS="${VISION_FINISH_TARGET_STEPS:-30392}"
export VISION_FINISH_SOURCE_RUN="${VISION_FINISH_SOURCE_RUN:-runs/moonvit_rwkv_eight_hour}"
export VISION_FINISH_SOURCE_CHECKPOINT="${VISION_FINISH_SOURCE_CHECKPOINT:-runs/moonvit_rwkv_eight_hour/last.pt}"
export VISION_ENABLE_GROUNDING_LEVERS=1

# Pooled MoonViT cache format is unchanged by the learned residual resampler.
export SKIP_CACHE_VERIFY="${SKIP_CACHE_VERIFY:-1}"

exec scripts/run_vision_finish_grounded.sh
