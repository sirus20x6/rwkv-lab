#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if (($# != 1)) || [[ ! "$1" =~ ^[0-9]+$ ]]; then
  echo "usage: $0 CACHE_PROCESS_PID" >&2
  exit 2
fi
cache_pid="$1"

# The cache producer is owned by another launcher session, so its exit status
# is not waitable here. The exhaustive finalizer below is the authoritative
# success check and fails if the producer stopped early or left a bad payload.
while kill -0 "$cache_pid" 2>/dev/null; do
  sleep 30
done

PYTHONPATH=src python scripts/finalize_vision_cache.py \
  --train curated_vision/vision_eight_hour.jsonl \
  --eval curated_vision/vision_eight_hour_eval.jsonl \
  --cache /home/sirus/.cache/moe-mla/moonvit_features_eight_hour \
  --output curated_vision/vision_eight_hour.cache.json \
  --workers 32

# Exercise every immutable launch contract without starting or loading the
# trainer. A future operator launch therefore has no preparation fallback.
VISION_PREFLIGHT_ONLY=1 scripts/run_vision_eight_hour.sh
echo "eight-hour vision run is ready"
