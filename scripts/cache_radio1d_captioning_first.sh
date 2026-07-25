#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
PYTHON_BIN="${VISION_PYTHON:-/usr/bin/python}"
CACHE="${RADIO_CACHE:-/thearray/downloads/cache/moe-mla/radio1d_captioning_first}"
OLD_CACHE="${RADIO_SOURCE_CACHE:-/thearray/downloads/cache/moe-mla/radio1d_eight_hour_grounded}"
TRAIN="${RADIO_TRAIN:-curated_vision/captioning_first_train.jsonl}"
EVAL="${RADIO_EVAL:-curated_vision/captioning_first_eval.jsonl}"
REVISION="e18692120c7a3203496e1a99056a4149ede135fc"
THRESHOLD="${RADIO_ADAPTIVE_TOKEN_THRESHOLD:-12}"

mkdir -p "$CACHE"
exec 9>"$CACHE/.builder.lock"
flock -n 9 || { echo "another process owns the captioning-first cache" >&2; exit 75; }

REUSE_MARKER="$CACHE/.reuse-${REVISION}-${THRESHOLD}.complete"
if [[ ! -f "$REUSE_MARKER" ]]; then
  PYTHONPATH=src "$PYTHON_BIN" scripts/reuse_radio1d_cache.py \
    --source-cache "$OLD_CACHE" \
    --source-manifest curated_vision/vision_eight_hour.jsonl \
                      curated_vision/vision_eight_hour_eval.jsonl \
    --destination-cache "$CACHE" \
    --destination-manifest "$TRAIN" "$EVAL" \
    --revision "$REVISION" --adaptive-token-threshold "$THRESHOLD"
  touch "$REUSE_MARKER"
fi

cache_manifest() {
  local manifest="$1"
  shift
  PYTHONPATH=src "$PYTHON_BIN" scripts/cache_radio1d_features.py \
    --manifest "$manifest" --cache-dir "$CACHE" \
    --model models/vision/C-RADIOv4-1D-H --revision "$REVISION" \
    --batch-size "${RADIO_TILE_BATCH:-8}" \
    --max-detail-tiles "${RADIO_MAX_DETAIL_TILES:-48}" \
    --adaptive-token-threshold "$THRESHOLD" --trust-existing "$@"
}

SHARDS="${RADIO_CACHE_SHARDS:-4}"
[[ "$SHARDS" =~ ^[1-9][0-9]*$ ]] || {
  echo "RADIO_CACHE_SHARDS must be a positive integer" >&2
  exit 2
}
TRAIN_ROWS="$(wc -l < "$TRAIN")"
CHUNK="$(( (TRAIN_ROWS + SHARDS - 1) / SHARDS ))"
pids=()
trap 'jobs -pr | xargs -r kill' EXIT INT TERM
for (( shard=0; shard<SHARDS; shard++ )); do
  start="$(( shard * CHUNK ))"
  (( start < TRAIN_ROWS )) || break
  cache_manifest "$TRAIN" --start "$start" --limit "$CHUNK" &
  pids+=("$!")
done
for pid in "${pids[@]}"; do
  wait "$pid"
done
trap - EXIT INT TERM

cache_manifest "$EVAL"
