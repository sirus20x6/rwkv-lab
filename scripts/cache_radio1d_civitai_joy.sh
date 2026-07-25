#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
PYTHON_BIN="${VISION_PYTHON:-/usr/bin/python}"
CACHE="${RADIO_CACHE:-/thearray/downloads/cache/moe-mla/radio1d_captioning_first}"
TRAIN="${RADIO_TRAIN:-curated_vision/captioning_first_civitai_joy_qwen36_train.jsonl}"
EVAL="${RADIO_EVAL:-curated_vision/captioning_first_civitai_joy_qwen36_eval.jsonl}"
REVISION="e18692120c7a3203496e1a99056a4149ede135fc"
THRESHOLD="${RADIO_ADAPTIVE_TOKEN_THRESHOLD:-12}"
for manifest in "$TRAIN" "$EVAL"; do
  [[ -f "$manifest" ]] || {
    echo "missing Qwen3.6-materialized Civitai/Joy manifest: $manifest" >&2
    echo "finish recaptioning, then run scripts/apply_qwen36_civitai_joy_recaptions.py" >&2
    exit 1
  }
done
TRAIN_SHA="$(sha256sum "$TRAIN" | cut -d' ' -f1)"
EVAL_SHA="$(sha256sum "$EVAL" | cut -d' ' -f1)"
MARKER="$CACHE/.civitai-joy-${TRAIN_SHA}-${EVAL_SHA}.complete"

mkdir -p "$CACHE"
# Share the canonical cache-writer lock with the original captioning-first
# builder. Both jobs publish into the same content-addressed directory.
exec 9>"$CACHE/.builder.lock"
flock -n 9 || { echo "another RADIO cache builder is active" >&2; exit 75; }

cache_manifest() {
  PYTHONPATH=src "$PYTHON_BIN" scripts/cache_radio1d_features.py \
    --manifest "$1" --cache-dir "$CACHE" \
    --model models/vision/C-RADIOv4-1D-H --revision "$REVISION" \
    --batch-size "${RADIO_TILE_BATCH:-8}" \
    --max-detail-tiles "${RADIO_MAX_DETAIL_TILES:-48}" \
    --adaptive-token-threshold "$THRESHOLD" --trust-existing
}

cache_manifest "$TRAIN"
cache_manifest "$EVAL"

# Never certify a moving manifest. A changed mix can safely reuse everything
# just cached, but must run this command again to cover and certify its delta.
[[ "$(sha256sum "$TRAIN" | cut -d' ' -f1)" == "$TRAIN_SHA" \
   && "$(sha256sum "$EVAL" | cut -d' ' -f1)" == "$EVAL_SHA" ]] || {
  echo "Civitai/Joy manifests changed during caching; rerun to certify the new generation" >&2
  exit 74
}
touch "$MARKER"
echo "cache ready: $MARKER"
