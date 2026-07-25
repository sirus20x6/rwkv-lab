#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
PYTHON_BIN="${VISION_PYTHON:-/usr/bin/python}"
RUN="${RADIO_RUN:-runs/radio1d_rwkv_captioning_first}"
TRAIN="${RADIO_TRAIN:-curated_vision/captioning_first_train.jsonl}"
EVAL="${RADIO_EVAL:-curated_vision/captioning_first_eval.jsonl}"
CACHE="${RADIO_CACHE:-/thearray/downloads/cache/moe-mla/radio1d_captioning_first}"
SOURCE_RUN="${RADIO_SOURCE_RUN:-runs/radio1d_rwkv_eight_hour_grounded_sandwich}"

if [[ -n "${RADIO_SOURCE_CHECKPOINT:-}" ]]; then
  SOURCE_CHECKPOINT="$RADIO_SOURCE_CHECKPOINT"
else
  SOURCE_CHECKPOINT="$($PYTHON_BIN scripts/vision_run_evidence.py \
    "$SOURCE_RUN" --resolve-best)"
fi
for required in "$TRAIN" "$EVAL" "$SOURCE_CHECKPOINT" \
                models/vision/C-RADIOv4-1D-H \
                models/rwkv7-g1h-2.9b-20260710-ctx10240.pth; do
  [[ -e "$required" ]] || {
    echo "missing captioning-first RADIO input: $required" >&2
    exit 1
  }
done

mkdir -p "$RUN" "$CACHE"
exec 9>"$RUN/.launcher.lock"
flock -n 9 || { echo "another launcher owns $RUN" >&2; exit 75; }

resume=(--resume auto)
if [[ ! -f "$RUN/last.pt" ]]; then
  # The RADIO geometry is unchanged, so carry the complete trained bridge,
  # grounding, deep-injection, loop, NextLat, and Engram state into the new
  # data phase while intentionally resetting optimizer and sampler state.
  resume=(--resume none --init-adapters-from "$SOURCE_CHECKPOINT")
fi

while true; do
  set +e
  PYTHONPATH=src "$PYTHON_BIN" -m rwkv_lab.vision_train \
  --vision-backend radio1d \
  --data "$TRAIN" --eval-data "$EVAL" \
  --out "$RUN" --steps "${RADIO_STEPS:-45000}" \
  --feature-cache "$CACHE" \
  --radio-model models/vision/C-RADIOv4-1D-H \
  --radio-revision e18692120c7a3203496e1a99056a4149ede135fc \
  --radio-max-detail-tiles "${RADIO_MAX_DETAIL_TILES:-48}" \
  --radio-adaptive-token-threshold "${RADIO_ADAPTIVE_TOKEN_THRESHOLD:-12}" \
  --radio-tile-batch "${RADIO_TILE_BATCH:-8}" \
  --radio-adaptive-complexity \
  --radio-complexity-budget-ratio "${RADIO_COMPLEXITY_BUDGET_RATIO:-0.75}" \
  --radio-complexity-token-quantum "${RADIO_COMPLEXITY_TOKEN_QUANTUM:-16}" \
  --batch 1 --min-batch 1 --max-batch "${RADIO_MAX_BATCH:-4}" \
  --target-batch-tokens "${RADIO_BATCH_TOKENS:-4096}" \
  --activation-checkpoint-min-tokens \
    "${RADIO_ACTIVATION_CHECKPOINT_TOKENS:-4096}" \
  --max-text-tokens 768 --prefix-tokens 256 \
  --sandwich-prompt --sandwich-lead-prompt $'An image follows:\n' \
  --vision-resampler-layers 0 \
  --deep-vision-layers 8,16,24 --deep-vision-rank 256 \
  --grounding-early-tokens 24 --grounding-early-weight 3 \
  --grounding-contrastive-weight 0.1 \
  --grounding-contrastive-dim 512 --grounding-temperature 0.07 \
  --structured-head --structured-weight 1.0 \
  --structured-coordinate-weight "${RADIO_STRUCTURED_COORDINATE_WEIGHT:-4.0}" \
  --structured-invalid-box-weight "${RADIO_INVALID_BOX_WEIGHT:-0.5}" \
  --structured-invalid-box-margin "${RADIO_INVALID_BOX_MARGIN:-1.0}" \
  --structured-width 256 --structured-object-queries 16 \
  --structured-spatial-layers 2 --structured-object-layers 2 \
  --structured-heads 8 \
  --loop-count 2 --loop-index --loop-start-step 1 --loop-ramp-steps 0 \
  --loop-gate-cap 0.25 --loop-lr 1e-5 \
  --engram --engram-sites 3,15 --engram-drow 128 --engram-rows 65536 \
  --engram-lr 1e-3 --engram-warmup-steps 0 --engram-boundary-id 0 \
  --nextlat-weight 0.1 --nextlat-hidden 1024 \
  --manifest-stat-workers 64 --prefetch-next-batch --allow-batch-resize \
  --checkpoint-every 50 --eval-every "${RADIO_EVAL_EVERY:-500}" \
  --eval-examples "${RADIO_EVAL_EXAMPLES:-448}" \
  --eval-batch-size "${RADIO_EVAL_BATCH_SIZE:-4}" \
  --eval-sample-every "${RADIO_EVAL_SAMPLE_EVERY:-1000}" \
  --eval-samples "${RADIO_EVAL_SAMPLES:-14}" \
  --eval-sample-exclude-sources joy,civitai,nsfw,porn,manga,pose_vr,grid \
  --eval-sample-max-new 768 --require-fused-ce \
  --restart-before-eval \
  --profile-steps "${RADIO_PROFILE_STEPS:-3}" \
  --allow-coco-prompt-migration \
  "${resume[@]}"
  status=$?
  set -e

  if [[ "$status" -eq 42 ]]; then
    echo "trainer requested a clean-process eval restart; resuming exact checkpoint" >&2
    resume=(--resume auto)
    continue
  fi
  exit "$status"
done
