#!/usr/bin/env bash
# C-RADIOv4-H all-teacher run:
#   SigLIP2-g 1536 + SAM3 1024 + lossless compact DINOv3 1536 = RWKV 4096.
set -uo pipefail
cd /workspace/rwkv-lab || exit 1

PYTHON_BIN="${VISION_PYTHON:-/usr/bin/python}"
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}src"
# Native aspect-ratio grids vary from tens to >10k tokens. Expandable segments
# keep those alternating allocation sizes from stranding unusable CUDA slivers.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

RUN="${V4H_RUN:-runs/radio_v4h_all_adaptors_7b}"
CACHE="${V4H_CACHE:-/workspace/downloads/cache/moe-mla/radio_v4h_all_adaptors_4096}"
TRAIN="${V4H_TRAIN:-curated_vision/captioning_ocr_fixed_train.jsonl}"
EVAL="${V4H_EVAL:-curated_vision/captioning_ocr_fixed_eval.jsonl}"
RWKV="${V4H_RWKV:-models/rwkv7-g1i_preview3260-7.2b-20260716-ctx12288.pth}"
REVISION="${V4H_REVISION:-c-radiov4-h-all-adaptors-4096-native-v2-calibrated}"
RADIO_MODEL="${V4H_RADIO_MODEL:-models/vision/C-RADIOv4-H}"
CACHE_ONLY="${V4H_CACHE_ONLY:-1}"
deep_grouped=(--no-deep-vision-grouped-precompute)
[[ "${V4H_DEEP_GROUPED_PRECOMPUTE:-1}" == "1" ]] &&
  deep_grouped=(--deep-vision-grouped-precompute)
cuda_prefetch=(--no-prefetch-cuda-transfer)
[[ "${V4H_PREFETCH_CUDA_TRANSFER:-1}" == "1" ]] &&
  cuda_prefetch=(--prefetch-cuda-transfer)
# --allow-loop-lr-change-on-resume was passed here. It was a bare boolean that
# never expired, so it silently accepted any V4H_LOOP_LR drift across the
# hundreds of restarts --restart-before-eval performs. It has been replaced by
# --allow-loop-lr-change-from <previous loop-lr>, which self-expires once the
# migrated step is saved, matching its two sibling allowances. Nothing needs it
# here -- every restart passes the same --loop-lr, so no loop-LR migration is
# ever requested.
#
# _structured_data_removal_migration only fires when a resumed checkpoint's
# saved data_fingerprint equals the named source AND differs from this run's.
# This launcher starts at --resume none and afterwards resumes its own
# checkpoints, whose fingerprint always matches, so the allowance was inert.
# Removed rather than kept as decoration.

required=("$TRAIN" "$EVAL" "$CACHE" "$RWKV")
cache_only=(--radio-v4h-cache-only)
if [[ "$CACHE_ONLY" != "1" ]]; then
  # vision_train skips load_radio_v4h entirely under --radio-v4h-cache-only,
  # so the encoder directory is only a real input on the encoding path.
  cache_only=(--no-radio-v4h-cache-only)
  required+=("$RADIO_MODEL")
fi
for input in "${required[@]}"; do
  [[ -e "$input" ]] || {
    echo "missing required input: $input" >&2
    exit 1
  }
done

# A cache entry can be missing or truncated while the directory still exists:
# cache_v4h_all_adaptors.py counts per-image OOM as a tolerated failure, so an
# OOM storm leaves a hole-ridden cache that passes an -e test. Under
# --radio-v4h-cache-only every one of those holes is fatal mid-training.
if [[ "${V4H_SKIP_CACHE_VERIFY:-0}" != "1" ]]; then
  "$PYTHON_BIN" scripts/verify_v4h_all_adaptors_cache.py \
    --manifest "$TRAIN" "$EVAL" \
    --cache-dir "$CACHE" \
    --revision "$REVISION" || {
    echo "feature cache $CACHE is incomplete for $TRAIN/$EVAL" >&2
    exit 1
  }
fi

mkdir -p "$RUN"
exec 9>"$RUN/.launcher.lock"
flock -n 9 || { echo "another launcher already owns $RUN" >&2; exit 75; }

resume=(--resume none)
[[ -f "$RUN/last.pt" ]] && resume=(--resume auto)

# The C-RADIOv4-H artifact digest hashes the producer sources, so editing
# radio_v4h.py -- even for a fix that writes no cache -- makes an existing
# checkpoint unresumable. V4H_FINGERPRINT_FROM names the digest the checkpoint
# actually carries so the migration can accept it once. Read it from last.pt,
# NOT from config.json: config.json is rewritten on every restart and drifts
# ahead of the checkpoint whenever the producer changes mid-run.
FINGERPRINT_ALLOWANCE=()
if [[ -n "${V4H_FINGERPRINT_FROM:-}" ]]; then
  FINGERPRINT_ALLOWANCE=(--allow-v4h-fingerprint-change-from
                         "$V4H_FINGERPRINT_FROM")
fi
STRUCTURED_LR_ALLOWANCE=()
if [[ -n "${STRUCTURED_LR_FROM:-}" ]]; then
  STRUCTURED_LR_ALLOWANCE=(--allow-structured-lr-change-from
                           "$STRUCTURED_LR_FROM")
fi

FAST_FAILURES=0
while true; do
  START=$SECONDS
  "$PYTHON_BIN" -m rwkv_lab.vision_train \
    --rwkv "$RWKV" \
    --vision-backend radio_v4h \
    --radio-v4h-model "$RADIO_MODEL" \
    --radio-v4h-revision "$REVISION" \
    --radio-v4h-native --radio-v4h-native-packing cells \
    --radio-v4h-feature-width 4096 "${cache_only[@]}" \
    --radio-v4h-max-edge 2048 --radio-v4h-bridge-rank 256 \
    --no-radio-adaptive-complexity \
    --data "$TRAIN" --eval-data "$EVAL" --out "$RUN" \
    --feature-cache "$CACHE" \
    --steps "${V4H_STEPS:-45000}" --lr "${V4H_LR:-2e-4}" \
    --weight-decay 0.01 \
    --batch 1 --min-batch 1 --max-batch "${V4H_MAX_BATCH:-8}" \
    --target-batch-tokens "${V4H_BATCH_TOKENS:-4096}" \
    --ocr-update-ratio "${OCR_UPDATE_RATIO:-0.15}" \
    --structured-update-ratio "${STRUCTURED_UPDATE_RATIO:-0}" \
    --activation-checkpoint-min-tokens "${V4H_CHECKPOINT_TOKENS:-4000}" \
    --activation-checkpoint-max-layers \
      "${V4H_ACTIVATION_CHECKPOINT_MAX_LAYERS:-16}" \
    --max-text-tokens 768 --prefix-tokens 256 \
    --sandwich-prompt --sandwich-lead-prompt $'An image follows:\n' \
    --prompt $'Describe this image:\n' \
    --vision-resampler-layers 0 \
    --deep-vision-layers 8,16,24 --deep-vision-rank 256 \
    "${deep_grouped[@]}" \
    --grounding-early-tokens 24 --grounding-early-weight 3 \
    --grounding-contrastive-weight 0.1 --grounding-contrastive-dim 512 \
    --grounding-temperature 0.07 \
    --structured-head --structured-weight 0 \
    --structured-lr "${STRUCTURED_LR:-0}" \
    --structured-coordinate-weight "${STRUCTURED_COORDINATE_WEIGHT:-4.0}" \
    --structured-invalid-box-weight 0.5 --structured-invalid-box-margin 1.0 \
    --structured-width 256 --structured-object-queries 16 \
    --structured-spatial-layers 2 --structured-object-layers 2 \
    --structured-heads 8 \
    --loop-count "${V4H_LOOP_COUNT:-2}" --loop-index \
    --loop-start-step 1 --loop-ramp-steps 0 \
    --loop-gate-cap 0.25 --loop-lr "${V4H_LOOP_LR:-5e-5}" \
    --allow-loop-count-increase-from 1 \
    "${FINGERPRINT_ALLOWANCE[@]}" \
    "${STRUCTURED_LR_ALLOWANCE[@]}" \
    --engram --engram-sites 3,15 --engram-drow 128 --engram-rows 65536 \
    --engram-lr 1e-3 --engram-warmup-steps 0 --engram-boundary-id 0 \
    --nextlat-weight 0.1 --nextlat-hidden 1024 \
    --manifest-stat-workers 64 --prefetch-next-batch \
    "${cuda_prefetch[@]}" --allow-batch-resize \
    --checkpoint-every 50 --eval-every 500 \
    --eval-examples 448 --eval-batch-size 48 --eval-batch-tokens 49152 \
    --eval-sample-every 1000 --eval-samples 14 \
    --eval-ocr-samples 4 --eval-structured-samples 0 \
    --eval-sample-exclude-sources joy,community_gallery,restricted,private_web,manga,pose_vr,grid \
    --eval-sample-max-new 768 --require-fused-ce \
    --restart-before-eval --profile-steps "${V4H_PROFILE_STEPS:-3}" \
    --operator-profile-steps "${V4H_OPERATOR_PROFILE_STEPS:-0}" \
    "${resume[@]}"
  CODE=$?
  ELAPSED=$((SECONDS - START))

  if [[ "$CODE" -eq 0 ]]; then
    echo "[supervisor] reached --steps; done."
    break
  fi
  if [[ "$CODE" -ne 42 ]]; then
    echo "[supervisor] exit $CODE after ${ELAPSED}s" >&2
    exit "$CODE"
  fi
  if [[ "$ELAPSED" -lt 60 ]]; then
    FAST_FAILURES=$((FAST_FAILURES + 1))
    if [[ "$FAST_FAILURES" -ge 3 ]]; then
      echo "[supervisor] three fast restarts; stopping" >&2
      exit 1
    fi
  else
    FAST_FAILURES=0
  fi
  echo "[supervisor] planned eval restart after ${ELAPSED}s"
  resume=(--resume auto)
done
