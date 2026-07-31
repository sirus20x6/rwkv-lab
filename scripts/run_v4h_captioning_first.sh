#!/usr/bin/env bash
# C-RADIOv4-H arm of the captioning-first run.
#
# WHAT CHANGED vs the RADIO1D arm:
#   * encoder: C-RADIOv4-H (652M) emitting native (H/16,W/16) spatial patches at
#     1280ch, instead of RADIO1D-H's nested 1D tokens at 2560ch. Same teachers
#     (SigLIP2-g, DINOv3-7B, SAM3) -- the difference is basis, not supervision.
#   * NO TILING. Each image is encoded whole at its own resolution (floored to
#     a multiple of 32, capped at 2048), so nothing is upscaled and no token is
#     spent on a letterboxed thumbnail. Tiling only existed because RADIO1D
#     emitted a fixed token count per tile; v4h's scales with resolution.
#   * downscale INTER_AREA / upscale INTER_LANCZOS4 via cv2 (~4x PIL here).
#   * --radio-adaptive-complexity is REJECTED here: it truncates RADIO1D's
#     nested prefix, and v4h tokens are positional.
#
# BRIDGE: v4h's 1280 channels reach the LM's 2560 by concatenating two
# horizontally adjacent cells -- parameter-free and lossless. RWKV-2.9B's 2560
# is exactly 2x1280, which is what makes this possible at all. The remaining
# 1.49M bridge is LayerNorm + a zero-init gated low-rank residual + per-token
# Fourier geometry; there are no tile/token index tables because a flat token
# index means different places in differently-shaped grids.
#
# --init-text-adapters-from carries loop/Engram/NextLat/grounding/structured
# state from the RADIO1D run and resets ONLY the vision bridge, which is the
# one component whose input width changed.
set -uo pipefail
cd /thearray/git/moe-mla

PYTHON_BIN="${VISION_PYTHON:-/usr/bin/python}"
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}src"

RUN="${V4H_RUN:-runs/radio_v4h_captioning_first}"
SRC="${V4H_SOURCE:-runs/radio1d_rwkv_captioning_first/last.pt}"
CACHE="${V4H_CACHE:-/thearray/downloads/cache/moe-mla/radio_v4h_native_captioning_first}"
REVISION="${V4H_REVISION:-c-radiov4-h-native-h16w32}"
TRAIN="${V4H_TRAIN:-curated_vision/captioning_first_train.jsonl}"
EVAL="${V4H_EVAL:-curated_vision/captioning_first_eval.jsonl}"
INIT_MODE="${V4H_INIT_MODE:-text}"

for required in "$TRAIN" "$EVAL" "$SRC" "$CACHE" \
                models/vision/C-RADIOv4-H \
                models/rwkv7-g1h-2.9b-20260710-ctx10240.pth; do
  [[ -e "$required" ]] || { echo "missing required input: $required" >&2; exit 1; }
done

mkdir -p "$RUN"
exec 9>"$RUN/.launcher.lock"
flock -n 9 || { echo "another launcher already owns $RUN" >&2; exit 75; }

# First launch warm-starts the text side; every eval restart resumes in place.
if [[ -f "$RUN/last.pt" ]]; then
  resume=(--resume auto)
else
  case "$INIT_MODE" in
    text) resume=(--resume none --init-text-adapters-from "$SRC") ;;
    full) resume=(--resume none --init-adapters-from "$SRC") ;;
    *) echo "V4H_INIT_MODE must be text or full" >&2; exit 2 ;;
  esac
fi

FAST_FAILURES=0
while true; do
  START=$SECONDS
  "$PYTHON_BIN" -m rwkv_lab.vision_train \
    --vision-backend radio_v4h \
    --radio-v4h-model models/vision/C-RADIOv4-H \
    --radio-v4h-revision "$REVISION" \
    --radio-v4h-native --radio-v4h-max-edge 2048 \
    --radio-v4h-bridge-rank 256 --radio-v4h-pair-axis columns \
    --no-radio-adaptive-complexity \
    --data "$TRAIN" --eval-data "$EVAL" --out "$RUN" \
    --feature-cache "$CACHE" \
    --radio-max-detail-tiles 48 --radio-tile-batch 8 \
    --steps "${V4H_STEPS:-45000}" --lr 2e-4 --weight-decay 0.01 \
    --batch 1 --min-batch 1 --max-batch 4 --target-batch-tokens 4096 \
    --ocr-update-ratio "${OCR_UPDATE_RATIO:-0}" \
    --structured-update-ratio "${STRUCTURED_UPDATE_RATIO:-0}" \
    --activation-checkpoint-min-tokens 4096 \
    --max-text-tokens 768 --prefix-tokens 256 \
    --sandwich-prompt --sandwich-lead-prompt $'An image follows:\n' \
    --prompt $'Describe this image:\n' \
    --vision-resampler-layers 0 \
    --deep-vision-layers 8,16,24 --deep-vision-rank 256 \
    --grounding-early-tokens 24 --grounding-early-weight 3 \
    --grounding-contrastive-weight 0.1 --grounding-contrastive-dim 512 \
    --grounding-temperature 0.07 \
    --structured-head --structured-weight "${STRUCTURED_WEIGHT:-1.0}" \
    --structured-lr "${STRUCTURED_LR:-0}" \
    --structured-coordinate-weight "${STRUCTURED_COORDINATE_WEIGHT:-4.0}" \
    --structured-invalid-box-weight 0.5 --structured-invalid-box-margin 1.0 \
    --structured-width 256 --structured-object-queries 16 \
    --structured-spatial-layers 2 --structured-object-layers 2 \
    --structured-heads 8 \
    --loop-count 2 --loop-index --loop-start-step 1 --loop-ramp-steps 0 \
    --loop-gate-cap 0.25 --loop-lr 1e-5 \
    --engram --engram-sites 3,15 --engram-drow 128 --engram-rows 65536 \
    --engram-lr 1e-3 --engram-warmup-steps 0 --engram-boundary-id 0 \
    --nextlat-weight 0.1 --nextlat-hidden 1024 \
    --manifest-stat-workers 64 --prefetch-next-batch --allow-batch-resize \
    --checkpoint-every 50 --eval-every 500 \
    --eval-examples 448 --eval-batch-size 4 \
    --eval-sample-every 1000 --eval-samples 14 \
    --eval-ocr-samples "${EVAL_OCR_SAMPLES:-4}" \
    --eval-structured-samples "${EVAL_STRUCTURED_SAMPLES:-4}" \
    --eval-sample-exclude-sources joy,civitai,nsfw,porn,manga,pose_vr,grid \
    --eval-sample-max-new 768 --require-fused-ce \
    --restart-before-eval --profile-steps 3 \
    "${resume[@]}"
  CODE=$?
  ELAPSED=$(( SECONDS - START ))

  if [ "$CODE" -eq 0 ]; then echo "[supervisor] reached --steps; done."; break; fi
  if [ "$CODE" -ne 42 ]; then
    echo "[supervisor] exit $CODE after ${ELAPSED}s — real failure, not an eval restart." >&2
    exit "$CODE"
  fi
  if [ "$ELAPSED" -lt 60 ]; then
    FAST_FAILURES=$(( FAST_FAILURES + 1 ))
    if [ "$FAST_FAILURES" -ge 3 ]; then
      echo "[supervisor] three restarts in under 60s each — stopping." >&2; exit 1
    fi
  else FAST_FAILURES=0; fi
  echo "[supervisor] planned eval restart after ${ELAPSED}s; resuming from last.pt"
  resume=(--resume auto)
done
