#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 || ( "$1" != moonvit && "$1" != compressor ) ]]; then
  echo "usage: $0 {moonvit|compressor}" >&2
  exit 2
fi
cd "$(dirname "$0")/.."
ARM="$1"
PYTHON_BIN="${VISION_PYTHON:-$PWD/.venv/bin/python}"
TRAIN="curated_vision/vision_next_shard_001_ocr10_train.jsonl"
EVAL="curated_vision/vision_next_shard_000_ocr10_eval.jsonl"
MOON_CACHE="/workspace/downloads/cache/moe-mla/moonvit_next_128_shard_001_ocr10"
FUSION_CACHE="/workspace/downloads/cache/moe-mla/fusion_so400m_next_128_shard_001_ocr10"
COMPRESSOR="models/vision/teacher-compressor-so400m-dinov2-sam-128x1024-v1.pt"
RUN="runs/vision_ab_${ARM}"

for required in "$PYTHON_BIN" "$TRAIN" "$EVAL" "$MOON_CACHE"; do
  [[ -e "$required" ]] || { echo "A/B input missing: $required" >&2; exit 1; }
done
extra=()
if [[ "$ARM" == compressor ]]; then
  for required in "$FUSION_CACHE" "$COMPRESSOR"; do
    [[ -e "$required" ]] || { echo "compressor A/B input missing: $required" >&2; exit 1; }
  done
  extra+=(--fusion-feature-cache "$FUSION_CACHE"
          --vision-compressor-checkpoint "$COMPRESSOR")
fi

mkdir -p "$RUN"
exec 9>"$RUN/.launcher.lock"
flock -n 9 || { echo "another launcher owns $RUN" >&2; exit 75; }

PYTHONPATH=src exec "$PYTHON_BIN" -m rwkv_lab.vision_train \
  --data "$TRAIN" --eval-data "$EVAL" --out "$RUN" \
  --steps "${VISION_AB_STEPS:-2000}" \
  --batch 4 --min-batch 2 --max-batch 16 --target-batch-tokens 4096 \
  --max-text-tokens 768 --prefix-tokens 128 \
  --feature-cache "$MOON_CACHE" --feature-cache-max-bytes $((32 * 1024 * 1024 * 1024)) \
  --moonvit-tap-layers 8,17,26 --vision-view-mode full-quadrants \
  --vision-resampler-layers 0 --sandwich-prompt \
  --deep-vision-layers 8,16,24 --deep-vision-rank 256 \
  --grounding-early-tokens 24 --grounding-early-weight 3 \
  --grounding-contrastive-weight 0.1 --grounding-contrastive-dim 512 \
  --grounding-temperature 0.07 \
  --loop-count 2 --loop-index --loop-start-step 1 --loop-ramp-steps 0 \
  --loop-gate-cap 0.25 --loop-lr 1e-5 \
  --engram --engram-sites 3,15 --engram-drow 128 --engram-rows 65536 \
  --engram-lr 1e-3 --engram-warmup-steps 0 --engram-boundary-id 0 \
  --nextlat-weight 0.1 --nextlat-hidden 1024 \
  --manifest-stat-workers 64 --prefetch-next-batch \
  --checkpoint-every 100 --eval-every 250 --eval-examples 64 \
  --eval-samples 8 --eval-sample-max-new 256 \
  --eval-sample-exclude-sources joy,community_gallery,restricted,private_web,manga,pose_vr,grid \
  --profile-steps 10 --require-fused-ce --resume auto \
  "${extra[@]}"
