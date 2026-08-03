#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
PYTHON_BIN="${VISION_PYTHON:-$PWD/.venv/bin/python}"
RUN="${VISION_COMPRESSOR_RUN:-runs/vision_teacher_compressor_so400m_ocr10}"
TRAIN="${VISION_COMPRESSOR_TRAIN:-curated_vision/vision_next_shard_000_ocr10_train.jsonl}"
EVAL="${VISION_COMPRESSOR_EVAL:-curated_vision/vision_next_shard_000_ocr10_eval.jsonl}"
MOON_CACHE="${VISION_COMPRESSOR_MOON_CACHE:-/thearray/downloads/cache/moe-mla/moonvit_next_128_shard_000_ocr10}"
FUSION_CACHE="${VISION_COMPRESSOR_FUSION_CACHE:-/thearray/downloads/cache/moe-mla/fusion_so400m_next_128_shard_000_ocr10}"
INIT_FROM="${VISION_COMPRESSOR_INIT_FROM:-}"

for required in "$PYTHON_BIN" "$TRAIN" "$EVAL" "$MOON_CACHE" "$FUSION_CACHE"; do
  if [[ ! -e "$required" ]]; then
    echo "compressor launch is not cache-ready; missing $required" >&2
    exit 1
  fi
done

mkdir -p "$RUN"
exec 9>"$RUN/.launcher.lock"
if ! flock -n 9; then
  echo "another process owns $RUN" >&2
  exit 75
fi

EXTRA_ARGS=()
if [[ -n "$INIT_FROM" ]]; then
  if [[ ! -f "$INIT_FROM" ]]; then
    echo "compressor initialization checkpoint is missing: $INIT_FROM" >&2
    exit 1
  fi
  EXTRA_ARGS+=(--init-from "$INIT_FROM")
fi

PYTHONPATH=src exec "$PYTHON_BIN" -m rwkv_lab.vision_teacher_compressor \
  --data "$TRAIN" --eval-data "$EVAL" \
  --moon-cache "$MOON_CACHE" --fusion-cache "$FUSION_CACHE" \
  --out "$RUN" --steps "${VISION_COMPRESSOR_STEPS:-6000}" \
  --batch "${VISION_COMPRESSOR_BATCH:-8}" \
  --workers "${VISION_COMPRESSOR_WORKERS:-8}" \
  --lr "${VISION_COMPRESSOR_LR:-2e-4}" \
  --teacher-dropout "${VISION_COMPRESSOR_TEACHER_DROPOUT:-0.15}" \
  --eval-every "${VISION_COMPRESSOR_EVAL_EVERY:-250}" \
  --checkpoint-every "${VISION_COMPRESSOR_CHECKPOINT_EVERY:-250}" \
  --resume auto "${EXTRA_ARGS[@]}"
