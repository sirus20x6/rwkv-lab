#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${MAGE_FLOW_PYTHON_BIN:-$REPO_ROOT/.venv-mage-flow/bin/python}"
REDDIT_ROOT="${MAGE_FLOW_REDDIT_ROOT:-/thearray/git/datasets/porn/reddit/subreddits}"
CAPTION_SOURCE="${MAGE_FLOW_CAPTION_SOURCE:-$REDDIT_ROOT/qwen3.6-35b-a3b-test-4096.captions.partial.jsonl}"
ARTIFACT_SOURCE="${MAGE_FLOW_ARTIFACT_SOURCE:-$REDDIT_ROOT/qwen3.6-35b-a3b-test-4096.artifacts.partial.jsonl}"
DATA_DIR="${MAGE_FLOW_DATA_DIR:-$REPO_ROOT/curated_vision/mage_flow_edit_reddit_cpt}"
RUN_DIR="${MAGE_FLOW_RUN_DIR:-$REPO_ROOT/runs/mage_flow_edit_reddit_cpt_plan}"
OUTPUT_DIR="${MAGE_FLOW_OUTPUT_DIR:-$REPO_ROOT/runs/mage_flow_edit_reddit_cpt}"

mkdir -p "$DATA_DIR"

PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}" \
"$PYTHON_BIN" -m rwkv_lab.mage_flow_pretrain prepare-reddit \
  --captions "$CAPTION_SOURCE" \
  --artifact-manifest "$ARTIFACT_SOURCE" \
  --image-root "$REDDIT_ROOT" \
  --train-output "$DATA_DIR/train.jsonl" \
  --eval-output "$DATA_DIR/eval.jsonl" \
  --train-count "${MAGE_FLOW_TRAIN_IMAGES:-5000}" \
  --eval-count "${MAGE_FLOW_EVAL_IMAGES:-128}" \
  --require-clean-artifacts \
  --allow-smaller-train \
  --train-count-multiple 8 \
  --pixel-budget 1048576 \
  --max-side 2048 \
  --max-aspect-ratio 4 \
  --workers "${MAGE_FLOW_PREP_WORKERS:-16}"

TRAIN_ROWS="$(wc -l < "$DATA_DIR/train.jsonl")"
GRAD_ACCUM="${MAGE_FLOW_GRAD_ACCUM:-4}"
PACKED_TOKENS="${MAGE_FLOW_PACKED_TOKENS:-10240}"
EFFECTIVE_IMAGES_PER_STEP=$((2 * GRAD_ACCUM))
if (( TRAIN_ROWS % EFFECTIVE_IMAGES_PER_STEP != 0 )); then
  printf 'Train rows (%d) must divide evenly by effective batch (%d).\n' \
    "$TRAIN_ROWS" "$EFFECTIVE_IMAGES_PER_STEP" >&2
  exit 1
fi
ONE_EPOCH_STEPS=$((TRAIN_ROWS / EFFECTIVE_IMAGES_PER_STEP))

PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}" \
"$PYTHON_BIN" -m rwkv_lab.mage_flow_pretrain plan \
  --train-manifest "$DATA_DIR/train.jsonl" \
  --eval-manifest "$DATA_DIR/eval.jsonl" \
  --run-dir "$RUN_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --max-steps "${MAGE_FLOW_MAX_STEPS:-$ONE_EPOCH_STEPS}" \
  --packed-sequence-tokens "$PACKED_TOKENS" \
  --gradient-accumulation-steps "$GRAD_ACCUM" \
  --learning-rate "${MAGE_FLOW_LR:-2e-6}" \
  --warmup-steps "${MAGE_FLOW_WARMUP_STEPS:-40}" \
  --caption-dropout "${MAGE_FLOW_CAPTION_DROPOUT:-0.1}" \
  --checkpoint-every "${MAGE_FLOW_CHECKPOINT_EVERY:-100}" \
  --eval-every "${MAGE_FLOW_EVAL_EVERY:-100}" \
  --eval-packs "${MAGE_FLOW_EVAL_PACKS:-32}" \
  --eval-gen-every "${MAGE_FLOW_EVAL_GEN_EVERY:-100}" \
  --eval-gen-samples "${MAGE_FLOW_EVAL_GEN_SAMPLES:-4}" \
  --eval-gen-steps "${MAGE_FLOW_EVAL_GEN_STEPS:-30}" \
  --eval-gen-cfg "${MAGE_FLOW_EVAL_GEN_CFG:-5.0}" \
  --seed "${MAGE_FLOW_SEED:-42}"

printf 'Prepared Mage-Flow run: %s (%d clean images, %d one-epoch steps)\n' \
  "$RUN_DIR/launch.sh" "$TRAIN_ROWS" "$ONE_EPOCH_STEPS"
