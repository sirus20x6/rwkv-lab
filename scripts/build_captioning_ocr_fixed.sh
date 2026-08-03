#!/usr/bin/env bash
# Build new OCR/manifests beside the active run's immutable inputs.
set -euo pipefail
cd /thearray/git/moe-mla

# The mixer derives its DoclingMatix target from the non-OCR source counts, so
# a hardcoded row count silently over-materializes whenever the mix changes
# (--coco-sam 0 --lvis-sam 0 removes 10k non-OCR rows). Ask the mixer for the
# number using the very flags it will be run with.
mix_flags=(--coco-sam 0 --lvis-sam 0)
train_rows="$(python scripts/build_captioning_first_mix.py \
  "${mix_flags[@]}" --print-ocr-target)"
# build_captioning_first_mix.py consumes exactly --eval-per-source (64) OCR eval
# rows off the front of the eval manifest, so build with margin rather than an
# exact-fit coupling between two scripts' defaults.
eval_rows="${OCR_EVAL_ROWS:-96}"

python scripts/build_doclingmatix_ocr_mix.py \
  --target-train-rows "$train_rows" \
  --eval-rows "$eval_rows" \
  --train-output curated_vision/ocr_curriculum_train.jsonl \
  --eval-output curated_vision/ocr_curriculum_eval.jsonl \
  --combined-train curated_vision/vision_next_ocr_curriculum_train.jsonl \
  --combined-eval curated_vision/vision_next_ocr_curriculum_eval.jsonl \
  --receipt curated_vision/ocr_curriculum.summary.json

python scripts/build_captioning_first_mix.py \
  --ocr-train curated_vision/ocr_curriculum_train.jsonl \
  --ocr-eval curated_vision/ocr_curriculum_eval.jsonl \
  "${mix_flags[@]}" \
  --output curated_vision/captioning_ocr_fixed_train.jsonl \
  --eval-output curated_vision/captioning_ocr_fixed_eval.jsonl
