#!/usr/bin/env bash
# OCR-curriculum phase: location crops, balanced OCR sampler, and all OCR evals.
set -euo pipefail
cd /thearray/git/moe-mla

export V4H_TRAIN="${V4H_TRAIN:-curated_vision/captioning_ocr_fixed_train.jsonl}"
export V4H_EVAL="${V4H_EVAL:-curated_vision/captioning_ocr_fixed_eval.jsonl}"
export V4H_RUN="${V4H_RUN:-runs/radio_v4h_captioning_ocr_fixed}"
# best/ republishes a new step-qualified checkpoint on every eval improvement,
# so that filename is transient. best.json is the atomic manifest naming the
# current winner; ckpt.pt is the legacy alias for runs predating the manifest.
if [[ -z "${V4H_SOURCE:-}" ]]; then
  best_dir="${V4H_BEST_DIR:-runs/radio_v4h_captioning_first/best}"
  best_name="$(python -c \
    'import json,sys; print(json.load(open(sys.argv[1]))["checkpoint"])' \
    "$best_dir/best.json" 2>/dev/null || true)"
  V4H_SOURCE="$best_dir/${best_name:-ckpt.pt}"
  [[ -f "$V4H_SOURCE" ]] || {
    echo "no published best checkpoint under $best_dir" >&2
    exit 1
  }
fi
export V4H_SOURCE
export V4H_CACHE="${V4H_CACHE:-/thearray/downloads/cache/moe-mla/radio_v4h_native_captioning_first}"
export V4H_INIT_MODE="${V4H_INIT_MODE:-full}"
export OCR_UPDATE_RATIO="${OCR_UPDATE_RATIO:-0.15}"
# This manifest is built with --coco-sam 0 --lvis-sam 0, so it carries no
# task="sam_mask" rows: multitask_balanced_indices aborts the run at startup for
# any non-zero structured ratio, and the structured head can never receive a
# gradient here. Keep the sampler, the loss, and its LR group switched off,
# matching scripts/run_v4h_all_adaptors_7b.sh on the same manifest.
export STRUCTURED_UPDATE_RATIO="${STRUCTURED_UPDATE_RATIO:-0}"
export STRUCTURED_WEIGHT="${STRUCTURED_WEIGHT:-0}"

exec scripts/run_v4h_captioning_first.sh
