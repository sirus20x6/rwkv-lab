#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
exec .venv/bin/python -m rwkv_lab.vision_rwkv_student_train \
  --data curated_vision/vision_next_shard_000_ocr10_train.jsonl \
  --eval-data curated_vision/vision_next_shard_000_ocr10_eval.jsonl \
  --moon-cache /thearray/downloads/cache/moe-mla/moonvit_next_128_shard_000_ocr10 \
  --fusion-cache /thearray/downloads/cache/moe-mla/fusion_so400m_next_128_shard_000_ocr10 \
  --out runs/vision_rwkv_student_1p22b \
  --hidden-size 2048 --layers 26 --head-size 64 --ffn-hidden 7168 \
  --batch 8 --workers 8 --steps 30000 \
  --eval-every 250 --eval-examples 32 --checkpoint-every 250 \
  --resume auto "$@"
