#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
exec .venv/bin/python -m rwkv_lab.vision_native_train \
  --baseline runs/moonvit_rwkv_eight_hour_grounded/best/ckpt_step_00021900.pt \
  --compressor models/vision/teacher-compressor-so400m-dinov2-sam-128x1024-v1.pt \
  --data curated_vision/vision_next_shard_000_train.jsonl \
  --eval-data curated_vision/vision_eight_hour_eval.jsonl \
  --out runs/native_compressor_rwkv_arm2 \
  --steps 5000 --batch 8 --workers 8 \
  --lr 2e-4 --eval-every 100 --eval-examples 64 \
  --checkpoint-every 50 --resume auto
