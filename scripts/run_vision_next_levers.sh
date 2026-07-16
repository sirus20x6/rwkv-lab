#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
PYTHON_BIN="${VISION_PYTHON:-$PWD/.venv/bin/python}"
TRAIN="${VISION_NEXT_TRAIN:-curated_vision/vision_eight_hour.jsonl}"
EVAL="${VISION_NEXT_EVAL:-curated_vision/vision_eight_hour_eval.jsonl}"
PREFIX="${VISION_NEXT_PREFIX_TOKENS:-128}"
TAPS="${VISION_NEXT_MOONVIT_TAPS:-8,17,26}"
LAYER_SITES="${VISION_NEXT_RWKV_LAYER_SITES:-8,16,24}"
VIEW_MODE="${VISION_NEXT_VIEW_MODE:-full-quadrants}"
RUN="${VISION_NEXT_RUN:-runs/moonvit_rwkv_next_levers_so400m_${PREFIX}}"
MOON_CACHE="${VISION_NEXT_MOON_CACHE:-/thearray/downloads/cache/moe-mla/moonvit_next_${PREFIX}}"
FUSION_CACHE="${VISION_NEXT_FUSION_CACHE:-/thearray/downloads/cache/moe-mla/fusion_so400m_next_${PREFIX}}"
RECEIPT="${VISION_NEXT_CACHE_RECEIPT:-curated_vision/vision_next_so400m_${PREFIX}.cache.json}"
SOURCE_RUN="${VISION_NEXT_SOURCE_RUN:-runs/moonvit_rwkv_eight_hour_grounded}"
SIGLIP_MODEL="${VISION_NEXT_SIGLIP2_MODEL:-models/vision/siglip2-so400m-patch16-512}"
SIGLIP_WIDTH="${VISION_NEXT_SIGLIP2_WIDTH:-1152}"

for required in "$PYTHON_BIN" "$TRAIN" "$EVAL" "$RECEIPT"; do
  if [[ ! -e "$required" ]]; then
    echo "next-lever launch is not cache-ready; missing $required" >&2
    echo "run scripts/prepare_vision_next_levers.sh first" >&2
    exit 1
  fi
done

export TRAIN EVAL PREFIX TAPS VIEW_MODE MOON_CACHE FUSION_CACHE RECEIPT SIGLIP_MODEL SIGLIP_WIDTH
PYTHONPATH=src "$PYTHON_BIN" - <<'PY'
import hashlib
import json
import os
from pathlib import Path

receipt = json.loads(Path(os.environ["RECEIPT"]).read_text())
train, evaluation = Path(os.environ["TRAIN"]), Path(os.environ["EVAL"])
expected = {
    "train_sha256": hashlib.sha256(train.read_bytes()).hexdigest(),
    "eval_sha256": hashlib.sha256(evaluation.read_bytes()).hexdigest(),
    "prefix_tokens": int(os.environ["PREFIX"]),
    "moonvit_taps": os.environ["TAPS"],
    "view_mode": os.environ["VIEW_MODE"],
    "moon_cache": str(Path(os.environ["MOON_CACHE"]).resolve()),
    "fusion_cache": str(Path(os.environ["FUSION_CACHE"]).resolve()),
    "siglip2_model": str(Path(os.environ["SIGLIP_MODEL"]).resolve()),
    "siglip2_width": int(os.environ["SIGLIP_WIDTH"]),
}
bad = [name for name, value in expected.items() if receipt.get(name) != value]
for kind in ("moon", "fusion"):
    path = Path(expected[f"{kind}_cache"])
    count = sum(item.suffix == ".pt" for item in path.iterdir())
    if count != int(receipt["expected_entries"]):
        bad.append(f"{kind}_cache_count")
    if path.stat().st_mtime_ns != int(receipt[f"{kind}_cache_mtime_ns"]):
        bad.append(f"{kind}_cache_mtime")
if bad:
    raise SystemExit(f"next-lever cache receipt is stale/incompatible: {bad}")
print({"kind": "next_lever_preflight", "cache_ready": True,
       "entries": receipt["expected_entries"], "prefix_tokens": expected["prefix_tokens"]})
PY

mkdir -p "$RUN"
exec 9>"$RUN/.launcher.lock"
if ! flock -n 9; then
  echo "another launcher owns $RUN" >&2
  exit 75
fi

resume_args=(--resume auto)
if [[ ! -f "$RUN/last.pt" ]]; then
  if [[ -n "${VISION_NEXT_SOURCE_CHECKPOINT:-}" ]]; then
    SOURCE_CHECKPOINT="$VISION_NEXT_SOURCE_CHECKPOINT"
  elif SOURCE_CHECKPOINT="$("$PYTHON_BIN" scripts/vision_run_evidence.py "$SOURCE_RUN" --resolve-best)"; then
    :
  elif [[ $? == 2 ]]; then
    SOURCE_CHECKPOINT="$SOURCE_RUN/last.pt"
  else
    echo "source best checkpoint publication is invalid: $SOURCE_RUN" >&2
    exit 1
  fi
  if [[ ! -f "$SOURCE_CHECKPOINT" || -L "$SOURCE_CHECKPOINT" ]]; then
    echo "missing source checkpoint: $SOURCE_CHECKPOINT" >&2
    exit 1
  fi
  resume_args=(--resume none --init-adapters-from "$SOURCE_CHECKPOINT")
fi

PYTHONPATH=src exec "$PYTHON_BIN" -m rwkv_lab.vision_train \
  --data "$TRAIN" --eval-data "$EVAL" \
  --out "$RUN" --steps "${VISION_NEXT_STEPS:-30392}" \
  --batch 4 --min-batch 2 --max-batch 16 \
  --target-batch-tokens "${VISION_NEXT_BATCH_TOKENS:-4096}" \
  --max-text-tokens 768 --prefix-tokens "$PREFIX" \
  --feature-cache "$MOON_CACHE" --fusion-feature-cache "$FUSION_CACHE" \
  --moonvit-tap-layers "$TAPS" --layer-vision-layers "$LAYER_SITES" \
  --layer-vision-rank 256 --vision-view-mode "$VIEW_MODE" \
  --sandwich-prompt --vision-fusion --vision-fusion-rank 512 \
  --siglip2-model "$SIGLIP_MODEL" --siglip2-width "$SIGLIP_WIDTH" \
  --vision-resampler-layers 2 --vision-resampler-width 1024 \
  --vision-resampler-heads 8 --deep-vision-layers 8,16,24 \
  --deep-vision-rank 256 --grounding-early-tokens 24 \
  --grounding-early-weight 3 --grounding-contrastive-weight 0.1 \
  --grounding-contrastive-dim 512 --grounding-temperature 0.07 \
  --loop-count 2 --loop-index --loop-start-step 1 --loop-ramp-steps 0 \
  --loop-gate-cap 0.25 --loop-lr 1e-5 \
  --engram --engram-sites 3,15 --engram-drow 128 --engram-rows 65536 \
  --engram-lr 1e-3 --engram-warmup-steps 0 --engram-boundary-id 0 \
  --nextlat-weight 0.1 --nextlat-hidden 1024 \
  --manifest-stat-workers 64 --preload-feature-cache \
  --background-feature-preload --prefetch-next-batch \
  --checkpoint-every 50 --eval-every 100 --eval-samples 12 \
  --eval-sample-exclude-sources joy,civitai,nsfw,porn,manga,pose_vr,grid \
  --eval-sample-max-new 768 --require-fused-ce \
  "${resume_args[@]}"
