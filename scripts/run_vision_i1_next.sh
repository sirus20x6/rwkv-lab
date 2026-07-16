#!/usr/bin/env bash
set -euo pipefail

# Phase 2: retain the completed stage-1 adapters, reset run-local optimizer and
# sampler state, and train on the image-verified i1 + curated Civitai mixture.
# The supplied Civitai eval split remains strictly held out.
# Run only after moonvit_rwkv_stage1_v3 has completed.
cd "$(dirname "$0")/.."

resolve_best_checkpoint() {
  python scripts/vision_run_evidence.py "$1" --resolve-best
}

if [[ ! -f curated_vision/vision_next_i1_civitai.jsonl ]] || \
   [[ ! -f curated_vision/vision_next_i1_civitai_eval.jsonl ]]; then
  echo "missing i1/Civitai manifests; run: python scripts/build_i1_civitai_mix.py" >&2
  exit 1
fi
RUN=runs/moonvit_rwkv_i1_civitai_phase2
mkdir -p "$RUN"
exec 9>"$RUN/.launcher.lock"
if ! flock -n 9; then
  echo "another phase-2 launcher already owns $RUN" >&2
  exit 75
fi
resume_args=(--resume auto)
if [[ -L "$RUN/last.pt" ]]; then
  echo "refusing symlinked phase-2 checkpoint: $RUN/last.pt" >&2
  exit 1
fi
if [[ ! -f "$RUN/last.pt" ]]; then
  RUN_EVIDENCE="$(python scripts/vision_run_evidence.py \
    "$RUN" --allow-step-zero-preloop)"
  startup_retry_args=()
  case "$RUN_EVIDENCE" in
    committed|exact)
      echo "refusing to restart an existing phase-2 run without $RUN/last.pt" >&2
      exit 1
      ;;
    startup)
      # --fresh archives a step-0 startup log so vision_train's overwrite guard
      # permits retry. The evidence classifier rejects every positive/unknown
      # step before this narrowly scoped cleanup is authorized.
      startup_retry_args=(--fresh)
      ;;
    pristine) ;;
    *)
      echo "invalid phase-2 run evidence classification: $RUN_EVIDENCE" >&2
      exit 1
      ;;
  esac
  if SOURCE_CHECKPOINT="$(resolve_best_checkpoint runs/moonvit_rwkv_stage1_v3)"; then
    :
  elif [[ $? == 2 ]]; then
    SOURCE_CHECKPOINT=runs/moonvit_rwkv_stage1_v3/last.pt
  else
    echo "stage-1 best checkpoint publication is invalid" >&2
    exit 1
  fi
  if [[ ! -f "$SOURCE_CHECKPOINT" || -L "$SOURCE_CHECKPOINT" ]]; then
    echo "missing stage-1 adapter checkpoint" >&2
    exit 1
  fi
  python - <<'PY'
import json
from pathlib import Path
status = json.loads(Path("runs/moonvit_rwkv_stage1_v3/status.json").read_text())
if status.get("state") not in {"complete", "stopped"}:
    raise SystemExit(
        f"stage-1 is {status.get('state')} at step {status.get('step')}; "
        "phase 2 must wait for its final checkpoint"
    )
PY
  resume_args=(--resume none --init-adapters-from "$SOURCE_CHECKPOINT"
               "${startup_retry_args[@]}")
fi

# A recovered run validates every feature as its exact batch is loaded and can
# regenerate a miss on demand. Avoid delaying each restart behind a full-corpus
# verification; set SKIP_CACHE_VERIFY=0 explicitly to force a maintenance scan.
if [[ -f "$RUN/last.pt" && -z "${SKIP_CACHE_VERIFY+x}" ]]; then
  SKIP_CACHE_VERIFY=1
fi

# Reuse every stage-1 entry and fill only new i1/Civitai/eval images. MoonViT's
# small cache-prefill kernels leave this large GPU underfilled in one process,
# so deterministic shards run concurrently. Entries are atomic and restartable.
if [[ "${SKIP_CACHE_VERIFY:-0}" != 1 ]]; then
  # Default to one MoonViT/CUDA context. Shards remain available for an explicit
  # multi-GPU launch, but multiple copies on one GPU regress throughput and VRAM.
  CACHE_SHARDS="${CACHE_SHARDS:-1}"
  cache_pids=()
  cleanup_cache() {
    if ((${#cache_pids[@]})); then
      kill "${cache_pids[@]}" 2>/dev/null || true
    fi
  }
  trap cleanup_cache EXIT INT TERM
  for ((shard=0; shard<CACHE_SHARDS; shard++)); do
    PYTHONPATH=src python -m rwkv_lab.vision_cache \
      --data curated_vision/vision_next_i1_civitai.jsonl \
             curated_vision/vision_next_i1_civitai_eval.jsonl \
      --cache caches/moonvit_features_stage1_v3 \
      --batch 32 \
      --workers 8 \
      --num-shards "$CACHE_SHARDS" \
      --shard-index "$shard" &
    cache_pids+=("$!")
  done
  for pid in "${cache_pids[@]}"; do
    wait "$pid"
  done
  cache_pids=()
  trap - EXIT INT TERM
fi

# This run is supervised by the watchdog and is intentionally continuous.
# Keep the trainer's required finite bound far beyond any practical run while
# still allowing an operator to override it deliberately.
VISION_TARGET_STEPS="${VISION_TARGET_STEPS:-1000000000}"

PYTHONPATH=src python -m rwkv_lab.vision_train \
  --data curated_vision/vision_next_i1_civitai.jsonl \
  --eval-data curated_vision/vision_next_i1_civitai_eval.jsonl \
  --steps "$VISION_TARGET_STEPS" \
  --batch 8 \
  --min-batch 4 \
  --max-batch 32 \
  --target-batch-tokens 3584 \
  --loop-token-budget-scale 1.0 \
  --max-text-tokens 768 \
  --out runs/moonvit_rwkv_i1_civitai_phase2 \
  --feature-cache caches/moonvit_features_stage1_v3 \
  --manifest-stat-workers 64 \
  --preload-feature-cache \
  --background-feature-preload \
  --checkpoint-every 50 \
  --eval-every 100 \
  --eval-samples 4 \
  --eval-sample-max-new 64 \
  --profile-steps 0 \
  --require-fused-ce \
  --loop-lr 1e-5 \
  --loop-start-step 1 \
  --loop-ramp-steps 0 \
  --engram \
  --engram-sites 3,15 \
  --engram-drow 128 \
  --engram-rows 65536 \
  --engram-lr 1e-3 \
  --engram-warmup-steps 1000 \
  --engram-boundary-id 0 \
  "${resume_args[@]}"
