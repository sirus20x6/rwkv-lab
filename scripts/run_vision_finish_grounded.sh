#!/usr/bin/env bash
set -euo pipefail

# Grounded finishing phase. The first launch imports every learned adapter from
# the phase-2 best checkpoint while intentionally creating a new optimizer,
# sampler, RNG stream, and eval baseline for the new dataset. Later launches
# resume this run's exact last.pt state.
cd "$(dirname "$0")/.."
PYTHON_BIN="${VISION_PYTHON:-$PWD/.venv/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "vision Python is not executable: $PYTHON_BIN" >&2
  exit 1
fi

resolve_best_checkpoint() {
  "$PYTHON_BIN" scripts/vision_run_evidence.py "$1" --resolve-best
}

RUN="${VISION_FINISH_RUN:-runs/moonvit_rwkv_finish_grounded}"
mkdir -p "$RUN"
exec 9>"$RUN/.launcher.lock"
if ! flock -n 9; then
  echo "another finishing launcher already owns $RUN" >&2
  exit 75
fi
TRAIN="${VISION_FINISH_TRAIN:-curated_vision/vision_finish_grounded.jsonl}"
EVAL="${VISION_FINISH_EVAL:-curated_vision/vision_finish_grounded_eval.jsonl}"
SUMMARY="${VISION_FINISH_SUMMARY:-curated_vision/vision_finish_grounded.summary.json}"
CACHE="${VISION_FINISH_CACHE:-caches/moonvit_features_stage1_v3}"
SAMPLER="${VISION_FINISH_SAMPLER:-}"
CACHE_RECEIPT="${VISION_FINISH_CACHE_RECEIPT:-}"
SOURCE_RUN="${VISION_FINISH_SOURCE_RUN:-runs/moonvit_rwkv_i1_civitai_phase2}"

for required in "$TRAIN" "$EVAL" "$SUMMARY"; do
  if [[ ! -f "$required" ]]; then
    echo "missing finishing artifact: $required" >&2
    exit 1
  fi
done
for required in "$SAMPLER" "$CACHE_RECEIPT"; do
  if [[ -n "$required" && ! -f "$required" ]]; then
    echo "missing optional finishing contract: $required" >&2
    exit 1
  fi
done

resume_args=(--resume auto)
if [[ -L "$RUN/last.pt" ]]; then
  echo "refusing symlinked finishing checkpoint: $RUN/last.pt" >&2
  exit 1
fi
if [[ -f "$RUN/last.pt" ]]; then
  SOURCE_CHECKPOINT="$RUN/last.pt"
else
  RUN_EVIDENCE="$("$PYTHON_BIN" scripts/vision_run_evidence.py \
    "$RUN" --allow-step-zero-preloop)"
  startup_retry_args=()
  case "$RUN_EVIDENCE" in
    committed|exact)
      echo "refusing to restart an existing finishing run without $RUN/last.pt" >&2
      exit 1
      ;;
    startup) startup_retry_args=(--fresh) ;;
    pristine) ;;
    *)
      echo "invalid finishing run evidence classification: $RUN_EVIDENCE" >&2
      exit 1
      ;;
  esac
  if [[ -n "${VISION_FINISH_SOURCE_CHECKPOINT:-}" ]]; then
    SOURCE_CHECKPOINT="$VISION_FINISH_SOURCE_CHECKPOINT"
  elif SOURCE_CHECKPOINT="$(resolve_best_checkpoint "$SOURCE_RUN")"; then
    :
  elif [[ $? == 2 ]]; then
    SOURCE_CHECKPOINT="$SOURCE_RUN/last.pt"
  else
    echo "source best checkpoint publication is invalid: $SOURCE_RUN" >&2
    exit 1
  fi
  if [[ ! -f "$SOURCE_CHECKPOINT" || -L "$SOURCE_CHECKPOINT" ]]; then
    echo "missing source adapter checkpoint: $SOURCE_CHECKPOINT" >&2
    exit 1
  fi
  resume_args=(--resume none --init-adapters-from "$SOURCE_CHECKPOINT"
               "${startup_retry_args[@]}")
fi
export SOURCE_CHECKPOINT TRAIN EVAL SUMMARY CACHE SAMPLER CACHE_RECEIPT

# Exact resumes already validate cache entries on use. Reserve the full cache
# walk for the first launch unless an operator explicitly requests it with
# SKIP_CACHE_VERIFY=0.
if [[ -f "$RUN/last.pt" && -z "${SKIP_CACHE_VERIFY+x}" ]]; then
  SKIP_CACHE_VERIFY=1
fi

# Verify immutable inputs and the warm-start payload before touching CUDA.
PYTHONPATH=src "$PYTHON_BIN" - <<'PY'
import hashlib
import json
import os
from pathlib import Path

import torch

train = Path(os.environ["TRAIN"])
evaluation = Path(os.environ["EVAL"])
summary = json.loads(Path(os.environ["SUMMARY"]).read_text())
actual = {
    "train_sha256": hashlib.sha256(train.read_bytes()).hexdigest(),
    "eval_sha256": hashlib.sha256(evaluation.read_bytes()).hexdigest(),
}
for name, digest in actual.items():
    if digest != summary.get(name):
        raise SystemExit(f"finishing manifest receipt mismatch: {name}")
if int(summary.get("train_eval_image_overlap", -1)) != 0:
    raise SystemExit("finishing receipt reports train/eval image leakage")
if int(summary.get("raw_civitai_rows", -1)) != 0:
    raise SystemExit("finishing receipt contains raw Civitai rows")
if int(summary.get("generationism_matches", -1)) != 0:
    raise SystemExit("finishing receipt contains generation prompt artifacts")
if int(summary.get("joy_caption_spam_matches", -1)) != 0:
    raise SystemExit("finishing receipt contains social-media caption artifacts")
if int(summary.get("blank_metadata_fields", 0)) != 0:
    raise SystemExit("finishing receipt contains blank structured metadata")
if int(summary.get("train_eval_i1_key_overlap", 0)) != 0:
    raise SystemExit("finishing receipt reports train/eval i1-key leakage")

sampler_path = os.environ.get("SAMPLER")
if sampler_path:
    sampler = json.loads(Path(sampler_path).read_text())
    expected_sampler = {
        "manifest_sha256": actual["train_sha256"],
        "rows": int(summary["rows"]),
        "max_text_tokens": 768,
        "prefix_tokens": 64,
        "target_batch_tokens": 3584,
        "min_batch": 4,
        "max_batch": 32,
        "truncated_captions": 0,
    }
    incompatible_sampler = [
        name for name, value in expected_sampler.items()
        if sampler.get(name) != value
    ]
    if incompatible_sampler:
        raise SystemExit(f"sampler receipt is incompatible: {incompatible_sampler}")
    target_steps = int(os.environ.get("VISION_FINISH_TARGET_STEPS", -1))
    if target_steps != int(sampler.get("steps_per_epoch", -2)):
        raise SystemExit(
            f"target steps {target_steps} != one epoch {sampler.get('steps_per_epoch')}"
        )

cache_receipt_path = os.environ.get("CACHE_RECEIPT")
if cache_receipt_path:
    receipt = json.loads(Path(cache_receipt_path).read_text())
    cache = Path(os.environ["CACHE"]).resolve()
    expected_cache = {
        "cache": str(cache),
        "train_sha256": actual["train_sha256"],
        "eval_sha256": actual["eval_sha256"],
        "prefix_tokens": 64,
        "max_input_patches": 1024,
        "vision_fingerprint": "c2a3f49bd46cd920618ce9093f66706da51effb677af3a282fbc6929082e9473",
        "expected_entries": int(summary["rows"]) + int(summary["eval_rows"]),
        "entries": int(summary["rows"]) + int(summary["eval_rows"]),
        "payloads_verified": True,
    }
    incompatible_cache = [
        name for name, value in expected_cache.items()
        if receipt.get(name) != value
    ]
    if incompatible_cache:
        raise SystemExit(f"cache receipt is incompatible: {incompatible_cache}")
    entries = sum(entry.name.endswith(".pt") for entry in os.scandir(cache))
    if entries != int(receipt.get("entries", -1)):
        raise SystemExit(f"cache entry count changed: {entries}")
    if cache.stat().st_mtime_ns != int(receipt.get("directory_mtime_ns", -1)):
        raise SystemExit("cache directory changed after readiness receipt")

checkpoint = Path(os.environ["SOURCE_CHECKPOINT"])
blob = torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)
if int(blob.get("schema", -1)) != 3:
    raise SystemExit(f"unsupported source checkpoint schema: {blob.get('schema')}")
missing = [name for name in
           ("projector", "nextlat", "engram", "loops", "optimizer", "sampler", "rng")
           if blob.get(name) is None]
if missing:
    raise SystemExit(f"source checkpoint lacks required state: {missing}")
saved = blob.get("args", {})
expected_structure = {
    "prefix_tokens": 64,
    "max_input_patches": 1024,
    "nextlat_hidden": 1024,
    "loop_count": 2,
    "loop_index": True,
    "loop_gate_cap": 0.25,
    "engram": True,
    "engram_sites": "3,15",
    "engram_drow": 128,
    "engram_rows": 65536,
    "engram_boundary_id": 0,
}
incompatible = [name for name, value in expected_structure.items()
                if saved.get(name) != value]
if incompatible:
    raise SystemExit(f"source adapter structure is incompatible: {incompatible}")
for name in ("rwkv", "moonvit"):
    model = Path(saved[name]).resolve()
    stat = model.stat()
    fingerprint = hashlib.sha256(
        f"{model}|{stat.st_size}|{stat.st_mtime_ns}".encode()).hexdigest()
    if fingerprint != saved.get(f"{name}_fingerprint"):
        raise SystemExit(f"source {name} weights no longer match the checkpoint")
print({
    "kind": "finish_preflight",
    "checkpoint": str(checkpoint),
    "source_step": int(blob.get("step", -1)),
    "train_rows": int(summary["rows"]),
    "eval_rows": int(summary["eval_rows"]),
    "train_sha256": actual["train_sha256"],
    "eval_sha256": actual["eval_sha256"],
    "steps_per_epoch": (int(sampler["steps_per_epoch"])
                        if sampler_path else None),
    "cache_ready": bool(cache_receipt_path),
})
PY

if [[ "${VISION_PREFLIGHT_ONLY:-0}" == 1 ]]; then
  exit 0
fi

# Fill only missing pooled MoonViT entries. Cache keys include the source image
# metadata and vision checkpoint fingerprint, and every write is atomic.
if [[ "${SKIP_CACHE_VERIFY:-0}" != 1 ]]; then
  # One GPU process owns one MoonViT instance. Multiple local shards duplicate
  # the tower and contend through separate CUDA contexts; override only when the
  # caller deliberately assigns shards to separate GPUs.
  CACHE_SHARDS="${CACHE_SHARDS:-1}"
  cache_pids=()
  cleanup_cache() {
    if ((${#cache_pids[@]})); then
      kill "${cache_pids[@]}" 2>/dev/null || true
    fi
  }
  trap cleanup_cache EXIT INT TERM
  for ((shard=0; shard<CACHE_SHARDS; shard++)); do
    PYTHONPATH=src "$PYTHON_BIN" -m rwkv_lab.vision_cache \
      --data "$TRAIN" "$EVAL" \
      --cache "$CACHE" \
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

VISION_FINISH_TARGET_STEPS="${VISION_FINISH_TARGET_STEPS:-${VISION_TARGET_STEPS:-1000000000}}"

grounding_args=()
if [[ "${VISION_ENABLE_GROUNDING_LEVERS:-0}" == 1 ]]; then
  grounding_args=(
    --vision-resampler-layers "${VISION_RESAMPLER_LAYERS:-2}"
    --vision-resampler-width "${VISION_RESAMPLER_WIDTH:-1024}"
    --vision-resampler-heads "${VISION_RESAMPLER_HEADS:-8}"
    --deep-vision-layers "${VISION_DEEP_LAYERS:-8,16,24}"
    --deep-vision-rank "${VISION_DEEP_RANK:-256}"
    --grounding-early-tokens "${VISION_GROUNDING_EARLY_TOKENS:-24}"
    --grounding-early-weight "${VISION_GROUNDING_EARLY_WEIGHT:-3.0}"
    --grounding-contrastive-weight "${VISION_GROUNDING_CONTRASTIVE_WEIGHT:-0.1}"
    --grounding-contrastive-dim "${VISION_GROUNDING_CONTRASTIVE_DIM:-512}"
    --grounding-temperature "${VISION_GROUNDING_TEMPERATURE:-0.07}"
  )
fi

PYTHONPATH=src "$PYTHON_BIN" -m rwkv_lab.vision_train \
  --data "$TRAIN" \
  --eval-data "$EVAL" \
  --steps "$VISION_FINISH_TARGET_STEPS" \
  --batch 8 \
  --min-batch 4 \
  --max-batch 32 \
  --target-batch-tokens 3584 \
  --loop-token-budget-scale 1.0 \
  --max-text-tokens 768 \
  --out "$RUN" \
  --feature-cache "$CACHE" \
  --manifest-stat-workers 64 \
  --preload-feature-cache \
  --background-feature-preload \
  --checkpoint-every 50 \
  --eval-every 100 \
  --eval-samples "${VISION_EVAL_SAMPLE_COUNT:-12}" \
  --eval-sample-exclude-sources "${VISION_EVAL_SAMPLE_EXCLUDE_SOURCES:-joy,civitai,nsfw,porn,manga,pose_vr,grid}" \
  --eval-sample-max-new "${VISION_EVAL_SAMPLE_MAX_NEW:-768}" \
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
  --engram-warmup-steps 0 \
  --engram-boundary-id 0 \
  "${grounding_args[@]}" \
  "${resume_args[@]}"
