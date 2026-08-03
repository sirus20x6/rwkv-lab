#!/usr/bin/env bash
# Resume runs/radio1d_rwkv_captioning_first from best/ckpt.pt (step 43000) for ~45,000 more steps.
#
# WHY THE LOOP: this run sets --restart-before-eval, so the trainer calls
# os._exit({42}) before every eval and expects a supervisor to relaunch it (FLA's
# recurrent inference path can leave CUDA state invalid after many training
# forwards). With eval_every=500 that is ~90 restarts over this budget.
#
# WHY RESUME CHANGES: only the FIRST launch may point at best/ckpt.pt. Every
# restart must use --resume auto, which picks up runs/radio1d_rwkv_captioning_first/last.pt.
# Re-pointing at best/ would rewind to step 43000 on every eval and the run
# would never advance.
#
# --steps is ABSOLUTE (`while step < args.steps`): 43000+45000=88000.
# lr is constant and engram_warmup/loop_ramp/loop_start all completed before
# step 43000, so extending the horizon distorts nothing.
#
# Fixes carried:
#   * --radio-adaptive-token-threshold 49 (was 12) — restores token scaling with
#     image size. Passed EXPLICITLY: config.json still records 12 and inheriting
#     it would silently re-enable the cliff this run exists to remove.
#   * letterbox content geometry in the RADIO bridge — zero-init, so the resume
#     is a bitwise no-op and the term is learned from here.
#
# --allow-batch-resize is REQUIRED: the threshold change is a budget difference.
# It quarantines best/ and resets eval claims — correct, since ppl 4.169 was
# measured on 128-tok/tile features and is not comparable to what follows.
set -uo pipefail
cd /thearray/git/moe-mla

# Match scripts/run_radio1d_captioning_first.sh: rwkv_lab is not installed, it
# is imported from ./src, and the run is guarded by a lock so two launchers can
# never own the same output directory.
PYTHON_BIN="${VISION_PYTHON:-/usr/bin/python}"
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}src"

RUN=runs/radio1d_rwkv_captioning_first
for required in curated_vision/captioning_first_train.jsonl \
                curated_vision/captioning_first_eval.jsonl \
                "$RUN/best/ckpt.pt" \
                models/vision/C-RADIOv4-1D-H \
                models/rwkv7-g1h-2.9b-20260710-ctx10240.pth; do
  [[ -e "$required" ]] || { echo "missing required input: $required" >&2; exit 1; }
done

mkdir -p "$RUN"
exec 9>"$RUN/.launcher.lock"
flock -n 9 || { echo "another launcher already owns $RUN" >&2; exit 75; }

RESUME="runs/radio1d_rwkv_captioning_first/best/ckpt.pt"
FAST_FAILURES=0
while true; do
  START=$SECONDS
  "$PYTHON_BIN" -m rwkv_lab.vision_train \
      --activation-checkpoint-min-tokens 4096 \
      --batch 1 \
      --checkpoint-every 50 \
      --data curated_vision/captioning_first_train.jsonl \
      --deep-vision-layers 8,16,24 \
      --deep-vision-rank 256 \
      --dinov2-model models/vision/dinov2-base \
      --engram --engram-boundary-id \
      0 --engram-drow \
      128 --engram-lr \
      0.001 --engram-rows \
      65536 --engram-sites \
      3,15 --engram-warmup-steps \
      0 --eval-batch-size \
      4 --eval-data \
      curated_vision/captioning_first_eval.jsonl --eval-every \
      500 --eval-examples \
      448 --eval-sample-every \
      1000 --eval-sample-exclude-sources \
      joy,civitai,nsfw,porn,manga,pose_vr,grid --eval-sample-max-new \
      768 --eval-samples \
      14 --feature-cache \
      /thearray/downloads/cache/moe-mla/radio1d_captioning_first --feature-cache-max-bytes \
      17179869184 --fusion-feature-cache \
      caches/siglip2_dinov2_sam_aligned_v1 --grad-clip \
      1.0 --grounding-contrastive-dim \
      512 --grounding-contrastive-weight \
      0.1 --grounding-early-tokens \
      24 --grounding-early-weight \
      3.0 --grounding-temperature \
      0.07 --layer-vision-layers \
      '' --layer-vision-rank \
      256 --log-every \
      1 --loop-count \
      2 --loop-gate-cap \
      0.25 --loop-index \
      --loop-lr 1e-05 \
      --loop-ramp-steps 0 \
      --loop-start-step 1 \
      --loop-token-budget-scale 1.0 \
      --lr 0.0002 \
      --manifest-stat-workers 64 \
      --max-batch 4 \
      --max-input-patches 1024 \
      --max-text-tokens 768 \
      --min-batch 1 \
      --moonvit models/kimi-k2.6-moonvit/model-00064-of-000064.safetensors \
      --moonvit-tap-layers '' \
      --nextlat-hidden 1024 \
      --nextlat-kl-weight 0.0 \
      --nextlat-weight 0.1 \
      --operator-profile-steps 0 \
      --out runs/radio1d_rwkv_captioning_first \
      --prefetch-next-batch --prefix-tokens \
      256 --profile-steps \
      3 --prompt \
      'Describe this image:
' --radio-adaptive-complexity \
      --radio-complexity-budget-ratio 0.75 \
      --radio-complexity-token-quantum 16 \
      --radio-max-detail-tiles 48 \
      --radio-model models/vision/C-RADIOv4-1D-H \
      --radio-revision e18692120c7a3203496e1a99056a4149ede135fc \
      --radio-tile-batch 8 \
      --require-fused-ce --restart-before-eval \
      --rwkv models/rwkv7-g1h-2.9b-20260710-ctx10240.pth \
      --no-sam-fusion --sam-fusion-tokens \
      128 --sam-model \
      models/vision/sam-vit-base --sandwich-lead-prompt \
      'An image follows:
' --sandwich-prompt \
      --seed 20260714 \
      --siglip2-model models/vision/siglip2-so400m-patch16-512 \
      --siglip2-width 1152 \
      --structured-coordinate-weight 4.0 \
      --structured-head --structured-heads \
      8 --structured-invalid-box-margin \
      1.0 --structured-invalid-box-weight \
      0.5 --structured-object-layers \
      2 --structured-object-queries \
      16 --structured-spatial-layers \
      2 --structured-weight \
      1.0 --structured-width \
      256 --target-batch-tokens \
      4096 --val-fraction \
      0.02 --vision-backend \
      radio1d --vision-compressor-checkpoint \
      '' --no-vision-fusion \
      --vision-fusion-rank 512 \
      --vision-resampler-heads 8 \
      --vision-resampler-layers 0 \
      --vision-resampler-width 1024 \
      --vision-view-mode full \
      --weight-decay 0.01 \
      --radio-adaptive-token-threshold 49 \
      --steps 88000 \
      --allow-batch-resize \
      --resume "$RESUME"
  CODE=$?
  ELAPSED=$(( SECONDS - START ))

  if [ "$CODE" -eq 0 ]; then
    echo "[supervisor] training reached --steps 88000; done."
    break
  fi
  if [ "$CODE" -ne 42 ]; then
    echo "[supervisor] exit $CODE after ${ELAPSED}s — real failure, not an eval restart." >&2
    exit "$CODE"
  fi

  # Exit 42 is the planned pre-eval restart. Guard against a crash loop that
  # merely looks like one.
  if [ "$ELAPSED" -lt 60 ]; then
    FAST_FAILURES=$(( FAST_FAILURES + 1 ))
    if [ "$FAST_FAILURES" -ge 3 ]; then
      echo "[supervisor] three restarts in under 60s each — stopping." >&2
      exit 1
    fi
  else
    FAST_FAILURES=0
  fi
  echo "[supervisor] planned eval restart after ${ELAPSED}s; resuming from last.pt"
  RESUME=auto
done
