#!/usr/bin/env bash
# Full rel re-sweep (OpenMOSE recipe step 1): re-convert every GDN layer with the
# normalized-MSE block loss (--block-loss rel = per-token relative L2 norm), warm-started
# from the raw-MSE best but with FRESH optimizer momentum (--no-warm-optimizer, since the
# loss objective changed). Judged by BLOCK ERROR, not ppl: under rel the per-layer ppl does
# NOT track the block match (residuals are expected and get fixed by the downstream logit-KL
# consolidation). IMPORTANT: convert_train's best/ is ppl-based and therefore useless here
# (ppl barely moves while block halves) -- so we bank the FINAL step_ checkpoint (block
# plateaus, so the last saved step ~= block-min) and assemble from those, NOT from best/.
#
# Recipe per layer: spectral_muon (NS-5 approx) + plus-norm row + equilibrate R + DDC 0.5
# (all defaults now), muon-lr 4e-6, io-first weights (w-block 20 / w-lmce 1 / smt 0 / dmt 0,
# no codec), scalar gate loop-4 (loop capacity is added later at consolidation via --loop-hyper).
set -u
cd /thearray/git/moe-mla || exit 1
PY=.venv/bin/python
DATA=/thearray/git/babyllm/data/cache/qwen3.6_fwedu_train
MODEL=Qwen3.5-9B-Base
LOG=runs/rel_sweep.log
LAYERS="0 1 2 4 5 6 8 9 10 12 13 14 16 17 18 20 21 22 24 25 26 28 29 30"

PATIENCE=6           # new-step block samples (past MIN_STOP_STEP) with no >MIN_DELTA drop -> plateaued
MIN_DELTA=0.003      # block-rel improvement (vs running-min) that counts as real progress
MIN_STOP_STEP=800    # don't early-stop before this (let block descend to its plateau first)
HARD_STOP_STEP=2500  # backstop cap
STALL_SECS=900       # no new train step for this long -> stuck, move on

log(){ printf '[%s] %s\n' "$(date '+%F %T')" "$*" >> "$LOG"; }
bestdir_for(){ if [ "$1" = 1 ]; then echo "runs/iso_L1_b8_10k_r2"; else echo "runs/iso_L${1}_b8_10k"; fi; }

log "===== REL SWEEP START (driver pid $$): layers $LAYERS ====="
for L in $LAYERS; do
  OUT=runs/iso_L${L}_rel
  SRC=$(bestdir_for "$L")
  mkdir -p "$OUT"
  printf '# L%s rel re-sweep: --block-loss rel, warm %s + fresh momentum, block-error stop.\n' "$L" "$SRC" > "$OUT/cmd.txt"
  log "L${L}: launching rel (warm $SRC, fresh momentum)"
  nohup $PY convert_train.py \
    --layer "$L" --model-dir "$MODEL" --data "$DATA" --out "$OUT" \
    --optimizer spectral_muon --lr 1e-3 --muon-lr 4e-6 \
    --loop-count 4 --loop-gate scalar \
    --init-rwkv-ckpt "$SRC" --no-warm-optimizer \
    --block-loss rel \
    --w-block 20 --w-lmce 1 --w-smt 0 --w-dmt 0 --codec-pretrain 0 \
    --batch-size 8 --fused-ce 1 --seq-len 1024 --steps 10000 \
    --eval-windows 64 --eval-every 200 --save-every 400 \
    --device cuda --dtype bfloat16 > "$OUT/train.log" 2>&1 &
  PID=$!
  best=""; best_step=""; patience=0; last_step=-1; last_ts=$(date +%s); reason=""
  sleep 25
  while kill -0 "$PID" 2>/dev/null; do
    sleep 30
    set -- $(grep '"kind": "train"' "$OUT/train.jsonl" 2>/dev/null | tail -1 | \
             sed -n 's/.*"step": \([0-9]*\).*"block": \([0-9.]*\).*/\1 \2/p')
    cs="${1:-}"; cb="${2:-}"; now=$(date +%s)
    case "$cs" in ''|*[!0-9]*) cs="" ;; esac
    case "$cb" in ''|*[!0-9.]*) cb="" ;; esac
    if [ -z "$cs" ] || [ -z "$cb" ]; then
      [ $((now - last_ts)) -gt $STALL_SECS ] && { reason="stall-no-block"; break; }
      continue
    fi
    if [ "$cs" = "$last_step" ]; then
      [ $((now - last_ts)) -gt $STALL_SECS ] && { reason="stall"; break; }
      continue
    fi
    last_step="$cs"; last_ts="$now"
    if [ -z "$best" ]; then
      best="$cb"; best_step="$cs"; patience=0
    else
      v=$(awk -v cb="$cb" -v b="$best" -v md="$MIN_DELTA" 'BEGIN{ if (cb < b - md) print "prog"; else print "flat"}')
      if [ "$v" = "prog" ]; then best="$cb"; best_step="$cs"; patience=0
      elif [ "$cs" -ge "$MIN_STOP_STEP" ]; then patience=$((patience+1)); fi
    fi
    log "L${L}: step=${cs} block=${cb} best=${best}@${best_step} patience=${patience}"
    [ "$patience" -ge "$PATIENCE" ] && { reason="block-plateau(best=${best}@${best_step})"; break; }
    [ "$cs" -ge "$HARD_STOP_STEP" ] && { reason="hardstop@${cs}(best=${best}@${best_step})"; break; }
  done
  kill -0 "$PID" 2>/dev/null || { [ -z "$reason" ] && reason="proc-exited"; }
  kill -TERM "$PID" 2>/dev/null
  for i in $(seq 1 900); do nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -q "^${PID}$" || break; sleep 1; done
  kill -0 "$PID" 2>/dev/null && { kill -9 "$PID" 2>/dev/null; sleep 5; }
  fin=$(ls -d "$OUT"/step_* 2>/dev/null | sort -t_ -k2 -n | tail -1)
  log "L${L}: DONE reason=${reason:-none} | assemble_ckpt=${fin:-<none>/ckpt.pt} | block_min=${best}@${best_step}"
done
log "===== REL SWEEP COMPLETE ====="
