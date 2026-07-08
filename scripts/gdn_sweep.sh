#!/usr/bin/env bash
# Autonomous GDN->RWKV isolation sweep. Converts each remaining GDN layer with the
# validated recipe (spectral_muon + plus-norm row + DDC 0.5 + muon-lr 4e-6 + batch-8,
# GDN-init vs pristine base), early-stopping when the held-out eval ppl BOTTOMS OUT
# (plateau: PATIENCE evals with no >MIN_DELTA improvement) or TRENDS BACK UP (ppl rises
# >UPTURN above the best). convert_train writes best/ atomically on every eval
# improvement, so each layer's banked ckpt is always its true minimum regardless of
# when we stop. Advances layer-by-layer with no human input.
set -u
cd /thearray/git/moe-mla || exit 1
PY=.venv/bin/python
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}src"
DATA=/thearray/git/babyllm/data/cache/qwen3.6_fwedu_train
MODEL=Qwen3.5-9B-Base
LOG=runs/gdn_sweep.log
LAYERS="8 9 10 12 13 14 16 17 18 20 21 22 24 25 26 28 29 30"

PATIENCE=3           # flat evals (past MIN_STOP_STEP) with no >MIN_DELTA gain -> bottomed
MIN_DELTA=0.003      # ppl improvement that counts as real progress
UPTURN=0.03          # ppl rising this far above best -> trending back up
MIN_STOP_STEP=600    # don't early-stop before this (let it descend first)
HARD_STOP_STEP=3000  # backstop: stop regardless once here
STALL_SECS=1200      # no new eval for this long -> stuck, move on

log(){ printf '[%s] %s\n' "$(date '+%F %T')" "$*" >> "$LOG"; }

log "===== GDN SWEEP START (driver pid $$): layers $LAYERS ====="
for L in $LAYERS; do
  OUT=runs/iso_L${L}_b8_10k
  mkdir -p "$OUT"
  printf '# L%s GDN-init isolation convert, gdn_sweep driver. eval 120.\n' "$L" > "$OUT/cmd.txt"
  log "L${L}: launching"
  nohup $PY -m rwkv_lab.convert_train \
    --layer "${L}" --model-dir "$MODEL" --data "$DATA" --out "$OUT" \
    --optimizer spectral_muon --lr 1e-3 --muon-lr 4e-6 \
    --sm-plus-norm row --sm-ddc-strength 0.5 --sm-ddc-mode both --sm-equilibrate R \
    --batch-size 8 --fused-ce 1 \
    --w-block 20 --w-lmce 1 --w-smt 0 --w-dmt 0 --codec-pretrain 0 \
    --seq-len 1024 --steps 10000 \
    --eval-windows 64 --eval-every 120 --save-every 600 \
    --device cuda --dtype bfloat16 > "$OUT/train.log" 2>&1 &
  PID=$!
  best=""; best_step=""; patience=0; last_step=-1; last_ts=$(date +%s); reason=""
  sleep 25
  while kill -0 "$PID" 2>/dev/null; do
    sleep 30
    set -- $(grep '"kind": "eval"' "$OUT/train.jsonl" 2>/dev/null | tail -1 | \
             sed -n 's/.*"step": \([0-9]*\).*"ppl": \([0-9.]*\).*/\1 \2/p')
    cs="${1:-}"; cp="${2:-}"; now=$(date +%s)
    case "$cs" in ''|*[!0-9]*) cs="" ;; esac
    case "$cp" in ''|*[!0-9.]*) cp="" ;; esac
    if [ -z "$cs" ] || [ -z "$cp" ]; then
      [ $((now - last_ts)) -gt $STALL_SECS ] && { reason="stall-no-eval"; break; }
      continue
    fi
    if [ "$cs" = "$last_step" ]; then
      [ $((now - last_ts)) -gt $STALL_SECS ] && { reason="stall"; break; }
      continue
    fi
    last_step="$cs"; last_ts="$now"
    if [ -z "$best" ]; then
      best="$cp"; best_step="$cs"; patience=0
    else
      v=$(awk -v cp="$cp" -v b="$best" -v md="$MIN_DELTA" -v up="$UPTURN" 'BEGIN{
        if (cp < b - md) print "prog"; else if (cp > b + up) print "up"; else print "flat"}')
      if [ "$v" = "prog" ]; then
        best="$cp"; best_step="$cs"; patience=0
      elif [ "$cs" -ge "$MIN_STOP_STEP" ]; then
        if [ "$v" = "up" ]; then reason="upturn@${cs}(best=${best}@${best_step})"; break; fi
        patience=$((patience+1))
      fi
    fi
    log "L${L}: step=${cs} ppl=${cp} best=${best}@${best_step} patience=${patience}"
    [ "$patience" -ge "$PATIENCE" ] && { reason="plateau(best=${best}@${best_step})"; break; }
    [ "$cs" -ge "$HARD_STOP_STEP" ] && { reason="hardstop@${cs}(best=${best}@${best_step})"; break; }
  done
  kill -0 "$PID" 2>/dev/null || { [ -z "$reason" ] && reason="proc-exited"; }
  kill -TERM "$PID" 2>/dev/null
  for i in $(seq 1 900); do
    nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -q "^${PID}$" || break
    sleep 1
  done
  kill -0 "$PID" 2>/dev/null && { kill -9 "$PID" 2>/dev/null; sleep 5; }
  bj=$(cat "$OUT/best/best.json" 2>/dev/null)
  log "L${L}: DONE reason=${reason:-none} | ${bj:-<no best>}"
done
log "===== GDN SWEEP COMPLETE ====="
