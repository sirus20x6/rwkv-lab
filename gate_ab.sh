#!/usr/bin/env bash
# Controlled A/B of the LoopedRWKV gate parameterizations (--loop-gate). Same layer,
# same recipe, same seed, fresh GDN-init — the ONLY difference between arms is the
# gate mode. loop-lr-mult 30 so the zero-init gates actually escape (the gdn_sweep
# left it at the 1.0 default, so its loops barely trained). Compare eval ppl vs STEP
# on the dashboard, plus the loop card's max|rw| to see which parameterization opens
# its gates fastest.
#
# SEQUENTIAL: one full-recipe 9B run peaks ~55GB (looped 4-pass graph), so two do NOT
# co-fit in 94GB — run the arms one at a time, each with the whole GPU (no contention,
# and the comparison is per-STEP so serial vs parallel is equally fair). All four land
# on the dashboard as gate_ab_<mode>. Watch: http://127.0.0.1:9124
set -u
cd /thearray/git/moe-mla || exit 1
PY=.venv/bin/python
DATA=/thearray/git/babyllm/data/cache/qwen3.6_fwedu_train
MODEL=Qwen3.5-9B-Base
LAYER=16
STEPS=1500
LOG=runs/gate_ab.log

log(){ printf '[%s] %s\n' "$(date '+%F %T')" "$*" | tee -a "$LOG"; }

launch(){  # $1 = gate mode
  local mode="$1" out="runs/gate_ab_$1"
  mkdir -p "$out"
  printf '# gate A/B: L%s fresh GDN-init, loop-count 4, --loop-gate %s, loop-lr-mult 30, seed 0\n' \
    "$LAYER" "$mode" > "$out/cmd.txt"
  log "launch $mode -> $out"
  nohup $PY convert_train.py \
    --layer "$LAYER" --model-dir "$MODEL" --data "$DATA" --out "$out" \
    --optimizer spectral_muon --lr 1e-3 --muon-lr 4e-6 \
    --sm-plus-norm row --sm-ddc-strength 0.5 --sm-ddc-mode both --sm-equilibrate R \
    --batch-size 8 --fused-ce 1 \
    --w-block 20 --w-lmce 1 --w-smt 0 --w-dmt 0 --codec-pretrain 0 \
    --seq-len 1024 --steps "$STEPS" \
    --loop-count 4 --loop-gate "$mode" --loop-lr-mult 30 \
    --eval-windows 32 --eval-every 100 --log-every 20 --save-every 0 \
    --seed 0 --device cuda --dtype bfloat16 > "$out/train.log" 2>&1 &
  echo $!
}

log "===== GATE A/B START (driver $$): L$LAYER, $STEPS steps, 4 arms SEQUENTIAL ====="
# scalar (baseline) and factored (our design) first — the key contrast — then the
# head/channel granularity middle. One at a time so each gets the full GPU.
for mode in scalar factored head channel; do
  pid=$(launch "$mode")
  log "$mode running (pid $pid); waiting..."
  wait "$pid"; log "$mode exited rc=$?"
done

log "===== GATE A/B COMPLETE — comparing best/ ppl ====="
for m in scalar head channel factored; do
  bj=$(cat "runs/gate_ab_$m/best/best.json" 2>/dev/null || echo '<none>')
  log "  $m: $bj"
done
