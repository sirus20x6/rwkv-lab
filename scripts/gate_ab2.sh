#!/usr/bin/env bash
# Gate A/B round 2: scalar vs factored under the TRAINER-OWNED loop-LR anneal
# (--loop-anneal-rw), plus a --loop-count 1 control for attribution.
#
# Why round 2: round 1's arms got materially different effective-LR schedules from
# the dashboard detector's reactive cooling (factored cooled at max|rw| 0.303,
# scalar at 2.1 — ingest/sampler lag), and its full-precision result was a tie
# (9.1717 vs 9.1730 on 256 paired windows). Round 2 removes the confound: both
# looped arms run the identical deterministic policy — full 30x boost until
# max|rw| crosses 0.1, then cosine to 1x over 400 steps — applied in-process with
# zero lag. The anneal-aware detector (loop_anneal=1) writes no loop controls.
# The control arm answers how much of the improvement the loop earns at all
# (round 1 had no control: core training gains were conflated with loop gains).
#
# Foreground-sequential (one ~56GB run at a time); runs land on the dashboard as
# gate_ab2_<arm>. Final ranking = offline paired re-eval of best/ ckpts (fp32 CE,
# 256 windows), NOT the in-training numbers. Watch: http://127.0.0.1:9124
set -u
cd /thearray/git/moe-mla || exit 1
PY=.venv/bin/python
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}src"
DATA=/thearray/git/babyllm/data/cache/qwen3.6_fwedu_train
MODEL=Qwen3.5-9B-Base
LAYER=16
STEPS=2000
LOG=runs/gate_ab2.log

log(){ printf '[%s] %s\n' "$(date '+%F %T')" "$*" | tee -a "$LOG"; }

run_arm(){  # $1 = arm name, $@ = extra convert_train flags; FOREGROUND (blocks)
  local arm="$1"; shift
  local out="runs/gate_ab2_$arm"
  mkdir -p "$out"
  printf '# gate A/B round 2: L%s fresh GDN-init, seed 0, %s steps, arm %s: %s\n' \
    "$LAYER" "$STEPS" "$arm" "$*" > "$out/cmd.txt"
  log "START $arm -> $out ($*)"
  $PY -m rwkv_lab.convert_train \
    --layer "$LAYER" --model-dir "$MODEL" --data "$DATA" --out "$out" \
    --optimizer spectral_muon --lr 1e-3 --muon-lr 4e-6 \
    --sm-plus-norm row --sm-ddc-strength 0.5 --sm-ddc-mode both --sm-equilibrate R \
    --batch-size 8 --fused-ce 1 \
    --w-block 20 --w-lmce 1 --w-smt 0 --w-dmt 0 --codec-pretrain 0 \
    --seq-len 1024 --steps "$STEPS" \
    --eval-windows 32 --eval-every 100 --log-every 20 --save-every 0 \
    --seed 0 --device cuda --dtype bfloat16 "$@" > "$out/train.log" 2>&1
  log "END $arm rc=$?"
}

log "===== GATE A/B ROUND 2 START (driver $$): L$LAYER, $STEPS steps ====="
run_arm control  --loop-count 1
run_arm scalar   --loop-count 4 --loop-gate scalar   --loop-lr-mult 30 --loop-anneal-rw 0.1 --loop-anneal-steps 400
run_arm factored --loop-count 4 --loop-gate factored --loop-lr-mult 30 --loop-anneal-rw 0.1 --loop-anneal-steps 400

log "===== GATE A/B ROUND 2 COMPLETE — best/ ppl per arm (in-training metric) ====="
for m in control scalar factored; do
  bj=$(cat "runs/gate_ab2_$m/best/best.json" 2>/dev/null || echo '<none>')
  log "  $m: $bj"
done
log "final ranking: offline paired re-eval (reeval_best.py, fp32 CE, 256 windows)"
