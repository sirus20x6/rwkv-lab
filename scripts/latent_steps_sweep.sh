#!/usr/bin/env bash
# FIXED-STEPS latent-prediction A/B — isolates the representation effect from throughput.
# The fixed-wall-clock sweep (latent_sweep.sh) penalizes expensive aux heads (they do fewer
# optimizer steps in 10 min). Here every config runs the SAME number of steps, so the aux head
# just costs more wall-clock; the LM-head val at equal steps answers "does the auxiliary
# prediction signal improve the representation?" independent of its compute cost.
set -u
cd /thearray/git/moe-mla
export PYTHONPATH=src PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
D=${DATA:-models/g1g_tokens_big.bin}; STEPS=${STEPS:-4000}
COMMON="--data $D --steps $STEPS --d-model 512 --n-layers 6 --batch 16 --seq-len 512 --eval-every 200 --seed 0"
run(){ name=$1; shift; echo "=== $(date +%H:%M) ls_$name (${STEPS} steps) ==="; python -m rwkv_lab.rwkv_pretrain $COMMON --out runs/ls_$name "$@"; }
run baseline
run nextlat  --nextlat-weight 0.1
run top      --top-weight 0.1
run lmtp     --lmtp-weight 0.1
run bst      --bst-weight 0.1
run jtp      --jtp-weight 0.1
echo "=== fixed-steps latent sweep done ==="
