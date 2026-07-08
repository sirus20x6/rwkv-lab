#!/usr/bin/env bash
# Factored-gate loop ladder — the scalar gate (default) may not engage from-scratch because it's
# a single zero-init scalar per pass; the factored gate is a richer head×channel gate that can
# turn loops on more selectively. Same from-scratch model/data/seed as loop_sweep.sh, so this is
# a direct scalar-vs-factored comparison at each depth. Scalar c2/c3/c4 come from loop_sweep.sh.
set -u
cd /thearray/git/moe-mla
export PYTHONPATH=src PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
D=${DATA:-models/g1g_tokens_big.bin}; MIN=${MIN:-10}
COMMON="--data $D --minutes $MIN --d-model 512 --n-layers 6 --batch 16 --seq-len 512 --eval-every 40 --seed 0"
run(){ name=$1; shift; echo "=== $(date +%H:%M) fg_$name ==="; python -m rwkv_lab.rwkv_pretrain $COMMON --out runs/fg_$name "$@"; }
run c2_factored  --loop-count 2 --loop-gate factored
run c3_factored  --loop-count 3 --loop-gate factored
run c4_factored  --loop-count 4 --loop-gate factored
echo "=== factored loop sweep done ==="
