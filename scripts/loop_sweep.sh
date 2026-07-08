#!/usr/bin/env bash
# Fixed-wall-clock (~10-min) A/B of recurrent-depth / loop levers on a from-scratch small
# RWKV-7 (our modules native). Same seed/data/model — only the loop config varies — so final
# val loss answers: does looping the same weights for more effective depth beat a single pass
# within a fixed compute budget, and which loop levers add value?
set -u
cd /thearray/git/moe-mla
export PYTHONPATH=src PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
D=${DATA:-models/g1g_tokens_big.bin}; MIN=${MIN:-10}
COMMON="--data $D --minutes $MIN --d-model 512 --n-layers 6 --batch 16 --seq-len 512 --eval-every 40 --seed 0"
run(){ name=$1; shift; echo "=== $(date +%H:%M) loop_$name ==="; python -m rwkv_lab.rwkv_pretrain $COMMON --out runs/loop_$name "$@"; }
run baseline                                              # loop=1, single pass
run c2            --loop-count 2
run c3            --loop-count 3
run c4            --loop-count 4
run c3_hyper      --loop-count 3 --loop-hyper 2           # + hyper-connection lanes
run c3_cart       --loop-count 3 --loop-cart-anchor 1     # + contractive LTI anchor
run c3_deq        --loop-count 3 --loop-deq 1             # + DEQ 1-step gradient (O(1) mem)
run c3_fphalt     --loop-count 3 --loop-fp-halt 1         # + fixed-point-residual halting
run c3_ponder     --loop-count 3 --loop-adaptive-halt 1   # + PonderNet adaptive depth
echo "=== loop sweep done ==="
