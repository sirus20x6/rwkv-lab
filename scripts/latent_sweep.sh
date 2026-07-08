#!/usr/bin/env bash
# Fixed-wall-clock (~10-min) A/B of latent-prediction / lookahead objectives on a from-scratch
# small RWKV-7 (our modules native). Same seed/data/model as loop_sweep.sh — only the aux
# objective varies — so final val loss (the plain LM head, unchanged) isolates whether the
# auxiliary prediction signal improves the primary next-token model.
#
# Requires the latent-prediction flags wired into rwkv_pretrain.py (LookaheadHeads on the
# post-norm final hidden). Objectives: NextLat (predict future latent), TOP (token-order
# prediction), L-MTP (leap multi-token), Belief-State (fwd+bwd), JTP (joint multi-token).
set -u
cd /thearray/git/moe-mla
export PYTHONPATH=src PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
D=${DATA:-models/g1g_tokens_big.bin}; MIN=${MIN:-10}
COMMON="--data $D --minutes $MIN --d-model 512 --n-layers 6 --batch 16 --seq-len 512 --eval-every 40 --seed 0"
run(){ name=$1; shift; echo "=== $(date +%H:%M) lat_$name ==="; python -m rwkv_lab.rwkv_pretrain $COMMON --out runs/lat_$name "$@"; }
run baseline                                    # no aux objective (same as loop baseline)
run nextlat    --nextlat-weight 0.5             # predict next latent (d=1 rollout) + KL
run top        --top-weight 0.5                 # token-order prediction over a future window
run lmtp       --lmtp-weight 0.5                # leap multi-token prediction head
run bst        --bst-weight 0.5                 # belief-state (forward+backward) objective
run jtp        --jtp-weight 0.5                 # joint multi-token prediction head
echo "=== latent sweep done ==="
