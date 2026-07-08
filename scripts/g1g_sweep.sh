#!/usr/bin/env bash
# Karpathy-style ~10-min partitioned A/B of RWKV-Lab optimizer levers on BlinkDL rwkv7-g1g-1.5b.
# Baseline -> plain Muon -> Muon + each spectral lever. Each run writes runs/g1g_<name>/train.jsonl
# (live in trainboard). Same seed/data/window so final val loss is comparable across levers.
set -u
cd /thearray/git/moe-mla
export PYTHONPATH=src
M=models/rwkv7-g1g-1.5b.pth; D=models/g1g_tokens.bin
MIN=${MIN:-10}; COMMON="--model $M --data $D --minutes $MIN --batch 8 --seq-len 1024 --eval-every 25 --seed 0"
run(){ name=$1; shift; echo "=== $(date +%H:%M) g1g_$name ==="; python -m rwkv_lab.rwkv_finetune $COMMON --out runs/g1g_$name "$@"; }
run baseline_adamw   --optimizer adamw
run muon_plain       --optimizer spectral_muon
run muon_rsav        --optimizer spectral_muon --sm-rsav 1
run muon_mona        --optimizer spectral_muon --sm-mona 1
run muon_tile        --optimizer spectral_muon --sm-tile-size 256
run muon_damuon      --optimizer spectral_muon --sm-da-muon 1
run muon_aro         --optimizer spectral_muon --sm-aro 1
run muon_ddc         --optimizer spectral_muon --sm-ddc-strength 0.1
run muon_specpow     --optimizer spectral_muon --sm-spectral-power 0.33
echo "=== sweep done ==="
