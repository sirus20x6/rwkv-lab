#!/usr/bin/env bash
# World-tokenize a text corpus into a flat uint16 .bin for rwkv_finetune, using ztok
# (github: sirus20x6/ztok) — a fast multithreaded Zig tokenizer, bit-identical to the
# BlinkDL World trie (verified via ztok's bench/rwkv_parity.py). ~300 MB/s vs the slow
# pure-Python trie. Output is directly np.memmap-able (uint16) by rwkv_finetune.
#   scripts/tokenize_ztok.sh <corpus.txt> <out.bin>
set -euo pipefail
ZTOK=${ZTOK:-/thearray/git/ztok/zig-out/bin/ztok}
VOCAB=${VOCAB:-/thearray/git/ztok/bench/vocabs/rwkv_vocab_v20230424.txt}
IN=${1:?usage: tokenize_ztok.sh <corpus.txt> <out.bin>}; OUT=${2:?missing out.bin}
"$ZTOK" tokenize-dataset --model "$VOCAB" --input "$IN" --output "$OUT" \
  --format bin --dtype u16 --doc-mode whole
