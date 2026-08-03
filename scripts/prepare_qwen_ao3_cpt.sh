#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${AO3_CPT_VENV:-$ROOT/.venv-ao3-cpt}"
MODEL="${AO3_CPT_MODEL:-/thearray/git/ob/text-generation-webui/models/Qwen3.6-35B-A3B-heretic}"
PREPARED="${AO3_CPT_PREPARED:-/thearray/downloads/completed/ao3/ao3_filthiest_top5pct_cpt}"
TOKENS="${AO3_CPT_TOKENS:-/thearray/downloads/completed/ao3/ao3_filthiest_top5pct_cpt_tokens}"
PACKS="${AO3_CPT_PACKS:-/thearray/downloads/completed/ao3/ao3_filthiest_top5pct_cpt_packs}"
CONFIG="${AO3_CPT_CONFIG:-$ROOT/experiments/qwen36_ao3_cpt.json}"

if [[ ! -f "$PREPARED/manifest.json" ]]; then
  echo "Prepared corpus is incomplete: $PREPARED/manifest.json" >&2
  exit 1
fi
if [[ ! -f "$TOKENS/manifest.json" ]]; then
  "$VENV/bin/python" "$ROOT/scripts/prepare_ao3_cpt_tokens.py" tokenize \
    --prepared "$PREPARED" \
    --tokenizer "$MODEL" \
    --output "$TOKENS"
fi
if [[ ! -f "$PACKS/train_ctx8192/manifest.json" ]]; then
  "$VENV/bin/python" "$ROOT/scripts/prepare_ao3_cpt_tokens.py" pack \
    --token-cache "$TOKENS" \
    --split train \
    --context-length 8192 \
    --output "$PACKS/train_ctx8192"
fi
if [[ ! -f "$PACKS/eval_ctx8192/manifest.json" ]]; then
  "$VENV/bin/python" "$ROOT/scripts/prepare_ao3_cpt_tokens.py" pack \
    --token-cache "$TOKENS" \
    --split eval \
    --context-length 8192 \
    --no-shuffle-documents \
    --output "$PACKS/eval_ctx8192"
fi
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
"$VENV/bin/python" -m rwkv_lab.qwen_ao3_cpt prepare-run --config "$CONFIG"
