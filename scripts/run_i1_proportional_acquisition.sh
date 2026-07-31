#!/usr/bin/env bash
set -euo pipefail

repo=/workspace/git/moe-mla
output=/workspace/git/datasets/i1
logs="$output/logs"
rate="${I1_DOWNLOAD_MIB_PER_SECOND:-8}"

mkdir -p "$logs"
cd "$repo"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYARROW_NUM_THREADS=1
export PYTHONPATH=.:src

run_logged() {
    local name="$1"
    shift
    "$@" >> "$logs/$name.log" 2>&1
    python -m scripts.materialize_i1_proportional_tranche --merge \
        >> "$logs/$name.log" 2>&1
}

run_logged redcaps \
    python -m scripts.fetch_i1_proportional_web_subsets \
    --subsets redcaps --mib-per-second "$rate"

run_logged gptedit \
    python -m scripts.fetch_i1_proportional_hf_subsets \
    --subsets gptedit --direct-url-mib-per-second "$rate"

run_logged textatlas \
    python -m scripts.fetch_i1_proportional_hf_subsets \
    --subsets textatlas --direct-url-mib-per-second "$rate"

run_logged megalith10m \
    python -m scripts.fetch_i1_proportional_hf_subsets \
    --subsets megalith10m --direct-url-mib-per-second "$rate"

run_logged rendered_text \
    python -m scripts.fetch_i1_rendered_text \
    --mib-per-second "$rate"

run_logged yfcc \
    python -m scripts.fetch_i1_proportional_web_subsets \
    --subsets yfcc --mib-per-second "$rate"

python -m scripts.materialize_i1_proportional_tranche --merge \
    > "$logs/final_validation.log" 2>&1
