#!/usr/bin/env bash
set -Eeuo pipefail

repo=/workspace/git/moe-mla
python_bin="$repo/.venv-mage-flow/bin/python"
module=rwkv_lab.mage_flow_terminal_train
cache_root=/mnt/hypercard/mageflow-cache/mageflow-steps-00005501-00012228
build_config="$cache_root/cache_build_config.json"
resume_config="$cache_root/resume_config.json"
entries=/mnt/hypercard/mageflow-cache/mageflow-steps-00000501-00005500/entries
status_file="$entries/cache_build_status.json"
log_file="$cache_root/recovery_handoff.log"
lock_file="$cache_root/handoff.lock"

mkdir -p "$cache_root"
exec >>"$log_file" 2>&1
exec 9>"$lock_file"
if ! flock -n 9; then
    echo "[$(date --iso-8601=seconds)] another cache handoff owns the lock"
    exit 1
fi
cd "$repo"

echo "[$(date --iso-8601=seconds)] completing full encoder cache"
PYTHONPATH=src "$python_bin" -m "$module" cache-encoders \
    --config "$build_config"

if [[ "$(jq -r '.state // "missing"' "$status_file")" != "complete" ]]; then
    echo "encoder cache did not validate complete"
    exit 1
fi
if [[ "$(jq -r '.coverage.complete // false' "$status_file")" != "true" ]]; then
    echo "encoder cache coverage validation failed"
    exit 1
fi

echo "[$(date --iso-8601=seconds)] cache validated; resuming from checkpoint 4500"
exec env \
    PYTHONPATH=src \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "$python_bin" -m "$module" train --config "$resume_config"
