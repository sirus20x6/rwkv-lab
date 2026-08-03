#!/usr/bin/env bash
set -Eeuo pipefail

repo=/thearray/git/moe-mla
python_bin="$repo/.venv-mage-flow/bin/python"
module=rwkv_lab.mage_flow_terminal_train
run_dir="$repo/runs/mage_flow_terminal_tread_loop_repa_fixed_v2"
status_file="$run_dir/status.json"
source_config="$repo/experiments/mageflow_terminal_repa_fixed_v2.json"
shared_entries=/mnt/hypercard/mageflow-cache/mageflow-steps-00000501-00005500/entries
full_manifest=/thearray/git/datasets/midjourney-v6-recap-continuation-30pct-b-512-1024/train.jsonl
next_cache=/mnt/hypercard/mageflow-cache/mageflow-steps-00005501-00012228
target_step=5500
final_step=12228
log_file="$next_cache/handoff.log"
lock_file="$next_cache/handoff.lock"

mkdir -p "$next_cache"
exec >>"$log_file" 2>&1
exec 9>"$lock_file"
if ! flock -n 9; then
    echo "[$(date --iso-8601=seconds)] another cache handoff owns the lock"
    exit 0
fi
cd "$repo"

echo "[$(date --iso-8601=seconds)] handoff armed for step $target_step"
while true; do
    if [[ ! -f "$status_file" ]]; then
        echo "[$(date --iso-8601=seconds)] waiting for trainer status"
        sleep 30
        continue
    fi
    state=$(jq -r '.state // "unknown"' "$status_file")
    step=$(jq -r '.step // 0' "$status_file")
    if [[ "$state" == "cache_span_complete" && "$step" -ge "$target_step" ]]; then
        checkpoint=$(jq -r '.checkpoint' "$status_file")
        break
    fi
    if [[ "$state" == "interrupted" || "$state" == "failed" || "$state" == "complete" ]]; then
        echo "trainer reached terminal state=$state before cache handoff at step=$step"
        exit 1
    fi
    if ! pgrep -f -- \
        "rwkv_lab.mage_flow_terminal_train train --config $source_config" \
        >/dev/null; then
        echo "trainer process disappeared while status remained state=$state step=$step"
        exit 1
    fi
    echo "[$(date --iso-8601=seconds)] training state=$state step=$step"
    sleep 30
done

if [[ ! -d "$checkpoint" ]]; then
    echo "checkpoint does not exist: $checkpoint"
    exit 1
fi
remaining_steps=$((final_step - step))
if [[ "$remaining_steps" -le 0 ]]; then
    echo "training is already at final step $step"
    exit 0
fi

echo "[$(date --iso-8601=seconds)] planning $remaining_steps remaining steps"
PYTHONPATH=src "$python_bin" -m "$module" prepare-cache-span \
    --config "$source_config" \
    --checkpoint "$checkpoint" \
    --optimizer-steps "$remaining_steps" \
    --output-dir "$next_cache"

build_config="$next_cache/cache_build_config.json"
resume_config="$next_cache/resume_config.json"
"$python_bin" - \
    "$build_config" "$resume_config" "$shared_entries" "$full_manifest" <<'PY'
import json
import os
import sys
from pathlib import Path

for name in sys.argv[1:3]:
    path = Path(name)
    values = json.loads(path.read_text(encoding="utf-8"))
    values["encoder_cache_dir"] = sys.argv[3]
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(values, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)

build_path = Path(sys.argv[1])
build = json.loads(build_path.read_text(encoding="utf-8"))
build["train_manifest"] = sys.argv[4]
temporary = build_path.with_name(build_path.name + ".tmp")
temporary.write_text(
    json.dumps(build, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
os.replace(temporary, build_path)

resume_path = Path(sys.argv[2])
resume = json.loads(resume_path.read_text(encoding="utf-8"))
resume["encoder_cache_coverage_manifest"] = sys.argv[4]
temporary = resume_path.with_name(resume_path.name + ".tmp")
temporary.write_text(
    json.dumps(resume, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
os.replace(temporary, resume_path)
PY

echo "[$(date --iso-8601=seconds)] building remaining encoder cache"
PYTHONPATH=src "$python_bin" -m "$module" cache-encoders \
    --config "$build_config"

cache_status="$shared_entries/cache_build_status.json"
if [[ "$(jq -r '.state // "missing"' "$cache_status")" != "complete" ]]; then
    echo "encoder cache did not validate complete"
    exit 1
fi
if [[ "$(jq -r '.coverage.complete // false' "$cache_status")" != "true" ]]; then
    echo "encoder cache coverage validation failed"
    exit 1
fi

echo "[$(date --iso-8601=seconds)] cache validated; resuming training"
exec env \
    PYTHONPATH=src \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "$python_bin" -m "$module" train --config "$resume_config"
