#!/usr/bin/env bash
set -euo pipefail
cd /workspace/git/moe-mla

# Every shard loads its own full C-RADIOv4-H copy, so without an explicit
# device they all land on cuda:0 and fight over one card's memory. Shards are
# assigned round-robin across V4H_CACHE_DEVICES, and the worker count defaults
# to one shard per device so no card is oversubscribed by accident.
devices_spec="${V4H_CACHE_DEVICES:-}"
if [[ -z "$devices_spec" ]]; then
  devices_spec="$(python -c 'import torch; print(",".join(
      f"cuda:{index}" for index in range(torch.cuda.device_count())) or "cpu")')"
fi
IFS=',' read -r -a devices <<<"$devices_spec"
[[ "${#devices[@]}" -ge 1 ]] || { echo "no cache devices resolved" >&2; exit 1; }

workers="${V4H_CACHE_WORKERS:-${#devices[@]}}"
[[ "$workers" -ge 1 ]] || { echo "V4H_CACHE_WORKERS must be positive" >&2; exit 1; }
if [[ "$workers" -gt "${#devices[@]}" ]]; then
  echo "warning: $workers shards share ${#devices[@]} device(s): ${devices[*]}" >&2
fi

cache="${V4H_CACHE:-/workspace/downloads/cache/moe-mla/radio_v4h_all_adaptors_4096}"
log_dir="${V4H_CACHE_LOG_DIR:-runs/cache_v4h_all_adaptors}"
mkdir -p "$log_dir"
reconfigure=()
[[ "${V4H_CACHE_ALLOW_RECONFIGURE:-0}" == "1" ]] \
  && reconfigure=(--allow-reconfigure)

pids=()
for ((shard=0; shard<workers; shard++)); do
  python scripts/cache_v4h_all_adaptors.py \
    --manifest curated_vision/captioning_ocr_fixed_train.jsonl \
               curated_vision/captioning_ocr_fixed_eval.jsonl \
    --cache-dir "$cache" \
    --num-shards "$workers" \
    --shard-index "$shard" \
    --device "${devices[shard % ${#devices[@]}]}" \
    "${reconfigure[@]}" \
    --max-failure-rate "${V4H_CACHE_MAX_FAILURE_RATE:-0.01}" \
    >"$log_dir/shard_${shard}.log" 2>&1 &
  pids+=("$!")
done

# Exit 3 from a shard means a few individual sources were skipped; exit 1 means
# that shard's slice of the cache is untrustworthy and must not be trained on.
failed=0
tolerated=0
for pid in "${pids[@]}"; do
  code=0
  wait "$pid" || code=$?
  case "$code" in
    0) ;;
    3) tolerated=1 ;;
    *) failed=1 ;;
  esac
done
if [[ "$failed" -ne 0 ]]; then
  echo "cache shards failed above the tolerated rate; see $log_dir" >&2
  exit 1
fi
if [[ "$tolerated" -ne 0 ]]; then
  echo "cache complete with tolerated per-source failures; see $log_dir" >&2
  exit 3
fi
exit 0
