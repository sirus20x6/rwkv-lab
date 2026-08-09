#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "usage: $0 RUN_DIRECTORY RUN_ID BASELINE_TOTAL WORKER_PID" >&2
  exit 64
fi

run_directory=$1
run_id=$2
baseline_total=$3
worker_pid=$4
baseline_log="${run_directory}/baseline-test.jsonl"
train_log="${run_directory}/train.jsonl"
last_completed=-1

# Hostd runs the worker under an isolated UID, so an unprivileged dashboard
# bridge cannot use signal permission as its liveness test. procfs presence is
# sufficient here: the bridge only emits progress and never controls the worker.
while [[ -d "/proc/${worker_pid}" ]]; do
  completed=0
  if [[ -f "${baseline_log}" ]]; then
    completed=$(wc -l <"${baseline_log}")
    completed=${completed//[[:space:]]/}
  fi
  if [[ "${completed}" != "${last_completed}" ]]; then
    /usr/bin/jq -cn \
      --arg run_id "${run_id}" \
      --argjson completed "${completed}" \
      --argjson total "${baseline_total}" \
      '{kind:"train",step:0,phase:"baseline_eval",run_id:$run_id,baseline_images_completed:$completed,baseline_images_total:$total}' \
      >>"${train_log}"
    last_completed=${completed}
  fi
  if (( completed >= baseline_total )); then
    exit 0
  fi
  sleep 30
done
