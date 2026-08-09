#!/usr/bin/env bash
set -euo pipefail

# CPU tests use processes (pytest-xdist), which avoids Python's GIL. CUDA tests
# share one device and run serially to avoid allocator/Inductor cache contention
# and the OOM failures caused by concurrent compiler captures.
#
#   scripts/test_parallel.sh [--gpu]
#
# The GPU phase is opt-in. It used to run unconditionally, which meant the
# ordinary local test command took a device that may be driving a display or a
# training run.
run_gpu=0
if [[ "${1:-}" == "--gpu" ]]; then
  run_gpu=1
  shift
fi
if [[ $# -ne 0 ]]; then
  echo "usage: scripts/test_parallel.sh [--gpu]" >&2
  exit 2
fi
if [[ "${RWKV_GPU_STRESS:-0}" == "1" && "$run_gpu" != "1" ]]; then
  echo "RWKV_GPU_STRESS=1 needs --gpu: the stress programs require a device" >&2
  exit 2
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
workers="${PYTEST_WORKERS:-4}"
# Bound each worker's native BLAS/OpenMP pool. Without this, four PyTorch
# workers each claim the whole machine and are slower than one process.
native_threads="${PYTEST_NATIVE_THREADS:-4}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${native_threads}}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-${native_threads}}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-${native_threads}}"
# Masked one process early: marker filtering happens after test modules are
# imported, so `-m "not gpu"` alone does not keep the driver closed.
python scripts/non_gpu_environment.py \
  python -m pytest -q -n "${workers}" --dist worksteal -m "not gpu" tests

if [[ "$run_gpu" == "1" ]]; then
  TRAINVM_TEST_ACCELERATOR_ACCESS=1 python -m pytest -q -m gpu tests
else
  echo "GPU suite skipped; pass --gpu to authorize accelerator tests"
fi

# These full-size compile/graph programs require an otherwise idle GPU and are
# deliberately outside normal pytest collection. Opt in for release validation.
if [[ "$run_gpu" == "1" && "${RWKV_GPU_STRESS:-0}" == "1" ]]; then
  export TRAINVM_TEST_ACCELERATOR_ACCESS=1
  python tests/test_compile_core.py
  python tests/test_dmt_graph.py
fi
