#!/usr/bin/env bash
set -euo pipefail

# CPU tests use processes (pytest-xdist), which avoids Python's GIL. CUDA tests
# share one device and run serially to avoid allocator/Inductor cache contention
# and the OOM failures caused by concurrent compiler captures.
workers="${PYTEST_WORKERS:-4}"
# Bound each worker's native BLAS/OpenMP pool. Without this, four PyTorch
# workers each claim the whole machine and are slower than one process.
native_threads="${PYTEST_NATIVE_THREADS:-4}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${native_threads}}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-${native_threads}}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-${native_threads}}"
python -m pytest -q -n "${workers}" --dist worksteal -m "not gpu" tests
python -m pytest -q -m gpu tests

# These full-size compile/graph programs require an otherwise idle GPU and are
# deliberately outside normal pytest collection. Opt in for release validation.
if [[ "${RWKV_GPU_STRESS:-0}" == "1" ]]; then
  python tests/test_compile_core.py
  python tests/test_dmt_graph.py
fi
