"""Make the src-layout package importable from tests/ (run: `pytest tests/`).

Also makes the session CPU-only unless accelerator access was explicitly
authorized, and sets the env flags several tests rely on (CPU-only kernels; no
torch.compile).

The masking is here, in top-level code, rather than in `pytest_configure`: a
root conftest is imported before any test module, but `pytest_configure` runs
after, and after is too late. Marker filtering is later still, which is the
whole bug -- see `scripts/non_gpu_environment.py`, which applies the same mask
one process earlier for callers that are not already Python.
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
# The checkout root, so `import scripts.<gate>` resolves to this tree rather
# than to whatever the invocation directory happens to contain. Tests that
# import or execute repository code must be anchored to the repository, for the
# same reason the trainvm gate binary is: a test that grades a different tree
# than the one it was pointed at reports a verdict about the wrong source.
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # tests/ helpers

from scripts.non_gpu_environment import (  # noqa: E402  (needs the sys.path lines)
    ACCELERATOR_ACCESS_ENV,
    NON_GPU_ENVIRONMENT,
    accelerator_access_enabled,
)

ACCELERATOR_ACCESS_ENABLED = accelerator_access_enabled()
if not ACCELERATOR_ACCESS_ENABLED:
    # Assign rather than setdefault. setdefault would let a training or gaming
    # shell's own device list leak into a run that calls itself CPU-only, which
    # is exactly the case this exists to stop.
    os.environ.update(NON_GPU_ENVIRONMENT)

import trainvm_binary  # noqa: E402  (needs the sys.path line above)
import ztok_binary  # noqa: E402  (same)

os.environ.setdefault("RWKV8_FORCE_PYREF", "1")   # CPU wkv7 reference (no fla/GPU)
os.environ.setdefault("CODA_NO_COMPILE", "1")     # skip torch.compile in tests

# These are idle-GPU benchmark/qualification programs with module-level work,
# not pytest test modules. Importing them during xdist collection would execute
# one full 4096-wide GPU workload in every worker. The parallel runner invokes
# them directly and sequentially when RWKV_GPU_STRESS=1.
collect_ignore = ["test_compile_core.py", "test_dmt_graph.py"]


def pytest_collection_modifyitems(items):
    """Keep GPU tests opt-in even when a caller forgets `-m "not gpu"`.

    Default-deny rather than relying on every invocation to pass the marker
    expression. `scripts/test_parallel.sh` ran the gpu suite unconditionally
    until this landed, so "the caller always passes -m" was not true.
    """
    if ACCELERATOR_ACCESS_ENABLED:
        return
    import pytest

    denied = pytest.mark.skip(
        reason=f"accelerator tests require {ACCELERATOR_ACCESS_ENV}=1",
    )
    for item in items:
        if item.get_closest_marker("gpu") is not None:
            item.add_marker(denied)


def pytest_report_header():
    """State which foreign tools this run can see, before any of them is used.

    These tests hand evidence to the native authority and believe its verdict,
    so the binary is part of the result. It used to be resolved from PATH in
    preference to the build tree and never recorded, which let a stale global
    install grade a fresh checkout invisibly. Reporting it unconditionally --
    including "none" -- means no run leaves the question open.

    ztok is here for the same reason and is reported the same way. It no longer
    decides anything (test_world_vocab.py grades against committed expectations
    now), but its presence still changes which tests run: the cross-check in
    test_world_vocab.py and the AO3 fixture in test_benchmark_runner.py both
    skip without it, and a skip that does not name what was missing is the same
    silence this header exists to end.
    """
    disposition = "explicitly enabled" if ACCELERATOR_ACCESS_ENABLED else "masked"
    return [
        f"accelerator access: {disposition}",
        trainvm_binary.report_line(),
        *ztok_binary.report_lines(),
    ]
