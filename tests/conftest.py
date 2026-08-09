"""Make the src-layout package importable from tests/ (run: `pytest tests/`).

Also sets the env flags several tests rely on (CPU-only kernels; no torch.compile)."""
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

import trainvm_binary  # noqa: E402  (needs the sys.path line above)
import ztok_binary  # noqa: E402  (same)

os.environ.setdefault("RWKV8_FORCE_PYREF", "1")   # CPU wkv7 reference (no fla/GPU)
os.environ.setdefault("CODA_NO_COMPILE", "1")     # skip torch.compile in tests

# These are idle-GPU benchmark/qualification programs with module-level work,
# not pytest test modules. Importing them during xdist collection would execute
# one full 4096-wide GPU workload in every worker. The parallel runner invokes
# them directly and sequentially when RWKV_GPU_STRESS=1.
collect_ignore = ["test_compile_core.py", "test_dmt_graph.py"]


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
    return [trainvm_binary.report_line(), *ztok_binary.report_lines()]
