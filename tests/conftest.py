"""Make the src-layout package importable from tests/ (run: `pytest tests/`).

Also sets the env flags several tests rely on (CPU-only kernels; no torch.compile)."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("RWKV8_FORCE_PYREF", "1")   # CPU wkv7 reference (no fla/GPU)
os.environ.setdefault("CODA_NO_COMPILE", "1")     # skip torch.compile in tests

# These are idle-GPU benchmark/qualification programs with module-level work,
# not pytest test modules. Importing them during xdist collection would execute
# one full 4096-wide GPU workload in every worker. The parallel runner invokes
# them directly and sequentially when RWKV_GPU_STRESS=1.
collect_ignore = ["test_compile_core.py", "test_dmt_graph.py"]
