"""Make the src-layout package importable from tests/ (run: `pytest tests/`).

Also sets the env flags several tests rely on (CPU-only kernels; no torch.compile)."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("RWKV8_FORCE_PYREF", "1")   # CPU wkv7 reference (no fla/GPU)
os.environ.setdefault("CODA_NO_COMPILE", "1")     # skip torch.compile in tests
