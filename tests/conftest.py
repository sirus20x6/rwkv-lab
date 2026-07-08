"""Make the flat top-level modules importable from tests/ (run: `pytest tests/`).

The project keeps its modules flat at the repo root (spectral_muon.py, looped_rwkv.py,
lookahead_module.py, ...), so tests import them by bare name. Inserting the repo root on
sys.path here lets pytest resolve those imports regardless of where it is invoked from.
Also sets the env flags several tests rely on (CPU-only kernels; no torch.compile)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("RWKV8_FORCE_PYREF", "1")   # CPU wkv7 reference (no fla/GPU)
os.environ.setdefault("CODA_NO_COMPILE", "1")     # skip torch.compile in tests
