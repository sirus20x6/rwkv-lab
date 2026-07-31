#!/usr/bin/env python3
"""Create a validated Qwen AO3 CPT run directory and launcher."""

from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from rwkv_lab.qwen_ao3_cpt import main


if __name__ == "__main__":
    main()
