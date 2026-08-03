#!/usr/bin/env python3
"""Prepare immutable AO3 source shards for continued pretraining."""

from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from rwkv_lab.ao3_cpt_data import main


if __name__ == "__main__":
    main()
