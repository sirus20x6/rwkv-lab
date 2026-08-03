#!/usr/bin/env python3
"""Atomically export a weights-only frozen compressor checkpoint."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rwkv_lab.vision_compressor_features import frozen_payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = frozen_payload(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, args.output)
    print({"kind": "frozen_compressor_export", "output": str(args.output),
           "source_step": payload["source_step"], "frozen": True}, flush=True)


if __name__ == "__main__":
    main()
