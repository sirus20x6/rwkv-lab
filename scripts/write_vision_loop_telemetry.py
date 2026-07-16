#!/usr/bin/env python3
"""Materialize dashboard loop telemetry from a vision adapter checkpoint."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from rwkv_lab.vision_loop import loop_telemetry_from_states


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()
    blob = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    saved = blob.get("args", {})
    payload = loop_telemetry_from_states(
        blob["loops"], loop_count=int(saved.get("loop_count", 1)),
        gate_cap=float(saved.get("loop_gate_cap", 0.25)),
        step=int(blob.get("step", 0)))
    output = args.output or args.checkpoint.with_name("loop_rw.json")
    temporary = output.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload) + "\n")
    os.replace(temporary, output)
    print(json.dumps({"output": str(output), "step": payload["step"],
                      "layers": payload["n_layers"],
                      "mean_max_rw": payload["mean_max_rw"],
                      "n_pinned": payload["n_pinned"]}, indent=2))


if __name__ == "__main__":
    main()
