#!/usr/bin/env python3
"""Run the MoonViT-only and frozen-compressor caption probes sequentially."""
from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
STATUS = ROOT / "runs/vision_representation_ab_status.json"
STEPS = 2000


def latest_step(path: Path) -> int:
    step = 0
    if path.is_file():
        for line in path.open():
            try:
                step = max(step, int(json.loads(line).get("step", 0)))
            except (ValueError, TypeError, json.JSONDecodeError):
                pass
    return step


def publish(**values) -> None:
    payload = {"updated": time.time(), **values}
    temporary = STATUS.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(STATUS)
    print(payload, flush=True)


def main() -> None:
    for arm in ("moonvit", "compressor"):
        run = ROOT / f"runs/vision_ab_{arm}"
        if latest_step(run / "train.jsonl") >= STEPS:
            publish(state="arm_complete", arm=arm, step=STEPS)
            continue
        publish(state="running", arm=arm, step=latest_step(run / "train.jsonl"))
        subprocess.run([str(ROOT / "scripts/run_vision_representation_ab_arm.sh"), arm],
                       cwd=ROOT, check=True)
        if latest_step(run / "train.jsonl") < STEPS:
            raise RuntimeError(f"{arm} exited before step {STEPS}")
        publish(state="arm_complete", arm=arm, step=STEPS)
    publish(state="complete", arms=["moonvit", "compressor"], step=STEPS)


if __name__ == "__main__":
    main()
