#!/usr/bin/env python3
"""Atomically replace ambiguous COCO mask prompts in existing manifests."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from build_captioning_first_mix import coco_sam_prompt


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PATHS = (
    ROOT / "datasets/captioning_first_work/coco_sam.jsonl",
    ROOT / "curated_vision/captioning_first_train.jsonl",
    ROOT / "curated_vision/captioning_first_eval.jsonl",
)
POLICY = "coco_polygon_mask16_v2_explicit_request"


def repair(path: Path, *, check: bool = False) -> tuple[int, int]:
    temporary = path.with_suffix(path.suffix + ".prompt-repair.tmp")
    seen = changed = 0
    output = None if check else temporary.open("w")
    try:
        with path.open() as source:
            for line in source:
                row = json.loads(line)
                if row.get("caption_variant") == "coco_instance_polygon_mask16":
                    seen += 1
                    prompt = coco_sam_prompt(str(row["text"]))
                    if row.get("prompt") != prompt or row.get("caption_policy") != POLICY:
                        row["prompt"] = prompt
                        row["caption_policy"] = POLICY
                        changed += 1
                if output is not None:
                    output.write(json.dumps(row, ensure_ascii=False) + "\n")
        if output is not None:
            output.flush()
            os.fsync(output.fileno())
            output.close()
            output = None
            os.replace(temporary, path)
    finally:
        if output is not None:
            output.close()
        temporary.unlink(missing_ok=True)
    return seen, changed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", type=Path, default=DEFAULT_PATHS)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    total_changed = 0
    for path in args.paths:
        seen, changed = repair(path, check=args.check)
        total_changed += changed
        print({"path": str(path), "coco_rows": seen, "changed": changed,
               "mode": "check" if args.check else "repair"}, flush=True)
    if args.check and total_changed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
