#!/usr/bin/env python3
"""Atomically combine image-disjoint JSONL manifests."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows, images = [], set()
    for source in args.input:
        with source.open() as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                image = str(row["image"])
                if image in images:
                    raise SystemExit(f"duplicate image across manifests: {image}")
                images.add(image)
                rows.append(line if line.endswith("\n") else line + "\n")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    with temporary.open("w") as handle:
        handle.writelines(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, args.output)
    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    print({"kind": "combined_manifest", "rows": len(rows),
           "unique_images": len(images), "sha256": digest,
           "output": str(args.output)}, flush=True)


if __name__ == "__main__":
    main()
