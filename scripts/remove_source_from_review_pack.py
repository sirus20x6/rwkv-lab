#!/usr/bin/env python3
"""Remove one source dataset and its copied assets from a review pack."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("pack", type=Path)
    parser.add_argument("source_dataset")
    args = parser.parse_args()
    pack = args.pack.resolve()
    manifest_path = pack / "review_manifest.jsonl"
    rows = [json.loads(line) for line in manifest_path.open() if line.strip()]
    removed = [row for row in rows if row.get("source_dataset") == args.source_dataset]
    retained = [row for row in rows if row.get("source_dataset") != args.source_dataset]
    for row in removed:
        image = row.get("image")
        if image:
            (pack / image).unlink(missing_ok=True)
    with manifest_path.open("w") as handle:
        for row in retained:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    key_path = pack / "source_key.jsonl"
    if key_path.exists():
        keys = [json.loads(line) for line in key_path.open() if line.strip()]
        with key_path.open("w") as handle:
            for row in keys:
                if row.get("source_dataset") != args.source_dataset:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"removed {len(removed)} items from {pack.name}")


if __name__ == "__main__":
    main()
