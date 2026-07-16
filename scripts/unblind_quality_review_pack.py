#!/usr/bin/env python3
"""Add source-dataset labels from a review pack's private key to its manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pack", type=Path, default=ROOT / "quality_review_pack")
    args = parser.parse_args()
    pack = args.pack.resolve()
    manifest_path, key_path = pack / "review_manifest.jsonl", pack / "source_key.jsonl"
    manifest = [json.loads(line) for line in manifest_path.open() if line.strip()]
    source_by_id = {row["id"]: row["source_dataset"] for row in (json.loads(line) for line in key_path.open() if line.strip())}
    for row in manifest:
        row["source_dataset"] = source_by_id[row["id"]]
    with manifest_path.open("w") as handle:
        for row in manifest:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"unblinded {len(manifest)} review items")


if __name__ == "__main__":
    main()
