#!/usr/bin/env python3
"""Publish one deterministic, non-overlapping slice of a JSONL vision corpus."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--size", type=int, default=40_000)
    parser.add_argument("--summary", type=Path)
    args = parser.parse_args()
    if args.index < 0 or args.size < 1:
        parser.error("--index must be nonnegative and --size positive")
    start, stop = args.index * args.size, (args.index + 1) * args.size
    selected: list[str] = []
    images: set[str] = set()
    with args.source.open() as handle:
        for row_index, line in enumerate(handle):
            if row_index < start:
                continue
            if row_index >= stop:
                break
            if not line.strip():
                raise SystemExit(f"blank source row at index {row_index}")
            row = json.loads(line)
            image = str(row.get("image", ""))
            if not image:
                raise SystemExit(f"source row {row_index} has no image")
            if image in images:
                raise SystemExit(f"duplicate image inside shard: {image}")
            images.add(image)
            selected.append(line if line.endswith("\n") else line + "\n")
    if len(selected) != args.size:
        raise SystemExit(
            f"source ended before requested shard: expected={args.size} got={len(selected)}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    with temporary.open("w") as handle:
        handle.writelines(selected)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, args.output)
    summary_path = args.summary or args.output.with_suffix(".summary.json")
    summary = {
        "schema": 1,
        "source": str(args.source.resolve()),
        "source_sha256": digest(args.source),
        "shard_index": args.index,
        "start_row_inclusive": start,
        "stop_row_exclusive": stop,
        "rows": len(selected),
        "unique_images": len(images),
        "manifest": str(args.output.resolve()),
        "manifest_sha256": digest(args.output),
    }
    summary_temp = summary_path.with_suffix(summary_path.suffix + ".tmp")
    summary_temp.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    os.replace(summary_temp, summary_path)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
