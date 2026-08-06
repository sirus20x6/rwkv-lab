#!/usr/bin/env python3
"""Build a deterministic legacy dashboard caption-comparison snapshot."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path


def _read_records(path: Path) -> dict[str, dict[str, object]]:
    records: dict[str, dict[str, object]] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        record = json.loads(line)
        record_id = str(record.get("id", ""))
        if not record_id or record_id in records:
            raise ValueError(f"{path}:{line_number}: missing or duplicate id")
        records[record_id] = record
    return records


def _rank(seed: str, record_id: str) -> bytes:
    return hashlib.sha256(f"{seed}:{record_id}".encode()).digest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--ppl", type=float, required=True)
    parser.add_argument("--per-group", type=int, default=6)
    parser.add_argument("--max-new", type=int, default=768)
    parser.add_argument("--seed", default="caption-before-after-v1")
    args = parser.parse_args()
    if args.step < 0 or args.ppl <= 0 or args.per_group < 1 or args.max_new < 1:
        raise ValueError("step, ppl, per-group, and max-new must be positive")
    before = _read_records(args.before)
    after = _read_records(args.after)
    if before.keys() != after.keys():
        raise ValueError("before and after files do not contain identical image IDs")

    groups: dict[str, list[str]] = defaultdict(list)
    for record_id, final in after.items():
        initial = before[record_id]
        image = Path(str(final.get("image", "")))
        if str(initial.get("image", "")) != str(image):
            raise ValueError(f"image identity changed for {record_id}")
        if not image.is_file():
            raise FileNotFoundError(image)
        if not str(initial.get("response", "")).strip() or not str(final.get("response", "")).strip():
            raise ValueError(f"empty comparison caption for {record_id}")
        group = str(final.get("i1_subset", "unknown"))
        if str(initial.get("i1_subset", "unknown")) != group:
            raise ValueError(f"source group changed for {record_id}")
        groups[group].append(record_id)

    selected: list[str] = []
    for group in sorted(groups):
        candidates = sorted(groups[group], key=lambda value: _rank(args.seed, value))
        if len(candidates) < args.per_group:
            raise ValueError(f"group {group!r} has fewer than {args.per_group} records")
        selected.extend(candidates[: args.per_group])

    items = []
    for record_id in selected:
        initial, final = before[record_id], after[record_id]
        generated_tokens = int(final.get("generated_tokens", 0))
        items.append(
            {
                "image": final["image"],
                "prompt": "",
                "reference": str(initial["response"]),
                "caption": str(final["response"]),
                "tokens": generated_tokens,
                "stopped_at_eod": 0 < generated_tokens < args.max_new,
                "source": (
                    f"before: untouched BF16 base · after: rank-256 LoRA step {args.step}"
                    f" · {final.get('i1_subset', 'unknown')}"
                ),
            }
        )
    artifact = {
        "step": args.step,
        "ppl": args.ppl,
        "eval_kind": "caption_before_after",
        "decoding": "greedy paired by frozen image id",
        "max_new": args.max_new,
        "complete": True,
        "items": items,
        "comparison": {
            "before": str(args.before.resolve()),
            "after": str(args.after.resolve()),
            "selection_seed": args.seed,
            "groups": {group: args.per_group for group in sorted(groups)},
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(f"refusing to replace {args.output}")
    temporary = args.output.with_name(f".{args.output.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, args.output)
    print(json.dumps({"output": str(args.output), "items": len(items), "groups": sorted(groups)}))


if __name__ == "__main__":
    main()
