#!/usr/bin/env python3
"""Build the deterministic first-stage caption mixture for MoonViT→RWKV.

Stage one deliberately holds out the user's forthcoming refined dataset.  It
uses locally paired images only and makes the sources visible on every row so
sampling and later quality audits are reproducible.
"""
from __future__ import annotations

import json
import random
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    "grid": (ROOT / "curated_vision/grid_caption_no_titles.jsonl", 0.68),
    "joy": (ROOT / "curated_vision/joy_matched_cleaned.jsonl", 0.30),
    "anime": (ROOT / "curated_vision/anime_b_cleaned.jsonl", 0.02),
}
OUT = ROOT / "curated_vision/vision_stage1_mix.jsonl"
SEED = 20260714


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def main() -> None:
    rng = random.Random(SEED)
    rows = {name: load(path) for name, (path, _) in SOURCES.items()}
    # Use every currently matched Joy row once. The 30% target determines the
    # epoch size; grid then contributes a broad, non-repeated sample.
    total = round(len(rows["joy"]) / SOURCES["joy"][1])
    mixed: list[dict] = []
    for name, (_, ratio) in SOURCES.items():
        target = round(total * ratio)
        source = rows[name]
        if target <= len(source):
            selected = rng.sample(source, target)
        else:  # anime is intentionally tiny; its capped 2% uses deterministic reuse.
            selected = [source[i % len(source)] for i in range(target)]
        for row in selected:
            mixed.append({**row, "stage1_source": name})
    rng.shuffle(mixed)
    with OUT.open("w") as handle:
        for row in mixed:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    counts = {name: sum(row["stage1_source"] == name for row in mixed) for name in SOURCES}
    (OUT.with_suffix(".summary.json")).write_text(json.dumps({
        "seed": SEED, "rows": len(mixed), "counts": counts,
        "ratios": {name: count / len(mixed) for name, count in counts.items()},
        "purpose": "stage-1 bridge/alignment; later refined dataset intentionally held out",
    }, indent=2) + "\n")
    print(json.dumps({"output": str(OUT), "rows": len(mixed), "counts": counts}, indent=2))


if __name__ == "__main__":
    main()
