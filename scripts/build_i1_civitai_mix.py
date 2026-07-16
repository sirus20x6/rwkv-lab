#!/usr/bin/env python3
"""Weave the normalized Civitai train split into the image-backed i1 mix.

The supplied Civitai eval split is normalized with the same rules but remains a
separate manifest. Negative prompts are not descriptions of visible content and
are therefore excluded from both targets.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = Path("/thearray/git/captioning/qwen3_vl_lora/dataset_civitai_all_normalized")
QUALITY_TAGS = {
    "masterpiece", "best quality", "amazing", "amazing quality", "high quality",
    "highest quality", "absurdres", "highres", "4k", "8k", "uhd",
    "ultra high resolution",
}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path,
                    default=ROOT / "curated_vision/vision_next_i1_25pct.jsonl")
    ap.add_argument("--civitai-train", type=Path,
                    default=DEFAULT_SOURCE / "train.jsonl")
    ap.add_argument("--civitai-eval", type=Path,
                    default=DEFAULT_SOURCE / "eval.jsonl")
    ap.add_argument("--output", type=Path,
                    default=ROOT / "curated_vision/vision_next_i1_civitai.jsonl")
    ap.add_argument("--eval-output", type=Path,
                    default=ROOT / "curated_vision/vision_next_i1_civitai_eval.jsonl")
    ap.add_argument("--seed", type=int, default=20260714)
    return ap.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    # Iterate physical lines: valid JSON may contain a literal U+2028, which
    # str.splitlines() incorrectly treats as a record boundary.
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def clean_caption(caption: str) -> str:
    text = re.sub(r"^\s*positive\s+prompt\s*:\s*", "", caption,
                  count=1, flags=re.IGNORECASE)
    text = re.split(r"\n\s*negative\s+prompt\s*:\s*", text,
                    maxsplit=1, flags=re.IGNORECASE)[0]
    # Remove only standalone comma-delimited quality incantations. Do not
    # rewrite descriptive prose or words that merely contain these phrases.
    parts = [part.strip() for part in text.split(",")]
    parts = [part for part in parts if part and part.casefold() not in QUALITY_TAGS]
    return ", ".join(parts).strip(" ,\n\t")


def normalize(rows: list[dict], split: str) -> tuple[list[dict], Counter]:
    output = []
    stats = Counter(input=len(rows))
    seen_images = set()
    for row in rows:
        image = Path(str(row.get("image", "")))
        text = clean_caption(str(row.get("caption", "")))
        if not image.is_file():
            stats["missing_image"] += 1
            continue
        if not text:
            stats["empty_after_cleaning"] += 1
            continue
        resolved = str(image.resolve())
        if resolved in seen_images:
            stats["duplicate_image"] += 1
            continue
        seen_images.add(resolved)
        if row.get("has_negative_prompt"):
            stats["negative_prompt_removed"] += 1
        output.append({
            "image": resolved,
            "text": text,
            "stage1_source": "civitai",
            "dataset_source": "civitai_all_normalized",
            "dataset_split": split,
            "image_id": row.get("image_id"),
        })
    stats["output"] = len(output)
    return output, stats


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    base = read_jsonl(args.base)
    civitai_train, train_stats = normalize(read_jsonl(args.civitai_train), "train")
    civitai_eval, eval_stats = normalize(read_jsonl(args.civitai_eval), "eval")

    train_paths = {str(Path(row["image"]).resolve()) for row in base}
    train_paths.update(row["image"] for row in civitai_train)
    leaked = [row for row in civitai_eval if row["image"] in train_paths]
    if leaked:
        raise SystemExit(f"refusing train/eval image leakage: {len(leaked)} overlapping images")

    combined = base + civitai_train
    random.Random(args.seed).shuffle(combined)
    write_jsonl(args.output, combined)
    write_jsonl(args.eval_output, civitai_eval)

    counts = Counter(row.get("stage1_source", "unknown") for row in combined)
    receipt = {
        "seed": args.seed,
        "rows": len(combined),
        "eval_rows": len(civitai_eval),
        "counts": dict(sorted(counts.items())),
        "ratios": {name: count / len(combined) for name, count in sorted(counts.items())},
        "civitai_train_stats": dict(train_stats),
        "civitai_eval_stats": dict(eval_stats),
        "train_eval_image_overlap": 0,
        "train_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
        "eval_sha256": hashlib.sha256(args.eval_output.read_bytes()).hexdigest(),
        "caption_policy": "strip Positive prompt header; discard Negative prompt section; remove standalone quality tags",
    }
    args.output.with_suffix(".summary.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
