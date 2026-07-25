#!/usr/bin/env python3
"""Add cleaned Civitai and Joy prose to the captioning-first RADIO curriculum."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CIVITAI = Path(
    "/thearray/git/captioning/qwen3_vl_lora/dataset_civitai_all_normalized")
PROMPT = "Describe this image accurately and only state visible details:\n"
QUALITY_TAGS = {
    "masterpiece", "best quality", "amazing", "amazing quality", "high quality",
    "highest quality", "absurdres", "highres", "4k", "8k", "uhd",
    "ultra high resolution",
}
GENERATIONISM = re.compile(
    r"(?ix)(?:\bmaster[_\s-]?piece\b|\bmaster[_\s-]?class\b|"
    r"\bbest\s+quality\b|\bamazing\s+quality\b|"
    r"\bhighest\s+quality\b|\babsurd\s*res\b|\bhigh\s*res\b|\bhi-res\b|"
    r"\bvery\s+aesthetic\b|\baward[- ]winning\b|\buhd\b|"
    r"\b(?:high|very\s+high|low|poor|excellent)\s+quality\b|"
    r"\b(?:highly|ultra|extremely)\s+detailed\b|"
    r"\b(?:best|excellent)\s+(?:detail|details|anatomy)\b|"
    r"\b(?:coherent|correct)\s+anatomy\b|\bbalanced\s+proportions\b|"
    r"\bperfect\s+resolution\b|\bup_[0-9]+\b|\baddmicrodetails\b|"
    r"\b(?:4k|8k)\s*(?:resolution)?\b|\bhdr\b|"
    r"<\s*(?:lora|lyco|embedding):|\blora:|"
    r"\b(?:positive|negative)\s+prompt\s*:|<segment:|\f)"
)
CIVITAI_META_INSTRUCTION = re.compile(
    r"(?ix)(?:"
    r"\b(?:make\s+this|rewrite\s+this|respond\s+with|answer\s+with|"
    r"generate\s+(?:an?|the|this)|firm\s+policy|new\s+civitai|"
    r"consider\s+(?:two|three|several)\s+alternatives)\b|"
    r"\bstyle\s+(?:replication|tags)\s*:|\bembedding:urn:|\bair:sd\d|"
    r"\bsource_anime\s+screenshot\b"
    r")"
)
JOY_CAPTION_SPAM = re.compile(
    r"(?ix)(?:https?://|www\.|\b[a-z0-9_-]+\.(?:com|net|org|tumblr)\b|"
    r"(?:^|\s)\#[a-z][\w-]*|[\U0001F300-\U0001FAFF\u2600-\u27BF]|"
    r"\b(?:check\s+out|follow\s+(?:the\s+)?(?:artist|creator|\w+\s+for)|"
    r"drop\s+a\s+comment|comment\s+below|what\s+do\s+you(?:\s+all)?\s+think|"
    r"shout[- ]?out|perfect\s+for|ideal\s+for|for\s+fans\s+of|"
    r"art\s+enthusiasts|your\s+feed|new\s+comic\s+strip\s+alert|"
    r"hey\s+everyone|dive\s+into|capture\s+the\s+charm|immerse\s+yourself|"
    r"no\s+(?:visible|noticeable)\s+jpe?g\s+artifacts?|"
    r"jpe?g\s+artifacts?\s+(?:are|is)|artist/username\s+if\s+known)\b)"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-train", type=Path,
                        default=ROOT / "curated_vision/captioning_first_train.jsonl")
    parser.add_argument("--base-eval", type=Path,
                        default=ROOT / "curated_vision/captioning_first_eval.jsonl")
    parser.add_argument("--civitai-train", type=Path,
                        default=DEFAULT_CIVITAI / "train.jsonl")
    parser.add_argument("--civitai-eval", type=Path,
                        default=DEFAULT_CIVITAI / "eval.jsonl")
    parser.add_argument("--joy", type=Path,
                        default=ROOT / "curated_vision/joy_matched_cleaned.jsonl")
    parser.add_argument("--output", type=Path, default=ROOT /
                        "curated_vision/captioning_first_civitai_joy_train.jsonl")
    parser.add_argument("--eval-output", type=Path, default=ROOT /
                        "curated_vision/captioning_first_civitai_joy_eval.jsonl")
    parser.add_argument("--adult-eval-per-source", type=int, default=64)
    parser.add_argument("--civitai-max-caption-repeats", type=int, default=4,
                        help="limit identical generation prompts dominating the caption prior")
    parser.add_argument("--metadata-workers", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260718)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def rooted(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def identity(row: dict) -> str:
    return str(rooted(str(row["image"])).resolve())


def clean_civitai_caption(value: str) -> str:
    text = re.sub(r"^\s*positive\s+prompt\s*:\s*", "", str(value), count=1,
                  flags=re.IGNORECASE)
    text = re.split(r"\n\s*negative\s+prompt\s*:\s*", text, maxsplit=1,
                    flags=re.IGNORECASE)[0]
    parts = [part.strip() for part in text.split(",")]
    parts = [part for part in parts
             if part and part.casefold() not in QUALITY_TAGS]
    text = ", ".join(parts).strip(" ,\n\t")
    # Quality incantations also occur embedded in prose rather than as clean
    # comma-delimited tags. Remove them wherever they appear, then repair only
    # punctuation/whitespace introduced by those deletions.
    text = GENERATIONISM.sub("", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    text = re.sub(r",(?:\s*,)+", ",", text)
    return text.strip(" ,;:\n\t")


def normalize_civitai(rows: list[dict], split: str) -> tuple[list[dict], Counter]:
    output: list[dict] = []
    stats = Counter(input=len(rows))
    seen: set[str] = set()
    for row in rows:
        image = rooted(str(row.get("image", "")))
        text = clean_civitai_caption(str(row.get("caption", "")))
        if not image.is_file():
            stats["missing_image"] += 1
            continue
        if not text:
            stats["empty_after_cleaning"] += 1
            continue
        if CIVITAI_META_INSTRUCTION.search(text):
            stats["meta_instruction"] += 1
            continue
        key = str(image.resolve())
        if key in seen:
            stats["duplicate_image"] += 1
            continue
        seen.add(key)
        stats["negative_prompt_removed"] += int(bool(row.get("has_negative_prompt")))
        output.append({
            "image": key,
            "text": text,
            "prompt": PROMPT,
            "stage1_source": "captioning_civitai",
            "source": "civitai",
            "task": "caption",
            "dataset_source": "civitai_all_normalized",
            "dataset_split": split,
            "image_id": row.get("image_id"),
            "caption_policy": "positive description only; negative prompt and quality tags removed",
        })
    stats["output"] = len(output)
    return output, stats


def cap_caption_repeats(rows: list[dict], maximum: int, seed: int,
                        stats: Counter) -> list[dict]:
    if maximum < 1:
        raise ValueError("caption repeat cap must be positive")
    shuffled = list(rows)
    random.Random(seed).shuffle(shuffled)
    counts: Counter[str] = Counter()
    output = []
    for row in shuffled:
        text = str(row["text"])
        if counts[text] >= maximum:
            stats["repeat_cap_removed"] += 1
            continue
        counts[text] += 1
        output.append(row)
    stats["output_after_repeat_cap"] = len(output)
    return output


def clean_joy_rows(rows: list[dict]) -> tuple[list[dict], Counter]:
    output = []
    stats = Counter(input=len(rows))
    seen = set()
    for row in rows:
        if row.get("question_type") != "caption":
            stats["non_caption"] += 1
            continue
        text = str(row.get("text", "")).strip()
        if not text:
            stats["empty"] += 1
            continue
        if GENERATIONISM.search(text):
            stats["generationism"] += 1
            continue
        if JOY_CAPTION_SPAM.search(text):
            stats["caption_spam"] += 1
            continue
        image = rooted(str(row.get("image", "")))
        if not image.is_file():
            stats["missing_image"] += 1
            continue
        key = str(image.resolve())
        if key in seen:
            stats["duplicate_image"] += 1
            continue
        seen.add(key)
        output.append({
            "image": key,
            "text": text,
            "prompt": PROMPT,
            "stage1_source": "captioning_joy",
            "source": "joy-captioning-20250408a",
            "task": "caption",
            "caption_policy": "matched prose only; response wrappers, generationisms, and promotional spam removed",
        })
    stats["output"] = len(output)
    return output, stats


def attach_dimensions(rows: list[dict], workers: int) -> None:
    def dimensions(row: dict) -> tuple[int, int]:
        if row.get("width") and row.get("height"):
            return int(row["width"]), int(row["height"])
        with Image.open(row["image"]) as image:
            return image.size

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for row, (width, height) in zip(rows, pool.map(dimensions, rows)):
            row["width"], row["height"] = width, height


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    args = parse_args()
    if (args.adult_eval_per_source < 1 or args.metadata_workers < 1
            or args.civitai_max_caption_repeats < 1):
        raise SystemExit("eval count, metadata workers, and caption repeat cap must be positive")
    base_train = read_jsonl(args.base_train)
    base_eval = read_jsonl(args.base_eval)
    civitai_train, civitai_train_stats = normalize_civitai(
        read_jsonl(args.civitai_train), "train")
    civitai_eval_all, civitai_eval_stats = normalize_civitai(
        read_jsonl(args.civitai_eval), "eval")
    civitai_train = cap_caption_repeats(
        civitai_train, args.civitai_max_caption_repeats,
        args.seed ^ 0xC17A1, civitai_train_stats)
    # Evaluation should measure a broad prompt set rather than repeatedly
    # scoring different images generated from the same exact target string.
    civitai_eval_all = cap_caption_repeats(
        civitai_eval_all, 1, args.seed ^ 0xE7A1, civitai_eval_stats)
    joy_all, joy_stats = clean_joy_rows(read_jsonl(args.joy))
    if min(len(civitai_eval_all), len(joy_all)) < args.adult_eval_per_source:
        raise SystemExit("not enough Civitai/Joy examples for the requested eval slice")

    rng = random.Random(args.seed)
    civitai_eval = rng.sample(civitai_eval_all, args.adult_eval_per_source)
    joy_eval = random.Random(args.seed ^ 0xA07).sample(
        joy_all, args.adult_eval_per_source)
    eval_ids = {identity(row) for row in base_eval + civitai_eval + joy_eval}
    train_ids = {identity(row) for row in base_train}
    adult_train = [row for row in civitai_train + joy_all
                   if identity(row) not in eval_ids and identity(row) not in train_ids]
    combined_train = base_train + adult_train
    combined_eval = base_eval + civitai_eval + joy_eval

    attach_dimensions(adult_train + civitai_eval + joy_eval, args.metadata_workers)
    rng.shuffle(combined_train)
    rng.shuffle(combined_eval)
    train_ids = {identity(row) for row in combined_train}
    final_eval_ids = {identity(row) for row in combined_eval}
    if len(train_ids) != len(combined_train):
        raise RuntimeError("combined training manifest contains duplicate image paths")
    if len(final_eval_ids) != len(combined_eval):
        raise RuntimeError("combined eval manifest contains duplicate image paths")
    overlap = train_ids & final_eval_ids
    if overlap:
        raise RuntimeError(f"train/eval image leakage: {len(overlap)}")

    write_jsonl(args.output, combined_train)
    write_jsonl(args.eval_output, combined_eval)
    source_counts = Counter(str(row.get("stage1_source") or "unknown")
                            for row in combined_train)
    eval_counts = Counter(str(row.get("stage1_source") or "unknown")
                          for row in combined_eval)
    task_counts = Counter(str(row.get("task") or "caption")
                          for row in combined_train)
    receipt = {
        "schema": 1,
        "seed": args.seed,
        "rows": len(combined_train),
        "eval_rows": len(combined_eval),
        "added_rows": len(adult_train),
        "source_counts": dict(sorted(source_counts.items())),
        "source_ratios": {name: count / len(combined_train)
                          for name, count in sorted(source_counts.items())},
        "eval_source_counts": dict(sorted(eval_counts.items())),
        "task_counts": dict(sorted(task_counts.items())),
        "civitai_train_stats": dict(civitai_train_stats),
        "civitai_eval_stats": dict(civitai_eval_stats),
        "joy_stats": dict(joy_stats),
        "adult_eval_per_source": args.adult_eval_per_source,
        "civitai_max_caption_repeats": args.civitai_max_caption_repeats,
        "train_eval_image_overlap": 0,
        "unique_train_images": len(train_ids),
        "unique_eval_images": len(final_eval_ids),
        "caption_policy": "retain grounded captioning-first curriculum; add cleaned Civitai descriptions and matched Joy prose",
        "train_sha256": digest(args.output),
        "eval_sha256": digest(args.eval_output),
    }
    summary = args.output.with_suffix(".summary.json")
    summary.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n",
                       encoding="utf-8")
    print(json.dumps({"output": str(args.output),
                      "eval_output": str(args.eval_output), **receipt},
                     indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
