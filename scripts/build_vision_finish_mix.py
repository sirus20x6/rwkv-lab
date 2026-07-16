#!/usr/bin/env python3
"""Build a grounded finishing mix without raw image-generation prompts.

The default train mixture is 63% real i1/Pexels prose, 17% matched Joy
prose, 15% recaptioned i1/Midjourney imagery, and 5% explicitly prompted tag
supervision. A source-stratified image-disjoint prose evaluation set replaces
the Civitai-prompt evaluation used by the broad alignment phase.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
CAPTION_COLUMNS = tuple(f"caption{i}" for i in range(1, 6))
TAG_PROMPT = "List relevant image tags:\n"
GRID_METADATA_PROMPT = "List relevant tags, categories, and cast:\n"
GENERATIONISM = re.compile(
    r"(?ix)(?:"
    r"\bmasterpiece\b|\bbest\s+quality\b|\bamazing\s+quality\b|"
    r"\bhighest\s+quality\b|\babsurd\s*res\b|\bhigh\s*res\b|\bhi-res\b|"
    r"\bvery\s+aesthetic\b|\baward[- ]winning\b|\buhd\b|"
    r"\b(?:high|very\s+high|low|poor)\s+quality\b|\bhighly\s+detailed\b|"
    r"\b(?:4k|8k)\s*(?:resolution)?\b|"
    r"<\s*(?:lora|lyco|embedding):|\blora:|"
    r"\b(?:positive|negative)\s+prompt\s*:|<segment:|\\f"
    r")"
)
NONVISUAL_TAG_PREFIX = re.compile(r"^(?:artist|copyright|meta):", re.IGNORECASE)
NONVISUAL_GRID_TAG = re.compile(
    r"(?ix)^(?:sclip|[^\s;,.]+\.(?:com|net|org|tv|xxx))$"
)
JOY_CAPTION_SPAM = re.compile(
    r"(?ix)(?:"
    r"https?://|www\.|\b[a-z0-9_-]+\.(?:com|net|org|tumblr)\b|"
    r"(?:^|\s)\#[a-z][\w-]*|[\U0001F300-\U0001FAFF\u2600-\u27BF]|"
    r"\b(?:check\s+out|follow\s+(?:the\s+)?(?:artist|creator|\w+\s+for)|"
    r"drop\s+a\s+comment|comment\s+below|what\s+do\s+you(?:\s+all)?\s+think|"
    r"shout[- ]?out|perfect\s+for|ideal\s+for|for\s+fans\s+of|"
    r"art\s+enthusiasts|your\s+feed|new\s+comic\s+strip\s+alert|"
    r"hey\s+everyone|dive\s+into|capture\s+the\s+charm|immerse\s+yourself|"
    r"no\s+(?:visible|noticeable)\s+jpe?g\s+artifacts?|"
    r"jpe?g\s+artifacts?\s+(?:are|is)|artist/username\s+if\s+known)\b"
    r")"
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--i1-matched", type=Path,
                    default=ROOT / "curated_vision/i1_matched_25pct.jsonl")
    ap.add_argument("--i1-captions", type=Path, default=ROOT / "i1-captions")
    ap.add_argument("--joy", type=Path,
                    default=ROOT / "curated_vision/joy_matched_cleaned.jsonl")
    ap.add_argument("--grid", type=Path,
                    default=ROOT / "curated_vision/grid_caption_no_titles.jsonl")
    ap.add_argument("--anime", type=Path,
                    default=ROOT / "curated_vision/anime_b_cleaned.jsonl")
    ap.add_argument("--output", type=Path,
                    default=ROOT / "curated_vision/vision_finish_grounded.jsonl")
    ap.add_argument("--eval-output", type=Path,
                    default=ROOT / "curated_vision/vision_finish_grounded_eval.jsonl")
    ap.add_argument("--total", type=int, default=12_000)
    ap.add_argument("--eval-per-source", type=int, default=128)
    ap.add_argument("--seed", type=int, default=20260714)
    return ap.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def image_path(row: dict) -> Path:
    path = Path(str(row["image"]))
    return path if path.is_absolute() else ROOT / path


def image_identity(row: dict) -> str:
    return str(image_path(row).resolve())


def strip_generationisms(text: str) -> str:
    text = GENERATIONISM.sub("", text).strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    text = re.sub(r",(?:\s*,)+", ",", text)
    text = re.sub(
        r"\b(This|That|It|The|A|An)\s*(?:,|\band\b)\s+",
        lambda match: match.group(1) + " ", text)
    return re.sub(r"^\s*[,;:]\s*|\s*[,;:]\s*$", "", text).strip()


def load_matching_captions(directory: Path, wanted: set[str]) -> dict[str, list[str]]:
    matches: dict[str, list[str]] = {}
    options = pc.SetLookupOptions(value_set=pa.array(sorted(wanted), type=pa.string()))
    for path in sorted(directory.glob("*.parquet")):
        parquet = pq.ParquetFile(path)
        available = [name for name in CAPTION_COLUMNS if name in parquet.schema_arrow.names]
        for group in range(parquet.metadata.num_row_groups):
            table = parquet.read_row_group(group, columns=["key", *available])
            table = table.filter(pc.is_in(table["key"], options=options))
            for row in table.to_pylist():
                captions = [strip_generationisms(str(row[name])) for name in available
                            if row.get(name) and str(row[name]).strip()]
                captions = [text for text in captions if text]
                if captions:
                    matches[str(row["key"])] = captions
        if len(matches) == len(wanted):
            break
    return matches


def caption_row(source: dict, text: str, source_name: str) -> dict:
    return {
        "image": source["image"],
        "text": text.strip(),
        "stage1_source": source_name,
        "task": "caption",
        **({"i1_key": source["i1_key"]} if source.get("i1_key") else {}),
    }


def layered_i1_rows(rows: list[dict], captions: dict[str, list[str]], target: int,
                    source_name: str, seed: int) -> list[dict]:
    rng = random.Random(seed)
    sources = [row for row in rows if str(row.get("i1_key")) in captions]
    rng.shuffle(sources)
    per_image: dict[str, list[str]] = {}
    for row in sources:
        choices = list(captions[str(row["i1_key"])])
        random.Random(f"{seed}:{row['i1_key']}").shuffle(choices)
        per_image[str(row["i1_key"])] = choices
    output: list[dict] = []
    layer = 0
    while len(output) < target:
        layer_rows = list(sources)
        random.Random(seed + layer + 1).shuffle(layer_rows)
        added = 0
        for row in layer_rows:
            choices = per_image[str(row["i1_key"])]
            if layer >= len(choices):
                continue
            output.append(caption_row(row, choices[layer], source_name))
            added += 1
            if len(output) == target:
                break
        if added == 0:
            break
        layer += 1
    if len(output) != target:
        raise RuntimeError(f"only built {len(output)}/{target} rows for {source_name}")
    return output


def cleaned_tag_text(row: dict) -> str:
    text = str(row["text"]).strip()
    if text.startswith("Tags:"):
        output = []
        for line in text.splitlines():
            label, separator, value = line.partition(":")
            if not separator or not value.strip():
                continue
            if label.strip().lower() == "tags":
                tags = [part.strip() for part in value.split(";")]
                tags = [part for part in tags
                        if part and not NONVISUAL_GRID_TAG.match(part)]
                if tags:
                    output.append(f"Tags: {'; '.join(tags)}")
            else:
                output.append(f"{label.strip()}: {value.strip()}")
        return "\n".join(output)
    if "," not in text:
        return text
    tags = [part.strip() for part in text.split(",")]
    return ", ".join(part for part in tags
                     if part and not NONVISUAL_TAG_PREFIX.match(part))


def tag_row(row: dict, source_name: str, *, prompt: str = TAG_PROMPT) -> dict:
    return {
        "image": row["image"],
        "text": cleaned_tag_text(row),
        "prompt": prompt,
        "stage1_source": source_name,
        "task": "tags",
    }


def select(rows: list[dict], count: int, seed: int) -> list[dict]:
    values = list(rows)
    random.Random(seed).shuffle(values)
    if len(values) < count:
        raise RuntimeError(f"only {len(values)} candidates for requested {count}")
    return values[:count]


def first_clean_caption(captions: list[str]) -> str:
    for text in captions:
        if text.strip() and not GENERATIONISM.search(text):
            return text.strip()
    raise RuntimeError("caption list contains no generationism-free text")


def main() -> None:
    args = parse_args()
    if args.total < 100 or args.eval_per_source < 1:
        raise SystemExit("--total must be at least 100 and --eval-per-source positive")
    required = (args.i1_matched, args.joy, args.grid, args.anime,
                args.i1_captions / "pexels", args.i1_captions / "midjourneyv6")
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise SystemExit(f"missing inputs: {missing}")

    i1 = read_jsonl(args.i1_matched)
    pexels = [row for row in i1 if row.get("i1_subset") == "pexels"]
    midjourney = [row for row in i1 if row.get("i1_subset") == "midjourneyv6"]
    joy = read_jsonl(args.joy)
    joy_captions = [
        row for row in joy
        if row.get("question_type") == "caption"
        and not GENERATIONISM.search(str(row.get("text", "")))
        and not JOY_CAPTION_SPAM.search(str(row.get("text", "")))
    ]
    joy_tags = [
        row for row in joy
        if row.get("question_type") == "all_tags"
        and len(str(row.get("text", ""))) <= 2_000
        and not GENERATIONISM.search(str(row.get("text", "")))
    ]
    grid = [row for row in read_jsonl(args.grid)
            if not GENERATIONISM.search(str(row.get("text", "")))]
    anime = [row for row in read_jsonl(args.anime)
             if not GENERATIONISM.search(str(row.get("text", "")))]

    print("joining all five i1 caption variants", flush=True)
    pexels_captions = load_matching_captions(
        args.i1_captions / "pexels", {str(row["i1_key"]) for row in pexels})
    midjourney_captions = load_matching_captions(
        args.i1_captions / "midjourneyv6", {str(row["i1_key"]) for row in midjourney})

    rng = random.Random(args.seed)
    pexels_eval = select([row for row in pexels if str(row["i1_key"]) in pexels_captions],
                         args.eval_per_source, args.seed ^ 0x501)
    midjourney_eval = select(
        [row for row in midjourney if str(row["i1_key"]) in midjourney_captions],
        args.eval_per_source, args.seed ^ 0x4D4A)
    joy_eval = select(joy_captions, args.eval_per_source, args.seed ^ 0xA07)
    held_out_images = {image_identity(row)
                       for row in pexels_eval + midjourney_eval + joy_eval}

    eval_rows = [
        caption_row(row, first_clean_caption(pexels_captions[str(row["i1_key"])]),
                    "eval_i1_pexels") for row in pexels_eval
    ] + [
        caption_row(row, first_clean_caption(midjourney_captions[str(row["i1_key"])]),
                    "eval_i1_midjourneyv6") for row in midjourney_eval
    ] + [
        caption_row(row, str(row["text"]), "eval_joy_caption") for row in joy_eval
    ]

    pexels_train = [row for row in pexels if image_identity(row) not in held_out_images]
    midjourney_train = [row for row in midjourney
                        if image_identity(row) not in held_out_images]
    joy_caption_train = [row for row in joy_captions
                         if image_identity(row) not in held_out_images]
    joy_tag_train = [row for row in joy_tags if image_identity(row) not in held_out_images]

    targets = {
        "finish_i1_pexels": round(args.total * 0.63),
        "finish_joy_caption": round(args.total * 0.17),
        "finish_i1_midjourneyv6": round(args.total * 0.15),
    }
    targets["tags"] = args.total - sum(targets.values())
    tag_targets = {
        "finish_tags_joy": round(targets["tags"] * 0.50),
        "finish_tags_grid": round(targets["tags"] * 0.42),
    }
    tag_targets["finish_tags_anime"] = targets["tags"] - sum(tag_targets.values())

    train_rows = layered_i1_rows(
        pexels_train, pexels_captions, targets["finish_i1_pexels"],
        "finish_i1_pexels", args.seed ^ 0x501)
    train_rows += [caption_row(row, str(row["text"]), "finish_joy_caption")
                   for row in select(joy_caption_train, targets["finish_joy_caption"],
                                     args.seed ^ 0xA07)]
    train_rows += layered_i1_rows(
        midjourney_train, midjourney_captions, targets["finish_i1_midjourneyv6"],
        "finish_i1_midjourneyv6", args.seed ^ 0x4D4A)
    train_rows += [tag_row(row, "finish_tags_joy") for row in
                   select(joy_tag_train, tag_targets["finish_tags_joy"], args.seed ^ 0x701)]
    train_rows += [tag_row(row, "finish_tags_grid", prompt=GRID_METADATA_PROMPT) for row in
                   select(grid, tag_targets["finish_tags_grid"], args.seed ^ 0x6A1D)]
    train_rows += [tag_row(row, "finish_tags_anime") for row in
                   select(anime, tag_targets["finish_tags_anime"], args.seed ^ 0xA11)]

    rng.shuffle(train_rows)
    rng.shuffle(eval_rows)
    if len(train_rows) != args.total:
        raise RuntimeError(f"train row count differs: {len(train_rows)} != {args.total}")
    missing_images = [row["image"] for row in train_rows + eval_rows
                      if not image_path(row).is_file()]
    if missing_images:
        raise RuntimeError(f"missing {len(missing_images)} selected images")
    train_images = {image_identity(row) for row in train_rows}
    eval_images = {image_identity(row) for row in eval_rows}
    overlap = train_images & eval_images
    if overlap:
        raise RuntimeError(f"train/eval image leakage: {len(overlap)}")
    contaminated = [row for row in train_rows + eval_rows
                    if GENERATIONISM.search(str(row["text"]))]
    if contaminated:
        raise RuntimeError(f"generationism filter missed {len(contaminated)} rows")
    spam = [row for row in train_rows + eval_rows
            if row["stage1_source"] in {"finish_joy_caption", "eval_joy_caption"}
            and JOY_CAPTION_SPAM.search(str(row["text"]))]
    if spam:
        raise RuntimeError(f"Joy caption spam filter missed {len(spam)} rows")

    write_jsonl(args.output, train_rows)
    write_jsonl(args.eval_output, eval_rows)
    source_counts = Counter(str(row["stage1_source"]) for row in train_rows)
    task_counts = Counter(str(row["task"]) for row in train_rows)
    receipt = {
        "seed": args.seed,
        "rows": len(train_rows),
        "eval_rows": len(eval_rows),
        "unique_train_images": len(train_images),
        "unique_eval_images": len(eval_images),
        "source_counts": dict(sorted(source_counts.items())),
        "source_ratios": {name: count / len(train_rows)
                          for name, count in sorted(source_counts.items())},
        "task_counts": dict(sorted(task_counts.items())),
        "train_eval_image_overlap": 0,
        "raw_civitai_rows": 0,
        "generationism_matches": 0,
        "joy_caption_spam_matches": 0,
        "caption_policy": "grounded prose by default; tag rows use an explicit per-row task prompt",
        "ocr_policy": "deferred until matched TextAtlas/RenderedText image manifests exist",
        "train_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
        "eval_sha256": hashlib.sha256(args.eval_output.read_bytes()).hexdigest(),
    }
    summary = args.output.with_suffix(".summary.json")
    summary.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "eval_output": str(args.eval_output),
                      **receipt}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
