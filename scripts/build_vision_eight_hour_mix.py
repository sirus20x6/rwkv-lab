#!/usr/bin/env python3
"""Assemble a resumable, image-verified eight-hour vision training epoch.

The default 400k-row mixture is 80% real Pexels photographs with i1 prose,
15% recaptioned Midjourney imagery, and 5% cleaned adult-domain Joy/grid
supervision.  Source shards are sampled deterministically, every selected image
is materialized as a normal local file, and the existing grounded evaluation
set is excluded by both local image identity and i1 key.

Selection and extraction receipts live below ``--work-dir``.  A stopped build
can therefore resume at the next source shard instead of rescanning or
rewriting completed shards.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import random
import re
import shutil
import sys
import tarfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from midjourney_alignment import (ALIGNMENT_SCHEMA, SOURCE_CAPTION_COLUMNS,
                                  align_group)

CAPTION_COLUMNS = tuple(f"caption{i}" for i in range(1, 6))
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
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
CAPTION_PREAMBLE_HINT = re.compile(
    r"(?ix)\b(?:description|caption|descriptive\s+(?:paragraph|passage)|"
    r"provided\s+(?:image|crops?)|image\s+provided)\b"
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--total", type=int, default=400_000)
    ap.add_argument("--pexels", type=int, default=320_000)
    ap.add_argument("--midjourney", type=int, default=60_000)
    ap.add_argument("--pexels-shards", type=int, default=16)
    ap.add_argument("--midjourney-shards", type=int, default=7)
    ap.add_argument("--seed", type=int, default=20260715)
    ap.add_argument("--pexels-source", type=Path, default=ROOT /
                    "datasets/i1_full_sources/pexels/train")
    ap.add_argument("--midjourney-source", type=Path, default=ROOT /
                    "datasets/i1_full_sources/midjourneyv6")
    ap.add_argument("--captions", type=Path, default=ROOT / "i1-captions")
    ap.add_argument("--joy", type=Path, default=ROOT /
                    "curated_vision/joy_matched_cleaned.jsonl")
    ap.add_argument("--grid", type=Path, default=ROOT /
                    "curated_vision/grid_caption_no_titles.jsonl")
    ap.add_argument("--eval", type=Path, default=ROOT /
                    "curated_vision/vision_finish_grounded_eval.jsonl")
    ap.add_argument("--image-dir", type=Path, default=ROOT /
                    "datasets/i1_eight_hour_images")
    ap.add_argument("--work-dir", type=Path, default=ROOT /
                    "datasets/i1_eight_hour_work")
    ap.add_argument("--staging-dir", type=Path, default=Path(
                    "/tmp/moe-mla-i1-staging"))
    ap.add_argument("--output", type=Path, default=ROOT /
                    "curated_vision/vision_eight_hour.jsonl")
    ap.add_argument("--eval-output", type=Path, default=ROOT /
                    "curated_vision/vision_eight_hour_eval.jsonl")
    return ap.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def rooted(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def image_identity(row: dict) -> str:
    return str(rooted(str(row["image"])).resolve())


def stable_index(seed: int, key: str, size: int) -> int:
    digest = hashlib.sha256(f"{seed}:{key}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % size


def strip_generationisms(text: str) -> str:
    text = GENERATIONISM.sub("", text).strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    text = re.sub(r",(?:\s*,)+", ",", text)
    text = re.sub(
        r"\b(This|That|It|The|A|An)\s*(?:,|\band\b)\s+",
        lambda match: match.group(1) + " ", text)
    return re.sub(r"^\s*[,;:]\s*|\s*[,;:]\s*$", "", text).strip()


def choose_clean_caption(captions: list[str], key: str, seed: int) -> tuple[str, int] | None:
    if not captions:
        return None
    start = stable_index(seed, key, len(captions))
    for offset in range(len(captions)):
        index = (start + offset) % len(captions)
        text = strip_generationisms(captions[index])
        if text:
            return text, index + 1
    return None


def load_matching_captions(directory: Path, wanted: set[str]) -> dict[str, list[str]]:
    """Read only exact source-key matches, one parquet row group at a time."""
    matches: dict[str, list[str]] = {}
    options = pc.SetLookupOptions(value_set=pa.array(sorted(wanted), type=pa.string()))
    for path in sorted(directory.glob("*.parquet")):
        parquet = pq.ParquetFile(path)
        columns = [name for name in CAPTION_COLUMNS if name in parquet.schema_arrow.names]
        for group in range(parquet.metadata.num_row_groups):
            table = parquet.read_row_group(group, columns=["key", *columns])
            table = table.filter(pc.is_in(table["key"], options=options))
            for row in table.to_pylist():
                values = [str(row[name]).strip() for name in columns
                          if row.get(name) and str(row[name]).strip()]
                if values:
                    matches[str(row["key"])] = values
        print(f"captions {directory.name}: {len(matches):,}/{len(wanted):,} after {path.name}",
              flush=True)
        if len(matches) == len(wanted):
            break
    return matches


def selected_source_files(files: list[Path], count: int, seed: int) -> list[Path]:
    if len(files) < count:
        raise RuntimeError(f"only {len(files)} source shards for requested {count}")
    return sorted(random.Random(seed).sample(files, count))


def pexels_candidates(shards: list[Path], held_out: set[str]) -> list[dict]:
    candidates = []
    seen = set()
    for index, path in enumerate(shards, 1):
        count = 0
        with tarfile.open(path) as archive:
            for member in archive:
                if not member.isfile() or Path(member.name).suffix.lower() not in IMAGE_SUFFIXES:
                    continue
                key = Path(member.name).stem
                if key in held_out or key in seen:
                    continue
                seen.add(key)
                candidates.append({"i1_key": key, "source_file": str(path),
                                   "member": member.name})
                count += 1
        print(f"indexed Pexels shard {index}/{len(shards)} {path.name}: {count:,}",
              flush=True)
    return candidates


def midjourney_candidates(shards: list[Path], held_out: set[str],
                          caption_dir: Path) -> list[dict]:
    """Index physical rows using i1's semantic suffix, not duplicate-row order."""
    source_groups: list[tuple[Path, str, list[tuple[int, list[str]]]]] = []
    wanted: set[str] = set()
    for index, path in enumerate(shards, 1):
        parquet = pq.ParquetFile(path)
        columns = ["id", *SOURCE_CAPTION_COLUMNS]
        grouped: dict[str, list[tuple[int, list[str]]]] = defaultdict(list)
        row_index = 0
        for batch in parquet.iter_batches(columns=columns, batch_size=16_384):
            for row in batch.to_pylist():
                numeric_id = str(int(row["id"]))
                grouped[numeric_id].append((
                    row_index,
                    [str(row.get(name) or "") for name in SOURCE_CAPTION_COLUMNS],
                ))
                row_index += 1
        malformed = [numeric_id for numeric_id, rows in grouped.items() if len(rows) != 4]
        if malformed:
            raise RuntimeError(
                f"{path} has {len(malformed)} Midjourney IDs without four source rows")
        for numeric_id, rows in grouped.items():
            source_groups.append((path, numeric_id, rows))
            wanted.update(f"{numeric_id}_{suffix}" for suffix in range(4))
        print(f"indexed Midjourney shard {index}/{len(shards)} {path.name}: "
              f"{row_index:,}", flush=True)

    captions = load_matching_captions(caption_dir, wanted)
    missing = sorted(wanted - captions.keys())
    if missing:
        print(f"Midjourney: {len(missing)} i1 suffixes have no caption row; "
              "they will be inferred for alignment and excluded", flush=True)
    candidates = []
    margins = []
    for path, numeric_id, rows in source_groups:
        aligned = align_group(
            [source_captions for _, source_captions in rows],
            [captions.get(f"{numeric_id}_{suffix}", []) for suffix in range(4)],
        )
        margins.append(aligned.margin)
        for source_offset, (row_index, _) in enumerate(rows):
            key = f"{numeric_id}_{aligned.row_to_suffix[source_offset]}"
            if key in captions and key not in held_out:
                candidates.append({
                    "i1_key": key,
                    "source_file": str(path),
                    "row_index": row_index,
                    "alignment_schema": ALIGNMENT_SCHEMA,
                    "alignment_margin": aligned.margin,
                })
    print(f"aligned Midjourney groups: {len(source_groups):,}; "
          f"minimum assignment margin: {min(margins):.6f}", flush=True)
    return candidates


def selection_contract(args: argparse.Namespace, source: str,
                       shards: list[Path], target: int) -> dict:
    return {
        "schema": 2 if source == "midjourneyv6" else 1,
        "source": source, "seed": args.seed, "target": target,
        "shards": [str(path.resolve()) for path in shards],
        **({"alignment_schema": ALIGNMENT_SCHEMA} if source == "midjourneyv6" else {}),
    }


def build_selection(args: argparse.Namespace, source: str, target: int,
                    shards: list[Path], candidates: list[dict],
                    held_out: set[str]) -> list[dict]:
    selection_path = args.work_dir / f"{source}_selection.jsonl"
    contract_path = args.work_dir / f"{source}_selection.json"
    contract = selection_contract(args, source, shards, target)
    if selection_path.is_file() and contract_path.is_file():
        if json.loads(contract_path.read_text()) != contract:
            raise RuntimeError(f"stale selection contract at {contract_path}")
        rows = read_jsonl(selection_path)
        if len(rows) != target:
            raise RuntimeError(f"incomplete selection at {selection_path}")
        print(f"reusing {source} selection: {len(rows):,}", flush=True)
        return rows

    random.Random(args.seed ^ (0x501 if source == "pexels" else 0x4D4A)).shuffle(candidates)
    reserve = min(len(candidates), target + max(2_000, target // 50))
    candidates = candidates[:reserve]
    wanted = {row["i1_key"] for row in candidates} - held_out
    captions = load_matching_captions(args.captions / source, wanted)
    selected = []
    for row in candidates:
        choice = choose_clean_caption(captions.get(row["i1_key"], []),
                                      row["i1_key"], args.seed)
        if choice is None:
            continue
        text, variant = choice
        selected.append({**row, "text": text, "caption_variant": variant})
        if len(selected) == target:
            break
    if len(selected) != target:
        raise RuntimeError(f"only selected {len(selected):,}/{target:,} clean {source} rows")
    write_jsonl(selection_path, selected)
    write_json(contract_path, contract)
    print(f"wrote {source} selection: {len(selected):,}", flush=True)
    return selected


def image_header(payload: bytes) -> tuple[int, int, str]:
    with Image.open(io.BytesIO(payload)) as image:
        width, height = image.size
        fmt = (image.format or "").lower()
    suffix = {"jpeg": ".jpg", "jpg": ".jpg", "png": ".png",
              "webp": ".webp"}.get(fmt)
    if width < 1 or height < 1 or suffix is None:
        raise ValueError("unsupported or empty image")
    return width, height, suffix


def materialize_image(payload: bytes, image_dir: Path, source: str,
                      key: str) -> tuple[str, str, int, int]:
    width, height, suffix = image_header(payload)
    digest = hashlib.sha256(payload).hexdigest()
    bucket = hashlib.sha256(key.encode()).hexdigest()[:2]
    path = image_dir / source / bucket / f"{key}{suffix}"
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.is_file() or path.stat().st_size != len(payload):
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_bytes(payload)
        temporary.replace(path)
    try:
        manifest_path = str(path.relative_to(ROOT))
    except ValueError:
        manifest_path = str(path.resolve())
    return manifest_path, digest, width, height


def part_path(work_dir: Path, source: str, source_file: Path) -> Path:
    return work_dir / "parts" / source / f"{source_file.stem}.jsonl"


def staged_source(path: Path, staging_dir: Path) -> Path:
    """Copy one HDD source shard sequentially into fast temporary storage."""
    staging_dir.mkdir(parents=True, exist_ok=True)
    prefix = hashlib.sha256(str(path.resolve()).encode()).hexdigest()[:12]
    target = staging_dir / f"{prefix}-{path.name}"
    expected = path.stat().st_size
    if target.is_file() and target.stat().st_size == expected:
        return target
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.unlink(missing_ok=True)
    with path.open("rb", buffering=16 * 1024 * 1024) as source:
        with temporary.open("wb", buffering=16 * 1024 * 1024) as destination:
            shutil.copyfileobj(source, destination, length=16 * 1024 * 1024)
    if temporary.stat().st_size != expected:
        raise RuntimeError(f"short staged copy of {path}")
    temporary.replace(target)
    return target


def valid_part(path: Path, expected: int) -> list[dict] | None:
    if not path.is_file():
        return None
    rows = read_jsonl(path)
    if len(rows) != expected or any(not rooted(row["image"]).is_file() for row in rows):
        return None
    return rows


def extract_pexels(args: argparse.Namespace, selection: list[dict]) -> list[dict]:
    grouped: dict[Path, list[dict]] = defaultdict(list)
    for row in selection:
        grouped[Path(row["source_file"])].append(row)
    output = []
    for index, source_file in enumerate(sorted(grouped), 1):
        chosen = grouped[source_file]
        receipt = part_path(args.work_dir, "pexels", source_file)
        rows = valid_part(receipt, len(chosen))
        if rows is None:
            local_source = staged_source(source_file, args.staging_dir)
            by_member = {row["member"]: row for row in chosen}
            rows = []
            with tarfile.open(local_source) as archive:
                for member in archive:
                    if not member.isfile():
                        continue
                    handle = archive.extractfile(member)
                    if handle is None:
                        continue
                    # The selection covers roughly 80% of each chosen tar and
                    # every image is followed by a small JSON sidecar. Reading
                    # every regular member keeps the HDD pool sequential;
                    # seeking past either kind is dramatically slower on
                    # RAID-Z despite transferring fewer logical bytes.
                    payload = handle.read()
                    if Path(member.name).suffix.lower() not in IMAGE_SUFFIXES:
                        continue
                    selected = by_member.get(member.name)
                    if selected is None:
                        continue
                    image, digest, width, height = materialize_image(
                        payload, args.image_dir, "pexels", selected["i1_key"])
                    rows.append({
                        "image": image, "text": selected["text"],
                        "stage1_source": "eight_hour_i1_pexels", "task": "caption",
                        "i1_subset": "pexels", "i1_key": selected["i1_key"],
                        "caption_variant": selected["caption_variant"],
                        "image_sha256": digest, "width": width, "height": height,
                    })
            if len(rows) != len(chosen):
                raise RuntimeError(f"extracted {len(rows)}/{len(chosen)} from {source_file}")
            write_jsonl(receipt, rows)
            local_source.unlink(missing_ok=True)
        output.extend(rows)
        print(f"materialized Pexels {index}/{len(grouped)}: {len(output):,}", flush=True)
    return output


def extract_midjourney(args: argparse.Namespace, selection: list[dict]) -> list[dict]:
    grouped: dict[Path, list[dict]] = defaultdict(list)
    for row in selection:
        grouped[Path(row["source_file"])].append(row)
    output = []
    for index, source_file in enumerate(sorted(grouped), 1):
        chosen = grouped[source_file]
        receipt = part_path(args.work_dir, "midjourneyv6", source_file)
        rows = valid_part(receipt, len(chosen))
        if rows is None:
            local_source = staged_source(source_file, args.staging_dir)
            by_index = {int(row["row_index"]): row for row in chosen}
            rows = []
            offset = 0
            parquet = pq.ParquetFile(local_source)
            for batch in parquet.iter_batches(columns=["image"], batch_size=1_024):
                images = batch.column(0).to_pylist()
                for local, value in enumerate(images):
                    selected = by_index.get(offset + local)
                    if selected is None or not value or not value.get("bytes"):
                        continue
                    image, digest, width, height = materialize_image(
                        value["bytes"], args.image_dir, "midjourneyv6",
                        selected["i1_key"])
                    rows.append({
                        "image": image, "text": selected["text"],
                        "stage1_source": "eight_hour_i1_midjourneyv6",
                        "task": "caption", "i1_subset": "midjourneyv6",
                        "i1_key": selected["i1_key"],
                        "caption_variant": selected["caption_variant"],
                        "image_sha256": digest, "width": width, "height": height,
                    })
                offset += len(images)
            if len(rows) != len(chosen):
                raise RuntimeError(f"extracted {len(rows)}/{len(chosen)} from {source_file}")
            write_jsonl(receipt, rows)
            local_source.unlink(missing_ok=True)
        output.extend(rows)
        print(f"materialized Midjourney {index}/{len(grouped)}: {len(output):,}",
              flush=True)
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
                tags = [part for part in tags if part and not NONVISUAL_GRID_TAG.match(part)]
                if tags:
                    output.append(f"Tags: {';'.join(tags)}")
            else:
                output.append(f"{label.strip()}: {value.strip()}")
        return "\n".join(output)
    tags = [part.strip() for part in text.split(",")]
    return ", ".join(part for part in tags
                     if part and not NONVISUAL_TAG_PREFIX.match(part))


def cleaned_caption_text(value: str) -> str:
    """Remove teacher-facing response wrappers without flattening real prose."""
    text = str(value).strip()
    text = re.sub(r"(?is)^(?:answer|response)\s*:\s*", "", text)
    lines = text.splitlines()
    while lines and not lines[0].strip():
        lines.pop(0)
    if (lines and lines[0].strip().endswith(":")
            and len(lines[0]) <= 240
            and CAPTION_PREAMBLE_HINT.search(lines[0])):
        lines.pop(0)
        while lines and not lines[0].strip():
            lines.pop(0)
    text = "\n".join(lines).strip()
    # Some i1 variants append a redundant markdown inventory after an already
    # complete prose caption. It teaches formatting habits, not visual facts.
    text = re.split(r"\n\s*```(?:markdown)?\s*\n", text, maxsplit=1,
                    flags=re.IGNORECASE)[0].strip()
    return text


def specialty_rows(args: argparse.Namespace, target: int,
                   held_out_images: set[str]) -> list[dict]:
    joy = read_jsonl(args.joy)
    grid = read_jsonl(args.grid)
    rng = random.Random(args.seed ^ 0x5EEC)

    joy_caption = [row for row in joy
                   if row.get("question_type") == "caption"
                   and image_identity(row) not in held_out_images
                   and not GENERATIONISM.search(str(row.get("text", "")))
                   and not JOY_CAPTION_SPAM.search(str(row.get("text", "")))]
    joy_tags = [row for row in joy
                if row.get("question_type") == "all_tags"
                and image_identity(row) not in held_out_images
                and len(str(row.get("text", ""))) <= 2_000
                and not GENERATIONISM.search(str(row.get("text", "")))]
    rng.shuffle(joy_caption)
    rng.shuffle(joy_tags)
    used_images = set()
    rows = []
    for row in joy_caption:
        identity = image_identity(row)
        if identity in used_images:
            continue
        used_images.add(identity)
        rows.append({"image": row["image"], "text": str(row["text"]).strip(),
                     "stage1_source": "eight_hour_joy_caption", "task": "caption"})
    for row in joy_tags:
        identity = image_identity(row)
        if identity in used_images:
            continue
        used_images.add(identity)
        rows.append({"image": row["image"], "text": cleaned_tag_text(row),
                     "prompt": TAG_PROMPT,
                     "stage1_source": "eight_hour_tags_joy", "task": "tags"})

    grid = [row for row in grid
            if image_identity(row) not in held_out_images
            and not GENERATIONISM.search(str(row.get("text", "")))]
    rng.shuffle(grid)
    for row in grid:
        if len(rows) == target:
            break
        identity = image_identity(row)
        if identity in used_images:
            continue
        text = cleaned_tag_text(row)
        if not text:
            continue
        used_images.add(identity)
        rows.append({"image": row["image"], "text": text,
                     "prompt": GRID_METADATA_PROMPT,
                     "stage1_source": "eight_hour_tags_grid", "task": "tags"})
    if len(rows) != target:
        raise RuntimeError(f"only built {len(rows):,}/{target:,} specialty rows")
    missing = [row["image"] for row in rows if not rooted(row["image"]).is_file()]
    if missing:
        raise RuntimeError(f"specialty selection has {len(missing)} missing images")
    return rows


def main() -> None:
    args = parse_args()
    specialty_target = args.total - args.pexels - args.midjourney
    if min(args.total, args.pexels, args.midjourney, specialty_target) < 1:
        raise SystemExit("all source targets must be positive and sum to --total")
    required = [args.pexels_source, args.midjourney_source,
                args.captions / "pexels", args.captions / "midjourneyv6",
                args.joy, args.grid, args.eval]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise SystemExit(f"missing required inputs: {missing}")
    args.work_dir.mkdir(parents=True, exist_ok=True)

    eval_rows = read_jsonl(args.eval)
    held_out_images = {image_identity(row) for row in eval_rows}
    held_out = defaultdict(set)
    for row in eval_rows:
        if row.get("i1_key") and row.get("stage1_source") == "eval_i1_pexels":
            held_out["pexels"].add(str(row["i1_key"]))
        if row.get("i1_key") and row.get("stage1_source") == "eval_i1_midjourneyv6":
            held_out["midjourneyv6"].add(str(row["i1_key"]))

    pexels_files = selected_source_files(
        sorted(args.pexels_source.glob("*.tar")), args.pexels_shards,
        args.seed ^ 0x501)
    midjourney_files = selected_source_files(
        sorted(args.midjourney_source.glob("*.parquet")), args.midjourney_shards,
        args.seed ^ 0x4D4A)

    pexels_selection_path = args.work_dir / "pexels_selection.jsonl"
    pexels_contract_path = args.work_dir / "pexels_selection.json"
    if pexels_selection_path.is_file() and pexels_contract_path.is_file():
        pexels_candidates_rows = []
    else:
        pexels_candidates_rows = pexels_candidates(pexels_files, held_out["pexels"])
    pexels_selection = build_selection(
        args, "pexels", args.pexels, pexels_files,
        pexels_candidates_rows, held_out["pexels"])

    midjourney_selection_path = args.work_dir / "midjourneyv6_selection.jsonl"
    midjourney_contract_path = args.work_dir / "midjourneyv6_selection.json"
    if (midjourney_selection_path.is_file()
            and midjourney_contract_path.is_file()):
        midjourney_candidates_rows = []
    else:
        midjourney_candidates_rows = midjourney_candidates(
            midjourney_files, held_out["midjourneyv6"],
            args.captions / "midjourneyv6")
    midjourney_selection = build_selection(
        args, "midjourneyv6", args.midjourney, midjourney_files,
        midjourney_candidates_rows, held_out["midjourneyv6"])

    pexels = extract_pexels(args, pexels_selection)
    midjourney = extract_midjourney(args, midjourney_selection)
    source_rows = []
    seen_source_hashes = set()
    source_content_duplicates_removed = 0
    for row in pexels + midjourney:
        digest = str(row["image_sha256"])
        if digest in seen_source_hashes:
            source_content_duplicates_removed += 1
            continue
        seen_source_hashes.add(digest)
        source_rows.append(row)
    specialty = specialty_rows(
        args, specialty_target + source_content_duplicates_removed,
        held_out_images)
    rows = source_rows + specialty
    caption_preambles_stripped = 0
    for row in rows:
        if row["task"] != "caption":
            continue
        cleaned = cleaned_caption_text(row["text"])
        if not cleaned:
            raise RuntimeError("caption cleaning produced an empty target")
        caption_preambles_stripped += int(cleaned != row["text"])
        row["text"] = cleaned
    random.Random(args.seed).shuffle(rows)

    if len(rows) != args.total:
        raise RuntimeError(f"row count differs: {len(rows):,} != {args.total:,}")
    train_images = {image_identity(row) for row in rows}
    if len(train_images) != len(rows):
        raise RuntimeError(f"train images are not unique: {len(train_images):,}/{len(rows):,}")
    overlap = train_images & held_out_images
    if overlap:
        raise RuntimeError(f"train/eval image leakage: {len(overlap)}")
    train_keys = defaultdict(set)
    for row in rows:
        if row.get("i1_key"):
            train_keys[str(row.get("i1_subset"))].add(str(row["i1_key"]))
    key_overlap = ((train_keys["pexels"] & held_out["pexels"])
                   | (train_keys["midjourneyv6"] & held_out["midjourneyv6"]))
    if key_overlap:
        raise RuntimeError(f"train/eval i1-key leakage: {len(key_overlap)}")
    contaminated = [row for row in rows + eval_rows
                    if GENERATIONISM.search(str(row["text"]))]
    if contaminated:
        raise RuntimeError(f"generationism filter missed {len(contaminated)} rows")
    blank_metadata = [row for row in rows
                      if row["task"] == "tags"
                      and re.search(r"(?m)^[^:\n]+:\s*$", row["text"])]
    if blank_metadata:
        raise RuntimeError(f"blank metadata fields remain in {len(blank_metadata)} rows")
    missing_images = [row["image"] for row in rows + eval_rows
                      if not rooted(row["image"]).is_file()]
    if missing_images:
        raise RuntimeError(f"manifest has {len(missing_images)} missing images")

    write_jsonl(args.output, rows)
    write_jsonl(args.eval_output, eval_rows)
    source_counts = Counter(str(row["stage1_source"]) for row in rows)
    task_counts = Counter(str(row["task"]) for row in rows)
    receipt = {
        "schema": 1, "seed": args.seed, "rows": len(rows),
        "eval_rows": len(eval_rows), "unique_train_images": len(train_images),
        "unique_eval_images": len(held_out_images),
        "source_counts": dict(sorted(source_counts.items())),
        "source_ratios": {name: count / len(rows)
                          for name, count in sorted(source_counts.items())},
        "task_counts": dict(sorted(task_counts.items())),
        "train_eval_image_overlap": 0, "train_eval_i1_key_overlap": 0,
        "raw_civitai_rows": 0, "generationism_matches": 0,
        "joy_caption_spam_matches": 0, "blank_metadata_fields": 0,
        "caption_preambles_stripped": caption_preambles_stripped,
        "source_content_duplicates_removed": source_content_duplicates_removed,
        "unique_source_content_hashes": len(seen_source_hashes),
        "caption_policy": "one deterministic clean i1 caption per unique source image; explicit prompts for tag tasks",
        "pexels_source_shards": [path.name for path in pexels_files],
        "midjourney_source_shards": [path.name for path in midjourney_files],
        "train_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
        "eval_sha256": hashlib.sha256(args.eval_output.read_bytes()).hexdigest(),
    }
    write_json(args.output.with_suffix(".summary.json"), receipt)
    print(json.dumps({"output": str(args.output), "eval_output": str(args.eval_output),
                      **receipt}, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
