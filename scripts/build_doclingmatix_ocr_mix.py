#!/usr/bin/env python3
"""Build a bounded, image-disjoint DoclingMatix OCR supplement.

The base vision shard stays immutable. This builder materializes a deterministic
OCR-only train/eval supplement plus combined manifests. Structured DocTags are
converted to plain reading-order text. Their locations also produce enlarged
text-region crops, retaining spatial supervision without teaching
dataset-specific coordinate syntax.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import io
import json
import math
import os
import random
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Sequence

import pyarrow.parquet as pq
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from rwkv_lab.generate import WorldVocab

DEFAULT_SOURCE = ROOT / "datasets/doclingmatix/default/train"
DEFAULT_IMAGES = ROOT / "datasets/doclingmatix_ocr_images"
DEFAULT_BASE_TRAIN = ROOT / "curated_vision/vision_next_shard_000_train.jsonl"
DEFAULT_BASE_EVAL = ROOT / "curated_vision/vision_eight_hour_eval.jsonl"
DEFAULT_TRAIN = ROOT / "curated_vision/vision_next_ocr10_train.jsonl"
DEFAULT_EVAL = ROOT / "curated_vision/vision_next_ocr10_eval.jsonl"
DEFAULT_COMBINED_TRAIN = (
    ROOT / "curated_vision/vision_next_shard_000_ocr10_train.jsonl")
DEFAULT_COMBINED_EVAL = (
    ROOT / "curated_vision/vision_next_shard_000_ocr10_eval.jsonl")
DEFAULT_RECEIPT = (
    ROOT / "curated_vision/vision_next_shard_000_ocr10.summary.json")
OCR_PROMPT = (
    "Transcribe all visible text in this document image. "
    "Preserve reading order and line breaks:\n")
CONVERSION_PROMPT = re.compile(
    r"^\s*convert\s+(?:this\s+)?page\s+to\s+docling[.!]?\s*$",
    re.IGNORECASE)
LOCATION_TAG = re.compile(r"<loc_\d+>")
LOCATED_ELEMENT = re.compile(
    r"<(?P<tag>[a-zA-Z0-9_]+)>"
    r"<loc_(?P<x1>\d+)><loc_(?P<y1>\d+)>"
    r"<loc_(?P<x2>\d+)><loc_(?P<y2>\d+)>"
    r"(?P<body>.*?)</(?P=tag)>",
    re.DOTALL,
)
ANY_TAG = re.compile(r"</?[^>]+>")
GLYPH_TOKEN = re.compile(r"GLYPH\([^)]*\)", re.IGNORECASE)
SPACE = re.compile(r"[ \t\f\v]+")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    ap.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGES)
    ap.add_argument("--base-train", type=Path, default=DEFAULT_BASE_TRAIN)
    ap.add_argument("--base-eval", type=Path, default=DEFAULT_BASE_EVAL)
    ap.add_argument("--train-output", type=Path, default=DEFAULT_TRAIN)
    ap.add_argument("--eval-output", type=Path, default=DEFAULT_EVAL)
    ap.add_argument("--combined-train", type=Path,
                    default=DEFAULT_COMBINED_TRAIN)
    ap.add_argument("--combined-eval", type=Path,
                    default=DEFAULT_COMBINED_EVAL)
    ap.add_argument("--receipt", type=Path, default=DEFAULT_RECEIPT)
    ap.add_argument("--ocr-ratio", type=float, default=0.10,
                    help="OCR share of the final combined training manifest")
    ap.add_argument("--target-train-rows", type=int, default=0,
                    help="exact OCR train rows; 0 derives it from --ocr-ratio")
    ap.add_argument("--eval-rows", type=int, default=64)
    ap.add_argument("--candidate-shards", type=int, default=18,
                    help="leading pinned shards used for the bounded supplement")
    ap.add_argument("--reserve-rows", type=int, default=2048,
                    help="extra parent pages materialized in the first tranche "
                         "beyond the optimistic estimate; a shortfall extends "
                         "the tranche at the measured yield instead of failing")
    ap.add_argument("--min-chars", type=int, default=160)
    ap.add_argument("--max-chars", type=int, default=2400)
    ap.add_argument("--max-tokens", type=int, default=700,
                    help="maximum complete transcription length in RWKV tokens")
    ap.add_argument("--crops-per-page", type=int, default=2,
                    help="maximum located text crops retained per full page")
    ap.add_argument("--crop-min-chars", type=int, default=4)
    ap.add_argument("--crop-max-chars", type=int, default=600)
    ap.add_argument("--crop-max-tokens", type=int, default=128)
    ap.add_argument("--crop-target-height", type=int, default=128,
                    help="located text is enlarged toward this height so glyphs "
                         "stay visible to the encoder; a crop is never shrunk "
                         "below its own size except to honour --crop-max-edge")
    ap.add_argument("--crop-max-edge", type=int, default=1536)
    ap.add_argument("--crop-padding", type=float, default=0.08,
                    help="fractional padding around each Docling 0..500 box")
    ap.add_argument("--vocab", type=Path, default=Path(os.environ.get(
        "VOCAB", "/workspace/git/ztok/bench/vocabs/rwkv_vocab_v20230424.txt")))
    ap.add_argument("--min-side", type=int, default=512)
    ap.add_argument("--seed", type=int, default=20260716)
    args = ap.parse_args()
    if not 0 < args.ocr_ratio < 1:
        ap.error("--ocr-ratio must be between zero and one")
    if min(args.eval_rows, args.candidate_shards, args.reserve_rows,
           args.min_chars, args.max_chars, args.max_tokens, args.min_side,
           args.crop_min_chars, args.crop_max_chars, args.crop_max_tokens,
           args.crop_target_height, args.crop_max_edge) < 1:
        ap.error("row, shard, character, and size limits must be positive")
    if args.min_chars > args.max_chars:
        ap.error("--min-chars cannot exceed --max-chars")
    if args.crop_min_chars > args.crop_max_chars:
        ap.error("--crop-min-chars cannot exceed --crop-max-chars")
    if args.target_train_rows < 0 or args.crops_per_page < 0:
        ap.error("target rows and crops per page must be non-negative")
    if not 0 <= args.crop_padding <= 1:
        ap.error("--crop-padding must be in [0, 1]")
    return args


def read_jsonl(path: Path) -> list[dict]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def atomic_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(16 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def target_ocr_rows(base_rows: int, ratio: float) -> int:
    """Smallest integer whose final share is at least ``ratio``."""
    if base_rows < 1 or not 0 < ratio < 1:
        raise ValueError("base_rows must be positive and ratio in (0, 1)")
    return math.ceil(base_rows * ratio / (1.0 - ratio))


def doctag_to_text(value: str) -> str:
    """Remove Docling markup while retaining compact reading-order text."""
    value = html.unescape(str(value))
    value = LOCATION_TAG.sub("", value)
    value = re.sub(r"</[^>]+>", "\n", value)
    value = re.sub(r"<[^>]+>", "\n", value)
    value = GLYPH_TOKEN.sub("", value)
    output: list[str] = []
    for raw_line in value.replace("\r", "\n").splitlines():
        line = SPACE.sub(" ", raw_line).strip()
        if not line or (output and line == output[-1]):
            continue
        output.append(line)
    return "\n".join(output).strip()


def located_text_blocks(value: str, *, min_chars: int,
                        max_chars: int) -> list[dict]:
    """Extract text-bearing Docling elements and their normalized page boxes."""
    blocks = []
    seen = set()
    for match in LOCATED_ELEMENT.finditer(html.unescape(str(value))):
        coordinates = tuple(int(match.group(name))
                            for name in ("x1", "y1", "x2", "y2"))
        if (any(coordinate > 500 for coordinate in coordinates)
                or coordinates[2] <= coordinates[0]
                or coordinates[3] <= coordinates[1]):
            continue
        text = doctag_to_text(match.group("body"))
        if not min_chars <= len(text) <= max_chars:
            continue
        visible = sum(character.isalnum() for character in text)
        if visible / max(1, len(text)) < 0.35:
            continue
        identity = (coordinates, text)
        if identity in seen:
            continue
        seen.add(identity)
        blocks.append({
            "kind": match.group("tag"), "box_500": coordinates, "text": text,
        })
    return blocks


def conversion_payload(texts: Sequence[dict], *, min_chars: int,
                       max_chars: int, crop_min_chars: int = 4,
                       crop_max_chars: int = 600) -> tuple[str, list[dict]] | None:
    for turn in reversed(texts):
        if not isinstance(turn, dict):
            continue
        user = str(turn.get("user") or "")
        assistant = str(turn.get("assistant") or "")
        if not CONVERSION_PROMPT.match(user) or not assistant.startswith("<doctag>"):
            continue
        if len(GLYPH_TOKEN.findall(assistant)) > 2:
            return None
        text = doctag_to_text(assistant)
        if not min_chars <= len(text) <= max_chars:
            return None
        visible = sum(character.isalnum() for character in text)
        if visible / max(len(text), 1) < 0.35:
            return None
        return text, located_text_blocks(
            assistant, min_chars=crop_min_chars, max_chars=crop_max_chars)
    return None


def conversion_target(texts: Sequence[dict], *, min_chars: int,
                      max_chars: int) -> str | None:
    payload = conversion_payload(
        texts, min_chars=min_chars, max_chars=max_chars)
    return None if payload is None else payload[0]


def stable_rank(seed: int, key: str) -> bytes:
    return hashlib.sha256(f"{seed}:{key}".encode()).digest()


def collect_candidates(paths: Sequence[Path], *, min_chars: int,
                       max_chars: int, crop_min_chars: int,
                       crop_max_chars: int, seed: int) -> list[dict]:
    candidates = []
    for shard_number, path in enumerate(paths, 1):
        parquet = pq.ParquetFile(path)
        row_offset = 0
        accepted = 0
        for group in range(parquet.metadata.num_row_groups):
            table = parquet.read_row_group(group, columns=["texts"])
            for local, texts in enumerate(table["texts"].to_pylist()):
                row_index = row_offset + local
                payload = conversion_payload(
                    texts or [], min_chars=min_chars, max_chars=max_chars,
                    crop_min_chars=crop_min_chars,
                    crop_max_chars=crop_max_chars)
                if payload is None:
                    continue
                text, blocks = payload
                key = f"{path.name}:{row_index}"
                candidates.append({
                    "key": key, "source_file": str(path.resolve()),
                    "row_index": row_index, "row_group": group,
                    "row_group_offset": local, "text": text, "blocks": blocks,
                    "rank": stable_rank(seed, key),
                })
                accepted += 1
            row_offset += len(table)
        print({"kind": "docling_ocr_candidates", "shard": shard_number,
               "total_shards": len(paths), "accepted": accepted,
               "candidates": len(candidates)}, flush=True)
    candidates.sort(key=lambda row: row["rank"])
    return candidates


def image_payload(value: object) -> bytes:
    if not isinstance(value, list) or len(value) != 1:
        raise ValueError("expected exactly one embedded image")
    image = value[0]
    if not isinstance(image, dict) or not image.get("bytes"):
        raise ValueError("embedded image bytes are missing")
    return bytes(image["bytes"])


def crop_scale(width: int, height: int, *, target_height: int,
               max_edge: int) -> float:
    """Resize factor for one located crop: enlarge for glyphs, cap for budget.

    Enlarge toward ``target_height``, never shrink for it: a 1600x40 line and a
    900x400 paragraph both already clear a 128px target, and a bare
    ``min(target_height / height, ...)`` scaled them DOWN (0.96x and 0.32x),
    destroying the glyphs the crop exists to show. ``max_edge`` is a hard
    encoder budget, so it still binds in both directions -- it shrinks an
    oversized crop and can cap an upscale short of the target height.
    """
    if width < 1 or height < 1 or target_height < 1 or max_edge < 1:
        raise ValueError("crop geometry and limits must be positive")
    return min(max(1.0, target_height / height), max_edge / max(width, height))


def materialize_candidates(candidates: Sequence[dict], image_dir: Path, *,
                           min_side: int, crops_per_page: int,
                           crop_target_height: int,
                           crop_max_edge: int, crop_padding: float,
                           seed: int,
                           created_crops: set[Path] | None = None) -> list[dict]:
    """Write page images and located crops for one candidate slice.

    ``created_crops`` accumulates every crop file this process actually wrote,
    so the caller can reclaim the ones that later filtering rejects.

    The --min-side floor is deliberately a page-only guard. Crops are enlarged
    toward --crop-target-height, and radio_v4h.native_image_size clamps
    whatever still arrives small with ``max(axis_step, min(max_edge, snapped))``
    -- at least one 16px patch row and one 32px patch column -- so a short crop
    is padded up by the encoder instead of yielding an empty grid. No separate
    crop floor is needed.
    """
    created_crops = set() if created_crops is None else created_crops
    grouped: dict[tuple[Path, int], list[dict]] = defaultdict(list)
    for candidate in candidates:
        grouped[(Path(candidate["source_file"]),
                 int(candidate["row_group"]))].append(candidate)

    materialized: dict[str, list[dict]] = {}
    for completed, ((source, group), selected) in enumerate(
            sorted(grouped.items(), key=lambda item: (str(item[0][0]), item[0][1])), 1):
        table = pq.ParquetFile(source).read_row_group(group, columns=["images"])
        images = table["images"].to_pylist()
        for candidate in selected:
            try:
                payload = image_payload(
                    images[int(candidate["row_group_offset"])])
            except (IndexError, TypeError, ValueError):
                # DoclingMatix declares a list-of-images schema. A minority of
                # rows contain no image or multiple pages; this bounded shard
                # deliberately keeps the one-page/one-target contract.
                continue
            digest = hashlib.sha256(payload).hexdigest()
            try:
                with Image.open(io.BytesIO(payload)) as image:
                    image.load()
                    width, height = image.size
                    fmt = str(image.format or "").lower()
                    page_image = image.convert("RGB")
            except (OSError, ValueError):
                continue
            suffix = {"jpeg": ".jpg", "jpg": ".jpg", "png": ".png",
                      "webp": ".webp"}.get(fmt)
            if suffix is None or min(width, height) < min_side:
                continue
            target = image_dir / digest[:2] / f"{digest}{suffix}"
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.is_file() or target.stat().st_size != len(payload):
                temporary = target.with_suffix(target.suffix + ".tmp")
                temporary.write_bytes(payload)
                temporary.replace(target)
            try:
                image_path = str(target.relative_to(ROOT))
            except ValueError:
                image_path = str(target.resolve())
            full_row = {
                "image": image_path, "text": candidate["text"],
                "prompt": OCR_PROMPT, "source": "doclingmatix_ocr",
                "stage1_source": "doclingmatix_ocr_transcription",
                "task": "ocr", "docling_key": candidate["key"],
                "docling_parent_key": candidate["key"],
                "ocr_granularity": "page",
                "image_sha256": digest, "width": width, "height": height,
            }
            output_rows = [full_row]
            blocks = sorted(
                candidate["blocks"],
                key=lambda block: stable_rank(
                    seed, f"{candidate['key']}:{block['box_500']}:{block['text']}"),
            )[:crops_per_page]
            for block_index, block in enumerate(blocks):
                x1, y1, x2, y2 = block["box_500"]
                left, top = x1 * width / 500, y1 * height / 500
                right, bottom = x2 * width / 500, y2 * height / 500
                pad_x = max(2.0, (right - left) * crop_padding)
                pad_y = max(2.0, (bottom - top) * crop_padding)
                crop_box = (
                    max(0, math.floor(left - pad_x)),
                    max(0, math.floor(top - pad_y)),
                    min(width, math.ceil(right + pad_x)),
                    min(height, math.ceil(bottom + pad_y)),
                )
                if crop_box[2] <= crop_box[0] or crop_box[3] <= crop_box[1]:
                    continue
                crop = page_image.crop(crop_box)
                scale = crop_scale(
                    crop.width, crop.height,
                    target_height=crop_target_height, max_edge=crop_max_edge)
                if scale != 1.0:
                    crop = crop.resize(
                        (max(1, round(crop.width * scale)),
                         max(1, round(crop.height * scale))),
                        Image.Resampling.LANCZOS,
                    )
                encoded = io.BytesIO()
                crop.save(encoded, format="PNG", optimize=True)
                crop_payload = encoded.getvalue()
                crop_digest = hashlib.sha256(crop_payload).hexdigest()
                crop_target = (
                    image_dir / "crops" / crop_digest[:2] /
                    f"{crop_digest}.png")
                crop_target.parent.mkdir(parents=True, exist_ok=True)
                if (not crop_target.is_file()
                        or crop_target.stat().st_size != len(crop_payload)):
                    temporary = crop_target.with_suffix(".png.tmp")
                    temporary.write_bytes(crop_payload)
                    temporary.replace(crop_target)
                    # Only files this run created may be reclaimed later; a
                    # pre-existing digest belongs to an earlier manifest that
                    # may still reference it.
                    created_crops.add(crop_target)
                try:
                    crop_path = str(crop_target.relative_to(ROOT))
                except ValueError:
                    crop_path = str(crop_target.resolve())
                output_rows.append({
                    "image": crop_path, "text": block["text"],
                    "prompt": OCR_PROMPT, "source": "doclingmatix_ocr",
                    "stage1_source": "doclingmatix_ocr_located_crop",
                    "task": "ocr",
                    "docling_key": f"{candidate['key']}:crop:{block_index}",
                    "docling_parent_key": candidate["key"],
                    "docling_box_500": list(block["box_500"]),
                    "docling_element": block["kind"],
                    "ocr_granularity": "located_crop",
                    "image_sha256": crop_digest,
                    "width": crop.width, "height": crop.height,
                })
            materialized[candidate["key"]] = output_rows
        print({"kind": "docling_ocr_materialize", "groups": completed,
               "total_groups": len(grouped), "images": len(materialized)},
              flush=True)
    return [
        row
        for candidate in candidates
        for row in materialized.get(candidate["key"], [])
    ]


def split_ocr_rows(rows: Sequence[dict], candidates: Sequence[dict], *,
                   eval_count: int,
                   train_count: int) -> tuple[list[dict], list[dict]]:
    """Take disjoint train/eval selections whose parent pages never overlap.

    Rows are grouped by parent page first: a crop carries distinct image bytes
    but the same page content, so it must never cross the held-out boundary.
    """
    groups: dict[str, list[dict]] = {}
    parent_hashes: set[str] = set()
    for row in rows:
        parent = str(row["docling_parent_key"])
        if row["ocr_granularity"] == "page":
            if row["image_sha256"] in parent_hashes:
                continue
            parent_hashes.add(row["image_sha256"])
            groups[parent] = [row]
        elif parent in groups:
            groups[parent].append(row)
    ordered_groups = [groups[candidate["key"]] for candidate in candidates
                      if candidate["key"] in groups]

    selected_hashes: set[str] = set()

    def take_grouped(start: int, count: int) -> tuple[list[dict], int]:
        selected: list[dict] = []
        position = start
        while position < len(ordered_groups) and len(selected) < count:
            group = ordered_groups[position]
            position += 1
            for row in group:
                digest = str(row["image_sha256"])
                if digest in selected_hashes:
                    continue
                selected_hashes.add(digest)
                selected.append(row)
                if len(selected) == count:
                    break
        return selected, position

    selected_eval, next_group = take_grouped(0, eval_count)
    selected_train, _ = take_grouped(next_group, train_count)
    return selected_train, selected_eval


def additional_parents(deficit: int, rows_done: int, parents_done: int, *,
                       minimum: int = 256) -> int:
    """Extra parent pages to materialize to cover a measured row deficit.

    The yield is measured, not assumed: pages without located blocks emit one
    row and --crop-max-tokens rejects crops only after their images exist, so
    the ``1 + crops_per_page`` upper bound routinely overestimates. A pessimistic
    one-row-per-parent fallback covers the degenerate no-yield case.
    """
    if deficit <= 0:
        return 0
    observed = (rows_done / parents_done
                if parents_done > 0 and rows_done > 0 else 1.0)
    return max(minimum, math.ceil(deficit / observed))


def rooted_image(row: dict) -> Path:
    path = Path(str(row["image"]))
    return (path if path.is_absolute() else ROOT / path).resolve()


def main() -> None:
    args = parse_args()
    required = [args.source, args.base_train, args.base_eval]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise SystemExit(f"missing OCR inputs: {missing}")
    base_train = read_jsonl(args.base_train)
    base_eval = read_jsonl(args.base_eval)
    train_target = (args.target_train_rows or
                    target_ocr_rows(len(base_train), args.ocr_ratio))
    wanted = train_target + args.eval_rows

    source_paths = sorted(args.source.glob("*.parquet"))
    if len(source_paths) < args.candidate_shards:
        raise SystemExit(
            f"only {len(source_paths)} DoclingMatix shards for "
            f"--candidate-shards {args.candidate_shards}")
    source_paths = source_paths[:args.candidate_shards]
    candidates = collect_candidates(
        source_paths, min_chars=args.min_chars,
        max_chars=args.max_chars, crop_min_chars=args.crop_min_chars,
        crop_max_chars=args.crop_max_chars, seed=args.seed)
    if not args.vocab.is_file():
        raise SystemExit(f"missing RWKV World vocabulary: {args.vocab}")
    vocab = WorldVocab(str(args.vocab))
    before_token_filter = len(candidates)
    candidates = [row for row in candidates
                  if len(vocab.encode(row["text"])) <= args.max_tokens]
    print({"kind": "docling_ocr_token_filter",
           "before": before_token_filter, "accepted": len(candidates),
           "max_tokens": args.max_tokens}, flush=True)
    # Rows per parent are NOT knowable before materialization: a page with no
    # located block yields one row, and --crop-max-tokens rejects crops only
    # after their images are on disk. Reserving `1 + crops_per_page` rows per
    # parent therefore under-provisions and used to abort hours in; reserving
    # `wanted` parents could never under-provision but multiplies the work.
    # Materialize the optimistic tranche, measure the real yield, and extend.
    expected_rows_per_parent = 1 + args.crops_per_page
    minimum_parents = math.ceil(wanted / expected_rows_per_parent)
    if len(candidates) < minimum_parents:
        raise SystemExit(
            f"only {len(candidates)} quality OCR candidates: even at "
            f"{expected_rows_per_parent} rows each they cannot reach "
            f"{wanted} rows")
    reserve = min(len(candidates), minimum_parents + args.reserve_rows)
    created_crops: set[Path] = set()
    rows: list[dict] = []
    materialized_parents = 0
    while True:
        fresh = materialize_candidates(
            candidates[materialized_parents:reserve], args.image_dir,
            min_side=args.min_side, crops_per_page=args.crops_per_page,
            crop_target_height=args.crop_target_height,
            crop_max_edge=args.crop_max_edge,
            crop_padding=args.crop_padding, seed=args.seed,
            created_crops=created_crops)
        rows.extend(
            row for row in fresh
            if (row["ocr_granularity"] == "page"
                or len(vocab.encode(row["text"])) <= args.crop_max_tokens))
        materialized_parents = reserve
        train_rows, eval_rows = split_ocr_rows(
            rows, candidates[:materialized_parents],
            eval_count=args.eval_rows, train_count=train_target)
        deficit = ((train_target - len(train_rows))
                   + (args.eval_rows - len(eval_rows)))
        if deficit <= 0:
            break
        if materialized_parents >= len(candidates):
            raise SystemExit(
                f"only {len(train_rows)} train and {len(eval_rows)} eval OCR "
                f"rows from all {len(candidates)} candidates, for "
                f"{train_target} and {args.eval_rows} requested")
        reserve = min(
            len(candidates),
            materialized_parents + additional_parents(
                deficit, len(rows), materialized_parents))
        print({"kind": "docling_ocr_reserve_extended",
               "materialized_parents": materialized_parents, "rows": len(rows),
               "deficit": deficit, "next_reserve": reserve}, flush=True)

    # Crops are written before --crop-max-tokens and group selection run, so a
    # build always leaves rejected crop images behind. Reclaim only the files
    # this process created: an identical digest that already existed belongs to
    # an earlier manifest that may still reference it.
    referenced = {rooted_image(row) for row in (*train_rows, *eval_rows)}
    orphan_crops = [path for path in created_crops
                    if path.resolve() not in referenced]
    for path in orphan_crops:
        path.unlink(missing_ok=True)
    print({"kind": "docling_ocr_crop_cleanup",
           "created": len(created_crops), "removed": len(orphan_crops)},
          flush=True)

    # Shuffle within each already-disjoint split so page/crop curriculum rows
    # are distributed across the epoch rather than adjacent.
    random.Random(args.seed ^ 0x0C7).shuffle(train_rows)
    random.Random(args.seed ^ 0xE7A1).shuffle(eval_rows)

    train_paths = {rooted_image(row) for row in train_rows}
    eval_paths = {rooted_image(row) for row in eval_rows}
    base_train_paths = {rooted_image(row) for row in base_train}
    base_eval_paths = {rooted_image(row) for row in base_eval}
    if train_paths & eval_paths:
        raise RuntimeError("OCR train/eval image overlap")
    train_parents = {str(row["docling_parent_key"]) for row in train_rows}
    eval_parents = {str(row["docling_parent_key"]) for row in eval_rows}
    if train_parents & eval_parents:
        raise RuntimeError("OCR train/eval parent-page overlap")
    if train_paths & base_eval_paths or eval_paths & base_train_paths:
        raise RuntimeError("OCR supplement overlaps the opposite base split")
    if any(not path.is_file() for path in train_paths | eval_paths):
        raise RuntimeError("materialized OCR manifest contains missing images")

    combined_train = [*base_train, *train_rows]
    combined_eval = [*base_eval, *eval_rows]
    random.Random(args.seed ^ 0xC0AB).shuffle(combined_train)
    random.Random(args.seed ^ 0xE0AB).shuffle(combined_eval)
    atomic_jsonl(args.train_output, train_rows)
    atomic_jsonl(args.eval_output, eval_rows)
    atomic_jsonl(args.combined_train, combined_train)
    atomic_jsonl(args.combined_eval, combined_eval)
    final_ratio = len(train_rows) / len(combined_train)
    receipt = {
        "schema": 2, "seed": args.seed,
        "source": "HuggingFaceM4/DoclingMatix",
        "source_receipt": str((args.source.parents[1] /
                               "tranche_000.receipt.json").resolve()),
        "candidate_shards": [path.name for path in source_paths],
        "candidate_rows_passing_quality": len(candidates),
        "materialized_parent_pages": materialized_parents,
        "orphan_crops_removed": len(orphan_crops),
        "base_train_rows": len(base_train), "base_eval_rows": len(base_eval),
        "ocr_train_rows": len(train_rows), "ocr_eval_rows": len(eval_rows),
        "ocr_train_page_rows": sum(
            row["ocr_granularity"] == "page" for row in train_rows),
        "ocr_train_crop_rows": sum(
            row["ocr_granularity"] == "located_crop" for row in train_rows),
        "ocr_eval_page_rows": sum(
            row["ocr_granularity"] == "page" for row in eval_rows),
        "ocr_eval_crop_rows": sum(
            row["ocr_granularity"] == "located_crop" for row in eval_rows),
        "combined_train_rows": len(combined_train),
        "combined_eval_rows": len(combined_eval),
        "requested_ocr_ratio": args.ocr_ratio,
        "requested_train_rows": train_target,
        "actual_ocr_ratio": final_ratio,
        "train_eval_image_overlap": 0,
        "train_eval_parent_overlap": 0,
        "target_policy": (
            "plain reading-order full-page transcription plus enlarged, "
            "location-derived short text crops; Docling syntax removed; parent "
            "pages remain split-disjoint"),
        "train_sha256": file_sha256(args.train_output),
        "eval_sha256": file_sha256(args.eval_output),
        "combined_train_sha256": file_sha256(args.combined_train),
        "combined_eval_sha256": file_sha256(args.combined_eval),
    }
    atomic_json(args.receipt, receipt)
    print(json.dumps({"kind": "docling_ocr_mix", "state": "ready",
                      "train": str(args.train_output),
                      "eval": str(args.eval_output),
                      "combined_train": str(args.combined_train),
                      "combined_eval": str(args.combined_eval),
                      "receipt": str(args.receipt), **receipt},
                     indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
