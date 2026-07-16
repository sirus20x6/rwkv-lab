#!/usr/bin/env python3
"""Build a bounded, image-disjoint DoclingMatix OCR supplement.

The base vision shard stays immutable. This builder materializes a deterministic
OCR-only train/eval supplement plus combined manifests. Structured DocTags are
converted to plain reading-order text so the captioner learns visible text, not
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
    ap.add_argument("--eval-rows", type=int, default=64)
    ap.add_argument("--candidate-shards", type=int, default=18,
                    help="leading pinned shards used for the bounded supplement")
    ap.add_argument("--reserve-rows", type=int, default=2048,
                    help="extra candidates retained before content deduplication")
    ap.add_argument("--min-chars", type=int, default=160)
    ap.add_argument("--max-chars", type=int, default=2400)
    ap.add_argument("--max-tokens", type=int, default=700,
                    help="maximum complete transcription length in RWKV tokens")
    ap.add_argument("--vocab", type=Path, default=Path(os.environ.get(
        "VOCAB", "/thearray/git/ztok/bench/vocabs/rwkv_vocab_v20230424.txt")))
    ap.add_argument("--min-side", type=int, default=512)
    ap.add_argument("--seed", type=int, default=20260716)
    args = ap.parse_args()
    if not 0 < args.ocr_ratio < 1:
        ap.error("--ocr-ratio must be between zero and one")
    if min(args.eval_rows, args.candidate_shards, args.reserve_rows,
           args.min_chars, args.max_chars, args.max_tokens, args.min_side) < 1:
        ap.error("row, shard, character, and size limits must be positive")
    if args.min_chars > args.max_chars:
        ap.error("--min-chars cannot exceed --max-chars")
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


def conversion_target(texts: Sequence[dict], *, min_chars: int,
                      max_chars: int) -> str | None:
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
        return text
    return None


def stable_rank(seed: int, key: str) -> bytes:
    return hashlib.sha256(f"{seed}:{key}".encode()).digest()


def collect_candidates(paths: Sequence[Path], *, min_chars: int,
                       max_chars: int, seed: int) -> list[dict]:
    candidates = []
    for shard_number, path in enumerate(paths, 1):
        parquet = pq.ParquetFile(path)
        row_offset = 0
        accepted = 0
        for group in range(parquet.metadata.num_row_groups):
            table = parquet.read_row_group(group, columns=["texts"])
            for local, texts in enumerate(table["texts"].to_pylist()):
                row_index = row_offset + local
                text = conversion_target(
                    texts or [], min_chars=min_chars, max_chars=max_chars)
                if text is None:
                    continue
                key = f"{path.name}:{row_index}"
                candidates.append({
                    "key": key, "source_file": str(path.resolve()),
                    "row_index": row_index, "row_group": group,
                    "row_group_offset": local, "text": text,
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


def materialize_candidates(candidates: Sequence[dict], image_dir: Path,
                           *, min_side: int) -> list[dict]:
    grouped: dict[tuple[Path, int], list[dict]] = defaultdict(list)
    for candidate in candidates:
        grouped[(Path(candidate["source_file"]),
                 int(candidate["row_group"]))].append(candidate)

    materialized: dict[str, dict] = {}
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
            materialized[candidate["key"]] = {
                "image": image_path, "text": candidate["text"],
                "prompt": OCR_PROMPT, "source": "doclingmatix_ocr",
                "stage1_source": "doclingmatix_ocr_transcription",
                "task": "ocr", "docling_key": candidate["key"],
                "image_sha256": digest, "width": width, "height": height,
            }
        print({"kind": "docling_ocr_materialize", "groups": completed,
               "total_groups": len(grouped), "images": len(materialized)},
              flush=True)
    return [materialized[row["key"]] for row in candidates
            if row["key"] in materialized]


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
    train_target = target_ocr_rows(len(base_train), args.ocr_ratio)
    wanted = train_target + args.eval_rows

    source_paths = sorted(args.source.glob("*.parquet"))
    if len(source_paths) < args.candidate_shards:
        raise SystemExit(
            f"only {len(source_paths)} DoclingMatix shards for "
            f"--candidate-shards {args.candidate_shards}")
    source_paths = source_paths[:args.candidate_shards]
    candidates = collect_candidates(
        source_paths, min_chars=args.min_chars,
        max_chars=args.max_chars, seed=args.seed)
    if not args.vocab.is_file():
        raise SystemExit(f"missing RWKV World vocabulary: {args.vocab}")
    vocab = WorldVocab(str(args.vocab))
    before_token_filter = len(candidates)
    candidates = [row for row in candidates
                  if len(vocab.encode(row["text"])) <= args.max_tokens]
    print({"kind": "docling_ocr_token_filter",
           "before": before_token_filter, "accepted": len(candidates),
           "max_tokens": args.max_tokens}, flush=True)
    reserve = wanted + args.reserve_rows
    if len(candidates) < reserve:
        raise SystemExit(
            f"only {len(candidates)} quality OCR candidates for {reserve} requested")
    rows = materialize_candidates(
        candidates[:reserve], args.image_dir, min_side=args.min_side)

    unique = []
    seen_hashes = set()
    for row in rows:
        if row["image_sha256"] in seen_hashes:
            continue
        seen_hashes.add(row["image_sha256"])
        unique.append(row)
        if len(unique) == wanted:
            break
    if len(unique) != wanted:
        raise SystemExit(
            f"only {len(unique)} unique materialized OCR rows for {wanted} requested")

    # Assign held-out rows first from a deterministically shuffled candidate
    # order, then shuffle training independently so source shard order is never
    # visible to the sampler.
    eval_rows = unique[:args.eval_rows]
    train_rows = unique[args.eval_rows:]
    random.Random(args.seed ^ 0x0C7).shuffle(train_rows)
    random.Random(args.seed ^ 0xE7A1).shuffle(eval_rows)

    train_paths = {rooted_image(row) for row in train_rows}
    eval_paths = {rooted_image(row) for row in eval_rows}
    base_train_paths = {rooted_image(row) for row in base_train}
    base_eval_paths = {rooted_image(row) for row in base_eval}
    if train_paths & eval_paths:
        raise RuntimeError("OCR train/eval image overlap")
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
        "schema": 1, "seed": args.seed,
        "source": "HuggingFaceM4/DoclingMatix",
        "source_receipt": str((args.source.parents[1] /
                               "tranche_000.receipt.json").resolve()),
        "candidate_shards": [path.name for path in source_paths],
        "candidate_rows_passing_quality": len(candidates),
        "base_train_rows": len(base_train), "base_eval_rows": len(base_eval),
        "ocr_train_rows": len(train_rows), "ocr_eval_rows": len(eval_rows),
        "combined_train_rows": len(combined_train),
        "combined_eval_rows": len(combined_eval),
        "requested_ocr_ratio": args.ocr_ratio,
        "actual_ocr_ratio": final_ratio,
        "train_eval_image_overlap": 0,
        "target_policy": (
            "plain reading-order transcription; Docling tags and coordinates "
            "removed; complete 160-2400 character and <=700 World-token targets"),
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
