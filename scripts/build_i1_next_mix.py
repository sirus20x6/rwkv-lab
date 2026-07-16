#!/usr/bin/env python3
"""Build an image-verified i1 mixture for the next vision training phase.

The i1 caption repository contains keys and captions, not image payloads.  This
script performs an exact key join against small, explicitly downloaded source
shards, decodes every selected image, and emits only matched pairs.  It uses a
balanced Pexels/Midjourney sample so the i1 addition is not all synthetic art.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import random
import sys
import tarfile
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from midjourney_alignment import SOURCE_CAPTION_COLUMNS, align_group

CAPTION_COLUMNS = tuple(f"caption{i}" for i in range(1, 6))
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path,
                    default=ROOT / "curated_vision/vision_stage1_mix.jsonl")
    ap.add_argument("--captions", type=Path, default=ROOT / "i1-captions")
    ap.add_argument("--midjourney-shard", type=Path, default=ROOT /
                    "datasets/i1_source_shards/midjourneyv6/train_000.parquet")
    ap.add_argument("--pexels-shard", type=Path, default=ROOT /
                    "datasets/i1_source_shards/pexels/train/00001.tar")
    ap.add_argument("--image-dir", type=Path,
                    default=ROOT / "datasets/i1_matched_images")
    ap.add_argument("--i1-output", type=Path,
                    default=ROOT / "curated_vision/i1_matched_25pct.jsonl")
    ap.add_argument("--output", type=Path,
                    default=ROOT / "curated_vision/vision_next_i1_25pct.jsonl")
    ap.add_argument("--i1-ratio", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=20260714)
    return ap.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def midjourney_keys(path: Path) -> list[str]:
    """Return all possible i1 keys without assuming duplicate-row ordering."""
    pf = pq.ParquetFile(path)
    seen: Counter[str] = Counter()
    for group in range(pf.metadata.num_row_groups):
        for raw_id in pf.read_row_group(group, columns=["id"])["id"].to_pylist():
            seen[str(int(raw_id))] += 1
    if any(count != 4 for count in seen.values()):
        raise RuntimeError("Midjourney source shard does not contain four images per id")
    return [f"{numeric_id}_{suffix}" for numeric_id in seen for suffix in range(4)]


def aligned_midjourney_row_keys(path: Path,
                                captions: dict[str, list[str]]) -> list[str]:
    """Map each physical parquet row to the semantically matching i1 suffix."""
    parquet = pq.ParquetFile(path)
    grouped: dict[str, list[tuple[int, list[str]]]] = defaultdict(list)
    row_index = 0
    for group in range(parquet.metadata.num_row_groups):
        table = parquet.read_row_group(
            group, columns=["id", *SOURCE_CAPTION_COLUMNS])
        for row in table.to_pylist():
            numeric_id = str(int(row["id"]))
            grouped[numeric_id].append((
                row_index,
                [str(row.get(name) or "") for name in SOURCE_CAPTION_COLUMNS],
            ))
            row_index += 1
    result: list[str | None] = [None] * row_index
    for numeric_id, rows in grouped.items():
        if len(rows) != 4:
            raise RuntimeError(f"Midjourney id {numeric_id} does not have four rows")
        targets = [captions.get(f"{numeric_id}_{suffix}", []) for suffix in range(4)]
        aligned = align_group([values for _, values in rows], targets)
        for source_offset, (index, _) in enumerate(rows):
            result[index] = f"{numeric_id}_{aligned.row_to_suffix[source_offset]}"
    if any(key is None for key in result):
        raise RuntimeError("Midjourney alignment left physical rows unmapped")
    return [str(key) for key in result]


def pexels_members(path: Path) -> dict[str, str]:
    result = {}
    with tarfile.open(path) as archive:
        for member in archive:
            suffix = Path(member.name).suffix.lower()
            if member.isfile() and suffix in IMAGE_SUFFIXES:
                result[Path(member.name).stem] = member.name
    return result


def load_matching_captions(directory: Path, wanted: set[str]) -> dict[str, list[str]]:
    """Scan caption columns a row group at a time and retain exact key matches."""
    matches: dict[str, list[str]] = {}
    arrow_wanted = pc.SetLookupOptions(value_set=pa.array(sorted(wanted), type=pa.string()))
    for path in sorted(directory.glob("*.parquet")):
        pf = pq.ParquetFile(path)
        available = [name for name in CAPTION_COLUMNS if name in pf.schema_arrow.names]
        for group in range(pf.metadata.num_row_groups):
            table = pf.read_row_group(group, columns=["key", *available])
            table = table.filter(pc.is_in(table["key"], options=arrow_wanted))
            for row in table.to_pylist():
                captions = [str(row[name]).strip() for name in available
                            if row.get(name) and str(row[name]).strip()]
                if captions:
                    matches[str(row["key"])] = captions
        if len(matches) == len(wanted):
            break
    return matches


def verified_image(data: bytes) -> tuple[str, int, int, str]:
    with Image.open(io.BytesIO(data)) as image:
        image.load()
        width, height = image.size
        fmt = (image.format or "JPEG").lower()
    suffix = {"jpeg": ".jpg", "jpg": ".jpg", "png": ".png",
              "webp": ".webp"}.get(fmt, ".img")
    return suffix, width, height, hashlib.sha256(data).hexdigest()


def choose_caption(captions: list[str], key: str, seed: int) -> tuple[str, int]:
    digest = hashlib.sha256(f"{seed}:{key}".encode()).digest()
    index = int.from_bytes(digest[:8], "big") % len(captions)
    return captions[index], index + 1


def write_image(data: bytes, output: Path, key: str,
                source: str) -> tuple[Path, str, int, int]:
    suffix, width, height, digest = verified_image(data)
    directory = output / source
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{key}{suffix}"
    if not path.exists() or path.stat().st_size != len(data):
        path.write_bytes(data)
    return path, digest, width, height


def extract_midjourney(path: Path, row_keys: list[str],
                       captions: dict[str, list[str]], wanted: int,
                       output: Path, seed: int, used_hashes: set[str]) -> list[dict]:
    rng = random.Random(seed ^ 0x4D4A)
    eligible = list(captions)
    rng.shuffle(eligible)
    rank = {key: i for i, key in enumerate(eligible)}
    pf = pq.ParquetFile(path)
    candidates: list[tuple[int, str, bytes]] = []
    row_index = 0
    for group in range(pf.metadata.num_row_groups):
        table = pf.read_row_group(group, columns=["image"])
        for row in table.to_pylist():
            key = row_keys[row_index]
            row_index += 1
            if key in rank:
                payload = row["image"]["bytes"]
                if payload:
                    candidates.append((rank[key], key, payload))
    rows = []
    for _, key, payload in sorted(candidates):
        try:
            image_path, digest, width, height = write_image(
                payload, output, key, "midjourneyv6")
        except Exception:
            continue
        if digest in used_hashes:
            continue
        used_hashes.add(digest)
        text, variant = choose_caption(captions[key], key, seed)
        rows.append({"image": str(image_path.relative_to(ROOT)), "text": text,
                     "stage1_source": "i1_midjourneyv6", "i1_subset": "midjourneyv6",
                     "i1_key": key, "caption_variant": variant,
                     "image_sha256": digest, "width": width, "height": height})
        if len(rows) == wanted:
            break
    return rows


def extract_pexels(path: Path, members: dict[str, str], captions: dict[str, list[str]],
                   wanted: int, output: Path, seed: int,
                   used_hashes: set[str]) -> list[dict]:
    rng = random.Random(seed ^ 0x504558)
    eligible = list(captions)
    rng.shuffle(eligible)
    selected = set(eligible)
    payloads = {}
    with tarfile.open(path) as archive:
        for member in archive:
            key = Path(member.name).stem
            if key not in selected or member.name != members.get(key):
                continue
            handle = archive.extractfile(member)
            if handle is not None:
                payloads[key] = handle.read()
    rows = []
    for key in eligible:
        payload = payloads.get(key)
        if not payload:
            continue
        try:
            image_path, digest, width, height = write_image(
                payload, output, key, "pexels")
        except Exception:
            continue
        if digest in used_hashes:
            continue
        used_hashes.add(digest)
        text, variant = choose_caption(captions[key], key, seed)
        rows.append({"image": str(image_path.relative_to(ROOT)), "text": text,
                     "stage1_source": "i1_pexels", "i1_subset": "pexels",
                     "i1_key": key, "caption_variant": variant,
                     "image_sha256": digest, "width": width, "height": height})
        if len(rows) == wanted:
            break
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    if not 0 < args.i1_ratio < 1:
        raise SystemExit("--i1-ratio must be between zero and one")
    required = [args.base, args.midjourney_shard, args.pexels_shard,
                args.captions / "midjourneyv6", args.captions / "pexels"]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise SystemExit(f"missing required inputs: {missing}")

    base = read_jsonl(args.base)
    target = math.ceil(len(base) * args.i1_ratio / (1 - args.i1_ratio))
    pexels_target = target // 2
    midjourney_target = target - pexels_target

    print("indexing source keys", flush=True)
    mj_keys = midjourney_keys(args.midjourney_shard)
    px_members = pexels_members(args.pexels_shard)
    print("joining i1 captions", flush=True)
    mj_captions = load_matching_captions(args.captions / "midjourneyv6", set(mj_keys))
    px_captions = load_matching_captions(args.captions / "pexels", set(px_members))
    print(f"matched captions: midjourney={len(mj_captions)} pexels={len(px_captions)}",
          flush=True)

    used_hashes: set[str] = set()
    mj_row_keys = aligned_midjourney_row_keys(args.midjourney_shard, mj_captions)
    midjourney = extract_midjourney(args.midjourney_shard, mj_row_keys, mj_captions,
                                    midjourney_target, args.image_dir,
                                    args.seed, used_hashes)
    pexels = extract_pexels(args.pexels_shard, px_members, px_captions,
                            pexels_target, args.image_dir, args.seed, used_hashes)
    i1_rows = midjourney + pexels
    if len(i1_rows) != target:
        raise SystemExit(f"only extracted {len(i1_rows)}/{target} verified i1 images")

    rng = random.Random(args.seed)
    rng.shuffle(i1_rows)
    combined = [dict(row) for row in base] + i1_rows
    rng.shuffle(combined)
    write_jsonl(args.i1_output, i1_rows)
    write_jsonl(args.output, combined)

    counts = Counter(row.get("stage1_source", "unknown") for row in combined)
    receipt = {
        "seed": args.seed, "base_manifest": str(args.base.relative_to(ROOT)),
        "base_rows": len(base), "i1_rows": len(i1_rows), "rows": len(combined),
        "i1_ratio": len(i1_rows) / len(combined), "counts": dict(sorted(counts.items())),
        "i1_sources": {"midjourneyv6": len(midjourney), "pexels": len(pexels)},
        "caption_matches": {"midjourneyv6": len(mj_captions), "pexels": len(px_captions)},
        "image_sha256_unique": len(used_hashes),
        "output_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
        "policy": "exact source-key join; successfully decoded images only; one deterministic caption variant",
    }
    args.output.with_suffix(".summary.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
