#!/usr/bin/env python3
"""Replace a materialized Midjourney stage's text with aligned i1 captions."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import pyarrow.parquet as pq

from repair_midjourney_i1_alignment import (
    SOURCE_CAPTION_COLUMNS,
    aligned_caption,
    build_mapping,
    load_i1_captions,
)


DEFAULT_STAGE = Path("/thearray/git/datasets/i1")
DEFAULT_CAPTIONS = Path("/thearray/git/moe-mla/i1-captions/midjourneyv6")
DEFAULT_SOURCE = Path(
    "/thearray/git/moe-mla/datasets/i1_full_sources/midjourneyv6"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=Path, default=DEFAULT_STAGE)
    parser.add_argument("--captions", type=Path, default=DEFAULT_CAPTIONS)
    parser.add_argument("--source-shards", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--seed", type=int, default=20260730)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def atomic_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    temporary = path.with_name(path.name + ".i1.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            )
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".i1.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_revision(path: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def selected_source_groups(
    source_directory: Path,
    metadata: list[dict[str, Any]],
) -> tuple[list[tuple[Path, str, list[tuple[int, list[str]]]]], set[str]]:
    families_by_shard: dict[str, set[str]] = defaultdict(set)
    expected_rows: set[tuple[str, int]] = set()
    for row in metadata:
        shard = str(row["source_shard"])
        families_by_shard[shard].add(str(row["family_id"]))
        expected_rows.add((shard, int(row["source_row"])))

    groups: list[tuple[Path, str, list[tuple[int, list[str]]]]] = []
    wanted_keys: set[str] = set()
    seen_rows: set[tuple[str, int]] = set()
    for index, shard in enumerate(sorted(families_by_shard), start=1):
        source_path = (source_directory / shard).resolve()
        if not source_path.is_file():
            raise RuntimeError(f"missing pinned source shard: {source_path}")
        wanted_families = families_by_shard[shard]
        grouped: dict[str, list[tuple[int, list[str]]]] = defaultdict(list)
        row_index = 0
        parquet = pq.ParquetFile(source_path)
        for batch in parquet.iter_batches(
            columns=["id", *SOURCE_CAPTION_COLUMNS],
            batch_size=16_384,
        ):
            for row in batch.to_pylist():
                numeric_id = f"{int(row['id']):09d}"
                if numeric_id in wanted_families:
                    grouped[numeric_id].append(
                        (
                            row_index,
                            [
                                str(row.get(column) or "")
                                for column in SOURCE_CAPTION_COLUMNS
                            ],
                        )
                    )
                    seen_rows.add((shard, row_index))
                row_index += 1
        malformed = {
            family: len(rows) for family, rows in grouped.items() if len(rows) != 4
        }
        missing_families = wanted_families - grouped.keys()
        if malformed or missing_families:
            raise RuntimeError(
                f"{shard}: malformed groups={malformed}, "
                f"missing families={len(missing_families)}"
            )
        for family, rows in grouped.items():
            i1_family = str(int(family))
            groups.append((source_path, i1_family, rows))
            wanted_keys.update(f"{i1_family}_{suffix}" for suffix in range(4))
        print(
            f"source alignment {index}/{len(families_by_shard)}: "
            f"{shard} ({len(grouped):,} families)",
            flush=True,
        )
    if not expected_rows <= seen_rows:
        raise RuntimeError(
            f"source alignment missed {len(expected_rows - seen_rows):,} selected rows"
        )
    return groups, wanted_keys


def main() -> None:
    args = parse_args()
    stage = args.stage.expanduser().resolve()
    captions = args.captions.expanduser().resolve()
    source_shards = args.source_shards.expanduser().resolve()
    receipt_path = stage / "i1_caption_receipt.json"
    if receipt_path.is_file():
        print(receipt_path.read_text(encoding="utf-8"))
        return

    metadata_path = stage / "metadata.jsonl"
    train_path = stage / "train.jsonl"
    report_path = stage / "report.json"
    for required in (metadata_path, train_path, report_path, captions, source_shards):
        if not required.exists():
            raise SystemExit(f"required input does not exist: {required}")

    metadata = read_jsonl(metadata_path)
    train = read_jsonl(train_path)
    if len(metadata) != len(train) or not metadata:
        raise RuntimeError(
            f"metadata/train mismatch: {len(metadata):,}/{len(train):,}"
        )
    metadata_keys = [
        (str(row["source_shard"]), int(row["source_row"])) for row in metadata
    ]
    train_keys = [
        (str(row["source_shard"]), int(row["source_row"])) for row in train
    ]
    if metadata_keys != train_keys or len(set(metadata_keys)) != len(metadata_keys):
        raise RuntimeError("metadata/train source identities differ or are duplicated")

    groups, wanted_keys = selected_source_groups(source_shards, metadata)
    i1_captions = load_i1_captions(captions, wanted_keys)
    by_source_row, _by_physical_key, alignment_stats = build_mapping(
        groups, i1_captions
    )

    provenance: Counter[str] = Counter()
    variant_counts: Counter[int] = Counter()
    changed = 0
    canonical_keys: set[str] = set()
    rewritten_metadata: list[dict[str, Any]] = []
    rewritten_train: list[dict[str, Any]] = []
    for metadata_row, train_row in zip(metadata, train, strict=True):
        shard = str(metadata_row["source_shard"])
        row_index = int(metadata_row["source_row"])
        item = by_source_row.get(
            (str((source_shards / shard).resolve()), row_index)
        )
        if item is None:
            raise RuntimeError(f"no i1 alignment for {shard}:{row_index}")
        text, variant, source = aligned_caption(item, i1_captions, args.seed)
        text = text.strip()
        if not text:
            raise RuntimeError(f"empty aligned caption for {shard}:{row_index}")
        canonical_key = str(item["canonical_key"])
        provenance[source] += 1
        variant_counts[int(variant)] += 1
        canonical_keys.add(canonical_key)
        changed += text != str(metadata_row["text"])
        rewritten_metadata.append(
            {
                **metadata_row,
                "text": text,
                "caption_source": source,
                "i1_key": canonical_key,
                "i1_caption_variant": int(variant),
                "i1_alignment_margin": float(item["alignment_margin"]),
            }
        )
        rewritten_train.append(
            {
                **train_row,
                "source": "i1_midjourneyv6",
                "caption": text,
                "conditioning_text": text,
                "conditioning_kind": source,
                "caption_provenance": source,
                "i1_key": canonical_key,
                "i1_caption_variant": int(variant),
            }
        )

    domain_counts = Counter(str(row["domain"]) for row in rewritten_train)
    if domain_counts != Counter(str(row["domain"]) for row in train):
        raise RuntimeError("recaptioning changed the route distribution")
    if len(canonical_keys) != len(rewritten_train):
        raise RuntimeError(
            f"i1 key collision: {len(canonical_keys):,}/{len(rewritten_train):,}"
        )
    missing_images = [
        row["image"] for row in rewritten_train if not Path(row["image"]).is_file()
    ]
    if missing_images:
        raise RuntimeError(f"{len(missing_images):,} materialized images are missing")

    atomic_jsonl(metadata_path, rewritten_metadata)
    atomic_jsonl(train_path, rewritten_train)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["caption_dataset"] = "zlab-princeton/i1-captions"
    report["caption_dataset_revision"] = git_revision(captions.parent)
    report["i1_captioning"] = {
        "rows": len(rewritten_train),
        "changed_from_routing_caption": changed,
        "canonical_i1_keys": len(canonical_keys),
        "caption_provenance_counts": dict(sorted(provenance.items())),
        "caption_variant_counts": {
            str(key): value for key, value in sorted(variant_counts.items())
        },
        "alignment": alignment_stats,
    }
    atomic_json(report_path, report)
    receipt = {
        "schema": "rwkv-lab.i1-ratioed-tranche-caption-receipt.v1",
        "stage": str(stage),
        "caption_dataset": "zlab-princeton/i1-captions",
        "caption_dataset_revision": git_revision(captions.parent),
        "source_dataset": "Photoroom/midjourney-v6-recap",
        "rows": len(rewritten_train),
        "domain_counts": dict(sorted(domain_counts.items())),
        "canonical_i1_keys": len(canonical_keys),
        "caption_provenance_counts": dict(sorted(provenance.items())),
        "changed_from_routing_caption": changed,
        "metadata_sha256": sha256(metadata_path),
        "train_sha256": sha256(train_path),
        "report_sha256": sha256(report_path),
    }
    atomic_json(receipt_path, receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
