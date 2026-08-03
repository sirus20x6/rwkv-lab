#!/usr/bin/env python3
"""Build a deterministic Gelbooru image/caption training subset."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sqlite3
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq


DEFAULT_DATASET_ROOT = Path("/thearray/git/datasets/gelbooru-masterpiece")
DEFAULT_IMAGES = DEFAULT_DATASET_ROOT / "sources"
DEFAULT_DATABASE = DEFAULT_DATASET_ROOT / "state/downloads.sqlite3"
DEFAULT_METADATA = DEFAULT_DATASET_ROOT / "state/stage_01_score_ge_100.parquet"
DEFAULT_DESTINATION = Path("/thearray/git/datasets/gelbooru-trainer-all")


@dataclass(frozen=True)
class Record:
    id: int
    score: int
    source: Path
    suffix: str
    caption: str
    rating: str
    width: int
    height: int
    md5: str

    @property
    def image_name(self) -> str:
        return f"{self.id}{self.suffix}"

    def json_record(self) -> dict[str, Any]:
        return {
            "file_name": f"images/{self.image_name}",
            "text": self.caption,
            "id": self.id,
            "score": self.score,
            "rating": self.rating,
            "width": self.width,
            "height": self.height,
            "md5": self.md5,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, default=DEFAULT_IMAGES)
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--destination", type=Path, default=DEFAULT_DESTINATION)
    parser.add_argument(
        "--count",
        type=parse_count,
        default=None,
        metavar="N|all",
        help="number of records to select; defaults to every eligible download",
    )
    parser.add_argument(
        "--exclude-metadata",
        type=Path,
        action="append",
        default=[],
        help="metadata.jsonl whose IDs and MD5 values must not be selected",
    )
    return parser.parse_args()


def parse_count(value: str) -> int | None:
    if value.lower() == "all":
        return None
    count = int(value)
    if count <= 0:
        raise argparse.ArgumentTypeError("count must be positive or 'all'")
    return count


def read_download_snapshot(database: Path) -> list[tuple[int, int, str]]:
    uri = f"file:{database.resolve()}?mode=ro"
    connection = sqlite3.connect(uri, uri=True)
    try:
        connection.execute("BEGIN")
        rows = connection.execute(
            """
            SELECT id, community_score, local_path
            FROM downloads
            WHERE status = 'downloaded'
            ORDER BY community_score DESC, id DESC
            """
        ).fetchall()
        connection.commit()
    finally:
        connection.close()

    return [(int(image_id), int(score), str(path)) for image_id, score, path in rows]


def read_metadata(metadata_path: Path, wanted_ids: set[int]) -> dict[int, dict[str, Any]]:
    columns = ["id", "width", "height", "rating", "score", "md5", "tags"]
    table = pq.read_table(metadata_path, columns=columns)
    result: dict[int, dict[str, Any]] = {}
    for row in table.to_pylist():
        image_id = int(row["id"])
        if image_id not in wanted_ids:
            continue
        if image_id in result:
            raise RuntimeError(f"metadata contains duplicate image ID {image_id}")
        result[image_id] = row
    return result


def read_exclusions(paths: list[Path]) -> tuple[set[int], set[str]]:
    ids: set[int] = set()
    md5_values: set[str] = set()
    for path in paths:
        with path.expanduser().resolve().open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                    ids.add(int(row["id"]))
                    md5 = str(row["md5"]).strip().lower()
                except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                    raise RuntimeError(
                        f"invalid exclusion row in {path} at line {line_number}"
                    ) from exc
                if md5:
                    md5_values.add(md5)
    return ids, md5_values


def resolve_source(local_path: str, images_root: Path) -> Path | None:
    candidate = Path(local_path)
    if not candidate.is_absolute():
        # Download paths are rooted at the parent of sources (for example,
        # "sources/0196/1961533.jpg").
        candidate = images_root.parent / candidate
    try:
        resolved = candidate.resolve(strict=True)
    except (FileNotFoundError, OSError):
        return None

    images_resolved = images_root.resolve(strict=True)
    if not resolved.is_relative_to(images_resolved):
        return None
    if "staging" in resolved.relative_to(images_resolved).parts:
        return None
    return resolved if resolved.is_file() else None


def select_records(
    downloads: list[tuple[int, int, str]],
    metadata: dict[int, dict[str, Any]],
    images_root: Path,
    count: int | None,
    excluded_ids: set[int] | None = None,
    excluded_md5: set[str] | None = None,
) -> tuple[list[Record], Counter[str]]:
    selected: list[Record] = []
    skipped: Counter[str] = Counter()
    seen_ids: set[int] = set()
    seen_md5: set[str] = set()
    excluded_ids = excluded_ids or set()
    excluded_md5 = excluded_md5 or set()

    for image_id, community_score, local_path in downloads:
        if image_id in excluded_ids:
            skipped["excluded_id"] += 1
            continue
        if image_id in seen_ids:
            skipped["duplicate_id"] += 1
            continue
        seen_ids.add(image_id)

        row = metadata.get(image_id)
        if row is None:
            skipped["missing_metadata"] += 1
            continue

        tags = row.get("tags")
        caption_parts = str(tags).strip().split() if tags is not None else []
        if not caption_parts:
            skipped["empty_tags"] += 1
            continue

        source = resolve_source(local_path, images_root)
        if source is None:
            skipped["missing_or_invalid_image"] += 1
            continue

        metadata_score = int(row["score"])
        if metadata_score != community_score:
            raise RuntimeError(
                f"score mismatch for ID {image_id}: "
                f"database={community_score}, metadata={metadata_score}"
            )

        md5 = str(row.get("md5") or "").strip().lower()
        if not md5:
            skipped["empty_md5"] += 1
            continue
        if md5 in excluded_md5:
            skipped["excluded_md5"] += 1
            continue
        if md5 in seen_md5:
            skipped["duplicate_md5"] += 1
            continue
        seen_md5.add(md5)

        suffix = source.suffix.lower()
        if not suffix:
            skipped["missing_extension"] += 1
            continue

        selected.append(
            Record(
                id=image_id,
                score=community_score,
                source=source,
                suffix=suffix,
                caption=", ".join(caption_parts),
                rating=str(row.get("rating") or ""),
                width=int(row["width"]),
                height=int(row["height"]),
                md5=md5,
            )
        )
        if count is not None and len(selected) == count:
            break

    if count is not None and len(selected) != count:
        raise RuntimeError(
            f"only {len(selected):,} valid unique records were available; "
            f"{count:,} required"
        )
    return selected, skipped


def selection_digest(records: list[Record]) -> str:
    digest = hashlib.sha256()
    for record in records:
        digest.update(
            f"{record.id}\t{record.score}\t{record.md5}\t{record.caption}\n".encode("utf-8")
        )
    return digest.hexdigest()


def validate_build(root: Path, records: list[Record]) -> dict[str, Any]:
    images_dir = root / "images"
    expected_names = {record.image_name for record in records}
    image_files = sorted(
        path for path in images_dir.iterdir() if path.is_file() and path.suffix != ".txt"
    )
    sidecars = sorted(images_dir.glob("*.txt"))
    metadata_lines = (root / "metadata.jsonl").read_text(encoding="utf-8").splitlines()

    expected_count = len(records)
    if len(image_files) != expected_count:
        raise RuntimeError(f"found {len(image_files)} images, expected {expected_count}")
    if len(sidecars) != expected_count:
        raise RuntimeError(f"found {len(sidecars)} sidecars, expected {expected_count}")
    if len(metadata_lines) != expected_count:
        raise RuntimeError(
            f"found {len(metadata_lines)} metadata records, expected {expected_count}"
        )
    if {path.name for path in image_files} != expected_names:
        raise RuntimeError("materialized image names do not match the selection")

    parsed = [json.loads(line) for line in metadata_lines]
    ids = [int(row["id"]) for row in parsed]
    md5_values = [str(row["md5"]) for row in parsed]
    if len(ids) != len(set(ids)):
        raise RuntimeError("duplicate IDs found in metadata.jsonl")
    if len(md5_values) != len(set(md5_values)):
        raise RuntimeError("duplicate MD5 values found in metadata.jsonl")

    expected_rows = [record.json_record() for record in records]
    if parsed != expected_rows:
        raise RuntimeError("metadata.jsonl does not exactly match the selected records")
    if [
        (int(row["score"]), int(row["id"])) for row in parsed
    ] != sorted(
        ((int(row["score"]), int(row["id"])) for row in parsed),
        reverse=True,
    ):
        raise RuntimeError("metadata.jsonl is not in score-descending, ID-descending order")

    for record in records:
        image_path = images_dir / record.image_name
        sidecar_path = image_path.with_suffix(".txt")
        if not image_path.is_file() or not sidecar_path.is_file():
            raise RuntimeError(f"missing image/caption pair for ID {record.id}")
        if sidecar_path.read_text(encoding="utf-8") != record.caption + "\n":
            raise RuntimeError(f"caption mismatch for ID {record.id}")

    extension_counts = Counter(path.suffix.lower() for path in image_files)
    apparent_bytes = sum(
        path.stat().st_size
        for path in root.rglob("*")
        if path.is_file()
    )
    return {
        "images": len(image_files),
        "sidecars": len(sidecars),
        "metadata_records": len(parsed),
        "extension_counts": dict(sorted(extension_counts.items())),
        "apparent_bytes": apparent_bytes,
    }


def materialize(
    destination: Path,
    records: list[Record],
    downloads_count: int,
    skipped: Counter[str],
) -> dict[str, Any]:
    destination = destination.resolve()
    if destination.exists():
        raise FileExistsError(
            f"destination already exists; refusing to overwrite: {destination}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent)
    )
    linked = 0
    copied = 0
    try:
        images_dir = temporary / "images"
        images_dir.mkdir()
        metadata_path = temporary / "metadata.jsonl"
        with metadata_path.open("w", encoding="utf-8", newline="\n") as metadata_file:
            for record in records:
                output_image = images_dir / record.image_name
                try:
                    os.link(record.source, output_image)
                    linked += 1
                except OSError:
                    shutil.copy2(record.source, output_image)
                    copied += 1

                output_image.with_suffix(".txt").write_text(
                    record.caption + "\n", encoding="utf-8", newline="\n"
                )
                metadata_file.write(
                    json.dumps(
                        record.json_record(),
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                    + "\n"
                )

        validation = validate_build(temporary, records)
        receipt = {
            "format_version": 1,
            "source_downloaded_snapshot_count": downloads_count,
            "selected_count": len(records),
            "sort": ["community_score DESC", "id DESC"],
            "selection_sha256": selection_digest(records),
            "score_max": records[0].score,
            "score_min": records[-1].score,
            "hardlinks": linked,
            "copies": copied,
            "skipped_before_selection_complete": dict(sorted(skipped.items())),
            "validation": validation,
        }
        (temporary / "build_receipt.json").write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        # Validate once more with every final file, including the receipt, present.
        validation = validate_build(temporary, records)
        receipt["validation"] = validation
        (temporary / "build_receipt.json").write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        os.replace(temporary, destination)
        return receipt
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main() -> None:
    args = parse_args()
    if args.destination.exists():
        raise FileExistsError(
            f"destination already exists; refusing to overwrite: {args.destination}"
        )

    downloads = read_download_snapshot(args.database)
    metadata = read_metadata(args.metadata, {image_id for image_id, _, _ in downloads})
    excluded_ids, excluded_md5 = read_exclusions(args.exclude_metadata)
    records, skipped = select_records(
        downloads,
        metadata,
        args.images,
        args.count,
        excluded_ids,
        excluded_md5,
    )
    receipt = materialize(args.destination, records, len(downloads), skipped)
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
