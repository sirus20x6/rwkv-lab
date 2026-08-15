#!/usr/bin/env python3
"""Remove unreliable DeepGHS classification fields from the I1 manifests."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from collections import Counter
from pathlib import Path


CLASSIFICATION_FIELDS = {
    "deepghs_image_type",
    "deepghs_image_type_confidence",
    "deepghs_image_type_scores",
    "deepghs_anime_real",
    "deepghs_anime_real_confidence",
    "deepghs_anime_real_scores",
    "classifier_caption",
    "caption_with_classifiers",
    "quality_viewer_manual_classification",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("/workspace/datasets/i1"),
    )
    return parser.parse_args()


def ensure_backup(source: Path, backup: Path) -> None:
    if backup.exists():
        return
    backup.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, backup)
    except OSError:
        shutil.copy2(source, backup)


def stripped_copy(source: Path) -> tuple[Path, int, Counter[str]]:
    temporary = source.with_name(
        f".{source.name}.strip-classification.{os.getpid()}.tmp"
    )
    rows = 0
    removed: Counter[str] = Counter()
    try:
        with (
            source.open(encoding="utf-8") as input_file,
            temporary.open("w", encoding="utf-8", buffering=1024 * 1024) as output,
        ):
            for line_no, line in enumerate(input_file, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise RuntimeError(f"{source}:{line_no}: malformed JSON") from exc
                for field in CLASSIFICATION_FIELDS:
                    if field in row:
                        row.pop(field)
                        removed[field] += 1
                if CLASSIFICATION_FIELDS.intersection(row):
                    raise RuntimeError(
                        f"{source}:{line_no}: classification field survived"
                    )
                output.write(json.dumps(row, ensure_ascii=False) + "\n")
                rows += 1
            output.flush()
            os.fsync(output.fileno())
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return temporary, rows, removed


def archive_artifacts(root: Path) -> list[dict[str, str]]:
    archive = root / "work" / "removed_classification_artifacts"
    candidates = [
        root / "deepghs_classification_receipt.json",
        root / "work" / "deepghs_classification.partial.jsonl",
        root / "work" / "quality_viewer_reclassifications.jsonl",
    ]
    moved: list[dict[str, str]] = []
    for source in candidates:
        if not source.exists():
            continue
        destination = archive / source.relative_to(root)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            raise RuntimeError(f"archive target already exists: {destination}")
        os.replace(source, destination)
        moved.append({"from": str(source), "to": str(destination)})
    return moved


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    sources = (root / "metadata.jsonl", root / "train.jsonl")
    backup_root = root / "work" / "pre_strip_classification_manifests"
    for source in sources:
        ensure_backup(source, backup_root / source.name)
    written: list[tuple[Path, Path, int, Counter[str]]] = []
    try:
        for source in sources:
            temporary, rows, removed = stripped_copy(source)
            written.append((source, temporary, rows, removed))
        results = [(rows, removed) for _, _, rows, removed in written]
        if results[0] != results[1]:
            raise RuntimeError("metadata/train stripping results do not match")
        for source, temporary, _, _ in written:
            os.replace(temporary, source)
    except Exception:
        for _, temporary, _, _ in written:
            temporary.unlink(missing_ok=True)
        raise
    moved = archive_artifacts(root)
    rows, removed = results[0]
    print(
        json.dumps(
            {
                "root": str(root),
                "rows_per_manifest": rows,
                "removed_fields": dict(sorted(removed.items())),
                "backup": str(backup_root),
                "archived_artifacts": moved,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
