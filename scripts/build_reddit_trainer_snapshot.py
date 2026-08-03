#!/usr/bin/env python3
"""Export a frozen, manifest-ordered Reddit image/caption snapshot."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import sys
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any


DEFAULT_ROOT = Path("/thearray/git/datasets/porn/reddit/subreddits")
DEFAULT_CAPTIONER = Path(
    "/thearray/git/datasets/captiontest/run_cached_multistage_captioner.py"
)
DEFAULT_PREFIX = "qwen3.6-multistage-38051"
DEFAULT_DESTINATION = Path("/thearray/git/datasets/porn/reddit/trainer-ready-3500")


@dataclass(frozen=True)
class FrozenRecord:
    relative_path: str
    source: Path
    caption: str

    def json_record(self) -> dict[str, str]:
        return {
            "file_name": self.relative_path,
            "text": self.caption,
            "source_relative_path": self.relative_path,
        }


class FrozenCheckpointStore:
    """The read-only subset of CheckpointStore used by image_disposition()."""

    def __init__(self, rows: dict[str, dict[str, dict[str, Any]]]):
        self.rows = rows

    def get(self, stage: str, relative_path: str) -> dict[str, Any] | None:
        return self.rows.get(stage, {}).get(relative_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--captioner", type=Path, default=DEFAULT_CAPTIONER)
    parser.add_argument("--prefix", default=DEFAULT_PREFIX)
    parser.add_argument("--destination", type=Path, default=DEFAULT_DESTINATION)
    parser.add_argument("--count", type=int, default=3500)
    parser.add_argument(
        "--exclude-metadata",
        type=Path,
        action="append",
        default=[],
        help="metadata.jsonl whose source paths must not be selected",
    )
    return parser.parse_args()


def import_captioner(path: Path) -> ModuleType:
    name = "_frozen_multistage_captioner"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not import captioner: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    if not callable(getattr(module, "image_disposition", None)):
        raise RuntimeError("captioner does not define image_disposition()")
    return module


def read_frozen_prefix(
    path: Path,
    *,
    size: int | None = None,
) -> tuple[bytes, dict[str, Any]]:
    if size is None:
        size = path.stat().st_size
    with path.open("rb") as handle:
        content = handle.read(size)
    if len(content) != size:
        raise RuntimeError(f"checkpoint shrank while being snapshotted: {path}")
    return content, {
        "bytes": size,
        "sha256": hashlib.sha256(content).hexdigest(),
    }


def parse_checkpoint_snapshot(
    path: Path,
    content: bytes,
) -> tuple[dict[str, dict[str, Any]], int]:
    rows: dict[str, dict[str, Any]] = {}
    lines = content.splitlines()
    nonempty_indexes = [index for index, line in enumerate(lines) if line.strip()]
    final_nonempty_index = nonempty_indexes[-1] if nonempty_indexes else None
    ignored_final_lines = 0
    for index, raw_line in enumerate(lines):
        if not raw_line.strip():
            continue
        try:
            row = json.loads(raw_line)
            relative_path = row["relative_path"]
            if not isinstance(row, dict):
                raise TypeError("checkpoint row is not an object")
            if not isinstance(relative_path, str) or not relative_path:
                raise TypeError("relative_path is not a nonempty string")
        except (UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError) as exc:
            if index == final_nonempty_index:
                ignored_final_lines += 1
                continue
            raise RuntimeError(
                f"malformed checkpoint record in {path} at line {index + 1}"
            ) from exc
        rows[relative_path] = row
    return rows, ignored_final_lines


def snapshot_checkpoints(
    root: Path,
    prefix: str,
    captioner: ModuleType,
) -> tuple[FrozenCheckpointStore, dict[str, Any]]:
    expected_stages = tuple(str(stage) for stage in captioner.ALL_STAGE_NAMES)
    paths = sorted(root.glob(f"{prefix}.*.partial.jsonl"))
    stage_paths: dict[str, Path] = {}
    marker = f"{prefix}."
    suffix = ".partial.jsonl"
    for path in paths:
        stage = path.name[len(marker) : -len(suffix)]
        if not stage or stage in stage_paths:
            raise RuntimeError(f"ambiguous checkpoint stage path: {path}")
        stage_paths[stage] = path

    missing = sorted(set(expected_stages) - set(stage_paths))
    extra = sorted(set(stage_paths) - set(expected_stages))
    if missing or extra:
        raise RuntimeError(
            f"checkpoint stage mismatch: missing={missing}, extra={extra}"
        )

    stage_rows: dict[str, dict[str, dict[str, Any]]] = {}
    snapshot_info: dict[str, Any] = {}
    # Capture all byte boundaries before reading any checkpoint so the stage
    # files form one logical snapshot even while the captioner keeps appending.
    frozen_sizes = {
        stage: stage_paths[stage].stat().st_size for stage in expected_stages
    }
    for stage in expected_stages:
        content, info = read_frozen_prefix(
            stage_paths[stage],
            size=frozen_sizes[stage],
        )
        rows, ignored = parse_checkpoint_snapshot(stage_paths[stage], content)
        stage_rows[stage] = rows
        snapshot_info[stage] = {
            **info,
            "records_after_collapse": len(rows),
            "ignored_malformed_final_lines": ignored,
        }
    return FrozenCheckpointStore(stage_rows), snapshot_info


def read_manifest_snapshot(path: Path) -> tuple[list[str], dict[str, Any]]:
    content, info = read_frozen_prefix(path)
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise RuntimeError(f"manifest is not UTF-8: {path}") from exc
    manifest = [line.strip() for line in text.splitlines() if line.strip()]
    if len(manifest) != len(set(manifest)):
        raise RuntimeError("manifest contains duplicate relative paths")
    return manifest, {**info, "records": len(manifest)}


def safe_source(root: Path, relative_path: str) -> Path | None:
    relative = Path(relative_path)
    if relative.is_absolute() or ".." in relative.parts:
        return None
    try:
        source = (root / relative).resolve(strict=True)
        root_resolved = root.resolve(strict=True)
    except (FileNotFoundError, OSError):
        return None
    if not source.is_relative_to(root_resolved) or not source.is_file():
        return None
    return source


def read_excluded_paths(paths: list[Path]) -> set[str]:
    excluded: set[str] = set()
    for path in paths:
        with path.expanduser().resolve().open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                    relative_path = row.get("source_relative_path") or row.get(
                        "file_name"
                    )
                except (json.JSONDecodeError, KeyError, TypeError) as exc:
                    raise RuntimeError(
                        f"invalid exclusion row in {path} at line {line_number}"
                    ) from exc
                if not isinstance(relative_path, str) or not relative_path:
                    raise RuntimeError(
                        f"invalid exclusion path in {path} at line {line_number}"
                    )
                excluded.add(relative_path)
    return excluded


def freeze_selection(
    *,
    root: Path,
    manifest: list[str],
    checkpoints: FrozenCheckpointStore,
    image_disposition: Any,
    count: int,
    excluded_paths: set[str] | None = None,
) -> tuple[list[FrozenRecord], Counter[str]]:
    selected: list[FrozenRecord] = []
    rejected: Counter[str] = Counter()
    sidecar_paths: set[str] = set()
    excluded_paths = excluded_paths or set()

    for relative_path in manifest:
        if relative_path in excluded_paths:
            rejected["excluded_training_path"] += 1
            continue
        disposition = image_disposition(checkpoints, relative_path)
        if disposition != "complete":
            rejected[disposition or "incomplete"] += 1
            continue

        audit = checkpoints.get("audit", relative_path)
        result = audit.get("result") if isinstance(audit, dict) else None
        caption_value = result.get("caption") if isinstance(result, dict) else None
        caption = caption_value.strip() if isinstance(caption_value, str) else ""
        if not caption:
            rejected["empty_audit_caption"] += 1
            continue

        source = safe_source(root, relative_path)
        if source is None:
            rejected["missing_or_invalid_source"] += 1
            continue

        sidecar_path = str(Path(relative_path).with_suffix(".txt"))
        if sidecar_path in sidecar_paths:
            raise RuntimeError(f"sidecar path collision: {sidecar_path}")
        sidecar_paths.add(sidecar_path)
        selected.append(
            FrozenRecord(
                relative_path=relative_path,
                source=source,
                caption=caption,
            )
        )
        if len(selected) == count:
            break

    if len(selected) != count:
        raise RuntimeError(
            f"only {len(selected):,} eligible records found; {count:,} required"
        )
    return selected, rejected


def selection_digest(records: list[FrozenRecord]) -> str:
    digest = hashlib.sha256()
    for record in records:
        digest.update(
            f"{record.relative_path}\t{record.caption}\n".encode("utf-8")
        )
    return digest.hexdigest()


def validate_export(
    root: Path,
    records: list[FrozenRecord],
    checkpoints: FrozenCheckpointStore,
    image_disposition: Any,
) -> dict[str, Any]:
    metadata_lines = (root / "metadata.jsonl").read_text(
        encoding="utf-8"
    ).splitlines()
    if len(metadata_lines) != len(records):
        raise RuntimeError("metadata.jsonl record count does not match selection")
    metadata = [json.loads(line) for line in metadata_lines]
    expected_metadata = [record.json_record() for record in records]
    if metadata != expected_metadata:
        raise RuntimeError("metadata.jsonl does not exactly match frozen selection")

    expected_images = {record.relative_path for record in records}
    expected_sidecars = {
        str(Path(record.relative_path).with_suffix(".txt")) for record in records
    }
    actual_images: set[str] = set()
    actual_sidecars: set[str] = set()
    for path in root.rglob("*"):
        if not path.is_file() or path.name in ("metadata.jsonl", "build_receipt.json"):
            continue
        relative = path.relative_to(root).as_posix()
        if path.suffix.lower() == ".txt":
            actual_sidecars.add(relative)
        else:
            actual_images.add(relative)
    if actual_images != expected_images:
        raise RuntimeError("exported image paths do not match frozen selection")
    if actual_sidecars != expected_sidecars:
        raise RuntimeError("exported sidecars do not match frozen selection")

    for record in records:
        if image_disposition(checkpoints, record.relative_path) != "complete":
            raise RuntimeError(
                f"frozen disposition changed for {record.relative_path}"
            )
        output = root / record.relative_path
        sidecar = output.with_suffix(".txt")
        if not output.is_file():
            raise RuntimeError(f"missing output image: {record.relative_path}")
        if sidecar.read_text(encoding="utf-8") != record.caption + "\n":
            raise RuntimeError(f"caption mismatch: {record.relative_path}")

    extensions = Counter(Path(path).suffix.lower() for path in actual_images)
    file_bytes = sum(path.stat().st_size for path in root.rglob("*") if path.is_file())
    return {
        "images": len(actual_images),
        "sidecars": len(actual_sidecars),
        "metadata_records": len(metadata),
        "empty_captions": sum(not row["text"].strip() for row in metadata),
        "non_complete_dispositions": 0,
        "extension_counts": dict(sorted(extensions.items())),
        "apparent_bytes": file_bytes,
    }


def materialize(
    *,
    destination: Path,
    records: list[FrozenRecord],
    checkpoints: FrozenCheckpointStore,
    image_disposition: Any,
    manifest_info: dict[str, Any],
    checkpoint_info: dict[str, Any],
    rejected: Counter[str],
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
    hardlinks = 0
    copies = 0
    try:
        with (temporary / "metadata.jsonl").open(
            "w", encoding="utf-8", newline="\n"
        ) as metadata_file:
            for record in records:
                output = temporary / record.relative_path
                output.parent.mkdir(parents=True, exist_ok=True)
                try:
                    os.link(record.source, output)
                    hardlinks += 1
                except OSError:
                    shutil.copy2(record.source, output)
                    copies += 1
                output.with_suffix(".txt").write_text(
                    record.caption + "\n",
                    encoding="utf-8",
                    newline="\n",
                )
                metadata_file.write(
                    json.dumps(
                        record.json_record(),
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                    + "\n"
                )

        validation = validate_export(
            temporary, records, checkpoints, image_disposition
        )
        receipt = {
            "format_version": 1,
            "selected_count": len(records),
            "selection_sha256": selection_digest(records),
            "hardlinks": hardlinks,
            "copies": copies,
            "manifest_snapshot": manifest_info,
            "checkpoint_snapshots": checkpoint_info,
            "rejected_before_selection_complete": dict(sorted(rejected.items())),
            "validation": validation,
        }
        (temporary / "build_receipt.json").write_text(
            json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        validation = validate_export(
            temporary, records, checkpoints, image_disposition
        )
        receipt["validation"] = validation
        (temporary / "build_receipt.json").write_text(
            json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
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
    if args.count <= 0:
        raise ValueError("--count must be positive")
    if args.destination.exists():
        raise FileExistsError(
            f"destination already exists; refusing to overwrite: {args.destination}"
        )

    captioner = import_captioner(args.captioner)
    checkpoints, checkpoint_info = snapshot_checkpoints(
        args.root, args.prefix, captioner
    )
    manifest, manifest_info = read_manifest_snapshot(
        args.root / f"{args.prefix}.manifest.txt"
    )
    excluded_paths = read_excluded_paths(args.exclude_metadata)
    records, rejected = freeze_selection(
        root=args.root,
        manifest=manifest,
        checkpoints=checkpoints,
        image_disposition=captioner.image_disposition,
        count=args.count,
        excluded_paths=excluded_paths,
    )
    receipt = materialize(
        destination=args.destination,
        records=records,
        checkpoints=checkpoints,
        image_disposition=captioner.image_disposition,
        manifest_info=manifest_info,
        checkpoint_info=checkpoint_info,
        rejected=rejected,
    )
    print(json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
