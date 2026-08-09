#!/usr/bin/env python3
"""Freeze the Qwen3.6 caption preference handoff for declarative DPO."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

EXPECTED_COUNTS = {"train": 11_906, "validation": 680, "test": 674}
PREFERENCE_SCHEMA = "qwen36-caption-preference.v1"
DATASET_SCHEMA = "rwkv-lab.qwen-caption-dpo-dataset.v1"
TARGET_SCHEMA = "rwkv-lab.qwen-caption-lora-targets.v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def rows(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: malformed JSON") from error
            if not isinstance(value, dict):
                raise TypeError(f"{path}:{line_number}: row is not an object")
            yield value


def safe_image_relative(row: Mapping[str, Any]) -> Path:
    value = row.get("file_name")
    relative = Path(value) if isinstance(value, str) else Path()
    if (
        relative.is_absolute()
        or relative.parts[:1] != ("images",)
        or ".." in relative.parts
        or relative != Path(*relative.parts)
    ):
        raise ValueError(f"unsafe image identity: {value!r}")
    return relative


def freeze_image(source: Path, destination: Path) -> str:
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        source_stat = source.stat()
        destination_stat = destination.stat()
        if (source_stat.st_dev, source_stat.st_ino) != (
            destination_stat.st_dev,
            destination_stat.st_ino,
        ):
            raise FileExistsError(destination)
        return "existing_hardlink"
    try:
        os.link(source, destination)
        return "hardlink"
    except OSError:
        shutil.copy2(source, destination)
        return "copy"


def preference_row(row: dict[str, Any], destination: Path) -> dict[str, Any]:
    if row.get("schema") != PREFERENCE_SCHEMA or row.get("split") != "train":
        raise ValueError("preference row has the wrong schema or split")
    for field in ("id", "chosen", "rejected", "image"):
        if not isinstance(row.get(field), str) or not row[field].strip():
            raise ValueError(f"preference row has invalid {field}")
    for prefix in ("chosen", "rejected"):
        tokens = row.get(f"reference_{prefix}_token_ids")
        count = row.get(f"reference_{prefix}_token_count")
        logp = row.get(f"reference_{prefix}_logp_sum")
        if (
            not isinstance(tokens, list)
            or not tokens
            or any(
                not isinstance(token, int)
                or isinstance(token, bool)
                or token < 0
                for token in tokens
            )
            or count != len(tokens)
            or not isinstance(logp, (float, int))
            or isinstance(logp, bool)
            or not math.isfinite(float(logp))
        ):
            raise ValueError(f"preference row has invalid {prefix} reference payload")
    relative = safe_image_relative(row)
    source = Path(row["image"])
    freeze_image(source, destination / relative)
    return {
        **row,
        "caption": row["chosen"].strip(),
        "image": str((destination / relative).resolve()),
    }


def heldout_row(
    row: dict[str, Any], split: str, destination: Path
) -> dict[str, Any]:
    if row.get("split") != split:
        raise ValueError(f"held-out row is not in {split}")
    caption = row.get("caption")
    if (
        not isinstance(row.get("id"), str)
        or not row["id"]
        or not isinstance(caption, str)
        or not caption.strip()
        or not isinstance(row.get("image"), str)
    ):
        raise ValueError("held-out row is incomplete")
    relative = safe_image_relative(row)
    freeze_image(Path(row["image"]), destination / relative)
    # Preference-only values are present to keep one exact declared schema.
    # The engine never reads them outside the training split.
    return {
        **row,
        "image": str((destination / relative).resolve()),
        "chosen": caption.strip(),
        "rejected": caption.strip(),
        "reference_chosen_logp_sum": 0.0,
        "reference_rejected_logp_sum": 0.0,
        "reference_chosen_token_ids": [],
        "reference_rejected_token_ids": [],
    }


def write_jsonl(path: Path, values: Iterator[dict[str, Any]]) -> int:
    count = 0
    with path.open("x", encoding="utf-8") as output:
        for row in values:
            output.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
        output.flush()
        os.fsync(output.fileno())
    return count


def target_manifest(source: Path, model: Path, destination: Path) -> dict[str, Any]:
    document = json.loads(source.read_text(encoding="utf-8"))
    if document.get("schema") != TARGET_SCHEMA or document.get("target_count") != 311:
        raise ValueError("source target manifest is incompatible")
    document.pop("target_digest", None)
    document["model_config_sha256"] = sha256(model / "config.json")
    document["weight_index_sha256"] = sha256(
        model / "model.safetensors.index.json"
    )
    document["target_digest"] = canonical_sha256(document)
    destination.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return document


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    preference = Path(args.preference_root).resolve()
    caption = Path(args.caption_root).resolve()
    model = Path(args.model).resolve()
    destination = Path(args.destination).resolve()
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"refusing nonempty destination: {destination}")
    destination.mkdir(parents=True, exist_ok=True)

    preference_manifest = json.loads(
        (preference / "manifest.json").read_text(encoding="utf-8")
    )
    train_source = preference / "train-preferences.jsonl"
    final_receipt = preference_manifest.get("final", {})
    if (
        final_receipt.get("rows") != EXPECTED_COUNTS["train"]
        or final_receipt.get("sha256") != sha256(train_source)
    ):
        raise ValueError("preference handoff receipt disagrees with training JSONL")

    source_caption_manifest = json.loads(
        (caption / "manifest.json").read_text(encoding="utf-8")
    )
    seen: set[str] = set()

    def unique(values: Iterator[dict[str, Any]]) -> Iterator[dict[str, Any]]:
        for row in values:
            identifier = row.get("id")
            if not isinstance(identifier, str) or identifier in seen:
                raise ValueError("dataset contains an empty or duplicate ID")
            seen.add(identifier)
            yield row

    counts = {
        "train": write_jsonl(
            destination / "train.jsonl",
            unique(preference_row(row, destination) for row in rows(train_source)),
        )
    }
    for split in ("validation", "test"):
        counts[split] = write_jsonl(
            destination / f"{split}.jsonl",
            unique(
                heldout_row(row, split, destination)
                for row in rows(caption / f"{split}.jsonl")
            ),
        )
    if counts != EXPECTED_COUNTS:
        raise ValueError(f"snapshot counts disagree: {counts}")
    shutil.copy2(
        caption / "validation-fixed-100.jsonl",
        destination / "validation-fixed-100.jsonl",
    )

    targets = target_manifest(
        caption / "lora-targets.json", model, destination / "lora-targets.json"
    )
    files = {
        f"{split}.jsonl": {
            "rows": count,
            "sha256": sha256(destination / f"{split}.jsonl"),
        }
        for split, count in counts.items()
    }
    files["validation-fixed-100.jsonl"] = {
        "rows": 100,
        "sha256": sha256(destination / "validation-fixed-100.jsonl"),
    }
    manifest: dict[str, Any] = {
        "schema": DATASET_SCHEMA,
        "counts": counts,
        "files": files,
        "unique_content_hashes": len(seen),
        "training_preference_source": {
            "path": str(train_source),
            "sha256": sha256(train_source),
            "manifest_sha256": sha256(preference / "manifest.json"),
        },
        "held_out_caption_source": {
            "root": str(caption),
            "dataset_digest": source_caption_manifest["dataset_digest"],
            "validation_rows": EXPECTED_COUNTS["validation"],
            "test_rows": EXPECTED_COUNTS["test"],
        },
        "reference_model": {
            "path": str(model),
            "merge_receipt_sha256": sha256(model / "merge-receipt.json"),
            "dtype": "bfloat16",
        },
        "lora_targets": {
            "path": str(destination / "lora-targets.json"),
            "target_digest": targets["target_digest"],
            "target_count": targets["target_count"],
        },
        "split_policy": (
            "all preference pairs train; original caption validation/test remain held out"
        ),
        "source_modified": False,
    }
    manifest["dataset_digest"] = canonical_sha256(manifest)
    (destination / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preference-root", required=True)
    parser.add_argument("--caption-root", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--destination", required=True)
    print(json.dumps(prepare(parser.parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
