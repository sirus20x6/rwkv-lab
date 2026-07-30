#!/usr/bin/env python3
"""Prepare canonical Mage-Flow expert train/eval manifests from frozen exports."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rwkv_lab.mage_flow_adaptation import prepare_domain_manifest

DEFAULT_OUTPUT = Path("/thearray/git/datasets/mageflow-expert-manifests")


@dataclass(frozen=True)
class Export:
    root: Path
    domain: str
    source: str
    caption_provenance: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--photo-train",
        type=Path,
        default=Path("/thearray/git/datasets/porn/reddit/trainer-ready-3500"),
    )
    parser.add_argument(
        "--animation-train",
        type=Path,
        default=Path("/thearray/git/datasets/gelbooru-trainer-all"),
    )
    parser.add_argument(
        "--animation-only-train",
        action="store_true",
        help="omit photo rows from training while retaining mixed-domain evaluation",
    )
    parser.add_argument("--animation-source", default="gelbooru_masterpiece")
    parser.add_argument("--animation-caption-provenance", default="gelbooru_tags")
    parser.add_argument(
        "--animation-holdout-count",
        type=int,
        default=0,
        help=(
            "deterministically remove this many rows from animation training and "
            "use them instead of --animation-eval"
        ),
    )
    parser.add_argument("--animation-holdout-seed", type=int, default=42)
    parser.add_argument("--max-aspect-ratio", type=float, default=4.0)
    parser.add_argument(
        "--photo-eval",
        type=Path,
        default=Path("/thearray/git/datasets/porn/reddit/trainer-eval-128"),
    )
    parser.add_argument(
        "--animation-eval",
        type=Path,
        default=Path("/thearray/git/datasets/gelbooru-eval-128"),
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--no-verify-images",
        action="store_true",
        help="skip full pixel decode while retaining header and geometry validation",
    )
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_export(export: Export) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    root = export.root.expanduser().resolve()
    metadata_path = root / "metadata.jsonl"
    rows: list[dict[str, Any]] = []
    with metadata_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                metadata = json.loads(line)
                file_name = str(metadata["file_name"])
                caption = str(metadata["text"]).strip()
            except (json.JSONDecodeError, KeyError, TypeError) as exc:
                raise RuntimeError(
                    f"invalid metadata in {metadata_path} at line {line_number}"
                ) from exc
            image = (root / file_name).resolve()
            if not image.is_relative_to(root) or not image.is_file():
                raise RuntimeError(f"missing or unsafe image path: {file_name}")
            if not caption:
                raise RuntimeError(f"empty caption at {metadata_path}:{line_number}")
            rows.append(
                {
                    "image": str(image),
                    "domain": export.domain,
                    "source": export.source,
                    "caption": caption,
                    "caption_provenance": export.caption_provenance,
                    **(
                        {"width": int(metadata["width"])}
                        if int(metadata.get("width") or 0) > 0
                        else {}
                    ),
                    **(
                        {"height": int(metadata["height"])}
                        if int(metadata.get("height") or 0) > 0
                        else {}
                    ),
                    **(
                        {"image_sha256": str(metadata["sha256"])}
                        if str(metadata.get("sha256") or "").strip()
                        else {}
                    ),
                    **(
                        {"content_class": str(metadata["content_class"])}
                        if str(metadata.get("content_class") or "").strip()
                        else {}
                    ),
                }
            )
    return rows, {
        "root": str(root),
        "metadata": str(metadata_path),
        "metadata_sha256": file_sha256(metadata_path),
        "records": len(rows),
        "domain": export.domain,
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            )


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def split_balanced_animation_holdout(
    rows: list[dict[str, Any]],
    *,
    count: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    """Select a stable SFW/NSFW holdout without depending on input ordering."""
    if count < 0 or count % 2:
        raise ValueError("animation holdout count must be a nonnegative even number")
    if count == 0:
        return rows, [], {}
    per_class = count // 2
    selected: list[dict[str, Any]] = []
    class_counts: dict[str, int] = {}
    for content_class in ("sfw", "nsfw"):
        candidates = [
            row for row in rows if row.get("content_class") == content_class
        ]
        if len(candidates) < per_class:
            raise ValueError(
                f"animation holdout needs {per_class} {content_class} rows, "
                f"found {len(candidates)}"
            )
        candidates.sort(
            key=lambda row: hashlib.sha256(
                f"{seed}:{row.get('image_sha256')}:{row['image']}".encode()
            ).digest()
        )
        chosen = candidates[:per_class]
        selected.extend(chosen)
        class_counts[content_class] = len(chosen)
    selected_ids = {
        str(row.get("image_sha256") or row["image"]) for row in selected
    }
    training = [
        row
        for row in rows
        if str(row.get("image_sha256") or row["image"]) not in selected_ids
    ]
    return training, selected, class_counts


def main() -> None:
    args = parse_args()
    destination = args.output_dir.expanduser().resolve()
    if destination.exists():
        raise FileExistsError(
            f"destination already exists; refusing to overwrite: {destination}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)

    animation_train = Export(
        args.animation_train,
        "animation",
        args.animation_source,
        args.animation_caption_provenance,
    )
    train_exports = (
        (animation_train,)
        if args.animation_only_train
        else (
            Export(args.photo_train, "photo", "reddit_qwen36_audit", "audit"),
            animation_train,
        )
    )
    eval_exports = (
        Export(args.photo_eval, "photo", "reddit_qwen36_audit", "audit"),
        Export(
            args.animation_eval,
            "animation",
            "gelbooru_masterpiece",
            "gelbooru_tags",
        ),
    )

    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent)
    )
    try:
        source_info: dict[str, list[dict[str, Any]]] = {"train": [], "eval": []}
        raw_train: list[dict[str, Any]] = []
        raw_eval: list[dict[str, Any]] = []
        for export in train_exports:
            rows, info = read_export(export)
            raw_train.extend(rows)
            source_info["train"].append(info)

        raw_train, animation_holdout, holdout_class_counts = (
            split_balanced_animation_holdout(
                raw_train,
                count=args.animation_holdout_count,
                seed=args.animation_holdout_seed,
            )
        )
        photo_eval_rows, photo_eval_info = read_export(eval_exports[0])
        raw_eval.extend(photo_eval_rows)
        source_info["eval"].append(photo_eval_info)
        if animation_holdout:
            raw_eval.extend(animation_holdout)
            source_info["eval"].append(
                {
                    "root": str(args.animation_train.expanduser().resolve()),
                    "records": len(animation_holdout),
                    "domain": "animation",
                    "source": args.animation_source,
                    "split_from_training": True,
                    "content_class_counts": holdout_class_counts,
                    "seed": args.animation_holdout_seed,
                }
            )
        else:
            animation_eval_rows, animation_eval_info = read_export(eval_exports[1])
            raw_eval.extend(animation_eval_rows)
            source_info["eval"].append(animation_eval_info)

        raw_train_path = temporary / "_raw_train.jsonl"
        raw_eval_path = temporary / "_raw_eval.jsonl"
        write_jsonl(raw_train_path, raw_train)
        write_jsonl(raw_eval_path, raw_eval)
        train_report = prepare_domain_manifest(
            raw_train_path,
            temporary / "train.jsonl",
            data_root=Path("/"),
            max_aspect_ratio=args.max_aspect_ratio,
            verify_images=not args.no_verify_images,
        )
        eval_report = prepare_domain_manifest(
            raw_eval_path,
            temporary / "eval.jsonl",
            data_root=Path("/"),
            max_aspect_ratio=args.max_aspect_ratio,
            verify_images=not args.no_verify_images,
        )
        raw_train_path.unlink()
        raw_eval_path.unlink()

        train_rows = load_jsonl(temporary / "train.jsonl")
        eval_rows = load_jsonl(temporary / "eval.jsonl")
        expected_train = len(raw_train)
        expected_eval = len(raw_eval)
        if len(eval_rows) != expected_eval:
            raise RuntimeError(
                f"trainer canonicalization retained {len(eval_rows):,} of "
                f"{expected_eval:,} eval images"
            )

        train_ids = {str(row["image_id"]) for row in train_rows}
        eval_ids = {str(row["image_id"]) for row in eval_rows}
        overlap = train_ids & eval_ids
        if overlap:
            raise RuntimeError(
                f"{len(overlap):,} exact image hashes leak between train and eval"
            )

        receipt = {
            "format_version": 1,
            "train_count": len(train_rows),
            "eval_count": len(eval_rows),
            "train_input_count": expected_train,
            "eval_input_count": expected_eval,
            "full_pixel_decode": not args.no_verify_images,
            "max_aspect_ratio": args.max_aspect_ratio,
            "train_canonicalization_counts": train_report["counts"],
            "eval_canonicalization_counts": eval_report["counts"],
            "train_domain_counts": train_report["audit"]["domain_counts"],
            "eval_domain_counts": eval_report["audit"]["domain_counts"],
            "cross_split_duplicate_images": 0,
            "animation_holdout_count": len(animation_holdout),
            "animation_holdout_seed": args.animation_holdout_seed,
            "animation_holdout_class_counts": holdout_class_counts,
            "train_manifest_sha256": file_sha256(temporary / "train.jsonl"),
            "eval_manifest_sha256": file_sha256(temporary / "eval.jsonl"),
            "sources": source_info,
        }
        (temporary / "build_receipt.json").write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        os.replace(temporary, destination)
        print(json.dumps(receipt, indent=2, sort_keys=True))
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


if __name__ == "__main__":
    main()
