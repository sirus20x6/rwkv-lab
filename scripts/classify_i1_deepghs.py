#!/usr/bin/env python3
"""Add opt-in DeepGHS classification captions to an I1 dataset snapshot.

The original ``caption``, ``text``, and ``conditioning_text`` fields are never
changed.  Classification is first written to an append-only cache.  Only after
every image has a valid result are metadata.jsonl and train.jsonl rewritten
atomically.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import shutil
import sys
import time
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from PIL import Image, ImageOps

IMAGE_TYPE_REPO = "deepghs/anime_classification"
IMAGE_TYPE_REVISION = "5ee62e06f5f4cd68a1c2f3bc5dc9805e827d37df"
IMAGE_TYPE_MODEL = "mobilenetv3_v1.5_dist"
IMAGE_TYPE_LABELS = ("3d", "bangumi", "comic", "illustration", "not_painting")

ANIME_REAL_REPO = "deepghs/anime_real_cls"
ANIME_REAL_REVISION = "097a6c1e9e62866a435810c883a45a7cb2c7077d"
ANIME_REAL_MODEL = "mobilenetv3_v1.4_dist"
ANIME_REAL_LABELS = ("anime", "real")

INPUT_SIZE = 384
SCHEMA_VERSION = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("/workspace/git/datasets/i1"),
        help="Dataset root containing metadata.jsonl and train.jsonl.",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--decode-workers", type=int, default=24)
    parser.add_argument("--checkpoint-every", type=int, default=2048)
    parser.add_argument(
        "--provider",
        choices=("cuda", "cpu"),
        default="cuda",
        help="ONNX Runtime execution provider.",
    )
    parser.add_argument(
        "--cache",
        type=Path,
        default=None,
        help="Append-only classification cache (defaults below ROOT/work).",
    )
    parser.add_argument(
        "--classify-only",
        action="store_true",
        help="Populate and validate the cache without rewriting manifests.",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"{path}:{line_no}: malformed JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise TypeError(f"{path}:{line_no}: expected a JSON object")
            rows.append(row)
    return rows


def image_key(row: dict[str, Any]) -> str:
    value = row.get("image_sha256") or row.get("image_id")
    if not isinstance(value, str) or not value:
        image = row.get("image")
        if not isinstance(image, str) or not image:
            raise RuntimeError("Manifest row has no image_sha256, image_id, or image path")
        value = hashlib.sha256(image.encode("utf-8")).hexdigest()
    return value


def load_cached_results(path: Path) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return results
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                # The only tolerable corruption is an interrupted final append.
                if any(rest.strip() for rest in handle):
                    raise RuntimeError(f"{path}:{line_no}: malformed non-final cache line")
                break
            if valid_result(item):
                results[item["key"]] = item
    return results


def valid_scores(value: Any, labels: tuple[str, ...]) -> bool:
    if not isinstance(value, dict) or set(value) != set(labels):
        return False
    scores = np.asarray([value[label] for label in labels], dtype=np.float64)
    return bool(
        np.isfinite(scores).all()
        and (scores >= -1e-6).all()
        and (scores <= 1.0 + 1e-6).all()
        and abs(float(scores.sum()) - 1.0) <= 2e-3
    )


def valid_result(item: Any) -> bool:
    return bool(
        isinstance(item, dict)
        and isinstance(item.get("key"), str)
        and item.get("image_type") in IMAGE_TYPE_LABELS
        and item.get("anime_real") in ANIME_REAL_LABELS
        and valid_scores(item.get("image_type_scores"), IMAGE_TYPE_LABELS)
        and valid_scores(item.get("anime_real_scores"), ANIME_REAL_LABELS)
    )


def load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source)
        if image.mode in ("RGBA", "LA") or "transparency" in image.info:
            rgba = image.convert("RGBA")
            background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
            image = Image.alpha_composite(background, rgba).convert("RGB")
        else:
            image = image.convert("RGB")
        image = image.resize(
            (INPUT_SIZE, INPUT_SIZE),
            resample=Image.Resampling.BILINEAR,
        )
        array = np.asarray(image, dtype=np.float32)
    array = np.transpose(array, (2, 0, 1))
    return np.ascontiguousarray(array / 127.5 - 1.0, dtype=np.float32)


def load_batch(
    paths: list[Path],
    executor: concurrent.futures.Executor,
) -> np.ndarray:
    arrays = list(executor.map(load_rgb, paths))
    return np.stack(arrays, axis=0)


def resolve_model(repo: str, revision: str, model: str) -> Path:
    return Path(
        hf_hub_download(
            repo_id=repo,
            filename=f"{model}/model.onnx",
            revision=revision,
        )
    )


def make_session(path: Path, provider: str) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    if provider == "cuda":
        if "CUDAExecutionProvider" not in ort.get_available_providers():
            raise RuntimeError("CUDAExecutionProvider is unavailable in ONNX Runtime")
        providers: list[Any] = [
            (
                "CUDAExecutionProvider",
                {
                    "cudnn_conv_algo_search": "HEURISTIC",
                    "do_copy_in_default_stream": "1",
                },
            ),
            "CPUExecutionProvider",
        ]
    else:
        providers = ["CPUExecutionProvider"]
    session = ort.InferenceSession(str(path), sess_options=options, providers=providers)
    expected = (
        "CUDAExecutionProvider" if provider == "cuda" else "CPUExecutionProvider"
    )
    if session.get_providers()[0] != expected:
        raise RuntimeError(
            f"{path}: requested {expected}, got providers {session.get_providers()}"
        )
    return session


def score_dict(labels: tuple[str, ...], values: np.ndarray) -> dict[str, float]:
    return {label: float(values[index]) for index, label in enumerate(labels)}


def batched(values: list[Any], size: int) -> Iterable[list[Any]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def classify(
    rows: list[dict[str, Any]],
    cache_path: Path,
    batch_size: int,
    decode_workers: int,
    checkpoint_every: int,
    provider: str,
) -> dict[str, dict[str, Any]]:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    results = load_cached_results(cache_path)
    pending = [row for row in rows if image_key(row) not in results]
    print(
        f"Classification cache: {len(results):,} complete, "
        f"{len(pending):,} pending",
        flush=True,
    )
    if not pending:
        return results

    image_type_path = resolve_model(
        IMAGE_TYPE_REPO, IMAGE_TYPE_REVISION, IMAGE_TYPE_MODEL
    )
    anime_real_path = resolve_model(
        ANIME_REAL_REPO, ANIME_REAL_REVISION, ANIME_REAL_MODEL
    )
    image_type_session = make_session(image_type_path, provider)
    anime_real_session = make_session(anime_real_path, provider)

    started = time.monotonic()
    completed_now = 0
    since_sync = 0
    with (
        concurrent.futures.ThreadPoolExecutor(max_workers=decode_workers) as executor,
        cache_path.open("a", encoding="utf-8", buffering=1024 * 1024) as output,
    ):
        for group in batched(pending, batch_size):
            paths = [Path(row["image"]) for row in group]
            for path in paths:
                if not path.is_file():
                    raise FileNotFoundError(path)
            inputs = load_batch(paths, executor)
            image_type_batch = image_type_session.run(
                None, {"input": inputs}
            )[0]
            anime_real_batch = anime_real_session.run(
                None, {"input": inputs}
            )[0]
            if image_type_batch.shape != (len(group), len(IMAGE_TYPE_LABELS)):
                raise RuntimeError(
                    f"Unexpected image-type output shape: {image_type_batch.shape}"
                )
            if anime_real_batch.shape != (len(group), len(ANIME_REAL_LABELS)):
                raise RuntimeError(
                    f"Unexpected anime-real output shape: {anime_real_batch.shape}"
                )

            for row, type_values, real_values in zip(
                group, image_type_batch, anime_real_batch, strict=True
            ):
                type_scores = score_dict(IMAGE_TYPE_LABELS, type_values)
                real_scores = score_dict(ANIME_REAL_LABELS, real_values)
                item = {
                    "schema_version": SCHEMA_VERSION,
                    "key": image_key(row),
                    "file_name": row.get("file_name"),
                    "image_type": max(type_scores, key=type_scores.__getitem__),
                    "image_type_scores": type_scores,
                    "anime_real": max(real_scores, key=real_scores.__getitem__),
                    "anime_real_scores": real_scores,
                }
                if not valid_result(item):
                    raise RuntimeError(
                        f"Invalid classifier output for {row.get('image')}: {item}"
                    )
                output.write(json.dumps(item, ensure_ascii=False) + "\n")
                results[item["key"]] = item
                completed_now += 1
                since_sync += 1

            if since_sync >= checkpoint_every:
                output.flush()
                os.fsync(output.fileno())
                since_sync = 0
            elapsed = max(time.monotonic() - started, 1e-6)
            total_complete = len(results)
            rate = completed_now / elapsed
            print(
                f"\rClassified {total_complete:,}/{len(rows):,} "
                f"({rate:.1f} images/s)",
                end="",
                file=sys.stderr,
                flush=True,
            )
        output.flush()
        os.fsync(output.fileno())
    print(file=sys.stderr)
    return results


def classifier_fields(result: dict[str, Any], original_caption: str) -> dict[str, Any]:
    image_type = result["image_type"]
    anime_real = result["anime_real"]
    classifier_caption = (
        f"image type: {image_type}; anime versus real: {anime_real}"
    )
    suffix = f"Image type: {image_type}. Anime versus real: {anime_real}."
    caption = original_caption.strip()
    return {
        "deepghs_image_type": image_type,
        "deepghs_image_type_confidence": result["image_type_scores"][image_type],
        "deepghs_image_type_scores": result["image_type_scores"],
        "deepghs_anime_real": anime_real,
        "deepghs_anime_real_confidence": result["anime_real_scores"][anime_real],
        "deepghs_anime_real_scores": result["anime_real_scores"],
        "classifier_caption": classifier_caption,
        "caption_with_classifiers": f"{caption} {suffix}".strip(),
    }


def write_jsonl_atomic(path: Path, rows: Iterable[dict[str, Any]]) -> Path:
    temporary = path.with_name(f".{path.name}.deepghs.tmp")
    with temporary.open("w", encoding="utf-8", buffering=1024 * 1024) as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    return temporary


def enrich_rows(
    rows: list[dict[str, Any]],
    results: dict[str, dict[str, Any]],
    *,
    caption_fields: tuple[str, ...],
) -> Iterable[dict[str, Any]]:
    for row in rows:
        result = results[image_key(row)]
        original_caption = ""
        for field in caption_fields:
            value = row.get(field)
            if isinstance(value, str) and value.strip():
                original_caption = value
                break
        if not original_caption:
            raise RuntimeError(f"Empty original caption for {row.get('image')}")
        enriched = dict(row)
        enriched.update(classifier_fields(result, original_caption))
        yield enriched


def ensure_backup(source: Path, backup: Path) -> None:
    backup.parent.mkdir(parents=True, exist_ok=True)
    if backup.exists():
        return
    try:
        os.link(source, backup)
    except OSError:
        shutil.copy2(source, backup)


def validate_enriched(
    path: Path,
    expected_rows: int,
) -> tuple[Counter[str], Counter[str]]:
    type_counts: Counter[str] = Counter()
    real_counts: Counter[str] = Counter()
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            row = json.loads(line)
            count += 1
            image_type = row.get("deepghs_image_type")
            anime_real = row.get("deepghs_anime_real")
            if image_type not in IMAGE_TYPE_LABELS:
                raise RuntimeError(f"{path}:{line_no}: invalid image type")
            if anime_real not in ANIME_REAL_LABELS:
                raise RuntimeError(f"{path}:{line_no}: invalid anime-real label")
            if not row.get("classifier_caption"):
                raise RuntimeError(f"{path}:{line_no}: empty classifier caption")
            if not row.get("caption_with_classifiers"):
                raise RuntimeError(f"{path}:{line_no}: empty enriched caption")
            type_counts[image_type] += 1
            real_counts[anime_real] += 1
    if count != expected_rows:
        raise RuntimeError(f"{path}: expected {expected_rows:,} rows, got {count:,}")
    return type_counts, real_counts


def main() -> int:
    args = parse_args()
    if args.batch_size < 1 or args.decode_workers < 1 or args.checkpoint_every < 1:
        raise SystemExit("Batch size, decode workers, and checkpoint interval must be > 0")

    root = args.root.resolve()
    metadata_path = root / "metadata.jsonl"
    train_path = root / "train.jsonl"
    cache_path = (
        args.cache.resolve()
        if args.cache
        else root / "work" / "deepghs_classification.partial.jsonl"
    )

    metadata_rows = read_jsonl(metadata_path)
    train_rows = read_jsonl(train_path)
    if len(metadata_rows) != len(train_rows):
        raise RuntimeError(
            f"Manifest row mismatch: metadata={len(metadata_rows):,}, "
            f"train={len(train_rows):,}"
        )
    metadata_keys = [image_key(row) for row in metadata_rows]
    train_keys = [image_key(row) for row in train_rows]
    if len(set(metadata_keys)) != len(metadata_keys):
        raise RuntimeError("metadata.jsonl contains duplicate image keys")
    if set(metadata_keys) != set(train_keys):
        raise RuntimeError("metadata.jsonl and train.jsonl image keys differ")

    results = classify(
        metadata_rows,
        cache_path,
        args.batch_size,
        args.decode_workers,
        args.checkpoint_every,
        args.provider,
    )
    missing = set(metadata_keys) - set(results)
    if missing:
        raise RuntimeError(f"Classification cache is missing {len(missing):,} rows")
    if args.classify_only:
        print(f"Validated classification cache with {len(metadata_rows):,} rows")
        return 0

    metadata_temp = write_jsonl_atomic(
        metadata_path,
        enrich_rows(metadata_rows, results, caption_fields=("caption", "text")),
    )
    train_temp = write_jsonl_atomic(
        train_path,
        enrich_rows(
            train_rows,
            results,
            caption_fields=("caption", "conditioning_text"),
        ),
    )
    metadata_counts = validate_enriched(metadata_temp, len(metadata_rows))
    train_counts = validate_enriched(train_temp, len(train_rows))
    if metadata_counts != train_counts:
        raise RuntimeError("Classifier distributions differ between output manifests")

    backup_root = root / "work" / "pre_deepghs_manifests"
    ensure_backup(metadata_path, backup_root / metadata_path.name)
    ensure_backup(train_path, backup_root / train_path.name)
    os.replace(metadata_temp, metadata_path)
    os.replace(train_temp, train_path)

    receipt = {
        "schema_version": SCHEMA_VERSION,
        "created_unix": time.time(),
        "dataset_root": str(root),
        "rows": len(metadata_rows),
        "original_caption_fields_preserved": True,
        "added_fields": [
            "deepghs_image_type",
            "deepghs_image_type_confidence",
            "deepghs_image_type_scores",
            "deepghs_anime_real",
            "deepghs_anime_real_confidence",
            "deepghs_anime_real_scores",
            "classifier_caption",
            "caption_with_classifiers",
        ],
        "image_type_model": {
            "repo": IMAGE_TYPE_REPO,
            "revision": IMAGE_TYPE_REVISION,
            "model": IMAGE_TYPE_MODEL,
            "labels": list(IMAGE_TYPE_LABELS),
        },
        "anime_real_model": {
            "repo": ANIME_REAL_REPO,
            "revision": ANIME_REAL_REVISION,
            "model": ANIME_REAL_MODEL,
            "labels": list(ANIME_REAL_LABELS),
        },
        "preprocessing": {
            "input_size": [INPUT_SIZE, INPUT_SIZE],
            "resize": "PIL bilinear, direct resize",
            "background": "white",
            "normalization": "(pixel / 255 - 0.5) / 0.5",
        },
        "image_type_counts": dict(sorted(metadata_counts[0].items())),
        "anime_real_counts": dict(sorted(metadata_counts[1].items())),
        "cache": str(cache_path),
        "backup_directory": str(backup_root),
    }
    receipt_path = root / "deepghs_classification_receipt.json"
    receipt_temp = receipt_path.with_name(f".{receipt_path.name}.tmp")
    receipt_temp.write_text(
        json.dumps(receipt, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(receipt_temp, receipt_path)

    print(json.dumps(receipt, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
