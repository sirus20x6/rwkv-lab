#!/usr/bin/env python3
"""Add resumable DeepGHS anime-aesthetic scores to an I1 snapshot.

The classifier emits seven ordered class probabilities.  The scalar score is
their normalized expected ordinal, with ``worst=0`` and ``masterpiece=1``.
Results are cached append-only before both manifests are rewritten atomically.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
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

MODEL_REPO = "deepghs/anime_aesthetic"
MODEL_REVISION = "a83ab545e1d2a869f1180b99b2a7dee22ec3b97e"
MODEL_NAME = "swinv2pv3_v0_448_ls0.2_x"
LABELS = (
    "masterpiece",
    "best",
    "great",
    "good",
    "normal",
    "low",
    "worst",
)
LABEL_VALUES = {
    label: (len(LABELS) - 1 - index) / (len(LABELS) - 1)
    for index, label in enumerate(LABELS)
}
INPUT_SIZE = 448
SCHEMA_VERSION = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("/thearray/git/datasets/i1"),
        help="Dataset root containing metadata.jsonl and train.jsonl.",
    )
    parser.add_argument("--batch-size", type=int, default=128)
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
        help="Append-only score cache (defaults below ROOT/work).",
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
    if isinstance(value, str) and value:
        return value
    image = row.get("image")
    if not isinstance(image, str) or not image:
        raise RuntimeError("Manifest row has no image_sha256, image_id, or image path")
    return hashlib.sha256(image.encode("utf-8")).hexdigest()


def valid_scores(value: Any) -> bool:
    if not isinstance(value, dict) or set(value) != set(LABELS):
        return False
    scores = np.asarray([value[label] for label in LABELS], dtype=np.float64)
    return bool(
        np.isfinite(scores).all()
        and (scores >= -1e-6).all()
        and (scores <= 1.0 + 1e-6).all()
        and abs(float(scores.sum()) - 1.0) <= 2e-3
    )


def expected_score(scores: dict[str, float]) -> float:
    return float(sum(scores[label] * LABEL_VALUES[label] for label in LABELS))


def valid_result(item: Any) -> bool:
    if not (
        isinstance(item, dict)
        and item.get("schema_version") == SCHEMA_VERSION
        and isinstance(item.get("key"), str)
        and item.get("label") in LABELS
        and valid_scores(item.get("scores"))
    ):
        return False
    confidence = item.get("confidence")
    score = item.get("score")
    if not (
        isinstance(confidence, (int, float))
        and math.isfinite(confidence)
        and 0.0 <= confidence <= 1.0
        and isinstance(score, (int, float))
        and math.isfinite(score)
        and 0.0 <= score <= 1.0
    ):
        return False
    scores = item["scores"]
    return bool(
        abs(confidence - scores[item["label"]]) <= 1e-6
        and abs(score - expected_score(scores)) <= 1e-6
    )


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
                if any(rest.strip() for rest in handle):
                    raise RuntimeError(f"{path}:{line_no}: malformed non-final cache line")
                break
            if valid_result(item):
                results[item["key"]] = item
    return results


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
    return np.stack(list(executor.map(load_rgb, paths)), axis=0)


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


def batched(values: list[Any], size: int) -> Iterable[list[Any]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def classify(
    rows: list[dict[str, Any]],
    cache_path: Path,
    *,
    batch_size: int,
    decode_workers: int,
    checkpoint_every: int,
    provider: str,
) -> dict[str, dict[str, Any]]:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    results = load_cached_results(cache_path)
    pending = [row for row in rows if image_key(row) not in results]
    print(
        f"Anime-aesthetic cache: {len(results):,} complete, "
        f"{len(pending):,} pending",
        flush=True,
    )
    if not pending:
        return results

    model_path = Path(
        hf_hub_download(
            repo_id=MODEL_REPO,
            filename=f"{MODEL_NAME}/model.onnx",
            revision=MODEL_REVISION,
        )
    )
    session = make_session(model_path, provider)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

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
            values = session.run([output_name], {input_name: inputs})[0]
            if values.shape != (len(group), len(LABELS)):
                raise RuntimeError(f"Unexpected classifier output shape: {values.shape}")

            for row, vector in zip(group, values, strict=True):
                scores = {
                    label: float(vector[index])
                    for index, label in enumerate(LABELS)
                }
                label = max(scores, key=scores.__getitem__)
                item = {
                    "schema_version": SCHEMA_VERSION,
                    "key": image_key(row),
                    "label": label,
                    "confidence": scores[label],
                    "score": expected_score(scores),
                    "scores": scores,
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
            rate = completed_now / elapsed
            print(
                f"\rScored {len(results):,}/{len(rows):,} "
                f"({rate:.1f} images/s)",
                end="",
                file=sys.stderr,
                flush=True,
            )
        output.flush()
        os.fsync(output.fileno())
    print(file=sys.stderr)
    return results


def enriched_rows(
    rows: list[dict[str, Any]],
    results: dict[str, dict[str, Any]],
) -> Iterable[dict[str, Any]]:
    for row in rows:
        result = results[image_key(row)]
        enriched = dict(row)
        enriched.update(
            {
                "deepghs_anime_aesthetic_score": result["score"],
                "deepghs_anime_aesthetic_label": result["label"],
                "deepghs_anime_aesthetic_confidence": result["confidence"],
                "deepghs_anime_aesthetic_scores": result["scores"],
            }
        )
        yield enriched


def write_jsonl_atomic(path: Path, rows: Iterable[dict[str, Any]]) -> Path:
    temporary = path.with_name(f".{path.name}.anime-aesthetic.tmp")
    with temporary.open("w", encoding="utf-8", buffering=1024 * 1024) as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    return temporary


def validate_enriched(
    path: Path,
    expected_rows: int,
) -> tuple[Counter[str], list[float]]:
    labels: Counter[str] = Counter()
    scores: list[float] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            row = json.loads(line)
            label = row.get("deepghs_anime_aesthetic_label")
            score = row.get("deepghs_anime_aesthetic_score")
            confidence = row.get("deepghs_anime_aesthetic_confidence")
            probabilities = row.get("deepghs_anime_aesthetic_scores")
            item = {
                "schema_version": SCHEMA_VERSION,
                "key": "validation",
                "label": label,
                "score": score,
                "confidence": confidence,
                "scores": probabilities,
            }
            if not valid_result(item):
                raise RuntimeError(f"{path}:{line_no}: invalid aesthetic result")
            labels[label] += 1
            scores.append(float(score))
    if len(scores) != expected_rows:
        raise RuntimeError(f"{path}: expected {expected_rows:,} rows, got {len(scores):,}")
    return labels, scores


def ensure_backup(source: Path, backup: Path) -> None:
    backup.parent.mkdir(parents=True, exist_ok=True)
    if backup.exists():
        return
    try:
        os.link(source, backup)
    except OSError:
        shutil.copy2(source, backup)


def score_summary(scores: list[float]) -> dict[str, float]:
    values = np.asarray(scores, dtype=np.float64)
    return {
        "minimum": float(values.min()),
        "p01": float(np.quantile(values, 0.01)),
        "p10": float(np.quantile(values, 0.10)),
        "median": float(np.quantile(values, 0.50)),
        "mean": float(values.mean()),
        "p90": float(np.quantile(values, 0.90)),
        "p99": float(np.quantile(values, 0.99)),
        "maximum": float(values.max()),
    }


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
        else root / "work" / "anime_aesthetic.partial.jsonl"
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
        batch_size=args.batch_size,
        decode_workers=args.decode_workers,
        checkpoint_every=args.checkpoint_every,
        provider=args.provider,
    )
    missing = set(metadata_keys) - set(results)
    if missing:
        raise RuntimeError(f"Score cache is missing {len(missing):,} rows")
    if args.classify_only:
        print(f"Validated anime-aesthetic cache with {len(metadata_rows):,} rows")
        return 0

    metadata_temp = write_jsonl_atomic(
        metadata_path,
        enriched_rows(metadata_rows, results),
    )
    train_temp = write_jsonl_atomic(
        train_path,
        enriched_rows(train_rows, results),
    )
    metadata_labels, metadata_scores = validate_enriched(
        metadata_temp, len(metadata_rows)
    )
    train_labels, train_scores = validate_enriched(train_temp, len(train_rows))
    if metadata_labels != train_labels or metadata_scores != train_scores:
        raise RuntimeError("Aesthetic results differ between output manifests")

    backup_root = root / "work" / "pre_anime_aesthetic_manifests"
    ensure_backup(metadata_path, backup_root / metadata_path.name)
    ensure_backup(train_path, backup_root / train_path.name)
    os.replace(metadata_temp, metadata_path)
    os.replace(train_temp, train_path)

    receipt = {
        "schema_version": SCHEMA_VERSION,
        "created_unix": time.time(),
        "dataset_root": str(root),
        "rows": len(metadata_rows),
        "model": {
            "repo": MODEL_REPO,
            "revision": MODEL_REVISION,
            "name": MODEL_NAME,
            "labels": list(LABELS),
        },
        "score": {
            "field": "deepghs_anime_aesthetic_score",
            "definition": "normalized expected ordinal",
            "label_values": LABEL_VALUES,
            "range": [0.0, 1.0],
        },
        "added_fields": [
            "deepghs_anime_aesthetic_score",
            "deepghs_anime_aesthetic_label",
            "deepghs_anime_aesthetic_confidence",
            "deepghs_anime_aesthetic_scores",
        ],
        "preprocessing": {
            "input_size": [INPUT_SIZE, INPUT_SIZE],
            "resize": "PIL bilinear, direct resize",
            "background": "white",
            "normalization": "(pixel / 255 - 0.5) / 0.5",
        },
        "label_counts": dict(sorted(metadata_labels.items())),
        "score_summary": score_summary(metadata_scores),
        "cache": str(cache_path),
        "backup_directory": str(backup_root),
        "original_fields_preserved": True,
    }
    receipt_path = root / "anime_aesthetic_receipt.json"
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
