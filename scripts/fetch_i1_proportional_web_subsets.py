#!/usr/bin/env python3
"""Materialize the RedCaps and YFCC portions of the proportional i1 tranche."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
import os
import sqlite3
import time
import threading
import zipfile
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pyarrow.parquet as pq
import requests
from PIL import Image, ImageOps

from scripts.materialize_i1_proportional_tranche import (
    DEFAULT_OUTPUT,
    SEED,
    TARGETS,
    atomic_jsonl,
)


ROOT = Path(__file__).resolve().parents[1]
CAPTIONS = ROOT / "i1-captions"
YFCC_DATABASE = ROOT / "datasets/i1_full_sources/yfcc/yfcc100m_dataset.sql"
REDCAPS_ARCHIVE = (
    DEFAULT_OUTPUT / "work/redcaps/redcaps_v1.0_annotations.zip.part"
)
CAPTION_COLUMNS = tuple(f"caption{index}" for index in range(1, 6))
SOURCE_REVISIONS = {
    "redcaps": "kdexd/red_caps@0400b045a9750532447784f85dacee06a275ad8f",
    "yfcc": "multimedia-commons/yfcc100m",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--captions", type=Path, default=CAPTIONS)
    parser.add_argument("--yfcc-database", type=Path, default=YFCC_DATABASE)
    parser.add_argument("--redcaps-archive", type=Path, default=REDCAPS_ARCHIVE)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--subsets",
        nargs="+",
        choices=("redcaps", "yfcc"),
        default=("redcaps", "yfcc"),
    )
    parser.add_argument("--mib-per-second", type=float, default=8.0)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--redcaps-reserve-multiplier", type=float, default=10.0)
    parser.add_argument("--yfcc-reserve-multiplier", type=float, default=2.0)
    return parser.parse_args()


def robust_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.is_file():
        return rows
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                rows.append(row)
    return rows


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def append_jsonl_rows(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def allocate_by_size(sizes: dict[Path, int], total: int) -> dict[Path, int]:
    denominator = sum(sizes.values())
    result = {path: total * size // denominator for path, size in sizes.items()}
    remaining = total - sum(result.values())
    order = sorted(
        sizes,
        key=lambda path: (total * sizes[path] % denominator, str(path)),
        reverse=True,
    )
    for path in order[:remaining]:
        result[path] += 1
    return result


def choose_caption(row: dict[str, Any], key: str, seed: int) -> tuple[str, str]:
    values = [
        (column, str(row.get(column) or "").strip())
        for column in CAPTION_COLUMNS
        if str(row.get(column) or "").strip()
    ]
    if not values:
        raise RuntimeError(f"no caption for {key}")
    digest = hashlib.sha256(f"{seed}:{key}".encode()).digest()
    return values[int.from_bytes(digest[:8], "big") % len(values)]


def sampled_caption_rows(
    captions: Path,
    output: Path,
    subset: str,
    wanted: int,
    seed: int,
) -> list[dict[str, Any]]:
    receipt = output / "work" / f"{subset}.sampled_captions.jsonl"
    existing = robust_jsonl(receipt)
    if len(existing) >= wanted:
        return existing[:wanted]
    files = sorted((captions / subset).glob("*.parquet"))
    sizes = {path: pq.ParquetFile(path).metadata.num_rows for path in files}
    quotas = allocate_by_size(sizes, wanted)
    rows: list[dict[str, Any]] = []
    for path in files:
        parquet = pq.ParquetFile(path)
        quota = quotas[path]
        file_seed = int.from_bytes(
            hashlib.sha256(f"{seed}:{subset}:{path.name}".encode()).digest()[:8],
            "big",
        )
        generator = np.random.default_rng(file_seed)
        indices = np.sort(
            generator.choice(parquet.metadata.num_rows, size=quota, replace=False)
        )
        offset = 0
        cursor = 0
        available = [
            name for name in CAPTION_COLUMNS if name in parquet.schema_arrow.names
        ]
        for group_index in range(parquet.metadata.num_row_groups):
            group_rows = parquet.metadata.row_group(group_index).num_rows
            end = offset + group_rows
            right = int(np.searchsorted(indices, end, side="left"))
            selected = indices[cursor:right] - offset
            if selected.size:
                table = parquet.read_row_group(
                    group_index, columns=["key", *available]
                )
                for raw in table.take(selected).to_pylist():
                    key = str(raw["key"])
                    variant, text = choose_caption(raw, key, seed)
                    rows.append(
                        {
                            "i1_subset": subset,
                            "i1_key": key,
                            "text": text,
                            "caption_variant": variant,
                        }
                    )
            cursor = right
            offset = end
        print(
            json.dumps(
                {
                    "phase": "caption_sample",
                    "subset": subset,
                    "file": path.name,
                    "rows": len(rows),
                    "wanted": wanted,
                }
            ),
            flush=True,
        )
    if len(rows) != wanted:
        raise RuntimeError(f"sampled {len(rows):,}/{wanted:,} {subset} captions")
    atomic_jsonl(receipt, rows)
    return rows


def yfcc_archive_url(download_url: str) -> str:
    digest = hashlib.md5(download_url.encode(), usedforsecurity=False).hexdigest()
    compact = "".join(f"{int(digest[index:index + 2], 16):x}" for index in range(0, 32, 2))
    return (
        "https://multimedia-commons.s3-us-west-2.amazonaws.com/data/images/"
        f"{compact[:3]}/{compact[3:6]}/{compact}.jpg"
    )


def attach_yfcc_urls(
    database: Path, candidates: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    by_id = {int(row["i1_key"]): row for row in candidates}
    result: list[dict[str, Any]] = []
    connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
    try:
        # The source is a 65 GB SQLite database on the array. Ascending
        # primary-key probes turn 100k random seeks into mostly forward I/O.
        keys = sorted(by_id)
        for start in range(0, len(keys), 500):
            batch = keys[start : start + 500]
            marks = ",".join("?" for _ in batch)
            query = (
                "SELECT photoid, downloadurl FROM yfcc100m_dataset "
                f"WHERE photoid IN ({marks})"
            )
            for photo_id, download_url in connection.execute(query, batch):
                if not download_url:
                    continue
                row = dict(by_id[int(photo_id)])
                row["source_url"] = yfcc_archive_url(str(download_url))
                result.append(row)
    finally:
        connection.close()
    return result


def attach_redcaps_urls(
    archive: Path, candidates: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    if not zipfile.is_zipfile(archive):
        raise RuntimeError(f"RedCaps archive is incomplete or invalid: {archive}")
    by_key = {str(row["i1_key"]): row for row in candidates}
    result: list[dict[str, Any]] = []
    found: set[str] = set()
    with zipfile.ZipFile(archive) as handle:
        names = sorted(
            name
            for name in handle.namelist()
            if name.endswith(".json")
            and (
                name.startswith("annotations/")
                or "/annotations/" in name
            )
        )
        for name in names:
            with handle.open(name) as member:
                data = json.load(io.TextIOWrapper(member, encoding="utf-8"))
            for annotation in data.get("annotations", ()):
                key = str(annotation.get("image_id") or "")
                url = str(annotation.get("url") or "")
                if key in by_key and key not in found and url:
                    row = dict(by_key[key])
                    row["source_url"] = url
                    result.append(row)
                    found.add(key)
            print(
                json.dumps(
                    {
                        "phase": "redcaps_url_join",
                        "file": name,
                        "matched": len(result),
                        "wanted": len(candidates),
                    }
                ),
                flush=True,
            )
    return result


def rate_limited_get(
    session: requests.Session, url: str, mib_per_second: float
) -> bytes:
    byte_rate = mib_per_second * 1024 * 1024
    started = time.monotonic()
    payload = bytearray()
    with session.get(url, stream=True, timeout=(15, 60)) as response:
        response.raise_for_status()
        content_type = response.headers.get("content-type", "").lower()
        if "text/html" in content_type:
            raise RuntimeError(f"unexpected content type {content_type}")
        for chunk in response.iter_content(256 * 1024):
            if not chunk:
                continue
            payload.extend(chunk)
            delay = len(payload) / byte_rate - (time.monotonic() - started)
            if delay > 0:
                time.sleep(delay)
    return bytes(payload)


def verify_image(payload: bytes) -> tuple[str, int, int, str]:
    with Image.open(io.BytesIO(payload)) as image:
        image_format = str(image.format or "").upper()
        encoded_width, encoded_height = image.size
        if encoded_width * encoded_height > 64_000_000:
            raise RuntimeError(
                f"image exceeds 64 MP: {encoded_width}x{encoded_height}"
            )
        image.load()
        image = ImageOps.exif_transpose(image)
        width, height = image.size
    extension = {"JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp"}.get(image_format)
    if extension is None or width < 16 or height < 16:
        raise RuntimeError(f"invalid {image_format} image at {width}x{height}")
    return extension, width, height, hashlib.sha256(payload).hexdigest()


def existing_global_hashes(output: Path, excluding: str) -> set[str]:
    hashes: set[str] = set()
    for path in sorted((output / "subsets").glob("*.jsonl")):
        if path.stem == excluding:
            continue
        hashes.update(
            str(row.get("image_sha256") or "") for row in robust_jsonl(path)
        )
    hashes.discard("")
    return hashes


def materialize(
    output: Path,
    subset: str,
    row: dict[str, Any],
    payload: bytes,
) -> dict[str, Any]:
    extension, width, height, digest = verify_image(payload)
    name = hashlib.sha256(f"{subset}:{row['i1_key']}".encode()).hexdigest()
    relative = Path("images") / subset / name[:2] / f"{name}{extension}"
    destination = output / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.is_file():
        temporary = destination.with_name(destination.name + ".tmp")
        temporary.write_bytes(payload)
        os.replace(temporary, destination)
    return {
        "file_name": str(relative),
        "image": str(destination.resolve()),
        "text": row["text"],
        "caption": row["text"],
        "i1_subset": subset,
        "i1_key": str(row["i1_key"]),
        "caption_variant": row["caption_variant"],
        "caption_policy": "deterministic_existing_i1_variant",
        "image_sha256": digest,
        "width": width,
        "height": height,
        "source_kind": "rate_limited_source_download",
        "source_revision": SOURCE_REVISIONS[subset],
        "source_url": row["source_url"],
    }


def download_candidates(
    output: Path,
    subset: str,
    candidates: Iterable[dict[str, Any]],
    target: int,
    rate: float,
    workers: int,
) -> list[dict[str, Any]]:
    attempts_path = output / "work" / f"{subset}.download_attempts.jsonl"
    attempts = robust_jsonl(attempts_path)
    attempted = {str(row["i1_key"]) for row in attempts}
    successes = {
        str(row["i1_key"]): row
        for row in attempts
        if row.get("status") == "success"
        and Path(str(row.get("image") or "")).is_file()
        and int(row.get("width") or 0) * int(row.get("height") or 0)
        <= 64_000_000
    }
    hashes = existing_global_hashes(output, subset)
    hashes.update(
        str(row["image_sha256"]) for row in successes.values()
    )
    thread_state = threading.local()

    def fetch(candidate: dict[str, Any]) -> dict[str, Any]:
        key = str(candidate["i1_key"])
        session = getattr(thread_state, "session", None)
        if session is None:
            session = requests.Session()
            session.headers["User-Agent"] = "rwkv-lab-i1-proportional-tranche/1.0"
            thread_state.session = session
        try:
            payload = rate_limited_get(session, candidate["source_url"], rate)
            row = materialize(output, subset, candidate, payload)
            row["status"] = "success"
            return row
        except Exception as error:
            return {
                "status": "failure",
                "i1_subset": subset,
                "i1_key": key,
                "source_url": candidate["source_url"],
                "error": f"{type(error).__name__}: {error}",
            }

    iterator = iter(candidates)
    exhausted = False
    with ThreadPoolExecutor(max_workers=workers) as executor:
        while len(successes) < target and not exhausted:
            batch: list[dict[str, Any]] = []
            while len(batch) < workers * 4:
                try:
                    candidate = next(iterator)
                except StopIteration:
                    exhausted = True
                    break
                if str(candidate["i1_key"]) not in attempted:
                    batch.append(candidate)
            if not batch:
                continue
            results = list(executor.map(fetch, batch))
            finalized_results: list[dict[str, Any]] = []
            for row in results:
                key = str(row["i1_key"])
                if (
                    row["status"] == "success"
                    and row["image_sha256"] in hashes
                ):
                    row = {
                        "status": "failure",
                        "i1_subset": subset,
                        "i1_key": key,
                        "source_url": row["source_url"],
                        "error": "RuntimeError: duplicate image content",
                    }
                elif row["status"] == "success":
                    successes[key] = row
                    hashes.add(row["image_sha256"])
                attempted.add(key)
                finalized_results.append(row)
            append_jsonl_rows(attempts_path, finalized_results)
            failures = sum(
                row["status"] == "failure" for row in finalized_results
            )
            print(
                json.dumps(
                    {
                        "phase": "download",
                        "subset": subset,
                        "successes": len(successes),
                        "target": target,
                        "attempts": len(attempted),
                        "batch_failures": failures,
                    }
                ),
                flush=True,
            )
    rows = list(successes.values())
    if len(rows) < target:
        raise RuntimeError(f"downloaded {len(rows):,}/{target:,} {subset} images")
    rows.sort(key=lambda row: str(row["i1_key"]))
    rows = rows[:target]
    for row in rows:
        row.pop("status", None)
    atomic_jsonl(output / "subsets" / f"{subset}.jsonl", rows)
    return rows


def fetch_subset(args: argparse.Namespace, subset: str) -> list[dict[str, Any]]:
    target = TARGETS[subset]
    complete = args.output / "subsets" / f"{subset}.jsonl"
    if complete.is_file():
        rows = robust_jsonl(complete)
        if len(rows) != target:
            raise RuntimeError(f"{complete} has {len(rows):,}/{target:,} rows")
        return rows
    multiplier = (
        args.redcaps_reserve_multiplier
        if subset == "redcaps"
        else args.yfcc_reserve_multiplier
    )
    candidate_count = int(np.ceil(target * multiplier))
    candidates = sampled_caption_rows(
        args.captions, args.output, subset, candidate_count, args.seed
    )
    if subset == "redcaps":
        candidates = attach_redcaps_urls(args.redcaps_archive, candidates)
    else:
        candidates = attach_yfcc_urls(args.yfcc_database, candidates)
    return download_candidates(
        args.output, subset, candidates, target, args.mib_per_second, args.workers
    )


def main() -> None:
    args = parse_args()
    args.output = args.output.expanduser().resolve()
    args.captions = args.captions.expanduser().resolve()
    args.yfcc_database = args.yfcc_database.expanduser().resolve()
    args.redcaps_archive = args.redcaps_archive.expanduser().resolve()
    if args.mib_per_second <= 0 or args.workers < 1:
        raise SystemExit("--mib-per-second and --workers must be positive")
    for subset in args.subsets:
        rows = fetch_subset(args, subset)
        print(
            json.dumps({"complete": subset, "rows": len(rows), "target": TARGETS[subset]}),
            flush=True,
        )


if __name__ == "__main__":
    main()
