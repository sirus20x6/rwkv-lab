#!/usr/bin/env python3
"""Resumably download a pinned DoclingMatix OCR tranche."""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download


ROOT = Path(__file__).resolve().parents[1]
REPO_ID = "HuggingFaceM4/DoclingMatix"
REPO_TYPE = "dataset"
PARQUET_REVISION = "e6dab61363330892bdc86bc44e1ada8fdff817b0"
MAIN_REVISION = "b8713960364fbfc6319c7f0be3c4b76d2e118141"
DEFAULT_DESTINATION = ROOT / "datasets/doclingmatix"
DEFAULT_SHARDS = 218


def shard_names(api: HfApi, count: int) -> list[str]:
    files = api.list_repo_files(
        REPO_ID, repo_type=REPO_TYPE, revision=PARQUET_REVISION)
    shards = sorted(name for name in files
                    if name.startswith("default/train/")
                    and name.endswith(".parquet"))
    if not 1 <= count <= len(shards):
        raise ValueError(f"requested {count} shards from repository with {len(shards)}")
    return shards[:count]


def download_file(filename: str, destination: Path) -> Path:
    result = hf_hub_download(
        repo_id=REPO_ID, repo_type=REPO_TYPE, revision=PARQUET_REVISION,
        filename=filename, local_dir=destination)
    return Path(result)


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def receipt(destination: Path, shards: list[str]) -> dict:
    rows = size = 0
    records = []
    for name in shards:
        path = destination / name
        if not path.is_file():
            raise FileNotFoundError(path)
        file_rows = pq.ParquetFile(path).metadata.num_rows
        file_size = path.stat().st_size
        rows += file_rows
        size += file_size
        records.append({"file": name, "bytes": file_size, "rows": file_rows})
    return {
        "schema": 1,
        "repo_id": REPO_ID,
        "revision": PARQUET_REVISION,
        "shards": len(shards),
        "rows": rows,
        "bytes": size,
        "files": records,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path,
                        default=DEFAULT_DESTINATION)
    parser.add_argument("--shards", type=int, default=DEFAULT_SHARDS,
                        help="leading train shards to fetch (default: 218)")
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    return args


def main() -> None:
    args = parse_args()
    args.destination.mkdir(parents=True, exist_ok=True)
    api = HfApi()
    shards = shard_names(api, args.shards)
    # Keep the dataset card beside the pinned converted shards.
    hf_hub_download(
        repo_id=REPO_ID, repo_type=REPO_TYPE, revision=MAIN_REVISION,
        filename="README.md", local_dir=args.destination)
    print({"kind": "doclingmatix", "state": "downloading",
           "destination": str(args.destination.resolve()),
           "shards": len(shards), "workers": args.workers,
           "revision": PARQUET_REVISION}, flush=True)
    completed = 0
    with concurrent.futures.ThreadPoolExecutor(
            max_workers=args.workers) as pool:
        futures = {pool.submit(download_file, name, args.destination): name
                   for name in shards}
        for future in concurrent.futures.as_completed(futures):
            path = future.result()
            completed += 1
            print({"kind": "doclingmatix", "completed": completed,
                   "total": len(shards), "file": path.name}, flush=True)
    result = receipt(args.destination, shards)
    receipt_path = args.destination / "tranche_000.receipt.json"
    atomic_json(receipt_path, result)
    print({"kind": "doclingmatix", "state": "ready",
           "rows": result["rows"], "bytes": result["bytes"],
           "receipt": str(receipt_path)}, flush=True)


if __name__ == "__main__":
    main()
