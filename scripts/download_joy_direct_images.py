#!/usr/bin/env python3
"""Resumably download JoyCaption rows that provide direct image URLs.

This intentionally excludes hash-only rows: those require an IPFS provider
that still pins the referenced content.  It uses one concurrent request per
host and a short host delay to avoid overloading source sites.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import os
import threading
import time
import urllib.parse
import urllib.request
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]


def records(data_dir: Path):
    for path in sorted(data_dir.glob("*.parquet")):
        for batch in pq.ParquetFile(path).iter_batches(columns=["filehash", "urls"], batch_size=4096):
            for digest, urls in zip(batch.column(0).to_pylist(), batch.column(1).to_pylist()):
                if urls:
                    yield digest.hex(), urls


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=ROOT / "joy-captioning-20250408a")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--host-delay", type=float, default=0.25)
    parser.add_argument("--limit", type=int, default=0, help="0 downloads every direct-URL row")
    args = parser.parse_args()
    dataset = args.dataset.resolve()
    output = dataset / "images_direct"
    output.mkdir(exist_ok=True)
    locks: dict[str, threading.Lock] = defaultdict(threading.Lock)
    last_request: dict[str, float] = defaultdict(float)
    stats = {"done": 0, "ok": 0, "skipped": 0, "failed": 0}
    stats_lock = threading.Lock()

    def download(record: tuple[str, list[str]]) -> None:
        filehash, urls = record
        if any(output.glob(f"{filehash}.*")):
            outcome = "skipped"
        else:
            outcome = "failed"
            for url in urls:
                host = urllib.parse.urlparse(url).netloc
                try:
                    # Serialize only each host; different sources still download in parallel.
                    with locks[host]:
                        delay = args.host_delay - (time.monotonic() - last_request[host])
                        if delay > 0:
                            time.sleep(delay)
                        last_request[host] = time.monotonic()
                        request = urllib.request.Request(url, headers={"User-Agent": "rwkv-lab-research/1.0"})
                        with urllib.request.urlopen(request, timeout=45) as response:
                            content_type = response.headers.get_content_type()
                            if not content_type.startswith("image/"):
                                continue
                            suffix = {"image/jpeg": ".jpg", "image/png": ".png", "image/webp": ".webp"}.get(content_type, ".img")
                            destination = output / f"{filehash}{suffix}"
                            temporary = destination.with_suffix(destination.suffix + ".part")
                            with temporary.open("wb") as handle:
                                while chunk := response.read(1024 * 1024):
                                    handle.write(chunk)
                            os.replace(temporary, destination)
                    outcome = "ok"
                    break
                except Exception:
                    for part in output.glob(f"{filehash}.*.part"):
                        part.unlink(missing_ok=True)
        with stats_lock:
            stats["done"] += 1
            stats[outcome] += 1
            if stats["done"] % 100 == 0:
                print(f"processed={stats['done']} ok={stats['ok']} skipped={stats['skipped']} failed={stats['failed']}", flush=True)

    source = records(dataset / "data")
    if args.limit:
        source = (item for _, item in zip(range(args.limit), source))
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        for _ in pool.map(download, source):
            pass
    print(" ".join(f"{key}={value}" for key, value in stats.items()), flush=True)


if __name__ == "__main__":
    main()
