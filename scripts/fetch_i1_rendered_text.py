#!/usr/bin/env python3
"""Fetch the proportional RenderedText slice from pinned WebDataset shards."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tarfile
import time
from pathlib import Path
from typing import Any

import requests

from scripts.fetch_i1_proportional_hf_subsets import (
    CAPTIONS,
    SOURCE_SPECS,
    finalize_subset,
    materialize_candidate,
    read_jsonl,
)
from scripts.materialize_i1_proportional_tranche import (
    DEFAULT_OUTPUT,
    SEED,
    TARGETS,
    atomic_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--captions", type=Path, default=CAPTIONS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--mib-per-second", type=float, default=8.0)
    return parser.parse_args()


def download_shard(
    url: str, destination: Path, mib_per_second: float
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    existing = destination.stat().st_size if destination.is_file() else 0
    headers = {
        "User-Agent": "rwkv-lab-i1-proportional-tranche/1.0",
        **({"Range": f"bytes={existing}-"} if existing else {}),
    }
    with requests.get(
        url, stream=True, timeout=(15, 60), headers=headers
    ) as response:
        response.raise_for_status()
        append = existing > 0 and response.status_code == 206
        if existing and not append:
            existing = 0
        mode = "ab" if append else "wb"
        started = time.monotonic()
        received = 0
        with destination.open(mode) as handle:
            for chunk in response.iter_content(256 * 1024):
                if not chunk:
                    continue
                handle.write(chunk)
                received += len(chunk)
                delay = (
                    received / (mib_per_second * 1024 * 1024)
                    - (time.monotonic() - started)
                )
                if delay > 0:
                    time.sleep(delay)
            handle.flush()
            os.fsync(handle.fileno())
    if not shard_is_complete(destination):
        raise RuntimeError(f"downloaded shard is not a valid tar: {destination}")


def shard_is_complete(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        with tarfile.open(path) as archive:
            for _ in archive:
                pass
    except (OSError, tarfile.TarError):
        return False
    return True


def collect_candidates(
    output: Path,
    target: int,
    rate: float,
) -> list[dict[str, Any]]:
    wanted = target + max(128, math.ceil(target * 0.10))
    state_path = output / "work/rendered_text.candidates.jsonl"
    candidates = [
        row for row in read_jsonl(state_path) if Path(str(row["image"])).is_file()
    ]
    by_key = {str(row["i1_key"]): row for row in candidates}
    spec = SOURCE_SPECS["rendered_text"]
    shard_index = 0
    while len(by_key) < wanted:
        filename = f"{shard_index:06}.tar"
        url = (
            f"https://huggingface.co/datasets/{spec['repo']}/resolve/"
            f"{spec['revision']}/{filename}?download=true"
        )
        shard = output / "work/rendered_text/shards" / filename
        if not shard_is_complete(shard):
            download_shard(url, shard, rate)
        with tarfile.open(shard) as archive:
            members = sorted(
                (
                    member
                    for member in archive
                    if member.isfile() and member.name.lower().endswith(".png")
                ),
                key=lambda member: member.name,
            )
            for member in members:
                key = Path(member.name).stem
                if key in by_key:
                    continue
                extracted = archive.extractfile(member)
                if extracted is None:
                    continue
                try:
                    row = materialize_candidate(
                        output,
                        "rendered_text",
                        key,
                        extracted.read(),
                        config=None,
                        source_url=url.split("?", 1)[0],
                    )
                except Exception as error:
                    print(
                        json.dumps(
                            {
                                "phase": "candidate_failure",
                                "subset": "rendered_text",
                                "key": key,
                                "error": f"{type(error).__name__}: {error}",
                            }
                        ),
                        flush=True,
                    )
                    continue
                by_key[key] = row
                if len(by_key) % 100 == 0 or len(by_key) == wanted:
                    atomic_jsonl(state_path, by_key.values())
                    print(
                        json.dumps(
                            {
                                "phase": "candidates",
                                "subset": "rendered_text",
                                "rows": len(by_key),
                                "wanted": wanted,
                                "shard": filename,
                            }
                        ),
                        flush=True,
                    )
                if len(by_key) >= wanted:
                    break
        shard_index += 1
    atomic_jsonl(state_path, by_key.values())
    return list(by_key.values())


def main() -> None:
    args = parse_args()
    args.output = args.output.expanduser().resolve()
    args.captions = args.captions.expanduser().resolve()
    if args.mib_per_second <= 0:
        raise SystemExit("--mib-per-second must be positive")
    target = TARGETS["rendered_text"]
    complete = args.output / "subsets/rendered_text.jsonl"
    if complete.is_file():
        rows = read_jsonl(complete)
        if len(rows) != target:
            raise RuntimeError(f"{complete} has {len(rows):,}/{target:,} rows")
    else:
        candidates = collect_candidates(args.output, target, args.mib_per_second)
        rows = finalize_subset(
            args.output,
            args.captions,
            "rendered_text",
            candidates,
            target,
            args.seed,
        )
    print(json.dumps({"complete": "rendered_text", "rows": len(rows), "target": target}))


if __name__ == "__main__":
    main()
