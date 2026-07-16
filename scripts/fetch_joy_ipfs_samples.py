#!/usr/bin/env python3
"""Backfill missing JoyCaption review images from their IPFS filehashes."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
from pathlib import Path

from build_blinded_review_pack import cidv0_from_sha256, fetch


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pack", type=Path, default=ROOT / "quality_review_pack")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    pack = args.pack.resolve()
    manifest_path, key_path = pack / "review_manifest.jsonl", pack / "source_key.jsonl"
    manifest = [json.loads(line) for line in manifest_path.open() if line.strip()]
    key = [json.loads(line) for line in key_path.open() if line.strip()]
    by_id = {row["id"]: row for row in manifest}
    targets = [row for row in key if row["source_dataset"] == "joy-captioning-20250408a" and by_id[row["id"]]["image_unavailable"]]

    def download(row: dict) -> tuple[str, str, bool]:
        cid = cidv0_from_sha256(bytes.fromhex(row["original"]))
        destination = pack / "images" / f"{row['id']}.jpg"
        return row["id"], f"ipfs://{cid}", fetch(f"ipfs://{cid}", destination)

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(download, targets))
    urls = {item_id: url for item_id, url, _ in results}
    available = {item_id for item_id, _url, success in results if success}
    for row in manifest:
        if row["id"] in urls:
            row["image"] = f"images/{row['id']}.jpg" if row["id"] in available else None
            row["image_unavailable"] = row["id"] not in available
    for row in key:
        if row["id"] in urls:
            row["url"] = urls[row["id"]]
    with manifest_path.open("w") as handle:
        for row in manifest:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with key_path.open("w") as handle:
        for row in key:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"retrieved {len(available)} / {len(targets)} Joy IPFS images")


if __name__ == "__main__":
    main()
