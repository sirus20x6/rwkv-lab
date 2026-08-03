#!/usr/bin/env python3
"""Materialize a balanced, caption-routed Midjourney v6 expert stage."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import shutil
import subprocess
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import duckdb
import pyarrow.parquet as pq
from PIL import Image

from rwkv_lab.mage_flow_adaptation import prepare_domain_manifest

DEFAULT_ROUTING = Path("/thearray/git/datasets/midjourney-v6-recap-routing")
DEFAULT_OUTPUT = Path("/thearray/git/datasets/midjourney-v6-recap-stage-100k")
DEFAULT_PRIOR_EVAL = Path(
    "/thearray/git/datasets/mageflow-anima-balanced-animation-holdout-128/eval.jsonl"
)
DEFAULT_REPO = "Photoroom/midjourney-v6-recap"
DEFAULT_REVISION = "21c628db81401da88c5b33507230528cf3fe4a12"
SOURCE_NAME = "photoroom_midjourney_v6_recap"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--routing-dir", type=Path, default=DEFAULT_ROUTING)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--prior-eval", type=Path, default=DEFAULT_PRIOR_EVAL)
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--train-per-domain", type=int, default=50_000)
    parser.add_argument("--new-eval-per-domain", type=int, default=64)
    parser.add_argument("--prior-eval-per-domain", type=int, default=64)
    parser.add_argument("--pool-reserve-fraction", type=float, default=0.10)
    parser.add_argument("--workers", type=int, default=3)
    return parser.parse_args()


def stable_digest(namespace: str, *values: Any) -> bytes:
    payload = ":".join([namespace, *(str(value) for value in values)])
    return hashlib.sha256(payload.encode()).digest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def select_shards(
    connection: duckdb.DuckDBPyConnection,
    routing_glob: Path,
    *,
    revision: str,
    needed_per_domain: int,
) -> tuple[list[str], dict[str, int]]:
    rows = connection.execute(
        """
        SELECT source_shard,
          sum((domain = 'photo')::INT)::BIGINT AS photo,
          sum((domain = 'animation')::INT)::BIGINT AS animation
        FROM read_parquet(?)
        GROUP BY source_shard
        """,
        [str(routing_glob)],
    ).fetchall()
    rows.sort(key=lambda row: stable_digest("shard", revision, row[0]))
    counts = {"photo": 0, "animation": 0}
    selected: list[str] = []
    for shard, photo, animation in rows:
        selected.append(str(shard))
        counts["photo"] += int(photo)
        counts["animation"] += int(animation)
        if all(counts[domain] >= needed_per_domain for domain in counts):
            return selected, counts
    raise RuntimeError(
        f"routing snapshot cannot supply {needed_per_domain:,} rows per domain"
    )


def routing_rows_for_shard(path: Path) -> list[dict[str, Any]]:
    table = pq.read_table(path)
    return [
        row for row in table.to_pylist() if row["domain"] in {"photo", "animation"}
    ]


def download_shard(
    destination: Path,
    *,
    repo: str,
    revision: str,
    shard: str,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    # The dataset uses Hugging Face's Xet large-file backend. The Xet-aware
    # client is dramatically faster and more reliable here than a redirected
    # single-stream HTTP transfer.
    destination.unlink(missing_ok=True)
    environment = os.environ.copy()
    environment["HF_XET_HIGH_PERFORMANCE"] = "1"
    subprocess.run(
        [
            "hf",
            "download",
            repo,
            shard,
            "--repo-type",
            "dataset",
            "--revision",
            revision,
            "--local-dir",
            str(destination.parent),
            "--quiet",
        ],
        check=True,
        env=environment,
    )
    if not destination.is_file():
        raise RuntimeError(f"Hugging Face download did not create {destination}")


def image_extension(image_format: str) -> str:
    return {
        "JPEG": ".jpg",
        "PNG": ".png",
        "WEBP": ".webp",
    }[image_format.upper()]


def extract_shard(
    shard_path: Path,
    routing_rows: list[dict[str, Any]],
    *,
    pool_dir: Path,
    metadata_path: Path,
) -> None:
    table = pq.read_table(shard_path, columns=["image"])
    images = table.column("image")
    temporary_pool = pool_dir.with_name(pool_dir.name + ".incomplete")
    if temporary_pool.exists():
        shutil.rmtree(temporary_pool)
    temporary_pool.mkdir(parents=True)
    records = []
    for route in sorted(routing_rows, key=lambda row: int(row["source_row"])):
        source_row = int(route["source_row"])
        value = images[source_row].as_py()
        payload = value.get("bytes") if isinstance(value, dict) else None
        if not isinstance(payload, bytes) or not payload:
            raise RuntimeError(
                f"{route['source_shard']}:{source_row} has no embedded image bytes"
            )
        with Image.open(io.BytesIO(payload)) as decoded:
            decoded.load()
            width, height = decoded.size
            image_format = str(decoded.format or "").upper()
        extension = image_extension(image_format)
        file_name = f"{source_row:05d}{extension}"
        destination = temporary_pool / file_name
        temporary = destination.with_name(destination.name + ".tmp")
        temporary.write_bytes(payload)
        os.replace(temporary, destination)
        records.append(
            {
                **route,
                "pool_image": str((pool_dir / file_name).resolve()),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "width": int(width),
                "height": int(height),
                "image_format": image_format,
                "byte_size": len(payload),
            }
        )
    temporary_metadata = metadata_path.with_name(metadata_path.name + ".tmp")
    with temporary_metadata.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(
                json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
            )
    os.replace(temporary_pool, pool_dir)
    os.replace(temporary_metadata, metadata_path)


def materialize_pool(
    selected_shards: list[str],
    *,
    routing_dir: Path,
    output_dir: Path,
    repo: str,
    revision: str,
    workers: int,
) -> list[dict[str, Any]]:
    pool_root = output_dir / "pool"
    metadata_root = output_dir / "pool_metadata"
    download_root = output_dir / ".downloads"
    pool_root.mkdir(parents=True, exist_ok=True)
    metadata_root.mkdir(parents=True, exist_ok=True)
    download_root.mkdir(parents=True, exist_ok=True)
    def process_shard(shard: str) -> str:
        stem = Path(shard).stem
        pool_dir = pool_root / stem
        metadata_path = metadata_root / f"{stem}.jsonl"
        if not (pool_dir.is_dir() and metadata_path.is_file()):
            routing_rows = routing_rows_for_shard(
                routing_dir / "routing_shards" / shard
            )
            local_shard = download_root / shard
            download_shard(
                local_shard, repo=repo, revision=revision, shard=shard
            )
            try:
                extract_shard(
                    local_shard,
                    routing_rows,
                    pool_dir=pool_dir,
                    metadata_path=metadata_path,
                )
            finally:
                local_shard.unlink(missing_ok=True)
        return shard

    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = {
            executor.submit(process_shard, shard): shard for shard in selected_shards
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            shard = future.result()
            print(
                json.dumps(
                    {
                        "kind": "pool_shard",
                        "completed": completed,
                        "total": len(selected_shards),
                        "shard": shard,
                    }
                ),
                flush=True,
            )
    records = []
    for shard in selected_shards:
        records.extend(load_jsonl(metadata_root / f"{Path(shard).stem}.jsonl"))
    return records


def deduplicate(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for record in sorted(
        records,
        key=lambda row: stable_digest(
            "dedup", row["source_shard"], row["source_row"]
        ),
    ):
        digest = str(record["sha256"])
        if digest not in seen:
            selected.append(record)
            seen.add(digest)
    return selected


def split_records(
    records: list[dict[str, Any]],
    *,
    train_per_domain: int,
    eval_per_domain: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], set[str]]:
    families: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        families[str(record["family_id"])].append(record)
    heldout_families: set[str] = set()
    heldout_counts = {"photo": 0, "animation": 0}
    for family_id in sorted(
        families, key=lambda value: stable_digest("eval-family", value)
    ):
        family_counts = Counter(row["domain"] for row in families[family_id])
        if not any(
            heldout_counts[domain] < eval_per_domain and family_counts[domain]
            for domain in heldout_counts
        ):
            continue
        heldout_families.add(family_id)
        for domain in heldout_counts:
            heldout_counts[domain] += family_counts[domain]
        if all(value >= eval_per_domain for value in heldout_counts.values()):
            break
    if not all(value >= eval_per_domain for value in heldout_counts.values()):
        raise RuntimeError(f"insufficient family-disjoint eval rows: {heldout_counts}")

    eval_records, train_records = [], []
    for domain in ("photo", "animation"):
        heldout = [
            row
            for row in records
            if row["domain"] == domain
            and str(row["family_id"]) in heldout_families
        ]
        heldout.sort(
            key=lambda row: stable_digest(
                "eval-row", row["source_shard"], row["source_row"]
            )
        )
        eval_records.extend(heldout[:eval_per_domain])

        candidates = [
            row
            for row in records
            if row["domain"] == domain
            and str(row["family_id"]) not in heldout_families
        ]
        candidates.sort(
            key=lambda row: stable_digest(
                "train-row", row["source_shard"], row["source_row"]
            )
        )
        if len(candidates) < train_per_domain:
            raise RuntimeError(
                f"only {len(candidates):,} unique {domain} train rows available"
            )
        train_records.extend(candidates[:train_per_domain])
    return train_records, eval_records, heldout_families


def link_records(
    records: list[dict[str, Any]],
    *,
    output_dir: Path,
    split: str,
) -> list[dict[str, Any]]:
    linked = []
    for record in records:
        source = Path(record["pool_image"])
        relative = (
            Path("images")
            / split
            / str(record["domain"])
            / Path(str(record["source_shard"])).stem
            / source.name
        )
        destination = output_dir / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            try:
                os.link(source, destination)
            except OSError:
                shutil.copy2(source, destination)
        linked.append({**record, "file_name": str(relative)})
    return linked


def raw_manifest_row(record: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    return {
        "image": str((output_dir / record["file_name"]).resolve()),
        "image_sha256": record["sha256"],
        "domain": record["domain"],
        "source": SOURCE_NAME,
        "caption": record["caption"],
        "caption_provenance": f"{record['caption_source']}_recaption",
        "width": record["width"],
        "height": record["height"],
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            )
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    routing_dir = args.routing_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    complete_receipt = output_dir / "report.json"
    if complete_receipt.is_file():
        print(complete_receipt.read_text(encoding="utf-8"))
        return
    needed = int(
        (args.train_per_domain + args.new_eval_per_domain)
        * (1.0 + args.pool_reserve_fraction)
    )
    connection = duckdb.connect()
    selected_shards, routed_counts = select_shards(
        connection,
        routing_dir / "routing_shards" / "*.parquet",
        revision=args.revision,
        needed_per_domain=needed,
    )
    atomic_json(
        output_dir / "selection_plan.json",
        {
            "repo": args.repo,
            "revision": args.revision,
            "selected_shards": selected_shards,
            "routed_counts": routed_counts,
            "train_per_domain": args.train_per_domain,
            "new_eval_per_domain": args.new_eval_per_domain,
            "prior_eval_per_domain": args.prior_eval_per_domain,
        },
    )
    pool = deduplicate(
        materialize_pool(
            selected_shards,
            routing_dir=routing_dir,
            output_dir=output_dir,
            repo=args.repo,
            revision=args.revision,
            workers=args.workers,
        )
    )
    train, new_eval, heldout_families = split_records(
        pool,
        train_per_domain=args.train_per_domain,
        eval_per_domain=args.new_eval_per_domain,
    )
    linked_train = link_records(train, output_dir=output_dir, split="train")
    linked_eval = link_records(new_eval, output_dir=output_dir, split="eval")

    train_metadata = [
        {
            "file_name": row["file_name"],
            "text": row["caption"],
            "domain": row["domain"],
            "sha256": row["sha256"],
            "width": row["width"],
            "height": row["height"],
            "family_id": row["family_id"],
            "source_shard": row["source_shard"],
            "source_row": row["source_row"],
            "caption_source": row["caption_source"],
            "llava_label": row["llava_label"],
            "gemini_label": row["gemini_label"],
            "qwen3_label": row["qwen3_label"],
        }
        for row in linked_train
    ]
    eval_metadata = [
        {
            "file_name": row["file_name"],
            "text": row["caption"],
            "domain": row["domain"],
            "sha256": row["sha256"],
            "width": row["width"],
            "height": row["height"],
            "family_id": row["family_id"],
            "source_shard": row["source_shard"],
            "source_row": row["source_row"],
            "caption_source": row["caption_source"],
        }
        for row in linked_eval
    ]
    write_jsonl(output_dir / "metadata.jsonl", train_metadata)
    write_jsonl(output_dir / "eval_metadata.jsonl", eval_metadata)

    raw_train = [raw_manifest_row(row, output_dir) for row in linked_train]
    new_eval_rows = [raw_manifest_row(row, output_dir) for row in linked_eval]
    prior_eval = load_jsonl(args.prior_eval.expanduser().resolve())
    retained_prior = []
    for domain in ("photo", "animation"):
        retained_prior.extend(
            [row for row in prior_eval if row["domain"] == domain][
                : args.prior_eval_per_domain
            ]
        )
    raw_eval = new_eval_rows + retained_prior
    write_jsonl(output_dir / "_raw_train.jsonl", raw_train)
    write_jsonl(output_dir / "_raw_eval.jsonl", raw_eval)
    train_report = prepare_domain_manifest(
        output_dir / "_raw_train.jsonl",
        output_dir / "train.jsonl",
        data_root=Path("/"),
        max_aspect_ratio=4.0,
        verify_images=False,
    )
    eval_report = prepare_domain_manifest(
        output_dir / "_raw_eval.jsonl",
        output_dir / "eval.jsonl",
        data_root=Path("/"),
        max_aspect_ratio=4.0,
        verify_images=False,
    )
    (output_dir / "_raw_train.jsonl").unlink()
    (output_dir / "_raw_eval.jsonl").unlink()

    train_ids = {row["sha256"] for row in train_metadata}
    eval_ids = {row["sha256"] for row in eval_metadata}
    if train_ids & eval_ids:
        raise RuntimeError("Midjourney train/eval image hash overlap")
    report = {
        "schema": "rwkv-lab.midjourney-v6-expert-stage.v1",
        "repo": args.repo,
        "revision": args.revision,
        "selected_shards": selected_shards,
        "pool_unique_images": len(pool),
        "train_count": len(train_metadata),
        "new_eval_count": len(eval_metadata),
        "combined_eval_count": int(eval_report["counts"]["output"]),
        "train_domain_counts": dict(Counter(row["domain"] for row in train_metadata)),
        "new_eval_domain_counts": dict(
            Counter(row["domain"] for row in eval_metadata)
        ),
        "combined_eval_domain_counts": eval_report["audit"]["domain_counts"],
        "heldout_family_count": len(heldout_families),
        "cross_split_duplicate_images": 0,
        "train_canonicalization_counts": train_report["counts"],
        "eval_canonicalization_counts": eval_report["counts"],
        "image_bytes": sum(int(row["byte_size"]) for row in linked_train + linked_eval),
    }
    atomic_json(complete_receipt, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
