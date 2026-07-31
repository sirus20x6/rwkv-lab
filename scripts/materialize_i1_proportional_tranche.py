#!/usr/bin/env python3
"""Build and validate the proportional 97,819-row i1 tranche incrementally."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = Path("/workspace/git/datasets/i1")
DEFAULT_REUSE_WORK = ROOT / "datasets/captioning_first_work"
TARGET_ROWS = 97_819
SEED = 20_260_730
POPULATIONS = {
    "fluxreason": 5_890_279,
    "gptedit": 1_553_575,
    "imagenet22k": 13_673_544,
    "inaturalist": 4_813_543,
    "megalith10m": 9_393_971,
    "midjourneyv6": 1_240_185,
    "pexels": 2_810_634,
    "places365-challenge2016": 7_221_597,
    "redcaps": 4_817_431,
    "rendered_text": 11_977_816,
    "textatlas": 5_396_890,
    "yfcc": 97_945_286,
}
REUSE_NAMES = {
    "fluxreason": "fluxreason",
    "imagenet22k": "imagenet22k",
    "inaturalist": "inaturalist",
    "midjourneyv6": "midjourneyv6",
    "pexels": "pexels",
    "places365-challenge2016": "places365",
}


def hamilton_targets(populations: dict[str, int], total: int) -> dict[str, int]:
    population = sum(populations.values())
    result = {name: total * size // population for name, size in populations.items()}
    remaining = total - sum(result.values())
    order = sorted(
        populations,
        key=lambda name: (total * populations[name] % population, name),
        reverse=True,
    )
    for name in order[:remaining]:
        result[name] += 1
    return result


TARGETS = hamilton_targets(POPULATIONS, TARGET_ROWS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--reuse-work", type=Path, default=DEFAULT_REUSE_WORK)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--reuse-local",
        action="store_true",
        help="materialize the six already verified local subsets",
    )
    parser.add_argument(
        "--merge",
        action="store_true",
        help="merge all complete per-subset receipts and validate current progress",
    )
    return parser.parse_args()


def stable_rank(seed: int, subset: str, key: str) -> bytes:
    return hashlib.sha256(f"{seed}:{subset}:{key}".encode()).digest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def atomic_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            )
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def source_image(row: dict[str, Any]) -> Path:
    path = Path(str(row["image"]))
    return path if path.is_absolute() else ROOT / path


def materialize_image(source: Path, output: Path, subset: str, key: str) -> Path:
    safe_key = hashlib.sha256(key.encode()).hexdigest()
    suffix = source.suffix.lower() or ".img"
    relative = Path("images") / subset / safe_key[:2] / f"{safe_key}{suffix}"
    destination = output / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        try:
            os.link(source, destination)
        except OSError:
            shutil.copy2(source, destination)
    return relative


def canonical_reused_row(
    row: dict[str, Any], output: Path, subset: str
) -> dict[str, Any]:
    key = str(row["i1_key"])
    source = source_image(row).resolve()
    if not source.is_file():
        raise RuntimeError(f"missing reused image: {source}")
    relative = materialize_image(source, output, subset, key)
    text = str(row.get("text") or "").strip()
    digest = str(row.get("image_sha256") or "").strip()
    if not text or len(digest) != 64:
        raise RuntimeError(f"invalid reused pair for {subset}:{key}")
    return {
        "file_name": str(relative),
        "image": str((output / relative).resolve()),
        "text": text,
        "caption": text,
        "i1_subset": subset,
        "i1_key": key,
        "caption_variant": row.get("caption_variant"),
        "caption_policy": row.get("caption_policy"),
        "image_sha256": digest,
        "width": int(row["width"]),
        "height": int(row["height"]),
        "source_kind": "verified_local_reuse",
    }


def reuse_subset(
    output: Path,
    reuse_work: Path,
    subset: str,
    work_name: str,
    target: int,
    seed: int,
    global_hashes: set[str],
) -> list[dict[str, Any]]:
    part = output / "subsets" / f"{subset}.jsonl"
    if part.is_file():
        rows = read_jsonl(part)
        if len(rows) != target:
            raise RuntimeError(f"{part} has {len(rows):,}/{target:,} rows")
        return rows
    source_rows = read_jsonl(reuse_work / f"{work_name}.jsonl")
    source_rows.sort(
        key=lambda row: stable_rank(seed, subset, str(row.get("i1_key") or ""))
    )
    selected: list[dict[str, Any]] = []
    seen_keys: set[str] = set()
    for source_row in source_rows:
        key = str(source_row.get("i1_key") or "")
        digest = str(source_row.get("image_sha256") or "")
        if not key or key in seen_keys or digest in global_hashes:
            continue
        try:
            row = canonical_reused_row(source_row, output, subset)
        except (OSError, RuntimeError, ValueError):
            continue
        selected.append(row)
        seen_keys.add(key)
        global_hashes.add(digest)
        if len(selected) == target:
            break
    if len(selected) != target:
        raise RuntimeError(
            f"only {len(selected):,}/{target:,} reusable rows for {subset}"
        )
    atomic_jsonl(part, selected)
    return selected


def merge(output: Path) -> dict[str, Any]:
    subsets: dict[str, list[dict[str, Any]]] = {}
    for subset in POPULATIONS:
        path = output / "subsets" / f"{subset}.jsonl"
        if path.is_file():
            subsets[subset] = read_jsonl(path)
    rows = [row for subset in POPULATIONS for row in subsets.get(subset, [])]
    keys = [(str(row["i1_subset"]), str(row["i1_key"])) for row in rows]
    hashes = [str(row["image_sha256"]) for row in rows]
    paths = [Path(str(row["image"])) for row in rows]
    if len(keys) != len(set(keys)):
        raise RuntimeError("duplicate subset/key identity in proportional tranche")
    if len(hashes) != len(set(hashes)):
        raise RuntimeError("duplicate image content in proportional tranche")
    if any(not path.is_file() for path in paths):
        raise RuntimeError("proportional tranche contains missing images")
    if any(not str(row.get("text") or "").strip() for row in rows):
        raise RuntimeError("proportional tranche contains empty captions")
    counts = Counter(str(row["i1_subset"]) for row in rows)
    over_target = {
        subset: counts[subset] - TARGETS[subset]
        for subset in counts
        if counts[subset] > TARGETS[subset]
    }
    if over_target:
        raise RuntimeError(f"subset rows exceed selection plan: {over_target}")
    atomic_jsonl(output / "metadata.jsonl", rows)
    atomic_jsonl(
        output / "train.jsonl",
        (
            {
                "image": row["image"],
                "caption": row["text"],
                "conditioning_text": row["text"],
                "conditioning_kind": "i1_caption",
                "source": f"i1_{row['i1_subset']}",
                "i1_subset": row["i1_subset"],
                "i1_key": row["i1_key"],
                "image_id": row["image_sha256"],
                "width": row["width"],
                "height": row["height"],
            }
            for row in rows
        ),
    )
    receipt = {
        "schema": "rwkv-lab.i1-proportional-tranche-progress.v1",
        "target_rows": TARGET_ROWS,
        "completed_rows": len(rows),
        "remaining_rows": TARGET_ROWS - len(rows),
        "complete": len(rows) == TARGET_ROWS,
        "targets": TARGETS,
        "counts": {subset: counts[subset] for subset in POPULATIONS},
        "completed_subsets": sorted(
            subset for subset in POPULATIONS if counts[subset] == TARGETS[subset]
        ),
        "unique_subset_keys": len(set(keys)),
        "unique_image_hashes": len(set(hashes)),
    }
    atomic_json(output / "progress.json", receipt)
    return receipt


def main() -> None:
    args = parse_args()
    output = args.output.expanduser().resolve()
    reuse_work = args.reuse_work.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    if args.reuse_local:
        global_hashes: set[str] = set()
        for path in sorted((output / "subsets").glob("*.jsonl")):
            global_hashes.update(
                str(row["image_sha256"]) for row in read_jsonl(path)
            )
        for subset, work_name in REUSE_NAMES.items():
            rows = reuse_subset(
                output,
                reuse_work,
                subset,
                work_name,
                TARGETS[subset],
                args.seed,
                global_hashes,
            )
            print(
                json.dumps(
                    {"subset": subset, "rows": len(rows), "target": TARGETS[subset]}
                ),
                flush=True,
            )
    if args.reuse_local or args.merge:
        print(json.dumps(merge(output), indent=2, sort_keys=True))
    if not args.reuse_local and not args.merge:
        raise SystemExit("select --reuse-local and/or --merge")


if __name__ == "__main__":
    main()
