#!/usr/bin/env python3
"""Fetch the four Hugging Face-backed portions of the proportional i1 tranche."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Iterable, Iterator

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import requests
from PIL import Image, ImageOps

from scripts.materialize_i1_proportional_tranche import (
    DEFAULT_OUTPUT,
    SEED,
    TARGETS,
    atomic_jsonl,
    stable_rank,
)


ROOT = Path(__file__).resolve().parents[1]
CAPTIONS = ROOT / "i1-captions"
SOURCE_SPECS = {
    "gptedit": {
        "repo": "UCSC-VLAA/gpt-edit-simpler",
        "revision": "021f1eb3c945a9e9600fa325388b5657e5d31da4",
        "image_column": "output",
        "population": 1_553_575,
    },
    "megalith10m": {
        "repo": "drawthingsai/megalith-10m",
        "revision": "829c00885e3926118674375b50c41d970aceb5ac",
        "image_column": None,
        "population": 9_393_971,
    },
    "rendered_text": {
        "repo": "wendlerc/RenderedText",
        "revision": "9a10151dc56b32f244403aa88127d146eb10e353",
        "image_column": "png",
        "population": 4_700,
    },
    "textatlas": {
        "repo": "CSU-JPG/TextAtlas5M",
        "revision": "f9f2a0f5000fbb078f718197acb45cfb9ceed551",
        "image_column": "image",
    },
}
TEXTATLAS_POPULATIONS = {
    "CleanTextSynth": 1_907_721,
    "CoverBook": 207_566,
    "LongWordsSubset-A": 259_897,
    "LongWordsSubset-M": 1_250_428,
    "PPT2Details": 298_565,
    "PPT2Structured": 96_401,
    "Paper2Text": 356_658,
    "StyledTextSynth": 425_826,
    "TextScenesHQ": 48_935,
    "TextVisionBlend": 546_829,
}
CAPTION_COLUMNS = tuple(f"caption{index}" for index in range(1, 6))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--captions", type=Path, default=CAPTIONS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--subsets",
        nargs="+",
        choices=tuple(SOURCE_SPECS),
        default=list(SOURCE_SPECS),
    )
    parser.add_argument("--shuffle-buffer", type=int, default=2_048)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument(
        "--direct-url-mib-per-second",
        type=float,
        default=8.0,
        help="per-process cap for Megalith's direct image URLs",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def proportional_targets(populations: dict[str, int], total: int) -> dict[str, int]:
    denominator = sum(populations.values())
    targets = {
        name: total * population // denominator
        for name, population in populations.items()
    }
    remaining = total - sum(targets.values())
    order = sorted(
        populations,
        key=lambda name: (total * populations[name] % denominator, name),
        reverse=True,
    )
    for name in order[:remaining]:
        targets[name] += 1
    return targets


def selected_caption(row: dict[str, Any], key: str, seed: int) -> tuple[str, str]:
    values = [
        (column, str(row.get(column) or "").strip())
        for column in CAPTION_COLUMNS
        if str(row.get(column) or "").strip()
    ]
    if not values:
        raise RuntimeError(f"no nonempty i1 caption for {key}")
    digest = hashlib.sha256(f"{seed}:{key}".encode()).digest()
    column, text = values[int.from_bytes(digest[:8], "big") % len(values)]
    return text, column


def load_matching_captions(
    directory: Path, wanted: set[str]
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    options = pc.SetLookupOptions(
        value_set=pa.array(sorted(wanted), type=pa.string())
    )
    for path in sorted(directory.glob("*.parquet")):
        parquet = pq.ParquetFile(path)
        available = [
            column for column in CAPTION_COLUMNS if column in parquet.schema_arrow.names
        ]
        for group in range(parquet.metadata.num_row_groups):
            table = parquet.read_row_group(group, columns=["key", *available])
            table = table.filter(pc.is_in(table["key"], options=options))
            for row in table.to_pylist():
                result[str(row["key"])] = row
        print(
            json.dumps(
                {
                    "phase": "caption_join",
                    "subset": directory.name,
                    "matched": len(result),
                    "wanted": len(wanted),
                    "file": path.name,
                }
            ),
            flush=True,
        )
        if len(result) == len(wanted):
            break
    return result


def row_key(subset: str, row: dict[str, Any], config: str | None) -> str:
    if subset == "gptedit":
        return str(row["id"])
    if subset == "megalith10m":
        return str(row["key"])
    if subset == "rendered_text":
        return str(row["__key__"])
    if subset == "textatlas":
        assert config is not None
        return f"{config}:{row['image_path']}"
    raise AssertionError(subset)


def raw_image_bytes(value: Any) -> bytes:
    if isinstance(value, dict) and isinstance(value.get("bytes"), bytes):
        return value["bytes"]
    if isinstance(value, bytes):
        return value
    raise RuntimeError("streamed image row has no original byte payload")


def rate_limited_get(url: str, mib_per_second: float) -> bytes:
    limit = mib_per_second * 1024 * 1024
    started = time.monotonic()
    payload = bytearray()
    with requests.get(
        url,
        stream=True,
        timeout=(15, 60),
        headers={"User-Agent": "rwkv-lab-i1-proportional-tranche/1.0"},
    ) as response:
        response.raise_for_status()
        for chunk in response.iter_content(256 * 1024):
            if not chunk:
                continue
            payload.extend(chunk)
            expected = len(payload) / limit
            elapsed = time.monotonic() - started
            if expected > elapsed:
                time.sleep(expected - elapsed)
    return bytes(payload)


def rate_limited_json(
    url: str, params: dict[str, Any], mib_per_second: float
) -> dict[str, Any]:
    prepared = requests.Request("GET", url, params=params).prepare().url
    last_error: Exception | None = None
    for attempt in range(6):
        try:
            payload = rate_limited_get(str(prepared), mib_per_second)
            value = json.loads(payload)
            if not isinstance(value, dict):
                raise RuntimeError("dataset viewer returned a non-object response")
            return value
        except (requests.RequestException, json.JSONDecodeError, RuntimeError) as error:
            last_error = error
            time.sleep(min(30, 2**attempt))
    raise RuntimeError(f"dataset viewer request failed: {last_error}")


def verified_payload(payload: bytes) -> tuple[str, int, int, str]:
    with Image.open(io.BytesIO(payload)) as image:
        # EXIF transposition may return a new PIL image without ``format``.
        image_format = str(image.format or "").upper()
        encoded_width, encoded_height = image.size
        if encoded_width * encoded_height > 64_000_000:
            raise RuntimeError(
                f"image exceeds 64 MP: {encoded_width}x{encoded_height}"
            )
        image.load()
        image = ImageOps.exif_transpose(image)
        width, height = image.size
    extension = {
        "JPEG": ".jpg",
        "PNG": ".png",
        "WEBP": ".webp",
    }.get(image_format)
    if extension is None or width < 16 or height < 16:
        raise RuntimeError(
            f"unsupported or invalid image: {image_format} {width}x{height}"
        )
    return extension, width, height, hashlib.sha256(payload).hexdigest()


def materialize_candidate(
    output: Path,
    subset: str,
    key: str,
    payload: bytes,
    *,
    config: str | None,
    source_url: str | None,
) -> dict[str, Any]:
    extension, width, height, digest = verified_payload(payload)
    name = hashlib.sha256(f"{subset}:{key}".encode()).hexdigest()
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
        "i1_subset": subset,
        "i1_key": key,
        "image_sha256": digest,
        "width": width,
        "height": height,
        "source_kind": "pinned_huggingface_stream",
        "source_revision": SOURCE_SPECS[subset]["revision"],
        **({"source_config": config} if config is not None else {}),
        **({"source_url": source_url} if source_url is not None else {}),
    }


def source_stream(
    subset: str,
    *,
    config: str | None,
    seed: int,
    shuffle_buffer: int,
    wanted: int,
    direct_url_rate: float,
) -> Iterator[dict[str, Any]]:
    spec = SOURCE_SPECS[subset]
    population = (
        TEXTATLAS_POPULATIONS[config]
        if subset == "textatlas" and config is not None
        else int(spec["population"])
    )
    # Use distributed 100-row windows so the slice spans the source rather
    # than taking one chronological prefix. The viewer serves cached assets
    # from the pinned commit recorded in SOURCE_SPECS.
    source_multiplier = 4.0 if subset == "megalith10m" else 2.0
    planned = max(wanted + 256, math.ceil(wanted * source_multiplier))
    batches = math.ceil(planned / 100)
    rotation = int.from_bytes(
        hashlib.sha256(f"{seed}:{subset}:{config or ''}".encode()).digest()[:8],
        "big",
    ) / 2**64
    for batch in range(batches):
        offset = int(((batch + rotation) / batches % 1.0) * population)
        length = min(100, population - offset)
        try:
            response = rate_limited_json(
                "https://datasets-server.huggingface.co/rows",
                {
                    "dataset": spec["repo"],
                    "config": config or "default",
                    "split": "train",
                    "offset": offset,
                    "length": length,
                },
                direct_url_rate,
            )
        except Exception as error:
            print(
                json.dumps(
                    {
                        "phase": "viewer_window_failure",
                        "subset": subset,
                        "config": config,
                        "offset": offset,
                        "length": length,
                        "error": f"{type(error).__name__}: {error}",
                    }
                ),
                flush=True,
            )
            continue
        for item in response.get("rows", ()):
            row = item.get("row")
            if isinstance(row, dict):
                yield row


def collect_candidates(
    output: Path,
    subset: str,
    *,
    config: str | None,
    wanted: int,
    seed: int,
    shuffle_buffer: int,
    direct_url_rate: float,
    workers: int,
) -> list[dict[str, Any]]:
    suffix = f".{config}" if config is not None else ""
    state_path = output / "work" / f"{subset}{suffix}.candidates.jsonl"
    candidates = [
        row for row in read_jsonl(state_path) if Path(str(row["image"])).is_file()
    ]
    by_key = {str(row["i1_key"]): row for row in candidates}
    if len(by_key) >= wanted:
        return list(by_key.values())
    failures = 0
    stream_seed = int.from_bytes(
        hashlib.sha256(f"{seed}:{subset}:{config or ''}".encode()).digest()[:8],
        "big",
    )
    stream = iter(source_stream(
        subset,
        config=config,
        seed=stream_seed,
        shuffle_buffer=shuffle_buffer,
        wanted=wanted,
        direct_url_rate=direct_url_rate,
    ))

    def fetch(row: dict[str, Any]) -> tuple[str, dict[str, Any] | None, str | None]:
        key = row_key(subset, row, config)
        try:
            if subset == "megalith10m":
                if str(row.get("status") or "") != "success":
                    return key, None, None
                url = str(row.get("url") or "")
                if not url:
                    return key, None, None
                payload = rate_limited_get(url, direct_url_rate)
            else:
                image = row[SOURCE_SPECS[subset]["image_column"]]
                url = str(image.get("src") or "")
                if not url:
                    raise RuntimeError("dataset viewer row has no image URL")
                if SOURCE_SPECS[subset]["revision"] not in url:
                    raise RuntimeError("dataset viewer asset is not from pinned revision")
                payload = rate_limited_get(url, direct_url_rate)
            candidate = materialize_candidate(
                output,
                subset,
                key,
                payload,
                config=config,
                source_url=url.split("?", 1)[0] if url else None,
            )
        except Exception as error:
            return key, None, f"{type(error).__name__}: {error}"
        return key, candidate, None

    exhausted = False
    with ThreadPoolExecutor(max_workers=workers) as executor:
        while len(by_key) < wanted and not exhausted:
            batch: list[dict[str, Any]] = []
            while len(batch) < workers * 4:
                try:
                    row = next(stream)
                except StopIteration:
                    exhausted = True
                    break
                key = row_key(subset, row, config)
                if key not in by_key:
                    batch.append(row)
            if not batch:
                continue
            for key, candidate, error in executor.map(fetch, batch):
                if error is not None:
                    failures += 1
                    if failures <= 20 or failures % 100 == 0:
                        print(
                            json.dumps(
                                {
                                    "phase": "candidate_failure",
                                    "subset": subset,
                                    "config": config,
                                    "key": key,
                                    "failures": failures,
                                    "error": error,
                                }
                            ),
                            flush=True,
                        )
                    continue
                if candidate is not None:
                    by_key[key] = candidate
            atomic_jsonl(state_path, by_key.values())
            print(
                json.dumps(
                    {
                        "phase": "candidates",
                        "subset": subset,
                        "config": config,
                        "rows": len(by_key),
                        "wanted": wanted,
                        "failures": failures,
                    }
                ),
                flush=True,
            )
    atomic_jsonl(state_path, by_key.values())
    if len(by_key) < wanted:
        raise RuntimeError(
            f"source stream ended at {len(by_key):,}/{wanted:,} for "
            f"{subset}:{config}"
        )
    return list(by_key.values())


def finalize_subset(
    output: Path,
    captions_root: Path,
    subset: str,
    candidates: list[dict[str, Any]],
    target: int,
    seed: int,
) -> list[dict[str, Any]]:
    captions = load_matching_captions(
        captions_root / subset, {str(row["i1_key"]) for row in candidates}
    )
    selected: list[dict[str, Any]] = []
    seen_hashes: set[str] = set()
    for row in sorted(
        candidates,
        key=lambda item: stable_rank(seed, subset, str(item["i1_key"])),
    ):
        key = str(row["i1_key"])
        caption_row = captions.get(key)
        digest = str(row["image_sha256"])
        if caption_row is None or digest in seen_hashes:
            continue
        text, variant = selected_caption(caption_row, key, seed)
        selected.append(
            {
                **row,
                "text": text,
                "caption": text,
                "caption_variant": variant,
                "caption_policy": "deterministic_existing_i1_variant",
            }
        )
        seen_hashes.add(digest)
        if len(selected) == target:
            break
    if len(selected) != target:
        raise RuntimeError(
            f"caption join produced {len(selected):,}/{target:,} rows for {subset}"
        )
    atomic_jsonl(output / "subsets" / f"{subset}.jsonl", selected)
    return selected


def fetch_subset(args: argparse.Namespace, subset: str) -> list[dict[str, Any]]:
    target = TARGETS[subset]
    complete = args.output / "subsets" / f"{subset}.jsonl"
    if complete.is_file():
        rows = read_jsonl(complete)
        if len(rows) != target:
            raise RuntimeError(f"{complete} has {len(rows):,}/{target:,} rows")
        return rows
    reserve = max(128, math.ceil(target * (0.35 if subset == "megalith10m" else 0.10)))
    if subset != "textatlas":
        candidates = collect_candidates(
            args.output,
            subset,
            config=None,
            wanted=target + reserve,
            seed=args.seed,
            shuffle_buffer=args.shuffle_buffer,
            direct_url_rate=args.direct_url_mib_per_second,
            workers=args.workers,
        )
    else:
        config_targets = proportional_targets(TEXTATLAS_POPULATIONS, target + reserve)
        candidates = []
        for config, config_target in config_targets.items():
            candidates.extend(
                collect_candidates(
                    args.output,
                    subset,
                    config=config,
                    wanted=config_target,
                    seed=args.seed,
                    shuffle_buffer=args.shuffle_buffer,
                    direct_url_rate=args.direct_url_mib_per_second,
                    workers=args.workers,
                )
            )
    return finalize_subset(
        args.output, args.captions, subset, candidates, target, args.seed
    )


def main() -> None:
    args = parse_args()
    args.output = args.output.expanduser().resolve()
    args.captions = args.captions.expanduser().resolve()
    if (
        args.shuffle_buffer < 1
        or args.direct_url_mib_per_second <= 0
        or args.workers < 1
    ):
        raise SystemExit("shuffle buffer, rate, and workers must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    receipt = {
        "schema": "rwkv-lab.i1-proportional-hf-acquisition.v1",
        "subsets": {},
    }
    for subset in args.subsets:
        rows = fetch_subset(args, subset)
        receipt["subsets"][subset] = {
            "rows": len(rows),
            "target": TARGETS[subset],
            "revision": SOURCE_SPECS[subset]["revision"],
        }
        atomic_json(args.output / "hf_acquisition_receipt.json", receipt)
        print(json.dumps({"complete": subset, **receipt["subsets"][subset]}), flush=True)


if __name__ == "__main__":
    main()
