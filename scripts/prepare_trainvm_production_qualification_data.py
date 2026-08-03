#!/usr/bin/env python3
"""Freeze small, real data inputs for TrainVM production qualification.

The output contains a balanced MageFlow train/eval snapshot and a deterministic
uint32 transformer token stream.  It is deliberately new-only and carries
source/content hashes so the much larger source datasets never need to become
qualification content roots.  The resulting files are inputs to
``materialize_trainvm_production_qualification.py``; this tool grants no launch
authority and does not compile code or initialize an accelerator.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from array import array
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

SNAPSHOT_VERSION = "trainvm.production-qualification-data/v1"
DOMAINS = ("animation", "photo")
MAXIMUM_LINE_BYTES = 4 * 1024 * 1024
MAXIMUM_ROWS = 10_000_000


class SnapshotError(ValueError):
    """Source data cannot produce the closed qualification snapshot."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _regular_file(value: Path, label: str) -> Path:
    if value.is_symlink():
        raise SnapshotError(f"{label} must not be a symlink")
    try:
        path = value.expanduser().resolve(strict=True)
    except OSError as error:
        raise SnapshotError(f"{label} is unavailable: {error}") from error
    if not path.is_file():
        raise SnapshotError(f"{label} must be a regular file")
    return path


def _directory(value: Path, label: str) -> Path:
    if value.is_symlink():
        raise SnapshotError(f"{label} must not be a symlink")
    try:
        path = value.expanduser().resolve(strict=True)
    except OSError as error:
        raise SnapshotError(f"{label} is unavailable: {error}") from error
    if not path.is_dir():
        raise SnapshotError(f"{label} must be a directory")
    return path


def _row_identity(row: Mapping[str, Any], image: Path) -> str:
    for field in ("image_id", "image_sha256"):
        value = row.get(field)
        if isinstance(value, str) and value:
            return value
    return _sha256(image)


def _candidate(
    row: Any,
    *,
    manifest: Path,
    line_number: int,
) -> dict[str, Any]:
    if not isinstance(row, dict):
        raise SnapshotError(f"{manifest}:{line_number} is not an object")
    domain = row.get("domain")
    if domain not in DOMAINS:
        raise SnapshotError(f"{manifest}:{line_number} has an unsupported domain")
    caption = row.get("caption") or row.get("text")
    if not isinstance(caption, str) or not caption.strip():
        raise SnapshotError(f"{manifest}:{line_number} has no caption")
    raw_image = row.get("image") or row.get("image_path")
    if not isinstance(raw_image, str) or not raw_image:
        raise SnapshotError(f"{manifest}:{line_number} has no image path")
    image = Path(raw_image).expanduser()
    if not image.is_absolute():
        image = manifest.parent / image
    try:
        image = image.resolve(strict=True)
    except OSError as error:
        raise SnapshotError(
            f"{manifest}:{line_number} image is unavailable: {error}"
        ) from error
    if not image.is_file():
        raise SnapshotError(f"{manifest}:{line_number} image is not a file")
    width = row.get("train_width")
    height = row.get("train_height")
    latent_tokens = row.get("latent_tokens")
    for field, value in (
        ("train_width", width),
        ("train_height", height),
        ("latent_tokens", latent_tokens),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise SnapshotError(f"{manifest}:{line_number} has invalid {field}")
    identity = _row_identity(row, image)
    if not identity or len(identity) > 512 or "\x00" in identity:
        raise SnapshotError(f"{manifest}:{line_number} has an invalid image identity")
    normalized = dict(row)
    normalized["caption"] = caption.strip()
    normalized["image_id"] = identity
    return {
        "domain": domain,
        "identity": identity,
        "image": image,
        "line_number": line_number,
        "sort_key": (latent_tokens, identity, line_number),
        "row": normalized,
    }


def _select(
    manifest: Path,
    *,
    per_domain: int,
    excluded: frozenset[str] = frozenset(),
) -> list[dict[str, Any]]:
    selected: dict[str, list[dict[str, Any]]] = {domain: [] for domain in DOMAINS}
    seen: set[str] = set()
    with manifest.open("rb") as handle:
        for line_number in range(1, MAXIMUM_ROWS + 2):
            encoded = handle.readline(MAXIMUM_LINE_BYTES + 1)
            if not encoded:
                break
            if len(encoded) > MAXIMUM_LINE_BYTES:
                raise SnapshotError(f"{manifest}:{line_number} exceeds the line bound")
            if not encoded.strip():
                continue
            try:
                value = json.loads(encoded)
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise SnapshotError(f"{manifest}:{line_number} is malformed") from error
            candidate = _candidate(value, manifest=manifest, line_number=line_number)
            identity = candidate["identity"]
            if identity in excluded or identity in seen:
                continue
            seen.add(identity)
            bucket = selected[candidate["domain"]]
            bucket.append(candidate)
            bucket.sort(key=lambda item: item["sort_key"])
            del bucket[per_domain:]
        else:
            raise SnapshotError(f"{manifest} exceeds the row bound")
    for domain, values in selected.items():
        if len(values) != per_domain:
            raise SnapshotError(
                f"{manifest} has only {len(values)} eligible {domain} rows; "
                f"need {per_domain}"
            )
    # Round-robin ordering ensures a two-sample gallery contains both routes.
    return [selected[domain][index] for index in range(per_domain) for domain in DOMAINS]


def _safe_suffix(path: Path) -> str:
    suffix = path.suffix.lower()
    if 1 < len(suffix) <= 12 and suffix[1:].isalnum():
        return suffix
    return ".image"


def _freeze_rows(
    selected: Sequence[Mapping[str, Any]],
    *,
    split: str,
    staging_root: Path,
    published_root: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    image_root = staging_root / "images" / split
    image_root.mkdir(parents=True)
    rows: list[dict[str, Any]] = []
    evidence: list[dict[str, Any]] = []
    for index, candidate in enumerate(selected):
        source = Path(candidate["image"])
        identity = str(candidate["identity"])
        filename = f"{index:03d}-{candidate['domain']}-{hashlib.sha256(identity.encode()).hexdigest()[:20]}{_safe_suffix(source)}"
        target = image_root / filename
        shutil.copyfile(source, target)
        source_digest = _sha256(source)
        target_digest = _sha256(target)
        if source_digest != target_digest:
            raise SnapshotError(f"copied image digest mismatch for {source}")
        row = dict(candidate["row"])
        published_target = (published_root / "images" / split / filename).resolve()
        row["image"] = str(published_target)
        row.pop("image_path", None)
        row["qualification_source_image_sha256"] = source_digest
        row["qualification_source_line"] = int(candidate["line_number"])
        rows.append(row)
        evidence.append(
            {
                "domain": candidate["domain"],
                "image_id": identity,
                "source_line": candidate["line_number"],
                "source_image_sha256": source_digest,
                "snapshot_path": str(published_target),
            }
        )
    return rows, evidence


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    row,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                    allow_nan=False,
                )
                + "\n"
            )


def _transformer_tokens(
    tokenizer_directory: Path,
    text_sources: Sequence[Path],
    minimum_tokens: int,
    output: Path,
) -> dict[str, Any]:
    if sys.byteorder != "little" or array("I").itemsize != 4:
        raise SnapshotError("uint32 token publication requires a little-endian 32-bit array")
    try:
        from tokenizers import Tokenizer
    except ImportError as error:
        raise SnapshotError("the tokenizers package is required") from error
    tokenizer_path = _regular_file(tokenizer_directory / "tokenizer.json", "tokenizer.json")
    config_path = _regular_file(tokenizer_directory / "config.json", "tokenizer config")
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
        text_config = config.get("text_config", config)
        eos = text_config["eos_token_id"]
    except (UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError) as error:
        raise SnapshotError("tokenizer model config has no EOS identity") from error
    if isinstance(eos, bool) or not isinstance(eos, int) or not 0 <= eos < 1 << 32:
        raise SnapshotError("tokenizer EOS identity is invalid")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    documents: list[list[int]] = []
    source_evidence = []
    for source in text_sources:
        text = source.read_text(encoding="utf-8")
        ids = tokenizer.encode(text, add_special_tokens=False).ids
        if not ids:
            raise SnapshotError(f"transformer text source tokenized empty: {source}")
        if any(isinstance(value, bool) or not 0 <= value < 1 << 32 for value in ids):
            raise SnapshotError(f"transformer text source has an invalid token: {source}")
        documents.append([*ids, eos])
        source_evidence.append(
            {"path": str(source), "sha256": _sha256(source), "tokens_with_eos": len(ids) + 1}
        )
    values = array("I")
    while len(values) < minimum_tokens:
        for document in documents:
            remaining = minimum_tokens - len(values)
            values.extend(document[:remaining])
            if len(values) == minimum_tokens:
                break
    output.parent.mkdir(parents=True)
    with output.open("xb") as handle:
        values.tofile(handle)
    return {
        "dtype": "uint32-le",
        "total_tokens": len(values),
        "eos_token_id": eos,
        "tokenizer": str(tokenizer_directory),
        "tokenizer_json_sha256": _sha256(tokenizer_path),
        "sources": source_evidence,
        "tokens_sha256": _sha256(output),
    }


def prepare(
    *,
    train_manifest: Path,
    eval_manifest: Path,
    tokenizer_directory: Path,
    text_sources: Sequence[Path],
    destination: Path,
    train_per_domain: int,
    eval_per_domain: int,
    transformer_tokens: int,
) -> Path:
    train_manifest = _regular_file(train_manifest, "MageFlow train manifest")
    eval_manifest = _regular_file(eval_manifest, "MageFlow eval manifest")
    tokenizer_directory = _directory(tokenizer_directory, "tokenizer directory")
    sources = [_regular_file(path, f"transformer text source {index}") for index, path in enumerate(text_sources)]
    if not sources:
        raise SnapshotError("at least one transformer text source is required")
    for label, value in (
        ("train_per_domain", train_per_domain),
        ("eval_per_domain", eval_per_domain),
        ("transformer_tokens", transformer_tokens),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 1_000_000:
            raise SnapshotError(f"{label} is outside the supported bound")
    if destination.exists() or destination.is_symlink():
        raise SnapshotError(f"destination already exists: {destination}")
    parent = destination.parent.resolve(strict=True)
    staging = parent / f".{destination.name}.tmp-{os.getpid()}"
    if staging.exists() or staging.is_symlink():
        raise SnapshotError(f"staging path already exists: {staging}")
    staging.mkdir(mode=0o700)
    try:
        train_selected = _select(train_manifest, per_domain=train_per_domain)
        train_ids = frozenset(str(item["identity"]) for item in train_selected)
        eval_selected = _select(
            eval_manifest,
            per_domain=eval_per_domain,
            excluded=train_ids,
        )
        train_rows, train_evidence = _freeze_rows(
            train_selected,
            split="train",
            staging_root=staging / "mageflow",
            published_root=destination / "mageflow",
        )
        eval_rows, eval_evidence = _freeze_rows(
            eval_selected,
            split="eval",
            staging_root=staging / "mageflow",
            published_root=destination / "mageflow",
        )
        train_output = staging / "mageflow" / "train.jsonl"
        eval_output = staging / "mageflow" / "eval.jsonl"
        _write_jsonl(train_output, train_rows)
        _write_jsonl(eval_output, eval_rows)
        token_output = staging / "transformer" / "tokens.bin"
        token_evidence = _transformer_tokens(
            tokenizer_directory,
            sources,
            transformer_tokens,
            token_output,
        )
        report = {
            "api_version": SNAPSHOT_VERSION,
            "mageflow": {
                "domains": list(DOMAINS),
                "train_manifest": str((destination / "mageflow" / "train.jsonl").resolve()),
                "eval_manifest": str((destination / "mageflow" / "eval.jsonl").resolve()),
                "image_root": str((destination / "mageflow" / "images").resolve()),
                "source_train_manifest": str(train_manifest),
                "source_train_manifest_sha256": _sha256(train_manifest),
                "source_eval_manifest": str(eval_manifest),
                "source_eval_manifest_sha256": _sha256(eval_manifest),
                "train": train_evidence,
                "eval": eval_evidence,
            },
            "transformer": {
                **token_evidence,
                "tokens_path": str((destination / "transformer" / "tokens.bin").resolve()),
            },
        }
        (staging / "snapshot.json").write_text(
            json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(staging, destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return destination / "snapshot.json"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--eval-manifest", type=Path, required=True)
    parser.add_argument("--tokenizer-directory", type=Path, required=True)
    parser.add_argument("--text-source", type=Path, action="append", required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--train-per-domain", type=int, default=4)
    parser.add_argument("--eval-per-domain", type=int, default=2)
    parser.add_argument("--transformer-tokens", type=int, default=32768)
    arguments = parser.parse_args(argv)
    try:
        report = prepare(
            train_manifest=arguments.train_manifest,
            eval_manifest=arguments.eval_manifest,
            tokenizer_directory=arguments.tokenizer_directory,
            text_sources=arguments.text_source,
            destination=arguments.destination,
            train_per_domain=arguments.train_per_domain,
            eval_per_domain=arguments.eval_per_domain,
            transformer_tokens=arguments.transformer_tokens,
        )
    except SnapshotError as error:
        print(f"qualification data snapshot rejected: {error}", file=sys.stderr)
        return 2
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
