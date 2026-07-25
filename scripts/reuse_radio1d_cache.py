#!/usr/bin/env python3
"""Hardlink byte-identical RADIO cache entries into a new manifest cache."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rwkv_lab.radio1d_cache import (  # noqa: E402
    CACHE_SCHEMA, LEGACY_CACHE_SCHEMA, RadioCacheMetadata, cache_is_current,
    cache_path,
)
from rwkv_lab.radio1d_rwkv import (  # noqa: E402
    DEFAULT_ADAPTIVE_TOKEN_THRESHOLD, RADIO_HIDDEN_SIZE,
)


DEFAULT_REVISION = "e18692120c7a3203496e1a99056a4149ede135fc"


def manifest_rows(paths: list[Path]):
    for manifest in paths:
        with manifest.open() as handle:
            for index, line in enumerate(handle):
                if not line.strip():
                    continue
                row = json.loads(line)
                value = row.get("image") or row.get("image_path")
                if not value:
                    raise ValueError(f"{manifest}:{index + 1} has no image path")
                source = Path(value)
                if not source.is_absolute():
                    source = ROOT / source
                yield manifest, index, source, row.get("image_sha256")


def cache_fingerprint(path: Path, revision: str, threshold: int) -> str | None:
    """Read and validate a cache header without loading its large tensor."""
    try:
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            header = handle.metadata() or {}
            if header.get("schema") not in (CACHE_SCHEMA, LEGACY_CACHE_SCHEMA):
                return None
            metadata = RadioCacheMetadata.from_json(header["radio_metadata"])
            shape = tuple(handle.get_slice("global_tokens").get_shape())
        if metadata.checkpoint_revision != revision:
            return None
        if metadata.adaptive_token_threshold != threshold:
            return None
        expected = (len(metadata.tiles), metadata.tokens_per_tile, RADIO_HIDDEN_SIZE)
        if shape != expected:
            return None
        return metadata.source_sha256
    except (KeyError, OSError, RuntimeError, ValueError):
        return None


def historical_cache_path(cache_dir: Path, source: Path) -> Path:
    """Reproduce cache_path without filesystem-touching realpath traversal.

    The historical manifests and repository/cache roots are absolute and were
    verified not to contain symlink components.  This avoids 400k directory
    lookups merely to discover which 60k deterministic cache names exist.
    """
    absolute = os.path.abspath(os.fsencode(source))
    name = hashlib.sha256(absolute).hexdigest()
    return cache_dir / name[:2] / f"{name}.safetensors"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-cache", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, nargs="+", required=True)
    parser.add_argument("--destination-cache", type=Path, required=True)
    parser.add_argument("--destination-manifest", type=Path, nargs="+", required=True)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--adaptive-token-threshold", type=int,
                        default=DEFAULT_ADAPTIVE_TOKEN_THRESHOLD)
    args = parser.parse_args()

    # Directory enumeration is sequential and cheap.  Calling is_file() for
    # every one of the 400k historical rows causes hundreds of thousands of
    # random metadata reads on large spinning arrays.
    existing_names = {
        path.name
        for shard in args.source_cache.iterdir() if shard.is_dir()
        for path in shard.glob("*.safetensors")
    }
    print(json.dumps({
        "phase": "inventory_source", "cache_files": len(existing_names),
    }, sort_keys=True), flush=True)

    by_sha: dict[str, Path] = {}
    candidates = valid = duplicate_hashes = scanned = 0
    for _manifest, _index, source, declared_sha in manifest_rows(
            args.source_manifest):
        scanned += 1
        if scanned == 1 or scanned % 25000 == 0:
            print(json.dumps({
                "phase": "index_source", "rows_scanned": scanned,
                "cache_candidates": candidates, "valid": valid,
                "unique_hashes": len(by_sha),
            }, sort_keys=True), flush=True)
        candidate = historical_cache_path(args.source_cache, source)
        if candidate.name not in existing_names:
            continue
        candidates += 1
        if declared_sha and len(declared_sha) == 64:
            # The original manifest SHA and deterministic cache filename form
            # the fast index.  Every reused entry is still header-validated
            # against this SHA at its destination below.
            actual_sha = declared_sha
        else:
            actual_sha = cache_fingerprint(
                candidate, args.revision, args.adaptive_token_threshold)
            if actual_sha is None:
                continue
        valid += 1
        if actual_sha in by_sha:
            duplicate_hashes += 1
        else:
            by_sha[actual_sha] = candidate

    args.destination_cache.mkdir(parents=True, exist_ok=True)
    rows = already = linked = missing = invalid = 0
    for manifest, index, source, source_sha in manifest_rows(
            args.destination_manifest):
        rows += 1
        if not source_sha or len(source_sha) != 64:
            invalid += 1
            continue
        target = cache_path(args.destination_cache, source)
        if cache_is_current(
                target, source, args.revision,
                adaptive_token_threshold=args.adaptive_token_threshold,
                source_sha256=source_sha):
            already += 1
            continue
        reusable = by_sha.get(source_sha)
        if reusable is None:
            missing += 1
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f".{target.name}.{os.getpid()}.link")
        temporary.unlink(missing_ok=True)
        try:
            os.link(reusable, temporary)
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
        if not cache_is_current(
                target, source, args.revision,
                adaptive_token_threshold=args.adaptive_token_threshold,
                source_sha256=source_sha):
            target.unlink(missing_ok=True)
            invalid += 1
            continue
        linked += 1
        if linked == 1 or linked % 1000 == 0:
            print(json.dumps({
                "phase": "link_destination", "manifest": str(manifest),
                "row": index, "linked": linked, "already": already,
                "missing": missing,
            }, sort_keys=True), flush=True)

    print(json.dumps({
        "status": "complete", "source_candidates": candidates,
        "source_valid": valid, "source_unique_hashes": len(by_sha),
        "source_duplicate_hashes": duplicate_hashes,
        "destination_rows": rows, "already_current": already,
        "hardlinked": linked, "requires_encoding": missing,
        "invalid_rows_or_links": invalid,
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
