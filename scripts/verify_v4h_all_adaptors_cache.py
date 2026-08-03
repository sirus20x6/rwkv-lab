#!/usr/bin/env python3
"""Audit every all-adaptor cache entry referenced by one or more manifests.

Two modes, because they cost three orders of magnitude apart:

``--fast`` (the default) audits from ``stat()`` alone. It reports presence and
file geometry -- the size must be one contiguous bf16 grid plus a header
smaller than one cell -- and nothing else. This is the same tradeoff, for the
same reason, that ``native_cache_token_count`` documents: opening a file and
touching one byte faults in a filesystem readahead window, measured at ~1.6 MB
of real block I/O per entry on the array holding these caches, so a header read
across ~107k entries is hours of seeks. ``stat()`` touches no data blocks.

``--deep`` (alias ``--exact``) is the old behaviour: open every entry's
safetensors header and compare schema, revision, source SHA, max edge, feature
width and tensor shape. Use it when the operator actually wants byte-level
assurance about the producing configuration, not just that the bytes are there.

The fast mode exists because it catches the failure the launcher cares about --
cache_v4h_all_adaptors.py counts a per-image OOM as a tolerated failure, so an
OOM storm leaves holes and truncated entries behind -- while remaining cheap
enough to run on every launch instead of being skipped with
V4H_SKIP_CACHE_VERIFY=1.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rwkv_lab.radio_v4h import (  # noqa: E402
    NATIVE_CACHE_SCHEMA, V4H_MAX_EDGE, V4H_PATCH, cache_path,
    native_cache_token_count)
from rwkv_lab.radio_v4h_adaptors import FUSED_ADAPTOR_WIDTH  # noqa: E402
from cache_v4h_all_adaptors import DEFAULT_REVISION  # noqa: E402

FAST_DETECTS = [
    "a missing entry (the hole an OOM storm leaves behind)",
    "a truncated or zero-length entry",
    "a size that is not one bf16 grid plus a sub-cell header",
    "a cell count that cannot come from a capped, column-paired native grid",
]
FAST_CANNOT_DETECT = [
    "a wrong checkpoint_revision -- the header is never read",
    "a source_sha256 that no longer matches the manifest",
    "a wrong max_edge",
    "a wrong feature width: an incorrect hidden_size is absorbed into the "
    "inferred header length and usually still yields a plausible size, exactly "
    "as native_cache_token_count's docstring warns",
    "a tensor shape that disagrees with its own recorded grid",
    "any corruption inside the payload bytes",
]
DEEP_DETECTS = FAST_DETECTS[:1] + [
    "a wrong schema, checkpoint_revision, max_edge or hidden_size",
    "a source_sha256 that no longer matches the manifest",
    "a tensor shape that disagrees with its own recorded grid metadata",
]
DEEP_CANNOT_DETECT = [
    "corruption inside the payload bytes -- only the header and the tensor "
    "index are read, never the grid itself",
]


def geometry_reason(target: Path, hidden_size: int, max_edge: int) -> str | None:
    """Fast-mode verdict for one entry, or None when it looks sound."""
    try:
        cells = native_cache_token_count(target, hidden_size=hidden_size)
    except FileNotFoundError:
        return "absent"
    except OSError:
        return "unreadable"
    except ValueError:
        return "bad-size-geometry"
    # A native grid is capped at max_edge/V4H_PATCH cells per axis and its width
    # is snapped to an even number of patches so column pairing has no ragged
    # edge, so an odd or oversized cell count cannot have been written here.
    limit = (max_edge // V4H_PATCH) ** 2
    if cells % 2 or not 0 < cells <= limit:
        return "implausible-grid"
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, nargs="+", required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    # The writer exposes --max-edge, so a cache built at a non-default cap is a
    # perfectly valid cache; hardcoding V4H_MAX_EDGE here reported all of it as
    # invalid.
    parser.add_argument("--max-edge", type=int, default=V4H_MAX_EDGE)
    parser.add_argument("--workers", type=int, default=64)
    parser.add_argument(
        "--deep", "--exact", dest="deep", action="store_true",
        help="open every entry's safetensors header and check the producing "
             "configuration (schema, revision, source SHA, max edge, feature "
             "width, tensor shape). Authoritative but expensive: each open "
             "faults in a filesystem readahead window, ~1.6 MB of real block "
             "I/O per entry on this array, which is hours across 107k entries.")
    parser.add_argument(
        "--fast", dest="deep", action="store_false",
        help="default. Audit presence and size geometry from stat() alone; no "
             "file contents are read, so nothing about the producing "
             "configuration is checked.")
    parser.set_defaults(deep=False)
    args = parser.parse_args()

    rows: dict[Path, str | None] = {}
    for manifest in args.manifest:
        with manifest.open() as handle:
            for line in handle:
                value = json.loads(line)
                source = Path(value.get("image") or value["image_path"])
                source = source if source.is_absolute() else ROOT / source
                rows[source.resolve()] = value.get("image_sha256")

    mode = "deep" if args.deep else "fast"
    print(json.dumps({
        "phase": "start", "mode": mode, "entries": len(rows),
        "audit": ("safetensors header + tensor index per entry"
                  if args.deep else
                  "stat() size geometry only -- no file contents are read"),
    }, sort_keys=True), file=sys.stderr, flush=True)

    def deep(item):
        source, source_sha256 = item
        target = cache_path(args.cache_dir, source)
        # The manifest SHA is the authoritative source identity and was checked
        # by every writer, so the source file itself is never opened or
        # statted: 107k stats in random manifest order turn this audit into
        # hours of seeks on spinning storage. Cache targets are sorted below so
        # their sharded directories are traversed sequentially. A row WITHOUT a
        # manifest SHA simply skips the identity comparison -- it must not fall
        # back to native_cache_is_current, which stats the source and
        # reintroduces exactly the I/O this avoids.
        try:
            with safe_open(str(target), framework="pt", device="cpu") as handle:
                header = handle.metadata() or {}
                keys = list(handle.keys())
                shape = (
                    list(handle.get_slice("grid").get_shape())
                    if keys == ["grid"] else None)
            meta = json.loads(header["radio_metadata"])
        except FileNotFoundError:
            return source, "absent"
        except (OSError, ValueError, KeyError, RuntimeError):
            return source, "unreadable-header"
        for reason, ok in (
                ("schema", header.get("schema") == NATIVE_CACHE_SCHEMA),
                ("revision", meta["checkpoint_revision"] == args.revision),
                ("source-sha256", not source_sha256
                 or meta["source_sha256"] == source_sha256),
                ("max-edge", meta["max_edge"] == args.max_edge),
                ("hidden-size", meta["hidden_size"] == FUSED_ADAPTOR_WIDTH),
                ("shape", shape == [1, *meta["grid"], FUSED_ADAPTOR_WIDTH])):
            if not ok:
                return source, reason
        return source, None

    def fast(item):
        source, _source_sha256 = item
        return source, geometry_reason(
            cache_path(args.cache_dir, source), FUSED_ADAPTOR_WIDTH,
            args.max_edge)

    check = deep if args.deep else fast
    invalid: list[tuple[Path, str]] = []
    reasons: dict[str, int] = {}
    ordered = sorted(
        rows.items(), key=lambda item: str(cache_path(args.cache_dir, item[0])))
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for source, reason in pool.map(check, ordered, chunksize=64):
            if reason is not None:
                invalid.append((source, reason))
                reasons[reason] = reasons.get(reason, 0) + 1
    elapsed = time.monotonic() - started

    print(json.dumps({
        "mode": mode,
        "audit": ("safetensors header + tensor index per entry"
                  if args.deep else
                  "stat() size geometry only -- no file contents were read"),
        "detects": DEEP_DETECTS if args.deep else FAST_DETECTS,
        "cannot_detect": DEEP_CANNOT_DETECT if args.deep else FAST_CANNOT_DETECT,
        "unique_sources": len(rows),
        "sound": len(rows) - len(invalid),
        "rejected": len(invalid),
        "rejected_by_reason": dict(sorted(reasons.items())),
        "examples": [{"source": str(path), "reason": reason}
                     for path, reason in invalid[:20]],
        "elapsed_seconds": round(elapsed, 3),
        "microseconds_per_entry": round(1e6 * elapsed / max(len(rows), 1), 1),
    }, indent=2))
    return 1 if invalid else 0


if __name__ == "__main__":
    raise SystemExit(main())
