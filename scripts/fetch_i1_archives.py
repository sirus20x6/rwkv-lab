#!/usr/bin/env python3
"""Fetch large non-Hugging-Face i1 archives in resumable byte ranges."""
from __future__ import annotations

import argparse
import fcntl
import http.client
import json
import os
import shutil
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "datasets/i1_full_sources"


@dataclass(frozen=True)
class Archive:
    url: str
    relative_path: str
    size: int


ARCHIVES: dict[str, Archive] = {
    "inaturalist": Archive(
        "https://ml-inat-competition-datasets.s3.amazonaws.com/2024/train.tar.gz",
        "inaturalist/train.tar.gz",
        472_688_822_915,
    ),
    "places365": Archive(
        "https://data.csail.mit.edu/places/places365/train_large_places365challenge.tar",
        "places365/train_large_places365challenge.tar",
        511_257_384_960,
    ),
    "yfcc_metadata": Archive(
        "https://multimedia-commons.s3-us-west-2.amazonaws.com/tools/etc/"
        "yfcc100m_dataset.sql",
        "yfcc/yfcc100m_dataset.sql",
        65_644_027_904,
    ),
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def human_bytes(value: int) -> str:
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    amount = float(value)
    for unit in units:
        if amount < 1024 or unit == units[-1]:
            return f"{amount:.2f} {unit}"
        amount /= 1024
    raise AssertionError("unreachable")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--sources", nargs="+", choices=tuple(ARCHIVES), default=list(ARCHIVES)
    )
    parser.add_argument("--chunk-gib", type=float, default=256.0)
    parser.add_argument("--reserve-tib", type=float, default=8.0)
    parser.add_argument(
        "--max-chunks", type=int, default=0,
        help="zero continues until complete or space-limited",
    )
    parser.add_argument(
        "--wait-for-space", action="store_true",
        help="wait and resume automatically when the free-space reserve is restored",
    )
    parser.add_argument("--space-poll-seconds", type=int, default=60)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def read_state(path: Path) -> dict:
    if not path.exists():
        return {"version": 1, "created_at": now(), "sources": {}, "chunks": []}
    state = json.loads(path.read_text())
    if (state.get("version") != 1
            or not isinstance(state.get("sources"), dict)
            or not isinstance(state.get("chunks"), list)):
        raise RuntimeError(f"unsupported or corrupt acquisition state: {path}")
    return state


def write_state(path: Path, state: dict) -> None:
    state["updated_at"] = now()
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path.parent, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def publish_complete_archive(partial: Path, final: Path, expected_size: int) -> None:
    """Durably publish only an exactly sized archive payload."""
    actual = partial.stat().st_size
    if actual != expected_size:
        raise RuntimeError(
            f"archive size verification failed for {partial}: "
            f"expected {expected_size}, got {actual}"
        )
    # A previous process can be killed after its last append but before its
    # per-range fsync. Re-establish payload durability before making the final
    # pathname visible.
    with partial.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(partial, final)
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(final.parent, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def archive_state_entry(state: dict, name: str, archive: Archive) -> dict:
    entry = state["sources"].setdefault(
        name,
        {
            "url": archive.url,
            "path": archive.relative_path,
            "expected_bytes": archive.size,
        },
    )
    if (entry.get("url") != archive.url
            or entry.get("path") != archive.relative_path
            or entry.get("expected_bytes") != archive.size):
        raise RuntimeError(f"state mismatch for {name}")
    validator = entry.get("validator")
    if validator is not None and (
            not isinstance(validator, dict)
            or validator.get("kind") != "strong_etag"
            or not isinstance(validator.get("value"), str)
            or not validator["value"]
            or validator["value"].lstrip().lower().startswith("w/")
            or not (validator["value"].startswith('"')
                    and validator["value"].endswith('"'))):
        raise RuntimeError(f"invalid archive validator state for {name}")
    return entry


def _strong_response_etag(response) -> str:
    """Return the response's strong object identity or fail closed."""
    raw = response.headers.get("ETag")
    etag = str(raw).strip() if raw is not None else ""
    if (not etag or etag.lower().startswith("w/")
            or not (etag.startswith('"') and etag.endswith('"'))):
        raise RuntimeError(
            "range response has no strong ETag; refusing an unpinned archive")
    return etag


def _validate_content_range(value: str, *, start: int, end: int,
                            expected_total: int) -> None:
    expected = f"bytes {start}-{end}/{expected_total}"
    if value.strip() != expected:
        raise RuntimeError(
            f"unexpected Content-Range: {value!r}; expected {expected!r}")


def fetch_range(url: str, output: Path, start: int, end: int, *,
                expected_total: int, expected_etag: str | None = None,
                pin_etag: Callable[[str], None] | None = None
                ) -> tuple[int, str]:
    """Append one exact range while binding every byte to one object ETag."""
    original_start = start
    existing_size = output.stat().st_size if output.exists() else 0
    if existing_size != original_start:
        raise RuntimeError(
            f"partial size does not match requested range start: "
            f"{existing_size} != {original_start}")
    failures = 0
    transient = (ConnectionError, ConnectionResetError, TimeoutError,
                 urllib.error.URLError, http.client.IncompleteRead)
    while start <= end:
        headers = {
            "Range": f"bytes={start}-{end}",
            "User-Agent": "i1-resumable-acquisition/1.0",
        }
        if expected_etag is not None:
            headers["If-Range"] = expected_etag
        request = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                if response.status != 206:
                    raise RuntimeError(
                        f"server did not honor byte range: HTTP {response.status}")
                _validate_content_range(
                    response.headers.get("Content-Range", ""), start=start,
                    end=end, expected_total=expected_total)
                response_etag = _strong_response_etag(response)
                if expected_etag is None:
                    # Commit object identity before the first append. If the
                    # process dies mid-response, a later resume can prove that
                    # the existing partial belongs to the same generation.
                    if pin_etag is not None:
                        pin_etag(response_etag)
                    expected_etag = response_etag
                elif response_etag != expected_etag:
                    raise RuntimeError(
                        "archive ETag changed during resume: "
                        f"expected {expected_etag!r}, got {response_etag!r}")
                with output.open("ab") as handle:
                    remaining = end - start + 1
                    while True:
                        block = response.read(8 * 1024 * 1024)
                        if not block:
                            break
                        if len(block) > remaining:
                            raise RuntimeError(
                                f"range response overran byte {end}: received "
                                f"{len(block)} bytes with {remaining} remaining")
                        handle.write(block)
                        remaining -= len(block)
                    handle.flush()
                    os.fsync(handle.fileno())
        except transient as error:
            # HTTPError subclasses URLError: split permanent HTTP rejections
            # (4xx other than timeout/rate-limit) out of the retry loop, or a
            # gone/forbidden object would be re-requested forever.
            if (isinstance(error, urllib.error.HTTPError)
                    and 400 <= error.code < 500
                    and error.code not in (408, 429)):
                raise RuntimeError(
                    f"permanent HTTP failure for {url}: "
                    f"HTTP {error.code} {error.reason}") from error
            # Closing the append handle flushes Python's buffer, but it does not
            # make the newly observed file length power-loss durable. Commit the
            # partial before using its size as the next resume boundary.
            if output.exists():
                with output.open("rb") as handle:
                    os.fsync(handle.fileno())
            resumed = output.stat().st_size if output.exists() else original_start
            if not original_start <= resumed <= end + 1:
                raise RuntimeError(f"invalid partial size after network failure: {resumed}") from error
            if resumed > start:
                # The failed response still appended bytes; that is progress,
                # so the consecutive-failure budget starts over.
                failures = 0
            failures += 1
            if failures >= 6:
                # Mirror fetch_i1_sources.py's six-attempt cap instead of
                # retrying a persistently failing range forever.
                raise RuntimeError(
                    f"giving up after {failures} consecutive transient "
                    f"failures for {url}; last error: "
                    f"{type(error).__name__}: {error}") from error
            print(
                f"RETRY {failures}: {type(error).__name__}; resuming byte {resumed}",
                flush=True,
            )
            start = resumed
            time.sleep(min(30, 2 * failures))
            continue
        actual = output.stat().st_size
        if actual <= start:
            raise RuntimeError(f"range made no progress from byte {start}")
        if actual > end + 1:
            raise RuntimeError(
                f"range response overran byte {end}: partial size is {actual}")
        start = actual
    expected = end - original_start + 1
    assert expected_etag is not None
    return expected, expected_etag


def main() -> int:
    args = parse_args()
    if (
        args.chunk_gib <= 0
        or args.reserve_tib < 0
        or args.max_chunks < 0
        or args.space_poll_seconds <= 0
    ):
        raise SystemExit("chunk and reserve values must be non-negative")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    state_path = output / "archive_acquisition_state.json"
    lock_path = output / ".archive_acquisition.lock"
    chunk_limit = int(args.chunk_gib * 1024**3)
    reserve = int(args.reserve_tib * 1024**4)

    with lock_path.open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("another archive acquisition process holds the lock")

        state = read_state(state_path)
        chunks_done = 0
        for name in args.sources:
            archive = ARCHIVES[name]
            final = output / archive.relative_path
            partial = final.with_suffix(final.suffix + ".part")
            final.parent.mkdir(parents=True, exist_ok=True)
            entry = archive_state_entry(state, name, archive)
            if final.exists():
                if final.stat().st_size != archive.size:
                    raise RuntimeError(f"incorrect completed size for {final}")
                entry["status"] = "complete"
                write_state(state_path, state)
                continue

            current = partial.stat().st_size if partial.exists() else 0
            if current > archive.size:
                raise RuntimeError(f"partial file is too large: {partial}")
            saved_validator = entry.get("validator")
            expected_etag = (str(saved_validator["value"])
                             if saved_validator is not None else None)
            if current and expected_etag is None:
                raise RuntimeError(
                    f"cannot safely resume {partial}: existing bytes have no pinned "
                    "strong ETag; preserve or remove the partial explicitly")
            for previous in state["chunks"]:
                if previous.get("source") == name and previous.get("status") == "running":
                    previous["status"] = "interrupted"
                    previous["downloaded_bytes"] = max(
                        0, min(current, int(previous["end"]) + 1)
                        - int(previous["start"]))
                    previous["finished_at"] = now()
            write_state(state_path, state)
            while current < archive.size:
                if args.max_chunks and chunks_done >= args.max_chunks:
                    print(f"Chunk limit reached after {chunks_done} chunk(s).", flush=True)
                    return 0
                amount = min(chunk_limit, archive.size - current)
                available = shutil.disk_usage(output).free
                if available - amount < reserve:
                    entry["status"] = "waiting_for_space" if args.wait_for_space else "space_limited"
                    entry["downloaded_bytes"] = current
                    write_state(state_path, state)
                    print(
                        f"SPACE_LIMIT: {name} needs {human_bytes(amount)} next; "
                        f"{human_bytes(available)} free, preserving {human_bytes(reserve)}.",
                        flush=True,
                    )
                    if args.wait_for_space:
                        time.sleep(args.space_poll_seconds)
                        continue
                    return 75
                end = current + amount - 1
                chunk = {
                    "id": len(state["chunks"]) + 1,
                    "source": name,
                    "start": current,
                    "end": end,
                    "expected_bytes": amount,
                    "status": "dry_run" if args.dry_run else "running",
                    "started_at": now(),
                }
                state["chunks"].append(chunk)
                entry["status"] = "downloading"
                write_state(state_path, state)
                print(
                    f"CHUNK {chunk['id']} {name}: bytes {current}-{end} "
                    f"({human_bytes(amount)})",
                    flush=True,
                )
                if args.dry_run:
                    chunk["finished_at"] = now()
                    write_state(state_path, state)
                    return 0
                def pin_etag(value: str) -> None:
                    nonlocal expected_etag
                    if expected_etag is not None and expected_etag != value:
                        raise RuntimeError(
                            f"archive ETag changed: {expected_etag!r} -> {value!r}")
                    expected_etag = value
                    entry["validator"] = {
                        "kind": "strong_etag", "value": value,
                    }
                    write_state(state_path, state)

                received, expected_etag = fetch_range(
                    archive.url, partial, current, end,
                    expected_total=archive.size, expected_etag=expected_etag,
                    pin_etag=pin_etag)
                current += received
                chunk["status"] = "complete"
                chunk["downloaded_bytes"] = received
                chunk["finished_at"] = now()
                entry["downloaded_bytes"] = current
                chunks_done += 1
                write_state(state_path, state)

            publish_complete_archive(partial, final, archive.size)
            entry["status"] = "complete"
            entry["completed_at"] = now()
            write_state(state_path, state)
            print(f"SOURCE {name} complete.", flush=True)

        print("All selected direct archives are complete.", flush=True)
        return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted; the next run will resume.", file=sys.stderr)
        raise SystemExit(130)
