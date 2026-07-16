#!/usr/bin/env python3
"""Resumably download the Hugging Face image sources used by i1.

The downloader pins every source repository to the revision seen on its first
run, downloads one bounded chunk at a time, and stops before crossing a free
space reserve.  Re-running it is safe: files with the expected byte size are
skipped and Hugging Face resumes partial blob downloads from its local cache.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import shutil
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

# Xet's reconstruction client can repeatedly rewrite sparse partial files when
# signed range URLs expire on very large parquet shards. Ordinary Hub HTTP uses
# the same resumable ``.incomplete`` file without that reconstruction loop.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

from huggingface_hub import HfApi, constants as hf_constants, hf_hub_download


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "datasets/i1_full_sources"
STATE_NAME = "acquisition_state.json"


@dataclass(frozen=True)
class Source:
    repo_id: str
    directory: str


# Ordered from the smallest/easiest proven joins toward the largest sources.
HF_SOURCES: dict[str, Source] = {
    "pexels": Source("animetimm/pexels-tagger-v0-w640-ws-full", "pexels"),
    "midjourneyv6": Source("Photoroom/midjourney-v6-recap", "midjourneyv6"),
    "fluxreason": Source("LucasFang/FLUX-Reason-6M", "fluxreason"),
    "imagenet22k": Source("timm/imagenet-22k-wds", "imagenet22k"),
    # The i1 README/code currently says ``madebyollin/megalith-10mm`` (two
    # trailing m's), which does not exist. The source metadata is at
    # madebyollin/megalith-10m and points to this archived image copy. Both use
    # the original nine-digit Megalith keys expected by i1-captions.
    "megalith10m": Source("drawthingsai/megalith-10m", "megalith10m"),
    "gptedit": Source("UCSC-VLAA/gpt-edit-simpler", "gptedit"),
    "textatlas": Source("CSU-JPG/TextAtlas5M", "textatlas"),
    "rendered_text": Source("wendlerc/RenderedText", "rendered_text"),
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--sources",
        nargs="+",
        choices=tuple(HF_SOURCES),
        default=list(HF_SOURCES),
        help="ordered source queue",
    )
    parser.add_argument(
        "--chunk-gib", type=float, default=256.0,
        help="maximum expected bytes admitted to one chunk",
    )
    parser.add_argument(
        "--reserve-tib", type=float, default=8.0,
        help="stop before free space falls below this value",
    )
    parser.add_argument(
        "--max-chunks", type=int, default=1,
        help="chunks to fetch; zero keeps going until complete or space-limited",
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


def free_bytes(path: Path) -> int:
    return shutil.disk_usage(path).free


def human_bytes(value: int) -> str:
    units = ("B", "KiB", "MiB", "GiB", "TiB", "PiB")
    amount = float(value)
    for unit in units:
        if amount < 1024 or unit == units[-1]:
            return f"{amount:.2f} {unit}"
        amount /= 1024
    raise AssertionError("unreachable")


def pin_source(api: HfApi, name: str, source: Source, state: dict) -> str:
    entry = state["sources"].setdefault(
        name, {"repo_id": source.repo_id, "directory": source.directory}
    )
    if (entry.get("repo_id") != source.repo_id
            or entry.get("directory") != source.directory):
        raise RuntimeError(f"state source mismatch for {name}")
    if not entry.get("revision"):
        entry["revision"] = api.dataset_info(source.repo_id).sha
        entry["pinned_at"] = now()
    return str(entry["revision"])


def repository_files(api: HfApi, source: Source, revision: str) -> list[tuple[str, int]]:
    files: list[tuple[str, int]] = []
    seen: set[str] = set()
    for item in api.list_repo_tree(
        source.repo_id,
        repo_type="dataset",
        revision=revision,
        recursive=True,
        expand=True,
    ):
        size = getattr(item, "size", None)
        path = getattr(item, "path", None)
        if path is not None and size is not None:
            filename = str(path)
            relative = PurePosixPath(filename)
            if (relative.is_absolute() or not relative.parts
                    or ".." in relative.parts or filename in seen):
                raise RuntimeError(
                    f"unsafe or duplicate repository path at {revision}: {filename!r}")
            byte_count = int(size)
            if byte_count < 0:
                raise RuntimeError(
                    f"negative repository file size at {revision}: {filename!r}")
            seen.add(filename)
            files.append((filename, byte_count))
    if not files:
        raise RuntimeError(
            f"repository tree is empty at pinned revision {revision}")
    return sorted(files)


def file_complete(path: Path, expected_size: int) -> bool:
    try:
        return path.is_file() and path.stat().st_size == expected_size
    except OSError:
        return False


def select_chunk(
    queue: list[tuple[str, Source, str, int]], limit: int
) -> list[tuple[str, Source, str, int]]:
    selected: list[tuple[str, Source, str, int]] = []
    total = 0
    for item in queue:
        size = item[3]
        if selected and total + size > limit:
            break
        selected.append(item)
        total += size
        if total >= limit:
            break
    return selected


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
    state_path = output / STATE_NAME
    lock_path = output / ".acquisition.lock"
    chunk_limit = int(args.chunk_gib * 1024**3)
    reserve = int(args.reserve_tib * 1024**4)

    with lock_path.open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("another i1 acquisition process already holds the lock")

        state = read_state(state_path)
        # A prior process may have been interrupted between per-file state
        # updates. Preserve its history, but do not leave a stale "running"
        # record that looks live to operators.
        for previous in state.get("chunks", []):
            if previous.get("status") == "running":
                previous["status"] = "interrupted"
                previous.setdefault("finished_at", now())
        api = HfApi()
        chunks_done = 0

        # Discover only the source currently being consumed. Some repositories
        # have thousands of files, and eagerly walking every tree needlessly
        # delays the first useful transfer.
        for name in args.sources:
            source = HF_SOURCES[name]
            revision = pin_source(api, name, source, state)
            # Persist the immutable revision before the potentially long tree
            # walk. If indexing fails, the next launch must resume this exact
            # repository generation instead of silently pinning whatever is
            # current at that later time.
            write_state(state_path, state)
            print(f"INDEX {name}@{revision[:12]}", flush=True)
            files = repository_files(api, source, revision)
            entry = state["sources"][name]
            entry["file_count"] = len(files)
            entry["expected_bytes"] = sum(size for _, size in files)
            entry["downloaded_bytes"] = sum(
                size for filename, size in files
                if file_complete(output / source.directory / filename, size)
            )
            entry["status"] = "downloading"
            write_state(state_path, state)

            destination = output / source.directory
            while True:
                pending = [
                    (name, source, filename, size)
                    for filename, size in files
                    if not file_complete(destination / filename, size)
                ]
                if not pending:
                    entry["status"] = "complete"
                    entry["completed_at"] = now()
                    write_state(state_path, state)
                    print(f"SOURCE {name} complete.", flush=True)
                    break
                if args.max_chunks and chunks_done >= args.max_chunks:
                    print(f"Chunk limit reached after {chunks_done} chunk(s).", flush=True)
                    return 0

                selected = select_chunk(pending, chunk_limit)
                expected = sum(item[3] for item in selected)
                available = free_bytes(output)
                if available - expected < reserve:
                    entry["status"] = "waiting_for_space" if args.wait_for_space else "space_limited"
                    write_state(state_path, state)
                    print(
                        "SPACE_LIMIT: next chunk needs "
                        f"{human_bytes(expected)} with {human_bytes(available)} free; "
                        f"preserving {human_bytes(reserve)} reserve.",
                        flush=True,
                    )
                    if args.wait_for_space:
                        time.sleep(args.space_poll_seconds)
                        continue
                    return 75

                chunk = {
                    "id": len(state["chunks"]) + 1,
                    "started_at": now(),
                    "status": "dry_run" if args.dry_run else "running",
                    "expected_bytes": expected,
                    "files": len(selected),
                    "first": f"{selected[0][0]}:{selected[0][2]}",
                    "last": f"{selected[-1][0]}:{selected[-1][2]}",
                    "downloaded_bytes": 0,
                    "completed_files": 0,
                }
                state["chunks"].append(chunk)
                write_state(state_path, state)
                print(
                    f"CHUNK {chunk['id']}: {len(selected)} files, "
                    f"{human_bytes(expected)} expected",
                    flush=True,
                )

                if args.dry_run:
                    chunk["finished_at"] = now()
                    write_state(state_path, state)
                    return 0

                for _, _, filename, size in selected:
                    available = free_bytes(output)
                    while available - size < reserve:
                        status = "waiting_for_space" if args.wait_for_space else "space_limited"
                        chunk["status"] = status
                        entry["status"] = status
                        write_state(state_path, state)
                        print("SPACE_LIMIT reached between files.", flush=True)
                        if not args.wait_for_space:
                            chunk["finished_at"] = now()
                            write_state(state_path, state)
                            return 75
                        time.sleep(args.space_poll_seconds)
                        available = free_bytes(output)
                    chunk["status"] = "running"
                    entry["status"] = "downloading"
                    print(
                        f"FETCH {name}:{filename} ({human_bytes(size)})",
                        flush=True,
                    )
                    downloaded: Path | None = None
                    for attempt in range(1, 7):
                        try:
                            downloaded = Path(
                                hf_hub_download(
                                    repo_id=source.repo_id,
                                    filename=filename,
                                    repo_type="dataset",
                                    revision=revision,
                                    local_dir=destination,
                                )
                            )
                            chunk.pop("last_error", None)
                            entry.pop("last_error", None)
                            break
                        except Exception as error:
                            message = f"{type(error).__name__}: {error}"
                            chunk["last_error"] = message
                            entry["last_error"] = message
                            write_state(state_path, state)
                            if hf_constants.HF_HUB_ENABLE_HF_TRANSFER:
                                # The Rust transfer helper is fast, but can
                                # exhaust its internal permits after parallel
                                # response failures. Continue the same partial
                                # file through the ordinary HTTP implementation.
                                hf_constants.HF_HUB_ENABLE_HF_TRANSFER = False
                                os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
                                print("HF_TRANSFER failed; falling back to standard HTTP.",
                                      file=sys.stderr, flush=True)
                            if attempt == 6:
                                raise
                            delay = min(60, 2 ** attempt)
                            print(f"RETRY {name}:{filename} attempt={attempt + 1} "
                                  f"in {delay}s: {message}", file=sys.stderr, flush=True)
                            time.sleep(delay)
                    if downloaded is None:
                        raise RuntimeError(f"download produced no path for {name}:{filename}")
                    if not file_complete(downloaded, size):
                        raise RuntimeError(
                            f"size verification failed for {downloaded}: "
                            f"expected {size}, got {downloaded.stat().st_size}"
                        )
                    chunk["downloaded_bytes"] += size
                    chunk["completed_files"] += 1
                    entry["downloaded_bytes"] = int(entry.get("downloaded_bytes", 0)) + size
                    write_state(state_path, state)

                chunk["status"] = "complete"
                chunk["finished_at"] = now()
                chunks_done += 1
                write_state(state_path, state)
                print(f"CHUNK {chunk['id']} complete.", flush=True)

        print("All selected Hugging Face sources are complete.", flush=True)

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted; the next run will resume.", file=sys.stderr)
        raise SystemExit(130)
