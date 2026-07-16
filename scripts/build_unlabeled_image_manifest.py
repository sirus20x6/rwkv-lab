#!/usr/bin/env python3
"""Build a resumable, deduplicated manifest from a local image tree.

The pipeline deliberately separates cheap filesystem inventory, exact-byte
deduplication, image decoding/perceptual hashing, and manifest export.  Its
SQLite database is the checkpoint, so an interrupted multi-terabyte scan can
resume without repeating completed work.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

import imagehash
from PIL import Image, ImageOps

try:
    import blake3
except ImportError:  # pragma: no cover - available in the project environment
    blake3 = None


IMAGE_SUFFIXES = {
    ".avif", ".bmp", ".gif", ".heic", ".heif", ".jpeg", ".jpg",
    ".png", ".tif", ".tiff", ".webp",
}
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IMAGE_ROOT = Path("/thearray/git/datasets/porn")
DEFAULT_DATABASE = Path(
    "/thearray/downloads/cache/moe-mla/local_porn_image_dedup.sqlite")
DEFAULT_MANIFEST = PROJECT_ROOT / "curated_vision/local_porn_unlabeled_dedup.jsonl"

# Do not admit material whose path makes adult status uncertain. This is a
# conservative manifest filter; it neither opens nor classifies excluded files.
DEFAULT_UNCERTAIN_AGE_RE = re.compile(
    r"(?:^|[^a-z])(?:barely[ ._-]*legal|child|ddlg|kid|loli(?:ta)?|minor|"
    r"pre[ ._-]*teen|school[ ._-]*girl|teen)(?:[^a-z]|$)", re.IGNORECASE)


SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;
CREATE TABLE IF NOT EXISTS files (
    path TEXT PRIMARY KEY,
    size INTEGER NOT NULL,
    mtime_ns INTEGER NOT NULL,
    excluded_reason TEXT,
    content_hash TEXT,
    width INTEGER,
    height INTEGER,
    phash TEXT,
    colorhash TEXT,
    decode_error TEXT,
    duplicate_of TEXT,
    duplicate_kind TEXT,
    scanned_ns INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS files_size_idx ON files(size);
CREATE INDEX IF NOT EXISTS files_hash_idx ON files(content_hash);
CREATE INDEX IF NOT EXISTS files_dup_idx ON files(duplicate_of);
CREATE TABLE IF NOT EXISTS phash_bands (
    band INTEGER NOT NULL,
    value INTEGER NOT NULL,
    path TEXT NOT NULL,
    PRIMARY KEY (band, value, path)
);
CREATE INDEX IF NOT EXISTS phash_bands_lookup ON phash_bands(band, value);
CREATE TABLE IF NOT EXISTS metadata (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


@dataclass(frozen=True)
class Decoded:
    path: str
    width: int | None
    height: int | None
    phash: str | None
    colorhash: str | None
    error: str | None


def connection(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path, timeout=120)
    db.executescript(SCHEMA)
    columns = {row[1] for row in db.execute("PRAGMA table_info(files)")}
    for name in ("colorhash", "duplicate_kind"):
        if name not in columns:
            db.execute(f"ALTER TABLE files ADD COLUMN {name} TEXT")
    db.commit()
    return db


def candidate_paths(root: Path) -> Iterator[Path]:
    for directory, names, files in os.walk(root):
        names.sort()
        files.sort()
        base = Path(directory)
        for name in files:
            path = base / name
            if path.suffix.lower() in IMAGE_SUFFIXES:
                yield path


def inventory(root: Path, db: sqlite3.Connection, commit_every: int,
              limit: int | None = None) -> int:
    root = root.resolve()
    seen = 0
    invalid_utf8 = 0
    upsert = """
        INSERT INTO files(path,size,mtime_ns,excluded_reason,scanned_ns)
        VALUES(?,?,?,?,?)
        ON CONFLICT(path) DO UPDATE SET
          size=excluded.size, mtime_ns=excluded.mtime_ns,
          excluded_reason=excluded.excluded_reason,
          content_hash=CASE WHEN files.size=excluded.size AND
            files.mtime_ns=excluded.mtime_ns THEN files.content_hash END,
          width=CASE WHEN files.size=excluded.size AND
            files.mtime_ns=excluded.mtime_ns THEN files.width END,
          height=CASE WHEN files.size=excluded.size AND
            files.mtime_ns=excluded.mtime_ns THEN files.height END,
          phash=CASE WHEN files.size=excluded.size AND
            files.mtime_ns=excluded.mtime_ns THEN files.phash END,
          colorhash=CASE WHEN files.size=excluded.size AND
            files.mtime_ns=excluded.mtime_ns THEN files.colorhash END,
          decode_error=CASE WHEN files.size=excluded.size AND
            files.mtime_ns=excluded.mtime_ns THEN files.decode_error END,
          duplicate_of=CASE WHEN files.size=excluded.size AND
            files.mtime_ns=excluded.mtime_ns THEN files.duplicate_of END,
          duplicate_kind=CASE WHEN files.size=excluded.size AND
            files.mtime_ns=excluded.mtime_ns THEN files.duplicate_kind END,
          scanned_ns=excluded.scanned_ns
    """
    scan_id = __import__("time").time_ns()
    for path in candidate_paths(root):
        seen += 1
        try:
            stat = path.stat()
        except OSError:
            continue
        absolute = str(path.resolve())
        try:
            absolute.encode("utf-8", errors="strict")
        except UnicodeEncodeError:
            # SQLite's Python binding rejects surrogate-escaped filesystem
            # names, and such a path could not be represented safely in the
            # JSONL consumed by the training pipeline either.
            invalid_utf8 += 1
            if seen % commit_every == 0:
                db.commit()
                print({"phase": "inventory", "seen": seen,
                       "invalid_utf8_paths": invalid_utf8}, flush=True)
            if limit is not None and seen >= limit:
                break
            continue
        relative = str(path.relative_to(root))
        excluded = "uncertain_age_path" if DEFAULT_UNCERTAIN_AGE_RE.search(
            relative) else None
        db.execute(upsert, (absolute, stat.st_size, stat.st_mtime_ns,
                            excluded, scan_id))
        if seen % commit_every == 0:
            db.commit()
            print({"phase": "inventory", "seen": seen,
                   "invalid_utf8_paths": invalid_utf8}, flush=True)
        if limit is not None and seen >= limit:
            break
    if limit is None:
        # Only prune disappeared paths after a complete walk. An interrupted or
        # deliberately limited inventory must leave the prior checkpoint intact.
        db.execute("DELETE FROM files WHERE scanned_ns != ?", (scan_id,))
    db.commit()
    db.execute("INSERT OR REPLACE INTO metadata VALUES('root', ?)", (str(root),))
    db.execute("""INSERT OR REPLACE INTO metadata
                  VALUES('inventory_invalid_utf8_paths', ?)""",
               (str(invalid_utf8),))
    db.commit()
    return seen


def digest(path: str) -> tuple[str, str | None]:
    try:
        if blake3 is not None:
            value = blake3.blake3()
        else:
            value = hashlib.blake2b(digest_size=32)
        with open(path, "rb", buffering=8 * 1024 * 1024) as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                value.update(chunk)
        return path, value.hexdigest()
    except OSError:
        return path, None


def exact_hashes(db: sqlite3.Connection, workers: int,
                 commit_every: int) -> int:
    # Files with a unique byte size cannot be exact duplicates. Avoid reading
    # those files in this phase; every image will be decoded in the next phase.
    rows = db.execute("""
        SELECT path FROM files WHERE excluded_reason IS NULL
          AND content_hash IS NULL AND size IN (
            SELECT size FROM files WHERE excluded_reason IS NULL
            GROUP BY size HAVING count(*) > 1)
    """).fetchall()
    completed = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        for path, value in pool.map(digest, (row[0] for row in rows)):
            if value is None:
                db.execute("UPDATE files SET decode_error=? WHERE path=?",
                           ("read_error", path))
            else:
                db.execute("UPDATE files SET content_hash=? WHERE path=?",
                           (value, path))
            completed += 1
            if completed % commit_every == 0:
                db.commit()
                print({"phase": "exact_hash", "done": completed,
                       "total": len(rows)}, flush=True)
    db.commit()

    groups = db.execute("""
        SELECT content_hash FROM files WHERE content_hash IS NOT NULL
        GROUP BY content_hash HAVING count(*) > 1
    """).fetchall()
    for (value,) in groups:
        members = db.execute("""
            SELECT path FROM files WHERE content_hash=?
            ORDER BY length(path), path
        """, (value,)).fetchall()
        keeper = members[0][0]
        db.executemany("UPDATE files SET duplicate_of=?, duplicate_kind='exact' WHERE path=?",
                       ((keeper, row[0]) for row in members[1:]))
    db.commit()
    return completed


def decode_image(path: str) -> Decoded:
    try:
        with Image.open(path) as source:
            source.seek(0)
            image = ImageOps.exif_transpose(source).convert("RGB")
            width, height = image.size
            value = str(imagehash.phash(image, hash_size=16))
            color = str(imagehash.colorhash(image, binbits=3))
        return Decoded(path, width, height, value, color, None)
    except Exception as error:  # Pillow raises several format-specific errors.
        return Decoded(path, None, None, None, None,
                       f"{type(error).__name__}: {str(error)[:240]}")


def perceptual_hashes(db: sqlite3.Connection, workers: int,
                      commit_every: int) -> int:
    rows = db.execute("""
        SELECT path FROM files WHERE excluded_reason IS NULL
          AND duplicate_of IS NULL AND phash IS NULL AND decode_error IS NULL
    """).fetchall()
    def record(items: Iterable[Decoded]) -> int:
        completed = 0
        for item in items:
            db.execute("""
                UPDATE files SET width=?,height=?,phash=?,colorhash=?,decode_error=?
                WHERE path=?
            """, (item.width, item.height, item.phash, item.colorhash,
                    item.error, item.path))
            completed += 1
            if completed % commit_every == 0:
                db.commit()
                print({"phase": "perceptual_hash", "done": completed,
                       "total": len(rows)}, flush=True)
        db.commit()
        return completed

    paths = (row[0] for row in rows)
    if workers == 1:
        return record(map(decode_image, paths))
    # Processes isolate Pillow decoders and scale across CPU cores. map preserves
    # bounded streaming via chunksize instead of materializing decoded images.
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
        return record(pool.map(decode_image, paths, chunksize=8))


def bands(value: str, count: int = 8) -> list[tuple[int, int]]:
    number = int(value, 16)
    bits = len(value) * 4
    width = bits // count
    mask = (1 << width) - 1
    return [(band, (number >> (band * width)) & mask)
            for band in range(count)]


def cluster_near_duplicates(db: sqlite3.Connection, distance: int,
                            min_side: int, commit_every: int) -> tuple[int, int]:
    db.execute("DELETE FROM phash_bands")
    # Preserve exact-duplicate assignments, but make perceptual clustering
    # deterministic and recomputable when the threshold changes.
    db.execute("""UPDATE files SET duplicate_of=NULL, duplicate_kind=NULL
                  WHERE duplicate_kind='perceptual'""")
    db.commit()
    rows = db.execute("""
        SELECT path,phash,colorhash,width,height,size FROM files
        WHERE excluded_reason IS NULL AND decode_error IS NULL
          AND duplicate_of IS NULL AND phash IS NOT NULL
        ORDER BY (width * height) DESC, size DESC, path
    """).fetchall()
    kept = duplicates = 0
    for index, (path, value, color, width, height, _size) in enumerate(rows, 1):
        if min(width, height) < min_side:
            db.execute("UPDATE files SET excluded_reason=? WHERE path=?",
                       ("below_min_side", path))
            continue
        pieces = bands(value)
        predicate = " OR ".join("(b.band=? AND b.value=?)" for _ in pieces)
        parameters = [part for pair in pieces for part in pair]
        candidate_rows = db.execute(f"""
            SELECT DISTINCT b.path,f.phash,f.colorhash
            FROM phash_bands b JOIN files f ON f.path=b.path
            WHERE {predicate}
        """, parameters).fetchall()
        number = int(value, 16)
        match = None
        best_distance = distance + 1
        for candidate, candidate_hash, candidate_color in candidate_rows:
            candidate_distance = (number ^ int(candidate_hash, 16)).bit_count()
            color_distance = (int(color, 16) ^ int(candidate_color, 16)).bit_count()
            if color_distance <= 4 and candidate_distance < best_distance:
                match, best_distance = candidate, candidate_distance
        if match is not None and best_distance <= distance:
            db.execute("""UPDATE files SET duplicate_of=?,
                          duplicate_kind='perceptual' WHERE path=?""",
                       (match, path))
            duplicates += 1
        else:
            db.executemany("INSERT INTO phash_bands VALUES(?,?,?)",
                           ((band, part, path) for band, part in pieces))
            kept += 1
        if index % commit_every == 0:
            db.commit()
            print({"phase": "cluster", "done": index, "total": len(rows),
                   "kept": kept, "near_duplicates": duplicates}, flush=True)
    db.commit()
    db.execute("INSERT OR REPLACE INTO metadata VALUES('phash_distance', ?)",
               (str(distance),))
    db.commit()
    return kept, duplicates


def export_manifest(db: sqlite3.Connection, output: Path) -> int:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    rows = db.execute("""
        SELECT path,width,height,size,content_hash,phash,colorhash FROM files
        WHERE excluded_reason IS NULL AND decode_error IS NULL
          AND duplicate_of IS NULL AND phash IS NOT NULL
        ORDER BY path
    """)
    count = 0
    with temporary.open("w") as handle:
        for path, width, height, size, content_hash, phash, colorhash in rows:
            record = {
                "image": path,
                "source_dataset": "local_unlabeled_porn_dedup",
                "task": "vision_distillation",
                "width": width,
                "height": height,
                "bytes": size,
                "phash256": phash,
                "colorhash": colorhash,
            }
            if content_hash:
                record["content_hash"] = content_hash
            handle.write(json.dumps(record, separators=(",", ":")) + "\n")
            count += 1
    os.replace(temporary, output)
    return count


def statistics(db: sqlite3.Connection) -> dict[str, int]:
    queries = {
        "inventoried": "SELECT count(*) FROM files",
        "excluded": "SELECT count(*) FROM files WHERE excluded_reason IS NOT NULL",
        "decode_errors": "SELECT count(*) FROM files WHERE decode_error IS NOT NULL",
        "duplicates": "SELECT count(*) FROM files WHERE duplicate_of IS NOT NULL",
        "keepers": """SELECT count(*) FROM files WHERE excluded_reason IS NULL
          AND decode_error IS NULL AND duplicate_of IS NULL AND phash IS NOT NULL""",
    }
    result = {name: db.execute(query).fetchone()[0]
              for name, query in queries.items()}
    invalid = db.execute("""SELECT value FROM metadata
                             WHERE key='inventory_invalid_utf8_paths'""").fetchone()
    result["invalid_utf8_paths"] = int(invalid[0]) if invalid else 0
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_IMAGE_ROOT,
                        help=f"image tree (default: {DEFAULT_IMAGE_ROOT})")
    parser.add_argument("--db", type=Path, default=DEFAULT_DATABASE,
                        help=f"resumable SQLite state (default: {DEFAULT_DATABASE})")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST,
                        help=f"output JSONL (default: {DEFAULT_MANIFEST})")
    parser.add_argument("--phase", choices=("inventory", "exact", "hash",
                                             "cluster", "export", "all"),
                        default="all")
    parser.add_argument("--workers", type=int, default=max(1, min(8,
                        (os.cpu_count() or 2) // 2)))
    parser.add_argument("--commit-every", type=int, default=500)
    parser.add_argument("--phash-distance", type=int, default=4)
    parser.add_argument("--min-side", type=int, default=256)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.workers < 1 or args.commit_every < 1:
        parser.error("--workers and --commit-every must be positive")
    if not 0 <= args.phash_distance <= 7:
        parser.error("--phash-distance must be between 0 and 7")
    if args.min_side < 1:
        parser.error("--min-side must be positive")
    return args


def main() -> None:
    args = parse_args()
    print({"phase": "configuration", "root": str(args.root.resolve()),
           "db": str(args.db.resolve()),
           "manifest": str(args.manifest.resolve()),
           "requested": args.phase, "workers": args.workers}, flush=True)
    db = connection(args.db)
    if args.phase in {"inventory", "all"}:
        print({"phase": "inventory", "seen": inventory(
            args.root, db, args.commit_every, args.limit)}, flush=True)
    if args.phase in {"exact", "all"}:
        print({"phase": "exact_hash", "done": exact_hashes(
            db, args.workers, args.commit_every)}, flush=True)
    if args.phase in {"hash", "all"}:
        print({"phase": "perceptual_hash", "done": perceptual_hashes(
            db, args.workers, args.commit_every)}, flush=True)
    if args.phase in {"cluster", "all"}:
        kept, duplicate = cluster_near_duplicates(
            db, args.phash_distance, args.min_side, args.commit_every)
        print({"phase": "cluster", "kept": kept,
               "near_duplicates": duplicate}, flush=True)
    if args.phase in {"export", "all"}:
        print({"phase": "export", "manifest": str(args.manifest),
               "rows": export_manifest(db, args.manifest)}, flush=True)
    print({"phase": "summary", **statistics(db)}, flush=True)


if __name__ == "__main__":
    main()
