#!/usr/bin/env python3
"""Reclaim exact-duplicate bytes while preserving every original path.

Rows already proven byte-identical by ``build_unlabeled_image_manifest.py`` are
replaced atomically with hard links to their canonical keeper.  Keeping every
pathname avoids invalidating manifests, captions, or external references.  A
row is skipped if either file changed since it was hashed.
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import time
from pathlib import Path


DEFAULT_DB = Path(
    "/thearray/downloads/cache/moe-mla/local_porn_image_dedup.sqlite")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--commit-every", type=int, default=500)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.commit_every < 1:
        parser.error("--commit-every must be positive")

    db = sqlite3.connect(args.db, timeout=120)
    db.execute("PRAGMA busy_timeout=120000")
    rows = db.execute("""
        SELECT d.path,d.duplicate_of,d.size,d.mtime_ns,
               k.size,k.mtime_ns,d.content_hash,k.content_hash
        FROM files d JOIN files k ON k.path=d.duplicate_of
        WHERE d.duplicate_kind='exact' AND d.path != d.duplicate_of
        ORDER BY d.path
    """).fetchall()
    linked = already = changed = missing = cross_device = 0
    reclaimed = 0
    started = time.time()
    for index, row in enumerate(rows, 1):
        (duplicate_name, keeper_name, size, duplicate_mtime,
         keeper_size, keeper_mtime, duplicate_hash, keeper_hash) = row
        duplicate, keeper = Path(duplicate_name), Path(keeper_name)
        try:
            duplicate_stat, keeper_stat = duplicate.stat(), keeper.stat()
        except OSError:
            missing += 1
            continue
        if (duplicate_hash is None or duplicate_hash != keeper_hash
                or size != keeper_size
                or duplicate_stat.st_size != size
                or keeper_stat.st_size != keeper_size
                or duplicate_stat.st_mtime_ns != duplicate_mtime
                or keeper_stat.st_mtime_ns != keeper_mtime):
            changed += 1
            continue
        if (duplicate_stat.st_dev, duplicate_stat.st_ino) == (
                keeper_stat.st_dev, keeper_stat.st_ino):
            already += 1
            continue
        if duplicate_stat.st_dev != keeper_stat.st_dev:
            cross_device += 1
            continue
        if args.dry_run:
            linked += 1
            if duplicate_stat.st_nlink == 1:
                reclaimed += duplicate_stat.st_blocks * 512
            continue

        temporary = duplicate.with_name(
            f".{duplicate.name}.dedupe-link-{os.getpid()}")
        try:
            temporary.unlink(missing_ok=True)
            os.link(keeper, temporary)
            os.replace(temporary, duplicate)
        finally:
            temporary.unlink(missing_ok=True)
        linked += 1
        if duplicate_stat.st_nlink == 1:
            reclaimed += duplicate_stat.st_blocks * 512
        # Preserve the exact-hash checkpoint across a future inventory pass:
        # the duplicate path now has the keeper inode's timestamp.
        db.execute("UPDATE files SET mtime_ns=? WHERE path=?",
                   (keeper_stat.st_mtime_ns, duplicate_name))
        if linked % args.commit_every == 0:
            db.commit()
            print(json.dumps({
                "kind": "exact_dedup", "done": index, "total": len(rows),
                "linked": linked, "already_linked": already,
                "reclaimed_gib": round(reclaimed / 2**30, 3),
            }), flush=True)
    if not args.dry_run:
        db.commit()
    result = {
        "kind": "exact_dedup_complete", "total": len(rows),
        "linked": linked, "already_linked": already,
        "changed_skipped": changed, "missing_skipped": missing,
        "cross_device_skipped": cross_device,
        "reclaimed_bytes": reclaimed,
        "reclaimed_gib": round(reclaimed / 2**30, 3),
        "dry_run": args.dry_run,
        "seconds": round(time.time() - started, 3),
    }
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
