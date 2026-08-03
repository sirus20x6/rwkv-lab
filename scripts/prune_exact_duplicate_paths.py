#!/usr/bin/env python3
"""Remove exact-duplicate path aliases after hardlink materialization.

Only aliases that are still the same inode as their canonical keeper are
removed.  A durable JSONL receipt is written before each committed unlink
batch, so every deleted pathname can be recreated as a hardlink if an external
reference is discovered later.  The canonical keeper is never removed.
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
DEFAULT_RECEIPT = Path(
    "/thearray/downloads/cache/moe-mla/local_porn_exact_prune_receipt.jsonl")


def exact_alias_rows(db: sqlite3.Connection):
    return db.execute("""
        SELECT d.path,d.duplicate_of,d.size,d.content_hash,k.content_hash
        FROM files d JOIN files k ON k.path=d.duplicate_of
        WHERE d.duplicate_kind='exact' AND d.path != d.duplicate_of
        ORDER BY d.path
    """).fetchall()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--receipt", type=Path, default=DEFAULT_RECEIPT)
    parser.add_argument("--commit-every", type=int, default=500)
    parser.add_argument("--execute", action="store_true",
                        help="unlink verified aliases; otherwise audit only")
    args = parser.parse_args()
    if args.commit_every < 1:
        parser.error("--commit-every must be positive")

    db = sqlite3.connect(args.db, timeout=120)
    db.execute("PRAGMA busy_timeout=120000")
    rows = exact_alias_rows(db)
    verified = removed = reconciled = changed = missing_keeper = 0
    started = time.time()
    receipt = None
    pending: list[tuple[str, str, os.stat_result]] = []

    if args.execute:
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        receipt = args.receipt.open("a", encoding="utf-8")

    def commit_batch() -> None:
        nonlocal removed
        if not pending:
            return
        assert receipt is not None
        timestamp = time.time_ns()
        for duplicate, keeper, stat in pending:
            receipt.write(json.dumps({
                "removed": duplicate, "keeper": keeper,
                "device": stat.st_dev, "inode": stat.st_ino,
                "size": stat.st_size, "recorded_ns": timestamp,
            }, ensure_ascii=False) + "\n")
        receipt.flush()
        os.fsync(receipt.fileno())
        for duplicate, _keeper, _stat in pending:
            Path(duplicate).unlink()
            db.execute("DELETE FROM phash_bands WHERE path=?", (duplicate,))
            db.execute("DELETE FROM files WHERE path=?", (duplicate,))
            removed += 1
        db.commit()
        pending.clear()

    try:
        for index, (duplicate_name, keeper_name, expected_size,
                    duplicate_hash, keeper_hash) in enumerate(rows, 1):
            duplicate, keeper = Path(duplicate_name), Path(keeper_name)
            try:
                keeper_stat = keeper.stat()
            except OSError:
                missing_keeper += 1
                continue
            try:
                duplicate_stat = duplicate.stat()
            except OSError:
                # A prior interrupted execute may have unlinked the alias after
                # its receipt was synced but before SQLite committed.
                if args.execute and duplicate_hash == keeper_hash:
                    db.execute("DELETE FROM phash_bands WHERE path=?",
                               (duplicate_name,))
                    db.execute("DELETE FROM files WHERE path=?",
                               (duplicate_name,))
                    reconciled += 1
                else:
                    changed += 1
                continue
            same_inode = ((duplicate_stat.st_dev, duplicate_stat.st_ino)
                          == (keeper_stat.st_dev, keeper_stat.st_ino))
            if (not same_inode or duplicate_hash is None
                    or duplicate_hash != keeper_hash
                    or duplicate_stat.st_size != expected_size
                    or keeper_stat.st_size != expected_size):
                changed += 1
                continue
            verified += 1
            if args.execute:
                pending.append((duplicate_name, keeper_name, duplicate_stat))
                if len(pending) >= args.commit_every:
                    commit_batch()
                    print(json.dumps({
                        "kind": "exact_path_prune", "done": index,
                        "total": len(rows), "removed": removed,
                        "reconciled": reconciled,
                    }), flush=True)
        if args.execute:
            commit_batch()
            db.commit()
    finally:
        if receipt is not None:
            receipt.close()
        db.close()

    print(json.dumps({
        "kind": "exact_path_prune_complete", "total": len(rows),
        "verified": verified, "removed": removed,
        "reconciled": reconciled, "changed_skipped": changed,
        "missing_keeper_skipped": missing_keeper,
        "execute": args.execute, "receipt": str(args.receipt),
        "seconds": round(time.time() - started, 3),
    }), flush=True)


if __name__ == "__main__":
    main()
