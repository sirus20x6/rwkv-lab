#!/usr/bin/env python3
"""Extract every source caption for the materialized I1 review dataset."""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
from collections import defaultdict
from pathlib import Path

import pyarrow.dataset as ds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("/thearray/git/datasets/i1/metadata.jsonl"),
    )
    parser.add_argument(
        "--captions-root",
        type=Path,
        default=Path("/thearray/git/moe-mla/i1-captions"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "/thearray/git/datasets/i1/work/quality_viewer_captions.sqlite3"
        ),
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def wanted_keys(metadata: Path) -> dict[str, set[str]]:
    wanted: dict[str, set[str]] = defaultdict(set)
    with metadata.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            subset = row.get("i1_subset")
            key = row.get("i1_key")
            if not isinstance(subset, str) or not isinstance(key, str):
                raise RuntimeError(
                    f"{metadata}:{line_no}: missing i1_subset or i1_key"
                )
            wanted[subset].add(key)
    return dict(wanted)


def initialize(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        PRAGMA journal_mode=OFF;
        PRAGMA synchronous=OFF;
        PRAGMA temp_store=MEMORY;
        CREATE TABLE caption_variants (
            subset TEXT NOT NULL,
            i1_key TEXT NOT NULL,
            name TEXT NOT NULL,
            text TEXT NOT NULL,
            PRIMARY KEY (subset, i1_key, name)
        ) WITHOUT ROWID;
        CREATE TABLE subset_status (
            subset TEXT PRIMARY KEY,
            expected_items INTEGER NOT NULL,
            found_items INTEGER NOT NULL,
            caption_values INTEGER NOT NULL
        ) WITHOUT ROWID;
        CREATE TABLE cache_metadata (
            name TEXT PRIMARY KEY,
            value TEXT NOT NULL
        ) WITHOUT ROWID;
        """
    )


def extract_subset(
    connection: sqlite3.Connection,
    captions_root: Path,
    subset: str,
    keys: set[str],
) -> tuple[int, int]:
    source = captions_root / subset
    parquet_files = sorted(source.glob("*.parquet"))
    if not parquet_files:
        raise RuntimeError(f"{source}: no Parquet caption files")
    dataset = ds.dataset(source, format="parquet")
    columns = [name for name in dataset.schema.names if name != "key"]
    if not columns:
        raise RuntimeError(f"{source}: no caption columns")
    table = dataset.to_table(
        columns=["key", *columns],
        filter=ds.field("key").isin(sorted(keys)),
        use_threads=True,
    )
    found: set[str] = set()
    values = 0
    statement = (
        "INSERT INTO caption_variants(subset,i1_key,name,text)"
        " VALUES(?,?,?,?)"
    )
    for batch in table.to_batches(max_chunksize=2_048):
        pending: list[tuple[str, str, str, str]] = []
        for row in batch.to_pylist():
            key = str(row["key"])
            found.add(key)
            for name in columns:
                text = row.get(name)
                if text is None:
                    continue
                text = str(text).strip()
                if not text:
                    continue
                pending.append((subset, key, name, text))
        connection.executemany(statement, pending)
        values += len(pending)
    missing = keys - found
    if missing:
        examples = ", ".join(sorted(missing)[:5])
        raise RuntimeError(
            f"{subset}: found {len(found):,}/{len(keys):,} keys; "
            f"missing examples: {examples}"
        )
    connection.execute(
        "INSERT INTO subset_status VALUES(?,?,?,?)",
        (subset, len(keys), len(found), values),
    )
    connection.commit()
    print(
        json.dumps(
            {
                "subset": subset,
                "items": len(found),
                "caption_columns": columns,
                "caption_values": values,
            }
        ),
        flush=True,
    )
    return len(found), values


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    if output.exists() and not args.force:
        raise SystemExit(f"{output} already exists; pass --force to rebuild")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    temporary.unlink(missing_ok=True)
    wanted = wanted_keys(args.metadata)
    expected = sum(map(len, wanted.values()))
    connection = sqlite3.connect(temporary)
    try:
        initialize(connection)
        found = values = 0
        for subset, keys in sorted(wanted.items()):
            subset_found, subset_values = extract_subset(
                connection, args.captions_root, subset, keys
            )
            found += subset_found
            values += subset_values
        connection.executemany(
            "INSERT INTO cache_metadata VALUES(?,?)",
            [
                ("schema_version", "1"),
                ("metadata_path", str(args.metadata.resolve())),
                ("expected_items", str(expected)),
                ("found_items", str(found)),
                ("caption_values", str(values)),
            ],
        )
        connection.commit()
        integrity = connection.execute("PRAGMA integrity_check").fetchone()
        if integrity != ("ok",):
            raise RuntimeError(f"SQLite integrity check failed: {integrity}")
        if found != expected:
            raise RuntimeError(f"found {found:,}/{expected:,} requested items")
    except Exception:
        connection.close()
        temporary.unlink(missing_ok=True)
        raise
    connection.close()
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, output)
    print(
        json.dumps(
            {
                "output": str(output),
                "items": found,
                "caption_values": values,
                "bytes": output.stat().st_size,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
