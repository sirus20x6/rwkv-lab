#!/usr/bin/env python3
"""Classify Midjourney v6 rows as photo, animation, or ambiguous from captions."""

from __future__ import annotations

import argparse
import json
import os
import re
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import duckdb

DEFAULT_REPO = "Photoroom/midjourney-v6-recap"
DEFAULT_REVISION = "21c628db81401da88c5b33507230528cf3fe4a12"
DEFAULT_OUTPUT = Path("/workspace/git/datasets/midjourney-v6-recap-routing")
CLASSIFIER_VERSION = "caption-medium-v1"

ANIMATION_PATTERN = (
    r"\b(?:anime|manga|cartoon|comic(?:[ -]book)?|illustration|illustrated|"
    r"painting|painted|painterly|watercolou?r|gouache|oil[ -](?:painting|on[ -]"
    r"canvas)|digital[ -](?:painting|art|artwork)|artwork|drawing|sketch|"
    r"line[ -]art|vector[ -](?:art|illustration|graphic)|pixel[ -]art|"
    r"3d[ -](?:render|rendering)|cgi|claymation|stop[ -]motion|"
    r"graphic[ -](?:art|design|illustration)|concept[ -]art|poster[ -](?:art|"
    r"design|illustration)|collage|mixed[ -]media|cel[ -]shaded|"
    r"storybook[ -](?:art|illustration)|rendered[ -](?:image|scene))\b"
)
PHOTO_PATTERN = (
    r"\b(?:photograph|photography|photographic|photo[ -]?realistic|"
    r"photorealistic|photo[ -]portrait|studio[ -]portrait|cinematic[ -]still|"
    r"film[ -]still|product[ -](?:photo|photograph|shot)|editorial[ -](?:photo|"
    r"photograph)|documentary[ -](?:photo|photograph)|macro[ -](?:photo|"
    r"photograph|photography)|dslr|camera[ -](?:photo|photograph|shot|settings)|"
    r"shot[ -]on[ -](?:a[ -]|an[ -])?(?:camera|canon|nikon|sony|leica|"
    r"fujifilm)|real[ -]life[ -]photograph)\b"
)

_ANIMATION_RE = re.compile(ANIMATION_PATTERN, re.IGNORECASE)
_PHOTO_RE = re.compile(PHOTO_PATTERN, re.IGNORECASE)
_SHARD_RE = re.compile(r"train_\d{3}\.parquet")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def classify_caption(text: str | None) -> str:
    """Return one explicit-medium vote without guessing from depicted content."""
    if not isinstance(text, str) or not text.strip():
        return "missing"
    animation = bool(_ANIMATION_RE.search(text))
    photo = bool(_PHOTO_RE.search(text))
    if animation and photo:
        return "conflict"
    if animation:
        return "animation"
    if photo:
        return "photo"
    return "unknown"


def route_caption_votes(labels: list[str]) -> str:
    """Require two agreeing captioners and no contradictory/conflicting vote."""
    counts = Counter(labels)
    if counts["conflict"]:
        return "ambiguous"
    if counts["photo"] >= 2 and counts["animation"] == 0:
        return "photo"
    if counts["animation"] >= 2 and counts["photo"] == 0:
        return "animation"
    return "ambiguous"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--max-shards",
        type=int,
        default=None,
        help="process only the first N shards (testing only)",
    )
    return parser.parse_args()


def quote_sql(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def discover_shards(repo: str, revision: str) -> list[str]:
    api = f"https://huggingface.co/api/datasets/{repo}/revision/{revision}"
    with urllib.request.urlopen(api) as response:
        metadata = json.load(response)
    resolved = str(metadata.get("sha") or "")
    if resolved != revision:
        raise RuntimeError(f"dataset revision changed: expected {revision}, got {resolved}")
    shards = sorted(
        str(item["rfilename"])
        for item in metadata.get("siblings", [])
        if _SHARD_RE.fullmatch(str(item.get("rfilename") or ""))
    )
    if not shards:
        raise RuntimeError("no train_NNN.parquet shards discovered")
    return shards


def label_sql(column: str, status: str) -> str:
    animation = f"regexp_matches({column}, {quote_sql(ANIMATION_PATTERN)}, 'i')"
    photo = f"regexp_matches({column}, {quote_sql(PHOTO_PATTERN)}, 'i')"
    present = f"coalesce({status}, false) AND length(trim(coalesce({column}, ''))) > 0"
    return (
        f"CASE WHEN NOT ({present}) THEN 'missing' "
        f"WHEN {animation} AND {photo} THEN 'conflict' "
        f"WHEN {animation} THEN 'animation' "
        f"WHEN {photo} THEN 'photo' ELSE 'unknown' END"
    )


def scan_shard(
    connection: duckdb.DuckDBPyConnection,
    *,
    repo: str,
    revision: str,
    shard: str,
    output: Path,
) -> None:
    url = f"https://huggingface.co/datasets/{repo}/resolve/{revision}/{shard}"
    destination = output.resolve()
    temporary = destination.with_name(destination.name + ".tmp")
    source = (
        f"read_parquet({quote_sql(url)}, filename=true, file_row_number=true)"
    )
    llava_label = label_sql("llava", "llava_status")
    gemini_label = label_sql("gemini", "gemini_status")
    qwen_label = label_sql("qwen3", "qwen3_status")
    query = f"""
        COPY (
          WITH labelled AS (
            SELECT
              id AS family_id,
              {quote_sql(shard)} AS source_shard,
              file_row_number::BIGINT AS source_row,
              prompt,
              CASE
                WHEN coalesce(qwen3_status, false)
                  AND length(trim(coalesce(qwen3, ''))) > 0 THEN qwen3
                WHEN coalesce(gemini_status, false)
                  AND length(trim(coalesce(gemini, ''))) > 0 THEN gemini
                ELSE llava
              END AS caption,
              CASE
                WHEN coalesce(qwen3_status, false)
                  AND length(trim(coalesce(qwen3, ''))) > 0 THEN 'qwen3'
                WHEN coalesce(gemini_status, false)
                  AND length(trim(coalesce(gemini, ''))) > 0 THEN 'gemini'
                ELSE 'llava'
              END AS caption_source,
              {llava_label} AS llava_label,
              {gemini_label} AS gemini_label,
              {qwen_label} AS qwen3_label
            FROM {source}
          ),
          voted AS (
            SELECT *,
              (llava_label = 'photo')::INT
                + (gemini_label = 'photo')::INT
                + (qwen3_label = 'photo')::INT AS photo_votes,
              (llava_label = 'animation')::INT
                + (gemini_label = 'animation')::INT
                + (qwen3_label = 'animation')::INT AS animation_votes,
              (llava_label = 'conflict')::INT
                + (gemini_label = 'conflict')::INT
                + (qwen3_label = 'conflict')::INT AS conflict_votes
            FROM labelled
          )
          SELECT
            {quote_sql(CLASSIFIER_VERSION)} AS classifier_version,
            family_id,
            source_shard,
            source_row,
            prompt,
            caption,
            caption_source,
            llava_label,
            gemini_label,
            qwen3_label,
            photo_votes,
            animation_votes,
            conflict_votes,
            CASE
              WHEN conflict_votes > 0 THEN 'ambiguous'
              WHEN photo_votes >= 2 AND animation_votes = 0 THEN 'photo'
              WHEN animation_votes >= 2 AND photo_votes = 0 THEN 'animation'
              ELSE 'ambiguous'
            END AS domain
          FROM voted
        ) TO {quote_sql(str(temporary))} (
          FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 10000
        )
    """
    connection.execute(query)
    os.replace(temporary, destination)


def summarize(
    connection: duckdb.DuckDBPyConnection,
    shard_glob: Path,
) -> dict[str, Any]:
    source = f"read_parquet({quote_sql(str(shard_glob))})"
    domain_rows = connection.execute(
        f"SELECT domain, count(*) FROM {source} GROUP BY domain ORDER BY domain"
    ).fetchall()
    caption_rows = connection.execute(
        f"""
        SELECT caption_source, count(*)
        FROM {source}
        GROUP BY caption_source
        ORDER BY caption_source
        """
    ).fetchall()
    family_rows = connection.execute(
        f"""
        SELECT domain, count(DISTINCT family_id)
        FROM {source}
        GROUP BY domain
        ORDER BY domain
        """
    ).fetchall()
    total = connection.execute(f"SELECT count(*) FROM {source}").fetchone()[0]
    return {
        "schema": "rwkv-lab.midjourney-v6-caption-routing.v1",
        "classifier_version": CLASSIFIER_VERSION,
        "row_count": int(total),
        "domain_counts": {str(key): int(value) for key, value in domain_rows},
        "domain_family_counts": {
            str(key): int(value) for key, value in family_rows
        },
        "caption_source_counts": {
            str(key): int(value) for key, value in caption_rows
        },
    }


def write_audit_samples(
    connection: duckdb.DuckDBPyConnection,
    shard_glob: Path,
    destination: Path,
) -> None:
    source = f"read_parquet({quote_sql(str(shard_glob))})"
    rows = connection.execute(
        f"""
        WITH ranked AS (
          SELECT *,
            row_number() OVER (
              PARTITION BY domain
              ORDER BY md5(source_shard || ':' || source_row::VARCHAR)
            ) AS sample_rank
          FROM {source}
        )
        SELECT * EXCLUDE (sample_rank)
        FROM ranked
        WHERE sample_rank <= 100
        ORDER BY domain, source_shard, source_row
        """
    ).fetch_arrow_table()
    temporary = destination.with_name(destination.name + ".tmp")
    connection.register("audit_rows", rows)
    connection.execute(
        f"""
        COPY audit_rows TO {quote_sql(str(temporary))}
        (FORMAT JSON, ARRAY false)
        """
    )
    os.replace(temporary, destination)


def main() -> None:
    args = parse_args()
    output = args.output_dir.expanduser().resolve()
    shard_dir = output / "routing_shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    shards = discover_shards(args.repo, args.revision)
    if args.max_shards is not None:
        shards = shards[: args.max_shards]

    connection = duckdb.connect()
    connection.execute("INSTALL httpfs")
    connection.execute("LOAD httpfs")
    completed = []
    for index, shard in enumerate(shards, start=1):
        destination = shard_dir / shard
        if not destination.is_file():
            scan_shard(
                connection,
                repo=args.repo,
                revision=args.revision,
                shard=shard,
                output=destination,
            )
        completed.append(shard)
        atomic_json(
            output / "scan_state.json",
            {
                "schema": "rwkv-lab.midjourney-v6-caption-routing.v1",
                "repo": args.repo,
                "revision": args.revision,
                "classifier_version": CLASSIFIER_VERSION,
                "completed_shards": completed,
                "shard_count": len(shards),
                "updated_at": utc_now(),
            },
        )
        print(json.dumps({"completed": index, "total": len(shards), "shard": shard}))

    summary = summarize(connection, shard_dir / "*.parquet")
    summary.update(
        {
            "repo": args.repo,
            "revision": args.revision,
            "shard_count": len(shards),
            "completed_at": utc_now(),
        }
    )
    atomic_json(output / "summary.json", summary)
    write_audit_samples(
        connection, shard_dir / "*.parquet", output / "audit_samples.jsonl"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
