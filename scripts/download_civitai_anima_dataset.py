#!/usr/bin/env python3
"""Inventory and download prompt-bearing images from Civitai's official Anima gallery.

The public gallery is mutable, so inventory follows one cursor chain and records
its starting cutoff. SQLite is the durable checkpoint for both API pagination
and image transfers. API responses are requested with ``withMeta=true`` and
``flatMeta=true``; without those flags Civitai intentionally omits prompts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sqlite3
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests
from PIL import Image


MODEL_ID = 2_458_426
API_ROOT = "https://civitai.com/api/v1"
DEFAULT_ROOT = Path("/thearray/git/datasets/anima-civitai-official")
USER_AGENT = "mageflow-anima-dataset/1.0"
ALLOWED_IMAGE_HOSTS = {"image.civitai.com", "imagecache.civitai.com"}
ALLOWED_LEVELS = {"None"}
MAX_IMAGE_BYTES = 100 * 1024 * 1024

MINOR_PATTERN = re.compile(
    r"\b(?:loli(?:con)?|shota(?:con)?|underage|pre[- ]?teen|child(?:like)?|"
    r"young\s+(?:girl|boy)|schoolgirl|schoolboy)\b",
    re.IGNORECASE,
)
SEXUAL_PATTERN = re.compile(
    r"\b(?:nude|naked|sex(?:ual)?|explicit|nsfw|nipples?|areolae?|genitals?|"
    r"pussy|vagina|penis|cock|cum|masturbat\w*|intercourse|oral|anal|"
    r"breasts?|cleavage|erection|orgasm)\b",
    re.IGNORECASE,
)
UNDERAGE_NUMBER_PATTERN = re.compile(
    r"\b(?:age[ :=-]*)?(?:[0-9]|1[0-7])(?:\s*[- ]?years?\s*old|\s*yo)\b",
    re.IGNORECASE,
)

SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=FULL;
CREATE TABLE IF NOT EXISTS settings (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS images (
    id INTEGER PRIMARY KEY,
    post_id INTEGER,
    url TEXT NOT NULL,
    width INTEGER NOT NULL,
    height INTEGER NOT NULL,
    media_type TEXT NOT NULL,
    nsfw_level TEXT NOT NULL,
    created_at TEXT,
    username TEXT,
    base_model TEXT,
    model_version_ids_json TEXT NOT NULL,
    prompt TEXT,
    negative_prompt TEXT,
    meta_json TEXT,
    raw_json TEXT NOT NULL,
    eligibility TEXT NOT NULL,
    eligibility_reason TEXT NOT NULL,
    download_status TEXT NOT NULL DEFAULT 'pending',
    attempts INTEGER NOT NULL DEFAULT 0,
    file_name TEXT,
    sha256 TEXT,
    size_bytes INTEGER,
    actual_width INTEGER,
    actual_height INTEGER,
    image_format TEXT,
    error TEXT,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS images_eligibility_idx
    ON images(eligibility, download_status, id);
CREATE INDEX IF NOT EXISTS images_sha256_idx ON images(sha256);
CREATE TABLE IF NOT EXISTS pages (
    cursor TEXT PRIMARY KEY,
    next_cursor TEXT,
    item_count INTEGER NOT NULL,
    response_sha256 TEXT NOT NULL,
    fetched_at TEXT NOT NULL
);
"""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def connect(root: Path) -> sqlite3.Connection:
    state = root / "state"
    state.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(state / "downloads.sqlite3", timeout=60)
    connection.row_factory = sqlite3.Row
    connection.executescript(SCHEMA)
    return connection


def setting(connection: sqlite3.Connection, key: str) -> str | None:
    row = connection.execute(
        "SELECT value FROM settings WHERE key=?", (key,)
    ).fetchone()
    return None if row is None else str(row["value"])


def set_setting(connection: sqlite3.Connection, key: str, value: Any) -> None:
    connection.execute(
        """
        INSERT INTO settings(key, value) VALUES (?, ?)
        ON CONFLICT(key) DO UPDATE SET value=excluded.value
        """,
        (key, str(value)),
    )


def request_json(
    session: requests.Session,
    url: str,
    *,
    params: dict[str, Any] | None = None,
    attempts: int = 8,
) -> tuple[dict[str, Any], bytes]:
    delay = 1.0
    for attempt in range(attempts):
        try:
            response = session.get(url, params=params, timeout=(20, 120))
            if response.status_code == 429:
                wait = float(response.headers.get("Retry-After", delay))
                time.sleep(min(max(wait, 1.0), 120.0))
                delay = min(delay * 2, 120.0)
                continue
            response.raise_for_status()
            body = response.content
            payload = response.json()
            if not isinstance(payload, dict):
                raise RuntimeError("Civitai API response is not an object")
            return payload, body
        except (requests.RequestException, ValueError, RuntimeError):
            if attempt + 1 == attempts:
                raise
            time.sleep(delay)
            delay = min(delay * 2, 120.0)
    raise AssertionError("unreachable")


def safety_reason(prompt: str, nsfw_level: str) -> str | None:
    if nsfw_level not in ALLOWED_LEVELS:
        return f"content_level_{nsfw_level or 'unknown'}"
    sexual = bool(SEXUAL_PATTERN.search(prompt))
    if sexual and (
        MINOR_PATTERN.search(prompt) or UNDERAGE_NUMBER_PATTERN.search(prompt)
    ):
        return "sexualized_minor_prompt"
    return None


def classify_image(
    item: dict[str, Any],
    official_version_ids: set[int] | None = None,
) -> tuple[str, str, str, str]:
    meta = item.get("meta")
    if not isinstance(meta, dict):
        return "excluded", "missing_metadata", "", ""
    prompt = str(meta.get("prompt") or "").strip()
    negative = str(meta.get("negativePrompt") or "").strip()
    if not prompt:
        return "excluded", "empty_prompt", "", negative
    if str(item.get("type") or "image") != "image":
        return "excluded", "non_image_media", prompt, negative
    if official_version_ids is not None:
        try:
            observed_versions = {
                int(value) for value in (item.get("modelVersionIds") or [])
            }
        except (TypeError, ValueError):
            observed_versions = set()
        if not observed_versions.intersection(official_version_ids):
            return "excluded", "no_official_anima_version", prompt, negative
    url = str(item.get("url") or "")
    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.hostname not in ALLOWED_IMAGE_HOSTS:
        return "excluded", "unsafe_image_url", prompt, negative
    if int(item.get("width") or 0) < 16 or int(item.get("height") or 0) < 16:
        return "excluded", "invalid_geometry", prompt, negative
    nsfw_level = str(item.get("nsfwLevel") or "Unknown")
    reason = safety_reason(prompt, nsfw_level)
    if reason:
        return "quarantined", reason, prompt, negative
    return "eligible", "prompt_bearing_official_gallery_image", prompt, negative


def upsert_items(
    connection: sqlite3.Connection,
    items: list[Any],
    *,
    official_version_ids: set[int],
) -> Counter[str]:
    counts: Counter[str] = Counter()
    now = utc_now()
    for value in items:
        if not isinstance(value, dict):
            counts["invalid_item"] += 1
            continue
        try:
            image_id = int(value["id"])
            url = str(value["url"])
            width = int(value["width"])
            height = int(value["height"])
        except (KeyError, TypeError, ValueError):
            counts["invalid_item"] += 1
            continue
        eligibility, reason, prompt, negative = classify_image(
            value, official_version_ids
        )
        meta = value.get("meta")
        version_ids = value.get("modelVersionIds")
        if not isinstance(version_ids, list):
            version_ids = []
        connection.execute(
            """
            INSERT INTO images(
                id, post_id, url, width, height, media_type, nsfw_level,
                created_at, username, base_model, model_version_ids_json,
                prompt, negative_prompt, meta_json, raw_json, eligibility,
                eligibility_reason, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                post_id=excluded.post_id,
                url=excluded.url,
                width=excluded.width,
                height=excluded.height,
                media_type=excluded.media_type,
                nsfw_level=excluded.nsfw_level,
                created_at=excluded.created_at,
                username=excluded.username,
                base_model=excluded.base_model,
                model_version_ids_json=excluded.model_version_ids_json,
                prompt=excluded.prompt,
                negative_prompt=excluded.negative_prompt,
                meta_json=excluded.meta_json,
                raw_json=excluded.raw_json,
                eligibility=excluded.eligibility,
                eligibility_reason=excluded.eligibility_reason,
                updated_at=excluded.updated_at
            """,
            (
                image_id,
                value.get("postId"),
                url,
                width,
                height,
                str(value.get("type") or "image"),
                str(value.get("nsfwLevel") or "Unknown"),
                value.get("createdAt"),
                value.get("username"),
                value.get("baseModel"),
                canonical_json(version_ids),
                prompt or None,
                negative or None,
                canonical_json(meta) if isinstance(meta, dict) else None,
                canonical_json(value),
                eligibility,
                reason,
                now,
            ),
        )
        counts[f"{eligibility}:{reason}"] += 1
    return counts


def authenticated_session(token_env: str) -> requests.Session:
    token = os.environ.get(token_env, "").strip()
    if not token:
        raise RuntimeError(f"required Civitai credential is unset: {token_env}")
    session = requests.Session()
    session.headers.update(
        {
            "Authorization": f"Bearer {token}",
            "Accept": "application/json",
            "User-Agent": USER_AGENT,
        }
    )
    return session


def inventory(args: argparse.Namespace) -> None:
    root = args.root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    connection = connect(root)
    session = authenticated_session(args.token_env)

    model, model_body = request_json(session, f"{API_ROOT}/models/{MODEL_ID}")
    if int(model.get("id") or 0) != MODEL_ID or str(model.get("name")) != "Anima":
        raise RuntimeError("Civitai model identity changed")
    atomic_text(root / "state" / "model.json", json.dumps(model, indent=2) + "\n")
    official_version_ids = {
        int(version["id"])
        for version in model.get("modelVersions", [])
        if isinstance(version, dict) and version.get("id") is not None
    }
    if not official_version_ids:
        raise RuntimeError("official Anima model has no published version IDs")
    set_setting(connection, "model_id", MODEL_ID)
    set_setting(
        connection,
        "official_version_ids",
        canonical_json(sorted(official_version_ids)),
    )
    set_setting(connection, "model_response_sha256", hashlib.sha256(model_body).hexdigest())

    if args.refresh:
        set_setting(connection, "next_cursor", "")
        set_setting(connection, "inventory_complete", "0")
        set_setting(connection, "inventory_started_at", utc_now())
        connection.commit()
    elif setting(connection, "inventory_started_at") is None:
        set_setting(connection, "next_cursor", "")
        set_setting(connection, "inventory_complete", "0")
        set_setting(connection, "inventory_started_at", utc_now())
        connection.commit()

    if setting(connection, "inventory_complete") == "1" and not args.refresh:
        print("inventory already complete; use --refresh to begin a new cursor snapshot")
        materialize(root, connection)
        return

    cursor = setting(connection, "next_cursor") or ""
    page_number = 0
    while True:
        params: dict[str, Any] = {
            "modelId": MODEL_ID,
            "limit": 200,
            "sort": "Newest",
            "period": "AllTime",
            "withMeta": "true",
            "flatMeta": "true",
        }
        if cursor:
            params["cursor"] = cursor
        payload, body = request_json(session, f"{API_ROOT}/images", params=params)
        items = payload.get("items")
        metadata = payload.get("metadata")
        if not isinstance(items, list) or not isinstance(metadata, dict):
            raise RuntimeError("malformed Civitai image page")
        counts = upsert_items(
            connection, items, official_version_ids=official_version_ids
        )
        next_cursor_value = metadata.get("nextCursor")
        next_cursor = "" if next_cursor_value is None else str(next_cursor_value)
        fetched = utc_now()
        connection.execute(
            """
            INSERT OR REPLACE INTO pages(
                cursor, next_cursor, item_count, response_sha256, fetched_at
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (
                cursor,
                next_cursor or None,
                len(items),
                hashlib.sha256(body).hexdigest(),
                fetched,
            ),
        )
        set_setting(connection, "next_cursor", next_cursor)
        set_setting(connection, "inventory_updated_at", fetched)
        connection.commit()
        page_number += 1
        print(
            canonical_json(
                {
                    "kind": "inventory_page",
                    "page": page_number,
                    "cursor": cursor or None,
                    "next_cursor": next_cursor or None,
                    "items": len(items),
                    "classified": dict(sorted(counts.items())),
                }
            ),
            flush=True,
        )
        if not next_cursor:
            set_setting(connection, "inventory_complete", "1")
            set_setting(connection, "inventory_completed_at", utc_now())
            connection.commit()
            break
        cursor = next_cursor
        time.sleep(args.api_delay)
    materialize(root, connection)


class RateLimiter:
    def __init__(self, requests_per_second: float):
        self.interval = 1.0 / max(requests_per_second, 0.01)
        self.lock = threading.Lock()
        self.next_time = 0.0

    def wait(self) -> None:
        with self.lock:
            now = time.monotonic()
            delay = max(0.0, self.next_time - now)
            self.next_time = max(now, self.next_time) + self.interval
        if delay:
            time.sleep(delay)


def image_extension(image_format: str) -> str:
    values = {
        "JPEG": ".jpg",
        "PNG": ".png",
        "WEBP": ".webp",
        "AVIF": ".avif",
    }
    try:
        return values[image_format.upper()]
    except KeyError as error:
        raise RuntimeError(f"unsupported downloaded image format: {image_format}") from error


def download_one(
    root: Path,
    row: sqlite3.Row,
    limiter: RateLimiter,
) -> dict[str, Any]:
    image_id = int(row["id"])
    url = str(row["url"])
    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.hostname not in ALLOWED_IMAGE_HOSTS:
        raise RuntimeError("image URL left the Civitai image host allowlist")
    shard = f"{image_id // 1000:06d}"
    directory = root / "images" / shard
    directory.mkdir(parents=True, exist_ok=True)
    temporary = directory / f".{image_id}.part"
    try:
        limiter.wait()
        headers = {"Accept": "image/*", "User-Agent": USER_AGENT}
        with requests.get(
            url, headers=headers, stream=True, timeout=(20, 180)
        ) as response:
            response.raise_for_status()
            content_length = int(response.headers.get("Content-Length") or 0)
            if content_length > MAX_IMAGE_BYTES:
                raise RuntimeError(f"image exceeds {MAX_IMAGE_BYTES} byte limit")
            digest = hashlib.sha256()
            size = 0
            with temporary.open("wb") as handle:
                for chunk in response.iter_content(1024 * 1024):
                    if not chunk:
                        continue
                    size += len(chunk)
                    if size > MAX_IMAGE_BYTES:
                        raise RuntimeError(
                            f"image exceeds {MAX_IMAGE_BYTES} byte limit"
                        )
                    digest.update(chunk)
                    handle.write(chunk)
                handle.flush()
                os.fsync(handle.fileno())
        with Image.open(temporary) as decoded:
            decoded.verify()
        with Image.open(temporary) as decoded:
            width, height = decoded.size
            image_format = str(decoded.format or "").upper()
        extension = image_extension(image_format)
        final = directory / f"{image_id}{extension}"
        os.replace(temporary, final)
        sidecar = final.with_suffix(".txt")
        atomic_text(sidecar, str(row["prompt"]).strip() + "\n")
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return {
        "id": image_id,
        "file_name": str(final.relative_to(root)),
        "sha256": digest.hexdigest(),
        "size_bytes": size,
        "actual_width": width,
        "actual_height": height,
        "image_format": image_format,
    }


def mark_download(
    database: Path,
    result: dict[str, Any] | None,
    *,
    image_id: int,
    error: str | None,
) -> None:
    connection = sqlite3.connect(database, timeout=60)
    try:
        if result is not None:
            connection.execute(
                """
                UPDATE images SET
                    download_status='downloaded', attempts=attempts+1,
                    file_name=?, sha256=?, size_bytes=?, actual_width=?,
                    actual_height=?, image_format=?, error=NULL, updated_at=?
                WHERE id=?
                """,
                (
                    result["file_name"],
                    result["sha256"],
                    result["size_bytes"],
                    result["actual_width"],
                    result["actual_height"],
                    result["image_format"],
                    utc_now(),
                    image_id,
                ),
            )
        else:
            connection.execute(
                """
                UPDATE images SET download_status='error', attempts=attempts+1,
                    error=?, updated_at=? WHERE id=?
                """,
                ((error or "unknown error")[:2000], utc_now(), image_id),
            )
        connection.commit()
    finally:
        connection.close()


def download(args: argparse.Namespace) -> None:
    root = args.root.expanduser().resolve()
    connection = connect(root)
    if setting(connection, "inventory_complete") != "1":
        raise RuntimeError("inventory must complete before downloads begin")
    rows = connection.execute(
        """
        SELECT * FROM images
        WHERE eligibility='eligible' AND download_status!='downloaded'
        ORDER BY id
        """
    ).fetchall()
    if args.limit is not None:
        rows = rows[: args.limit]
    connection.close()
    database = root / "state" / "downloads.sqlite3"
    limiter = RateLimiter(args.requests_per_second)
    completed = errors = 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(download_one, root, row, limiter): int(row["id"])
            for row in rows
        }
        for future in as_completed(futures):
            image_id = futures[future]
            try:
                result = future.result()
                mark_download(database, result, image_id=image_id, error=None)
                completed += 1
            except Exception as error:  # noqa: BLE001 - durable per-image errors
                mark_download(database, None, image_id=image_id, error=str(error))
                errors += 1
            if (completed + errors) % 25 == 0 or completed + errors == len(rows):
                print(
                    canonical_json(
                        {
                            "kind": "download_progress",
                            "processed": completed + errors,
                            "scheduled": len(rows),
                            "downloaded": completed,
                            "errors": errors,
                        }
                    ),
                    flush=True,
                )
    connection = connect(root)
    materialize(root, connection)
    connection.close()
    if errors:
        raise RuntimeError(f"{errors} image downloads failed; rerun to retry")


def materialize(root: Path, connection: sqlite3.Connection) -> dict[str, Any]:
    eligible = connection.execute(
        """
        SELECT * FROM images
        WHERE eligibility='eligible' AND download_status='downloaded'
        ORDER BY id
        """
    ).fetchall()
    metadata_lines = []
    for row in eligible:
        metadata_lines.append(
            canonical_json(
                {
                    "file_name": row["file_name"],
                    "text": row["prompt"],
                    "negative_prompt": row["negative_prompt"] or "",
                    "civitai_image_id": int(row["id"]),
                    "civitai_post_id": row["post_id"],
                    "creator": row["username"],
                    "source_url": f"https://civitai.com/images/{int(row['id'])}",
                    "model_id": MODEL_ID,
                    "model_version_ids": json.loads(row["model_version_ids_json"]),
                    "width": row["actual_width"],
                    "height": row["actual_height"],
                    "sha256": row["sha256"],
                    "generation_metadata": (
                        json.loads(row["meta_json"]) if row["meta_json"] else None
                    ),
                }
            )
        )
    atomic_text(
        root / "metadata.jsonl",
        "".join(line + "\n" for line in metadata_lines),
    )

    inventory_counts = {
        str(row["value"]): int(row["count"])
        for row in connection.execute(
            """
            SELECT eligibility || ':' || eligibility_reason AS value, count(*) AS count
            FROM images GROUP BY value ORDER BY value
            """
        )
    }
    download_counts = {
        str(row["download_status"]): int(row["count"])
        for row in connection.execute(
            """
            SELECT download_status, count(*) AS count
            FROM images GROUP BY download_status ORDER BY download_status
            """
        )
    }
    extension_counts: Counter[str] = Counter()
    total_bytes = 0
    missing_files = 0
    for row in eligible:
        path = root / str(row["file_name"])
        sidecar = path.with_suffix(".txt")
        if not path.is_file() or not sidecar.is_file():
            missing_files += 1
        extension_counts[path.suffix.lower()] += 1
        total_bytes += int(row["size_bytes"] or 0)
    duplicate_hashes = int(
        connection.execute(
            """
            SELECT count(*) FROM (
                SELECT sha256 FROM images
                WHERE download_status='downloaded' AND sha256 IS NOT NULL
                GROUP BY sha256 HAVING count(*) > 1
            )
            """
        ).fetchone()[0]
    )
    report = {
        "schema": "rwkv-lab.civitai-anima-dataset.v1",
        "model_id": MODEL_ID,
        "source": f"https://civitai.com/models/{MODEL_ID}",
        "api_endpoint": f"{API_ROOT}/images",
        "inventory_started_at": setting(connection, "inventory_started_at"),
        "inventory_completed_at": setting(connection, "inventory_completed_at"),
        "inventory_complete": setting(connection, "inventory_complete") == "1",
        "inventory_records": int(
            connection.execute("SELECT count(*) FROM images").fetchone()[0]
        ),
        "inventory_counts": inventory_counts,
        "download_counts": download_counts,
        "downloaded_prompt_pairs": len(eligible),
        "downloaded_bytes": total_bytes,
        "extension_counts": dict(sorted(extension_counts.items())),
        "duplicate_sha256_groups": duplicate_hashes,
        "missing_image_or_sidecar": missing_files,
        "content_levels_downloaded": sorted(ALLOWED_LEVELS),
        "license_observation": {
            "observed_at": "2026-07-28",
            "civitai_terms": "https://civitai.com/content/tos",
            "public_user_content_license": (
                "Civitai Terms section 9.2; preserve creator and source provenance"
            ),
            "anima_model_license": (
                "CircleStone Labs Non-Commercial License v1.2; training use "
                "requires non-commercial purpose or a separate commercial license"
            ),
        },
    }
    atomic_text(root / "report.json", json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def validate(args: argparse.Namespace) -> None:
    root = args.root.expanduser().resolve()
    connection = connect(root)
    report = materialize(root, connection)
    errors = []
    if not report["inventory_complete"]:
        errors.append("inventory is incomplete")
    if report["missing_image_or_sidecar"]:
        errors.append(
            f"{report['missing_image_or_sidecar']} downloaded records lack files"
        )
    metadata_count = sum(
        1 for line in (root / "metadata.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    if metadata_count != report["downloaded_prompt_pairs"]:
        errors.append("metadata.jsonl count does not match downloaded records")
    empty_prompts = int(
        connection.execute(
            """
            SELECT count(*) FROM images
            WHERE download_status='downloaded' AND trim(coalesce(prompt, ''))=''
            """
        ).fetchone()[0]
    )
    if empty_prompts:
        errors.append(f"{empty_prompts} downloaded records have empty prompts")
    connection.close()
    if errors:
        raise RuntimeError("; ".join(errors))
    print(json.dumps(report, indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("inventory", "download", "validate", "all")
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--token-env", default="CIVITAI_TOKEN")
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--api-delay", type=float, default=0.75)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--requests-per-second", type=float, default=4.0)
    parser.add_argument(
        "--limit",
        type=int,
        help="download at most this many pending records (for qualification)",
    )
    args = parser.parse_args()
    if (
        args.api_delay < 0
        or args.workers < 1
        or args.requests_per_second <= 0
        or (args.limit is not None and args.limit < 1)
    ):
        parser.error("invalid pacing or worker configuration")
    return args


def main() -> None:
    args = parse_args()
    if args.command in {"inventory", "all"}:
        inventory(args)
    if args.command in {"download", "all"}:
        download(args)
    if args.command == "validate":
        validate(args)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("interrupted; SQLite state is resumable", file=sys.stderr)
        raise SystemExit(130)
