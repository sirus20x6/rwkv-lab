#!/usr/bin/env python3
"""Build a reaction-ranked 50/50 SFW/NSFW snapshot from Anima's Civitai gallery.

The public Civitai v1 image endpoint currently accepts ``sort=Most Reactions``
while returning model-gallery results in cursor/newest order.  This tool uses
the site's ranked gallery endpoint instead, merges each official Anima version,
and freezes exactly ``--per-class`` records from each content class using
Civitai's authoritative all-time ``reactionCount`` metric.

The inventory and downloads are resumable.  Prompts and generation metadata are
read from the official API; no captions are generated or modified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sqlite3
import struct
import sys
import threading
import time
import zlib
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests
from PIL import Image, ImageFile, UnidentifiedImageError

MODEL_ID = 2_458_426
MODEL_NAME = "Anima"
API_ROOT = "https://civitai.red/api/v1"
TRPC_ROOT = "https://civitai.red/api/trpc"
SITE_ROOT = "https://civitai.red"
DEFAULT_ROOT = Path("/thearray/git/datasets/anima-civitai-balanced-10000")
USER_AGENT = "mageflow-anima-balanced-dataset/1.0"
ALLOWED_IMAGE_HOSTS = {"image.civitai.com", "imagecache.civitai.com"}
SFW_LEVELS = {"None"}
ADULT_LEVELS = {"Mature", "X", "XXX"}
REACTION_KEYS = ("likeCount", "heartCount", "laughCount", "cryCount")
NSFW_LEVEL_BY_FLAG = {
    1: "None",
    2: "Soft",
    4: "Mature",
    8: "X",
    16: "XXX",
}
CLASS_BROWSING_LEVEL = {"sfw": 1, "nsfw": 4 | 8 | 16}
MAX_IMAGE_BYTES = 100 * 1024 * 1024
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
PIL_FALLBACK_LOCK = threading.Lock()

# For an adult-class record, any positive-prompt minor marker is disqualifying.
# The SFW feed uses the narrower conjunction of minor and sexual markers.
MINOR_PATTERN = re.compile(
    r"\b(?:loli(?:con)?|shota(?:con)?|underage|minor|pre[- ]?teen|teenager|"
    r"child(?:ren|like)?|kid(?:s)?|toddler|infant|young\s+(?:girl|boy)|"
    r"schoolgirl|schoolboy)\b",
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
CREATE TABLE IF NOT EXISTS pages (
    content_class TEXT NOT NULL,
    cursor TEXT NOT NULL,
    next_cursor TEXT,
    item_count INTEGER NOT NULL,
    response_sha256 TEXT NOT NULL,
    fetched_at TEXT NOT NULL,
    PRIMARY KEY(content_class, cursor)
);
CREATE TABLE IF NOT EXISTS images (
    id INTEGER PRIMARY KEY,
    content_class TEXT NOT NULL,
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
    like_count INTEGER NOT NULL,
    heart_count INTEGER NOT NULL,
    laugh_count INTEGER NOT NULL,
    cry_count INTEGER NOT NULL,
    reaction_count INTEGER NOT NULL,
    eligibility TEXT NOT NULL,
    eligibility_reason TEXT NOT NULL,
    selected_rank INTEGER,
    download_status TEXT NOT NULL DEFAULT 'pending',
    attempts INTEGER NOT NULL DEFAULT 0,
    file_name TEXT,
    sha256 TEXT,
    size_bytes INTEGER,
    actual_width INTEGER,
    actual_height INTEGER,
    image_format TEXT,
    validation_warning TEXT,
    error TEXT,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS images_selection_idx
    ON images(content_class, eligibility, selected_rank);
CREATE INDEX IF NOT EXISTS images_download_idx
    ON images(selected_rank, download_status);
CREATE INDEX IF NOT EXISTS images_sha_idx ON images(sha256);
CREATE UNIQUE INDEX IF NOT EXISTS images_class_rank_idx
    ON images(content_class, selected_rank) WHERE selected_rank IS NOT NULL;
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
    (root / "state").mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(root / "state" / "downloads.sqlite3", timeout=60)
    connection.row_factory = sqlite3.Row
    connection.executescript(SCHEMA)
    columns = {
        str(row["name"])
        for row in connection.execute("PRAGMA table_info(images)").fetchall()
    }
    if "validation_warning" not in columns:
        connection.execute("ALTER TABLE images ADD COLUMN validation_warning TEXT")
        connection.commit()
    return connection


def setting(connection: sqlite3.Connection, key: str) -> str | None:
    row = connection.execute("SELECT value FROM settings WHERE key=?", (key,)).fetchone()
    return None if row is None else str(row["value"])


def set_setting(connection: sqlite3.Connection, key: str, value: Any) -> None:
    connection.execute(
        """
        INSERT INTO settings(key, value) VALUES (?, ?)
        ON CONFLICT(key) DO UPDATE SET value=excluded.value
        """,
        (key, str(value)),
    )


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


def request_json(
    session: requests.Session,
    url: str,
    *,
    params: dict[str, Any] | None = None,
    attempts: int = 8,
) -> tuple[Any, bytes]:
    delay = 1.0
    for attempt in range(attempts):
        try:
            response = session.get(url, params=params, timeout=(20, 120))
            if response.status_code == 429:
                wait = float(response.headers.get("Retry-After", delay))
                time.sleep(min(max(wait, 1.0), 120.0))
                delay = min(delay * 2.0, 120.0)
                continue
            response.raise_for_status()
            payload = response.json()
            return payload, response.content
        except (requests.RequestException, TypeError, ValueError):
            if attempt + 1 == attempts:
                raise
            time.sleep(delay)
            delay = min(delay * 2.0, 120.0)
    raise AssertionError("unreachable")


def devalue_unflatten(payload: dict[str, Any]) -> Any:
    """Decode the flattened devalue payload returned by image.getInfinite."""
    result = payload.get("result")
    if not isinstance(result, dict) or not isinstance(result.get("data"), str):
        raise TypeError("malformed Civitai tRPC response")
    flat = json.loads(result["data"])
    if not isinstance(flat, list) or not flat:
        raise TypeError("malformed Civitai devalue payload")
    memo: dict[int, Any] = {}

    def hydrate(index: int) -> Any:
        if index < 0:
            return None
        if index in memo:
            return memo[index]
        node = flat[index]
        if isinstance(node, list):
            hydrated_list: list[Any] = []
            memo[index] = hydrated_list
            hydrated_list.extend(
                hydrate(value)
                if isinstance(value, int) and not isinstance(value, bool)
                else value
                for value in node
            )
            return hydrated_list
        if isinstance(node, dict):
            hydrated_dict: dict[str, Any] = {}
            memo[index] = hydrated_dict
            for key, value in node.items():
                hydrated_dict[key] = (
                    hydrate(value)
                    if isinstance(value, int) and not isinstance(value, bool)
                    else value
                )
            return hydrated_dict
        memo[index] = node
        return node

    return hydrate(0)


def ranked_stream(
    *,
    token_env: str,
    model_version_id: int,
    content_class: str,
    candidate_limit: int,
    api_delay: float,
) -> list[dict[str, Any]]:
    session = authenticated_session(token_env)
    cursor: str | None = None
    found: dict[int, dict[str, Any]] = {}
    while len(found) < candidate_limit:
        query: dict[str, Any] = {
            "period": "AllTime",
            "sort": "Most Reactions",
            "types": ["image"],
            "withMeta": True,
            "modelVersionId": model_version_id,
            "browsingLevel": CLASS_BROWSING_LEVEL[content_class],
            "limit": 100,
        }
        if cursor:
            query["cursor"] = cursor
        payload, _ = request_json(
            session,
            f"{TRPC_ROOT}/image.getInfinite",
            params={"input": canonical_json({"json": query})},
        )
        decoded = devalue_unflatten(payload)
        if not isinstance(decoded, dict) or not isinstance(decoded.get("items"), list):
            raise TypeError("malformed ranked image page")
        for item in decoded["items"]:
            if not isinstance(item, dict):
                continue
            image_id = positive_int(item.get("id"))
            if not image_id:
                continue
            item["_official_version_ids"] = [model_version_id]
            found[image_id] = item
        next_cursor = decoded.get("nextCursor")
        if not next_cursor or not decoded["items"]:
            break
        cursor = str(next_cursor)
        if api_delay:
            time.sleep(api_delay)
    return list(found.values())


def generation_batch(
    *,
    token_env: str,
    image_ids: list[int],
) -> dict[int, dict[str, Any]]:
    if not image_ids:
        return {}
    session = authenticated_session(token_env)
    procedure = ",".join("image.getGenerationData" for _ in image_ids)
    inputs = {
        str(index): {"json": {"id": image_id}}
        for index, image_id in enumerate(image_ids)
    }
    payload, _ = request_json(
        session,
        f"{TRPC_ROOT}/{procedure}",
        params={"batch": "1", "input": canonical_json(inputs)},
    )
    if not isinstance(payload, list) or len(payload) != len(image_ids):
        raise TypeError("malformed generation-data batch")
    results: dict[int, dict[str, Any]] = {}
    for image_id, value in zip(image_ids, payload, strict=True):
        if not isinstance(value, dict):
            continue
        result = value.get("result")
        data = result.get("data") if isinstance(result, dict) else None
        decoded = data.get("json") if isinstance(data, dict) else None
        if isinstance(decoded, dict):
            results[image_id] = decoded
    return results


def original_image_url(value: Any) -> str:
    url_or_uuid = str(value or "").strip()
    if url_or_uuid.startswith("https://"):
        return url_or_uuid
    if not re.fullmatch(r"[0-9a-fA-F-]{32,40}", url_or_uuid):
        return ""
    return (
        "https://image.civitai.com/xG1nkqKTMzGDvpLrqFT7WA/"
        f"{url_or_uuid}/original=true/{url_or_uuid}"
    )


def nsfw_level_name(value: Any) -> str:
    if isinstance(value, str) and value in {*SFW_LEVELS, *ADULT_LEVELS, "Soft"}:
        return value
    return NSFW_LEVEL_BY_FLAG.get(positive_int(value), "Unknown")


def positive_int(value: Any) -> int:
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


def reaction_counts(item: dict[str, Any]) -> dict[str, int]:
    stats = item.get("stats")
    if not isinstance(stats, dict):
        stats = {}
    return {key: positive_int(stats.get(key)) for key in REACTION_KEYS}


def classify_image(
    item: dict[str, Any],
    content_class: str,
    official_version_ids: set[int],
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
    try:
        observed_versions = {
            int(value) for value in (item.get("modelVersionIds") or [])
        }
    except (TypeError, ValueError):
        observed_versions = set()
    if not observed_versions.intersection(official_version_ids):
        return "excluded", "no_official_anima_version", prompt, negative
    parsed = urlparse(str(item.get("url") or ""))
    if parsed.scheme != "https" or parsed.hostname not in ALLOWED_IMAGE_HOSTS:
        return "excluded", "unsafe_image_url", prompt, negative
    if positive_int(item.get("width")) < 16 or positive_int(item.get("height")) < 16:
        return "excluded", "invalid_geometry", prompt, negative

    nsfw_level = str(item.get("nsfwLevel") or "Unknown")
    has_minor_marker = bool(
        MINOR_PATTERN.search(prompt) or UNDERAGE_NUMBER_PATTERN.search(prompt)
    )
    if content_class == "sfw":
        if nsfw_level not in SFW_LEVELS:
            return "excluded", f"wrong_sfw_level_{nsfw_level}", prompt, negative
        if has_minor_marker and SEXUAL_PATTERN.search(prompt):
            return "quarantined", "sexualized_minor_prompt", prompt, negative
    elif content_class == "nsfw":
        if nsfw_level not in ADULT_LEVELS:
            return "excluded", f"wrong_adult_level_{nsfw_level}", prompt, negative
        if has_minor_marker:
            return "quarantined", "minor_marker_in_adult_prompt", prompt, negative
    else:
        raise ValueError(f"unknown content class: {content_class}")
    return "eligible", f"prompt_bearing_official_anima_{content_class}", prompt, negative


def upsert_items(
    connection: sqlite3.Connection,
    items: list[Any],
    *,
    content_class: str,
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
            value, content_class, official_version_ids
        )
        counts_by_kind = reaction_counts(value)
        reaction_count = sum(counts_by_kind.values())
        meta = value.get("meta")
        version_ids = value.get("modelVersionIds")
        if not isinstance(version_ids, list):
            version_ids = []
        existing = connection.execute(
            "SELECT content_class FROM images WHERE id=?", (image_id,)
        ).fetchone()
        if existing is not None and str(existing["content_class"]) != content_class:
            counts["cross_feed_duplicate"] += 1
            continue
        connection.execute(
            """
            INSERT INTO images(
                id, content_class, post_id, url, width, height, media_type,
                nsfw_level, created_at, username, base_model,
                model_version_ids_json, prompt, negative_prompt, meta_json,
                raw_json, like_count, heart_count, laugh_count, cry_count,
                reaction_count, eligibility, eligibility_reason, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                      ?, ?, ?, ?, ?)
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
                like_count=excluded.like_count,
                heart_count=excluded.heart_count,
                laugh_count=excluded.laugh_count,
                cry_count=excluded.cry_count,
                reaction_count=excluded.reaction_count,
                eligibility=excluded.eligibility,
                eligibility_reason=excluded.eligibility_reason,
                updated_at=excluded.updated_at
            """,
            (
                image_id,
                content_class,
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
                counts_by_kind["likeCount"],
                counts_by_kind["heartCount"],
                counts_by_kind["laughCount"],
                counts_by_kind["cryCount"],
                reaction_count,
                eligibility,
                reason,
                now,
            ),
        )
        counts[f"{eligibility}:{reason}"] += 1
    return counts


def inventory(args: argparse.Namespace) -> None:
    root = args.root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    connection = connect(root)
    session = authenticated_session(args.token_env)
    model, model_body = request_json(session, f"{API_ROOT}/models/{MODEL_ID}")
    if int(model.get("id") or 0) != MODEL_ID or str(model.get("name")) != MODEL_NAME:
        raise RuntimeError("Civitai model identity changed")
    official_version_ids = {
        int(version["id"])
        for version in model.get("modelVersions", [])
        if isinstance(version, dict) and version.get("id") is not None
    }
    if not official_version_ids:
        raise RuntimeError("official Anima model has no published version IDs")
    atomic_text(root / "state" / "model.json", json.dumps(model, indent=2) + "\n")
    set_setting(connection, "model_id", MODEL_ID)
    set_setting(connection, "api_root", API_ROOT)
    set_setting(connection, "official_version_ids", canonical_json(sorted(official_version_ids)))
    set_setting(connection, "model_response_sha256", hashlib.sha256(model_body).hexdigest())
    connection.commit()

    for content_class, nsfw in (("sfw", "false"), ("nsfw", "true")):
        complete_key = f"inventory_complete_{content_class}"
        if args.refresh:
            connection.execute("DELETE FROM pages WHERE content_class=?", (content_class,))
            connection.execute(
                "DELETE FROM images WHERE content_class=? AND selected_rank IS NULL",
                (content_class,),
            )
            set_setting(connection, f"next_cursor_{content_class}", "")
            set_setting(connection, complete_key, "0")
            set_setting(connection, f"inventory_started_at_{content_class}", utc_now())
            connection.commit()
        elif setting(connection, f"inventory_started_at_{content_class}") is None:
            set_setting(connection, f"next_cursor_{content_class}", "")
            set_setting(connection, complete_key, "0")
            set_setting(connection, f"inventory_started_at_{content_class}", utc_now())
            connection.commit()
        if setting(connection, complete_key) == "1" and not args.refresh:
            print(canonical_json({"kind": "inventory_skip", "class": content_class}))
            continue

        cursor = setting(connection, f"next_cursor_{content_class}") or ""
        page_number = 0
        while True:
            params: dict[str, Any] = {
                "modelId": MODEL_ID,
                "limit": 200,
                "sort": "Newest",
                "period": "AllTime",
                "nsfw": nsfw,
                "withMeta": "true",
                "flatMeta": "true",
            }
            if cursor:
                params["cursor"] = cursor
            payload, body = request_json(session, f"{API_ROOT}/images", params=params)
            items = payload.get("items")
            metadata = payload.get("metadata")
            if not isinstance(items, list) or not isinstance(metadata, dict):
                raise TypeError("malformed Civitai image page")
            counts = upsert_items(
                connection,
                items,
                content_class=content_class,
                official_version_ids=official_version_ids,
            )
            next_value = metadata.get("nextCursor")
            next_cursor = "" if next_value is None else str(next_value)
            fetched = utc_now()
            connection.execute(
                """
                INSERT OR REPLACE INTO pages(
                    content_class, cursor, next_cursor, item_count,
                    response_sha256, fetched_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    content_class,
                    cursor,
                    next_cursor or None,
                    len(items),
                    hashlib.sha256(body).hexdigest(),
                    fetched,
                ),
            )
            set_setting(connection, f"next_cursor_{content_class}", next_cursor)
            set_setting(connection, f"inventory_updated_at_{content_class}", fetched)
            connection.commit()
            page_number += 1
            print(
                canonical_json(
                    {
                        "kind": "inventory_page",
                        "class": content_class,
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
                set_setting(connection, complete_key, "1")
                set_setting(
                    connection, f"inventory_completed_at_{content_class}", utc_now()
                )
                connection.commit()
                break
            cursor = next_cursor
            time.sleep(args.api_delay)
    connection.close()


def ranked_inventory(args: argparse.Namespace) -> None:
    """Inventory the authoritative all-time reaction ordering used by the site."""
    root = args.root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    connection = connect(root)
    session = authenticated_session(args.token_env)
    model, model_body = request_json(session, f"{API_ROOT}/models/{MODEL_ID}")
    if not isinstance(model, dict):
        raise TypeError("malformed Civitai model response")
    if int(model.get("id") or 0) != MODEL_ID or str(model.get("name")) != MODEL_NAME:
        raise RuntimeError("Civitai model identity changed")
    official_version_ids = {
        int(version["id"])
        for version in model.get("modelVersions", [])
        if isinstance(version, dict) and version.get("id") is not None
    }
    if not official_version_ids:
        raise RuntimeError("official Anima model has no published version IDs")
    atomic_text(root / "state" / "model.json", json.dumps(model, indent=2) + "\n")
    set_setting(connection, "model_id", MODEL_ID)
    set_setting(connection, "api_root", API_ROOT)
    set_setting(
        connection,
        "official_version_ids",
        canonical_json(sorted(official_version_ids)),
    )
    set_setting(
        connection, "model_response_sha256", hashlib.sha256(model_body).hexdigest()
    )
    set_setting(connection, "ranked_inventory_started_at", utc_now())
    set_setting(connection, "selection_frozen", "0")
    connection.execute("UPDATE images SET selected_rank=NULL")
    connection.commit()

    candidate_limit = args.per_class + args.rank_buffer
    streams: list[tuple[str, int]] = [
        (content_class, version_id)
        for content_class in ("sfw", "nsfw")
        for version_id in sorted(official_version_ids)
    ]
    stream_results: dict[tuple[str, int], list[dict[str, Any]]] = {}
    with ThreadPoolExecutor(max_workers=args.inventory_workers) as executor:
        futures = {
            executor.submit(
                ranked_stream,
                token_env=args.token_env,
                model_version_id=version_id,
                content_class=content_class,
                candidate_limit=candidate_limit,
                api_delay=args.api_delay,
            ): (content_class, version_id)
            for content_class, version_id in streams
        }
        for future in as_completed(futures):
            key = futures[future]
            values = future.result()
            stream_results[key] = values
            print(
                canonical_json(
                    {
                        "kind": "ranked_stream",
                        "class": key[0],
                        "model_version_id": key[1],
                        "records": len(values),
                    }
                ),
                flush=True,
            )

    merged: dict[str, dict[int, dict[str, Any]]] = {"sfw": {}, "nsfw": {}}
    associations: dict[str, dict[int, set[int]]] = {"sfw": {}, "nsfw": {}}
    for (content_class, version_id), values in stream_results.items():
        for item in values:
            image_id = positive_int(item.get("id"))
            if not image_id:
                continue
            associations[content_class].setdefault(image_id, set()).add(version_id)
            previous = merged[content_class].get(image_id)
            if previous is None or positive_int(item.get("reactionCount")) > positive_int(
                previous.get("reactionCount")
            ):
                merged[content_class][image_id] = item

    windows: dict[str, list[dict[str, Any]]] = {}
    for content_class in ("sfw", "nsfw"):
        ordered = sorted(
            merged[content_class].values(),
            key=lambda item: (
                -positive_int(item.get("reactionCount")),
                -positive_int(item.get("id")),
            ),
        )
        windows[content_class] = ordered[:candidate_limit]
        if len(windows[content_class]) < args.per_class:
            raise RuntimeError(
                f"ranked {content_class} candidate pool has "
                f"{len(windows[content_class])}; {args.per_class} required"
            )

    window_ids = {
        positive_int(item.get("id"))
        for values in windows.values()
        for item in values
    }
    existing_raw: dict[int, dict[str, Any]] = {}
    ordered_window_ids = sorted(window_ids)
    for id_chunk_start in range(0, len(ordered_window_ids), 500):
        id_chunk = ordered_window_ids[id_chunk_start : id_chunk_start + 500]
        placeholders = ",".join("?" for _ in id_chunk)
        for row in connection.execute(
            f"SELECT id, raw_json FROM images WHERE id IN ({placeholders})",
            id_chunk,
        ):
            try:
                raw = json.loads(row["raw_json"])
            except (TypeError, ValueError):
                continue
            meta = raw.get("meta") if isinstance(raw, dict) else None
            if isinstance(meta, dict) and str(meta.get("prompt") or "").strip():
                existing_raw[int(row["id"])] = raw

    missing_ids = sorted(window_ids.difference(existing_raw))
    generation_data: dict[int, dict[str, Any]] = {}
    batches = [
        missing_ids[start : start + args.generation_batch_size]
        for start in range(0, len(missing_ids), args.generation_batch_size)
    ]
    with ThreadPoolExecutor(max_workers=args.inventory_workers) as executor:
        futures = {
            executor.submit(
                generation_batch,
                token_env=args.token_env,
                image_ids=batch,
            ): batch
            for batch in batches
        }
        for completed_batches, future in enumerate(as_completed(futures), 1):
            generation_data.update(future.result())
            if completed_batches % 25 == 0 or completed_batches == len(batches):
                print(
                    canonical_json(
                        {
                            "kind": "generation_metadata",
                            "batches": completed_batches,
                            "total_batches": len(batches),
                            "records": len(generation_data),
                        }
                    ),
                    flush=True,
                )

    connection.execute(
        """
        UPDATE images SET eligibility='excluded',
            eligibility_reason='outside_ranked_candidate_window',
            selected_rank=NULL, updated_at=?
        """,
        (utc_now(),),
    )
    summaries: dict[str, Counter[str]] = {}
    for content_class, candidates in windows.items():
        counts: Counter[str] = Counter()
        for candidate in candidates:
            image_id = positive_int(candidate.get("id"))
            raw = existing_raw.get(image_id)
            if raw is None:
                generation = generation_data.get(image_id, {})
                user = candidate.get("user")
                if not isinstance(user, dict):
                    user = {}
                raw = {
                    "id": image_id,
                    "postId": candidate.get("postId"),
                    "url": original_image_url(candidate.get("url")),
                    "width": positive_int(candidate.get("width")),
                    "height": positive_int(candidate.get("height")),
                    "type": candidate.get("type") or "image",
                    "nsfwLevel": nsfw_level_name(candidate.get("nsfwLevel")),
                    "createdAt": candidate.get("createdAt"),
                    "username": user.get("username"),
                    "baseModel": candidate.get("baseModel"),
                    "modelVersionIds": sorted(
                        associations[content_class].get(image_id, set())
                    ),
                    "stats": candidate.get("stats") or {},
                    "meta": generation.get("meta"),
                }
            else:
                raw["stats"] = candidate.get("stats") or raw.get("stats") or {}
                raw["modelVersionIds"] = sorted(
                    {
                        *(
                            positive_int(value)
                            for value in (raw.get("modelVersionIds") or [])
                        ),
                        *associations[content_class].get(image_id, set()),
                    }
                    - {0}
                )
            classified = upsert_items(
                connection,
                [raw],
                content_class=content_class,
                official_version_ids=official_version_ids,
            )
            counts.update(classified)
            connection.execute(
                """
                UPDATE images SET reaction_count=?, updated_at=? WHERE id=?
                """,
                (
                    positive_int(candidate.get("reactionCount")),
                    utc_now(),
                    image_id,
                ),
            )
        summaries[content_class] = counts

    for content_class in ("sfw", "nsfw"):
        eligible_count = int(
            connection.execute(
                """
                SELECT count(*) FROM images
                WHERE content_class=? AND eligibility='eligible'
                """,
                (content_class,),
            ).fetchone()[0]
        )
        if eligible_count < args.per_class:
            raise RuntimeError(
                f"ranked {content_class} window produced {eligible_count} eligible "
                f"records; increase --rank-buffer"
            )
        set_setting(connection, f"inventory_complete_{content_class}", "1")
        set_setting(connection, f"inventory_completed_at_{content_class}", utc_now())
    set_setting(connection, "inventory_method", "site_trpc_most_reactions_all_time")
    set_setting(connection, "rank_candidate_limit_per_version", candidate_limit)
    set_setting(connection, "ranked_inventory_completed_at", utc_now())
    connection.commit()
    connection.close()
    print(
        canonical_json(
            {
                "kind": "ranked_inventory_complete",
                "candidate_limit_per_version": candidate_limit,
                "classified": {
                    key: dict(sorted(value.items())) for key, value in summaries.items()
                },
            }
        )
    )


def freeze(args: argparse.Namespace) -> None:
    root = args.root.expanduser().resolve()
    connection = connect(root)
    if any(
        setting(connection, f"inventory_complete_{content_class}") != "1"
        for content_class in ("sfw", "nsfw")
    ):
        raise RuntimeError("both SFW and NSFW inventories must be complete before freeze")
    already_frozen = setting(connection, "selection_frozen") == "1"
    if already_frozen and not args.refreeze:
        print("selection already frozen; use --refreeze to replace it")
        connection.close()
        return
    connection.execute("UPDATE images SET selected_rank=NULL")
    summary: dict[str, Any] = {}
    for content_class in ("sfw", "nsfw"):
        rows = connection.execute(
            """
            SELECT id, reaction_count FROM images
            WHERE content_class=? AND eligibility='eligible'
            ORDER BY reaction_count DESC, id DESC
            LIMIT ?
            """,
            (content_class, args.per_class),
        ).fetchall()
        if len(rows) != args.per_class:
            raise RuntimeError(
                f"{content_class} has {len(rows)} eligible records; "
                f"{args.per_class} required"
            )
        connection.executemany(
            "UPDATE images SET selected_rank=? WHERE id=?",
            ((rank, int(row["id"])) for rank, row in enumerate(rows, 1)),
        )
        summary[content_class] = {
            "selected": len(rows),
            "maximum_reactions": int(rows[0]["reaction_count"]),
            "minimum_reactions": int(rows[-1]["reaction_count"]),
        }
    set_setting(connection, "per_class", args.per_class)
    set_setting(connection, "selection_frozen", "1")
    set_setting(connection, "selection_frozen_at", utc_now())
    set_setting(
        connection,
        "selection_order",
        "reaction_count DESC, civitai_image_id DESC",
    )
    connection.commit()
    materialize(root, connection, require_downloads=False)
    connection.close()
    print(canonical_json({"kind": "selection_frozen", **summary}))


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
    extensions = {
        "JPEG": ".jpg",
        "PNG": ".png",
        "WEBP": ".webp",
        "AVIF": ".avif",
    }
    try:
        return extensions[image_format.upper()]
    except KeyError as error:
        raise RuntimeError(f"unsupported downloaded image format: {image_format}") from error


def png_critical_chunks_valid(path: Path) -> bool:
    """Return true only when all required/critical PNG chunks have valid CRCs."""
    seen: set[bytes] = set()
    with path.open("rb") as handle:
        if handle.read(len(PNG_SIGNATURE)) != PNG_SIGNATURE:
            return False
        while True:
            length_bytes = handle.read(4)
            if len(length_bytes) != 4:
                return False
            length = struct.unpack(">I", length_bytes)[0]
            if length > MAX_IMAGE_BYTES:
                return False
            chunk_type = handle.read(4)
            data = handle.read(length)
            crc_bytes = handle.read(4)
            if len(chunk_type) != 4 or len(data) != length or len(crc_bytes) != 4:
                return False
            expected_crc = struct.unpack(">I", crc_bytes)[0]
            actual_crc = zlib.crc32(chunk_type)
            actual_crc = zlib.crc32(data, actual_crc) & 0xFFFFFFFF
            is_critical = 65 <= chunk_type[0] <= 90
            if is_critical and actual_crc != expected_crc:
                return False
            seen.add(chunk_type)
            if chunk_type == b"IEND":
                break
    return {b"IHDR", b"IDAT", b"IEND"}.issubset(seen)


def inspect_image(path: Path) -> tuple[int, int, str, str | None]:
    try:
        with Image.open(path) as decoded:
            decoded.verify()
        with Image.open(path) as decoded:
            width, height = decoded.size
            image_format = str(decoded.format or "").upper()
        return width, height, image_format, None
    except (SyntaxError, UnidentifiedImageError):
        if not png_critical_chunks_valid(path):
            raise
        with PIL_FALLBACK_LOCK:
            previous = ImageFile.LOAD_TRUNCATED_IMAGES
            try:
                ImageFile.LOAD_TRUNCATED_IMAGES = True
                with Image.open(path) as decoded:
                    decoded.load()
                    width, height = decoded.size
                    image_format = str(decoded.format or "").upper()
            finally:
                ImageFile.LOAD_TRUNCATED_IMAGES = previous
        if image_format != "PNG":
            raise RuntimeError("strict image validation failed for non-PNG media")
        return width, height, image_format, "corrupt_ancillary_png_crc"


def download_one(root: Path, row: sqlite3.Row, limiter: RateLimiter) -> dict[str, Any]:
    image_id = int(row["id"])
    parsed = urlparse(str(row["url"]))
    if parsed.scheme != "https" or parsed.hostname not in ALLOWED_IMAGE_HOSTS:
        raise RuntimeError("image URL left the Civitai image host allowlist")
    content_class = str(row["content_class"])
    shard = f"{image_id // 1000:06d}"
    directory = root / "images" / content_class / shard
    directory.mkdir(parents=True, exist_ok=True)
    temporary = directory / f".{image_id}.part"
    try:
        limiter.wait()
        with requests.get(
            str(row["url"]),
            headers={"Accept": "image/*", "User-Agent": USER_AGENT},
            stream=True,
            timeout=(20, 180),
        ) as response:
            response.raise_for_status()
            content_length = positive_int(response.headers.get("Content-Length"))
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
        width, height, image_format, validation_warning = inspect_image(temporary)
        final = directory / f"{image_id}{image_extension(image_format)}"
        os.replace(temporary, final)
        atomic_text(final.with_suffix(".txt"), str(row["prompt"]).strip() + "\n")
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
        "validation_warning": validation_warning,
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
                UPDATE images SET download_status='downloaded',
                    attempts=attempts+1, file_name=?, sha256=?, size_bytes=?,
                    actual_width=?, actual_height=?, image_format=?,
                    validation_warning=?, error=NULL, updated_at=? WHERE id=?
                """,
                (
                    result["file_name"],
                    result["sha256"],
                    result["size_bytes"],
                    result["actual_width"],
                    result["actual_height"],
                    result["image_format"],
                    result["validation_warning"],
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
    if setting(connection, "selection_frozen") != "1":
        raise RuntimeError("freeze the ranked selection before downloading")
    rows = connection.execute(
        """
        SELECT * FROM images
        WHERE selected_rank IS NOT NULL AND download_status!='downloaded'
        ORDER BY content_class, selected_rank
        """
    ).fetchall()
    connection.close()
    if args.limit is not None:
        rows = rows[: args.limit]
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
            processed = completed + errors
            if processed % 25 == 0 or processed == len(rows):
                print(
                    canonical_json(
                        {
                            "kind": "download_progress",
                            "processed": processed,
                            "scheduled": len(rows),
                            "downloaded": completed,
                            "errors": errors,
                        }
                    ),
                    flush=True,
                )
    connection = connect(root)
    materialize(root, connection, require_downloads=False)
    connection.close()
    if errors:
        raise RuntimeError(f"{errors} image downloads failed; rerun to retry")


def seed_downloads(args: argparse.Namespace) -> None:
    """Reuse verified originals from an existing resumable Anima download."""
    root = args.root.expanduser().resolve()
    seed_root = args.seed_root.expanduser().resolve()
    seed_database = seed_root / "state" / "downloads.sqlite3"
    if not seed_database.is_file():
        raise RuntimeError(f"seed database does not exist: {seed_database}")
    connection = connect(root)
    if setting(connection, "selection_frozen") != "1":
        raise RuntimeError("freeze the ranked selection before seeding downloads")
    connection.execute("ATTACH DATABASE ? AS seed", (str(seed_database),))
    rows = connection.execute(
        """
        SELECT target.id, target.content_class, target.prompt,
               source.file_name AS source_file_name, source.sha256,
               source.size_bytes, source.actual_width, source.actual_height,
               source.image_format
        FROM main.images AS target
        JOIN seed.images AS source ON source.id=target.id
        WHERE target.selected_rank IS NOT NULL
          AND target.download_status!='downloaded'
          AND source.download_status='downloaded'
        ORDER BY target.content_class, target.selected_rank
        """
    ).fetchall()
    seeded = hardlinked = copied = skipped = 0
    for row in rows:
        source = seed_root / str(row["source_file_name"])
        source_sidecar = source.with_suffix(".txt")
        if not source.is_file() or not source_sidecar.is_file():
            skipped += 1
            continue
        prompt = str(row["prompt"]).strip()
        if source_sidecar.read_text(encoding="utf-8").strip() != prompt:
            skipped += 1
            continue
        image_id = int(row["id"])
        content_class = str(row["content_class"])
        shard = f"{image_id // 1000:06d}"
        destination = (
            root / "images" / content_class / shard / f"{image_id}{source.suffix.lower()}"
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            try:
                os.link(source, destination)
                hardlinked += 1
            except OSError:
                shutil.copy2(source, destination)
                copied += 1
        atomic_text(destination.with_suffix(".txt"), prompt + "\n")
        connection.execute(
            """
            UPDATE images SET download_status='downloaded', file_name=?,
                sha256=?, size_bytes=?, actual_width=?, actual_height=?,
                image_format=?, error=NULL, updated_at=? WHERE id=?
            """,
            (
                str(destination.relative_to(root)),
                row["sha256"],
                row["size_bytes"],
                row["actual_width"],
                row["actual_height"],
                row["image_format"],
                utc_now(),
                image_id,
            ),
        )
        seeded += 1
    connection.commit()
    connection.execute("DETACH DATABASE seed")
    materialize(root, connection, require_downloads=False)
    connection.close()
    print(
        canonical_json(
            {
                "kind": "seed_downloads",
                "seeded": seeded,
                "hardlinked": hardlinked,
                "copied": copied,
                "skipped": skipped,
            }
        )
    )


def replace_selected_duplicates(args: argparse.Namespace) -> None:
    """Quarantine lower-ranked duplicate bytes and promote ranked reserves."""
    root = args.root.expanduser().resolve()
    connection = connect(root)
    duplicate_groups = connection.execute(
        """
        SELECT sha256 FROM images
        WHERE selected_rank IS NOT NULL AND download_status='downloaded'
          AND sha256 IS NOT NULL
        GROUP BY sha256 HAVING count(*) > 1
        """
    ).fetchall()
    displaced: list[int] = []
    for group in duplicate_groups:
        rows = connection.execute(
            """
            SELECT * FROM images
            WHERE selected_rank IS NOT NULL AND download_status='downloaded'
              AND sha256=?
            ORDER BY reaction_count DESC, id DESC
            """,
            (group["sha256"],),
        ).fetchall()
        for row in rows[1:]:
            image_id = int(row["id"])
            source = root / str(row["file_name"])
            quarantine = (
                root
                / "state"
                / "replaced_duplicates"
                / str(row["content_class"])
                / source.name
            )
            quarantine.parent.mkdir(parents=True, exist_ok=True)
            if source.is_file():
                os.replace(source, quarantine)
            source_sidecar = source.with_suffix(".txt")
            quarantine_sidecar = quarantine.with_suffix(".txt")
            if source_sidecar.is_file():
                os.replace(source_sidecar, quarantine_sidecar)
            connection.execute(
                """
                UPDATE images SET selected_rank=NULL, eligibility='excluded',
                    eligibility_reason='duplicate_sha256_selected',
                    download_status='excluded_duplicate', file_name=?,
                    updated_at=? WHERE id=?
                """,
                (str(quarantine.relative_to(root)), utc_now(), image_id),
            )
            displaced.append(image_id)
    set_setting(connection, "selection_frozen", "0")
    connection.commit()
    connection.close()
    freeze(args)
    print(
        canonical_json(
            {
                "kind": "replace_selected_duplicates",
                "duplicate_groups": len(duplicate_groups),
                "displaced_ids": displaced,
            }
        )
    )


def selected_rows(connection: sqlite3.Connection) -> list[sqlite3.Row]:
    return connection.execute(
        """
        SELECT * FROM images WHERE selected_rank IS NOT NULL
        ORDER BY content_class, selected_rank
        """
    ).fetchall()


def metadata_record(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "file_name": row["file_name"],
        "text": row["prompt"],
        "negative_prompt": row["negative_prompt"] or "",
        "content_class": row["content_class"],
        "class_rank": int(row["selected_rank"]),
        "reaction_count": int(row["reaction_count"]),
        "reactions": {
            "like": int(row["like_count"]),
            "heart": int(row["heart_count"]),
            "laugh": int(row["laugh_count"]),
            "cry": int(row["cry_count"]),
        },
        "civitai_image_id": int(row["id"]),
        "civitai_post_id": row["post_id"],
        "creator": row["username"],
        "source_url": f"{SITE_ROOT}/images/{int(row['id'])}",
        "model_id": MODEL_ID,
        "model_version_ids": json.loads(row["model_version_ids_json"]),
        "nsfw_level": row["nsfw_level"],
        "width": row["actual_width"],
        "height": row["actual_height"],
        "sha256": row["sha256"],
        "validation_warning": row["validation_warning"],
        "generation_metadata": (
            json.loads(row["meta_json"]) if row["meta_json"] else None
        ),
    }


def materialize(
    root: Path,
    connection: sqlite3.Connection,
    *,
    require_downloads: bool,
) -> dict[str, Any]:
    rows = selected_rows(connection)
    downloaded = [row for row in rows if row["download_status"] == "downloaded"]
    if require_downloads and len(downloaded) != len(rows):
        raise RuntimeError("not all frozen records have downloaded")
    atomic_text(
        root / "metadata.jsonl",
        "".join(canonical_json(metadata_record(row)) + "\n" for row in downloaded),
    )
    inventory_counts = {
        str(row["key"]): int(row["count"])
        for row in connection.execute(
            """
            SELECT content_class || ':' || eligibility || ':' ||
                   eligibility_reason AS key, count(*) AS count
            FROM images GROUP BY key ORDER BY key
            """
        )
    }
    selected_counts = {
        str(row["content_class"]): int(row["count"])
        for row in connection.execute(
            """
            SELECT content_class, count(*) AS count FROM images
            WHERE selected_rank IS NOT NULL GROUP BY content_class
            """
        )
    }
    downloaded_counts = {
        str(row["content_class"]): int(row["count"])
        for row in connection.execute(
            """
            SELECT content_class, count(*) AS count FROM images
            WHERE selected_rank IS NOT NULL AND download_status='downloaded'
            GROUP BY content_class
            """
        )
    }
    extension_counts: Counter[str] = Counter()
    total_bytes = 0
    missing = 0
    for row in downloaded:
        path = root / str(row["file_name"])
        if not path.is_file() or not path.with_suffix(".txt").is_file():
            missing += 1
        extension_counts[path.suffix.lower()] += 1
        total_bytes += int(row["size_bytes"] or 0)
    duplicate_sha_groups = int(
        connection.execute(
            """
            SELECT count(*) FROM (
                SELECT sha256 FROM images
                WHERE selected_rank IS NOT NULL AND download_status='downloaded'
                  AND sha256 IS NOT NULL
                GROUP BY sha256 HAVING count(*) > 1
            )
            """
        ).fetchone()[0]
    )
    validation_warning_counts = {
        str(row["validation_warning"]): int(row["count"])
        for row in connection.execute(
            """
            SELECT validation_warning, count(*) AS count FROM images
            WHERE selected_rank IS NOT NULL AND download_status='downloaded'
              AND validation_warning IS NOT NULL
            GROUP BY validation_warning ORDER BY validation_warning
            """
        )
    }
    report = {
        "schema": "rwkv-lab.civitai-anima-balanced.v1",
        "model_id": MODEL_ID,
        "source": f"{SITE_ROOT}/models/{MODEL_ID}",
        "api_endpoint": f"{API_ROOT}/images",
        "inventory_method": setting(connection, "inventory_method"),
        "selection_order": setting(connection, "selection_order"),
        "selection_frozen": setting(connection, "selection_frozen") == "1",
        "selection_frozen_at": setting(connection, "selection_frozen_at"),
        "per_class": positive_int(setting(connection, "per_class")),
        "inventory_complete": {
            key: setting(connection, f"inventory_complete_{key}") == "1"
            for key in ("sfw", "nsfw")
        },
        "inventory_records": int(
            connection.execute("SELECT count(*) FROM images").fetchone()[0]
        ),
        "inventory_counts": inventory_counts,
        "selected_counts": selected_counts,
        "downloaded_counts": downloaded_counts,
        "downloaded_prompt_pairs": len(downloaded),
        "downloaded_bytes": total_bytes,
        "extension_counts": dict(sorted(extension_counts.items())),
        "duplicate_sha256_groups": duplicate_sha_groups,
        "validation_warning_counts": validation_warning_counts,
        "missing_image_or_sidecar": missing,
        "content_policy": {
            "sfw_levels": sorted(SFW_LEVELS),
            "adult_levels": sorted(ADULT_LEVELS),
            "adult_minor_marker_policy": "quarantine",
        },
        "license_observation": {
            "observed_at": "2026-07-28",
            "civitai_terms": f"{SITE_ROOT}/content/tos",
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
    report = materialize(root, connection, require_downloads=True)
    errors: list[str] = []
    expected = args.per_class
    for content_class in ("sfw", "nsfw"):
        if report["selected_counts"].get(content_class, 0) != expected:
            errors.append(f"{content_class} selection is not {expected}")
        if report["downloaded_counts"].get(content_class, 0) != expected:
            errors.append(f"{content_class} download count is not {expected}")
    if report["missing_image_or_sidecar"]:
        errors.append(f"{report['missing_image_or_sidecar']} records lack files")
    with (root / "metadata.jsonl").open(encoding="utf-8") as metadata_file:
        metadata_count = sum(1 for line in metadata_file if line.strip())
    if metadata_count != expected * 2:
        errors.append(f"metadata.jsonl has {metadata_count}, expected {expected * 2}")
    empty_prompts = int(
        connection.execute(
            """
            SELECT count(*) FROM images WHERE selected_rank IS NOT NULL
              AND trim(coalesce(prompt, ''))=''
            """
        ).fetchone()[0]
    )
    duplicate_ids = int(
        connection.execute(
            """
            SELECT count(*) FROM (
                SELECT id FROM images WHERE selected_rank IS NOT NULL
                GROUP BY id HAVING count(*) > 1
            )
            """
        ).fetchone()[0]
    )
    unsafe_adult = int(
        connection.execute(
            """
            SELECT count(*) FROM images WHERE selected_rank IS NOT NULL
              AND content_class='nsfw' AND (
                eligibility!='eligible' OR nsfw_level NOT IN ('Mature', 'X', 'XXX')
              )
            """
        ).fetchone()[0]
    )
    if empty_prompts:
        errors.append(f"{empty_prompts} selected prompts are empty")
    if duplicate_ids:
        errors.append(f"{duplicate_ids} duplicate image IDs")
    if unsafe_adult:
        errors.append(f"{unsafe_adult} adult records violate classification")
    connection.close()
    if errors:
        raise RuntimeError("; ".join(errors))
    print(json.dumps(report, indent=2, sort_keys=True))


def status(args: argparse.Namespace) -> None:
    connection = connect(args.root.expanduser().resolve())
    report = materialize(
        args.root.expanduser().resolve(), connection, require_downloads=False
    )
    connection.close()
    print(json.dumps(report, indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=(
            "inventory",
            "ranked-inventory",
            "freeze",
            "seed",
            "deduplicate",
            "download",
            "validate",
            "status",
            "all",
        ),
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument(
        "--seed-root",
        type=Path,
        default=Path("/thearray/git/datasets/anima-civitai-official"),
    )
    parser.add_argument("--token-env", default="CIVITAI_TOKEN")
    parser.add_argument("--per-class", type=int, default=5_000)
    parser.add_argument("--rank-buffer", type=int, default=1_000)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--refreeze", action="store_true")
    parser.add_argument("--api-delay", type=float, default=0.25)
    parser.add_argument("--inventory-workers", type=int, default=7)
    parser.add_argument("--generation-batch-size", type=int, default=10)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--requests-per-second", type=float, default=3.0)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if (
        args.per_class < 1
        or args.rank_buffer < 0
        or args.api_delay < 0
        or args.inventory_workers < 1
        or not 1 <= args.generation_batch_size <= 20
        or args.workers < 1
        or args.requests_per_second <= 0
        or (args.limit is not None and args.limit < 1)
    ):
        parser.error("invalid count, pacing, worker, or limit")
    return args


def main() -> None:
    args = parse_args()
    if args.command == "inventory":
        inventory(args)
    if args.command in {"ranked-inventory", "all"}:
        ranked_inventory(args)
    if args.command in {"freeze", "all"}:
        freeze(args)
    if args.command == "seed":
        seed_downloads(args)
    if args.command == "deduplicate":
        replace_selected_duplicates(args)
    if args.command in {"download", "all"}:
        download(args)
    if args.command == "validate":
        validate(args)
    if args.command == "status":
        status(args)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("interrupted; SQLite state is resumable", file=sys.stderr)
        raise SystemExit(130)
