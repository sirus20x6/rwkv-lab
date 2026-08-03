from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any


class CanonicalJsonError(ValueError):
    pass


def _object_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise CanonicalJsonError(f"duplicate JSON member: {key}")
        value[key] = item
    return value


def canonical_loads(raw: bytes, *, maximum_bytes: int) -> dict[str, Any]:
    if not raw or len(raw) > maximum_bytes:
        raise CanonicalJsonError("canonical JSON size is invalid")
    try:
        text = raw.decode("utf-8", errors="strict")
        value = json.loads(
            text,
            object_pairs_hook=_object_without_duplicates,
            parse_constant=lambda value: (_ for _ in ()).throw(
                CanonicalJsonError(f"non-finite JSON number: {value}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CanonicalJsonError("canonical JSON decoding failed") from error
    if not isinstance(value, dict):
        raise CanonicalJsonError("canonical JSON root must be an object")
    if canonical_dumps(value) != raw:
        raise CanonicalJsonError("JSON is not in canonical wire form")
    return value


def canonical_dumps(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def sha256_digest(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def exact_fields(value: Mapping[str, Any], fields: frozenset[str]) -> None:
    if set(value) != fields:
        raise CanonicalJsonError("JSON object fields are inexact")


def is_uint64(value: Any, *, positive: bool = False) -> bool:
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and (value > 0 if positive else value >= 0)
        and value <= (1 << 64) - 1
    )


def is_bounded_text(value: Any, maximum: int) -> bool:
    return (
        isinstance(value, str)
        and 0 < len(value.encode("utf-8")) <= maximum
        and "\x00" not in value
        and "\n" not in value
        and "\r" not in value
    )


def is_digest(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 71
        and value.startswith("sha256:")
        and all(character in "0123456789abcdef" for character in value[7:])
    )


def deep_freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({key: deep_freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(deep_freeze(item) for item in value)
    return value
