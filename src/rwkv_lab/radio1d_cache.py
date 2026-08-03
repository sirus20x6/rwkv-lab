"""Atomic, fingerprinted BF16 caches for tiled RADIO1D-H global tokens."""
from __future__ import annotations

import hashlib
import json
import os
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from torch import Tensor

from rwkv_lab.radio1d_rwkv import (
    DEFAULT_ADAPTIVE_TOKEN_THRESHOLD, RADIO_COMPACT_TOKENS_PER_TILE,
    RADIO_HIDDEN_SIZE, RADIO_TILE_SIZE, RADIO_TOKENS_PER_TILE, TILE_SCHEMA,
    RadioTile, tokens_per_tile_for_tile_count,
)


LEGACY_CACHE_SCHEMA = "radio1d-global-cache-v1"
CACHE_SCHEMA = "radio1d-global-cache-v2"
TOKEN_POLICY = "native-256-through-threshold-then-native-128-v1"


def sha256_file(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def cache_path(cache_dir: Path, source_path: Path) -> Path:
    resolved = os.path.realpath(os.fsencode(source_path))
    name = hashlib.sha256(resolved).hexdigest()
    return cache_dir / name[:2] / f"{name}.safetensors"


@dataclass(frozen=True)
class CachedTile:
    source_box: tuple[float, float, float, float]
    role: str
    index: int
    row: int
    column: int
    grid_rows: int
    grid_columns: int

    @classmethod
    def from_tile(cls, tile: RadioTile) -> "CachedTile":
        return cls(tile.source_box, tile.role, tile.index, tile.row, tile.column,
                   tile.grid_rows, tile.grid_columns)


@dataclass(frozen=True)
class RadioCacheMetadata:
    source_path: str
    source_sha256: str
    source_size: int
    source_mtime_ns: int
    checkpoint_revision: str
    tiles: tuple[CachedTile, ...]
    schema: str = CACHE_SCHEMA
    tile_schema: str = TILE_SCHEMA
    tile_size: int = RADIO_TILE_SIZE
    tokens_per_tile: int = RADIO_TOKENS_PER_TILE
    hidden_size: int = RADIO_HIDDEN_SIZE
    adaptive_token_threshold: int = DEFAULT_ADAPTIVE_TOKEN_THRESHOLD
    token_policy: str = TOKEN_POLICY

    def validate(self) -> None:
        if self.schema not in (CACHE_SCHEMA, LEGACY_CACHE_SCHEMA) or self.tile_schema != TILE_SCHEMA:
            raise ValueError("unsupported RADIO cache schema")
        if (self.tile_size, self.hidden_size) != (
                RADIO_TILE_SIZE, RADIO_HIDDEN_SIZE):
            raise ValueError("cache has a lossy or incompatible RADIO contract")
        if len(self.source_sha256) != 64 or not self.tiles:
            raise ValueError("cache source fingerprint or tile layout is missing")
        if not self.checkpoint_revision:
            raise ValueError("checkpoint revision is required")
        if [tile.index for tile in self.tiles] != list(range(len(self.tiles))):
            raise ValueError("tiles are not in canonical sequence order")
        expected = tokens_per_tile_for_tile_count(
            len(self.tiles), threshold=self.adaptive_token_threshold)
        if self.tokens_per_tile != expected or self.token_policy != TOKEN_POLICY:
            raise ValueError("cache does not follow the adaptive RADIO token policy")

    def to_json(self) -> str:
        self.validate()
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"),
                          ensure_ascii=True)

    @classmethod
    def from_json(cls, value: str) -> "RadioCacheMetadata":
        raw = json.loads(value)
        # Legacy caches did not identify their policy. They are compatible only
        # when their stored width already matches today's deterministic policy;
        # long 256-token entries therefore fail validation and are regenerated.
        raw.setdefault("adaptive_token_threshold", DEFAULT_ADAPTIVE_TOKEN_THRESHOLD)
        raw.setdefault("token_policy", TOKEN_POLICY)
        raw["tiles"] = tuple(CachedTile(
            source_box=tuple(tile["source_box"]), role=tile["role"],
            index=tile["index"], row=tile["row"], column=tile["column"],
            grid_rows=tile["grid_rows"], grid_columns=tile["grid_columns"])
            for tile in raw["tiles"])
        result = cls(**raw)
        result.validate()
        return result


def validate_features(metadata: RadioCacheMetadata, features: Tensor) -> None:
    metadata.validate()
    expected = (len(metadata.tiles), metadata.tokens_per_tile, RADIO_HIDDEN_SIZE)
    if tuple(features.shape) != expected:
        raise ValueError(f"cached features {tuple(features.shape)} != {expected}")
    if features.device.type != "cpu" or features.dtype != torch.bfloat16:
        raise ValueError("cached RADIO features must be CPU BF16")
    if not bool(torch.isfinite(features).all()):
        raise ValueError("cached RADIO features contain non-finite values")


def make_metadata(source: Path, revision: str,
                  tiles: Sequence[RadioTile], *,
                  adaptive_token_threshold: int = DEFAULT_ADAPTIVE_TOKEN_THRESHOLD
                  ) -> RadioCacheMetadata:
    stat = source.stat()
    return RadioCacheMetadata(
        source_path=str(source.resolve()), source_sha256=sha256_file(source),
        source_size=stat.st_size, source_mtime_ns=stat.st_mtime_ns,
        checkpoint_revision=revision,
        tiles=tuple(CachedTile.from_tile(tile) for tile in tiles),
        tokens_per_tile=tokens_per_tile_for_tile_count(
            len(tiles), threshold=adaptive_token_threshold),
        adaptive_token_threshold=adaptive_token_threshold)


def save_cache(path: Path, metadata: RadioCacheMetadata, features: Tensor) -> None:
    validate_features(metadata, features)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Cache generation may come from the trainer and its exact-next-batch
    # prefetch worker at the same time.  A PID-only temporary lets those two
    # valid writers truncate/unlink one another's file.  The thread id keeps
    # each atomic producer private without changing the final cache identity.
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}-{threading.get_ident()}.tmp")
    try:
        save_file({"global_tokens": features.contiguous()}, str(temporary),
                  metadata={"schema": CACHE_SCHEMA,
                            "radio_metadata": metadata.to_json()})
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def load_cache(path: Path) -> tuple[RadioCacheMetadata, Tensor]:
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        header = handle.metadata() or {}
    if (header.get("schema") not in (CACHE_SCHEMA, LEGACY_CACHE_SCHEMA)
            or "radio_metadata" not in header):
        raise ValueError(f"{path} is not a supported RADIO cache entry")
    metadata = RadioCacheMetadata.from_json(header["radio_metadata"])
    values = load_file(str(path), device="cpu")
    if set(values) != {"global_tokens"}:
        raise ValueError("RADIO cache contains unexpected tensors")
    features = values["global_tokens"]
    validate_features(metadata, features)
    return metadata, features


def cache_is_current(
        path: Path, source: Path, revision: str, *,
        adaptive_token_threshold: int = DEFAULT_ADAPTIVE_TOKEN_THRESHOLD,
        source_sha256: str | None = None) -> bool:
    if not path.is_file():
        return False
    try:
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            header = handle.metadata() or {}
        if header.get("schema") not in (CACHE_SCHEMA, LEGACY_CACHE_SCHEMA):
            return False
        metadata = RadioCacheMetadata.from_json(header["radio_metadata"])
        stat = source.stat()
        source_identity_matches = (
            metadata.source_mtime_ns == stat.st_mtime_ns
            or (source_sha256 is not None
                and len(source_sha256) == 64
                and metadata.source_sha256 == source_sha256)
        )
        # Deliberately NOT comparing the raw adaptive_token_threshold. What must
        # match is the token width this entry actually holds, which the next
        # clause checks against today's policy for this tile count. Gating on the
        # threshold as well discards every entry whenever the knob moves, even
        # when the resulting width is identical: raising it from 12 to 49 leaves
        # 96% of a corpus byte-for-byte correct yet invalidated all 142k entries
        # (533 GB) instead of the ~4k whose width genuinely changed. The stored
        # threshold is retained as provenance.
        return (metadata.checkpoint_revision == revision
                and metadata.tokens_per_tile == tokens_per_tile_for_tile_count(
                    len(metadata.tiles), threshold=adaptive_token_threshold)
                and metadata.source_size == stat.st_size
                and source_identity_matches)
    except (OSError, ValueError, RuntimeError):
        return False
