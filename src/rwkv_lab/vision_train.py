"""Train a frozen MoonViT -> RWKV caption model with recoverable runs.

The pretrained MoonViT and RWKV weights remain frozen.  Training updates only
the image-prefix projector, NextLat auxiliary predictor, per-layer factored
TimeMix loop adapters, and optional Engram recall adapters. Runs checkpoint
atomically and resume automatically, including the exact sampler and RNG state.
"""
from __future__ import annotations

import argparse
import atexit
import contextlib
import fcntl
import hashlib
import json
import math
import os
import pickle
import random
import signal
import sys
import threading
import time
import zipfile
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Callable, Sequence

import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn

from rwkv_lab.generate import SEP, WorldVocab
from rwkv_lab.fused_ce import HAS_FUSED_CE, weighted_logits_cross_entropy
from rwkv_lab.deep_vision import DeepVisionInjector, LayerMatchedVisionInjector
from rwkv_lab.engram_lmb import (
    LexicalMemoryBank,
    RecallResult,
    attach_engram,
    engram_parameters,
    float_growth_params,
    token_rosa_recall,
)
from rwkv_lab.lookahead_module import NextLatPredictor, nextlat_loss
from rwkv_lab.moonvit import (MoonViT, MoonViTPrefixProjector, feature_cache_key,
                              valid_pooled_feature as _valid_pooled_feature,
                              valid_pooled_feature_archive as _valid_pooled_feature_archive,
                              valid_pooled_feature_payload as _valid_pooled_feature_payload,
                              valid_torch_archive_storages)
from rwkv_lab.rwkv_finetune import load_g1g_fla
from rwkv_lab.vision_loop import (
    install_factored_timemix,
    load_loop_adapter_state,
    loop_adapter_state,
    loop_training_metrics,
    reset_loop_adapters,
    set_loop_enabled,
    set_loop_scale,
    write_loop_telemetry,
)
from rwkv_lab.vision_grounding import ImageTextContrastiveHead, early_token_weights
from rwkv_lab.vision_fusion import (
    AlignedFrozenVisionFeatures,
    VisionFusionResidual,
    VisionTowerConfig,
    aligned_feature_cache_key,
    valid_aligned_feature,
)

ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT_SCHEMA = 3
_CACHE_LOAD_POOL = ThreadPoolExecutor(max_workers=16, thread_name_prefix="vision-cache")
# Background warming must leave I/O and CPU headroom for the exact next-batch
# prefetch. Sixty-four concurrent torch ZIP readers previously flooded the same
# disk queue the trainer was waiting on, making "background" preload a foreground
# stall on a cold cache.
_FEATURE_PRELOAD_POOL = ThreadPoolExecutor(max_workers=8, thread_name_prefix="vision-preload")
_NEXT_BATCH_POOL = ThreadPoolExecutor(max_workers=1, thread_name_prefix="vision-next-batch")
_FEATURE_MEMORY_CACHE: dict[Path, torch.Tensor] = {}


def _acquire_run_lock(out: Path):
    """Hold an advisory exclusive lock for one trainer process per run."""
    path = out / ".trainer.lock"
    handle = path.open("w")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        raise RuntimeError(f"another vision trainer already owns run {out}") from None
    return handle


def _load_raw_tensor_archive(path: Path, *, shape: tuple[int, ...],
                             stride: tuple[int, ...], storage_offset: int,
                             dtype: torch.dtype, storage_bytes: int) -> torch.Tensor:
    """Load a one-storage ``torch.save`` tensor without unpickling each file."""
    try:
        with zipfile.ZipFile(path) as archive:
            members = [info for info in archive.infolist()
                       if info.filename.endswith("/data/0")]
            metadata = [info for info in archive.infolist()
                        if info.filename.endswith("/data.pkl")]
            if len(members) != 1 or members[0].file_size != storage_bytes:
                raise ValueError("unexpected tensor archive layout")
            dtype_markers = {
                torch.float16: b"\nHalfStorage\n",
                torch.bfloat16: b"\nBFloat16Storage\n",
                torch.float32: b"\nFloatStorage\n",
            }
            marker = dtype_markers.get(dtype)
            if len(metadata) != 1 or marker is None \
                    or marker not in archive.read(metadata[0]):
                # fp16 and bf16 have identical storage sizes. Reading one as
                # the other silently produces plausible-shaped garbage, so a
                # dtype mismatch must take the safe torch.load fallback.
                raise ValueError("tensor archive dtype does not match template")
            payload = bytearray(archive.read(members[0]))
        storage = torch.frombuffer(payload, dtype=dtype)
        return torch.as_strided(storage, shape, stride, storage_offset)
    except zipfile.BadZipFile:
        # A PyTorch ZIP whose storage CRC fails must not fall through to
        # torch.load: PyTorch accepts finite bit flips without checking the CRC.
        # Only a genuine legacy non-ZIP serialization may use the compatibility
        # path (and its caller will reject it for lacking an integrity receipt).
        try:
            with path.open("rb") as handle:
                is_zip_archive = handle.read(4).startswith(b"PK")
        except OSError:
            raise
        if is_zip_archive:
            raise
        return torch.load(path, map_location="cpu", weights_only=True)
    except (FileNotFoundError, OSError, ValueError):
        # Preserve layout/dtype compatibility through the safe general loader.
        # Callers additionally verify the loaded storage against the ZIP CRC.
        return torch.load(path, map_location="cpu", weights_only=True)


def load_examples(path: str | Path, *, root: Path = ROOT,
                  stat_workers: int = 1) -> list[dict]:
    """Read valid image-caption rows without silently accepting missing files."""
    candidates = []
    source = Path(path)
    # Iterate physical lines. str.splitlines() also splits valid JSON strings at
    # Unicode line-separator characters (U+2028/U+2029), corrupting such rows.
    with source.open() as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not row.get("image") or not row.get("text"):
                continue
            image = Path(row["image"])
            image = image if image.is_absolute() else root / image
            text = row["text"].strip()
            if text:
                candidates.append((row, image, text, line_number))

    def inspect(candidate: tuple[dict, Path, str, int]) -> dict | None:
        row, image, text, line_number = candidate
        try:
            image = image.resolve()
            source_stat = image.stat()
            if not image.is_file():
                return None
        except OSError:
            return None
        item = dict(row)
        item.update(image=image, text=text, manifest=str(source), line=line_number,
                    _source_size=source_stat.st_size,
                    _source_mtime_ns=source_stat.st_mtime_ns,
                    _source_dev=source_stat.st_dev,
                    _source_ino=source_stat.st_ino)
        return item

    if stat_workers > 1:
        with ThreadPoolExecutor(max_workers=stat_workers,
                                thread_name_prefix="vision-manifest") as pool:
            inspected = pool.map(inspect, candidates)
            rows = [item for item in inspected if item is not None]
    else:
        rows = [item for candidate in candidates if (item := inspect(candidate)) is not None]
    return rows


def _image_file_identity(row: dict) -> tuple[str, int, int] | tuple[str, str]:
    """Identify one physical file, collapsing symlinks and hard links."""
    device, inode = row.get("_source_dev"), row.get("_source_ino")
    if (isinstance(device, int) and not isinstance(device, bool)
            and isinstance(inode, int) and not isinstance(inode, bool)
            and inode > 0):
        return "inode", device, inode
    return "path", str(Path(row["image"]).resolve())


def _row_feature_cache_key(row: dict, *, max_input_patches: int,
                           prefix_tokens: int, vision_fingerprint: str,
                           tap_layers: Sequence[int] = (),
                           view_mode: str = "full") -> str:
    """Build a cache key without repeating source filesystem metadata reads."""
    return feature_cache_key(
        row["image"], max_input_patches=max_input_patches,
        prefix_tokens=prefix_tokens, vision_fingerprint=vision_fingerprint,
        source_size=row.get("_source_size"),
        source_mtime_ns=row.get("_source_mtime_ns"),
        tap_layers=tap_layers, view_mode=view_mode,
    )


def prepare_examples(rows: Sequence[dict], vocab: WorldVocab, *, prompt: str,
                     max_text_tokens: int,
                     sandwich_prompt: bool = False) -> tuple[list[dict], list[int]]:
    """Tokenize once, append EOD, and retain EOD when a caption is truncated."""
    prompt_cache: dict[str, list[int]] = {}
    prepared, lengths = [], []
    for row in rows:
        row_prompt = str(row.get("prompt") or prompt)
        prompt_tokens = prompt_cache.get(row_prompt)
        if prompt_tokens is None:
            prompt_tokens = vocab.encode(row_prompt)
            prompt_cache[row_prompt] = prompt_tokens
        prompt_copies = 2 if sandwich_prompt else 1
        prompt_width = prompt_copies * len(prompt_tokens)
        if prompt_width + 2 > max_text_tokens:
            raise ValueError("max_text_tokens is too small for prompt + one caption token + EOD")
        room = max_text_tokens - prompt_width - 1
        caption = vocab.encode(row["text"])
        if not caption:
            continue
        tokens = prompt_tokens * prompt_copies + caption[:room] + [SEP]
        item = dict(row)
        item["tokens"] = tokens
        item["prompt_len"] = prompt_width
        item["vision_insert"] = len(prompt_tokens) if sandwich_prompt else 0
        item["prompt"] = row_prompt
        item["truncated"] = len(caption) > room
        prepared.append(item)
        lengths.append(len(tokens))
    return prepared, lengths


def split_examples(rows: Sequence[dict], *, val_fraction: float) -> tuple[list[int], list[int]]:
    """Stable image-disjoint split that is unchanged by manifest row ordering."""
    if not 0.0 < val_fraction < 0.5:
        raise ValueError("val_fraction must be between 0 and 0.5")
    groups: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        identity = repr(_image_file_identity(row))
        groups.setdefault(identity, []).append(index)
    if len(groups) < 2:
        raise ValueError("image-disjoint validation needs at least two unique images")
    train, val = [], []
    cutoff = int(val_fraction * 1_000_000)
    buckets = []
    for identity in sorted(groups):
        bucket = int.from_bytes(hashlib.sha256(identity.encode()).digest()[:8], "big") % 1_000_000
        buckets.append((bucket, identity))
        (val if bucket < cutoff else train).extend(groups[identity])
    if not val:
        _, identity = min(buckets)
        selected = set(groups[identity])
        train = [index for index in train if index not in selected]
        val.extend(groups[identity])
    if not train:
        _, identity = max(buckets)
        selected = set(groups[identity])
        val = [index for index in val if index not in selected]
        train.extend(groups[identity])
    return train, val


def dataset_fingerprint(rows: Sequence[dict], train_indices: Sequence[int],
                        val_indices: Sequence[int], *, explicit_eval: bool) -> str:
    """Fingerprint image/text/token identities with schema-3 compatibility."""
    if explicit_eval:
        train_index_set = set(train_indices)
        lines = (
            f"{'train' if index in train_index_set else 'eval'}\0"
            f"{row['image'].resolve()}\0{row['text']}\0{','.join(map(str, row['tokens']))}"
            for index, row in enumerate(rows)
        )
    else:
        # Preserve the schema-3 fingerprint used by existing resumable runs.
        lines = (
            f"{row['image'].resolve()}\0{row['text']}\0{','.join(map(str, row['tokens']))}"
            for row in rows
        )
    return hashlib.sha256("\n".join(lines).encode()).hexdigest()


def image_metadata_fingerprint(rows: Sequence[dict]) -> str:
    """Cheaply pin the image bytes represented by a resumable run.

    Full hashing hundreds of gigabytes at every launch is not practical. The
    manifest loader already captured each file's size and nanosecond mtime for
    the feature-cache key, so bind that same identity into new checkpoints.
    Older schema-3 checkpoints did not carry this field and remain loadable;
    once re-saved, subsequent resumes enforce it.
    """
    lines = (
        f"{row['image'].resolve()}\0{row.get('_source_size', -1)}\0"
        f"{row.get('_source_mtime_ns', -1)}"
        for row in rows
    )
    return hashlib.sha256("\n".join(lines).encode()).hexdigest()


class EpochBatchSampler:
    """No-replacement epoch sampler with recoverable order and position.

    Random windows are locally sorted by token length.  This retains stochastic
    batches while avoiding the worst padding overhead from the caption tail.
    """
    def __init__(self, indices: Sequence[int], lengths: Sequence[int], *, batch_size: int,
                 seed: int, bucket_batches: int = 32):
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.indices = list(indices)
        self.lengths = list(lengths)
        self.batch_size = int(batch_size)
        self.bucket_batches = max(1, int(bucket_batches))
        self.generator = torch.Generator().manual_seed(seed)
        self.epoch = 0
        self.position = 0
        self.order: list[int] = []
        self._new_epoch()

    def _new_epoch(self) -> None:
        shuffled = [self.indices[i] for i in torch.randperm(len(self.indices), generator=self.generator).tolist()]
        window = self.batch_size * self.bucket_batches
        ordered = []
        for start in range(0, len(shuffled), window):
            ordered.extend(sorted(shuffled[start:start + window], key=self.lengths.__getitem__))
        self.order = ordered
        self.position = 0

    def next_batch(self) -> list[int]:
        if self.position >= len(self.order):
            self.epoch += 1
            self._new_epoch()
        end = min(self.position + self.batch_size, len(self.order))
        result = self.order[self.position:end]
        self.position = end
        return result

    def next_budget_batch(self, token_costs: Sequence[int], *, target_tokens: int,
                          min_items: int, max_items: int) -> list[int]:
        """Take a no-replacement batch sized to a padded-token budget."""
        self.ensure_epoch()
        result = self.peek_budget_batch(
            token_costs, target_tokens=target_tokens,
            min_items=min_items, max_items=max_items)
        self.commit_batch(result)
        return result

    def ensure_epoch(self) -> None:
        """Prepare a new deterministic order once the previous epoch is consumed."""
        if self.position >= len(self.order):
            self.epoch += 1
            self._new_epoch()

    def commit_batch(self, indices: Sequence[int]) -> None:
        """Atomically consume a batch previously returned by ``peek_budget_batch``."""
        expected = self.order[self.position:self.position + len(indices)]
        if list(indices) != expected:
            raise ValueError("cannot commit a batch that is not the sampler's current prefix")
        self.position += len(indices)

    def peek_budget_batch(self, token_costs: Sequence[int], *, target_tokens: int,
                          min_items: int, max_items: int,
                          position_offset: int = 0) -> list[int]:
        """Return the next batch without advancing recoverable sampler state."""
        if min_items < 1 or max_items < min_items:
            raise ValueError("budget batch limits must satisfy 1 <= min_items <= max_items")
        position = self.position + int(position_offset)
        if position_offset < 0:
            raise ValueError("position_offset must be non-negative")
        if position >= len(self.order):
            return []
        if target_tokens <= 0:
            end = min(position + self.batch_size, len(self.order))
            return self.order[position:end]
        if max_items == min_items:
            end = min(position + min_items, len(self.order))
            return self.order[position:end]
        available = min(max_items, len(self.order) - position)
        take = min(min_items, available)
        max_cost = max(token_costs[index]
                       for index in self.order[position:position + take])
        # Batches are locally length-sorted. Account for padding using the
        # actual longest item, including when a batch crosses a bucket boundary.
        while take < available:
            candidate = self.order[position + take]
            candidate_max = max(max_cost, token_costs[candidate])
            padded = (take + 1) * candidate_max
            if padded > target_tokens:
                break
            max_cost = candidate_max
            take += 1
        return self.order[position:position + take]

    def state_dict(self) -> dict:
        return {"epoch": self.epoch, "position": self.position, "order": self.order,
                "generator_state": self.generator.get_state()}

    def load_state_dict(self, state: dict) -> None:
        order = [int(i) for i in state["order"]]
        if sorted(order) != sorted(self.indices):
            raise ValueError("checkpoint sampler does not match the current training split")
        position = int(state["position"])
        if not 0 <= position <= len(order):
            raise ValueError("invalid checkpoint sampler position")
        self.epoch, self.position, self.order = int(state["epoch"]), position, order
        self.generator.set_state(state["generator_state"])


def make_batch(rows: Sequence[dict], vocab: WorldVocab | None = None, *,
               prompt: str = "Describe this image:\n", device: str = "cuda",
               max_text_tokens: int = 384) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pad a prepared batch and mask prompt/padding from caption CE.

    The backwards-compatible tokenizer path is useful to small callers/tests;
    the trainer itself pre-tokenizes every row exactly once.
    """
    if not rows:
        raise ValueError("cannot make an empty batch")
    if "tokens" not in rows[0]:
        if vocab is None:
            raise ValueError("unprepared rows require a vocab")
        rows, _ = prepare_examples(rows, vocab, prompt=prompt, max_text_tokens=max_text_tokens)
    width = max(len(row["tokens"]) for row in rows)
    ids = torch.zeros(len(rows), width, dtype=torch.long, device=device)
    labels = torch.full_like(ids, -100)
    mask = torch.zeros_like(ids, dtype=torch.bool)
    for i, row in enumerate(rows):
        tokens = torch.tensor(row["tokens"], dtype=torch.long, device=device)
        length, prompt_len = len(tokens), int(row["prompt_len"])
        ids[i, :length] = tokens
        labels[i, prompt_len:length] = tokens[prompt_len:]
        mask[i, :length] = True
    return ids, labels, mask


def visual_insert_positions(rows: Sequence[dict]) -> tuple[int, ...]:
    """Return the per-row text offset at which the visual span is inserted."""
    starts = tuple(int(row.get("vision_insert", 0)) for row in rows)
    for row, start in zip(rows, starts):
        if start < 0 or start > int(row.get("prompt_len", 0)):
            raise ValueError("visual insertion must occur inside the masked prompt")
    return starts


def insert_visual_span(text: torch.Tensor, visual: torch.Tensor,
                       starts: Sequence[int]) -> torch.Tensor:
    """Insert a fixed-width visual span into every padded text row."""
    if text.shape[0] != visual.shape[0] or len(starts) != text.shape[0]:
        raise ValueError("visual span batch does not match text batch")
    rows = []
    for batch, start in enumerate(starts):
        if start < 0 or start > text.shape[1]:
            raise ValueError("visual insertion falls outside text sequence")
        rows.append(torch.cat((text[batch, :start], visual[batch],
                               text[batch, start:]), dim=0))
    return torch.stack(rows)


def remove_visual_span(sequence: torch.Tensor, starts: Sequence[int],
                       width: int) -> torch.Tensor:
    """Undo :func:`insert_visual_span` while retaining padded text layout."""
    if len(starts) != sequence.shape[0] or width < 0:
        raise ValueError("visual removal contract does not match sequence")
    rows = []
    for batch, start in enumerate(starts):
        end = start + width
        if start < 0 or end > sequence.shape[1]:
            raise ValueError("visual span falls outside sequence")
        rows.append(torch.cat((sequence[batch, :start], sequence[batch, end:]), dim=0))
    return torch.stack(rows)


def insert_boundary_ids(ids: torch.Tensor, starts: Sequence[int], width: int,
                        boundary: int) -> torch.Tensor:
    placeholders = torch.full((ids.shape[0], width), int(boundary),
                              dtype=ids.dtype, device=ids.device)
    return insert_visual_span(ids, placeholders, starts)


def supervised_positions(rows: Sequence[dict], prefix_tokens: int, *,
                         device: str = "cuda") -> torch.Tensor:
    """Build LM-head selectors on CPU metadata, avoiding CUDA ``nonzero`` sync."""
    positions = [
        (batch, prefix_tokens + target - 1)
        for batch, row in enumerate(rows)
        for target in range(int(row["prompt_len"]), len(row["tokens"]))
    ]
    if not positions:
        raise ValueError("batch contains no supervised caption tokens")
    return torch.tensor(positions, dtype=torch.long, device=device)


def cached_features(rows: Sequence[dict], vision: MoonViT,
                    projector: MoonViTPrefixProjector, cache_dir: Path | None) -> list[torch.Tensor]:
    """Load pooled frozen features, batch-encoding cache misses by image grid."""
    result: list[torch.Tensor | None] = [None] * len(rows)
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
    missing: list[tuple[int, Path | None, Image.Image]] = []
    device = vision.patch_embed.proj.weight.device
    fingerprint = getattr(vision, "cache_fingerprint", "unknown")
    stages = int(getattr(vision, "feature_stages", 1))
    tap_layers = tuple(getattr(vision, "tap_layers", ()))
    view_mode = str(getattr(vision, "view_mode", "full"))

    def load_one(row: dict) -> tuple[Path | None, torch.Tensor | None]:
        path = None if cache_dir is None else cache_dir / _row_feature_cache_key(
            row, max_input_patches=vision.max_input_patches,
            prefix_tokens=projector.prefix_tokens, vision_fingerprint=fingerprint,
            tap_layers=tap_layers, view_mode=view_mode)
        if path is None:
            return None, None
        memory_item = _FEATURE_MEMORY_CACHE.get(path)
        if memory_item is not None and _valid_pooled_feature(
                memory_item, projector.prefix_tokens, stages):
            return path, memory_item
        if memory_item is not None:
            _FEATURE_MEMORY_CACHE.pop(path, None)
        try:
            # Loading many small files directly onto CUDA serialized a device
            # transfer (and filesystem latency) per caption.  Read in parallel
            # on CPU, then make one contiguous host->device transfer below.
            item = torch.load(path, map_location="cpu", weights_only=True)
            if not _valid_pooled_feature_archive(
                    path, item, projector.prefix_tokens, stages):
                path.unlink(missing_ok=True)
                return path, None
            return path, item
        except (OSError, EOFError, RuntimeError, pickle.UnpicklingError,
                zipfile.BadZipFile):
            path.unlink(missing_ok=True)
            return path, None

    loaded = list(_CACHE_LOAD_POOL.map(load_one, rows))
    for index, (row, (path, item)) in enumerate(zip(rows, loaded)):
        if item is None:
            with Image.open(row["image"]) as image:
                missing.append((index, path, image.convert("RGB")))
        else:
            if path is not None:
                _FEATURE_MEMORY_CACHE.setdefault(path, item)
            result[index] = item
    if missing:
        raw = vision.encode_many([image for _, _, image in missing])
        for (index, path, _), item in zip(missing, raw):
            item = projector.pool_features(item).squeeze(0).detach()
            if not _valid_pooled_feature_payload(
                    item, projector.prefix_tokens, stages):
                raise FloatingPointError(
                    f"MoonViT produced an invalid pooled feature for row {index}")
            if path is not None:
                # Cache filling can overlap an external prefill process. Use a
                # writer-unique temporary so two valid producers never corrupt
                # each other's archive before the atomic replace.
                temporary = path.with_name(
                    f".{path.name}.{os.getpid()}-{threading.get_ident()}.tmp")
                try:
                    torch.save(item.cpu(), temporary)
                    os.replace(temporary, path)
                finally:
                    temporary.unlink(missing_ok=True)
            result[index] = item

    cpu_indices = [i for i, item in enumerate(result)
                   if item is not None and item.device.type == "cpu"]
    if cpu_indices:
        packed = torch.stack([result[i] for i in cpu_indices]).to(device=device)
        for i, item in zip(cpu_indices, packed.unbind(0)):
            result[i] = item
    if any(item is None for item in result):
        raise RuntimeError("feature cache loader left an unresolved item")
    return list(result)  # type: ignore[return-value]


def _row_fusion_cache_path(row: dict, cache_dir: Path, *, tokens: int,
                           tower_fingerprint: str) -> Path:
    return cache_dir / aligned_feature_cache_key(
        row["image"], tokens=tokens, tower_fingerprint=tower_fingerprint,
        source_size=row.get("_source_size"),
        source_mtime_ns=row.get("_source_mtime_ns"))


def cached_fusion_features(
        rows: Sequence[dict], tower: AlignedFrozenVisionFeatures,
        prefix_tokens: int, cache_dir: Path) -> list[torch.Tensor]:
    """Load aligned frozen three-tower features and fill cache misses."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    paths = [_row_fusion_cache_path(
        row, cache_dir, tokens=prefix_tokens,
        tower_fingerprint=tower.cache_fingerprint) for row in rows]

    def load_one(path: Path) -> torch.Tensor | None:
        resident = _FEATURE_MEMORY_CACHE.get(path)
        if resident is not None and valid_aligned_feature(
                resident, prefix_tokens, tower.width):
            return resident
        if resident is not None:
            _FEATURE_MEMORY_CACHE.pop(path, None)
        try:
            item = torch.load(path, map_location="cpu", weights_only=True)
            if (not valid_aligned_feature(item, prefix_tokens, tower.width)
                    or not valid_torch_archive_storages(path, item)):
                path.unlink(missing_ok=True)
                return None
            _FEATURE_MEMORY_CACHE[path] = item
            return item
        except (OSError, EOFError, RuntimeError, pickle.UnpicklingError,
                zipfile.BadZipFile):
            path.unlink(missing_ok=True)
            return None

    result = list(_CACHE_LOAD_POOL.map(load_one, paths))
    missing = [index for index, item in enumerate(result) if item is None]
    if missing:
        if not tower.loaded:
            print("loading frozen SigLIP2 + DINOv2 + SAM for fusion cache misses",
                  flush=True)
            tower.load_pretrained(device="cuda", dtype=torch.bfloat16)
        images = []
        for index in missing:
            with Image.open(rows[index]["image"]) as image:
                images.append(image.convert("RGB"))
        encoded = tower(images, tokens=prefix_tokens, device="cuda")
        for index, item in zip(missing, encoded.unbind(0)):
            item = item.detach().to(device="cpu", dtype=torch.bfloat16)
            if not valid_aligned_feature(item, prefix_tokens, tower.width):
                raise FloatingPointError(
                    f"three-tower fusion produced invalid features for row {index}")
            path = paths[index]
            temporary = path.with_name(
                f".{path.name}.{os.getpid()}-{threading.get_ident()}.tmp")
            try:
                torch.save(item, temporary)
                os.replace(temporary, path)
            finally:
                temporary.unlink(missing_ok=True)
            _FEATURE_MEMORY_CACHE[path] = item
            result[index] = item
    if any(item is None for item in result):
        raise RuntimeError("fusion cache loader left an unresolved item")
    packed = torch.stack(result).to(device="cuda")  # type: ignore[arg-type]
    return list(packed.unbind(0))


def preload_feature_cache(rows: Sequence[dict], vision: MoonViT,
                          projector: MoonViTPrefixProjector,
                          cache_dir: Path,
                          stop_event: threading.Event | None = None) -> tuple[int, int]:
    """Deserialize the fixed MoonViT cache once instead of once per epoch.

    This machine has enough system RAM for the pooled feature corpus. Keeping
    CPU tensors resident removes thousands of small ``torch.load`` calls from
    the hot training loop without pinning 30+ GiB or consuming GPU memory.
    Missing entries remain on the ordinary encode-and-fill fallback path.
    """
    fingerprint = getattr(vision, "cache_fingerprint", "unknown")
    stages = int(getattr(vision, "feature_stages", 1))
    tap_layers = tuple(getattr(vision, "tap_layers", ()))
    view_mode = str(getattr(vision, "view_mode", "full"))
    paths = list(dict.fromkeys(
        cache_dir / _row_feature_cache_key(
            row, max_input_patches=vision.max_input_patches,
            prefix_tokens=projector.prefix_tokens,
            vision_fingerprint=fingerprint, tap_layers=tap_layers,
            view_mode=view_mode)
        for row in rows
    ))
    pending = [path for path in paths if path not in _FEATURE_MEMORY_CACHE]
    if stop_event is not None and stop_event.is_set():
        return 0, 0

    template_path = None
    template_item = None
    for candidate in pending:
        if stop_event is not None and stop_event.is_set():
            return 0, 0
        if not candidate.is_file():
            continue
        try:
            item = torch.load(candidate, map_location="cpu", weights_only=True)
        except (OSError, EOFError, RuntimeError, pickle.UnpicklingError,
                zipfile.BadZipFile):
            continue
        if _valid_pooled_feature_archive(
                candidate, item, projector.prefix_tokens, stages):
            template_path, template_item = candidate, item
            break
    layout = None
    if template_item is not None:
        layout = {
            "shape": tuple(template_item.shape),
            "stride": tuple(template_item.stride()),
            "storage_offset": int(template_item.storage_offset()),
            "dtype": template_item.dtype,
            "storage_bytes": int(template_item.untyped_storage().nbytes()),
        }

    def load_one(path: Path) -> tuple[Path, torch.Tensor | None]:
        try:
            if path == template_path:
                return path, template_item
            if layout is not None:
                item = _load_raw_tensor_archive(path, **layout)
            else:
                item = torch.load(path, map_location="cpu", weights_only=True)
            if not _valid_pooled_feature_archive(
                    path, item, projector.prefix_tokens, stages):
                return path, None
            return path, item
        except (OSError, EOFError, RuntimeError, pickle.UnpicklingError,
                zipfile.BadZipFile):
            return path, None

    loaded = 0
    resident_bytes = 0
    # Executor.map eagerly queued the entire corpus. Global ThreadPoolExecutor
    # workers are joined by Python at process exit, so SIGINT could save the
    # checkpoint and then appear hung while thousands of "background" reads
    # drained. Keep only a small bounded window in flight and stop submitting as
    # soon as the trainer exits.
    iterator = iter(pending)
    in_flight = set()

    def fill() -> None:
        while len(in_flight) < 16 and not (stop_event and stop_event.is_set()):
            try:
                path = next(iterator)
            except StopIteration:
                break
            in_flight.add(_FEATURE_PRELOAD_POOL.submit(load_one, path))

    fill()
    while in_flight:
        done, in_flight = wait(
            in_flight, timeout=0.5, return_when=FIRST_COMPLETED)
        if not done and stop_event and stop_event.is_set():
            for future in in_flight:
                future.cancel()
            break
        for future in done:
            path, item = future.result()
            if item is None:
                continue
            _FEATURE_MEMORY_CACHE[path] = item
            loaded += 1
            resident_bytes += item.numel() * item.element_size()
            if loaded % 4096 == 0:
                print({"kind": "feature_preload", "loaded": loaded,
                       "total": len(pending)}, flush=True)
        if stop_event and stop_event.is_set():
            for future in in_flight:
                future.cancel()
            break
        fill()
    return loaded, resident_bytes


def prefetch_cached_feature_rows(rows: Sequence[dict], vision: MoonViT,
                                 projector: MoonViTPrefixProjector,
                                 cache_dir: Path | None) -> int:
    """Bring an exact future batch into RAM without touching CUDA or sampler state."""
    if cache_dir is None:
        return 0
    fingerprint = getattr(vision, "cache_fingerprint", "unknown")
    stages = int(getattr(vision, "feature_stages", 1))
    tap_layers = tuple(getattr(vision, "tap_layers", ()))
    view_mode = str(getattr(vision, "view_mode", "full"))
    paths = [cache_dir / _row_feature_cache_key(
        row, max_input_patches=vision.max_input_patches,
        prefix_tokens=projector.prefix_tokens,
        vision_fingerprint=fingerprint, tap_layers=tap_layers,
        view_mode=view_mode) for row in rows]

    def load_one(path: Path) -> tuple[Path, torch.Tensor | None, bool]:
        existing = _FEATURE_MEMORY_CACHE.get(path)
        if existing is not None and _valid_pooled_feature(
                existing, projector.prefix_tokens, stages):
            return path, existing, True
        if existing is not None:
            _FEATURE_MEMORY_CACHE.pop(path, None)
        try:
            item = torch.load(path, map_location="cpu", weights_only=True)
            if not _valid_pooled_feature_archive(
                    path, item, projector.prefix_tokens, stages):
                return path, None, False
            return path, item, False
        except (OSError, EOFError, RuntimeError, pickle.UnpicklingError,
                zipfile.BadZipFile):
            return path, None, False

    ready = 0
    for path, item, _resident in _CACHE_LOAD_POOL.map(load_one, paths):
        if item is not None:
            _FEATURE_MEMORY_CACHE.setdefault(path, item)
            ready += 1
    return ready


def prefetch_fusion_feature_rows(
        rows: Sequence[dict], tower: AlignedFrozenVisionFeatures | None,
        prefix_tokens: int, cache_dir: Path | None) -> int:
    if tower is None or cache_dir is None:
        return 0
    paths = [_row_fusion_cache_path(
        row, cache_dir, tokens=prefix_tokens,
        tower_fingerprint=tower.cache_fingerprint) for row in rows]

    def load_one(path: Path) -> tuple[Path, torch.Tensor | None]:
        item = _FEATURE_MEMORY_CACHE.get(path)
        if item is not None and valid_aligned_feature(
                item, prefix_tokens, tower.width):
            return path, item
        try:
            item = torch.load(path, map_location="cpu", weights_only=True)
            if (not valid_aligned_feature(item, prefix_tokens, tower.width)
                    or not valid_torch_archive_storages(path, item)):
                return path, None
            return path, item
        except (OSError, EOFError, RuntimeError, pickle.UnpicklingError,
                zipfile.BadZipFile):
            return path, None

    ready = 0
    for path, item in _CACHE_LOAD_POOL.map(load_one, paths):
        if item is not None:
            _FEATURE_MEMORY_CACHE.setdefault(path, item)
            ready += 1
    return ready


def prefetch_training_batch(rows: Sequence[dict], vision: MoonViT,
                            projector: MoonViTPrefixProjector,
                            cache_dir: Path | None,
                            engram: LexicalMemoryBank | None,
                            fusion_tower: AlignedFrozenVisionFeatures | None = None,
                            fusion_cache_dir: Path | None = None,
                            ) -> tuple[int, RecallResult | None]:
    """Prepare disk features and CPU lexical recall while the GPU is busy."""
    ready = prefetch_cached_feature_rows(rows, vision, projector, cache_dir)
    ready += prefetch_fusion_feature_rows(
        rows, fusion_tower, projector.prefix_tokens, fusion_cache_dir)
    if engram is None:
        return ready, None
    ids, _, _ = make_batch(rows, device="cpu")
    boundary = 0 if engram.boundary_id is None else int(engram.boundary_id)
    starts = visual_insert_positions(rows)
    recall = token_rosa_recall(
        insert_boundary_ids(ids, starts, projector.prefix_tokens, boundary),
        engram.table.vocab_size,
        engram.boundary_id)
    return ready, recall


def multimodal_loss(rwkv: nn.Module, projector: nn.Module, vision: MoonViT,
                    images: Sequence[Image.Image], ids: torch.Tensor,
                    labels: torch.Tensor, text_mask: torch.Tensor, *,
                    nextlat: NextLatPredictor | None = None,
                    nextlat_weight: float = 0.0, nextlat_kl_weight: float = 0.0,
                    engram: LexicalMemoryBank | None = None,
                    features: list[torch.Tensor] | None = None,
                    selected_positions: torch.Tensor | None = None,
                    engram_recall: RecallResult | None = None,
                    deep_vision: DeepVisionInjector | None = None,
                    layer_vision: LayerMatchedVisionInjector | None = None,
                    visual_starts: Sequence[int] | None = None,
                    fusion_adapter: VisionFusionResidual | None = None,
                    fusion_features: Sequence[torch.Tensor] | None = None,
                    grounding: ImageTextContrastiveHead | None = None,
                    grounding_contrastive_weight: float = 0.0,
                    grounding_early_tokens: int = 0,
                    grounding_early_weight: float = 1.0,
                    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if features is None:
        with torch.no_grad():
            features = vision(images)
    prefix = projector(features)
    fusion_residual = None
    if fusion_adapter is not None:
        if fusion_features is None:
            raise ValueError("fusion adapter requires cached three-tower features")
        fusion_residual = fusion_adapter(fusion_features)
        prefix = prefix + fusion_residual.to(prefix.dtype)
    text = rwkv.model.embeddings(ids)
    prefix = prefix.to(dtype=text.dtype)
    starts = tuple(int(value) for value in (
        visual_starts if visual_starts is not None else (0,) * ids.shape[0]))
    embeds = insert_visual_span(text, prefix, starts)
    ignore = torch.full((ids.shape[0], prefix.shape[1]), -100,
                        dtype=labels.dtype, device=labels.device)
    full_labels = insert_visual_span(labels, ignore, starts)
    attention_mask = insert_visual_span(
        text_mask, torch.ones_like(ignore, dtype=torch.bool), starts)
    if engram is not None:
        # The vision prefix has no vocabulary IDs. Treat it as a sequence of
        # boundaries so lexical recall begins fresh at the textual prompt and
        # can never manufacture matches from identical placeholder IDs.
        boundary = 0 if engram.boundary_id is None else int(engram.boundary_id)
        engram.set_input_ids(
            insert_boundary_ids(ids, starts, prefix.shape[1], boundary),
            recall=engram_recall)
    with contextlib.ExitStack() as stack:
        if deep_vision is not None:
            stack.enter_context(deep_vision.use_prefix(prefix, starts))
        if layer_vision is not None:
            staged = torch.stack(features)
            stack.enter_context(layer_vision.use_features(staged, starts))
        output = rwkv.model(inputs_embeds=embeds, attention_mask=attention_mask,
                            output_hidden_states=False, use_cache=False, return_dict=True)
    hidden = output.last_hidden_state
    # Apply the 65,536-way head only to states that predict caption targets.
    # Image-prefix, prompt-only, and padding states never materialize logits.
    if selected_positions is None:
        # Compatibility fallback for small external callers. The trainer passes
        # selectors built from row metadata because CUDA nonzero synchronizes.
        selected_positions = (full_labels[:, 1:] != -100).nonzero()
    selected_positions = selected_positions.to(device=hidden.device, dtype=torch.long)
    batch_positions = selected_positions[:, 0]
    sequence_positions = selected_positions[:, 1]
    causal_hidden = hidden[batch_positions, sequence_positions]
    selected_targets = full_labels[batch_positions, sequence_positions + 1]
    selected_logits = rwkv.lm_head(causal_hidden)
    if engram is not None:
        selected_logits = engram.logit_bias_at(
            selected_logits, batch_positions, sequence_positions,
            inplace=True)
    token_weights = early_token_weights(
        full_labels, batch_positions, sequence_positions,
        token_count=grounding_early_tokens, weight=grounding_early_weight)
    ce, raw_ce = weighted_logits_cross_entropy(
        selected_logits, selected_targets, token_weights)
    loss = ce
    # Leave scalar telemetry on-device until after backward/optimizer. Python
    # conversion here would force a forward-to-CPU barrier every training step.
    metrics = {"ce_loss": raw_ce, "grounded_ce_loss": ce.detach()}
    if grounding is not None and grounding_contrastive_weight:
        target_embeddings = rwkv.model.embeddings(selected_targets).detach()
        contrastive, accuracy = grounding(
            prefix, target_embeddings, batch_positions)
        loss = loss + grounding_contrastive_weight * contrastive
        metrics.update(grounding_contrastive_loss=contrastive.detach(),
                       grounding_retrieval_accuracy=accuracy.detach())
    if deep_vision is not None:
        metrics["deep_vision_inj_rms"] = deep_vision.injection_rms().detach()
    if layer_vision is not None:
        metrics["layer_vision_inj_rms"] = layer_vision.injection_rms().detach()
    if fusion_residual is not None:
        metrics["vision_fusion_rms"] = (
            fusion_residual.detach().float().square().mean().sqrt())
    if nextlat is not None and nextlat_weight:
        h_text = remove_visual_span(hidden, starts, prefix.shape[1])
        valid = text_mask[:, 1:]
        if valid.shape[1] == 0:
            raise ValueError("NextLat needs at least two unpadded tokens per batch")
        if nextlat_kl_weight:
            # The KL path creates full-vocabulary logits; retain its per-row
            # implementation so padding never enters that expensive objective.
            terms = []
            for i, length in enumerate(text_mask.sum(-1).tolist()):
                if length >= 2:
                    terms.append(nextlat_loss(
                        nextlat, h_text[i:i + 1, :length], text[i:i + 1, :length].detach(),
                        rwkv.lm_head.weight, getattr(rwkv.lm_head, "bias", None),
                        kl_weight=nextlat_kl_weight))
            latent, kl = (torch.stack([term[j] for term in terms]).mean() for j in (0, 1))
        else:
            # Default one-step NextLat is one batched MLP call, not one launch
            # per caption.  Reduce only over real (unpadded) state transitions.
            predicted = nextlat(h_text[:, :-1], text[:, 1:].detach())
            per_token = F.smooth_l1_loss(predicted.float(), h_text[:, 1:].detach().float(),
                                         reduction="none").mean(-1)
            latent = per_token.masked_select(valid).mean()
            kl = latent.new_zeros(())
        auxiliary = latent + nextlat_kl_weight * kl
        loss = loss + nextlat_weight * auxiliary
        metrics.update(nextlat_loss=latent.detach(), nextlat_kl=kl.detach(),
                       aux_loss=auxiliary.detach())
    return loss, metrics


def _cpu_state(module: nn.Module | None) -> dict | None:
    return None if module is None else {k: v.detach().cpu() for k, v in module.state_dict().items()}


def _fsync_directory(path: Path) -> None:
    """Commit directory-entry changes needed to recover after host power loss."""
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _durable_replace(temporary: Path, path: Path) -> None:
    """Publish a closed file atomically and durably, preserving the old target on write failure."""
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    _fsync_directory(path.parent)


def _atomic_json(path: Path, value: dict, *, durable: bool = False) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str) + "\n")
    if durable:
        _durable_replace(temporary, path)
    else:
        os.replace(temporary, path)


def _publish_eval_contract_reset(path: Path, *, step: int,
                                 reasons: Sequence[str]) -> None:
    """Durably publish whether historical eval minima belong to this contract."""
    normalized = list(dict.fromkeys(str(reason).strip() for reason in reasons
                                    if str(reason).strip()))
    if step < 0 or not normalized:
        raise ValueError("eval contract receipt requires a non-negative step and reason")
    _atomic_json(path, {
        "schema": 1,
        "reset": True,
        "step": int(step),
        "reasons": normalized,
    }, durable=True)


def _trainer_run_artifact_paths(out: Path) -> list[Path]:
    """Return active run identity/artifacts, excluding live coordination files."""
    fixed_names = (
        "train.jsonl", "train.tmp", "train.eval-reset.tmp",
        "last.pt", "last.tmp", "best", "eval_samples",
        "pre_loop.pt", "pre_loop.tmp", "loop_rw.json", "loop_rw.tmp",
        "config.json", "config.json.tmp", "status.json", "status.json.tmp",
        "eval_contract_reset.json", "eval_contract_reset.json.tmp",
        "overnight_caption_smoke.json", "overnight_caption_smoke.json.tmp",
        "overnight_inference.log",
    )
    candidates = [out / name for name in fixed_names
                  if os.path.lexists(out / name)]
    candidates.extend(path for path in sorted(out.glob("operator_step_*"))
                      if path not in candidates)
    candidates.extend(
        path for path in sorted(out.glob("overnight_caption_smoke.failed-*"))
        if path not in candidates)
    return candidates


def _archive_fresh_run_artifacts(out: Path, *, stamp: str | None = None) -> Path | None:
    """Move active trainer-owned artifacts out of a deliberate fresh run.

    Keeping only ``train.jsonl``/``last.pt``/``best`` was insufficient: stale
    qualitative cards, loop telemetry, rollback checkpoints, profiles, config,
    and status could all be rendered as if they belonged to the new run. Lock
    and watchdog files are intentionally not touched because another process may
    hold their inode while launching this trainer.
    """
    candidates = _trainer_run_artifact_paths(out)
    if not candidates:
        return None

    stamp = stamp or time.strftime("%Y%m%d-%H%M%S")
    archive = out / f".fresh-backup-{stamp}"
    suffix = 1
    while os.path.lexists(archive):
        archive = out / f".fresh-backup-{stamp}.{suffix}"
        suffix += 1
    archive.mkdir()
    _fsync_directory(out)
    for path in candidates:
        os.replace(path, archive / path.name)
    _fsync_directory(archive)
    _fsync_directory(out)
    return archive


def _sync_log(handle) -> None:
    """Make a sparse recovery-boundary record survive an abrupt host failure."""
    handle.flush()
    os.fsync(handle.fileno())


def _publish_eval_due(handle, *, step: int, checkpoint_path: Path,
                      train_record: dict | None,
                      save_checkpoint: Callable[[], None]) -> None:
    """Durably order the train row, exact checkpoint, and eval obligation."""
    if train_record is not None:
        handle.write(json.dumps(train_record) + "\n")
        # If the checkpoint below survives a host failure, the dashboard row
        # describing that committed optimizer update must survive with it.
        _sync_log(handle)
    save_checkpoint()
    handle.write(json.dumps({
        "kind": "checkpoint", "step": step,
        "reason": "eval_due", "path": str(checkpoint_path),
    }) + "\n")
    handle.write(json.dumps({"kind": "eval_due", "step": step}) + "\n")
    _sync_log(handle)


def _fail_nonterminal_status(path: Path, *, reason: str,
                             error: str | None = None) -> bool:
    """Turn a stranded loading/training status into an explicit failure."""
    try:
        current = json.loads(path.read_text()) if path.is_file() else {}
    except (OSError, json.JSONDecodeError):
        current = {}
    state = str(current.get("state", ""))
    if state in {"complete", "paused", "failed", "stopped"}:
        return False
    failed = {
        **current, "state": "failed", "previous_state": state or None,
        "reason": reason, "updated": time.time(),
    }
    if error:
        failed["error"] = error
    try:
        _atomic_json(path, failed, durable=True)
    except OSError:
        return False
    return True


def _trim_log(path: Path, checkpoint_step: int) -> None:
    """Discard only records newer than the recovered checkpoint."""
    if not path.exists():
        return
    kept = []
    for line in path.read_text().splitlines():
        try:
            record = json.loads(line)
            record_step = int(record.get("step", 0))
        except (json.JSONDecodeError, TypeError, ValueError):
            continue
        if record_step <= checkpoint_step:
            kept.append(json.dumps(record))
    temporary = path.with_suffix(".tmp")
    temporary.write_text("\n".join(kept) + ("\n" if kept else ""))
    _durable_replace(temporary, path)


def _invalidate_step_evaluation(path: Path, checkpoint_step: int) -> bool:
    """Durably discard eval claims invalidated by a same-step model mutation."""
    if not path.exists():
        return False
    eval_kinds = {"eval_due", "eval", "eval_artifact"}
    eval_checkpoint_reasons = {"eval_due", "best_eval_promoted"}
    kept: list[str] = []
    removed = False
    # Keep malformed/unrelated lines byte-for-byte. _trim_log has already
    # bounded ordinary resume history; this function removes only claims whose
    # model/evaluation contract is known to have changed at checkpoint_step.
    for line in path.read_text().splitlines():
        discard = False
        try:
            record = json.loads(line)
            same_step = int(record.get("step", -1)) == checkpoint_step
            discard = bool(
                same_step
                and (record.get("kind") in eval_kinds
                     or (record.get("kind") == "checkpoint"
                         and record.get("reason") in eval_checkpoint_reasons)))
        except (json.JSONDecodeError, TypeError, ValueError):
            pass
        if discard:
            removed = True
        else:
            kept.append(line)
    if not removed:
        return False
    temporary = path.with_suffix(".eval-reset.tmp")
    temporary.write_text("\n".join(kept) + ("\n" if kept else ""))
    _durable_replace(temporary, path)
    return True


def _pending_eval_work(
        path: Path, checkpoint_step: int, *,
        eval_expected: bool = False) -> tuple[str, dict | None] | None:
    """Return unfinished scalar or qualitative work for a committed eval.

    ``eval_due`` makes the scalar evaluation durable across an interrupt. The
    scalar ``eval`` row advances the obligation to caption generation, and only
    ``eval_artifact`` clears it.
    """
    # A scheduled checkpoint step is itself sufficient evidence that scalar
    # eval is owed. This closes the tiny crash window after durable checkpoint
    # publication but before the separate eval_due log record reaches disk.
    pending: tuple[str, dict | None] | None = (
        ("loss", None) if eval_expected else None)
    if not path.exists():
        return pending
    for line in path.read_text().splitlines():
        try:
            record = json.loads(line)
            record_step = int(record.get("step", -1))
        except (json.JSONDecodeError, TypeError, ValueError):
            continue
        if record_step != checkpoint_step:
            continue
        if record.get("kind") == "eval_due":
            pending = ("loss", None)
        elif record.get("kind") == "eval":
            pending = ("captions", record)
        elif record.get("kind") == "eval_artifact":
            pending = None
    if pending is not None and pending[0] == "captions":
        scalar = pending[1] or {}
        if scalar.get("qualitative_complete") is True:
            return None
        raw_artifact = scalar.get("sample_artifact")
        if raw_artifact is None:
            return None
        artifact = Path(str(raw_artifact))
        candidates = ([artifact] if artifact.is_absolute() else
                      [artifact, path.parent / "eval_samples" / artifact.name])
        for candidate in candidates:
            try:
                payload = json.loads(candidate.read_text())
                # Artifacts predating the resumable ``complete`` field were
                # written only after decoding and are therefore complete.
                complete = bool(payload.get("complete", True))
                artifact_ppl = float(payload.get("ppl"))
                scalar_ppl = float(scalar.get("ppl"))
                same_eval = math.isclose(
                    artifact_ppl, scalar_ppl, rel_tol=1e-9, abs_tol=1e-12)
                if (int(payload.get("step", -1)) == checkpoint_step
                        and complete and same_eval):
                    return None
            except (OSError, json.JSONDecodeError, TypeError, ValueError):
                continue
    return pending


def _quarantine_best(best_dir: Path, label: str) -> Path | None:
    """Preserve a best directory while removing it from the active run branch."""
    if not best_dir.exists():
        return None
    candidate = best_dir.with_name(f"{best_dir.name}.{label}")
    suffix = 1
    while candidate.exists():
        candidate = best_dir.with_name(f"{best_dir.name}.{label}.{suffix}")
        suffix += 1
    best_dir.rename(candidate)
    # The absence of the active best directory is part of the resumed branch
    # contract. Do not rely on a later checkpoint save to incidentally commit
    # this rename; some recovery paths perform no subsequent serialization.
    _fsync_directory(best_dir.parent)
    return candidate


def _valid_best_manifest_metadata(payload: object) -> bool:
    """Validate the step and metric identity shared by new and legacy manifests."""
    if not isinstance(payload, dict):
        return False
    step = payload.get("step")
    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        return False
    found_metric = False
    for name in ("loss", "ppl"):
        if name not in payload:
            continue
        found_metric = True
        value = payload[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return False
        rendered = float(value)
        if not math.isfinite(rendered):
            return False
        if name == "loss" and rendered < 0:
            return False
        if name == "ppl" and rendered <= 0:
            return False
    return found_metric


def _contained_regular_checkpoint(best_dir: Path, candidate: Path) -> bool:
    """Require a real checkpoint file physically selected inside ``best_dir``."""
    try:
        if candidate.is_symlink() or not candidate.is_file():
            return False
        return candidate.resolve(strict=True).parent == best_dir.resolve(strict=True)
    except OSError:
        return False


def _best_checkpoint_path(best_dir: Path) -> Path | None:
    """Resolve the checkpoint named by the atomic best manifest.

    New manifests point at an immutable, step-qualified file so the metadata
    and checkpoint can be selected as one atomic publication.  ``ckpt.pt`` is
    retained as a compatibility alias for older runs and callers.
    """
    info_path = best_dir / "best.json"
    # Once a manifest exists it is the authoritative atomic publication. Never
    # pair malformed/new metadata with an unrelated legacy alias underneath it.
    if os.path.lexists(info_path):
        if not info_path.is_file():
            return None
        try:
            payload = json.loads(info_path.read_text())
        except (OSError, json.JSONDecodeError, TypeError):
            return None
        if not isinstance(payload, dict):
            return None
        if not _valid_best_manifest_metadata(payload):
            return None
        if "checkpoint" in payload:
            name = payload["checkpoint"]
            if (not isinstance(name, str) or not name
                    or Path(name).name != name or Path(name).suffix != ".pt"):
                return None
            candidate = best_dir / name
            return candidate if _contained_regular_checkpoint(best_dir, candidate) else None
    legacy = best_dir / "ckpt.pt"
    return legacy if _contained_regular_checkpoint(best_dir, legacy) else None


def _quarantine_future_best(best_dir: Path, checkpoint_step: int) -> Path | None:
    """Preserve, but stop advertising, a winner from an abandoned future branch."""
    info_path = best_dir / "best.json"
    if not info_path.is_file() or _best_checkpoint_path(best_dir) is None:
        return None
    try:
        best_step = int(json.loads(info_path.read_text())["step"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None
    if best_step <= checkpoint_step:
        return None
    return _quarantine_best(
        best_dir, f"future-step-{best_step}-resume-{checkpoint_step}")


def _save_checkpoint(path: Path, *, step: int, projector: nn.Module,
                     nextlat: nn.Module | None, engram: LexicalMemoryBank | None,
                     deep_vision: DeepVisionInjector | None,
                     grounding: ImageTextContrastiveHead | None,
                     wrappers: list[nn.Module], optimizer,
                     sampler: EpochBatchSampler, args: argparse.Namespace,
                     layer_vision: LayerMatchedVisionInjector | None = None,
                     vision_fusion: VisionFusionResidual | None = None) -> None:
    blob = {
        "schema": CHECKPOINT_SCHEMA,
        "step": step,
        "projector": _cpu_state(projector),
        "nextlat": _cpu_state(nextlat),
        "engram": _cpu_state(engram),
        "deep_vision": _cpu_state(deep_vision),
        "layer_vision": _cpu_state(layer_vision),
        "vision_fusion": _cpu_state(vision_fusion),
        "grounding": _cpu_state(grounding),
        "loops": loop_adapter_state(wrappers),
        "optimizer": optimizer.state_dict(),
        "sampler": sampler.state_dict(),
        "rng": {
            "python": random.getstate(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all(),
        },
        "args": vars(args),
    }
    temporary = path.with_suffix(".tmp")
    torch.save(blob, temporary)
    # Atomic rename protects against process interruption; the file and parent
    # directory fsyncs additionally guarantee that a successful return remains
    # recoverable after an abrupt host reset or power loss.
    _durable_replace(temporary, path)


def _resumed_last_checkpoint_step(resume_path: Path | None,
                                  checkpoint_path: Path, step: int, *,
                                  contract_changed: bool = False) -> int | None:
    """Return ``step`` only when resume loaded the run's advertised last file.

    An explicit resume may deliberately branch from an older checkpoint while
    a newer, stale ``last.pt`` remains in the output directory. File existence
    and checkpoint cadence cannot establish that the stale path represents the
    current in-memory state; inode identity can.
    """
    if resume_path is None or contract_changed:
        return None
    return int(step) if _same_checkpoint_file(resume_path, checkpoint_path) else None


def _final_checkpoint_required(step: int,
                               last_checkpoint_step: int | None) -> bool:
    """Whether the run's advertised checkpoint lacks the committed final state."""
    return last_checkpoint_step != int(step)


def _resume_checkpoint_publication_required(
        resume_path: Path | None, last_checkpoint_step: int | None) -> bool:
    """Whether a loaded resume branch/contract still needs to become ``last.pt``."""
    return resume_path is not None and last_checkpoint_step is None


def _same_checkpoint_file(left: Path | None, right: Path | None) -> bool:
    """Whether two existing paths identify the same durable checkpoint inode."""
    if left is None or right is None:
        return False
    try:
        return os.path.samefile(left, right)
    except OSError:
        return False


def _resume_requires_best_quarantine(
        resume_path: Path | None, checkpoint_path: Path,
        best_checkpoint_path: Path | None) -> bool:
    """Whether an explicit branch is unrelated to both active checkpoint heads."""
    return bool(
        resume_path is not None
        and not _same_checkpoint_file(resume_path, checkpoint_path)
        and not _same_checkpoint_file(resume_path, best_checkpoint_path))


def _budget_resume_differences(saved_args: dict,
                               args: argparse.Namespace) -> list[str]:
    """Return token-budget geometry changed by an exact resume request."""
    saved_batch = int(saved_args.get("batch", 8))
    defaults = {
        "batch": 8,
        "max_batch": saved_batch,
        "min_batch": 0,
        "target_batch_tokens": 0,
        "loop_token_budget_scale": 1.0,
    }
    return [name for name, default in defaults.items()
            if saved_args.get(name, default) != getattr(args, name, default)]


def _loop_lr_resume_difference(saved_args: dict,
                               args: argparse.Namespace) -> bool:
    """Reject an LR change unless this resume will really reset loop state."""
    reset_pending = bool(
        args.reset_loop_on_resume
        and not saved_args.get("loop_reset_committed", False))
    return bool(
        saved_args.get("loop_lr") != getattr(args, "loop_lr", None)
        and not reset_pending)


def _preserve_loop_reset_outcome(args: argparse.Namespace,
                                 committed: bool) -> None:
    """Carry a one-time reset receipt into every descendant checkpoint."""
    if committed:
        args.loop_reset_committed = True


def _resume_contract_changed(*, text_limit_migrated: bool,
                             budget_differences: Sequence[str]) -> bool:
    """Whether accepted resume settings must be republished in the checkpoint."""
    return bool(text_limit_migrated or budget_differences)


def _resume_invalidates_step_evaluation(*, text_limit_migrated: bool,
                                        unrelated_branch: bool,
                                        loop_reset_pending: bool) -> bool:
    """Whether prior same-step evaluation belongs to a different model contract."""
    return bool(text_limit_migrated or unrelated_branch or loop_reset_pending)


def _promote_checkpoint(source: Path, best_dir: Path, *, step: int,
                        loss: float) -> None:
    """Publish an already-saved checkpoint as one consistent eval winner.

    `last.pt` is atomically replaced on later saves, so a hard link preserves
    this exact inode without serializing hundreds of MB a second time.  The
    atomic manifest points at an immutable file; a crash can therefore expose
    either the complete old winner or the complete new winner, never new model
    bytes under old metrics.  ``ckpt.pt`` remains a best-effort legacy alias.
    """
    best_dir.mkdir(parents=True, exist_ok=True)
    destination = best_dir / f"ckpt_step_{int(step):08d}.pt"
    temporary = best_dir / f".{destination.name}.tmp"
    temporary.unlink(missing_ok=True)
    os.link(source, temporary)
    os.replace(temporary, destination)
    _fsync_directory(best_dir)
    _atomic_json(best_dir / "best.json", {
        "step": int(step), "loss": float(loss),
        "ppl": math.exp(min(float(loss), 20.0)),
        "checkpoint": destination.name,
    }, durable=True)
    alias = best_dir / "ckpt.pt"
    alias_temporary = best_dir / ".ckpt.pt.tmp"
    alias_temporary.unlink(missing_ok=True)
    os.link(destination, alias_temporary)
    os.replace(alias_temporary, alias)
    # Successful publication makes abandoned pre-manifest files and older
    # immutable winners unreachable.  Keep only the manifest target and alias.
    for candidate in best_dir.glob("ckpt_step_*.pt"):
        if candidate != destination:
            candidate.unlink(missing_ok=True)


def _load_checkpoint(path: Path, *, projector: nn.Module, nextlat: nn.Module | None,
                     engram: LexicalMemoryBank | None, wrappers: list[nn.Module], optimizer,
                     sampler: EpochBatchSampler,
                     args: argparse.Namespace,
                     deep_vision: DeepVisionInjector | None = None,
                     layer_vision: LayerMatchedVisionInjector | None = None,
                     vision_fusion: VisionFusionResidual | None = None,
                     grounding: ImageTextContrastiveHead | None = None,
                     ) -> tuple[int, bool, bool, bool]:
    blob = torch.load(path, map_location="cpu", weights_only=False)
    if not valid_torch_archive_storages(path, blob):
        raise ValueError(f"checkpoint archive integrity check failed: {path}")
    if int(blob.get("schema", -1)) != CHECKPOINT_SCHEMA:
        raise ValueError(f"unsupported vision checkpoint schema {blob.get('schema')}")
    saved_args = blob.get("args", {})
    saved_limit = int(saved_args.get("max_text_tokens", 0))
    migrating_text_limit = bool(
        args.allow_text_limit_increase_from
        and saved_limit == args.allow_text_limit_increase_from
        and args.max_text_tokens > saved_limit
        and saved_args.get("data_fingerprint")
        == getattr(args, "previous_data_fingerprint", None)
    )
    migration_compatible = ({"data_fingerprint", "max_text_tokens"}
                            if migrating_text_limit else set())
    compatibility = ("data_fingerprint", "rwkv_fingerprint", "moonvit_fingerprint",
                     "rwkv", "moonvit", "prefix_tokens",
                     "max_text_tokens", "max_input_patches", "loop_count",
                     "loop_index", "loop_gate_cap", "loop_start_step",
                     "loop_ramp_steps",
                     "lr", "weight_decay", "grad_clip", "nextlat_weight",
                     "nextlat_hidden", "nextlat_kl_weight", "val_fraction",
                     "eval_every", "eval_examples")
    differences = [name for name in compatibility
                   if name not in migration_compatible
                   if saved_args.get(name) != getattr(args, name, None)]
    if _loop_lr_resume_difference(saved_args, args):
        differences.append("loop_lr")
    saved_image_fingerprint = saved_args.get("image_metadata_fingerprint")
    if (saved_image_fingerprint is not None and
            saved_image_fingerprint != getattr(args, "image_metadata_fingerprint", None)):
        differences.append("image_metadata_fingerprint")
    saved_fused_ce = saved_args.get("fused_ce_enabled")
    if saved_fused_ce is not None and bool(saved_fused_ce) != bool(HAS_FUSED_CE):
        differences.append("fused_ce_enabled")
    budget_differences = _budget_resume_differences(saved_args, args)
    if budget_differences and not args.allow_batch_resize:
        differences.extend(budget_differences)
    if differences:
        raise ValueError(f"checkpoint is incompatible with current settings: {differences}")
    engram_compatibility = ("engram_sites", "engram_drow", "engram_rows",
                            "engram_boundary_id", "engram_lr", "engram_warmup_steps")
    if bool(saved_args.get("engram", False)) != bool(args.engram):
        raise ValueError("checkpoint Engram enablement does not match this run")
    if args.engram:
        engram_differences = [name for name in engram_compatibility
                              if saved_args.get(name) != getattr(args, name)]
        if engram_differences:
            raise ValueError(f"checkpoint Engram configuration differs: {engram_differences}")
    new_contract_defaults = {
        "vision_resampler_layers": 0, "vision_resampler_width": 1024,
        "vision_resampler_heads": 8, "deep_vision_layers": "",
        "deep_vision_rank": 256, "moonvit_tap_layers": "",
        "layer_vision_layers": "", "layer_vision_rank": 256,
        "vision_view_mode": "full", "sandwich_prompt": False,
        "vision_fusion": False, "vision_fusion_rank": 512,
        "vision_fusion_fingerprint": "",
        "grounding_contrastive_weight": 0.0,
        "grounding_contrastive_dim": 512, "grounding_temperature": 0.07,
        "grounding_early_tokens": 0, "grounding_early_weight": 1.0,
    }
    new_differences = [
        name for name, default in new_contract_defaults.items()
        if saved_args.get(name, default) != getattr(args, name, default)
    ]
    if new_differences:
        raise ValueError(
            f"checkpoint grounding/vision configuration differs: {new_differences}")
    projector.load_state_dict(blob["projector"])
    if (nextlat is None) != (blob.get("nextlat") is None):
        raise ValueError("checkpoint NextLat configuration does not match this run")
    if nextlat is not None:
        nextlat.load_state_dict(blob["nextlat"])
    if (engram is None) != (blob.get("engram") is None):
        raise ValueError("checkpoint Engram state does not match this run")
    if engram is not None:
        engram.load_state_dict(blob["engram"])
    if (deep_vision is None) != (blob.get("deep_vision") is None):
        raise ValueError("checkpoint deep-vision configuration does not match this run")
    if deep_vision is not None:
        deep_vision.load_state_dict(blob["deep_vision"])
    if (layer_vision is None) != (blob.get("layer_vision") is None):
        raise ValueError("checkpoint layer-vision configuration does not match this run")
    if layer_vision is not None:
        layer_vision.load_state_dict(blob["layer_vision"])
    if (vision_fusion is None) != (blob.get("vision_fusion") is None):
        raise ValueError("checkpoint vision-fusion configuration does not match this run")
    if vision_fusion is not None:
        vision_fusion.load_state_dict(blob["vision_fusion"])
    if (grounding is None) != (blob.get("grounding") is None):
        raise ValueError("checkpoint grounding-head configuration does not match this run")
    if grounding is not None:
        grounding.load_state_dict(blob["grounding"])
    load_loop_adapter_state(wrappers, blob["loops"])
    optimizer.load_state_dict(blob["optimizer"])
    sampler.load_state_dict(blob["sampler"])
    random.setstate(blob["rng"]["python"])
    torch.set_rng_state(blob["rng"]["torch"])
    torch.cuda.set_rng_state_all(blob["rng"]["cuda"])
    contract_changed = _resume_contract_changed(
        text_limit_migrated=migrating_text_limit,
        budget_differences=budget_differences)
    return (int(blob["step"]), migrating_text_limit,
            bool(saved_args.get("loop_reset_committed", False)),
            contract_changed)


def _initialize_adapters(path: Path, *, projector: nn.Module,
                         nextlat: nn.Module | None, engram: nn.Module | None,
                         wrappers: list[nn.Module], args: argparse.Namespace,
                         deep_vision: DeepVisionInjector | None = None,
                         layer_vision: LayerMatchedVisionInjector | None = None,
                         vision_fusion: VisionFusionResidual | None = None,
                         grounding: ImageTextContrastiveHead | None = None) -> int:
    """Warm-start trainable vision adapters without inheriting run state.

    Unlike exact resume, this intentionally accepts a different dataset and
    sampler. Optimizer moments, RNG, step, and logs start fresh for the new
    phase, while every compatible learned adapter is retained. A destination
    may explicitly introduce a fresh Engram when the source predates Engram,
    but an existing source Engram is never silently discarded.
    """
    blob = torch.load(path, map_location="cpu", weights_only=False)
    if not valid_torch_archive_storages(path, blob):
        raise ValueError(f"checkpoint archive integrity check failed: {path}")
    if int(blob.get("schema", -1)) != CHECKPOINT_SCHEMA:
        raise ValueError(f"unsupported vision checkpoint schema {blob.get('schema')}")
    saved = blob.get("args", {})
    structural = ("rwkv_fingerprint", "moonvit_fingerprint",
                  "max_input_patches", "nextlat_hidden", "loop_count",
                  "loop_index", "loop_gate_cap")
    differences = [name for name in structural
                   if saved.get(name) != getattr(args, name, None)]
    if differences:
        raise ValueError(f"adapter initialization is incompatible: {differences}")
    if (nextlat is None) != (blob.get("nextlat") is None):
        raise ValueError("adapter initialization NextLat configuration does not match")
    source_engram = blob.get("engram")
    saved_engram_enabled = bool(saved.get("engram", source_engram is not None))
    if saved_engram_enabled != (source_engram is not None):
        raise ValueError("adapter initialization checkpoint has inconsistent Engram state")
    if source_engram is not None and engram is None:
        raise ValueError("adapter initialization would discard the source Engram")
    if source_engram is not None:
        engram_structural = (
            "engram_sites", "engram_drow", "engram_rows", "engram_boundary_id"
        )
        engram_differences = [
            name for name in engram_structural
            if saved.get(name) != getattr(args, name, None)
        ]
        if engram_differences:
            raise ValueError(
                "adapter initialization Engram is incompatible: "
                f"{engram_differences}"
            )
    source_resampler_layers = int(saved.get("vision_resampler_layers", 0))
    destination_resampler_layers = int(getattr(args, "vision_resampler_layers", 0))
    if source_resampler_layers and source_resampler_layers != destination_resampler_layers:
        raise ValueError("adapter initialization resampler geometry is incompatible")
    projector_state = dict(blob["projector"])
    source_prefix = int(saved.get("prefix_tokens", 0))
    destination_prefix = int(getattr(args, "prefix_tokens", 0))
    if source_prefix != destination_prefix:
        if source_prefix < 1 or destination_prefix < 1:
            raise ValueError("adapter initialization has an invalid prefix geometry")
        for name in ("position", "resampler.queries"):
            value = projector_state.get(name)
            if value is None:
                continue
            if value.ndim != 3 or value.shape[0] != 1 or value.shape[1] != source_prefix:
                raise ValueError(f"cannot resize malformed projector tensor {name}")
            resized = F.interpolate(
                value.float().transpose(1, 2), size=destination_prefix,
                mode="linear", align_corners=False).transpose(1, 2)
            projector_state[name] = resized.to(value.dtype)
    projector_info = projector.load_state_dict(
        projector_state, strict=not (
            source_resampler_layers == 0 and destination_resampler_layers > 0))
    if not source_resampler_layers and destination_resampler_layers > 0:
        invalid_missing = [key for key in projector_info.missing_keys
                           if not key.startswith("resampler.")]
        if invalid_missing or projector_info.unexpected_keys:
            raise ValueError(
                "adapter initialization projector migration has unexpected keys: "
                f"missing={invalid_missing[:3]} "
                f"unexpected={projector_info.unexpected_keys[:3]}")
    if nextlat is not None:
        nextlat.load_state_dict(blob["nextlat"])
    if source_engram is not None:
        assert engram is not None
        engram.load_state_dict(source_engram)
    source_deep = blob.get("deep_vision")
    if source_deep is not None and deep_vision is None:
        raise ValueError("adapter initialization would discard deep vision")
    if source_deep is not None:
        deep_vision.load_state_dict(source_deep)
    source_layer = blob.get("layer_vision")
    if source_layer is not None and layer_vision is None:
        raise ValueError("adapter initialization would discard layer-matched vision")
    if source_layer is not None:
        layer_vision.load_state_dict(source_layer)
    source_fusion = blob.get("vision_fusion")
    if source_fusion is not None and vision_fusion is None:
        raise ValueError("adapter initialization would discard three-tower fusion")
    if source_fusion is not None:
        if (saved.get("vision_fusion_fingerprint") !=
                getattr(args, "vision_fusion_fingerprint", None)):
            raise ValueError("adapter initialization fusion towers do not match")
        vision_fusion.load_state_dict(source_fusion)
    source_grounding = blob.get("grounding")
    if source_grounding is not None and grounding is None:
        raise ValueError("adapter initialization would discard grounding head")
    if source_grounding is not None:
        grounding.load_state_dict(source_grounding)
    load_loop_adapter_state(wrappers, blob["loops"])
    return int(blob.get("step", 0))


def _optimizer(projector: nn.Module, nextlat: nn.Module | None,
               engram: LexicalMemoryBank | None, wrappers: list[nn.Module],
               args: argparse.Namespace, *,
               deep_vision: DeepVisionInjector | None = None,
               layer_vision: LayerMatchedVisionInjector | None = None,
               vision_fusion: VisionFusionResidual | None = None,
               grounding: ImageTextContrastiveHead | None = None):
    loop_gate, loop_norm = [], []
    for wrapper in wrappers:
        gate_names = wrapper.loop.loop_param_names()
        for name, parameter in wrapper.loop.named_parameters():
            if not parameter.requires_grad or name.startswith("core."):
                continue
            (loop_gate if name in gate_names else loop_norm).append(parameter)
    groups = [
        {"params": list(projector.parameters()), "lr": args.lr, "weight_decay": args.weight_decay,
         "name": "projector"},
        {"params": loop_gate, "lr": args.loop_lr, "weight_decay": 0.0, "name": "loop_gates"},
        {"params": loop_norm, "lr": args.lr, "weight_decay": 0.0, "name": "loop_norms"},
    ]
    if nextlat is not None:
        groups.append({"params": list(nextlat.parameters()), "lr": args.lr,
                       "weight_decay": args.weight_decay, "name": "nextlat"})
    if engram is not None:
        groups.append({"params": engram_parameters(engram), "lr": args.engram_lr,
                       "weight_decay": 0.0, "name": "engram"})
    if deep_vision is not None:
        groups.append({"params": list(deep_vision.parameters()), "lr": args.lr,
                       "weight_decay": args.weight_decay, "name": "deep_vision"})
    if layer_vision is not None:
        groups.append({"params": list(layer_vision.parameters()), "lr": args.lr,
                       "weight_decay": args.weight_decay, "name": "layer_vision"})
    if vision_fusion is not None:
        groups.append({"params": list(vision_fusion.parameters()), "lr": args.lr,
                       "weight_decay": args.weight_decay, "name": "vision_fusion"})
    if grounding is not None:
        groups.append({"params": list(grounding.parameters()), "lr": args.lr,
                       "weight_decay": args.weight_decay, "name": "grounding"})
    groups = [group for group in groups if group["params"]]
    kwargs = dict(betas=(0.9, 0.95))
    if torch.cuda.is_available():
        kwargs["fused"] = True
    return torch.optim.AdamW(groups, **kwargs), [p for group in groups for p in group["params"]]


def _reset_loop_optimizer_state(optimizer, wrappers: Sequence[nn.Module], args) -> None:
    """Discard moments from an unsafe loop activation, leaving bridge state intact."""
    loop_parameters = {parameter for wrapper in wrappers
                       for name, parameter in wrapper.loop.named_parameters()
                       if parameter.requires_grad and not name.startswith("core.")}
    for parameter in loop_parameters:
        optimizer.state.pop(parameter, None)
    for group in optimizer.param_groups:
        if group.get("name") == "loop_gates":
            group["lr"] = args.loop_lr
        elif group.get("name") == "loop_norms":
            group["lr"] = args.lr


def _loop_runtime_scale(next_step: int, *, start_step: int, ramp_steps: int) -> float:
    if next_step < start_step:
        return 0.0
    if ramp_steps <= 0:
        return 1.0
    return min(1.0, (next_step - start_step + 1) / ramp_steps)


def _require_finite_metric(name: str, value: float | torch.Tensor) -> float:
    """Convert a scalar metric while refusing invalid JSON/checkpoint state."""
    rendered = float(value)
    if not math.isfinite(rendered):
        raise FloatingPointError(f"non-finite {name}: {rendered}")
    return rendered


def _parse_engram_sites(spec: str, n_layers: int) -> list[int]:
    try:
        sites = sorted({int(value.strip()) for value in spec.split(",") if value.strip()})
    except ValueError as exc:
        raise ValueError(f"invalid --engram-sites {spec!r}") from exc
    if not sites:
        raise ValueError("--engram-sites must name at least one layer")
    invalid = [site for site in sites if not 0 <= site < n_layers]
    if invalid:
        raise ValueError(f"Engram layers out of range for {n_layers} layers: {invalid}")
    return sites


def _engram_metrics(engram: LexicalMemoryBank | None) -> dict[str, float | bool]:
    if engram is None:
        return {"engram_enabled": False}
    h_rms = [site.last_inj_h_rms for site in engram.sites.values()
             if site.last_inj_h_rms is not None]
    v_rms = [site.last_inj_v_rms for site in engram.sites.values()
             if site.last_inj_v_rms is not None]
    gates = [value for site in engram.sites.values()
             for value in (site.last_gate_h_mean, site.last_gate_v_mean)
             if value is not None]
    rms = h_rms + v_rms
    anchor = next(engram.parameters()).new_zeros((), dtype=torch.float32)
    inj_rms = torch.stack(rms).square().mean().sqrt() if rms else anchor
    gate_mean = torch.stack(gates).mean() if gates else anchor
    values = [inj_rms, gate_mean]
    recall_names: list[str] = []
    if engram.last_recall is not None:
        rr = engram.last_recall
        valid = rr.valid.float()
        count = valid.sum()
        denominator = count.clamp_min(1)
        valid_rate = count / valid.numel()
        beyond = ((rr.dist > 32).float() * valid).sum() / denominator
        maximum = (rr.mlen * rr.valid).max().float()
        # Histogram selection keeps the tensor shape static. Boolean indexing a
        # CUDA mask produces a dynamic-size tensor and forces a hidden host sync
        # before median/quantile can launch.
        width = rr.mlen.shape[1]
        histogram = torch.bincount(
            rr.mlen.clamp(0, width).reshape(-1),
            weights=valid.reshape(-1), minlength=width + 1)
        median_rank = torch.div(count.long() + 1, 2, rounding_mode="floor")
        median = torch.searchsorted(histogram.cumsum(0), median_rank).float()
        median = median * (count > 0)
        values.extend((valid_rate, beyond, median, maximum))
        recall_names = ["valid_rate", "frac_beyond_32", "mlen_p50", "mlen_max"]
    rendered = torch.stack([value.float() for value in values]).tolist()
    result: dict[str, float | bool] = {
        "engram_enabled": True,
        "engram_inj_rms": rendered[0],
        "engram_gate_mean": rendered[1],
    }
    for name, value in zip(recall_names, rendered[2:]):
        result[f"engram_recall_{name}"] = value
        engram.recall_stats[name] = value
    return result


def assert_training_contract(rwkv: nn.Module, vision: nn.Module,
                             wrappers: Sequence[nn.Module], trainable: Sequence[nn.Parameter]) -> None:
    """Fail at startup, not hundreds of steps later, on freeze/device mistakes."""
    trainable_ids = [id(parameter) for parameter in trainable]
    if len(trainable_ids) != len(set(trainable_ids)):
        raise RuntimeError("optimizer contains a trainable parameter more than once")
    allowed_rwkv = {
        id(parameter)
        for wrapper in wrappers
        for name, parameter in wrapper.loop.named_parameters()
        if not name.startswith("core.")
    }
    attached_engram = getattr(rwkv, "engram", None)
    if isinstance(attached_engram, LexicalMemoryBank):
        allowed_rwkv.update(id(parameter) for parameter in attached_engram.parameters())
    leaked = [name for name, parameter in rwkv.named_parameters()
              if parameter.requires_grad and id(parameter) not in allowed_rwkv]
    if leaked:
        raise RuntimeError(f"unexpected trainable RWKV parameters: {leaked[:5]}")
    missing = [name for name, parameter in rwkv.named_parameters()
               if parameter.requires_grad and id(parameter) not in trainable_ids]
    if missing:
        raise RuntimeError(f"RWKV adapters missing from optimizer: {missing[:5]}")
    if any(parameter.requires_grad for parameter in vision.parameters()):
        raise RuntimeError("frozen MoonViT has trainable parameters")
    off_device = [tuple(parameter.shape) for parameter in trainable if parameter.device.type != "cuda"]
    if off_device:
        raise RuntimeError(f"trainable parameters are not on CUDA: {off_device[:5]}")
    for layer, wrapper in enumerate(wrappers):
        if any(parameter.requires_grad for parameter in wrapper.inner.parameters()):
            raise RuntimeError(f"frozen RWKV TimeMix core is trainable at layer {layer}")


@torch.no_grad()
def evaluate(rows: Sequence[dict], indices: Sequence[int], *, rwkv: nn.Module,
             projector: nn.Module, vision: MoonViT,
             engram: LexicalMemoryBank | None, cache_dir: Path | None,
             batch_size: int, max_examples: int,
             deep_vision: DeepVisionInjector | None = None,
             layer_vision: LayerMatchedVisionInjector | None = None,
             fusion_tower: AlignedFrozenVisionFeatures | None = None,
             fusion_adapter: VisionFusionResidual | None = None,
             fusion_cache_dir: Path | None = None,
             progress: Callable[[int, int], None] | None = None) -> float:
    if not indices:
        return float("nan")
    losses, weights = [], []
    total = min(len(indices), max_examples)
    for start in range(0, total, batch_size):
        chosen = indices[start:start + batch_size]
        batch_rows = [rows[i] for i in chosen]
        ids, labels, mask = make_batch(batch_rows, device="cuda")
        positions = supervised_positions(
            batch_rows, projector.prefix_tokens, device="cuda")
        features = cached_features(batch_rows, vision, projector, cache_dir)
        fusion_features = (cached_fusion_features(
            batch_rows, fusion_tower, projector.prefix_tokens, fusion_cache_dir)
            if fusion_tower is not None and fusion_adapter is not None
            and fusion_cache_dir is not None else None)
        starts = visual_insert_positions(batch_rows)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss, _ = multimodal_loss(rwkv, projector, vision, (), ids, labels, mask,
                                      engram=engram, features=features,
                                      selected_positions=positions,
                                      deep_vision=deep_vision,
                                      layer_vision=layer_vision,
                                      visual_starts=starts,
                                      fusion_adapter=fusion_adapter,
                                      fusion_features=fusion_features)
        weight = int((labels != -100).sum())
        losses.append(float(loss) * weight)
        weights.append(weight)
        if progress is not None:
            progress(min(start + len(chosen), total), total)
    return sum(losses) / max(1, sum(weights))


def select_eval_sample_indices(rows: Sequence[dict], indices: Sequence[int],
                               count: int) -> list[int]:
    """Choose a stable, source-stratified spread for qualitative eval."""
    if count <= 0 or not indices:
        return []
    groups: dict[str, list[int]] = {}
    for index in indices:
        row = rows[index]
        source = str(row.get("stage1_source") or row.get("source") or "unknown")
        groups.setdefault(source, []).append(index)
    names = sorted(groups)
    n = min(count, len(indices))
    quotas = {name: n // len(names) for name in names}
    for name in names[:n % len(names)]:
        quotas[name] += 1

    selected_by_source: dict[str, list[int]] = {}
    for name in names:
        values, quota = groups[name], min(quotas[name], len(groups[name]))
        if quota == 0:
            selected_by_source[name] = []
        elif quota == 1:
            selected_by_source[name] = [values[len(values) // 2]]
        else:
            selected_by_source[name] = [
                values[(i * (len(values) - 1)) // (quota - 1)] for i in range(quota)
            ]

    # A tiny source can leave unused quota. Fill deterministically from any
    # remaining rows, then round-robin sources so the rendered card order is mixed.
    chosen = {index for values in selected_by_source.values() for index in values}
    for index in indices:
        if len(chosen) >= n:
            break
        if index not in chosen:
            row = rows[index]
            source = str(row.get("stage1_source") or row.get("source") or "unknown")
            selected_by_source[source].append(index)
            chosen.add(index)
    output: list[int] = []
    offset = 0
    while len(output) < n:
        for name in names:
            values = selected_by_source[name]
            if offset < len(values):
                output.append(values[offset])
        offset += 1
    return output[:n]


def filter_eval_sample_indices(rows: Sequence[dict], indices: Sequence[int],
                               excluded_sources: Sequence[str]) -> list[int]:
    """Exclude qualitative sources without changing the scalar eval contract.

    Filters are case-insensitive source-name fragments.  This lets an operator
    keep a broad held-out set for loss/PPL while ensuring the dashboard gallery
    is drawn only from sources appropriate for human review.
    """
    denied = tuple(value.strip().casefold() for value in excluded_sources
                   if value.strip())
    if not denied:
        return list(indices)
    output = []
    for index in indices:
        row = rows[index]
        source = str(row.get("stage1_source") or row.get("source") or "unknown")
        if not any(fragment in source.casefold() for fragment in denied):
            output.append(index)
    return output


@torch.no_grad()
def write_eval_samples(rows: Sequence[dict], indices: Sequence[int], *, step: int,
                       ppl: float, rwkv: nn.Module, projector: nn.Module,
                       vision: MoonViT, engram: LexicalMemoryBank | None,
                       cache_dir: Path | None, vocab: WorldVocab, prompt: str,
                       out: Path, count: int, max_new: int,
                       deep_vision: DeepVisionInjector | None = None,
                       layer_vision: LayerMatchedVisionInjector | None = None,
                       sandwich_prompt: bool = False,
                       fusion_tower: AlignedFrozenVisionFeatures | None = None,
                       fusion_adapter: VisionFusionResidual | None = None,
                       fusion_cache_dir: Path | None = None,
                       progress: Callable[[int, int], None] | None = None) -> Path | None:
    """Greedily caption a fixed spread of held-out images for chart drill-down."""
    if count <= 0 or max_new <= 0 or not indices:
        return None
    chosen = select_eval_sample_indices(rows, indices, count)
    sample_rows = [rows[index] for index in chosen]
    prompt_ids = [vocab.encode(str(row.get("prompt") or prompt)) for row in sample_rows]
    generated: list[list[int]] = [[] for _ in sample_rows]
    stopped = [False] * len(sample_rows)
    artifact = out / "eval_samples" / f"step_{step:08d}.json"
    artifact.parent.mkdir(parents=True, exist_ok=True)

    def persist(*, complete: bool, generation_steps: int) -> None:
        _atomic_json(artifact, {
            "step": step, "ppl": ppl, "decoding": "greedy", "max_new": max_new,
            "complete": complete, "generation_steps": generation_steps,
            "items": [{
                "image": str(row["image"].resolve()),
                "prompt": str(row.get("prompt") or prompt),
                "reference": row["text"],
                "caption": vocab.decode(tokens).strip(),
                "tokens": len(tokens), "stopped_at_eod": stopped[i],
                "source": str(row.get("stage1_source") or row.get("source") or "unknown"),
            } for i, (row, tokens) in enumerate(zip(sample_rows, generated))],
        }, durable=complete)

    # Publish the image/reference skeleton before the expensive autoregressive
    # phase. The dashboard can render and poll it while captions are still being
    # produced, and a hard stop cannot erase the already-computed scalar eval.
    persist(complete=False, generation_steps=0)
    generation_steps = 0
    try:
        features = cached_features(sample_rows, vision, projector, cache_dir)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            prefix = projector(features)
            if fusion_tower is not None and fusion_adapter is not None:
                if fusion_cache_dir is None:
                    raise ValueError("fusion eval requires its feature cache")
                fusion_features = cached_fusion_features(
                    sample_rows, fusion_tower, projector.prefix_tokens,
                    fusion_cache_dir)
                prefix = prefix + fusion_adapter(fusion_features).to(prefix.dtype)
        for generation_step in range(max_new):
            sequences = [((row_prompt * (2 if sandwich_prompt else 1)) + item)
                         for row_prompt, item in zip(prompt_ids, generated)]
            starts = tuple(len(row_prompt) if sandwich_prompt else 0
                           for row_prompt in prompt_ids)
            lengths = [len(item) for item in sequences]
            width = max(lengths)
            ids = torch.zeros(len(sequences), width, dtype=torch.long, device="cuda")
            text_mask = torch.zeros_like(ids, dtype=torch.bool)
            for i, sequence in enumerate(sequences):
                ids[i, :len(sequence)] = torch.tensor(sequence, dtype=torch.long, device="cuda")
                text_mask[i, :len(sequence)] = True
            with torch.autocast("cuda", dtype=torch.bfloat16):
                text = rwkv.model.embeddings(ids)
                typed_prefix = prefix.to(text.dtype)
                embeds = insert_visual_span(text, typed_prefix, starts)
                attention_mask = insert_visual_span(
                    text_mask, torch.ones(
                        len(sequences), prefix.shape[1], dtype=torch.bool,
                        device="cuda"), starts)
                if engram is not None:
                    boundary = 0 if engram.boundary_id is None else int(engram.boundary_id)
                    engram.set_input_ids(insert_boundary_ids(
                        ids, starts, prefix.shape[1], boundary))
                with contextlib.ExitStack() as stack:
                    if deep_vision is not None:
                        stack.enter_context(deep_vision.use_prefix(typed_prefix, starts))
                    if layer_vision is not None:
                        stack.enter_context(layer_vision.use_features(
                            torch.stack(features), starts))
                    hidden = rwkv.model(
                        inputs_embeds=embeds, attention_mask=attention_mask,
                        output_hidden_states=False, use_cache=False,
                        return_dict=True).last_hidden_state
                positions = torch.tensor(
                    [prefix.shape[1] + length - 1 for length in lengths],
                    dtype=torch.long, device="cuda")
                logits = rwkv.lm_head(
                    hidden[torch.arange(len(sequences), device="cuda"), positions]).float()
                if engram is not None:
                    logits = engram.logit_bias_at(
                        logits, torch.arange(len(sequences), device="cuda"), positions,
                        inplace=True)
                logits[:, 0] = -torch.inf
                next_tokens = logits.argmax(-1).tolist()
            for i, token in enumerate(next_tokens):
                if stopped[i]:
                    continue
                if token == SEP:
                    stopped[i] = True
                else:
                    generated[i].append(int(token))
            generation_steps = generation_step + 1
            if generation_steps % 8 == 0 or all(stopped):
                persist(complete=False, generation_steps=generation_steps)
                if progress is not None:
                    progress(generation_steps, max_new)
            if all(stopped):
                break
    except BaseException:
        persist(complete=False, generation_steps=generation_steps)
        raise
    persist(complete=True, generation_steps=generation_steps)
    return artifact


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--eval-data", nargs="+", default=None,
                    help="held-out image/text manifests; never sampled for training")
    ap.add_argument("--rwkv", default="models/rwkv7-g1h-2.9b-20260710-ctx10240.pth")
    ap.add_argument("--moonvit", default="models/kimi-k2.6-moonvit/model-00064-of-000064.safetensors")
    ap.add_argument("--out", default="runs/moonvit_rwkv_stage1_v3")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--min-batch", type=int, default=0,
                    help="minimum captions for token-budget batches; 0 uses --batch")
    ap.add_argument("--max-batch", type=int, default=8,
                    help="maximum captions when token-budget batching is enabled")
    ap.add_argument("--target-batch-tokens", type=int, default=0,
                    help="padded image-prefix + text tokens per step; 0 keeps fixed --batch")
    ap.add_argument("--loop-token-budget-scale", type=float, default=1.0,
                    help="multiply token budget after factored loops become active")
    ap.add_argument("--allow-batch-resize", action="store_true",
                    help="explicitly resume while changing token-budget batch geometry")
    ap.add_argument("--steps", type=int, default=16000)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--loop-lr", type=float, default=1e-5)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--prefix-tokens", type=int, default=64)
    ap.add_argument("--vision-resampler-layers", type=int, default=0,
                    help="learned-query residual resampler blocks; 0 disables")
    ap.add_argument("--vision-resampler-width", type=int, default=1024)
    ap.add_argument("--vision-resampler-heads", type=int, default=8)
    ap.add_argument("--deep-vision-layers", default="",
                    help="decoder layers receiving zero-init visual reinjection")
    ap.add_argument("--deep-vision-rank", type=int, default=256)
    ap.add_argument("--moonvit-tap-layers", default="",
                    help="MoonViT blocks retained in staged feature caches")
    ap.add_argument("--layer-vision-layers", default="",
                    help="RWKV layers receiving the corresponding MoonViT tap")
    ap.add_argument("--layer-vision-rank", type=int, default=256)
    ap.add_argument("--vision-view-mode", choices=("full", "full-quadrants"),
                    default="full")
    ap.add_argument("--vision-fusion", action=argparse.BooleanOptionalAction,
                    default=False,
                    help="add frozen SigLIP2+DINOv2+SAM aligned residual features")
    ap.add_argument("--vision-fusion-rank", type=int, default=512)
    ap.add_argument("--siglip2-model",
                    default="models/vision/siglip2-so400m-patch16-512")
    ap.add_argument("--siglip2-width", type=int, default=1152)
    ap.add_argument("--dinov2-model", default="models/vision/dinov2-base")
    ap.add_argument("--sam-model", default="models/vision/sam-vit-base")
    ap.add_argument("--fusion-feature-cache",
                    default="caches/siglip2_dinov2_sam_aligned_v1")
    ap.add_argument("--grounding-early-tokens", type=int, default=0,
                    help="caption-opening targets receiving extra CE weight")
    ap.add_argument("--grounding-early-weight", type=float, default=1.0)
    ap.add_argument("--grounding-contrastive-weight", type=float, default=0.0,
                    help="in-batch image/text contrastive auxiliary weight")
    ap.add_argument("--grounding-contrastive-dim", type=int, default=512)
    ap.add_argument("--grounding-temperature", type=float, default=0.07)
    ap.add_argument("--prompt", default="Describe this image:\n")
    ap.add_argument("--sandwich-prompt", action=argparse.BooleanOptionalAction,
                    default=False,
                    help="train prompt -> image -> repeated prompt -> caption")
    ap.add_argument("--max-text-tokens", type=int, default=384)
    ap.add_argument("--allow-text-limit-increase-from", type=int, default=0,
                    help="resume a checkpoint made at this smaller text limit; verifies its old fingerprint")
    ap.add_argument("--max-input-patches", type=int, default=1024)
    ap.add_argument("--feature-cache", default="caches/moonvit_features_stage1_v3")
    ap.add_argument("--preload-feature-cache", action="store_true",
                    help="keep deserialized cached MoonViT features in system RAM")
    ap.add_argument("--background-feature-preload", action="store_true",
                    help="warm the RAM cache asynchronously while training starts")
    ap.add_argument("--manifest-stat-workers", type=int, default=1,
                    help="parallel source-file checks during manifest loading")
    ap.add_argument("--prefetch-next-batch", action=argparse.BooleanOptionalAction, default=True,
                    help="CPU-load the exact next batch while the GPU trains the current one")
    ap.add_argument("--loop-count", type=int, default=2)
    ap.add_argument("--loop-start-step", type=int, default=250,
                    help="warm up the bridge before enabling per-layer TimeMix refinement")
    ap.add_argument("--loop-ramp-steps", type=int, default=1000,
                    help="linearly ramp effective loop gates after activation; 0 disables")
    ap.add_argument("--loop-gate-cap", type=float, default=0.25)
    ap.add_argument("--loop-index", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--reset-loop-on-resume", action="store_true",
                    help="one-time recovery: zero loop adapters and their optimizer moments")
    ap.add_argument("--nextlat-weight", type=float, default=0.1)
    ap.add_argument("--nextlat-hidden", type=int, default=1024)
    ap.add_argument("--nextlat-kl-weight", type=float, default=0.0)
    ap.add_argument("--engram", action="store_true",
                    help="train a lexical Engram memory while leaving RWKV frozen")
    ap.add_argument("--engram-sites", default="3,15",
                    help="comma-separated zero-based RWKV layer indices")
    ap.add_argument("--engram-drow", type=int, default=128,
                    help="width of each learned Engram table row")
    ap.add_argument("--engram-rows", type=int, default=65536,
                    help="hashed rows per Engram view (capped at tokenizer vocabulary)")
    ap.add_argument("--engram-lr", type=float, default=1e-3,
                    help="Engram-only AdamW learning rate; Engram receives no weight decay")
    ap.add_argument("--engram-warmup-steps", type=int, default=1000,
                    help="linear 0-to-1 ramp for Engram residual injection")
    ap.add_argument("--engram-boundary-id", type=int, default=0,
                    help="token ID separating the image prefix from caption recall")
    ap.add_argument("--val-fraction", type=float, default=0.02)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--eval-examples", type=int, default=64)
    ap.add_argument("--eval-samples", type=int, default=4,
                    help="held-out images greedily captioned at each eval marker")
    ap.add_argument("--eval-sample-exclude-sources", default="",
                    help="comma-separated, case-insensitive source fragments excluded only from the qualitative gallery")
    ap.add_argument("--eval-sample-max-new", type=int, default=64)
    ap.add_argument("--checkpoint-every", type=int, default=50)
    ap.add_argument("--log-every", type=int, default=1)
    ap.add_argument("--profile-steps", type=int, default=10)
    ap.add_argument("--operator-profile-steps", type=int, default=0,
                    help="capture PyTorch CPU/CUDA operator profiles through this step")
    ap.add_argument("--require-fused-ce", action=argparse.BooleanOptionalAction, default=False,
                    help="fail before loading models if flash-attn Triton CE is unavailable")
    ap.add_argument("--seed", type=int, default=20260714)
    ap.add_argument("--resume", default="auto", help="auto, none, or a checkpoint path")
    ap.add_argument("--init-adapters-from", default=None,
                    help="warm-start adapters from another run; resets optimizer/sampler/step")
    ap.add_argument("--fresh", action="store_true",
                    help="explicitly archive an existing run and start over; incompatible with resume")
    args = ap.parse_args()

    if args.init_adapters_from and args.resume != "none":
        raise SystemExit("--init-adapters-from requires --resume none")

    if args.steps < 0:
        raise SystemExit("--steps must be non-negative")
    if args.batch < 1 or args.max_batch < 1:
        raise SystemExit("--batch and --max-batch must be positive")
    if args.loop_count < 1:
        raise SystemExit("--loop-count must be at least 1")
    if args.max_batch < args.batch:
        raise SystemExit("--max-batch must be at least --batch")
    if args.min_batch < 0 or (args.min_batch and args.min_batch > args.max_batch):
        raise SystemExit("--min-batch must be zero or between 1 and --max-batch")
    if args.allow_text_limit_increase_from < 0:
        raise SystemExit("--allow-text-limit-increase-from must be non-negative")
    if args.max_text_tokens < 2 or args.prefix_tokens < 1:
        raise SystemExit("--max-text-tokens and --prefix-tokens must be positive")
    if args.max_input_patches < 4:
        raise SystemExit("--max-input-patches must be at least 4")
    if args.vision_resampler_layers < 0 or args.vision_resampler_width < 1:
        raise SystemExit("vision resampler layers must be non-negative and width positive")
    if (args.vision_resampler_heads < 1
            or args.vision_resampler_width % args.vision_resampler_heads):
        raise SystemExit("vision resampler width must be divisible by its positive head count")
    if args.deep_vision_rank < 1:
        raise SystemExit("--deep-vision-rank must be positive")
    if args.layer_vision_rank < 1:
        raise SystemExit("--layer-vision-rank must be positive")
    if args.vision_fusion_rank < 1:
        raise SystemExit("--vision-fusion-rank must be positive")
    if args.siglip2_width < 1:
        raise SystemExit("--siglip2-width must be positive")
    try:
        moonvit_taps = tuple(sorted({int(value.strip()) for value in
                                    args.moonvit_tap_layers.split(",")
                                    if value.strip()}))
        layer_vision_sites = tuple(sorted({int(value.strip()) for value in
                                          args.layer_vision_layers.split(",")
                                          if value.strip()}))
    except ValueError as exc:
        raise SystemExit("invalid MoonViT/RWKV layer-match specification") from exc
    if bool(moonvit_taps) != bool(layer_vision_sites):
        raise SystemExit("MoonViT taps and layer-vision sites must be enabled together")
    if len(moonvit_taps) != len(layer_vision_sites):
        raise SystemExit("MoonViT taps and layer-vision sites need the same count")
    if any(index < 0 or index >= 27 for index in moonvit_taps):
        raise SystemExit("MoonViT tap layers must be between 0 and 26")
    args.moonvit_tap_layers = ",".join(map(str, moonvit_taps))
    args.layer_vision_layers = ",".join(map(str, layer_vision_sites))
    if args.grounding_early_tokens < 0 or args.grounding_early_weight < 1:
        raise SystemExit("grounding early-token count must be non-negative and weight at least 1")
    if (args.grounding_contrastive_weight < 0
            or args.grounding_contrastive_dim < 1
            or args.grounding_temperature <= 0):
        raise SystemExit("invalid grounding contrastive configuration")
    if (args.allow_text_limit_increase_from
            and args.max_text_tokens <= args.allow_text_limit_increase_from):
        raise SystemExit("text-limit migration requires a strictly larger --max-text-tokens")
    if not 0 < args.loop_token_budget_scale <= 1:
        raise SystemExit("--loop-token-budget-scale must be in (0, 1]")
    if args.loop_ramp_steps < 0:
        raise SystemExit("--loop-ramp-steps must be non-negative")
    if args.loop_start_step < 0:
        raise SystemExit("--loop-start-step must be non-negative")
    if args.target_batch_tokens < 0:
        raise SystemExit("--target-batch-tokens must be non-negative")
    if args.lr <= 0 or args.loop_lr < 0 or args.engram_lr < 0:
        raise SystemExit("--lr must be positive; adapter-specific learning rates non-negative")
    if args.grad_clip <= 0:
        raise SystemExit("--grad-clip must be positive")
    if args.nextlat_weight < 0 or args.nextlat_kl_weight < 0 or args.nextlat_hidden < 1:
        raise SystemExit("NextLat weights must be non-negative and hidden size positive")
    if args.engram_drow < 1 or args.engram_rows < 1:
        raise SystemExit("--engram-drow and --engram-rows must be positive")
    if args.manifest_stat_workers < 1:
        raise SystemExit("--manifest-stat-workers must be positive")
    if args.engram_warmup_steps < 0:
        raise SystemExit("--engram-warmup-steps must be non-negative")
    if args.eval_every < 0 or args.checkpoint_every < 0:
        raise SystemExit("eval/checkpoint intervals must be non-negative")
    if args.log_every < 1:
        raise SystemExit("--log-every must be positive")
    if args.eval_every and args.eval_examples < 1:
        raise SystemExit("--eval-examples must be positive when evaluation is enabled")
    if args.eval_samples < 0 or args.eval_sample_max_new < 0:
        raise SystemExit("qualitative eval sizes must be non-negative")
    if args.profile_steps < 0 or args.operator_profile_steps < 0:
        raise SystemExit("profiling step counts must be non-negative")
    args.fused_ce_enabled = HAS_FUSED_CE
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    try:
        run_lock = _acquire_run_lock(out)
    except RuntimeError as error:
        # Do not touch status/config when another live trainer owns the run.
        raise SystemExit(str(error)) from error
    # ThreadPoolExecutor workers are joined before ordinary atexit callbacks.
    # Retain the lock until that shutdown completes so a replacement trainer
    # cannot overlap old cache workers that are still draining.
    atexit.register(run_lock.close)
    log_path, checkpoint_path = out / "train.jsonl", out / "last.pt"
    eval_contract_reset_path = out / "eval_contract_reset.json"
    best_dir = out / "best"
    if args.fresh:
        if args.resume != "none":
            raise SystemExit("--fresh requires --resume none")
        archived = _archive_fresh_run_artifacts(out)
        if archived is not None:
            print(f"archived prior trainer artifacts at {archived}", flush=True)
        # Fresh starts a new eval contract even before its first winner. This
        # also suppresses an old SQLite minimum while log archival is observed.
        _publish_eval_contract_reset(
            eval_contract_reset_path, step=0, reasons=("fresh",))
    else:
        existing_artifacts = _trainer_run_artifact_paths(out)
        if args.resume == "none" and existing_artifacts:
            raise SystemExit(
                f"refusing to overwrite existing run {out}; use --resume auto "
                "(default) or explicit --fresh")
        if (args.resume == "auto" and not checkpoint_path.exists()
                and existing_artifacts):
            raise SystemExit(
                f"{out} contains run artifacts but no recoverable last.pt; "
                "use --fresh only if restarting is intentional")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    _atomic_json(out / "config.json", {"schema": CHECKPOINT_SCHEMA, **vars(args)})
    status_path = out / "status.json"
    _atomic_json(status_path, {"state": "loading_data", "updated": time.time()})
    # Model/data/checkpoint failures happen before the training loop's guarded
    # section. Never strand the dashboard/watchdog in a fictional loading state.
    atexit.register(
        _fail_nonterminal_status, status_path,
        reason="process_exit_before_terminal_state")
    previous_excepthook = sys.excepthook

    def record_unhandled(exc_type, exc_value, traceback) -> None:
        _fail_nonterminal_status(
            status_path, reason="unhandled_exception",
            error=f"{exc_type.__name__}: {exc_value}")
        previous_excepthook(exc_type, exc_value, traceback)

    sys.excepthook = record_unhandled

    # This is an environment/startup failure, not a CLI-shape error.  Check it
    # only after installing the terminal-status guard so a supervised launcher
    # cannot leave an old paused/loading status behind and restart forever.
    if args.require_fused_ce and not HAS_FUSED_CE:
        message = "--require-fused-ce requested but flash-attn Triton CE is unavailable"
        _fail_nonterminal_status(
            status_path, reason="runtime_requirement_unavailable", error=message)
        raise SystemExit(message)

    raw_train = [row for source in args.data
                 for row in load_examples(source, stat_workers=args.manifest_stat_workers)]
    if not raw_train:
        raise SystemExit("No usable image/text rows found")
    # Deduplicate exact image-caption pairs before sampling.
    def task_identity(row: dict) -> tuple[str, str, str]:
        return (
            repr(_image_file_identity(row)), row["text"],
            str(row.get("prompt") or args.prompt),
        )

    unique_train = {}
    for row in raw_train:
        unique_train[task_identity(row)] = row
    vocab = WorldVocab()
    unique_train_rows = list(unique_train.values())
    train_rows, train_lengths = prepare_examples(unique_train_rows, vocab,
                                                  prompt=args.prompt,
                                                  max_text_tokens=args.max_text_tokens,
                                                  sandwich_prompt=args.sandwich_prompt)
    unique_eval_rows: list[dict] = []
    if args.eval_data:
        raw_eval = [row for source in args.eval_data
                    for row in load_examples(source, stat_workers=args.manifest_stat_workers)]
        train_images = {_image_file_identity(row) for row in unique_train.values()}
        eval_images = {_image_file_identity(row) for row in raw_eval}
        image_overlap = train_images & eval_images
        if image_overlap:
            raise SystemExit(
                f"explicit eval is not image-disjoint: {len(image_overlap)} overlapping images"
            )
        unique_eval = {}
        for row in raw_eval:
            identity = task_identity(row)
            if identity not in unique_train:
                unique_eval[identity] = row
        unique_eval_rows = list(unique_eval.values())
        eval_rows, eval_lengths = prepare_examples(unique_eval_rows, vocab,
                                                    prompt=args.prompt,
                                                    max_text_tokens=args.max_text_tokens,
                                                    sandwich_prompt=args.sandwich_prompt)
        if not eval_rows:
            raise SystemExit("No usable held-out image/text rows found")
        rows = train_rows + eval_rows
        lengths = train_lengths + eval_lengths
        train_indices = list(range(len(train_rows)))
        val_indices = list(range(len(train_rows), len(rows)))
    else:
        rows, lengths = train_rows, train_lengths
        train_indices, val_indices = split_examples(rows, val_fraction=args.val_fraction)
    qualitative_val_indices = filter_eval_sample_indices(
        rows, val_indices, args.eval_sample_exclude_sources.split(","))
    if args.eval_samples and not qualitative_val_indices:
        raise SystemExit(
            "qualitative eval source filter excluded every held-out image")
    args.data_fingerprint = dataset_fingerprint(
        rows, train_indices, val_indices, explicit_eval=bool(args.eval_data))
    args.image_metadata_fingerprint = image_metadata_fingerprint(rows)
    if args.allow_text_limit_increase_from:
        previous_train, _ = prepare_examples(
            unique_train_rows, vocab, prompt=args.prompt,
            max_text_tokens=args.allow_text_limit_increase_from,
            sandwich_prompt=args.sandwich_prompt)
        if args.eval_data:
            previous_eval, _ = prepare_examples(
                unique_eval_rows, vocab, prompt=args.prompt,
                max_text_tokens=args.allow_text_limit_increase_from,
                sandwich_prompt=args.sandwich_prompt)
            previous_rows = previous_train + previous_eval
        else:
            previous_rows = previous_train
        if len(previous_rows) != len(rows):
            raise SystemExit("text-limit migration changed the usable row set")
        args.previous_data_fingerprint = dataset_fingerprint(
            previous_rows, train_indices, val_indices,
            explicit_eval=bool(args.eval_data))
    def checkpoint_fingerprint(path: str) -> str:
        resolved = Path(path).resolve()
        stat = resolved.stat()
        return hashlib.sha256(
            f"{resolved}|{stat.st_size}|{stat.st_mtime_ns}".encode()).hexdigest()
    args.rwkv_fingerprint = checkpoint_fingerprint(args.rwkv)
    args.moonvit_fingerprint = checkpoint_fingerprint(args.moonvit)
    args.vision_fusion_fingerprint = (VisionTowerConfig(
        siglip2=args.siglip2_model, dinov2=args.dinov2_model,
        sam=args.sam_model, siglip_width=args.siglip2_width).fingerprint()
        if args.vision_fusion else "")
    _atomic_json(out / "config.json", {"schema": CHECKPOINT_SCHEMA, **vars(args)})
    sampler = EpochBatchSampler(train_indices, lengths, batch_size=args.batch, seed=args.seed)
    token_costs = [args.prefix_tokens + length for length in lengths]
    train_truncated = sum(rows[index]["truncated"] for index in train_indices)
    val_truncated = sum(rows[index]["truncated"] for index in val_indices)
    print(f"data: {len(train_indices)} train / {len(val_indices)} val; "
          f"{train_truncated} train + {val_truncated} val captions truncated", flush=True)

    _atomic_json(out / "status.json", {"state": "loading_rwkv", "updated": time.time()})
    print(f"loading frozen RWKV: {args.rwkv}", flush=True)
    rwkv = load_g1g_fla(args.rwkv, device="cuda")
    rwkv.requires_grad_(False)
    rwkv.eval()
    engram = None
    if args.engram:
        engram_sites = _parse_engram_sites(
            args.engram_sites, int(rwkv.config.num_hidden_layers))
        vocab_size = int(rwkv.config.vocab_size)
        engram = LexicalMemoryBank(
            hidden_size=int(rwkv.config.hidden_size), vocab_size=vocab_size,
            layer_sites=engram_sites, d_row=args.engram_drow,
            table_rows=min(args.engram_rows, vocab_size),
            num_heads=int(rwkv.config.num_heads), max_loops=args.loop_count,
            boundary_id=args.engram_boundary_id)
        # Match the frozen stream dtype. One-dimensional growth/gate tensors
        # are promoted back to fp32 so their zero-origin updates are retained.
        engram.to(device="cuda", dtype=torch.bfloat16)
        float_growth_params(engram)
        # Attach while the original FLA TimeMix module is still directly on
        # each layer. Its value hook remains valid after factored-loop wrapping.
        attach_engram(rwkv, engram, resolve="model.layers")
        rwkv.engram = engram
        print(f"Engram: sites={engram_sites} d_row={args.engram_drow} "
              f"rows={min(args.engram_rows, vocab_size)} "
              f"params={sum(p.numel() for p in engram.parameters())/1e6:.1f}M",
              flush=True)
    wrappers = install_factored_timemix(
        rwkv, n_loops=args.loop_count, gate_cap=args.loop_gate_cap,
        loop_index=args.loop_index)

    _atomic_json(out / "status.json", {"state": "loading_moonvit", "updated": time.time()})
    print(f"loading frozen MoonViT: {args.moonvit}", flush=True)
    moonvit_taps = tuple(int(value) for value in args.moonvit_tap_layers.split(",")
                         if value)
    vision = MoonViT.from_checkpoint(args.moonvit, device="cuda",
                                     max_input_patches=args.max_input_patches,
                                     tap_layers=moonvit_taps,
                                     view_mode=args.vision_view_mode)
    vision.requires_grad_(False)
    vision.eval()
    # Trainable modules intentionally remain fp32; autocast supplies bf16 matmuls
    # while AdamW retains real fp32 master parameters and moments.
    projector = MoonViTPrefixProjector(
        rwkv.config.hidden_size, args.prefix_tokens,
        resampler_layers=args.vision_resampler_layers,
        resampler_width=args.vision_resampler_width,
        resampler_heads=args.vision_resampler_heads).cuda().float()
    fusion_tower = None
    vision_fusion = None
    fusion_cache_dir = None
    if args.vision_fusion:
        fusion_tower = AlignedFrozenVisionFeatures(VisionTowerConfig(
            siglip2=args.siglip2_model, dinov2=args.dinov2_model,
            sam=args.sam_model, siglip_width=args.siglip2_width))
        fusion_tower.requires_grad_(False).eval()
        vision_fusion = VisionFusionResidual(
            int(rwkv.config.hidden_size), rank=args.vision_fusion_rank,
            source_width=fusion_tower.width).cuda().float()
        fusion_cache_dir = Path(args.fusion_feature_cache)
        print(f"three-tower fusion: SigLIP2+DINOv2+SAM rank={args.vision_fusion_rank} "
              f"cache={fusion_cache_dir} fingerprint={fusion_tower.cache_fingerprint[:12]}",
              flush=True)
    deep_vision = None
    if args.deep_vision_layers.strip():
        deep_sites = _parse_engram_sites(
            args.deep_vision_layers, int(rwkv.config.num_hidden_layers))
        deep_vision = DeepVisionInjector(
            int(rwkv.config.hidden_size), deep_sites,
            rank=args.deep_vision_rank).cuda().float()
        deep_vision.install(rwkv.model.layers)
        print(f"deep vision: sites={deep_sites} rank={args.deep_vision_rank} "
              f"params={sum(p.numel() for p in deep_vision.parameters())/1e6:.1f}M",
              flush=True)
    layer_vision = None
    if args.layer_vision_layers.strip():
        layer_sites = _parse_engram_sites(
            args.layer_vision_layers, int(rwkv.config.num_hidden_layers))
        layer_vision = LayerMatchedVisionInjector(
            int(rwkv.config.hidden_size), layer_sites,
            rank=args.layer_vision_rank).cuda().float()
        layer_vision.install(rwkv.model.layers)
        print(f"layer-matched vision: taps={moonvit_taps} sites={layer_sites} "
              f"rank={args.layer_vision_rank} "
              f"params={sum(p.numel() for p in layer_vision.parameters())/1e6:.1f}M",
              flush=True)
    grounding = (ImageTextContrastiveHead(
        int(rwkv.config.hidden_size), width=args.grounding_contrastive_dim,
        temperature=args.grounding_temperature).cuda().float()
        if args.grounding_contrastive_weight else None)
    nextlat = (NextLatPredictor(rwkv.config.hidden_size, hidden=args.nextlat_hidden).cuda().float()
               if args.nextlat_weight else None)
    optimizer, trainable = _optimizer(
        projector, nextlat, engram, wrappers, args,
        deep_vision=deep_vision, layer_vision=layer_vision,
        vision_fusion=vision_fusion,
        grounding=grounding)
    assert_training_contract(rwkv, vision, wrappers, trainable)
    last_checkpoint_step: int | None = None

    def save_last_checkpoint(checkpoint_step: int) -> None:
        """Save ``last.pt`` and remember which committed state it contains."""
        nonlocal last_checkpoint_step
        _save_checkpoint(
            checkpoint_path, step=checkpoint_step, projector=projector,
            nextlat=nextlat, engram=engram, deep_vision=deep_vision,
            layer_vision=layer_vision, grounding=grounding, wrappers=wrappers,
            optimizer=optimizer, sampler=sampler, args=args,
            vision_fusion=vision_fusion)
        # Update only after the atomic durable save succeeds.
        last_checkpoint_step = checkpoint_step

    loop_trainable = [parameter for group in optimizer.param_groups
                      if str(group.get("name", "")).startswith("loop_")
                      for parameter in group["params"]]
    bridge_trainable = [parameter for group in optimizer.param_groups
                        if not str(group.get("name", "")).startswith("loop_")
                        and group.get("name") != "engram"
                        for parameter in group["params"]]
    engram_trainable = [parameter for group in optimizer.param_groups
                        if group.get("name") == "engram"
                        for parameter in group["params"]]

    if args.resume == "auto":
        resume_path = checkpoint_path if checkpoint_path.exists() else None
    elif args.resume == "none":
        resume_path = None
    else:
        resume_path = Path(args.resume)
        if not resume_path.is_file():
            raise SystemExit(f"resume checkpoint does not exist: {resume_path}")
    step = 0
    text_limit_migrated = False
    loop_reset_already_applied = False
    loop_reset_performed = False
    resume_contract_changed = False
    unrelated_resume_branch = False
    if resume_path is not None:
        print(f"resuming exact training state from {resume_path}", flush=True)
        (step, text_limit_migrated, loop_reset_already_applied,
         resume_contract_changed) = _load_checkpoint(
             resume_path, projector=projector, nextlat=nextlat, engram=engram,
             wrappers=wrappers, optimizer=optimizer, sampler=sampler, args=args,
             deep_vision=deep_vision, layer_vision=layer_vision,
             vision_fusion=vision_fusion, grounding=grounding)
        _preserve_loop_reset_outcome(args, loop_reset_already_applied)
        last_checkpoint_step = _resumed_last_checkpoint_step(
            resume_path, checkpoint_path, step,
            contract_changed=resume_contract_changed)
        unrelated_resume_branch = _resume_requires_best_quarantine(
            resume_path, checkpoint_path, _best_checkpoint_path(best_dir))
        _trim_log(log_path, step)
        if unrelated_resume_branch:
            quarantined_best = _quarantine_best(
                best_dir, f"before-explicit-resume-step-{step}")
        else:
            quarantined_best = _quarantine_future_best(best_dir, step)
        if quarantined_best is not None:
            reason = ("unrelated active best" if unrelated_resume_branch
                      else "future-branch best checkpoint")
            print(f"preserved {reason} at {quarantined_best}; "
                  f"it is no longer advertised by resumed step {step}",
                  flush=True)
        if text_limit_migrated:
            migrated_best = _quarantine_best(
                best_dir,
                f"before-text-limit-{args.allow_text_limit_increase_from}-to-{args.max_text_tokens}",
            )
            if migrated_best is not None:
                print(
                    f"preserved pre-migration best checkpoint at {migrated_best}; "
                    "the new eval contract starts with no claimed winner",
                    flush=True,
                )
        if text_limit_migrated:
            print(
                f"increased text limit from {args.allow_text_limit_increase_from} "
                f"to {args.max_text_tokens}; retained optimizer and sampler state",
                flush=True,
            )
        loop_reset_pending = bool(
            args.reset_loop_on_resume and not loop_reset_already_applied)
        if loop_reset_pending:
            reset_best = _quarantine_best(best_dir, f"before-loop-reset-step-{step}")
            if reset_best is not None:
                print(
                    f"preserved pre-reset best checkpoint at {reset_best}; "
                    "the reset loop branch starts with no claimed winner",
                    flush=True,
                )
        invalidates_eval_contract = _resume_invalidates_step_evaluation(
                text_limit_migrated=text_limit_migrated,
                unrelated_branch=unrelated_resume_branch,
                loop_reset_pending=loop_reset_pending)
        if invalidates_eval_contract:
            reset_reasons = []
            if unrelated_resume_branch:
                reset_reasons.append("explicit_resume_branch")
            if text_limit_migrated:
                reset_reasons.append("text_limit_migration")
            if loop_reset_pending:
                reset_reasons.append("loop_reset")
            # Publish independently of best/ existence. A missing or corrupt
            # active best must not let abandoned log minima masquerade as the
            # winner of the accepted contract.
            _publish_eval_contract_reset(
                eval_contract_reset_path, step=step, reasons=reset_reasons)
            if _invalidate_step_evaluation(log_path, step):
                print(f"invalidated prior evaluation claims at mutated step {step}",
                      flush=True)
        if loop_reset_pending:
            reset_loop_adapters(wrappers)
            _reset_loop_optimizer_state(optimizer, wrappers, args)
            # Persist an outcome marker, not merely the CLI request. A run may
            # be launched with the flag in a context where no resume/reset was
            # performed; only this point proves the mutation was committed.
            args.loop_reset_committed = True
            save_last_checkpoint(step)
            loop_reset_performed = True
            print(f"reset loop adapters and optimizer state at recovered step {step}", flush=True)
        elif args.reset_loop_on_resume:
            print(f"loop reset was already committed at recovered step {step}; "
                  "not applying the one-time recovery twice", flush=True)
    elif args.init_adapters_from:
        init_path = Path(args.init_adapters_from)
        if not init_path.is_file():
            raise SystemExit(f"adapter initialization checkpoint does not exist: {init_path}")
        source_step = _initialize_adapters(
            init_path, projector=projector, nextlat=nextlat, engram=engram,
            wrappers=wrappers, args=args, deep_vision=deep_vision,
            layer_vision=layer_vision, vision_fusion=vision_fusion,
            grounding=grounding)
        print(f"initialized adapters from {init_path} at source step {source_step}; "
              "optimizer, sampler, and step reset for new phase", flush=True)
    elif log_path.exists() and not args.fresh:
        raise SystemExit(f"{log_path} exists but no recoverable checkpoint was found; use --fresh only if intentional")

    # An explicit resume can branch from an older file, while an in-place text
    # or batch migration can change the contract represented by the same file.
    # Publish either change after all one-time migrations/resets, before cache
    # preload or other lengthy startup work. A later --resume auto must recover
    # the accepted branch and current arguments even if this process stops
    # before its first optimizer update or periodic checkpoint.
    if _resume_checkpoint_publication_required(
            resume_path, last_checkpoint_step):
        save_last_checkpoint(step)
        print(f"published resumed branch and contract as {checkpoint_path} at step {step}",
              flush=True)

    best_eval_loss = float("inf")
    best_info_path = best_dir / "best.json"
    if (not text_limit_migrated and best_info_path.is_file()
            and _best_checkpoint_path(best_dir) is not None):
        try:
            best_info = json.loads(best_info_path.read_text())
            # An explicit resume from an older checkpoint must not claim a
            # winner from a future branch of training.
            if int(best_info["step"]) <= step:
                candidate_best = (float(best_info["loss"])
                                  if "loss" in best_info
                                  else math.log(float(best_info["ppl"])))
                if math.isfinite(candidate_best):
                    best_eval_loss = candidate_best
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            best_eval_loss = float("inf")

    cache_dir = Path(args.feature_cache) if args.feature_cache else None
    preload_stop = threading.Event()
    if args.preload_feature_cache:
        if cache_dir is None:
            raise SystemExit("--preload-feature-cache requires --feature-cache")
        def warm_feature_cache() -> None:
            try:
                # Warm the exact unconsumed sampler order first, then held-out
                # eval and already-consumed rows. This makes the background job
                # useful immediately instead of walking an unordered path set.
                priority = (sampler.order[sampler.position:] + list(val_indices)
                            + sampler.order[:sampler.position])
                preload_rows = [rows[index] for index in priority]
                loaded_features, resident_bytes = preload_feature_cache(
                    preload_rows, vision, projector, cache_dir,
                    stop_event=preload_stop)
                print({"kind": "feature_preload_complete", "features": loaded_features,
                       "resident_gib": round(resident_bytes / 2**30, 2)}, flush=True)
            except Exception as error:
                # Cache misses still use the ordinary verified loader, so a
                # failed background warmup must not terminate a recoverable run.
                print({"kind": "feature_preload_failed", "error": repr(error)}, flush=True)

        if args.background_feature_preload:
            print({"kind": "feature_preload_started", "mode": "background"}, flush=True)
            threading.Thread(target=warm_feature_cache, name="feature-preload-driver",
                             daemon=True).start()
        else:
            _atomic_json(out / "status.json", {
                "state": "preloading_features", "step": step, "updated": time.time()})
            warm_feature_cache()
    loop_enabled = step >= args.loop_start_step and args.loop_count > 1
    set_loop_enabled(wrappers, loop_enabled)
    # Restore the exact runtime scale of the committed model state. This is
    # observable when resume first drains a scheduled eval/caption obligation;
    # advancing to step+1 here would make that recovered eval differ from the
    # uninterrupted eval at ``step``. The training loop advances scales before
    # its next forward pass.
    set_loop_scale(wrappers, _loop_runtime_scale(
        step, start_step=args.loop_start_step, ramp_steps=args.loop_ramp_steps))
    if engram is not None:
        engram.set_warmup(
            1.0 if args.engram_warmup_steps <= 0
            else min(1.0, step / args.engram_warmup_steps))
    log = log_path.open("a", buffering=1)
    started = time.time()
    _atomic_json(out / "status.json", {"state": "training", "step": step,
                                       "resumed": resume_path is not None, "updated": time.time()})
    source_counts = Counter(str(rows[index].get("stage1_source") or
                                rows[index].get("source") or "unknown")
                            for index in train_indices)
    eval_source_counts = Counter(str(rows[index].get("stage1_source") or
                                     rows[index].get("source") or "unknown")
                                 for index in val_indices)
    startup = {"kind": "startup", "step": step, "resumed": resume_path is not None,
               "initialized_from": args.init_adapters_from,
               "loop_reset_performed": loop_reset_performed,
               "text_limit_migrated": text_limit_migrated,
               "max_text_tokens": args.max_text_tokens,
               "train_examples": len(train_indices), "val_examples": len(val_indices),
               "source_counts": dict(sorted(source_counts.items())),
               "eval_source_counts": dict(sorted(eval_source_counts.items())),
               "trainable_parameters": sum(p.numel() for p in trainable),
               "vision_resampler_parameters": (
                   sum(p.numel() for p in projector.resampler.parameters())
                   if projector.resampler is not None else 0),
               "deep_vision_parameters": (
                   sum(p.numel() for p in deep_vision.parameters())
                   if deep_vision is not None else 0),
               "grounding_parameters": (
                   sum(p.numel() for p in grounding.parameters())
                   if grounding is not None else 0),
               "engram_parameters": (sum(p.numel() for p in engram.parameters())
                                     if engram is not None else 0),
               "fused_ce": HAS_FUSED_CE,
               "frozen_rwkv_parameters": sum(p.numel() for p in rwkv.parameters() if not p.requires_grad),
               "frozen_vision_parameters": sum(p.numel() for p in vision.parameters()),
               "sampler_epoch": sampler.epoch, "sampler_position": sampler.position}
    write_loop_telemetry(out / "loop_rw.json", wrappers, step=step)
    log.write(json.dumps(startup) + "\n")
    if loop_reset_performed:
        log.write(json.dumps({"kind": "loop_reset", "step": step,
                              "loop_lr": args.loop_lr,
                              "loop_ramp_steps": args.loop_ramp_steps}) + "\n")
    print(startup, flush=True)

    def schedule_next_batch_prefetch(step_number: int, *, position_offset: int = 0):
        if not args.prefetch_next_batch:
            return None, None
        target = args.target_batch_tokens
        if step_number >= args.loop_start_step and args.loop_count > 1:
            target = int(target * args.loop_token_budget_scale)
        future_indices = sampler.peek_budget_batch(
            token_costs, target_tokens=target,
            min_items=(args.min_batch or args.batch), max_items=args.max_batch,
            position_offset=position_offset)
        if not future_indices:
            return None, None
        future_rows = [rows[index] for index in future_indices]
        future = _NEXT_BATCH_POOL.submit(
            prefetch_training_batch, future_rows, vision, projector, cache_dir,
            engram, fusion_tower, fusion_cache_dir)
        return future_indices, future

    def run_evaluation(eval_step: int, prior_eval: dict | None = None, *,
                       checkpoint_saved: bool = False) -> bool:
        """Run one scalar+qualitative eval; return whether it saved ``last.pt``."""
        nonlocal best_eval_loss

        def eval_progress(phase: str):
            def update(done: int, total: int) -> None:
                _atomic_json(out / "status.json", {
                    "state": "evaluating", "phase": phase, "step": eval_step,
                    "progress": done, "total": total, "updated": time.time(),
                })
            return update

        if prior_eval is None:
            eval_progress("loss")(0, min(len(val_indices), args.eval_examples))
            val_loss = evaluate(
                rows, val_indices, rwkv=rwkv, projector=projector,
                vision=vision, engram=engram, cache_dir=cache_dir,
                batch_size=args.batch, max_examples=args.eval_examples,
                deep_vision=deep_vision, layer_vision=layer_vision,
                fusion_tower=fusion_tower, fusion_adapter=vision_fusion,
                fusion_cache_dir=fusion_cache_dir,
                progress=eval_progress("loss"))
            val_loss = _require_finite_metric("validation loss", val_loss)
            improved = val_loss < best_eval_loss
            if improved:
                # Persist the winner before qualitative decoding. Caption
                # generation is intentionally much longer than scalar eval; a
                # pause there must not lose an already-proven best model.
                previous_mask = signal.pthread_sigmask(
                    signal.SIG_BLOCK, {signal.SIGINT})
                try:
                    _sync_log(log)
                    if not checkpoint_saved:
                        save_last_checkpoint(eval_step)
                    _promote_checkpoint(
                        checkpoint_path, best_dir, step=eval_step, loss=val_loss)
                    best_eval_loss = val_loss
                    checkpoint_saved = True
                finally:
                    signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)
                log.write(json.dumps({
                    "kind": "checkpoint", "step": eval_step,
                    "reason": "best_eval_promoted",
                    "path": str(_best_checkpoint_path(best_dir)),
                }) + "\n")
            ppl = math.exp(min(val_loss, 20.0))
            expected_artifact = (
                out / "eval_samples" / f"step_{eval_step:08d}.json"
                if (args.eval_samples > 0 and args.eval_sample_max_new > 0
                    and qualitative_val_indices)
                else None
            )
            # The scalar evaluation is complete now. Log it before greedy
            # decoding so the eval marker and best header survive a hard stop
            # during the much longer qualitative phase.
            log.write(json.dumps({
                "kind": "eval", "step": eval_step, "loss": val_loss,
                "val_loss": val_loss, "ppl": ppl,
                "sample_artifact": (str(expected_artifact)
                                    if expected_artifact else None),
                "best": improved,
                "qualitative_complete": expected_artifact is None,
            }) + "\n")
            _sync_log(log)
        else:
            # Scalar eval and any best promotion are already durable. Resume
            # only the qualitative phase instead of recomputing or changing the
            # recorded winner.
            val_loss = _require_finite_metric(
                "resumed validation loss",
                prior_eval.get("val_loss", prior_eval.get("loss")))
            ppl = _require_finite_metric(
                "resumed validation ppl",
                prior_eval.get("ppl", math.exp(min(val_loss, 20.0))))
            improved = bool(prior_eval.get("best", False))
        eval_progress("captions")(0, args.eval_sample_max_new)
        sample_artifact = write_eval_samples(
            rows, qualitative_val_indices, step=eval_step, ppl=ppl, rwkv=rwkv,
            projector=projector, vision=vision, engram=engram,
            cache_dir=cache_dir, vocab=vocab, prompt=args.prompt, out=out,
            count=args.eval_samples, max_new=args.eval_sample_max_new,
            deep_vision=deep_vision, layer_vision=layer_vision,
            sandwich_prompt=args.sandwich_prompt,
            fusion_tower=fusion_tower, fusion_adapter=vision_fusion,
            fusion_cache_dir=fusion_cache_dir,
            progress=eval_progress("captions"))
        evaluation = {
            "kind": "eval_artifact", "step": eval_step, "loss": val_loss,
            "val_loss": val_loss, "ppl": ppl,
            "sample_artifact": (str(sample_artifact) if sample_artifact else None),
            "best": improved, "qualitative_complete": True,
        }
        log.write(json.dumps(evaluation) + "\n")
        print(evaluation, flush=True)
        _atomic_json(out / "status.json", {
            "state": "training", "step": eval_step, "updated": time.time(),
        })
        return checkpoint_saved

    prefetched_indices, prefetch_future = schedule_next_batch_prefetch(step + 1)
    resume_eval_work = _pending_eval_work(
        log_path, step,
        eval_expected=bool(step > 0 and args.eval_every
                           and step % args.eval_every == 0))

    interrupted = False
    # False only while optimizer.step may have partially mutated parameters but
    # the sampler/public step have not committed. Outside that narrow window,
    # an unexpected failure can preserve an exact recovery point.
    checkpoint_state_valid = True
    try:
        while step < args.steps or resume_eval_work is not None:
            if resume_eval_work is not None:
                phase, prior_eval = resume_eval_work
                print({"kind": "eval_resume", "step": step, "phase": phase},
                      flush=True)
                run_evaluation(step, prior_eval=prior_eval)
                resume_eval_work = None
                continue
            next_step = step + 1
            desired_loop = next_step >= args.loop_start_step and args.loop_count > 1
            if desired_loop != loop_enabled:
                # Save the exact last warmup state before exercising a delayed
                # architecture path for the first time. Keep a named rollback
                # copy: periodic last.pt checkpoints must never overwrite it.
                pre_loop_path = out / "pre_loop.pt"
                _sync_log(log)
                _save_checkpoint(pre_loop_path, step=step, projector=projector,
                                 nextlat=nextlat, engram=engram,
                                 deep_vision=deep_vision, layer_vision=layer_vision,
                                 grounding=grounding,
                                 wrappers=wrappers,
                                 optimizer=optimizer,
                                 sampler=sampler, args=args,
                                 vision_fusion=vision_fusion)
                log.write(json.dumps({"kind": "checkpoint", "step": step,
                                      "reason": "before_loop_activation",
                                      "path": str(pre_loop_path)}) + "\n")
                loop_enabled = desired_loop
                set_loop_enabled(wrappers, loop_enabled)
                log.write(json.dumps({"kind": "loop_enabled", "step": next_step,
                                      "enabled": loop_enabled}) + "\n")
            loop_scale = (_loop_runtime_scale(
                next_step, start_step=args.loop_start_step, ramp_steps=args.loop_ramp_steps)
                if loop_enabled else 0.0)
            set_loop_scale(wrappers, loop_scale)
            engram_scale = 0.0
            if engram is not None:
                engram_scale = min(1.0, next_step / max(args.engram_warmup_steps, 1))
                engram.set_warmup(engram_scale)
            profile = next_step <= args.profile_steps
            if profile:
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            target_tokens = args.target_batch_tokens
            if loop_enabled:
                target_tokens = int(target_tokens * args.loop_token_budget_scale)
            sampler.ensure_epoch()
            indices = sampler.peek_budget_batch(
                token_costs, target_tokens=target_tokens,
                min_items=(args.min_batch or args.batch), max_items=args.max_batch)
            prefetch_wait_s = 0.0
            prefetch_ready = 0
            prefetched_recall = None
            if prefetch_future is not None and prefetched_indices == indices:
                wait_started = time.perf_counter()
                try:
                    prefetch_ready, prefetched_recall = prefetch_future.result()
                except Exception as error:
                    print({"kind": "next_batch_prefetch_failed",
                           "error": repr(error)}, flush=True)
                prefetch_wait_s = time.perf_counter() - wait_started
            elif prefetch_future is not None:
                prefetch_future.cancel()
            prefetched_indices, prefetch_future = schedule_next_batch_prefetch(
                next_step + 1, position_offset=len(indices))
            batch_rows = [rows[i] for i in indices]
            ids, labels, text_mask = make_batch(batch_rows, device="cuda")
            positions = supervised_positions(
                batch_rows, args.prefix_tokens, device="cuda")
            if profile:
                torch.cuda.synchronize()
            t_data = time.perf_counter()
            features = cached_features(batch_rows, vision, projector, cache_dir)
            fusion_features = (cached_fusion_features(
                batch_rows, fusion_tower, args.prefix_tokens, fusion_cache_dir)
                if fusion_tower is not None and vision_fusion is not None
                and fusion_cache_dir is not None else None)
            if profile:
                torch.cuda.synchronize()
            t_features = time.perf_counter()
            operator_profile = None
            if next_step <= args.operator_profile_steps:
                operator_profile = torch.profiler.profile(activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ])
                operator_profile.start()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss, metrics = multimodal_loss(
                    rwkv, projector, vision, (), ids, labels, text_mask,
                    nextlat=nextlat, nextlat_weight=args.nextlat_weight,
                    nextlat_kl_weight=args.nextlat_kl_weight,
                    engram=engram, features=features,
                    selected_positions=positions,
                    engram_recall=prefetched_recall,
                    deep_vision=deep_vision, layer_vision=layer_vision,
                    visual_starts=visual_insert_positions(batch_rows),
                    fusion_adapter=vision_fusion,
                    fusion_features=fusion_features,
                    grounding=grounding,
                    grounding_contrastive_weight=args.grounding_contrastive_weight,
                    grounding_early_tokens=args.grounding_early_tokens,
                    grounding_early_weight=args.grounding_early_weight)
            if profile:
                torch.cuda.synchronize()
            t_forward = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            # Loop gates can have very large raw gradients even while their
            # effective, ramped contribution is tiny. A single global clip made
            # those gradients scale projector/NextLat updates almost to zero.
            # Clip the bridge and recurrent adapters independently.
            grad_norm = torch.nn.utils.clip_grad_norm_(
                bridge_trainable, args.grad_clip, error_if_nonfinite=False)
            loop_grad_norm = torch.nn.utils.clip_grad_norm_(
                loop_trainable, args.grad_clip, error_if_nonfinite=False)
            engram_grad_norm = torch.nn.utils.clip_grad_norm_(
                engram_trainable, args.grad_clip, error_if_nonfinite=False)
            metric_names = list(metrics)
            # Gradient clipping already requires one safety barrier before the
            # optimizer. Materialize loss and auxiliary scalars in that same
            # transfer so a non-finite objective can never be discovered only
            # after weights and sampler state have been committed.
            safety_values = torch.stack((
                loss.detach(), grad_norm.to(loss.device),
                loop_grad_norm.to(loss.device), engram_grad_norm.to(loss.device),
                *(metrics[name] for name in metric_names),
            )).float().tolist()
            if not all(math.isfinite(value) for value in safety_values):
                raise FloatingPointError(
                    f"non-finite loss/gradient/auxiliary metrics: {safety_values}")
            loss_value = safety_values[0]
            norm_values = safety_values[1:4]
            metrics = dict(zip(metric_names, safety_values[4:]))
            # Treat the optimizer update, sampler advance, and public step as
            # one commit. SIGINT remains pending until all three agree, so an
            # interrupt checkpoint either repeats an unfinished batch or
            # resumes strictly after a completed one—never skips it.
            previous_mask = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGINT})
            eval_checkpoint_saved = False
            checkpoint_state_valid = False
            try:
                optimizer.step()
                sampler.commit_batch(indices)
                step = next_step
                checkpoint_state_valid = True
                if profile:
                    torch.cuda.synchronize()
                if operator_profile is not None:
                    operator_profile.stop()
                t_backward = time.perf_counter()
                loss_value = _require_finite_metric("training loss", loss_value)
                metrics = {name: _require_finite_metric(name, value)
                           for name, value in metrics.items()}
                record = {
                    "kind": "train", "step": step, "loss": loss_value,
                    "grad_norm": norm_values[0], "loop_grad_norm": norm_values[1],
                    "engram_grad_norm": norm_values[2],
                    "elapsed_s": round(time.time() - started, 1),
                    "batch_captions": len(indices),
                    "text_tokens": sum(len(row["tokens"]) for row in batch_rows),
                    "max_text_tokens": ids.shape[1], "sampler_epoch": sampler.epoch,
                    "sampler_position": sampler.position, "loop_enabled": loop_enabled,
                    "loop_count": args.loop_count, "loop_gate": "factored",
                    "loop_index": args.loop_index, "loop_scale": loop_scale,
                    "engram_scale": engram_scale, **metrics,
                    "batch_prefetch_ready": prefetch_ready,
                    "batch_prefetch_wait_s": round(prefetch_wait_s, 4),
                }
                record.update(loop_training_metrics(wrappers))
                record.update(_engram_metrics(engram))
                if "aux_loss" in metrics:
                    weighted = args.nextlat_weight * metrics["aux_loss"]
                    record["nextlat_weighted_loss"] = weighted
                    record["nextlat_to_ce_ratio"] = weighted / max(
                        metrics["ce_loss"], 1e-12)
                record["source_counts"] = dict(sorted(Counter(
                    str(row.get("stage1_source") or row.get("source") or "unknown")
                    for row in batch_rows).items()))
                if profile:
                    record.update(data_s=round(t_data - t0, 4),
                                  feature_s=round(t_features - t_data, 4),
                                  forward_s=round(t_forward - t_features, 4),
                                  backward_s=round(t_backward - t_forward, 4),
                                  step_s=round(t_backward - t0, 4))
                train_record = record if step % args.log_every == 0 else None
                if args.eval_every and step % args.eval_every == 0:
                    # Evaluation is a scheduled side effect of this committed
                    # model state. Publish its visible train row first, then the
                    # exact checkpoint, then the obligation. On recovery, the
                    # checkpoint step also implies the obligation if a host
                    # failure landed before the final log sync.
                    _publish_eval_due(
                        log, step=step, checkpoint_path=checkpoint_path,
                        train_record=train_record,
                        save_checkpoint=lambda: save_last_checkpoint(step),
                    )
                    eval_checkpoint_saved = True
                elif train_record is not None:
                    # The dashboard-visible step belongs to the same commit as
                    # weights and sampler state. A pending pause cannot save N
                    # while leaving the visible training series at N-1.
                    log.write(json.dumps(train_record) + "\n")
            finally:
                signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)
            if operator_profile is not None:
                trace = out / f"operator_step_{next_step:06d}.json.gz"
                operator_profile.export_chrome_trace(str(trace))
                table = operator_profile.key_averages().table(
                    sort_by="self_cuda_time_total", row_limit=30)
                (out / f"operator_step_{next_step:06d}.txt").write_text(table + "\n")
                print(table, flush=True)
            if step % 10 == 0:
                write_loop_telemetry(out / "loop_rw.json", wrappers, step=step)
                _atomic_json(out / "status.json", {"state": "training", "step": step,
                                                   "updated": time.time()})
            if step <= args.profile_steps or step % 10 == 0:
                print(record, flush=True)

            checkpoint_saved = eval_checkpoint_saved
            if args.eval_every and step % args.eval_every == 0:
                checkpoint_saved = run_evaluation(
                    step, checkpoint_saved=checkpoint_saved)

            if (args.checkpoint_every and step % args.checkpoint_every == 0
                    and not checkpoint_saved):
                _atomic_json(out / "status.json", {"state": "checkpointing", "step": step,
                                                   "updated": time.time()})
                _sync_log(log)
                save_last_checkpoint(step)
                log.write(json.dumps({"kind": "checkpoint", "step": step,
                                      "path": str(checkpoint_path)}) + "\n")
                _sync_log(log)
                checkpoint_saved = True
                _atomic_json(out / "status.json", {"state": "training", "step": step,
                                                   "updated": time.time()})
    except KeyboardInterrupt:
        # A dashboard/operator pause must be an exact, recoverable stop rather
        # than losing every completed update since the periodic checkpoint.
        # The atomic replacement preserves the previous checkpoint if a second
        # signal arrives while serialization is in progress.
        preload_stop.set()
        if prefetch_future is not None:
            prefetch_future.cancel()
        _atomic_json(out / "status.json", {"state": "checkpointing", "step": step,
                                           "reason": "interrupt", "updated": time.time()})
        _sync_log(log)
        save_last_checkpoint(step)
        log.write(json.dumps({"kind": "checkpoint", "step": step,
                              "reason": "interrupt",
                              "path": str(checkpoint_path)}) + "\n")
        _sync_log(log)
        _atomic_json(out / "status.json", {"state": "paused", "step": step,
                                           "updated": time.time()}, durable=True)
        write_loop_telemetry(out / "loop_rw.json", wrappers, step=step)
        print(f"paused: {step} steps; exact checkpoint {checkpoint_path}", flush=True)
        interrupted = True
    except BaseException as error:
        preload_stop.set()
        if prefetch_future is not None:
            prefetch_future.cancel()
        recovery_error = None
        recovered = False
        if checkpoint_state_valid:
            try:
                # Best effort: preserve the dashboard row before publishing a
                # recovery checkpoint, but do not sacrifice exact model state
                # merely because the log descriptor itself is already broken.
                try:
                    _sync_log(log)
                except (OSError, ValueError):
                    pass
                save_last_checkpoint(step)
                recovered = True
                try:
                    log.write(json.dumps({
                        "kind": "checkpoint", "step": step,
                        "reason": "failure_recovery", "path": str(checkpoint_path),
                    }) + "\n")
                    _sync_log(log)
                except (OSError, ValueError):
                    pass
            except BaseException as save_error:
                recovery_error = f"{type(save_error).__name__}: {save_error}"
        failure_status = {
            "state": "failed", "step": step,
            "error": f"{type(error).__name__}: {error}",
            "exact_checkpoint_saved": recovered,
            "checkpoint_state_valid": checkpoint_state_valid,
            "updated": time.time(),
        }
        if recovery_error is not None:
            failure_status["checkpoint_error"] = recovery_error
        _atomic_json(out / "status.json", failure_status, durable=True)
        raise
    finally:
        preload_stop.set()
        if prefetch_future is not None:
            prefetch_future.cancel()
        try:
            # Keep dashboard history aligned with the last durable checkpoint
            # across host failure. This is one shutdown sync, not a per-step
            # barrier; periodic checkpoint paths sync independently above.
            _sync_log(log)
        except (OSError, ValueError):
            pass
        log.close()

    if interrupted:
        return

    # Reaching the target is itself a commit boundary. Keep an operator SIGINT
    # pending until the final checkpoint and terminal status agree; otherwise a
    # pause in this small window leaves a finished run advertised as training
    # and invites the watchdog to launch it again.
    previous_mask = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGINT})
    try:
        _atomic_json(out / "status.json", {"state": "checkpointing", "step": step,
                                           "reason": "final", "updated": time.time()})
        if _final_checkpoint_required(step, last_checkpoint_step):
            save_last_checkpoint(step)
        write_loop_telemetry(out / "loop_rw.json", wrappers, step=step)
        _atomic_json(out / "status.json", {"state": "complete", "step": step,
                                           "updated": time.time()}, durable=True)
    except BaseException:
        _atomic_json(out / "status.json", {"state": "failed", "step": step,
                                           "phase": "final_checkpoint",
                                           "updated": time.time()}, durable=True)
        raise
    finally:
        signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)
    print(f"complete: {step} steps; checkpoint {checkpoint_path}", flush=True)


if __name__ == "__main__":
    main()
