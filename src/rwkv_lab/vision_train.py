"""Train a frozen MoonViT -> RWKV caption model with recoverable runs.

The pretrained MoonViT and RWKV weights remain frozen.  Training updates only
the image-prefix projector, NextLat auxiliary predictor, per-layer factored
TimeMix loop adapters, and optional Engram recall adapters. Runs checkpoint
atomically and resume automatically, including the exact sampler and RNG state.
"""
from __future__ import annotations

import argparse
import atexit
import copy
import contextlib
import fcntl
import gc
import hashlib
import heapq
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
from collections import Counter, OrderedDict
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Sequence

import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn

from rwkv_lab.activation_checkpointing import selective_activation_checkpointing
from rwkv_lab.generate import SEP, WorldVocab
from rwkv_lab.fused_ce import HAS_FUSED_CE, weighted_logits_cross_entropy
from rwkv_lab.deep_vision import DeepVisionInjector, LayerMatchedVisionInjector
from rwkv_lab.engram_lmb import (
    BatchedStreamingEngramState,
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
    capture_loop_refinement_caches,
    install_factored_timemix,
    load_loop_adapter_state,
    loop_adapter_state,
    loop_telemetry_payload,
    loop_training_metric_tensors,
    reset_loop_adapters,
    reset_loop_inference_cache,
    restore_loop_refinement_caches,
    select_legacy_fla_cache_rows,
    select_loop_refinement_cache_rows,
    set_loop_enabled,
    set_loop_scale,
    stack_legacy_fla_caches,
    stack_loop_refinement_cache_snapshots,
    write_loop_telemetry,
    write_loop_telemetry_payload,
)
from rwkv_lab.vision_grounding import ImageTextContrastiveHead, early_token_weights
from rwkv_lab.vision_ocr import ocr_generation_metrics
from rwkv_lab.vision_structured import (
    BOX_RE,
    StructuredSpatialHead,
    parse_structured_target,
    structured_detection_loss,
    structured_generation_metrics,
    structured_prediction_instances,
    structured_target_from_row,
)
from rwkv_lab.vision_fusion import (
    AlignedFrozenVisionFeatures,
    SamAlignedFrozenFeatures,
    VisionFusionResidual,
    VisionTowerConfig,
    aligned_feature_cache_key,
    valid_aligned_feature,
)
from rwkv_lab.vision_compressor_features import (
    CanonicalLatentPrefixProjector,
    FrozenTeacherCompressor,
)
from rwkv_lab.radio1d_cache import (
    cache_is_current as radio_cache_is_current,
    cache_path as radio_cache_path,
    load_cache as load_radio_cache,
    make_metadata as make_radio_metadata,
    save_cache as save_radio_cache,
)
from rwkv_lab.radio1d_rwkv import (
    DEFAULT_ADAPTIVE_TOKEN_THRESHOLD,
    DEFAULT_COMPLEXITY_BUDGET_RATIO,
    DEFAULT_COMPLEXITY_TOKEN_QUANTUM,
    DEFAULT_MAX_DETAIL_TILES,
    RadioFeatureProjector,
    RadioPrefixInjector,
    build_radio_tiles,
    choose_detail_grid,
    encode_radio_tiles,
    fourier_box_features,
    load_radio1d_h,
    adaptive_tokens_per_tile,
    tokens_per_tile_for_tile_count,
)

ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT_SCHEMA = 3
EVAL_RESTART_EXIT_CODE = 42
_CACHE_LOAD_POOL = ThreadPoolExecutor(max_workers=16, thread_name_prefix="vision-cache")
# Background warming must leave I/O and CPU headroom for the exact next-batch
# prefetch. Sixty-four concurrent torch ZIP readers previously flooded the same
# disk queue the trainer was waiting on, making "background" preload a foreground
# stall on a cold cache.
_FEATURE_PRELOAD_POOL = ThreadPoolExecutor(max_workers=8, thread_name_prefix="vision-preload")
_NEXT_BATCH_POOL = ThreadPoolExecutor(max_workers=1, thread_name_prefix="vision-next-batch")
_TELEMETRY_WRITE_POOL = ThreadPoolExecutor(
    max_workers=1, thread_name_prefix="vision-telemetry")
_CHECKPOINT_WRITE_POOL = ThreadPoolExecutor(
    max_workers=1, thread_name_prefix="vision-checkpoint")
# RADIO is a frozen eval-only tower, but its remote implementation is not
# documented as re-entrant.  Prefetch may overlap RWKV work; serialize RADIO
# calls themselves so evaluation and a cache worker can never mutate/read the
# same tower concurrently.
_RADIO_ENCODE_LOCK = threading.Lock()


@dataclass(frozen=True)
class BatchPrefetchResult:
    """Observable result of preparing exactly one sampler-peeked batch."""

    ready: int
    recall: RecallResult | None
    resident_hits: int = 0
    disk_hits: int = 0
    generated: int = 0
    elapsed_s: float = 0.0
    native_features: tuple[torch.Tensor, ...] | None = None
    text_batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None
    positions: torch.Tensor | None = None


def _cuda_recall(
        recall: RecallResult | None, device: torch.device) -> RecallResult | None:
    if recall is None:
        return None
    return RecallResult(*(value.to(device=device, non_blocking=True)
                          for value in recall))


class _CudaBatchPrefetch:
    """Stage a completed pinned CPU batch on a dedicated CUDA stream."""

    def __init__(self, cpu_future, *, device: torch.device,
                 stream: torch.cuda.Stream):
        self.cpu_future = cpu_future
        self.device = device
        self.stream = stream
        self.ready = threading.Event()
        self.cuda_event = torch.cuda.Event()
        self.value: BatchPrefetchResult | None = None
        self.error: BaseException | None = None
        cpu_future.add_done_callback(self._stage)

    def _stage(self, future) -> None:
        try:
            cpu = future.result()
            with torch.cuda.device(self.device), torch.cuda.stream(self.stream):
                native = (
                    None if cpu.native_features is None else
                    tuple(value.to(
                        device=self.device, non_blocking=True)
                          for value in cpu.native_features)
                )
                text_batch = (
                    None if cpu.text_batch is None else
                    tuple(value.to(
                        device=self.device, non_blocking=True)
                          for value in cpu.text_batch)
                )
                positions = (
                    None if cpu.positions is None else
                    cpu.positions.to(device=self.device, non_blocking=True)
                )
                self.value = replace(
                    cpu, native_features=native, text_batch=text_batch,
                    positions=positions,
                    recall=_cuda_recall(cpu.recall, self.device))
                self.cuda_event.record(self.stream)
        except BaseException as error:
            self.error = error
        finally:
            self.ready.set()

    def result(self) -> BatchPrefetchResult:
        self.ready.wait()
        if self.error is not None:
            raise self.error
        if self.value is None:
            raise RuntimeError("CUDA prefetch completed without a batch")
        current = torch.cuda.current_stream(self.device)
        current.wait_event(self.cuda_event)
        for value in (
                *(self.value.native_features or ()),
                *(self.value.text_batch or ()),
                *((self.value.positions,) if self.value.positions is not None else ()),
                *(tuple(self.value.recall) if self.value.recall is not None else ())):
            value.record_stream(current)
        return self.value

    def cancel(self) -> bool:
        return self.cpu_future.cancel()


def _pin_tensor(value: torch.Tensor) -> torch.Tensor:
    """Pin a CPU tensor when CUDA host registration is available."""
    if value.device.type != "cpu" or value.is_pinned():
        return value
    try:
        return value.pin_memory()
    except RuntimeError:
        # CPU-only tests and a host that has exhausted its pinned-memory quota
        # retain the exact pageable fallback.
        return value


def _pin_recall(recall: RecallResult | None) -> RecallResult | None:
    if recall is None:
        return None
    return RecallResult(*(_pin_tensor(value) for value in recall))


class _BoundedFeatureCache:
    """Thread-safe byte-bounded LRU for resident cached feature tensors.

    Without a bound, every feature ever loaded stayed resident for the whole
    process (~590KB/image MoonViT + ~560KB/image fusion), growing RSS on large
    corpora until the OOM killer ended the run. ``max_bytes == 0`` disables
    eviction; ``--preload-feature-cache`` opts into that unbounded mode because
    the operator asserts the corpus fits in system RAM. Prefetch/preload run on
    background threads, so every access holds the lock.
    """

    def __init__(self, max_bytes: int = 16 * 2**30) -> None:
        self.max_bytes = int(max_bytes)
        self._lock = threading.Lock()
        self._items: OrderedDict[Path, object] = OrderedDict()
        self._sizes: dict[Path, int] = {}
        self._total_bytes = 0

    @staticmethod
    def _entry_bytes(item) -> int:
        if isinstance(item, (tuple, list)):
            return sum(_BoundedFeatureCache._entry_bytes(part) for part in item)
        if torch.is_tensor(item):
            return item.numel() * item.element_size()
        return 0

    @property
    def total_bytes(self) -> int:
        with self._lock:
            return self._total_bytes

    def __contains__(self, key) -> bool:
        with self._lock:
            return key in self._items

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)

    def get(self, key, default=None):
        with self._lock:
            if key not in self._items:
                return default
            self._items.move_to_end(key)
            return self._items[key]

    def pop(self, key, default=None):
        with self._lock:
            if key not in self._items:
                return default
            self._total_bytes -= self._sizes.pop(key, 0)
            return self._items.pop(key)

    def clear(self) -> None:
        with self._lock:
            self._items.clear()
            self._sizes.clear()
            self._total_bytes = 0

    def setdefault(self, key, item):
        with self._lock:
            if key in self._items:
                self._items.move_to_end(key)
                return self._items[key]
            self._store_locked(key, item)
            return item

    def __setitem__(self, key, item) -> None:
        with self._lock:
            self._store_locked(key, item)

    def _store_locked(self, key, item) -> None:
        if key in self._items:
            self._total_bytes -= self._sizes.pop(key, 0)
            del self._items[key]
        self._items[key] = item
        size = self._entry_bytes(item)
        self._sizes[key] = size
        self._total_bytes += size
        if self.max_bytes <= 0:
            return
        # Keep at least the newest entry: callers already hold a reference to
        # it, and a single oversized item must not thrash the whole cache.
        while self._total_bytes > self.max_bytes and len(self._items) > 1:
            old_key, _ = self._items.popitem(last=False)
            self._total_bytes -= self._sizes.pop(old_key, 0)


_FEATURE_MEMORY_CACHE = _BoundedFeatureCache()


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
                             dtype: torch.dtype, storage_bytes: int,
                             on_fallback: Callable[[], None] | None = None
                             ) -> torch.Tensor:
    """Load a one-storage ``torch.save`` tensor without unpickling each file.

    ``on_fallback`` is invoked whenever the raw fast path cannot be used and
    the safe (slow) ``torch.load`` compatibility loader runs instead, so
    callers can surface how much of a corpus silently lost the speedup.
    """
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
        if on_fallback is not None:
            on_fallback()
        return torch.load(path, map_location="cpu", weights_only=True)
    except (FileNotFoundError, OSError, ValueError):
        # Preserve layout/dtype compatibility through the safe general loader.
        # Callers additionally verify the loaded storage against the ZIP CRC.
        if on_fallback is not None:
            on_fallback()
        return torch.load(path, map_location="cpu", weights_only=True)


def load_examples(path: str | Path, *, root: Path = ROOT,
                  stat_workers: int = 1,
                  require_all: bool = False) -> list[dict]:
    """Read valid image-caption rows without silently accepting missing files."""
    candidates = []
    invalid_fields: list[int] = []
    source = Path(path)
    # Iterate physical lines. str.splitlines() also splits valid JSON strings at
    # Unicode line-separator characters (U+2028/U+2029), corrupting such rows.
    with source.open() as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not row.get("image") or not row.get("text"):
                invalid_fields.append(line_number)
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
    if require_all and (invalid_fields or len(rows) != len(candidates)):
        missing = len(candidates) - len(rows)
        examples = invalid_fields[:5]
        raise ValueError(
            f"manifest {source} rejected {missing} missing/unreadable images and "
            f"{len(invalid_fields)} rows without image/text"
            + (f" (invalid field lines: {examples})" if examples else ""))
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
                     sandwich_prompt: bool = False,
                     sandwich_lead_prompt: str = "") -> tuple[list[dict], list[int]]:
    """Tokenize once, append EOD, and retain EOD when a caption is truncated."""
    prompt_cache: dict[str, list[int]] = {}
    prepared, lengths = [], []
    for row in rows:
        row_prompt = str(row.get("prompt") or prompt)
        prompt_tokens = prompt_cache.get(row_prompt)
        if prompt_tokens is None:
            prompt_tokens = vocab.encode(row_prompt)
            prompt_cache[row_prompt] = prompt_tokens
        if sandwich_prompt:
            lead_text = sandwich_lead_prompt or row_prompt
            lead_tokens = prompt_cache.get(lead_text)
            if lead_tokens is None:
                lead_tokens = vocab.encode(lead_text)
                prompt_cache[lead_text] = lead_tokens
        else:
            lead_tokens = []
        prompt_width = len(lead_tokens) + len(prompt_tokens)
        if prompt_width + 2 > max_text_tokens:
            raise ValueError("max_text_tokens is too small for prompt + one caption token + EOD")
        room = max_text_tokens - prompt_width - 1
        caption_text = str(row["text"])
        caption = vocab.encode(caption_text)
        if not caption:
            continue
        tokens = lead_tokens + prompt_tokens + caption[:room] + [SEP]
        item = dict(row)
        item["tokens"] = tokens
        item["prompt_len"] = prompt_width
        item["vision_insert"] = len(lead_tokens) if sandwich_prompt else 0
        item["prompt"] = row_prompt
        item["truncated"] = len(caption) > room
        if str(row.get("task") or "") == "sam_mask":
            coordinate_tokens, coordinate_starts = (
                structured_coordinate_token_offsets(
                    caption_text, caption[:room], vocab))
            item["structured_coordinate_tokens"] = tuple(
                prompt_width + offset for offset in coordinate_tokens)
            item["structured_coordinate_starts"] = tuple(
                prompt_width + offset for offset in coordinate_starts)
            item["structured_boundary_lead_tokens"] = tuple(
                vocab.encode(value)[0] for value in ("000", "999"))
        prepared.append(item)
        lengths.append(len(tokens))
    return prepared, lengths


def structured_coordinate_token_offsets(
        text: str, tokens: Sequence[int], vocab: WorldVocab,
        ) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Locate token offsets covering the four numeric fields of every box.

    Offsets are computed in bytes because WorldVocab is byte-tokenized while
    labels before ``box=`` may contain non-ASCII category names.  The returned
    start offsets identify the first (normally two-digit) token of each
    coordinate, which lets the LM objective explicitly suppress its common
    ``00``/``99`` collapse without banning real edge-touching targets.
    """
    spans = []
    for match in BOX_RE.finditer(text):
        for group in range(1, 5):
            char_start, char_end = match.span(group)
            spans.append((len(text[:char_start].encode("utf-8")),
                          len(text[:char_end].encode("utf-8"))))
    if not spans or not tokens:
        return (), ()
    token_spans = []
    cursor = 0
    for token in tokens:
        value = vocab.tok[int(token)]
        token_spans.append((cursor, cursor + len(value)))
        cursor += len(value)
    covered, starts = set(), []
    for span_start, span_end in spans:
        first = None
        for index, (token_start, token_end) in enumerate(token_spans):
            if token_end <= span_start:
                continue
            if token_start >= span_end:
                break
            covered.add(index)
            if first is None:
                first = index
        if first is not None:
            starts.append(first)
    return tuple(sorted(covered)), tuple(starts)


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


def multitask_balanced_indices(
        rows: Sequence[dict], indices: Sequence[int], *,
        target_ratios: dict[str, float], seed: int) -> list[int]:
    """Repeat minority-task slots to satisfy simultaneous epoch shares."""
    output = list(indices)
    targets = {
        task.casefold(): float(ratio)
        for task, ratio in target_ratios.items() if ratio
    }
    if any(not 0 < ratio < 1 for ratio in targets.values()):
        raise ValueError("target task ratios must be in (0, 1)")
    if sum(targets.values()) >= 1:
        raise ValueError("target task ratios must sum to less than one")
    if not output or not targets:
        return output
    matching = {
        task: [
            index for index in output
            if str(rows[index].get("task") or "caption").casefold() == task
        ]
        for task in targets
    }
    absent = [task for task, values in matching.items() if not values]
    if absent:
        raise ValueError(f"cannot balance absent tasks {absent}")
    slots = {task: len(values) for task, values in matching.items()}
    other = len(output) - sum(slots.values())
    while True:
        total = other + sum(slots.values())
        updated = {
            task: max(slots[task], math.ceil(ratio * total))
            for task, ratio in targets.items()
        }
        if updated == slots:
            break
        slots = updated
    for task in sorted(targets):
        values = matching[task]
        ranked = sorted(
            values,
            key=lambda index: hashlib.sha256(
                f"{seed}:{task}:{rows[index].get('image')}:{index}".encode()
            ).digest(),
        )
        extra = slots[task] - len(values)
        output.extend(ranked[offset % len(ranked)] for offset in range(extra))
    return output


def task_balanced_indices(rows: Sequence[dict], indices: Sequence[int], *,
                          task: str, target_ratio: float,
                          seed: int) -> list[int]:
    """Compatibility wrapper for balancing one task."""
    if not 0 <= target_ratio < 1:
        raise ValueError("target task ratio must be in [0, 1)")
    return multitask_balanced_indices(
        rows, indices,
        target_ratios=({task: target_ratio} if target_ratio else {}),
        seed=seed)


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
                 seed: int, bucket_batches: int = 32,
                 group_keys: Sequence[int] | None = None):
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.indices = list(indices)
        self.lengths = list(lengths)
        self.batch_size = int(batch_size)
        self.bucket_batches = max(1, int(bucket_batches))
        self.group_keys = list(group_keys) if group_keys is not None else None
        if self.group_keys is not None and len(self.group_keys) != len(self.lengths):
            raise ValueError("sampler group keys must align with lengths")
        self.generator = torch.Generator().manual_seed(seed)
        self.epoch = 0
        self.position = 0
        self.order: list[int] = []
        self._new_epoch()

    def _new_epoch(self) -> None:
        shuffled = [self.indices[i] for i in torch.randperm(len(self.indices), generator=self.generator).tolist()]
        window = self.batch_size * self.bucket_batches
        ordered: list[int] = []
        if self.group_keys is None:
            for start in range(0, len(shuffled), window):
                ordered.extend(sorted(shuffled[start:start + window],
                                      key=self.lengths.__getitem__))
        else:
            by_key: dict[int, list[tuple[int, int]]] = {}
            for occurrence, index in enumerate(shuffled):
                by_key.setdefault(
                    int(self.group_keys[index]), []).append((occurrence, index))
            # Preserve the global shuffled order as closely as exact-shape
            # batching permits. The prior round-robin-over-keys policy gave a
            # tiny 2-row shape the same early-epoch weight as a 2,000-row
            # shape; because OCR/structured sources correlate with geometry,
            # the first several thousand optimizer steps received less than
            # half their configured structured share. A heap keyed by each
            # group's next global-shuffle rank keeps bounded same-shape runs
            # while retaining row-proportional prefix distributions.
            # The run length is the same ``window`` the round-robin policy used.
            # Proportionality is the heap's job, not the run's, and shortening
            # the run silently costs throughput instead: ``peek_budget_batch``
            # truncates a budget batch at the first group-key change, so a run
            # of ``r`` rows caps every batch at ``r`` however much the token
            # budget and ``--max-batch`` would otherwise admit.
            offsets = {key: 0 for key in by_key}
            heap = [(group[0][0], key) for key, group in by_key.items()]
            heapq.heapify(heap)
            run = window
            while heap:
                _, key = heapq.heappop(heap)
                start = offsets[key]
                end = min(start + run, len(by_key[key]))
                ordered.extend(sorted(
                    (index for _, index in by_key[key][start:end]),
                    key=self.lengths.__getitem__))
                offsets[key] = end
                if end < len(by_key[key]):
                    heapq.heappush(heap, (by_key[key][end][0], key))
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
            if self.group_keys is not None:
                first_key = self.group_keys[self.order[position]]
                while end > position + 1 and any(
                        self.group_keys[index] != first_key
                        for index in self.order[position:end]):
                    end -= 1
            return self.order[position:end]
        if max_items == min_items:
            end = min(position + min_items, len(self.order))
            if self.group_keys is not None:
                first_key = self.group_keys[self.order[position]]
                while end > position + 1 and self.group_keys[
                        self.order[end - 1]] != first_key:
                    end -= 1
            return self.order[position:end]
        available = min(max_items, len(self.order) - position)
        if self.group_keys is not None:
            first_key = self.group_keys[self.order[position]]
            available = next((offset for offset in range(1, available)
                              if self.group_keys[self.order[position + offset]]
                              != first_key), available)
        take = min(min_items, available)
        max_cost = max(token_costs[index]
                       for index in self.order[position:position + take])
        # Batches are locally length-sorted. Account for padding using the
        # actual longest item, including when a batch crosses a bucket boundary.
        while take < available:
            candidate = self.order[position + take]
            if (self.group_keys is not None
                    and self.group_keys[candidate]
                    != self.group_keys[self.order[position]]):
                break
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
    # Assemble the padded batch on the host: per-row torch.tensor(...,
    # device="cuda") issued one synchronous transfer per caption. Three pinned
    # bulk copies replace them without changing any produced value.
    ids = torch.zeros(len(rows), width, dtype=torch.long)
    labels = torch.full_like(ids, -100)
    mask = torch.zeros_like(ids, dtype=torch.bool)
    for i, row in enumerate(rows):
        tokens = torch.as_tensor(row["tokens"], dtype=torch.long)
        length, prompt_len = len(tokens), int(row["prompt_len"])
        ids[i, :length] = tokens
        labels[i, prompt_len:length] = tokens[prompt_len:]
        mask[i, :length] = True
    target = torch.device(device)
    if target.type == "cuda":
        ids = ids.pin_memory().to(target, non_blocking=True)
        labels = labels.pin_memory().to(target, non_blocking=True)
        mask = mask.pin_memory().to(target, non_blocking=True)
    else:
        ids, labels, mask = ids.to(target), labels.to(target), mask.to(target)
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
    if any(start < 0 or start > text.shape[1] for start in starts):
        raise ValueError("visual insertion falls outside text sequence")
    if len(set(starts)) == 1:
        # Uniform insertion point (every non-sandwich batch): one batched cat
        # instead of a per-row cat plus a stack.
        start = starts[0]
        return torch.cat((text[:, :start], visual, text[:, start:]), dim=1)
    rows = []
    for batch, start in enumerate(starts):
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


def visual_prefix_width(rows: Sequence[dict], projector: nn.Module) -> int:
    widths = {int(row.get("_visual_tokens", 0)) for row in rows
              if row.get("_visual_tokens") is not None}
    if widths:
        if len(widths) != 1:
            raise ValueError("visual batch contains different RADIO token counts")
        width = widths.pop()
        if width < 1:
            raise ValueError("RADIO visual token width must be positive")
        return width
    return int(projector.prefix_tokens)


def cached_features(rows: Sequence[dict], vision: MoonViT,
                    projector: MoonViTPrefixProjector, cache_dir: Path | None) -> list[torch.Tensor]:
    """Load pooled frozen features, batch-encoding cache misses by image grid."""
    result: list[torch.Tensor | None] = [None] * len(rows)
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
    missing: list[tuple[int, Path | None, Image.Image]] = []
    device = vision.patch_embed.proj.weight.device
    fingerprint = getattr(vision, "cache_fingerprint", "unknown")
    stages = int(getattr(vision, "feature_stages", 0))
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


def cached_radio_features(
        rows: Sequence[dict], vision: nn.Module, cache_dir: Path,
        *, revision: str, max_detail_tiles: int, tile_batch: int,
        adaptive_token_threshold: int,
        telemetry: dict[str, int] | None = None,
        ) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Load or atomically create tiled RADIO globals in stable row order.

    All cache misses in an exact-tile-count training batch are flattened into
    one tile stream.  This fills ``tile_batch`` forwards across image
    boundaries instead of issuing one under-filled RADIO call per image.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    output: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None] = [
        None] * len(rows)
    misses: list[tuple[int, Path, object, list]] = []
    resident_hits = disk_hits = 0
    for index, row in enumerate(rows):
        source = Path(row["image"])
        target = radio_cache_path(cache_dir, source)
        resident = _FEATURE_MEMORY_CACHE.get(target)
        if resident is not None:
            output[index] = resident
            resident_hits += 1
            continue
        if radio_cache_is_current(
                target, source, revision,
                adaptive_token_threshold=adaptive_token_threshold,
                source_sha256=row.get("image_sha256")):
            metadata, tokens = load_radio_cache(target)
            boxes = torch.tensor(
                [tile.source_box for tile in metadata.tiles], dtype=torch.float32)
            roles = torch.tensor(
                [0 if tile.role == "thumbnail" else 1 for tile in metadata.tiles],
                dtype=torch.long)
            item = (tokens, boxes, roles)
            _FEATURE_MEMORY_CACHE[target] = item
            output[index] = item
            disk_hits += 1
        else:
            with Image.open(source) as image:
                tiles = build_radio_tiles(
                    image, max_detail_tiles=max_detail_tiles)
            metadata = make_radio_metadata(
                source, revision, tiles,
                adaptive_token_threshold=adaptive_token_threshold)
            misses.append((index, target, metadata, tiles))

    tile_counts = {
        int(item[0].shape[0]) for item in output if item is not None}
    tile_counts.update(len(tiles) for _, _, _, tiles in misses)
    if len(tile_counts) > 1:
        raise ValueError("RADIO batch crossed an exact tile-count bucket")
    token_counts = {
        int(item[0].shape[1]) for item in output if item is not None}
    token_counts.update(tokens_per_tile_for_tile_count(
        len(tiles), threshold=adaptive_token_threshold)
        for _, _, _, tiles in misses)
    if len(token_counts) > 1:
        raise ValueError("RADIO batch crossed an adaptive token-policy bucket")

    if misses:
        flat_tiles = [tile for _, _, _, tiles in misses for tile in tiles]
        tokens_per_tile = token_counts.pop()
        # The shared default CUDA stream preserves ordering with the trainer's
        # RWKV kernels.  The lock additionally protects the remote RADIO module
        # from a simultaneous evaluation/cache forward in another host thread.
        with _RADIO_ENCODE_LOCK:
            packed = encode_radio_tiles(
                vision, flat_tiles, batch_size=tile_batch,
                num_tokens=tokens_per_tile)
        offset = 0
        for index, target, metadata, tiles in misses:
            end = offset + len(tiles)
            tokens = packed[offset:end].contiguous()
            offset = end
            save_radio_cache(target, metadata, tokens)
            boxes = torch.tensor(
                [tile.source_box for tile in metadata.tiles], dtype=torch.float32)
            roles = torch.tensor(
                [0 if tile.role == "thumbnail" else 1 for tile in metadata.tiles],
                dtype=torch.long)
            item = (tokens, boxes, roles)
            _FEATURE_MEMORY_CACHE[target] = item
            output[index] = item
        if offset != packed.shape[0]:
            raise RuntimeError("packed RADIO cache split did not consume every tile")

    if telemetry is not None:
        telemetry.update(resident_hits=resident_hits, disk_hits=disk_hits,
                         generated=len(misses))
    if any(item is None for item in output):
        raise RuntimeError("RADIO cache loader left an unresolved row")
    return list(output)  # type: ignore[return-value]


def runtime_cached_features(rows: Sequence[dict], vision: nn.Module,
                            projector: nn.Module,
                            cache_dir: Path | None):
    if isinstance(projector, RadioFeatureProjector):
        if cache_dir is None:
            raise ValueError("RADIO training requires a resumable feature cache")
        if getattr(vision, "v4h_native", False):
            return cached_native_v4h_features(
                rows, vision, Path(getattr(
                    vision, "v4h_cache_dir", cache_dir)))
        if hasattr(vision, "v4h_lattice"):
            from rwkv_lab.radio_v4h import load_v4h_features
            # Read-only: a cache miss raises rather than encoding inline, so the
            # prefetch worker can never touch CUDA from a foreign thread.
            return load_v4h_features(
                rows, Path(getattr(vision, "v4h_cache_dir", cache_dir)),
                revision=str(getattr(vision, "radio_revision")),
                lattice=int(getattr(vision, "v4h_lattice")), root=ROOT,
                pair_axis=str(getattr(vision, "v4h_pair_axis", "columns")))
        return cached_radio_features(
            rows, vision, cache_dir,
            revision=str(getattr(vision, "radio_revision")),
            max_detail_tiles=int(getattr(vision, "radio_max_detail_tiles")),
            tile_batch=int(getattr(vision, "radio_tile_batch")),
            adaptive_token_threshold=int(getattr(
                vision, "radio_adaptive_token_threshold")))
    return cached_features(rows, vision, projector, cache_dir)


def cached_native_v4h_features(
        rows: Sequence[dict], vision: nn.Module, cache_dir: Path,
        *, telemetry: dict[str, int] | None = None
        ) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Load native V4H features through the shared bounded resident LRU.

    Both next-batch prefetch and the foreground consume this exact function.
    Consequently a reported disk hit means the worker really decoded and
    packed a safetensors grid, while the foreground normally records a resident
    hit instead of repeating that I/O.
    """
    from rwkv_lab.radio_v4h import cache_path, load_native_features

    cache_dir = Path(cache_dir)
    revision = str(getattr(vision, "radio_revision"))
    max_edge = int(getattr(vision, "v4h_max_edge", 2048))
    hidden_size = int(getattr(vision, "v4h_feature_width", 1280))
    packing = str(getattr(vision, "v4h_native_packing", "pair_columns"))
    output: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None] = [
        None] * len(rows)
    misses: list[tuple[int, dict, tuple[Path, str, int, int, str]]] = []
    resident_hits = 0
    for index, row in enumerate(rows):
        source = Path(row["image"])
        source = source if source.is_absolute() else ROOT / source
        target = cache_path(cache_dir, source)
        memory_key = (target, revision, max_edge, hidden_size, packing)
        resident = _FEATURE_MEMORY_CACHE.get(memory_key)
        if resident is None:
            misses.append((index, row, memory_key))
        else:
            output[index] = resident
            resident_hits += 1

    def load_one(entry: tuple[
            int, dict, tuple[Path, str, int, int, str]]):
        index, row, memory_key = entry
        item = load_native_features(
            [row], cache_dir, revision=revision, root=ROOT,
            max_edge=max_edge, hidden_size=hidden_size, packing=packing)[0]
        return index, memory_key, item

    disk_hits = 0
    for index, memory_key, item in _CACHE_LOAD_POOL.map(load_one, misses):
        output[index] = _FEATURE_MEMORY_CACHE.setdefault(memory_key, item)
        disk_hits += 1
    if telemetry is not None:
        telemetry.update(
            resident_hits=resident_hits, disk_hits=disk_hits, generated=0)
    if any(item is None for item in output):
        raise RuntimeError("native V4H cache loader left an unresolved row")
    return list(output)  # type: ignore[return-value]


def _row_fusion_cache_path(row: dict, cache_dir: Path, *, tokens: int,
                           tower_fingerprint: str) -> Path:
    return cache_dir / aligned_feature_cache_key(
        row["image"], tokens=tokens, tower_fingerprint=tower_fingerprint,
        source_size=row.get("_source_size"),
        source_mtime_ns=row.get("_source_mtime_ns"))


def image_aspect_tensor(rows: Sequence[dict],
                        device: torch.device | str) -> torch.Tensor | None:
    """Per-row width/height, the missing half of RADIO's tile geometry.

    ``source_box`` says which image region a tile covers, but every tile is
    letterboxed into a square canvas and that transform depends on the source
    aspect ratio. Without it, a coordinate in feature space cannot be mapped to
    a coordinate in the image frame that ``box=``/``mask16=`` targets use.
    Returns ``None`` when any row lacks dimensions, so the bridge stays on its
    previous behavior rather than silently inventing geometry.
    """
    values = []
    for row in rows:
        width, height = row.get("_image_width"), row.get("_image_height")
        if not width or not height:
            return None
        values.append(float(width) / float(height))
    return torch.tensor(values, dtype=torch.float32, device=device)


from rwkv_lab.radio_v4h import DEFAULT_NATIVE_REVISION as _DEFAULT_NATIVE_REVISION

RADIO_BACKENDS = ("radio1d", "radio_v4h")


def is_radio_backend(args: argparse.Namespace) -> bool:
    """Both RADIO backends share tiling, sampler bucketing and token costing."""
    return getattr(args, "vision_backend", "") in RADIO_BACKENDS


def uses_native_prefix(args: argparse.Namespace) -> bool:
    """True when the visual prefix is a native grid with no tile structure.

    Every consumer that would otherwise assume a tiled prefix -- the projector
    forward and the deep-vision reinjection guard -- must ask this one
    question, so the two cannot drift apart.
    """
    return (getattr(args, "vision_backend", "") == "radio_v4h"
            and bool(getattr(args, "radio_v4h_native", False)))


def native_visual_token_count(grid_h: int, grid_w: int, *, packing: str) -> int:
    """Tokens one native C-RADIOv4-H grid contributes under ``packing``.

    ``pair_columns`` pairs adjacent cells within a row, which is
    ``grid_h * (grid_w // 2)`` -- not ``(grid_h * grid_w) // 2``. The two differ
    for every odd width (3x5 gives 6 against 7), and the sampler's bucket key is
    computed from this number, so an approximation here surfaces as a shape
    error deep inside ``forward_native`` instead of a plan that never runs.
    ``native_token_boxes`` rejects an odd width outright, so this refuses it at
    planning time with the geometry named.
    """
    if grid_h < 1 or grid_w < 1:
        raise ValueError(f"native grid {grid_h}x{grid_w} is not positive")
    if packing != "pair_columns":
        return grid_h * grid_w
    if grid_w % 2:
        raise ValueError(
            f"a {grid_h}x{grid_w} native grid cannot be column-paired; "
            "--radio-v4h-native-packing pair_columns needs an even grid width")
    return grid_h * (grid_w // 2)


def uses_v4h_adaptors(args: argparse.Namespace) -> bool:
    """Whether this run's native cache was written by the fused adaptor stack.

    The answer decides which artifacts the cache fingerprint covers, so it has
    to be about the adaptors themselves rather than about a width that happens
    to correlate with them today: a 4096-wide generic cache, or a fused cache
    built at some other width, would otherwise fingerprint the wrong file set
    and quietly stop guarding against a stale cache. ``--radio-v4h-adaptors``
    states it; the inferred default reproduces every existing invocation.
    """
    from rwkv_lab.radio_v4h_adaptors import FUSED_ADAPTOR_WIDTH

    explicit = getattr(args, "radio_v4h_adaptors", None)
    if explicit is not None:
        return bool(explicit)
    return int(getattr(args, "radio_v4h_feature_width", 0)) == FUSED_ADAPTOR_WIDTH


def v4h_run_artifact_fingerprint(args: argparse.Namespace) -> str:
    """Fingerprint the encoder artifacts that produced this run's features.

    Under ``--radio-v4h-cache-only`` the encoder is never loaded, and a box that
    trains from a populated cache need not hold the multi-GB checkpoint at all.
    Hashing artifacts this process will never read therefore cannot be a startup
    precondition. When they are present the fingerprint is computed exactly as
    before -- so a cache-only resume keeps the value it saved -- and when they
    are absent the run degrades to the cache contract that
    ``radio_v4h_cache_fingerprint`` pins (cache path, revision, packing, widths,
    lattice), plus each entry's own producer metadata, which the loader checks
    per row. Only a non-cache-only run, which is about to load the encoder,
    still fails outright on missing artifacts.
    """
    from rwkv_lab.radio_v4h import v4h_artifact_fingerprint

    try:
        return v4h_artifact_fingerprint(
            args.radio_v4h_model,
            include_adaptors=uses_v4h_adaptors(args),
            fingerprint_cache=(
                Path(args.feature_cache) / ".v4h_artifact_fingerprint.json"))
    except FileNotFoundError:
        if not args.radio_v4h_cache_only:
            raise
        print("C-RADIOv4-H encoder artifacts are absent; --radio-v4h-cache-only "
              "pins this cache by its recorded contract instead", flush=True)
        return ""


def _vision_tower_config(args: argparse.Namespace) -> VisionTowerConfig:
    """One construction site, so the cache fingerprint cannot drift from the tower."""
    return VisionTowerConfig(
        siglip2=args.siglip2_model, dinov2=args.dinov2_model,
        sam=args.sam_model, siglip_width=args.siglip2_width,
        sam_crop_padding=bool(getattr(args, "sam_crop_padding", False)))


def fusion_feature_tokens(tower: nn.Module, projector: nn.Module) -> int:
    tokens = int(getattr(tower, "fusion_tokens",
                         getattr(projector, "prefix_tokens", 0)))
    if tokens < 1:
        raise ValueError("fusion tower has no positive fixed token span")
    return tokens


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
            print("loading frozen vision tower for fusion cache misses", flush=True)
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
    stages = int(getattr(vision, "feature_stages", 0))
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

    def load_one(path: Path) -> tuple[Path, torch.Tensor | None, bool]:
        fallback = [False]

        def note_fallback() -> None:
            fallback[0] = True

        try:
            if path == template_path:
                return path, template_item, False
            if layout is not None:
                item = _load_raw_tensor_archive(
                    path, **layout, on_fallback=note_fallback)
            else:
                fallback[0] = True
                item = torch.load(path, map_location="cpu", weights_only=True)
            if not _valid_pooled_feature_archive(
                    path, item, projector.prefix_tokens, stages):
                return path, None, fallback[0]
            return path, item, fallback[0]
        except (OSError, EOFError, RuntimeError, pickle.UnpicklingError,
                zipfile.BadZipFile):
            return path, None, fallback[0]

    loaded = 0
    resident_bytes = 0
    fallback_loads = 0
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
            path, item, used_fallback = future.result()
            if item is None:
                continue
            _FEATURE_MEMORY_CACHE[path] = item
            loaded += 1
            fallback_loads += int(used_fallback)
            resident_bytes += item.numel() * item.element_size()
            if loaded % 4096 == 0:
                print({"kind": "feature_preload", "loaded": loaded,
                       "total": len(pending)}, flush=True)
        if stop_event and stop_event.is_set():
            for future in in_flight:
                future.cancel()
            break
        fill()
    if pending:
        # A half-migrated cache silently loses the raw fast path per file;
        # make the split visible so operators notice the regression.
        print({"kind": "feature_preload_paths",
               "raw_fast_path": loaded - fallback_loads,
               "torch_load_fallback": fallback_loads,
               "total": len(pending)}, flush=True)
    return loaded, resident_bytes


def prefetch_cached_feature_rows(rows: Sequence[dict], vision: MoonViT,
                                 projector: MoonViTPrefixProjector,
                                 cache_dir: Path | None) -> int:
    """Bring an exact future batch into RAM without touching CUDA or sampler state."""
    if cache_dir is None:
        return 0
    fingerprint = getattr(vision, "cache_fingerprint", "unknown")
    stages = int(getattr(vision, "feature_stages", 0))
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


def prefetch_cached_radio_rows(
        rows: Sequence[dict], vision: nn.Module, cache_dir: Path, *,
        adaptive_token_threshold: int) -> tuple[int, int, int]:
    """Load existing RADIO archives into RAM without ever touching CUDA.

    A next-batch worker used to call ``cached_radio_features`` directly. On a
    miss that function encodes RADIO tiles on CUDA, but CUDA's current context
    and default stream are thread-local. The resulting invalid-context errors
    could poison the main training stream and surface later as an illegal
    access during evaluation. Cache creation belongs exclusively to the main
    trainer thread; a prefetch miss is simply left unresolved here.
    """
    revision = str(getattr(vision, "radio_revision"))

    def load_one(row: dict) -> tuple[
            Path, tuple[torch.Tensor, ...] | None, str]:
        source = Path(row["image"])
        target = radio_cache_path(cache_dir, source)
        resident = _FEATURE_MEMORY_CACHE.get(target)
        if resident is not None:
            return target, resident, "resident"
        try:
            if not radio_cache_is_current(
                    target, source, revision,
                    adaptive_token_threshold=adaptive_token_threshold,
                    source_sha256=row.get("image_sha256")):
                return target, None, "miss"
            metadata, tokens = load_radio_cache(target)
            boxes = torch.tensor(
                [tile.source_box for tile in metadata.tiles],
                dtype=torch.float32)
            roles = torch.tensor(
                [0 if tile.role == "thumbnail" else 1
                 for tile in metadata.tiles], dtype=torch.long)
            return target, (tokens, boxes, roles), "disk"
        except (OSError, EOFError, RuntimeError, pickle.UnpicklingError,
                zipfile.BadZipFile):
            return target, None, "miss"

    ready = resident_hits = disk_hits = 0
    for target, item, source in _CACHE_LOAD_POOL.map(load_one, rows):
        if item is not None:
            _FEATURE_MEMORY_CACHE.setdefault(target, item)
            ready += 1
            resident_hits += int(source == "resident")
            disk_hits += int(source == "disk")
    return ready, resident_hits, disk_hits


def prefetch_training_batch(rows: Sequence[dict], vision: nn.Module,
                            projector: nn.Module,
                            cache_dir: Path | None,
                            engram: LexicalMemoryBank | None,
                            fusion_tower: AlignedFrozenVisionFeatures | None = None,
                            fusion_cache_dir: Path | None = None,
                            ) -> BatchPrefetchResult:
    """Prepare exactly one future sampler batch without advancing its state."""
    started = time.perf_counter()
    stats: dict[str, int] = {}
    native_features = None
    if isinstance(projector, RadioFeatureProjector) and getattr(vision, "v4h_native", False):
        if cache_dir is None:
            raise ValueError("native V4H prefetch requires a resumable cache")
        loaded = cached_native_v4h_features(
            rows, vision, Path(getattr(
                vision, "v4h_cache_dir", cache_dir)), telemetry=stats)
        # Stack and pin on the worker while the current GPU step is running.
        # The foreground projector can consume this block directly instead of
        # restacking pageable per-row tensors on its critical path.
        native_features = tuple(_pin_tensor(torch.stack([
            item[part] for item in loaded])) for part in range(3))
        # Geometry is immutable for a cached native grid. Evaluate its
        # parameter-free sin/cos basis on the worker instead of launching trig
        # kernels on the critical CUDA stream every time this shape appears.
        box_fourier = fourier_box_features(native_features[1])
        native_features = (*native_features, _pin_tensor(box_fourier))
        ready = len(rows)
        visual_width = int({int(row["_visual_tokens"]) for row in rows}.pop())
    elif isinstance(projector, RadioFeatureProjector) and hasattr(vision, "v4h_lattice"):
        # v4h caches are pre-built and read-only; nothing to warm on the worker.
        stats.update(resident_hits=0, disk_hits=len(rows), generated=0)
        ready = len(rows)
        visual_width = int(getattr(vision, "v4h_lattice")) ** 2 * int(
            {int(row["_radio_tiles"]) for row in rows}.pop())
    elif isinstance(projector, RadioFeatureProjector):
        if cache_dir is None:
            raise ValueError("RADIO prefetch requires a resumable feature cache")
        threshold = int(getattr(vision, "radio_adaptive_token_threshold"))
        ready, resident_hits, disk_hits = prefetch_cached_radio_rows(
            rows, vision, cache_dir,
            adaptive_token_threshold=threshold)
        tile_counts = {int(row["_radio_tiles"]) for row in rows}
        if len(tile_counts) != 1:
            raise ValueError("prefetched RADIO rows do not share one tile count")
        tile_count = tile_counts.pop()
        maximum = tokens_per_tile_for_tile_count(
            tile_count, threshold=threshold)
        visual_width = tile_count * maximum
        if projector.adaptive_complexity:
            visual_width = tile_count * adaptive_tokens_per_tile(
                maximum, ratio=projector.complexity_budget_ratio,
                quantum=projector.complexity_token_quantum)
        stats.update(resident_hits=resident_hits, disk_hits=disk_hits,
                     generated=0)
    else:
        ready = prefetch_cached_feature_rows(
            rows, vision, projector, cache_dir)  # type: ignore[arg-type]
        visual_width = int(getattr(projector, "prefix_tokens"))
        stats["disk_hits"] = ready
    ready += prefetch_fusion_feature_rows(
        rows, fusion_tower,
        (fusion_feature_tokens(fusion_tower, projector)
         if fusion_tower is not None else 0), fusion_cache_dir)
    ids, labels, mask = make_batch(rows, device="cpu")
    text_batch = tuple(_pin_tensor(value) for value in (ids, labels, mask))
    positions = _pin_tensor(supervised_positions(
        rows, visual_width, device="cpu"))
    recall = None
    if engram is not None:
        boundary = 0 if engram.boundary_id is None else int(engram.boundary_id)
        starts = visual_insert_positions(rows)
        recall = _pin_recall(token_rosa_recall(
            insert_boundary_ids(ids, starts, visual_width, boundary),
            engram.table.vocab_size,
            engram.boundary_id))
    return BatchPrefetchResult(
        ready=ready, recall=recall, native_features=native_features,
        text_batch=text_batch, positions=positions,
        elapsed_s=time.perf_counter() - started, **stats)


def add_fusion_residual(prefix: torch.Tensor,
                        residual: torch.Tensor) -> torch.Tensor:
    if (prefix.ndim != 3 or residual.ndim != 3
            or prefix.shape[0] != residual.shape[0]
            or prefix.shape[2] != residual.shape[2]):
        raise ValueError(
            f"fusion residual {tuple(residual.shape)} is incompatible with "
            f"prefix {tuple(prefix.shape)}")
    if residual.shape[1] > prefix.shape[1]:
        raise ValueError("fusion residual is longer than the visual prefix")
    residual = residual.to(prefix.dtype)
    if residual.shape[1] == prefix.shape[1]:
        return prefix + residual
    output = prefix.clone()
    output[:, :residual.shape[1]] = (
        output[:, :residual.shape[1]] + residual)
    return output


def structured_language_masks(
        rows: Sequence[dict], full_labels: torch.Tensor,
        *, visual_width: int, visual_starts: Sequence[int],
        ) -> tuple[torch.Tensor, torch.Tensor]:
    """Dense masks for structured coordinate targets after visual insertion."""
    coordinate = torch.zeros_like(full_labels, dtype=torch.bool)
    starts = torch.zeros_like(full_labels, dtype=torch.bool)
    for batch, (row, visual_start) in enumerate(zip(rows, visual_starts)):
        for key, destination in (
                ("structured_coordinate_tokens", coordinate),
                ("structured_coordinate_starts", starts)):
            for target in row.get(key, ()):
                target = int(target)
                full_target = target + (visual_width
                                        if target >= visual_start else 0)
                if 0 <= full_target < full_labels.shape[1]:
                    destination[batch, full_target] = True
    return coordinate, starts


def invalid_boundary_token_loss(
        logits: torch.Tensor, targets: torch.Tensor, start_mask: torch.Tensor,
        boundary_tokens: Sequence[int], *, margin: float = 1.0,
        ) -> torch.Tensor:
    """Rank false 00/99 coordinate starts below the correct target token.

    A boundary lead is excluded when it is the teacher target, so real boxes
    beginning with 00 or 99 are never penalized.  This acts directly on the
    autoregressive coordinate stream whose greedy generations otherwise tend
    to collapse to 000/999 despite a healthy continuous DETR auxiliary head.
    """
    if not boundary_tokens:
        return logits.new_zeros(())
    chosen_logits = logits[start_mask]
    chosen_targets = targets[start_mask]
    if chosen_logits.shape[0] == 0:
        return logits.new_zeros(())
    correct = chosen_logits.gather(1, chosen_targets[:, None]).squeeze(1)
    bad_ids = torch.as_tensor(
        tuple(dict.fromkeys(map(int, boundary_tokens))),
        dtype=torch.long, device=logits.device)
    bad = chosen_logits.index_select(1, bad_ids)
    allowed = bad_ids.unsqueeze(0) != chosen_targets.unsqueeze(1)
    terms = F.softplus(bad.float() - correct.float().unsqueeze(1) + margin)
    return (terms * allowed).sum() / allowed.sum().clamp_min(1)


def apply_coordinate_token_weights(
        token_weights: torch.Tensor | None, coordinate_mask: torch.Tensor,
        weight: float) -> torch.Tensor | None:
    """Compose coordinate emphasis with optional pre-existing CE weights."""
    if weight == 1:
        return token_weights
    if token_weights is None:
        token_weights = torch.ones_like(coordinate_mask, dtype=torch.float32)
    return token_weights * torch.where(
        coordinate_mask,
        token_weights.new_tensor(weight),
        token_weights.new_tensor(1.0))


def project_visual_prefix(projector: nn.Module, features,
                          *, image_aspect: torch.Tensor | None = None):
    """Turn vision-tower features into the visual prefix.

    Tiled and native RADIO prefixes come from different methods, and every
    consumer has to choose between them the same way. Two consumers once chose
    differently: training ran natively for 1,000 steps while the qualitative
    sampler still called the tiled path, which raised
    ``features must be [batch,tiles,128|1,2560]`` and killed the run at its
    first sample. Ask this function instead of dispatching in place.
    """
    if not isinstance(projector, RadioFeatureProjector):
        return projector(features)
    if getattr(projector, "native_mode", False):
        return projector.forward_native(features)
    return projector(features, image_aspect=image_aspect)


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
                    vision_compressor: FrozenTeacherCompressor | None = None,
                    grounding: ImageTextContrastiveHead | None = None,
                    grounding_contrastive_weight: float = 0.0,
                    grounding_early_tokens: int = 0,
                    grounding_early_weight: float = 1.0,
                    structured_head: StructuredSpatialHead | None = None,
                    structured_rows: Sequence[dict] | None = None,
                    structured_weight: float = 0.0,
                    structured_coordinate_weight: float = 1.0,
                    structured_invalid_box_weight: float = 0.0,
                    structured_invalid_box_margin: float = 1.0,
                    return_per_example_ce: bool = False,
                    activation_checkpoint_min_tokens: int = 0,
                    activation_checkpoint_max_layers: int = 0,
                    image_aspect: torch.Tensor | None = None,
                    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if features is None:
        with torch.no_grad():
            features = vision(images)
    if vision_compressor is not None:
        if fusion_features is None:
            raise ValueError("frozen compressor requires paired fusion features")
        canonical = vision_compressor(features, fusion_features)
        prefix = projector(list(canonical.unbind(0)))
    else:
        prefix = project_visual_prefix(projector, features,
                                       image_aspect=image_aspect)
    fusion_residual = None
    if fusion_adapter is not None:
        if fusion_features is None:
            raise ValueError("fusion adapter requires cached three-tower features")
        fusion_residual = fusion_adapter(fusion_features)
        prefix = add_fusion_residual(prefix, fusion_residual)
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
    if engram_recall is not None:
        # Recall and the supervised-position selectors are both built ahead of
        # this forward from ``visual_prefix_width`` -- the manifest geometry --
        # while the prefix inserted here is the grid that actually loaded. EXIF
        # rotation can make those disagree, and a disagreement misaligns every
        # recalled position and every loss offset without raising anywhere.
        planned_width = (int(engram_recall.recalled.shape[1])
                         - int(ids.shape[1]))
        if planned_width != int(prefix.shape[1]):
            raise ValueError(
                "visual prefix width disagrees with the planned geometry: the "
                f"loaded grid contributes {int(prefix.shape[1])} tokens while "
                f"Engram recall and the supervised positions were built for "
                f"{planned_width}")
    if engram is not None:
        # The vision prefix has no vocabulary IDs. Treat it as a sequence of
        # boundaries so lexical recall begins fresh at the textual prompt and
        # can never manufacture matches from identical placeholder IDs.
        boundary = 0 if engram.boundary_id is None else int(engram.boundary_id)
        engram.set_input_ids(
            insert_boundary_ids(ids, starts, prefix.shape[1], boundary),
            recall=engram_recall)
    checkpoint_exclusions: set[int] = set()
    if deep_vision is not None:
        checkpoint_exclusions.update(deep_vision.layer_indices)
    if layer_vision is not None:
        checkpoint_exclusions.update(layer_vision.layer_indices)
    if engram is not None:
        checkpoint_exclusions.update(int(index) for index in engram.sites)
    with selective_activation_checkpointing(
            rwkv.model.layers, sequence_tokens=embeds.shape[1],
            min_tokens=activation_checkpoint_min_tokens,
            excluded_layers=checkpoint_exclusions,
            max_layers=activation_checkpoint_max_layers) as checkpointed_layers, \
            contextlib.ExitStack() as stack:
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
    coordinate_mask = coordinate_start_mask = None
    if (structured_rows is not None
            and (structured_coordinate_weight != 1
                 or structured_invalid_box_weight
                 # The coordinate CE metrics are exactly zero without
                 # coordinate targets, so an all-caption/OCR eval batch skips
                 # building two dense [batch, sequence] masks per forward.
                 or (return_per_example_ce
                     and any(row.get("structured_coordinate_tokens")
                             for row in structured_rows)))):
        coordinate_targets, coordinate_starts = structured_language_masks(
            structured_rows, full_labels, visual_width=prefix.shape[1],
            visual_starts=starts)
        coordinate_mask = coordinate_targets[
            batch_positions, sequence_positions + 1]
        coordinate_start_mask = coordinate_starts[
            batch_positions, sequence_positions + 1]
        token_weights = apply_coordinate_token_weights(
            token_weights, coordinate_mask, structured_coordinate_weight)
    invalid_box_loss = selected_logits.new_zeros(())
    if structured_invalid_box_weight and coordinate_start_mask is not None:
        boundary_tokens = next((
            tuple(map(int, row.get("structured_boundary_lead_tokens", ())))
            for row in structured_rows
            if row.get("structured_boundary_lead_tokens")), ())
        invalid_box_loss = invalid_boundary_token_loss(
            selected_logits, selected_targets, coordinate_start_mask,
            boundary_tokens, margin=structured_invalid_box_margin)
    ce, raw_ce = weighted_logits_cross_entropy(
        selected_logits, selected_targets, token_weights)
    loss = ce + structured_invalid_box_weight * invalid_box_loss
    # Leave scalar telemetry on-device until after backward/optimizer. Python
    # conversion here would force a forward-to-CPU barrier every training step.
    metrics = {"ce_loss": raw_ce, "grounded_ce_loss": ce.detach()}
    if coordinate_mask is not None:
        metrics.update(
            structured_coordinate_tokens=coordinate_mask.sum().detach(),
            structured_invalid_box_loss=invalid_box_loss.detach(),
            structured_invalid_box_weighted_loss=(
                structured_invalid_box_weight * invalid_box_loss.detach()),
        )
    if return_per_example_ce:
        token_nll = F.cross_entropy(
            selected_logits.float(), selected_targets, reduction="none")
        per_example_sum = token_nll.new_zeros(ids.shape[0])
        per_example_count = token_nll.new_zeros(ids.shape[0])
        per_example_sum.scatter_add_(0, batch_positions, token_nll)
        per_example_count.scatter_add_(
            0, batch_positions, torch.ones_like(token_nll))
        metrics["_eval_ce_sums"] = per_example_sum
        metrics["_eval_ce_counts"] = per_example_count
        if coordinate_mask is not None:
            coordinate_weights = coordinate_mask.to(token_nll.dtype)
            metrics["_eval_coordinate_ce_sum"] = (
                token_nll * coordinate_weights).sum()
            metrics["_eval_coordinate_ce_count"] = coordinate_weights.sum()
    if (isinstance(projector, RadioFeatureProjector)
            and projector.last_token_counts is not None):
        routed = projector.last_token_counts.float()
        metrics.update(
            radio_tokens_per_tile_mean=routed.mean().detach(),
            radio_tokens_per_tile_min=routed.min().detach(),
            radio_tokens_per_tile_max=routed.max().detach(),
            radio_visual_tokens=loss.new_tensor(prefix.shape[1], dtype=torch.float32),
        )
    metrics["activation_checkpointed_layers"] = loss.new_tensor(
        len(checkpointed_layers), dtype=torch.float32)
    if grounding is not None and grounding_contrastive_weight:
        target_embeddings = rwkv.model.embeddings(selected_targets).detach()
        contrastive, accuracy = grounding(
            prefix, target_embeddings, batch_positions)
        loss = loss + grounding_contrastive_weight * contrastive
        metrics.update(grounding_contrastive_loss=contrastive.detach(),
                       grounding_retrieval_accuracy=accuracy.detach())
    if structured_head is not None and structured_weight:
        if structured_rows is None or len(structured_rows) != ids.shape[0]:
            raise ValueError("structured head requires every source row")
        selected_rows = [index for index, row in enumerate(structured_rows)
                         if str(row.get("task") or "") == "sam_mask"]
        if selected_rows:
            row_index = torch.tensor(
                selected_rows, dtype=torch.long, device=prefix.device)
            contexts = []
            targets = []
            mask_shapes = []
            for index in selected_rows:
                row = structured_rows[index]
                start = int(row.get("vision_insert", 0))
                end = int(row["prompt_len"])
                if not 0 <= start < end <= text.shape[1]:
                    raise ValueError("structured task prompt span is invalid")
                contexts.append(text[index, start:end].mean(dim=0))
                shape = row.get("_structured_mask_shape")
                if shape is None:
                    targets.append(parse_structured_target(
                        str(row.get("text") or ""), device=prefix.device))
                else:
                    shape = tuple(map(int, shape))
                    mask_shapes.append(shape)
                    targets.append(structured_target_from_row(
                        row, mask_shape=shape, device=prefix.device))
            if mask_shapes and len(mask_shapes) != len(selected_rows):
                raise ValueError(
                    "native structured mask geometry is missing for part of "
                    "the selected batch")
            prediction = structured_head(
                prefix.index_select(0, row_index), torch.stack(contexts),
                mask_shapes=(mask_shapes or None))
            structured, structured_metrics = structured_detection_loss(
                prediction, targets)
            loss = loss + structured_weight * structured
            metrics.update(structured_metrics)
            metrics["structured_weighted_loss"] = (
                structured_weight * structured.detach())
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


def _module_state_view(module: nn.Module | None) -> dict | None:
    """Return detached state references for the single-copy snapshot walker."""
    return (None if module is None else
            {key: value.detach() for key, value in module.state_dict().items()})


def _loop_adapter_state_view(
        wrappers: Sequence[nn.Module]) -> list[dict[str, torch.Tensor]]:
    return [
        {name: value.detach() for name, value in wrapper.loop.state_dict().items()
         if not name.startswith("core.")}
        for wrapper in wrappers
    ]


def _checkpoint_cpu_snapshot(value):
    """Detach a checkpoint tree from mutable live training state.

    ``Optimizer.state_dict`` is a shallow view whose tensors continue to be
    mutated by every optimizer step.  A background serializer therefore needs
    an owned CPU copy, rather than merely moving the eventual ``torch.save`` to
    another thread.
    """
    if torch.is_tensor(value):
        return value.detach().to(device="cpu", copy=True)
    if isinstance(value, dict):
        return {copy.deepcopy(key): _checkpoint_cpu_snapshot(item)
                for key, item in value.items()}
    if isinstance(value, list):
        return [_checkpoint_cpu_snapshot(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_checkpoint_cpu_snapshot(item) for item in value)
    return copy.deepcopy(value)


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
        "radio_native_plan.json", "radio_native_plan.json.tmp",
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
        except (json.JSONDecodeError, AttributeError, TypeError, ValueError):
            # Keep malformed/unstepped lines byte-for-byte, matching
            # _invalidate_step_evaluation. Only records provably newer than
            # the recovered checkpoint may be discarded.
            kept.append(line)
            continue
        if record_step <= checkpoint_step:
            kept.append(line)
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


_MIGRATION_RECEIPTS = frozenset({
    "batch_budget_migrated", "loop_count_migrated_from",
    "loop_lr_migrated_from", "radio_v4h_fingerprint_bootstrapped",
    "structured_lr_migrated_from",
    "structured_data_removed_from_fingerprint",
    "v4h_fingerprint_migrated_from",
})


def persisted_args(args: argparse.Namespace) -> dict:
    """The resumable configuration, without this resume's migration receipts.

    ``_load_checkpoint`` records what one resume migrated so the same process
    can decide publication and print its one-time notes. Those receipts describe
    an event, not a configuration: writing them into ``config.json`` and every
    later checkpoint would leave a run permanently claiming a migration that
    some unrelated resume performed months earlier. Anything a later resume
    genuinely reads back (``loop_reset_committed``,
    ``structured_head_initialized_step``) is configuration and stays.
    """
    return {name: value for name, value in vars(args).items()
            if name not in _MIGRATION_RECEIPTS}


def _checkpoint_snapshot(*, step: int, projector: nn.Module,
                         nextlat: nn.Module | None,
                         engram: LexicalMemoryBank | None,
                         deep_vision: DeepVisionInjector | None,
                         grounding: ImageTextContrastiveHead | None,
                         structured_head: StructuredSpatialHead | None,
                         wrappers: list[nn.Module], optimizer,
                         sampler: EpochBatchSampler, args: argparse.Namespace,
                         layer_vision: LayerMatchedVisionInjector | None = None,
                         vision_fusion: VisionFusionResidual | None = None) -> dict:
    """Stage one immutable, self-contained checkpoint on CPU."""
    return _checkpoint_cpu_snapshot({
        "schema": CHECKPOINT_SCHEMA,
        "step": step,
        "projector": _module_state_view(projector),
        "nextlat": _module_state_view(nextlat),
        "engram": _module_state_view(engram),
        "deep_vision": _module_state_view(deep_vision),
        "layer_vision": _module_state_view(layer_vision),
        "vision_fusion": _module_state_view(vision_fusion),
        "grounding": _module_state_view(grounding),
        "structured_head": _module_state_view(structured_head),
        "loops": _loop_adapter_state_view(wrappers),
        "optimizer": optimizer.state_dict(),
        "sampler": sampler.state_dict(),
        "rng": {
            "python": random.getstate(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all(),
        },
        "args": persisted_args(args),
    })


def _write_checkpoint_snapshot(path: Path, blob: dict) -> None:
    """Serialize and atomically publish an immutable CPU checkpoint snapshot."""
    temporary = path.with_suffix(".tmp")
    torch.save(blob, temporary)
    # Atomic rename protects against process interruption; the file and parent
    # directory fsyncs additionally guarantee that a successful return remains
    # recoverable after an abrupt host reset or power loss.
    _durable_replace(temporary, path)


class _AsyncCheckpointWriter:
    """Own at most one immutable snapshot being durably published."""

    def __init__(self, path: Path):
        self.path = path
        self.future = None
        self.step: int | None = None

    @property
    def pending(self) -> bool:
        return self.future is not None

    def submit(self, step: int, snapshot: dict) -> None:
        if self.future is not None:
            raise RuntimeError("checkpoint writer already has a pending snapshot")
        self.step = int(step)
        self.future = _CHECKPOINT_WRITE_POOL.submit(
            _write_checkpoint_snapshot, self.path, snapshot)

    def finish(self, *, wait_for_write: bool = False) -> int | None:
        if self.future is None:
            return None
        if not wait_for_write and not self.future.done():
            return None
        future, completed_step = self.future, self.step
        # Clear first so a surfaced writer failure cannot be raised a second
        # time from an outer recovery/finally path.
        self.future = self.step = None
        future.result()
        return completed_step


def _save_checkpoint(path: Path, *, step: int, projector: nn.Module,
                     nextlat: nn.Module | None, engram: LexicalMemoryBank | None,
                     deep_vision: DeepVisionInjector | None,
                     grounding: ImageTextContrastiveHead | None,
                     structured_head: StructuredSpatialHead | None,
                     wrappers: list[nn.Module], optimizer,
                     sampler: EpochBatchSampler, args: argparse.Namespace,
                     layer_vision: LayerMatchedVisionInjector | None = None,
                     vision_fusion: VisionFusionResidual | None = None) -> None:
    blob = _checkpoint_snapshot(
        step=step, projector=projector, nextlat=nextlat, engram=engram,
        deep_vision=deep_vision, grounding=grounding,
        structured_head=structured_head, wrappers=wrappers,
        optimizer=optimizer, sampler=sampler, args=args,
        layer_vision=layer_vision, vision_fusion=vision_fusion)
    _write_checkpoint_snapshot(path, blob)


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
        "radio_adaptive_complexity": False,
        "radio_complexity_budget_ratio": DEFAULT_COMPLEXITY_BUDGET_RATIO,
        "radio_complexity_token_quantum": DEFAULT_COMPLEXITY_TOKEN_QUANTUM,
        # Sits with the other RADIO budget knobs rather than the hard contract
        # list: like radio_adaptive_complexity it changes how much of each
        # tile's prefix survives, and the same machinery already handles the
        # consequences (stale feature-cache entries are re-encoded on demand,
        # best/ is quarantined, and prior eval claims are reset).
        "radio_adaptive_token_threshold": DEFAULT_ADAPTIVE_TOKEN_THRESHOLD,
    }
    return [name for name, default in defaults.items()
            if saved_args.get(name, default) != getattr(args, name, default)]


def _budget_eval_contract_differences(
        budget_differences: Sequence[str]) -> list[str]:
    """Budget changes that alter the recovered model's current forward pass."""
    semantic = {
        "radio_adaptive_complexity",
        "radio_complexity_budget_ratio",
        "radio_complexity_token_quantum",
        "radio_adaptive_token_threshold",
    }
    return [name for name in budget_differences if name in semantic]


def _loop_lr_resume_difference(saved_args: dict,
                               args: argparse.Namespace) -> bool:
    """Reject an LR change unless this resume explicitly migrates or resets."""
    reset_pending = bool(
        args.reset_loop_on_resume
        and not saved_args.get("loop_reset_committed", False))
    return bool(
        saved_args.get("loop_lr") != getattr(args, "loop_lr", None)
        and not reset_pending
        and not _loop_lr_resume_migration(saved_args, args))


def _loop_lr_resume_migration(saved_args: dict,
                              args: argparse.Namespace) -> bool:
    """Accept only an explicitly named LR change from the checkpoint's LR.

    Naming the source expires the permission the moment it is used, exactly as
    ``--allow-loop-count-increase-from`` and
    ``--allow-structured-data-removal-from`` do. A bare boolean cannot: the
    launcher passes it on every one of hundreds of automatic restarts, so any
    later drift in the requested LR would be waived silently forever.
    """
    saved = saved_args.get("loop_lr")
    requested = getattr(args, "loop_lr", None)
    if saved is None or requested is None:
        return False
    if getattr(args, "allow_loop_lr_change_on_resume", False):
        # Deprecated permissive spelling; see the argparse note. Retains the
        # original never-expiring behaviour for in-flight supervisor loops.
        return float(saved) != float(requested)
    source = getattr(args, "allow_loop_lr_change_from", None)
    return bool(
        source is not None
        and float(saved) == float(source)
        and float(saved) != float(requested))


def _v4h_fingerprint_migration(saved_args: dict,
                               args: argparse.Namespace) -> bool:
    """Accept a named C-RADIOv4-H artifact-fingerprint change on resume.

    ``v4h_artifact_fingerprint`` deliberately hashes the *producer sources*
    (``radio_v4h.py`` and, for fused caches, ``radio_v4h_adaptors.py``) so that
    changing the snapping, packing or cache conventions invalidates caches those
    conventions produced. The consequence is that any edit to those modules --
    including a pure bug fix that writes no cache at all -- changes
    ``radio_v4h_fingerprint``, and with it the ``radio_v4h_cache_fingerprint``
    that embeds it, so every existing radio_v4h checkpoint stops resuming.

    That is the correct default for a run that *writes* cache. It is too strict
    for ``--radio-v4h-cache-only``, which never encodes: what matters there is
    that the entries match, and the cache contract plus each entry's recorded
    producer metadata already establish that per row.

    Naming the source expires the permission the moment it is used, exactly as
    the loop-count, loop-LR and structured-data allowances do: once the migrated
    step is saved the checkpoint carries the new fingerprint, so a launcher that
    passes the flag on every automatic restart cannot waive a later, real change.
    """
    source = str(getattr(args, "allow_v4h_fingerprint_change_from", "") or "")
    saved = saved_args.get("radio_v4h_fingerprint")
    current = getattr(args, "radio_v4h_fingerprint", None)
    return bool(
        source and saved is not None and current is not None
        and str(saved) == source and str(saved) != str(current))


def _structured_data_removal_migration(
        saved_args: dict, args: argparse.Namespace) -> bool:
    """Allow exactly one audited transition from bbox data to no bbox data."""
    source = str(getattr(
        args, "allow_structured_data_removal_from", "") or "")
    saved_fingerprint = str(saved_args.get("data_fingerprint") or "")
    return bool(
        source
        and source == saved_fingerprint
        and saved_fingerprint != str(args.data_fingerprint)
        and int(getattr(args, "structured_train_examples", -1)) == 0
        and int(getattr(args, "structured_eval_examples", -1)) == 0
        and float(getattr(args, "structured_update_ratio", -1.0)) == 0.0
        and float(getattr(args, "structured_weight", -1.0)) == 0.0
        and float(saved_args.get("structured_update_ratio", 0.0)) > 0.0)


def _structured_lr_resume_migration(
        saved_args: dict, args: argparse.Namespace) -> bool:
    """Accept one explicitly named structured-head LR transition."""
    saved = saved_args.get("structured_lr")
    requested = getattr(args, "structured_lr", None)
    source = getattr(args, "allow_structured_lr_change_from", None)
    if saved is None or requested is None or source is None:
        return False
    return float(saved) == float(source) and float(saved) != float(requested)


def _preserve_loop_reset_outcome(args: argparse.Namespace,
                                 committed: bool) -> None:
    """Carry a one-time reset receipt into every descendant checkpoint."""
    if committed:
        args.loop_reset_committed = True


def _resume_contract_changed(*, text_limit_migrated: bool,
                             budget_differences: Sequence[str],
                             coco_prompt_migrated: bool = False,
                             loop_count_migrated: bool = False) -> bool:
    """Whether weights/eval semantics changed at the recovered optimizer step.

    A loop-LR migration is deliberately not an input here: it changes only the
    next update, so the recovered weights and their existing same-step
    evaluation remain exact. That migration still needs checkpoint publication,
    which ``resume_publication_changed`` decides separately.
    """
    return bool(text_limit_migrated or budget_differences or coco_prompt_migrated
                or loop_count_migrated)


def _loop_count_resume_migration(saved_args: dict,
                                 args: argparse.Namespace) -> bool:
    """Accept only an explicitly named increase from the checkpoint's depth."""
    source = int(saved_args.get("loop_count", 0))
    allowed_source = int(getattr(
        args, "allow_loop_count_increase_from", 0) or 0)
    return bool(
        allowed_source > 0
        and source == allowed_source
        and int(args.loop_count) > source)


def _expanded_loop_state_dict(module: nn.Module, saved: dict[str, torch.Tensor],
                              *, source_loop_count: int) -> dict[str, torch.Tensor]:
    """Pad Engram's loop-index state while requiring every other tensor exact."""
    current = module.state_dict()
    if set(current) != set(saved):
        missing = sorted(set(current) - set(saved))
        unexpected = sorted(set(saved) - set(current))
        raise ValueError(
            "checkpoint Engram state keys differ during loop migration: "
            f"missing={missing[:3]}, unexpected={unexpected[:3]}")
    expanded = {}
    for name, value in saved.items():
        target = current[name]
        if target.shape == value.shape:
            expanded[name] = value
            continue
        can_grow = bool(
            name.endswith(".loop_scale")
            and value.ndim == 1 and target.ndim == 1
            and value.shape[0] == source_loop_count
            and target.shape[0] > value.shape[0])
        if not can_grow:
            raise ValueError(
                f"incompatible Engram tensor during loop migration: {name} "
                f"{tuple(value.shape)} -> {tuple(target.shape)}")
        grown = torch.zeros_like(target, device=value.device)
        grown[:value.shape[0]].copy_(value.to(grown.dtype))
        expanded[name] = grown
    return expanded


def _resume_invalidates_step_evaluation(*, text_limit_migrated: bool,
                                        unrelated_branch: bool,
                                        loop_reset_pending: bool,
                                        contract_changed: bool = False) -> bool:
    """Whether prior same-step evaluation belongs to a different model contract."""
    return bool(text_limit_migrated or unrelated_branch or loop_reset_pending
                or contract_changed)


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


_STRUCTURED_HEAD_PARAMETERS = frozenset({
    "structured_head", "structured_weight", "structured_width",
    "structured_object_queries", "structured_spatial_layers",
    "structured_object_layers", "structured_heads", "structured_lr",
})
"""Settings that only exist because a structured head exists.

Appending a head to a running checkpoint has to introduce these, so they are
exempt from the resume contract for that one migration. Nothing else may be:
``structured_update_ratio`` changes the *sampler composition*, and waiving it
here would let a head-append silently rebuild the training split -- surfacing
much later as ``EpochBatchSampler.load_state_dict``'s "does not match the
current training split" rather than as the contract error it is.
"""


def _load_checkpoint(path: Path, *, projector: nn.Module, nextlat: nn.Module | None,
                     engram: LexicalMemoryBank | None, wrappers: list[nn.Module], optimizer,
                     sampler: EpochBatchSampler,
                     args: argparse.Namespace,
                     deep_vision: DeepVisionInjector | None = None,
                     layer_vision: LayerMatchedVisionInjector | None = None,
                     vision_fusion: VisionFusionResidual | None = None,
                     grounding: ImageTextContrastiveHead | None = None,
                     structured_head: StructuredSpatialHead | None = None,
                     ) -> tuple[int, bool, bool, bool]:
    blob = torch.load(path, map_location="cpu", weights_only=False)
    if not valid_torch_archive_storages(path, blob):
        raise ValueError(f"checkpoint archive integrity check failed: {path}")
    if int(blob.get("schema", -1)) != CHECKPOINT_SCHEMA:
        raise ValueError(f"unsupported vision checkpoint schema {blob.get('schema')}")
    saved_args = blob.get("args", {})
    if (saved_args.get("structured_lr") == 0.0
            and not saved_args.get("structured_lr_zero_is_freeze", False)):
        # 0.0 used to be this option's "unset" sentinel, which made a frozen
        # structured head inexpressible. None is the sentinel now, so a legacy
        # checkpoint's 0.0 reads as the same "inherit --lr" it always meant.
        saved_args = {**saved_args, "structured_lr": None}
    adding_structured_head = bool(
        structured_head is not None
        and blob.get("structured_head") is None
        and not saved_args.get("structured_head", False))
    saved_limit = int(saved_args.get("max_text_tokens", 0))
    migrating_text_limit = bool(
        args.allow_text_limit_increase_from
        and saved_limit == args.allow_text_limit_increase_from
        and args.max_text_tokens > saved_limit
        and saved_args.get("data_fingerprint")
        == getattr(args, "previous_data_fingerprint", None)
    )
    migrating_coco_prompt = bool(
        getattr(args, "allow_coco_prompt_migration", False)
        and saved_args.get("data_fingerprint")
        == getattr(args, "previous_coco_prompt_data_fingerprint", None)
        and saved_args.get("data_fingerprint") != args.data_fingerprint
    )
    migrating_structured_data = _structured_data_removal_migration(
        saved_args, args)
    migrating_structured_lr = _structured_lr_resume_migration(
        saved_args, args)
    migrating_loop_count = _loop_count_resume_migration(saved_args, args)
    migrating_loop_lr = _loop_lr_resume_migration(saved_args, args)
    source_loop_count = int(saved_args.get("loop_count", 0))
    migration_compatible = set()
    if migrating_text_limit:
        migration_compatible.update(("data_fingerprint", "max_text_tokens"))
    if migrating_coco_prompt:
        migration_compatible.add("data_fingerprint")
    if migrating_structured_data:
        migration_compatible.add("data_fingerprint")
    if migrating_loop_count:
        migration_compatible.add("loop_count")
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
    if saved_image_fingerprint is not None:
        if migrating_structured_data:
            # Dropping the sam_mask rows legitimately changes the all-rows
            # digest, so that digest cannot decide this resume. Every surviving
            # caption/OCR image must still be byte-identical, which the digest
            # restricted to the surviving rows answers exactly. Checkpoints
            # written before that digest existed keep the previous behaviour of
            # skipping the check rather than failing an audited migration.
            saved_surviving = saved_args.get(
                "unstructured_image_metadata_fingerprint")
            if (saved_surviving is not None and saved_surviving != getattr(
                    args, "unstructured_image_metadata_fingerprint", None)):
                differences.append("image_metadata_fingerprint")
        elif saved_image_fingerprint != getattr(
                args, "image_metadata_fingerprint", None):
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
        "sandwich_lead_prompt": "", "vision_backend": "moonvit",
        "radio_model": "models/vision/C-RADIOv4-1D-H",
        "radio_revision": "e18692120c7a3203496e1a99056a4149ede135fc",
        "radio_max_detail_tiles": DEFAULT_MAX_DETAIL_TILES,
        # radio_adaptive_token_threshold intentionally lives in
        # _budget_resume_differences instead, so it is changeable under
        # --allow-batch-resize rather than being an unresumable hard error.
        "radio_tile_batch": 8, "radio_fingerprint": "",
        "radio_v4h_model": "models/vision/C-RADIOv4-H",
        "radio_v4h_revision": _DEFAULT_NATIVE_REVISION,
        "radio_v4h_lattice": 16, "radio_v4h_native": True,
        "radio_v4h_native_packing": "pair_columns",
        "radio_v4h_feature_width": 1280,
        "radio_v4h_max_edge": 2048, "radio_v4h_pair_axis": "columns",
        "radio_v4h_bridge_rank": 256,
        "radio_v4h_fingerprint": "",
        "radio_v4h_cache_fingerprint": "",
        "vision_fusion": False, "vision_fusion_rank": 512,
        "vision_fusion_fingerprint": "",
        "sam_fusion": False, "sam_fusion_tokens": 128,
        "sam_crop_padding": False,
        "vision_compressor_checkpoint": "",
        "vision_compressor_fingerprint": "",
        "grounding_contrastive_weight": 0.0,
        "grounding_contrastive_dim": 512, "grounding_temperature": 0.07,
        "grounding_early_tokens": 0, "grounding_early_weight": 1.0,
        "structured_head": False, "structured_weight": 1.0,
        "structured_width": 256, "structured_object_queries": 16,
        "structured_spatial_layers": 2, "structured_object_layers": 2,
        "structured_heads": 8,
        "ocr_update_ratio": 0.0,
        "structured_update_ratio": 0.0,
        "structured_lr": None,
    }
    bootstrapping_v4h_fingerprint = bool(
        saved_args.get("vision_backend", "moonvit") == "radio_v4h"
        and ("radio_v4h_fingerprint" not in saved_args
             or "radio_v4h_cache_fingerprint" not in saved_args))
    migrating_v4h_fingerprint = _v4h_fingerprint_migration(saved_args, args)
    if migrating_v4h_fingerprint:
        args.v4h_fingerprint_migrated_from = str(
            saved_args.get("radio_v4h_fingerprint"))
    new_differences = [
        name for name, default in new_contract_defaults.items()
        if not (adding_structured_head
                and name in _STRUCTURED_HEAD_PARAMETERS)
        if not (migrating_structured_data
                and name in {"structured_update_ratio", "structured_weight"})
        if not (migrating_structured_lr and name == "structured_lr")
        if not (migrating_v4h_fingerprint
                and name in {"radio_v4h_fingerprint",
                             "radio_v4h_cache_fingerprint"})
        if saved_args.get(
            name,
            (getattr(args, name, default)
             if bootstrapping_v4h_fingerprint
             and name in {"radio_v4h_fingerprint",
                          "radio_v4h_cache_fingerprint"}
             else default)
        ) != getattr(args, name, default)
    ]
    if new_differences:
        raise ValueError(
            f"checkpoint grounding/vision configuration differs: {new_differences}")
    # The letterbox content-geometry embedding is zero-initialized, so a
    # checkpoint predating it resumes as an exact no-op and simply starts
    # learning the term. Only that module may be absent; anything else missing
    # still fails loudly.
    projector_result = projector.load_state_dict(blob["projector"], strict=False)
    unexpected = list(projector_result.unexpected_keys)
    missing = [key for key in projector_result.missing_keys
               if not key.startswith("bridge.content_embedding.")]
    if missing or unexpected:
        raise ValueError(
            f"checkpoint projector does not match this run: missing={missing[:5]} "
            f"unexpected={unexpected[:5]}")
    content_geometry_growth = len(projector_result.missing_keys)
    if (nextlat is None) != (blob.get("nextlat") is None):
        raise ValueError("checkpoint NextLat configuration does not match this run")
    if nextlat is not None:
        nextlat.load_state_dict(blob["nextlat"])
    if (engram is None) != (blob.get("engram") is None):
        raise ValueError("checkpoint Engram state does not match this run")
    if engram is not None:
        engram_state = (
            _expanded_loop_state_dict(
                engram, blob["engram"],
                source_loop_count=source_loop_count)
            if migrating_loop_count else blob["engram"])
        engram.load_state_dict(engram_state)
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
    if not adding_structured_head:
        if (structured_head is None) != (blob.get("structured_head") is None):
            raise ValueError(
                "checkpoint structured-head configuration does not match this run")
        if structured_head is not None:
            structured_head.load_state_dict(blob["structured_head"])
    load_loop_adapter_state(
        wrappers, blob["loops"],
        allow_loop_count_increase_from=(
            source_loop_count if migrating_loop_count else 0))
    if adding_structured_head and content_geometry_growth:
        raise ValueError(
            "cannot append a structured head and grow the visual bridge in one "
            "resume; migrate them in separate runs")
    if migrating_loop_count and (adding_structured_head or content_geometry_growth):
        raise ValueError(
            "loop-count migration cannot be combined with another module growth")
    if migrating_loop_count:
        _load_optimizer_with_loop_growth(
            optimizer, blob["optimizer"],
            source_loop_count=source_loop_count)
    elif adding_structured_head:
        _load_optimizer_with_appended_group(
            optimizer, blob["optimizer"], group_name="structured_head")
    elif content_geometry_growth:
        _load_optimizer_with_grown_groups(
            optimizer, blob["optimizer"], expected_growth=content_geometry_growth)
    else:
        optimizer.load_state_dict(blob["optimizer"])
    if migrating_loop_lr:
        _set_optimizer_group_lr(
            optimizer, group_name="loop_gates", lr=float(args.loop_lr))
    if migrating_structured_lr:
        _set_optimizer_group_lr(
            optimizer, group_name="structured_head",
            lr=float(args.structured_lr))
    if not migrating_structured_data:
        sampler.load_state_dict(blob["sampler"])
    random.setstate(blob["rng"]["python"])
    torch.set_rng_state(blob["rng"]["torch"])
    torch.cuda.set_rng_state_all(blob["rng"]["cuda"])
    contract_changed = _resume_contract_changed(
        text_limit_migrated=migrating_text_limit,
        budget_differences=_budget_eval_contract_differences(
            budget_differences),
        coco_prompt_migrated=migrating_coco_prompt,
        loop_count_migrated=migrating_loop_count)
    contract_changed = (contract_changed or adding_structured_head
                        or bool(content_geometry_growth)
                        or migrating_structured_data)
    if adding_structured_head:
        args.structured_head_initialized_step = int(blob["step"])
    if migrating_loop_count:
        args.loop_count_migrated_from = source_loop_count
    if migrating_loop_lr:
        args.loop_lr_migrated_from = float(saved_args["loop_lr"])
    if migrating_structured_lr:
        args.structured_lr_migrated_from = float(saved_args["structured_lr"])
    if budget_differences:
        args.batch_budget_migrated = True
    if bootstrapping_v4h_fingerprint:
        args.radio_v4h_fingerprint_bootstrapped = True
    if migrating_structured_data:
        args.structured_data_removed_from_fingerprint = (
            saved_args["data_fingerprint"])
    return (int(blob["step"]), migrating_text_limit,
            bool(saved_args.get("loop_reset_committed", False)),
            contract_changed)


def _initialize_adapters(path: Path, *, projector: nn.Module,
                         nextlat: nn.Module | None, engram: nn.Module | None,
                         wrappers: list[nn.Module], args: argparse.Namespace,
                         deep_vision: DeepVisionInjector | None = None,
                         layer_vision: LayerMatchedVisionInjector | None = None,
                         vision_fusion: VisionFusionResidual | None = None,
                         grounding: ImageTextContrastiveHead | None = None,
                         structured_head: StructuredSpatialHead | None = None,
                         replace_vision_bridge: bool = False) -> int:
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
    structural = (("rwkv_fingerprint", "nextlat_hidden", "loop_count",
                   "loop_index", "loop_gate_cap") if replace_vision_bridge else
                  ("rwkv_fingerprint", "moonvit_fingerprint",
                   "max_input_patches", "nextlat_hidden", "loop_count",
                   "loop_index", "loop_gate_cap"))
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
    if (not replace_vision_bridge and source_resampler_layers
            and source_resampler_layers != destination_resampler_layers):
        raise ValueError("adapter initialization resampler geometry is incompatible")
    projector_state = dict(blob["projector"])
    source_prefix = int(saved.get("prefix_tokens", 0))
    destination_prefix = int(getattr(args, "prefix_tokens", 0))
    if not replace_vision_bridge and source_prefix != destination_prefix:
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
    # Always non-strict, then validate: a checkpoint predating the letterbox
    # content-geometry embedding is missing exactly those zero-init keys, and a
    # strict load would reject it outright the way exact resume used to.
    projector_info = (None if replace_vision_bridge
                      else projector.load_state_dict(projector_state, strict=False))
    if not replace_vision_bridge:
        assert projector_info is not None
        growing_resampler = (not source_resampler_layers
                             and destination_resampler_layers > 0)
        invalid_missing = [
            key for key in projector_info.missing_keys
            if not key.startswith("bridge.content_embedding.")
            and not (growing_resampler and key.startswith("resampler."))]
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
    source_structured = blob.get("structured_head")
    if source_structured is not None and structured_head is None:
        raise ValueError("adapter initialization would discard structured head")
    if source_structured is not None:
        structured_head.load_state_dict(source_structured)
    load_loop_adapter_state(wrappers, blob["loops"])
    return int(blob.get("step", 0))


def _optimizer(projector: nn.Module, nextlat: nn.Module | None,
               engram: LexicalMemoryBank | None, wrappers: list[nn.Module],
               args: argparse.Namespace, *,
               deep_vision: DeepVisionInjector | None = None,
               layer_vision: LayerMatchedVisionInjector | None = None,
               vision_fusion: VisionFusionResidual | None = None,
               grounding: ImageTextContrastiveHead | None = None,
               structured_head: StructuredSpatialHead | None = None):
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
    # Keep this last: exact-resume migration can append it to an existing
    # optimizer without renumbering or losing any prior Adam moments.
    if structured_head is not None:
        groups.append({"params": list(structured_head.parameters()),
                       "lr": (args.lr if args.structured_lr is None
                              else args.structured_lr),
                       "weight_decay": args.weight_decay,
                       "name": "structured_head"})
    groups = [group for group in groups if group["params"]]
    kwargs = dict(betas=(0.9, 0.95))
    if torch.cuda.is_available():
        kwargs["fused"] = True
    return torch.optim.AdamW(groups, **kwargs), [p for group in groups for p in group["params"]]


def _load_optimizer_with_grown_groups(optimizer, saved: dict, *,
                                      expected_growth: int) -> None:
    """Restore moments when a group gained members from a new zero-init module.

    ``Optimizer.load_state_dict`` matches groups by size, so adding parameters to
    an existing group would otherwise discard every prior moment. New parameters
    are appended by ``nn.Module`` registration order, so old positions map
    one-to-one onto the first slots of each group and the newcomers simply start
    without state — correct for a zero-initialized term.
    """
    current = optimizer.state_dict()
    old_groups, new_groups = saved.get("param_groups", []), current.get("param_groups", [])
    if len(old_groups) != len(new_groups):
        raise ValueError("optimizer group count changed; cannot migrate moments")
    mapping, growth = {}, 0
    for old, new in zip(old_groups, new_groups):
        if len(new["params"]) < len(old["params"]):
            raise ValueError("an optimizer group shrank; refusing to guess a mapping")
        growth += len(new["params"]) - len(old["params"])
        mapping.update(zip(old["params"], new["params"]))
    if growth != expected_growth:
        raise ValueError(
            f"optimizer grew by {growth} parameters, expected {expected_growth}")
    optimizer.load_state_dict({
        "state": {mapping[key]: value
                  for key, value in saved.get("state", {}).items()
                  if key in mapping},
        "param_groups": [{**old, "params": new["params"]}
                         for old, new in zip(old_groups, new_groups)],
    })


def _load_optimizer_with_loop_growth(optimizer, saved: dict, *,
                                     source_loop_count: int) -> None:
    """Restore Adam moments while zero-padding pass-indexed parameter state."""
    current = optimizer.state_dict()
    old_groups = saved.get("param_groups", [])
    new_groups = current.get("param_groups", [])
    if len(old_groups) != len(new_groups):
        raise ValueError(
            "optimizer group count changed during loop-count migration")
    migrated_state = {}
    migrated_groups = []
    for old_group, new_group, live_group in zip(
            old_groups, new_groups, optimizer.param_groups):
        if (old_group.get("name") != new_group.get("name")
                or len(old_group["params"]) != len(new_group["params"])
                or len(live_group["params"]) != len(new_group["params"])):
            raise ValueError(
                "optimizer groups changed during loop-count migration")
        group_name = old_group.get("name")
        for old_id, new_id, parameter in zip(
                old_group["params"], new_group["params"],
                live_group["params"]):
            old_state = saved.get("state", {}).get(old_id)
            if old_state is None:
                continue
            parameter_state = {}
            for name, value in old_state.items():
                if not torch.is_tensor(value) or value.ndim == 0 \
                        or value.shape == parameter.shape:
                    parameter_state[name] = value
                    continue
                can_grow = bool(
                    group_name in {"loop_gates", "engram"}
                    and value.ndim == parameter.ndim
                    and value.shape[0] == source_loop_count
                    and parameter.shape[0] > value.shape[0]
                    and value.shape[1:] == parameter.shape[1:])
                if not can_grow:
                    raise ValueError(
                        "optimizer tensor shape changed unexpectedly during "
                        f"loop migration: group={group_name} state={name} "
                        f"{tuple(value.shape)} -> {tuple(parameter.shape)}")
                grown = torch.zeros(
                    parameter.shape, dtype=value.dtype, device=value.device)
                grown[:value.shape[0]].copy_(value)
                parameter_state[name] = grown
            migrated_state[new_id] = parameter_state
        migrated_groups.append({**old_group, "params": new_group["params"]})
    optimizer.load_state_dict({
        "state": migrated_state,
        "param_groups": migrated_groups,
    })


def _set_optimizer_group_lr(optimizer, *, group_name: str, lr: float) -> None:
    """Change exactly one named group after restoring checkpoint hyperparams."""
    matches = [
        group for group in optimizer.param_groups
        if group.get("name") == group_name
    ]
    if len(matches) != 1:
        raise ValueError(
            f"expected one optimizer group named {group_name}, found "
            f"{len(matches)}")
    matches[0]["lr"] = float(lr)


def _load_optimizer_with_appended_group(optimizer, saved: dict, *,
                                        group_name: str) -> None:
    """Restore old moments while retaining one newly appended parameter group."""
    current = optimizer.state_dict()
    old_groups = saved.get("param_groups", [])
    new_groups = current.get("param_groups", [])
    if len(new_groups) != len(old_groups) + 1:
        raise ValueError("optimizer migration expected exactly one appended group")
    for old, new in zip(old_groups, new_groups):
        if old.get("name") != new.get("name") or len(old["params"]) != len(new["params"]):
            raise ValueError("optimizer groups changed before structured-head append")
    if new_groups[-1].get("name") != group_name:
        raise ValueError(f"new optimizer group is not {group_name}")
    optimizer.load_state_dict({
        "state": saved.get("state", {}),
        "param_groups": [*old_groups, new_groups[-1]],
    })


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


def _engram_metric_tensors(
        engram: LexicalMemoryBank | None
        ) -> tuple[dict[str, torch.Tensor], bool]:
    if engram is None:
        return {}, False
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
    metrics = {
        "engram_inj_rms": inj_rms,
        "engram_gate_mean": gate_mean,
    }
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
        metrics.update({
            "engram_recall_valid_rate": valid_rate,
            "engram_recall_frac_beyond_32": beyond,
            "engram_recall_mlen_p50": median,
            "engram_recall_mlen_max": maximum,
        })
    return metrics, True


def _render_adapter_training_metrics(
        wrappers: list, engram: LexicalMemoryBank | None
        ) -> dict[str, float | bool]:
    """Materialize all non-safety adapter telemetry in one host transfer."""
    loop_metrics = loop_training_metric_tensors(wrappers)
    engram_metrics, engram_enabled = _engram_metric_tensors(engram)
    tensors = {**loop_metrics, **engram_metrics}
    rendered = (
        torch.stack([value.float() for value in tensors.values()]).tolist()
        if tensors else [])
    result: dict[str, float | bool] = dict(zip(tensors, rendered))
    result["engram_enabled"] = engram_enabled
    if engram is not None:
        for name in ("valid_rate", "frac_beyond_32", "mlen_p50", "mlen_max"):
            key = f"engram_recall_{name}"
            if key in result:
                engram.recall_stats[name] = float(result[key])
    return result


def _engram_metrics(engram: LexicalMemoryBank | None) -> dict[str, float | bool]:
    """Compatibility renderer for tests and non-training callers."""
    return _render_adapter_training_metrics([], engram)


def assert_training_contract(rwkv: nn.Module, vision: nn.Module,
                             wrappers: Sequence[nn.Module], trainable: Sequence[nn.Parameter],
                             vision_compressor: nn.Module | None = None) -> None:
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
    if (vision_compressor is not None
            and any(parameter.requires_grad for parameter in vision_compressor.parameters())):
        raise RuntimeError("frozen teacher compressor has trainable parameters")
    off_device = [tuple(parameter.shape) for parameter in trainable if parameter.device.type != "cuda"]
    if off_device:
        raise RuntimeError(f"trainable parameters are not on CUDA: {off_device[:5]}")
    for layer, wrapper in enumerate(wrappers):
        if any(parameter.requires_grad for parameter in wrapper.inner.parameters()):
            raise RuntimeError(f"frozen RWKV TimeMix core is trainable at layer {layer}")


def eval_task_name(row: dict) -> str:
    task = str(row.get("task") or "caption").casefold()
    if task == "ocr":
        return "ocr"
    if task == "sam_mask":
        return "structured"
    return "caption"


def rotate_batch(values: Sequence):
    """Deterministically derange a batch for image-conditioning ablations."""
    if len(values) < 2:
        raise ValueError("a shuffled-image ablation requires at least two rows")
    return list(values[1:]) + [values[0]]


def rotate_feature_batch(values):
    """Rotate rows without confusing a prebatched native feature tuple."""
    if torch.is_tensor(values):
        if values.shape[0] < 2:
            raise ValueError(
                "a shuffled-image ablation requires at least two rows")
        return torch.roll(values, shifts=-1, dims=0)
    if (isinstance(values, tuple) and values
            and all(torch.is_tensor(value) for value in values)):
        batch = int(values[0].shape[0])
        if batch < 2 or any(value.shape[0] != batch for value in values):
            raise ValueError(
                "prebatched feature parts need one shared batch dimension")
        return tuple(torch.roll(value, shifts=-1, dims=0)
                     for value in values)
    return rotate_batch(values)


def stratified_eval_indices(rows: Sequence[dict], indices: Sequence[int],
                            max_examples: int) -> list[int]:
    """Round-robin tasks, then exhaust small tasks before caption overflow."""
    limit = min(len(indices), max_examples)
    groups: dict[str, list[int]] = {}
    for index in indices:
        groups.setdefault(eval_task_name(rows[index]), []).append(index)
    selected = []
    positions = {name: 0 for name in groups}
    names = sorted(groups)
    while len(selected) < limit:
        advanced = False
        for name in names:
            position = positions[name]
            if position < len(groups[name]) and len(selected) < limit:
                selected.append(groups[name][position])
                positions[name] += 1
                advanced = True
        if not advanced:
            break
    return selected


def build_eval_batches(rows: Sequence[dict], indices: Sequence[int], *,
                       batch_size: int, token_budget: int = 0
                       ) -> list[list[int]]:
    """Pack exact-shape eval rows under both row and padded-token limits."""
    if batch_size < 1:
        raise ValueError("eval batch size must be positive")

    def visual_width(index: int) -> int:
        return int(rows[index].get("_visual_tokens", 0) or 0)

    def group_key(index: int) -> tuple[int, bool]:
        # OCR has a second shuffled-image control forward. Isolating those rows
        # prevents one OCR sample from making a large caption batch run twice.
        return visual_width(index), eval_task_name(rows[index]) == "ocr"

    ordered = sorted(
        indices,
        key=lambda index: (
            group_key(index),
            visual_width(index) + len(rows[index]["tokens"]),
        ))
    batches: list[list[int]] = []
    batch_costs: list[int] = []
    for index in ordered:
        width = visual_width(index)
        cost = width + len(rows[index]["tokens"])
        starts_new = not batches or len(batches[-1]) >= batch_size
        if batches and group_key(batches[-1][0]) != group_key(index):
            starts_new = True
        if (batches and token_budget > 0 and
                (len(batches[-1]) + 1) * max(batch_costs[-1], cost)
                > token_budget):
            starts_new = True
        if starts_new:
            batches.append([])
            batch_costs.append(0)
        batches[-1].append(index)
        batch_costs[-1] = max(batch_costs[-1], cost)
    return batches


_STRUCTURED_EXAMPLE_METRICS = (
    "structured_loss", "structured_object_loss", "structured_box_l1",
    "structured_giou_loss", "structured_mask_bce",
    "structured_mask_dice_loss",
)
_STRUCTURED_INSTANCE_METRICS = (
    "structured_box_iou_instance", "structured_box_giou",
    "structured_mask_dice",
)


def _structured_eval_metric_name(name: str) -> str:
    """Keep auxiliary detection loss distinct from structured text NLL."""
    return "structured_aux_loss" if name == "structured_loss" else name


def _ocr_conditioning_coverage(
        *, shuffled_examples: float, total_examples: float,
        shuffled_tokens: float, total_tokens: float) -> dict[str, float]:
    if total_examples <= 0:
        return {}
    return {
        "ocr_image_conditioning_eval_examples": shuffled_examples,
        "ocr_image_conditioning_example_coverage": (
            shuffled_examples / total_examples),
        "ocr_image_conditioning_token_coverage": (
            shuffled_tokens / max(1.0, total_tokens)),
    }


def _pack_eval_metrics(metrics: dict[str, torch.Tensor], *,
                       shuffled: dict[str, torch.Tensor] | None = None
                       ) -> torch.Tensor:
    """Pack all per-batch metrics for one deferred device-to-host transfer."""
    loss_sums = metrics.pop("_eval_ce_sums").detach().float().reshape(-1)
    token_counts = metrics.pop("_eval_ce_counts").detach().float().reshape(-1)
    zero = loss_sums.new_zeros(1)

    def scalar(name: str) -> torch.Tensor:
        value = metrics.get(name)
        return (value.detach().float().reshape(1)
                if torch.is_tensor(value) else zero)

    values = [
        loss_sums, token_counts,
        scalar("_eval_coordinate_ce_sum"),
        scalar("_eval_coordinate_ce_count"),
        *(scalar(name) for name in _STRUCTURED_EXAMPLE_METRICS),
        *(scalar(name) for name in _STRUCTURED_INSTANCE_METRICS),
        scalar("structured_examples"),
        scalar("structured_positive_instances"),
    ]
    if shuffled is not None:
        values.extend((
            shuffled["_eval_ce_sums"].detach().float().reshape(-1),
            shuffled["_eval_ce_counts"].detach().float().reshape(-1),
        ))
    return torch.cat(values)


@torch.no_grad()
def evaluate(rows: Sequence[dict], indices: Sequence[int], *, rwkv: nn.Module,
             projector: nn.Module, vision: MoonViT,
             engram: LexicalMemoryBank | None, cache_dir: Path | None,
             batch_size: int, max_examples: int, token_budget: int = 0,
             deep_vision: DeepVisionInjector | None = None,
             layer_vision: LayerMatchedVisionInjector | None = None,
             fusion_tower: AlignedFrozenVisionFeatures | None = None,
             fusion_adapter: VisionFusionResidual | None = None,
             fusion_cache_dir: Path | None = None,
             vision_compressor: FrozenTeacherCompressor | None = None,
             structured_head: StructuredSpatialHead | None = None,
             structured_weight: float = 0.0,
             structured_coordinate_weight: float = 1.0,
             structured_invalid_box_weight: float = 0.0,
             structured_invalid_box_margin: float = 1.0,
             progress: Callable[[int, int], None] | None = None,
             ) -> dict[str, float]:
    if not indices:
        return {"loss": float("nan"), "ppl": float("nan")}
    eval_started = time.perf_counter()
    selected = stratified_eval_indices(rows, indices, max_examples)
    batches = build_eval_batches(
        rows, selected, batch_size=batch_size, token_budget=token_budget)
    completed = 0
    task_loss_sums: Counter[str] = Counter()
    task_token_counts: Counter[str] = Counter()
    structured_sums: Counter[str] = Counter()
    structured_examples = structured_instances = 0.0
    structured_coordinate_loss_sum = structured_coordinate_token_count = 0.0
    shuffled_ocr_loss_sum = shuffled_ocr_correct_sum = 0.0
    shuffled_ocr_token_count = 0.0
    shuffled_ocr_examples = 0.0
    ocr_eval_examples = float(sum(
        eval_task_name(rows[index]) == "ocr" for index in selected))
    packed_metrics: list[tuple[list[dict], torch.Tensor, bool]] = []
    cache_wait_s = 0.0
    metric_sync_s = 0.0
    prefetch = bool(
        batches and cache_dir is not None
        and isinstance(projector, RadioFeatureProjector)
        and getattr(vision, "v4h_native", False))

    def prepare_batch(chosen: Sequence[int]):
        batch_rows = [rows[index] for index in chosen]
        return prefetch_training_batch(
            batch_rows, vision, projector, cache_dir, engram,
            fusion_tower, fusion_cache_dir)

    feature_pool = (ThreadPoolExecutor(
        max_workers=1, thread_name_prefix="vision-eval-cache")
        if prefetch else None)
    transfer_stream = (
        torch.cuda.Stream(device=torch.cuda.current_device())
        if feature_pool is not None else None)

    def submit_batch(chosen: Sequence[int]):
        if feature_pool is None:
            return None
        future = feature_pool.submit(prepare_batch, chosen)
        return _CudaBatchPrefetch(
            future, device=torch.device("cuda", torch.cuda.current_device()),
            stream=transfer_stream)

    feature_future = submit_batch(batches[0]) if feature_pool is not None else None
    try:
      for batch_number, chosen in enumerate(batches):
        batch_rows = [rows[i] for i in chosen]
        wait_started = time.perf_counter()
        if feature_future is not None:
            prepared = feature_future.result()
        else:
            prepared = prepare_batch(chosen)
        cache_wait_s += time.perf_counter() - wait_started
        if feature_pool is not None and batch_number + 1 < len(batches):
            feature_future = submit_batch(batches[batch_number + 1])
        else:
            feature_future = None
        ids, labels, mask = (
            value.to("cuda", non_blocking=True)
            for value in prepared.text_batch)
        positions = prepared.positions.to("cuda", non_blocking=True)
        features = (
            prepared.native_features
            if prepared.native_features is not None else
            runtime_cached_features(
                batch_rows, vision, projector, cache_dir))
        eval_recall = prepared.recall
        fusion_features = (cached_fusion_features(
            batch_rows, fusion_tower,
            fusion_feature_tokens(fusion_tower, projector), fusion_cache_dir)
            if fusion_tower is not None and (fusion_adapter is not None
                                             or vision_compressor is not None)
            and fusion_cache_dir is not None else None)
        starts = visual_insert_positions(batch_rows)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _loss, batch_metrics = multimodal_loss(
                                      rwkv, projector, vision, (), ids, labels, mask,
                                      engram=engram, features=features,
                                      engram_recall=eval_recall,
                                      selected_positions=positions,
                                      deep_vision=deep_vision,
                                      layer_vision=layer_vision,
                                      visual_starts=starts,
                                      fusion_adapter=fusion_adapter,
                                      fusion_features=fusion_features,
                                      vision_compressor=vision_compressor,
                                      structured_head=structured_head,
                                      structured_rows=batch_rows,
                                      structured_weight=structured_weight,
                                      structured_coordinate_weight=(
                                          structured_coordinate_weight),
                                      structured_invalid_box_weight=(
                                          structured_invalid_box_weight),
                                      structured_invalid_box_margin=(
                                          structured_invalid_box_margin),
                                      return_per_example_ce=True,
                                      image_aspect=image_aspect_tensor(
                                          batch_rows, "cuda"))
        shuffled_metrics = None
        if len(batch_rows) > 1 and any(
                eval_task_name(row) == "ocr" for row in batch_rows):
            shuffled_rows = rotate_batch(batch_rows)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                _shuffled_loss, shuffled_metrics = multimodal_loss(
                    rwkv, projector, vision, (), ids, labels, mask,
                    engram=engram, features=rotate_feature_batch(features),
                    engram_recall=eval_recall,
                    selected_positions=positions,
                    deep_vision=deep_vision,
                    layer_vision=layer_vision,
                    visual_starts=starts,
                    fusion_adapter=fusion_adapter,
                    fusion_features=(rotate_feature_batch(fusion_features)
                                     if fusion_features is not None else None),
                    vision_compressor=vision_compressor,
                    return_per_example_ce=True,
                    image_aspect=image_aspect_tensor(shuffled_rows, "cuda"))
        packed_metrics.append((
            batch_rows,
            _pack_eval_metrics(batch_metrics, shuffled=shuffled_metrics),
            shuffled_metrics is not None,
        ))
        completed += len(chosen)
        if progress is not None:
            progress(completed, len(selected))
    finally:
        if feature_pool is not None:
            feature_pool.shutdown(wait=True, cancel_futures=True)

    if not packed_metrics:
        # A non-empty split can still stratify down to nothing to evaluate.
        return {"loss": float("nan"), "ppl": float("nan")}
    sync_started = time.perf_counter()
    host_metrics = (
        torch.cat([packed for _, packed, _ in packed_metrics]).cpu().tolist())
    metric_sync_s = time.perf_counter() - sync_started
    offset = 0
    scalar_count = (
        2 + len(_STRUCTURED_EXAMPLE_METRICS)
        + len(_STRUCTURED_INSTANCE_METRICS) + 2)
    for batch_rows, packed, has_shuffled in packed_metrics:
        width = len(batch_rows)
        count = packed.numel()
        values = host_metrics[offset:offset + count]
        offset += count
        cursor = 0
        loss_sums = values[cursor:cursor + width]
        cursor += width
        token_counts = values[cursor:cursor + width]
        cursor += width
        coordinate_loss_sum, coordinate_token_count = values[cursor:cursor + 2]
        cursor += 2
        example_values = values[
            cursor:cursor + len(_STRUCTURED_EXAMPLE_METRICS)]
        cursor += len(_STRUCTURED_EXAMPLE_METRICS)
        instance_values = values[
            cursor:cursor + len(_STRUCTURED_INSTANCE_METRICS)]
        cursor += len(_STRUCTURED_INSTANCE_METRICS)
        examples, instances = values[cursor:cursor + 2]
        cursor += 2
        if cursor != 2 * width + scalar_count:
            raise RuntimeError("eval metric packet layout drifted")
        structured_coordinate_loss_sum += coordinate_loss_sum
        structured_coordinate_token_count += coordinate_token_count
        for row, loss_sum, token_count in zip(
                batch_rows, loss_sums, token_counts):
            task = eval_task_name(row)
            task_loss_sums[task] += float(loss_sum)
            task_token_counts[task] += float(token_count)
        if examples:
            for name, value in zip(_STRUCTURED_EXAMPLE_METRICS, example_values):
                structured_sums[name] += float(value) * examples
            structured_examples += examples
        if instances:
            for name, value in zip(_STRUCTURED_INSTANCE_METRICS, instance_values):
                structured_sums[name] += float(value) * instances
            structured_instances += instances
        if has_shuffled:
            shuffled_sums = values[cursor:cursor + width]
            cursor += width
            shuffled_counts = values[cursor:cursor + width]
            cursor += width
            for row, correct_sum, shuffled_sum, shuffled_count in zip(
                    batch_rows, loss_sums, shuffled_sums, shuffled_counts):
                if eval_task_name(row) != "ocr":
                    continue
                shuffled_ocr_examples += 1.0
                shuffled_ocr_correct_sum += float(correct_sum)
                shuffled_ocr_loss_sum += float(shuffled_sum)
                shuffled_ocr_token_count += float(shuffled_count)
    total_loss = sum(task_loss_sums.values()) / max(
        1.0, sum(task_token_counts.values()))
    result = {"loss": total_loss, "ppl": math.exp(min(total_loss, 20.0))}
    for task in ("caption", "ocr", "structured"):
        if task_token_counts[task]:
            task_loss = task_loss_sums[task] / task_token_counts[task]
            result[f"{task}_loss"] = task_loss
            result[f"{task}_ppl"] = math.exp(min(task_loss, 20.0))
            result[f"{task}_eval_tokens"] = task_token_counts[task]
    if structured_examples:
        for name in ("structured_loss", "structured_object_loss",
                     "structured_box_l1", "structured_giou_loss",
                     "structured_mask_bce", "structured_mask_dice_loss"):
            output_name = _structured_eval_metric_name(name)
            result[output_name] = structured_sums[name] / structured_examples
        result["structured_eval_examples"] = structured_examples
    if structured_instances:
        result["structured_box_iou_instance"] = (
            structured_sums["structured_box_iou_instance"]
            / structured_instances)
        result["structured_box_giou"] = (
            structured_sums["structured_box_giou"] / structured_instances)
        result["structured_mask_dice"] = (
            structured_sums["structured_mask_dice"] / structured_instances)
        result["structured_eval_instances"] = structured_instances
    if structured_coordinate_token_count:
        coordinate_loss = (
            structured_coordinate_loss_sum / structured_coordinate_token_count)
        result["structured_coordinate_loss"] = coordinate_loss
        result["structured_coordinate_ppl"] = math.exp(min(coordinate_loss, 20.0))
        result["structured_coordinate_eval_tokens"] = (
            structured_coordinate_token_count)
    if shuffled_ocr_token_count:
        shuffled_loss = shuffled_ocr_loss_sum / shuffled_ocr_token_count
        paired_correct_loss = shuffled_ocr_correct_sum / shuffled_ocr_token_count
        result["ocr_shuffled_loss"] = shuffled_loss
        result["ocr_shuffled_ppl"] = math.exp(min(shuffled_loss, 20.0))
        result["ocr_image_conditioning_nll_delta"] = (
            shuffled_loss - paired_correct_loss)
        result["ocr_image_conditioning_eval_tokens"] = (
            shuffled_ocr_token_count)
    result.update(_ocr_conditioning_coverage(
        shuffled_examples=shuffled_ocr_examples,
        total_examples=ocr_eval_examples,
        shuffled_tokens=shuffled_ocr_token_count,
        total_tokens=task_token_counts["ocr"]))
    result.update(
        eval_wall_s=time.perf_counter() - eval_started,
        eval_cache_wait_s=cache_wait_s,
        eval_metric_sync_s=metric_sync_s,
        eval_batches=float(len(batches)),
        eval_mean_batch=(len(selected) / max(1, len(batches))),
        eval_visual_token_budget=float(token_budget),
        eval_cache_prefetch=float(prefetch),
        eval_recall_prefetch=float(prefetch and engram is not None),
    )
    return result


def select_eval_sample_indices(rows: Sequence[dict], indices: Sequence[int],
                               count: int,
                               required_tasks: dict[str, int] | None = None
                               ) -> list[int]:
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
    output = output[:n]
    requirements = {
        str(task): min(n, max(0, int(quota)))
        for task, quota in (required_tasks or {}).items()
    }
    if not requirements:
        return output
    if sum(requirements.values()) > n:
        raise ValueError("required qualitative task counts exceed gallery size")

    selected = set(output)
    task_counts = Counter(eval_task_name(rows[index]) for index in output)
    for task, quota in sorted(requirements.items()):
        candidates = [
            index for index in indices
            if eval_task_name(rows[index]) == task and index not in selected
        ]
        while task_counts[task] < quota and candidates:
            candidate = candidates.pop(len(candidates) // 2)
            replace = next((
                position for position in range(len(output) - 1, -1, -1)
                if task_counts[eval_task_name(rows[output[position]])]
                > requirements.get(eval_task_name(rows[output[position]]), 0)
            ), None)
            if replace is None:
                break
            removed = output[replace]
            selected.remove(removed)
            task_counts[eval_task_name(rows[removed])] -= 1
            output[replace] = candidate
            selected.add(candidate)
            task_counts[task] += 1
    return output


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


def qualitative_eval_due(step: int, *, eval_every: int,
                         sample_every: int) -> bool:
    """Return whether a scalar-eval step also owes free caption decoding.

    A zero qualitative cadence preserves the historical behavior by following
    ``eval_every``. This separates frequent teacher-forced PPL from the much
    more expensive human-review gallery without changing either computation.
    """
    cadence = sample_every or eval_every
    return bool(step > 0 and cadence > 0 and step % cadence == 0)


@torch.no_grad()
def write_eval_samples(rows: Sequence[dict], indices: Sequence[int], *, step: int,
                       ppl: float, rwkv: nn.Module, projector: nn.Module,
                       vision: MoonViT, engram: LexicalMemoryBank | None,
                       cache_dir: Path | None, vocab: WorldVocab, prompt: str,
                       out: Path, count: int, max_new: int,
                       ocr_count: int = 2,
                       structured_count: int = 2,
                       structured_head: StructuredSpatialHead | None = None,
                       deep_vision: DeepVisionInjector | None = None,
                       layer_vision: LayerMatchedVisionInjector | None = None,
                       sandwich_prompt: bool = False,
                       sandwich_lead_prompt: str = "",
                       fusion_tower: AlignedFrozenVisionFeatures | None = None,
                       fusion_adapter: VisionFusionResidual | None = None,
                       fusion_cache_dir: Path | None = None,
                       vision_compressor: FrozenTeacherCompressor | None = None,
                       wrappers: Sequence[nn.Module] = (),
                       progress: Callable[[int, int], None] | None = None) -> Path | None:
    """Greedily caption a fixed spread of held-out images for chart drill-down."""
    if count <= 0 or max_new <= 0 or not indices:
        return None
    chosen = select_eval_sample_indices(
        rows, indices, count,
        required_tasks={
            "ocr": min(ocr_count, count),
            "structured": min(structured_count, max(0, count - ocr_count)),
        })
    sample_rows = [rows[index] for index in chosen]
    prompt_ids = [vocab.encode(str(row.get("prompt") or prompt)) for row in sample_rows]
    lead_ids = [vocab.encode(
        sandwich_lead_prompt or str(row.get("prompt") or prompt))
        for row in sample_rows]
    generated: list[list[int]] = [[] for _ in sample_rows]
    structured_head_outputs: list[dict | None] = [
        None for _ in sample_rows]
    stopped = [False] * len(sample_rows)
    item_complete = [False] * len(sample_rows)
    item_steps = [0] * len(sample_rows)
    artifact = out / "eval_samples" / f"step_{step:08d}.json"
    artifact.parent.mkdir(parents=True, exist_ok=True)
    total_work = len(sample_rows) * max_new
    generation_started = time.perf_counter()
    prefill_elapsed_s = 0.0
    prefill_model_calls = 0
    decode_elapsed_s = 0.0
    decode_model_calls = 0
    decode_model_row_steps = 0
    decode_dense_row_steps = 0

    def persist(*, complete: bool, generation_steps: int,
                progress_completed: int) -> None:
        decoded = [vocab.decode(tokens).strip() for tokens in generated]
        ocr_indices = [
            i for i, row in enumerate(sample_rows)
            if str(row.get("task") or "").casefold() == "ocr"
        ]
        structured_pairs = [
            (str(row["text"]), decoded[i])
            for i, row in enumerate(sample_rows)
            if str(row.get("task") or "").casefold() == "sam_mask"
        ]
        payload = {
            "step": step, "ppl": ppl, "decoding": "greedy_recurrent_cache",
            "max_new": max_new,
            "complete": complete, "generation_steps": generation_steps,
            "progress_completed": progress_completed,
            "progress_total": total_work,
            "generation_elapsed_s": time.perf_counter() - generation_started,
            "prefill_elapsed_s": prefill_elapsed_s,
            "prefill_model_calls": prefill_model_calls,
            "decode_elapsed_s": decode_elapsed_s,
            "decode_model_calls": decode_model_calls,
            "decode_model_row_steps": decode_model_row_steps,
            "decode_dense_row_steps": decode_dense_row_steps,
            "items": [{
                "image": str(row["image"].resolve()),
                "prompt": str(row.get("prompt") or prompt),
                "reference": row["text"],
                "caption": decoded[i],
                # Parallel native-grid output from the task-specific head. It
                # is independent of the autoregressive caption and therefore
                # remains usable even when text generation hits max_new.
                "structured_head": structured_head_outputs[i],
                "tokens": len(tokens), "stopped_at_eod": stopped[i],
                "complete": item_complete[i], "generation_steps": item_steps[i],
                "source": str(row.get("stage1_source") or row.get("source") or "unknown"),
            } for i, (row, tokens) in enumerate(zip(sample_rows, generated))],
        }
        # Structured metrics are deferred to the completed artifact for the same
        # reason as the OCR ones below, but for accuracy rather than cost: a
        # partial decode truncates the final instance mid-line, and a truncated
        # line counts against ``precision_at_50``'s denominator, so intermediate
        # samples read as a systematic regression that resolves itself at the
        # end. An absent number is better than a misleading one.
        if complete and structured_pairs:
            payload["structured_generation"] = structured_generation_metrics(
                structured_pairs)
        # Edit distance over long document targets is intentionally deferred
        # until the completed artifact. Intermediate polling writes should stay
        # cheap enough not to interrupt autoregressive decoding.
        if complete and ocr_indices:
            payload["ocr_generation"] = ocr_generation_metrics(
                [(str(sample_rows[i]["text"]), decoded[i]) for i in ocr_indices],
                stopped_at_eod=[stopped[i] for i in ocr_indices],
                maxed_out=[
                    not stopped[i] and item_steps[i] >= max_new
                    for i in ocr_indices
                ],
            )
        _atomic_json(artifact, payload, durable=complete)

    # Publish the image/reference skeleton before the expensive autoregressive
    # phase. The dashboard can render and poll it while captions are still being
    # produced, and a hard stop cannot erase the already-computed scalar eval.
    persist(complete=False, generation_steps=0, progress_completed=0)
    generation_steps = 0
    completed_work = 0
    try:
        # RADIO images have variable visual-prefix widths. Prefill each row
        # independently so no fake visual tokens enter recurrence, then stack
        # the resulting positionless RWKV states and decode the gallery as one
        # batch. This combines exact variable-tile prefills with one model call
        # per token round instead of one call per image per token.
        row_caches = [None] * len(sample_rows)
        refinement_snapshots = [None] * len(sample_rows)
        has_refinement_passes = any(
            wrapper.loop.n_loops > 1 for wrapper in wrappers)
        first_tokens = [0] * len(sample_rows)
        engram_streams = [None] * len(sample_rows) if engram is not None else []
        prefill_started = time.perf_counter()
        prefill_groups: dict[tuple, list[int]] = {}
        for i, row in enumerate(sample_rows):
            sequence = ((lead_ids[i] + prompt_ids[i]) if sandwich_prompt
                        else prompt_ids[i])
            start = len(lead_ids[i]) if sandwich_prompt else 0
            shape_value = row.get("_structured_mask_shape")
            group_key = (
                int(row.get("_visual_tokens", 0) or 0),
                len(sequence), start,
                str(row.get("task") or "") == "sam_mask",
                tuple(map(int, shape_value)) if shape_value is not None else None,
            )
            prefill_groups.setdefault(group_key, []).append(i)

        for group_indices in prefill_groups.values():
            prefill_model_calls += 1
            group_rows = [sample_rows[i] for i in group_indices]
            row_features = runtime_cached_features(
                group_rows, vision, projector, cache_dir)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                if vision_compressor is not None:
                    if fusion_tower is None or fusion_cache_dir is None:
                        raise ValueError("compressor eval requires paired feature caches")
                    fusion_features = cached_fusion_features(
                        group_rows, fusion_tower,
                        fusion_feature_tokens(fusion_tower, projector),
                        fusion_cache_dir)
                    canonical = vision_compressor(row_features, fusion_features)
                    prefix = projector(list(canonical.unbind(0)))
                else:
                    prefix = project_visual_prefix(
                        projector, row_features,
                        image_aspect=image_aspect_tensor(group_rows, "cuda"))
                if (vision_compressor is None and fusion_tower is not None
                        and fusion_adapter is not None):
                    if fusion_cache_dir is None:
                        raise ValueError("fusion eval requires its feature cache")
                    fusion_features = cached_fusion_features(
                        group_rows, fusion_tower,
                        fusion_feature_tokens(fusion_tower, projector),
                        fusion_cache_dir)
                    prefix = add_fusion_residual(
                        prefix, fusion_adapter(fusion_features))

            sequences = [
                ((lead_ids[i] + prompt_ids[i]) if sandwich_prompt
                 else prompt_ids[i])
                for i in group_indices
            ]
            starts = tuple(
                len(lead_ids[i]) if sandwich_prompt else 0
                for i in group_indices)
            ids = _pin_tensor(torch.tensor(
                sequences, dtype=torch.long)).to("cuda", non_blocking=True)
            reset_loop_inference_cache(wrappers)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                text = rwkv.model.embeddings(ids)
                typed_prefix = prefix.to(text.dtype)
                if (structured_head is not None
                        and str(group_rows[0].get("task") or "") == "sam_mask"):
                    mask_shapes = (
                        [tuple(map(int, row["_structured_mask_shape"]))
                         for row in group_rows]
                        if group_rows[0].get("_structured_mask_shape") is not None
                        else None)
                    head_prediction = structured_head(
                        typed_prefix, text[:, starts[0]:].mean(dim=1),
                        mask_shapes=mask_shapes)
                    for local_row, sample_index in enumerate(group_indices):
                        structured_head_outputs[sample_index] = {
                            "format": (
                                "native_feature_grid"
                                if mask_shapes is not None else "mask16"),
                            "instances": structured_prediction_instances(
                                head_prediction, row=local_row),
                        }
                embeds = insert_visual_span(text, typed_prefix, starts)
                attention_mask = torch.ones(
                    embeds.shape[:2], dtype=torch.bool, device=embeds.device)
                if engram is not None:
                    boundary = 0 if engram.boundary_id is None else int(engram.boundary_id)
                    lexical_ids = insert_boundary_ids(
                        ids, starts, prefix.shape[1], boundary)
                    engram.set_input_ids(lexical_ids)
                with contextlib.ExitStack() as stack:
                    if deep_vision is not None:
                        # Reinject RADIO only during the visual prefill. Its
                        # effect is retained by RWKV's recurrent state.
                        stack.enter_context(
                            deep_vision.use_prefix(typed_prefix, starts))
                    if layer_vision is not None:
                        stack.enter_context(layer_vision.use_features(
                            torch.stack(row_features), starts))
                    output = rwkv.model(
                        inputs_embeds=embeds, attention_mask=attention_mask,
                        output_hidden_states=False, use_cache=True,
                        return_dict=True)
                position = prefix.shape[1] + len(sequences[0]) - 1
                logits = rwkv.lm_head(
                    output.last_hidden_state[:, position]).float()
                if engram is not None:
                    logits = engram.logit_bias_at(
                        logits, torch.arange(
                            len(group_indices), dtype=torch.long,
                            device="cuda"),
                        torch.full(
                            (len(group_indices),), position,
                            dtype=torch.long, device="cuda"),
                        inplace=True)
                logits[:, 0] = -torch.inf
                tokens = logits.argmax(-1).tolist()
            group_cache = output.past_key_values
            group_refinement = (
                capture_loop_refinement_caches(
                    wrappers, owner=group_cache)
                if has_refinement_passes else None)
            for local_row, (sample_index, token) in enumerate(
                    zip(group_indices, tokens)):
                row_caches[sample_index] = select_legacy_fla_cache_rows(
                    group_cache, [local_row])
                first_tokens[sample_index] = int(token)
                if group_refinement is not None:
                    refinement_snapshots[sample_index] = (
                        select_loop_refinement_cache_rows(
                            group_refinement, [local_row]))
            if has_refinement_passes:
                reset_loop_inference_cache(wrappers)
            if engram is not None:
                # Visual boundary tokens reset suffix recall. Everything
                # before the final boundary is unreachable during decoding,
                # so seed the exact streaming automaton and causal ShortConv
                # windows from only that suffix.
                assert engram.last_recall is not None
                for local_row, sample_index in enumerate(group_indices):
                    last_boundary = starts[local_row] + prefix.shape[1] - 1
                    stream_ids = lexical_ids[
                        local_row:local_row + 1, last_boundary:]
                    stream_recall = type(engram.last_recall)(*(
                        value[local_row:local_row + 1, last_boundary:]
                        for value in engram.last_recall))
                    engram_streams[sample_index] = engram.begin_streaming(
                        stream_ids, recall=stream_recall)
        prefill_elapsed_s = time.perf_counter() - prefill_started

        if any(value is None for value in row_caches):
            raise RuntimeError("grouped eval prefill did not produce every cache")
        cache = stack_legacy_fla_caches(row_caches)
        if has_refinement_passes:
            if any(value is None for value in refinement_snapshots):
                raise RuntimeError(
                    "grouped eval prefill did not produce every refinement cache")
            stacked_refinement = stack_loop_refinement_cache_snapshots(
                refinement_snapshots)
            restore_loop_refinement_caches(
                wrappers, stacked_refinement, owner=cache)
        batched_engram = (BatchedStreamingEngramState(engram_streams)
                          if engram is not None else None)
        tokens = torch.tensor(first_tokens, dtype=torch.long, device="cuda")
        active_rows = list(range(len(sample_rows)))
        decode_started = time.perf_counter()

        for decode_round in range(1, max_new + 1):
            token_values = tokens.tolist()  # one synchronization per batch round
            completed_this_round = False
            remaining_positions = []
            for position, (i, token) in enumerate(zip(active_rows, token_values)):
                generation_steps += 1
                item_steps[i] += 1
                if token == SEP:
                    stopped[i] = True
                    item_complete[i] = True
                    completed_this_round = True
                else:
                    generated[i].append(int(token))
                    if item_steps[i] >= max_new:
                        item_complete[i] = True
                        completed_this_round = True
                    else:
                        remaining_positions.append(position)

            completed_work = sum(
                max_new if complete else steps
                for complete, steps in zip(item_complete, item_steps))
            if completed_this_round or decode_round % 64 == 0:
                persist(complete=False, generation_steps=generation_steps,
                        progress_completed=completed_work)
                if progress is not None:
                    progress(completed_work, total_work)
            if not remaining_positions:
                break

            # Recurrent states are independent along batch dimension. Remove
            # completed rows before the next model call so short captions do
            # not pay for every remaining 768-token OCR generation round.
            if len(remaining_positions) != len(active_rows):
                current_refinement = (
                    capture_loop_refinement_caches(wrappers, owner=cache)
                    if has_refinement_passes else None)
                cache = select_legacy_fla_cache_rows(
                    cache, remaining_positions)
                if current_refinement is not None:
                    selected_refinement = select_loop_refinement_cache_rows(
                        current_refinement, remaining_positions)
                    restore_loop_refinement_caches(
                        wrappers, selected_refinement, owner=cache)
                if batched_engram is not None:
                    batched_engram.select_rows(remaining_positions)
                selected = torch.tensor(
                    remaining_positions, dtype=torch.long, device=tokens.device)
                tokens = tokens.index_select(0, selected)
                token_values = [token_values[position]
                                for position in remaining_positions]
                active_rows = [active_rows[position]
                               for position in remaining_positions]

            token_ids = tokens.unsqueeze(1)
            if batched_engram is not None:
                # Reuse the host token list from the single argmax sync above.
                batched_engram.step(token_values)
            decode_model_calls += 1
            decode_model_row_steps += len(active_rows)
            decode_dense_row_steps += len(sample_rows)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output = rwkv.model(
                    inputs_embeds=rwkv.model.embeddings(token_ids),
                    attention_mask=torch.ones_like(token_ids, dtype=torch.bool),
                    past_key_values=cache, use_cache=True,
                    output_hidden_states=False, return_dict=True)
                logits = rwkv.lm_head(output.last_hidden_state[:, -1]).float()
                if engram is not None:
                    rows_selector = torch.arange(
                        len(active_rows), dtype=torch.long, device="cuda")
                    logits = engram.logit_bias_at(
                        logits, rows_selector, torch.zeros_like(rows_selector),
                        inplace=True)
                logits[:, 0] = -torch.inf
                tokens = logits.argmax(-1)
            cache = output.past_key_values

        decode_elapsed_s = time.perf_counter() - decode_started
        completed_work = total_work
        if progress is not None:
            progress(completed_work, total_work)
    except BaseException:
        persist(complete=False, generation_steps=generation_steps,
                progress_completed=completed_work)
        raise
    persist(complete=True, generation_steps=generation_steps,
            progress_completed=total_work)
    return artifact


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--eval-data", nargs="+", default=None,
                    help="held-out image/text manifests; never sampled for training")
    ap.add_argument("--rwkv", default="models/rwkv7-g1h-2.9b-20260710-ctx10240.pth")
    ap.add_argument("--vision-backend", choices=("moonvit", "radio1d", "radio_v4h"),
                    default="moonvit")
    ap.add_argument("--moonvit", default="models/kimi-k2.6-moonvit/model-00064-of-000064.safetensors")
    ap.add_argument("--radio-model", default="models/vision/C-RADIOv4-1D-H")
    ap.add_argument("--radio-revision",
                    default="e18692120c7a3203496e1a99056a4149ede135fc")
    ap.add_argument("--radio-max-detail-tiles", type=int,
                    default=DEFAULT_MAX_DETAIL_TILES)
    ap.add_argument("--radio-adaptive-token-threshold", type=int,
                    default=DEFAULT_ADAPTIVE_TOKEN_THRESHOLD,
                    help="use 256 native tokens/tile through this total tile count, then 128")
    ap.add_argument("--radio-tile-batch", type=int, default=8)
    ap.add_argument("--radio-v4h-model", default="models/vision/C-RADIOv4-H")
    ap.add_argument("--radio-v4h-revision",
                    default=_DEFAULT_NATIVE_REVISION,
                    help="must match what wrote the cache; a mismatch presents "
                         "as an entirely absent cache, not as a version error")
    ap.add_argument("--radio-v4h-lattice", type=int, default=16,
                    help="pooled tokens per tile axis for --vision-backend radio_v4h")
    ap.add_argument("--radio-v4h-native", action=argparse.BooleanOptionalAction,
                    default=True,
                    help="encode whole images at native resolution instead of "
                         "tiling; tiling only existed because RADIO1D emitted a "
                         "fixed token count per tile")
    ap.add_argument("--radio-v4h-native-packing",
                    choices=("pair_columns", "cells"),
                    default="pair_columns",
                    help="legacy generic caches concatenate adjacent 1280-wide "
                         "cells; fused 4096-wide adaptor caches use one token "
                         "per native cell")
    ap.add_argument("--radio-v4h-feature-width", type=int, default=1280,
                    help="cached native feature width before optional pairing")
    ap.add_argument("--radio-v4h-adaptors",
                    action=argparse.BooleanOptionalAction, default=None,
                    help="whether the cache was written by the fused teacher "
                         "adaptors, which the artifact fingerprint must then "
                         "cover; inferred from --radio-v4h-feature-width when "
                         "unset")
    ap.add_argument("--radio-v4h-cache-only",
                    action=argparse.BooleanOptionalAction, default=False,
                    help="skip loading the frozen image encoder because native "
                         "cache misses are fatal during training")
    ap.add_argument("--radio-v4h-max-edge", type=int, default=2048)
    ap.add_argument("--radio-v4h-pair-axis", choices=("columns", "rows"),
                    default="columns",
                    help="axis paired to reach the LM width; 'columns' retains "
                         "more horizontal resolution, the scarce axis for text")
    ap.add_argument("--radio-v4h-bridge-rank", type=int, default=256,
                    help="bridge bottleneck; 256 matches the RADIO1D arm exactly "
                         "because paired tokens need no input projection")
    ap.add_argument("--radio-adaptive-complexity",
                    action=argparse.BooleanOptionalAction, default=False,
                    help="content-route each tile's RADIO prefix under a fixed unpadded batch total")
    ap.add_argument("--radio-complexity-budget-ratio", type=float,
                    default=DEFAULT_COMPLEXITY_BUDGET_RATIO)
    ap.add_argument("--radio-complexity-token-quantum", type=int,
                    default=DEFAULT_COMPLEXITY_TOKEN_QUANTUM)
    ap.add_argument("--out", default="runs/moonvit_rwkv_stage1_v3")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--min-batch", type=int, default=0,
                    help="minimum captions for token-budget batches; 0 uses --batch")
    ap.add_argument("--max-batch", type=int, default=8,
                    help="maximum captions when token-budget batching is enabled")
    ap.add_argument("--target-batch-tokens", type=int, default=0,
                    help="padded image-prefix + text tokens per step; 0 keeps fixed --batch")
    ap.add_argument("--ocr-update-ratio", type=float, default=0.0,
                    help="repeat OCR examples to reach this training-sampler "
                         "share; 0 preserves the manifest's natural ratio")
    ap.add_argument("--structured-update-ratio", type=float, default=0.0,
                    help="repeat sam_mask examples to reach this training-sampler "
                         "share; balanced simultaneously with OCR")
    ap.add_argument("--activation-checkpoint-min-tokens", type=int, default=0,
                    help="checkpoint safe decoder layers at or above this full sequence length; 0 disables")
    ap.add_argument("--activation-checkpoint-max-layers", type=int, default=0,
                    help="maximum eligible decoder layers checkpointed on long "
                         "rows; 0 checkpoints every eligible layer")
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
    ap.add_argument("--deep-vision-grouped-precompute",
                    action=argparse.BooleanOptionalAction, default=False,
                    help="precompute all deep-vision adapters with grouped "
                         "matrix multiplications before the frozen decoder")
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
    ap.add_argument("--sam-fusion", action=argparse.BooleanOptionalAction,
                    default=False,
                    help="add a frozen global SAM ViT-B residual to RADIO. NOTE: "
                         "C-RADIOv4-1D-H already distills SAM3 as a dense teacher, "
                         "so this adds an older, smaller SAM on top of a stronger "
                         "one the backbone has absorbed; measure before adopting")
    ap.add_argument("--sam-fusion-tokens", type=int, default=128)
    ap.add_argument("--vision-fusion-rank", type=int, default=512)
    ap.add_argument("--siglip2-model",
                    default="models/vision/siglip2-so400m-patch16-512")
    ap.add_argument("--siglip2-width", type=int, default=1152)
    ap.add_argument("--dinov2-model", default="models/vision/dinov2-base")
    ap.add_argument("--sam-model", default="models/vision/sam-vit-base")
    ap.add_argument("--sam-crop-padding", action=argparse.BooleanOptionalAction,
                    default=False,
                    help="crop SAM's zero-padded canvas and pool it on a 2D "
                         "lattice; off by default because the shipped frozen "
                         "compressor was trained on the uncropped features. "
                         "Enabling it repartitions the fusion feature cache.")
    ap.add_argument("--fusion-feature-cache",
                    default="caches/siglip2_dinov2_sam_aligned_v1")
    ap.add_argument("--vision-compressor-checkpoint", default="",
                    help="frozen six-teacher 128x1024 compressor; replaces the Moon-only prefix bridge")
    ap.add_argument("--grounding-early-tokens", type=int, default=0,
                    help="caption-opening targets receiving extra CE weight")
    ap.add_argument("--grounding-early-weight", type=float, default=1.0)
    ap.add_argument("--grounding-contrastive-weight", type=float, default=0.0,
                    help="in-batch image/text contrastive auxiliary weight")
    ap.add_argument("--grounding-contrastive-dim", type=int, default=512)
    ap.add_argument("--grounding-temperature", type=float, default=0.07)
    ap.add_argument("--structured-head", action=argparse.BooleanOptionalAction,
                    default=False,
                    help="set-matched DETR-style box/mask auxiliary for sam_mask rows")
    ap.add_argument("--structured-weight", type=float, default=1.0)
    ap.add_argument("--structured-lr", type=float, default=None,
                    help="structured-head AdamW rate; omit to follow --lr. "
                         "0 freezes the head rather than meaning 'unset'")
    ap.add_argument("--structured-coordinate-weight", type=float, default=1.0,
                    help="extra teacher-forced CE weight on box coordinate tokens")
    ap.add_argument("--structured-invalid-box-weight", type=float, default=0.0,
                    help="unlikelihood weight against false 000/999 coordinate starts")
    ap.add_argument("--structured-invalid-box-margin", type=float, default=1.0,
                    help="logit margin separating false boundary starts from the target")
    ap.add_argument("--structured-width", type=int, default=256)
    ap.add_argument("--structured-object-queries", type=int, default=16)
    ap.add_argument("--structured-spatial-layers", type=int, default=2)
    ap.add_argument("--structured-object-layers", type=int, default=2)
    ap.add_argument("--structured-heads", type=int, default=8)
    ap.add_argument("--prompt", default="Describe this image:\n")
    ap.add_argument("--sandwich-prompt", action=argparse.BooleanOptionalAction,
                    default=False,
                    help="train prompt -> image -> repeated prompt -> caption")
    ap.add_argument("--sandwich-lead-prompt", default="",
                    help="asymmetric text before the image; empty repeats the task prompt")
    ap.add_argument("--max-text-tokens", type=int, default=384)
    ap.add_argument("--allow-text-limit-increase-from", type=int, default=0,
                    help="resume a checkpoint made at this smaller text limit; verifies its old fingerprint")
    ap.add_argument("--max-input-patches", type=int, default=1024)
    ap.add_argument("--feature-cache", default="caches/moonvit_features_stage1_v3")
    ap.add_argument("--preload-feature-cache", action="store_true",
                    help="keep deserialized cached MoonViT features in system RAM")
    ap.add_argument("--feature-cache-max-bytes", type=int, default=16 * 2**30,
                    help="approximate RAM budget for the in-process feature LRU; "
                         "0 = unbounded; --preload-feature-cache implies unbounded")
    ap.add_argument("--background-feature-preload", action="store_true",
                    help="warm the RAM cache asynchronously while training starts")
    ap.add_argument("--manifest-stat-workers", type=int, default=1,
                    help="parallel source-file checks during manifest loading")
    ap.add_argument("--prefetch-next-batch", action=argparse.BooleanOptionalAction, default=True,
                    help="CPU-load the exact next batch while the GPU trains the current one")
    ap.add_argument("--prefetch-cuda-transfer",
                    action=argparse.BooleanOptionalAction, default=False,
                    help="copy the pinned prefetched batch on a dedicated CUDA "
                         "stream while the current step is still computing")
    ap.add_argument("--loop-count", type=int, default=2)
    ap.add_argument("--loop-start-step", type=int, default=250,
                    help="warm up the bridge before enabling per-layer TimeMix refinement")
    ap.add_argument("--loop-ramp-steps", type=int, default=1000,
                    help="linearly ramp effective loop gates after activation; 0 disables")
    ap.add_argument("--loop-gate-cap", type=float, default=0.25)
    ap.add_argument("--loop-index", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--reset-loop-on-resume", action="store_true",
                    help="one-time recovery: zero loop adapters and their optimizer moments")
    ap.add_argument("--allow-loop-count-increase-from", type=int, default=0,
                    help="one-time exact-resume migration from this smaller "
                         "pass count; appends zero-init loop and optimizer state")
    ap.add_argument("--allow-loop-lr-change-from", type=float, default=None,
                    help="one-time exact-resume migration from this checkpoint "
                         "--loop-lr; retains loop tensors and Adam moments. "
                         "Naming the source expires the permission once the "
                         "migrated step is saved, so a launcher that passes it "
                         "on every restart cannot waive a later LR drift")
    # Deprecated legacy spelling of the above. A supervisor loop that bash
    # already parsed into memory keeps relaunching the command line it started
    # with, so removing this outright would kill a live run at its next
    # --restart-before-eval boundary with "unrecognized arguments". Accepted
    # with its original permissive meaning; prefer the self-expiring
    # --allow-loop-lr-change-from for new launches.
    ap.add_argument("--allow-loop-lr-change-on-resume", action="store_true",
                    help=argparse.SUPPRESS)
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
    ap.add_argument("--restart-before-eval",
                    action=argparse.BooleanOptionalAction, default=False,
                    help="recreate CUDA only before recurrent qualitative "
                         "generation; scalar teacher-forced eval stays in-process")
    ap.add_argument("--eval-examples", type=int, default=64)
    ap.add_argument("--eval-batch-size", type=int, default=0,
                    help="teacher-forced eval batch cap; 0 uses --batch")
    ap.add_argument("--eval-batch-tokens", type=int, default=0,
                    help="teacher-forced padded visual+text token budget; "
                         "0 disables budget packing")
    ap.add_argument("--eval-samples", type=int, default=4,
                    help="held-out images greedily captioned at each eval marker")
    ap.add_argument("--eval-ocr-samples", type=int, default=2,
                    help="minimum OCR rows reserved in each qualitative gallery")
    ap.add_argument("--eval-structured-samples", type=int, default=2,
                    help="minimum structured rows reserved in each qualitative gallery")
    ap.add_argument("--eval-sample-every", type=int, default=0,
                    help="qualitative caption cadence; 0 follows --eval-every")
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
    ap.add_argument("--allow-coco-prompt-migration", action="store_true",
                    help="resume only when the prior data fingerprint exactly matches "
                         "the legacy ambiguous COCO mask prompt; preserves optimizer "
                         "and sampler state while resetting eval claims")
    ap.add_argument("--allow-structured-data-removal-from", default="",
                    help="one-time source checkpoint data fingerprint allowed "
                         "to resume after every sam_mask train/eval row is "
                         "removed; preserves model/optimizer and resets sampler "
                         "and eval claims")
    ap.add_argument("--allow-structured-lr-change-from", type=float,
                    default=None,
                    help="one-time source structured-head learning rate allowed "
                         "to change on exact resume; preserves tensors and Adam "
                         "moments, then expires when the migrated step is saved")
    ap.add_argument("--allow-v4h-fingerprint-change-from", default="",
                    help="one-time source checkpoint radio_v4h_fingerprint "
                         "allowed to resume after the C-RADIOv4-H producer "
                         "sources change; the artifact digest hashes those "
                         "modules, so any edit to them -- even one that writes "
                         "no cache -- otherwise makes every existing radio_v4h "
                         "checkpoint unresumable. Naming the source expires the "
                         "permission once the migrated step is saved")
    ap.add_argument("--init-adapters-from", default=None,
                    help="warm-start adapters from another run; resets optimizer/sampler/step")
    ap.add_argument("--init-text-adapters-from", default=None,
                    help="warm-start non-vision adapters while replacing the vision bridge")
    ap.add_argument("--fresh", action="store_true",
                    help="explicitly archive an existing run and start over; incompatible with resume")
    args = ap.parse_args()
    # Historical checkpoints used 0 as an "inherit --lr" sentinel. New
    # checkpoints carry this marker so an explicit zero remains a real frozen
    # optimizer group on every subsequent exact resume.
    args.structured_lr_zero_is_freeze = True

    # A supervisor's default SIGTERM must reuse the KeyboardInterrupt
    # checkpoint-and-pause path instead of dying mid-step and stranding
    # status.json at "training". Background work here is thread-based (no
    # forked dataloader workers), and CPython only delivers signal handlers in
    # the main thread of the process that installed them.
    def _sigterm_to_interrupt(signum, frame) -> None:
        raise KeyboardInterrupt

    # Explicitly install SIGINT too. A trainer launched by an asynchronous
    # shell can inherit SIGINT=SIG_IGN, which otherwise makes the dashboard's
    # documented graceful-stop signal a no-op.
    signal.signal(signal.SIGINT, _sigterm_to_interrupt)
    signal.signal(signal.SIGTERM, _sigterm_to_interrupt)

    if args.feature_cache_max_bytes < 0:
        raise SystemExit("--feature-cache-max-bytes must be non-negative")
    # Preload mode is the operator's assertion that the corpus fits in RAM;
    # everything else gets a bounded LRU so long runs cannot grow until OOM.
    _FEATURE_MEMORY_CACHE.max_bytes = (
        0 if args.preload_feature_cache else int(args.feature_cache_max_bytes))

    if args.init_adapters_from and args.init_text_adapters_from:
        raise SystemExit("choose only one adapter initialization source")
    if ((args.init_adapters_from or args.init_text_adapters_from)
            and args.resume != "none"):
        raise SystemExit("adapter initialization requires --resume none")
    if args.init_text_adapters_from and not is_radio_backend(args):
        raise SystemExit("--init-text-adapters-from is reserved for a replaced RADIO bridge")

    if args.steps < 0:
        raise SystemExit("--steps must be non-negative")
    if args.batch < 1 or args.max_batch < 1:
        raise SystemExit("--batch and --max-batch must be positive")
    if args.loop_count < 1:
        raise SystemExit("--loop-count must be at least 1")
    if args.allow_loop_count_increase_from < 0:
        raise SystemExit("--allow-loop-count-increase-from must be non-negative")
    if args.max_batch < args.batch:
        raise SystemExit("--max-batch must be at least --batch")
    if args.min_batch < 0 or (args.min_batch and args.min_batch > args.max_batch):
        raise SystemExit("--min-batch must be zero or between 1 and --max-batch")
    if (not 0 <= args.ocr_update_ratio < 1
            or not 0 <= args.structured_update_ratio < 1
            or args.ocr_update_ratio + args.structured_update_ratio >= 1):
        raise SystemExit(
            "OCR/structured update ratios must be non-negative and sum below one")
    if args.allow_text_limit_increase_from < 0:
        raise SystemExit("--allow-text-limit-increase-from must be non-negative")
    if args.max_text_tokens < 2 or args.prefix_tokens < 1:
        raise SystemExit("--max-text-tokens and --prefix-tokens must be positive")
    if args.max_input_patches < 4:
        raise SystemExit("--max-input-patches must be at least 4")
    if (args.radio_max_detail_tiles < 1 or args.radio_tile_batch < 1
            or args.radio_adaptive_token_threshold < 1):
        raise SystemExit("RADIO detail-tile and encoder-batch counts must be positive")
    if (not 0 < args.radio_complexity_budget_ratio <= 1
            or args.radio_complexity_token_quantum < 1
            or 128 % args.radio_complexity_token_quantum
            or 256 % args.radio_complexity_token_quantum):
        raise SystemExit(
            "RADIO complexity ratio must be in (0,1] and its quantum must divide 128 and 256")
    if args.vision_backend == "radio_v4h" and args.radio_adaptive_complexity:
        raise SystemExit(
            "--radio-adaptive-complexity truncates RADIO1D's nested prefix; "
            "C-RADIOv4-H tokens are positional, so the budget is set by the "
            "pooled lattice. Drop the flag for --vision-backend radio_v4h.")
    if args.vision_backend == "radio_v4h":
        native_input_width = (
            2 * args.radio_v4h_feature_width
            if args.radio_v4h_native_packing == "pair_columns"
            else args.radio_v4h_feature_width)
        if args.radio_v4h_feature_width < 1:
            raise SystemExit("--radio-v4h-feature-width must be positive")
        if args.radio_v4h_native and native_input_width < 1:
            raise SystemExit("native C-RADIO input width is invalid")
    if is_radio_backend(args):
        incompatible = []
        if args.vision_fusion:
            incompatible.append("--vision-fusion")
        if args.vision_compressor_checkpoint:
            incompatible.append("--vision-compressor-checkpoint")
        if args.moonvit_tap_layers.strip() or args.layer_vision_layers.strip():
            incompatible.append("MoonViT layer taps")
        if args.vision_resampler_layers:
            incompatible.append("--vision-resampler-layers")
        if incompatible:
            raise SystemExit(
                "RADIO1D already supplies the unpooled teacher representation; "
                f"remove incompatible options: {incompatible}")
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
    if (args.structured_weight < 0
            or (args.structured_lr is not None and args.structured_lr < 0)
            or args.structured_coordinate_weight < 1
            or args.structured_invalid_box_weight < 0
            or args.structured_invalid_box_margin < 0
            or args.structured_width < 1
            or args.structured_object_queries < 1
            or args.structured_spatial_layers < 1
            or args.structured_object_layers < 1
            or args.structured_heads < 1
            or args.structured_width % args.structured_heads):
        raise SystemExit("structured-head geometry/weight is invalid")
    if args.structured_head and not is_radio_backend(args):
        raise SystemExit("--structured-head requires a RADIO vision backend")
    if args.sam_fusion_tokens < 1:
        raise SystemExit("--sam-fusion-tokens must be positive")
    if args.sam_fusion and args.vision_fusion:
        raise SystemExit("--sam-fusion and --vision-fusion are mutually exclusive")
    if args.sam_fusion and not is_radio_backend(args):
        raise SystemExit("--sam-fusion is defined only for variable-prefix RADIO1D")
    if args.sam_fusion:
        # --vision-fusion is already rejected under RADIO1D on the grounds that
        # the backbone supplies the teacher representation. That argument applies
        # at least as strongly here: C-RADIOv4-1D-H is distilled from SigLIP2-g,
        # DINOv3-7B and SAM3, with SAM3 contributing dense (use_summary=false)
        # structure specifically. Warn rather than refuse, so the redundancy can
        # still be measured against the ocr_/structured_ eval splits.
        print("warn: --sam-fusion adds SAM ViT-B to a backbone already distilled "
              "from SAM3; treat it as an ablation, not a default", flush=True)
    minimum_radio_tokens = (adaptive_tokens_per_tile(
        256, ratio=args.radio_complexity_budget_ratio,
        quantum=args.radio_complexity_token_quantum)
        if args.radio_adaptive_complexity else 256)
    if args.sam_fusion and args.sam_fusion_tokens > minimum_radio_tokens:
        raise SystemExit(
            "--sam-fusion-tokens exceeds the smallest possible RADIO prefix "
            f"({minimum_radio_tokens})")
    if args.siglip2_width < 1:
        raise SystemExit("--siglip2-width must be positive")
    if args.vision_compressor_checkpoint:
        if args.prefix_tokens != 128:
            raise SystemExit("the frozen compressor requires --prefix-tokens 128")
        if args.vision_fusion:
            raise SystemExit("compressor input and --vision-fusion residual are mutually exclusive")
        if args.layer_vision_layers.strip():
            raise SystemExit("compressor input cannot use raw MoonViT layer injection")
        if not Path(args.vision_compressor_checkpoint).is_file():
            raise SystemExit("frozen compressor checkpoint does not exist")
    try:
        moonvit_taps = tuple(sorted({int(value.strip()) for value in
                                    args.moonvit_tap_layers.split(",")
                                    if value.strip()}))
        layer_vision_sites = tuple(sorted({int(value.strip()) for value in
                                          args.layer_vision_layers.split(",")
                                          if value.strip()}))
    except ValueError as exc:
        raise SystemExit("invalid MoonViT/RWKV layer-match specification") from exc
    # Staged caches are also useful without raw layer injection: the ordinary
    # bridge consumes their deepest tap. Injection, however, still requires a
    # one-to-one set of source taps.
    if layer_vision_sites and not moonvit_taps:
        raise SystemExit("layer-vision sites require MoonViT taps")
    if layer_vision_sites and len(moonvit_taps) != len(layer_vision_sites):
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
    if args.activation_checkpoint_min_tokens < 0:
        raise SystemExit("--activation-checkpoint-min-tokens must be non-negative")
    if args.activation_checkpoint_max_layers < 0:
        raise SystemExit("--activation-checkpoint-max-layers must be non-negative")
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
    if (args.eval_every < 0 or args.eval_sample_every < 0
            or args.eval_batch_size < 0 or args.eval_batch_tokens < 0
            or args.checkpoint_every < 0):
        raise SystemExit("eval/checkpoint intervals must be non-negative")
    if args.log_every < 1:
        raise SystemExit("--log-every must be positive")
    if args.eval_every and args.eval_examples < 1:
        raise SystemExit("--eval-examples must be positive when evaluation is enabled")
    if (args.eval_samples < 0 or args.eval_sample_max_new < 0
            or args.eval_ocr_samples < 0 or args.eval_structured_samples < 0):
        raise SystemExit("qualitative eval sizes must be non-negative")
    if args.eval_ocr_samples + args.eval_structured_samples > args.eval_samples:
        raise SystemExit(
            "reserved OCR + structured gallery counts exceed --eval-samples")
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
    _atomic_json(out / "config.json",
                 {"schema": CHECKPOINT_SCHEMA, **persisted_args(args)})
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
                 for row in load_examples(
                     source, stat_workers=args.manifest_stat_workers,
                     require_all=True)]
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
                                                  sandwich_prompt=args.sandwich_prompt,
                                                  sandwich_lead_prompt=args.sandwich_lead_prompt)
    unique_eval_rows: list[dict] = []
    if args.eval_data:
        raw_eval = [row for source in args.eval_data
                    for row in load_examples(
                        source, stat_workers=args.manifest_stat_workers,
                        require_all=True)]
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
                                                    sandwich_prompt=args.sandwich_prompt,
                                                    sandwich_lead_prompt=args.sandwich_lead_prompt)
        if not eval_rows:
            raise SystemExit("No usable held-out image/text rows found")
        rows = train_rows + eval_rows
        lengths = train_lengths + eval_lengths
        train_indices = list(range(len(train_rows)))
        val_indices = list(range(len(train_rows), len(rows)))
    else:
        rows, lengths = train_rows, train_lengths
        train_indices, val_indices = split_examples(rows, val_fraction=args.val_fraction)
    if is_radio_backend(args):
        native_plan_path = out / "radio_native_plan.json"
        native_plan_signature = None
        native_plan_entries = None
        native_plan_cacheable = bool(
            args.vision_backend == "radio_v4h"
            and args.radio_v4h_native and args.radio_v4h_cache_only)
        if native_plan_cacheable:
            signature = hashlib.sha256()
            signature.update(json.dumps({
                "schema": 1,
                "cache": str(Path(args.feature_cache).resolve()),
                "cache_mtime_ns": Path(args.feature_cache).stat().st_mtime_ns,
                "revision": args.radio_v4h_revision,
                "packing": args.radio_v4h_native_packing,
                "feature_width": args.radio_v4h_feature_width,
                "max_edge": args.radio_v4h_max_edge,
            }, sort_keys=True).encode())
            for row in rows:
                signature.update(json.dumps((
                    str(row["image"]), row.get("image_sha256"),
                    row.get("width"), row.get("height"),
                    row.get("exif_orientation"), row.get("task"),
                ), separators=(",", ":"), default=str).encode())
            native_plan_signature = signature.hexdigest()
            try:
                cached_plan = json.loads(native_plan_path.read_text())
                if (cached_plan.get("schema") == 1
                        and cached_plan.get("signature")
                        == native_plan_signature
                        and len(cached_plan.get("entries", ())) == len(rows)):
                    native_plan_entries = cached_plan["entries"]
                    print({
                        "kind": "radio_tile_plan_cache_hit",
                        "rows": len(rows), "path": str(native_plan_path),
                    }, flush=True)
            except (OSError, json.JSONDecodeError, AttributeError, TypeError):
                pass
        generated_native_plan = []
        print("planning aspect-aware RADIO tile buckets", flush=True)
        for index, row in enumerate(rows):
            if native_plan_entries is not None:
                entry = native_plan_entries[index]
                if not isinstance(entry, list) or len(entry) != 6:
                    raise SystemExit(
                        f"invalid cached native plan entry {index}")
                tiles, width, height, visual_tokens, mask_h, mask_w = map(
                    int, entry)
                row["_radio_tiles"] = tiles
                row["_image_width"], row["_image_height"] = width, height
                row["_visual_tokens"] = visual_tokens
                if mask_h and mask_w:
                    row["_structured_mask_shape"] = (mask_h, mask_w)
                continue
            width, height = row.get("width"), row.get("height")
            if not width or not height:
                with Image.open(row["image"]) as image:
                    width, height = image.size
            grid_rows, grid_columns = choose_detail_grid(
                int(width), int(height),
                max_detail_tiles=args.radio_max_detail_tiles)
            details = grid_rows * grid_columns
            tiles = 1 if details == 1 else details + 1
            row["_radio_tiles"] = tiles
            # Retained so the bridge can reconstruct each tile's letterbox
            # transform without reopening the image every batch.
            row["_image_width"], row["_image_height"] = int(width), int(height)
            if args.vision_backend == "radio_v4h" and args.radio_v4h_native:
                from rwkv_lab.radio_v4h import (
                    cache_path as v4h_cache_path,
                    native_cache_token_count,
                    native_grid_for,
                )
                pairing = args.radio_v4h_native_packing == "pair_columns"
                if args.radio_v4h_cache_only:
                    source = Path(row["image"])
                    source = source if source.is_absolute() else ROOT / source
                    raw_cells = native_cache_token_count(
                        v4h_cache_path(Path(args.feature_cache), source),
                        hidden_size=args.radio_v4h_feature_width)
                    # The cheap header read yields cells, not a grid, so an odd
                    # grid width can only be caught by ``native_token_boxes``
                    # when the row loads. Whenever the width is even -- the only
                    # case that loads at all -- halving the cells is exact.
                    visual_tokens = raw_cells // 2 if pairing else raw_cells
                else:
                    gh, gw = native_grid_for(
                        row, root=ROOT, max_edge=args.radio_v4h_max_edge)
                    try:
                        visual_tokens = native_visual_token_count(
                            gh, gw, packing=args.radio_v4h_native_packing)
                    except ValueError as error:
                        raise SystemExit(
                            f"{row['image']}: {error}") from error
                if str(row.get("task") or "") == "sam_mask":
                    # Native mask supervision needs the 2-D feature geometry,
                    # not merely its flattened token count. New manifests carry
                    # EXIF orientation, so this is metadata-only; old manifests
                    # pay a one-time image-header fallback.
                    gh, gw = native_grid_for(
                        row, root=ROOT, max_edge=args.radio_v4h_max_edge)
                    expected_tokens = native_visual_token_count(
                        gh, gw, packing=args.radio_v4h_native_packing)
                    if expected_tokens != visual_tokens:
                        raise SystemExit(
                            f"{row['image']}: manifest/native grid predicts "
                            f"{expected_tokens} visual tokens but the cache "
                            f"contains {visual_tokens}")
                    row["_structured_mask_shape"] = (
                        (gh, gw) if not pairing else (gh, gw // 2))
                # The projection-free bridge stacks flattened tokens and their
                # per-row boxes, so equal cell counts are the exact batching
                # requirement.  Cache geometry is authoritative here: EXIF
                # rotation can make manifest width/height differ from the grid
                # that the encoder actually wrote.
                row["_radio_tiles"] = visual_tokens
                row["_visual_tokens"] = visual_tokens
                generated_native_plan.append([
                    visual_tokens, int(width), int(height), visual_tokens,
                    *(row.get("_structured_mask_shape") or (0, 0)),
                ])
                if (index + 1) % 50_000 == 0:
                    print({"kind": "radio_tile_plan", "done": index + 1,
                           "total": len(rows)}, flush=True)
                continue
            if args.vision_backend == "radio_v4h":
                routed_tokens = args.radio_v4h_lattice ** 2
            else:
                maximum_tokens = tokens_per_tile_for_tile_count(
                    tiles, threshold=args.radio_adaptive_token_threshold)
                routed_tokens = (adaptive_tokens_per_tile(
                    maximum_tokens, ratio=args.radio_complexity_budget_ratio,
                    quantum=args.radio_complexity_token_quantum)
                    if args.radio_adaptive_complexity else maximum_tokens)
            row["_visual_tokens"] = tiles * routed_tokens
            if (index + 1) % 50_000 == 0:
                print({"kind": "radio_tile_plan", "done": index + 1,
                       "total": len(rows)}, flush=True)
        if (native_plan_cacheable and native_plan_entries is None
                and len(generated_native_plan) == len(rows)):
            _atomic_json(native_plan_path, {
                "schema": 1, "signature": native_plan_signature,
                "entries": generated_native_plan,
            })
    qualitative_val_indices = filter_eval_sample_indices(
        rows, val_indices, args.eval_sample_exclude_sources.split(","))
    if args.eval_samples and not qualitative_val_indices:
        raise SystemExit(
            "qualitative eval source filter excluded every held-out image")
    args.data_fingerprint = dataset_fingerprint(
        rows, train_indices, val_indices, explicit_eval=bool(args.eval_data))
    args.image_metadata_fingerprint = image_metadata_fingerprint(rows)
    # The same digest restricted to the rows a structured-data removal keeps.
    # Removing sam_mask rows changes the all-rows digest by construction; this
    # one still has to be identical across that migration, so every surviving
    # caption/OCR image stays as protected as it was before.
    args.unstructured_image_metadata_fingerprint = image_metadata_fingerprint(
        [row for row in rows
         if str(row.get("task") or "").casefold() != "sam_mask"])
    args.structured_train_examples = sum(
        str(rows[index].get("task") or "").casefold() == "sam_mask"
        for index in train_indices)
    args.structured_eval_examples = sum(
        str(rows[index].get("task") or "").casefold() == "sam_mask"
        for index in val_indices)
    if args.allow_coco_prompt_migration:
        legacy_prompt = (
            "Segment the annotated objects. For each instance, give its category, "
            "normalized box [x1,y1,x2,y2], and 16x16 mask row spans:\n")

        def legacy_coco_rows(source_rows: Sequence[dict]) -> list[dict]:
            migrated = []
            for row in source_rows:
                item = dict(row)
                if item.get("caption_variant") == "coco_instance_polygon_mask16":
                    item["prompt"] = legacy_prompt
                migrated.append(item)
            return migrated

        previous_train, _ = prepare_examples(
            legacy_coco_rows(unique_train_rows), vocab, prompt=args.prompt,
            max_text_tokens=args.max_text_tokens,
            sandwich_prompt=args.sandwich_prompt,
            sandwich_lead_prompt=args.sandwich_lead_prompt)
        if args.eval_data:
            previous_eval, _ = prepare_examples(
                legacy_coco_rows(unique_eval_rows), vocab, prompt=args.prompt,
                max_text_tokens=args.max_text_tokens,
                sandwich_prompt=args.sandwich_prompt,
                sandwich_lead_prompt=args.sandwich_lead_prompt)
            previous_rows = previous_train + previous_eval
        else:
            previous_rows = previous_train
        if len(previous_rows) != len(rows):
            raise SystemExit("COCO prompt migration changed the usable row set")
        args.previous_coco_prompt_data_fingerprint = dataset_fingerprint(
            previous_rows, train_indices, val_indices,
            explicit_eval=bool(args.eval_data))
    if args.allow_text_limit_increase_from:
        previous_train, _ = prepare_examples(
            unique_train_rows, vocab, prompt=args.prompt,
            max_text_tokens=args.allow_text_limit_increase_from,
            sandwich_prompt=args.sandwich_prompt,
            sandwich_lead_prompt=args.sandwich_lead_prompt)
        if args.eval_data:
            previous_eval, _ = prepare_examples(
                unique_eval_rows, vocab, prompt=args.prompt,
                max_text_tokens=args.allow_text_limit_increase_from,
                sandwich_prompt=args.sandwich_prompt,
                sandwich_lead_prompt=args.sandwich_lead_prompt)
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
    args.moonvit_fingerprint = (checkpoint_fingerprint(args.moonvit)
                                if args.vision_backend == "moonvit" else "")
    args.radio_fingerprint = (checkpoint_fingerprint(args.radio_model)
                              if args.vision_backend == "radio1d" else "")
    if args.vision_backend == "radio_v4h":
        args.radio_v4h_fingerprint = v4h_run_artifact_fingerprint(args)
        v4h_cache_contract = {
            "schema": "radio-v4h-runtime-cache-contract-v1",
            "cache": str(Path(args.feature_cache).resolve()),
            "artifact": args.radio_v4h_fingerprint,
            "revision": args.radio_v4h_revision,
            "native": bool(args.radio_v4h_native),
            "packing": args.radio_v4h_native_packing,
            "feature_width": int(args.radio_v4h_feature_width),
            "max_edge": int(args.radio_v4h_max_edge),
            "lattice": int(args.radio_v4h_lattice),
            "pair_axis": args.radio_v4h_pair_axis,
        }
        args.radio_v4h_cache_fingerprint = hashlib.sha256(
            json.dumps(
                v4h_cache_contract, sort_keys=True,
                separators=(",", ":")).encode()).hexdigest()
    else:
        args.radio_v4h_fingerprint = ""
        args.radio_v4h_cache_fingerprint = ""
    if args.sam_fusion:
        args.vision_fusion_fingerprint = SamAlignedFrozenFeatures(
            args.sam_model, tokens=args.sam_fusion_tokens).cache_fingerprint
    elif args.vision_fusion or args.vision_compressor_checkpoint:
        args.vision_fusion_fingerprint = _vision_tower_config(args).fingerprint()
    else:
        args.vision_fusion_fingerprint = ""
    args.vision_compressor_fingerprint = (
        checkpoint_fingerprint(args.vision_compressor_checkpoint)
        if args.vision_compressor_checkpoint else "")
    try:
        sampler_train_indices = multitask_balanced_indices(
            rows, train_indices,
            target_ratios={
                "ocr": args.ocr_update_ratio,
                "sam_mask": args.structured_update_ratio,
            },
            seed=args.seed)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    natural_ocr = sum(eval_task_name(rows[index]) == "ocr"
                      for index in train_indices)
    sampled_ocr = sum(eval_task_name(rows[index]) == "ocr"
                      for index in sampler_train_indices)
    natural_structured = sum(eval_task_name(rows[index]) == "structured"
                             for index in train_indices)
    sampled_structured = sum(eval_task_name(rows[index]) == "structured"
                             for index in sampler_train_indices)
    args.ocr_manifest_rows = natural_ocr
    args.ocr_sampler_slots = sampled_ocr
    args.ocr_sampler_ratio = sampled_ocr / max(1, len(sampler_train_indices))
    args.structured_manifest_rows = natural_structured
    args.structured_sampler_slots = sampled_structured
    args.structured_sampler_ratio = (
        sampled_structured / max(1, len(sampler_train_indices)))
    _atomic_json(out / "config.json",
                 {"schema": CHECKPOINT_SCHEMA, **persisted_args(args)})
    sampler = EpochBatchSampler(
        sampler_train_indices, lengths, batch_size=args.batch, seed=args.seed,
        group_keys=([int(row["_radio_tiles"]) for row in rows]
                    if is_radio_backend(args) else None))
    token_costs = [
        (int(row["_visual_tokens"]) if is_radio_backend(args)
         else args.prefix_tokens) + length
        for row, length in zip(rows, lengths)
    ]
    train_truncated = sum(rows[index]["truncated"] for index in train_indices)
    val_truncated = sum(rows[index]["truncated"] for index in val_indices)
    print(f"data: {len(train_indices)} train / {len(val_indices)} val; "
          f"{train_truncated} train + {val_truncated} val captions truncated", flush=True)
    if len(sampler_train_indices) != len(train_indices):
        print({"kind": "task_balanced_sampler",
               "manifest_rows": {
                   "ocr": natural_ocr, "structured": natural_structured},
               "sampler_slots": {
                   "ocr": sampled_ocr, "structured": sampled_structured},
               "total_sampler_slots": len(sampler_train_indices),
               "ratios": {
                   "ocr": args.ocr_sampler_ratio,
                   "structured": args.structured_sampler_ratio}}, flush=True)

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

    moonvit_taps = tuple(int(value) for value in args.moonvit_tap_layers.split(",")
                         if value)
    vision_compressor = None
    if args.vision_backend == "radio_v4h":
        from rwkv_lab.radio_v4h import load_radio_v4h
        from rwkv_lab.radio1d_rwkv import RadioRWKVBridge
        _atomic_json(out / "status.json", {
            "state": "loading_radio_v4h", "updated": time.time()})
        if args.radio_v4h_cache_only:
            print("using read-only C-RADIOv4-H feature cache", flush=True)
            vision = nn.Identity()
        else:
            print(f"loading frozen C-RADIOv4-H: {args.radio_v4h_model}", flush=True)
            vision = load_radio_v4h(args.radio_v4h_model)
        vision.radio_revision = args.radio_v4h_revision
        vision.radio_max_detail_tiles = args.radio_max_detail_tiles
        vision.radio_tile_batch = args.radio_tile_batch
        vision.v4h_cache_dir = Path(args.feature_cache)
        vision.v4h_lattice = args.radio_v4h_lattice
        vision.v4h_pair_axis = args.radio_v4h_pair_axis
        vision.v4h_feature_width = args.radio_v4h_feature_width
        vision.v4h_native_packing = args.radio_v4h_native_packing
        # hidden_size is the LM width (the bridge output goes straight into the
        # token stream); input_size is the cached encoder width after optional
        # column packing. Native mode requires those widths to match, keeping
        # the path projection-free for both the 2560-wide generic cache and the
        # 4096-wide all-adaptor cache.
        input_width = (
            2 * args.radio_v4h_feature_width
            if args.radio_v4h_native_packing == "pair_columns"
            else args.radio_v4h_feature_width)
        if args.radio_v4h_native and input_width != int(rwkv.config.hidden_size):
            raise SystemExit(
                f"projection-free native C-RADIO width {input_width} does not "
                f"match RWKV width {rwkv.config.hidden_size}")
        projector = RadioFeatureProjector(
            bridge=RadioRWKVBridge(
                hidden_size=int(rwkv.config.hidden_size),
                input_size=input_width,
                rank=args.radio_v4h_bridge_rank,
                tokens_per_tile=(1 if args.radio_v4h_native
                                 else args.radio_v4h_lattice ** 2),
                max_tiles=(1 if args.radio_v4h_native
                           else args.radio_max_detail_tiles + 1),
                letterbox_geometry=not args.radio_v4h_native),
        ).cuda().float()
        projector.native_mode = uses_native_prefix(args)
        vision.v4h_native = bool(args.radio_v4h_native)
        vision.v4h_max_edge = args.radio_v4h_max_edge
        print(f"v4h bridge: rank={args.radio_v4h_bridge_rank} "
              f"input={input_width} packing={args.radio_v4h_native_packing} "
              f"lattice={args.radio_v4h_lattice}x{args.radio_v4h_lattice} "
              f"params={sum(p.numel() for p in projector.parameters())/1e6:.2f}M",
              flush=True)
    elif args.vision_backend == "radio1d":
        _atomic_json(out / "status.json", {
            "state": "loading_radio1d", "updated": time.time()})
        print(f"loading frozen RADIO1D-H: {args.radio_model}", flush=True)
        vision = load_radio1d_h(args.radio_model)
        vision.radio_revision = args.radio_revision
        vision.radio_max_detail_tiles = args.radio_max_detail_tiles
        vision.radio_tile_batch = args.radio_tile_batch
        vision.radio_adaptive_token_threshold = args.radio_adaptive_token_threshold
        projector = RadioFeatureProjector(
            adaptive_complexity=args.radio_adaptive_complexity,
            complexity_budget_ratio=args.radio_complexity_budget_ratio,
            complexity_token_quantum=args.radio_complexity_token_quantum,
        ).cuda().float()
    else:
        _atomic_json(out / "status.json", {
            "state": "loading_moonvit", "updated": time.time()})
        print(f"loading frozen MoonViT: {args.moonvit}", flush=True)
        vision = MoonViT.from_checkpoint(
            args.moonvit, device="cuda",
            max_input_patches=args.max_input_patches,
            tap_layers=moonvit_taps, view_mode=args.vision_view_mode)
        vision.requires_grad_(False)
        vision.eval()
    # Trainable modules intentionally remain fp32; autocast supplies bf16 matmuls
    # while AdamW retains real fp32 master parameters and moments.
    if args.vision_compressor_checkpoint:
        vision_compressor = FrozenTeacherCompressor.from_checkpoint(
            args.vision_compressor_checkpoint, device="cuda",
            dtype=torch.bfloat16)
        projector = CanonicalLatentPrefixProjector(
            int(rwkv.config.hidden_size), args.prefix_tokens).cuda().float()
        print(f"frozen teacher compressor: {args.vision_compressor_checkpoint} "
              f"params={sum(p.numel() for p in vision_compressor.parameters())/1e6:.1f}M",
              flush=True)
    elif args.vision_backend == "moonvit":
        projector = MoonViTPrefixProjector(
            rwkv.config.hidden_size, args.prefix_tokens,
            resampler_layers=args.vision_resampler_layers,
            resampler_width=args.vision_resampler_width,
            resampler_heads=args.vision_resampler_heads).cuda().float()
    fusion_tower = None
    vision_fusion = None
    fusion_cache_dir = None
    if args.vision_fusion or args.sam_fusion or vision_compressor is not None:
        fusion_tower = (SamAlignedFrozenFeatures(
            args.sam_model, tokens=args.sam_fusion_tokens)
            if args.sam_fusion
            else AlignedFrozenVisionFeatures(_vision_tower_config(args)))
        fusion_tower.requires_grad_(False).eval()
        if args.vision_fusion or args.sam_fusion:
            vision_fusion = VisionFusionResidual(
                int(rwkv.config.hidden_size), rank=args.vision_fusion_rank,
                source_width=fusion_tower.width).cuda().float()
        fusion_cache_dir = Path(args.fusion_feature_cache)
        mode = (f"SAM global residual rank={args.vision_fusion_rank}"
                if args.sam_fusion else
                f"residual rank={args.vision_fusion_rank}" if args.vision_fusion
                else "frozen-compressor input")
        towers = "SAM" if args.sam_fusion else "SigLIP2+DINOv2+SAM"
        print(f"frozen fusion features: {towers} {mode} "
              f"cache={fusion_cache_dir} fingerprint={fusion_tower.cache_fingerprint[:12]}",
              flush=True)
    deep_vision = None
    if args.deep_vision_layers.strip():
        deep_sites = _parse_engram_sites(
            args.deep_vision_layers, int(rwkv.config.num_hidden_layers))
        radio_token_quantum = (
            None if uses_native_prefix(args)
            else args.radio_complexity_token_quantum
            if args.radio_adaptive_complexity else 128)
        deep_vision = (RadioPrefixInjector(
            int(rwkv.config.hidden_size), deep_sites,
            rank=args.deep_vision_rank,
            token_quantum=radio_token_quantum,
            grouped_precompute=args.deep_vision_grouped_precompute).cuda().float()
            if is_radio_backend(args) else DeepVisionInjector(
                int(rwkv.config.hidden_size), deep_sites,
                rank=args.deep_vision_rank,
                grouped_precompute=args.deep_vision_grouped_precompute).cuda().float())
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
    structured_head = (StructuredSpatialHead(
        input_width=int(rwkv.config.hidden_size), width=args.structured_width,
        object_queries=args.structured_object_queries,
        spatial_layers=args.structured_spatial_layers,
        object_layers=args.structured_object_layers,
        heads=args.structured_heads).cuda().float()
        if args.structured_head else None)
    nextlat = (NextLatPredictor(rwkv.config.hidden_size, hidden=args.nextlat_hidden).cuda().float()
               if args.nextlat_weight else None)
    optimizer, trainable = _optimizer(
        projector, nextlat, engram, wrappers, args,
        deep_vision=deep_vision, layer_vision=layer_vision,
        vision_fusion=vision_fusion,
        grounding=grounding, structured_head=structured_head)
    assert_training_contract(rwkv, vision, wrappers, trainable,
                             vision_compressor=vision_compressor)
    last_checkpoint_step: int | None = None
    checkpoint_writer = _AsyncCheckpointWriter(checkpoint_path)
    log = None

    def finish_async_last_checkpoint(*, wait_for_write: bool) -> int | None:
        """Publish completion metadata only after the durable rename succeeds."""
        nonlocal last_checkpoint_step
        completed_step = checkpoint_writer.finish(
            wait_for_write=wait_for_write)
        if completed_step is None:
            return None
        last_checkpoint_step = completed_step
        if log is not None and not log.closed:
            log.write(json.dumps({
                "kind": "checkpoint", "step": completed_step,
                "reason": "periodic_async", "path": str(checkpoint_path),
            }) + "\n")
            _sync_log(log)
        return completed_step

    def checkpoint_snapshot(checkpoint_step: int) -> dict:
        return _checkpoint_snapshot(
            step=checkpoint_step, projector=projector,
            nextlat=nextlat, engram=engram, deep_vision=deep_vision,
            layer_vision=layer_vision, grounding=grounding,
            structured_head=structured_head, wrappers=wrappers,
            optimizer=optimizer, sampler=sampler, args=args,
            vision_fusion=vision_fusion)

    def save_last_checkpoint(checkpoint_step: int) -> None:
        """Save ``last.pt`` and remember which committed state it contains."""
        nonlocal last_checkpoint_step
        # Never let two writers share ``last.tmp``. A synchronous boundary
        # first makes any earlier periodic publication durable.
        finish_async_last_checkpoint(wait_for_write=True)
        _write_checkpoint_snapshot(
            checkpoint_path, checkpoint_snapshot(checkpoint_step))
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
    resume_publication_changed = False
    unrelated_resume_branch = False
    if resume_path is not None:
        print(f"resuming exact training state from {resume_path}", flush=True)
        (step, text_limit_migrated, loop_reset_already_applied,
         resume_contract_changed) = _load_checkpoint(
             resume_path, projector=projector, nextlat=nextlat, engram=engram,
             wrappers=wrappers, optimizer=optimizer, sampler=sampler, args=args,
             deep_vision=deep_vision, layer_vision=layer_vision,
             vision_fusion=vision_fusion, grounding=grounding,
             structured_head=structured_head)
        resume_publication_changed = bool(
            resume_contract_changed
            or getattr(args, "batch_budget_migrated", False)
            or getattr(args, "loop_lr_migrated_from", None) is not None
            or getattr(args, "structured_lr_migrated_from", None) is not None
            or getattr(args, "radio_v4h_fingerprint_bootstrapped", False))
        _preserve_loop_reset_outcome(args, loop_reset_already_applied)
        last_checkpoint_step = _resumed_last_checkpoint_step(
            resume_path, checkpoint_path, step,
            contract_changed=resume_publication_changed)
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
        if resume_contract_changed and not text_limit_migrated:
            migrated_best = _quarantine_best(
                best_dir, f"before-resume-contract-step-{step}")
            if migrated_best is not None:
                print(
                    f"preserved pre-migration best checkpoint at {migrated_best}; "
                    "the adaptive visual contract starts with no claimed winner",
                    flush=True,
                )
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
        if getattr(args, "loop_count_migrated_from", 0):
            print(
                f"increased factored loop count from "
                f"{args.loop_count_migrated_from} to {args.loop_count}; "
                "retained optimizer and sampler state and zero-initialized "
                "the new refinement pass",
                flush=True,
            )
        if getattr(args, "loop_lr_migrated_from", None) is not None:
            print(
                f"changed factored-loop learning rate from "
                f"{args.loop_lr_migrated_from:g} to {args.loop_lr:g}; "
                "retained loop tensors and Adam moments",
                flush=True,
            )
        if getattr(args, "structured_lr_migrated_from", None) is not None:
            print(
                f"changed structured-head learning rate from "
                f"{args.structured_lr_migrated_from:g} to "
                f"{args.structured_lr:g}; retained head tensors and Adam "
                "moments",
                flush=True,
            )
        if getattr(args, "structured_data_removed_from_fingerprint", None):
            print(
                "removed every structured bbox/mask train and eval row; "
                "retained model and optimizer state, reset sampler state, and "
                "disabled structured sampling/loss",
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
                loop_reset_pending=loop_reset_pending,
                contract_changed=resume_contract_changed)
        if invalidates_eval_contract:
            reset_reasons = []
            if unrelated_resume_branch:
                reset_reasons.append("explicit_resume_branch")
            if text_limit_migrated:
                reset_reasons.append("text_limit_migration")
            if loop_reset_pending:
                reset_reasons.append("loop_reset")
            if resume_contract_changed:
                reset_reasons.append("resume_contract_migration")
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
    elif args.init_adapters_from or args.init_text_adapters_from:
        init_path = Path(args.init_adapters_from or args.init_text_adapters_from)
        if not init_path.is_file():
            raise SystemExit(f"adapter initialization checkpoint does not exist: {init_path}")
        source_step = _initialize_adapters(
            init_path, projector=projector, nextlat=nextlat, engram=engram,
            wrappers=wrappers, args=args, deep_vision=deep_vision,
            layer_vision=layer_vision, vision_fusion=vision_fusion,
            grounding=grounding, structured_head=structured_head,
            replace_vision_bridge=bool(args.init_text_adapters_from))
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
    if (not resume_contract_changed and not text_limit_migrated
            and best_info_path.is_file()
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
               "initialized_from": (
                   args.init_adapters_from or args.init_text_adapters_from),
               "vision_backend": args.vision_backend,
               "loop_reset_performed": loop_reset_performed,
               "text_limit_migrated": text_limit_migrated,
               "max_text_tokens": args.max_text_tokens,
               "activation_checkpoint_min_tokens": args.activation_checkpoint_min_tokens,
               "activation_checkpoint_max_layers": args.activation_checkpoint_max_layers,
               "prefetch_cuda_transfer": args.prefetch_cuda_transfer,
               "train_examples": len(train_indices), "val_examples": len(val_indices),
               "source_counts": dict(sorted(source_counts.items())),
               "eval_source_counts": dict(sorted(eval_source_counts.items())),
               "trainable_parameters": sum(p.numel() for p in trainable),
               "radio_bridge_parameters": (
                   sum(p.numel() for p in projector.parameters())
                   if isinstance(projector, RadioFeatureProjector) else 0),
               "radio_adaptive_complexity": (
                   projector.adaptive_complexity
                   if isinstance(projector, RadioFeatureProjector) else False),
               "radio_complexity_budget_ratio": (
                   projector.complexity_budget_ratio
                   if isinstance(projector, RadioFeatureProjector) else None),
               "vision_resampler_parameters": (
                   sum(p.numel() for p in projector.resampler.parameters())
                   if getattr(projector, "resampler", None) is not None else 0),
               "deep_vision_parameters": (
                   sum(p.numel() for p in deep_vision.parameters())
                   if deep_vision is not None else 0),
               "grounding_parameters": (
                   sum(p.numel() for p in grounding.parameters())
                   if grounding is not None else 0),
               "structured_head_parameters": (
                   sum(p.numel() for p in structured_head.parameters())
                   if structured_head is not None else 0),
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

    # ``--profile-steps`` is a count of live steps to inspect, including after
    # a resume.  Treating it as an absolute step number made restart profiling
    # silently disappear on long-running jobs, exactly when a new hot-path
    # optimization needs before/after timing evidence.
    profile_until_step = step + args.profile_steps
    operator_profile_until_step = step + args.operator_profile_steps
    telemetry_future = None
    prefetch_transfer_stream = (
        torch.cuda.Stream(device=torch.cuda.current_device())
        if args.prefetch_next_batch and args.prefetch_cuda_transfer else None)

    def queue_loop_telemetry(telemetry_step: int) -> None:
        """Snapshot tiny gate tensors now; write JSON off the training thread."""
        nonlocal telemetry_future
        if telemetry_future is not None:
            if not telemetry_future.done():
                return
            telemetry_future.result()
        payload = loop_telemetry_payload(wrappers, step=telemetry_step)
        telemetry_future = _TELEMETRY_WRITE_POOL.submit(
            write_loop_telemetry_payload, out / "loop_rw.json", payload)

    def flush_loop_telemetry() -> None:
        nonlocal telemetry_future
        if telemetry_future is not None:
            telemetry_future.result()
            telemetry_future = None

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
        if prefetch_transfer_stream is not None:
            future = _CudaBatchPrefetch(
                future, device=torch.device(
                    "cuda", torch.cuda.current_device()),
                stream=prefetch_transfer_stream)
        return future_indices, future

    def run_evaluation(eval_step: int, prior_eval: dict | None = None, *,
                       checkpoint_saved: bool = False,
                       recurrent_context_fresh: bool = False) -> bool:
        """Run one scalar+qualitative eval; return whether it saved ``last.pt``."""
        nonlocal best_eval_loss

        cadence_due = qualitative_eval_due(
            eval_step, eval_every=args.eval_every,
            sample_every=args.eval_sample_every)
        # Resume an interrupted gallery only when that step is still part of
        # the current qualitative cadence.  This lets an operator safely widen
        # the cadence after a costly gallery was interrupted without forcing
        # the obsolete work to finish on restart.
        caption_due = cadence_due and (
            prior_eval is None or prior_eval.get("sample_artifact") is not None
        )

        def eval_progress(phase: str):
            def update(done: int, total: int) -> None:
                _atomic_json(out / "status.json", {
                    "state": "evaluating", "phase": phase, "step": eval_step,
                    "progress": done, "total": total, "updated": time.time(),
                })
            return update

        if prior_eval is None:
            eval_progress("loss")(0, min(len(val_indices), args.eval_examples))
            eval_metrics = evaluate(
                rows, val_indices, rwkv=rwkv, projector=projector,
                vision=vision, engram=engram, cache_dir=cache_dir,
                batch_size=(args.eval_batch_size or args.batch),
                max_examples=args.eval_examples,
                token_budget=args.eval_batch_tokens,
                deep_vision=deep_vision, layer_vision=layer_vision,
                fusion_tower=fusion_tower, fusion_adapter=vision_fusion,
                fusion_cache_dir=fusion_cache_dir,
                vision_compressor=vision_compressor,
                structured_head=structured_head,
                structured_weight=args.structured_weight,
                structured_coordinate_weight=args.structured_coordinate_weight,
                structured_invalid_box_weight=args.structured_invalid_box_weight,
                structured_invalid_box_margin=args.structured_invalid_box_margin,
                progress=eval_progress("loss"))
            val_loss = _require_finite_metric(
                "validation loss", eval_metrics["loss"])
            ppl = _require_finite_metric(
                "validation ppl", eval_metrics["ppl"])
            eval_details = {
                name: value for name, value in eval_metrics.items()
                if name not in {"loss", "ppl"}
            }
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
            expected_artifact = (
                out / "eval_samples" / f"step_{eval_step:08d}.json"
                if (caption_due and args.eval_samples > 0
                    and args.eval_sample_max_new > 0
                    and qualitative_val_indices)
                else None
            )
            # The scalar evaluation is complete now. Log it before greedy
            # decoding so the eval marker and best header survive a hard stop
            # during the much longer qualitative phase.
            log.write(json.dumps({
                "kind": "eval", "step": eval_step, "loss": val_loss,
                "val_loss": val_loss, "ppl": ppl,
                **eval_details,
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
            eval_details = {
                name: value for name, value in prior_eval.items()
                if (name.startswith(("caption_", "ocr_", "structured_", "eval_"))
                    and isinstance(value, (int, float)))
            }
        if not caption_due:
            evaluation = {
                "kind": "eval_artifact", "step": eval_step,
                "loss": val_loss, "val_loss": val_loss, "ppl": ppl,
                **eval_details,
                "sample_artifact": None, "best": improved,
                "qualitative_complete": True,
            }
            log.write(json.dumps(evaluation) + "\n")
            print(evaluation, flush=True)
            _atomic_json(out / "status.json", {
                "state": "training", "step": eval_step,
                "updated": time.time(),
            })
            return checkpoint_saved
        if args.restart_before_eval and not recurrent_context_fresh:
            # Teacher-forced scalar eval uses the ordinary training path and
            # does not need a process reload. Recreate CUDA only for FLA's
            # long-lived recurrent generation path after the scalar record is
            # durable, so recovery resumes directly at captions.
            if prefetch_future is not None:
                prefetch_future.cancel()
            _sync_log(log)
            _atomic_json(out / "status.json", {
                "state": "restarting_for_qualitative", "step": eval_step,
                "updated": time.time(),
            }, durable=True)
            print({"kind": "qualitative_process_restart", "step": eval_step,
                   "exit_code": EVAL_RESTART_EXIT_CODE}, flush=True)
            os._exit(EVAL_RESTART_EXIT_CODE)
        caption_work = min(args.eval_samples, len(qualitative_val_indices)) \
            * args.eval_sample_max_new
        eval_progress("captions")(0, caption_work)
        sample_artifact = write_eval_samples(
            rows, qualitative_val_indices, step=eval_step, ppl=ppl, rwkv=rwkv,
            projector=projector, vision=vision, engram=engram,
            cache_dir=cache_dir, vocab=vocab, prompt=args.prompt, out=out,
            count=args.eval_samples, max_new=args.eval_sample_max_new,
            ocr_count=args.eval_ocr_samples,
            structured_count=args.eval_structured_samples,
            structured_head=structured_head,
            deep_vision=deep_vision, layer_vision=layer_vision,
            sandwich_prompt=args.sandwich_prompt,
            sandwich_lead_prompt=args.sandwich_lead_prompt,
            wrappers=wrappers,
            fusion_tower=fusion_tower, fusion_adapter=vision_fusion,
            fusion_cache_dir=fusion_cache_dir,
            vision_compressor=vision_compressor,
            progress=eval_progress("captions"))
        evaluation = {
            "kind": "eval_artifact", "step": eval_step, "loss": val_loss,
            "val_loss": val_loss, "ppl": ppl,
            **eval_details,
            "sample_artifact": (str(sample_artifact) if sample_artifact else None),
            "best": improved, "qualitative_complete": True,
        }
        log.write(json.dumps(evaluation) + "\n")
        print(evaluation, flush=True)
        _atomic_json(out / "status.json", {
            "state": "training", "step": eval_step, "updated": time.time(),
        })
        return checkpoint_saved

    resume_eval_work = _pending_eval_work(
        log_path, step,
        eval_expected=bool(step > 0 and args.eval_every
                           and step % args.eval_every == 0))
    if resume_eval_work is None:
        prefetched_indices, prefetch_future = schedule_next_batch_prefetch(step + 1)
    else:
        prefetched_indices, prefetch_future = None, None

    interrupted = False
    # False only while optimizer.step may have partially mutated parameters but
    # the sampler/public step have not committed. Outside that narrow window,
    # an unexpected failure can preserve an exact recovery point.
    checkpoint_state_valid = True
    try:
        while step < args.steps or resume_eval_work is not None:
            # The disk/fsync half of a periodic checkpoint overlaps subsequent
            # GPU work. Surface errors and advertise the checkpoint from the
            # training thread only after its atomic publication has completed.
            finish_async_last_checkpoint(wait_for_write=False)
            if resume_eval_work is not None:
                phase, prior_eval = resume_eval_work
                print({"kind": "eval_resume", "step": step, "phase": phase},
                      flush=True)
                run_evaluation(
                    step, prior_eval=prior_eval, recurrent_context_fresh=True)
                resume_eval_work = None
                prefetched_indices, prefetch_future = (
                    schedule_next_batch_prefetch(step + 1))
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
                                 structured_head=structured_head,
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
            profile = next_step <= profile_until_step
            if profile:
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
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
            prefetch_resident_hits = 0
            prefetch_disk_hits = 0
            prefetch_generated = 0
            prefetch_elapsed_s = 0.0
            prefetched_recall = None
            prefetched_native_features = None
            prefetched_text_batch = None
            prefetched_positions = None
            if prefetch_future is not None and prefetched_indices == indices:
                wait_started = time.perf_counter()
                try:
                    prefetch_result = prefetch_future.result()
                    prefetch_ready = prefetch_result.ready
                    prefetched_recall = prefetch_result.recall
                    prefetched_native_features = prefetch_result.native_features
                    prefetched_text_batch = prefetch_result.text_batch
                    prefetched_positions = prefetch_result.positions
                    prefetch_resident_hits = prefetch_result.resident_hits
                    prefetch_disk_hits = prefetch_result.disk_hits
                    prefetch_generated = prefetch_result.generated
                    prefetch_elapsed_s = prefetch_result.elapsed_s
                except Exception as error:
                    print({"kind": "next_batch_prefetch_failed",
                           "error": repr(error)}, flush=True)
                prefetch_wait_s = time.perf_counter() - wait_started
            elif prefetch_future is not None:
                prefetch_future.cancel()
            prefetched_indices, prefetch_future = schedule_next_batch_prefetch(
                next_step + 1, position_offset=len(indices))
            batch_rows = [rows[i] for i in indices]
            if prefetched_text_batch is None:
                ids, labels, text_mask = make_batch(
                    batch_rows, device="cuda")
            else:
                ids, labels, text_mask = (
                    value.to("cuda", non_blocking=True)
                    for value in prefetched_text_batch)
            positions = (
                supervised_positions(
                    batch_rows, visual_prefix_width(batch_rows, projector),
                    device="cuda")
                if prefetched_positions is None else
                prefetched_positions.to("cuda", non_blocking=True))
            if profile:
                torch.cuda.synchronize()
            t_data = time.perf_counter()
            features = (
                runtime_cached_features(
                    batch_rows, vision, projector, cache_dir)
                if prefetched_native_features is None else
                prefetched_native_features)
            fusion_features = (cached_fusion_features(
                batch_rows, fusion_tower,
                fusion_feature_tokens(fusion_tower, projector), fusion_cache_dir)
                if fusion_tower is not None and (vision_fusion is not None
                                                 or vision_compressor is not None)
                and fusion_cache_dir is not None else None)
            if profile:
                torch.cuda.synchronize()
            t_features = time.perf_counter()
            operator_profile = None
            if next_step <= operator_profile_until_step:
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
                    vision_compressor=vision_compressor,
                    grounding=grounding,
                    grounding_contrastive_weight=args.grounding_contrastive_weight,
                    grounding_early_tokens=args.grounding_early_tokens,
                    grounding_early_weight=args.grounding_early_weight,
                    structured_head=structured_head,
                    structured_rows=batch_rows,
                    structured_weight=args.structured_weight,
                    structured_coordinate_weight=args.structured_coordinate_weight,
                    structured_invalid_box_weight=args.structured_invalid_box_weight,
                    structured_invalid_box_margin=args.structured_invalid_box_margin,
                    activation_checkpoint_min_tokens=(
                        args.activation_checkpoint_min_tokens),
                    activation_checkpoint_max_layers=(
                        args.activation_checkpoint_max_layers),
                    image_aspect=image_aspect_tensor(batch_rows, "cuda"))
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
                    "batch_prefetch_resident_hits": prefetch_resident_hits,
                    "batch_prefetch_disk_hits": prefetch_disk_hits,
                    "batch_prefetch_generated": prefetch_generated,
                    "batch_prefetch_elapsed_s": round(prefetch_elapsed_s, 4),
                }
                record.update(_render_adapter_training_metrics(
                    wrappers, engram))
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
                                  step_s=round(t_backward - t0, 4),
                                  peak_allocated_gib=round(
                                      torch.cuda.max_memory_allocated()
                                      / (1024 ** 3), 3),
                                  peak_reserved_gib=round(
                                      torch.cuda.max_memory_reserved()
                                      / (1024 ** 3), 3))
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
                queue_loop_telemetry(step)
                _atomic_json(out / "status.json", {"state": "training", "step": step,
                                                   "updated": time.time()})
            if step <= profile_until_step or step % 10 == 0:
                print(record, flush=True)

            checkpoint_saved = eval_checkpoint_saved
            if args.eval_every and step % args.eval_every == 0:
                # Release the completed training step before changing to eval
                # sequence geometry. Retained gradients plus allocator
                # fragmentation pushed the largest RADIO buckets into an OOM
                # fallback, after which the asynchronous CUDA failure surfaced
                # misleadingly in the structured matcher.
                optimizer.zero_grad(set_to_none=True)
                del (loss, grad_norm, loop_grad_norm, engram_grad_norm,
                     features, fusion_features, ids, labels, text_mask,
                     positions, prefetched_recall)
                gc.collect()
                torch.cuda.empty_cache()
                checkpoint_saved = run_evaluation(
                    step, checkpoint_saved=checkpoint_saved,
                    recurrent_context_fresh=False)

            if (args.checkpoint_every and step % args.checkpoint_every == 0
                    and not checkpoint_saved):
                _atomic_json(out / "status.json", {
                    "state": "checkpoint_staging", "step": step,
                    "updated": time.time(),
                })
                _sync_log(log)
                # Only the immutable CPU snapshot blocks this thread. ZIP
                # serialization, fsync, and atomic publication overlap the
                # following optimizer steps.
                finish_async_last_checkpoint(wait_for_write=True)
                checkpoint_writer.submit(step, checkpoint_snapshot(step))
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
        flush_loop_telemetry()
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
        finish_async_last_checkpoint(wait_for_write=True)
        flush_loop_telemetry()
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
        flush_loop_telemetry()
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
