"""Full-backbone continued pre-training for Microsoft Mage-Flow-Edit-Base.

The official Mage-Flow repository currently exposes the model and packed
inference path but not a public trainer.  This module keeps the model
implementation external and pinned, while supplying the missing reproducible
training boundary:

* canonical image/caption manifests with geometry and de-duplication;
* native-aspect-ratio, token-budgeted packing;
* the published rectified-flow velocity objective;
* full NR-MMDiT optimization (not LoRA), with frozen Mage-VAE and Qwen3-VL;
* Accelerate/DeepSpeed checkpoint and exact data-position resume.

Image/caption rows are generation examples.  This is intentional: the
Mage-Flow paper mixes generation pairs into both stages of Edit training to
preserve the generative prior.  Fabricating identity edits from a single image
would instead teach the source-conditioned model to copy its input.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import shlex
import shutil
import signal
import struct
import threading
import time
import zlib
from collections.abc import Iterable, Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from queue import Full, Queue
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from rwkv_lab.trainvm_adapters import WorkerTrainingComponents
    from rwkv_lab.trainvm_worker import (
        WorkerControlRuntime,
        WorkerExecutionPhases,
        WorkerObservability,
        WorkerStepProfiler,
    )

RUN_SCHEMA = "rwkv-lab.mage-flow-pretrain.v1"
OFFICIAL_REPOSITORY = "https://github.com/microsoft/Mage"
OFFICIAL_SOURCE_REVISION = "ef932e2cc3e94bb026d937a6cffae65492adc0fb"
DEFAULT_MODEL_ID = "microsoft/Mage-Flow-Edit-Base"
DEFAULT_MODEL_REVISION = "8654a7bc0283ab2946385230b5b2eb944e0b76ea"
CAPTION_FIELDS = ("caption", "text")
IMAGE_FIELDS = ("image", "image_path")
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_PIL_ANCILLARY_CRC_LOCK = threading.Lock()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temp, path)


def _first_text(row: dict[str, Any], fields: Sequence[str]) -> str | None:
    for field in fields:
        value = row.get(field)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def native_size(
    width: int,
    height: int,
    *,
    pixel_budget: int = 1024 * 1024,
    multiple: int = 16,
    max_side: int = 2048,
    max_aspect_ratio: float = 4.0,
) -> tuple[int, int]:
    """Scale to an approximate pixel budget without cropping or distortion."""
    width, height = int(width), int(height)
    if width < 1 or height < 1:
        raise ValueError(f"invalid image size {width}x{height}")
    if pixel_budget < multiple * multiple:
        raise ValueError("pixel_budget is too small")
    aspect = max(width / height, height / width)
    if aspect > max_aspect_ratio:
        raise ValueError(
            f"aspect ratio {aspect:.3f} exceeds configured maximum "
            f"{max_aspect_ratio:.3f}"
        )
    scale = math.sqrt(pixel_budget / float(width * height))
    out_w = max(multiple, round(width * scale / multiple) * multiple)
    out_h = max(multiple, round(height * scale / multiple) * multiple)
    if max(out_w, out_h) > max_side:
        side_scale = max_side / float(max(out_w, out_h))
        out_w = max(multiple, math.floor(out_w * side_scale / multiple) * multiple)
        out_h = max(multiple, math.floor(out_h * side_scale / multiple) * multiple)
    return out_w, out_h


def latent_tokens(width: int, height: int) -> int:
    if width % 16 or height % 16:
        raise ValueError(f"Mage-VAE geometry must be divisible by 16: {width}x{height}")
    return (width // 16) * (height // 16)


def canonical_caption_row(
    row: dict[str, Any],
    *,
    data_root: Path,
    pixel_budget: int,
    max_side: int,
    max_aspect_ratio: float,
    verify_image: bool,
) -> tuple[dict[str, Any] | None, str | None]:
    """Convert common caption-manifest schemas into the training contract."""
    task = str(row.get("task") or "caption").strip().lower()
    if task not in {"caption", "generation", "text_to_image", "t2i"}:
        return None, "non_caption_task"
    image_value = _first_text(row, IMAGE_FIELDS)
    caption = _first_text(row, CAPTION_FIELDS)
    if image_value is None:
        return None, "missing_image"
    if caption is None:
        return None, "missing_caption"
    image_path = Path(image_value).expanduser()
    if not image_path.is_absolute():
        image_path = data_root / image_path
    image_path = image_path.resolve()
    if not image_path.is_file():
        return None, "missing_image_file"

    width = int(row.get("width") or 0)
    height = int(row.get("height") or 0)
    if verify_image or width < 1 or height < 1:
        try:
            from PIL import Image, ImageOps

            with Image.open(image_path) as image:
                image = ImageOps.exif_transpose(image)
                width, height = image.size
                if verify_image:
                    image.load()
        except (OSError, SyntaxError, ValueError):
            return None, "decode_error"
    try:
        train_w, train_h = native_size(
            width,
            height,
            pixel_budget=pixel_budget,
            max_side=max_side,
            max_aspect_ratio=max_aspect_ratio,
        )
    except ValueError:
        return None, "geometry_rejected"

    identity = row.get("image_sha256")
    if not isinstance(identity, str) or not identity:
        identity = hashlib.sha256(str(image_path).encode("utf-8")).hexdigest()
    result = {
        "task": "generation",
        "image": str(image_path),
        "caption": caption,
        "width": width,
        "height": height,
        "train_width": train_w,
        "train_height": train_h,
        "latent_tokens": latent_tokens(train_w, train_h),
        "image_id": identity,
    }
    for key in (
        "source",
        "stage1_source",
        "caption_variant",
        "aesthetic_score",
        "subreddit",
        "caption_model",
        "watermarks",
        "censorship",
    ):
        if key in row:
            result[key] = row[key]
    return result, None


def prepare_manifest(
    input_path: Path,
    output_path: Path,
    *,
    data_root: Path,
    pixel_budget: int = 1024 * 1024,
    max_side: int = 2048,
    max_aspect_ratio: float = 4.0,
    verify_images: bool = True,
    exclude_manifests: Sequence[Path] = (),
    workers: int = 16,
) -> dict[str, Any]:
    """Build an atomic, de-duplicated Mage-Flow generation manifest."""
    from collections import Counter

    excluded: set[str] = set()
    for path in exclude_manifests:
        with path.expanduser().resolve().open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                identity = row.get("image_id") or row.get("image_sha256")
                if identity:
                    excluded.add(str(identity))
                image = _first_text(row, IMAGE_FIELDS)
                if image:
                    candidate = Path(image).expanduser()
                    if not candidate.is_absolute():
                        candidate = data_root / candidate
                    excluded.add(str(candidate.resolve()))

    counters: Counter[str] = Counter()
    seen: set[str] = set()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp = output_path.with_name(output_path.name + ".tmp")

    def process(item: tuple[int, str]):
        line_number, line = item
        if not line.strip():
            return line_number, None, "blank"
        try:
            raw = json.loads(line)
        except json.JSONDecodeError:
            return line_number, None, "invalid_json"
        canonical, reason = canonical_caption_row(
            raw,
            data_root=data_root,
            pixel_budget=pixel_budget,
            max_side=max_side,
            max_aspect_ratio=max_aspect_ratio,
            verify_image=verify_images,
        )
        return line_number, canonical, reason

    def chunks(source, size: int = 1024):
        chunk = []
        for item in enumerate(source, 1):
            chunk.append(item)
            if len(chunk) >= size:
                yield chunk
                chunk = []
        if chunk:
            yield chunk

    with (
        input_path.open(encoding="utf-8") as source,
        temp.open("w", encoding="utf-8") as destination,
        ThreadPoolExecutor(max_workers=max(1, workers)) as pool,
    ):
        for chunk in chunks(source):
            for line_number, canonical, reason in pool.map(process, chunk):
                if reason == "blank":
                    continue
                counters["input"] += 1
                if canonical is None:
                    counters[reason or "rejected"] += 1
                    continue
                identity = str(canonical["image_id"])
                path_identity = str(canonical["image"])
                if identity in excluded or path_identity in excluded:
                    counters["excluded"] += 1
                    continue
                if identity in seen:
                    counters["duplicate"] += 1
                    continue
                seen.add(identity)
                canonical["source_line"] = line_number
                destination.write(json.dumps(canonical, ensure_ascii=False) + "\n")
                counters["output"] += 1
    os.replace(temp, output_path)
    report = {
        "schema": RUN_SCHEMA,
        "created_at": _utc_now(),
        "input": str(input_path.resolve()),
        "output": str(output_path.resolve()),
        "data_root": str(data_root.resolve()),
        "pixel_budget": pixel_budget,
        "max_side": max_side,
        "max_aspect_ratio": max_aspect_ratio,
        "verify_images": verify_images,
        "workers": workers,
        "excluded_manifests": [str(path.resolve()) for path in exclude_manifests],
        "counts": dict(sorted(counters.items())),
    }
    _json_dump(output_path.with_suffix(output_path.suffix + ".report.json"), report)
    return report


def prepare_reddit_split(
    caption_path: Path,
    image_root: Path,
    train_output: Path,
    eval_output: Path,
    *,
    train_count: int = 5_000,
    eval_count: int = 128,
    seed: int = 42,
    artifact_path: Path | None = None,
    pixel_budget: int = 1024 * 1024,
    max_side: int = 2048,
    max_aspect_ratio: float = 4.0,
    workers: int = 16,
    require_clean_artifacts: bool = False,
    allow_smaller_train: bool = False,
    train_count_multiple: int = 1,
) -> dict[str, Any]:
    """Freeze an append-only Reddit caption prefix into exact train/eval splits."""
    from collections import Counter

    caption_path = caption_path.expanduser().resolve()
    image_root = image_root.expanduser().resolve()
    train_output = train_output.expanduser().resolve()
    eval_output = eval_output.expanduser().resolve()
    if train_count < 1 or eval_count < 1 or train_count_multiple < 1:
        raise ValueError("train_count and eval_count must be positive")
    required = train_count + eval_count

    artifacts: dict[str, dict[str, Any]] = {}
    if artifact_path is not None:
        artifact_path = artifact_path.expanduser().resolve()
        with artifact_path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                relative_path = _first_text(row, ("relative_path",))
                if relative_path:
                    artifacts[relative_path] = row

    # Read once so a concurrently appended caption file has a fixed EOF for this
    # invocation. Selection then consumes the earliest valid prefix only.
    source_lines = caption_path.read_text(encoding="utf-8").splitlines()
    counters: Counter[str] = Counter()

    def process(item: tuple[int, str]):
        line_number, line = item
        try:
            raw = json.loads(line)
        except json.JSONDecodeError:
            return line_number, None, "invalid_json"
        if str(raw.get("finish_reason") or "stop").lower() != "stop":
            return line_number, None, "truncated_caption"
        relative_path = _first_text(raw, ("relative_path", "image", "image_path"))
        if relative_path is None:
            return line_number, None, "missing_image"
        subreddit = Path(relative_path).parts[0] if Path(relative_path).parts else ""
        artifact = artifacts.get(relative_path)
        if require_clean_artifacts:
            if artifact is None:
                return line_number, None, "artifact_unscanned"
            if artifact.get("watermarks") or artifact.get("censorship"):
                return line_number, None, "artifact_flagged"
        prepared = {
            **raw,
            "image": relative_path,
            "source": f"reddit/{subreddit}" if subreddit else "reddit",
            "subreddit": subreddit,
            "caption_model": raw.get("model"),
        }
        if artifact is not None:
            prepared["watermarks"] = artifact.get("watermarks") or []
            prepared["censorship"] = artifact.get("censorship") or []
        canonical, reason = canonical_caption_row(
            prepared,
            data_root=image_root,
            pixel_budget=pixel_budget,
            max_side=max_side,
            max_aspect_ratio=max_aspect_ratio,
            verify_image=True,
        )
        return line_number, canonical, reason

    accepted: list[dict[str, Any]] = []
    seen: set[str] = set()
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        for line_number, canonical, reason in pool.map(
            process, enumerate(source_lines, 1)
        ):
            counters["input"] += 1
            if canonical is None:
                counters[reason or "rejected"] += 1
                continue
            identity = str(canonical["image_id"])
            if identity in seen:
                counters["duplicate"] += 1
                continue
            seen.add(identity)
            canonical["source_line"] = line_number
            accepted.append(canonical)
            if len(accepted) == required:
                break

    if len(accepted) < required:
        if not allow_smaller_train:
            raise RuntimeError(
                f"only {len(accepted)} valid unique rows are available; need {required}"
            )
        actual_train_count = min(train_count, len(accepted) - eval_count)
        actual_train_count -= actual_train_count % train_count_multiple
        if actual_train_count < 1:
            raise RuntimeError(
                f"only {len(accepted)} valid unique rows are available; "
                f"cannot reserve {eval_count} eval rows and a positive train split"
            )
        accepted = accepted[: actual_train_count + eval_count]
    else:
        actual_train_count = train_count

    random.Random(seed).shuffle(accepted)
    eval_rows = accepted[:eval_count]
    train_rows = accepted[eval_count : eval_count + actual_train_count]

    def write_rows(path: Path, rows: Sequence[dict[str, Any]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_name(path.name + ".tmp")
        with temp.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        os.replace(temp, path)

    write_rows(train_output, train_rows)
    write_rows(eval_output, eval_rows)
    selected = train_rows + eval_rows
    report = {
        "schema": RUN_SCHEMA,
        "created_at": _utc_now(),
        "caption_input": str(caption_path),
        "caption_input_lines_at_snapshot": len(source_lines),
        "artifact_input": str(artifact_path) if artifact_path else None,
        "image_root": str(image_root),
        "train_output": str(train_output),
        "eval_output": str(eval_output),
        "requested_train_count": train_count,
        "train_count": len(train_rows),
        "eval_count": len(eval_rows),
        "train_count_multiple": train_count_multiple,
        "require_clean_artifacts": require_clean_artifacts,
        "seed": seed,
        "pixel_budget": pixel_budget,
        "max_side": max_side,
        "max_aspect_ratio": max_aspect_ratio,
        "counts": dict(sorted(counters.items())),
        "selected_artifact_coverage": sum(
            "watermarks" in row or "censorship" in row for row in selected
        ),
        "selected_watermark_flags": sum(
            bool(row.get("watermarks")) for row in selected
        ),
        "selected_censorship_flags": sum(
            bool(row.get("censorship")) for row in selected
        ),
    }
    _json_dump(
        train_output.parent / "reddit_split.report.json",
        report,
    )
    return report


@dataclass
class MageFlowTrainConfig:
    train_manifest: str
    output_dir: str
    eval_manifest: str | None = None
    model_id: str = DEFAULT_MODEL_ID
    model_path: str | None = None
    model_revision: str = DEFAULT_MODEL_REVISION
    official_source_revision: str = OFFICIAL_SOURCE_REVISION
    max_steps: int = 16_000
    packed_sequence_tokens: int = 10_240
    gradient_accumulation_steps: int = 4
    learning_rate: float = 1.0e-5
    min_learning_rate_ratio: float = 0.1
    warmup_steps: int = 500
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_epsilon: float = 1.0e-8
    max_grad_norm: float = 1.0
    caption_dropout: float = 0.1
    timestep_sampling: str = "uniform"
    timestep_shift: float = 1.0
    checkpoint_every: int = 500
    eval_every: int = 500
    eval_packs: int = 8
    eval_gen_every: int = 100
    eval_gen_samples: int = 4
    eval_gen_steps: int = 30
    eval_gen_cfg: float = 5.0
    eval_gen_step_zero: bool = True
    eval_gen_screen_prompts: bool = True
    keep_last_n: int = 3
    prefetch_packs: int = 3
    seed: int = 42
    resume_from: str | None = None
    mixed_precision: str = "bf16"
    gradient_checkpointing: bool = True
    vae_sample_posterior: bool = True
    compile_vae_encoder: bool = True
    compile_transformer_blocks: bool = False
    compile_transformer_mode: str = "default"
    compile_transformer_dynamic: bool = False
    tracker_project: str | None = None

    @classmethod
    def from_path(cls, path: Path) -> MageFlowTrainConfig:
        return cls(**json.loads(path.read_text(encoding="utf-8")))

    def validate(self) -> None:
        if not Path(self.train_manifest).expanduser().is_file():
            raise ValueError(f"train manifest not found: {self.train_manifest}")
        if self.eval_manifest and not Path(self.eval_manifest).expanduser().is_file():
            raise ValueError(f"eval manifest not found: {self.eval_manifest}")
        if self.model_id != DEFAULT_MODEL_ID:
            raise ValueError(
                "this full-pretraining contract is qualified for "
                f"{DEFAULT_MODEL_ID}, got {self.model_id}"
            )
        if self.model_revision != DEFAULT_MODEL_REVISION:
            raise ValueError("unqualified Mage-Flow model revision")
        if self.official_source_revision != OFFICIAL_SOURCE_REVISION:
            raise ValueError("unqualified Microsoft Mage source revision")
        if self.model_path and not Path(self.model_path).expanduser().is_dir():
            raise ValueError(f"model path not found: {self.model_path}")
        if self.max_steps < 1 or self.packed_sequence_tokens < 1:
            raise ValueError("max_steps and packed_sequence_tokens must be positive")
        if self.gradient_accumulation_steps < 1:
            raise ValueError("gradient_accumulation_steps must be positive")
        if self.learning_rate <= 0 or not 0 <= self.min_learning_rate_ratio <= 1:
            raise ValueError("invalid learning-rate configuration")
        if not 0 <= self.caption_dropout < 1:
            raise ValueError("caption_dropout must be in [0, 1)")
        if self.timestep_sampling not in {"uniform", "shifted_uniform", "logit_normal"}:
            raise ValueError("unsupported timestep_sampling")
        if self.timestep_shift <= 0:
            raise ValueError("timestep_shift must be positive")
        if self.checkpoint_every < 1 or self.eval_every < 1 or self.eval_packs < 1:
            raise ValueError("checkpoint/eval intervals must be positive")
        if self.eval_gen_every < 0 or self.eval_gen_samples < 0:
            raise ValueError("generation eval interval/sample count cannot be negative")
        if self.eval_gen_samples and (
            self.eval_gen_every < 1
            or self.eval_gen_steps < 1
            or self.eval_gen_cfg <= 0
        ):
            raise ValueError("invalid generation-eval configuration")
        if self.keep_last_n < 1:
            raise ValueError("keep_last_n must be positive")
        if self.mixed_precision != "bf16":
            raise ValueError("Mage-Flow full training is qualified only for bf16")


def resolved_worker_component_contract(
    config: MageFlowTrainConfig,
    worker_components: WorkerTrainingComponents | None,
    worker_controls: WorkerControlRuntime | None = None,
) -> tuple[float, Mapping[str, Mapping[str, str]] | None, str | None]:
    """Bind the legacy scalar configuration to the sealed composition."""

    if worker_components is None:
        return config.learning_rate, None, None
    optimizer = dict(worker_components.configuration("optimizer", category="optimizer"))
    schedule = dict(
        worker_components.configuration(
            "learning_rate", category="learning_rate_schedule"
        )
    )
    expected_optimizer = {
        "learning_rate": (
            optimizer["learning_rate"]
            if worker_controls is not None
            and "learning_rate" in worker_controls.effective_values
            else config.learning_rate
        ),
        "beta1": config.adam_beta1,
        "beta2": config.adam_beta2,
        "epsilon": config.adam_epsilon,
        "foreach": True,
        "fused": False,
    }
    if optimizer != expected_optimizer:
        raise ValueError(
            "authority optimizer composition disagrees with full-backbone configuration"
        )
    if dict(
        worker_components.configuration(
            "parameter_router", category="parameter_router"
        )
    ):
        raise ValueError("full-backbone parameter-router configuration must be empty")
    if schedule != {
        "warmup_steps": config.warmup_steps,
        "max_steps": config.max_steps,
        "minimum_ratio": config.min_learning_rate_ratio,
    }:
        raise ValueError(
            "authority LR-schedule composition disagrees with full-backbone configuration"
        )
    if dict(
        worker_components.configuration(
            "weight_decay", category="weight_decay_schedule"
        )
    ) != {"weight_decay": config.weight_decay}:
        raise ValueError(
            "authority weight-decay composition disagrees with full-backbone configuration"
        )
    if dict(
        worker_components.configuration(
            "gradient_clipping", category="gradient_clipping"
        )
    ) != {
        "max_norm": config.max_grad_norm,
        "norm_type": 2.0,
        "error_if_nonfinite": False,
    }:
        raise ValueError(
            "authority gradient-clipping composition disagrees with full-backbone configuration"
        )
    return (
        float(config.learning_rate),
        worker_components.evidence(),
        worker_components.composition.composition_digest,
    )


def load_manifest(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            required = (
                "image",
                "caption",
                "train_width",
                "train_height",
                "latent_tokens",
            )
            missing = [key for key in required if key not in row]
            if missing:
                raise ValueError(f"{path}:{line_number} missing {missing}")
            rows.append(row)
    if not rows:
        raise ValueError(f"empty manifest: {path}")
    return rows


def epoch_packs(
    rows: Sequence[dict[str, Any]],
    *,
    token_budget: int,
    seed: int,
    epoch: int,
    rank: int = 0,
    world_size: int = 1,
    shuffle: bool = True,
) -> list[list[int]]:
    """Deterministic rank-local sequential first-fit packs."""
    indices = list(range(len(rows)))
    if shuffle:
        random.Random(seed + epoch * 1_000_003).shuffle(indices)
    indices = indices[rank::world_size]
    packs: list[list[int]] = []
    current: list[int] = []
    current_tokens = 0
    for index in indices:
        tokens = int(rows[index].get("packed_tokens", rows[index]["latent_tokens"]))
        if current and current_tokens + tokens > token_budget:
            packs.append(current)
            current, current_tokens = [], 0
        current.append(index)
        current_tokens += tokens
        if current_tokens >= token_budget:
            packs.append(current)
            current, current_tokens = [], 0
    if current:
        packs.append(current)
    return packs


def _png_has_only_ancillary_crc_errors(path: Path) -> bool:
    """Accept a PNG fallback only when corruption is confined to ancillary CRCs."""
    seen: set[bytes] = set()
    ancillary_crc_error = False
    file_size = path.stat().st_size
    with path.open("rb") as handle:
        if handle.read(len(PNG_SIGNATURE)) != PNG_SIGNATURE:
            return False
        while True:
            length_bytes = handle.read(4)
            if len(length_bytes) != 4:
                return False
            length = struct.unpack(">I", length_bytes)[0]
            if length > file_size:
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
            if actual_crc != expected_crc:
                if is_critical:
                    return False
                ancillary_crc_error = True
            seen.add(chunk_type)
            if chunk_type == b"IEND":
                break
    required = {b"IHDR", b"IDAT", b"IEND"}
    return ancillary_crc_error and required.issubset(seen)


def _load_image_tensor(row: dict[str, Any]):
    import numpy as np
    import torch
    from PIL import Image, ImageFile, ImageOps

    width, height = int(row["train_width"]), int(row["train_height"])
    image_path = Path(row["image"])

    def decode() -> Any:
        with Image.open(image_path) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
            image = image.resize((width, height), Image.Resampling.BICUBIC)
            return np.asarray(image, dtype=np.uint8).copy()

    try:
        array = decode()
    except (OSError, SyntaxError):
        if not _png_has_only_ancillary_crc_errors(image_path):
            raise
        # Pillow's switch is process-global, so serialize this exceptional path.
        with _PIL_ANCILLARY_CRC_LOCK:
            previous = ImageFile.LOAD_TRUNCATED_IMAGES
            try:
                ImageFile.LOAD_TRUNCATED_IMAGES = True
                array = decode()
            finally:
                ImageFile.LOAD_TRUNCATED_IMAGES = previous
    tensor = torch.from_numpy(array).permute(2, 0, 1).float().div_(127.5).sub_(1.0)
    return tensor


def _prefetched(iterable: Iterable[Any], transform, *, depth: int) -> Iterator[Any]:
    if depth <= 0:
        for item in iterable:
            yield transform(item)
        return
    queue: Queue[Any] = Queue(maxsize=depth)
    sentinel = object()
    stopped = threading.Event()

    def publish(value: Any) -> bool:
        while not stopped.is_set():
            try:
                queue.put(value, timeout=0.1)
                return True
            except Full:
                continue
        return False

    def producer() -> None:
        try:
            for item in iterable:
                if stopped.is_set() or not publish(transform(item)):
                    return
        except Exception as error:  # noqa: BLE001 - propagate worker failures
            publish(error)
        finally:
            publish(sentinel)

    thread = threading.Thread(
        target=producer, name="mage-flow-data-prefetch", daemon=True
    )
    thread.start()
    try:
        while True:
            item = queue.get()
            if item is sentinel:
                break
            if isinstance(item, BaseException):
                raise item
            yield item
    finally:
        stopped.set()


def _sample_timesteps(config: MageFlowTrainConfig, count: int, device):
    import torch

    if config.timestep_sampling == "logit_normal":
        values = torch.sigmoid(torch.randn(count, device=device))
    else:
        values = torch.rand(count, device=device)
    if config.timestep_sampling == "shifted_uniform":
        shift = config.timestep_shift
        values = shift * values / (1.0 + (shift - 1.0) * values)
    return values.clamp_(1.0e-5, 1.0 - 1.0e-5)


def _lens_to_cu(lens: Sequence[int], device):
    import torch

    values = torch.tensor(lens, dtype=torch.int32, device=device)
    return torch.cat(
        [
            torch.zeros(1, dtype=torch.int32, device=device),
            torch.cumsum(values, dim=0, dtype=torch.int32),
        ]
    )


def _encode_pack(
    model,
    rows,
    images,
    config: MageFlowTrainConfig,
    device,
    *,
    caption_dropout: float | None = None,
):
    """Return exact official packed conditioning and rectified-flow targets."""
    import torch
    from mage_flow.models.utils import PROMPT_TEMPLATE
    from mage_flow.pipeline import _encode_texts_packed

    dropout = config.caption_dropout if caption_dropout is None else caption_dropout
    if dropout <= 0:
        captions = [str(row["caption"]) for row in rows]
    else:
        captions = [
            " " if random.random() < dropout else str(row["caption"]) for row in rows
        ]
    template_info = PROMPT_TEMPLATE["mage-flow"]
    txt_flat, _vec, text_lens = _encode_texts_packed(
        model,
        captions,
        template_info["template"],
        int(template_info["start_idx"]),
        device,
    )
    txt = txt_flat.reshape(1, -1, txt_flat.shape[-1]).to(
        device=device, dtype=torch.bfloat16
    )
    txt_cu = _lens_to_cu(text_lens, device)

    gpu_images = [image.to(device=device, non_blocking=True) for image in images]
    clean, image_shapes = model.compute_vae_encodings(gpu_images, with_ids=False)
    clean = clean.to(device=device, dtype=torch.bfloat16)
    image_lens = [int(row["latent_tokens"]) for row in rows]
    if clean.shape[1] != sum(image_lens):
        raise RuntimeError(
            f"VAE token mismatch: got {clean.shape[1]}, expected {sum(image_lens)}"
        )
    noise = torch.randn_like(clean)
    timesteps = _sample_timesteps(config, len(rows), device)
    token_t = torch.repeat_interleave(
        timesteps, torch.tensor(image_lens, device=device)
    ).view(1, -1, 1)
    noised, velocity = rectified_flow_path(clean, noise, token_t)
    # FP32 timesteps intentionally promote the interpolation for numerical
    # accuracy. The BF16 transformer projection still requires its activation
    # dtype to match the loaded BF16 weights.
    noised = noised.to(dtype=clean.dtype)
    img_shapes = [[shape[0] for shape in image_shapes]]
    return {
        "img": noised,
        "txt": txt,
        "timesteps": timesteps,
        "img_shapes": img_shapes,
        "img_cu_seqlens": _lens_to_cu(image_lens, device),
        "txt_cu_seqlens": txt_cu,
        "velocity": velocity,
        "image_lens": image_lens,
    }


def rectified_flow_loss(prediction, target):
    """Mean target-token/channel velocity error."""
    squared_sum = (prediction.float() - target.float()).square().sum()
    loss = squared_sum / float(target.numel())
    return loss, loss.detach()


def annotate_packed_token_lengths(model, rows: Sequence[dict[str, Any]]) -> None:
    """Measure exact Qwen token lengths so packs include image *and* text."""
    from mage_flow.models.utils import PROMPT_TEMPLATE

    tokenizer = model.txt_enc.tokenizer
    info = PROMPT_TEMPLATE["mage-flow"]
    template, drop_idx = info["template"], int(info["start_idx"])
    max_length = int(model.txt_enc.tokenizer_max_length) + drop_idx
    batch_size = 512
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        encoded = tokenizer(
            [template.format(str(row["caption"])) for row in batch],
            max_length=max_length,
            truncation=True,
            padding=False,
        )["input_ids"]
        for row, token_ids in zip(batch, encoded, strict=True):
            text_tokens = max(1, len(token_ids) - drop_idx)
            row["text_tokens"] = text_tokens
            row["packed_tokens"] = int(row["latent_tokens"]) + text_tokens


def rectified_flow_path(clean, noise, token_timesteps):
    """Released-checkpoint path and denoising velocity.

    Sigma decreases during inference, so ``noise-clean`` is the velocity that
    moves a sigma=1 noise sample to clean data at sigma=0.
    """
    noised = (1.0 - token_timesteps) * clean + token_timesteps * noise
    return noised, noise - clean


def _write_rank_state(
    directory: Path,
    *,
    rank: int,
    global_step: int,
    epoch: int,
    pack_index: int,
    component_composition_digest: str | None = None,
    parameter_routing: Mapping[str, Any] | None = None,
    control_state: Mapping[str, object] | None = None,
) -> None:
    _json_dump(
        directory / f"trainer_state_rank{rank:04d}.json",
        {
            "schema": RUN_SCHEMA,
            "global_step": global_step,
            "epoch": epoch,
            "pack_index": pack_index,
            "component_composition_digest": component_composition_digest,
            "parameter_routing": parameter_routing,
            "worker_controls": control_state,
            "saved_at": _utc_now(),
        },
    )


def _checkpoint_sort_key(path: Path) -> int:
    try:
        return int(path.name.rsplit("-", 1)[-1])
    except ValueError:
        return -1


def _prune_checkpoints(output_dir: Path, keep: int) -> None:
    checkpoints = sorted(
        (path for path in output_dir.glob("checkpoint-*") if path.is_dir()),
        key=_checkpoint_sort_key,
    )
    for path in checkpoints[:-keep]:
        shutil.rmtree(path)


def _save_checkpoint(
    accelerator,
    output_dir: Path,
    *,
    global_step: int,
    epoch: int,
    pack_index: int,
    keep_last_n: int,
    component_composition_digest: str | None = None,
    parameter_routing: Mapping[str, Any] | None = None,
    control_state: Mapping[str, object] | None = None,
) -> Path:
    final = output_dir / f"checkpoint-{global_step:08d}"
    temp = output_dir / f".checkpoint-{global_step:08d}.incomplete"
    accelerator.wait_for_everyone()
    if final.is_dir():
        return final
    if accelerator.is_main_process:
        if temp.exists():
            shutil.rmtree(temp)
        temp.mkdir(parents=True)
    accelerator.wait_for_everyone()
    accelerator.save_state(str(temp))
    _write_rank_state(
        temp,
        rank=accelerator.process_index,
        global_step=global_step,
        epoch=epoch,
        pack_index=pack_index,
        component_composition_digest=component_composition_digest,
        parameter_routing=parameter_routing,
        control_state=control_state,
    )
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        os.replace(temp, final)
        _prune_checkpoints(output_dir, keep_last_n)
    accelerator.wait_for_everyone()
    return final


def _export_transformer(
    accelerator,
    transformer,
    *,
    model_dir: Path,
    output_dir: Path,
    model_id: str,
    model_revision: str,
) -> Path:
    """Write an inference-compatible BF16 transformer replacement."""
    import torch

    state = accelerator.get_state_dict(transformer)
    export_dir = output_dir / "hf_transformer"
    if accelerator.is_main_process:
        from safetensors.torch import save_file

        export_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(
            model_dir / "transformer" / "config.json", export_dir / "config.json"
        )
        cpu_state = {
            key: value.detach().to(device="cpu", dtype=torch.bfloat16).contiguous()
            for key, value in state.items()
        }
        save_file(
            cpu_state,
            export_dir / "diffusion_pytorch_model.safetensors",
            metadata={"format": "pt"},
        )
        _json_dump(
            export_dir / "export_contract.json",
            {
                "schema": RUN_SCHEMA,
                "created_at": _utc_now(),
                "base_model": model_id,
                "base_model_revision": model_revision,
                "usage": (
                    "Replace transformer/ in a snapshot of the pinned base model "
                    "with this directory; keep the base VAE, text encoder, scheduler, "
                    "and model_index.json."
                ),
            },
        )
    accelerator.wait_for_everyone()
    return export_dir


def _evaluate(
    accelerator,
    transformer,
    model,
    rows: Sequence[dict[str, Any]],
    config: MageFlowTrainConfig,
    device,
) -> float:
    import torch

    packs = epoch_packs(
        rows,
        token_budget=config.packed_sequence_tokens,
        seed=config.seed + 91_919,
        epoch=0,
        rank=accelerator.process_index,
        world_size=accelerator.num_processes,
        shuffle=False,
    )[: config.eval_packs]
    values = []
    transformer.eval()
    cuda_devices = (
        [device.index if device.index is not None else torch.cuda.current_device()]
        if device.type == "cuda"
        else []
    )
    with torch.no_grad(), torch.random.fork_rng(devices=cuda_devices):
        torch.manual_seed(config.seed + 7_777)
        for indices in packs:
            batch_rows = [rows[index] for index in indices]
            images = [_load_image_tensor(row) for row in batch_rows]
            flow = _encode_pack(
                model,
                batch_rows,
                images,
                config,
                device,
                caption_dropout=0.0,
            )
            prediction = transformer(
                img=flow["img"],
                txt=flow["txt"],
                timesteps=flow["timesteps"],
                img_shapes=flow["img_shapes"],
                img_cu_seqlens=flow["img_cu_seqlens"],
                txt_cu_seqlens=flow["txt_cu_seqlens"],
            )
            values.append(
                (prediction.float() - flow["velocity"].float()).square().mean()
            )
    transformer.train()
    local = (
        torch.stack(values).mean()
        if values
        else torch.tensor(float("nan"), device=device)
    )
    return float(accelerator.reduce(local, reduction="mean").item())


def _generate_eval_snapshot(
    pipeline,
    transformer,
    rows: Sequence[dict[str, Any]],
    config: MageFlowTrainConfig,
    device,
    output_dir: Path,
    *,
    step: int,
    baseline: bool = False,
) -> Path | None:
    """Generate a deterministic held-out visual snapshot for the dashboard."""
    import torch

    count = min(config.eval_gen_samples, len(rows))
    if count < 1:
        return None
    selection_path = output_dir / "eval_generation_selection.json"
    selected_indices: list[int] = []
    if selection_path.is_file():
        saved_selection = json.loads(selection_path.read_text(encoding="utf-8"))
        if (
            saved_selection.get("model_revision") == config.model_revision
            and saved_selection.get("screen_prompts", True)
            == config.eval_gen_screen_prompts
        ):
            for item in saved_selection.get("items", []):
                index = int(item["index"])
                if not 0 <= index < len(rows):
                    selected_indices = []
                    break
                caption_hash = hashlib.sha256(
                    str(rows[index]["caption"]).encode("utf-8")
                ).hexdigest()
                if caption_hash != item.get("caption_sha256"):
                    selected_indices = []
                    break
                selected_indices.append(index)
    if len(selected_indices) < count:
        selected_indices = []
        for index, row in enumerate(rows):
            allowed = True
            if config.eval_gen_screen_prompts:
                verdict = pipeline.model.txt_enc.screen_text(str(row["caption"]))
                allowed = not verdict.violates
            if allowed:
                selected_indices.append(index)
                if len(selected_indices) == count:
                    break
        if len(selected_indices) < count:
            raise RuntimeError(
                f"only {len(selected_indices)} of {count} held-out generation "
                "prompts passed the pinned model's mandatory content gate"
            )
        _json_dump(
            selection_path,
            {
                "schema": RUN_SCHEMA,
                "created_at": _utc_now(),
                "model_revision": config.model_revision,
                "screen_prompts": config.eval_gen_screen_prompts,
                "items": [
                    {
                        "index": index,
                        "image": str(rows[index]["image"]),
                        "caption_sha256": hashlib.sha256(
                            str(rows[index]["caption"]).encode("utf-8")
                        ).hexdigest(),
                    }
                    for index in selected_indices
                ],
            },
        )
    selected_indices = selected_indices[:count]
    selection_sha256 = hashlib.sha256(
        json.dumps(selected_indices, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    artifact = output_dir / "eval_samples" / f"step_{step:08d}.json"
    if artifact.is_file():
        existing = json.loads(artifact.read_text(encoding="utf-8"))
        if (
            existing.get("selection_sha256") == selection_sha256
            and existing.get("prompt_screening", True)
            == config.eval_gen_screen_prompts
        ):
            return artifact

    selected = [rows[index] for index in selected_indices]
    prompts = [str(row["caption"]) for row in selected]
    seeds = [config.seed + 100_000 + index for index in range(count)]
    heights = [int(row["train_height"]) for row in selected]
    widths = [int(row["train_width"]) for row in selected]
    image_dir = output_dir / "eval_generations" / f"step_{step:08d}"
    image_dir.mkdir(parents=True, exist_ok=True)

    model = pipeline.model
    previous_transformer = model.transformer
    text_encoder = model.txt_enc
    had_instance_screen = "screen_text" in vars(text_encoder)
    previous_instance_screen = vars(text_encoder).get("screen_text")
    was_training = bool(transformer.training)
    cuda_devices = (
        [device.index if device.index is not None else torch.cuda.current_device()]
        if device.type == "cuda"
        else []
    )
    try:
        model.transformer = transformer
        transformer.eval()
        if not config.eval_gen_screen_prompts:
            class AllowedVerdict:
                violates = False

            text_encoder.screen_text = lambda _prompt: AllowedVerdict()
        with torch.random.fork_rng(devices=cuda_devices):
            images = pipeline.generate(
                prompts,
                seeds=seeds,
                steps=config.eval_gen_steps,
                cfg=config.eval_gen_cfg,
                heights=heights,
                widths=widths,
                device=str(device),
            )
    finally:
        if not config.eval_gen_screen_prompts:
            if had_instance_screen:
                text_encoder.screen_text = previous_instance_screen
            else:
                del text_encoder.screen_text
        model.transformer = previous_transformer
        transformer.train(was_training)

    items = []
    for index, (row, image) in enumerate(zip(selected, images, strict=True)):
        image_path = (image_dir / f"sample_{index:02d}.png").resolve()
        temporary = image_path.with_name(image_path.name + ".tmp")
        image.save(temporary, format="PNG")
        os.replace(temporary, image_path)
        label = "pinned base model" if baseline else f"training step {step}"
        items.append(
            {
                "image_id": str(row["image_id"]),
                "image": str(image_path),
                "prompt": prompts[index],
                "route": "full_backbone",
                "reference": f"held-out target image: {row['image']}",
                "caption": (
                    f"{label} · seed {seeds[index]} · "
                    f"{widths[index]}×{heights[index]} · "
                    f"{config.eval_gen_steps} steps · CFG {config.eval_gen_cfg:g}"
                ),
                "tokens": 0,
                "stopped_at_eod": True,
                "source": str(row.get("source", "mage_flow_eval_generation")),
                "target_image": str(row["image"]),
                "seed": seeds[index],
                "width": widths[index],
                "height": heights[index],
                "sampling_attributes": {
                    "cfg": f"{config.eval_gen_cfg:g}",
                    "height": str(heights[index]),
                    "route": "full_backbone",
                    "sampler": "mage_flow_rectified_flow",
                    "steps": str(config.eval_gen_steps),
                    "width": str(widths[index]),
                },
            }
        )
    _json_dump(
        artifact,
        {
            "schema": RUN_SCHEMA,
            "eval_kind": "image_generation",
            "step": step,
            "ppl": 0.0,
            "decoding": "mage_flow_rectified_flow",
            "max_new": 0,
            "complete": True,
            "generation_steps": config.eval_gen_steps,
            "base_model": config.model_id,
            "base_model_revision": config.model_revision,
            "baseline": baseline,
            "prompt_screening": config.eval_gen_screen_prompts,
            "selection_sha256": selection_sha256,
            "items": items,
        },
    )
    return artifact


def train(
    config: MageFlowTrainConfig,
    *,
    worker_components: WorkerTrainingComponents | None = None,
    worker_step_profiler: WorkerStepProfiler | None = None,
    worker_observability: WorkerObservability | None = None,
    worker_controls: WorkerControlRuntime | None = None,
    worker_execution_phases: WorkerExecutionPhases | None = None,
) -> None:
    """Execute a full NR-MMDiT continued-pretraining run."""
    config.validate()
    try:
        import torch
        from accelerate import Accelerator
        from accelerate.utils import (
            DummyOptim,
            DummyScheduler,
            ProjectConfiguration,
            set_seed,
        )
        from huggingface_hub import snapshot_download
        from mage_flow import MageFlowPipeline

        from rwkv_lab.mage_flow_optimizations import compile_transformer_blocks
        from rwkv_lab.training_components import (
            AdamWConfiguration,
            LinearWarmupCosineConfiguration,
            OptimizerImplementation,
            ScheduleImplementation,
            build_registered_optimizer,
            build_registered_schedule,
        )
        from rwkv_lab.trainvm_adapters.mageflow_controls import MageFlowMutableControls
    except ImportError as error:
        raise RuntimeError(
            "Mage-Flow training dependencies are missing. Run the generated "
            "bootstrap script in a dedicated Python 3.11 environment."
        ) from error

    output_dir = Path(config.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    _json_dump(
        output_dir / "status.json",
        {"schema": RUN_SCHEMA, "state": "initializing", "step": 0, "updated_at": _utc_now()},
    )
    project = ProjectConfiguration(
        project_dir=str(output_dir), logging_dir=str(output_dir / "logs")
    )
    log_with = "wandb" if config.tracker_project else None
    accelerator = Accelerator(
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        mixed_precision=config.mixed_precision,
        project_config=project,
        log_with=log_with,
    )
    if worker_components is not None and accelerator.num_processes != 1:
        raise RuntimeError(
            "the TrainVM full-backbone profile currently requires one worker process"
        )
    set_seed(config.seed, device_specific=True)
    if config.tracker_project:
        accelerator.init_trackers(config.tracker_project, config=asdict(config))

    model_dir = (
        str(Path(config.model_path).expanduser().resolve())
        if config.model_path
        else snapshot_download(
            repo_id=config.model_id,
            revision=config.model_revision,
        )
    )
    pipeline = MageFlowPipeline.from_pretrained(
        model_dir, device=str(accelerator.device)
    )
    model = pipeline.model
    model.vae.sample_posterior = config.vae_sample_posterior
    model.config.compile_vae_encoder = config.compile_vae_encoder
    if config.compile_vae_encoder and worker_execution_phases is None:
        model.maybe_compile_vae_encoder()
    model.vae.eval().requires_grad_(False)
    model.txt_enc.eval().requires_grad_(False)
    transformer = model.transformer
    transformer.checkpoint = config.gradient_checkpointing
    transformer.train().requires_grad_(True)

    optimizer_learning_rate, component_evidence, component_digest = (
        resolved_worker_component_contract(config, worker_components, worker_controls)
    )
    parameter_routing = (
        worker_components.parameter_routing(
            transformer.named_parameters(),
            {},
            base_learning_rate=optimizer_learning_rate,
        )
        if worker_components is not None
        else None
    )

    train_rows = load_manifest(Path(config.train_manifest).expanduser().resolve())
    eval_rows = (
        load_manifest(Path(config.eval_manifest).expanduser().resolve())
        if config.eval_manifest
        else []
    )
    annotate_packed_token_lengths(model, train_rows)
    if eval_rows:
        annotate_packed_token_lengths(model, eval_rows)

    deepspeed_config = (
        accelerator.state.deepspeed_plugin.deepspeed_config
        if accelerator.state.deepspeed_plugin is not None
        else {}
    )
    if worker_components is not None and (
        "optimizer" in deepspeed_config or "scheduler" in deepspeed_config
    ):
        raise RuntimeError(
            "DeepSpeed may not replace a TrainVM-authority optimizer or schedule"
        )
    weight_decay_schedule = None
    if worker_components is not None:
        assert parameter_routing is not None
        optimizer = worker_components.optimizer(parameter_routing.groups)
        weight_decay_schedule = worker_components.weight_decay_schedule(optimizer)
    elif "optimizer" in deepspeed_config:
        optimizer = DummyOptim(
            transformer.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
    else:
        optimizer = build_registered_optimizer(
            OptimizerImplementation.TORCH_ADAMW_V1,
            transformer.parameters(),
            AdamWConfiguration(
                learning_rate=config.learning_rate,
                beta1=config.adam_beta1,
                beta2=config.adam_beta2,
                epsilon=config.adam_epsilon,
                weight_decay=config.weight_decay,
            ),
        )
    if worker_components is not None:
        scheduler = worker_components.learning_rate_schedule(optimizer)
    elif "scheduler" in deepspeed_config:
        scheduler = DummyScheduler(
            optimizer,
            total_num_steps=config.max_steps,
            warmup_num_steps=config.warmup_steps,
        )
    else:
        scheduler = build_registered_schedule(
            ScheduleImplementation.LINEAR_WARMUP_COSINE_V1,
            optimizer,
            LinearWarmupCosineConfiguration(
                warmup_steps=config.warmup_steps,
                max_steps=config.max_steps,
                minimum_ratio=config.min_learning_rate_ratio,
            ),
        )
    mutable_scheduler = scheduler
    transformer, optimizer, scheduler = accelerator.prepare(
        transformer, optimizer, scheduler
    )
    mutable_controls = (
        MageFlowMutableControls(config, mutable_scheduler, worker_controls)
        if worker_controls is not None
        else None
    )

    global_step, start_epoch, start_pack = 0, 0, 0
    if config.resume_from:
        resume = Path(config.resume_from).expanduser().resolve()
        accelerator.load_state(str(resume))
        rank_state = json.loads(
            (
                resume / f"trainer_state_rank{accelerator.process_index:04d}.json"
            ).read_text()
        )
        global_step = int(rank_state["global_step"])
        start_epoch = int(rank_state["epoch"])
        start_pack = int(rank_state["pack_index"])
        if worker_components is not None:
            if rank_state.get("component_composition_digest") != component_digest:
                raise ValueError(
                    "resume checkpoint component composition disagrees with invocation"
                )
            expected_routing = (
                parameter_routing.report if parameter_routing is not None else None
            )
            if rank_state.get("parameter_routing") != expected_routing:
                raise ValueError(
                    "resume checkpoint parameter routing disagrees with invocation"
                )
        if worker_controls is not None:
            state = rank_state.get("worker_controls")
            if not isinstance(state, Mapping):
                raise ValueError("resume checkpoint omits worker control state")
            worker_controls.verify_checkpoint_state(state)
    last_epoch, next_pack = start_epoch, start_pack

    unwrapped_transformer = accelerator.unwrap_model(transformer)
    if worker_execution_phases is None:
        compile_report = compile_transformer_blocks(
            unwrapped_transformer,
            enabled=config.compile_transformer_blocks,
            mode=config.compile_transformer_mode,
            dynamic=config.compile_transformer_dynamic,
        )
    else:
        from rwkv_lab.trainvm_adapters.mageflow_phases import (
            run_mageflow_execution_phases,
        )
        from rwkv_lab.trainvm_worker import ExecutionPhase

        compile_report = compile_transformer_blocks(
            unwrapped_transformer,
            enabled=False,
            mode=config.compile_transformer_mode,
            dynamic=config.compile_transformer_dynamic,
        )
        compile_request = worker_execution_phases.request(ExecutionPhase.COMPILE)
        warmup_request = worker_execution_phases.request(ExecutionPhase.WARMUP)
        needs_workload = any(
            request is not None and request.enabled
            for request in (compile_request, warmup_request)
        )
        phase_rows: list[dict[str, Any]] = []
        phase_images: list[Any] = []
        if needs_workload:
            phase_epoch = start_epoch
            phase_pack = start_pack
            while not phase_rows:
                phase_packs = epoch_packs(
                    train_rows,
                    token_budget=config.packed_sequence_tokens,
                    seed=config.seed,
                    epoch=phase_epoch,
                    rank=accelerator.process_index,
                    world_size=accelerator.num_processes,
                    shuffle=True,
                )
                if phase_pack < len(phase_packs):
                    phase_rows = [
                        train_rows[index] for index in phase_packs[phase_pack]
                    ]
                    break
                phase_epoch += 1
                phase_pack = 0
                if phase_epoch > start_epoch + 1:
                    raise RuntimeError(
                        "MageFlow execution phase could not resolve a training pack"
                    )
            phase_images = [_load_image_tensor(row) for row in phase_rows]

        def phase_extra_state() -> Mapping[str, Any]:
            return {
                "component_composition_digest": component_digest,
                "controls": (
                    worker_controls.checkpoint_state()
                    if worker_controls is not None
                    else {}
                ),
                "data_cursor": {
                    "epoch": start_epoch,
                    "optimizer_step": global_step,
                    "pack_index": start_pack,
                },
                "parameter_routing": (
                    parameter_routing.report if parameter_routing is not None else None
                ),
                "scheduler": scheduler.state_dict(),
            }

        def phase_training_workload() -> None:
            flow = _encode_pack(
                model,
                phase_rows,
                phase_images,
                config,
                accelerator.device,
            )
            prediction = transformer(
                img=flow["img"],
                txt=flow["txt"],
                timesteps=flow["timesteps"],
                img_shapes=flow["img_shapes"],
                img_cu_seqlens=flow["img_cu_seqlens"],
                txt_cu_seqlens=flow["txt_cu_seqlens"],
            )
            loss, _observed = rectified_flow_loss(prediction, flow["velocity"])
            loss.backward()

        def compile_phase() -> Mapping[str, Any]:
            if config.compile_vae_encoder:
                model.maybe_compile_vae_encoder()
            return compile_transformer_blocks(
                unwrapped_transformer,
                enabled=True,
                mode=config.compile_transformer_mode,
                dynamic=config.compile_transformer_dynamic,
            )

        compile_report = run_mageflow_execution_phases(
            worker_execution_phases,
            trajectory_model=unwrapped_transformer,
            optimizer=optimizer,
            optimizer_step=global_step,
            disabled_compile_report=compile_report,
            compile_workload=compile_phase,
            training_workload=phase_training_workload,
            extra_state=phase_extra_state,
            synchronize=lambda: torch.cuda.synchronize(accelerator.device),
        )

    if (
        global_step == 0
        and eval_rows
        and config.eval_gen_samples
        and config.eval_gen_step_zero
    ):
        if accelerator.is_main_process:
            _json_dump(
                output_dir / "status.json",
                {
                    "schema": RUN_SCHEMA,
                    "state": "evaluating_generation",
                    "step": 0,
                    "updated_at": _utc_now(),
                },
            )
            _generate_eval_snapshot(
                pipeline,
                transformer,
                eval_rows,
                config,
                accelerator.device,
                output_dir,
                step=0,
                baseline=True,
            )
        accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        _json_dump(
            output_dir / "run_contract.json",
            {
                "schema": RUN_SCHEMA,
                "created_at": _utc_now(),
                "config": asdict(config),
                "official_repository": OFFICIAL_REPOSITORY,
                "official_source_revision": config.official_source_revision,
                "model_revision": config.model_revision,
                "torch": torch.__version__,
                "world_size": accelerator.num_processes,
                "train_examples": len(train_rows),
                "eval_examples": len(eval_rows),
                "component_composition": component_evidence,
                "component_composition_digest": component_digest,
                "parameter_routing": (
                    parameter_routing.report if parameter_routing is not None else None
                ),
                "regional_compile": compile_report,
            },
        )
    accelerator.wait_for_everyone()

    stop_requested = {"value": False}

    def handle_stop(signum, _frame):
        stop_requested["value"] = True
        if accelerator.is_main_process:
            print(
                f"received signal {signum}; saving after the current optimizer step",
                flush=True,
            )

    signal.signal(signal.SIGINT, handle_stop)
    signal.signal(signal.SIGTERM, handle_stop)
    if accelerator.is_main_process:
        _json_dump(
            output_dir / "status.json",
            {
                "schema": RUN_SCHEMA,
                "state": "training",
                "step": global_step,
                "updated_at": _utc_now(),
            },
        )

    epoch = start_epoch
    update_started = time.perf_counter()
    accumulated_samples = 0
    accumulated_microbatches = 0
    accumulated_target_tokens = 0
    accumulated_sequence_tokens = 0
    accumulated_loss = None
    while global_step < config.max_steps:
        packs = epoch_packs(
            train_rows,
            token_budget=config.packed_sequence_tokens,
            seed=config.seed,
            epoch=epoch,
            rank=accelerator.process_index,
            world_size=accelerator.num_processes,
            shuffle=True,
        )
        first_pack = start_pack if epoch == start_epoch else 0
        pack_stream = (
            (pack_index, [train_rows[index] for index in indices])
            for pack_index, indices in enumerate(packs[first_pack:], first_pack)
        )

        def load_pack(item):
            pack_index, batch_rows = item
            return (
                pack_index,
                batch_rows,
                [_load_image_tensor(row) for row in batch_rows],
            )

        loaded_packs = _prefetched(
            pack_stream, load_pack, depth=config.prefetch_packs
        )
        if worker_step_profiler is not None:
            loaded_packs = worker_step_profiler.track_input(loaded_packs)
        for pack_index, batch_rows, images in loaded_packs:
            if mutable_controls is not None:
                worker_controls.microbatch(global_step + 1, mutable_controls.apply)
            accumulated_samples += len(batch_rows)
            with accelerator.accumulate(transformer):
                flow = _encode_pack(
                    model, batch_rows, images, config, accelerator.device
                )
                prediction = transformer(
                    img=flow["img"],
                    txt=flow["txt"],
                    timesteps=flow["timesteps"],
                    img_shapes=flow["img_shapes"],
                    img_cu_seqlens=flow["img_cu_seqlens"],
                    txt_cu_seqlens=flow["txt_cu_seqlens"],
                )
                loss, observed_mse = rectified_flow_loss(prediction, flow["velocity"])
                accelerator.backward(loss)
                accumulated_microbatches += 1
                accumulated_target_tokens += sum(flow["image_lens"])
                accumulated_sequence_tokens += sum(
                    int(row["packed_tokens"]) for row in batch_rows
                )
                accumulated_loss = (
                    observed_mse.detach()
                    if accumulated_loss is None
                    else accumulated_loss + observed_mse.detach()
                )
                if accelerator.sync_gradients:
                    if mutable_controls is not None:
                        worker_controls.optimizer_step(
                            global_step + 1, mutable_controls.apply
                        )
                    grad_norm = (
                        worker_components.gradient_clipping(
                            unwrapped_transformer.parameters()
                        )
                        if worker_components is not None
                        else accelerator.clip_grad_norm_(
                            transformer.parameters(), config.max_grad_norm
                        )
                    )
                else:
                    grad_norm = torch.tensor(float("nan"), device=accelerator.device)
                if accelerator.sync_gradients and weight_decay_schedule is not None:
                    weight_decay_schedule.step(global_step)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if not accelerator.sync_gradients:
                continue
            global_step += 1
            step_seconds = time.perf_counter() - update_started
            if worker_step_profiler is not None:
                worker_step_profiler.step(global_step)
            last_epoch, next_pack = epoch, pack_index + 1
            if accumulated_loss is None or accumulated_microbatches < 1:
                raise RuntimeError("optimizer update has no accumulated loss")
            mean_loss = float(
                (accumulated_loss / accumulated_microbatches).item()
            )
            metrics = {
                "train/loss": mean_loss,
                "train/optimization_loss": mean_loss,
                "train/grad_norm": float(grad_norm),
                "train/lr": float(scheduler.get_last_lr()[0]),
                "train/samples": accumulated_samples,
                "train/target_tokens": accumulated_target_tokens,
                "train/sequence_tokens": accumulated_sequence_tokens,
                "train/epoch": epoch,
                "train/images_per_second": accumulated_samples / step_seconds,
                "train/step_seconds": step_seconds,
            }
            accelerator.log(metrics, step=global_step)
            if worker_observability is not None:
                worker_observability.optimizer_step(global_step)
                worker_observability.publish_if_declared(
                    "train.loss",
                    metrics["train/loss"],
                    step=global_step,
                    sample_weight=accumulated_samples,
                )
                worker_observability.publish_if_declared(
                    "train.images_per_second",
                    metrics["train/images_per_second"],
                    step=global_step,
                )
                worker_observability.publish_if_declared(
                    "system.gpu_memory_used",
                    int(torch.cuda.memory_allocated(accelerator.device)),
                    step=global_step,
                )
            update_started = time.perf_counter()
            accumulated_samples = 0
            accumulated_microbatches = 0
            accumulated_target_tokens = 0
            accumulated_sequence_tokens = 0
            accumulated_loss = None
            if accelerator.is_main_process:
                with (output_dir / "train.jsonl").open(
                    "a", encoding="utf-8"
                ) as handle:
                    handle.write(
                        json.dumps(
                            {
                                "kind": "train",
                                "step": global_step,
                                "loss": metrics["train/loss"],
                                "lr": metrics["train/lr"],
                                "gnorm": metrics["train/grad_norm"],
                                **metrics,
                            }
                        )
                        + "\n"
                    )
                _json_dump(
                    output_dir / "status.json",
                    {
                        "schema": RUN_SCHEMA,
                        "state": "training",
                        "step": global_step,
                        "updated_at": _utc_now(),
                    },
                )
            if eval_rows and global_step % config.eval_every == 0:
                if mutable_controls is not None:
                    worker_controls.evaluation(global_step, mutable_controls.apply)
                eval_loss = _evaluate(
                    accelerator,
                    transformer,
                    model,
                    eval_rows,
                    config,
                    accelerator.device,
                )
                accelerator.log({"eval/loss": eval_loss}, step=global_step)
                if worker_observability is not None:
                    worker_observability.publish_if_declared(
                        "eval.loss", eval_loss, step=global_step
                    )
                if accelerator.is_main_process:
                    with (output_dir / "train.jsonl").open(
                        "a", encoding="utf-8"
                    ) as handle:
                        handle.write(
                            json.dumps(
                                {
                                    "kind": "eval",
                                    "step": global_step,
                                    "loss": eval_loss,
                                    "eval/loss": eval_loss,
                                }
                            )
                            + "\n"
                        )
            checkpoint_requested = bool(
                worker_controls is not None
                and worker_controls.checkpoint_boundary_requested
            )
            if global_step % config.checkpoint_every == 0 or checkpoint_requested:
                if mutable_controls is not None:
                    worker_controls.checkpoint(global_step, mutable_controls.apply)
                checkpoint = _save_checkpoint(
                    accelerator,
                    output_dir,
                    global_step=global_step,
                    epoch=epoch,
                    pack_index=pack_index + 1,
                    keep_last_n=config.keep_last_n,
                    component_composition_digest=component_digest,
                    parameter_routing=(
                        parameter_routing.report
                        if parameter_routing is not None
                        else None
                    ),
                    control_state=(
                        worker_controls.checkpoint_state()
                        if worker_controls is not None
                        else None
                    ),
                )
                if (
                    worker_controls is not None
                    and worker_controls.checkpoint_completion_requested
                ):
                    worker_controls.publish_requested_checkpoint_directory(
                        str(checkpoint),
                        optimizer_step=global_step,
                        resume_grade="compatible",
                        state_components=(
                            "component_composition",
                            "control_revision",
                            "data_cursor",
                            "lr_schedule",
                            "model",
                            "optimizer",
                            "parameter_routing",
                            "rng_accelerator",
                            "rng_python",
                            "rng_torch",
                        ),
                    )
            if (
                eval_rows
                and config.eval_gen_samples
                and global_step % config.eval_gen_every == 0
            ):
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    _json_dump(
                        output_dir / "status.json",
                        {
                            "schema": RUN_SCHEMA,
                            "state": "evaluating_generation",
                            "step": global_step,
                            "updated_at": _utc_now(),
                        },
                    )
                    _generate_eval_snapshot(
                        pipeline,
                        accelerator.unwrap_model(transformer),
                        eval_rows,
                        config,
                        accelerator.device,
                        output_dir,
                        step=global_step,
                    )
                    _json_dump(
                        output_dir / "status.json",
                        {
                            "schema": RUN_SCHEMA,
                            "state": "training",
                            "step": global_step,
                            "updated_at": _utc_now(),
                        },
                    )
                accelerator.wait_for_everyone()
            if stop_requested["value"]:
                if mutable_controls is not None:
                    worker_controls.checkpoint(global_step, mutable_controls.apply)
                checkpoint = _save_checkpoint(
                    accelerator,
                    output_dir,
                    global_step=global_step,
                    epoch=epoch,
                    pack_index=pack_index + 1,
                    keep_last_n=config.keep_last_n,
                    component_composition_digest=component_digest,
                    parameter_routing=(
                        parameter_routing.report
                        if parameter_routing is not None
                        else None
                    ),
                    control_state=(
                        worker_controls.checkpoint_state()
                        if worker_controls is not None
                        else None
                    ),
                )
                if accelerator.is_main_process:
                    _json_dump(
                        output_dir / "interrupted.json",
                        {
                            "schema": RUN_SCHEMA,
                            "state": "interrupted",
                            "interrupted_at": _utc_now(),
                            "global_step": global_step,
                            "checkpoint": str(checkpoint),
                        },
                    )
                    _json_dump(
                        output_dir / "status.json",
                        {
                            "schema": RUN_SCHEMA,
                            "state": "interrupted",
                            "step": global_step,
                            "updated_at": _utc_now(),
                        },
                    )
                accelerator.end_training()
                return
            if global_step >= config.max_steps:
                break
        if global_step >= config.max_steps:
            break
        epoch += 1
        last_epoch, next_pack = epoch, 0
        start_pack = 0

    if eval_rows and config.eval_gen_samples:
        if mutable_controls is not None:
            worker_controls.evaluation(global_step, mutable_controls.apply)
        if accelerator.is_main_process:
            _generate_eval_snapshot(
                pipeline,
                transformer,
                eval_rows,
                config,
                accelerator.device,
                output_dir,
                step=global_step,
            )
        accelerator.wait_for_everyone()
    if mutable_controls is not None:
        worker_controls.checkpoint(global_step, mutable_controls.apply)
    final_checkpoint = _save_checkpoint(
        accelerator,
        output_dir,
        global_step=global_step,
        epoch=last_epoch,
        pack_index=next_pack,
        keep_last_n=config.keep_last_n,
        component_composition_digest=component_digest,
        parameter_routing=(
            parameter_routing.report if parameter_routing is not None else None
        ),
        control_state=(
            worker_controls.checkpoint_state()
            if worker_controls is not None
            else None
        ),
    )
    export_dir = _export_transformer(
        accelerator,
        transformer,
        model_dir=Path(model_dir),
        output_dir=output_dir,
        model_id=config.model_id,
        model_revision=config.model_revision,
    )
    if accelerator.is_main_process:
        _json_dump(
            output_dir / "complete.json",
            {
                "schema": RUN_SCHEMA,
                "completed_at": _utc_now(),
                "global_step": global_step,
                "checkpoint": str(final_checkpoint),
                "transformer_export": str(export_dir),
            },
        )
        _json_dump(
            output_dir / "status.json",
            {
                "schema": RUN_SCHEMA,
                "state": "complete",
                "step": global_step,
                "updated_at": _utc_now(),
            },
        )
    accelerator.end_training()


def deepspeed_config(config: MageFlowTrainConfig) -> dict[str, Any]:
    """Single/multi-GPU ZeRO-2 with exact AdamW states offloaded to host RAM."""
    return {
        "bf16": {"enabled": True},
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": config.learning_rate,
                "betas": [config.adam_beta1, config.adam_beta2],
                "eps": config.adam_epsilon,
                "weight_decay": config.weight_decay,
                "adam_w_mode": True,
            },
        },
        "scheduler": {
            "type": "WarmupCosineLR",
            "params": {
                "total_num_steps": config.max_steps,
                "warmup_num_steps": config.warmup_steps,
                "warmup_min_ratio": 0.0,
                "cos_min_ratio": config.min_learning_rate_ratio,
                "warmup_type": "linear",
            },
        },
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "gradient_clipping": config.max_grad_norm,
        "train_micro_batch_size_per_gpu": 1,
        "train_batch_size": "auto",
        "zero_optimization": {
            "stage": 2,
            "offload_optimizer": {
                "device": "cpu",
                "pin_memory": True,
            },
            "contiguous_gradients": True,
            "overlap_comm": False,
            "reduce_scatter": True,
        },
        "steps_per_print": 100,
        "wall_clock_breakdown": False,
    }


def accelerate_config(deepspeed_path: Path) -> dict[str, Any]:
    return {
        "compute_environment": "LOCAL_MACHINE",
        "debug": False,
        "distributed_type": "DEEPSPEED",
        "downcast_bf16": "no",
        "enable_cpu_affinity": True,
        "machine_rank": 0,
        "main_training_function": "main",
        "num_machines": 1,
        "num_processes": 1,
        "rdzv_backend": "static",
        "same_network": True,
        "use_cpu": False,
        "deepspeed_config": {
            "deepspeed_config_file": str(deepspeed_path.resolve()),
            "zero3_init_flag": False,
        },
    }


def prepare_run(config: MageFlowTrainConfig, run_dir: Path) -> dict[str, Any]:
    import yaml

    config.validate()
    run_dir.mkdir(parents=True, exist_ok=True)
    config_path = run_dir / "train_config.json"
    deepspeed_path = run_dir / "deepspeed_zero2_cpu.json"
    accelerate_path = run_dir / "accelerate.yaml"
    _json_dump(config_path, asdict(config))
    _json_dump(deepspeed_path, deepspeed_config(config))
    with accelerate_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(accelerate_config(deepspeed_path), handle, sort_keys=False)
    launch = run_dir / "launch.sh"
    repo_root = Path(__file__).resolve().parents[2]
    launch.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        'RUN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"\n'
        f"REPO_ROOT={shlex.quote(str(repo_root))}\n"
        'export HF_HOME="${MAGE_FLOW_HF_HOME:-$REPO_ROOT/.hf_cache}"\n'
        'VENV="${MAGE_FLOW_VENV:-$REPO_ROOT/.venv-mage-flow}"\n'
        'exec "$VENV/bin/accelerate" launch --config_file "$RUN_DIR/accelerate.yaml" '
        '-m rwkv_lab.mage_flow_pretrain train --config "$RUN_DIR/train_config.json"\n',
        encoding="utf-8",
    )
    launch.chmod(0o755)
    receipt = {
        "schema": RUN_SCHEMA,
        "created_at": _utc_now(),
        "official_repository": OFFICIAL_REPOSITORY,
        "official_source_revision": config.official_source_revision,
        "model_id": config.model_id,
        "model_revision": config.model_revision,
        "config": str(config_path.resolve()),
        "accelerate_config": str(accelerate_path.resolve()),
        "deepspeed_config": str(deepspeed_path.resolve()),
        "launcher": str(launch.resolve()),
    }
    _json_dump(run_dir / "preparation_receipt.json", receipt)
    return receipt


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare and run full Mage-Flow-Edit-Base continued pre-training"
    )
    sub = parser.add_subparsers(dest="action", required=True)

    prep = sub.add_parser("prepare-data", help="canonicalize an image/caption JSONL")
    prep.add_argument("--input", required=True, type=Path)
    prep.add_argument("--output", required=True, type=Path)
    prep.add_argument("--data-root", type=Path, default=Path.cwd())
    prep.add_argument("--pixel-budget", type=int, default=1024 * 1024)
    prep.add_argument("--max-side", type=int, default=2048)
    prep.add_argument("--max-aspect-ratio", type=float, default=4.0)
    prep.add_argument("--exclude-manifest", action="append", type=Path, default=[])
    prep.add_argument("--workers", type=int, default=16)
    prep.add_argument("--no-verify-images", action="store_true")

    reddit = sub.add_parser(
        "prepare-reddit",
        help="freeze an append-only Reddit caption JSONL into exact train/eval splits",
    )
    reddit.add_argument("--captions", required=True, type=Path)
    reddit.add_argument("--image-root", required=True, type=Path)
    reddit.add_argument("--train-output", required=True, type=Path)
    reddit.add_argument("--eval-output", required=True, type=Path)
    reddit.add_argument("--artifact-manifest", type=Path)
    reddit.add_argument("--train-count", type=int, default=5_000)
    reddit.add_argument("--eval-count", type=int, default=128)
    reddit.add_argument("--pixel-budget", type=int, default=1024 * 1024)
    reddit.add_argument("--max-side", type=int, default=2048)
    reddit.add_argument("--max-aspect-ratio", type=float, default=4.0)
    reddit.add_argument("--workers", type=int, default=16)
    reddit.add_argument("--seed", type=int, default=42)
    reddit.add_argument("--require-clean-artifacts", action="store_true")
    reddit.add_argument("--allow-smaller-train", action="store_true")
    reddit.add_argument("--train-count-multiple", type=int, default=1)

    plan = sub.add_parser("plan", help="write a pinned resumable run directory")
    plan.add_argument("--train-manifest", required=True, type=Path)
    plan.add_argument("--eval-manifest", type=Path)
    plan.add_argument("--run-dir", required=True, type=Path)
    plan.add_argument("--output-dir", required=True, type=Path)
    plan.add_argument("--max-steps", type=int, default=16_000)
    plan.add_argument("--packed-sequence-tokens", type=int, default=10_240)
    plan.add_argument("--gradient-accumulation-steps", type=int, default=4)
    plan.add_argument("--learning-rate", type=float, default=1.0e-5)
    plan.add_argument("--warmup-steps", type=int, default=500)
    plan.add_argument("--caption-dropout", type=float, default=0.1)
    plan.add_argument("--checkpoint-every", type=int, default=500)
    plan.add_argument("--eval-every", type=int, default=500)
    plan.add_argument("--eval-packs", type=int, default=8)
    plan.add_argument("--eval-gen-every", type=int, default=100)
    plan.add_argument("--eval-gen-samples", type=int, default=4)
    plan.add_argument("--eval-gen-steps", type=int, default=30)
    plan.add_argument("--eval-gen-cfg", type=float, default=5.0)
    plan.add_argument("--no-eval-gen-step-zero", action="store_true")
    plan.add_argument("--no-eval-gen-prompt-screen", action="store_true")
    plan.add_argument("--seed", type=int, default=42)

    run = sub.add_parser("train", help="launch from a prepared train_config.json")
    run.add_argument("--config", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    if args.action == "prepare-data":
        report = prepare_manifest(
            args.input.expanduser().resolve(),
            args.output.expanduser().resolve(),
            data_root=args.data_root.expanduser().resolve(),
            pixel_budget=args.pixel_budget,
            max_side=args.max_side,
            max_aspect_ratio=args.max_aspect_ratio,
            verify_images=not args.no_verify_images,
            exclude_manifests=args.exclude_manifest,
            workers=args.workers,
        )
        print(json.dumps(report, indent=2, sort_keys=True))
        return
    if args.action == "prepare-reddit":
        report = prepare_reddit_split(
            args.captions,
            args.image_root,
            args.train_output,
            args.eval_output,
            train_count=args.train_count,
            eval_count=args.eval_count,
            seed=args.seed,
            artifact_path=args.artifact_manifest,
            pixel_budget=args.pixel_budget,
            max_side=args.max_side,
            max_aspect_ratio=args.max_aspect_ratio,
            workers=args.workers,
            require_clean_artifacts=args.require_clean_artifacts,
            allow_smaller_train=args.allow_smaller_train,
            train_count_multiple=args.train_count_multiple,
        )
        print(json.dumps(report, indent=2, sort_keys=True))
        return
    if args.action == "plan":
        config = MageFlowTrainConfig(
            train_manifest=str(args.train_manifest.expanduser().resolve()),
            eval_manifest=(
                str(args.eval_manifest.expanduser().resolve())
                if args.eval_manifest
                else None
            ),
            output_dir=str(args.output_dir.expanduser().resolve()),
            max_steps=args.max_steps,
            packed_sequence_tokens=args.packed_sequence_tokens,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            learning_rate=args.learning_rate,
            warmup_steps=args.warmup_steps,
            caption_dropout=args.caption_dropout,
            checkpoint_every=args.checkpoint_every,
            eval_every=args.eval_every,
            eval_packs=args.eval_packs,
            eval_gen_every=args.eval_gen_every,
            eval_gen_samples=args.eval_gen_samples,
            eval_gen_steps=args.eval_gen_steps,
            eval_gen_cfg=args.eval_gen_cfg,
            eval_gen_step_zero=not args.no_eval_gen_step_zero,
            eval_gen_screen_prompts=not args.no_eval_gen_prompt_screen,
            seed=args.seed,
        )
        print(
            json.dumps(
                prepare_run(config, args.run_dir.expanduser().resolve()),
                indent=2,
                sort_keys=True,
            )
        )
        return
    train(MageFlowTrainConfig.from_path(args.config.expanduser().resolve()))


if __name__ == "__main__":
    main()
