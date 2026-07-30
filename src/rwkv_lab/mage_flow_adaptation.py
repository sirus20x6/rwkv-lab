"""CPU-testable foundations for the staged Mage-Flow adaptation project.

This module deliberately stops before model inference or training.  It provides
the durable contracts needed by Stages 0--2:

* a fixed generation/editing benchmark suite;
* safetensors layout and state-key compatibility reports;
* explicit general/photo/animation manifests with a dedicated uncaptioned
  condition that cannot be confused with the classifier-free-guidance null;
* deterministic, source-balanced, homogeneous-domain microbatches;
* zero-output residual image-FFN experts injected into late Mage-Flow blocks;
* independently saveable photo and animation expert checkpoints.

The existing :mod:`rwkv_lab.mage_flow_pretrain` path remains the full-backbone
Mage-Flow-Edit continued-pretraining workflow.  It is not changed or reused as
an appearance-expert trainer.
"""

# ruff: noqa: ISC004

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

MAGE_FLOW_BASE_ID = "microsoft/Mage-Flow-Base"
MAGE_FLOW_BASE_REVISION = "59a9cfd58cf6ecef28245852c6bdace3f12428a2"
MAGE_SOURCE_REVISION = "ef932e2cc3e94bb026d937a6cffae65492adc0fb"

BENCHMARK_SCHEMA = "rwkv-lab.mage-flow-benchmark.v1"
DOMAIN_MANIFEST_SCHEMA = "rwkv-lab.mage-flow-domain-data.v1"
EXPERT_SCHEMA = "rwkv-lab.mage-flow-appearance-expert.v1"

DOMAINS = ("general", "photo", "animation")
EXPERT_DOMAINS = ("photo", "animation")
UNCAPTIONED_IMAGE_CONDITION = "<uncaptioned-image>"
CFG_NULL_CONDITION = "<cfg-null>"
DEFAULT_DOMAIN_WEIGHTS = {"general": 0.10, "photo": 0.45, "animation": 0.45}


@dataclass(frozen=True)
class BenchmarkCase:
    case_id: str
    category: str
    prompt: str
    seed: int
    width: int
    height: int
    mode: str = "generation"
    reference_fixture: str | None = None


_BENCHMARK_PROMPTS: tuple[tuple[str, str, int, int], ...] = (
    (
        "photorealistic_people",
        "A candid documentary photograph of three generations cooking together "
        "in a small sunlit kitchen, natural skin texture and hands visible.",
        768,
        1024,
    ),
    (
        "photorealistic_people",
        "A studio portrait of a freckled violin maker holding a half-finished "
        "instrument, realistic hair, hands, wood grain, and softbox reflections.",
        1024,
        768,
    ),
    (
        "environments",
        "A rain-soaked elevated railway crossing a dense city at blue hour, wet "
        "asphalt reflections, pedestrians at several depths, documentary photo.",
        1344,
        768,
    ),
    (
        "environments",
        "An alpine research station after fresh snowfall under hard morning "
        "light, wind-carved drifts and distant atmospheric perspective.",
        1344,
        768,
    ),
    (
        "products_machinery",
        "A photorealistic exploded-view arrangement of a mechanical wristwatch "
        "on a workbench, coherent gears, screws, tweezers, and engraved metal.",
        1024,
        1024,
    ),
    (
        "products_machinery",
        "Commercial product photograph of a translucent cordless drill with its "
        "motor, gearbox, battery cells, and trigger visible in correct alignment.",
        1024,
        1024,
    ),
    (
        "animals",
        "A wet border collie leaping from a forest stream, individual water "
        "droplets frozen by a fast shutter and anatomically correct legs.",
        1024,
        768,
    ),
    (
        "animals",
        "A macro wildlife photograph of a jumping spider on a curled green leaf, "
        "accurate eyes, fine hairs, shallow depth of field, no fantasy features.",
        1024,
        768,
    ),
    (
        "anime_characters",
        "Anime key visual of two bicycle couriers racing downhill through a "
        "coastal town, distinct silhouettes, expressive faces, dynamic staging.",
        1344,
        768,
    ),
    (
        "anime_characters",
        "A quiet anime character sheet for an elderly botanist, front, side, and "
        "three-quarter views with consistent clothing and facial proportions.",
        1344,
        768,
    ),
    (
        "western_animation",
        "A polished western-animation frame of a nervous young dragon presenting "
        "a burnt pie at a village contest, strong poses and readable expressions.",
        1344,
        768,
    ),
    (
        "western_animation",
        "Graphic television-animation lineup of four detectives with clearly "
        "different body shapes, outfits, attitudes, and clean flat colors.",
        1344,
        768,
    ),
    (
        "painterly_illustration",
        "A painterly editorial illustration of a night-shift nurse riding the "
        "last bus home, layered gouache texture and restrained color harmony.",
        1024,
        1024,
    ),
    (
        "painterly_illustration",
        "A richly brushed fantasy landscape where terraced farms spiral around "
        "an ancient observatory, coherent scale and warm late-afternoon light.",
        1344,
        768,
    ),
    (
        "line_art",
        "Precise black ink line art of a repair cafe with six people fixing a "
        "lamp, radio, bicycle, and toaster, clean overlaps and no gray shading.",
        1344,
        768,
    ),
    (
        "line_art",
        "Technical pen-and-ink cutaway of a compact greenhouse, consistent line "
        "weight, legible structure, irrigation pipes, shelves, and open vents.",
        1024,
        1024,
    ),
    (
        "typography",
        'A realistic enamel shop sign reading exactly "NORTH STAR REPAIRS" with '
        'a smaller line reading "CLOCKS • RADIOS • CAMERAS", front-facing.',
        1344,
        768,
    ),
    (
        "typography",
        'A clean illustrated festival poster reading exactly "RIVERLIGHT 2026" '
        'and "AUGUST 14–16", with all lettering large and unobstructed.',
        768,
        1024,
    ),
    (
        "multi_object_composition",
        "A catalog photograph of twelve distinct camping tools arranged in a "
        "four-by-three grid, one object per cell, consistent scale and lighting.",
        1024,
        1024,
    ),
    (
        "multi_object_composition",
        "An animated banquet scene with eight distinct guests simultaneously "
        "passing dishes, each pair of hands and each object clearly attributable.",
        1344,
        768,
    ),
    (
        "spatial_relationships",
        "A red cube behind a glass sphere, a brass cone left of both, and a small "
        "blue cylinder balanced on the cube, photographed at eye level.",
        1024,
        768,
    ),
    (
        "spatial_relationships",
        "A child under a footbridge points toward a kite above the bridge while "
        "a cyclist crosses from right to left behind the child.",
        1344,
        768,
    ),
)

_EDIT_CASES: tuple[tuple[str, str, str], ...] = (
    (
        "editing_local",
        "Replace only the table lamp with a green banker lamp; preserve every "
        "other object, the camera, and the lighting.",
        "edit_room_v1.png",
    ),
    (
        "editing_background",
        "Move the subject to a snowy railway platform at dusk while preserving "
        "identity, pose, clothing, and framing.",
        "edit_person_v1.png",
    ),
    (
        "editing_pose",
        "Make the character wave with their left hand while preserving identity, "
        "costume design, background, and drawing style.",
        "edit_character_v1.png",
    ),
    (
        "reference_consistency",
        "Show the referenced character repairing a bicycle, retaining the exact "
        "face, hair, clothing motifs, proportions, and illustration style.",
        "reference_character_v1.png",
    ),
)


def benchmark_suite() -> list[BenchmarkCase]:
    """Return the immutable v1 prompt/seed suite."""
    cases = [
        BenchmarkCase(
            case_id=f"{category}-{index % 2 + 1:02d}",
            category=category,
            prompt=prompt,
            seed=20_260_100 + index,
            width=width,
            height=height,
        )
        for index, (category, prompt, width, height) in enumerate(_BENCHMARK_PROMPTS)
    ]
    for index, (category, prompt, fixture) in enumerate(_EDIT_CASES):
        cases.append(
            BenchmarkCase(
                case_id=f"{category}-01",
                category=category,
                prompt=prompt,
                seed=20_260_200 + index,
                width=1024,
                height=1024,
                mode="editing",
                reference_fixture=fixture,
            )
        )
    return cases


def write_benchmark_suite(path: Path) -> dict[str, Any]:
    """Write a fixed suite manifest without requiring model weights."""
    payload = {
        "schema": BENCHMARK_SCHEMA,
        "base_model": MAGE_FLOW_BASE_ID,
        "base_revision": MAGE_FLOW_BASE_REVISION,
        "source_revision": MAGE_SOURCE_REVISION,
        "cases": [asdict(case) for case in benchmark_suite()],
        "vae_categories": [
            "photographs",
            "skin_hair",
            "small_faces",
            "thin_mechanical_geometry",
            "text_signs",
            "anime_line_art",
            "flat_color_animation",
            "gradients_painterly_textures",
            "high_frequency_patterns",
            "consecutive_video_frames",
        ],
        "vae_metrics": [
            "psnr",
            "ssim",
            "lpips",
            "edge_reconstruction_error",
            "ocr_accuracy",
            "temporal_reconstruction_jitter",
        ],
    }
    _atomic_json(path, payload)
    return payload


def checkpoint_key_report(
    expected_keys: Iterable[str], checkpoint_keys: Iterable[str]
) -> dict[str, Any]:
    """Report state compatibility without mutating or allocating a model."""
    expected, actual = set(expected_keys), set(checkpoint_keys)
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    return {
        "expected_key_count": len(expected),
        "checkpoint_key_count": len(actual),
        "missing_keys": missing,
        "unexpected_keys": unexpected,
        "compatible": not missing and not unexpected,
    }


def inspect_safetensors_layout(paths: Sequence[Path]) -> dict[str, Any]:
    """Inspect checkpoint tensor names, shapes, and dtypes on CPU."""
    from safetensors import safe_open

    tensors: dict[str, dict[str, Any]] = {}
    duplicates: list[str] = []
    files = []
    for source in paths:
        source = source.expanduser().resolve()
        with safe_open(str(source), framework="pt", device="cpu") as handle:
            keys = sorted(handle.keys())
            files.append({"path": str(source), "tensor_count": len(keys)})
            for key in keys:
                if key in tensors:
                    duplicates.append(key)
                    continue
                view = handle.get_slice(key)
                tensors[key] = {
                    "shape": list(view.get_shape()),
                    "dtype": str(view.get_dtype()),
                    "file": str(source),
                }
    return {
        "files": files,
        "tensor_count": len(tensors),
        "duplicate_keys": sorted(set(duplicates)),
        "tensors": tensors,
    }


def _image_batch(tensor: torch.Tensor, name: str) -> torch.Tensor:
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 4:
        raise ValueError(f"{name} must have shape [N,C,H,W] or [C,H,W]")
    if tensor.shape[1] not in {1, 3, 4}:
        raise ValueError(f"{name} must have 1, 3, or 4 channels")
    tensor = tensor.detach().to(device="cpu", dtype=torch.float32)
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{name} contains non-finite values")
    if tensor.min().item() < 0 or tensor.max().item() > 1:
        raise ValueError(f"{name} values must be normalized to [0, 1]")
    return tensor


def reconstruction_psnr(reference: torch.Tensor, reconstructed: torch.Tensor) -> float:
    """Peak signal-to-noise ratio for normalized image batches."""
    reference = _image_batch(reference, "reference")
    reconstructed = _image_batch(reconstructed, "reconstructed")
    if reference.shape != reconstructed.shape:
        raise ValueError("reference and reconstructed shapes differ")
    mse = (reference - reconstructed).square().mean()
    if mse.item() == 0:
        return float("inf")
    return float((-10.0 * torch.log10(mse)).item())


def reconstruction_ssim(
    reference: torch.Tensor,
    reconstructed: torch.Tensor,
    *,
    window_size: int = 11,
) -> float:
    """Dependency-free local SSIM for normalized image batches."""
    from torch.nn import functional

    reference = _image_batch(reference, "reference")
    reconstructed = _image_batch(reconstructed, "reconstructed")
    if reference.shape != reconstructed.shape:
        raise ValueError("reference and reconstructed shapes differ")
    limit = min(window_size, reference.shape[-2], reference.shape[-1])
    size = limit if limit % 2 else limit - 1
    if size < 1:
        raise ValueError("images have invalid spatial geometry")
    padding = size // 2

    def average(value):
        return functional.avg_pool2d(value, kernel_size=size, stride=1, padding=padding)

    mu_x, mu_y = average(reference), average(reconstructed)
    var_x = average(reference.square()) - mu_x.square()
    var_y = average(reconstructed.square()) - mu_y.square()
    covariance = average(reference * reconstructed) - mu_x * mu_y
    c1, c2 = 0.01**2, 0.03**2
    score = ((2 * mu_x * mu_y + c1) * (2 * covariance + c2)) / (
        (mu_x.square() + mu_y.square() + c1) * (var_x + var_y + c2)
    )
    return float(score.mean().clamp(-1, 1).item())


def edge_reconstruction_error(
    reference: torch.Tensor, reconstructed: torch.Tensor
) -> float:
    """Mean absolute Sobel-gradient error on normalized images."""
    from torch.nn import functional

    reference = _image_batch(reference, "reference")
    reconstructed = _image_batch(reconstructed, "reconstructed")
    if reference.shape != reconstructed.shape:
        raise ValueError("reference and reconstructed shapes differ")
    channels = reference.shape[1]
    kernels = torch.tensor(
        [
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
        ],
        dtype=reference.dtype,
    ).unsqueeze(1)
    kernels = kernels.repeat(channels, 1, 1, 1)

    def gradients(value):
        result = functional.conv2d(value, kernels, padding=1, groups=channels)
        return result.view(value.shape[0], channels, 2, *value.shape[-2:])

    return float((gradients(reference) - gradients(reconstructed)).abs().mean().item())


def temporal_reconstruction_jitter(
    reference_frames: torch.Tensor, reconstructed_frames: torch.Tensor
) -> float:
    """Frame-to-frame variation introduced by reconstruction error."""
    reference = _image_batch(reference_frames, "reference_frames")
    reconstructed = _image_batch(reconstructed_frames, "reconstructed_frames")
    if reference.shape != reconstructed.shape:
        raise ValueError("reference and reconstructed frame shapes differ")
    if reference.shape[0] < 2:
        raise ValueError("temporal jitter requires at least two frames")
    residual = reconstructed - reference
    return float((residual[1:] - residual[:-1]).abs().mean().item())


def normalized_ocr_accuracy(
    reference_texts: Sequence[str], reconstructed_texts: Sequence[str]
) -> float:
    """Normalized character edit similarity for OCR outputs."""
    if len(reference_texts) != len(reconstructed_texts) or not reference_texts:
        raise ValueError("OCR text sequences must be non-empty and equally sized")

    def normalize(value: str) -> str:
        return re.sub(r"\s+", " ", value.strip().casefold())

    def distance(left: str, right: str) -> int:
        previous = list(range(len(right) + 1))
        for left_index, left_character in enumerate(left, 1):
            current = [left_index]
            for right_index, right_character in enumerate(right, 1):
                current.append(
                    min(
                        current[-1] + 1,
                        previous[right_index] + 1,
                        previous[right_index - 1] + (left_character != right_character),
                    )
                )
            previous = current
        return previous[-1]

    scores = []
    for expected, observed in zip(reference_texts, reconstructed_texts, strict=True):
        expected, observed = normalize(expected), normalize(observed)
        denominator = max(1, len(expected), len(observed))
        scores.append(1.0 - distance(expected, observed) / denominator)
    return sum(scores) / len(scores)


def vae_reconstruction_report(
    reference: torch.Tensor,
    reconstructed: torch.Tensor,
    *,
    reference_texts: Sequence[str] | None = None,
    reconstructed_texts: Sequence[str] | None = None,
    lpips_fn: Any | None = None,
    temporal: bool = False,
) -> dict[str, float]:
    """Compute the Stage 0 reconstruction metrics from decoded tensors."""
    checked_reference = _image_batch(reference, "reference")
    checked_reconstructed = _image_batch(reconstructed, "reconstructed")
    report = {
        "psnr": reconstruction_psnr(checked_reference, checked_reconstructed),
        "ssim": reconstruction_ssim(checked_reference, checked_reconstructed),
        "edge_reconstruction_error": edge_reconstruction_error(
            checked_reference, checked_reconstructed
        ),
    }
    if temporal:
        report["temporal_reconstruction_jitter"] = temporal_reconstruction_jitter(
            checked_reference, checked_reconstructed
        )
    if reference_texts is not None or reconstructed_texts is not None:
        if reference_texts is None or reconstructed_texts is None:
            raise ValueError("both OCR text sequences are required")
        report["ocr_accuracy"] = normalized_ocr_accuracy(
            reference_texts, reconstructed_texts
        )
    if lpips_fn is not None:
        # LPIPS implementations conventionally consume [-1, 1] RGB tensors.
        with torch.no_grad():
            value = lpips_fn(
                checked_reference[:, :3] * 2 - 1,
                checked_reconstructed[:, :3] * 2 - 1,
            )
        report["lpips"] = float(torch.as_tensor(value).float().mean().item())
    return report


def _atomic_json(path: Path, payload: Any) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _text(value: Any) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None


def _join_tags(value: Any) -> str | None:
    if isinstance(value, str):
        return value.strip() or None
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        items = [str(item).strip() for item in value if str(item).strip()]
        return ", ".join(items) or None
    return None


def select_caption(row: Mapping[str, Any]) -> tuple[str, str, bool]:
    """Apply the explicit caption hierarchy and preserve an uncaptioned sentinel."""
    for field, kind in (
        ("curated_caption", "curated"),
        ("human_caption", "human"),
    ):
        if value := _text(row.get(field)):
            return value, kind, True

    provenance = (
        str(row.get("caption_provenance") or row.get("caption_kind") or "")
        .strip()
        .lower()
    )
    generic = _text(row.get("caption")) or _text(row.get("text"))
    if generic and provenance in {"curated", "human"}:
        return generic, provenance, True
    if value := _text(row.get("dense_caption")):
        return value, "dense_automatic", True
    if generic:
        return generic, provenance or "caption_unknown", True
    if value := _text(row.get("short_caption")):
        return value, "short_automatic", True

    tags = _join_tags(row.get("tags"))
    entities = _join_tags(row.get("entities"))
    combined = ", ".join(item for item in (tags, entities) if item)
    if combined:
        return combined, "tags_entities", True
    return UNCAPTIONED_IMAGE_CONDITION, "uncaptioned_image", False


def cfg_null_condition() -> dict[str, Any]:
    """Return the true CFG-null case, kept distinct from missing captions."""
    return {
        "conditioning_text": CFG_NULL_CONDITION,
        "conditioning_kind": "cfg_null",
        "is_captioned": False,
        "domain": "general",
        "training_scope": "cfg",
    }


def canonical_domain_row(
    row: Mapping[str, Any],
    *,
    data_root: Path,
    pixel_budget: int = 512 * 512,
    max_side: int = 2048,
    max_aspect_ratio: float = 4.0,
    verify_image: bool = True,
) -> tuple[dict[str, Any] | None, str | None]:
    """Normalize one explicitly labelled image-domain sample."""
    from rwkv_lab.mage_flow_pretrain import latent_tokens, native_size

    domain = str(row.get("domain") or "").strip().lower()
    if domain not in DOMAINS:
        return None, "invalid_or_missing_domain"
    image_value = _text(row.get("image")) or _text(row.get("image_path"))
    if not image_value:
        return None, "missing_image"
    image = Path(image_value).expanduser()
    if not image.is_absolute():
        image = data_root / image
    image = image.resolve()
    if not image.is_file():
        return None, "missing_image_file"

    width, height = int(row.get("width") or 0), int(row.get("height") or 0)
    if verify_image or width < 1 or height < 1:
        try:
            from PIL import Image, ImageOps

            with Image.open(image) as decoded:
                decoded = ImageOps.exif_transpose(decoded)
                width, height = decoded.size
                if verify_image:
                    decoded.load()
        except (OSError, SyntaxError, ValueError):
            return None, "decode_error"
    try:
        train_width, train_height = native_size(
            width,
            height,
            pixel_budget=pixel_budget,
            max_side=max_side,
            max_aspect_ratio=max_aspect_ratio,
        )
    except ValueError:
        return None, "geometry_rejected"

    caption, caption_kind, is_captioned = select_caption(row)
    if not is_captioned and domain == "general":
        return None, "uncaptioned_general_has_no_expert"
    identity = _text(row.get("image_sha256")) or _text(row.get("image_id"))
    if not identity:
        identity = _file_sha256(image)
    source = _text(row.get("source")) or "unknown"
    result = {
        "schema": DOMAIN_MANIFEST_SCHEMA,
        "image": str(image),
        "image_id": identity,
        "domain": domain,
        "source": source,
        "caption": caption,
        "conditioning_text": caption,
        "conditioning_kind": caption_kind,
        "is_captioned": is_captioned,
        "training_scope": (
            "expert_and_selected_shared" if is_captioned else "expert_only"
        ),
        "width": width,
        "height": height,
        "train_width": train_width,
        "train_height": train_height,
        "latent_tokens": latent_tokens(train_width, train_height),
    }
    for field in (
        "quality_score",
        "aesthetic_score",
        "caption_model",
        "caption_provenance",
        "license",
        "cache_key",
    ):
        if field in row:
            result[field] = row[field]
    return result, None


def prepare_domain_manifest(
    input_path: Path,
    output_path: Path,
    *,
    data_root: Path,
    pixel_budget: int = 512 * 512,
    max_side: int = 2048,
    max_aspect_ratio: float = 4.0,
    verify_images: bool = True,
) -> dict[str, Any]:
    """Prepare an atomic domain manifest and reject exact duplicate images."""
    counters: Counter[str] = Counter()
    seen: dict[str, str] = {}
    rows: list[dict[str, Any]] = []
    with input_path.expanduser().resolve().open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            counters["input"] += 1
            try:
                raw = json.loads(line)
            except json.JSONDecodeError:
                counters["invalid_json"] += 1
                continue
            canonical, reason = canonical_domain_row(
                raw,
                data_root=data_root.expanduser().resolve(),
                pixel_budget=pixel_budget,
                max_side=max_side,
                max_aspect_ratio=max_aspect_ratio,
                verify_image=verify_images,
            )
            if canonical is None:
                counters[reason or "rejected"] += 1
                continue
            identity = str(canonical["image_id"])
            if identity in seen:
                if seen[identity] != canonical["domain"]:
                    counters["cross_domain_duplicate"] += 1
                else:
                    counters["duplicate"] += 1
                continue
            seen[identity] = str(canonical["domain"])
            rows.append(canonical)
            counters["output"] += 1

    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(output_path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(temporary, output_path)
    audit = audit_domain_rows(rows)
    audit["input_cross_domain_duplicate_count"] = counters["cross_domain_duplicate"]
    audit["passed"] = (
        audit["passed"] and audit["input_cross_domain_duplicate_count"] == 0
    )
    report = {
        "schema": DOMAIN_MANIFEST_SCHEMA,
        "input": str(input_path.expanduser().resolve()),
        "output": str(output_path),
        "counts": dict(sorted(counters.items())),
        "audit": audit,
    }
    _atomic_json(output_path.with_suffix(output_path.suffix + ".report.json"), report)
    return report


def _normalized_caption(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value).strip().casefold())


def audit_domain_rows(
    rows: Sequence[Mapping[str, Any]], *, max_uncaptioned_fraction: float = 0.15
) -> dict[str, Any]:
    """Audit duplicates, cross-domain leakage, source balance, and captions."""
    domain_counts: Counter[str] = Counter()
    kind_counts: Counter[str] = Counter()
    source_counts: dict[str, Counter[str]] = defaultdict(Counter)
    identities: dict[str, set[str]] = defaultdict(set)
    identity_counts: Counter[str] = Counter()
    captions: dict[str, set[str]] = defaultdict(set)
    invalid_domains = 0
    uncaptioned = 0
    for row in rows:
        domain = str(row.get("domain", ""))
        if domain not in DOMAINS:
            invalid_domains += 1
            continue
        domain_counts[domain] += 1
        kind_counts[str(row.get("conditioning_kind", "missing"))] += 1
        source_counts[domain][str(row.get("source", "unknown"))] += 1
        identity = str(row.get("image_id", ""))
        identities[identity].add(domain)
        identity_counts[identity] += 1
        caption = _normalized_caption(row.get("caption", ""))
        if caption and caption != UNCAPTIONED_IMAGE_CONDITION:
            captions[caption].add(domain)
        if not bool(row.get("is_captioned", False)):
            uncaptioned += 1

    cross_domain_images = sorted(
        identity
        for identity, assigned in identities.items()
        if identity and len(assigned) > 1
    )
    missing_identities = identity_counts.get("", 0)
    duplicate_rows = sum(
        count - 1
        for identity, count in identity_counts.items()
        if identity and count > 1
    )
    cross_domain_captions = sum(len(domains) > 1 for domains in captions.values())
    uncaptioned_fraction = uncaptioned / max(1, len(rows))
    dominance = {}
    for domain, counts in source_counts.items():
        total = sum(counts.values())
        source, count = counts.most_common(1)[0]
        dominance[domain] = {
            "source": source,
            "fraction": count / total,
            "counts": dict(sorted(counts.items())),
        }
    warnings = []
    if cross_domain_captions:
        warnings.append("normalized captions occur in more than one domain")
    for domain, item in dominance.items():
        if item["fraction"] > 0.80 and domain_counts[domain] > 1:
            warnings.append(f"{domain} source {item['source']} exceeds 80%")
    passed = (
        not invalid_domains
        and not missing_identities
        and not cross_domain_images
        and duplicate_rows == 0
        and uncaptioned_fraction <= max_uncaptioned_fraction
    )
    return {
        "passed": passed,
        "row_count": len(rows),
        "invalid_domain_count": invalid_domains,
        "missing_image_id_count": missing_identities,
        "domain_counts": dict(sorted(domain_counts.items())),
        "conditioning_kind_counts": dict(sorted(kind_counts.items())),
        "duplicate_row_count": duplicate_rows,
        "cross_domain_image_count": len(cross_domain_images),
        "cross_domain_image_ids": cross_domain_images[:100],
        "cross_domain_normalized_caption_count": cross_domain_captions,
        "uncaptioned_count": uncaptioned,
        "uncaptioned_fraction": uncaptioned_fraction,
        "max_uncaptioned_fraction": max_uncaptioned_fraction,
        "source_dominance": dominance,
        "warnings": warnings,
    }


def load_domain_manifest(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.expanduser().resolve().open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            required = {
                "image",
                "image_id",
                "domain",
                "source",
                "conditioning_text",
                "conditioning_kind",
                "is_captioned",
                "training_scope",
                "train_width",
                "train_height",
                "latent_tokens",
            }
            missing = sorted(required - row.keys())
            if missing:
                raise ValueError(f"{path}:{line_number} missing {missing}")
            if row["domain"] not in DOMAINS:
                raise ValueError(f"{path}:{line_number} invalid domain")
            if (
                not row["is_captioned"]
                and row["conditioning_text"] != UNCAPTIONED_IMAGE_CONDITION
            ):
                raise ValueError(
                    f"{path}:{line_number} missing dedicated uncaptioned condition"
                )
            rows.append(row)
    if not rows:
        raise ValueError(f"empty domain manifest: {path}")
    return rows


def _domain_batch_counts(
    batch_count: int, weights: Mapping[str, float]
) -> Counter[str]:
    if batch_count < 1:
        raise ValueError("batch_count must be positive")
    if set(weights) - set(DOMAINS):
        raise ValueError("unsupported domain in weights")
    total = sum(float(value) for value in weights.values())
    if total <= 0 or any(float(value) < 0 for value in weights.values()):
        raise ValueError("domain weights must be non-negative with a positive sum")
    quotas = {
        domain: batch_count * float(value) / total for domain, value in weights.items()
    }
    counts = Counter({domain: math.floor(quota) for domain, quota in quotas.items()})
    remaining = batch_count - sum(counts.values())
    order = sorted(
        quotas,
        key=lambda domain: (quotas[domain] - counts[domain], domain),
        reverse=True,
    )
    for domain in order[:remaining]:
        counts[domain] += 1
    return counts


def homogeneous_domain_batches(
    rows: Sequence[Mapping[str, Any]],
    *,
    batch_size: int,
    seed: int,
    epoch: int,
    batch_count: int | None = None,
    weights: Mapping[str, float] = DEFAULT_DOMAIN_WEIGHTS,
) -> list[list[int]]:
    """Build deterministic hard-routed batches, balanced by domain and source.

    Sampling cycles independently through every source.  It may repeat examples
    when a small source must contribute to the requested balanced epoch.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    batch_count = batch_count or math.ceil(len(rows) / batch_size)
    counts = _domain_batch_counts(batch_count, weights)
    buckets: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    for index, row in enumerate(rows):
        domain = str(row.get("domain", ""))
        if domain not in DOMAINS:
            raise ValueError(f"row {index} has invalid domain {domain!r}")
        buckets[domain][str(row.get("source", "unknown"))].append(index)
    for domain, count in counts.items():
        if count and not buckets.get(domain):
            raise ValueError(f"no rows available for weighted domain {domain}")

    rng = random.Random(seed + epoch * 1_000_003)
    domain_schedule = [domain for domain, count in counts.items() for _ in range(count)]
    rng.shuffle(domain_schedule)

    state: dict[tuple[str, str], tuple[list[int], int]] = {}
    source_offsets: Counter[str] = Counter()
    for domain, sources in buckets.items():
        for source, indices in sources.items():
            shuffled = list(indices)
            rng.shuffle(shuffled)
            state[(domain, source)] = (shuffled, 0)

    def next_index(domain: str) -> int:
        source_names = sorted(buckets[domain])
        source = source_names[source_offsets[domain] % len(source_names)]
        source_offsets[domain] += 1
        values, offset = state[(domain, source)]
        if offset >= len(values):
            rng.shuffle(values)
            offset = 0
        result = values[offset]
        state[(domain, source)] = (values, offset + 1)
        return result

    return [
        [next_index(domain) for _ in range(batch_size)] for domain in domain_schedule
    ]


def assert_homogeneous_batch(
    rows: Sequence[Mapping[str, Any]], indices: Sequence[int]
) -> str:
    domains = {str(rows[index]["domain"]) for index in indices}
    if len(domains) != 1:
        raise ValueError(f"mixed-domain microbatch: {sorted(domains)}")
    domain = next(iter(domains))
    if domain not in DOMAINS:
        raise ValueError(f"invalid batch domain {domain!r}")
    return domain


class ResidualFFNExpert(nn.Module):
    """Narrow nonlinear correction whose output projection starts at zero."""

    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU(approximate="tanh")
        self.fc2 = nn.Linear(hidden_dim, dim)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, inputs):
        return self.fc2(self.act(self.fc1(inputs)))


class RoutedImageFFN(nn.Module):
    """Released image FFN plus one hard-routed residual expert."""

    def __init__(
        self,
        shared_ffn,
        *,
        dim: int,
        hidden_dim: int,
        domains: Sequence[str] = EXPERT_DOMAINS,
    ):
        super().__init__()
        self.shared_ffn = shared_ffn
        self.experts = nn.ModuleDict(
            {domain: ResidualFFNExpert(dim, hidden_dim) for domain in domains}
        )
        self.scales = nn.ParameterDict(
            {
                domain: nn.Parameter(torch.ones((), dtype=torch.float32))
                for domain in domains
            }
        )
        self.active_domain = "general"
        self.dim = dim
        self.hidden_dim = hidden_dim

    def set_domain(self, domain: str) -> None:
        if domain not in DOMAINS:
            raise ValueError(f"unsupported Mage-Flow domain {domain!r}")
        self.active_domain = domain

    def forward(self, inputs, *args, **kwargs):
        shared = self.shared_ffn(inputs, *args, **kwargs)
        if self.active_domain == "general":
            return shared
        delta = self.experts[self.active_domain](inputs)
        scale = self.scales[self.active_domain].to(
            device=delta.device, dtype=delta.dtype
        )
        return shared + scale * delta


@dataclass(frozen=True)
class AppearanceExpertConfig:
    block_indices: tuple[int, ...]
    model_dim: int
    shared_ffn_multiplier: float
    expert_width_fraction: float
    expert_hidden_dim: int
    expert_hidden_alignment: int
    requested_parameter_fraction: float | None
    actual_parameter_fraction: float
    expert_parameter_count: int
    base_parameter_count: int
    combined_shared_fraction: float
    domains: tuple[str, ...] = EXPERT_DOMAINS


class AppearanceExpertController:
    """Route and serialize the wrappers registered in a Mage-Flow transformer."""

    def __init__(
        self,
        wrappers: Mapping[int, Any],
        config: AppearanceExpertConfig,
        *,
        base_model: str,
        base_revision: str,
    ):
        self.wrappers = dict(sorted(wrappers.items()))
        self.config = config
        self.base_model = base_model
        self.base_revision = base_revision

    def set_domain(self, domain: str) -> None:
        if domain not in DOMAINS:
            raise ValueError(f"unsupported Mage-Flow domain {domain!r}")
        for wrapper in self.wrappers.values():
            wrapper.set_domain(domain)

    @contextmanager
    def route(self, domain: str) -> Iterator[None]:
        previous = {
            index: wrapper.active_domain for index, wrapper in self.wrappers.items()
        }
        self.set_domain(domain)
        try:
            yield
        finally:
            for index, wrapper in self.wrappers.items():
                wrapper.set_domain(previous[index])

    def parameters(self, domain: str | None = None) -> Iterator[Any]:
        selected = EXPERT_DOMAINS if domain is None else (domain,)
        for name in selected:
            if name not in self.config.domains:
                raise ValueError(f"expert domain {name!r} is not installed")
            for wrapper in self.wrappers.values():
                yield from wrapper.experts[name].parameters()
                yield wrapper.scales[name]

    def parameter_count(self, domain: str) -> int:
        return sum(parameter.numel() for parameter in self.parameters(domain))


def inject_appearance_experts(
    transformer,
    *,
    final_block_fraction: float = 2 / 3,
    expert_parameter_fraction: float | None = 0.15,
    expert_width_fraction: float | None = None,
    shared_ffn_multiplier: float = 4.0,
    expert_hidden_alignment: int = 128,
    block_indices: Sequence[int] | None = None,
    base_model: str = MAGE_FLOW_BASE_ID,
    base_revision: str = MAGE_FLOW_BASE_REVISION,
    expert_dtype: torch.dtype | None = None,
) -> AppearanceExpertController:
    """Wrap late image FFNs while keeping the released FFNs intact."""
    blocks = transformer.transformer_blocks
    depth = len(blocks)
    if depth < 1:
        raise ValueError("transformer has no blocks")
    if not 0 < final_block_fraction <= 1:
        raise ValueError("final_block_fraction must be in (0, 1]")
    if (expert_parameter_fraction is None) == (expert_width_fraction is None):
        raise ValueError(
            "set exactly one of expert_parameter_fraction or expert_width_fraction"
        )
    if expert_parameter_fraction is not None and not 0 < expert_parameter_fraction <= 1:
        raise ValueError("expert_parameter_fraction must be in (0, 1]")
    if expert_width_fraction is not None and expert_width_fraction <= 0:
        raise ValueError("expert_width_fraction must be positive")
    if expert_hidden_alignment < 1:
        raise ValueError("expert_hidden_alignment must be positive")
    if block_indices is None:
        selected_count = max(1, round(depth * final_block_fraction))
        block_indices = tuple(range(depth - selected_count, depth))
    else:
        block_indices = tuple(sorted({int(index) for index in block_indices}))
    if not block_indices or min(block_indices) < 0 or max(block_indices) >= depth:
        raise ValueError("expert block index is out of range")

    dim = int(getattr(transformer, "inner_dim", getattr(blocks[0], "dim", 0)))
    if dim < 1:
        raise ValueError("cannot infer Mage-Flow transformer width")
    base_parameter_count = sum(
        parameter.numel() for parameter in transformer.parameters()
    )
    shared_hidden = round(dim * shared_ffn_multiplier)
    if expert_parameter_fraction is not None:
        target_parameters = base_parameter_count * expert_parameter_fraction
        parameters_per_block = target_parameters / len(block_indices)
        raw_hidden = (parameters_per_block - dim - 1) / (2 * dim + 1)
        if raw_hidden < 1:
            raise ValueError("requested expert parameter fraction is too small")
        hidden_dim = max(
            expert_hidden_alignment,
            round(raw_hidden / expert_hidden_alignment) * expert_hidden_alignment,
        )
    else:
        hidden_dim = max(1, round(shared_hidden * float(expert_width_fraction)))
    actual_width_fraction = hidden_dim / shared_hidden
    parameters_per_block = 2 * dim * hidden_dim + hidden_dim + dim + 1
    expert_parameter_count = len(block_indices) * parameters_per_block
    actual_parameter_fraction = expert_parameter_count / base_parameter_count
    combined_shared_fraction = base_parameter_count / (
        base_parameter_count + len(EXPERT_DOMAINS) * expert_parameter_count
    )
    wrappers = {}
    for index in block_indices:
        block = blocks[index]
        if hasattr(block.img_mlp, "experts") and hasattr(block.img_mlp, "shared_ffn"):
            raise ValueError(f"block {index} already has appearance experts")
        wrapper = RoutedImageFFN(
            block.img_mlp,
            dim=dim,
            hidden_dim=hidden_dim,
            domains=EXPERT_DOMAINS,
        )
        reference = next(block.img_mlp.parameters(), None)
        if reference is not None:
            wrapper.experts.to(
                device=reference.device,
                dtype=expert_dtype or reference.dtype,
            )
            wrapper.scales.to(device=reference.device)
        block.img_mlp = wrapper
        wrappers[index] = wrapper
    config = AppearanceExpertConfig(
        block_indices=tuple(block_indices),
        model_dim=dim,
        shared_ffn_multiplier=shared_ffn_multiplier,
        expert_width_fraction=actual_width_fraction,
        expert_hidden_dim=hidden_dim,
        expert_hidden_alignment=expert_hidden_alignment,
        requested_parameter_fraction=expert_parameter_fraction,
        actual_parameter_fraction=actual_parameter_fraction,
        expert_parameter_count=expert_parameter_count,
        base_parameter_count=base_parameter_count,
        combined_shared_fraction=combined_shared_fraction,
    )
    return AppearanceExpertController(
        wrappers,
        config,
        base_model=base_model,
        base_revision=base_revision,
    )


def freeze_for_expert_training(
    transformer, controller: AppearanceExpertController
) -> None:
    """Freeze the released transformer and enable only both appearance experts."""
    transformer.requires_grad_(False)
    for parameter in controller.parameters():
        parameter.requires_grad_(True)


def _expert_tensors(
    controller: AppearanceExpertController, domain: str
) -> dict[str, Any]:
    if domain not in controller.config.domains:
        raise ValueError(f"expert domain {domain!r} is not installed")
    tensors = {}
    for index, wrapper in controller.wrappers.items():
        for name, value in wrapper.experts[domain].state_dict().items():
            tensors[f"blocks.{index}.expert.{name}"] = value.detach().cpu().contiguous()
        tensors[f"blocks.{index}.scale"] = (
            wrapper.scales[domain].detach().cpu().contiguous()
        )
    return tensors


def _expert_keys(controller: AppearanceExpertController, domain: str) -> set[str]:
    if domain not in controller.config.domains:
        raise ValueError(f"expert domain {domain!r} is not installed")
    keys = set()
    for index, wrapper in controller.wrappers.items():
        keys.update(
            f"blocks.{index}.expert.{name}"
            for name in wrapper.experts[domain].state_dict()
        )
        keys.add(f"blocks.{index}.scale")
    return keys


def save_appearance_expert(
    controller: AppearanceExpertController,
    domain: str,
    path: Path,
    *,
    dtype: torch.dtype | None = None,
) -> dict[str, Any]:
    """Save one lightweight domain expert, excluding all shared weights."""
    from safetensors.torch import save_file

    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema": EXPERT_SCHEMA,
        "domain": domain,
        "base_model": controller.base_model,
        "base_revision": controller.base_revision,
        "config": asdict(controller.config),
    }
    metadata = {
        "schema": EXPERT_SCHEMA,
        "domain": domain,
        "base_model": controller.base_model,
        "base_revision": controller.base_revision,
        "config_json": json.dumps(asdict(controller.config), sort_keys=True),
    }
    tensors = _expert_tensors(controller, domain)
    if dtype is not None:
        tensors = {
            name: (
                value.to(dtype=dtype) if value.is_floating_point() else value
            ).contiguous()
            for name, value in tensors.items()
        }
    temporary = path.with_name(path.name + ".tmp")
    save_file(tensors, str(temporary), metadata=metadata)
    os.replace(temporary, path)
    manifest["path"] = str(path)
    manifest["sha256"] = _file_sha256(path)
    manifest["tensor_count"] = len(tensors)
    _atomic_json(path.with_suffix(path.suffix + ".json"), manifest)
    return manifest


def load_appearance_expert(
    controller: AppearanceExpertController,
    domain: str,
    path: Path,
    *,
    strict: bool = True,
) -> dict[str, Any]:
    """Load one expert and report every missing or unexpected tensor."""
    import torch
    from safetensors import safe_open
    from safetensors.torch import load_file

    path = path.expanduser().resolve()
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}
    if metadata.get("schema") != EXPERT_SCHEMA:
        raise ValueError("unsupported appearance-expert checkpoint schema")
    if metadata.get("domain") != domain:
        raise ValueError(
            f"checkpoint domain {metadata.get('domain')!r} does not match {domain!r}"
        )
    if metadata.get("base_model") != controller.base_model:
        raise ValueError("appearance expert targets a different base model")
    if metadata.get("base_revision") != controller.base_revision:
        raise ValueError("appearance expert targets a different base revision")

    tensors = load_file(str(path), device="cpu")
    report = checkpoint_key_report(_expert_keys(controller, domain), tensors)
    if strict and not report["compatible"]:
        raise ValueError(
            "appearance expert is incompatible: "
            f"missing={report['missing_keys']} unexpected={report['unexpected_keys']}"
        )
    with torch.no_grad():
        for index, wrapper in controller.wrappers.items():
            prefix = f"blocks.{index}.expert."
            state = {
                key.removeprefix(prefix): value
                for key, value in tensors.items()
                if key.startswith(prefix)
            }
            wrapper.experts[domain].load_state_dict(state, strict=strict)
            scale_key = f"blocks.{index}.scale"
            if scale_key in tensors:
                wrapper.scales[domain].copy_(
                    tensors[scale_key].to(
                        device=wrapper.scales[domain].device,
                        dtype=wrapper.scales[domain].dtype,
                    )
                )
    return {**report, "domain": domain, "path": str(path)}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.expanduser().resolve().open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    benchmark = subparsers.add_parser("write-benchmark")
    benchmark.add_argument("--output", type=Path, required=True)

    prepare = subparsers.add_parser("prepare-domain")
    prepare.add_argument("--input", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--data-root", type=Path, required=True)
    prepare.add_argument("--pixel-budget", type=int, default=512 * 512)
    prepare.add_argument("--max-side", type=int, default=2048)
    prepare.add_argument("--max-aspect-ratio", type=float, default=4.0)
    prepare.add_argument("--no-verify-images", action="store_true")

    audit = subparsers.add_parser("audit-domain")
    audit.add_argument("--manifest", type=Path, required=True)
    audit.add_argument("--max-uncaptioned-fraction", type=float, default=0.15)
    audit.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "write-benchmark":
        result = write_benchmark_suite(args.output)
    elif args.command == "prepare-domain":
        result = prepare_domain_manifest(
            args.input,
            args.output,
            data_root=args.data_root,
            pixel_budget=args.pixel_budget,
            max_side=args.max_side,
            max_aspect_ratio=args.max_aspect_ratio,
            verify_images=not args.no_verify_images,
        )
    elif args.command == "audit-domain":
        result = audit_domain_rows(
            _read_jsonl(args.manifest),
            max_uncaptioned_fraction=args.max_uncaptioned_fraction,
        )
        if args.output:
            _atomic_json(args.output, result)
    else:  # pragma: no cover
        raise AssertionError(args.command)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
