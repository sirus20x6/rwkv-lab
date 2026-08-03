"""Native-resolution cache contracts for agglomerative vision teachers.

The v1 captioning caches pooled every tower to a fixed token count.  Those
payloads are intentionally incompatible with this module.  Native-v2 keeps
each teacher on its original spatial grid and records the image transform that
maps grid positions back to the source image.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from torch import Tensor


NATIVE_CACHE_SCHEMA = "agglomerative-vision-native-v2"
TEACHERS = frozenset(("moonvit", "siglip2", "dinov2", "sam"))
FLOAT_DTYPES = frozenset((torch.bfloat16, torch.float16, torch.float32))


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False)


def sha256_file(path: str | Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def preprocessing_fingerprint(config: Mapping[str, object]) -> str:
    return hashlib.sha256(canonical_json(dict(config)).encode()).hexdigest()


@dataclass(frozen=True)
class GridGeometry:
    """Map a teacher grid to one rectangular region of the source image.

    Coordinates use half-open source-pixel bounds. ``content_yxyx`` identifies
    the non-padding region in the processed tensor. It permits exact handling
    of SAM padding and MoonViT quadrant views without interpolating targets.
    """

    source_hw: tuple[int, int]
    source_yxyx: tuple[int, int, int, int]
    processed_hw: tuple[int, int]
    content_yxyx: tuple[int, int, int, int]
    grid_hw: tuple[int, int]

    def validate(self) -> None:
        sh, sw = self.source_hw
        sy0, sx0, sy1, sx1 = self.source_yxyx
        ph, pw = self.processed_hw
        cy0, cx0, cy1, cx1 = self.content_yxyx
        gh, gw = self.grid_hw
        if min(sh, sw, ph, pw, gh, gw) < 1:
            raise ValueError("geometry dimensions must be positive")
        if not (0 <= sy0 < sy1 <= sh and 0 <= sx0 < sx1 <= sw):
            raise ValueError("source_yxyx is outside source_hw")
        if not (0 <= cy0 < cy1 <= ph and 0 <= cx0 < cx1 <= pw):
            raise ValueError("content_yxyx is outside processed_hw")


@dataclass(frozen=True)
class NativeEntryMetadata:
    sample_id: str
    teacher: str
    source_path: str
    source_sha256: str
    teacher_fingerprint: str
    preprocessing_fingerprint: str
    geometries: tuple[GridGeometry, ...]
    schema: str = NATIVE_CACHE_SCHEMA

    def validate(self) -> None:
        if self.schema != NATIVE_CACHE_SCHEMA:
            raise ValueError(f"unsupported native cache schema {self.schema!r}")
        if self.teacher not in TEACHERS:
            raise ValueError(f"unknown teacher {self.teacher!r}")
        if not self.sample_id or not self.source_path:
            raise ValueError("sample_id and source_path are required")
        for name, value in (("source_sha256", self.source_sha256),
                            ("teacher_fingerprint", self.teacher_fingerprint),
                            ("preprocessing_fingerprint",
                             self.preprocessing_fingerprint)):
            if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
                raise ValueError(f"{name} must be a lowercase SHA-256 digest")
        if not self.geometries:
            raise ValueError("at least one geometry is required")
        for geometry in self.geometries:
            geometry.validate()

    def to_json(self) -> str:
        self.validate()
        return canonical_json(asdict(self))

    @classmethod
    def from_json(cls, value: str) -> "NativeEntryMetadata":
        raw = json.loads(value)
        raw["geometries"] = tuple(GridGeometry(
            source_hw=tuple(item["source_hw"]),
            source_yxyx=tuple(item["source_yxyx"]),
            processed_hw=tuple(item["processed_hw"]),
            content_yxyx=tuple(item["content_yxyx"]),
            grid_hw=tuple(item["grid_hw"]),
        ) for item in raw["geometries"])
        metadata = cls(**raw)
        metadata.validate()
        return metadata


def _validate_tensor(name: str, tensor: Tensor) -> None:
    if not torch.is_tensor(tensor) or tensor.device.type != "cpu":
        raise ValueError(f"{name} must be a CPU tensor")
    if tensor.dtype not in FLOAT_DTYPES:
        raise ValueError(f"{name} has unsupported dtype {tensor.dtype}")
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"{name} contains non-finite values")


def _validate_dense_geometry(metadata: NativeEntryMetadata, tensor: Tensor,
                             index: int = 0) -> None:
    expected = metadata.geometries[index].grid_hw
    if tuple(tensor.shape[:2]) != expected:
        raise ValueError(
            f"dense grid {tuple(tensor.shape[:2])} != geometry {expected}")


def validate_native_tensors(metadata: NativeEntryMetadata,
                            tensors: Mapping[str, Tensor]) -> None:
    """Strictly validate one teacher's complete native-resolution payload."""
    metadata.validate()
    if not tensors:
        raise ValueError("native teacher payload is empty")
    for name, tensor in tensors.items():
        _validate_tensor(name, tensor)

    names = set(tensors)
    if metadata.teacher == "siglip2":
        if names != {"dense", "summary"}:
            raise ValueError("SigLIP2 requires dense and summary tensors")
        dense, summary = tensors["dense"], tensors["summary"]
        if dense.ndim != 3 or dense.shape[-1] != 1152:
            raise ValueError("SigLIP2 dense shape must be [H,W,1152]")
        if tuple(summary.shape) != (1152,):
            raise ValueError("SigLIP2 summary shape must be [1152]")
        _validate_dense_geometry(metadata, dense)
    elif metadata.teacher == "dinov2":
        if names != {"dense", "summary"}:
            raise ValueError("DINOv2 requires dense and summary tensors")
        dense, summary = tensors["dense"], tensors["summary"]
        if dense.ndim != 3 or dense.shape[-1] != 768:
            raise ValueError("DINOv2 dense shape must be [H,W,768]")
        if tuple(summary.shape) != (768,):
            raise ValueError("DINOv2 summary shape must be [768]")
        _validate_dense_geometry(metadata, dense)
    elif metadata.teacher == "sam":
        if names != {"dense"}:
            raise ValueError("SAM requires only its native dense tensor")
        dense = tensors["dense"]
        if tuple(dense.shape) != (64, 64, 256):
            raise ValueError("SAM dense shape must be [64,64,256]")
        _validate_dense_geometry(metadata, dense)
    else:
        expected = {
            f"tap_{tap}.view_{view}"
            for tap in (8, 17, 26)
            for view in range(len(metadata.geometries))
        }
        if names != expected:
            missing, extra = sorted(expected - names), sorted(names - expected)
            raise ValueError(f"MoonViT tensor set mismatch missing={missing} extra={extra}")
        for view, geometry in enumerate(metadata.geometries):
            for tap in (8, 17, 26):
                tensor = tensors[f"tap_{tap}.view_{view}"]
                if tensor.ndim != 4 or tuple(tensor.shape[-2:]) != (4, 1152):
                    raise ValueError(
                        "MoonViT native shape must be [H,W,4,1152]")
                if tuple(tensor.shape[:2]) != geometry.grid_hw:
                    raise ValueError("MoonViT grid does not match view geometry")


def save_native_entry(path: str | Path, metadata: NativeEntryMetadata,
                      tensors: Mapping[str, Tensor]) -> None:
    """Atomically save a validated native-v2 entry."""
    validate_native_tensors(metadata, tensors)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    cpu = {name: tensor.detach().contiguous() for name, tensor in tensors.items()}
    try:
        save_file(cpu, str(temporary), metadata={
            "native_metadata": metadata.to_json(),
            "schema": NATIVE_CACHE_SCHEMA,
        })
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, target)
        directory_fd = os.open(target.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def load_native_entry(path: str | Path) -> tuple[NativeEntryMetadata,
                                                 dict[str, Tensor]]:
    target = Path(path)
    with safe_open(str(target), framework="pt", device="cpu") as handle:
        header = handle.metadata() or {}
    if header.get("schema") != NATIVE_CACHE_SCHEMA or "native_metadata" not in header:
        raise ValueError(f"{target} is not a native-v2 teacher cache entry")
    metadata = NativeEntryMetadata.from_json(header["native_metadata"])
    tensors = load_file(str(target), device="cpu")
    validate_native_tensors(metadata, tensors)
    return metadata, tensors


@dataclass
class NativeCacheReceipt:
    teacher: str
    teacher_fingerprint: str
    preprocessing_fingerprint: str
    manifest_fingerprint: str
    completed: list[str]
    schema: str = NATIVE_CACHE_SCHEMA

    def validate(self) -> None:
        if self.schema != NATIVE_CACHE_SCHEMA or self.teacher not in TEACHERS:
            raise ValueError("invalid native cache receipt identity")
        for value in (self.teacher_fingerprint, self.preprocessing_fingerprint,
                      self.manifest_fingerprint):
            if len(value) != 64:
                raise ValueError("receipt fingerprints must be SHA-256 digests")
        if len(self.completed) != len(set(self.completed)):
            raise ValueError("receipt has duplicate completed sample IDs")

    @classmethod
    def load(cls, path: str | Path) -> "NativeCacheReceipt":
        receipt = cls(**json.loads(Path(path).read_text()))
        receipt.validate()
        return receipt

    def save(self, path: str | Path) -> None:
        self.validate()
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
        try:
            temporary.write_text(canonical_json(asdict(self)) + "\n")
            with temporary.open("rb") as handle:
                os.fsync(handle.fileno())
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)

    def mark_completed(self, sample_ids: Sequence[str]) -> None:
        seen = set(self.completed)
        for sample_id in sample_ids:
            if sample_id not in seen:
                self.completed.append(sample_id)
                seen.add(sample_id)

