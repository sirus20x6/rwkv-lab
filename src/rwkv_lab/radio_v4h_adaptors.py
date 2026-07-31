"""Frozen C-RADIOv4-H teacher-adaptor fusion.

C-RADIOv4-H was distilled from three teachers and ships a spatial adaptor for
each one:

* SigLIP2-g: 1536 channels
* SAM3: 1024 channels
* DINOv3-7B: 4096 channels

The 4096-wide DINO output is the affine image of a 1520-wide latent.  A reduced
QR factorization of its final ``[W | b]`` matrix therefore gives at most 1521
orthogonal coordinates that preserve the complete DINO output and its inner
products.  We pad those coordinates to 1536 and concatenate all three teacher
spaces into one native 4096-wide RWKV token:

    1536 SigLIP2 + 1024 SAM3 + 1536 compact DINO = 4096.

Two honest caveats about that compact block:

* The QR rank is exactly 1521, so DINO occupies 1521 of its 1536 slots and
  channels 1521:1536 are structurally, permanently zero -- 15 dead channels,
  0.37% of the fused token.  1536 is chosen anyway because it is what makes the
  three teachers sum to 4096 (see ``DINO_COMPACT_WIDTH``).
* "Lossless" is an fp32 statement.  Production loads at bfloat16, where the
  reconstruction is accurate to ~2e-3 relative rather than the fp32 ~1e-6 --
  the ordinary cost of an 8-bit mantissa, and below the quantization the bf16
  feature cache applies to these vectors anyway.

Everything in this module is frozen.  The compaction is derived only from the
pinned NVIDIA checkpoint; it is not a learned vision-to-language projection.
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import types
from pathlib import Path

import torch
from torch import Tensor, nn

SIGLIP2_SPATIAL_WIDTH = 1536
SAM3_SPATIAL_WIDTH = 1024
DINO_SPATIAL_WIDTH = 4096
# 1536, not the 1521 the QR rank actually needs: the three teacher widths have
# to sum to the LM's 4096, and 1536 is the value that does it. The consequence
# is explicit rather than hidden -- compact.weight[DINO_QR_RANK:] is zeroed at
# construction and never written, so the last 15 channels of the fused token
# carry exactly 0.0 for every image, forever. They cost 0.37% of the token and
# buy alignment with the 4096-wide bridge.
DINO_COMPACT_WIDTH = 1536
DINO_QR_RANK = 1521          # 1520 latent dimensions plus the bias column
FUSED_ADAPTOR_WIDTH = (
    SIGLIP2_SPATIAL_WIDTH + SAM3_SPATIAL_WIDTH + DINO_COMPACT_WIDTH
)
# Fixed corpus calibration measured over a path-hash-random sample of native
# caption/OCR/structured grids. Scalar multiplication preserves every
# teacher's within-space geometry and confidence variation, unlike per-token
# LayerNorm, while preventing the compact DINO coordinates from entering the
# shared bridge at roughly one fifth of SAM3's scale.
#
# The DINO figure was measured across the whole 1536-wide block, which includes
# the 15 dead channels, so it understates the live standard deviation by
# sqrt(1521/1536) = 0.9951 and the scale is 0.49% high. That is far inside the
# calibration's own sampling noise, and correcting it would fork the numerics of
# the already-written fused cache under an unchanged revision string, so it is
# documented rather than adjusted.
SIGLIP2_FUSION_SCALE = 1.0 / 0.739328316837559
SAM3_FUSION_SCALE = 1.0 / 1.1187075244576177
DINO_FUSION_SCALE = 1.0 / 0.23292758564542096
ADAPTOR_NAMES = ("siglip2-g", "sam3", "dino_v3_7b")
DEFAULT_ADAPTOR_CHECKPOINT = "c-radio_v4-h_half.pth.tar"
DEFAULT_COMPACTOR = "dino_v3_7b_qr_compact_1536.safetensors"


def _remote_module(model_path: Path, filename: str):
    """Import one pinned RADIO source file as part of its local package.

    Deliberately the SAME synthetic package ``load_radio_v4h`` uses, and named
    from a digest rather than ``hash()``: ``hash(str)`` is salted per process,
    so the old name changed between runs and between a worker and its parent,
    and the private package it created held a second, unrelated copy of every
    class ``hf_model`` had already imported (MLP2, AttnFDHead, Block). Sharing
    the package means an already-imported submodule is reused and the adaptor
    heads are instances of the same classes the encoder was built from.
    """
    from rwkv_lab.radio_v4h import _pinned_package_name

    path = Path(model_path).resolve()
    package = _pinned_package_name(path, "radio_v4h")
    if package not in sys.modules:
        container = types.ModuleType(package)
        container.__path__ = [str(path)]
        container.__package__ = package
        sys.modules[package] = container
    module_name = f"{package}.{Path(filename).stem}"
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, path / filename)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import {filename} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _load_checkpoint(model_path: Path, checkpoint: Path | None = None):
    path = Path(model_path).resolve()
    checkpoint = Path(checkpoint or path / DEFAULT_ADAPTOR_CHECKPOINT).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"C-RADIO adaptor checkpoint is missing: {checkpoint}")
    with torch.serialization.safe_globals([argparse.Namespace]):
        blob = torch.load(
            checkpoint, map_location="cpu", mmap=True, weights_only=True)
    if not isinstance(blob, dict) or "args" not in blob or "state_dict" not in blob:
        raise ValueError(f"unsupported C-RADIO adaptor checkpoint: {checkpoint}")
    return blob["args"], blob["state_dict"]


def _feature_state(state: dict[str, Tensor], name: str) -> dict[str, Tensor]:
    """Translate the training checkpoint namespace to GenericAdaptor format."""
    prefix = f"_feature_projections.{name}"
    translated = {
        "feature" + key[len(prefix):]: value
        for key, value in state.items()
        if key.startswith(prefix)
    }
    if not translated:
        raise KeyError(f"checkpoint contains no spatial adaptor {name!r}")
    return translated


def _build_feature_head(factory, args, state: dict[str, Tensor], teacher: dict):
    version = (
        teacher.get("spatial_mlp_version")
        or getattr(args, "spatial_mlp_version", None)
        or args.mlp_version
    )
    extra: dict[str, int | None] = {}
    if teacher.get("fd_upsample_factor") is not None:
        extra["upsample_factor"] = teacher["fd_upsample_factor"]
        extra["upsample_rank"] = teacher.get("fd_upsample_rank")
    return factory.create_mlp_from_state(
        version, _feature_state(state, teacher["name"]), "feature.",
        spectral_weights=getattr(args, "spectral_heads", False),
        is_summary=False, **extra)


def build_dino_qr_compactor(
    dino_head: nn.Module,
    *,
    output_width: int = DINO_COMPACT_WIDTH,
) -> tuple[nn.Linear, Tensor]:
    """Replace DINO's affine expansion with lossless orthogonal coordinates.

    Returns ``(compact_linear, reconstruction_basis)``.  If ``z`` is the input
    to DINO's original final linear, then:

        original(z) == reconstruction_basis @ compact_linear(z)[:rank]

    to fp32 precision (~1e-6 relative); in bfloat16 the same identity holds to
    ~2e-3, which is mantissa rounding, not a defect in the factorization.  The
    unused tail ``compact_linear.weight[rank:]`` is exactly zero padding and
    stays zero for every input -- ``output_width`` is chosen for the fused
    token's arithmetic, not for the rank.
    """
    final = dino_head.final[-1]
    if not isinstance(final, nn.Linear):
        raise TypeError("DINO spatial head does not end in nn.Linear")
    if final.in_features != 1520 or final.out_features != DINO_SPATIAL_WIDTH:
        raise ValueError(
            "unexpected DINO final geometry "
            f"{final.in_features}->{final.out_features}")
    if output_width < final.in_features + 1:
        raise ValueError("compact width cannot preserve DINO's affine rank")

    weight = final.weight.detach().float().cpu()
    bias = final.bias.detach().float().cpu()
    affine = torch.cat((weight, bias[:, None]), dim=1)
    basis, coordinates = torch.linalg.qr(affine, mode="reduced")
    rank_width = coordinates.shape[0]

    compact = nn.Linear(final.in_features, output_width, bias=True)
    with torch.no_grad():
        compact.weight.zero_()
        compact.bias.zero_()
        compact.weight[:rank_width].copy_(coordinates[:, :-1])
        compact.bias[:rank_width].copy_(coordinates[:, -1])
    compact.requires_grad_(False)
    return compact, basis.contiguous()


def save_dino_qr_compactor(path: Path, compact: nn.Linear, basis: Tensor) -> None:
    from safetensors.torch import save_file

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{os.getpid()}.tmp"
    save_file({
        "weight": compact.weight.detach().cpu().contiguous(),
        "bias": compact.bias.detach().cpu().contiguous(),
        "reconstruction_basis": basis.detach().cpu().contiguous(),
    }, str(temporary), metadata={
        "schema": "radio-v4h-dino-qr-v1",
        "source_width": str(DINO_SPATIAL_WIDTH),
        "compact_width": str(DINO_COMPACT_WIDTH),
    })
    temporary.replace(path)


def load_dino_qr_compactor(path: Path) -> tuple[nn.Linear, Tensor]:
    from safetensors import safe_open

    with safe_open(str(path), framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}
        if metadata.get("schema") != "radio-v4h-dino-qr-v1":
            raise ValueError(f"unsupported DINO compactor schema in {path}")
        weight = handle.get_tensor("weight")
        bias = handle.get_tensor("bias")
        basis = handle.get_tensor("reconstruction_basis")
    compact = nn.Linear(weight.shape[1], weight.shape[0], bias=True)
    compact.load_state_dict({"weight": weight, "bias": bias})
    compact.requires_grad_(False)
    return compact, basis


class V4HAdaptorFusion(nn.Module):
    """Emit one 4096-wide token per C-RADIO patch using every v4 teacher."""

    def __init__(self, siglip2: nn.Module, sam3: nn.Module, dino: nn.Module):
        super().__init__()
        self.siglip2 = siglip2
        self.sam3 = sam3
        self.dino = dino
        self.register_buffer(
            "fusion_scales",
            torch.tensor((
                SIGLIP2_FUSION_SCALE, SAM3_FUSION_SCALE, DINO_FUSION_SCALE),
                dtype=torch.float32),
            persistent=True)

    def forward(self, features: Tensor) -> Tensor:
        """Fuse the three heads at the input dtype.

        NVIDIA's ``GenericAdaptor.forward`` casts head outputs to fp32. We do
        not: the fused grid is stored as bf16 either way, so an fp32 detour buys
        at most the difference between a bf16 matmul epilogue and a bf16 cast of
        an fp32 one (~1e-3 relative, under the cache's own bf16 quantization),
        while changing it now would fork the numerics of the already-written
        fused cache without changing its revision string. Revisit only together
        with a revision bump and a full re-encode.
        """
        if features.ndim != 3 or features.shape[-1] != 1280:
            raise ValueError(
                f"expected generic C-RADIO features [B,L,1280], got "
                f"{tuple(features.shape)}")
        siglip2 = self.siglip2(features)
        sam3 = self.sam3(features)
        dino = self.dino(features)
        expected = (
            SIGLIP2_SPATIAL_WIDTH, SAM3_SPATIAL_WIDTH, DINO_COMPACT_WIDTH)
        actual = (siglip2.shape[-1], sam3.shape[-1], dino.shape[-1])
        if actual != expected:
            raise RuntimeError(f"adaptor widths {actual} do not match {expected}")
        scales = self.fusion_scales.to(device=features.device, dtype=features.dtype)
        return torch.cat((
            siglip2 * scales[0], sam3 * scales[1], dino * scales[2]), dim=-1)


def load_v4h_adaptor_fusion(
    model_path: str | Path,
    *,
    checkpoint: str | Path | None = None,
    compactor: str | Path | None = None,
    device: str | torch.device = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    build_compactor: bool = False,
) -> tuple[V4HAdaptorFusion, Tensor | None]:
    """Load all three frozen spatial heads from NVIDIA's original archive.

    ``build_compactor`` computes the DINO QR basis when no saved compactor is
    available and returns the reconstruction basis for validation.  Production
    cache workers should load the precomputed safetensors file instead.
    """
    path = Path(model_path).resolve()
    args, state = _load_checkpoint(path, Path(checkpoint) if checkpoint else None)
    teachers = {teacher["name"]: teacher for teacher in args.teachers}
    missing = set(ADAPTOR_NAMES) - set(teachers)
    if missing:
        raise ValueError(f"C-RADIO checkpoint is missing adaptors {sorted(missing)}")
    factory = _remote_module(path, "adaptor_module_factory.py")

    siglip2 = _build_feature_head(factory, args, state, teachers["siglip2-g"])
    sam3 = _build_feature_head(factory, args, state, teachers["sam3"])
    dino = _build_feature_head(factory, args, state, teachers["dino_v3_7b"])

    basis = None
    compactor_path = Path(compactor or path / DEFAULT_COMPACTOR)
    if compactor_path.is_file():
        compact, basis = load_dino_qr_compactor(compactor_path)
    elif build_compactor:
        compact, basis = build_dino_qr_compactor(dino)
    else:
        raise FileNotFoundError(
            f"DINO compactor is missing: {compactor_path}; build it once with "
            "scripts/build_v4h_dino_compactor.py")
    dino.final[-1] = compact

    fusion = V4HAdaptorFusion(siglip2, sam3, dino)
    fusion.eval().requires_grad_(False)
    return fusion.to(device=device, dtype=dtype), basis


__all__ = [
    "ADAPTOR_NAMES", "DEFAULT_ADAPTOR_CHECKPOINT", "DEFAULT_COMPACTOR",
    "DINO_COMPACT_WIDTH", "DINO_FUSION_SCALE", "DINO_QR_RANK",
    "DINO_SPATIAL_WIDTH",
    "FUSED_ADAPTOR_WIDTH", "SAM3_FUSION_SCALE", "SAM3_SPATIAL_WIDTH",
    "SIGLIP2_FUSION_SCALE", "SIGLIP2_SPATIAL_WIDTH", "V4HAdaptorFusion",
    "build_dino_qr_compactor", "load_dino_qr_compactor",
    "load_v4h_adaptor_fusion", "save_dino_qr_compactor",
]
