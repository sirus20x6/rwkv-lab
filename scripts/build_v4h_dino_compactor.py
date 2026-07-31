#!/usr/bin/env python3
"""Build and verify the fixed 4096 -> 1536 DINOv3 adaptor compactor."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rwkv_lab.radio_v4h_adaptors import (  # noqa: E402
    DEFAULT_COMPACTOR, load_v4h_adaptor_fusion, save_dino_qr_compactor)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", type=Path, default=ROOT / "models/vision/C-RADIOv4-H")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or args.model / DEFAULT_COMPACTOR

    fusion, basis = load_v4h_adaptor_fusion(
        args.model, device="cpu", dtype=torch.float32, build_compactor=True,
        compactor=output)
    if basis is None:
        raise RuntimeError("compactor builder did not return a reconstruction basis")
    compact = fusion.dino.final[-1]

    # Verify against the original affine map before committing the artifact.
    from rwkv_lab.radio_v4h_adaptors import _load_checkpoint
    _, state = _load_checkpoint(args.model)
    weight = state[
        "_feature_projections.dino_v3_7b.final.2.weight"].float()
    bias = state[
        "_feature_projections.dino_v3_7b.final.2.bias"].float()
    generator = torch.Generator().manual_seed(20260725)
    latent = torch.randn(8, weight.shape[1], generator=generator)
    expected = torch.nn.functional.linear(latent, weight, bias)

    def reconstruction_error(dtype: torch.dtype) -> float:
        # Cast copies, never the module: an in-place round trip through bf16
        # would quantize the very weights this script is about to save.
        coordinates = torch.nn.functional.linear(
            latent.to(dtype), compact.weight.detach().to(dtype),
            compact.bias.detach().to(dtype))[:, :basis.shape[1]]
        reconstructed = coordinates.float() @ basis.T
        return ((reconstructed - expected).norm()
                / expected.norm().clamp_min(1e-12)).item()

    # Two gates, because "lossless" is a statement about a dtype. fp32 is where
    # the QR identity is exact; bf16 is what production actually loads
    # (load_v4h_adaptor_fusion defaults to bfloat16), and there an 8-bit
    # mantissa costs ~1e-3 relative no matter how good the factorization is.
    # Gating only fp32 asserted a bound the deployed path misses by ~85x.
    relative = reconstruction_error(torch.float32)
    if relative > 2e-5:
        raise RuntimeError(f"DINO QR reconstruction error is too high: {relative}")
    bfloat16_relative = reconstruction_error(torch.bfloat16)
    if bfloat16_relative > 5e-3:
        raise RuntimeError(
            "DINO QR reconstruction in bfloat16 is worse than mantissa "
            f"rounding explains: {bfloat16_relative}")
    save_dino_qr_compactor(output, compact, basis)
    print({
        "output": str(output),
        "rank_width": int(basis.shape[1]),
        "compact_width": int(compact.out_features),
        "dead_channels": int(compact.out_features - basis.shape[1]),
        "relative_reconstruction_error": relative,
        "bfloat16_relative_reconstruction_error": bfloat16_relative,
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
