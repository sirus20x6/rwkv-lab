"""Unit-scaled maximal-update parametrization utilities.

Reference: Blake et al., "u-muP: The Unit-Scaled Maximal Update
Parametrization", arXiv:2407.17465, https://arxiv.org/abs/2407.17465.

The paper's useful contract for this lab is *scale transfer*: tune at a base
width, then preserve order-one activations and width-correct parameter updates
at a target width.  This module keeps that contract explicit and framework
independent.  It provides:

* unit-variance initialization for embeddings and linear maps;
* depth-scaled residual-output initialization;
* Adam-style per-parameter learning-rate multipliers derived from the ratio of
  base fan-in to target fan-in; and
* metadata on each parameter so experiment capsules can audit the applied
  scaling rule.

This is the practical u-muP subset needed by RWKV-Lab, not a claim that every
operator in the paper is reproduced.  In particular, recurrent CUDA kernels
keep their native parametrization; the trainable projections around them are
the maximal-update objects controlled here.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import torch
import torch.nn as nn


@dataclass(frozen=True)
class UMuPConfig:
    """Base/target geometry used for scale-transfer rules."""

    base_width: int
    width: int
    depth: int = 1
    base_depth: int = 1

    def __post_init__(self):
        for name in ("base_width", "width", "depth", "base_depth"):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")

    @property
    def width_ratio(self) -> float:
        return self.width / self.base_width

    @property
    def residual_scale(self) -> float:
        # Unit-scaled residual branches contribute O(1/depth) variance each.
        return math.sqrt(self.base_depth / self.depth)


def _role(name: str, p: torch.Tensor) -> str:
    low = name.lower()
    if p.ndim < 2 or any(x in low for x in ("norm", "ln", "bias", "gate", "scale")):
        return "scalar"
    if "emb" in low:
        return "embedding"
    if "head" in low or "readout" in low:
        return "readout"
    if any(x in low for x in ("output", "out_proj", ".o.", ".o.weight", "ffn.value")):
        return "residual_output"
    return "matrix"


def initialize_u_mup(module: nn.Module, config: UMuPConfig) -> dict[str, dict[str, float | str]]:
    """Initialize trainable tensors and return an auditable scaling manifest.

    Matrix rows receive unit-variance outputs for unit-variance inputs
    (std=fan_in^-1/2). Residual outputs additionally receive the paper's
    depth-normalization principle. Embeddings are unit variance; scalar gates
    and normalization parameters retain the module's intentional initialization.
    """

    manifest: dict[str, dict[str, float | str]] = {}
    with torch.no_grad():
        for name, p in module.named_parameters():
            role = _role(name, p)
            if not p.requires_grad:
                continue
            # Preserve deliberate no-op initializations (residual output ends,
            # gates, adapters). Re-randomizing those would break composability.
            if p.numel() and not torch.count_nonzero(p):
                std = 0.0
            elif role == "embedding":
                std = 1.0
                nn.init.normal_(p, std=std)
            elif p.ndim >= 2:
                std = p.shape[-1] ** -0.5
                if role == "residual_output":
                    std *= config.residual_scale
                nn.init.normal_(p, std=std)
            else:
                std = float("nan")
            manifest[name] = {
                "role": role,
                "init_std": float(std),
                "lr_mult": float(lr_multiplier(name, p, config)),
            }
    module._u_mup_manifest = manifest  # type: ignore[attr-defined]
    return manifest


def lr_multiplier(name: str, p: torch.Tensor, config: UMuPConfig) -> float:
    """Adam learning-rate multiplier preserving maximal feature updates.

    For hidden matrices Adam's elementwise normalization removes gradient
    magnitude, so their target-width step is reduced by base_fan_in/fan_in.
    Readouts use the square-root rule because they map width to fixed logits;
    embeddings and scalar controls remain width independent.
    """

    role = _role(name, p)
    if role in ("scalar", "embedding") or p.ndim < 2:
        return 1.0
    base_fan_in = min(config.base_width, int(p.shape[-1]))
    ratio = base_fan_in / max(int(p.shape[-1]), 1)
    return math.sqrt(ratio) if role == "readout" else ratio


def parameter_groups(
    named_parameters: Iterable[tuple[str, nn.Parameter]],
    *,
    lr: float,
    weight_decay: float,
    config: UMuPConfig,
) -> list[dict]:
    """Build optimizer groups keyed by identical u-muP LR multipliers."""

    buckets: dict[float, list[nn.Parameter]] = {}
    names: dict[float, list[str]] = {}
    for name, p in named_parameters:
        if not p.requires_grad:
            continue
        mult = float(lr_multiplier(name, p, config))
        buckets.setdefault(mult, []).append(p)
        names.setdefault(mult, []).append(name)
    return [
        {
            "params": params,
            "lr": lr * mult,
            "weight_decay": weight_decay,
            "u_mup_lr_mult": mult,
            "u_mup_names": names[mult],
        }
        for mult, params in sorted(buckets.items())
    ]
