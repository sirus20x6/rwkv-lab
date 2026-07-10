"""Shared memory-efficient language-model cross entropy.

The fast path uses flash-attn's Triton CE, whose backward reuses the bf16
logit buffer.  The portable fallback bounds memory by projecting token chunks.
Keeping this here prevents the large trainers from drifting onto different CE
implementations and precision behavior.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

try:  # Optional: CUDA-stack specific, so it is intentionally not a package dep.
    from flash_attn.ops.triton.cross_entropy import cross_entropy_loss as _flash_ce
    HAS_FUSED_CE = True
except Exception:  # pragma: no cover - depends on the local CUDA stack
    _flash_ce = None
    HAS_FUSED_CE = False


def lmhead_cross_entropy(
    hidden: torch.Tensor,
    lm_head: torch.nn.Module,
    labels: torch.Tensor,
    *,
    chunk: int = 2048,
    fused: bool = True,
    ignore_index: int | None = None,
) -> torch.Tensor:
    """Apply ``lm_head`` and return mean CE without an fp32 full-logit copy.

    ignore_index: positions whose label equals it are excluded from the mean (packed-row padding);
    None keeps the plain all-positions mean."""
    flat_h = hidden.reshape(-1, hidden.shape[-1])
    flat_labels = labels.reshape(-1)
    weight = lm_head.weight
    bias = getattr(lm_head, "bias", None)

    if fused and HAS_FUSED_CE and flat_h.is_cuda:
        logits = F.linear(flat_h.to(weight.dtype), weight, bias)
        losses, _ = _flash_ce(logits, flat_labels, inplace_backward=True)
        if ignore_index is not None:
            mask = flat_labels != ignore_index
            return (losses.float() * mask).sum() / mask.sum().clamp_min(1)
        return losses.float().mean()

    total = hidden.new_zeros((), dtype=torch.float32)
    for start in range(0, flat_h.shape[0], chunk):
        end = min(start + chunk, flat_h.shape[0])
        logits = F.linear(flat_h[start:end].to(weight.dtype), weight, bias)
        total = total + F.cross_entropy(
            logits.float(), flat_labels[start:end], reduction="sum",
            ignore_index=(-100 if ignore_index is None else ignore_index),
        )
    denom = (flat_labels != ignore_index).sum().clamp_min(1) if ignore_index is not None \
        else flat_h.shape[0]
    return total / denom
