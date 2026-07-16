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


def logits_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    fused: bool = True,
) -> torch.Tensor:
    """Mean CE for precomputed 2-D logits without an fp32 vocabulary copy."""
    flat_logits = logits.reshape(-1, logits.shape[-1])
    flat_labels = labels.reshape(-1)
    if fused and HAS_FUSED_CE and flat_logits.is_cuda:
        losses, _ = _flash_ce(flat_logits, flat_labels, inplace_backward=True)
        return losses.float().mean()
    return F.cross_entropy(flat_logits.float(), flat_labels)


def weighted_logits_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    weights: torch.Tensor | None = None,
    *,
    fused: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Weighted CE plus an unweighted detached metric from one logit buffer.

    This keeps the flash-attn in-place backward optimization available when
    caption-opening tokens receive extra grounding weight.
    """
    flat_logits = logits.reshape(-1, logits.shape[-1])
    flat_labels = labels.reshape(-1)
    if fused and HAS_FUSED_CE and flat_logits.is_cuda:
        losses, _ = _flash_ce(flat_logits, flat_labels, inplace_backward=True)
        losses = losses.float()
    else:
        losses = F.cross_entropy(
            flat_logits.float(), flat_labels, reduction="none")
    raw = losses.mean().detach()
    if weights is None:
        return losses.mean(), raw
    flat_weights = weights.reshape(-1).to(device=losses.device, dtype=losses.dtype)
    if flat_weights.shape != losses.shape:
        raise ValueError("cross-entropy weights do not match labels")
    return ((losses * flat_weights).sum() / flat_weights.sum().clamp_min(1), raw)


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
    None defaults to -100, so -100-labelled positions are always excluded from
    both the CE sum and the denominator (masked mean) in both paths."""
    flat_h = hidden.reshape(-1, hidden.shape[-1])
    flat_labels = labels.reshape(-1)
    weight = lm_head.weight
    bias = getattr(lm_head, "bias", None)
    effective_ignore = -100 if ignore_index is None else ignore_index

    if fused and HAS_FUSED_CE and flat_h.is_cuda:
        logits = F.linear(flat_h.to(weight.dtype), weight, bias)
        losses, _ = _flash_ce(logits, flat_labels, inplace_backward=True)
        mask = flat_labels != effective_ignore
        return (losses.float() * mask).sum() / mask.sum().clamp_min(1)

    total = hidden.new_zeros((), dtype=torch.float32)
    for start in range(0, flat_h.shape[0], chunk):
        end = min(start + chunk, flat_h.shape[0])
        logits = F.linear(flat_h[start:end].to(weight.dtype), weight, bias)
        total = total + F.cross_entropy(
            logits.float(), flat_labels[start:end], reduction="sum",
            ignore_index=effective_ignore,
        )
    denom = (flat_labels != effective_ignore).sum().clamp_min(1)
    return total / denom
