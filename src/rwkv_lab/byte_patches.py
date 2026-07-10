"""Entropy-driven dynamic byte patches.

Reference: Pagnoni et al., "Byte Latent Transformer: Patches Scale Better Than
Tokens", arXiv:2412.09871, https://arxiv.org/abs/2412.09871.

BLT replaces a fixed tokenizer with byte-level local encoders and dynamically
sized patches, placing boundaries where next-byte entropy is high. The helpers
below implement that boundary rule plus exact pool/unpool bookkeeping so an
RWKV global recurrent core can operate over patches. The local byte encoder is
deliberately small and replaceable; this is not a reproduction of BLT's full
Transformer architecture.
"""
from __future__ import annotations

from dataclasses import dataclass
import torch
import torch.nn as nn


def entropy_patch_ids(entropy: torch.Tensor, *, threshold: float,
                      min_patch: int = 1, max_patch: int = 16) -> torch.Tensor:
    """Assign monotonically increasing patch ids to ``[batch,time]`` bytes."""

    if entropy.ndim != 2 or min_patch <= 0 or max_patch < min_patch:
        raise ValueError("invalid entropy shape or patch bounds")
    B, T = entropy.shape
    ids = torch.zeros(B, T, dtype=torch.long, device=entropy.device)
    for b in range(B):
        patch, length = 0, 1
        for t in range(1, T):
            split = length >= max_patch or (length >= min_patch and float(entropy[b, t]) >= threshold)
            if split:
                patch, length = patch + 1, 1
            else:
                length += 1
            ids[b, t] = patch
    return ids


@dataclass
class PatchBatch:
    values: torch.Tensor
    counts: torch.Tensor
    patch_ids: torch.Tensor
    mask: torch.Tensor

    def unpool(self) -> torch.Tensor:
        index = self.patch_ids.unsqueeze(-1).expand(*self.patch_ids.shape, self.values.shape[-1])
        return self.values.gather(1, index)


def pool_patches(hidden: torch.Tensor, patch_ids: torch.Tensor) -> PatchBatch:
    """Mean-pool byte states into padded patches while preserving exact mapping."""

    if hidden.ndim != 3 or patch_ids.shape != hidden.shape[:2]:
        raise ValueError("hidden must be [B,T,C] and patch_ids [B,T]")
    B, _, C = hidden.shape
    n = int(patch_ids.max()) + 1
    values = hidden.new_zeros(B, n, C)
    counts = hidden.new_zeros(B, n)
    values.scatter_add_(1, patch_ids[..., None].expand(-1, -1, C), hidden)
    counts.scatter_add_(1, patch_ids, torch.ones_like(patch_ids, dtype=hidden.dtype))
    values = values / counts.clamp_min(1).unsqueeze(-1)
    return PatchBatch(values, counts, patch_ids, counts > 0)


class BytePatchEncoder(nn.Module):
    """Local byte embedding and entropy predictor feeding a patch-level core."""

    def __init__(self, d_model: int, *, threshold: float = 3.0,
                 min_patch: int = 1, max_patch: int = 16):
        super().__init__()
        self.byte_embedding = nn.Embedding(256, d_model)
        self.entropy_head = nn.Linear(d_model, 256)
        self.threshold, self.min_patch, self.max_patch = threshold, min_patch, max_patch

    def forward(self, byte_ids: torch.Tensor) -> PatchBatch:
        if byte_ids.ndim != 2 or torch.any((byte_ids < 0) | (byte_ids > 255)):
            raise ValueError("byte_ids must be [B,T] values in [0,255]")
        hidden = self.byte_embedding(byte_ids)
        probs = self.entropy_head(hidden).float().softmax(-1)
        entropy = -(probs * probs.clamp_min(1e-9).log()).sum(-1)
        ids = entropy_patch_ids(entropy.detach(), threshold=self.threshold,
                                min_patch=self.min_patch, max_patch=self.max_patch)
        return pool_patches(hidden, ids)
