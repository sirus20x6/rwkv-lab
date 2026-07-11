"""Triplet-block diffusion head for RWKV hidden states.

Reference: Lin et al., "Triplet-Block Diffusion RWKV" (B^3D-RWKV),
arXiv:2605.25969, https://arxiv.org/abs/2605.25969; official code:
https://github.com/leonardodalinky/B3D-RWKV.

The paper combines recurrent inter-block processing with bidirectional discrete
diffusion inside small triplet blocks. This module is an isolated auxiliary head:
the base RWKV remains unchanged, and adoption requires quality/throughput campaigns.
"""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class TripletBlockDiffusionHead(nn.Module):
    def __init__(self, d_model: int, vocab_size: int, *, block_size: int = 3,
                 heads: int = 4, mask_token_id: int = 0):
        super().__init__()
        if block_size < 2 or d_model % heads:
            raise ValueError("diffusion block must be >=2 and d_model divisible by heads")
        self.block_size, self.mask_token_id = int(block_size), int(mask_token_id)
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.noise_embedding = nn.Embedding(128, d_model)
        self.attention = nn.MultiheadAttention(d_model, heads, batch_first=True)
        self.norm = nn.RMSNorm(d_model)
        self.output = nn.Linear(d_model, vocab_size, bias=False)
        nn.init.zeros_(self.output.weight)  # exact zero-logit auxiliary at registration

    def corrupt(self, ids: torch.Tensor, noise_level: torch.Tensor,
                *, generator: torch.Generator | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        if ids.ndim != 2 or noise_level.shape != ids.shape[:1]:
            raise ValueError("diffusion ids must be [batch,time] and noise [batch]")
        probability = noise_level.float().clamp(0, 1)[:, None]
        mask = torch.rand(ids.shape, device=ids.device, generator=generator) < probability
        return ids.masked_fill(mask, self.mask_token_id), mask

    def forward(self, recurrent_hidden: torch.Tensor, noisy_ids: torch.Tensor,
                noise_step: torch.Tensor) -> torch.Tensor:
        B, T, D = recurrent_hidden.shape
        if noisy_ids.shape != (B, T) or noise_step.shape != (B,):
            raise ValueError("diffusion input geometry mismatch")
        pad = (-T) % self.block_size
        x = recurrent_hidden + self.token_embedding(noisy_ids)
        x = x + self.noise_embedding(noise_step.clamp(0, 127))[:, None]
        if pad:
            x = F.pad(x, (0, 0, 0, pad))
        blocks = x.view(B * ((T + pad) // self.block_size), self.block_size, D)
        mixed, _ = self.attention(blocks, blocks, blocks, need_weights=False)
        return self.output(self.norm(mixed + blocks)).view(B, T + pad, -1)[:, :T]

    def loss(self, recurrent_hidden: torch.Tensor, ids: torch.Tensor,
             noise_step: torch.Tensor, *, generator: torch.Generator | None = None) -> torch.Tensor:
        noisy, masked = self.corrupt(ids, noise_step.float() / 127.0, generator=generator)
        logits = self(recurrent_hidden, noisy, noise_step)
        if not masked.any():
            return logits.sum() * 0
        return F.cross_entropy(logits[masked].float(), ids[masked])
