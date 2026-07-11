"""Compressed Convolutional Attention correctness oracle.

Figliolia et al. (2026), https://arxiv.org/abs/2510.04476. Community lead:
https://discord.com/channels/992359628979568762/1426889957221466153/1511375668875890688
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class CCAAttention(nn.Module):
    def __init__(self, d_model: int, latent_dim: int, kernel_size: int = 3):
        super().__init__(); self.latent_dim = latent_dim
        self.q = nn.Linear(d_model, latent_dim, bias=False); self.k = nn.Linear(d_model, latent_dim, bias=False)
        self.v = nn.Linear(d_model, latent_dim, bias=False); self.out = nn.Linear(latent_dim, d_model, bias=False)
        self.conv = nn.Conv1d(latent_dim, latent_dim, kernel_size, groups=latent_dim,
                              padding=kernel_size-1, bias=False)
    def forward(self, x, causal=True):
        q, k, v = self.q(x), self.k(x), self.v(x)
        k = self.conv(k.transpose(1,2))[..., :x.shape[1]].transpose(1,2)
        score = q.float() @ k.float().transpose(-1,-2) / self.latent_dim**0.5
        if causal:
            score = score.masked_fill(torch.ones_like(score, dtype=torch.bool).triu(1), -torch.inf)
        return self.out(F.softmax(score, -1).to(v.dtype) @ v)
    def receipt(self, context: int):
        return {"schema":"rwkv-lab.cca.v1", "latent_dim":self.latent_dim,
                "cache_elements_per_token":2*self.latent_dim,
                "attention_score_flops":context*context*self.latent_dim}
