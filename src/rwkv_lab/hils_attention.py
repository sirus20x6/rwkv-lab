"""CPU-readable Hierarchical Landmark Sparse attention reference.

Reference: Hu et al., "Hierarchical Sparse Attention Done Right: Toward
Infinite Context Modeling", arXiv:2607.02980,
https://arxiv.org/abs/2607.02980. Official implementation:
https://github.com/Tencent-Hunyuan/HiLS-Attention.

HiLS learns compressed chunk retrieval under the LM loss, then performs
independent intra-chunk attention and fuses chunk outputs with retrieval scores.
This implementation favors semantic clarity and causal correctness; it is an
experimental hybrid-layer oracle, not a production sparse kernel.
"""
from __future__ import annotations

import math
import torch
from torch import nn
import torch.nn.functional as F


class HiLSAttention(nn.Module):
    def __init__(self, d_model: int, *, heads: int, chunk_size: int = 64,
                 top_chunks: int = 4, landmark_dim: int = 16):
        super().__init__()
        if d_model % heads or chunk_size < 1 or top_chunks < 1:
            raise ValueError("invalid HiLS geometry")
        self.d_model, self.heads = int(d_model), int(heads)
        self.head_dim = d_model // heads
        self.chunk_size, self.top_chunks = int(chunk_size), int(top_chunks)
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.landmark_q = nn.Linear(self.head_dim, landmark_dim, bias=False)
        self.landmark_k = nn.Linear(self.head_dim, landmark_dim, bias=False)
        self.output = nn.Linear(d_model, d_model, bias=False)
        nn.init.zeros_(self.output.weight)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        if hidden.ndim != 3:
            raise ValueError("HiLS hidden must be [batch,time,channels]")
        B, T, _ = hidden.shape; H, D, C = self.heads, self.head_dim, self.chunk_size
        q, k, v = self.qkv(hidden).chunk(3, dim=-1)
        q, k, v = [item.view(B, T, H, D).transpose(1, 2) for item in (q, k, v)]
        chunks = (T + C - 1) // C
        landmark = []
        for index in range(chunks):
            start, end = index * C, min(T, (index + 1) * C)
            landmark.append(self.landmark_k(k[:, :, start:end]).mean(dim=2))
        landmarks = torch.stack(landmark, dim=2)  # B,H,chunks,L
        outputs = []
        scale = 1.0 / math.sqrt(D)
        for token in range(T):
            current = token // C
            retrieval = torch.einsum("bhl,bhcl->bhc", self.landmark_q(q[:, :, token]),
                                     landmarks[:, :, :current + 1])
            selected = retrieval.topk(min(self.top_chunks, current + 1), dim=-1).indices
            per_chunk, chunk_scores = [], []
            for rank in range(selected.shape[-1]):
                # Gather one selected chunk independently for each batch/head.
                chunk_index = selected[:, :, rank]
                positions = chunk_index[..., None] * C + torch.arange(C, device=hidden.device)
                valid = positions <= token
                positions = positions.clamp_max(T - 1)
                gather = positions[..., None].expand(B, H, C, D)
                kc, vc = torch.gather(k, 2, gather), torch.gather(v, 2, gather)
                logits = torch.einsum("bhd,bhcd->bhc", q[:, :, token], kc) * scale
                logits = logits.masked_fill(~valid, -torch.inf)
                weights = F.softmax(logits, dim=-1)
                per_chunk.append(torch.einsum("bhc,bhcd->bhd", weights, vc))
                chunk_scores.append(torch.gather(retrieval, -1, chunk_index[..., None]).squeeze(-1))
            fusion = F.softmax(torch.stack(chunk_scores, dim=-1), dim=-1)
            stacked = torch.stack(per_chunk, dim=-2)
            outputs.append(torch.einsum("bhr,bhrd->bhd", fusion, stacked))
        out = torch.stack(outputs, dim=2).transpose(1, 2).reshape(B, T, self.d_model)
        return self.output(out)
