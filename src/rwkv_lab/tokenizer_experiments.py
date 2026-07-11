"""Opt-in byte-aware embedding and superword-tokenizer experiments.

Byte-position and token-length embeddings follow BlinkDL's RWKV community idea:
https://discord.com/channels/992359628979568762/992372861924823080/1098006666986930276.
Superword corpus arms follow SuperBPE (Liu et al., COLM 2025),
https://arxiv.org/abs/2503.13423, with the faster reference implementations from
Schmidt et al., https://arxiv.org/abs/2604.05192. This module does not silently
retokenize an existing checkpoint; tokenization remains an explicit corpus arm.
"""
from __future__ import annotations

import ast
from pathlib import Path
from typing import Mapping

import torch
from torch import nn


def load_world_token_bytes(path: str | Path) -> dict[int, bytes]:
    result = {0: b""}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        first, last = line.index(" "), line.rindex(" ")
        value = ast.literal_eval(line[first + 1:last])
        result[int(line[:first])] = value.encode() if isinstance(value, str) else bytes(value)
    return result


class ByteAwareEmbedding(nn.Module):
    """Add token-length and position-specific UTF-8 byte embeddings.

    Both additive branches are zero-initialized, making installation an exact
    no-op. ``base`` remains the checkpoint's ordinary token embedding.
    """
    def __init__(self, base: nn.Embedding, token_bytes: Mapping[int, bytes], *,
                 max_bytes: int = 16, byte_dim: int = 0):
        super().__init__()
        if max_bytes < 1:
            raise ValueError("max_bytes must be positive")
        self.base = base; self.max_bytes = int(max_bytes)
        width = int(byte_dim or base.embedding_dim)
        self.byte = nn.Embedding(256 * max_bytes, width)
        self.length = nn.Embedding(max_bytes + 1, width)
        self.project = (nn.Identity() if width == base.embedding_dim
                        else nn.Linear(width, base.embedding_dim, bias=False))
        nn.init.zeros_(self.byte.weight); nn.init.zeros_(self.length.weight)
        if isinstance(self.project, nn.Linear): nn.init.zeros_(self.project.weight)
        table = torch.zeros(base.num_embeddings, max_bytes, dtype=torch.long)
        lengths = torch.zeros(base.num_embeddings, dtype=torch.long)
        mask = torch.zeros(base.num_embeddings, max_bytes, dtype=torch.bool)
        for token, raw in token_bytes.items():
            if not 0 <= token < base.num_embeddings: continue
            raw = raw[:max_bytes]; lengths[token] = len(raw)
            for position, value in enumerate(raw):
                table[token, position] = position * 256 + value; mask[token, position] = True
        self.register_buffer("byte_ids", table, persistent=True)
        self.register_buffer("length_ids", lengths, persistent=True)
        self.register_buffer("byte_mask", mask, persistent=True)

    @property
    def weight(self):
        return self.base.weight

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        byte = self.byte(self.byte_ids[ids])
        byte = (byte * self.byte_mask[ids].unsqueeze(-1)).sum(-2)
        extra = self.project(byte + self.length(self.length_ids[ids]))
        return self.base(ids) + extra.to(self.base.weight.dtype)


def install_byte_aware_embedding(model: nn.Module, token_bytes: Mapping[int, bytes], *,
                                 max_bytes: int = 16, byte_dim: int = 0) -> ByteAwareEmbedding:
    if not isinstance(getattr(model, "emb", None), nn.Embedding):
        raise ValueError("byte-aware installation requires model.emb to be nn.Embedding")
    wrapped = ByteAwareEmbedding(model.emb, token_bytes, max_bytes=max_bytes, byte_dim=byte_dim)
    wrapped.to(device=model.emb.weight.device)
    model.emb = wrapped
    return wrapped


def superword_training_command(*, implementation: str, corpus: str, output: str,
                               vocab_size: int = 200_000, transition_vocab: int = 0,
                               ztok: str = "ztok") -> list[str]:
    """Build an auditable command for ztok's cited SuperBPE trainer."""
    if implementation != "superbpe":
        raise ValueError("ztok currently implements the SuperBPE curriculum")
    if not 256 <= vocab_size <= 1_000_000:
        raise ValueError("vocab_size outside the supported experiment range")
    if transition_vocab and not 256 < transition_vocab < vocab_size:
        raise ValueError("transition_vocab must lie between 256 and vocab_size")
    command = [ztok, "train", "--kind", "superbpe", "--input", str(corpus),
               "--output", str(output), "--vocab-size", str(vocab_size), "--cl100k"]
    if transition_vocab:
        command += ["--superword-phase-vocab", str(transition_vocab)]
    return command
