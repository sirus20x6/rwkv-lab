"""Auxiliary objectives that make caption training depend on the image."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class ImageTextContrastiveHead(nn.Module):
    """Align projected image prefixes with their reference-caption token bags.

    Every other caption in the batch is an explicit shuffled-image negative.
    Caption targets use the frozen RWKV embedding table and are detached, so
    the auxiliary cannot improve by rewriting the language model.
    """

    def __init__(self, hidden_size: int, width: int = 512, temperature: float = 0.07):
        super().__init__()
        if width < 1 or temperature <= 0:
            raise ValueError("invalid image-text contrastive geometry")
        self.temperature = float(temperature)
        self.vision_norm = nn.LayerNorm(hidden_size)
        self.text_norm = nn.LayerNorm(hidden_size)
        self.vision_projection = nn.Linear(hidden_size, width, bias=False)
        self.text_projection = nn.Linear(hidden_size, width, bias=False)

    def forward(self, prefix: torch.Tensor, target_embeddings: torch.Tensor,
                batch_positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch = prefix.shape[0]
        if batch < 2:
            zero = prefix.new_zeros((), dtype=torch.float32)
            return zero, zero
        text_sum = target_embeddings.new_zeros(batch, target_embeddings.shape[-1])
        text_sum.index_add_(0, batch_positions, target_embeddings)
        counts = torch.bincount(batch_positions, minlength=batch).clamp_min(1)
        text_mean = text_sum / counts[:, None].to(text_sum.dtype)
        vision_mean = prefix.mean(dim=1)
        image_vectors = F.normalize(
            self.vision_projection(self.vision_norm(vision_mean)).float(), dim=-1)
        text_vectors = F.normalize(
            self.text_projection(self.text_norm(text_mean)).float(), dim=-1)
        similarities = image_vectors @ text_vectors.transpose(0, 1) / self.temperature
        targets = torch.arange(batch, device=similarities.device)
        loss = 0.5 * (
            F.cross_entropy(similarities, targets)
            + F.cross_entropy(similarities.transpose(0, 1), targets))
        accuracy = 0.5 * (
            (similarities.argmax(-1) == targets).float().mean()
            + (similarities.argmax(0) == targets).float().mean())
        return loss, accuracy


def early_token_weights(full_labels: torch.Tensor, batch_positions: torch.Tensor,
                        sequence_positions: torch.Tensor, *, token_count: int,
                        weight: float) -> torch.Tensor | None:
    """Return per-target weights for the grounding-critical caption opening."""
    if token_count <= 0 or weight == 1.0:
        return None
    supervised = full_labels != -100
    first_targets = supervised.to(torch.int64).argmax(dim=1)
    target_positions = sequence_positions + 1
    offsets = target_positions - first_targets[batch_positions]
    early = (offsets >= 0) & (offsets < token_count)
    return torch.where(
        early,
        torch.full_like(offsets, float(weight), dtype=torch.float32),
        torch.ones_like(offsets, dtype=torch.float32),
    )
