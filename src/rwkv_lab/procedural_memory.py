"""Neural Procedural Memory via contrastive activation steering.

Zhao et al. (2026), https://arxiv.org/abs/2606.29824. Community lead:
https://discord.com/channels/992359628979568762/992362722035507270/1521742648812244994
"""
from __future__ import annotations
from dataclasses import dataclass
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class Procedure:
    key: torch.Tensor
    steering: torch.Tensor
    source: str


class ProceduralMemory:
    def __init__(self, procedures: list[Procedure] | None = None): self.procedures = procedures or []

    def add_contrast(self, key: torch.Tensor, successful: torch.Tensor,
                     failed: torch.Tensor, *, source: str) -> None:
        if successful.shape != failed.shape: raise ValueError("contrast activations must match")
        direction = (successful.float().mean(0) - failed.float().mean(0))
        self.procedures.append(Procedure(F.normalize(key.float(), dim=-1), direction, source))

    def retrieve(self, query: torch.Tensor, top_k: int = 4) -> list[Procedure]:
        if not self.procedures: return []
        keys = torch.stack([p.key.to(query.device) for p in self.procedures])
        score = F.normalize(query.float(), dim=-1) @ keys.T
        return [self.procedures[int(i)] for i in score.topk(min(top_k, len(self.procedures))).indices]

    def synthesize(self, query: torch.Tensor, top_k: int = 4) -> torch.Tensor:
        selected = self.retrieve(query, top_k)
        if not selected: raise ValueError("procedural memory is empty")
        return torch.stack([p.steering.to(query.device) for p in selected]).mean(0)

    def intervene(self, hidden: torch.Tensor, query: torch.Tensor, *, strength: float = 1.0,
                  top_k: int = 4) -> torch.Tensor:
        return hidden + strength * self.synthesize(query, top_k).to(hidden.dtype)
