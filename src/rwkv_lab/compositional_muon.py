"""Partner-whitened Compositional Muon for QK/OV weight pairs.

Tilde Research (2026), https://github.com/tilde-research/comp-muon-release.
Community lead: https://discord.com/channels/992359628979568762/992362722035507270/1512505466549174352
"""
from __future__ import annotations
import torch


def inverse_gram_root(weight: torch.Tensor, damping: float = 1e-4) -> torch.Tensor:
    gram = weight.float().T @ weight.float()
    values, vectors = torch.linalg.eigh(gram + damping * torch.eye(gram.shape[0], device=gram.device))
    return (vectors * values.clamp_min(damping).rsqrt()) @ vectors.T


def matrix_sign(value: torch.Tensor) -> torch.Tensor:
    u, _, vh = torch.linalg.svd(value.float(), full_matrices=False)
    return u @ vh


@torch.no_grad()
def compositional_pair_update(left: torch.Tensor, right: torch.Tensor,
                              left_grad: torch.Tensor, right_grad: torch.Tensor, *,
                              lr: float, damping: float = 1e-4) -> dict:
    """Update factors whose effective operator is ``left @ right.T``."""
    c_right, c_left = inverse_gram_root(right, damping), inverse_gram_root(left, damping)
    dl = -0.5 * lr * matrix_sign(left_grad.float() @ c_right) @ c_right
    dr = -0.5 * lr * matrix_sign(right_grad.float() @ c_left) @ c_left
    before = left.float() @ right.float().T
    left.add_(dl.to(left.dtype)); right.add_(dr.to(right.dtype))
    change = left.float() @ right.float().T - before
    return {"schema": "rwkv-lab.compositional-muon.v1", "operator_update_norm": float(change.norm()),
            "left_update_norm": float(dl.norm()), "right_update_norm": float(dr.norm())}
