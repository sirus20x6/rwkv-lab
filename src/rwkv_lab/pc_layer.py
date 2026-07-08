"""pc_layer.py — PC-Layer polynomial weight preconditioning (2606.06470).

A WEIGHT REPARAMETERIZATION (not an optimizer): in the forward pass the raw weight
W is replaced by g(W) = norm_recover(soft_flatten(W)), which softly reshapes the
singular-value spectrum (lifts small σ, saturates large σ toward 1) WITHOUT an SVD —
just matmuls (an odd polynomial in W·Wᵀ). Gradients flow to the raw W; the
norm-recovery rescale is stop-grad (per the paper). It is optimizer-orthogonal
(stacks with AdamW or SpectralMuon) and MERGEABLE at inference (torch.nn.utils
.parametrize.remove_parametrizations bakes g(W) into the weight → zero inference cost).

Live: a single shared `strength` (a 1-element list) blends g(W) with W as
  W_eff = (1-s)·W + s·g(W),  s in [0,1]  (s=0 → identity / off).
Set strength[0] from the trainer (launch arg or live control).
"""
from __future__ import annotations

import torch
from torch import nn

try:
    from torch.nn.utils import parametrize
except Exception:  # pragma: no cover
    parametrize = None


def _spectral_norm(W, iters=4):
    v = torch.randn(W.size(1), device=W.device, dtype=W.dtype)
    for _ in range(iters):
        u = W @ v
        u = u / (u.norm() + 1e-12)
        v = W.mT @ u
        v = v / (v.norm() + 1e-12)
    return (W @ v).norm()


def pc_transform(W, level):
    """g(W): soft spectral flattening + Frobenius-norm recovery (stop-grad scale)."""
    Wf = W.float()
    with torch.no_grad():
        nrm = _spectral_norm(Wf)
    X = Wf / (nrm + 1e-7)
    a, b, c = 3.4445, -4.7750, 2.0315
    transpose = X.size(0) > X.size(1)
    if transpose:
        X = X.mT
    for _ in range(max(1, int(level))):            # level light flatten steps (degree grows with level)
        A = X @ X.mT
        X = a * X + (b * A + c * (A @ A)) @ X
    if transpose:
        X = X.mT
    with torch.no_grad():                          # norm-recovery (stop-grad), per the paper
        scale = Wf.norm() / (X.norm() + 1e-7)
    return (X * scale).to(W.dtype)


class _PCParam(nn.Module):
    def __init__(self, level, strength_ref):
        super().__init__()
        self.level = level
        self.strength_ref = strength_ref           # 1-element list: live strength in [0,1]

    def forward(self, W):
        s = float(self.strength_ref[0])
        if W.ndim != 2 or s <= 0.0:
            return W
        g = pc_transform(W, self.level)
        return (1.0 - s) * W + s * g

    def right_inverse(self, W):                    # identity init so registration is lossless
        return W


def apply_pc_layer(student, level=2):
    """Register PC-Layer on every 2D nn.Linear weight in `student`. Returns
    (strength_ref, n_applied); set strength_ref[0] (launch or live) to enable."""
    if parametrize is None:
        return [0.0], 0
    strength_ref = [0.0]
    n = 0
    for m in student.modules():
        if isinstance(m, nn.Linear) and getattr(m, "weight", None) is not None and m.weight.ndim == 2:
            try:
                parametrize.register_parametrization(m, "weight", _PCParam(level, strength_ref))
                n += 1
            except Exception:
                pass
    return strength_ref, n
