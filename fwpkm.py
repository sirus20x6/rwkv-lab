"""Fast-weight Product Key Memory (arXiv:2601.00671) — a large, sparse-lookup key-value
memory drop-in to augment the model with episodic capacity.

This implements the RETRIEVAL architecture + the two memorization/addressing objectives:
  * Product keys: two sub-key codebooks K1,K2 of size sqrt(N) each; their Cartesian product
    gives N = sqrt(N)^2 keys without materializing them. A query is split in half, each half
    scored against its codebook, Top-k per half combined into a k*k candidate set, final Top-k
    selected, softmax-weighted over the selected value rows. Cost is O(sqrt(N)), not O(N).
  * IDW scoring: s = -log(eps + ||q_half - K||^2) (inverse-distance) instead of dot-product —
    pushes keys toward cluster centroids.
  * Gated residual: o = g * v_retrieved + (1-g) * v_projected, g = sigmoid(gate). The gate bias
    is init strongly negative so g~=0 and o_proj~=0 => the layer is a near-no-op at init.
  * Objectives (exposed as .last_mem_loss / .last_addr_loss for the trainer): a lookahead
    value-memorization MSE (write the next value into the touched slots) and an anti-collapse
    negative-entropy addressing loss on slot usage.

SCOPE / honest deviation: the paper's distinctive "fast weight" is a TEST-TIME-TRAINING update
that rewrites V and the keys IN-FORWARD per chunk by a unit gradient step. Here V and the keys
are ordinary trainable Parameters optimized by the OUTER optimizer via these same two objectives
(memorization + addressing) — the same product-key capacity and losses, without the fragile
in-forward TTT loop. The in-forward TTT variant is flagged as a deeper follow-up.
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class FwPKM(nn.Module):
    def __init__(self, d_model: int, *, sqrt_n: int = 64, d_key: int = 128, d_val: int | None = None,
                 topk: int = 8, gate_bias: float = -4.0):
        super().__init__()
        self.C = d_model
        self.sqrt_n = int(sqrt_n)
        self.n = self.sqrt_n ** 2
        self.dk = int(d_key)
        self.dk2 = self.dk // 2
        self.dv = int(d_val) if d_val else d_model
        self.k = int(topk)
        self.rms_q, self.rms_v, self.rms_g, self.rms_o = (nn.RMSNorm(d_model) for _ in range(4))
        self.q = nn.Linear(d_model, self.dk, bias=False)
        self.v = nn.Linear(d_model, self.dv, bias=False)
        self.g = nn.Linear(d_model, 1)
        self.o = nn.Linear(self.dv, d_model, bias=False)
        nn.init.constant_(self.g.bias, float(gate_bias))       # g ~= sigmoid(-4) ~ 0.018 at init
        nn.init.zeros_(self.o.weight)                          # near-no-op residual add at init
        self.K1 = nn.Parameter(torch.randn(self.sqrt_n, self.dk2) / math.sqrt(self.dk2))
        self.K2 = nn.Parameter(torch.randn(self.sqrt_n, self.dk2) / math.sqrt(self.dk2))
        self.V = nn.Parameter(torch.zeros(self.n, self.dv))    # value memory (episodic)
        self.last_mem_loss = None
        self.last_addr_loss = None

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        B, T, C = h.shape
        q = self.q(self.rms_q(h)).float()                      # [B,T,dk]
        v = self.v(self.rms_v(h))                              # [B,T,dv]
        g = torch.sigmoid(self.g(self.rms_g(h)))              # [B,T,1]
        q1, q2 = q[..., :self.dk2], q[..., self.dk2:]
        S = self.sqrt_n
        # IDW score per half: s = -log(eps + ||q - K||^2)
        s1 = -torch.log(1e-3 + torch.cdist(q1.reshape(-1, self.dk2), self.K1.float()).pow(2)).view(B, T, S)
        s2 = -torch.log(1e-3 + torch.cdist(q2.reshape(-1, self.dk2), self.K2.float()).pow(2)).view(B, T, S)
        sv1, i1 = s1.topk(self.k, dim=-1)                      # [B,T,k]
        sv2, i2 = s2.topk(self.k, dim=-1)
        cand = sv1[..., :, None] + sv2[..., None, :]           # [B,T,k,k] pairwise sums
        wsel, isel = cand.reshape(B, T, self.k * self.k).topk(self.k, dim=-1)   # final k pairs
        ii, jj = isel // self.k, isel % self.k                 # into i1 / i2
        gi = i1.gather(-1, ii); gj = i2.gather(-1, jj)         # global sub-key indices [B,T,k]
        rows = (gi * S + gj)                                   # flat V rows [B,T,k]
        w = torch.softmax(wsel, dim=-1)                        # [B,T,k]
        vsel = self.V[rows.reshape(-1)].view(B, T, self.k, self.dv)
        vhat = (w[..., None] * vsel).sum(-2)                   # [B,T,dv]
        o = g * vhat + (1.0 - g) * v
        out = self.o(self.rms_o(o))
        self._objectives(v, vhat, g, s1, s2, i1, i2, w)
        return out

    def _objectives(self, v, vhat, g, s1, s2, i1, i2, w):
        """Lookahead value-memorization MSE + anti-collapse addressing entropy (stored)."""
        if v.shape[1] < 2:
            self.last_mem_loss = self.last_addr_loss = None
            return
        vt = F.normalize(v[:, 1:].detach().float(), dim=-1)    # z-ish next-value target
        self.last_mem_loss = 0.5 * (g[:, :-1] * (vt - vhat[:, :-1].float()).pow(2)).mean()
        addr = v.new_zeros((), dtype=torch.float32)
        for sm, im in ((s1, i1), (s2, i2)):
            pk = torch.softmax(sm.gather(-1, im), dim=-1)      # top-k usage per token
            usage = torch.zeros(sm.shape[0], sm.shape[1], self.sqrt_n, device=sm.device)
            usage.scatter_(-1, im, pk)
            pbar = usage.mean(dim=(0, 1)).clamp_min(1e-9)      # marginal slot usage [sqrt_n]
            addr = addr + (pbar * pbar.log()).sum()            # = -entropy
        self.last_addr_loss = addr
