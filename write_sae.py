"""WriteSAE — Sparse Autoencoders for Recurrent State (arXiv:2605.12770, Young).

A diagnostic that decomposes a linear-attention model's accumulated per-head recurrent STATE
`S_t ∈ R^{d_v×d_k}` (the matrix cache, e.g. RWKV-7 / GDN wkv state) into a sparse set of interpretable
features — and, crucially, features shaped like the model's WRITE so they can be substituted back into
the cache for causal analysis. This is an analysis/interpretability tool, NOT a capability lever: it
tells you what the state stores; it does not change the model.

What makes it recurrence-specific (vs a plain activation SAE), all three from the paper:
  1. Rank-`R` matrix decoder atoms `Σ_r v_ir w_irᵀ`, shaped like the native write (`k v^T`). GDN writes
     rank-1; RWKV-7 writes rank-2 — so `atom_rank` is configurable (default 1 = paper's primary cell).
  2. Bilinear matched-filter encoder `a_i = Σ_r v_irᵀ S w_ir`: the read direction IS the write direction.
  3. Matched-Frobenius-norm cache substitution: at a firing, swap the native write for the atom rescaled
     to the native write's norm (`causal_substitute`), so an intervention is norm-preserving.

Sparsity is a hard TopK (k=32 in the paper) + an AuxK dead-feature revival term (the paper's exact
`L_dead` is unstated; we use the standard OpenAI-SAE AuxK: dead atoms reconstruct the residual). The
mean-state estimator `M` (also unstated in the paper) is a running mean, updated via `update_mean`.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class WriteSAE(nn.Module):
    def __init__(self, d_v: int, d_k: int, n_feat: int = 2048, k: int = 32,
                 atom_rank: int = 1, k_aux: int = 64, dead_after: int = 100, aux_weight: float = 1e-2):
        super().__init__()
        self.d_v, self.d_k, self.n_feat = int(d_v), int(d_k), int(n_feat)
        self.k, self.k_aux = int(k), int(k_aux)
        self.atom_rank = int(atom_rank)
        self.dead_after = int(dead_after)
        self.aux_weight = float(aux_weight)
        R = self.atom_rank
        self.V = nn.Parameter(torch.randn(n_feat, R, d_v) / (d_v ** 0.5))   # atom left factors v_ir
        self.W = nn.Parameter(torch.randn(n_feat, R, d_k) / (d_k ** 0.5))   # atom right factors w_ir
        self.register_buffer("M", torch.zeros(d_v, d_k))                    # running mean state
        self.register_buffer("_mean_n", torch.zeros(()))
        self.register_buffer("steps_since_fired", torch.zeros(n_feat, dtype=torch.long))

    # --- mean-state estimator (paper's M; estimator unstated -> running mean) ---
    @torch.no_grad()
    def update_mean(self, S: torch.Tensor):
        b = S.reshape(-1, self.d_v, self.d_k).float()
        n = b.shape[0]
        self._mean_n += n
        self.M += (b.sum(0) - self.M * n) / self._mean_n.clamp_min(1)

    def _scores(self, S: torch.Tensor) -> torch.Tensor:
        # bilinear matched filter: a_i = Σ_r v_irᵀ S w_ir  (read dir == write dir)
        return torch.einsum("nrd,bde,nre->bn", self.V, S, self.W)

    def encode(self, S: torch.Tensor, track: bool = False):
        """S [..., d_v, d_k] -> (a_sparse [.., n_feat], all_scores). Hard TopK-k over features."""
        lead = S.shape[:-2]
        Sf = (S.reshape(-1, self.d_v, self.d_k).float() - self.M)
        a = self._scores(Sf)                                               # [N, n_feat]
        topv, topi = a.topk(self.k, dim=-1)
        a_sparse = torch.zeros_like(a).scatter(-1, topi, topv)
        if track and self.training:
            with torch.no_grad():
                self.steps_since_fired += 1
                self.steps_since_fired[topi.reshape(-1).unique()] = 0
        return a_sparse.reshape(*lead, self.n_feat), a.reshape(*lead, self.n_feat)

    def decode(self, a_sparse: torch.Tensor) -> torch.Tensor:
        """a_sparse [..., n_feat] -> reconstructed centered state [..., d_v, d_k]."""
        lead = a_sparse.shape[:-1]
        a = a_sparse.reshape(-1, self.n_feat)
        Shat = torch.einsum("bn,nrd,nre->bde", a, self.V, self.W)
        return Shat.reshape(*lead, self.d_v, self.d_k)

    def loss(self, S: torch.Tensor):
        """TopK reconstruction of the mean-centered state + AuxK dead-feature revival."""
        Sf = S.reshape(-1, self.d_v, self.d_k).float()
        x = Sf - self.M
        a_sparse, a_all = self.encode(Sf, track=True)
        recon = self.decode(a_sparse)
        resid = x - recon
        l_recon = resid.pow(2).flatten(1).sum(-1).mean()
        # AuxK: let atoms silent > dead_after steps reconstruct the residual (revives dead features)
        l_aux = x.new_zeros(())
        dead = self.steps_since_fired > self.dead_after
        if dead.any() and self.k_aux > 0:
            a_dead = a_all.masked_fill(~dead.unsqueeze(0), float("-inf"))
            kk = min(self.k_aux, int(dead.sum()))
            tv, ti = a_dead.topk(kk, dim=-1)
            tv = torch.nan_to_num(tv, neginf=0.0)
            aux = torch.zeros_like(a_all).scatter(-1, ti, tv)
            l_aux = (resid.detach() - self.decode(aux)).pow(2).flatten(1).sum(-1).mean()
        return l_recon + self.aux_weight * l_aux, {"recon": float(l_recon.detach()),
                                                   "aux": float(l_aux.detach()),
                                                   "alive": int((~dead).sum())}

    @torch.no_grad()
    def atom(self, i: int) -> torch.Tensor:
        """The rank-R matrix atom Σ_r v_ir w_irᵀ [d_v, d_k]."""
        return torch.einsum("rd,re->de", self.V[i], self.W[i])

    @torch.no_grad()
    def causal_substitute(self, S_prev: torch.Tensor, native_write: torch.Tensor, atom_idx: int):
        """Matched-Frobenius-norm cache substitution: replace the native write with atom `atom_idx`
        rescaled to the native write's norm. Returns the new state S_prev + σ·atom (norm-preserving)."""
        A = self.atom(atom_idx)
        sigma = native_write.norm() / A.norm().clamp_min(1e-8)
        return S_prev + sigma * A
