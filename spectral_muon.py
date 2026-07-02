"""spectral_muon.py — one configurable Muon-family optimizer collecting the 2026
spectral-optimizer levers as flags. 2D weight matrices get the Muon update; all
other params (norms/biases/embeddings/scalars) use a built-in AdamW fallback.
Routing is by per-group `use_muon` (matches this repo's GuardedMuonClip convention).

Defaults reproduce vanilla Muon (orthogonalize momentum via Newton–Schulz → UVᵀ).
Per 2D matrix each step:
  1. momentum   m = μ·m + g'            (g' = g + α·EMA(Δg) if MONA)        [2605.26842]
  2. Muon²      m ← m / (√v + ε)        Adam-style precond before NS         [2604.09967]
  3. MuonEq     equilibrate rows/cols of m                                   [2603.28254]
  4. orthog.    p==0 → Newton–Schulz polar (UVᵀ);  p∈(0,1] → U·Σ^p·Vᵀ (eigh)  [2606.13867]
  5. Aurora     equal-row-norm for tall matrices                            [2606.27715]
  6. MUON+      row/col-normalize the orthogonalized update                 [2602.21545]
  7. scale by `scale·√(max(m,n))` (the repo's MuonClip amplifier) and lr; decoupled WD
`cubic=True` uses the odd-cubic NS schedule (~1/3 fewer matmuls).            [2606.00371]

The lever knobs live in each param group, so a trainer may live-tune them by
setting e.g. opt.param_groups[i]["spectral_power"] = v between steps.

Precision: all optimizer state (momentum, MONA acc, Muon² v, Adam moments) is kept
in fp32 regardless of param dtype — bf16 moments quantize away small updates (the
known bf16-AdamW failure). The NS polar iteration runs in bf16 (KJ Muon convention,
~2x faster than fp32+TF32); the eigh/SVD power paths need fp32 (cuSOLVER).
"""
from __future__ import annotations

import torch
from torch.optim.optimizer import Optimizer

_QUINTIC = (3.4445, -4.7750, 2.0315)  # Keller-Jordan Muon NS coefficients


def _ns_quintic(X, steps):
    a, b, c = _QUINTIC
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    return X


def _ns_cubic(X, steps):
    # odd cubic: 2 matmuls/step vs 3 for quintic (~1/3 cheaper) [2606.00371]
    a, b = 1.5, -0.5
    for _ in range(steps):
        X = a * X + b * ((X @ X.mT) @ X)
    return X


@torch.no_grad()
def _power_eigh(G, power, rtol=1e-3):
    """U.Sigma^p.Vt via eigh of the smaller-dim symmetric Gram (a matmul + symmetric
    eigensolver instead of a full SVD of G). Identity: U.Sigma^p.Vt = G.(GtG)^((p-1)/2).
    Forming the Gram squares the condition number, so directions with Sigma below
    rtol*Sigma_max sit beneath the eigh noise floor and are ZEROED — mirroring SVD's
    Sigma^p->0 on the null space. Without that, eigh noise on those directions gets
    amplified by the negative inner power Sigma^(p-1) and the update explodes (a dense
    full-rank momentum is unaffected; this is the robustness guard for ill-conditioned /
    rank-deficient G). Solves GtG when n<=m else GGt — the smaller dimension, so the
    rank-r low-rank factors cost only an r x r eigh."""
    m, n = G.shape[-2], G.shape[-1]
    if n <= m:
        evals, V = torch.linalg.eigh(G.mT @ G)            # GtG = V Sigma^2 Vt  (n x n)
        s = evals.clamp_min(0.0).sqrt()                   # Sigma
        inner = torch.where(s > rtol * s.amax(), s.pow(power - 1.0), s.new_zeros(()))
        return G @ ((V * inner) @ V.mT)                   # U Sigma . Sigma^(p-1) Vt = U Sigma^p Vt
    evals, U = torch.linalg.eigh(G @ G.mT)                # GGt = U Sigma^2 Ut  (m x m)
    s = evals.clamp_min(0.0).sqrt()
    inner = torch.where(s > rtol * s.amax(), s.pow(power - 1.0), s.new_zeros(()))
    return ((U * inner) @ U.mT) @ G


_power_fallback_warned = [False]


def orthogonalize(G, steps=5, cubic=False, power=0.0, power_method="eigh"):
    """(Fractional-power) orthogonalized factor of G. power>0 uses the math-identical
    eigh-on-Gram path (power_method 'eigh', default — see _power_eigh; measured ~14x
    faster than gesvd at 4096², 161ms vs 2.3s on Blackwell) OR exact gesvd ('svd',
    debug/verification only). The NS polar path runs in bf16 (~2x faster than
    fp32+TF32); the power paths need fp32 (cuSOLVER)."""
    if power and power > 0.0:
        try:
            if power_method == "eigh":
                return _power_eigh(G.float(), power)
            U, S, Vh = torch.linalg.svd(G.float(), full_matrices=False)
            return (U * S.clamp_min(0).pow(power)) @ Vh
        except Exception as e:  # solver failure -> NS polar (spectral_power OFF for this matrix/step)
            if not _power_fallback_warned[0]:
                _power_fallback_warned[0] = True
                print(f"[spectral_muon] WARNING: power_method={power_method!r} failed ({e!r}); "
                      "falling back to plain NS polar — spectral_power is silently ignored "
                      "wherever this recurs (warning printed once).", flush=True)
    X = G.bfloat16()
    transpose = X.size(-2) > X.size(-1)
    if transpose:
        X = X.mT
    X = X / (X.norm() + 1e-7)
    X = _ns_cubic(X, steps) if cubic else _ns_quintic(X, steps)
    if transpose:
        X = X.mT
    return X


def _rms(x, dim):
    return x.pow(2).mean(dim=dim, keepdim=True).clamp_min(1e-12).sqrt()


def _ddc_project(U, W, mode, strength):
    """Dead-Direction Conditioner (abelian subset, 2606.29176): remove the part of the
    update U that merely RESCALES channels of W — the per-channel gauge / "dead"
    directions where the loss is flat. 'row' = output-channel scale (RMSNorm-after /
    next-layer rescale gauge), 'col' = input-channel scale (RMSNorm-before), 'both' =
    both. Keeps the step on the loss-relevant quotient (resists over-training drift into
    degenerate flat minima). strength in [0,1] = fraction of the gauge component removed."""
    out = U
    if "row" in mode or mode == "both":
        Wn = W / (W.norm(dim=1, keepdim=True) + 1e-8)
        out = out - strength * (out * Wn).sum(dim=1, keepdim=True) * Wn
    if "col" in mode or mode == "both":
        Wn = W / (W.norm(dim=0, keepdim=True) + 1e-8)
        out = out - strength * (out * Wn).sum(dim=0, keepdim=True) * Wn
    return out


class SpectralMuon(Optimizer):
    def __init__(self, param_groups, *, momentum=0.95, nesterov=False,
                 ns_steps=5, cubic=False, spectral_power=0.0, power_method="eigh",
                 second_moment=False, sm_beta2=0.99, sm_eps=1e-8,
                 equilibrate="none", plus_norm="none", row_uniform=False,
                 mona=False, mona_beta=0.9, mona_alpha=0.1, scale=0.4,
                 ddc_strength=0.0, ddc_mode="both",
                 weight_decay=0.0, adam_betas=(0.9, 0.95), adam_eps=1e-8):
        defaults = dict(momentum=momentum, nesterov=nesterov, ns_steps=ns_steps, cubic=cubic,
                        spectral_power=spectral_power, power_method=power_method, second_moment=second_moment,
                        sm_beta2=sm_beta2, sm_eps=sm_eps, equilibrate=equilibrate,
                        plus_norm=plus_norm, row_uniform=row_uniform, mona=mona,
                        mona_beta=mona_beta, mona_alpha=mona_alpha, scale=scale,
                        ddc_strength=ddc_strength, ddc_mode=ddc_mode,
                        weight_decay=weight_decay, adam_betas=adam_betas, adam_eps=adam_eps,
                        use_muon=False, lr=3e-4)
        super().__init__(param_groups, defaults)

    def load_state_dict(self, state_dict):
        # Optimizer.load_state_dict casts float state to each param's dtype (bf16 for a
        # bf16 model), silently re-quantizing the fp32 state on every resume; undo it.
        # Old bf16-state ckpts upcast losslessly through the same path.
        super().load_state_dict(state_dict)
        for st in self.state.values():
            for k, v in st.items():
                if torch.is_tensor(v) and v.is_floating_point() and v.dtype != torch.float32:
                    st[k] = v.float()

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for grp in self.param_groups:
            for p in grp["params"]:
                if p.grad is None:
                    continue
                if grp.get("use_muon") and p.ndim == 2 and min(p.shape) > 1:
                    self._muon_step(p, p.grad, grp, self.state[p])
                else:
                    self._adam_step(p, p.grad, grp, self.state[p])
        return loss

    def _muon_step(self, p, g, grp, st):
        lr, mu = grp["lr"], grp["momentum"]
        if "mom" not in st:
            st["mom"] = torch.zeros_like(g, dtype=torch.float32)  # fp32 state: see module docstring
            if grp["mona"]:
                st["gprev"] = torch.zeros_like(g, dtype=torch.float32)
                st["acc"] = torch.zeros_like(g, dtype=torch.float32)
            if grp["second_moment"]:
                st["v"] = torch.zeros_like(g, dtype=torch.float32)
        gg = g
        if grp["mona"]:                                   # MONA curvature/Nesterov term
            d = g - st["gprev"]; st["gprev"].copy_(g)
            st["acc"].mul_(grp["mona_beta"]).add_(d, alpha=1 - grp["mona_beta"])
            gg = g + grp["mona_alpha"] * st["acc"]
        m = st["mom"]; m.mul_(mu).add_(gg)
        u = gg.add(m, alpha=mu) if grp["nesterov"] else m
        if grp["second_moment"]:                          # Muon²
            v = st["v"]; v.mul_(grp["sm_beta2"]).addcmul_(gg, gg, value=1 - grp["sm_beta2"])
            u = u / (v.sqrt() + grp["sm_eps"])
        eq = grp["equilibrate"]                           # MuonEq (pre-orthogonalization)
        if "R" in eq:
            u = u / _rms(u, 1)
        if "C" in eq:
            u = u / _rms(u, 0)
        o = orthogonalize(u, steps=grp["ns_steps"], cubic=grp["cubic"],
                          power=grp["spectral_power"], power_method=grp["power_method"]).to(p.dtype)
        if grp["row_uniform"] and o.size(0) >= o.size(1):  # Aurora (tall matrices)
            o = o / _rms(o, 1)
        pn = grp["plus_norm"]                              # MUON+ (post-orthogonalization)
        if pn == "row":
            o = o / _rms(o, 1)
        elif pn == "col":
            o = o / _rms(o, 0)
        if grp["ddc_strength"] > 0.0:                      # DDC: project out the rescale gauge
            o = _ddc_project(o, p, grp["ddc_mode"], grp["ddc_strength"])
        scale = grp["scale"] * (max(p.shape) ** 0.5)
        if grp["weight_decay"]:
            p.mul_(1.0 - lr * grp["weight_decay"])
        p.add_(o, alpha=-lr * scale)

    def _adam_step(self, p, g, grp, st):
        b1, b2 = grp["adam_betas"]; eps = grp["adam_eps"]; lr = grp["lr"]
        if "exp_avg" not in st:
            st["exp_avg"] = torch.zeros_like(g, dtype=torch.float32)  # fp32: bf16 moments
            st["exp_sq"] = torch.zeros_like(g, dtype=torch.float32)   # quantize away fine updates
            st["t"] = 0
        st["t"] += 1; t = st["t"]
        ea, es = st["exp_avg"], st["exp_sq"]
        ea.mul_(b1).add_(g, alpha=1 - b1)
        es.mul_(b2).addcmul_(g, g, value=1 - b2)
        denom = (es.sqrt() / (1 - b2 ** t) ** 0.5).add_(eps)
        upd = (ea / (1 - b1 ** t)) / denom
        if grp["ddc_strength"] > 0.0 and p.ndim == 2 and min(p.shape) > 1:
            upd = _ddc_project(upd, p, grp["ddc_mode"], grp["ddc_strength"])
        if grp["weight_decay"]:
            p.mul_(1.0 - lr * grp["weight_decay"])
        p.add_(upd, alpha=-lr)
