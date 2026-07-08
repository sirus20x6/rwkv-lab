"""llr.py — Heavy-Tail Layerwise Learning Rate (2605.22297).

Sets a per-param-GROUP LR multiplier from each group's spectral heavy-tailedness:
periodically fit a power law to the group's eigenvalue spectrum (ESD of WᵀW) via the
Hill estimator (PL_Alpha_Hill) and map the exponent α linearly across groups to a
multiplier in [1, s_max] — higher-α ("more trainable" / less-converged) groups get a
larger LR. The multiplier is written to g["llr_mult"]; the trainer applies it in its
per-group LR block (alongside lr_scale), so it composes with the schedule and never
fights it. Active only early in training (first `active_frac` of steps), refreshed
every `every` steps, with the spread `s_max` live-tunable.

True "layerwise" needs per-LAYER groups (the consolidation stage). In the single-layer
convert trainer the groups are the one layer's matrices, so it rebalances *within* the
layer — small effect, but the same code becomes full layerwise LR at consolidation.
Works with AdamW and Muon/SpectralMuon. Reversed mapping collapses, so the sign matters.
"""
from __future__ import annotations

import torch


def hill_alpha(svals, k_frac=0.5):
    """PL_Alpha_Hill on eigenvalues λ = σ² (top-k tail). α = 1 + k / Σ ln(λ_i/λ_k)."""
    ev = (svals.detach().float() ** 2)
    ev, _ = ev.sort(descending=True)
    n = ev.numel()
    if n < 4:
        return float("nan")
    k = max(2, int(k_frac * n))
    tail = ev[:k]
    lam_k = tail[-1].clamp_min(1e-12)
    s = torch.log(tail / lam_k).sum().clamp_min(1e-9)
    return float(1.0 + k / float(s))


class LayerwiseLR:
    def __init__(self, opt, s_max=5.0, every=200, active_frac=0.2, total_steps=1, k_frac=0.5):
        self.opt = opt
        self.s_max = s_max
        self.every = max(1, every)
        self.active_frac = active_frac
        self.total = max(1, total_steps)
        self.k_frac = k_frac
        for g in opt.param_groups:                 # default multiplier = 1 (no-op until set)
            g.setdefault("llr_mult", 1.0)

    @torch.no_grad()
    def update(self, step, s_max=None):
        if step > self.active_frac * self.total or (step % self.every) != 0:
            return
        s = self.s_max if s_max is None else s_max
        # aux-head groups (lookahead/rosa_soft) are not student layers: heavy-tail
        # layerwise scaling is meaningless there, and svdvals on the lm_head-sized
        # TOP matrix would cost minutes per update. rwkv_loop is exempt too: it has
        # its own live multiplier (loop_lr_mult) and now carries 2D tensors (LoRA
        # A/B, hyper_alpha) that would otherwise pick up a compounding llr_mult.
        # NaN -> llr_mult pinned to 1.0.
        alphas = [float("nan") if g.get("name") in ("lookahead", "rosa_soft", "rwkv_loop")
                  else self._group_alpha(g) for g in self.opt.param_groups]
        valid = [a for a in alphas if a == a]      # drop NaN
        if not valid:
            return
        amin, amax = min(valid), max(valid)
        for g, a in zip(self.opt.param_groups, alphas):
            if a != a or amax <= amin:
                g["llr_mult"] = 1.0
            else:
                g["llr_mult"] = 1.0 + (s - 1.0) * (a - amin) / (amax - amin)

    def _group_alpha(self, g):
        svs = []
        for p in g["params"]:
            if p.ndim == 2 and min(p.shape) > 1:
                try:
                    svs.append(torch.linalg.svdvals(p.detach().float()))
                except Exception:
                    pass
        if not svs:
            return float("nan")
        return hill_alpha(torch.cat(svs), self.k_frac)
