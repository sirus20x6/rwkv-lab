"""distill_objectives.py — alignment-invariant / relational distillation losses for
CROSS-ARCHITECTURE feature & state matching (transformer/GDN teacher -> RWKV student).

Pointwise MSE ‖h_S - h_T‖² assumes the student and teacher share a coordinate basis,
which different architectures do NOT (cross-arch hidden cosine ~0, 2606.06021 — OPRD: On-Policy Representation Distillation). These
terms instead match DIRECTION / STRUCTURE / DYNAMICS, so the basis/dim mismatch is
tolerated without a learned projector. All operate on [B,T,C] feature sequences (e.g.
the block outputs you already compute); add them as extra weighted terms beside your
block/SMT/DMT MSE (default weights 0 -> opt-in).

  cosine_match   1 - mean per-token cosine            (2602.05262 — ReGLA: Efficient Receptive-Field Modeling with Gated… / 2606.26488 — What Survives When You Compress a Recursive Reasoner for…)
  cka_loss       1 - linear CKA of feature Grams       (2606.05682 — Beyond Output Matching: Preserving Internal Geometry in…; dim-agnostic)
  flow_loss      PHF transition direction + Gram       (2606.29340; offset/scale/rot-inv)
  OPRDBridge     frozen low-rank PCA subspace match    (2606.06021)
  agreement_weight  trust-region per-token gating       (2606.01249)
  entropy_gated_kl  FKL where teacher uncertain, RKL else (2603.07079; needs teacher logits)
  carry_fidelity    label-free cosine drift monitor     (2606.26488 — What Survives When You Compress a Recursive Reasoner for…)
  taylor_calibrate_*  closed-form gate-init helpers      (2606.16429; arch-specific wiring)
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def _flat(x):                       # [B,T,C] -> [N,C]
    return x.reshape(-1, x.shape[-1])


def cosine_match(s, t, eps=1e-6):
    """1 - mean per-token cosine similarity. Direction-invariant, scale-robust."""
    s, t = _flat(s.float()), _flat(t.float())
    return (1.0 - F.cosine_similarity(s, t, dim=-1, eps=eps)).mean()


def cka_loss(s, t, eps=1e-6):
    """1 - linear CKA between feature Grams. Invariant to orthogonal transform + isotropic
    scaling, and works even if student/teacher feature dims differ (uses [C,C] Grams)."""
    X, Y = _flat(s.float()), _flat(t.float())
    X = X - X.mean(0, keepdim=True)
    Y = Y - Y.mean(0, keepdim=True)
    hsic_xy = (Y.T @ X).pow(2).sum()            # ||Y^T X||_F^2
    hsic_xx = (X.T @ X).pow(2).sum()
    hsic_yy = (Y.T @ Y).pow(2).sum()
    cka = hsic_xy / (hsic_xx.sqrt() * hsic_yy.sqrt() + eps)
    return 1.0 - cka


def flow_loss(s, t, eps=1e-6, max_gram=2048):
    """PHF (2606.29340): match how features MOVE along the sequence — transition-direction
    cosine + transition-Gram Frobenius. Invariant to per-token offset, per-step scale, and
    independent orthogonal transforms of the two trajectories ('move like the teacher')."""
    s, t = s.float(), t.float()
    dsn = F.normalize(_flat(s[:, 1:] - s[:, :-1]), dim=-1, eps=eps)
    dtn = F.normalize(_flat(t[:, 1:] - t[:, :-1]), dim=-1, eps=eps)
    dir_loss = (1.0 - (dsn * dtn).sum(-1)).mean()
    n = dsn.shape[0]
    if n > max_gram:                            # cap the [N,N] trajectory Gram
        idx = torch.randperm(n, device=dsn.device)[:max_gram]
        dsn, dtn = dsn[idx], dtn[idx]
    geo_loss = (dsn @ dsn.T - dtn @ dtn.T).pow(2).mean()
    return dir_loss + geo_loss


@torch.no_grad()
def agreement_weight(s, t, eps=1e-6):
    """Trust-region per-token weight (2606.01249 spirit): teacher-student cosine agreement,
    clamped to [0,1]. Down-weights diverged steps where the (rolled-out) target is
    unreliable, so compounding cross-arch mismatch doesn't dominate the gradient. [B,T,1]."""
    cos = F.cosine_similarity(s.float(), t.float(), dim=-1, eps=eps).clamp_min(0.0)
    return cos.unsqueeze(-1)


@torch.no_grad()
def carry_fidelity(s, t, eps=1e-6):
    """Label-free drift monitor (2606.26488 — What Survives When You Compress a Recursive Reasoner for…): cosine fidelity of the pooled feature
    trajectory. Monotonic with accuracy loss; ~<0.8 flags global-dynamics collapse even
    while pointwise block-MSE looks fine. Return a float for the dashboard."""
    s = _flat(s.float().mean(dim=1).unsqueeze(1))
    t = _flat(t.float().mean(dim=1).unsqueeze(1))
    return float(F.cosine_similarity(s, t, dim=-1, eps=eps).mean())


class OPRDBridge:
    """OPRD-Bridge (2606.06021): frozen low-rank PCA projectors map teacher & student
    features into a shared rank-r subspace; L2-normalize and match there. Projectors are
    fit ONCE (first call) then frozen — joint-training them hurts. Keep rank small (~8);
    higher rank degrades (bias-variance)."""

    def __init__(self, rank=8):
        self.rank = rank
        self.Pt = None
        self.Ps = None

    @torch.no_grad()
    def _fit(self, X):                          # [N,C] -> top-r right singular vectors [C,r]
        Xc = X - X.mean(0, keepdim=True)
        try:
            _, _, Vh = torch.linalg.svd(Xc, full_matrices=False)
            return Vh[: self.rank].T.contiguous()
        except Exception:
            return None

    def loss(self, s, t, eps=1e-6):
        sf, tf = _flat(s.float()), _flat(t.float())
        if self.Pt is None:
            self.Pt, self.Ps = self._fit(tf), self._fit(sf)
        if self.Pt is None or self.Ps is None:
            return s.new_zeros(())
        ps = F.normalize(sf @ self.Ps, dim=-1, eps=eps)
        pt = F.normalize(tf @ self.Pt, dim=-1, eps=eps)
        return (ps - pt).pow(2).mean()


def entropy_gated_kl(student_logits, teacher_logits, tau=0.8):
    """EOPD (2603.07079): forward-KL where the teacher is UNCERTAIN (entropy>tau, mode-
    covering -> preserves diversity), reverse-KL elsewhere (mode-seeking). Needs teacher
    logits (a second forward); provided for when you add the teacher-logit path."""
    sl, tl = student_logits.float(), teacher_logits.float()
    logp_s, logp_t = F.log_softmax(sl, -1), F.log_softmax(tl, -1)
    p_s, p_t = logp_s.exp(), logp_t.exp()
    H = -(p_t * logp_t).sum(-1)
    fkl = (p_t * (logp_t - logp_s)).sum(-1)
    rkl = (p_s * (logp_s - logp_t)).sum(-1)
    return torch.where(H > tau, fkl, rkl).mean()


# --- Taylor-Calibrate init helpers (2606.16429) -----------------------------------
# Closed-form gate initialization from TEACHER attention statistics, so the student's
# recurrent dynamics start in the right regime and the MSE has less to repair (88x better
# worst-case init PPL). These return the *values* to write into the student's gates; the
# WIRING (which RWKV time-mix params are the decay / write / output gates, and how to read
# the teacher's attention distance/entropy) is architecture-specific — call from build().
def taylor_calibrate_decay_bias(mean_attn_distance):
    """dt_bias = softplus^{-1}(ln2 / d): set the student's decay timescale so its half-life
    matches the teacher's mean attention distance d."""
    d = max(float(mean_attn_distance), 1.0)
    target = math.log(2.0) / d
    return math.log(math.expm1(target))         # softplus^{-1}(target)


def taylor_calibrate_value_rescale(y_teacher, y_student, eps=1e-8):
    """OLS rescale for the value/output projection: sigma* = <y_T,y_S>/<y_S,y_S>."""
    num = float((y_teacher.float() * y_student.float()).sum())
    den = float((y_student.float() ** 2).sum()) + eps
    return num / den
