"""Taylor-Calibrate (arXiv:2606.16429) — principled init for attention→RWKV-7 conversion.

The paper derives a closed-form init for a GATED-DELTANET student from softmax-attention
teacher statistics. Its decay/write-gate formulas are GDN-parameterization-specific; the
authors note only the value-side amplitude match transfers architecture-agnostically. Here
we port the two parts that map cleanly onto our RWKV-7 core (rwkv8_deltanet):

  1. HALF-LIFE DECAY (the portable, principled part). Set the per-head decay bias w0 so the
     RWKV state's memory half-life ≈ the teacher head's average attention look-back distance
     d_h. RWKV per-step decay at init (w1=0 ⇒ data-independent) is
         decay = exp(-exp(w)),   w = -softplus(-w0) - 0.5.
     Want decay^{d_h} = 1/2 ⇒ decay = 2^{-1/d_h} ⇒ exp(w) = ln2/d_h ⇒ w* = log(ln2/d_h),
     then invert the parametrization:  w0 = -log(exp(-w* - 0.5) - 1).
     (Valid for d_h > ~1.14, i.e. always in practice; the −0.5 offset only caps the SHORTEST
     representable half-life, not the longest.)

  2. VALUE RMS-MATCH (adaptation of the paper's §A.5 per-head OLS). The exact OLS assumes a
     GDN value path; our value contributes through the whole wkv7 recurrence, so we instead
     scale value.weight per head so the block output RMS matches the teacher's — an amplitude
     calibration in the same spirit, honestly not the closed-form OLS.

Both are init-only helpers; run them after RADLADS weight transfer, before distillation.
NOTE: the paper reports Phase-1 init ALONE is weak — its value is a better *starting point*
for the block/logit distillation we already run, not a standalone conversion.
"""
from __future__ import annotations

import math
import torch


@torch.no_grad()
def teacher_lookback_distance(attn_probs: torch.Tensor, eps: float = 1e-9):
    """Average causal look-back distance per head from teacher attention probabilities.

    attn_probs: [B, H, T, T] causal, rows sum to 1 (softmax over keys s<=t).
    Returns d_h: [H] = E_{b,t}[ sum_s a_{t,s} (t - s) ].
    """
    B, H, T, _ = attn_probs.shape
    t_idx = torch.arange(T, device=attn_probs.device)
    dist = (t_idx.view(T, 1) - t_idx.view(1, T)).clamp_min(0).to(attn_probs.dtype)  # [T,T] (t-s)
    per = (attn_probs * dist).sum(-1)                       # [B,H,T]
    return per.mean(dim=(0, 2)).clamp_min(1.0 + 1e-3)       # [H]; >1 so the log is finite


@torch.no_grad()
def set_halflife_decay(core, d_h: torch.Tensor) -> torch.Tensor:
    """Set core.w0 [1,1,C] so head h's memory half-life ≈ d_h[h]. Returns the w0 values [H]."""
    H, N = core.num_heads, core.head_size
    d = d_h.to(torch.float32).clamp_min(1.2)               # keep exp(-w*-0.5)-1 > 0
    w_star = torch.log(math.log(2.0) / d)                  # target log-decay, < -0.5
    w0 = -torch.log(torch.expm1(-w_star - 0.5))            # invert w = -softplus(-w0)-0.5
    core.w0.data.copy_(w0.repeat_interleave(N).view(1, 1, H * N).to(core.w0.dtype))
    return w0


@torch.no_grad()
def value_rms_match(core, teacher_out: torch.Tensor, student_out: torch.Tensor,
                    lo: float = 0.2, hi: float = 5.0) -> torch.Tensor:
    """Per-head amplitude calibration: scale core.value.weight so the student block output RMS
    matches the teacher's. teacher_out/student_out: [B, T, C] block outputs on the SAME input.
    Returns the per-head scale [H]. Clipped to [lo, hi] (paper leaves clip bounds unspecified)."""
    H, N = core.num_heads, core.head_size
    def _rms_per_head(y):
        yh = y.float().reshape(-1, H, N)
        return yh.pow(2).mean(dim=(0, 2)).clamp_min(1e-12).sqrt()   # [H]
    scale = (_rms_per_head(teacher_out) / _rms_per_head(student_out)).clamp(lo, hi)  # [H]
    w = core.value.weight.data                             # [C_out=H*N, C_in]
    w.mul_(scale.repeat_interleave(N).view(-1, 1).to(w.dtype))
    return scale


@torch.no_grad()
def taylor_calibrate(core, attn_probs, teacher_out=None, student_out=None):
    """Run the portable Taylor-Calibrate init on an RWKV core. attn_probs drives the half-life
    decay; if teacher_out+student_out are given, also RMS-match the value path. Returns a dict."""
    d_h = teacher_lookback_distance(attn_probs)
    w0 = set_halflife_decay(core, d_h)
    info = {"d_h_mean": float(d_h.mean()), "d_h_min": float(d_h.min()), "d_h_max": float(d_h.max())}
    if teacher_out is not None and student_out is not None:
        s = value_rms_match(core, teacher_out, student_out)
        info["value_scale_mean"] = float(s.mean())
    return info
