"""grokking_metrics.py — flag-gated diagnostics for *memorization vs grokking*.

These feed the trainboard dashboard via the existing JSONL sink: any key you add
to a `{"kind":"train"|"eval", ...}` record is auto-stashed in `extra_json` and is
queryable with no schema change. The dashboard panels/alerts look for:

  train: rosa_inj_rms, engram_inj_rms, rosa_e_gap, stable_rank, wnorm_rms
  eval : block_val, gen_gap

Design rules (a diagnostic must never break a run):
  * every public fn is pure (no I/O) and wrapped to return None/nan on failure;
  * everything runs under no_grad and casts to fp32 (bf16-safe);
  * the heavy one (stable_rank, an SVD) is meant for a COARSE cadence, not per step.

Typical use in a trainer:

    import grokking_metrics as gm
    if args.log_grokking_metrics:
        rec.update(gm.injection_stats(rosa_inj=ros, engram_inj=eng, ref=x))
        if args.grok_spec_every and step % args.grok_spec_every == 0:
            rec["stable_rank"] = gm.stable_rank(student.timemix.r_proj.weight)
            rec["wnorm_rms"]   = gm.weight_norm_rms(student.parameters())
"""
from __future__ import annotations

import math
from typing import Iterable, Optional

try:
    import torch
except Exception:  # pragma: no cover - torch is always present in trainers
    torch = None  # type: ignore


def _f(x) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


@torch.no_grad() if torch is not None else (lambda f: f)
def rms(t) -> float:
    """Root-mean-square of a tensor, in fp32. nan on failure (never raises)."""
    try:
        t = t.detach().float()
        if t.numel() == 0:
            return float("nan")
        return _f(t.pow(2).mean().sqrt())
    except Exception:
        return float("nan")


@torch.no_grad() if torch is not None else (lambda f: f)
def injection_rms(inj, ref=None) -> Optional[float]:
    """RMS of a residual-stream injection. If `ref` (e.g. the token embedding /
    hidden state) is given, returns the RELATIVE magnitude ||inj|| / ||ref||, which
    is the natural 'is this path doing anything' scale (0 at identity-init, grows as
    the recall path groks on). Returns None if `inj` is None."""
    if inj is None:
        return None
    a = rms(inj)
    if ref is None:
        return a
    b = rms(ref)
    if not (b == b) or b == 0.0:  # nan or zero ref
        return a
    return a / b


@torch.no_grad() if torch is not None else (lambda f: f)
def injection_stats(*, rosa_inj=None, engram_inj=None, ref=None) -> dict:
    """Convenience: relative RMS for the ROSA and Engram injections in one dict.
    Keys are omitted when the corresponding injection is None, so a run with only
    Engram (or only ROSA) emits only the field it has."""
    out: dict = {}
    r = injection_rms(rosa_inj, ref)
    e = injection_rms(engram_inj, ref)
    if r is not None:
        out["rosa_inj_rms"] = r
    if e is not None:
        out["engram_inj_rms"] = e
    return out


@torch.no_grad() if torch is not None else (lambda f: f)
def affine_gap(e0, e1) -> float:
    """Mean |e1 - e0| for ROSA's learned affine readout (rosa.py RosaLayer). 0 at
    identity-init; growth is the structural signature that the recall path is
    learning to read its retrieved bits."""
    try:
        return _f((e1.detach().float() - e0.detach().float()).abs().mean())
    except Exception:
        return float("nan")


@torch.no_grad() if torch is not None else (lambda f: f)
def gate_open(gate_param) -> float:
    """sigmoid(gate) for a scalar/vector gate (the placeholder ROSA/Engram modules
    fuse via sigmoid(gate)*value). A proxy for how far the path has opened."""
    try:
        return _f(torch.sigmoid(gate_param.detach().float()).mean())
    except Exception:
        return float("nan")


@torch.no_grad() if torch is not None else (lambda f: f)
def active_fraction(inj, dim: int = -1, thresh: float = 1e-3) -> float:
    """Fraction of routes/units (slices along `dim`) whose per-unit RMS exceeds
    `thresh` — a dead-route / lottery-ticket monitor for ROSA routes / Engram slots."""
    try:
        t = inj.detach().float()
        per = t.pow(2).mean(dim=tuple(i for i in range(t.ndim) if i != (dim % t.ndim))).sqrt()
        return _f((per > thresh).float().mean())
    except Exception:
        return float("nan")


@torch.no_grad() if torch is not None else (lambda f: f)
def stable_rank(W) -> float:
    """Stable rank ||W||_F^2 / sigma_max^2 of a 2-D weight matrix — a smooth,
    SVD-light proxy for effective rank. Falling toward 1 = low-rank collapse;
    healthy generalizing structure sits well above 1. HEAVY-ish: coarse cadence."""
    try:
        m = W.detach().float()
        if m.ndim > 2:
            m = m.reshape(m.shape[0], -1)
        if m.ndim != 2 or m.numel() == 0:
            return float("nan")
        fro2 = m.pow(2).sum()
        smax = torch.linalg.matrix_norm(m, ord=2)  # largest singular value
        if not torch.isfinite(smax) or smax <= 0:
            return float("nan")
        return _f(fro2 / (smax * smax))
    except Exception:
        return float("nan")


@torch.no_grad() if torch is not None else (lambda f: f)
def weight_norm_rms(params: Iterable) -> float:
    """RMS of all trainable weights — the classic grokking 'clock'. Cheap."""
    try:
        tot = 0.0
        n = 0
        for p in params:
            if p is None or not getattr(p, "requires_grad", False):
                continue
            pf = p.detach().float()
            tot += _f(pf.pow(2).sum())
            n += pf.numel()
        if n == 0:
            return float("nan")
        return math.sqrt(tot / n)
    except Exception:
        return float("nan")
