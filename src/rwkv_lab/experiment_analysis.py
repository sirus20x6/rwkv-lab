"""Statistical and decision helpers for paired experiment campaigns."""
from __future__ import annotations

import itertools
import math
from statistics import NormalDist
import numpy as np


def paired_stats(baseline, candidate, *, bootstrap=10_000, permutations=20_000,
                 seed=0, alpha=0.05) -> dict:
    """Paired effect, bootstrap CI, sign-flip permutation p, and power guidance."""
    a = np.asarray(baseline, dtype=np.float64)
    b = np.asarray(candidate, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 1 or not len(a):
        raise ValueError("paired samples must be non-empty one-dimensional arrays of equal length")
    d = b - a
    mean = float(d.mean())
    sd = float(d.std(ddof=1)) if len(d) > 1 else 0.0
    effect = mean / sd if sd > 0 else (math.copysign(math.inf, mean) if mean else 0.0)
    rng = np.random.default_rng(seed)
    if len(d) == 1:
        lo = hi = mean
    else:
        idx = rng.integers(0, len(d), size=(bootstrap, len(d)))
        boot = d[idx].mean(axis=1)
        lo, hi = (float(x) for x in np.quantile(boot, [alpha / 2, 1 - alpha / 2]))

    observed = abs(mean)
    if len(d) <= 16:
        signs = np.asarray(list(itertools.product((-1.0, 1.0), repeat=len(d))))
    else:
        signs = rng.choice((-1.0, 1.0), size=(permutations, len(d)))
    null = np.abs((signs * d).mean(axis=1))
    p = float((np.count_nonzero(null >= observed - 1e-15) + 1) / (len(null) + 1))
    if abs(mean) > 0 and sd > 0:
        recommended_n = max(len(d), int(math.ceil(((1.96 + 0.84) * sd / abs(mean)) ** 2)))
    else:
        recommended_n = len(d) if sd == 0 and mean != 0 else max(len(d) + 1, 8)
    return {"n": int(len(d)), "baseline_mean": float(a.mean()), "candidate_mean": float(b.mean()),
            "delta": mean, "ci_low": lo, "ci_high": hi, "p_value": p,
            "effect_size": effect, "recommended_n": recommended_n,
            "significant": bool(lo > 0 or hi < 0), "paired_deltas": d.tolist()}


def holm_adjust(items: dict[str, dict], alpha=0.05) -> dict[str, dict]:
    """Holm family-wise correction; returns copied stats with adjusted p/significance."""
    out = {k: dict(v) for k, v in items.items()}
    ordered = sorted(out, key=lambda k: out[k]["p_value"])
    running = 0.0
    m = len(ordered)
    reject_chain = True
    for rank, key in enumerate(ordered):
        raw = float(out[key]["p_value"])
        adjusted = min(1.0, max(running, (m - rank) * raw))
        running = adjusted
        reject_chain = reject_chain and raw <= alpha / (m - rank)
        out[key]["p_adjusted"] = adjusted
        out[key]["significant"] = bool(out[key]["significant"] and reject_chain)
    return out


def alpha_spending(look: int, total_looks: int, *, alpha=0.05,
                   method="obrien_fleming") -> dict:
    """Return a valid per-look alpha allocation for repeated interim analyses.

    ``cumulative`` is the alpha spent through this look and ``increment`` is the
    amount available to this look.  Testing each look at its increment is a
    conservative alpha-spending design (a union bound controls family-wise type-I
    error even though successive rung measurements are correlated).
    """
    if total_looks < 1 or not 1 <= look <= total_looks:
        raise ValueError("look must be in [1, total_looks]")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between zero and one")

    def cumulative(k):
        t = k / total_looks
        if method in ("obrien_fleming", "obf"):
            z = NormalDist().inv_cdf(1 - alpha / 2)
            return min(alpha, 2 * (1 - NormalDist().cdf(z / math.sqrt(t))))
        if method == "pocock":
            return alpha * math.log(1 + (math.e - 1) * t)
        if method in ("linear", "bonferroni"):
            return alpha * t
        raise ValueError(f"unknown alpha-spending method: {method}")

    spent = cumulative(look)
    previous = cumulative(look - 1) if look > 1 else 0.0
    return {"look": look, "total_looks": total_looks, "information_fraction": look / total_looks,
            "method": "obrien_fleming" if method == "obf" else method,
            "family_alpha": float(alpha), "cumulative": float(spent),
            "increment": float(max(spent - previous, np.finfo(float).eps))}


def sequential_holm(items: dict[str, dict], look: int, total_looks: int, *,
                    alpha=0.05, method="obrien_fleming") -> dict[str, dict]:
    """Holm-correct one interim look using its pre-registered alpha spend."""
    spend = alpha_spending(look, total_looks, alpha=alpha, method=method)
    out = holm_adjust(items, alpha=spend["increment"])
    for stats in out.values():
        stats["sequential"] = dict(spend)
        # holm_adjust also requires the CI to exclude zero. paired_stats should be
        # called with this same look alpha so both pieces use the same boundary.
        stats["significant"] = bool(stats["significant"] and
                                    stats["p_adjusted"] <= spend["increment"])
    return out


def pareto_front(rows: list[dict], *, maximize=("acc",), minimize=("train_seconds", "peak_alloc_mb")) -> list[bool]:
    """Return a nondominated flag per row; missing objectives make a row non-Pareto."""
    flags = []
    keys = tuple(maximize) + tuple(minimize)
    for i, row in enumerate(rows):
        if any(row.get(k) is None for k in keys):
            flags.append(False); continue
        dominated = False
        for j, other in enumerate(rows):
            if i == j or any(other.get(k) is None for k in keys):
                continue
            weak = all(other[k] >= row[k] for k in maximize) and all(other[k] <= row[k] for k in minimize)
            strict = any(other[k] > row[k] for k in maximize) or any(other[k] < row[k] for k in minimize)
            if weak and strict:
                dominated = True; break
        flags.append(not dominated)
    return flags
