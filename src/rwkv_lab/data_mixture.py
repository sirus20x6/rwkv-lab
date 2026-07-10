"""Data-mixture optimization for multi-domain pretraining.

References:

* Liu et al., "RegMix: Data Mixture as Regression for Language Model
  Pre-training", arXiv:2407.01492, https://arxiv.org/abs/2407.01492.
* Ye et al., "Data Mixing Made Efficient: A Bivariate Scaling Law for
  Language Model Pretraining", arXiv:2502.15950,
  https://arxiv.org/abs/2502.15950.

RegMix's key systems insight is adopted directly: fit a cheap surrogate to
small mixture runs, then search the simplex instead of training one large model
per candidate. ``loss_features`` also accepts per-domain expert losses, the
observable used by the MDE-style bivariate surrogate. This module chooses a
mixture; corpus construction and provenance remain in ``build_corpus``.
"""
from __future__ import annotations

from dataclasses import dataclass
import itertools
import numpy as np


def _simplex(weights) -> np.ndarray:
    w = np.asarray(weights, dtype=np.float64)
    if w.ndim != 1 or len(w) < 2 or np.any(w < 0) or not np.isfinite(w).all():
        raise ValueError("mixture weights must be a finite non-negative vector")
    total = float(w.sum())
    if total <= 0:
        raise ValueError("mixture weights must have positive mass")
    return w / total


@dataclass(frozen=True)
class MixtureObservation:
    weights: tuple[float, ...]
    validation_loss: float
    expert_losses: tuple[float, ...] = ()


class MixtureSurrogate:
    """Ridge-regression surrogate over simplex and optional expert losses."""

    def __init__(self, *, ridge: float = 1e-4, pairwise: bool = True):
        self.ridge, self.pairwise = float(ridge), bool(pairwise)
        self.coef_: np.ndarray | None = None
        self.n_domains: int | None = None
        self.n_expert: int = 0

    def _features(self, weights, expert_losses=()) -> np.ndarray:
        w = _simplex(weights)
        e = np.asarray(expert_losses, dtype=np.float64)
        out = [1.0, *w, *(w * w)]
        if self.pairwise:
            out.extend(w[i] * w[j] for i, j in itertools.combinations(range(len(w)), 2))
        if len(e):
            if len(e) != len(w) or not np.isfinite(e).all():
                raise ValueError("expert_losses must contain one finite value per domain")
            # MDE-style mixture/expert interactions let the fit express transfer and interference.
            out.extend(e)
            out.extend(w * e)
        return np.asarray(out, dtype=np.float64)

    def fit(self, observations: list[MixtureObservation]) -> "MixtureSurrogate":
        if not observations:
            raise ValueError("at least one mixture observation is required")
        self.n_domains = len(observations[0].weights)
        self.n_expert = len(observations[0].expert_losses)
        if any(len(o.weights) != self.n_domains or len(o.expert_losses) != self.n_expert
               for o in observations):
            raise ValueError("all observations must have the same domain geometry")
        X = np.stack([self._features(o.weights, o.expert_losses) for o in observations])
        y = np.asarray([o.validation_loss for o in observations], dtype=np.float64)
        reg = self.ridge * np.eye(X.shape[1]); reg[0, 0] = 0.0
        self.coef_ = np.linalg.pinv(X.T @ X + reg) @ X.T @ y
        return self

    def predict(self, weights, expert_losses=()) -> float:
        if self.coef_ is None:
            raise RuntimeError("fit the surrogate before prediction")
        return float(self._features(weights, expert_losses) @ self.coef_)

    def search(self, *, candidates: int = 10000, seed: int = 0,
               expert_losses=()) -> tuple[np.ndarray, float]:
        if self.n_domains is None:
            raise RuntimeError("fit the surrogate before search")
        rng = np.random.default_rng(seed)
        draws = rng.dirichlet(np.ones(self.n_domains), size=max(1, int(candidates)))
        # Include vertices and the uniform mixture so small searches retain useful anchors.
        draws = np.concatenate((draws, np.eye(self.n_domains),
                                np.full((1, self.n_domains), 1 / self.n_domains)))
        scores = np.asarray([self.predict(w, expert_losses) for w in draws])
        i = int(scores.argmin())
        return draws[i], float(scores[i])


class DomainMixtureSampler:
    """Deterministic domain sampler whose weights can change between rungs."""

    def __init__(self, weights, *, seed: int = 0):
        self.weights = _simplex(weights)
        self.rng = np.random.default_rng(seed)
        self.counts = np.zeros(len(self.weights), dtype=np.int64)

    def update(self, weights) -> None:
        w = _simplex(weights)
        if len(w) != len(self.weights):
            raise ValueError("cannot change the number of domains")
        self.weights = w

    def sample(self, size: int = 1) -> np.ndarray:
        ids = self.rng.choice(len(self.weights), size=int(size), p=self.weights)
        self.counts += np.bincount(ids, minlength=len(self.weights))
        return ids
