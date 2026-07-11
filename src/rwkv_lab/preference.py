"""Reference-tested post-training losses for preferences and learned rewards.

Paper references (also mirrored in README.md):

* DPO — Rafailov et al., https://arxiv.org/abs/2305.18290
* KTO — Ethayarajh et al., https://arxiv.org/abs/2402.01306
* ORPO — Hong et al., https://arxiv.org/abs/2403.07691
* SimPO — Meng et al., https://arxiv.org/abs/2405.14734
* Outcome reward modeling / InstructGPT — Ouyang et al., https://arxiv.org/abs/2203.02155
* Process reward modeling — Lightman et al., https://arxiv.org/abs/2305.20050

The functions accept sequence log-probabilities rather than model objects.  This keeps them usable
by native RWKV, converted Hugging Face models, and sharded training, while making unit-level parity
tests possible.  Learned reward models are intentionally separate from promotion decisions: the
independent held-out gates in ``rlvr_evaluation.py`` remain authoritative.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn
import torch.nn.functional as F


PreferenceKind = Literal["dpo", "kto", "orpo", "simpo"]


class OutcomeRewardHead(nn.Module):
    """Scalar reward at the last response token, as used by pairwise outcome supervision."""

    def __init__(self, hidden_size: int, *, bias: bool = False):
        super().__init__()
        self.proj = nn.Linear(hidden_size, 1, bias=bias)

    def forward(self, hidden: torch.Tensor, response_mask: torch.Tensor) -> torch.Tensor:
        if hidden.ndim != 3 or response_mask.shape != hidden.shape[:2]:
            raise ValueError("outcome reward expects hidden [B,T,D] and response mask [B,T]")
        positions = torch.arange(hidden.shape[1], device=hidden.device).expand_as(response_mask)
        last = positions.masked_fill(~response_mask.bool(), -1).max(-1).values
        if torch.any(last < 0):
            raise ValueError("every outcome-reward sequence needs a response token")
        chosen = hidden[torch.arange(hidden.shape[0], device=hidden.device), last]
        return self.proj(chosen).squeeze(-1)


class ProcessRewardHead(nn.Module):
    """One scalar per labeled reasoning step; selection/masking stays explicit at the caller."""

    def __init__(self, hidden_size: int, *, bias: bool = False):
        super().__init__()
        self.proj = nn.Linear(hidden_size, 1, bias=bias)

    def forward(self, hidden: torch.Tensor, step_positions: torch.Tensor) -> torch.Tensor:
        if hidden.ndim != 3 or step_positions.ndim != 2 or step_positions.shape[0] != hidden.shape[0]:
            raise ValueError("process reward expects hidden [B,T,D] and step positions [B,S]")
        if torch.any(step_positions < 0) or torch.any(step_positions >= hidden.shape[1]):
            raise ValueError("process-reward step position is outside the sequence")
        batch = torch.arange(hidden.shape[0], device=hidden.device)[:, None]
        return self.proj(hidden[batch, step_positions]).squeeze(-1)


def sequence_logps(logits: torch.Tensor, labels: torch.Tensor, *, average: bool = False,
                   ignore_index: int = -100) -> torch.Tensor:
    """Causal sequence log p over non-masked target tokens.

    ``labels`` has the same ``[batch,time]`` shape as the model input. Position ``t`` is predicted
    by logits at ``t-1``. Empty response masks are rejected instead of producing a plausible zero.
    """
    if logits.ndim != 3 or labels.ndim != 2 or logits.shape[:2] != labels.shape:
        raise ValueError("logits must be [B,T,V] and labels [B,T]")
    target = labels[:, 1:]
    mask = target != ignore_index
    if not torch.all(mask.sum(-1) > 0):
        raise ValueError("every sequence needs at least one unmasked target token")
    safe = target.masked_fill(~mask, 0)
    token_logps = logits[:, :-1].log_softmax(-1).gather(-1, safe.unsqueeze(-1)).squeeze(-1)
    result = (token_logps * mask).sum(-1)
    return result / mask.sum(-1) if average else result


@dataclass(frozen=True)
class PreferenceLoss:
    loss: torch.Tensor
    chosen_reward: torch.Tensor
    rejected_reward: torch.Tensor
    margin: torch.Tensor


def dpo_loss(policy_chosen: torch.Tensor, policy_rejected: torch.Tensor,
             reference_chosen: torch.Tensor, reference_rejected: torch.Tensor, *,
             beta: float = 0.1, label_smoothing: float = 0.0) -> PreferenceLoss:
    """DPO Eq. 7 with optional conservative-label smoothing."""
    if not 0.0 <= label_smoothing < 0.5:
        raise ValueError("DPO label_smoothing must be in [0, 0.5)")
    chosen_reward = beta * (policy_chosen - reference_chosen)
    rejected_reward = beta * (policy_rejected - reference_rejected)
    margin = chosen_reward - rejected_reward
    loss = (-(1.0 - label_smoothing) * F.logsigmoid(margin)
            - label_smoothing * F.logsigmoid(-margin))
    return PreferenceLoss(loss, chosen_reward.detach(), rejected_reward.detach(), margin.detach())


def simpo_loss(chosen_average_logp: torch.Tensor, rejected_average_logp: torch.Tensor, *,
               beta: float = 2.0, gamma: float = 1.0) -> PreferenceLoss:
    """SimPO length-normalized, reference-free objective (paper Eq. 6).

    ``gamma`` is the target reward margin, not ``gamma/beta``; the internal logit is
    ``beta * (chosen - rejected) - gamma``.
    """
    chosen_reward = beta * chosen_average_logp
    rejected_reward = beta * rejected_average_logp
    margin = chosen_reward - rejected_reward - gamma
    return PreferenceLoss(-F.logsigmoid(margin), chosen_reward.detach(),
                          rejected_reward.detach(), margin.detach())


def _log_odds(logp: torch.Tensor) -> torch.Tensor:
    # Sequence probabilities may round to one for synthetic tests; clamp before log1p.
    logp = torch.minimum(logp, torch.full_like(logp, -torch.finfo(logp.dtype).eps))
    return logp - torch.log1p(-torch.exp(logp))


def orpo_loss(chosen_logp: torch.Tensor, rejected_logp: torch.Tensor, chosen_nll: torch.Tensor, *,
              beta: float = 0.1) -> PreferenceLoss:
    """ORPO: chosen-response SFT NLL plus the odds-ratio preference penalty."""
    chosen_reward = _log_odds(chosen_logp)
    rejected_reward = _log_odds(rejected_logp)
    margin = chosen_reward - rejected_reward
    loss = chosen_nll - beta * F.logsigmoid(margin)
    return PreferenceLoss(loss, chosen_reward.detach(), rejected_reward.detach(), margin.detach())


def kto_loss(policy_logp: torch.Tensor, reference_logp: torch.Tensor,
             desirable: torch.Tensor, *, beta: float = 0.1,
             kl: torch.Tensor | None = None, desirable_weight: float = 1.0,
             undesirable_weight: float = 1.0) -> torch.Tensor:
    """KTO binary-feedback loss with a detached batch KL baseline.

    KTO's pointwise desirability labels do not need chosen/rejected pairs.  In production, mix both
    labels in each batch so the empirical KL baseline and class weights remain well-conditioned.
    """
    if policy_logp.shape != reference_logp.shape or desirable.shape != policy_logp.shape:
        raise ValueError("KTO log-probabilities and labels must have identical shapes")
    log_ratio = policy_logp - reference_logp
    if kl is None:
        kl = log_ratio.detach().mean().clamp_min(0.0)
    desirable = desirable.bool()
    good = 1.0 - torch.sigmoid(beta * (log_ratio - kl))
    bad = 1.0 - torch.sigmoid(beta * (kl - log_ratio))
    weights = torch.where(desirable, torch.as_tensor(desirable_weight, device=log_ratio.device),
                          torch.as_tensor(undesirable_weight, device=log_ratio.device))
    return torch.where(desirable, good, bad) * weights


def reward_model_loss(chosen_reward: torch.Tensor, rejected_reward: torch.Tensor, *,
                      margin: torch.Tensor | float = 0.0) -> torch.Tensor:
    """Bradley–Terry pairwise outcome-reward loss."""
    return -F.logsigmoid(chosen_reward - rejected_reward - margin)


def process_reward_loss(step_logits: torch.Tensor, step_labels: torch.Tensor,
                        step_mask: torch.Tensor | None = None) -> torch.Tensor:
    """Masked binary loss for process-reward labels attached to reasoning steps."""
    if step_logits.shape != step_labels.shape:
        raise ValueError("process-reward logits and labels must have identical shapes")
    raw = F.binary_cross_entropy_with_logits(step_logits, step_labels.to(step_logits.dtype),
                                             reduction="none")
    if step_mask is None:
        return raw.mean()
    if step_mask.shape != raw.shape or not torch.any(step_mask):
        raise ValueError("process-reward mask must select at least one matching step")
    mask = step_mask.to(raw.dtype)
    return (raw * mask).sum() / mask.sum()


def binary_calibration(logits: torch.Tensor, labels: torch.Tensor, *, bins: int = 10) -> dict[str, float]:
    """Accuracy, Brier score, and expected calibration error for reward-model receipts."""
    if logits.shape != labels.shape or logits.numel() == 0 or bins <= 0:
        raise ValueError("calibration needs non-empty matching tensors and positive bins")
    probabilities = logits.detach().float().sigmoid().flatten()
    truth = labels.detach().float().flatten()
    accuracy = ((probabilities >= 0.5) == truth.bool()).float().mean()
    brier = (probabilities - truth).square().mean()
    bucket = (probabilities * bins).long().clamp_(0, bins - 1)
    counts = torch.zeros(bins, device=probabilities.device).scatter_add_(
        0, bucket, torch.ones_like(probabilities))
    confidence = torch.zeros_like(counts).scatter_add_(0, bucket, probabilities)
    empirical = torch.zeros_like(counts).scatter_add_(0, bucket, truth)
    denom = counts.clamp_min(1)
    ece = (counts / probabilities.numel() *
           (confidence / denom - empirical / denom).abs()).sum()
    return {"accuracy": float(accuracy), "brier": float(brier), "ece": float(ece)}


def adversarial_reward_audit(clean_logits: torch.Tensor, adversarial_logits: torch.Tensor,
                             adversarial_labels: torch.Tensor) -> dict[str, float]:
    """Measure whether explicitly corrupted/reordered steps receive unjustified high reward."""
    if adversarial_logits.shape != adversarial_labels.shape or adversarial_logits.numel() == 0:
        raise ValueError("adversarial audit needs matching non-empty logits and labels")
    clean_mean = clean_logits.detach().float().sigmoid().mean()
    adversarial = adversarial_logits.detach().float().sigmoid()
    labels = adversarial_labels.detach().bool()
    negatives = ~labels
    false_positive = ((adversarial >= 0.5) & negatives).float().sum() / negatives.sum().clamp_min(1)
    return {"clean_probability": float(clean_mean),
            "adversarial_probability": float(adversarial.mean()),
            "clean_adversarial_margin": float(clean_mean - adversarial.mean()),
            "adversarial_false_positive_rate": float(false_positive)}
