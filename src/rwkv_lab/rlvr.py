"""Reinforcement learning with verifiable rewards (RLVR) primitives.

References and the exact pieces adopted here:

* Liu et al., "Understanding R1-Zero-Like Training: A Critical Perspective",
  arXiv:2503.20783, https://arxiv.org/abs/2503.20783 — Dr.GRPO's removal of
  response-length and group-standard-deviation bias.
* Yu et al., "DAPO: An Open-Source LLM Reinforcement Learning System at
  Scale", arXiv:2503.14476, https://arxiv.org/abs/2503.14476 — asymmetric
  clipping and dynamic removal of all-correct/all-incorrect groups.
* Zheng et al., "Group Sequence Policy Optimization", arXiv:2507.18071,
  https://arxiv.org/abs/2507.18071 — sequence-level likelihood ratios and
  clipping rather than token-level ratios.
* Zhao et al., "Absolute Zero: Reinforced Self-play Reasoning with Zero Data",
  arXiv:2505.03335, https://arxiv.org/abs/2505.03335 — programmatic verifier
  interface; task proposal/orchestration intentionally belongs to Adamaton.

The module is deliberately rollout-engine agnostic. It owns the differentiable
policy objectives and deterministic verifiers; Adamaton can supply curricula
and RWKV-Lab generation can supply sampled responses.
"""
from __future__ import annotations

from dataclasses import dataclass
import ast
import math
from typing import Callable, Sequence

import torch
import torch.nn.functional as F


@dataclass
class PolicyLossOutput:
    loss: torch.Tensor
    advantages: torch.Tensor
    approx_kl: torch.Tensor
    clip_fraction: torch.Tensor
    active_groups: torch.Tensor


def token_log_probs(logits: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
    """Return log p(tokens) for aligned ``[..., vocab]`` logits."""

    return logits.log_softmax(-1).gather(-1, tokens.unsqueeze(-1)).squeeze(-1)


def group_advantages(rewards: torch.Tensor, group_ids: torch.Tensor, *,
                     standardize: bool = True, drop_constant: bool = False,
                     eps: float = 1e-6) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute leave-shape group-relative advantages and active-row mask."""

    if rewards.ndim != 1 or group_ids.shape != rewards.shape:
        raise ValueError("rewards and group_ids must be equal 1-D tensors")
    adv = torch.zeros_like(rewards, dtype=torch.float32)
    active = torch.ones_like(rewards, dtype=torch.bool)
    for gid in torch.unique(group_ids):
        idx = group_ids == gid
        vals = rewards[idx].float()
        constant = bool((vals.max() - vals.min()).abs() <= eps)
        if drop_constant and constant:
            active[idx] = False
            continue
        centered = vals - vals.mean()
        if standardize and not constant:
            centered = centered / vals.std(unbiased=False).clamp_min(eps)
        adv[idx] = centered
    return adv, active


def policy_loss(
    logp: torch.Tensor,
    old_logp: torch.Tensor,
    rewards: torch.Tensor,
    group_ids: torch.Tensor,
    mask: torch.Tensor,
    *,
    algorithm: str = "gspo",
    clip_low: float = 0.2,
    clip_high: float = 0.2,
    reference_logp: torch.Tensor | None = None,
    kl_coef: float = 0.0,
) -> PolicyLossOutput:
    """GSPO, Dr.GRPO, or DAPO clipped policy objective.

    Shapes: logp/old_logp/mask are ``[responses,tokens]``; rewards and
    group_ids are ``[responses]``. Padding never contributes.
    """

    if algorithm not in ("gspo", "dr_grpo", "dapo"):
        raise ValueError("algorithm must be gspo, dr_grpo, or dapo")
    m = mask.float()
    lengths = m.sum(-1).clamp_min(1.0)
    standardize = algorithm != "dr_grpo"
    adv, active = group_advantages(rewards, group_ids, standardize=standardize,
                                   drop_constant=(algorithm == "dapo"))
    row_mask = active.float()

    if algorithm == "gspo":
        # GSPO Eq. concept: one geometrically averaged likelihood ratio per sequence.
        seq_delta = ((logp - old_logp) * m).sum(-1) / lengths
        ratio = seq_delta.exp()
        clipped = ratio.clamp(1.0 - clip_low, 1.0 + clip_high)
        objective = torch.minimum(ratio * adv, clipped * adv)
        clip_fraction = ((ratio != clipped) & active).float().sum() / row_mask.sum().clamp_min(1)
    else:
        ratio_tok = (logp - old_logp).exp()
        clipped = ratio_tok.clamp(1.0 - clip_low, 1.0 + clip_high)
        token_obj = torch.minimum(ratio_tok * adv[:, None], clipped * adv[:, None]) * m
        if algorithm == "dr_grpo":
            # Dr.GRPO uses a constant token normalizer, avoiding response-length bias.
            denom = m.shape[-1]
            objective = token_obj.sum(-1) / max(denom, 1)
        else:
            objective = token_obj.sum(-1) / lengths
        changed = ((ratio_tok != clipped) * m.bool()).any(-1)
        clip_fraction = (changed.float() * row_mask).sum() / row_mask.sum().clamp_min(1)

    pg = -(objective * row_mask).sum() / row_mask.sum().clamp_min(1)
    if reference_logp is not None:
        # Positive sampled reverse-KL estimator used as a stability diagnostic/penalty.
        kl_rows = ((logp - reference_logp) * m).sum(-1) / lengths
        kl = (kl_rows * row_mask).sum() / row_mask.sum().clamp_min(1)
        pg = pg + kl_coef * kl
    else:
        kl = (((old_logp - logp) * m).sum(-1) / lengths * row_mask).sum() \
            / row_mask.sum().clamp_min(1)
    return PolicyLossOutput(pg, adv, kl.detach(), clip_fraction.detach(), active)


class ExactAnswerVerifier:
    """Whitespace-normalized exact verifier suitable for deterministic RLVR."""

    def __init__(self, expected: str | Sequence[str], *, case_sensitive: bool = False):
        self.expected = [expected] if isinstance(expected, str) else list(expected)
        self.case_sensitive = case_sensitive

    def _norm(self, value: str) -> str:
        value = " ".join(value.split())
        return value if self.case_sensitive else value.casefold()

    def __call__(self, response: str) -> float:
        got = self._norm(response)
        return float(any(got == self._norm(x) for x in self.expected))


class PythonExpressionVerifier:
    """Safe arithmetic-expression verifier used by self-generated curricula.

    It accepts literals and arithmetic operators only—no names, calls,
    attributes, comprehensions, or imports—so it can run inside a trainer
    without becoming the task sandbox (which remains an Adamaton concern).
    """

    _OPS = (ast.Expression, ast.Constant, ast.UnaryOp, ast.BinOp, ast.Add, ast.Sub,
            ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow, ast.USub, ast.UAdd)

    def __init__(self, expected: float, *, atol: float = 1e-6):
        self.expected, self.atol = float(expected), float(atol)

    def _evaluate(self, node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return self._evaluate(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) \
                and not isinstance(node.value, bool):
            value = float(node.value)
        elif isinstance(node, ast.UnaryOp):
            value = self._evaluate(node.operand)
            value = -value if isinstance(node.op, ast.USub) else value
        elif isinstance(node, ast.BinOp):
            left, right = self._evaluate(node.left), self._evaluate(node.right)
            if isinstance(node.op, ast.Add): value = left + right
            elif isinstance(node.op, ast.Sub): value = left - right
            elif isinstance(node.op, ast.Mult): value = left * right
            elif isinstance(node.op, ast.Div): value = left / right
            elif isinstance(node.op, ast.FloorDiv): value = left // right
            elif isinstance(node.op, ast.Mod): value = left % right
            elif isinstance(node.op, ast.Pow) and abs(right) <= 12: value = left ** right
            else: raise ValueError("operator is not verifier-safe")
        else:
            raise ValueError("expression is not verifier-safe")
        if not math.isfinite(value) or abs(value) > 1e100:
            raise ValueError("arithmetic result exceeds verifier bounds")
        return value

    def __call__(self, response: str) -> float:
        try:
            if len(response) > 256:
                return 0.0
            tree = ast.parse(response.strip(), mode="eval")
            nodes = list(ast.walk(tree))
            if len(nodes) > 64 or any(not isinstance(n, self._OPS) for n in nodes):
                return 0.0
            value = self._evaluate(tree)
            return float(math.isfinite(value) and abs(value - self.expected) <= self.atol)
        except Exception:
            return 0.0


def verify_batch(responses: Sequence[str], verifiers: Sequence[Callable[[str], float]]) -> torch.Tensor:
    if len(responses) != len(verifiers):
        raise ValueError("one verifier is required per response")
    return torch.tensor([float(v(r)) for r, v in zip(responses, verifiers)], dtype=torch.float32)
