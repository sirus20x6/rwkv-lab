"""Guarded end-to-end test-time training transaction.

Tandon et al., *End-to-End Test-Time Training for Long Context* (2025),
https://test-time-training.github.io/e2e.pdf and official code
https://github.com/test-time-training/e2e. RWKV-Lab requires explicit parameter
scope, update budget, held-out gate, and rollback; baseline inference never mutates weights.
"""
from __future__ import annotations
from dataclasses import dataclass
import copy
import torch


@dataclass(frozen=True)
class TTTPolicy:
    steps: int = 1
    learning_rate: float = 1e-5
    max_parameters: int = 1_000_000
    max_regression: float = 0.0


def guarded_test_time_train(module: torch.nn.Module, loss_fn, evaluate_fn, policy: TTTPolicy = TTTPolicy()) -> dict:
    params = [p for p in module.parameters() if p.requires_grad]
    count = sum(p.numel() for p in params)
    if count > policy.max_parameters: raise ValueError("TTT parameter budget exceeded")
    snapshot = copy.deepcopy(module.state_dict()); before = float(evaluate_fn(module))
    opt = torch.optim.SGD(params, lr=policy.learning_rate)
    losses = []
    try:
        for _ in range(policy.steps):
            opt.zero_grad(set_to_none=True); loss = loss_fn(module)
            if not torch.isfinite(loss): raise FloatingPointError("non-finite TTT loss")
            loss.backward(); opt.step(); losses.append(float(loss.detach()))
        after = float(evaluate_fn(module)); accepted = after >= before - policy.max_regression
    except Exception:
        module.load_state_dict(snapshot); raise
    if not accepted: module.load_state_dict(snapshot)
    return {"schema": "rwkv-lab.guarded-ttt.v1", "parameters": count, "steps": policy.steps,
            "losses": losses, "score_before": before, "score_after": after,
            "accepted": accepted, "rolled_back": not accepted}
