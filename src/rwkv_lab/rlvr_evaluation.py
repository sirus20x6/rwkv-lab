"""Leakage audits, curriculum gates, and promotion statistics for RLVR.

Paper-derived mechanisms:

* Efron, "Bootstrap Methods: Another Look at the Jackknife" (1979),
  https://doi.org/10.1214/aos/1176344552 — paired non-parametric confidence
  intervals over fixed held-out tasks.
* DeepSeek-R1, https://arxiv.org/abs/2501.12948 — cold-start supervised data
  before RL and staged training rather than assuming a weak parent already
  emits useful verifier successes.

The gates are deliberately deterministic for a given seed and never inspect
private verifier internals. They operate only on task identities/families and
the scalar rewards returned by the verifier boundary.
"""

from __future__ import annotations

from collections import defaultdict
import hashlib
import math
import random
import re
from statistics import fmean
from typing import Any, Mapping, Sequence


def _family(task) -> str:
    return str((task.metadata or {}).get("family") or "unspecified")


def _normalized_prompt(prompt: str) -> str:
    return re.sub(r"\s+", " ", prompt).strip().casefold()


def audit_task_splits(train_tasks: Sequence, eval_tasks: Sequence) -> dict[str, Any]:
    """Detect identity and exact normalized-prompt leakage across splits."""

    train_ids, eval_ids = [t.id for t in train_tasks], [t.id for t in eval_tasks]
    train_prompts = {_normalized_prompt(t.prompt) for t in train_tasks}
    eval_prompts = {_normalized_prompt(t.prompt) for t in eval_tasks}
    id_overlap = sorted(set(train_ids) & set(eval_ids))
    prompt_overlap = sorted(train_prompts & eval_prompts)
    duplicate_ids = len(train_ids + eval_ids) - len(set(train_ids + eval_ids))
    duplicate_train_prompts = len(train_tasks) - len(train_prompts)
    duplicate_eval_prompts = len(eval_tasks) - len(eval_prompts)
    families: dict[str, int] = defaultdict(int)
    for task in eval_tasks:
        families[_family(task)] += 1
    return {
        "train_tasks": len(train_tasks),
        "eval_tasks": len(eval_tasks),
        "eval_families": dict(sorted(families.items())),
        "duplicate_ids": duplicate_ids,
        "duplicate_train_prompts": duplicate_train_prompts,
        "duplicate_eval_prompts": duplicate_eval_prompts,
        "id_overlap": id_overlap[:16],
        "normalized_prompt_overlap_count": len(prompt_overlap),
        "normalized_prompt_overlap_sha256": [
            hashlib.sha256(prompt.encode()).hexdigest()
            for prompt in prompt_overlap[:16]
        ],
        "passed": not duplicate_ids
        and not duplicate_eval_prompts
        and not id_overlap
        and not prompt_overlap,
    }


def stratified_tasks(tasks: Sequence, limit: int, *, seed: int) -> list:
    """Select a deterministic round-robin sample across hidden task families."""

    if limit >= len(tasks):
        return list(tasks)
    buckets: dict[str, list] = defaultdict(list)
    for task in tasks:
        buckets[_family(task)].append(task)
    rng = random.Random(seed)
    for values in buckets.values():
        rng.shuffle(values)
    selected = []
    families = sorted(buckets)
    while len(selected) < limit:
        progressed = False
        for family in families:
            if buckets[family] and len(selected) < limit:
                selected.append(buckets[family].pop())
                progressed = True
        if not progressed:
            break
    return selected


def curriculum_pool(
    tasks: Sequence, *, step: int, total_steps: int, stages: Sequence[int]
) -> list:
    """Expose progressively harder metadata.difficulty stages."""

    if not stages:
        return list(tasks)
    ordered = sorted(set(int(stage) for stage in stages))
    fraction = min(max(step, 0), max(total_steps - 1, 0)) / max(total_steps, 1)
    stage_index = min(int(fraction * len(ordered)), len(ordered) - 1)
    ceiling = ordered[stage_index]
    eligible = [
        task
        for task in tasks
        if int((task.metadata or {}).get("difficulty", ordered[0])) <= ceiling
    ]
    return eligible or list(tasks)


def reward_diversity(
    rewards,
    group_size: int,
    *,
    minimum_rate: float,
    maximum_rate: float,
    minimum_active_groups: int,
) -> dict[str, Any]:
    """Require successes, failures, and enough mixed groups before relative RL."""

    values = [float(value) for value in rewards]
    if not values or len(values) % group_size:
        raise ValueError("preflight rewards must form complete rollout groups")
    rate = fmean(values)
    active = 0
    for start in range(0, len(values), group_size):
        group = values[start : start + group_size]
        active += int(max(group) - min(group) > 1e-8)
    gates = {
        "minimum_reward_rate": rate >= minimum_rate,
        "maximum_reward_rate": rate <= maximum_rate,
        "active_groups": active >= minimum_active_groups,
    }
    return {
        "passed": all(gates.values()),
        "reward_rate": rate,
        "active_groups": active,
        "groups": len(values) // group_size,
        "thresholds": {
            "minimum_rate": minimum_rate,
            "maximum_rate": maximum_rate,
            "minimum_active_groups": minimum_active_groups,
        },
        "gates": gates,
    }


def task_reward_summary(tasks: Sequence, rewards, group_size: int) -> dict[str, Any]:
    """Collapse repeated rollouts into paired task and family reward vectors."""

    values = [float(value) for value in rewards]
    if len(values) != len(tasks) * group_size:
        raise ValueError("reward count does not match tasks × group size")
    task_rewards, family_values = {}, defaultdict(list)
    for index, task in enumerate(tasks):
        score = fmean(values[index * group_size : (index + 1) * group_size])
        task_rewards[task.id] = score
        family_values[_family(task)].append(score)
    return {
        "task_rewards": task_rewards,
        "family_rewards": {
            name: fmean(rows) for name, rows in sorted(family_values.items())
        },
    }


def paired_bootstrap(
    baseline: Mapping[str, float],
    candidate: Mapping[str, float],
    *,
    samples: int,
    confidence: float,
    seed: int,
) -> dict[str, Any]:
    """Paired percentile-bootstrap interval over held-out task deltas."""

    ids = sorted(set(baseline) & set(candidate))
    if not ids or set(ids) != set(baseline) or set(ids) != set(candidate):
        raise ValueError("baseline and candidate must cover the same held-out task ids")
    deltas = [float(candidate[key]) - float(baseline[key]) for key in ids]
    mean = fmean(deltas)
    if len(deltas) == 1 or samples <= 0:
        low = high = mean
    else:
        rng = random.Random(seed)
        boot = sorted(
            fmean(deltas[rng.randrange(len(deltas))] for _ in deltas)
            for _ in range(samples)
        )
        tail = (1.0 - confidence) / 2.0
        low = boot[min(int(math.floor(tail * len(boot))), len(boot) - 1)]
        high = boot[min(int(math.ceil((1.0 - tail) * len(boot))) - 1, len(boot) - 1)]
    return {
        "tasks": len(ids),
        "delta_mean": mean,
        "ci_low": low,
        "ci_high": high,
        "confidence": confidence,
        "bootstrap_samples": max(0, int(samples)),
        "paired_deltas": dict(zip(ids, deltas)),
    }


def promotion_gates(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    minimum_delta: float,
    updates_applied: int,
    maximum_family_regression: float,
    require_confidence: bool,
    bootstrap_samples: int,
    confidence: float,
    seed: int,
    split_audit: Mapping[str, Any],
    rollout_tokens: int,
    elapsed_seconds: float,
    maximum_rollout_tokens: int = 0,
    maximum_train_seconds: float = 0.0,
) -> dict[str, Any]:
    """Evaluate independent quality, leakage, regression, and budget gates."""

    paired = paired_bootstrap(
        baseline["task_rewards"],
        candidate["task_rewards"],
        samples=bootstrap_samples,
        confidence=confidence,
        seed=seed,
    )
    base_families = baseline.get("family_rewards", {})
    candidate_families = candidate.get("family_rewards", {})
    family_deltas = {
        name: float(candidate_families.get(name, -math.inf)) - float(value)
        for name, value in base_families.items()
    }
    gates = {
        "informative_update": updates_applied > 0,
        "minimum_point_delta": paired["delta_mean"] >= minimum_delta,
        "confidence_lower_bound": (
            not require_confidence or paired["ci_low"] >= minimum_delta
        ),
        "family_regression": all(
            delta >= -maximum_family_regression for delta in family_deltas.values()
        ),
        "split_contamination": bool(split_audit.get("passed")),
        "rollout_token_budget": (
            maximum_rollout_tokens <= 0 or rollout_tokens <= maximum_rollout_tokens
        ),
        "wall_clock_budget": (
            maximum_train_seconds <= 0 or elapsed_seconds <= maximum_train_seconds
        ),
    }
    return {
        "eligible": all(gates.values()),
        "gates": gates,
        "paired": paired,
        "family_deltas": family_deltas,
        "budget": {
            "rollout_tokens": rollout_tokens,
            "maximum_rollout_tokens": maximum_rollout_tokens,
            "elapsed_seconds": elapsed_seconds,
            "maximum_train_seconds": maximum_train_seconds,
        },
    }
