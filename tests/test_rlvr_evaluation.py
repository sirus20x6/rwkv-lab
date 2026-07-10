from __future__ import annotations

from rwkv_lab.rlvr_evaluation import (
    audit_task_splits,
    curriculum_pool,
    paired_bootstrap,
    promotion_gates,
    reward_diversity,
    stratified_tasks,
    task_reward_summary,
)
from rwkv_lab.rlvr_train import RLVRTask


def task(name, split, family, difficulty=1, prompt=None):
    return RLVRTask(
        name,
        prompt or f"solve {name}",
        {"kind": "numeric", "expected": 1},
        split,
        {"family": family, "difficulty": difficulty},
    )


def test_split_audit_rejects_normalized_prompt_leakage():
    train = [task("a", "train", "add", prompt="Compute 1 + 1")]
    evaluate = [task("b", "eval", "add", prompt="  compute  1 + 1 ")]
    audit = audit_task_splits(train, evaluate)
    assert not audit["passed"]
    assert audit["normalized_prompt_overlap_count"] == 1


def test_curriculum_and_stratified_family_selection():
    tasks = [
        task("easy-a", "train", "add", 1),
        task("easy-b", "train", "sub", 1),
        task("hard", "train", "mul", 3),
    ]
    assert {
        row.id for row in curriculum_pool(tasks, step=0, total_steps=10, stages=[1, 3])
    } == {"easy-a", "easy-b"}
    assert {row.metadata["family"] for row in stratified_tasks(tasks, 3, seed=2)} == {
        "add",
        "sub",
        "mul",
    }


def test_reward_diversity_requires_mixed_groups():
    passed = reward_diversity(
        [0, 1, 0, 0], 2, minimum_rate=0.01, maximum_rate=0.99, minimum_active_groups=1
    )
    failed = reward_diversity(
        [0, 0, 0, 0], 2, minimum_rate=0.01, maximum_rate=0.99, minimum_active_groups=1
    )
    assert passed["passed"] and passed["active_groups"] == 1
    assert not failed["passed"]


def test_task_reward_summary_and_paired_promotion_gates():
    tasks = [task("a", "eval", "add"), task("b", "eval", "sub")]
    baseline = task_reward_summary(tasks, [0, 0, 0, 0], 2)
    candidate = task_reward_summary(tasks, [1, 1, 1, 1], 2)
    stats = paired_bootstrap(
        baseline["task_rewards"],
        candidate["task_rewards"],
        samples=100,
        confidence=0.95,
        seed=4,
    )
    assert stats["ci_low"] == stats["ci_high"] == 1
    decision = promotion_gates(
        baseline,
        candidate,
        minimum_delta=0.1,
        updates_applied=1,
        maximum_family_regression=0,
        require_confidence=True,
        bootstrap_samples=100,
        confidence=0.95,
        seed=4,
        split_audit={"passed": True},
        rollout_tokens=100,
        elapsed_seconds=2,
        maximum_rollout_tokens=200,
        maximum_train_seconds=3,
    )
    assert decision["eligible"] and all(decision["gates"].values())


def test_family_regression_blocks_aggregate_improvement():
    baseline = {
        "task_rewards": {"a": 0, "b": 1},
        "family_rewards": {"add": 0, "sub": 1},
    }
    candidate = {
        "task_rewards": {"a": 1, "b": 0},
        "family_rewards": {"add": 1, "sub": 0},
    }
    decision = promotion_gates(
        baseline,
        candidate,
        minimum_delta=0,
        updates_applied=1,
        maximum_family_regression=0.1,
        require_confidence=False,
        bootstrap_samples=10,
        confidence=0.95,
        seed=1,
        split_audit={"passed": True},
        rollout_tokens=1,
        elapsed_seconds=1,
    )
    assert not decision["eligible"] and not decision["gates"]["family_regression"]
