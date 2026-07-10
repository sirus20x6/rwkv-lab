from __future__ import annotations

import json
import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from rwkv_lab.recursive_improve import (
    PROPOSAL_RESPONSE_SCHEMA,
    run_loop,
    validate_proposal,
)


def args(tmp_path: Path, **overrides):
    values = {
        "ckpt": str(tmp_path / "parent.pt"),
        "out": str(tmp_path / "loop"),
        "heldout_tasks": str(tmp_path / "heldout.jsonl"),
        "proposal_command": "proposal",
        "proposal_timeout": 2.0,
        "max_proposal_bytes": 100_000,
        "max_proposal_tasks": 8,
        "verifier_command": "",
        "verifier_timeout": 2.0,
        "rounds": 1,
        "max_consecutive_rejections": 2,
        "max_total_rollout_tokens": 1000,
        "max_round_seconds": 30,
        "algorithm": "gspo",
        "steps": 4,
        "prompts_per_step": 2,
        "group_size": 4,
        "max_new": 16,
        "lr": 1e-6,
        "max_lr": 1e-5,
        "reference": "rollout",
        "rollout_engine": "auto",
        "sft_steps": 2,
        "sft_batch_size": 1,
        "sft_lr": 2e-5,
        "preflight_prompts": 2,
        "min_preflight_reward": 0.01,
        "max_preflight_reward": 0.99,
        "min_preflight_active_groups": 1,
        "eval_prompts": 4,
        "eval_group_size": 2,
        "min_heldout_delta": 0.01,
        "confidence": 0.95,
        "bootstrap_samples": 100,
        "require_confidence": True,
        "max_family_regression": 0.0,
        "log_samples": 0,
        "seed": 0,
        "device": "cpu",
        "resume": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def proposal(**training):
    return {
        "schema": PROPOSAL_RESPONSE_SCHEMA,
        "proposal_id": "p1",
        "rationale": "start with easy addition",
        "training": training,
        "train_tasks": [
            {
                "id": "train-1",
                "split": "train",
                "prompt": "Compute 1+1",
                "verifier": {"kind": "numeric", "expected": 2},
                "metadata": {"family": "add", "sft_answer": "2"},
            }
        ],
    }


def test_proposal_controls_are_strictly_capped(tmp_path):
    cfg = args(tmp_path)
    assert (
        validate_proposal(proposal(steps=4, algorithm="dapo"), cfg)["training"]["steps"]
        == 4
    )
    with pytest.raises(ValueError, match="outside"):
        validate_proposal(proposal(steps=5), cfg)
    with pytest.raises(ValueError, match="unsupported"):
        validate_proposal(proposal(weight_decay=1), cfg)


def test_recursive_loop_promotes_only_trainer_eligible_checkpoint(
    monkeypatch, tmp_path
):
    cfg = args(tmp_path)
    Path(cfg.ckpt).write_bytes(b"parent")
    Path(cfg.heldout_tasks).write_text(
        json.dumps(
            {
                "id": "eval-1",
                "split": "eval",
                "prompt": "Compute 2+2",
                "verifier": {"kind": "numeric", "expected": 4},
            }
        )
        + "\n"
    )

    def fake_run(command, **kwargs):
        if command == ["proposal"]:
            return SimpleNamespace(
                returncode=0, stdout=json.dumps(proposal()), stderr=""
            )
        candidate_dir = Path(command[command.index("--out") + 1])
        candidate_dir.mkdir(parents=True, exist_ok=True)
        checkpoint = candidate_dir / "rlvr.pt"
        checkpoint.write_bytes(b"candidate")
        result = {
            "status": "complete",
            "checkpoint": str(checkpoint),
            "checkpoint_parent_sha256": hashlib.sha256(b"parent").hexdigest(),
            "training_status": "complete",
            "total_rollout_tokens": 10,
            "promotion": {
                "eligible": True,
                "heldout_delta": 0.2,
                "gates": {"heldout": True},
            },
        }
        (candidate_dir / "result.json").write_text(json.dumps(result))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("rwkv_lab.recursive_improve.subprocess.run", fake_run)
    result = run_loop(cfg)
    assert result["status"] == "complete" and result["promotions"] == 1
    assert result["current_checkpoint"].endswith("rlvr.pt")
    assert result["iterations"][0]["accepted"]
