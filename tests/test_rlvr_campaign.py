from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from rwkv_lab.rlvr_campaign import Arm, build_command, run_campaign, summarize


def campaign_args(tmp_path: Path, **overrides):
    values = {
        "ckpt": str(tmp_path / "parent.pt"),
        "out": str(tmp_path / "campaign"),
        "tasks": "",
        "heldout_tasks": "",
        "algorithms": "gspo,dr_grpo,dapo",
        "seeds": "0,1,2",
        "steps": 2,
        "prompts_per_step": 1,
        "group_size": 4,
        "epochs": 1,
        "max_new": 12,
        "rollout_engine": "auto",
        "temperature": 1.0,
        "eval_temperature": 0.0,
        "top_p": 1.0,
        "top_k": 0,
        "lr": 1e-6,
        "weight_decay": 0.0,
        "warmup": 1,
        "grad_clip": 1.0,
        "kl_coef": 0.01,
        "reference": "rollout",
        "reference_ckpt": "",
        "eval_every": 1,
        "eval_prompts": 2,
        "eval_group_size": 2,
        "min_heldout_delta": 0.01,
        "confidence": 0.95,
        "bootstrap_samples": 100,
        "require_confidence": True,
        "max_family_regression": 0.0,
        "max_rollout_tokens": 0,
        "max_train_seconds": 0.0,
        "train_tasks": 32,
        "eval_tasks": 8,
        "difficulty": 1,
        "curriculum_stages": "1,2",
        "sft_steps": 2,
        "sft_batch_size": 1,
        "sft_lr": 2e-5,
        "preflight_prompts": 2,
        "min_preflight_reward": 0.01,
        "max_preflight_reward": 0.99,
        "min_preflight_active_groups": 1,
        "save_every": 2,
        "log_samples": 0,
        "verifier_command": "",
        "verifier_timeout": 10.0,
        "device": "cpu",
        "resume_existing": True,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_build_command_preserves_equal_budget_and_arm_identity(tmp_path):
    command = build_command(campaign_args(tmp_path), Arm("dapo", 7), tmp_path / "arm")
    assert command[1:3] == ["-m", "rwkv_lab.rlvr_train"]
    assert command[command.index("--algorithm") + 1] == "dapo"
    assert command[command.index("--seed") + 1] == "7"
    assert command[command.index("--steps") + 1] == "2"
    assert command[command.index("--group-size") + 1] == "4"


def test_summary_reports_only_complete_runs():
    rows = [
        {
            "algorithm": "gspo",
            "status": "complete",
            "baseline_heldout": {"reward": 0.1},
            "heldout": {"reward": 0.3},
            "promotion": {"eligible": True, "heldout_delta": 0.2, "updates_applied": 2},
        },
        {
            "algorithm": "gspo",
            "status": "complete",
            "baseline_heldout": {"reward": 0.2},
            "heldout": {"reward": 0.2},
            "promotion": {
                "eligible": False,
                "heldout_delta": 0.0,
                "updates_applied": 1,
            },
        },
        {"algorithm": "gspo", "status": "failed"},
    ]
    result = summarize(rows)["gspo"]
    assert result["runs"] == 2
    assert result["heldout_mean"] == 0.25
    assert result["baseline_mean"] == pytest.approx(0.15)
    assert result["delta_mean"] == pytest.approx(0.1)
    assert result["promotions"] == 1
    assert result["updates_applied"] == 3


def test_campaign_aggregates_all_arms(monkeypatch, tmp_path):
    args = campaign_args(tmp_path, algorithms="gspo,dapo", seeds="3,4")
    Path(args.ckpt).write_bytes(b"parent")

    def fake_run(command, **_kwargs):
        run_dir = Path(command[command.index("--out") + 1])
        algorithm = command[command.index("--algorithm") + 1]
        result = {
            "status": "complete",
            "baseline_heldout": {"reward": 0.1},
            "heldout": {"reward": 0.2},
            "promotion": {
                "eligible": algorithm == "gspo",
                "heldout_delta": 0.1,
                "updates_applied": 1,
            },
        }
        (run_dir / "result.json").write_text(json.dumps(result))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("rwkv_lab.rlvr_campaign.subprocess.run", fake_run)
    result = run_campaign(args)
    stored = json.loads((Path(args.out) / "campaign.json").read_text())
    assert result["status"] == stored["status"] == "complete"
    assert result["completed_arms"] == 4
    assert result["summary"]["gspo"]["promotions"] == 2
    assert result["summary"]["dapo"]["promotions"] == 0


def test_campaign_refuses_mismatched_resume(monkeypatch, tmp_path):
    args = campaign_args(tmp_path, algorithms="gspo", seeds="0")
    Path(args.ckpt).write_bytes(b"parent")

    def fake_run(command, **_kwargs):
        run_dir = Path(command[command.index("--out") + 1])
        (run_dir / "result.json").write_text(
            json.dumps(
                {
                    "status": "complete",
                    "baseline_heldout": {"reward": 0},
                    "heldout": {"reward": 0},
                    "promotion": {
                        "eligible": False,
                        "heldout_delta": 0,
                        "updates_applied": 0,
                    },
                }
            )
        )
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("rwkv_lab.rlvr_campaign.subprocess.run", fake_run)
    run_campaign(args)
    args.steps = 3
    with pytest.raises(ValueError, match="configuration differs"):
        run_campaign(args)
