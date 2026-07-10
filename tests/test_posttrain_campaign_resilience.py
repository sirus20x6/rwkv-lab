import json
import subprocess
from types import SimpleNamespace

from rwkv_lab.posttrain_campaign import _execute_arm


def _args(tmp_path):
    return SimpleNamespace(
        checkpoint=str(tmp_path / "parent.pt"), data=str(tmp_path / "train.jsonl"),
        eval_data=str(tmp_path / "eval.jsonl"), rank=4, adapter_alpha=8.0, steps=2,
        batch_size=2, learning_rate=1e-4, beta=0.1, gamma=1.0, max_length=64,
        token_budget=100, packing="reset", base_quantization="none",
        quant_block_size=64, quant_backend="auto", token_cache="", targets="",
        activation_offload=False, device="cpu", resume=True, retries=1,
        retry_delay=0.0, arm_timeout=10.0,
    )


def test_arm_retry_state_and_completed_resume(tmp_path, monkeypatch):
    args = _args(tmp_path)
    run_dir = tmp_path / "run"
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        if len(calls) == 1:
            raise subprocess.TimeoutExpired(command, timeout=10)
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "posttrain-result.json").write_text(json.dumps({
            "schema": "rwkv-lab.posttrain-result.v1", "objective": "sft", "seed": 7,
        }))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    first = _execute_arm(args, "sft", 7, "explore", run_dir, "cpu")
    assert first["returncode"] == 0 and first["attempts"] == 2 and not first["resumed"]
    state = json.loads((run_dir / "arm-state.json").read_text())
    assert state["status"] == "complete"
    assert state["attempts"][0]["timed_out"] and state["attempts"][1]["status"] == "complete"

    second = _execute_arm(args, "sft", 7, "explore", run_dir, "cpu")
    assert second["resumed"] and second["attempts"] == 2 and len(calls) == 2


def test_arm_command_identity_refuses_mutated_resume(tmp_path, monkeypatch):
    args = _args(tmp_path)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    state = {"schema": "rwkv-lab.posttrain-arm-state.v1", "objective": "sft", "seed": 1,
             "phase": "explore", "command_sha256": "wrong", "attempts": [], "status": "failed"}
    (run_dir / "arm-state.json").write_text(json.dumps(state))
    try:
        _execute_arm(args, "sft", 1, "explore", run_dir, "cpu")
    except ValueError as exc:
        assert "command differs" in str(exc)
    else:
        raise AssertionError("mutated arm command should fail closed")
