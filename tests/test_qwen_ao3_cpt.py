import json
from dataclasses import asdict, replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from rwkv_lab.qwen_ao3_cpt import (
    SCHEMA,
    PackedRows,
    QwenAO3Config,
    _learning_rate_schedule,
    _optimizer_configuration,
    _resolve_auto_resume,
    _resume_contract,
    _write_status,
    resolved_worker_component_contract,
)


def config(tmp_path, **overrides):
    model = tmp_path / "model"
    train = tmp_path / "train"
    eval_ = tmp_path / "eval"
    for path in (model, train, eval_):
        path.mkdir(exist_ok=True)
    values = {
        "model_dir": str(model),
        "train_pack_dir": str(train),
        "eval_pack_dir": str(eval_),
        "run_dir": str(tmp_path / "run"),
    }
    values.update(overrides)
    return QwenAO3Config(**values)


def test_config_rejects_invalid_rank_and_context(tmp_path):
    with pytest.raises(ValueError, match="context_length"):
        config(tmp_path, context_length=64).validate()
    with pytest.raises(ValueError, match="expert_rank"):
        config(tmp_path, expert_rank=0).validate()


def test_packed_rows_validates_size_and_returns_int64_tensor(tmp_path):
    directory = tmp_path / "pack"
    directory.mkdir()
    values = np.arange(16, dtype=np.uint32).reshape(2, 8)
    values.tofile(directory / "train.ctx8.bin")
    (directory / "manifest.json").write_text(
        json.dumps(
            {
                "context_length": 8,
                "rows": 2,
                "packed_file": "train.ctx8.bin",
            }
        )
    )
    rows = PackedRows(directory, 8)
    tensor = rows.tensor(1, torch.device("cpu"))
    assert tensor.dtype == torch.int64
    assert tensor.tolist() == [list(range(8, 16))]


def test_packed_rows_rejects_truncated_file(tmp_path):
    directory = tmp_path / "pack"
    directory.mkdir()
    np.arange(4, dtype=np.uint32).tofile(directory / "train.ctx8.bin")
    (directory / "manifest.json").write_text(
        json.dumps(
            {
                "context_length": 8,
                "rows": 1,
                "packed_file": "train.ctx8.bin",
            }
        )
    )
    with pytest.raises(ValueError, match="packed size"):
        PackedRows(directory, 8)


def test_auto_resume_resolves_latest_checkpoint(tmp_path):
    checkpoint = tmp_path / "run" / "step-000010"
    checkpoint.mkdir(parents=True)
    (tmp_path / "run" / "latest.json").write_text(
        json.dumps(
            {
                "schema": SCHEMA,
                "step": 10,
                "checkpoint": str(checkpoint),
            }
        )
    )
    resolved = _resolve_auto_resume(config(tmp_path), tmp_path / "run")
    assert resolved.resume == str(checkpoint.resolve())


def test_explicit_resume_is_not_replaced(tmp_path):
    resolved = _resolve_auto_resume(
        config(tmp_path, resume="/explicit/checkpoint"), tmp_path / "run"
    )
    assert resolved.resume == "/explicit/checkpoint"


def test_auto_resume_recovers_checkpoint_when_latest_is_missing(tmp_path):
    checkpoint = tmp_path / "run" / "step-000010"
    checkpoint.mkdir(parents=True)
    (checkpoint / "trainer-state.pt").touch()
    (checkpoint / "state.json").write_text("{}")
    resolved = _resolve_auto_resume(config(tmp_path), tmp_path / "run")
    assert resolved.resume == str(checkpoint.resolve())


def test_log_cadence_is_not_part_of_resume_contract(tmp_path):
    first = config(tmp_path, log_every=10)
    second = config(tmp_path, log_every=1)
    assert _resume_contract(first) == _resume_contract(second)
    live = replace(
        first,
        learning_rate=first.learning_rate * 0.2,
        min_learning_rate=first.min_learning_rate * 0.2,
        eval_every=25,
    )
    mutable = ("learning_rate", "eval_every")
    assert _resume_contract(
        first, mutable_control_keys=mutable
    ) == _resume_contract(live, mutable_control_keys=mutable)
    assert _resume_contract(first) != _resume_contract(live)


def test_status_heartbeat_is_atomic_and_parseable(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    _write_status(run, state="training", step=3, cursor=27, total_steps=100)
    status = json.loads((run / "status.json").read_text())
    assert status["state"] == "training"
    assert status["step"] == 3
    assert status["cursor"] == 27
    assert status["total_steps"] == 100
    assert not (run / ".status.json.tmp").exists()


def test_qwen_optimizer_uses_registered_fused_adamw_configuration(tmp_path):
    optimizer = _optimizer_configuration(config(tmp_path))
    assert optimizer.fused
    assert not optimizer.foreach
    assert optimizer.beta1 == pytest.approx(0.9)
    assert optimizer.beta2 == pytest.approx(0.95)
    schedule = _learning_rate_schedule(config(tmp_path), total_steps=100)
    assert schedule.max_steps == 100
    assert schedule.minimum_ratio == pytest.approx(0.1)


def test_qwen_worker_components_are_exact_and_enter_checkpoint_identity(tmp_path):
    value = config(tmp_path, max_steps=100)
    total_steps = 80
    configurations = {
        "optimizer": {
            key: item
            for key, item in asdict(_optimizer_configuration(value)).items()
            if key != "weight_decay"
        },
        "weight_decay": {"weight_decay": value.weight_decay},
        "learning_rate": asdict(_learning_rate_schedule(value, total_steps)),
        "gradient_clipping": {
            "max_norm": value.max_grad_norm,
            "norm_type": 2.0,
            "error_if_nonfinite": True,
        },
    }

    class Components:
        composition = SimpleNamespace(composition_digest="sha256:" + "c" * 64)

        def configuration(self, slot, *, category):
            assert category in {
                "optimizer",
                "learning_rate_schedule",
                "weight_decay_schedule",
                "gradient_clipping",
            }
            return configurations[slot]

        def evidence(self):
            return {
                slot: {
                    "category": slot,
                    "implementation": slot,
                    "descriptor_digest": "sha256:" + "d" * 64,
                }
                for slot in configurations
            }

    components = Components()
    evidence, composition_digest = resolved_worker_component_contract(
        value, total_steps, components
    )
    assert set(evidence) == {
        "optimizer",
        "learning_rate",
        "weight_decay",
        "gradient_clipping",
    }
    assert composition_digest == components.composition.composition_digest

    live = replace(
        value,
        learning_rate=value.learning_rate * 0.2,
        min_learning_rate=value.min_learning_rate * 0.2,
    )
    controls = SimpleNamespace(
        effective_values={"learning_rate": live.learning_rate}
    )
    live_evidence, live_digest = resolved_worker_component_contract(
        live, total_steps, components, controls
    )
    assert live_evidence == evidence
    assert live_digest == composition_digest

    configurations["learning_rate"] = {
        **configurations["learning_rate"],
        "minimum_ratio": 0.5,
    }
    with pytest.raises(ValueError, match="learning_rate_schedule composition"):
        resolved_worker_component_contract(value, total_steps, components)
