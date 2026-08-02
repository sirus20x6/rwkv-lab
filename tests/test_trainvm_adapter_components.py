from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest
import torch

from rwkv_lab.trainvm_adapters import AdapterComponentError, WorkerTrainingComponents
from rwkv_lab.trainvm_worker import load_resolved_training_composition


def canonical(value: object) -> bytes:
    return json.dumps(value, separators=(",", ":"), sort_keys=True).encode()


def digest(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def composition():
    root = Path(__file__).resolve().parents[1]
    registry = json.loads(
        (root / "docs/experiment-vm/examples/training-components.v1.json").read_text(
            encoding="utf-8"
        )
    )
    requested = {
        "optimizer": (
            "torch_adamw_no_decay",
            {
                "learning_rate": 1e-3,
                "beta1": 0.9,
                "beta2": 0.999,
                "epsilon": 1e-8,
                "foreach": True,
                "fused": False,
            },
        ),
        "weight_decay": (
            "constant",
            {"weight_decay": 0.01},
        ),
        "learning_rate": (
            "linear_warmup_cosine",
            {"warmup_steps": 0, "max_steps": 4, "minimum_ratio": 0.1},
        ),
        "gradient_clipping": (
            "global_norm",
            {
                "max_norm": 1.0,
                "norm_type": 2.0,
                "error_if_nonfinite": True,
            },
        ),
        "gradient_accumulation": (
            "fixed",
            {"microbatches_per_optimizer_step": 4},
        ),
        "objective": (
            "linear_head_cross_entropy",
            {"chunk_size": 2, "prefer_fused": False},
        ),
    }
    components = {}
    for slot, (name, configuration) in requested.items():
        descriptor = next(
            item for item in registry["components"] if item["key"]["name"] == name
        )
        components[slot] = {
            "configuration": configuration,
            "descriptor": descriptor,
            "descriptor_digest": digest(canonical(descriptor)),
        }
    body = {
        "api_version": "trainvm.resolved-training-composition/v1",
        "components": components,
        "model_family": "rwkv",
        "registry_digest": digest(canonical(registry)),
    }
    return load_resolved_training_composition(
        {**body, "composition_digest": digest(canonical(body))}
    )


def test_worker_component_bridge_builds_optimizer_and_schedule() -> None:
    runtime = WorkerTrainingComponents(composition(), "rwkv")
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = runtime.optimizer([parameter])
    decay = runtime.weight_decay_schedule(optimizer)
    schedule = runtime.learning_rate_schedule(optimizer)
    parameter.grad = torch.tensor([4.0])
    gradient_norm = runtime.gradient_clipping([parameter])
    accumulation = runtime.gradient_accumulation()
    objective = runtime.objective()

    assert isinstance(optimizer, torch.optim.AdamW)
    assert optimizer.param_groups[0]["weight_decay"] == pytest.approx(0.01)
    assert decay.state_dict() == {}
    assert isinstance(schedule, torch.optim.lr_scheduler.LRScheduler)
    assert schedule.get_last_lr() == pytest.approx([1e-3])
    assert gradient_norm == pytest.approx(4.0)
    assert parameter.grad == pytest.approx(torch.tensor([1.0]))
    assert accumulation.microbatches_per_optimizer_step == 4
    assert accumulation.scale_loss(torch.tensor(8.0)) == pytest.approx(2.0)
    hidden = torch.randn(1, 2, 3)
    head = torch.nn.Linear(3, 5)
    labels = torch.tensor([[1, 2]])
    torch.testing.assert_close(
        objective(hidden, head, labels),
        torch.nn.functional.cross_entropy(
            head(hidden).reshape(-1, 5), labels.reshape(-1)
        ),
    )
    assert runtime.evidence()["optimizer"]["category"] == "optimizer"


def test_worker_component_bridge_rejects_family_and_slot_category_confusion() -> None:
    resolved = composition()
    with pytest.raises(AdapterComponentError, match="different model family"):
        WorkerTrainingComponents(resolved, "transformer")

    runtime = WorkerTrainingComponents(resolved, "rwkv")
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    with pytest.raises(ValueError, match="not 'optimizer'"):
        runtime.optimizer([parameter], slot="learning_rate")


def test_worker_optimizer_and_schedule_resume_the_same_next_update() -> None:
    runtime = WorkerTrainingComponents(composition(), "rwkv")
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = runtime.optimizer([parameter])
    schedule = runtime.learning_rate_schedule(optimizer)
    for _ in range(2):
        parameter.grad = torch.tensor([0.25])
        optimizer.step()
        schedule.step()

    resumed_parameter = torch.nn.Parameter(parameter.detach().clone())
    resumed_optimizer = runtime.optimizer([resumed_parameter])
    resumed_schedule = runtime.learning_rate_schedule(resumed_optimizer)
    resumed_optimizer.load_state_dict(copy.deepcopy(optimizer.state_dict()))
    resumed_schedule.load_state_dict(copy.deepcopy(schedule.state_dict()))
    assert resumed_schedule.last_epoch == schedule.last_epoch
    assert resumed_schedule.get_last_lr() == pytest.approx(schedule.get_last_lr())

    parameter.grad = torch.tensor([0.5])
    resumed_parameter.grad = torch.tensor([0.5])
    optimizer.step()
    schedule.step()
    resumed_optimizer.step()
    resumed_schedule.step()
    assert resumed_parameter.detach() == pytest.approx(parameter.detach())
    assert resumed_schedule.get_last_lr() == pytest.approx(schedule.get_last_lr())
