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
            "torch_adamw",
            {
                "learning_rate": 1e-3,
                "beta1": 0.9,
                "beta2": 0.999,
                "epsilon": 1e-8,
                "weight_decay": 0.01,
                "foreach": True,
                "fused": False,
            },
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
    schedule = runtime.learning_rate_schedule(optimizer)
    parameter.grad = torch.tensor([4.0])
    gradient_norm = runtime.gradient_clipping([parameter])

    assert isinstance(optimizer, torch.optim.AdamW)
    assert isinstance(schedule, torch.optim.lr_scheduler.LRScheduler)
    assert schedule.get_last_lr() == pytest.approx([1e-3])
    assert gradient_norm == pytest.approx(4.0)
    assert parameter.grad == pytest.approx(torch.tensor([1.0]))
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
