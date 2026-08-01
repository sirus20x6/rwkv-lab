from types import SimpleNamespace

import pytest
import torch

from rwkv_lab.rwkv_pretrain import (
    build_optimizer,
    resolved_worker_component_contract,
)
from rwkv_lab.training_components import PowerCoolConfiguration, ScheduleImplementation


def test_rwkv_adamw_path_uses_registered_scalar_cpu_backend():
    model = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.LayerNorm(4))
    optimizer = build_optimizer(
        model.named_parameters(),
        "adamw",
        lr=3.0e-4,
        wd=0.1,
    )
    assert isinstance(optimizer, torch.optim.AdamW)
    assert optimizer.defaults["lr"] == pytest.approx(3.0e-4)
    assert optimizer.defaults["betas"] == pytest.approx((0.9, 0.95))
    assert optimizer.defaults["weight_decay"] == pytest.approx(0.1)
    assert optimizer.defaults["foreach"] is False
    assert optimizer.defaults["fused"] is False


def test_rwkv_worker_components_drive_powercool_and_optimizer() -> None:
    schedule = PowerCoolConfiguration(
        warmup_steps=10,
        max_steps=100,
        minimum_ratio=0.1,
        cooldown_fraction=0.2,
        power=2.0,
    )

    class Components:
        composition = SimpleNamespace(composition_digest="sha256:" + "e" * 64)

        def configuration(self, slot, *, category):
            assert (slot, category) == ("optimizer", "optimizer")
            return {
                "learning_rate": 3.0e-4,
                "beta1": 0.9,
                "beta2": 0.95,
                "epsilon": 1.0e-8,
                "weight_decay": 0.1,
                "foreach": False,
                "fused": False,
            }

        def learning_rate_configuration(self):
            return ScheduleImplementation.POWERCOOL_V1, schedule

        def evidence(self):
            return {"optimizer": {"implementation": "test"}}

        def optimizer(self, parameters):
            return torch.optim.AdamW(parameters, lr=3.0e-4)

    components = Components()
    args = SimpleNamespace(
        optimizer="adamw",
        lr_schedule="powercool",
        u_mup_base_width=0,
        lr=3.0e-4,
        weight_decay=0.1,
    )
    resolved, evidence, composition_digest = resolved_worker_component_contract(
        args, schedule, components
    )
    assert resolved == schedule
    assert evidence == components.evidence()
    assert composition_digest == components.composition.composition_digest

    model = torch.nn.Linear(3, 4)
    optimizer = build_optimizer(
        model.named_parameters(),
        "adamw",
        lr=args.lr,
        wd=args.weight_decay,
        worker_components=components,
    )
    assert isinstance(optimizer, torch.optim.AdamW)

    args.lr_schedule = "cosine"
    with pytest.raises(ValueError, match="requires PowerCool"):
        resolved_worker_component_contract(args, schedule, components)
