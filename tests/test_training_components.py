import json
from pathlib import Path

import pytest
import torch

from rwkv_lab.mage_flow_optimizations import FP32MasterAdamW
from rwkv_lab.training_components import (
    AdamWConfiguration,
    AppearanceExpertRoutingConfiguration,
    LinearWarmupCosineConfiguration,
    OptimizerImplementation,
    ParameterRouterImplementation,
    ScheduleImplementation,
    TerminalExpertRoutingConfiguration,
    build_registered_optimizer,
    build_registered_parameter_routing,
    build_registered_schedule,
    linear_warmup_cosine_multiplier,
    optimizer_from_resolved_component,
    parameter_routing_from_resolved_component,
    schedule_from_resolved_component,
    supported_implementation_ids,
    supported_worker_capabilities,
)


def test_linear_warmup_cosine_has_one_optimizer_step_domain_trajectory():
    configuration = LinearWarmupCosineConfiguration(
        warmup_steps=2, max_steps=6, minimum_ratio=0.1
    )
    trajectory = [
        linear_warmup_cosine_multiplier(step, configuration) for step in range(8)
    ]
    assert trajectory[0] == pytest.approx(1.0e-8)
    assert trajectory[1] == pytest.approx(0.5)
    assert trajectory[2] == pytest.approx(1.0)
    assert trajectory[4] == pytest.approx(0.55)
    assert trajectory[6:] == pytest.approx([0.1, 0.1])


def test_registered_factories_preserve_group_rates_and_exact_state():
    expert = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.bfloat16))
    backbone = torch.nn.Parameter(torch.tensor([2.0], dtype=torch.bfloat16))
    configuration = AdamWConfiguration(
        learning_rate=1.0e-3,
        beta1=0.9,
        beta2=0.95,
        epsilon=1.0e-8,
        weight_decay=0.01,
    )
    optimizer = build_registered_optimizer(
        OptimizerImplementation.FP32_MASTER_ADAMW_V1,
        [
            {"params": [expert], "lr": 1.0e-3, "group_name": "expert"},
            {"params": [backbone], "lr": 5.0e-4, "group_name": "backbone"},
        ],
        configuration,
    )
    assert isinstance(optimizer, FP32MasterAdamW)
    assert [group["lr"] for group in optimizer.param_groups] == pytest.approx(
        [1.0e-3, 5.0e-4]
    )

    schedule = build_registered_schedule(
        ScheduleImplementation.LINEAR_WARMUP_COSINE_V1,
        optimizer,
        LinearWarmupCosineConfiguration(warmup_steps=0, max_steps=2, minimum_ratio=0.2),
    )
    assert schedule.state_dict()["last_epoch"] == 0
    assert FP32MasterAdamW.master_state_key in optimizer.state_dict()


def test_component_catalog_and_runtime_dispatch_are_exactly_aligned():
    root = Path(__file__).resolve().parents[1]
    document = json.loads(
        (
            root / "docs/experiment-vm/examples/mageflow-training-components.json"
        ).read_text(encoding="utf-8")
    )
    implementations = {
        component["implementation"] for component in document["components"]
    }
    capabilities = {
        capability
        for component in document["components"]
        for capability in component["required_capabilities"]
    }
    assert document["api_version"] == "trainvm.training-components/v1"
    assert implementations == supported_implementation_ids()
    assert capabilities == supported_worker_capabilities()
    grades = {
        component["key"]["category"]: component["state_grade"]
        for component in document["components"]
    }
    assert grades["optimizer"] == "exact"
    assert grades["learning_rate_schedule"] == "exact"
    assert grades["parameter_router"] == "stateless"


def test_resolved_worker_component_dispatch_is_closed_and_typed():
    root = Path(__file__).resolve().parents[1]
    components = json.loads(
        (
            root / "docs/experiment-vm/examples/mageflow-training-components.json"
        ).read_text(encoding="utf-8")
    )["components"]
    optimizer_descriptor = next(
        component
        for component in components
        if component["key"]["name"] == "torch_adamw"
    )
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer_component = {
        "descriptor": optimizer_descriptor,
        "descriptor_digest": "sha256:" + "a" * 64,
        "configuration": {
            "learning_rate": 1.0e-3,
            "beta1": 0.9,
            "beta2": 0.999,
            "epsilon": 1.0e-8,
            "weight_decay": 0.01,
            "foreach": True,
        },
    }
    optimizer = optimizer_from_resolved_component(optimizer_component, [parameter])
    assert isinstance(optimizer, torch.optim.AdamW)

    schedule_descriptor = next(
        component
        for component in components
        if component["key"]["category"] == "learning_rate_schedule"
    )
    schedule = schedule_from_resolved_component(
        {
            "descriptor": schedule_descriptor,
            "descriptor_digest": "sha256:" + "b" * 64,
            "configuration": {
                "warmup_steps": 0,
                "max_steps": 10,
                "minimum_ratio": 0.1,
            },
        },
        optimizer,
    )
    assert isinstance(schedule, torch.optim.lr_scheduler.LRScheduler)

    forged = dict(optimizer_component)
    forged["runtime_import"] = "malicious.module"
    with pytest.raises(ValueError, match="unknown fields"):
        optimizer_from_resolved_component(forged, [parameter])


def test_registered_mageflow_routers_own_expert_backbone_and_repa_once():
    expert = torch.nn.Parameter(torch.ones(2))
    backbone = torch.nn.Parameter(torch.ones(3))
    repa = torch.nn.Parameter(torch.ones(4))
    appearance = build_registered_parameter_routing(
        ParameterRouterImplementation.MAGEFLOW_APPEARANCE_EXPERT_V1,
        [("expert", expert), ("backbone", backbone)],
        {"expert": frozenset({id(expert)})},
        base_learning_rate=2.0e-5,
        configuration=AppearanceExpertRoutingConfiguration(
            shared_backbone_multiplier=0.5
        ),
    )
    assert [group["group_name"] for group in appearance.groups] == [
        "experts",
        "shared_backbone",
    ]
    assert [group["lr"] for group in appearance.groups] == pytest.approx(
        [2.0e-5, 1.0e-5]
    )

    terminal = build_registered_parameter_routing(
        ParameterRouterImplementation.MAGEFLOW_TERMINAL_EXPERT_V1,
        [("expert", expert), ("backbone", backbone), ("repa", repa)],
        {
            "expert": frozenset({id(expert)}),
            "repa": frozenset({id(repa)}),
        },
        base_learning_rate=2.0e-5,
        configuration=TerminalExpertRoutingConfiguration(
            shared_backbone_multiplier=0.5,
            repa_projection_multiplier=2.0,
        ),
    )
    assert [group["group_name"] for group in terminal.groups] == [
        "terminal_expert",
        "shared_backbone",
        "vae_repa_projection",
    ]
    assert [group["lr"] for group in terminal.groups] == pytest.approx(
        [2.0e-5, 1.0e-5, 4.0e-5]
    )


def test_resolved_parameter_router_dispatch_uses_exact_catalog_defaults():
    root = Path(__file__).resolve().parents[1]
    components = json.loads(
        (
            root / "docs/experiment-vm/examples/mageflow-training-components.json"
        ).read_text(encoding="utf-8")
    )["components"]
    descriptor = next(
        component
        for component in components
        if component["key"]["name"] == "mageflow_appearance_expert"
    )
    expert = torch.nn.Parameter(torch.ones(1))
    backbone = torch.nn.Parameter(torch.ones(1))
    result = parameter_routing_from_resolved_component(
        {
            "descriptor": descriptor,
            "descriptor_digest": "sha256:" + "c" * 64,
            "configuration": {"shared_backbone_multiplier": 0.5},
        },
        [("expert", expert), ("backbone", backbone)],
        {"expert": frozenset({id(expert)})},
        base_learning_rate=2.0e-5,
    )
    assert result.report["passed"]
    assert [group["lr"] for group in result.groups] == pytest.approx([2.0e-5, 1.0e-5])


@pytest.mark.parametrize(
    "configuration",
    [
        AdamWConfiguration,
        LinearWarmupCosineConfiguration,
    ],
)
def test_invalid_component_configuration_fails_before_tensor_construction(
    configuration,
):
    with pytest.raises((TypeError, ValueError)):
        if configuration is AdamWConfiguration:
            configuration(learning_rate=float("nan"))
        else:
            configuration(warmup_steps=-1, max_steps=10)
