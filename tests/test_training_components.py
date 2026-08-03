import ast
import json
from pathlib import Path

import pytest
import torch

from rwkv_lab.mage_flow_optimizations import FP32MasterAdamW
from rwkv_lab.training_components import (
    ActivationImplementation,
    AdamWConfiguration,
    AdamWNoDecayConfiguration,
    AppearanceExpertRoutingConfiguration,
    BFloat16PrecisionConfiguration,
    BFloat16PrecisionPolicy,
    ConstantLearningRateConfiguration,
    ConstantWeightDecayConfiguration,
    ContextLengthCurriculum,
    ContextLengthCurriculumConfiguration,
    CurriculumImplementation,
    FixedGradientAccumulation,
    FixedGradientAccumulationConfiguration,
    FP32ParametersBFloat16ComputeConfiguration,
    FP32ParametersBFloat16ComputePolicy,
    FullBackboneRoutingConfiguration,
    GlobalNormClippingConfiguration,
    GradientAccumulationImplementation,
    GradientClippingImplementation,
    LayerNormConfiguration,
    LayerNormFactory,
    LinearHeadCrossEntropyConfiguration,
    LinearHeadCrossEntropyObjective,
    LinearWarmupConstantConfiguration,
    LinearWarmupCosineConfiguration,
    NormalizationImplementation,
    ObjectiveImplementation,
    OptimizerImplementation,
    ParameterRouterImplementation,
    PowerCoolConfiguration,
    PrecisionImplementation,
    RegisteredActivation,
    RWKVMatrixOptimizerRoutingConfiguration,
    ScheduleImplementation,
    SpectralMuonConfiguration,
    TerminalExpertRoutingConfiguration,
    WeightDecayScheduleImplementation,
    build_registered_activation,
    build_registered_curriculum,
    build_registered_gradient_accumulation,
    build_registered_gradient_clipping,
    build_registered_normalization,
    build_registered_objective,
    build_registered_optimizer,
    build_registered_parameter_routing,
    build_registered_precision_policy,
    build_registered_schedule,
    build_registered_weight_decay_schedule,
    constant_learning_rate_multiplier,
    linear_warmup_constant_multiplier,
    linear_warmup_cosine_multiplier,
    optimizer_from_resolved_component,
    parameter_routing_from_resolved_component,
    powercool_multiplier,
    rebase_learning_rate_schedule,
    schedule_configuration_from_resolved_component,
    schedule_from_resolved_component,
    supported_implementation_ids,
    supported_worker_capabilities,
)


def test_runtime_categories_have_one_way_dependency_boundaries():
    root = Path(__file__).resolve().parents[1]
    runtime = root / "src/rwkv_lab/training_runtime"
    category_names = {
        "activations",
        "curricula",
        "gradient_accumulation",
        "gradient_clipping",
        "normalizations",
        "optimizers",
        "objectives",
        "routers",
        "precision",
        "schedules",
        "weight_decay_schedules",
    }
    family_fragments = {
        "mage_flow",
        "rwkv_pretrain",
        "rwkv_finetune",
        "qwen",
        "transformer",
        "vision_train",
    }

    for category in sorted(category_names):
        source = (runtime / f"{category}.py").read_text(encoding="utf-8")
        imports = []
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                prefix = "." * node.level
                imports.append(prefix + (node.module or ""))

        sibling_imports = {
            imported.removeprefix("rwkv_lab.training_runtime.").removeprefix(".")
            for imported in imports
            if imported.startswith((".", "rwkv_lab.training_runtime."))
        }
        assert not sibling_imports.intersection(category_names - {category})
        assert not any(
            fragment in imported
            for imported in imports
            for fragment in family_fragments
        )


def test_training_components_remains_a_logic_free_stable_facade():
    root = Path(__file__).resolve().parents[1]
    source = (root / "src/rwkv_lab/training_components.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    forbidden = (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
    assert not [node for node in tree.body if isinstance(node, forbidden)]


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


def test_constant_learning_rate_has_one_stateless_trajectory():
    configuration = ConstantLearningRateConfiguration()
    assert [
        constant_learning_rate_multiplier(step, configuration) for step in range(8)
    ] == [1.0] * 8
    with pytest.raises(ValueError, match="nonnegative"):
        constant_learning_rate_multiplier(-1, configuration)


def test_linear_warmup_constant_matches_rlvr_optimizer_step_trajectory():
    configuration = LinearWarmupConstantConfiguration(warmup_steps=4)
    assert [
        linear_warmup_constant_multiplier(step, configuration) for step in range(6)
    ] == pytest.approx([0.25, 0.5, 0.75, 1.0, 1.0, 1.0])
    assert linear_warmup_constant_multiplier(
        0, LinearWarmupConstantConfiguration(warmup_steps=0)
    ) == pytest.approx(1.0)


def test_powercool_has_one_shared_optimizer_step_trajectory():
    configuration = PowerCoolConfiguration(
        warmup_steps=2,
        max_steps=10,
        minimum_ratio=0.1,
        cooldown_fraction=0.2,
        power=2.0,
    )
    trajectory = [powercool_multiplier(step, configuration) for step in range(11)]
    assert trajectory[:2] == pytest.approx([0.5, 1.0])
    assert trajectory[7] == pytest.approx(1.0)
    assert trajectory[8] == pytest.approx(1.0)
    assert trajectory[9] < 1.0
    assert trajectory[10] == pytest.approx(0.1)


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


def test_live_learning_rate_rebase_preserves_phase_groups_and_checkpoint_state():
    first = torch.nn.Parameter(torch.tensor([1.0]))
    second = torch.nn.Parameter(torch.tensor([2.0]))
    optimizer = torch.optim.AdamW(
        [
            {"params": [first], "lr": 1.0e-3},
            {"params": [second], "lr": 5.0e-4},
        ]
    )
    schedule = build_registered_schedule(
        ScheduleImplementation.LINEAR_WARMUP_COSINE_V1,
        optimizer,
        LinearWarmupCosineConfiguration(
            warmup_steps=0, max_steps=10, minimum_ratio=0.1
        ),
    )
    optimizer.step()
    schedule.step()
    old_epoch = schedule.last_epoch
    old_rates = schedule.get_last_lr()

    rebase_learning_rate_schedule(
        schedule,
        old_base_learning_rate=1.0e-3,
        new_base_learning_rate=2.0e-4,
    )

    assert schedule.last_epoch == old_epoch
    assert schedule.base_lrs == pytest.approx([2.0e-4, 1.0e-4])
    assert schedule.get_last_lr() == pytest.approx(
        [value * 0.2 for value in old_rates]
    )
    assert [group["lr"] for group in optimizer.param_groups] == pytest.approx(
        schedule.get_last_lr()
    )
    assert [group["initial_lr"] for group in optimizer.param_groups] == pytest.approx(
        schedule.base_lrs
    )
    state = schedule.state_dict()
    assert state["base_lrs"] == pytest.approx([2.0e-4, 1.0e-4])
    assert state["_last_lr"] == pytest.approx(schedule.get_last_lr())


@pytest.mark.parametrize("old,new", [(0.0, 1.0e-4), (1.0e-4, float("nan"))])
def test_live_learning_rate_rebase_rejects_invalid_bases(old, new):
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.AdamW([parameter], lr=1.0e-3)
    schedule = build_registered_schedule(
        ScheduleImplementation.LINEAR_WARMUP_COSINE_V1,
        optimizer,
        LinearWarmupCosineConfiguration(warmup_steps=0, max_steps=2),
    )

    with pytest.raises(ValueError, match="finite and positive"):
        rebase_learning_rate_schedule(
            schedule,
            old_base_learning_rate=old,
            new_base_learning_rate=new,
        )


def test_component_catalog_and_runtime_dispatch_are_exactly_aligned():
    root = Path(__file__).resolve().parents[1]
    document = json.loads(
        (root / "docs/experiment-vm/examples/training-components.v1.json").read_text(
            encoding="utf-8"
        )
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
    assert grades["objective"] == "stateless"
    assert grades["activation"] == "stateless"
    assert grades["normalization"] == "stateless"
    assert grades["precision"] == "stateless"
    assert grades["parameter_router"] == "stateless"
    assert grades["gradient_accumulation"] == "stateless"
    assert grades["curriculum"] == "stateless"
    assert grades["gradient_clipping"] == "stateless"
    assert grades["weight_decay_schedule"] == "stateless"


def test_spectral_muon_factory_requires_explicit_topology_routes():
    from rwkv_lab.spectral_muon import SpectralMuon

    matrix = torch.nn.Parameter(torch.randn(4, 4))
    vector = torch.nn.Parameter(torch.randn(4))
    configuration = SpectralMuonConfiguration(
        learning_rate=0.02,
        ns_steps=2,
        rsav=True,
    )
    optimizer = build_registered_optimizer(
        OptimizerImplementation.SPECTRAL_MUON_NO_DECAY_V1,
        (
            {"params": [matrix], "lr": 0.02, "use_muon": True},
            {"params": [vector], "lr": 0.002, "use_muon": False},
        ),
        configuration,
    )

    assert isinstance(optimizer, SpectralMuon)
    assert optimizer.param_groups[0]["use_muon"] is True
    assert optimizer.param_groups[1]["use_muon"] is False
    with pytest.raises(TypeError, match="explicitly select use_muon"):
        build_registered_optimizer(
            OptimizerImplementation.SPECTRAL_MUON_NO_DECAY_V1,
            ({"params": [matrix], "lr": 0.02},),
            configuration,
        )
def test_registered_layer_norm_factory_owns_construction_not_model_state():
    factory = build_registered_normalization(
        NormalizationImplementation.LAYER_NORM_V1,
        LayerNormConfiguration(epsilon=2.0e-5),
    )
    layer = factory(8)

    assert isinstance(factory, LayerNormFactory)
    assert isinstance(layer, torch.nn.LayerNorm)
    assert layer.eps == pytest.approx(2.0e-5)
    assert layer.elementwise_affine
    assert layer.bias is not None
    assert factory.state_dict() == {}
    assert set(layer.state_dict()) == {"weight", "bias"}


def test_context_curriculum_has_optimizer_step_domain_and_constant_token_budget():
    curriculum = build_registered_curriculum(
        CurriculumImplementation.CONTEXT_LENGTH_V1,
        ContextLengthCurriculumConfiguration(
            maximum_sequence_length=1024,
            base_batch_size=8,
            stages="0:256,0.25:512,0.75:1024",
        ),
    )

    assert isinstance(curriculum, ContextLengthCurriculum)
    assert curriculum.for_step(0, 100) == (256, 32)
    assert curriculum.for_step(25, 100) == (512, 16)
    assert curriculum.for_step(75, 100) == (1024, 8)
    assert curriculum.state_dict() == {}


def test_registered_global_norm_clipping_has_typed_reference_semantics():
    parameter = torch.nn.Parameter(torch.tensor([3.0, 4.0]))
    parameter.grad = torch.tensor([6.0, 8.0])
    observed = build_registered_gradient_clipping(
        GradientClippingImplementation.GLOBAL_NORM_V1,
        [parameter],
        GlobalNormClippingConfiguration(
            max_norm=2.5,
            norm_type=2.0,
            error_if_nonfinite=True,
        ),
    )
    assert observed == pytest.approx(10.0)
    assert parameter.grad == pytest.approx(torch.tensor([1.5, 2.0]))


def test_fixed_gradient_accumulation_owns_microbatch_count_and_loss_scaling():
    policy = build_registered_gradient_accumulation(
        GradientAccumulationImplementation.FIXED_V1,
        FixedGradientAccumulationConfiguration(
            microbatches_per_optimizer_step=4
        ),
    )
    assert isinstance(policy, FixedGradientAccumulation)
    assert tuple(policy.microbatch_indices()) == (0, 1, 2, 3)
    loss = torch.tensor(8.0, requires_grad=True)
    policy.scale_loss(loss).backward()
    assert loss.grad == pytest.approx(torch.tensor(0.25))
    assert policy.state_dict() == {}


def test_linear_head_cross_entropy_objective_matches_reference_value_and_gradient():
    hidden = torch.randn(2, 3, 4, requires_grad=True)
    head = torch.nn.Linear(4, 7, bias=False)
    labels = torch.randint(0, 7, (2, 3))
    objective = build_registered_objective(
        ObjectiveImplementation.LINEAR_HEAD_CROSS_ENTROPY_V1,
        LinearHeadCrossEntropyConfiguration(chunk_size=2, prefer_fused=False),
    )
    assert isinstance(objective, LinearHeadCrossEntropyObjective)
    observed = objective(hidden, head, labels)
    reference = torch.nn.functional.cross_entropy(
        head(hidden).float().reshape(-1, 7), labels.reshape(-1)
    )
    torch.testing.assert_close(observed, reference)
    observed.backward()
    observed_gradient = hidden.grad.detach().clone()
    hidden.grad = None
    reference.backward()
    torch.testing.assert_close(observed_gradient, hidden.grad)


def test_bfloat16_precision_policy_owns_module_and_reduction_dtypes():
    policy = build_registered_precision_policy(
        PrecisionImplementation.BF16_PARAMETERS_FP32_REDUCTIONS_V1,
        BFloat16PrecisionConfiguration(),
    )
    assert isinstance(policy, BFloat16PrecisionPolicy)
    module = policy.convert_module(torch.nn.Linear(3, 4), "cpu")
    assert module.weight.dtype is torch.bfloat16
    reduced = policy.reduce(torch.ones(2, dtype=torch.bfloat16))
    assert reduced.dtype is torch.float32
    assert policy.state_dict() == {}


def test_fp32_parameter_bfloat16_compute_policy_is_truthful_and_stateless():
    policy = build_registered_precision_policy(
        PrecisionImplementation.FP32_PARAMETERS_BF16_COMPUTE_V1,
        FP32ParametersBFloat16ComputeConfiguration(),
    )
    assert isinstance(policy, FP32ParametersBFloat16ComputePolicy)
    module = policy.convert_module(torch.nn.Linear(3, 4), "cpu")
    assert module.weight.dtype is torch.float32
    assert policy.compute_dtype is torch.bfloat16
    assert policy.reduce(torch.ones(2, dtype=torch.bfloat16)).dtype is torch.float32
    assert policy.state_dict() == {}


def test_registered_activations_are_independent_forward_and_installation_policies():
    squared_relu = build_registered_activation(
        ActivationImplementation.SQUARED_RELU_V1
    )
    silu = build_registered_activation(ActivationImplementation.SILU_V1)
    assert isinstance(squared_relu, RegisteredActivation)
    value = torch.tensor([-1.0, 2.0])
    torch.testing.assert_close(squared_relu(value), torch.tensor([0.0, 4.0]))
    torch.testing.assert_close(silu(value), torch.nn.functional.silu(value))

    class InstallationPoint(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.activation = ""

        def set_activation(self, value):
            self.activation = value

    module = InstallationPoint()
    silu.install(module)
    assert module.activation == "silu"


def test_no_decay_optimizer_and_decay_schedule_have_independent_contracts():
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = build_registered_optimizer(
        OptimizerImplementation.TORCH_ADAMW_NO_DECAY_V2,
        [parameter],
        AdamWNoDecayConfiguration(
            learning_rate=1.0e-3,
            beta1=0.9,
            beta2=0.95,
            epsilon=1.0e-8,
            foreach=False,
            fused=False,
        ),
    )
    assert optimizer.param_groups[0]["weight_decay"] == 0.0
    schedule = build_registered_weight_decay_schedule(
        WeightDecayScheduleImplementation.CONSTANT_V1,
        optimizer,
        ConstantWeightDecayConfiguration(weight_decay=0.2),
    )
    assert optimizer.param_groups[0]["weight_decay"] == pytest.approx(0.2)
    optimizer.param_groups[0]["weight_decay_multiplier"] = 0.5
    schedule.step(1)
    assert optimizer.param_groups[0]["weight_decay"] == pytest.approx(0.1)
    assert schedule.state_dict() == {}


def test_resolved_worker_component_dispatch_is_closed_and_typed():
    root = Path(__file__).resolve().parents[1]
    components = json.loads(
        (root / "docs/experiment-vm/examples/training-components.v1.json").read_text(
            encoding="utf-8"
        )
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
            "fused": False,
        },
    }
    optimizer = optimizer_from_resolved_component(optimizer_component, [parameter])
    assert isinstance(optimizer, torch.optim.AdamW)

    sparse_descriptor = next(
        component
        for component in components
        if component["key"]["name"] == "torch_sparse_adam"
    )
    embedding = torch.nn.Embedding(8, 2, sparse=True)
    sparse_optimizer = optimizer_from_resolved_component(
        {
            "descriptor": sparse_descriptor,
            "descriptor_digest": "sha256:" + "b" * 64,
            "configuration": {
                "learning_rate": 5.0e-4,
                "beta1": 0.9,
                "beta2": 0.95,
                "epsilon": 1.0e-8,
            },
        },
        embedding.parameters(),
    )
    assert isinstance(sparse_optimizer, torch.optim.SparseAdam)

    schedule_descriptor = next(
        component
        for component in components
        if component["key"]["name"] == "linear_warmup_cosine"
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

    powercool_descriptor = next(
        component for component in components if component["key"]["name"] == "powercool"
    )
    powercool = schedule_from_resolved_component(
        {
            "descriptor": powercool_descriptor,
            "descriptor_digest": "sha256:" + "d" * 64,
            "configuration": {
                "warmup_steps": 0,
                "max_steps": 10,
                "minimum_ratio": 0.1,
                "cooldown_fraction": 0.2,
                "power": 2.0,
            },
        },
        optimizer,
    )
    assert isinstance(powercool, torch.optim.lr_scheduler.LRScheduler)

    implementation, typed = schedule_configuration_from_resolved_component(
        {
            "descriptor": powercool_descriptor,
            "descriptor_digest": "sha256:" + "d" * 64,
            "configuration": {
                "warmup_steps": 0,
                "max_steps": 10,
                "minimum_ratio": 0.1,
                "cooldown_fraction": 0.2,
                "power": 2.0,
            },
        }
    )
    assert implementation is ScheduleImplementation.POWERCOOL_V1
    assert isinstance(typed, PowerCoolConfiguration)

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

    rwkv_matrix = build_registered_parameter_routing(
        ParameterRouterImplementation.RWKV_MATRIX_OPTIMIZER_V1,
        [("matrix", expert), ("fallback", backbone)],
        {"muon": frozenset({id(expert)})},
        base_learning_rate=1.0e-4,
        configuration=RWKVMatrixOptimizerRoutingConfiguration(
            fallback_multiplier=0.1
        ),
    )
    assert [group["group_name"] for group in rwkv_matrix.groups] == [
        "muon",
        "adam_fallback",
    ]
    assert [group["lr"] for group in rwkv_matrix.groups] == pytest.approx(
        [1.0e-4, 1.0e-5]
    )


def test_resolved_parameter_router_dispatch_uses_exact_catalog_defaults():
    root = Path(__file__).resolve().parents[1]
    components = json.loads(
        (root / "docs/experiment-vm/examples/training-components.v1.json").read_text(
            encoding="utf-8"
        )
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


def test_full_backbone_parameter_router_claims_every_trainable_tensor():
    first = torch.nn.Parameter(torch.ones(2))
    second = torch.nn.Parameter(torch.ones(3))
    frozen = torch.nn.Parameter(torch.ones(5), requires_grad=False)
    result = build_registered_parameter_routing(
        ParameterRouterImplementation.MAGEFLOW_FULL_BACKBONE_V1,
        [("first", first), ("second", second), ("frozen", frozen)],
        {},
        base_learning_rate=3.0e-5,
        configuration=FullBackboneRoutingConfiguration(),
    )
    assert len(result.groups) == 1
    assert result.groups[0]["group_name"] == "full_backbone"
    assert result.groups[0]["params"] == [first, second]
    assert result.report["trainable_parameter_count"] == 5
    assert result.report["frozen_tensor_count"] == 1


@pytest.mark.parametrize(
    "configuration",
    [
        AdamWConfiguration,
        ConstantLearningRateConfiguration,
        LinearWarmupConstantConfiguration,
        LinearWarmupCosineConfiguration,
        PowerCoolConfiguration,
    ],
)
def test_invalid_component_configuration_fails_before_tensor_construction(
    configuration,
):
    with pytest.raises((TypeError, ValueError)):
        if configuration is AdamWConfiguration:
            configuration(learning_rate=float("nan"))
        elif configuration is ConstantLearningRateConfiguration:
            configuration.from_resolved({"unexpected": True})
        elif configuration is LinearWarmupConstantConfiguration:
            configuration(warmup_steps=-1)
        elif configuration is LinearWarmupCosineConfiguration:
            configuration(warmup_steps=-1, max_steps=10)
        else:
            configuration(warmup_steps=11, max_steps=10)


def test_adamw_execution_modes_are_explicit_and_exclusive():
    fused = AdamWConfiguration(
        learning_rate=1.0e-4,
        foreach=False,
        fused=True,
    )
    assert fused.fused and not fused.foreach
    with pytest.raises(ValueError, match="mutually exclusive"):
        AdamWConfiguration(
            learning_rate=1.0e-4,
            foreach=True,
            fused=True,
        )
