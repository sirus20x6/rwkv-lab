"""Stable facade for the category-separated TrainVM tensor runtime.

Authority schemas and composition locks live in C++.  Each tensor category has
one owned module beneath :mod:`rwkv_lab.training_runtime`; this facade preserves
the existing trainer API without re-coupling the implementations.
"""

from rwkv_lab.training_runtime.activations import (
    ActivationImplementation,
    RegisteredActivation,
    activation_from_resolved_component,
    build_registered_activation,
)
from rwkv_lab.training_runtime.catalog import (
    supported_implementation_ids,
    supported_worker_capabilities,
)
from rwkv_lab.training_runtime.gradient_accumulation import (
    FixedGradientAccumulation,
    FixedGradientAccumulationConfiguration,
    GradientAccumulationImplementation,
    build_registered_gradient_accumulation,
    gradient_accumulation_from_resolved_component,
)
from rwkv_lab.training_runtime.gradient_clipping import (
    GlobalNormClippingConfiguration,
    GradientClippingImplementation,
    build_registered_gradient_clipping,
    gradient_clipping_from_resolved_component,
)
from rwkv_lab.training_runtime.objectives import (
    LinearHeadCrossEntropyConfiguration,
    LinearHeadCrossEntropyObjective,
    ObjectiveImplementation,
    build_registered_objective,
    objective_from_resolved_component,
)
from rwkv_lab.training_runtime.optimizers import (
    AdamWConfiguration,
    AdamWNoDecayConfiguration,
    OptimizerImplementation,
    build_registered_optimizer,
    optimizer_from_resolved_component,
)
from rwkv_lab.training_runtime.precision import (
    BFloat16PrecisionConfiguration,
    BFloat16PrecisionPolicy,
    PrecisionImplementation,
    build_registered_precision_policy,
    precision_policy_from_resolved_component,
)
from rwkv_lab.training_runtime.routers import (
    AppearanceExpertRoutingConfiguration,
    ParameterRouterImplementation,
    TerminalExpertRoutingConfiguration,
    build_registered_parameter_routing,
    parameter_routing_from_resolved_component,
)
from rwkv_lab.training_runtime.schedules import (
    LinearWarmupCosineConfiguration,
    PowerCoolConfiguration,
    ScheduleImplementation,
    build_registered_schedule,
    linear_warmup_cosine_multiplier,
    powercool_multiplier,
    schedule_configuration_from_resolved_component,
    schedule_from_resolved_component,
)
from rwkv_lab.training_runtime.weight_decay_schedules import (
    ConstantWeightDecayConfiguration,
    ConstantWeightDecaySchedule,
    WeightDecayScheduleImplementation,
    build_registered_weight_decay_schedule,
    weight_decay_schedule_from_resolved_component,
)

__all__ = [
    "ActivationImplementation",
    "AdamWConfiguration",
    "AdamWNoDecayConfiguration",
    "AppearanceExpertRoutingConfiguration",
    "BFloat16PrecisionConfiguration",
    "BFloat16PrecisionPolicy",
    "ConstantWeightDecayConfiguration",
    "ConstantWeightDecaySchedule",
    "FixedGradientAccumulation",
    "FixedGradientAccumulationConfiguration",
    "GlobalNormClippingConfiguration",
    "GradientAccumulationImplementation",
    "GradientClippingImplementation",
    "LinearHeadCrossEntropyConfiguration",
    "LinearHeadCrossEntropyObjective",
    "LinearWarmupCosineConfiguration",
    "ObjectiveImplementation",
    "OptimizerImplementation",
    "ParameterRouterImplementation",
    "PowerCoolConfiguration",
    "PrecisionImplementation",
    "RegisteredActivation",
    "ScheduleImplementation",
    "TerminalExpertRoutingConfiguration",
    "WeightDecayScheduleImplementation",
    "activation_from_resolved_component",
    "build_registered_activation",
    "build_registered_gradient_accumulation",
    "build_registered_gradient_clipping",
    "build_registered_objective",
    "build_registered_optimizer",
    "build_registered_parameter_routing",
    "build_registered_precision_policy",
    "build_registered_schedule",
    "build_registered_weight_decay_schedule",
    "gradient_accumulation_from_resolved_component",
    "gradient_clipping_from_resolved_component",
    "linear_warmup_cosine_multiplier",
    "objective_from_resolved_component",
    "optimizer_from_resolved_component",
    "parameter_routing_from_resolved_component",
    "powercool_multiplier",
    "precision_policy_from_resolved_component",
    "schedule_configuration_from_resolved_component",
    "schedule_from_resolved_component",
    "supported_implementation_ids",
    "supported_worker_capabilities",
    "weight_decay_schedule_from_resolved_component",
]
