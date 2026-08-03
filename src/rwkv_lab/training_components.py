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
from rwkv_lab.training_runtime.curricula import (
    ContextLengthCurriculum,
    ContextLengthCurriculumConfiguration,
    ContextStage,
    CurriculumImplementation,
    build_registered_curriculum,
    context_batch_for_stages,
    curriculum_from_resolved_component,
    parse_context_stages,
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
from rwkv_lab.training_runtime.normalizations import (
    LayerNormConfiguration,
    LayerNormFactory,
    NormalizationImplementation,
    build_registered_normalization,
    normalization_from_resolved_component,
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
    SparseAdamConfiguration,
    build_registered_optimizer,
    optimizer_from_resolved_component,
)
from rwkv_lab.training_runtime.precision import (
    BFloat16PrecisionConfiguration,
    BFloat16PrecisionPolicy,
    FP32ParametersBFloat16ComputeConfiguration,
    FP32ParametersBFloat16ComputePolicy,
    PrecisionImplementation,
    build_registered_precision_policy,
    precision_policy_from_resolved_component,
)
from rwkv_lab.training_runtime.routers import (
    AppearanceExpertRoutingConfiguration,
    FullBackboneRoutingConfiguration,
    ParameterRouterImplementation,
    TerminalExpertRoutingConfiguration,
    build_registered_parameter_routing,
    parameter_routing_from_resolved_component,
)
from rwkv_lab.training_runtime.schedules import (
    ConstantLearningRateConfiguration,
    LinearWarmupConstantConfiguration,
    LinearWarmupCosineConfiguration,
    PowerCoolConfiguration,
    ScheduleImplementation,
    build_registered_schedule,
    constant_learning_rate_multiplier,
    linear_warmup_constant_multiplier,
    linear_warmup_cosine_multiplier,
    powercool_multiplier,
    rebase_learning_rate_schedule,
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
    "ConstantLearningRateConfiguration",
    "ConstantWeightDecayConfiguration",
    "ConstantWeightDecaySchedule",
    "ContextLengthCurriculum",
    "ContextLengthCurriculumConfiguration",
    "ContextStage",
    "CurriculumImplementation",
    "FP32ParametersBFloat16ComputeConfiguration",
    "FP32ParametersBFloat16ComputePolicy",
    "FixedGradientAccumulation",
    "FixedGradientAccumulationConfiguration",
    "FullBackboneRoutingConfiguration",
    "GlobalNormClippingConfiguration",
    "GradientAccumulationImplementation",
    "GradientClippingImplementation",
    "LayerNormConfiguration",
    "LayerNormFactory",
    "LinearHeadCrossEntropyConfiguration",
    "LinearHeadCrossEntropyObjective",
    "LinearWarmupConstantConfiguration",
    "LinearWarmupCosineConfiguration",
    "NormalizationImplementation",
    "ObjectiveImplementation",
    "OptimizerImplementation",
    "ParameterRouterImplementation",
    "PowerCoolConfiguration",
    "PrecisionImplementation",
    "RegisteredActivation",
    "ScheduleImplementation",
    "SparseAdamConfiguration",
    "TerminalExpertRoutingConfiguration",
    "WeightDecayScheduleImplementation",
    "activation_from_resolved_component",
    "build_registered_activation",
    "build_registered_curriculum",
    "build_registered_gradient_accumulation",
    "build_registered_gradient_clipping",
    "build_registered_normalization",
    "build_registered_objective",
    "build_registered_optimizer",
    "build_registered_parameter_routing",
    "build_registered_precision_policy",
    "build_registered_schedule",
    "build_registered_weight_decay_schedule",
    "constant_learning_rate_multiplier",
    "context_batch_for_stages",
    "curriculum_from_resolved_component",
    "gradient_accumulation_from_resolved_component",
    "gradient_clipping_from_resolved_component",
    "linear_warmup_constant_multiplier",
    "linear_warmup_cosine_multiplier",
    "normalization_from_resolved_component",
    "objective_from_resolved_component",
    "optimizer_from_resolved_component",
    "parameter_routing_from_resolved_component",
    "parse_context_stages",
    "powercool_multiplier",
    "precision_policy_from_resolved_component",
    "rebase_learning_rate_schedule",
    "schedule_configuration_from_resolved_component",
    "schedule_from_resolved_component",
    "supported_implementation_ids",
    "supported_worker_capabilities",
    "weight_decay_schedule_from_resolved_component",
]
