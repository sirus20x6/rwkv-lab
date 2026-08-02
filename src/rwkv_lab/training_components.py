"""Stable facade for the category-separated TrainVM tensor runtime.

Authority schemas and composition locks live in C++.  Each tensor category has
one owned module beneath :mod:`rwkv_lab.training_runtime`; this facade preserves
the existing trainer API without re-coupling the implementations.
"""

from rwkv_lab.training_runtime.catalog import (
    supported_implementation_ids,
    supported_worker_capabilities,
)
from rwkv_lab.training_runtime.gradient_clipping import (
    GlobalNormClippingConfiguration,
    GradientClippingImplementation,
    build_registered_gradient_clipping,
    gradient_clipping_from_resolved_component,
)
from rwkv_lab.training_runtime.optimizers import (
    AdamWConfiguration,
    OptimizerImplementation,
    build_registered_optimizer,
    optimizer_from_resolved_component,
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

__all__ = [
    "AdamWConfiguration",
    "AppearanceExpertRoutingConfiguration",
    "GlobalNormClippingConfiguration",
    "GradientClippingImplementation",
    "LinearWarmupCosineConfiguration",
    "OptimizerImplementation",
    "ParameterRouterImplementation",
    "PowerCoolConfiguration",
    "ScheduleImplementation",
    "TerminalExpertRoutingConfiguration",
    "build_registered_gradient_clipping",
    "build_registered_optimizer",
    "build_registered_parameter_routing",
    "build_registered_schedule",
    "gradient_clipping_from_resolved_component",
    "linear_warmup_cosine_multiplier",
    "optimizer_from_resolved_component",
    "parameter_routing_from_resolved_component",
    "powercool_multiplier",
    "schedule_configuration_from_resolved_component",
    "schedule_from_resolved_component",
    "supported_implementation_ids",
    "supported_worker_capabilities",
]
