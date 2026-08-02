from __future__ import annotations

from .gradient_clipping import GradientClippingImplementation
from .optimizers import OptimizerImplementation
from .routers import ParameterRouterImplementation
from .schedules import ScheduleImplementation


def supported_implementation_ids() -> frozenset[str]:
    return frozenset(
        implementation.value
        for implementation in (
            *OptimizerImplementation,
            *ScheduleImplementation,
            *ParameterRouterImplementation,
            *GradientClippingImplementation,
        )
    )


def supported_worker_capabilities() -> frozenset[str]:
    return frozenset(
        {
            "optimizer.torch_adamw.v1",
            "optimizer.fp32_master_adamw.v1",
            "schedule.linear_warmup_cosine.v1",
            "schedule.powercool.v1",
            "parameter_router.mageflow_appearance_expert.v1",
            "parameter_router.mageflow_terminal_expert.v1",
            "gradient_clipping.global_norm.v1",
        }
    )
