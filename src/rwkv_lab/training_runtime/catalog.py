from __future__ import annotations

from .gradient_accumulation import GradientAccumulationImplementation
from .gradient_clipping import GradientClippingImplementation
from .optimizers import OptimizerImplementation
from .routers import ParameterRouterImplementation
from .schedules import ScheduleImplementation
from .weight_decay_schedules import WeightDecayScheduleImplementation


def supported_implementation_ids() -> frozenset[str]:
    return frozenset(
        implementation.value
        for implementation in (
            *OptimizerImplementation,
            *ScheduleImplementation,
            *ParameterRouterImplementation,
            *GradientClippingImplementation,
            *GradientAccumulationImplementation,
            *WeightDecayScheduleImplementation,
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
            "gradient_accumulation.fixed.v1",
            "weight_decay_schedule.constant.v1",
            "optimizer.torch_adamw_no_decay.v2",
            "optimizer.fp32_master_adamw_no_decay.v2",
        }
    )
