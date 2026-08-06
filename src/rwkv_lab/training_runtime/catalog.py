from __future__ import annotations

from .activations import ActivationImplementation
from .curricula import CurriculumImplementation
from .gradient_accumulation import GradientAccumulationImplementation
from .gradient_clipping import GradientClippingImplementation
from .model_loaders import ModelLoaderImplementation
from .normalizations import NormalizationImplementation
from .objectives import ObjectiveImplementation
from .optimizers import OptimizerImplementation
from .precision import PrecisionImplementation
from .routers import ParameterRouterImplementation
from .schedules import ScheduleImplementation
from .trainability import TrainabilityImplementation
from .weight_decay_schedules import WeightDecayScheduleImplementation


def supported_implementation_ids() -> frozenset[str]:
    return frozenset(
        implementation.value
        for implementation in (
            *ActivationImplementation,
            *ModelLoaderImplementation,
            *TrainabilityImplementation,
            *CurriculumImplementation,
            *NormalizationImplementation,
            *OptimizerImplementation,
            *ObjectiveImplementation,
            # Scaled precision codecs (FP8, NVFP4) are runtime-allowlisted for
            # qualification but remain absent from the unchanged native authority
            # registry, so they are deliberately not listed here. The unscaled
            # policies are registered natively and must be.
            PrecisionImplementation.BF16_PARAMETERS_FP32_REDUCTIONS_V1,
            PrecisionImplementation.FP32_PARAMETERS_BF16_COMPUTE_V1,
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
            "activation.silu.v1",
            "model_loader.hf_causal.v1",
            "model_loader.hf_multimodal.v1",
            "trainability.full.v1",
            "trainability.frozen.v1",
            "trainability.named_rules.v1",
            "trainability.lora.v1",
            "activation.squared_relu.v1",
            "curriculum.context_length.v1",
            "normalization.layer_norm.v1",
            "optimizer.torch_adamw.v1",
            "optimizer.torch_sparse_adam.v1",
            "optimizer.fp32_master_adamw.v1",
            "schedule.constant.v1",
            "schedule.linear_warmup_constant.v1",
            "schedule.linear_warmup_cosine.v1",
            "schedule.powercool.v1",
            "parameter_router.mageflow_appearance_expert.v1",
            "parameter_router.mageflow_full_backbone.v1",
            "parameter_router.mageflow_terminal_expert.v1",
            "gradient_clipping.global_norm.v1",
            "gradient_accumulation.fixed.v1",
            "weight_decay_schedule.constant.v1",
            "optimizer.torch_adamw_no_decay.v2",
            "optimizer.fp32_master_adamw_no_decay.v2",
            "objective.linear_head_cross_entropy.v1",
            "precision.bf16_parameters_fp32_reductions.v1",
            "precision.fp32_parameters_bf16_compute.v1",
        }
    )
