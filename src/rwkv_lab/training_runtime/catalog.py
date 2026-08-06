from __future__ import annotations

from .activation_memory import ActivationMemoryImplementation
from .activations import ActivationImplementation
from .artifact_renderers import ArtifactRendererImplementation
from .checkpoint_policies import CheckpointPolicyImplementation
from .curricula import CurriculumImplementation
from .data_pipeline import (
    BatchingImplementation,
    CollatorImplementation,
    DataSourceImplementation,
    SampleMapperImplementation,
    SampleProcessorImplementation,
    SamplerImplementation,
    SplitSelectorImplementation,
)
from .evaluation_schedules import EvaluationScheduleImplementation
from .evaluators import EvaluatorImplementation
from .generation_policies import GenerationPolicyImplementation
from .gradient_accumulation import GradientAccumulationImplementation
from .gradient_clipping import GradientClippingImplementation
from .model_loaders import ModelLoaderImplementation
from .normalizations import NormalizationImplementation
from .objectives import ObjectiveImplementation
from .optimizers import OptimizerImplementation
from .precision import PrecisionImplementation
from .qualitative_samples import QualitativeSampleImplementation
from .routers import ParameterRouterImplementation
from .schedules import ScheduleImplementation
from .trainability import TrainabilityImplementation
from .weight_decay_schedules import WeightDecayScheduleImplementation


def supported_implementation_ids() -> frozenset[str]:
    return frozenset(
        implementation.value
        for implementation in (
            *ActivationMemoryImplementation,
            *ActivationImplementation,
            *ArtifactRendererImplementation,
            *CheckpointPolicyImplementation,
            *DataSourceImplementation,
            *SampleProcessorImplementation,
            *SampleMapperImplementation,
            *CollatorImplementation,
            *SamplerImplementation,
            *BatchingImplementation,
            *SplitSelectorImplementation,
            *EvaluationScheduleImplementation,
            *EvaluatorImplementation,
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
            *QualitativeSampleImplementation,
            *ScheduleImplementation,
            *ParameterRouterImplementation,
            *GradientClippingImplementation,
            *GradientAccumulationImplementation,
            *GenerationPolicyImplementation,
            *WeightDecayScheduleImplementation,
        )
    )


def supported_worker_capabilities() -> frozenset[str]:
    return frozenset(
        {
            "activation.silu.v1",
            "activation_memory.hf_gradient_checkpointing.v1",
            "artifact_renderer.caption_triplet.v1",
            "artifact_renderer.evidence_envelope.v1",
            "checkpoint_policy.atomic_retained.v1",
            "data_source.jsonl_image_caption.v1",
            "data_source.jsonl_frozen_image_splits.v1",
            "data_source.jsonl_frozen_token_splits.v1",
            "data_source.jsonl_token_corpus.v1",
            "sample_processor.image_caption.v1",
            "sample_processor.token_ids.v1",
            "sample_mapper.assistant_conversation.v2",
            "sample_mapper.assistant_only.v1",
            "sample_mapper.causal_tokens.v1",
            "collator.padded.v1",
            "sampler.deterministic.v1",
            "batching.fixed.v1",
            "batching.bucketed.v1",
            "split_selector.deterministic_holdout.v1",
            "split_selector.frozen_named.v1",
            "evaluation_schedule.launch_gate_periodic.v1",
            "evaluation_schedule.launch_gate_periodic.v2",
            "evaluator.scalar_loss.v1",
            "model_loader.hf_causal.v1",
            "model_loader.hf_multimodal.v1",
            "model_loader.rwkv_checkpoint.v1",
            "model_loader.rwkv_scratch.v1",
            "trainability.full.v1",
            "trainability.frozen.v1",
            "trainability.named_rules.v1",
            "trainability.lora.v1",
            "trainability.lora_target_manifest.v2",
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
            "generation_policy.greedy.v1",
            "weight_decay_schedule.constant.v1",
            "optimizer.torch_adamw_no_decay.v2",
            "optimizer.fp32_master_adamw_no_decay.v2",
            "objective.linear_head_cross_entropy.v1",
            "precision.bf16_parameters_fp32_reductions.v1",
            "precision.fp32_parameters_bf16_compute.v1",
            "qualitative_sample.fixed_held_out.v1",
            "qualitative_sample.fixed_held_out.v2",
        }
    )
