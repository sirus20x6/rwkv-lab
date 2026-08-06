from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import torch

from rwkv_lab.training_components import (
    BFloat16PrecisionPolicy,
    ConstantLearningRateConfiguration,
    ConstantWeightDecaySchedule,
    ContextLengthCurriculum,
    FixedGradientAccumulation,
    Float8PrecisionPolicy,
    FP32ParametersBFloat16ComputePolicy,
    LayerNormFactory,
    LinearHeadCrossEntropyObjective,
    LinearWarmupConstantConfiguration,
    LinearWarmupCosineConfiguration,
    NVFP4PrecisionPolicy,
    PowerCoolConfiguration,
    RegisteredActivation,
    ScheduleImplementation,
    activation_from_resolved_component,
    artifact_renderer_from_resolved_component,
    batching_from_resolved_component,
    build_data_pipeline,
    checkpoint_policy_from_resolved_component,
    collator_from_resolved_component,
    curriculum_from_resolved_component,
    data_source_from_resolved_component,
    evaluation_schedule_from_resolved_component,
    evaluator_from_resolved_component,
    gradient_accumulation_from_resolved_component,
    gradient_clipping_from_resolved_component,
    model_loader_from_resolved_component,
    normalization_from_resolved_component,
    objective_from_resolved_component,
    optimizer_from_resolved_component,
    parameter_routing_from_resolved_component,
    precision_policy_from_resolved_component,
    qualitative_sample_from_resolved_component,
    sample_mapper_from_resolved_component,
    sample_processor_from_resolved_component,
    sampler_from_resolved_component,
    schedule_configuration_from_resolved_component,
    schedule_from_resolved_component,
    split_selector_from_resolved_component,
    trainability_from_resolved_component,
    weight_decay_schedule_from_resolved_component,
)
from rwkv_lab.training_parameter_routing import ParameterRoutingResult
from rwkv_lab.trainvm_worker import ResolvedTrainingComposition


class AdapterComponentError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class WorkerTrainingComponents:
    """The only bridge from an authority composition into tensor factories.

    Family adapters own parameter discovery and topology installation. This
    object owns slot/category checks and converts only already-verified worker
    components into the generic runtime envelopes.
    """

    composition: ResolvedTrainingComposition
    expected_model_family: str

    def __post_init__(self) -> None:
        if self.composition.model_family != self.expected_model_family:
            raise AdapterComponentError(
                "worker training composition targets a different model family"
            )

    def optimizer(
        self,
        parameters: Iterable[torch.nn.Parameter] | Iterable[Mapping[str, Any]],
        *,
        slot: str = "optimizer",
    ) -> torch.optim.Optimizer:
        component = self.composition.require(slot, category="optimizer")
        return optimizer_from_resolved_component(
            component.runtime_envelope(), parameters
        )

    def model_loader(self, *, slot: str = "model"):
        component = self.composition.require(slot, category="model_loader")
        return model_loader_from_resolved_component(component.runtime_envelope())

    def trainability(self, *, slot: str = "trainability"):
        component = self.composition.require(slot, category="trainability")
        return trainability_from_resolved_component(component.runtime_envelope())

    def evaluator(self, *, slot: str = "evaluator"):
        component = self.composition.require(slot, category="evaluator")
        evaluator = evaluator_from_resolved_component(component.runtime_envelope())
        split = self.composition.require(
            evaluator.configuration.split_slot, category="split_selector"
        )
        if split.configuration["selection"] != "held_out":
            raise AdapterComponentError(
                "evaluator split slot does not select the held-out partition"
            )
        return evaluator

    def evaluation_schedule(self, *, slot: str = "evaluation_schedule"):
        component = self.composition.require(slot, category="evaluation_schedule")
        return evaluation_schedule_from_resolved_component(component.runtime_envelope())

    def qualitative_samples(self, *, slot: str = "qualitative_samples"):
        component = self.composition.require(slot, category="qualitative_sample")
        return qualitative_sample_from_resolved_component(component.runtime_envelope())

    def artifact_renderer(self, *, slot: str = "artifact_renderer"):
        component = self.composition.require(slot, category="artifact_renderer")
        return artifact_renderer_from_resolved_component(component.runtime_envelope())

    def checkpoint_policy(self, *, slot: str = "checkpoint_policy"):
        component = self.composition.require(slot, category="checkpoint_policy")
        return checkpoint_policy_from_resolved_component(component.runtime_envelope())

    def data_pipeline(
        self,
        *,
        source_slot: str = "data",
        processor_slot: str = "processor",
        mapper_slot: str = "sample_mapping",
        collator_slot: str = "collation",
        sampler_slot: str = "sampler",
        batching_slot: str = "batching",
        split_slot: str = "split",
    ):
        """Build the generic, digest-bound input pipeline for an adapter.

        Batching groups samples with compatible shapes. Gradient accumulation
        remains a separate optimizer-step policy and is intentionally absent
        from this factory.
        """

        source = self.composition.require(source_slot, category="data_source")
        processor = self.composition.require(
            processor_slot, category="sample_processor"
        )
        mapper = self.composition.require(mapper_slot, category="sample_mapper")
        collator = self.composition.require(collator_slot, category="collator")
        sampler = self.composition.require(sampler_slot, category="sampler")
        batching = self.composition.require(batching_slot, category="batching")
        split = self.composition.require(split_slot, category="split_selector")
        return build_data_pipeline(
            source=data_source_from_resolved_component(source.runtime_envelope()),
            processor=sample_processor_from_resolved_component(
                processor.runtime_envelope()
            ),
            mapper=sample_mapper_from_resolved_component(mapper.runtime_envelope()),
            collator=collator_from_resolved_component(collator.runtime_envelope()),
            sampler=sampler_from_resolved_component(sampler.runtime_envelope()),
            batching=batching_from_resolved_component(batching.runtime_envelope()),
            split_selector=split_selector_from_resolved_component(
                split.runtime_envelope()
            ),
        )

    def require_implementation(
        self,
        slot: str,
        *,
        category: str,
        allowed: frozenset[str],
    ) -> str:
        """Fail before tensor construction when an adapter narrows a slot.

        The native adapter contract applies the same key allowlist at compile
        time. This worker-side check is deliberate defense in depth for a
        sealed invocation and protects direct contract tests from reaching an
        optimizer step with incompatible dense/sparse gradient mechanics.
        """

        component = self.composition.require(slot, category=category)
        if not allowed or component.implementation not in allowed:
            raise AdapterComponentError(
                f"resolved training slot {slot!r} selects an implementation "
                "outside the adapter allowlist"
            )
        return component.implementation

    def activation(
        self,
        *,
        slot: str = "activation",
    ) -> RegisteredActivation:
        component = self.composition.require(slot, category="activation")
        return activation_from_resolved_component(component.runtime_envelope())

    def curriculum(
        self,
        *,
        slot: str = "curriculum",
    ) -> ContextLengthCurriculum:
        component = self.composition.require(slot, category="curriculum")
        return curriculum_from_resolved_component(component.runtime_envelope())

    def objective(
        self,
        *,
        slot: str = "objective",
    ) -> LinearHeadCrossEntropyObjective:
        component = self.composition.require(slot, category="objective")
        return objective_from_resolved_component(component.runtime_envelope())

    def normalization(
        self,
        *,
        slot: str = "normalization",
    ) -> LayerNormFactory:
        component = self.composition.require(slot, category="normalization")
        return normalization_from_resolved_component(component.runtime_envelope())

    def precision(
        self,
        *,
        slot: str = "precision",
    ) -> (
        BFloat16PrecisionPolicy
        | FP32ParametersBFloat16ComputePolicy
        | Float8PrecisionPolicy
        | NVFP4PrecisionPolicy
    ):
        component = self.composition.require(slot, category="precision")
        return precision_policy_from_resolved_component(component.runtime_envelope())

    def configuration(
        self, slot: str, *, category: str
    ) -> Mapping[str, bool | int | float | str | tuple[str, ...]]:
        return self.composition.require(slot, category=category).configuration

    def learning_rate_schedule(
        self,
        optimizer: torch.optim.Optimizer,
        *,
        slot: str = "learning_rate",
    ) -> torch.optim.lr_scheduler.LRScheduler:
        component = self.composition.require(slot, category="learning_rate_schedule")
        return schedule_from_resolved_component(component.runtime_envelope(), optimizer)

    def learning_rate_configuration(
        self, *, slot: str = "learning_rate"
    ) -> tuple[
        ScheduleImplementation,
        ConstantLearningRateConfiguration
        | LinearWarmupConstantConfiguration
        | LinearWarmupCosineConfiguration
        | PowerCoolConfiguration,
    ]:
        component = self.composition.require(slot, category="learning_rate_schedule")
        return schedule_configuration_from_resolved_component(
            component.runtime_envelope()
        )

    def gradient_clipping(
        self,
        parameters: Iterable[torch.Tensor],
        *,
        slot: str = "gradient_clipping",
    ) -> torch.Tensor:
        component = self.composition.require(slot, category="gradient_clipping")
        return gradient_clipping_from_resolved_component(
            component.runtime_envelope(), parameters
        )

    def gradient_accumulation(
        self,
        *,
        slot: str = "gradient_accumulation",
    ) -> FixedGradientAccumulation:
        component = self.composition.require(slot, category="gradient_accumulation")
        return gradient_accumulation_from_resolved_component(
            component.runtime_envelope()
        )

    def weight_decay_schedule(
        self,
        optimizer: torch.optim.Optimizer,
        *,
        slot: str = "weight_decay",
    ) -> ConstantWeightDecaySchedule:
        component = self.composition.require(slot, category="weight_decay_schedule")
        return weight_decay_schedule_from_resolved_component(
            component.runtime_envelope(), optimizer
        )

    def parameter_routing(
        self,
        named_parameters: Iterable[tuple[str, torch.nn.Parameter]],
        role_parameter_ids: Mapping[str, frozenset[int]],
        *,
        base_learning_rate: float,
        slot: str = "parameter_router",
    ) -> ParameterRoutingResult:
        component = self.composition.require(slot, category="parameter_router")
        return parameter_routing_from_resolved_component(
            component.runtime_envelope(),
            named_parameters,
            role_parameter_ids,
            base_learning_rate=base_learning_rate,
        )

    def evidence(self) -> Mapping[str, Mapping[str, str]]:
        return {
            slot: {
                "category": component.category,
                "implementation": component.implementation,
                "descriptor_digest": component.descriptor_digest,
            }
            for slot, component in self.composition.components.items()
        }
