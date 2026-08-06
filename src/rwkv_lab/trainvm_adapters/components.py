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
    curriculum_from_resolved_component,
    gradient_accumulation_from_resolved_component,
    gradient_clipping_from_resolved_component,
    model_loader_from_resolved_component,
    normalization_from_resolved_component,
    objective_from_resolved_component,
    optimizer_from_resolved_component,
    parameter_routing_from_resolved_component,
    precision_policy_from_resolved_component,
    schedule_configuration_from_resolved_component,
    schedule_from_resolved_component,
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
