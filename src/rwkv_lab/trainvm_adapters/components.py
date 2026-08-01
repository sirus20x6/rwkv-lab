from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import torch

from rwkv_lab.training_components import (
    optimizer_from_resolved_component,
    parameter_routing_from_resolved_component,
    schedule_from_resolved_component,
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

    def configuration(
        self, slot: str, *, category: str
    ) -> Mapping[str, bool | int | float | str]:
        return self.composition.require(slot, category=category).configuration

    def learning_rate_schedule(
        self,
        optimizer: torch.optim.Optimizer,
        *,
        slot: str = "learning_rate",
    ) -> torch.optim.lr_scheduler.LRScheduler:
        component = self.composition.require(slot, category="learning_rate_schedule")
        return schedule_from_resolved_component(component.runtime_envelope(), optimizer)

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
