"""Exhaustive, auditable optimizer parameter ownership.

Model adapters identify topology-specific ownership sets.  This module turns
those sets into optimizer groups only after proving that every trainable tensor
belongs to exactly one route.  Optimizer implementations never rediscover the
model topology and model adapters never duplicate group validation.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True, slots=True)
class ParameterRoute:
    name: str
    learning_rate_multiplier: float
    parameter_ids: frozenset[int]
    required: bool = False

    def __post_init__(self) -> None:
        if re.fullmatch(r"[a-z][a-z0-9_]*", self.name) is None:
            raise ValueError(
                "parameter route names must be lowercase ASCII identifiers"
            )
        multiplier = self.learning_rate_multiplier
        if (
            isinstance(multiplier, bool)
            or not isinstance(multiplier, (int, float))
            or not math.isfinite(multiplier)
            or multiplier <= 0
        ):
            raise ValueError(
                "parameter route learning-rate multiplier must be positive"
            )
        if not isinstance(self.parameter_ids, frozenset) or not all(
            isinstance(parameter_id, int) and parameter_id > 0
            for parameter_id in self.parameter_ids
        ):
            raise TypeError(
                "parameter route ownership must be a frozenset of identities"
            )
        if not isinstance(self.required, bool):
            raise TypeError("parameter route required flag must be boolean")


@dataclass(frozen=True, slots=True)
class ParameterRoutingResult:
    groups: tuple[dict[str, Any], ...]
    report: dict[str, Any]


def route_trainable_parameters(
    named_parameters: Iterable[tuple[str, torch.nn.Parameter]],
    routes: Sequence[ParameterRoute],
    *,
    base_learning_rate: float,
) -> ParameterRoutingResult:
    """Build deterministic optimizer groups after exclusive-ownership proof."""

    if (
        isinstance(base_learning_rate, bool)
        or not isinstance(base_learning_rate, (int, float))
        or not math.isfinite(base_learning_rate)
        or base_learning_rate <= 0
    ):
        raise ValueError("base learning rate must be positive and finite")
    if not routes:
        raise ValueError("at least one parameter route is required")
    route_names = [route.name for route in routes]
    if len(route_names) != len(set(route_names)):
        raise ValueError("parameter route names must be unique")

    names_by_id: dict[int, list[str]] = {}
    parameters_by_id: dict[int, torch.nn.Parameter] = {}
    ids_by_name: dict[str, int] = {}
    for name, parameter in named_parameters:
        if not isinstance(name, str) or not name:
            raise ValueError("parameter names must be nonempty strings")
        if not isinstance(parameter, torch.nn.Parameter):
            raise TypeError(f"{name!r} is not a torch Parameter")
        parameter_id = id(parameter)
        previous_id = ids_by_name.setdefault(name, parameter_id)
        if previous_id != parameter_id:
            raise ValueError(f"parameter name {name!r} identifies multiple tensors")
        parameters_by_id.setdefault(parameter_id, parameter)
        aliases = names_by_id.setdefault(parameter_id, [])
        if name not in aliases:
            aliases.append(name)

    trainable_ids = {
        parameter_id
        for parameter_id, parameter in parameters_by_id.items()
        if parameter.requires_grad
    }
    if not trainable_ids:
        raise RuntimeError("training scope contains no trainable parameters")

    route_members: dict[str, list[int]] = {route.name: [] for route in routes}
    unclaimed: list[str] = []
    overlapping: list[dict[str, Any]] = []
    for parameter_id in trainable_ids:
        matching = [
            route.name for route in routes if parameter_id in route.parameter_ids
        ]
        canonical_name = names_by_id[parameter_id][0]
        if not matching:
            unclaimed.append(canonical_name)
        elif len(matching) > 1:
            overlapping.append({"parameter": canonical_name, "routes": matching})
        else:
            route_members[matching[0]].append(parameter_id)
    if unclaimed or overlapping:
        raise RuntimeError(
            "parameter ownership is not exhaustive and exclusive: "
            f"unclaimed={sorted(unclaimed)!r}, overlaps={overlapping!r}"
        )

    groups: list[dict[str, Any]] = []
    route_reports: list[dict[str, Any]] = []
    for route in routes:
        member_ids = sorted(
            route_members[route.name], key=lambda item: names_by_id[item][0]
        )
        if route.required and not member_ids:
            raise RuntimeError(
                f"required parameter route {route.name!r} contains no trainable tensors"
            )
        learning_rate = float(base_learning_rate * route.learning_rate_multiplier)
        if member_ids:
            groups.append(
                {
                    "params": [parameters_by_id[item] for item in member_ids],
                    "lr": learning_rate,
                    "initial_lr": learning_rate,
                    "group_name": route.name,
                }
            )
        route_reports.append(
            {
                "name": route.name,
                "learning_rate": learning_rate,
                "learning_rate_multiplier": float(route.learning_rate_multiplier),
                "trainable_tensor_count": len(member_ids),
                "trainable_parameter_count": sum(
                    parameters_by_id[item].numel() for item in member_ids
                ),
                "parameter_names": [names_by_id[item][0] for item in member_ids],
            }
        )

    aliases = [
        {"canonical_name": names[0], "aliases": names[1:]}
        for names in names_by_id.values()
        if len(names) > 1
    ]
    return ParameterRoutingResult(
        groups=tuple(groups),
        report={
            "base_learning_rate": float(base_learning_rate),
            "trainable_tensor_count": len(trainable_ids),
            "trainable_parameter_count": sum(
                parameters_by_id[item].numel() for item in trainable_ids
            ),
            "frozen_tensor_count": len(parameters_by_id) - len(trainable_ids),
            "aliases": sorted(aliases, key=lambda item: item["canonical_name"]),
            "routes": route_reports,
            "passed": True,
        },
    )


__all__ = [
    "ParameterRoute",
    "ParameterRoutingResult",
    "route_trainable_parameters",
]
