from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any

from .resolved import resolved_component_parts


class GenerationPolicyImplementation(str, Enum):
    GREEDY_V1 = "rwkv_lab.generation_policy.greedy.v1"


@dataclass(frozen=True, slots=True)
class GreedyGenerationConfiguration:
    maximum_new_tokens: int
    generation_batch_size: int
    use_cache: bool

    def __post_init__(self) -> None:
        if (
            not isinstance(self.maximum_new_tokens, int)
            or isinstance(self.maximum_new_tokens, bool)
            or not 1 <= self.maximum_new_tokens <= 4096
        ):
            raise ValueError("maximum_new_tokens must be in [1, 4096]")
        if (
            not isinstance(self.generation_batch_size, int)
            or isinstance(self.generation_batch_size, bool)
            or not 1 <= self.generation_batch_size <= 256
        ):
            raise ValueError("generation_batch_size must be in [1, 256]")
        if not isinstance(self.use_cache, bool):
            raise TypeError("use_cache must be boolean")

    @classmethod
    def from_resolved(
        cls, configuration: Mapping[str, Any]
    ) -> GreedyGenerationConfiguration:
        if set(configuration) != {
            "generation_batch_size",
            "maximum_new_tokens",
            "use_cache",
        }:
            raise ValueError("resolved greedy generation configuration is inexact")
        return cls(**configuration)


@dataclass(frozen=True, slots=True)
class GreedyGenerationPolicy:
    implementation: GenerationPolicyImplementation
    configuration: GreedyGenerationConfiguration

    @property
    def digest(self) -> str:
        encoded = json.dumps(
            {
                "do_sample": False,
                "generation_batch_size": self.configuration.generation_batch_size,
                "implementation": self.implementation.value,
                "maximum_new_tokens": self.configuration.maximum_new_tokens,
                "use_cache": self.configuration.use_cache,
            },
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        return "sha256:" + hashlib.sha256(encoded).hexdigest()

    def kwargs(self) -> Mapping[str, bool | int]:
        return {
            "do_sample": False,
            "max_new_tokens": self.configuration.maximum_new_tokens,
            "use_cache": self.configuration.use_cache,
        }


def generation_policy_from_resolved_component(
    component: Mapping[str, Any],
) -> GreedyGenerationPolicy:
    implementation, configuration = resolved_component_parts(
        component, "generation_policy"
    )
    try:
        selected = GenerationPolicyImplementation(implementation)
    except ValueError as error:
        raise ValueError(
            "resolved generation policy implementation is not allowlisted"
        ) from error
    return GreedyGenerationPolicy(
        selected,
        GreedyGenerationConfiguration.from_resolved(configuration),
    )


__all__ = [
    "GenerationPolicyImplementation",
    "GreedyGenerationConfiguration",
    "GreedyGenerationPolicy",
    "generation_policy_from_resolved_component",
]
