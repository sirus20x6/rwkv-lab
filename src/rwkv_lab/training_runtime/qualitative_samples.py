from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any

from .resolved import resolved_component_parts


class QualitativeSampleImplementation(str, Enum):
    FIXED_HELD_OUT_V1 = "rwkv_lab.qualitative_sample.fixed_held_out.v1"


def _digest(value: object) -> str:
    encoded = json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class FixedHeldOutConfiguration:
    identity_field: str
    identities_digest: str
    selector_digest: str
    sample_count: int

    def __post_init__(self) -> None:
        if (
            not self.identity_field
            or not self.identity_field[0].isascii()
            or not self.identity_field[0].isalpha()
            or any(
                not character.isascii()
                or not (character.isalnum() or character in {"_", "-", ".", ":"})
                for character in self.identity_field
            )
        ):
            raise ValueError("held-out identity field must be symbolic")
        for value, label in (
            (self.identities_digest, "identities_digest"),
            (self.selector_digest, "selector_digest"),
        ):
            if (
                len(value) != 71
                or not value.startswith("sha256:")
                or any(character not in "0123456789abcdef" for character in value[7:])
            ):
                raise ValueError(f"{label} must be a lowercase sha256 digest")
        if (
            not isinstance(self.sample_count, int)
            or isinstance(self.sample_count, bool)
            or not 1 <= self.sample_count <= 1_000_000
        ):
            raise ValueError("sample_count must be a positive bounded integer")

    @classmethod
    def from_resolved(
        cls, configuration: Mapping[str, Any]
    ) -> FixedHeldOutConfiguration:
        if set(configuration) != {
            "identity_field",
            "identities_digest",
            "selector_digest",
            "sample_count",
        }:
            raise ValueError("resolved held-out sample configuration is inexact")
        return cls(**configuration)


@dataclass(frozen=True, slots=True)
class FixedHeldOutSamples:
    configuration: FixedHeldOutConfiguration

    def bind(
        self, identities: Sequence[str], *, selector_digest: str
    ) -> tuple[str, ...]:
        if selector_digest != self.configuration.selector_digest:
            raise ValueError(
                "held-out sample selector differs from the locked split membership"
            )
        bound = tuple(identities)
        if len(bound) != self.configuration.sample_count:
            raise ValueError(
                "held-out sample count differs from its authority declaration"
            )
        if any(not isinstance(identity, str) or not identity for identity in bound):
            raise ValueError("held-out sample identities must be nonempty strings")
        if len(set(bound)) != len(bound):
            raise ValueError("held-out sample identities must be unique")
        if _digest(bound) != self.configuration.identities_digest:
            raise ValueError(
                "held-out sample identities differ from the locked identity digest"
            )
        return bound


def qualitative_sample_from_resolved_component(
    component: Mapping[str, Any],
) -> FixedHeldOutSamples:
    implementation, configuration = resolved_component_parts(
        component, "qualitative_sample"
    )
    if implementation != QualitativeSampleImplementation.FIXED_HELD_OUT_V1:
        raise ValueError(
            "resolved qualitative sample implementation is not allowlisted"
        )
    return FixedHeldOutSamples(FixedHeldOutConfiguration.from_resolved(configuration))


__all__ = [
    "FixedHeldOutConfiguration",
    "FixedHeldOutSamples",
    "QualitativeSampleImplementation",
    "qualitative_sample_from_resolved_component",
]
