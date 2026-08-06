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
    FIXED_HELD_OUT_V2 = "rwkv_lab.qualitative_sample.fixed_held_out.v2"


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


@dataclass(frozen=True, slots=True)
class DerivedFixedHeldOutConfiguration:
    """Operator policy whose integrity receipts are measured at binding time."""

    identity_field: str
    sample_count: int

    def __post_init__(self) -> None:
        FixedHeldOutConfiguration(
            identity_field=self.identity_field,
            identities_digest="sha256:" + "0" * 64,
            selector_digest="sha256:" + "0" * 64,
            sample_count=self.sample_count,
        )

    @classmethod
    def from_resolved(
        cls, configuration: Mapping[str, Any]
    ) -> DerivedFixedHeldOutConfiguration:
        if set(configuration) != {"identity_field", "sample_count"}:
            raise ValueError("resolved derived held-out configuration is inexact")
        return cls(**configuration)


@dataclass(frozen=True, slots=True)
class HeldOutSampleBinding:
    identities: tuple[str, ...]
    identities_digest: str
    selector_digest: str


@dataclass(frozen=True, slots=True)
class DerivedFixedHeldOutSamples:
    configuration: DerivedFixedHeldOutConfiguration

    def bind(
        self, identities: Sequence[str], *, selector_digest: str
    ) -> HeldOutSampleBinding:
        bound = tuple(identities)
        if len(bound) != self.configuration.sample_count:
            raise ValueError("held-out sample count differs from declared policy")
        if any(not isinstance(identity, str) or not identity for identity in bound):
            raise ValueError("held-out sample identities must be nonempty strings")
        if len(set(bound)) != len(bound):
            raise ValueError("held-out sample identities must be unique")
        if (
            len(selector_digest) != 71
            or not selector_digest.startswith("sha256:")
            or any(
                character not in "0123456789abcdef"
                for character in selector_digest[7:]
            )
        ):
            raise ValueError("selector_digest must be a lowercase sha256 digest")
        return HeldOutSampleBinding(bound, _digest(bound), selector_digest)


def qualitative_sample_from_resolved_component(
    component: Mapping[str, Any],
) -> FixedHeldOutSamples | DerivedFixedHeldOutSamples:
    implementation, configuration = resolved_component_parts(
        component, "qualitative_sample"
    )
    if implementation == QualitativeSampleImplementation.FIXED_HELD_OUT_V1:
        return FixedHeldOutSamples(
            FixedHeldOutConfiguration.from_resolved(configuration)
        )
    if implementation == QualitativeSampleImplementation.FIXED_HELD_OUT_V2:
        return DerivedFixedHeldOutSamples(
            DerivedFixedHeldOutConfiguration.from_resolved(configuration)
        )
    else:
        raise ValueError(
            "resolved qualitative sample implementation is not allowlisted"
        )


__all__ = [
    "DerivedFixedHeldOutConfiguration",
    "DerivedFixedHeldOutSamples",
    "FixedHeldOutConfiguration",
    "FixedHeldOutSamples",
    "HeldOutSampleBinding",
    "QualitativeSampleImplementation",
    "qualitative_sample_from_resolved_component",
]
