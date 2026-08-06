from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any

from .resolved import resolved_component_parts


class CheckpointPolicyImplementation(str, Enum):
    ATOMIC_RETAINED_V1 = "rwkv_lab.checkpoint_policy.atomic_retained.v1"


@dataclass(frozen=True, slots=True)
class AtomicCheckpointPolicyConfiguration:
    every_steps: int
    keep_last: int
    keep_best: int = 0
    publish_final: bool = True
    resume_grade: str = "compatible"

    def __post_init__(self) -> None:
        for value, label, minimum, maximum in (
            (self.every_steps, "every_steps", 1, 1_000_000_000),
            (self.keep_last, "keep_last", 1, 1_000_000),
            (self.keep_best, "keep_best", 0, 1_000_000),
        ):
            if (
                not isinstance(value, int)
                or isinstance(value, bool)
                or not minimum <= value <= maximum
            ):
                raise ValueError(f"{label} must be a bounded integer")
        if self.resume_grade not in {"exact", "compatible", "restart_only"}:
            raise ValueError("checkpoint resume grade is unsupported")

    @classmethod
    def from_resolved(
        cls, configuration: Mapping[str, Any]
    ) -> AtomicCheckpointPolicyConfiguration:
        if set(configuration) != {
            "every_steps",
            "keep_last",
            "keep_best",
            "publish_final",
            "resume_grade",
        }:
            raise ValueError("resolved checkpoint policy configuration is inexact")
        return cls(**configuration)


def _digest(value: str, label: str) -> str:
    if (
        len(value) != 71
        or not value.startswith("sha256:")
        or any(character not in "0123456789abcdef" for character in value[7:])
    ):
        raise ValueError(f"{label} must be a lowercase sha256 digest")
    return value


@dataclass(frozen=True, slots=True)
class AtomicCheckpointPolicy:
    configuration: AtomicCheckpointPolicyConfiguration

    def due(self, step: int, *, final: bool = False) -> bool:
        if not isinstance(step, int) or isinstance(step, bool) or step < 0:
            raise ValueError("checkpoint step must be a nonnegative integer")
        return (final and self.configuration.publish_final) or (
            step > 0 and step % self.configuration.every_steps == 0
        )

    def retained_steps(
        self, published_steps: Sequence[int], best_steps: Sequence[int] = ()
    ) -> tuple[int, ...]:
        published = tuple(sorted(set(published_steps)))
        best = tuple(best_steps)
        if (
            len(set(best)) != len(best)
            or any(step < 0 for step in (*published, *best))
            or not set(best) <= set(published)
        ):
            raise ValueError("checkpoint retention input is malformed")
        keep = set(published[-self.configuration.keep_last :])
        if self.configuration.keep_best:
            keep.update(best[: self.configuration.keep_best])
        return tuple(sorted(keep))

    def component_state(
        self,
        *,
        last_published_step: int,
        publication_manifest: str,
        retention_manifest: str,
        atomic_publication_complete: bool,
    ) -> Mapping[str, object]:
        if not atomic_publication_complete:
            raise ValueError(
                "checkpoint state cannot precede atomic publication completion"
            )
        if last_published_step < 0:
            raise ValueError("last published checkpoint step must be nonnegative")
        return MappingProxyType(
            {
                "last_published_step": last_published_step,
                "publication_manifest": _digest(
                    publication_manifest, "publication_manifest"
                ),
                "retention_manifest": _digest(retention_manifest, "retention_manifest"),
            }
        )


def checkpoint_policy_from_resolved_component(
    component: Mapping[str, Any],
) -> AtomicCheckpointPolicy:
    implementation, configuration = resolved_component_parts(
        component, "checkpoint_policy"
    )
    if implementation != CheckpointPolicyImplementation.ATOMIC_RETAINED_V1:
        raise ValueError("resolved checkpoint policy implementation is not allowlisted")
    return AtomicCheckpointPolicy(
        AtomicCheckpointPolicyConfiguration.from_resolved(configuration)
    )


__all__ = [
    "AtomicCheckpointPolicy",
    "AtomicCheckpointPolicyConfiguration",
    "CheckpointPolicyImplementation",
    "checkpoint_policy_from_resolved_component",
]
