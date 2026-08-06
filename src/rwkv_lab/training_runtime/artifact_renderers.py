from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any

from .resolved import resolved_component_parts


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


class ArtifactRendererImplementation(str, Enum):
    EVIDENCE_ENVELOPE_V1 = "rwkv_lab.artifact_renderer.evidence_envelope.v1"
    CAPTION_TRIPLET_V1 = "rwkv_lab.artifact_renderer.caption_triplet.v1"


@dataclass(frozen=True, slots=True)
class ArtifactRendererConfiguration:
    modality: str
    schema: str

    def __post_init__(self) -> None:
        if self.modality not in {"audio", "image", "multimodal", "text", "video"}:
            raise ValueError("artifact renderer modality is unsupported")
        if self.schema not in {
            "trainvm.eval-evidence.v1",
            "trainvm.caption-triplet.v1",
        }:
            raise ValueError("artifact renderer schema is unsupported")

    @classmethod
    def from_resolved(
        cls, configuration: Mapping[str, Any]
    ) -> ArtifactRendererConfiguration:
        if set(configuration) != {"modality", "schema"}:
            raise ValueError("resolved artifact renderer configuration is inexact")
        return cls(**configuration)


@dataclass(frozen=True, slots=True)
class ArtifactRenderer:
    implementation: ArtifactRendererImplementation
    configuration: ArtifactRendererConfiguration

    def render(
        self,
        *,
        sample_identity: str,
        step: int,
        evidence: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        if (
            not sample_identity
            or not isinstance(step, int)
            or isinstance(step, bool)
            or step < 0
        ):
            raise ValueError(
                "rendered evidence requires an identity and nonnegative step"
            )
        if self.implementation is ArtifactRendererImplementation.CAPTION_TRIPLET_V1:
            required = {"teacher_target", "baseline", "current"}
            if set(evidence) != required or any(
                not isinstance(evidence[field], str) or not evidence[field].strip()
                for field in required
            ):
                raise ValueError(
                    "caption evidence requires aligned nonempty target/baseline/current"
                )
        elif not evidence:
            raise ValueError("rendered evaluation evidence must be nonempty")
        return MappingProxyType(
            {
                "schema": self.configuration.schema,
                "modality": self.configuration.modality,
                "sample_identity": sample_identity,
                "step": step,
                "evidence": _freeze(evidence),
            }
        )


def artifact_renderer_from_resolved_component(
    component: Mapping[str, Any],
) -> ArtifactRenderer:
    implementation, configuration = resolved_component_parts(
        component, "artifact_renderer"
    )
    try:
        selected = ArtifactRendererImplementation(implementation)
    except ValueError as error:
        raise ValueError(
            "resolved artifact renderer implementation is not allowlisted"
        ) from error
    typed = ArtifactRendererConfiguration.from_resolved(configuration)
    expected = (
        "trainvm.caption-triplet.v1"
        if selected is ArtifactRendererImplementation.CAPTION_TRIPLET_V1
        else "trainvm.eval-evidence.v1"
    )
    if typed.schema != expected:
        raise ValueError("artifact renderer implementation and schema disagree")
    return ArtifactRenderer(selected, typed)


__all__ = [
    "ArtifactRenderer",
    "ArtifactRendererConfiguration",
    "ArtifactRendererImplementation",
    "artifact_renderer_from_resolved_component",
]
