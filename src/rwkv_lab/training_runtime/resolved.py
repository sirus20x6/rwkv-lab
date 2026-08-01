from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def resolved_component_parts(
    component: Mapping[str, Any], expected_category: str
) -> tuple[str, Mapping[str, Any]]:
    """Decode the exact authority envelope shared by every runtime category."""

    if set(component) != {"configuration", "descriptor", "descriptor_digest"}:
        raise ValueError("resolved component envelope has missing or unknown fields")
    descriptor = component["descriptor"]
    configuration = component["configuration"]
    if not isinstance(descriptor, Mapping) or not isinstance(configuration, Mapping):
        raise TypeError(
            "resolved component descriptor and configuration must be objects"
        )
    key = descriptor.get("key")
    implementation = descriptor.get("implementation")
    if (
        not isinstance(key, Mapping)
        or key.get("category") != expected_category
        or not isinstance(implementation, str)
    ):
        raise ValueError("resolved component category or implementation is invalid")
    return implementation, configuration
