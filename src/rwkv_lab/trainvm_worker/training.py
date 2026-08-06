from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from ._canonical import (
    canonical_dumps,
    deep_freeze,
    is_bounded_text,
    is_digest,
    sha256_digest,
)

RESOLVED_TRAINING_API_VERSION = "trainvm.resolved-training-composition/v1"
MAXIMUM_COMPONENT_SLOTS = 64
_COMPOSITION_REQUIRED_FIELDS = frozenset(
    {
        "api_version",
        "components",
        "composition_digest",
        "model_family",
        "registry_digest",
    }
)
# Blocks the native side attaches to the same envelope when a composition
# declares them (training_component_registry.cpp composition_body): the lowered
# research-topology block, and the lowered post-training arm. They are absent
# entirely when undeclared, never null.
#
# The envelope check stays exact rather than permissive. It must still refuse a
# field neither side knows about — an unrecognised key in a digest-bound
# envelope is exactly the drift this check exists to catch — so a new optional
# block has to be added here deliberately, and the parity test in
# tests/test_trainvm_resolved_composition_parity.py fails until it is.
_COMPOSITION_OPTIONAL_FIELDS = frozenset({"post_training", "topologies"})
_COMPOSITION_FIELDS = _COMPOSITION_REQUIRED_FIELDS | _COMPOSITION_OPTIONAL_FIELDS
_COMPONENT_FIELDS = frozenset({"configuration", "descriptor", "descriptor_digest"})
_DESCRIPTOR_REQUIRED_FIELDS = frozenset(
    {
        "backend",
        "configuration",
        "implementation",
        "key",
        "model_families",
        "reference_implementation",
        "required_capabilities",
        "state",
        "state_grade",
    }
)
_DESCRIPTOR_OPTIONAL_FIELDS = frozenset({"step_domain"})
_KEY_FIELDS = frozenset({"category", "name", "version"})
_CATEGORIES = frozenset(
    {
        "optimizer",
        "model_loader",
        "trainability",
        "parameter_router",
        "learning_rate_schedule",
        "weight_decay_schedule",
        "activation",
        "normalization",
        "objective",
        "precision",
        "gradient_clipping",
        "gradient_accumulation",
        "curriculum",
        "metric_reducer",
        "data_source",
        "sample_processor",
        "sample_mapper",
        "collator",
        "sampler",
        "batching",
        "split_selector",
    }
)


class TrainingCompositionError(ValueError):
    pass


def _state_value(field: Mapping[str, Any], value: Any) -> bool:
    field_type = field.get("type")
    if field_type == "boolean":
        valid = isinstance(value, bool)
    elif field_type == "integer":
        valid = isinstance(value, int) and not isinstance(value, bool)
    elif field_type == "number":
        valid = isinstance(value, (int, float)) and not isinstance(value, bool)
    elif field_type in {"string", "path", "enumeration"}:
        valid = isinstance(value, str) and len(value.encode("utf-8")) <= 4096
    elif field_type == "string_list":
        valid = (
            isinstance(value, (tuple, list))
            and len(value) <= 256
            and all(
                isinstance(item, str) and len(item.encode("utf-8")) <= 4096
                for item in value
            )
        )
    else:
        return False
    if not valid:
        return False
    if field.get("string_format") == "sha256_digest":
        values = value if isinstance(value, (tuple, list)) else (value,)
        return all(is_digest(item) for item in values)
    return True


def _symbolic(value: Any, *, leading_digit: bool = False) -> bool:
    if not is_bounded_text(value, 192):
        return False
    first = value[0]
    if not (
        first.isascii() and (first.isalpha() or (leading_digit and first.isdigit()))
    ):
        return False
    return all(
        character.isascii()
        and (character.isalnum() or character in {"_", "-", ".", ":"})
        for character in value
    )


def _configuration_value(value: Any) -> bool:
    if isinstance(value, (bool, int)):
        return True
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, str):
        return len(value.encode("utf-8")) <= 4096
    return (
        isinstance(value, list)
        and len(value) <= 256
        and all(
            isinstance(item, str) and len(item.encode("utf-8")) <= 4096
            for item in value
        )
    )


@dataclass(frozen=True, slots=True)
class ResolvedTrainingComponent:
    descriptor: Mapping[str, Any]
    configuration: Mapping[str, bool | int | float | str | tuple[str, ...]]
    descriptor_digest: str

    @property
    def category(self) -> str:
        return self.descriptor["key"]["category"]

    @property
    def implementation(self) -> str:
        return self.descriptor["implementation"]

    def runtime_envelope(self) -> Mapping[str, Any]:
        return MappingProxyType(
            {
                "configuration": self.configuration,
                "descriptor": self.descriptor,
                "descriptor_digest": self.descriptor_digest,
            }
        )


@dataclass(frozen=True, slots=True)
class ResolvedTrainingComposition:
    model_family: str
    components: Mapping[str, ResolvedTrainingComponent]
    registry_digest: str
    composition_digest: str
    # None when the composition declares none. Both are inside
    # composition_digest, so a change to either is a different composition.
    topologies: Any | None = None
    post_training: Any | None = None

    def require(
        self, slot: str, *, category: str | None = None
    ) -> ResolvedTrainingComponent:
        try:
            component = self.components[slot]
        except KeyError as error:
            raise TrainingCompositionError(
                f"resolved training composition has no {slot!r} slot"
            ) from error
        if category is not None and component.category != category:
            raise TrainingCompositionError(
                f"resolved training slot {slot!r} is {component.category!r}, not {category!r}"
            )
        return component

    def validate_resume_state(self, state: Any) -> Mapping[str, Any]:
        """Validate exact per-component state before a trainer restores tensors."""

        stateful = {
            slot: component
            for slot, component in self.components.items()
            if component.descriptor["state_grade"] != "stateless"
        }
        if not isinstance(state, dict) or set(state) != set(stateful):
            raise TrainingCompositionError(
                "training component resume state has incomplete slot coverage"
            )
        for slot, component in stateful.items():
            slot_state = state[slot]
            fields = {field["name"]: field for field in component.descriptor["state"]}
            if not isinstance(slot_state, dict) or set(slot_state) != set(fields):
                raise TrainingCompositionError(
                    f"training component resume state for {slot!r} is incomplete"
                )
            for name, field in fields.items():
                if not _state_value(field, slot_state[name]):
                    raise TrainingCompositionError(
                        f"training component resume field {slot}.{name} is invalid"
                    )
        return deep_freeze(state)


def _load_component(value: Any, model_family: str) -> ResolvedTrainingComponent:
    if not isinstance(value, dict) or set(value) != _COMPONENT_FIELDS:
        raise TrainingCompositionError(
            "resolved training component envelope is inexact"
        )
    descriptor = value["descriptor"]
    configuration = value["configuration"]
    descriptor_digest = value["descriptor_digest"]
    if not isinstance(descriptor, dict) or not isinstance(configuration, dict):
        raise TrainingCompositionError(
            "resolved training component descriptor and configuration must be objects"
        )
    descriptor_fields = set(descriptor)
    if not _DESCRIPTOR_REQUIRED_FIELDS.issubset(
        descriptor_fields
    ) or not descriptor_fields.issubset(
        _DESCRIPTOR_REQUIRED_FIELDS | _DESCRIPTOR_OPTIONAL_FIELDS
    ):
        raise TrainingCompositionError(
            "resolved training component descriptor is inexact"
        )
    key = descriptor["key"]
    if not isinstance(key, dict) or set(key) != _KEY_FIELDS:
        raise TrainingCompositionError("resolved training component key is inexact")
    category = key["category"]
    families = descriptor["model_families"]
    if (
        category not in _CATEGORIES
        or not _symbolic(key["name"])
        or not _symbolic(key["version"], leading_digit=True)
        or not _symbolic(descriptor["implementation"])
        or not isinstance(families, list)
        or not families
        or any(not isinstance(item, str) for item in families)
        or (model_family not in families and "*" not in families)
    ):
        raise TrainingCompositionError(
            "resolved training component identity is invalid"
        )
    if not all(
        _symbolic(name) and _configuration_value(item)
        for name, item in configuration.items()
    ):
        raise TrainingCompositionError(
            "resolved training configuration contains an unsupported value"
        )
    if (
        not is_digest(descriptor_digest)
        or sha256_digest(canonical_dumps(descriptor)) != descriptor_digest
    ):
        raise TrainingCompositionError("resolved training descriptor digest disagrees")
    return ResolvedTrainingComponent(
        descriptor=deep_freeze(descriptor),
        configuration=deep_freeze(configuration),
        descriptor_digest=descriptor_digest,
    )


def load_resolved_training_composition(value: Any) -> ResolvedTrainingComposition:
    if (
        not isinstance(value, dict)
        or not _COMPOSITION_REQUIRED_FIELDS.issubset(value)
        or not set(value).issubset(_COMPOSITION_FIELDS)
    ):
        raise TrainingCompositionError(
            "resolved training composition envelope is inexact"
        )
    if value["api_version"] != RESOLVED_TRAINING_API_VERSION:
        raise TrainingCompositionError(
            "resolved training composition version is unsupported"
        )
    model_family = value["model_family"]
    components = value["components"]
    if (
        not _symbolic(model_family)
        or not isinstance(components, dict)
        or not 1 <= len(components) <= MAXIMUM_COMPONENT_SLOTS
        or any(not _symbolic(slot) for slot in components)
        or not is_digest(value["registry_digest"])
        or not is_digest(value["composition_digest"])
    ):
        raise TrainingCompositionError(
            "resolved training composition semantics are invalid"
        )
    body = dict(value)
    del body["composition_digest"]
    if sha256_digest(canonical_dumps(body)) != value["composition_digest"]:
        raise TrainingCompositionError("resolved training composition digest disagrees")
    loaded = {
        slot: _load_component(component, model_family)
        for slot, component in components.items()
    }
    return ResolvedTrainingComposition(
        model_family=model_family,
        components=MappingProxyType(loaded),
        registry_digest=value["registry_digest"],
        composition_digest=value["composition_digest"],
        # Preserved, not dropped. A worker that silently ignored a declared
        # topology or post-training arm would be a quieter version of the bug
        # this fixes.
        topologies=deep_freeze(value["topologies"]) if "topologies" in value else None,
        post_training=(
            deep_freeze(value["post_training"]) if "post_training" in value else None
        ),
    )
