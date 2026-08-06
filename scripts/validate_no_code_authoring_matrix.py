#!/usr/bin/env python3
"""Validate the no-code experiment-authoring qualification matrix.

The matrix declares what a release qualification must prove.  It deliberately
does not launch trainers or carry results: execution belongs to a sealed
TrainVM plan and measured evidence belongs to an immutable qualification
receipt.  Keeping this declaration executable as a validator prevents the
eventual acceptance runner from quietly covering one convenient family or
dropping the "no code/build/deploy mutation" requirement.

Usage:
    python scripts/validate_no_code_authoring_matrix.py [path]
"""

from __future__ import annotations

import json
import pathlib
import sys
from collections.abc import Mapping

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from scripts.gate_verdict import verdict_line

DEFAULT = pathlib.Path(
    "docs/experiment-vm/no-code-authoring-matrix.v1.json"
)

REQUIRED_FAMILIES = {
    "hf_multimodal_sft",
    "transformer_lm",
    "rwkv_lm",
    "mageflow",
}

REQUIRED_EVIDENCE = {
    "source_tree_unchanged",
    "adapter_registry_unchanged",
    "handler_dispatch_unchanged",
    "worker_deployment_unchanged",
    "dashboard_source_unchanged",
    "service_configuration_unchanged",
    "validated_before_accelerator_lease",
    "immediate_dashboard_registration",
    "step_zero_examples",
    "bounded_training",
    "checkpoint_publication",
    "pause_resume",
    "plan_identity",
    "plan_diff",
    "authoring_duration",
    "unsupported_boundary",
}

FORBIDDEN_MUTATIONS = {
    "repository_source",
    "adapter_registry",
    "handler_dispatch",
    "sealed_worker_deployment",
    "dashboard_source",
    "service_configuration",
}

REQUIRED_VARIATION_AXES = {
    "hf_multimodal_sft": {
        "dataset",
        "lora_rank",
        "learning_rate",
        "batching",
        "evaluation_schedule",
    },
    "transformer_lm": {
        "dataset",
        "trainability",
        "objective",
    },
    "rwkv_lm": {
        "optimizer",
        "learning_rate_schedule",
        "activation",
    },
    "mageflow": {
        "expert_route",
        "freeze_policy",
        "learning_rate_groups",
    },
}

# This file is a coverage declaration, never a launcher or evidence receipt.
FORBIDDEN_KEYS = {
    "argv",
    "arguments",
    "command",
    "env",
    "environment",
    "executable",
    "executable_path",
    "interpreter",
    "module",
    "working_directory",
    "result",
    "results",
    "run_id",
    "measured_authoring_seconds",
    "measured_steps",
}


def _walk_reserved_keys(value: object, path: str = "") -> list[str]:
    failures: list[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            child_path = f"{path}/{key}"
            if key in FORBIDDEN_KEYS:
                failures.append(f"{child_path}: reserved key {key!r}")
            failures.extend(_walk_reserved_keys(child, child_path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            failures.extend(_walk_reserved_keys(child, f"{path}/{index}"))
    return failures


def _recipe_field(value: object) -> bool:
    if not isinstance(value, str) or not value or len(value) > 192:
        return False
    parts = value.split(".")
    return all(
        part
        and part[0].isascii()
        and part[0].isalpha()
        and all(
            character.isascii()
            and (character.isalnum() or character in {"_", "-"})
            for character in part
        )
        for part in parts
    )


def validate(document: Mapping[str, object]) -> list[str]:
    failures = _walk_reserved_keys(document)
    if document.get("api_version") != "trainvm.no-code-authoring-matrix/v1":
        failures.append(
            "api_version must be trainvm.no-code-authoring-matrix/v1"
        )
    if document.get("authority") != "qualification_declaration_only":
        failures.append(
            "authority must be qualification_declaration_only; the matrix "
            "cannot grant execution or report results"
        )

    forbidden_mutations = set(document.get("forbidden_mutations", []))
    if forbidden_mutations != FORBIDDEN_MUTATIONS:
        missing = sorted(FORBIDDEN_MUTATIONS - forbidden_mutations)
        extra = sorted(forbidden_mutations - FORBIDDEN_MUTATIONS)
        failures.append(
            "forbidden_mutations must match the release gate exactly; "
            f"missing={missing}, extra={extra}"
        )

    required_evidence = set(document.get("required_evidence", []))
    if required_evidence != REQUIRED_EVIDENCE:
        missing = sorted(REQUIRED_EVIDENCE - required_evidence)
        extra = sorted(required_evidence - REQUIRED_EVIDENCE)
        failures.append(
            "required_evidence must match the implemented qualification "
            f"contract; missing={missing}, extra={extra}"
        )

    scenarios = document.get("scenarios", [])
    if not isinstance(scenarios, list) or not scenarios:
        failures.append("scenarios must be a non-empty array")
        scenarios = []

    seen_ids: set[str] = set()
    families: set[str] = set()
    for scenario in scenarios:
        if not isinstance(scenario, Mapping):
            failures.append("scenario entries must be objects")
            continue
        identifier = str(scenario.get("id", "<missing id>"))
        if identifier in seen_ids:
            failures.append(f"{identifier}: duplicate scenario id")
        seen_ids.add(identifier)

        family = scenario.get("family")
        if isinstance(family, str):
            families.add(family)
        if family not in REQUIRED_FAMILIES:
            failures.append(f"{identifier}: unreviewed family {family!r}")
            continue
        if not str(scenario.get("recipe", "")).strip():
            failures.append(f"{identifier}: recipe identity is required")

        axes = scenario.get("variation_axes", [])
        if not isinstance(axes, list) or len(axes) != len(set(axes)):
            failures.append(
                f"{identifier}: variation_axes must be a unique string array"
            )
            axes = []
        expected_axes = REQUIRED_VARIATION_AXES[family]
        if set(axes) != expected_axes:
            failures.append(
                f"{identifier}: variation_axes mismatch; "
                f"missing={sorted(expected_axes - set(axes))}, "
                f"extra={sorted(set(axes) - expected_axes)}"
            )

        variants = scenario.get("variants", [])
        if not isinstance(variants, list) or not variants:
            failures.append(f"{identifier}: variants must be non-empty")
            variants = []
        variant_axes: set[str] = set()
        variant_ids: set[str] = set()
        for variant in variants:
            if not isinstance(variant, Mapping):
                failures.append(f"{identifier}: variant must be an object")
                continue
            variant_id = str(variant.get("id", "<missing variant id>"))
            if variant_id in variant_ids:
                failures.append(
                    f"{identifier}: duplicate variant id {variant_id!r}"
                )
            variant_ids.add(variant_id)
            axis = variant.get("axis")
            if isinstance(axis, str):
                variant_axes.add(axis)
            overrides = variant.get("override_fields", [])
            if not isinstance(overrides, list) or not overrides:
                failures.append(
                    f"{identifier}/{variant_id}: override_fields must be non-empty"
                )
            elif len(set(overrides)) != len(overrides) or any(
                not _recipe_field(field) for field in overrides
            ):
                failures.append(
                    f"{identifier}/{variant_id}: every override field must be "
                    "a unique symbolic recipe field"
                )
        missing_variants = expected_axes - variant_axes
        if missing_variants:
            failures.append(
                f"{identifier}: variants do not exercise axes "
                f"{sorted(missing_variants)}"
            )

        if scenario.get("unsupported_request_disposition") != (
            "new_implementation_required"
        ):
            failures.append(
                f"{identifier}: unsupported requests must fail as "
                "new_implementation_required"
            )
        if not str(scenario.get("step_zero_modality", "")).strip():
            failures.append(f"{identifier}: step_zero_modality is required")

    missing_families = REQUIRED_FAMILIES - families
    if missing_families:
        failures.append(
            f"missing required families {sorted(missing_families)}"
        )
    return failures


def main() -> int:
    path = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT
    document = json.loads(path.read_text(encoding="utf-8"))
    failures = validate(document)
    scenarios = document.get("scenarios", [])
    variants = sum(
        len(scenario.get("variants", []))
        for scenario in scenarios
        if isinstance(scenario, Mapping)
    )
    for failure in failures:
        print(f"FAIL: {failure}")
    print(
        verdict_line(
            "no-code authoring matrix",
            failures,
            f"{len(scenarios)} families and {variants} variants",
        )
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
