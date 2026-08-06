#!/usr/bin/env python3
"""Validate measured no-code experiment-authoring qualification evidence.

The companion matrix declares coverage.  This validator consumes the immutable
receipt emitted after those scenarios actually run and refuses a green result
unless every declared variant proves the complete no-code lifecycle.

Usage:
    python scripts/validate_no_code_authoring_receipt.py RECEIPT [MATRIX]
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import sys
from collections.abc import Mapping

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from scripts.gate_verdict import verdict_line

DEFAULT_MATRIX = pathlib.Path(
    "docs/experiment-vm/no-code-authoring-matrix.v1.json"
)

IDENTITY_FIELDS = (
    "source_tree_sha256",
    "adapter_registry_sha256",
    "handler_dispatch_sha256",
    "worker_deployment_sha256",
    "dashboard_source_sha256",
    "service_configuration_sha256",
)


def _sha256(path: pathlib.Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _is_sha256(value: object) -> bool:
    if not isinstance(value, str) or not value.startswith("sha256:"):
        return False
    digest = value.removeprefix("sha256:")
    return len(digest) == 64 and all(character in "0123456789abcdef" for character in digest)


def _is_plan_hash(value: object) -> bool:
    """TrainVM CompiledPlan stores the canonical plan hash without a prefix."""
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _is_json_pointer(value: object) -> bool:
    if not isinstance(value, str) or not value.startswith("/") or len(value) > 4096:
        return False
    # RFC 6901 permits only ~0 and ~1 escape sequences. Empty path segments are
    # legal JSON Pointer, but recipe targets deliberately reject them so a
    # human-visible diff cannot contain ambiguous repeated separators.
    return all(
        part
        and all(
            character != "~" or index + 1 < len(part) and part[index + 1] in "01"
            for index, character in enumerate(part)
        )
        for part in value[1:].split("/")
    )


def _mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


def validate(
    receipt: Mapping[str, object],
    matrix: Mapping[str, object],
    matrix_digest: str,
) -> list[str]:
    failures: list[str] = []
    if receipt.get("api_version") != "trainvm.no-code-authoring-qualification/v1":
        failures.append(
            "api_version must be trainvm.no-code-authoring-qualification/v1"
        )
    if receipt.get("matrix_sha256") != matrix_digest:
        failures.append("matrix_sha256 does not bind the supplied matrix")
    if receipt.get("complete") is not True:
        failures.append("receipt must explicitly declare complete=true")

    baseline = _mapping(receipt.get("baseline_identities"))
    for field in IDENTITY_FIELDS:
        if not _is_sha256(baseline.get(field)):
            failures.append(f"baseline_identities/{field} must be a sha256 digest")

    declared_scenarios = {
        str(item["id"]): item
        for item in matrix.get("scenarios", [])
        if isinstance(item, Mapping) and "id" in item
    }
    observed_list = receipt.get("scenarios", [])
    if not isinstance(observed_list, list):
        failures.append("scenarios must be an array")
        observed_list = []
    observed_scenarios: dict[str, Mapping[str, object]] = {}
    for item in observed_list:
        if not isinstance(item, Mapping):
            failures.append("scenario evidence entries must be objects")
            continue
        identifier = str(item.get("id", "<missing id>"))
        if identifier in observed_scenarios:
            failures.append(f"{identifier}: duplicate scenario evidence")
        observed_scenarios[identifier] = item

    missing_scenarios = set(declared_scenarios) - set(observed_scenarios)
    extra_scenarios = set(observed_scenarios) - set(declared_scenarios)
    if missing_scenarios:
        failures.append(f"missing scenario evidence {sorted(missing_scenarios)}")
    if extra_scenarios:
        failures.append(f"undeclared scenario evidence {sorted(extra_scenarios)}")

    for scenario_id, declaration in declared_scenarios.items():
        evidence = observed_scenarios.get(scenario_id)
        if evidence is None:
            continue
        if evidence.get("recipe") != declaration.get("recipe"):
            failures.append(f"{scenario_id}: recipe identity mismatch")

        unsupported = _mapping(evidence.get("unsupported_probe"))
        if unsupported.get("disposition") != "new_implementation_required":
            failures.append(
                f"{scenario_id}: unsupported probe must report "
                "new_implementation_required"
            )
        if unsupported.get("accelerator_leases_created") != 0:
            failures.append(
                f"{scenario_id}: unsupported probe acquired an accelerator lease"
            )

        declared_variants = {
            str(item["id"]): item
            for item in declaration.get("variants", [])
            if isinstance(item, Mapping) and "id" in item
        }
        observed_variants_raw = evidence.get("variants", [])
        if not isinstance(observed_variants_raw, list):
            failures.append(f"{scenario_id}: variants must be an array")
            observed_variants_raw = []
        observed_variants: dict[str, Mapping[str, object]] = {}
        for item in observed_variants_raw:
            if not isinstance(item, Mapping):
                failures.append(f"{scenario_id}: variant evidence must be an object")
                continue
            identifier = str(item.get("id", "<missing id>"))
            if identifier in observed_variants:
                failures.append(f"{scenario_id}/{identifier}: duplicate evidence")
            observed_variants[identifier] = item
        missing_variants = set(declared_variants) - set(observed_variants)
        extra_variants = set(observed_variants) - set(declared_variants)
        if missing_variants:
            failures.append(
                f"{scenario_id}: missing variant evidence {sorted(missing_variants)}"
            )
        if extra_variants:
            failures.append(
                f"{scenario_id}: undeclared variant evidence {sorted(extra_variants)}"
            )

        for variant_id, variant_declaration in declared_variants.items():
            variant = observed_variants.get(variant_id)
            if variant is None:
                continue
            prefix = f"{scenario_id}/{variant_id}"
            if not isinstance(variant.get("authoring_duration_ms"), int) or int(
                variant.get("authoring_duration_ms", -1)
            ) < 0:
                failures.append(f"{prefix}: authoring_duration_ms must be nonnegative")
            if not _is_plan_hash(variant.get("plan_hash")):
                failures.append(
                    f"{prefix}: plan_hash must be the canonical 64-hex "
                    "CompiledPlan identity"
                )
            plan_identity = _mapping(variant.get("plan_identity"))
            required_plan_identity_fields = {
                "dry_run_preview_plan_hash",
                "launch_expected_plan_hash",
                "preflight_plan_hash",
                "queued_run_plan_hash",
            }
            if set(plan_identity) != required_plan_identity_fields:
                failures.append(f"{prefix}: plan_identity envelope is not exact")
            else:
                plan_hashes = {
                    plan_identity.get(field)
                    for field in required_plan_identity_fields
                }
                if any(not _is_plan_hash(value) for value in plan_hashes):
                    failures.append(
                        f"{prefix}: plan_identity contains a noncanonical plan hash"
                    )
                elif plan_hashes != {variant.get("plan_hash")}:
                    failures.append(
                        f"{prefix}: preview, launch fence, preflight, and queued run "
                        "do not share one plan identity"
                    )
            if not _is_sha256(variant.get("recipe_expansion_sha256")):
                failures.append(
                    f"{prefix}: recipe_expansion_sha256 must bind the authority expansion"
                )
            changed_paths = variant.get("plan_diff_paths", [])
            if not isinstance(changed_paths, list) or not changed_paths:
                failures.append(f"{prefix}: plan_diff_paths must be non-empty")
                changed_paths = []
            elif any(not _is_json_pointer(path) for path in changed_paths):
                failures.append(f"{prefix}: plan_diff_paths contains an invalid JSON pointer")

            required_fields = set(variant_declaration.get("override_fields", []))
            raw_bindings = variant.get("override_bindings", [])
            if not isinstance(raw_bindings, list):
                failures.append(f"{prefix}: override_bindings must be an array")
                raw_bindings = []
            bindings: dict[str, str] = {}
            for raw_binding in raw_bindings:
                binding = _mapping(raw_binding)
                field = binding.get("field")
                target = binding.get("target")
                if set(binding) != {"field", "target", "provenance"}:
                    failures.append(f"{prefix}: override binding envelope is not exact")
                    continue
                if not isinstance(field, str) or not field or field in bindings:
                    failures.append(f"{prefix}: override binding field is missing or duplicate")
                    continue
                if not _is_json_pointer(target):
                    failures.append(f"{prefix}: override binding target is not a JSON pointer")
                    continue
                if binding.get("provenance") != "instance":
                    failures.append(f"{prefix}: override binding is not instance-provenanced")
                bindings[field] = str(target)
            if set(bindings) != required_fields:
                failures.append(
                    f"{prefix}: override bindings do not exactly cover declared fields "
                    f"{sorted(required_fields)}"
                )
            required_paths = set(bindings.values())
            if not required_paths.issubset(set(changed_paths)):
                failures.append(
                    f"{prefix}: plan diff omits authority-bound override targets "
                    f"{sorted(required_paths - set(changed_paths))}"
                )

            preflight = _mapping(variant.get("preflight"))
            if preflight.get("passed") is not True:
                failures.append(f"{prefix}: preflight did not pass")
            if preflight.get("before_submission") is not True:
                failures.append(f"{prefix}: preflight did not precede submission")
            if preflight.get("accelerator_leases_created") != 0:
                failures.append(f"{prefix}: preflight acquired an accelerator lease")
            if not _is_sha256(preflight.get("receipt_sha256")):
                failures.append(f"{prefix}: preflight receipt identity is missing")

            run = _mapping(variant.get("run"))
            if not str(run.get("run_id", "")).strip():
                failures.append(f"{prefix}: run_id is missing")
            if run.get("dashboard_registered_at_optimizer_step") != 0:
                failures.append(
                    f"{prefix}: dashboard was not registered at optimizer step 0"
                )
            if not isinstance(run.get("step_zero_examples"), int) or int(
                run.get("step_zero_examples", 0)
            ) <= 0:
                failures.append(f"{prefix}: step-zero examples are missing")
            step_zero = _mapping(run.get("step_zero_evidence"))
            if set(step_zero) != {
                "artifact_id",
                "baseline_present",
                "checkpoint_artifact_id",
                "checkpoint_manifest_sha256",
                "current_present",
                "example_count",
                "modality",
                "optimizer_step",
                "target_present",
            }:
                failures.append(f"{prefix}: step-zero evidence envelope is not exact")
            else:
                if step_zero.get("optimizer_step") != 0:
                    failures.append(f"{prefix}: example artifact is not from step zero")
                if step_zero.get("modality") != declaration.get("step_zero_modality"):
                    failures.append(f"{prefix}: step-zero example modality mismatch")
                if not str(step_zero.get("artifact_id", "")).strip():
                    failures.append(f"{prefix}: step-zero example artifact is missing")
                if not str(step_zero.get("checkpoint_artifact_id", "")).strip() or not _is_sha256(
                    step_zero.get("checkpoint_manifest_sha256")
                ):
                    failures.append(
                        f"{prefix}: step-zero examples are not checkpoint-bound"
                    )
                if not isinstance(step_zero.get("example_count"), int) or int(
                    step_zero.get("example_count", 0)
                ) <= 0 or step_zero.get("example_count") != run.get("step_zero_examples"):
                    failures.append(
                        f"{prefix}: step-zero artifact count disagrees with the run"
                    )
                for role in ("target", "baseline", "current"):
                    if step_zero.get(f"{role}_present") is not True:
                        failures.append(
                            f"{prefix}: step-zero examples omit the {role} view"
                        )
            if not isinstance(run.get("optimizer_steps"), int) or int(
                run.get("optimizer_steps", 0)
            ) <= 0:
                failures.append(f"{prefix}: bounded training did not run")
            if not str(run.get("checkpoint_artifact_id", "")).strip():
                failures.append(f"{prefix}: checkpoint artifact is missing")
            if run.get("pause_resume_proven") is not True:
                failures.append(f"{prefix}: pause/resume was not proven")

            after = _mapping(variant.get("after_identities"))
            for field in IDENTITY_FIELDS:
                if after.get(field) != baseline.get(field):
                    failures.append(
                        f"{prefix}: forbidden mutation changed {field}"
                    )
    return failures


def main() -> int:
    if len(sys.argv) not in {2, 3}:
        print("usage: validate_no_code_authoring_receipt.py RECEIPT [MATRIX]", file=sys.stderr)
        return 2
    receipt_path = pathlib.Path(sys.argv[1])
    matrix_path = pathlib.Path(sys.argv[2]) if len(sys.argv) == 3 else DEFAULT_MATRIX
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    failures = validate(receipt, matrix, _sha256(matrix_path))
    for failure in failures:
        print(f"FAIL: {failure}")
    print(
        verdict_line(
            "no-code authoring qualification",
            failures,
            f"{len(matrix.get('scenarios', []))} declared scenarios",
        )
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
