"""Measured no-code authoring evidence must cover the full declared matrix."""

from __future__ import annotations

import copy
import hashlib
import json
import pathlib
import subprocess
import sys

import pytest

REPOSITORY = pathlib.Path(__file__).resolve().parents[1]
VALIDATOR = REPOSITORY / "scripts/validate_no_code_authoring_receipt.py"
MATRIX_PATH = (
    REPOSITORY
    / "docs"
    / "experiment-vm"
    / "no-code-authoring-matrix.v1.json"
)
SHA = "sha256:" + "a" * 64
PLAN_HASH = "a" * 64
IDENTITY_FIELDS = (
    "source_tree_sha256",
    "adapter_registry_sha256",
    "handler_dispatch_sha256",
    "worker_deployment_sha256",
    "dashboard_source_sha256",
    "service_configuration_sha256",
)


def matrix_digest() -> str:
    return "sha256:" + hashlib.sha256(MATRIX_PATH.read_bytes()).hexdigest()


@pytest.fixture
def matrix() -> dict:
    return json.loads(MATRIX_PATH.read_text(encoding="utf-8"))


@pytest.fixture
def receipt(matrix) -> dict:
    identities = {field: SHA for field in IDENTITY_FIELDS}
    return {
        "api_version": "trainvm.no-code-authoring-qualification/v1",
        "matrix_sha256": matrix_digest(),
        "complete": True,
        "baseline_identities": dict(identities),
        "scenarios": [
            {
                "id": scenario["id"],
                "recipe": scenario["recipe"],
                "unsupported_probe": {
                    "disposition": "new_implementation_required",
                    "accelerator_leases_created": 0,
                },
                "variants": [
                    {
                        "id": variant["id"],
                        "authoring_duration_ms": 10,
                        "plan_hash": PLAN_HASH,
                        "plan_identity": {
                            "dry_run_preview_plan_hash": PLAN_HASH,
                            "launch_expected_plan_hash": PLAN_HASH,
                            "preflight_plan_hash": PLAN_HASH,
                            "queued_run_plan_hash": PLAN_HASH,
                        },
                        "recipe_expansion_sha256": SHA,
                        "override_bindings": [
                            {
                                "field": field,
                                "target": "/spec/qualification/" + field.replace(".", "/"),
                                "provenance": "instance",
                            }
                            for field in variant["override_fields"]
                        ],
                        "plan_diff_paths": [
                            "/spec/qualification/" + field.replace(".", "/")
                            for field in variant["override_fields"]
                        ],
                        "preflight": {
                            "passed": True,
                            "before_submission": True,
                            "accelerator_leases_created": 0,
                            "receipt_sha256": SHA,
                        },
                        "run": {
                            "run_id": f"run-{scenario['id']}-{variant['id']}",
                            "dashboard_registered_at_optimizer_step": 0,
                            "step_zero_examples": 1,
                            "step_zero_evidence": {
                                "artifact_id": "artifact-step-zero",
                                "baseline_present": True,
                                "checkpoint_artifact_id": "checkpoint-step-zero",
                                "checkpoint_manifest_sha256": SHA,
                                "current_present": True,
                                "example_count": 1,
                                "modality": scenario["step_zero_modality"],
                                "optimizer_step": 0,
                                "target_present": True,
                            },
                            "optimizer_steps": 1,
                            "checkpoint_artifact_id": "artifact-checkpoint",
                            "pause_resume_proven": True,
                        },
                        "after_identities": dict(identities),
                    }
                    for variant in scenario["variants"]
                ],
            }
            for scenario in matrix["scenarios"]
        ],
    }


def write_receipt(tmp_path: pathlib.Path, receipt: dict) -> pathlib.Path:
    path = tmp_path / "receipt.json"
    path.write_text(json.dumps(receipt), encoding="utf-8")
    return path


def run_validator(path: pathlib.Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(VALIDATOR), str(path), str(MATRIX_PATH)],
        cwd=REPOSITORY,
        capture_output=True,
        text=True,
        check=False,
    )


def find_variant(receipt: dict, scenario_id: str, variant_id: str) -> dict:
    scenario = next(item for item in receipt["scenarios"] if item["id"] == scenario_id)
    return next(item for item in scenario["variants"] if item["id"] == variant_id)


def test_complete_measured_receipt_passes(tmp_path, receipt):
    result = run_validator(write_receipt(tmp_path, receipt))
    assert result.returncode == 0, result.stdout + result.stderr


def test_declaration_alone_cannot_masquerade_as_evidence(tmp_path, matrix):
    result = run_validator(write_receipt(tmp_path, matrix))
    assert result.returncode == 1
    assert "qualification/v1" in result.stdout
    assert "complete=true" in result.stdout
    assert "baseline_identities/source_tree_sha256" in result.stdout


def test_every_declared_variant_requires_evidence(tmp_path, receipt):
    receipt["scenarios"][0]["variants"].pop()
    result = run_validator(write_receipt(tmp_path, receipt))
    assert result.returncode == 1
    assert "missing variant evidence" in result.stdout


def test_preflight_must_happen_before_any_accelerator_lease(tmp_path, receipt):
    variant = receipt["scenarios"][0]["variants"][0]
    variant["preflight"]["accelerator_leases_created"] = 1
    result = run_validator(write_receipt(tmp_path, receipt))
    assert result.returncode == 1
    assert "preflight acquired an accelerator lease" in result.stdout


def test_step_zero_examples_are_not_optional(tmp_path, receipt):
    variant = receipt["scenarios"][0]["variants"][0]
    variant["run"]["step_zero_examples"] = 0
    result = run_validator(write_receipt(tmp_path, receipt))
    assert result.returncode == 1
    assert "step-zero examples are missing" in result.stdout


@pytest.mark.parametrize(
    "field",
    [
        "dry_run_preview_plan_hash",
        "launch_expected_plan_hash",
        "preflight_plan_hash",
        "queued_run_plan_hash",
    ],
)
def test_every_authoring_phase_must_share_one_plan_hash(
    tmp_path, receipt, field
):
    variant = receipt["scenarios"][0]["variants"][0]
    variant["plan_identity"][field] = "b" * 64
    result = run_validator(write_receipt(tmp_path, receipt))
    assert result.returncode == 1
    assert "do not share one plan identity" in result.stdout


@pytest.mark.parametrize("role", ["target", "baseline", "current"])
def test_step_zero_artifact_requires_all_user_visible_views(
    tmp_path, receipt, role
):
    variant = receipt["scenarios"][0]["variants"][0]
    variant["run"]["step_zero_evidence"][f"{role}_present"] = False
    result = run_validator(write_receipt(tmp_path, receipt))
    assert result.returncode == 1
    assert f"omit the {role} view" in result.stdout


def test_step_zero_artifact_must_be_checkpoint_bound(tmp_path, receipt):
    variant = receipt["scenarios"][0]["variants"][0]
    variant["run"]["step_zero_evidence"]["checkpoint_manifest_sha256"] = "bad"
    result = run_validator(write_receipt(tmp_path, receipt))
    assert result.returncode == 1
    assert "step-zero examples are not checkpoint-bound" in result.stdout


@pytest.mark.parametrize("field", IDENTITY_FIELDS)
def test_forbidden_source_build_or_deploy_mutation_fails(
    tmp_path, receipt, field
):
    modified = copy.deepcopy(receipt)
    variant = find_variant(
        modified,
        "qwen.multimodal-caption-lora",
        "alternate-learning-rate",
    )
    variant["after_identities"][field] = "sha256:" + "b" * 64
    result = run_validator(write_receipt(tmp_path, modified))
    assert result.returncode == 1
    assert f"forbidden mutation changed {field}" in result.stdout


def test_plan_diff_must_include_the_authority_bound_override(tmp_path, receipt):
    variant = find_variant(
        receipt,
        "qwen.multimodal-caption-lora",
        "alternate-lora-rank",
    )
    variant["plan_diff_paths"] = ["/unrelated"]
    result = run_validator(write_receipt(tmp_path, receipt))
    assert result.returncode == 1
    assert "plan diff omits authority-bound override targets" in result.stdout


def test_override_binding_must_be_instance_provenanced(tmp_path, receipt):
    variant = find_variant(
        receipt,
        "qwen.multimodal-caption-lora",
        "alternate-learning-rate",
    )
    variant["override_bindings"][0]["provenance"] = "template"
    result = run_validator(write_receipt(tmp_path, receipt))
    assert result.returncode == 1
    assert "override binding is not instance-provenanced" in result.stdout


def test_override_binding_must_cover_the_declared_recipe_fields(tmp_path, receipt):
    variant = find_variant(
        receipt,
        "qwen.multimodal-caption-lora",
        "alternate-lora-rank",
    )
    variant["override_bindings"].pop()
    result = run_validator(write_receipt(tmp_path, receipt))
    assert result.returncode == 1
    assert "override bindings do not exactly cover declared fields" in result.stdout


def test_unsupported_request_fails_before_a_lease(tmp_path, receipt):
    receipt["scenarios"][0]["unsupported_probe"][
        "accelerator_leases_created"
    ] = 1
    result = run_validator(write_receipt(tmp_path, receipt))
    assert result.returncode == 1
    assert "unsupported probe acquired an accelerator lease" in result.stdout


def test_the_validator_states_a_verdict_in_both_directions(tmp_path, receipt, matrix):
    """Its last line must say which way it went, and differ between outcomes.

    `tests/test_gate_verdict.py` runs every gate that uses `verdict_line` and
    checks this end to end, but only for gates that take no arguments. This one
    needs a receipt and a matrix, so it is named in that file's
    NEEDS_ARGUMENTS map as the place the property is asserted -- and this is
    that assertion. Without it the pointer was decoration: the map said "the
    verdict is checked over there" while nothing over there checked it, which is
    exactly the shape both files exist to prevent.
    """
    good, bad = tmp_path / "ok", tmp_path / "bad"
    good.mkdir()
    bad.mkdir()

    passing = run_validator(write_receipt(good, receipt))
    assert passing.returncode == 0, passing.stdout + passing.stderr
    passing_last = passing.stdout.strip().splitlines()[-1]

    failing = run_validator(write_receipt(bad, matrix))
    assert failing.returncode == 1
    failing_last = failing.stdout.strip().splitlines()[-1]

    assert "PASSED" in passing_last and "FAILED" not in passing_last
    assert "FAILED" in failing_last
    assert passing_last != failing_last, (
        "a reader who sees only the final line must be able to tell the two "
        "runs apart")
