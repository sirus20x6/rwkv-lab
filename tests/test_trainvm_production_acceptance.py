from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "verify_trainvm_production_acceptance.py"
)
SPEC = importlib.util.spec_from_file_location("trainvm_production_acceptance", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()


def _digest(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _publish(root: Path, name: str, value: Any) -> dict[str, str]:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    path.write_bytes(encoded)
    return {"path": name, "sha256": _digest(encoded)}


def _hostd_receipt() -> dict[str, Any]:
    cases = [
        {
            "crash_point": point,
            "executor": executor,
            "status": "qualified",
            "unqualified_reason": "none",
            "crash_delivered": True,
            "crashed_pid": index + 100,
            "detail": "qualified",
            "invariants": ["no_double_launch"],
            "evidence": {"case": str(index)},
        }
        for index, (point, executor) in enumerate(MODULE.HOSTD_CRASH_POINTS)
    ]
    receipt = {
        "api_version": "trainvm.hostd-crash-qualification/v1",
        "host": {"root_authority": True},
        "cases": cases,
        "findings": [],
        "declared_points": 16,
        "qualified_points": 16,
        "unqualified_points": 0,
        "gate_open": True,
        "blocking_points": [],
        "receipt_digest": "",
    }
    receipt["receipt_digest"] = _digest(_canonical(receipt))
    return receipt


def _run_documents(family: str) -> dict[str, Any]:
    run_id = f"{family}-production-run"
    plan_hash = _digest(family.encode())
    run = {
        "run_id": run_id,
        "experiment_name": f"{family} qualification",
        "plan_hash": plan_hash,
        "desired_state": "completed",
        "observed_state": "completed",
        "current_node_id": "",
        "current_attempt_id": "",
        "run_revision": 12,
        "optimizer_step": 4,
        "last_heartbeat_ns": 100,
        "last_event_sequence": 20,
        "failure_summary": "",
    }
    event_types = [
        "run.created",
        "host.resource_grant_recorded",
        "host.process_prepared",
        "host.process_committed",
        "worker.ready",
        "metric.sampled",
        "artifact.published",
        "node.attempt_restarted",
        "host.process_exited",
        "host.resource_release_receipt_recorded",
        "run.observed_state_changed",
    ]
    timeline = [
        {
            "sequence": index,
            "event_id": f"event-{index}",
            "run_id": run_id,
            "event_type": event_type,
            **(
                {
                    "optimizer_step": 2,
                    "payload": {"checkpoint_artifact_id": "checkpoint-1"},
                }
                if event_type == "node.attempt_restarted"
                else {}
            ),
        }
        for index, event_type in enumerate(event_types, 1)
    ]
    metrics = [
        {
            "run_id": run_id,
            "name": "train.loss",
            "value": 1.0,
            "optimizer_step": 4,
        },
        {
            "run_id": run_id,
            "name": "eval.loss",
            "value": 0.9,
            "optimizer_step": 4,
        },
    ]
    artifacts = [
        {"artifact_id": "checkpoint-1", "kind": "checkpoint"},
        {"artifact_id": "trace-1", "kind": "opaque"},
    ]
    checkpoints = [
        {
            "artifact_id": "checkpoint-1",
            "valid": True,
            "optimizer_step": 4,
        }
    ]
    galleries = {
        "galleries": [
            {
                "artifact_id": "gallery-1",
                "item_count": 2,
                "checkpoint_manifest_digest": _digest(b"checkpoint"),
            }
        ],
        "history_truncated": False,
    }
    profiles = [
        {
            "artifact_id": "trace-1",
            "capture_steps": 2,
            "summary": {"accelerator_time_us": 12.5},
        }
    ]
    adapter = {
        "mageflow": "rwkv-lab.mageflow-terminal-expert",
        "rwkv": "rwkv-lab.rwkv-scratch",
        "transformer": "rwkv-lab.transformer-mla",
    }[family]
    canonical_plan = {
        "apiVersion": "trainvm.openai/v1",
        "spec": {
            "components": {
                "trainer": {
                    "adapter": adapter,
                    "version": "1.0.0",
                    "runtime": "python_worker",
                }
            }
        },
    }
    gallery = {
        "artifact_id": "gallery-1",
        "items": [
            {
                "generated_image_url": "/generated?v=abc",
                "target_image_url": "/target?v=def",
            }
        ],
    }
    body = {
        "api_version": "trainvm.reproducibility-capsule/v1",
        "journal_id": "journal-id",
        "through_sequence": len(timeline),
        "run": run,
        "plan_hash": plan_hash,
        "canonical_plan": canonical_plan,
        "observability": {},
        "latest_metrics": metrics,
        "artifacts": artifacts,
        "artifact_history_truncated": False,
        "execution_phases": [],
        "control_events": [],
    }
    return {
        "run": run,
        "plan": {
            "journal_id": "journal-id",
            "run_id": run_id,
            "run_revision": 12,
            "plan_hash": plan_hash,
            "canonical_plan": canonical_plan,
        },
        "timeline": timeline,
        "metrics": metrics,
        "artifacts": artifacts,
        "checkpoints": checkpoints,
        "galleries": galleries,
        "gallery": gallery,
        "profiles": profiles,
        "capsule": {"body": body, "capsule_digest": _digest(_canonical(body))},
    }


def _bundle(root: Path) -> Path:
    commit = "a" * 40
    suite_names = sorted(MODULE.REQUIRED_ACCEPTANCE_SUITES)
    manifest: dict[str, Any] = {
        "api_version": MODULE.API_VERSION,
        "source_commit": commit,
        "acceptance": _publish(
            root,
            "acceptance.json",
            {
                "api_version": "trainvm.acceptance/v1",
                "commit": commit,
                "dirty_worktree": False,
                "scope": "gpu_unit",
                "gate_open": True,
                "generated_at": "2026-08-02T00:00:00Z",
                "suites": [
                    {"name": name, "status": "passed", "detail": f"{name}.log"}
                    for name in suite_names
                ],
            },
        ),
        "hostd_crash": _publish(root, "hostd-crash.json", _hostd_receipt()),
        "host_authority": _publish(
            root,
            "host-authority.json",
            {
                "api_version": "trainvm.hostd-authority-status/v1",
                "coordinator": {"startup_audit_passed": True, "poison_reason": ""},
                "startup_phase": "admitting",
                "ledger_verified": True,
                "process_launch_enabled": True,
                "mutation_enabled": True,
                "lease_renewal_poisoned": False,
                "remaining_unclosed_process_records": 0,
                "remaining_terminal_release_records": 0,
                "active_process_count": 0,
                "active_fence_count": 0,
                "resource_inventory_observed": True,
                "degraded_resource_count": 0,
            },
        ),
        "journal_verification": _publish(
            root,
            "journal.json",
            {"valid": True, "events": 100, "reason": ""},
        ),
        "runs": {},
    }
    for family in MODULE.FAMILIES:
        documents = _run_documents(family)
        manifest["runs"][family] = {
            name: _publish(root, f"{family}/{name}.json", document)
            for name, document in documents.items()
        }
    path = root / "bundle.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return path


def test_complete_production_bundle_opens_gate(tmp_path: Path) -> None:
    receipt = MODULE.verify_bundle(_bundle(tmp_path))
    assert receipt["gate_open"] is True
    assert receipt["source_commit"] == "a" * 40
    assert set(receipt["runs"]) == set(MODULE.FAMILIES)
    assert MODULE.SHA256.fullmatch(receipt["receipt_digest"])


def test_bundle_rejects_changed_evidence_bytes(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    (tmp_path / "rwkv" / "metrics.json").write_text("[]\n")
    with pytest.raises(MODULE.QualificationError, match="digest mismatch"):
        MODULE.verify_bundle(bundle)


def test_bundle_rejects_duplicate_fields(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    bundle.write_text(
        '{"api_version":"trainvm.production-acceptance-bundle/v1",'
        '"api_version":"trainvm.production-acceptance-bundle/v1"}\n'
    )
    with pytest.raises(MODULE.QualificationError, match="duplicate JSON field"):
        MODULE.verify_bundle(bundle)


def test_bundle_rejects_worker_launch_without_hostd_commit(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    manifest = json.loads(bundle.read_text())
    timeline_path = tmp_path / manifest["runs"]["transformer"]["timeline"]["path"]
    timeline = json.loads(timeline_path.read_text())
    timeline = [
        event for event in timeline if event["event_type"] != "host.process_committed"
    ]
    manifest["runs"]["transformer"]["timeline"] = _publish(
        tmp_path, "transformer/timeline.json", timeline
    )
    bundle.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    with pytest.raises(MODULE.QualificationError, match="real hostd process path"):
        MODULE.verify_bundle(bundle)


def test_bundle_rejects_kernel_only_gpu_skip(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    manifest = json.loads(bundle.read_text())
    acceptance_path = tmp_path / manifest["acceptance"]["path"]
    acceptance = json.loads(acceptance_path.read_text())
    next(item for item in acceptance["suites"] if item["name"] == "gpu")[
        "status"
    ] = "skipped"
    manifest["acceptance"] = _publish(tmp_path, "acceptance.json", acceptance)
    bundle.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    with pytest.raises(MODULE.QualificationError, match="not_passed=.*gpu"):
        MODULE.verify_bundle(bundle)
