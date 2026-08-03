#!/usr/bin/env python3
"""Verify the evidence required for a TrainVM production-parity claim.

This verifier deliberately does not launch trainers or accept shell commands.  A
qualification runner captures the dashboard and native receipts into a directory,
then this script checks the immutable evidence offline.  Every referenced document
is SHA-256 bound by the bundle so paths cannot silently change after capture.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

API_VERSION = "trainvm.production-acceptance-bundle/v1"
RECEIPT_VERSION = "trainvm.production-acceptance-receipt/v1"
FAMILIES = ("mageflow", "rwkv", "transformer")
RUN_EVIDENCE = (
    "run",
    "plan",
    "timeline",
    "metrics",
    "artifacts",
    "checkpoints",
    "galleries",
    "gallery",
    "profiles",
    "capsule",
)
REQUIRED_ACCEPTANCE_SUITES = frozenset(
    {
        "native-configure",
        "native-build",
        "native-ctest",
        "compatibility-catalog",
        "schema-golden",
        "coverage-gate",
        "python-cpu",
        "go",
        "gpu",
    }
)
REQUIRED_HOST_EVENTS = frozenset(
    {
        "host.resource_grant_recorded",
        "host.process_prepared",
        "host.process_committed",
        "worker.ready",
        "host.process_exited",
        "host.resource_release_receipt_recorded",
    }
)
HOSTD_CRASH_POINTS = (
    ("intent_prepare_window", "durable_ledger"),
    ("intent_commit_window", "durable_ledger"),
    ("spawn_prepare_window", "durable_ledger"),
    ("spawn_commit_window", "durable_ledger"),
    ("exit_prepare_window", "durable_ledger"),
    ("release_prepare_window", "durable_ledger"),
    ("release_reply_lost", "durable_ledger"),
    ("live_worker_exact_adoption", "real_process"),
    ("live_worker_orphan_termination", "real_process"),
    ("mismatched_identity_refuses_adoption", "real_process"),
    ("absent_pid_refuses_adoption", "real_process"),
    ("intent_cgroup_termination", "real_cgroup"),
    ("terminal_cgroup_cleanup", "real_cgroup"),
    ("privileged_stopped_child_before_spawn_commit", "privileged_launch"),
    ("privileged_device_policy_recovery", "privileged_launch"),
    ("privileged_daemon_socket_restart", "privileged_launch"),
)
SHA256 = re.compile(r"^sha256:[0-9a-f]{64}$")
COMMIT = re.compile(r"^[0-9a-f]{40}$")


class QualificationError(ValueError):
    """An evidence bundle cannot support a production acceptance claim."""


@dataclass(frozen=True)
class LoadedEvidence:
    label: str
    path: Path
    digest: str
    document: Any


def _strict_object(value: Any, keys: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        observed = sorted(value) if isinstance(value, dict) else type(value).__name__
        raise QualificationError(
            f"{label} must have exactly {sorted(keys)}; observed {observed}"
        )
    return value


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _digest_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _digest_document_without(value: dict[str, Any], field: str) -> str:
    material = dict(value)
    material[field] = ""
    return _digest_bytes(_canonical_bytes(material))


def _object_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise QualificationError(f"duplicate JSON field: {key}")
        result[key] = value
    return result


def _invalid_constant(value: str) -> Any:
    raise QualificationError(f"non-finite JSON number: {value}")


def _load_json(path: Path, label: str) -> Any:
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise QualificationError(f"cannot read {label}: {error}") from error
    try:
        return (
            json.loads(
                raw,
                object_pairs_hook=_object_without_duplicates,
                parse_constant=_invalid_constant,
            ),
            raw,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise QualificationError(f"{label} is not valid JSON: {error}") from error


def _load_evidence(
    bundle_root: Path, reference: Any, label: str
) -> LoadedEvidence:
    reference = _strict_object(reference, {"path", "sha256"}, f"{label} reference")
    relative = reference["path"]
    expected = reference["sha256"]
    if not isinstance(expected, str) or not SHA256.fullmatch(expected):
        raise QualificationError(f"{label} reference is malformed")
    if (
        not isinstance(relative, str)
        or not relative
        or "\x00" in relative
        or Path(relative).is_absolute()
    ):
        raise QualificationError(f"{label} reference is malformed")
    lexical = Path(os.path.normpath(relative))
    if lexical == Path(".") or lexical.parts[0] == "..":
        raise QualificationError(f"{label} reference escapes the bundle")
    unresolved = bundle_root.joinpath(lexical)
    try:
        path = unresolved.resolve(strict=True)
        path.relative_to(bundle_root)
        metadata = unresolved.lstat()
    except OSError as error:
        raise QualificationError(f"cannot stat {label}: {error}") from error
    except ValueError as error:
        raise QualificationError(f"{label} reference escapes the bundle") from error
    cursor = bundle_root
    for part in lexical.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise QualificationError(f"{label} reference traverses a symlink")
    if not path.is_file() or metadata.st_size > 64 * 1024 * 1024:
        raise QualificationError(f"{label} must be a bounded regular non-symlink file")
    document, raw = _load_json(path, label)
    actual = _digest_bytes(raw)
    if actual != expected:
        raise QualificationError(
            f"{label} digest mismatch: expected {expected}, observed {actual}"
        )
    return LoadedEvidence(label, path, actual, document)


def _verify_acceptance(value: Any, commit: str) -> str:
    if not isinstance(value, dict):
        raise QualificationError("developer acceptance receipt must be an object")
    if value.get("api_version") != "trainvm.acceptance/v2":
        raise QualificationError("developer acceptance receipt version is unsupported")
    if value.get("scope") != "gpu_unit" or value.get("gate_open") is not True:
        raise QualificationError(
            "developer acceptance did not pass its complete GPU-unit scope"
        )
    if value.get("commit") != commit or value.get("dirty_worktree") is not False:
        raise QualificationError(
            "developer acceptance was not run at the clean production source commit"
        )
    suites = value.get("suites")
    if not isinstance(suites, list):
        raise QualificationError("developer acceptance suites are missing")
    statuses: dict[str, str] = {}
    for suite in suites:
        if not isinstance(suite, dict):
            raise QualificationError("developer acceptance suite is malformed")
        name, status = suite.get("name"), suite.get("status")
        if not isinstance(name, str) or name in statuses or not isinstance(status, str):
            raise QualificationError("developer acceptance suite identities are malformed")
        statuses[name] = status
    missing = sorted(REQUIRED_ACCEPTANCE_SUITES.difference(statuses))
    failed = sorted(
        name for name in REQUIRED_ACCEPTANCE_SUITES if statuses.get(name) != "passed"
    )
    if missing or failed:
        raise QualificationError(
            f"developer acceptance is incomplete; missing={missing}, not_passed={failed}"
        )
    native_binaries = value.get("native_binaries")
    if (
        not isinstance(native_binaries, dict)
        or set(native_binaries) != {"hostd_crash_qualification"}
        or not isinstance(
            native_binaries.get("hostd_crash_qualification"), str
        )
        or not SHA256.fullmatch(native_binaries["hostd_crash_qualification"])
    ):
        raise QualificationError(
            "developer acceptance does not bind the hostd crash qualification binary"
        )
    return cast(str, native_binaries["hostd_crash_qualification"])


def _verify_hostd_crash(value: Any, expected_binary_digest: str) -> None:
    if not isinstance(value, dict):
        raise QualificationError("hostd crash receipt must be an object")
    if value.get("api_version") != "trainvm.hostd-crash-qualification/v2":
        raise QualificationError("hostd crash receipt version is unsupported")
    if value.get("qualification_binary_digest") != expected_binary_digest:
        raise QualificationError(
            "hostd crash receipt was not produced by the accepted native binary"
        )
    cases = value.get("cases")
    if (
        value.get("gate_open") is not True
        or value.get("declared_points") != 16
        or value.get("qualified_points") != 16
        or value.get("unqualified_points") != 0
        or value.get("blocking_points") != []
        or value.get("findings") != []
        or not isinstance(cases, list)
        or len(cases) != 16
    ):
        raise QualificationError("the complete privileged hostd crash gate is not open")
    for case, (expected_point, expected_executor) in zip(
        cases, HOSTD_CRASH_POINTS, strict=True
    ):
        if not isinstance(case, dict):
            raise QualificationError("hostd crash case is malformed")
        point = case.get("crash_point")
        if (
            point != expected_point
            or case.get("executor") != expected_executor
            or case.get("status") != "qualified"
            or case.get("unqualified_reason") != "none"
            or not isinstance(case.get("invariants"), list)
            or not case["invariants"]
        ):
            raise QualificationError("hostd crash case does not prove its declared point")
    digest = value.get("receipt_digest")
    if not isinstance(digest, str) or digest != _digest_document_without(
        value, "receipt_digest"
    ):
        raise QualificationError("hostd crash receipt digest is invalid")


def _verify_journal(value: Any) -> None:
    if not isinstance(value, dict) or value.get("valid") is not True:
        raise QualificationError("native journal verification did not pass")
    events = value.get("events")
    if isinstance(events, bool) or not isinstance(events, int) or events < 1:
        raise QualificationError("native journal verification contains no durable events")
    if value.get("reason") not in ("", None):
        raise QualificationError("native journal verification reports a failure reason")


def _finite_metric(metric: Any, run_id: str) -> bool:
    if not isinstance(metric, dict) or metric.get("run_id") != run_id:
        return False
    value = metric.get("value")
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
    )


def _is_eval_metric(name: Any) -> bool:
    return isinstance(name, str) and (
        name.startswith(("eval.", "eval/"))
        or "validation" in name.lower()
        or name.lower().startswith("val.")
    )


def _verify_capsule(value: Any, run_id: str, plan_hash: str) -> None:
    if not isinstance(value, dict) or set(value) != {"body", "capsule_digest"}:
        raise QualificationError("reproducibility capsule is malformed")
    body = value["body"]
    if (
        not isinstance(body, dict)
        or body.get("api_version") != "trainvm.reproducibility-capsule/v1"
        or not isinstance(body.get("run"), dict)
        or body["run"].get("run_id") != run_id
        or body.get("plan_hash") != plan_hash
    ):
        raise QualificationError("reproducibility capsule lost run identity")
    if value["capsule_digest"] != _digest_bytes(_canonical_bytes(body)):
        raise QualificationError("reproducibility capsule digest is invalid")


def _verify_run(family: str, documents: dict[str, Any]) -> dict[str, Any]:
    run = documents["run"]
    if not isinstance(run, dict):
        raise QualificationError(f"{family} run summary is malformed")
    run_id, plan_hash = run.get("run_id"), run.get("plan_hash")
    if (
        not isinstance(run_id, str)
        or not run_id
        or not isinstance(plan_hash, str)
        or not plan_hash
        or run.get("observed_state") != "completed"
        or run.get("desired_state") != "completed"
        or run.get("failure_summary") not in ("", None)
    ):
        raise QualificationError(f"{family} run did not complete successfully")

    plan = documents["plan"]
    if (
        not isinstance(plan, dict)
        or plan.get("run_id") != run_id
        or plan.get("plan_hash") != plan_hash
        or not isinstance(plan.get("canonical_plan"), dict)
    ):
        raise QualificationError(f"{family} compiled plan identity is inconsistent")
    canonical_plan = plan["canonical_plan"]
    spec = canonical_plan.get("spec")
    components = spec.get("components") if isinstance(spec, dict) else None
    if not isinstance(components, dict):
        raise QualificationError(f"{family} compiled plan has no component authority")
    adapters = {
        component.get("adapter")
        for component in components.values()
        if isinstance(component, dict) and isinstance(component.get("adapter"), str)
    }
    family_adapter = {
        "mageflow": lambda value: value.startswith("rwkv-lab.mageflow-"),
        "rwkv": lambda value: value.startswith("rwkv-lab.rwkv-"),
        "transformer": lambda value: value.startswith("rwkv-lab.transformer-")
        or value == "rwkv-lab.qwen-ao3",
    }[family]
    if not any(family_adapter(adapter) for adapter in adapters):
        raise QualificationError(f"{family} plan does not invoke its registered trainer family")

    timeline = documents["timeline"]
    if not isinstance(timeline, list) or not timeline:
        raise QualificationError(f"{family} timeline is empty")
    sequences: list[int] = []
    event_types: set[str] = set()
    for event in timeline:
        if not isinstance(event, dict) or event.get("run_id") != run_id:
            raise QualificationError(f"{family} timeline contains foreign events")
        sequence, event_type = event.get("sequence"), event.get("event_type")
        if (
            isinstance(sequence, bool)
            or not isinstance(sequence, int)
            or sequence < 1
            or not isinstance(event_type, str)
        ):
            raise QualificationError(f"{family} timeline event is malformed")
        sequences.append(sequence)
        event_types.add(event_type)
    if sequences != sorted(set(sequences)):
        raise QualificationError(f"{family} timeline is not a complete ordered prefix")
    missing_events = sorted(REQUIRED_HOST_EVENTS.difference(event_types))
    if missing_events:
        raise QualificationError(
            f"{family} did not traverse the real hostd process path: {missing_events}"
        )
    resumed_checkpoint_ids = {
        event.get("payload", {}).get("checkpoint_artifact_id")
        for event in timeline
        if event.get("event_type") == "node.attempt_restarted"
        and isinstance(event.get("payload"), dict)
        and isinstance(event["payload"].get("checkpoint_artifact_id"), str)
        and event["payload"]["checkpoint_artifact_id"]
    }
    if not resumed_checkpoint_ids:
        raise QualificationError(f"{family} has no durable checkpoint resume evidence")
    resumed_steps = {
        event.get("optimizer_step")
        for event in timeline
        if event.get("event_type") == "node.attempt_restarted"
        and isinstance(event.get("optimizer_step"), int)
        and not isinstance(event.get("optimizer_step"), bool)
    }
    final_step = run.get("optimizer_step")
    if (
        not resumed_steps
        or isinstance(final_step, bool)
        or not isinstance(final_step, int)
        or final_step <= max(resumed_steps)
    ):
        raise QualificationError(f"{family} made no optimizer progress after resume")

    metrics = documents["metrics"]
    if not isinstance(metrics, list) or not metrics:
        raise QualificationError(f"{family} emitted no live metrics")
    if not all(_finite_metric(metric, run_id) for metric in metrics):
        raise QualificationError(f"{family} metrics contain non-finite or foreign samples")
    if not any(_is_eval_metric(metric.get("name")) for metric in metrics):
        raise QualificationError(f"{family} emitted no evaluation metric")
    if not any(
        isinstance(metric.get("optimizer_step"), int)
        and not isinstance(metric.get("optimizer_step"), bool)
        and metric["optimizer_step"] > max(resumed_steps)
        for metric in metrics
    ):
        raise QualificationError(f"{family} emitted no live metric after resume")

    artifacts = documents["artifacts"]
    if not isinstance(artifacts, list) or not artifacts:
        raise QualificationError(f"{family} published no artifacts")
    if not any(isinstance(item, dict) and item.get("kind") == "checkpoint" for item in artifacts):
        raise QualificationError(f"{family} published no checkpoint artifact")

    checkpoints = documents["checkpoints"]
    if (
        not isinstance(checkpoints, list)
        or not checkpoints
        or not all(isinstance(item, dict) and item.get("valid") is True for item in checkpoints)
    ):
        raise QualificationError(f"{family} has no fully verified checkpoint")
    checkpoint_ids = {item.get("artifact_id") for item in checkpoints}
    if resumed_checkpoint_ids.isdisjoint(checkpoint_ids):
        raise QualificationError(
            f"{family} restart is not bound to a verified published checkpoint"
        )

    profiles = documents["profiles"]
    if not isinstance(profiles, list) or not profiles:
        raise QualificationError(f"{family} published no GPU trace profile")
    if not all(
        isinstance(profile, dict)
        and profile.get("capture_steps", 0) > 0
        and isinstance(profile.get("summary"), dict)
        and profile["summary"].get("accelerator_time_us", 0) > 0
        for profile in profiles
    ):
        raise QualificationError(f"{family} GPU trace evidence is malformed")

    galleries = documents["galleries"]
    if family == "mageflow":
        history = galleries.get("galleries") if isinstance(galleries, dict) else None
        if (
            not isinstance(history, list)
            or not history
            or not any(
                isinstance(item, dict)
                and item.get("item_count", 0) > 0
                and item.get("checkpoint_manifest_digest")
                for item in history
            )
        ):
            raise QualificationError(
                "MageFlow published no checkpoint-bound side-by-side eval gallery"
            )
        gallery = documents["gallery"]
        items = gallery.get("items") if isinstance(gallery, dict) else None
        if (
            not isinstance(items, list)
            or not items
            or gallery.get("artifact_id") not in {
                item.get("artifact_id") for item in history if isinstance(item, dict)
            }
            or not all(
                isinstance(item, dict)
                and isinstance(item.get("generated_image_url"), str)
                and item["generated_image_url"]
                and (
                    isinstance(item.get("target_image_url"), str)
                    and bool(item["target_image_url"])
                    or isinstance(item.get("source_image_url"), str)
                    and bool(item["source_image_url"])
                )
                for item in items
            )
        ):
            raise QualificationError(
                "MageFlow eval gallery does not expose generated/original side-by-side items"
            )

    _verify_capsule(documents["capsule"], run_id, plan_hash)
    return {
        "run_id": run_id,
        "plan_hash": plan_hash,
        "optimizer_step": run.get("optimizer_step", 0),
        "events": len(timeline),
        "metrics": len(metrics),
        "artifacts": len(artifacts),
        "checkpoints": len(checkpoints),
        "profiles": len(profiles),
    }


def verify_bundle(path: Path) -> dict[str, Any]:
    path = path.absolute()
    if path.is_symlink() or not path.is_file():
        raise QualificationError("bundle manifest must be a regular non-symlink file")
    manifest, _ = _load_json(path, "production acceptance bundle")
    manifest = cast(
        dict[str, Any],
        _strict_object(
        manifest,
        {
            "api_version",
            "source_commit",
            "acceptance",
            "hostd_crash",
            "host_authority",
            "journal_verification",
            "runs",
        },
        "production acceptance bundle",
        ),
    )
    if manifest["api_version"] != API_VERSION:
        raise QualificationError("production acceptance bundle version is unsupported")
    commit = manifest["source_commit"]
    if not isinstance(commit, str) or not COMMIT.fullmatch(commit):
        raise QualificationError("production acceptance source commit is malformed")
    root = path.parent.resolve(strict=True)

    loaded: list[LoadedEvidence] = []
    for key in ("acceptance", "hostd_crash", "host_authority", "journal_verification"):
        loaded.append(_load_evidence(root, manifest[key], key))
    by_label = {item.label: item for item in loaded}
    crash_binary_digest = _verify_acceptance(
        by_label["acceptance"].document, commit
    )
    _verify_hostd_crash(
        by_label["hostd_crash"].document, crash_binary_digest
    )
    _verify_journal(by_label["journal_verification"].document)
    authority = by_label["host_authority"].document
    coordinator = authority.get("coordinator") if isinstance(authority, dict) else None
    if (
        not isinstance(authority, dict)
        or authority.get("api_version") != "trainvm.hostd-authority-status/v1"
        or authority.get("startup_phase") != "admitting"
        or authority.get("ledger_verified") is not True
        or authority.get("process_launch_enabled") is not True
        or authority.get("mutation_enabled") is not True
        or authority.get("lease_renewal_poisoned") is not False
        or authority.get("remaining_unclosed_process_records") != 0
        or authority.get("remaining_terminal_release_records") != 0
        or authority.get("active_process_count") != 0
        or authority.get("active_fence_count") != 0
        or authority.get("resource_inventory_observed") is not True
        or authority.get("degraded_resource_count") != 0
        or not isinstance(coordinator, dict)
        or coordinator.get("startup_audit_passed") is not True
        or coordinator.get("poison_reason") not in ("", None)
    ):
        raise QualificationError(
            "dashboard did not observe healthy, admitted, fully released hostd authority"
        )

    runs = manifest["runs"]
    if not isinstance(runs, dict) or set(runs) != set(FAMILIES):
        raise QualificationError(f"production runs must be exactly {list(FAMILIES)}")
    run_results: dict[str, Any] = {}
    for family in FAMILIES:
        references = _strict_object(
            runs[family], set(RUN_EVIDENCE), f"{family} run evidence"
        )
        documents: dict[str, Any] = {}
        for name in RUN_EVIDENCE:
            item = _load_evidence(root, references[name], f"{family}.{name}")
            loaded.append(item)
            documents[name] = item.document
        run_results[family] = _verify_run(family, documents)

    evidence = {item.label: item.digest for item in loaded}
    body = {
        "api_version": RECEIPT_VERSION,
        "source_commit": commit,
        "gate_open": True,
        "evidence": dict(sorted(evidence.items())),
        "runs": run_results,
    }
    return {**body, "receipt_digest": _digest_bytes(_canonical_bytes(body))}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--receipt", type=Path)
    arguments = parser.parse_args()
    try:
        receipt = verify_bundle(arguments.bundle)
    except QualificationError as error:
        print(f"production acceptance rejected: {error}", file=sys.stderr)
        return 1
    encoded = json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    if arguments.receipt is None:
        print(encoded, end="")
    else:
        arguments.receipt.write_text(encoded)
        print(arguments.receipt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
