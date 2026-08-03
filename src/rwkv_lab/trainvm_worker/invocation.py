from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from ._canonical import (
    CanonicalJsonError,
    canonical_dumps,
    canonical_loads,
    deep_freeze,
    exact_fields,
    is_bounded_text,
    is_digest,
    is_uint64,
    sha256_digest,
)
from .training import ResolvedTrainingComposition, load_resolved_training_composition

INVOCATION_API_VERSION = "trainvm.worker-invocation/v2"
LEGACY_INVOCATION_API_VERSION = "trainvm.worker-invocation/v1"
MAXIMUM_INVOCATION_BYTES = 48 * 1024
_FIELDS = frozenset(
    {
        "adapter",
        "api_version",
        "attempt_id",
        "controls",
        "dispatch_id",
        "effective_control_revision",
        "execution",
        "host_id",
        "inputs",
        "invocation_digest",
        "node_id",
        "observability",
        "plan_hash",
        "plan_revision",
        "publishes",
        "resources",
        "resume",
        "run_id",
        "training",
        "workspace",
    }
)
_ADAPTER_FIELDS = frozenset({"adapter", "contract", "operation", "runtime", "version"})
_RESUME_FIELDS = frozenset(
    {"api_version", "checkpoint", "optimizer_step", "pause_command_id", "resume_command_id"}
)
_ARTIFACT_FIELDS = frozenset(
    {
        "artifact_id", "logical_name", "kind", "schema", "uri", "size_bytes",
        "fingerprint_algorithm", "fingerprint", "complete", "producer_node_id",
        "producer_attempt_id", "parent_artifact_ids", "published_at_ns",
    }
)


class InvocationError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class WorkerInvocation:
    run_id: str
    node_id: str
    attempt_id: str
    dispatch_id: str
    plan_hash: str
    plan_revision: int
    host_id: str
    adapter: Mapping[str, Any]
    workspace: Mapping[str, Any]
    resources: Mapping[str, Any]
    inputs: Mapping[str, Any]
    controls: Mapping[str, Any]
    effective_control_revision: int
    publishes: Mapping[str, Any]
    observability: Mapping[str, Any]
    execution: Mapping[str, Any] | None
    training: ResolvedTrainingComposition | None
    resume: Mapping[str, Any] | None
    invocation_digest: str


def _object_or_none(value: Any) -> bool:
    return value is None or isinstance(value, dict)


def _load_resume(value: Any, *, node_id: str, attempt_id: str) -> Mapping[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise InvocationError("worker invocation resume authority is not an object")
    exact_fields(value, _RESUME_FIELDS)
    checkpoint = value.get("checkpoint")
    if not isinstance(checkpoint, dict):
        raise InvocationError("worker invocation resume checkpoint is not an object")
    exact_fields(checkpoint, _ARTIFACT_FIELDS)
    parents = checkpoint.get("parent_artifact_ids")
    valid_parents = (
        isinstance(parents, list)
        and all(is_bounded_text(parent, 1024) for parent in parents)
        and len(parents) == len(set(parents))
    )
    valid = (
        value.get("api_version") == "trainvm.resume-checkpoint/v1"
        and is_uint64(value.get("optimizer_step"))
        and is_bounded_text(value.get("pause_command_id"), 1024)
        and is_bounded_text(value.get("resume_command_id"), 1024)
        and is_bounded_text(checkpoint.get("artifact_id"), 1024)
        and is_bounded_text(checkpoint.get("logical_name"), 1024)
        and checkpoint.get("kind") == "checkpoint"
        and is_bounded_text(checkpoint.get("schema"), 512)
        and is_bounded_text(checkpoint.get("uri"), 4096)
        and is_uint64(checkpoint.get("size_bytes"))
        and checkpoint.get("fingerprint_algorithm") == "manifest_sha256"
        and is_digest(checkpoint.get("fingerprint"))
        and checkpoint.get("complete") is True
        and checkpoint.get("producer_node_id") == node_id
        and is_bounded_text(checkpoint.get("producer_attempt_id"), 1024)
        and checkpoint.get("producer_attempt_id") != attempt_id
        and valid_parents
        and is_uint64(checkpoint.get("published_at_ns"))
    )
    if not valid:
        raise InvocationError("worker invocation resume checkpoint lineage is invalid")
    return deep_freeze(value)


def load_worker_invocation(
    raw: bytes,
    *,
    expected_digest: str | None = None,
    expected_run_id: str | None = None,
    expected_node_id: str | None = None,
    expected_attempt_id: str | None = None,
    expected_plan_revision: int | None = None,
) -> WorkerInvocation:
    try:
        document = canonical_loads(raw, maximum_bytes=MAXIMUM_INVOCATION_BYTES)
        api_version = document.get("api_version")
        fields = (
            _FIELDS - {"resume"}
            if api_version == LEGACY_INVOCATION_API_VERSION
            else _FIELDS
        )
        exact_fields(document, fields)
        adapter = document["adapter"]
        if not isinstance(adapter, dict):
            raise InvocationError("worker invocation adapter is not an object")
        exact_fields(adapter, _ADAPTER_FIELDS)
        digest = document["invocation_digest"]
        body = dict(document)
        del body["invocation_digest"]
        objects = (
            "workspace",
            "resources",
            "inputs",
            "controls",
            "publishes",
            "observability",
        )
        valid = (
            document["api_version"]
            in {INVOCATION_API_VERSION, LEGACY_INVOCATION_API_VERSION}
            and is_bounded_text(document["run_id"], 1024)
            and is_bounded_text(document["node_id"], 1024)
            and is_bounded_text(document["attempt_id"], 1024)
            and is_bounded_text(document["dispatch_id"], 4096)
            and isinstance(document["plan_hash"], str)
            and len(document["plan_hash"]) == 64
            and all(
                character in "0123456789abcdef" for character in document["plan_hash"]
            )
            and is_uint64(document["plan_revision"], positive=True)
            and is_digest(document["host_id"])
            and all(isinstance(document[name], dict) for name in objects)
            and _object_or_none(document["execution"])
            and _object_or_none(document["training"])
            and _object_or_none(document.get("resume"))
            and is_uint64(document["effective_control_revision"])
            and all(is_bounded_text(adapter[name], 256) for name in _ADAPTER_FIELDS)
            and adapter["runtime"] not in {"builtin", "external_worker"}
            and is_digest(digest)
            and sha256_digest(canonical_dumps(body)) == digest
        )
        if not valid:
            raise InvocationError("worker invocation semantics are invalid")
        expected = (
            (expected_digest, digest, "digest"),
            (expected_run_id, document["run_id"], "run"),
            (expected_node_id, document["node_id"], "node"),
            (expected_attempt_id, document["attempt_id"], "attempt"),
            (expected_plan_revision, document["plan_revision"], "plan revision"),
        )
        for wanted, actual, label in expected:
            if wanted is not None and wanted != actual:
                raise InvocationError(f"worker invocation {label} binding disagrees")
        return WorkerInvocation(
            run_id=document["run_id"],
            node_id=document["node_id"],
            attempt_id=document["attempt_id"],
            dispatch_id=document["dispatch_id"],
            plan_hash=document["plan_hash"],
            plan_revision=document["plan_revision"],
            host_id=document["host_id"],
            adapter=deep_freeze(adapter),
            workspace=deep_freeze(document["workspace"]),
            resources=deep_freeze(document["resources"]),
            inputs=deep_freeze(document["inputs"]),
            controls=deep_freeze(document["controls"]),
            effective_control_revision=document["effective_control_revision"],
            publishes=deep_freeze(document["publishes"]),
            observability=deep_freeze(document["observability"]),
            execution=deep_freeze(document["execution"]),
            training=(
                load_resolved_training_composition(document["training"])
                if document["training"] is not None
                else None
            ),
            resume=_load_resume(
                document.get("resume"),
                node_id=document["node_id"],
                attempt_id=document["attempt_id"],
            ),
            invocation_digest=digest,
        )
    except InvocationError:
        raise
    except (CanonicalJsonError, KeyError, TypeError, ValueError) as error:
        raise InvocationError("worker invocation decoding failed closed") from error
