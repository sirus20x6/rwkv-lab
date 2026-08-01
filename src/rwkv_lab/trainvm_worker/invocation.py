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

INVOCATION_API_VERSION = "trainvm.worker-invocation/v1"
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
        "run_id",
        "training",
        "workspace",
    }
)
_ADAPTER_FIELDS = frozenset({"adapter", "contract", "operation", "runtime", "version"})


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
    invocation_digest: str


def _object_or_none(value: Any) -> bool:
    return value is None or isinstance(value, dict)


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
        exact_fields(document, _FIELDS)
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
            document["api_version"] == INVOCATION_API_VERSION
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
            invocation_digest=digest,
        )
    except InvocationError:
        raise
    except (CanonicalJsonError, KeyError, TypeError, ValueError) as error:
        raise InvocationError("worker invocation decoding failed closed") from error
