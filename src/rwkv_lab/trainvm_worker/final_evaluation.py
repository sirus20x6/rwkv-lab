"""Canonical terminal-evaluation closure publication.

The closure is intentionally a small, direct immutable report rather than a
generic tree snapshot.  Its bytes are the reducer input, are content addressed,
and remain outside the 64 KiB worker envelope while the controller re-reads and
verifies them from the sealed run workspace.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from ._canonical import canonical_dumps, is_bounded_text, is_digest, sha256_digest

try:
    from trainvm.v1 import trainvm_pb2 as wire
except ImportError as error:  # pragma: no cover - installation contract
    raise RuntimeError(
        "TrainVM final-evaluation publication requires the 'trainvm-worker' project extra"
    ) from error


FINAL_EVALUATION_SCHEMA = "rwkv-lab.final-evaluation.v1"
FINAL_EVALUATION_API_VERSION = "rwkv-lab.final-evaluation/v1"
MAXIMUM_MANIFEST_BYTES = 32 * 1024 * 1024


class FinalEvaluationPublicationError(RuntimeError):
    pass


class _Session(Protocol):
    bootstrap: object
    invocation: object

    def artifact(self, **values: object) -> int: ...


@dataclass(frozen=True, slots=True)
class FinalEvaluationPublicationRequest:
    optimizer_step: int
    checkpoint_artifact_id: str
    checkpoint_fingerprint: str
    required_members: tuple[str, ...]
    member_context_digests: Mapping[str, str]
    records: tuple[Mapping[str, Any], ...]
    output_receipts: tuple[Mapping[str, str], ...]
    required_scalars: tuple[Mapping[str, str], ...]
    parent_artifact_ids: tuple[str, ...]
    output_name: str = "final_evaluation"


@dataclass(frozen=True, slots=True)
class PublishedFinalEvaluation:
    artifact_id: str
    manifest_path: Path
    manifest_sha256: str
    optimizer_step: int
    worker_sequence: int


def _text(value: object, label: str, maximum: int = 1024) -> str:
    if not is_bounded_text(value, maximum):
        raise FinalEvaluationPublicationError(f"final-evaluation {label} is invalid")
    assert isinstance(value, str)
    return value


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


class FinalEvaluationPublisher:
    def __init__(self, session: _Session, *, output_name: str = "final_evaluation"):
        self._session = session
        invocation = session.invocation
        publishes = getattr(invocation, "publishes", None)
        workspace = getattr(invocation, "workspace", None)
        if not isinstance(publishes, Mapping) or not isinstance(workspace, Mapping):
            raise FinalEvaluationPublicationError(
                "final-evaluation invocation authority is malformed"
            )
        publication = publishes.get(output_name)
        declaration = (
            publication.get("declaration") if isinstance(publication, Mapping) else None
        )
        if (
            not isinstance(publication, Mapping)
            or not isinstance(declaration, Mapping)
            or declaration.get("type") != "report"
            or declaration.get("schema") != FINAL_EVALUATION_SCHEMA
            or declaration.get("immutability") != "immutable"
            or declaration.get("fingerprint") != "manifest_sha256"
            or not is_digest(publication.get("finalization_policy_digest"))
        ):
            raise FinalEvaluationPublicationError(
                "final-evaluation output lacks closed controller policy authority"
            )
        self._output_name = output_name
        self._logical_name = _text(publication.get("logical_name"), "logical name")
        self._policy_digest = str(publication["finalization_policy_digest"])
        run_directory = workspace.get("run_directory")
        if not isinstance(run_directory, str) or not Path(run_directory).is_absolute():
            raise FinalEvaluationPublicationError("run directory is invalid")
        self._run_root = Path(run_directory).resolve(strict=True)
        self._revisions = self._run_root / "trainvm_artifacts" / "final_evaluation"
        self._revisions.mkdir(mode=0o750, parents=True, exist_ok=True)
        if (
            self._revisions.is_symlink()
            or self._run_root not in self._revisions.parents
        ):
            raise FinalEvaluationPublicationError("publication root escaped the run")

    def publish(
        self, request: FinalEvaluationPublicationRequest
    ) -> PublishedFinalEvaluation:
        if request.output_name != self._output_name:
            raise FinalEvaluationPublicationError(
                "output name disagrees with publisher"
            )
        members = tuple(_text(value, "member id") for value in request.required_members)
        if not members or members != tuple(sorted(set(members))):
            raise FinalEvaluationPublicationError("member IDs are not canonical")
        contexts = request.member_context_digests
        if set(contexts) != set(members) or any(
            not is_digest(contexts[member]) for member in members
        ):
            raise FinalEvaluationPublicationError("member contexts are incomplete")
        output_receipts = tuple(dict(value) for value in request.output_receipts)
        if tuple(value.get("output_name") for value in output_receipts) != tuple(
            sorted(value.get("output_name") for value in output_receipts)
        ):
            raise FinalEvaluationPublicationError("output receipts are not canonical")
        parents = tuple(
            _text(value, "parent artifact ID") for value in request.parent_artifact_ids
        )
        if len(parents) != len(set(parents)) or set(parents) != {
            str(value.get("artifact_id")) for value in output_receipts
        }:
            raise FinalEvaluationPublicationError(
                "closure parents must exactly equal output receipt artifacts"
            )
        records = tuple(dict(value) for value in request.records)
        if tuple(value.get("member_id") for value in records) != members:
            raise FinalEvaluationPublicationError("member ledger is not canonical")
        resolved = sum(value.get("disposition") == "success" for value in records)
        manifest = {
            "api_version": FINAL_EVALUATION_API_VERSION,
            "policy_digest": self._policy_digest,
            "optimizer_step": request.optimizer_step,
            "checkpoint_artifact_id": _text(
                request.checkpoint_artifact_id, "checkpoint artifact ID"
            ),
            "checkpoint_fingerprint": request.checkpoint_fingerprint,
            "membership_digest": sha256_digest(canonical_dumps(list(members))),
            "membership_count": len(members),
            "resolved_member_count": resolved,
            "failed_member_count": len(members) - resolved,
            "required_members": list(members),
            "output_receipts": list(output_receipts),
            "required_scalars": [dict(value) for value in request.required_scalars],
            "records": list(records),
        }
        if not is_digest(request.checkpoint_fingerprint):
            raise FinalEvaluationPublicationError("checkpoint fingerprint is invalid")
        manifest_bytes = canonical_dumps(manifest)
        if len(manifest_bytes) > MAXIMUM_MANIFEST_BYTES:
            raise FinalEvaluationPublicationError("closure exceeds its byte bound")
        manifest_sha256 = sha256_digest(manifest_bytes)
        bootstrap = self._session.bootstrap
        artifact_id = (
            "final-evaluation-"
            + hashlib.sha256(
                canonical_dumps(
                    [
                        bootstrap.run_id,
                        bootstrap.node_id,
                        bootstrap.attempt_id,
                        request.optimizer_step,
                        manifest_sha256,
                    ]
                )
            ).hexdigest()
        )
        revision = self._revisions / artifact_id
        temporary = Path(tempfile.mkdtemp(prefix="revision-", dir=self._revisions))
        try:
            path = temporary / "manifest.json"
            descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o440)
            with os.fdopen(descriptor, "wb") as output:
                output.write(manifest_bytes)
                output.flush()
                os.fsync(output.fileno())
            os.chmod(temporary, 0o550)
            _fsync_directory(temporary)
            try:
                os.rename(temporary, revision)
                _fsync_directory(self._revisions)
            except OSError:
                existing = revision / "manifest.json"
                if not existing.is_file() or existing.read_bytes() != manifest_bytes:
                    raise FinalEvaluationPublicationError(
                        "final-evaluation revision identity collision"
                    )
        finally:
            if temporary.exists():
                os.chmod(temporary, 0o750)
                shutil.rmtree(temporary)
        manifest_path = revision / "manifest.json"
        sequence = self._session.artifact(
            artifact_id=artifact_id,
            logical_name=self._logical_name,
            kind=wire.ARTIFACT_KIND_REPORT,
            schema=FINAL_EVALUATION_SCHEMA,
            uri=manifest_path.resolve(strict=True).as_uri(),
            size_bytes=len(manifest_bytes),
            fingerprint_algorithm="manifest_sha256",
            fingerprint=manifest_sha256,
            parent_artifact_ids=parents,
            optimizer_step=request.optimizer_step,
            wait=True,
        )
        return PublishedFinalEvaluation(
            artifact_id,
            manifest_path,
            manifest_sha256,
            request.optimizer_step,
            sequence,
        )


__all__ = [
    "FINAL_EVALUATION_SCHEMA",
    "FinalEvaluationPublicationError",
    "FinalEvaluationPublicationRequest",
    "FinalEvaluationPublisher",
    "PublishedFinalEvaluation",
]
