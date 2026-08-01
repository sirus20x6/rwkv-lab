from __future__ import annotations

import fcntl
import os
import stat
from dataclasses import dataclass

from ._canonical import (
    CanonicalJsonError,
    canonical_dumps,
    canonical_loads,
    exact_fields,
    is_bounded_text,
    is_digest,
    is_uint64,
    sha256_digest,
)

BOOTSTRAP_API_VERSION = "trainvm.worker-bootstrap/v1"
MAXIMUM_BOOTSTRAP_BYTES = 16 * 1024
# Linux UAPI values are stable. Some minimal Python builds omit their symbolic
# fcntl names even though the kernel operations are available.
_F_GET_SEALS = getattr(fcntl, "F_GET_SEALS", 1034)
_F_SEAL_SEAL = getattr(fcntl, "F_SEAL_SEAL", 0x0001)
_F_SEAL_SHRINK = getattr(fcntl, "F_SEAL_SHRINK", 0x0002)
_F_SEAL_GROW = getattr(fcntl, "F_SEAL_GROW", 0x0004)
_F_SEAL_WRITE = getattr(fcntl, "F_SEAL_WRITE", 0x0008)
_FIELDS = frozenset(
    {
        "adapter",
        "adapter_version",
        "api_version",
        "attempt_id",
        "bootstrap_digest",
        "capabilities",
        "code_fingerprint",
        "concurrency_key",
        "controller_target",
        "fencing_token",
        "last_acked_controller_sequence",
        "launch_nonce",
        "lease_id",
        "node_id",
        "run_id",
    }
)


class BootstrapError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class WorkerBootstrap:
    controller_target: str
    run_id: str
    node_id: str
    attempt_id: str
    launch_nonce: str
    adapter: str
    adapter_version: str
    code_fingerprint: str
    capabilities: tuple[str, ...]
    last_acked_controller_sequence: int
    concurrency_key: str
    lease_id: str
    fencing_token: int
    bootstrap_digest: str


def load_worker_bootstrap(raw: bytes) -> WorkerBootstrap:
    try:
        document = canonical_loads(raw, maximum_bytes=MAXIMUM_BOOTSTRAP_BYTES)
        exact_fields(document, _FIELDS)
        digest = document["bootstrap_digest"]
        body = dict(document)
        del body["bootstrap_digest"]
        capabilities = body["capabilities"]
        valid = (
            body["api_version"] == BOOTSTRAP_API_VERSION
            and is_bounded_text(body["controller_target"], 4096)
            and body["controller_target"].startswith("unix:/")
            and is_bounded_text(body["run_id"], 1024)
            and is_bounded_text(body["node_id"], 1024)
            and is_bounded_text(body["attempt_id"], 1024)
            and is_bounded_text(body["launch_nonce"], 1024)
            and is_bounded_text(body["adapter"], 256)
            and is_bounded_text(body["adapter_version"], 256)
            and is_digest(body["code_fingerprint"])
            and isinstance(capabilities, list)
            and len(capabilities) <= 256
            and all(is_bounded_text(item, 256) for item in capabilities)
            and capabilities == sorted(set(capabilities))
            and is_uint64(body["last_acked_controller_sequence"])
            and is_bounded_text(body["concurrency_key"], 1024)
            and is_bounded_text(body["lease_id"], 1024)
            and is_uint64(body["fencing_token"], positive=True)
            and is_digest(digest)
            and sha256_digest(canonical_dumps(body)) == digest
        )
        if not valid:
            raise BootstrapError("worker bootstrap semantics are invalid")
        return WorkerBootstrap(
            controller_target=body["controller_target"],
            run_id=body["run_id"],
            node_id=body["node_id"],
            attempt_id=body["attempt_id"],
            launch_nonce=body["launch_nonce"],
            adapter=body["adapter"],
            adapter_version=body["adapter_version"],
            code_fingerprint=body["code_fingerprint"],
            capabilities=tuple(capabilities),
            last_acked_controller_sequence=body[
                "last_acked_controller_sequence"
            ],
            concurrency_key=body["concurrency_key"],
            lease_id=body["lease_id"],
            fencing_token=body["fencing_token"],
            bootstrap_digest=digest,
        )
    except BootstrapError:
        raise
    except (CanonicalJsonError, KeyError, TypeError, ValueError) as error:
        raise BootstrapError("worker bootstrap decoding failed closed") from error


def read_worker_bootstrap_fd(descriptor: int) -> WorkerBootstrap:
    """Read a sealed, regular memfd without changing its shared file offset."""
    try:
        metadata = os.fstat(descriptor)
        required_seals = (
            _F_SEAL_WRITE | _F_SEAL_GROW | _F_SEAL_SHRINK | _F_SEAL_SEAL
        )
        seals = fcntl.fcntl(descriptor, _F_GET_SEALS)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_size <= 0
            or metadata.st_size > MAXIMUM_BOOTSTRAP_BYTES
            or seals & required_seals != required_seals
        ):
            raise BootstrapError("worker bootstrap descriptor is not sealed authority")
        raw = os.pread(descriptor, metadata.st_size, 0)
        if len(raw) != metadata.st_size:
            raise BootstrapError("worker bootstrap descriptor was read incompletely")
        return load_worker_bootstrap(raw)
    except BootstrapError:
        raise
    except OSError as error:
        raise BootstrapError("worker bootstrap descriptor inspection failed") from error
