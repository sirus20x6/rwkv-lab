#!/usr/bin/env python3
"""Run the native worker-evidence suite over a report this host measured.

The C++ suite proves the admission rules against reports it writes itself.
That is enough to prove the rules, and not enough to prove the worker probe
produces something the rules accept: two sides of one contract, each tested
against its own idea of the other, is how a wire format drifts.

So this driver runs the real probe from src/rwkv_lab/trainvm_worker, on this
host, through the real bootstrap decoder, and hands the resulting bytes to the
same suite. The portable (CPU) path is used deliberately: it needs no
accelerator, no driver and no CUDA stack, so it asserts the same thing on a
hosted runner as on the training host.

The runtime closure fingerprint is supplied rather than measured, here and in
production both. It is whatever the pre-import guard verified for the
deployment; re-deriving it in the probe would be a second answer to a question
the guard already answered under stricter conditions. The native fixture binds
its sealed launch to whatever the report carries, so the value below is a
binding input, not an assertion about this host.
"""

from __future__ import annotations

import json
import pathlib
import subprocess
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "src"))

from rwkv_lab.trainvm_worker._canonical import canonical_dumps, sha256_digest
from rwkv_lab.trainvm_worker.bootstrap import load_worker_bootstrap
from rwkv_lab.trainvm_worker.runtime_evidence import (
    REPORT_API_VERSION,
    measure_worker_runtime_evidence,
    worker_runtime_evidence_bytes,
)

# Matches the identity the native fixture seals; the probe echoes it from the
# bootstrap rather than inventing it, which is the property being carried over.
VERIFIED_CLOSURE_DIGEST = "sha256:" + "c" * 64
AUTHORITY_DERIVED_FIELDS = (
    "boot_id",
    "host_id",
    "inventory_receipt_digest",
    "launch_spec_digest",
    "namespace_digest",
    "receipt_digest",
    "receipt_name",
    "resource_binding_digest",
)


def bootstrap_document() -> bytes:
    body = {
        "adapter": "rwkv-lab.test-trainer",
        "adapter_version": "1.0.0",
        "api_version": "trainvm.worker-bootstrap/v1",
        "attempt_id": "train@1",
        "capabilities": ["artifact.manifest.v1"],
        "code_fingerprint": "sha256:" + "1" * 64,
        "concurrency_key": "gpu:0",
        "controller_target": "unix:/run/trainvm/controller.sock",
        "fencing_token": 9,
        "last_acked_controller_sequence": 0,
        "launch_nonce": "nonce-1",
        "lease_id": "lease-1",
        "node_id": "train",
        "run_id": "run-1",
    }
    document = dict(body)
    document["bootstrap_digest"] = sha256_digest(canonical_dumps(body))
    return canonical_dumps(document)


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: verify_worker_runtime_evidence.py <native-suite>")
        return 2
    bootstrap = load_worker_bootstrap(bootstrap_document())
    report = measure_worker_runtime_evidence(
        bootstrap,
        runtime_closure_fingerprint=VERIFIED_CLOSURE_DIGEST,
        selected_devices=(),
        placement_specific=False,
    )
    raw = worker_runtime_evidence_bytes(report)
    decoded = json.loads(raw)

    problems: list[str] = []
    if decoded.get("api_version") != REPORT_API_VERSION:
        problems.append(f"report api_version must be {REPORT_API_VERSION!r}")
    if decoded.get("compute_device_vendor") != "cpu":
        problems.append("the portable probe must report the cpu vendor")
    for field in AUTHORITY_DERIVED_FIELDS:
        if field in decoded:
            problems.append(
                f"the worker probe must not author the authority-derived "
                f"field {field!r}"
            )
    for field in ("compute_device_uuid", "compute_device_pci_address"):
        if field in decoded:
            problems.append(
                f"a portable probe must not carry placement identity: {field}"
            )
    for field, expected in (
        ("run_id", bootstrap.run_id),
        ("node_id", bootstrap.node_id),
        ("attempt_id", bootstrap.attempt_id),
        ("launch_nonce", bootstrap.launch_nonce),
        ("concurrency_key", bootstrap.concurrency_key),
        ("lease_id", bootstrap.lease_id),
        ("fencing_token", bootstrap.fencing_token),
    ):
        if decoded.get(field) != expected:
            problems.append(f"report {field} must echo the sealed bootstrap")
    if raw != canonical_dumps(decoded):
        problems.append("report bytes are not in canonical wire form")
    if problems:
        for problem in problems:
            print(f"  - {problem}")
        print("verify_worker_runtime_evidence: FAILED")
        return 1

    with tempfile.TemporaryDirectory() as directory:
        path = pathlib.Path(directory) / "worker-runtime-evidence.json"
        path.write_bytes(raw)
        completed = subprocess.run([sys.argv[1], str(path)], check=False)
    if completed.returncode != 0:
        print("verify_worker_runtime_evidence: FAILED")
        return completed.returncode
    print(
        "measured host abi "
        f"{decoded['host_abi_digest']}, compatibility "
        f"{decoded['compute_compatibility_digest']}, "
        f"{len(decoded['runtime_versions'])} runtime versions"
    )
    print("verify_worker_runtime_evidence: PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
