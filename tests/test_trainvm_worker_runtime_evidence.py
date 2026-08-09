"""The worker side of the cache runtime evidence transport.

The native suite owns the admission rules. These tests own the other half: that
the probe measures the host it is running on, echoes the sealed bootstrap
rather than inventing an identity, and -- the property this whole design rests
on -- cannot express an authority-derived field even if someone later wants it
to. They run in the CPU job, on a runner with no accelerator and no driver.
"""

from __future__ import annotations

import json

import pytest

from rwkv_lab.trainvm_worker._canonical import canonical_dumps, sha256_digest
from rwkv_lab.trainvm_worker.bootstrap import load_worker_bootstrap
from rwkv_lab.trainvm_worker.runtime_evidence import (
    REPORT_API_VERSION,
    RuntimeEvidenceError,
    host_abi_identity,
    measure_worker_runtime_evidence,
    runtime_versions,
    worker_runtime_evidence_bytes,
)

CLOSURE = "sha256:" + "c" * 64

# Names the authority derives from host identity, the sealed launch and the
# inventory receipt. A worker that could write any of them could name the
# receipt its own cache lookup reads, which is the whole attack.
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


def _bootstrap(**overrides):
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
    body.update(overrides)
    document = dict(body)
    document["bootstrap_digest"] = sha256_digest(canonical_dumps(body))
    return load_worker_bootstrap(canonical_dumps(document))


def _portable_report(**kwargs):
    return measure_worker_runtime_evidence(
        _bootstrap(), runtime_closure_fingerprint=CLOSURE, **kwargs
    )


def test_portable_report_measures_this_host_and_omits_authority_fields():
    report = _portable_report()
    assert report["api_version"] == REPORT_API_VERSION
    assert report["compute_device_vendor"] == "cpu"
    assert report["driver_version"] == "none"
    for field in AUTHORITY_DERIVED_FIELDS:
        assert field not in report
    # Placement identity is absent, not null: a portable claim that carried it
    # is refused by the authority, so the probe must not produce one.
    assert "compute_device_uuid" not in report
    assert "compute_device_pci_address" not in report
    assert report["runtime_closure_fingerprint"] == CLOSURE
    assert report["host_abi_digest"].startswith("sha256:")
    assert report["compute_compatibility_digest"].startswith("sha256:")


def test_report_echoes_the_sealed_bootstrap_identity():
    bootstrap = _bootstrap(
        attempt_id="train@7", lease_id="lease-9", fencing_token=41
    )
    report = measure_worker_runtime_evidence(
        bootstrap, runtime_closure_fingerprint=CLOSURE
    )
    assert report["run_id"] == bootstrap.run_id
    assert report["node_id"] == bootstrap.node_id
    assert report["attempt_id"] == "train@7"
    assert report["launch_nonce"] == bootstrap.launch_nonce
    assert report["concurrency_key"] == bootstrap.concurrency_key
    assert report["lease_id"] == "lease-9"
    assert report["fencing_token"] == 41


def test_wire_form_is_canonical_and_round_trips():
    raw = worker_runtime_evidence_bytes(_portable_report())
    assert raw == canonical_dumps(json.loads(raw))


def test_runtime_versions_are_canonical_and_present():
    versions = runtime_versions()
    names = [entry["name"] for entry in versions]
    assert names == sorted(set(names))
    assert "python" in names
    assert all(entry["version"] for entry in versions)


def test_host_abi_identity_covers_the_linked_abi_not_the_build_string():
    identity = host_abi_identity()
    assert set(identity) == {
        "libc",
        "machine",
        "python_cache_tag",
        "python_implementation",
        "python_platform",
        "python_version",
        "release",
        "sysname",
    }
    # `uname -v` moves on every distribution rebuild without changing anything
    # a compiled artifact binds to; including it would make every namespace
    # cold for no reason.
    assert "version" not in identity


def test_an_invalid_verified_closure_digest_is_refused():
    with pytest.raises(RuntimeEvidenceError):
        _portable_report_with_closure("not-a-digest")


def _portable_report_with_closure(closure: str):
    return measure_worker_runtime_evidence(
        _bootstrap(), runtime_closure_fingerprint=closure
    )


def test_a_device_the_probe_cannot_describe_is_refused_not_guessed():
    with pytest.raises(RuntimeEvidenceError):
        _portable_report(
            selected_devices=[{"vendor": "acme", "architecture": "x"}]
        )
    with pytest.raises(RuntimeEvidenceError):
        _portable_report(
            selected_devices=[
                {"vendor": "nvidia", "architecture": "sm_120", "pci_address": "a"},
                {"vendor": "amd", "architecture": "gfx", "pci_address": "b"},
            ]
        )
    with pytest.raises(RuntimeEvidenceError):
        _portable_report(
            selected_devices=[{"vendor": "nvidia", "architecture": ""}]
        )


def test_placement_specificity_without_a_device_cannot_be_satisfied():
    # A placement-specific namespace names one exact device. With no device
    # fenced there is nothing to name, and the probe fails rather than emitting
    # a portable claim under a placement-specific request.
    with pytest.raises(RuntimeEvidenceError):
        _portable_report(placement_specific=True)


def test_compatibility_digest_discriminates_beyond_the_named_architecture():
    from rwkv_lab.trainvm_worker.runtime_evidence import _cpu_features

    report = _portable_report()
    # Two hosts can agree on "x86_64" and still not share a compiled kernel.
    # The architecture string in the namespace claim cannot carry that, so the
    # instruction set has to reach the digest that can -- proved by rebuilding
    # the digest with the measured features removed and requiring it to move.
    without_features = sha256_digest(
        canonical_dumps(
            {
                "architecture": report["compute_architecture"],
                "attributes": {"cpu_features": []},
                "driver_version": report["driver_version"],
                "vendor": report["compute_device_vendor"],
            }
        )
    )
    assert _cpu_features(), "this host reports no CPU feature flags"
    assert report["compute_compatibility_digest"] != without_features
