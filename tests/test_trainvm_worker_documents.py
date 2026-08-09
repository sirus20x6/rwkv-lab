from __future__ import annotations

import ctypes
import fcntl
import hashlib
import json
import os
from pathlib import Path

import pytest

from rwkv_lab.trainvm_worker import (
    BootstrapError,
    InvocationError,
    TrainingCompositionError,
    load_worker_bootstrap,
    load_worker_invocation,
    read_worker_bootstrap_fd,
)


def canonical(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode()


def digest(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def memfd_create(name: str) -> int:
    function = ctypes.CDLL(None, use_errno=True).memfd_create
    function.argtypes = (ctypes.c_char_p, ctypes.c_uint)
    function.restype = ctypes.c_int
    descriptor = function(name.encode(), 0x0002)  # MFD_ALLOW_SEALING
    if descriptor < 0:
        raise OSError(ctypes.get_errno(), "memfd_create failed")
    return descriptor


def bootstrap_document() -> bytes:
    body = {
        "adapter": "rwkv-lab.mageflow",
        "adapter_version": "1.0.0",
        "api_version": "trainvm.worker-bootstrap/v1",
        "attempt_id": "attempt-1",
        "capabilities": ["artifact.manifest.v1", "metric.scalar.v1"],
        "code_fingerprint": "sha256:" + "a" * 64,
        "concurrency_key": "gpu:0",
        "controller_target": "unix:/run/user/1000/trainvm.sock",
        "fencing_token": 4,
        "last_acked_controller_sequence": 7,
        "launch_nonce": "launch-1",
        "lease_id": "lease-1",
        "node_id": "train",
        "run_id": "run-1",
    }
    return canonical({**body, "bootstrap_digest": digest(canonical(body))})


def training_composition() -> dict[str, object]:
    root = Path(__file__).resolve().parents[1]
    registry = json.loads(
        (root / "docs/experiment-vm/examples/training-components.v1.json").read_text(
            encoding="utf-8"
        )
    )
    descriptor = next(
        item for item in registry["components"] if item["key"]["name"] == "torch_adamw"
    )
    component = {
        "configuration": {
            "learning_rate": 3e-4,
            "beta1": 0.9,
            "beta2": 0.999,
            "epsilon": 1e-8,
            "weight_decay": 0.01,
            "foreach": True,
            "fused": False,
        },
        "descriptor": descriptor,
        "descriptor_digest": digest(canonical(descriptor)),
    }
    body = {
        "api_version": "trainvm.resolved-training-composition/v1",
        "components": {"optimizer": component},
        "model_family": "rwkv",
        "registry_digest": digest(canonical(registry)),
    }
    return {**body, "composition_digest": digest(canonical(body))}


def invocation_document(
    *, training: object = None, resume: object = None, execution: object = None
) -> bytes:
    body = {
        "adapter": {
            "adapter": "rwkv-lab.mageflow",
            "contract": "rwkv-lab.mageflow.train/v1",
            "operation": "train",
            "runtime": "python_worker",
            "version": "1.0.0",
        },
        "api_version": (
            "trainvm.worker-invocation/v2"
            if resume is not None
            else "trainvm.worker-invocation/v1"
        ),
        "attempt_id": "attempt-1",
        "controls": {"learning_rate": 2e-6},
        "dispatch_id": "dispatch-1",
        "effective_control_revision": 2,
        "execution": execution,
        "host_id": "sha256:" + "b" * 64,
        "inputs": {"caption": "雪"},
        "node_id": "train",
        "observability": {},
        "plan_hash": "c" * 64,
        "plan_revision": 3,
        "publishes": {},
        "resources": {},
        **({"resume": resume} if resume is not None else {}),
        "run_id": "run-1",
        "training": training,
        "workspace": {},
    }
    return canonical({**body, "invocation_digest": digest(canonical(body))})


def test_bootstrap_is_exact_content_addressed_and_reads_sealed_memfd() -> None:
    raw = bootstrap_document()
    value = load_worker_bootstrap(raw)
    assert value.bootstrap_digest == (
        "sha256:55996b8d641668a7c0b989df4f94561e93109fe45f3cdd819e8e10fe763e106b"
    )
    assert value.capabilities == ("artifact.manifest.v1", "metric.scalar.v1")
    descriptor = memfd_create("trainvm-bootstrap")
    try:
        os.write(descriptor, raw)
        fcntl.fcntl(
            descriptor,
            getattr(fcntl, "F_ADD_SEALS", 1033),
            getattr(fcntl, "F_SEAL_WRITE", 0x0008)
            | getattr(fcntl, "F_SEAL_GROW", 0x0004)
            | getattr(fcntl, "F_SEAL_SHRINK", 0x0002)
            | getattr(fcntl, "F_SEAL_SEAL", 0x0001),
        )
        assert read_worker_bootstrap_fd(descriptor) == value
    finally:
        os.close(descriptor)


def test_bootstrap_rejects_tampering_duplicates_and_unsealed_fd() -> None:
    changed = json.loads(bootstrap_document())
    changed["attempt_id"] = "attempt-2"
    with pytest.raises(BootstrapError):
        load_worker_bootstrap(canonical(changed))
    with pytest.raises(BootstrapError):
        load_worker_bootstrap(b'{"a":1,"a":2}')
    descriptor = memfd_create("trainvm-bootstrap")
    try:
        os.write(descriptor, bootstrap_document())
        with pytest.raises(BootstrapError):
            read_worker_bootstrap_fd(descriptor)
    finally:
        os.close(descriptor)


def test_invocation_is_immutable_and_bound_to_welcome_identity() -> None:
    assert json.loads(invocation_document())["invocation_digest"] == (
        "sha256:b0b5333370726a775514778a019a88e08c6d445cc8758e4c60aa26c399686781"
    )
    value = load_worker_invocation(
        invocation_document(),
        expected_digest=digest(
            canonical(
                {
                    key: item
                    for key, item in json.loads(invocation_document()).items()
                    if key != "invocation_digest"
                }
            )
        ),
        expected_run_id="run-1",
        expected_node_id="train",
        expected_attempt_id="attempt-1",
        expected_plan_revision=3,
    )
    assert value.inputs["caption"] == "雪"
    with pytest.raises(TypeError):
        value.controls["learning_rate"] = 1e-3  # type: ignore[index]


def test_invocation_rejects_tampering_noncanonical_json_and_wrong_binding() -> None:
    raw = invocation_document()
    changed = json.loads(raw)
    changed["inputs"]["caption"] = "tampered"
    with pytest.raises(InvocationError):
        load_worker_invocation(canonical(changed))
    with pytest.raises(InvocationError):
        load_worker_invocation(json.dumps(json.loads(raw), indent=2).encode())
    with pytest.raises(InvocationError, match="attempt binding"):
        load_worker_invocation(raw, expected_attempt_id="attempt-2")


def test_invocation_exposes_typed_content_addressed_training_composition() -> None:
    value = load_worker_invocation(invocation_document(training=training_composition()))
    assert value.training is not None
    assert value.training.model_family == "rwkv"
    optimizer = value.training.require("optimizer", category="optimizer")
    assert optimizer.implementation == "rwkv_lab.optimizer.torch_adamw.v1"
    assert optimizer.runtime_envelope()["configuration"]["learning_rate"] == 3e-4
    with pytest.raises(TrainingCompositionError, match="not 'activation'"):
        value.training.require("optimizer", category="activation")


def test_invocation_rejects_forged_nested_training_digests() -> None:
    forged = training_composition()
    forged["composition_digest"] = "sha256:" + "0" * 64
    with pytest.raises(InvocationError):
        load_worker_invocation(invocation_document(training=forged))

    forged = training_composition()
    forged["components"]["optimizer"]["descriptor"]["implementation"] = (
        "rwkv_lab.optimizer.forged.v1"
    )
    body = dict(forged)
    del body["composition_digest"]
    forged["composition_digest"] = digest(canonical(body))
    with pytest.raises(InvocationError):
        load_worker_invocation(invocation_document(training=forged))


def test_v2_invocation_binds_resume_to_prior_attempt() -> None:
    checkpoint = {
        "artifact_id": "checkpoint-1",
        "logical_name": "checkpoint",
        "kind": "checkpoint",
        "schema": "rwkv-lab.mageflow-checkpoint.v1",
        "uri": "file:///run/checkpoint/manifest.json",
        "size_bytes": 4096,
        "fingerprint_algorithm": "manifest_sha256",
        "fingerprint": "sha256:" + "d" * 64,
        "complete": True,
        "producer_node_id": "train",
        "producer_attempt_id": "attempt-0",
        "parent_artifact_ids": ["base-1"],
        "published_at_ns": 1234,
    }
    resume = {
        "api_version": "trainvm.resume-checkpoint/v1",
        "checkpoint": checkpoint,
        "optimizer_step": 12,
        "pause_command_id": "pause-1",
        "resume_command_id": "resume-1",
    }

    invocation = load_worker_invocation(invocation_document(resume=resume))

    assert invocation.resume is not None
    assert invocation.resume["checkpoint"]["producer_attempt_id"] == "attempt-0"
    resume["checkpoint"]["producer_attempt_id"] = "attempt-1"
    with pytest.raises(InvocationError, match="lineage"):
        load_worker_invocation(invocation_document(resume=resume))


def test_worker_accepts_every_category_the_shipped_registry_uses() -> None:
    """The worker's category gate must cover the registry the controller ships.

    rwkv_lab.trainvm_worker.training._CATEGORIES is a literal because deriving
    it would import torch during invocation decoding, before the runtime guard
    has cleared third-party imports. That duplication is only safe if something
    keeps it in step with the vocabulary the controller actually resolves plans
    against. It had drifted: a plan naming activation_memory or
    generation_policy — both implemented by this worker and both registered in
    training-components.v1.json — was rejected as "resolved training component
    identity is invalid", which reads like a corrupt plan rather than a stale
    list.
    """
    from rwkv_lab.trainvm_worker.training import _CATEGORIES

    registry = json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "docs/experiment-vm/examples/training-components.v1.json"
        ).read_text()
    )
    shipped = {
        component["key"]["category"] for component in registry["components"]
    }
    assert shipped <= _CATEGORIES, (
        "the shipped component registry uses categories this worker rejects: "
        f"{sorted(shipped - _CATEGORIES)}"
    )
