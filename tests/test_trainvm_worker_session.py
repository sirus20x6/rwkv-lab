from __future__ import annotations

import json
import threading
import time
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pytest
import torch
from test_trainvm_worker_documents import bootstrap_document, invocation_document

from rwkv_lab import rwkv_pretrain
from rwkv_lab.rwkv_pretrain import perform_rwkv_optimizer_step
from rwkv_lab.trainvm_adapters.rwkv_scratch import RWKVTextEvalPolicy
from rwkv_lab.trainvm_worker import (
    CheckpointDisposition,
    CheckpointPublisher,
    CommandKind,
    ControlDisposition,
    EvalEvidencePart,
    EvalExample,
    EvalExamplesPublicationRequest,
    EvalExamplesPublisher,
    ExecutionPhase,
    ExecutionPhaseDisposition,
    LifecycleDisposition,
    WorkerControlError,
    WorkerControlRuntime,
    WorkerSession,
    WorkerSessionError,
    controls_from_invocation,
    load_worker_bootstrap,
    observability_from_invocation,
    state_fingerprint,
)
from rwkv_lab.trainvm_worker._canonical import canonical_dumps, sha256_digest
from rwkv_lab.trainvm_worker.session import wire


def resume_authority(optimizer_step: int) -> dict[str, object]:
    return {
        "api_version": "trainvm.resume-checkpoint/v1",
        "checkpoint": {
            "artifact_id": "checkpoint-resume",
            "logical_name": "checkpoint",
            "kind": "checkpoint",
            "schema": "test.checkpoint.v1",
            "uri": "file:///run/checkpoint-resume/manifest.json",
            "size_bytes": 4096,
            "fingerprint_algorithm": "manifest_sha256",
            "fingerprint": "sha256:" + "d" * 64,
            "complete": True,
            "producer_node_id": "train",
            "producer_attempt_id": "attempt-0",
            "parent_artifact_ids": [],
            "published_at_ns": 1234,
        },
        "optimizer_step": optimizer_step,
        "pause_command_id": "pause-1",
        "resume_command_id": "resume-1",
    }


class FakeController:
    def __init__(
        self,
        *,
        send_control: bool = False,
        send_checkpoint: bool = False,
        send_cancel: bool = False,
        execution: object = None,
        invocation: bytes | None = None,
        step_zero_eval_gate_required: bool = False,
        step_zero_eval_gate_satisfied: bool = False,
        reject_eval_examples: bool = False,
        attempt_baseline_optimizer_step: int = 0,
    ) -> None:
        self.received: list[wire.WorkerToController] = []
        self.send_control = send_control
        self.send_checkpoint = send_checkpoint
        self.send_cancel = send_cancel
        self.execution = execution
        self.invocation = invocation
        self.step_zero_eval_gate_required = step_zero_eval_gate_required
        self.step_zero_eval_gate_satisfied = step_zero_eval_gate_satisfied
        self.reject_eval_examples = reject_eval_examples
        self.attempt_baseline_optimizer_step = attempt_baseline_optimizer_step
        self.control_sent = threading.Event()

    def __call__(
        self, requests: Iterable[wire.WorkerToController]
    ) -> Iterable[wire.ControllerToWorker]:
        iterator = iter(requests)
        hello_message = next(iterator)
        self.received.append(hello_message)
        hello = hello_message.hello
        raw_invocation = self.invocation or invocation_document(
            execution=self.execution,
            resume=(
                resume_authority(self.attempt_baseline_optimizer_step)
                if self.attempt_baseline_optimizer_step
                else None
            ),
        )
        invocation = json.loads(raw_invocation)
        phase_requests: list[wire.WorkerExecutionPhaseRequest] = []
        for phase_name, phase_value in (
            ("compile", wire.WorkerExecutionPhaseRequest.PHASE_COMPILE),
            ("warmup", wire.WorkerExecutionPhaseRequest.PHASE_WARMUP),
        ):
            if not isinstance(self.execution, dict) or phase_name not in self.execution:
                continue
            declaration = self.execution[phase_name]
            body = {
                "api_version": "trainvm.worker-execution-phase-request/v1",
                "enabled": declaration["enabled"],
                "invocation_digest": invocation["invocation_digest"],
                "phase": phase_name,
            }
            request = wire.WorkerExecutionPhaseRequest(
                phase=phase_value,
                enabled=declaration["enabled"],
            )
            if "steps" in declaration:
                request.steps = declaration["steps"]
                body["steps"] = declaration["steps"]
            request.request_digest = sha256_digest(canonical_dumps(body))
            phase_requests.append(request)
        yield wire.ControllerToWorker(
            welcome=wire.WorkerWelcome(
                disposition=wire.WorkerWelcome.DISPOSITION_ACCEPTED,
                journal_id="journal-1",
                plan_hash=invocation["plan_hash"],
                plan_revision=invocation["plan_revision"],
                run_id=hello.run_id,
                run_revision=9,
                node_id=hello.node_id,
                attempt_id=hello.attempt_id,
                launch_nonce=hello.launch_nonce,
                concurrency_key=hello.concurrency_key,
                lease_id=hello.lease_id,
                fencing_token=hello.fencing_token,
                dispatch_id=invocation["dispatch_id"],
                component="trainer",
                operation="train",
                acknowledged_worker_sequence=0,
                canonical_invocation_json=raw_invocation,
                invocation_digest=invocation["invocation_digest"],
                execution_phase_requests=phase_requests,
                step_zero_eval_gate_required=self.step_zero_eval_gate_required,
                step_zero_eval_gate_satisfied=self.step_zero_eval_gate_satisfied,
                attempt_baseline_optimizer_step=(
                    self.attempt_baseline_optimizer_step
                ),
            )
        )
        if self.send_control:
            yield wire.ControllerToWorker(
                command=wire.WorkerCommand(
                    controller_sequence=18,
                    command_id="control-1",
                    controls=wire.ControlPatchCommand(
                        expected_control_revision=0,
                        assignments=[
                            wire.ControlAssignment(
                                key="learning_rate",
                                value=wire.ScalarValue(number_value=1e-6),
                            )
                        ],
                        control_revision=8,
                        apply_point=wire.APPLY_POINT_NEXT_OPTIMIZER_STEP,
                    ),
                )
            )
            self.control_sent.set()
        if self.send_checkpoint:
            yield wire.ControllerToWorker(
                command=wire.WorkerCommand(
                    controller_sequence=19,
                    command_id="checkpoint-1",
                    checkpoint=wire.CheckpointCommand(reason="operator snapshot"),
                )
            )
            self.control_sent.set()
        if self.send_cancel:
            yield wire.ControllerToWorker(
                command=wire.WorkerCommand(
                    controller_sequence=20,
                    command_id="cancel-1",
                    cancel=wire.CancelCommand(
                        reason="operator stop",
                        graceful_timeout={"seconds": 7},
                    ),
                )
            )
            self.control_sent.set()
        for message in iterator:
            self.received.append(message)
            selected = message.WhichOneof("message")
            sequence = getattr(message, selected).worker_sequence
            if (
                self.reject_eval_examples
                and selected == "artifact"
                and message.artifact.kind == wire.ARTIFACT_KIND_EVAL_EXAMPLES
            ):
                raise RuntimeError("controller rejected step-zero eval evidence")
            if selected == "event":
                yield wire.ControllerToWorker(
                    receipt=wire.WorkerReceipt(
                        event_id=message.event.event_id,
                        acknowledged_worker_sequence=sequence,
                        run_id=hello.run_id,
                        committed_run_revision=10,
                        observed_state=wire.OBSERVED_STATE_COMPLETED,
                    )
                )
                return
            yield wire.ControllerToWorker(acknowledge_worker_sequence=sequence)


def wait_for_commands(session: WorkerSession) -> tuple:
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        values = session.poll_commands()
        if values:
            return values
        time.sleep(0.001)
    raise AssertionError("controller command was not delivered")


def step_zero_invocation(run_directory: Path) -> bytes:
    document = json.loads(invocation_document())
    document.pop("invocation_digest")
    document["workspace"] = {
        "run_directory": str(run_directory),
        "allowed_read_roots": [str(run_directory)],
        "allowed_write_roots": [str(run_directory)],
    }
    document["observability"] = {
        "heartbeat_seconds": 30,
        "metrics": [
            {
                "name": "perplexity",
                "type": "gauge",
                "unit": "ratio",
                "step_domain": "optimizer_step",
                "aggregation": "last",
            },
            {
                "name": "validation_loss",
                "type": "gauge",
                "unit": "ratio",
                "step_domain": "optimizer_step",
                "aggregation": "last",
            },
        ],
        "retain_raw_metrics_days": 7,
    }
    document["publishes"] = {
        "checkpoint": {
            "logical_name": "checkpoint",
            "declaration": {
                "type": "checkpoint",
                "schema": "rwkv-lab.rwkv-scratch-checkpoint.v1",
                "immutability": "immutable",
                "fingerprint": "manifest_sha256",
            },
        },
        "eval_examples": {
            "logical_name": "eval_examples",
            "declaration": {
                "type": "eval_examples",
                "schema": "rwkv-lab.eval-examples.v1",
                "immutability": "append_only",
                "fingerprint": "manifest_sha256",
            },
        },
    }
    return canonical_dumps(
        {
            **document,
            "invocation_digest": sha256_digest(canonical_dumps(document)),
        }
    )


def test_session_orders_hello_telemetry_and_terminal_receipt() -> None:
    controller = FakeController()
    session = WorkerSession(
        load_worker_bootstrap(bootstrap_document()), connector=controller
    )
    invocation = session.start()
    assert invocation.controls["learning_rate"] == 2e-6
    assert session.heartbeat(2, "training", wait=True) == 1
    assert (
        session.metric(
            "loss",
            0.25,
            unit="ratio",
            step_domain="optimizer_step",
            step=2,
            wait=True,
        )
        == 2
    )
    receipt = session.finish("node.completed", {"ok": True}, optimizer_step=2)
    assert receipt.acknowledged_worker_sequence == 3
    assert [message.WhichOneof("message") for message in controller.received] == [
        "hello",
        "heartbeat",
        "metric",
        "event",
    ]
    assert controller.received[-1].event.canonical_json_payload == b'{"ok":true}'
    session.close()


def test_control_command_is_typed_and_acknowledged_at_safe_point() -> None:
    controller = FakeController(send_control=True)
    session = WorkerSession(
        load_worker_bootstrap(bootstrap_document()), connector=controller
    )
    session.start()
    assert controller.control_sent.wait(2)
    (command,) = wait_for_commands(session)
    assert command.kind is CommandKind.CONTROLS
    assert command.controller_sequence == 18
    assert command.control_revision == 8
    assert command.apply_point == wire.APPLY_POINT_NEXT_OPTIMIZER_STEP
    assert command.assignments[0].key == "learning_rate"
    assert command.assignments[0].value == 1e-6
    session.acknowledge_controls(
        command,
        ControlDisposition.APPLIED,
        effective_values={"learning_rate": 1e-6},
        effective_step=4,
        wait=True,
    )
    session.finish("node.completed", {"ok": True}, optimizer_step=4)
    acknowledgement = controller.received[-2].control_ack
    assert acknowledgement.control_revision == 8
    assert acknowledgement.effective_step == 4
    assert acknowledgement.effective_values[0].key == "learning_rate"
    session.close()


def test_checkpoint_command_is_typed_and_acknowledges_published_artifact() -> None:
    controller = FakeController(send_checkpoint=True)
    session = WorkerSession(
        load_worker_bootstrap(bootstrap_document()), connector=controller
    )
    session.start()
    assert controller.control_sent.wait(2)
    (command,) = wait_for_commands(session)
    assert command.kind is CommandKind.CHECKPOINT
    assert command.controller_sequence == 19
    assert command.reason == "operator snapshot"
    session.acknowledge_checkpoint(
        command,
        CheckpointDisposition.APPLIED,
        optimizer_step=7,
        artifact_id="checkpoint-artifact",
        wait=True,
    )
    session.finish("node.completed", {"ok": True}, optimizer_step=7)
    acknowledgement = controller.received[-2].checkpoint_ack
    assert acknowledgement.optimizer_step == 7
    assert acknowledgement.artifact_id == "checkpoint-artifact"
    session.close()


def test_execution_phase_request_is_bound_and_receipted_with_state_proof() -> None:
    controller = FakeController(
        execution={
            "component": "trainer",
            "operation": "train",
            "compile": {"enabled": True},
            "warmup": {"enabled": True, "steps": 3},
        }
    )
    session = WorkerSession(
        load_worker_bootstrap(bootstrap_document()), connector=controller
    )
    session.start()
    requests = session.execution_phase_requests
    assert [request.phase for request in requests] == [
        ExecutionPhase.COMPILE,
        ExecutionPhase.WARMUP,
    ]
    warmup = requests[1]
    fingerprint = state_fingerprint(
        {"model": "m1", "optimizer": "o1", "rng": 17, "cursor": 0}
    )
    assert (
        session.execution_phase_receipt(
            warmup,
            ExecutionPhaseDisposition.COMPLETED,
            steps_executed=3,
            state_fingerprint_before=fingerprint,
            state_fingerprint_after=fingerprint,
            started_at_ns=10,
            completed_at_ns=20,
            wait=True,
        )
        == 1
    )
    session.finish("node.completed", {"ok": True}, optimizer_step=1)
    phase_receipt = controller.received[-2].phase_receipt
    assert phase_receipt.request_digest == warmup.request_digest
    assert phase_receipt.steps_executed == 3
    assert phase_receipt.concurrency_key == "gpu:0"
    session.close()


def test_cancel_command_is_typed_and_has_a_fenced_lifecycle_ack() -> None:
    controller = FakeController(send_cancel=True)
    session = WorkerSession(
        load_worker_bootstrap(bootstrap_document()), connector=controller
    )
    session.start()
    assert controller.control_sent.wait(2)
    (command,) = wait_for_commands(session)
    assert command.kind is CommandKind.CANCEL
    assert command.controller_sequence == 20
    assert command.reason == "operator stop"
    assert command.graceful_timeout_seconds == 7
    session.acknowledge_lifecycle(
        command,
        LifecycleDisposition.APPLIED,
        wait=True,
    )
    session.finish("node.completed", {"ok": True}, optimizer_step=7)
    acknowledgement = controller.received[-2].lifecycle_ack
    assert acknowledgement.kind == wire.LifecycleAcknowledgement.KIND_CANCEL
    assert (
        acknowledgement.disposition == wire.LifecycleAcknowledgement.DISPOSITION_APPLIED
    )
    session.close()


def test_identical_artifact_republication_reuses_the_durable_receipt() -> None:
    controller = FakeController()
    session = WorkerSession(
        load_worker_bootstrap(bootstrap_document()), connector=controller
    )
    session.start()
    values = {
        "artifact_id": "checkpoint-repeat",
        "logical_name": "checkpoint",
        "kind": wire.ARTIFACT_KIND_CHECKPOINT,
        "schema": "test.checkpoint.v1",
        "uri": "file:///run/checkpoint-repeat",
        "size_bytes": 12,
        "fingerprint_algorithm": "manifest_sha256",
        "fingerprint": "a" * 64,
        "optimizer_step": 1,
    }
    assert session.artifact(**values) == 1
    assert session.artifact(**values) == 1
    session.finish("node.completed", {"ok": True}, optimizer_step=1)
    assert [message.WhichOneof("message") for message in controller.received] == [
        "hello",
        "artifact",
        "event",
    ]
    session.close()


def test_checkpoint_artifact_requires_authoritative_optimizer_step() -> None:
    controller = FakeController()
    session = WorkerSession(
        load_worker_bootstrap(bootstrap_document()), connector=controller
    )
    session.start()
    values = {
        "artifact_id": "checkpoint-unstepped",
        "logical_name": "checkpoint",
        "kind": wire.ARTIFACT_KIND_CHECKPOINT,
        "schema": "test.checkpoint.v1",
        "uri": "file:///run/checkpoint-unstepped",
        "size_bytes": 12,
        "fingerprint_algorithm": "manifest_sha256",
        "fingerprint": "a" * 64,
    }
    with pytest.raises(WorkerSessionError, match="require an optimizer step"):
        session.artifact(**values)
    session.finish("node.completed", {"ok": True}, optimizer_step=0)
    session.close()


def test_real_rwkv_main_publishes_complete_step_zero_before_first_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    controller = FakeController(
        invocation=step_zero_invocation(tmp_path),
        step_zero_eval_gate_required=True,
    )
    session = WorkerSession(
        load_worker_bootstrap(bootstrap_document()), connector=controller
    )
    invocation = session.start()
    controls = controls_from_invocation(session, invocation)
    observability = observability_from_invocation(session, invocation)
    optimizer_events: list[str] = []
    real_build_optimizer = rwkv_pretrain.build_optimizer

    class RecordingOptimizer:
        def __init__(self, optimizer) -> None:
            self._optimizer = optimizer

        def __getattr__(self, name):
            return getattr(self._optimizer, name)

        def step(self, *args, **kwargs):
            message_types = [
                message.WhichOneof("message") for message in controller.received
            ]
            assert message_types[:5] == [
                "hello",
                "metric",
                "metric",
                "artifact",
                "artifact",
            ]
            assert [
                (message.metric.name, message.metric.step)
                for message in controller.received[1:3]
            ] == [("validation_loss", 0), ("perplexity", 0)]
            checkpoint_artifact = controller.received[3].artifact
            examples_artifact = controller.received[4].artifact
            assert checkpoint_artifact.kind == wire.ARTIFACT_KIND_CHECKPOINT
            assert examples_artifact.kind == wire.ARTIFACT_KIND_EVAL_EXAMPLES
            assert examples_artifact.parent_artifact_ids == [
                checkpoint_artifact.artifact_id
            ]
            assert session.step_zero_eval_gate_satisfied
            assert session.acknowledged_worker_sequence >= (
                controller.received[4].artifact.worker_sequence
            )
            optimizer_events.append("optimizer.step")
            return self._optimizer.step(*args, **kwargs)

    def build_recording_optimizer(*args, **kwargs):
        return RecordingOptimizer(real_build_optimizer(*args, **kwargs))

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(rwkv_pretrain, "build_optimizer", build_recording_optimizer)
    tokens = tmp_path / "tokens.uint16"
    np.arange(32, dtype=np.uint16).tofile(tokens)
    final_checkpoint = tmp_path / "checkpoint-final.pt"
    policy = RWKVTextEvalPolicy(
        heldout_tokens={"heldout-1": (3, 4, 5)},
        identity_field="id",
        identities_digest="sha256:" + "1" * 64,
        selector_digest="sha256:" + "2" * 64,
        evaluator_component_digest="sha256:" + "3" * 64,
        metric_names=("perplexity", "validation_loss"),
        generation_policy_digest="sha256:" + "4" * 64,
    )

    result = rwkv_pretrain.main(
        [
            "--data",
            str(tokens),
            "--out",
            str(tmp_path),
            "--save",
            str(final_checkpoint),
            "--steps",
            "1",
            "--d-model",
            "8",
            "--n-layers",
            "1",
            "--head-size",
            "8",
            "--seq-len",
            "2",
            "--batch",
            "1",
            "--val-windows",
            "1",
            "--eval-every",
            "50",
            "--log-every",
            "1",
            "--warmup",
            "0",
            "--gpu-data",
            "off",
            "--no-cpu-prefetch",
        ],
        worker_observability=observability,
        worker_controls=controls,
        worker_eval_examples=policy,
    )

    assert result["step"] == 1
    assert optimizer_events == ["optimizer.step"]
    step_zero_state = torch.load(
        tmp_path / "checkpoint-step-zero" / "state.pt",
        map_location="cpu",
        weights_only=False,
    )
    assert step_zero_state["step"] == 0
    assert step_zero_state["arch"]["d_model"] == 8
    assert step_zero_state["arch"]["n_layers"] == 1
    assert step_zero_state["arch"]["head_size"] == 8
    assert [
        controller.received[index].artifact.optimizer_step for index in (3, 4)
    ] == [0, 0]
    assert all(
        controller.received[index].artifact.producer_attempt_id == "attempt-1"
        for index in (3, 4)
    )
    session.finish("node.completed", {"ok": True}, optimizer_step=1)
    session.close()


def test_rwkv_step_zero_publication_durably_opens_gate_before_optimizer_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw_invocation = step_zero_invocation(tmp_path)
    controller = FakeController(
        invocation=raw_invocation,
        step_zero_eval_gate_required=True,
    )
    session = WorkerSession(
        load_worker_bootstrap(bootstrap_document()), connector=controller
    )
    invocation = session.start()
    controls = controls_from_invocation(session, invocation)
    observability = observability_from_invocation(session, invocation)

    optimizer_events: list[str] = []

    class Optimizer:
        def step(self) -> None:
            optimizer_events.append("optimizer.step")

    with pytest.raises(WorkerControlError, match="optimizer mutation is blocked"):
        perform_rwkv_optimizer_step(
            Optimizer(),
            controls,
            next_step=1,
            control_applier=lambda *_: None,
        )
    assert optimizer_events == []

    validation_sequence = observability.publish_if_declared(
        "validation_loss", 4.0, step=0
    )
    perplexity_sequence = observability.publish_if_declared(
        "perplexity", 54.598, step=0
    )
    assert validation_sequence == 1
    assert perplexity_sequence == 2

    checkpoint_source = tmp_path / "checkpoint-step-zero"
    checkpoint_source.mkdir()
    (checkpoint_source / "state.pt").write_bytes(b"step-zero-state")
    checkpoint = CheckpointPublisher(session).publish(
        checkpoint_source,
        optimizer_step=0,
        resume_grade="terminal_checkpoint",
        state_components=("model", "optimizer", "rng_torch"),
    )
    evaluation = EvalExamplesPublisher(session, output_name="eval_examples").publish(
        EvalExamplesPublicationRequest(
            output_name="eval_examples",
            optimizer_step=0,
            series_id="rwkv-token-predictions",
            identity_field="id",
            identities_digest="sha256:" + "1" * 64,
            selector_digest="sha256:" + "2" * 64,
            evaluator_component_digest="sha256:" + "3" * 64,
            metric_names=("perplexity", "validation_loss"),
            checkpoint_artifact_id=checkpoint.artifact_id,
            checkpoint_manifest_digest=checkpoint.manifest_sha256,
            policy_digest="sha256:" + "4" * 64,
            examples=(
                EvalExample(
                    example_id="rwkv-heldout",
                    heldout_item_id="heldout-1",
                    heldout_item_digest="sha256:" + "5" * 64,
                    input=(EvalEvidencePart(kind="text", text="token_ids: 3 4"),),
                    target=(EvalEvidencePart(kind="text", text="token_id: 5"),),
                    prediction=(EvalEvidencePart(kind="text", text="token_id: 7"),),
                ),
            ),
            parent_artifact_ids=(checkpoint.artifact_id,),
        )
    )

    # Evaluation publication returns only after the cumulative controller ACK;
    # therefore both earlier scalar samples and both same-attempt artifacts are
    # durable when the local gate transitions.
    assert session.acknowledged_worker_sequence == evaluation.worker_sequence
    assert session.acknowledged_worker_sequence >= perplexity_sequence
    assert session.step_zero_eval_gate_satisfied
    assert [message.WhichOneof("message") for message in controller.received] == [
        "hello",
        "metric",
        "metric",
        "artifact",
        "artifact",
    ]
    assert [message.metric.step for message in controller.received[1:3]] == [0, 0]
    assert all(
        message.artifact.producer_attempt_id == "attempt-1"
        for message in controller.received[3:5]
    )

    original_pre_step = controls.pre_optimizer_step

    def observed_pre_step(next_step, applier):
        optimizer_events.append("pre_optimizer_step")
        return original_pre_step(next_step, applier)

    monkeypatch.setattr(controls, "pre_optimizer_step", observed_pre_step)
    perform_rwkv_optimizer_step(
        Optimizer(), controls, next_step=1, control_applier=lambda *_: None
    )
    assert optimizer_events == ["pre_optimizer_step", "optimizer.step"]
    session.finish("node.completed", {"ok": True}, optimizer_step=1)
    session.close()

    # A fresh session with an unsatisfied Welcome cannot inherit the local gate
    # transition. Controller-side attempt isolation is covered by the native
    # durable-gate authority regression.
    restarted_controller = FakeController(
        invocation=raw_invocation,
        step_zero_eval_gate_required=True,
    )
    restarted = WorkerSession(
        load_worker_bootstrap(bootstrap_document()), connector=restarted_controller
    )
    restarted_invocation = restarted.start()
    restarted_controls = controls_from_invocation(restarted, restarted_invocation)
    restarted_optimizer = Optimizer()
    with pytest.raises(WorkerControlError, match="optimizer mutation is blocked"):
        perform_rwkv_optimizer_step(
            restarted_optimizer,
            restarted_controls,
            next_step=1,
            control_applier=lambda *_: None,
        )
    assert optimizer_events == ["pre_optimizer_step", "optimizer.step"]
    restarted.finish("node.completed", {"ok": True}, optimizer_step=0)
    restarted.close()


def test_rwkv_step_zero_publication_failure_cannot_mutate_optimizer(
    tmp_path: Path,
) -> None:
    controller = FakeController(
        invocation=step_zero_invocation(tmp_path),
        step_zero_eval_gate_required=True,
        reject_eval_examples=True,
    )
    session = WorkerSession(
        load_worker_bootstrap(bootstrap_document()), connector=controller
    )
    invocation = session.start()
    controls = controls_from_invocation(session, invocation)
    session.metric(
        "validation_loss",
        4.0,
        unit="ratio",
        step_domain="optimizer_step",
        step=0,
        wait=False,
    )
    with pytest.raises(WorkerSessionError, match="WorkerControl stream failed"):
        session.artifact(
            artifact_id="eval-examples-failed",
            logical_name="eval_examples",
            kind=wire.ARTIFACT_KIND_EVAL_EXAMPLES,
            schema="rwkv-lab.eval-examples.v1",
            uri="file:///run/eval-examples-failed/manifest.json",
            size_bytes=2,
            fingerprint_algorithm="manifest_sha256",
            fingerprint="sha256:" + "a" * 64,
            parent_artifact_ids=("checkpoint-step-zero",),
            optimizer_step=0,
            canonical_manifest_json=b"{}",
            wait=True,
        )
    assert not session.step_zero_eval_gate_satisfied

    optimizer_events: list[str] = []

    class Optimizer:
        def step(self) -> None:
            optimizer_events.append("optimizer.step")

    with pytest.raises(WorkerControlError, match="optimizer mutation is blocked"):
        perform_rwkv_optimizer_step(
            Optimizer(), controls, next_step=1, control_applier=lambda *_: None
        )
    assert optimizer_events == []
    session.close()


def test_resumed_eval_examples_ack_latches_exact_attempt_baseline() -> None:
    controller = FakeController(
        step_zero_eval_gate_required=True,
        attempt_baseline_optimizer_step=7,
    )
    session = WorkerSession(
        load_worker_bootstrap(bootstrap_document()), connector=controller
    )
    session.start()
    assert session.attempt_baseline_optimizer_step == 7
    assert session.step_zero_eval_gate_required
    assert not session.step_zero_eval_gate_satisfied
    controls = WorkerControlRuntime(session, {}, 0)
    with pytest.raises(WorkerControlError, match="optimizer mutation is blocked"):
        controls.pre_optimizer_step(8, lambda *_: None)

    def publish(artifact_id: str, optimizer_step: int) -> int:
        return session.artifact(
            artifact_id=artifact_id,
            logical_name="eval_examples",
            kind=wire.ARTIFACT_KIND_EVAL_EXAMPLES,
            schema="rwkv-lab.eval-examples.v1",
            uri=f"file:///run/{artifact_id}",
            size_bytes=2,
            fingerprint_algorithm="manifest_sha256",
            fingerprint="a" * 64,
            optimizer_step=optimizer_step,
            canonical_manifest_json=b"{}",
            wait=True,
        )

    assert publish("wrong-baseline", 0) == 1
    assert not session.step_zero_eval_gate_satisfied
    assert publish("resumed-baseline", 7) == 2
    assert session.step_zero_eval_gate_satisfied
    assert controls.pre_optimizer_step(8, lambda *_: None) == ()
    session.finish("node.completed", {"ok": True}, optimizer_step=7)
    session.close()
