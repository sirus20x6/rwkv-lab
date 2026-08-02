from __future__ import annotations

import json
import threading
import time
from collections.abc import Iterable

from test_trainvm_worker_documents import bootstrap_document, invocation_document

from rwkv_lab.trainvm_worker import (
    CheckpointDisposition,
    CommandKind,
    ControlDisposition,
    LifecycleDisposition,
    WorkerSession,
    load_worker_bootstrap,
)
from rwkv_lab.trainvm_worker.session import wire


class FakeController:
    def __init__(
        self,
        *,
        send_control: bool = False,
        send_checkpoint: bool = False,
        send_cancel: bool = False,
    ) -> None:
        self.received: list[wire.WorkerToController] = []
        self.send_control = send_control
        self.send_checkpoint = send_checkpoint
        self.send_cancel = send_cancel
        self.control_sent = threading.Event()

    def __call__(
        self, requests: Iterable[wire.WorkerToController]
    ) -> Iterable[wire.ControllerToWorker]:
        iterator = iter(requests)
        hello_message = next(iterator)
        self.received.append(hello_message)
        hello = hello_message.hello
        invocation = json.loads(invocation_document())
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
                canonical_invocation_json=invocation_document(),
                invocation_digest=invocation["invocation_digest"],
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
        acknowledgement.disposition
        == wire.LifecycleAcknowledgement.DISPOSITION_APPLIED
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
