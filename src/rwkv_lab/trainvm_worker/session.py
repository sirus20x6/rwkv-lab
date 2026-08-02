from __future__ import annotations

import queue
import threading
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any, Self

try:
    import grpc
    from google.protobuf.timestamp_pb2 import Timestamp

    from trainvm.v1 import trainvm_pb2 as wire
    from trainvm.v1 import trainvm_pb2_grpc as wire_grpc
except ImportError as error:  # pragma: no cover - depends on installation extra
    raise RuntimeError(
        "TrainVM worker sessions require the 'trainvm-worker' project extra"
    ) from error

from ._canonical import canonical_dumps
from .bootstrap import WorkerBootstrap
from .invocation import WorkerInvocation, load_worker_invocation

MAXIMUM_WORKER_MESSAGE_BYTES = 64 * 1024


class WorkerSessionError(RuntimeError):
    pass


class CommandKind(str, Enum):
    PAUSE = "pause"
    RESUME = "resume"
    CHECKPOINT = "checkpoint"
    CANCEL = "cancel"
    CONTROLS = "controls"


class ControlDisposition(str, Enum):
    APPLIED = "applied"
    REJECTED = "rejected"
    RESTART_REQUIRED = "restart_required"


class CheckpointDisposition(str, Enum):
    APPLIED = "applied"
    REJECTED = "rejected"


class LifecycleDisposition(str, Enum):
    APPLIED = "applied"
    REJECTED = "rejected"


@dataclass(frozen=True, slots=True)
class ControlAssignment:
    key: str
    value: bool | int | float | str


@dataclass(frozen=True, slots=True)
class WorkerCommand:
    controller_sequence: int
    command_id: str
    kind: CommandKind
    apply_point: int = wire.APPLY_POINT_UNSPECIFIED
    control_revision: int = 0
    expected_control_revision: int = 0
    requires_pause: bool = False
    assignments: tuple[ControlAssignment, ...] = ()
    checkpoint_first: bool = False
    release_resources: bool = False
    reason: str = ""
    graceful_timeout_seconds: float = 0.0


@dataclass(frozen=True, slots=True)
class WorkerReceipt:
    event_id: str
    acknowledged_worker_sequence: int
    run_id: str
    committed_run_revision: int
    observed_state: int
    next_node_id: str
    next_attempt_id: str


def _timestamp_now() -> Timestamp:
    value = Timestamp()
    value.GetCurrentTime()
    return value


def _scalar(value: bool | int | float | str) -> wire.ScalarValue:  # noqa: PYI041
    if isinstance(value, bool):
        return wire.ScalarValue(boolean_value=value)
    if isinstance(value, int):
        if value < -(1 << 63) or value > (1 << 63) - 1:
            raise WorkerSessionError("integer scalar is outside sint64")
        return wire.ScalarValue(integer_value=value)
    if isinstance(value, float):
        if not (-float("inf") < value < float("inf")):
            raise WorkerSessionError("floating scalar must be finite")
        return wire.ScalarValue(number_value=value)
    if isinstance(value, str):
        return wire.ScalarValue(string_value=value)
    raise WorkerSessionError("unsupported scalar type")


def _from_scalar(value: wire.ScalarValue) -> bool | int | float | str:
    selected = value.WhichOneof("value")
    if selected is None:
        raise WorkerSessionError("control assignment has no scalar value")
    return getattr(value, selected)


def _command(value: wire.WorkerCommand) -> WorkerCommand:
    selected = value.WhichOneof("command")
    if not value.controller_sequence or not value.command_id or selected is None:
        raise WorkerSessionError("controller command identity is incomplete")
    if selected == "pause":
        return WorkerCommand(
            value.controller_sequence,
            value.command_id,
            CommandKind.PAUSE,
            checkpoint_first=value.pause.checkpoint_first,
            release_resources=value.pause.release_resources,
        )
    if selected == "resume":
        return WorkerCommand(
            value.controller_sequence, value.command_id, CommandKind.RESUME
        )
    if selected == "checkpoint":
        return WorkerCommand(
            value.controller_sequence,
            value.command_id,
            CommandKind.CHECKPOINT,
            reason=value.checkpoint.reason,
        )
    if selected == "cancel":
        timeout = value.cancel.graceful_timeout
        return WorkerCommand(
            value.controller_sequence,
            value.command_id,
            CommandKind.CANCEL,
            reason=value.cancel.reason,
            graceful_timeout_seconds=timeout.seconds + timeout.nanos / 1_000_000_000,
        )
    if selected == "controls":
        controls = value.controls
        assignments = tuple(
            ControlAssignment(item.key, _from_scalar(item.value))
            for item in controls.assignments
        )
        if (
            not controls.control_revision
            or controls.apply_point == wire.APPLY_POINT_UNSPECIFIED
        ):
            raise WorkerSessionError("control command revision is inconsistent")
        return WorkerCommand(
            value.controller_sequence,
            value.command_id,
            CommandKind.CONTROLS,
            apply_point=controls.apply_point,
            control_revision=controls.control_revision,
            expected_control_revision=controls.expected_control_revision,
            requires_pause=controls.requires_pause,
            assignments=assignments,
        )
    raise WorkerSessionError("unsupported controller command")


class WorkerSession:
    """One replay-aware WorkerControl stream owned by a trainer process.

    The receive thread performs no tensor work. Trainers poll commands at their
    own safe points, then publish an exact acknowledgement after applying them.
    """

    def __init__(
        self,
        bootstrap: WorkerBootstrap,
        *,
        connector: Callable[
            [Iterable[wire.WorkerToController]], Iterable[wire.ControllerToWorker]
        ]
        | None = None,
    ) -> None:
        self.bootstrap = bootstrap
        self._connector = connector
        self._channel: grpc.Channel | None = None
        self._outgoing: queue.Queue[wire.WorkerToController | None] = queue.Queue()
        self._commands: queue.Queue[WorkerCommand] = queue.Queue()
        self._condition = threading.Condition()
        self._thread: threading.Thread | None = None
        self._welcome: wire.WorkerWelcome | None = None
        self._invocation: WorkerInvocation | None = None
        self._receipt: WorkerReceipt | None = None
        self._error: BaseException | None = None
        self._closed = False
        self._next_worker_sequence = 1
        self._acknowledged_worker_sequence = 0
        self._last_controller_sequence = bootstrap.last_acked_controller_sequence
        self._published_artifacts: dict[str, tuple[tuple[object, ...], int]] = {}

    @property
    def invocation(self) -> WorkerInvocation:
        with self._condition:
            if self._invocation is None:
                raise WorkerSessionError("worker session has no Welcome invocation")
            return self._invocation

    @property
    def completed_before_connect(self) -> bool:
        with self._condition:
            return bool(
                self._welcome
                and self._welcome.disposition
                == wire.WorkerWelcome.DISPOSITION_ALREADY_COMPLETED
            )

    @property
    def acknowledged_worker_sequence(self) -> int:
        with self._condition:
            return self._acknowledged_worker_sequence

    def _requests(self) -> Iterable[wire.WorkerToController]:
        while True:
            item = self._outgoing.get()
            if item is None:
                return
            yield item

    def _hello(self) -> wire.WorkerToController:
        value = self.bootstrap
        return wire.WorkerToController(
            hello=wire.WorkerHello(
                run_id=value.run_id,
                node_id=value.node_id,
                attempt_id=value.attempt_id,
                launch_nonce=value.launch_nonce,
                adapter=value.adapter,
                adapter_version=value.adapter_version,
                code_fingerprint=value.code_fingerprint,
                capabilities=value.capabilities,
                last_acked_controller_sequence=value.last_acked_controller_sequence,
                concurrency_key=value.concurrency_key,
                lease_id=value.lease_id,
                fencing_token=value.fencing_token,
            )
        )

    def start(self, timeout: float = 30.0) -> WorkerInvocation:
        with self._condition:
            if self._thread is not None:
                raise WorkerSessionError("worker session was already started")
            if self._connector is None:
                self._channel = grpc.insecure_channel(self.bootstrap.controller_target)
                self._connector = wire_grpc.WorkerControlStub(self._channel).Connect
            self._outgoing.put(self._hello())
            self._thread = threading.Thread(
                target=self._receive,
                name=f"trainvm-worker-{self.bootstrap.attempt_id}",
                daemon=True,
            )
            self._thread.start()
            deadline = time.monotonic() + timeout
            while self._invocation is None and self._error is None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    self.close()
                    raise WorkerSessionError("timed out waiting for WorkerWelcome")
                self._condition.wait(remaining)
            self._raise_if_failed()
            assert self._invocation is not None
            return self._invocation

    def _receive(self) -> None:
        try:
            assert self._connector is not None
            for message in self._connector(self._requests()):
                selected = message.WhichOneof("message")
                if selected == "welcome":
                    self._accept_welcome(message.welcome)
                elif selected == "acknowledge_worker_sequence":
                    self._accept_ack(message.acknowledge_worker_sequence)
                elif selected == "command":
                    self._accept_command(message.command)
                elif selected == "receipt":
                    self._accept_receipt(message.receipt)
                else:
                    raise WorkerSessionError("controller response variant is missing")
        except BaseException as error:  # noqa: BLE001 - preserve thread failures
            with self._condition:
                if not self._closed:
                    self._error = error
                self._condition.notify_all()
        finally:
            with self._condition:
                if not self._closed and self._receipt is None:
                    self._error = self._error or WorkerSessionError(
                        "WorkerControl stream closed before its terminal receipt"
                    )
                self._condition.notify_all()

    def _accept_welcome(self, welcome: wire.WorkerWelcome) -> None:
        with self._condition:
            if self._welcome is not None:
                raise WorkerSessionError("controller sent more than one Welcome")
            value = self.bootstrap
            if (
                welcome.disposition
                not in {
                    wire.WorkerWelcome.DISPOSITION_ACCEPTED,
                    wire.WorkerWelcome.DISPOSITION_REPLAYED,
                    wire.WorkerWelcome.DISPOSITION_ALREADY_COMPLETED,
                }
                or welcome.run_id != value.run_id
                or welcome.node_id != value.node_id
                or welcome.attempt_id != value.attempt_id
                or welcome.launch_nonce != value.launch_nonce
                or welcome.concurrency_key != value.concurrency_key
                or welcome.lease_id != value.lease_id
                or welcome.fencing_token != value.fencing_token
            ):
                raise WorkerSessionError("WorkerWelcome disagrees with sealed bootstrap")
            invocation = load_worker_invocation(
                welcome.canonical_invocation_json,
                expected_digest=welcome.invocation_digest,
                expected_run_id=welcome.run_id,
                expected_node_id=welcome.node_id,
                expected_attempt_id=welcome.attempt_id,
                expected_plan_revision=welcome.plan_revision,
            )
            if (
                invocation.plan_hash != welcome.plan_hash
                or invocation.dispatch_id != welcome.dispatch_id
                or invocation.adapter["adapter"] != value.adapter
                or invocation.adapter["version"] != value.adapter_version
                or invocation.adapter["operation"] != welcome.operation
            ):
                raise WorkerSessionError("WorkerWelcome invocation binding disagrees")
            self._welcome = wire.WorkerWelcome()
            self._welcome.CopyFrom(welcome)
            self._invocation = invocation
            self._acknowledged_worker_sequence = welcome.acknowledged_worker_sequence
            self._next_worker_sequence = welcome.acknowledged_worker_sequence + 1
            self._condition.notify_all()

    def _accept_ack(self, sequence: int) -> None:
        with self._condition:
            if self._welcome is None or sequence <= self._acknowledged_worker_sequence:
                raise WorkerSessionError("worker acknowledgement is stale or precedes Welcome")
            if sequence >= self._next_worker_sequence:
                raise WorkerSessionError("controller acknowledged an unsent worker sequence")
            self._acknowledged_worker_sequence = sequence
            self._condition.notify_all()

    def _accept_command(self, command: wire.WorkerCommand) -> None:
        decoded = _command(command)
        with self._condition:
            if self._welcome is None:
                raise WorkerSessionError("controller command preceded Welcome")
            if decoded.controller_sequence <= self._last_controller_sequence:
                raise WorkerSessionError("controller command sequence is stale")
            self._last_controller_sequence = decoded.controller_sequence
            self._commands.put(decoded)
            self._condition.notify_all()

    def _accept_receipt(self, receipt: wire.WorkerReceipt) -> None:
        with self._condition:
            if self._welcome is None or self._receipt is not None:
                raise WorkerSessionError("terminal receipt is out of order")
            if (
                receipt.run_id != self.bootstrap.run_id
                or not receipt.event_id
                or receipt.acknowledged_worker_sequence
                < self._acknowledged_worker_sequence
            ):
                raise WorkerSessionError("terminal receipt identity is invalid")
            self._acknowledged_worker_sequence = receipt.acknowledged_worker_sequence
            self._receipt = WorkerReceipt(
                event_id=receipt.event_id,
                acknowledged_worker_sequence=receipt.acknowledged_worker_sequence,
                run_id=receipt.run_id,
                committed_run_revision=receipt.committed_run_revision,
                observed_state=receipt.observed_state,
                next_node_id=receipt.next_node_id,
                next_attempt_id=receipt.next_attempt_id,
            )
            self._condition.notify_all()

    def poll_commands(self, maximum: int | None = None) -> tuple[WorkerCommand, ...]:
        commands: list[WorkerCommand] = []
        while maximum is None or len(commands) < maximum:
            try:
                commands.append(self._commands.get_nowait())
            except queue.Empty:
                break
        self._raise_if_failed()
        return tuple(commands)

    def _allocate_sequence(self) -> int:
        with self._condition:
            self._raise_if_failed()
            if self._welcome is None or self._receipt is not None or self._closed:
                raise WorkerSessionError("worker session is not open for publication")
            sequence = self._next_worker_sequence
            self._next_worker_sequence += 1
            return sequence

    def _send(
        self, message: wire.WorkerToController, sequence: int, wait: bool
    ) -> int:
        if message.ByteSize() > MAXIMUM_WORKER_MESSAGE_BYTES:
            raise WorkerSessionError("worker message exceeds 64 KiB")
        self._outgoing.put(message)
        if wait:
            self.wait_for_ack(sequence)
        return sequence

    def wait_for_ack(self, sequence: int, timeout: float = 30.0) -> None:
        deadline = time.monotonic() + timeout
        with self._condition:
            while self._acknowledged_worker_sequence < sequence:
                self._raise_if_failed()
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise WorkerSessionError(
                        f"timed out waiting for worker sequence {sequence}"
                    )
                self._condition.wait(remaining)

    def heartbeat(
        self, optimizer_step: int, phase: str, *, wait: bool = False
    ) -> int:
        sequence = self._allocate_sequence()
        return self._send(
            wire.WorkerToController(
                heartbeat=wire.WorkerHeartbeat(
                    worker_sequence=sequence,
                    optimizer_step=optimizer_step,
                    phase=phase,
                    observed_at=_timestamp_now(),
                )
            ),
            sequence,
            wait,
        )

    def metric(
        self,
        name: str,
        value: bool | int | float | str,  # noqa: PYI041
        *,
        unit: str,
        step_domain: str,
        step: int,
        sample_weight: float = 1.0,
        labels: Mapping[str, str] | None = None,
        wait: bool = False,
    ) -> int:
        sequence = self._allocate_sequence()
        return self._send(
            wire.WorkerToController(
                metric=wire.MetricSample(
                    name=name,
                    value=_scalar(value),
                    unit=unit,
                    step_domain=step_domain,
                    step=step,
                    sample_weight=sample_weight,
                    labels=dict(labels or {}),
                    observed_at=_timestamp_now(),
                    worker_sequence=sequence,
                )
            ),
            sequence,
            wait,
        )

    def artifact(
        self,
        *,
        artifact_id: str,
        logical_name: str,
        kind: int,
        schema: str,
        uri: str,
        size_bytes: int,
        fingerprint_algorithm: str,
        fingerprint: str,
        parent_artifact_ids: Iterable[str] = (),
        wait: bool = True,
    ) -> int:
        parents = tuple(parent_artifact_ids)
        identity = (
            logical_name,
            kind,
            schema,
            uri,
            size_bytes,
            fingerprint_algorithm,
            fingerprint,
            parents,
        )
        with self._condition:
            previous = self._published_artifacts.get(artifact_id)
        if previous is not None:
            if previous[0] != identity:
                raise WorkerSessionError(
                    "artifact identity was reused with different immutable content"
                )
            return previous[1]
        sequence = self._allocate_sequence()
        result = self._send(
            wire.WorkerToController(
                artifact=wire.ArtifactManifest(
                    artifact_id=artifact_id,
                    logical_name=logical_name,
                    kind=kind,
                    schema=schema,
                    uri=uri,
                    size_bytes=size_bytes,
                    fingerprint_algorithm=fingerprint_algorithm,
                    fingerprint=fingerprint,
                    complete=True,
                    producer_node_id=self.bootstrap.node_id,
                    producer_attempt_id=self.bootstrap.attempt_id,
                    parent_artifact_ids=parents,
                    published_at=_timestamp_now(),
                    worker_sequence=sequence,
                )
            ),
            sequence,
            wait,
        )
        with self._condition:
            self._published_artifacts[artifact_id] = (identity, result)
        return result

    def acknowledge_controls(
        self,
        command: WorkerCommand,
        disposition: ControlDisposition,
        *,
        effective_values: Mapping[str, bool | int | float | str] | None = None,
        effective_step: int = 0,
        diagnostics: Iterable[tuple[int, str, str, str, str]] = (),
        wait: bool = True,
    ) -> int:
        if command.kind is not CommandKind.CONTROLS:
            raise WorkerSessionError("only a control command has a control acknowledgement")
        dispositions = {
            ControlDisposition.APPLIED: wire.ControlPatchAcknowledgement.DISPOSITION_APPLIED,
            ControlDisposition.REJECTED: wire.ControlPatchAcknowledgement.DISPOSITION_REJECTED,
            ControlDisposition.RESTART_REQUIRED: wire.ControlPatchAcknowledgement.DISPOSITION_RESTART_REQUIRED,
        }
        sequence = self._allocate_sequence()
        acknowledgement = wire.ControlPatchAcknowledgement(
            control_revision=command.control_revision,
            disposition=dispositions[disposition],
            apply_point=command.apply_point,
            effective_step=effective_step,
            effective_values=[
                wire.ControlAssignment(key=key, value=_scalar(value))
                for key, value in (effective_values or {}).items()
            ],
            diagnostics=[
                wire.Diagnostic(
                    severity=severity,
                    code=code,
                    document_path=document_path,
                    message=message,
                    help=help_text,
                )
                for severity, code, document_path, message, help_text in diagnostics
            ],
            command_id=command.command_id,
            concurrency_key=self.bootstrap.concurrency_key,
            lease_id=self.bootstrap.lease_id,
            fencing_token=self.bootstrap.fencing_token,
            worker_sequence=sequence,
            acknowledged_at=_timestamp_now(),
        )
        return self._send(
            wire.WorkerToController(control_ack=acknowledgement),
            sequence,
            wait,
        )

    def acknowledge_checkpoint(
        self,
        command: WorkerCommand,
        disposition: CheckpointDisposition,
        *,
        optimizer_step: int = 0,
        artifact_id: str = "",
        diagnostics: Iterable[tuple[int, str, str, str, str]] = (),
        wait: bool = True,
    ) -> int:
        if command.kind is not CommandKind.CHECKPOINT:
            raise WorkerSessionError(
                "only a checkpoint command has a checkpoint acknowledgement"
            )
        applied = disposition is CheckpointDisposition.APPLIED
        if applied != bool(optimizer_step and artifact_id):
            raise WorkerSessionError(
                "checkpoint acknowledgement result is inconsistent"
            )
        dispositions = {
            CheckpointDisposition.APPLIED: wire.CheckpointAcknowledgement.DISPOSITION_APPLIED,
            CheckpointDisposition.REJECTED: wire.CheckpointAcknowledgement.DISPOSITION_REJECTED,
        }
        sequence = self._allocate_sequence()
        acknowledgement = wire.CheckpointAcknowledgement(
            command_id=command.command_id,
            disposition=dispositions[disposition],
            optimizer_step=optimizer_step,
            artifact_id=artifact_id,
            diagnostics=[
                wire.Diagnostic(
                    severity=severity,
                    code=code,
                    document_path=document_path,
                    message=message,
                    help=help_text,
                )
                for severity, code, document_path, message, help_text in diagnostics
            ],
            concurrency_key=self.bootstrap.concurrency_key,
            lease_id=self.bootstrap.lease_id,
            fencing_token=self.bootstrap.fencing_token,
            worker_sequence=sequence,
            acknowledged_at=_timestamp_now(),
        )
        return self._send(
            wire.WorkerToController(checkpoint_ack=acknowledgement),
            sequence,
            wait,
        )

    def acknowledge_lifecycle(
        self,
        command: WorkerCommand,
        disposition: LifecycleDisposition,
        *,
        optimizer_step: int = 0,
        artifact_id: str = "",
        diagnostics: Iterable[tuple[int, str, str, str, str]] = (),
        wait: bool = True,
    ) -> int:
        if command.kind not in {CommandKind.PAUSE, CommandKind.RESUME}:
            raise WorkerSessionError(
                "only pause and resume commands have lifecycle acknowledgements"
            )
        applied = disposition is LifecycleDisposition.APPLIED
        needs_checkpoint = command.kind is CommandKind.PAUSE and command.checkpoint_first
        if (
            (applied and needs_checkpoint) != bool(optimizer_step and artifact_id)
            or (not needs_checkpoint and (optimizer_step or artifact_id))
        ):
            raise WorkerSessionError(
                "lifecycle acknowledgement result is inconsistent"
            )
        kinds = {
            CommandKind.PAUSE: wire.LifecycleAcknowledgement.KIND_PAUSE,
            CommandKind.RESUME: wire.LifecycleAcknowledgement.KIND_RESUME,
        }
        dispositions = {
            LifecycleDisposition.APPLIED: wire.LifecycleAcknowledgement.DISPOSITION_APPLIED,
            LifecycleDisposition.REJECTED: wire.LifecycleAcknowledgement.DISPOSITION_REJECTED,
        }
        sequence = self._allocate_sequence()
        acknowledgement = wire.LifecycleAcknowledgement(
            command_id=command.command_id,
            kind=kinds[command.kind],
            disposition=dispositions[disposition],
            optimizer_step=optimizer_step,
            artifact_id=artifact_id,
            diagnostics=[
                wire.Diagnostic(
                    severity=severity,
                    code=code,
                    document_path=document_path,
                    message=message,
                    help=help_text,
                )
                for severity, code, document_path, message, help_text in diagnostics
            ],
            concurrency_key=self.bootstrap.concurrency_key,
            lease_id=self.bootstrap.lease_id,
            fencing_token=self.bootstrap.fencing_token,
            worker_sequence=sequence,
            acknowledged_at=_timestamp_now(),
        )
        return self._send(
            wire.WorkerToController(lifecycle_ack=acknowledgement),
            sequence,
            wait,
        )

    def finish(
        self,
        event_type: str,
        payload: Mapping[str, Any],
        *,
        optimizer_step: int | None = None,
        timeout: float = 30.0,
    ) -> WorkerReceipt:
        with self._condition:
            if self._welcome is None:
                raise WorkerSessionError("cannot finish before Welcome")
            welcome = self._welcome
        sequence = self._allocate_sequence()
        envelope = wire.EventEnvelope(
            event_id=welcome.dispatch_id + ":result",
            run_id=welcome.run_id,
            run_revision=welcome.run_revision,
            plan_revision=welcome.plan_revision,
            node_id=welcome.node_id,
            attempt_id=welcome.attempt_id,
            worker_sequence=sequence,
            event_type=event_type,
            event_version=1,
            wall_time=_timestamp_now(),
            monotonic_time_ns=time.monotonic_ns(),
            canonical_json_payload=canonical_dumps(dict(payload)),
        )
        if optimizer_step is not None:
            envelope.optimizer_step = optimizer_step
        self._send(wire.WorkerToController(event=envelope), sequence, False)
        deadline = time.monotonic() + timeout
        with self._condition:
            while self._receipt is None:
                self._raise_if_failed()
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise WorkerSessionError("timed out waiting for terminal receipt")
                self._condition.wait(remaining)
            return self._receipt

    def _raise_if_failed(self) -> None:
        if self._error is not None:
            raise WorkerSessionError("WorkerControl stream failed") from self._error

    def close(self) -> None:
        with self._condition:
            if self._closed:
                return
            self._closed = True
            self._outgoing.put(None)
            if self._channel is not None:
                self._channel.close()
            self._condition.notify_all()

    def __enter__(self) -> Self:
        self.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


__all__ = [
    "CheckpointDisposition",
    "CommandKind",
    "ControlAssignment",
    "ControlDisposition",
    "LifecycleDisposition",
    "WorkerCommand",
    "WorkerReceipt",
    "WorkerSession",
    "WorkerSessionError",
    "wire",
]
