from __future__ import annotations

from collections.abc import Mapping

import pytest

from rwkv_lab.trainvm_worker import (
    CheckpointDisposition,
    CheckpointPublicationRequest,
    CommandKind,
    ControlAssignment,
    ControlDisposition,
    LifecycleDisposition,
    SafePoint,
    WorkerCancellationRequested,
    WorkerCommand,
    WorkerControlError,
    WorkerControlRuntime,
    WorkerResourcesReleasedPause,
)
from rwkv_lab.trainvm_worker.session import wire


class FakeSession:
    def __init__(
        self, *commands: WorkerCommand, resume_on_heartbeat: bool = False
    ) -> None:
        self.commands = list(commands)
        self.acknowledgements: list[
            tuple[
                WorkerCommand,
                ControlDisposition,
                Mapping[str, bool | int | float | str] | None,
                int,
                tuple[tuple[int, str, str, str, str], ...],
            ]
        ] = []
        self.checkpoint_acknowledgements: list[
            tuple[WorkerCommand, CheckpointDisposition, int, str]
        ] = []
        self.lifecycle_acknowledgements: list[
            tuple[WorkerCommand, LifecycleDisposition, int, str]
        ] = []
        self.resume_on_heartbeat = resume_on_heartbeat
        self.heartbeats: list[tuple[int, str]] = []

    def poll_commands(self, maximum: int | None = None) -> tuple[WorkerCommand, ...]:
        count = len(self.commands) if maximum is None else maximum
        result = tuple(self.commands[:count])
        del self.commands[:count]
        return result

    def acknowledge_controls(
        self,
        command: WorkerCommand,
        disposition: ControlDisposition,
        *,
        effective_values: Mapping[str, bool | int | float | str] | None = None,
        effective_step: int = 0,
        diagnostics: tuple[tuple[int, str, str, str, str], ...] = (),
        wait: bool = True,
    ) -> int:
        assert wait is True
        self.acknowledgements.append(
            (
                command,
                disposition,
                effective_values,
                effective_step,
                diagnostics,
            )
        )
        return len(self.acknowledgements)

    def acknowledge_checkpoint(
        self,
        command: WorkerCommand,
        disposition: CheckpointDisposition,
        *,
        optimizer_step: int = 0,
        artifact_id: str = "",
        diagnostics: tuple[tuple[int, str, str, str, str], ...] = (),
        wait: bool = True,
    ) -> int:
        assert wait is True
        assert diagnostics == ()
        self.checkpoint_acknowledgements.append(
            (command, disposition, optimizer_step, artifact_id)
        )
        return len(self.checkpoint_acknowledgements)

    def acknowledge_lifecycle(
        self,
        command: WorkerCommand,
        disposition: LifecycleDisposition,
        *,
        optimizer_step: int = 0,
        artifact_id: str = "",
        diagnostics: tuple[tuple[int, str, str, str, str], ...] = (),
        wait: bool = True,
    ) -> int:
        assert wait is True
        assert diagnostics == ()
        self.lifecycle_acknowledgements.append(
            (command, disposition, optimizer_step, artifact_id)
        )
        return len(self.lifecycle_acknowledgements)

    def heartbeat(
        self, optimizer_step: int, phase: str, *, wait: bool = False
    ) -> int:
        assert wait is True
        self.heartbeats.append((optimizer_step, phase))
        if self.resume_on_heartbeat:
            self.resume_on_heartbeat = False
            self.commands.append(
                WorkerCommand(12, "resume-12", CommandKind.RESUME)
            )
        return len(self.heartbeats)


def control_command(
    revision: int,
    *,
    expected: int,
    apply_point: int,
    assignments: tuple[ControlAssignment, ...],
    requires_pause: bool = False,
) -> WorkerCommand:
    return WorkerCommand(
        controller_sequence=revision,
        command_id=f"control-{revision}",
        kind=CommandKind.CONTROLS,
        apply_point=apply_point,
        control_revision=revision,
        expected_control_revision=expected,
        requires_pause=requires_pause,
        assignments=assignments,
    )


def test_control_patch_applies_atomically_at_its_safe_point() -> None:
    session = FakeSession(
        control_command(
            4,
            expected=2,
            apply_point=wire.APPLY_POINT_NEXT_OPTIMIZER_STEP,
            assignments=(
                ControlAssignment("learning_rate", 1.0e-5),
                ControlAssignment("eval_every", 20),
            ),
        )
    )
    runtime = WorkerControlRuntime(
        session, {"learning_rate": 2.0e-5, "caption_dropout": 0.1}, 2
    )
    observed = []

    assert runtime.microbatch(8, lambda *_: pytest.fail("wrong safe point")) == ()
    assert session.acknowledgements == []
    applied = runtime.optimizer_step(
        9,
        lambda effective, assignments: observed.append(
            (dict(effective), dict(assignments))
        ),
    )

    assert observed == [
        (
            {
                "learning_rate": 1.0e-5,
                "caption_dropout": 0.1,
                "eval_every": 20,
            },
            {"learning_rate": 1.0e-5, "eval_every": 20},
        )
    ]
    assert len(applied) == 1
    assert applied[0].safe_point is SafePoint.NEXT_OPTIMIZER_STEP
    assert applied[0].effective_step == 9
    assert runtime.effective_revision == 4
    assert dict(runtime.effective_values) == observed[0][0]
    acknowledgement = session.acknowledgements[0]
    assert acknowledgement[1] is ControlDisposition.APPLIED
    assert dict(acknowledgement[2] or {}) == observed[0][1]
    assert acknowledgement[3] == 9
    assert acknowledgement[4] == ()
    assert runtime.checkpoint_state() == {
        "effective_control_revision": 4,
        "effective_controls": observed[0][0],
    }
    runtime.verify_checkpoint_state(runtime.checkpoint_state())
    with pytest.raises(WorkerControlError, match="checkpoint worker-control"):
        runtime.verify_checkpoint_state(
            {"effective_control_revision": 2, "effective_controls": observed[0][0]}
        )


def test_pre_optimizer_step_blocks_mutation_until_durable_eval_gate() -> None:
    session = FakeSession()
    session.step_zero_eval_gate_required = True
    session.step_zero_eval_gate_satisfied = False
    session.attempt_baseline_optimizer_step = 7
    runtime = WorkerControlRuntime(session, {}, 0)
    mutated: list[bool] = []

    with pytest.raises(WorkerControlError, match="must follow"):
        runtime.pre_optimizer_step(7, lambda *_: mutated.append(True))
    with pytest.raises(WorkerControlError, match="optimizer mutation is blocked"):
        runtime.pre_optimizer_step(8, lambda *_: mutated.append(True))
    assert mutated == []

    session.step_zero_eval_gate_satisfied = True
    assert runtime.pre_optimizer_step(8, lambda *_: mutated.append(True)) == ()
    # The control applier is only called when a patch exists. Reaching this
    # line without an exception is the pre-mutation permit.
    assert mutated == []


def test_pending_commands_remain_revision_ordered_across_safe_points() -> None:
    session = FakeSession(
        control_command(
            3,
            expected=0,
            apply_point=wire.APPLY_POINT_NEXT_EVAL,
            assignments=(ControlAssignment("eval_every", 10),),
        ),
        control_command(
            5,
            expected=3,
            apply_point=wire.APPLY_POINT_NEXT_OPTIMIZER_STEP,
            assignments=(ControlAssignment("learning_rate", 4.0e-6),),
        ),
    )
    runtime = WorkerControlRuntime(session, {}, 0)
    applied = []

    assert runtime.optimizer_step(1, lambda *_: pytest.fail("out of order")) == ()
    assert runtime.evaluation(
        2, lambda _effective, assignments: applied.append(dict(assignments))
    )[0].control_revision == 3
    assert runtime.optimizer_step(
        3, lambda _effective, assignments: applied.append(dict(assignments))
    )[0].control_revision == 5

    assert applied == [{"eval_every": 10}, {"learning_rate": 4.0e-6}]
    assert runtime.effective_revision == 5


def test_immediate_patch_applies_at_next_trainer_poll_without_effective_step() -> None:
    session = FakeSession(
        control_command(
            1,
            expected=0,
            apply_point=wire.APPLY_POINT_IMMEDIATE,
            assignments=(ControlAssignment("caption_dropout", 0.25),),
        )
    )
    runtime = WorkerControlRuntime(session, {"caption_dropout": 0.1}, 0)

    (applied,) = runtime.microbatch(12, lambda *_: None)

    assert applied.effective_step == 0
    assert session.acknowledgements[0][3] == 0


def test_stale_restart_and_duplicate_patches_are_acknowledged_without_mutation() -> None:
    session = FakeSession(
        control_command(
            4,
            expected=1,
            apply_point=wire.APPLY_POINT_NEXT_OPTIMIZER_STEP,
            assignments=(ControlAssignment("learning_rate", 1.0e-6),),
        ),
        control_command(
            5,
            expected=2,
            apply_point=wire.APPLY_POINT_RESTART,
            assignments=(ControlAssignment("mixed_precision", "fp16"),),
            requires_pause=True,
        ),
        control_command(
            6,
            expected=2,
            apply_point=wire.APPLY_POINT_NEXT_OPTIMIZER_STEP,
            assignments=(
                ControlAssignment("eval_every", 20),
                ControlAssignment("eval_every", 30),
            ),
        ),
    )
    runtime = WorkerControlRuntime(session, {"learning_rate": 2.0e-6}, 2)

    assert runtime.optimizer_step(7, lambda *_: pytest.fail("must not apply")) == ()

    assert [item[1] for item in session.acknowledgements] == [
        ControlDisposition.REJECTED,
        ControlDisposition.RESTART_REQUIRED,
        ControlDisposition.REJECTED,
    ]
    assert [item[4][0][1] for item in session.acknowledgements] == [
        "control.worker_revision_conflict",
        "control.restart_required",
        "control.duplicate_assignment",
    ]
    assert runtime.effective_revision == 2
    assert dict(runtime.effective_values) == {"learning_rate": 2.0e-6}


def test_adapter_rejection_is_atomic_and_does_not_disclose_exception() -> None:
    session = FakeSession(
        control_command(
            3,
            expected=2,
            apply_point=wire.APPLY_POINT_NEXT_MICROBATCH,
            assignments=(ControlAssignment("caption_dropout", 0.5),),
        )
    )
    runtime = WorkerControlRuntime(session, {"caption_dropout": 0.1}, 2)

    def reject(*_args: object) -> None:
        raise RuntimeError("secret trainer object at /private/path")

    assert runtime.microbatch(4, reject) == ()
    assert session.acknowledgements[0][1] is ControlDisposition.REJECTED
    assert session.acknowledgements[0][4][0][1] == "control.adapter_rejected"
    assert "secret" not in repr(session.acknowledgements)
    assert dict(runtime.effective_values) == {"caption_dropout": 0.1}
    assert runtime.effective_revision == 2


@pytest.mark.parametrize("value", [float("nan"), float("inf"), 1 << 80])
def test_invalid_initial_and_command_scalars_fail_closed(value: object) -> None:
    with pytest.raises(WorkerControlError, match="invalid scalar"):
        WorkerControlRuntime(FakeSession(), {"learning_rate": value}, 0)

    session = FakeSession(
        control_command(
            1,
            expected=0,
            apply_point=wire.APPLY_POINT_NEXT_MICROBATCH,
            assignments=(ControlAssignment("learning_rate", value),),  # type: ignore[arg-type]
        )
    )
    runtime = WorkerControlRuntime(session, {}, 0)
    assert runtime.microbatch(1, lambda *_: pytest.fail("must not apply")) == ()
    assert session.acknowledgements[0][4][0][1] == "control.invalid_assignment"


def test_cancel_command_is_acknowledged_at_safe_point_and_stops_adapter() -> None:
    command = WorkerCommand(1, "cancel-1", CommandKind.CANCEL, reason="stop")
    session = FakeSession(command)
    runtime = WorkerControlRuntime(session, {}, 0)

    with pytest.raises(WorkerCancellationRequested, match="stop"):
        runtime.checkpoint(1, lambda *_: None)
    assert session.lifecycle_acknowledgements == [
        (command, LifecycleDisposition.APPLIED, 0, "")
    ]


def test_retained_pause_waits_for_durable_resume_without_publishing() -> None:
    pause = WorkerCommand(
        11,
        "pause-11",
        CommandKind.PAUSE,
        checkpoint_first=False,
        release_resources=False,
    )
    session = FakeSession(pause, resume_on_heartbeat=True)
    runtime = WorkerControlRuntime(session, {}, 0)
    assert runtime.checkpoint_boundary_requested
    runtime.checkpoint(9, lambda *_: None)

    published = runtime.publish_requested_checkpoint(
        CheckpointPublicationRequest(
            source_directory="/run/checkpoint",
            optimizer_step=9,
            resume_grade="exact",
            state_components=("model", "optimizer"),
        )
    )

    assert published is None
    assert session.heartbeats == [(9, "paused")]
    assert [item[0].kind for item in session.lifecycle_acknowledgements] == [
        CommandKind.PAUSE,
        CommandKind.RESUME,
    ]
    assert all(
        item[1] is LifecycleDisposition.APPLIED
        for item in session.lifecycle_acknowledgements
    )


def test_resource_releasing_pause_publishes_checkpoint_then_retires_worker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pause = WorkerCommand(
        11,
        "pause-11",
        CommandKind.PAUSE,
        checkpoint_first=True,
        release_resources=True,
    )
    session = FakeSession(pause)
    runtime = WorkerControlRuntime(session, {}, 0)

    class Published:
        artifact_id = "pause-checkpoint"

    class Publisher:
        def __init__(self, _session: object, *, output_name: str) -> None:
            assert output_name == "checkpoint"

        def publish(self, *_args: object, **_kwargs: object) -> Published:
            return Published()

    monkeypatch.setattr(
        "rwkv_lab.trainvm_worker.checkpoint.CheckpointPublisher", Publisher
    )
    with pytest.raises(WorkerResourcesReleasedPause, match="released"):
        runtime.publish_requested_checkpoint(
            CheckpointPublicationRequest(
                source_directory="/run/checkpoint",
                optimizer_step=9,
                resume_grade="exact",
                state_components=("model", "optimizer"),
            )
        )

    assert session.heartbeats == []
    assert session.lifecycle_acknowledgements == [
        (pause, LifecycleDisposition.APPLIED, 9, "pause-checkpoint")
    ]


def test_paused_barrier_acknowledges_ordered_controls_before_resume() -> None:
    pause = WorkerCommand(11, "pause-11", CommandKind.PAUSE)
    restart_control = control_command(
        12,
        expected=0,
        apply_point=wire.APPLY_POINT_RESTART,
        assignments=(ControlAssignment("mixed_precision", "fp16"),),
        requires_pause=True,
    )
    live_control = control_command(
        13,
        expected=0,
        apply_point=wire.APPLY_POINT_NEXT_OPTIMIZER_STEP,
        assignments=(ControlAssignment("learning_rate", 1.0e-6),),
    )
    resume = WorkerCommand(14, "resume-14", CommandKind.RESUME)
    session = FakeSession(pause, restart_control, live_control, resume)
    runtime = WorkerControlRuntime(session, {"learning_rate": 2.0e-6}, 0)

    runtime.publish_requested_checkpoint(
        CheckpointPublicationRequest(
            source_directory="/run/checkpoint",
            optimizer_step=9,
            resume_grade="exact",
            state_components=("model", "optimizer"),
        )
    )

    assert [item[1] for item in session.acknowledgements] == [
        ControlDisposition.RESTART_REQUIRED,
        ControlDisposition.REJECTED,
    ]
    assert [item[4][0][1] for item in session.acknowledgements] == [
        "control.restart_required",
        "control.paused_barrier",
    ]
    assert [item[0].kind for item in session.lifecycle_acknowledgements] == [
        CommandKind.PAUSE,
        CommandKind.RESUME,
    ]


def test_cancel_stops_a_retained_paused_worker_without_resume() -> None:
    pause = WorkerCommand(11, "pause-11", CommandKind.PAUSE)
    cancel = WorkerCommand(
        12,
        "cancel-12",
        CommandKind.CANCEL,
        reason="cancel paused run",
        graceful_timeout_seconds=5,
    )
    session = FakeSession(pause, cancel)
    runtime = WorkerControlRuntime(session, {}, 0)

    with pytest.raises(WorkerCancellationRequested, match="cancel paused run"):
        runtime.publish_requested_checkpoint(
            CheckpointPublicationRequest(
                source_directory="/run/checkpoint",
                optimizer_step=9,
                resume_grade="exact",
                state_components=("model", "optimizer"),
            )
        )

    assert [item[0].kind for item in session.lifecycle_acknowledgements] == [
        CommandKind.PAUSE,
        CommandKind.CANCEL,
    ]
    assert all(
        item[1] is LifecycleDisposition.APPLIED
        for item in session.lifecycle_acknowledgements
    )


def test_checkpoint_command_blocks_later_controls_until_immutable_publication(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = WorkerCommand(10, "checkpoint-10", CommandKind.CHECKPOINT)
    later_control = control_command(
        3,
        expected=2,
        apply_point=wire.APPLY_POINT_NEXT_OPTIMIZER_STEP,
        assignments=(ControlAssignment("learning_rate", 1.0e-6),),
    )
    session = FakeSession(checkpoint, later_control)
    runtime = WorkerControlRuntime(session, {"learning_rate": 2.0e-6}, 2)
    assert runtime.optimizer_step(5, lambda *_: pytest.fail("must not overtake")) == ()
    assert runtime.checkpoint_boundary_requested
    assert runtime.checkpoint_requested

    class Published:
        artifact_id = "checkpoint-artifact"

    class Publisher:
        def __init__(self, _session: object, *, output_name: str) -> None:
            assert output_name == "checkpoint"

        def publish(self, *_args: object, **_kwargs: object) -> Published:
            return Published()

    monkeypatch.setattr(
        "rwkv_lab.trainvm_worker.checkpoint.CheckpointPublisher", Publisher
    )
    published = runtime.publish_requested_checkpoint(
        CheckpointPublicationRequest(
            source_directory="/run/checkpoint",
            optimizer_step=5,
            resume_grade="exact",
            state_components=("model", "optimizer"),
        )
    )

    assert published is not None
    assert session.checkpoint_acknowledgements == [
        (checkpoint, CheckpointDisposition.APPLIED, 5, "checkpoint-artifact")
    ]
    assert runtime.optimizer_step(6, lambda *_: None)[0].control_revision == 3


@pytest.mark.parametrize("effective_step", [0, -1, True])
def test_safe_point_step_must_be_positive(effective_step: object) -> None:
    runtime = WorkerControlRuntime(FakeSession(), {}, 0)
    with pytest.raises(WorkerControlError, match="effective step must be positive"):
        runtime.apply(
            SafePoint.NEXT_EVAL,
            effective_step=effective_step,  # type: ignore[arg-type]
            applier=lambda *_: None,
        )
