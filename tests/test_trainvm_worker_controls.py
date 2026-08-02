from __future__ import annotations

from collections.abc import Mapping

import pytest

from rwkv_lab.trainvm_worker import (
    CommandKind,
    ControlAssignment,
    ControlDisposition,
    SafePoint,
    WorkerCommand,
    WorkerControlError,
    WorkerControlRuntime,
)
from rwkv_lab.trainvm_worker.session import wire


class FakeSession:
    def __init__(self, *commands: WorkerCommand) -> None:
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


def test_lifecycle_commands_fail_closed_until_protocol_supports_them() -> None:
    session = FakeSession(
        WorkerCommand(1, "pause-1", CommandKind.PAUSE, checkpoint_first=True)
    )
    runtime = WorkerControlRuntime(session, {}, 0)

    with pytest.raises(WorkerControlError, match="lifecycle command unsupported"):
        runtime.checkpoint(1, lambda *_: None)


@pytest.mark.parametrize("effective_step", [0, -1, True])
def test_safe_point_step_must_be_positive(effective_step: object) -> None:
    runtime = WorkerControlRuntime(FakeSession(), {}, 0)
    with pytest.raises(WorkerControlError, match="effective step must be positive"):
        runtime.apply(
            SafePoint.NEXT_EVAL,
            effective_step=effective_step,  # type: ignore[arg-type]
            applier=lambda *_: None,
        )
