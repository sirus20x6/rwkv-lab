"""Ordering, not presence, for the HF multimodal SFT engine's optimizer boundary.

``tests/test_step_zero_interception_enumeration.py`` can only see that this
trainer *names* ``pre_optimizer_step`` somewhere: it is a static reader. That is
the weakest useful claim, and it is satisfied by a call placed after the
mutation it was meant to guard. These tests drive the real engine over the same
doubles ``test_hf_multimodal_sft_engine.py`` uses and assert the sequence.

The instrument is the one ``tests/test_rwkv_mutation_sentinel.py`` established:
a process-global ``register_optimizer_step_pre_hook`` appends to the *same*
list the fake controller appends its boundary crossings to, so what is asserted
is an interleaving rather than two independent counts. A boundary moved below
``stack.optimizer.step()``, deleted, or bypassed by a second optimizer shows up
here as a reordered or missing entry.

Every test here inherits one further guarantee from the shared harness rather
than restating it: the fake controller refuses a crossing whose attempt-baseline
eval-examples evidence is not durable. Deleting the engine's publication turns
six of these tests red, so a dedicated "evidence precedes the first mutation"
case was written, found to pass with the launch publication removed, and
dropped. What is asserted below is the part these tests can genuinely prove on
their own: the ordering, the fail-closed bypass, and the resume keying.
"""

from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
from torch.optim.optimizer import register_optimizer_step_pre_hook

from test_hf_multimodal_sft_engine import _engine_harness

from rwkv_lab.trainvm_adapters.hf_multimodal_sft import (
    HFMultimodalSFTError,
    run_hf_multimodal_sft,
)
from rwkv_lab.training_runtime.evaluation_schedules import (
    EvaluationScheduleConfiguration,
)
from rwkv_lab.trainvm_worker.mutation_sentinel import MutationSentinelError

FINAL_EVALUATION_OBSERVABILITY = {
    "metrics": ({"name": "eval.test_loss", "step_domain": "optimizer_step"},)
}

# A checkpoint at every step, so an interrupted run leaves a genuinely
# mid-run checkpoint to resume from rather than only its launch one.
RESUMABLE_CADENCE = EvaluationScheduleConfiguration(
    defer_full_scalar=False, qualitative_every_steps=1
)


@pytest.fixture
def observed_mutations():
    """Every optimizer mutation the process performs, in order."""

    events: list[tuple[str, int]] = []
    handle = register_optimizer_step_pre_hook(
        lambda optimizer, args, kwargs: events.append(("mutation", -1))
    )
    try:
        yield events
    finally:
        handle.remove()


def _recording_controls(harness, events, **keywords):
    """A harness controller whose crossings land in the shared ``events`` list."""

    controls = harness.Controls(**keywords)
    crossed = controls.pre_optimizer_step

    def pre_optimizer_step(step, applier):
        events.append(("boundary", step))
        return crossed(step, applier)

    controls.pre_optimizer_step = pre_optimizer_step
    return controls


def _run(harness, controls, directory, *, resume=None, maximum_steps=2):
    return run_hf_multimodal_sft(
        invocation=SimpleNamespace(
            attempt_id="attempt-1",
            observability=FINAL_EVALUATION_OBSERVABILITY,
        ),
        components=harness.Components(),
        run_directory=directory / "run",
        controls=controls,
        observability=harness.Observability(),
        step_profiler=SimpleNamespace(input_wait=nullcontext, step=lambda _step: None),
        resume_directory=resume,
        device="cpu",
    )


def _interrupted_at_step_two(harness, directory):
    """Run until the crash before step 2 and return its step-1 checkpoint request."""

    controls = harness.Controls(fail_before_step=2)
    with pytest.raises(RuntimeError, match="crash before optimizer step"):
        run_hf_multimodal_sft(
            invocation=SimpleNamespace(
                attempt_id="attempt-1",
                observability=FINAL_EVALUATION_OBSERVABILITY,
            ),
            components=harness.Components(),
            run_directory=directory / "run",
            controls=controls,
            observability=harness.Observability(),
            step_profiler=SimpleNamespace(
                input_wait=nullcontext, step=lambda _step: None
            ),
            resume_directory=None,
            device="cpu",
        )
    trained = [
        request
        for kind, request, _weights in controls.events
        if kind == "checkpoint" and request.optimizer_step == 1
    ]
    assert trained, "the interrupted run published no trained checkpoint"
    return trained[0]


def test_every_mutation_in_the_real_loop_is_preceded_by_its_boundary(
    tmp_path, observed_mutations
) -> None:
    """The load-bearing ordering assertion.

    The crossings and the mutations append to one list, so this is about
    sequence rather than counts. It fails if the boundary is missing, moved
    below ``stack.optimizer.step()``, or bypassed by another update path.
    """

    harness = _engine_harness(tmp_path, maximum_steps=2)
    controls = _recording_controls(harness, observed_mutations)
    step = _run(harness, controls, tmp_path)

    assert step == 2
    assert observed_mutations == [
        ("boundary", 1),
        ("mutation", -1),
        ("boundary", 2),
        ("mutation", -1),
    ]


def test_a_refused_boundary_stops_the_run_before_any_mutation(
    tmp_path, observed_mutations
) -> None:
    """A controller that refuses leaves the sentinel disarmed, so nothing mutates.

    This is what makes the boundary a gate rather than a notification. The
    refusal is the shape the real ``WorkerControlRuntime`` raises when the
    attempt-baseline evidence is not durable.
    """

    harness = _engine_harness(tmp_path, maximum_steps=2)
    controls = harness.Controls()

    def refuse(step, applier):
        raise RuntimeError("optimizer mutation is blocked until durable evidence")

    controls.pre_optimizer_step = refuse
    with pytest.raises(RuntimeError, match="blocked until durable evidence"):
        _run(harness, controls, tmp_path)

    assert observed_mutations == [], "the parameters were mutated past a refused gate"


def test_an_update_that_never_crosses_the_boundary_fails_closed(
    tmp_path, observed_mutations
) -> None:
    """A second optimizer, or a fused path, cannot reach the parameters.

    The sentinel's hook is process-global, so an update the engine did not
    authorize is refused even though it never goes through the loop's own
    crossing. Simulated by stepping an unrelated optimizer from inside a
    controller callback the engine invokes mid-loop.
    """

    harness = _engine_harness(tmp_path, maximum_steps=2)
    controls = harness.Controls()
    smuggled = torch.optim.SGD([torch.nn.Parameter(torch.zeros(2))], lr=0.1)
    evaluation = controls.evaluation

    def evaluation_with_bypass(step, applier):
        result = evaluation(step, applier)
        if step > 0:
            smuggled.step()
        return result

    controls.evaluation = evaluation_with_bypass
    with pytest.raises(MutationSentinelError, match="mandatory pre-optimizer-step"):
        _run(harness, controls, tmp_path)


def test_a_resumed_attempt_owes_its_evidence_at_its_own_baseline(
    tmp_path, observed_mutations
) -> None:
    """Resume must deadlock on neither a literal zero nor an unguarded update.

    A replacement attempt resumes at its checkpoint's step and is gated there.
    It republishes its own baseline evidence at that step -- keying it to a
    literal zero would leave it owing evidence at a step it can never reach
    again -- and every later mutation is still preceded by a crossing.
    """

    harness = _engine_harness(
        tmp_path, schedule_configuration=RESUMABLE_CADENCE, maximum_steps=3
    )
    request = _interrupted_at_step_two(harness, tmp_path)

    observed_mutations.clear()
    resumed = _recording_controls(
        harness, observed_mutations, attempt_baseline_optimizer_step=1
    )
    baseline_steps: list[int] = []
    publish_examples = resumed.publish_evaluation_examples

    def publish_evaluation_examples(publication):
        baseline_steps.append(publication.optimizer_step)
        return publish_examples(publication)

    resumed.publish_evaluation_examples = publish_evaluation_examples
    step = run_hf_multimodal_sft(
        invocation=SimpleNamespace(
            attempt_id="attempt-2",
            observability=FINAL_EVALUATION_OBSERVABILITY,
        ),
        components=harness.Components(),
        run_directory=tmp_path / "run",
        controls=resumed,
        observability=harness.Observability(),
        step_profiler=SimpleNamespace(input_wait=nullcontext, step=lambda _step: None),
        resume_directory=request.source_directory,
        resume_parent_artifact_ids=("checkpoint-1",),
        resume_checkpoint_manifest_digest="sha256:" + "1" * 64,
        device="cpu",
    )

    assert step == 3
    assert baseline_steps == [1], "the replacement attempt did not owe its own baseline"
    assert observed_mutations == [
        ("boundary", 2),
        ("mutation", -1),
        ("boundary", 3),
        ("mutation", -1),
    ]


def test_a_resumed_attempt_whose_baseline_disagrees_refuses_to_train(
    tmp_path, observed_mutations
) -> None:
    """Silence here would become a deadlock at the first crossing instead."""

    harness = _engine_harness(
        tmp_path, schedule_configuration=RESUMABLE_CADENCE, maximum_steps=3
    )
    request = _interrupted_at_step_two(harness, tmp_path)

    observed_mutations.clear()
    # The controller gates this attempt at zero while the engine resumed at
    # one: the evidence it would publish can never satisfy the gate.
    mismatched = _recording_controls(
        harness, observed_mutations, attempt_baseline_optimizer_step=0
    )
    with pytest.raises(HFMultimodalSFTError, match="does not gate"):
        run_hf_multimodal_sft(
            invocation=SimpleNamespace(
                attempt_id="attempt-2",
                observability=FINAL_EVALUATION_OBSERVABILITY,
            ),
            components=harness.Components(),
            run_directory=tmp_path / "run",
            controls=mismatched,
            observability=harness.Observability(),
            step_profiler=SimpleNamespace(
                input_wait=nullcontext, step=lambda _step: None
            ),
            resume_directory=request.source_directory,
            resume_parent_artifact_ids=("checkpoint-1",),
            resume_checkpoint_manifest_digest="sha256:" + "1" * 64,
            device="cpu",
        )
    assert observed_mutations == []


def test_a_reconnect_recovers_gate_satisfaction_without_republishing(
    tmp_path, observed_mutations
) -> None:
    """Durable evidence the controller replayed is not minted a second time.

    Recovering satisfaction this way never licenses skipping the boundary: the
    crossings below are still one per mutation.
    """

    harness = _engine_harness(
        tmp_path, schedule_configuration=RESUMABLE_CADENCE, maximum_steps=3
    )
    request = _interrupted_at_step_two(harness, tmp_path)

    observed_mutations.clear()
    reconnected = _recording_controls(
        harness,
        observed_mutations,
        attempt_baseline_optimizer_step=1,
        step_zero_eval_gate_satisfied=True,
    )
    republished: list[int] = []
    publish_examples = reconnected.publish_evaluation_examples

    def publish_evaluation_examples(publication):
        republished.append(publication.optimizer_step)
        return publish_examples(publication)

    reconnected.publish_evaluation_examples = publish_evaluation_examples
    step = run_hf_multimodal_sft(
        invocation=SimpleNamespace(
            attempt_id="attempt-1",
            observability=FINAL_EVALUATION_OBSERVABILITY,
        ),
        components=harness.Components(),
        run_directory=tmp_path / "run",
        controls=reconnected,
        observability=harness.Observability(),
        step_profiler=SimpleNamespace(input_wait=nullcontext, step=lambda _step: None),
        resume_directory=request.source_directory,
        resume_parent_artifact_ids=("checkpoint-1",),
        resume_checkpoint_manifest_digest="sha256:" + "1" * 64,
        device="cpu",
    )

    assert step == 3
    assert republished == [], "durable baseline evidence was published twice"
    assert observed_mutations == [
        ("boundary", 2),
        ("mutation", -1),
        ("boundary", 3),
        ("mutation", -1),
    ]
