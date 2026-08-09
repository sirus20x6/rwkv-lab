"""Evaluation cadence is independent of the training trajectory it observes.

The run this suite exists for (`qwen36-caption-distill-lora-r256-v2`, 745 steps,
one shared interval of 250) had no scalar baseline before step 250 and could not
schedule a cheap loss read without also launching a 100-image generation sweep.
These tests pin the properties that make it safe to fix that: the milestones are
separately declared, the baseline is anchored before any optimizer mutation, a
resumed attempt cannot re-label a trained checkpoint as the baseline, and moving
the cadence does not move the optimizer, the data order, or the RNG.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch
from test_hf_multimodal_sft_engine import _engine_harness

from rwkv_lab.training_runtime.evaluation_schedules import (
    EVALUATION_CADENCE_CONTROLS,
    EvaluationKind,
    EvaluationSchedule,
    EvaluationScheduleConfiguration,
    evaluation_schedule_from_resolved_component,
)

# The terminal final-evaluation closure requires every evaluator metric to have
# an exact observable scalar declared on the invocation. `_engine_harness`
# declares one evaluator metric, `test_loss`, so a run that reaches completion
# has to advertise `eval.test_loss` over the optimizer-step domain.
FINAL_EVALUATION_OBSERVABILITY = {
    "metrics": ({"name": "eval.test_loss", "step_domain": "optimizer_step"},)
}

ROOT = Path(__file__).resolve().parents[1]
CATALOG = json.loads(
    (ROOT / "docs/experiment-vm/examples/training-components.v1.json").read_text(
        encoding="utf-8"
    )
)


def _envelope(configuration):
    descriptor = next(
        component
        for component in CATALOG["components"]
        if component["key"]
        == {
            "category": "evaluation_schedule",
            "name": "milestone_cadence",
            "version": "3.0.0",
        }
    )
    encoded = json.dumps(descriptor, separators=(",", ":"), sort_keys=True).encode()
    return {
        "descriptor": descriptor,
        "descriptor_digest": "sha256:" + hashlib.sha256(encoded).hexdigest(),
        "configuration": configuration,
    }


def _resolved(**overrides):
    configuration = {
        "full_step_zero": True,
        "final": True,
        "full_every_steps": 0,
        "probe_every_steps": 0,
        "qualitative_every_steps": 0,
        "full_milestone_steps": [],
        "probe_milestone_steps": [],
        "qualitative_milestone_steps": [],
        "full_milestone_fractions": [],
        "probe_milestone_fractions": [],
        "qualitative_milestone_fractions": [],
        "probe_examples": 0,
        "mutable_cadence": False,
    }
    configuration.update(overrides)
    return _envelope(configuration)


def test_declared_milestones_and_fractions_resolve_against_the_run_length():
    schedule = evaluation_schedule_from_resolved_component(
        _resolved(
            probe_milestone_steps=["50", "100", "200", "300"],
            probe_milestone_fractions=["0.8"],
            probe_examples=64,
            qualitative_milestone_steps=["250", "500"],
        )
    )
    plan = schedule.plan(745)

    # A declared fraction resolves to a concrete step, and an explicit
    # milestone stays exactly where it was written.
    assert plan.steps_for(EvaluationKind.SCALAR_PROBE) == (50, 100, 200, 300, 596)
    # Step zero is a full scalar baseline, never a probe, and the final step is
    # a full read even though 745 is not a multiple of anything declared.
    assert plan.steps_for(EvaluationKind.SCALAR_FULL) == (0, 745)
    assert plan.steps_for(EvaluationKind.QUALITATIVE) == (0, 250, 500, 745)
    assert plan.steps_for(EvaluationKind.FINAL_AUDIT) == (745,)
    assert plan.counts == {
        "scalar_probe": 5,
        "scalar_full": 2,
        "qualitative": 4,
        "final_audit": 1,
    }


def test_loss_and_generation_schedules_are_independent():
    schedule = evaluation_schedule_from_resolved_component(
        _resolved(probe_every_steps=50, probe_examples=32, qualitative_every_steps=250)
    )
    plan = schedule.plan(745)
    probes = set(plan.steps_for(EvaluationKind.SCALAR_PROBE))
    generations = set(plan.steps_for(EvaluationKind.QUALITATIVE))

    # The interior probes exist precisely because they are not coupled to the
    # expensive generation sweep: 12 of the 14 interior loss reads carry no
    # generation at all.
    assert 50 in probes and 50 not in generations
    assert generations == {0, 250, 500, 745}
    assert probes == {step for step in range(50, 745, 50)}
    assert probes - generations == probes - {250, 500}
    assert len(probes - generations) == 12
    # And the coupling does not appear in the other direction either: a run may
    # generate captions at a step that takes no interior scalar read.
    generation_only = evaluation_schedule_from_resolved_component(
        _resolved(qualitative_every_steps=100)
    ).plan(500)
    assert generation_only.steps_for(EvaluationKind.QUALITATIVE) == (
        0,
        100,
        200,
        300,
        400,
        500,
    )
    assert generation_only.steps_for(EvaluationKind.SCALAR_PROBE) == ()
    assert generation_only.steps_for(EvaluationKind.SCALAR_FULL) == (0, 500)


def test_final_full_validation_happens_off_the_interval_grid():
    schedule = evaluation_schedule_from_resolved_component(
        _resolved(full_every_steps=250, qualitative_every_steps=250)
    )
    decision = schedule.for_step(745, final=True, total_steps=745)
    assert 745 % 250 != 0
    assert decision.full_scalar and decision.qualitative and decision.final_audit
    assert schedule.plan(745).steps_for(EvaluationKind.SCALAR_FULL)[-1] == 745


def test_step_zero_is_always_a_full_scalar_baseline_and_never_a_probe():
    schedule = evaluation_schedule_from_resolved_component(
        _resolved(probe_every_steps=1, probe_examples=8)
    )
    decision = schedule.for_step(0, total_steps=745)
    assert decision.full_scalar and decision.launch_gate and decision.qualitative
    assert not decision.probe
    with pytest.raises(ValueError, match="step-zero baseline is mandatory"):
        EvaluationScheduleConfiguration(full_step_zero=False)


def test_a_cadence_patch_cannot_move_which_examples_are_evaluated():
    configuration = EvaluationScheduleConfiguration(
        defer_full_scalar=False,
        probe_every_steps=100,
        probe_examples=64,
        mutable_cadence=True,
    )
    patched = configuration.with_cadence_assignments(
        {"evaluation.probe_every_steps": 25, "evaluation.qualitative_every_steps": 500}
    )
    assert patched.probe_every_steps == 25
    assert patched.qualitative_every_steps == 500
    # The whole point: the example-selecting identity is byte-identical.
    assert patched.example_identity == configuration.example_identity

    for rejected in (
        {"probe_examples": 8},
        {"evaluation.probe_examples": 8},
        {"hyperparameters.learning_rate": 0.1},
    ):
        with pytest.raises(ValueError, match="cadence"):
            configuration.with_cadence_assignments(rejected)
    with pytest.raises(ValueError, match="nonnegative integer"):
        configuration.with_cadence_assignments({"evaluation.probe_every_steps": -1})
    assert set(EVALUATION_CADENCE_CONTROLS) == {
        "evaluation.full_every_steps",
        "evaluation.probe_every_steps",
        "evaluation.qualitative_every_steps",
    }


def test_a_cadence_that_would_plan_absurd_milestone_counts_is_refused():
    """The plan is resolved up front, so an absurd cadence fails there.

    A per-step probe is a legitimate choice on a short run and a mistake on a
    long one. Refusing it while the plan is being built keeps the failure at
    authoring time instead of after the run has already begun paying for it.
    """

    schedule = EvaluationSchedule(
        EvaluationScheduleConfiguration(
            defer_full_scalar=False, probe_every_steps=1, probe_examples=1
        )
    )
    assert len(schedule.plan(1_000).milestones) == 1_001
    with pytest.raises(ValueError, match="more milestones than a run may carry"):
        schedule.plan(1_000_000)


def test_an_immutable_schedule_refuses_every_live_cadence_patch():
    configuration = EvaluationScheduleConfiguration(
        defer_full_scalar=False, full_every_steps=250
    )
    assert configuration.mutable_cadence is False
    with pytest.raises(ValueError, match="no live cadence"):
        configuration.with_cadence_assignments({"evaluation.full_every_steps": 10})
    # An empty patch is not a patch, and must not be treated as one.
    assert configuration.with_cadence_assignments({}) is configuration


def _run(harness_directory, *, schedule_configuration, maximum_steps):
    from contextlib import nullcontext
    from types import SimpleNamespace

    from rwkv_lab.trainvm_adapters.hf_multimodal_sft import run_hf_multimodal_sft

    harness_directory.mkdir(parents=True, exist_ok=True)
    harness = _engine_harness(
        harness_directory,
        schedule_configuration=schedule_configuration,
        maximum_steps=maximum_steps,
    )
    controls = harness.Controls()
    observability = harness.Observability()
    step = run_hf_multimodal_sft(
        invocation=SimpleNamespace(
            attempt_id="attempt-1",
            observability=FINAL_EVALUATION_OBSERVABILITY,
        ),
        components=harness.Components(),
        run_directory=harness_directory / "run",
        controls=controls,
        observability=observability,
        step_profiler=SimpleNamespace(input_wait=nullcontext, step=lambda _step: None),
        resume_directory=None,
        device="cpu",
    )
    assert step == maximum_steps
    return harness, controls, observability


def test_cadence_changes_do_not_perturb_the_optimizer_data_or_rng_trajectory(tmp_path):
    """The claim the whole card rests on, checked rather than asserted.

    Two runs of the same engine over the same doubles, differing only in the
    declared evaluation cadence. Every post-update parameter tensor, the torch
    and Python RNG states sampled at each optimizer step, and the exact order of
    consumed training samples must agree bit for bit. Anything that leaked from
    evaluation into training -- an eval batch drawing from the training sampler,
    a generation call advancing the global RNG, a dropout mask consumed by a
    scalar read -- shows up here as a divergence at the first step after the
    cadences differ.
    """

    sparse = EvaluationScheduleConfiguration(defer_full_scalar=False)
    dense = EvaluationScheduleConfiguration(
        defer_full_scalar=False,
        probe_every_steps=1,
        probe_examples=1,
        qualitative_every_steps=2,
        full_every_steps=3,
    )

    sparse_harness, sparse_controls, sparse_observability = _run(
        tmp_path / "sparse", schedule_configuration=sparse, maximum_steps=6
    )
    dense_harness, dense_controls, dense_observability = _run(
        tmp_path / "dense", schedule_configuration=dense, maximum_steps=6
    )

    def evaluated_steps(observability):
        return sorted(
            {
                step
                for name, _value, step in observability.metrics
                if name in {"eval.loss", "eval.probe_loss"}
            }
        )

    # The test proves nothing unless the two cadences really did differ.
    assert evaluated_steps(sparse_observability) == [0, 6]
    assert evaluated_steps(dense_observability) == [0, 1, 2, 3, 4, 5, 6]
    sparse_galleries = [
        request.step
        for kind, request, _weights in sparse_controls.events
        if kind == "gallery"
    ]
    dense_galleries = [
        request.step
        for kind, request, _weights in dense_controls.events
        if kind == "gallery"
    ]
    assert sparse_galleries != dense_galleries

    assert len(sparse_controls.trajectory) == len(dense_controls.trajectory) == 6
    for left, right in zip(
        sparse_controls.trajectory, dense_controls.trajectory, strict=True
    ):
        assert left[0] == right[0], "optimizer steps diverged"
        assert torch.equal(left[1], right[1]), f"head weight diverged at step {left[0]}"
        assert torch.equal(left[2], right[2]), (
            f"embedding weight diverged at step {left[0]}"
        )
        assert torch.equal(left[3], right[3]), f"torch RNG diverged at step {left[0]}"
        assert left[4] == right[4], f"python RNG diverged at step {left[0]}"

    assert sparse_harness.consumed["train"] == dense_harness.consumed["train"]
    assert sparse_harness.consumed["train"], "the harness consumed no training samples"
    # The evaluated subset is identical too: only how often it was read moved.
    assert set(sparse_harness.consumed["validation"]) == set(
        dense_harness.consumed["validation"]
    )
    assert len(dense_harness.consumed["validation"]) > len(
        sparse_harness.consumed["validation"]
    )


def test_resume_cannot_relabel_a_trained_checkpoint_as_the_baseline(tmp_path):
    """A replacement attempt inherits its baseline; it does not mint a new one."""

    from contextlib import nullcontext
    from types import SimpleNamespace

    from rwkv_lab.trainvm_adapters.hf_multimodal_sft import run_hf_multimodal_sft

    configuration = EvaluationScheduleConfiguration(
        defer_full_scalar=False, qualitative_every_steps=1
    )
    harness = _engine_harness(
        tmp_path, schedule_configuration=configuration, maximum_steps=2
    )
    controls = harness.Controls(fail_before_step=2)
    with pytest.raises(RuntimeError, match="crash before optimizer step"):
        run_hf_multimodal_sft(
            invocation=SimpleNamespace(
                attempt_id="attempt-1",
                observability=FINAL_EVALUATION_OBSERVABILITY,
            ),
            components=harness.Components(),
            run_directory=tmp_path / "run",
            controls=controls,
            observability=harness.Observability(),
            step_profiler=SimpleNamespace(
                input_wait=nullcontext, step=lambda _step: None
            ),
            resume_directory=None,
            device="cpu",
        )
    trained = [
        (index, request)
        for index, (kind, request, _weights) in enumerate(controls.events)
        if kind == "checkpoint" and request.optimizer_step > 0
    ]
    assert trained, "the run published no trained checkpoint to resume from"
    index, request = trained[0]
    assert request.optimizer_step == 1
    trained_state = json.loads(
        (request.source_directory / "engine-state.json").read_text(encoding="utf-8")
    )
    assert trained_state["runtime_state"]["launch_gallery_complete"] is True
    assert trained_state["runtime_state"]["baseline_complete"] is True
    assert trained_state["runtime_state"]["published_steps"] == [0, 1]

    resumed_controls = harness.Controls()
    resumed_observability = harness.Observability()
    step = run_hf_multimodal_sft(
        invocation=SimpleNamespace(
            attempt_id="attempt-2",
            observability=FINAL_EVALUATION_OBSERVABILITY,
        ),
        components=harness.Components(),
        run_directory=tmp_path / "run",
        controls=resumed_controls,
        observability=resumed_observability,
        step_profiler=SimpleNamespace(input_wait=nullcontext, step=lambda _step: None),
        resume_directory=request.source_directory,
        resume_parent_artifact_ids=(f"checkpoint-{index + 1}",),
        resume_checkpoint_manifest_digest="sha256:" + str(index + 1) * 64,
        device="cpu",
    )

    assert step == 2
    # The replacement attempt neither re-entered the launch gate nor published
    # anything against step zero: its baseline is the checkpoint it resumed.
    assert all(
        safe_point != ("evaluation", 0) for safe_point in resumed_controls.safe_points
    )
    assert all(
        step_at > 0
        for name, _value, step_at in resumed_observability.metrics
        if name in {"eval.loss", "eval.probe_loss", "eval.test_loss"}
    )
    assert all(
        published.step > 0
        for kind, published, _weights in resumed_controls.events
        if kind == "gallery"
    )
    assert all(
        published.optimizer_step > 0
        for kind, published, _weights in resumed_controls.events
        if kind == "checkpoint"
    )


def test_the_cadence_a_run_declares_is_the_cadence_it_publishes(tmp_path):
    _harness, _controls, observability = _run(
        tmp_path,
        schedule_configuration=EvaluationScheduleConfiguration(
            defer_full_scalar=False,
            probe_every_steps=2,
            probe_examples=1,
            qualitative_every_steps=3,
        ),
        maximum_steps=6,
    )
    planned = {
        name: value
        for name, value, step in observability.metrics
        if name.startswith("eval.planned") and step == 0
    }
    schedule = EvaluationSchedule(
        EvaluationScheduleConfiguration(
            defer_full_scalar=False,
            probe_every_steps=2,
            probe_examples=1,
            qualitative_every_steps=3,
        )
    )
    counts = schedule.plan(6).counts
    assert planned == {
        "eval.planned_scalar_full_milestones": counts["scalar_full"],
        "eval.planned_scalar_probe_milestones": counts["scalar_probe"],
        "eval.planned_qualitative_milestones": counts["qualitative"],
    }
    # Probe and full reads publish under distinct names, so a cheap bounded
    # read can never be plotted as a full validation of the split.
    probe_steps = sorted(
        step for name, _value, step in observability.metrics if name == "eval.probe_loss"
    )
    full_steps = sorted(
        step for name, _value, step in observability.metrics if name == "eval.loss"
    )
    assert probe_steps == list(schedule.plan(6).steps_for(EvaluationKind.SCALAR_PROBE))
    assert full_steps == list(schedule.plan(6).steps_for(EvaluationKind.SCALAR_FULL))
    assert set(probe_steps).isdisjoint(full_steps)


def _mutable_cadence():
    return EvaluationScheduleConfiguration(
        defer_full_scalar=False,
        probe_examples=1,
        qualitative_every_steps=3,
        mutable_cadence=True,
    )


def _run_with_patch(directory, *, patch, patch_at, maximum_steps=6):
    from contextlib import nullcontext
    from types import SimpleNamespace

    from rwkv_lab.trainvm_adapters.hf_multimodal_sft import run_hf_multimodal_sft

    directory.mkdir(parents=True, exist_ok=True)
    harness = _engine_harness(
        directory,
        schedule_configuration=_mutable_cadence(),
        maximum_steps=maximum_steps,
    )
    controls = harness.Controls(patch=patch, patch_at=patch_at)
    observability = harness.Observability()
    step = run_hf_multimodal_sft(
        invocation=SimpleNamespace(
            attempt_id="attempt-1",
            observability=FINAL_EVALUATION_OBSERVABILITY,
        ),
        components=harness.Components(),
        run_directory=directory / "run",
        controls=controls,
        observability=observability,
        step_profiler=SimpleNamespace(input_wait=nullcontext, step=lambda _step: None),
        resume_directory=None,
        device="cpu",
    )
    return step, harness, controls, observability


def test_a_live_cadence_patch_applies_at_an_evaluation_safe_point(tmp_path):
    step, _harness, _controls, observability = _run_with_patch(
        tmp_path / "patched",
        patch={"evaluation.probe_every_steps": 2},
        patch_at=("evaluation", 0),
    )
    assert step == 6
    probe_steps = sorted(
        at_step
        for name, _value, at_step in observability.metrics
        if name == "eval.probe_loss"
    )
    # The patch arrived at the step-zero evaluation, so the probe cadence it
    # declared is live for the rest of the run.
    assert probe_steps == [2, 4]
    revisions = [
        value
        for name, value, _step in observability.metrics
        if name == "eval.cadence_revision"
    ]
    # The patch lands at the step-zero gate, before the first plan is
    # published, and exactly one revision is recorded for the whole run.
    assert revisions and set(revisions) == {1}

    _step, _harness, _controls, unpatched = _run_with_patch(
        tmp_path / "unpatched", patch=None, patch_at=None
    )
    assert not [
        at_step
        for name, _value, at_step in unpatched.metrics
        if name == "eval.probe_loss"
    ]
    assert {
        value
        for name, value, _step in unpatched.metrics
        if name == "eval.cadence_revision"
    } == {0}


def test_only_the_evaluation_safe_point_carries_a_live_cadence_patch(tmp_path):
    from rwkv_lab.trainvm_adapters.hf_multimodal_sft import HFMultimodalSFTError

    for phase, at_step in (("optimizer_step", 1), ("microbatch", 1), ("checkpoint", 3)):
        with pytest.raises(HFMultimodalSFTError, match="evaluation cadence"):
            _run_with_patch(
                tmp_path / f"rejected-{phase}",
                patch={"evaluation.probe_every_steps": 2},
                patch_at=(phase, at_step),
            )


def test_a_live_patch_of_anything_but_cadence_is_refused(tmp_path):
    from rwkv_lab.trainvm_adapters.hf_multimodal_sft import HFMultimodalSFTError

    with pytest.raises(HFMultimodalSFTError, match="evaluation cadence controls"):
        _run_with_patch(
            tmp_path / "rejected-example-identity",
            patch={"evaluation.probe_examples": 4},
            patch_at=("evaluation", 0),
        )
