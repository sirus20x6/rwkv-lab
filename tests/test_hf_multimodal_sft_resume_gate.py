"""The resume half of criterion 4, driven against production controls.

`card-986f974e` requires that resume/reconnect **recover gate satisfaction from
journal/checkpoint evidence without permitting an unguarded update**.
`tests/test_step_zero_gate_refusals.py` (PR #199) established the four refusals
against the production worker stack, but it drives `rwkv_scratch`, which cannot
resume at all — `rwkv_lab.rwkv_scratch.v1.Train` declares both pause flags false
and a `terminal_checkpoint` resume grade, so no `invocation.resume` is ever
minted for it and its resume demonstration would be vacuous.
`rwkv_lab.hf_multimodal_sft.v1.Train` declares `exact_training_lifecycle()`, so
it is the armed route that owes this.

**Production controls, not a double.** Every other test of this engine —
`tests/test_hf_multimodal_sft_engine.py`, `tests/test_hf_multimodal_sft_mutation_boundary.py`,
`tests/test_evaluation_cadence_milestones.py` — drives the harness `Controls`
double whose `publish_evaluation_examples` accepts *any* request and sets
`step_zero_eval_gate_satisfied = True` unconditionally
(`tests/test_hf_multimodal_sft_engine.py:1428-1436`). On the resume path that
double can refuse only *absent* evidence, and only because the engine never
calls it: empty, malformed and wrong-step replacement evidence all open the gate
there. Here the real `WorkerSession`, the real `WorkerControlRuntime` built by
`controls_from_invocation`, the real `EvalExamplesPublisher`, `CheckpointPublisher`,
`EvalGalleryPublisher`, `ImmutableArtifactPublisher` and `FinalEvaluationPublisher`
decide, so what each case binds is the condition that ships.

Only three things are not production: the model and dataset (the tiny CPU object
graph from `_engine_harness`, since a real Hugging Face multimodal snapshot is
not available to a CPU-only pytest run), the controller (`FakeController`,
imported rather than re-declared so there is one opinion about the wire), and a
thin delegating recorder around the real runtime. The recorder decides nothing:
it appends to a list, optionally perturbs one field of the request the engine
built, and re-raises whatever production returns.

**Evidence defects are injected, not simulated.** An engine that publishes
correct evidence cannot be made to publish empty or wrong-step evidence from the
outside, and editing the engine to make it do so would be testing the edit. So
the recorder rewrites exactly one field of the real
`EvalExamplesPublicationRequest` the engine constructed and hands the result to
the real publisher. Every refusal below is production's.

MUTATION MATRIX — which production edit reddens which case, measured by applying
one mutant at a time to an isolated copy of the tree and parsing `--junitxml`.
Every run collected all six; the unmutated baseline was green.

  `eval_examples.py` `not 0 < len(request.examples)`     -> empty only
  `eval_examples.py` text-part incompatible-fields check -> malformed only
  `session.py` `optimizer_step == ...baseline...` latch  -> wrong-step only
  `hf_multimodal_sft.py` seeding `attempt_baseline_examples_published`
      from `controls.step_zero_eval_gate_satisfied`      -> reconnect only
  `hf_multimodal_sft.py` resumed-at-an-ungated-step raise-> ungated only
  `controls.py` `step_zero_eval_gate_required and not`   -> empty, malformed
                                                            and wrong-step
  `controls.py` `publish_artifact`'s publisher name      -> both positive
                                                            controls

The last two rows are shared conditions rather than a smell. Three different
upstream conditions catch three different evidence defects and one guard is what
refuses the mutation in each; and the two cases that run to completion are the
only ones that reach `publish_artifact` at all — which is how the `ImportError`
that broke it for every caller was found, having survived because no test had
ever driven this engine over the real runtime.
"""

from __future__ import annotations

import dataclasses
import json
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch.optim.optimizer import register_optimizer_step_pre_hook

from test_hf_multimodal_sft_engine import _engine_harness
from test_trainvm_worker_documents import bootstrap_document, invocation_document
from test_trainvm_worker_session import FakeController, resume_authority

from rwkv_lab.trainvm_adapters.hf_multimodal_sft import (
    HFMultimodalSFTError,
    run_hf_multimodal_sft,
)
from rwkv_lab.training_runtime.evaluation_schedules import (
    EvaluationScheduleConfiguration,
)
from rwkv_lab.trainvm_worker import (
    WorkerControlError,
    WorkerSession,
    controls_from_invocation,
    load_worker_bootstrap,
)
from rwkv_lab.trainvm_worker._canonical import canonical_dumps, sha256_digest
from rwkv_lab.trainvm_worker.eval_examples import EvalExamplesError
from rwkv_lab.trainvm_worker.session import wire

BLOCKED = "optimizer mutation is blocked until durable attempt-baseline"

# A checkpoint at every step, so the interrupted attempt leaves a genuinely
# mid-run checkpoint to resume from rather than only its launch one.
RESUMABLE_CADENCE = EvaluationScheduleConfiguration(
    defer_full_scalar=False, qualitative_every_steps=1
)
FINAL_EVALUATION_OBSERVABILITY = {
    "metrics": ({"name": "eval.test_loss", "step_domain": "optimizer_step"},)
}

RECIPE_PATH = (
    Path(__file__).resolve().parents[1]
    / "docs/experiment-vm/examples/hf-multimodal-sft.recipe-profiles.v1.json"
)
# The five output declarations the shipped recipe authors real runs from, read
# from the document rather than restated. Every publisher the engine reaches
# rejects a declaration whose type, schema, immutability or fingerprint is not
# the one it expects, so a hand-written copy that drifted from the recipe would
# make this suite refuse for a reason that has nothing to do with the gate.
RECIPE_ARTIFACTS: dict[str, dict[str, object]] = json.loads(
    RECIPE_PATH.read_text(encoding="utf-8")
)["recipes"][0]["template_document"]["spec"]["artifacts"]


def armed_invocation(root: Path, run_directory: Path, *, baseline: int = 0) -> bytes:
    """A v2 invocation declaring the five outputs this engine publishes.

    `required` + `type` + `schema` on the `eval_examples` declaration is what
    arms `invocation_requires_step_zero_eval_gate` on the controller. The
    baseline is not the controller's to assert unilaterally: `_accept_welcome`
    refuses a Welcome whose `attempt_baseline_optimizer_step` disagrees with the
    invocation's immutable resume authority, so a replacement attempt's baseline
    has to be minted the way production mints it — through `resume`.
    """

    document = json.loads(
        invocation_document(resume=resume_authority(baseline) if baseline else None)
    )
    document.pop("invocation_digest")
    document["workspace"] = {
        "run_directory": str(run_directory),
        "allowed_read_roots": [str(root)],
        "allowed_write_roots": [str(root)],
    }
    document["observability"] = {
        "heartbeat_seconds": 30,
        "metrics": [
            {
                "name": name,
                "type": "gauge",
                "unit": "ratio",
                "step_domain": "optimizer_step",
                "aggregation": "last",
            }
            for name in ("eval.loss", "eval.test_loss")
        ],
        "retain_raw_metrics_days": 7,
    }
    document["publishes"] = {
        name: {
            "logical_name": name,
            "declaration": declaration,
            # Only the final-evaluation output carries one; its publisher
            # refuses a publication whose closure policy the controller has not
            # pinned ("lacks closed controller policy authority").
            **(
                {"finalization_policy_digest": "sha256:" + "a" * 64}
                if name == "final_evaluation"
                else {}
            ),
        }
        for name, declaration in RECIPE_ARTIFACTS.items()
    }
    return canonical_dumps(
        {**document, "invocation_digest": sha256_digest(canonical_dumps(document))}
    )


class _Recorder:
    """A delegating view of the real runtime. It decides nothing.

    `__getattr__` forwards everything not named here, so the engine reaches the
    production property or method in every case the tests are not observing.
    """

    def __init__(self, runtime, events, *, perturb=None, interrupt_before=None):
        self._runtime = runtime
        self._events = events
        self._perturb = perturb
        self._interrupt_before = interrupt_before
        self.published_example_steps: list[int] = []
        self.checkpoint_sources: list[tuple[int, Path]] = []

    def __getattr__(self, name):
        return getattr(self._runtime, name)

    def microbatch(self, step, applier):
        if self._interrupt_before and step == self._interrupt_before:
            raise RuntimeError("simulated crash before optimizer step")
        return self._runtime.microbatch(step, applier)

    def pre_optimizer_step(self, step, applier):
        self._events.append(("boundary", step))
        return self._runtime.pre_optimizer_step(step, applier)

    def publish_policy_checkpoint(self, request, **keywords):
        self.checkpoint_sources.append(
            (request.optimizer_step, Path(request.source_directory))
        )
        return self._runtime.publish_policy_checkpoint(request, **keywords)

    def publish_evaluation_examples(self, request):
        self.published_example_steps.append(request.optimizer_step)
        if self._perturb is not None:
            request = self._perturb(request)
        return self._runtime.publish_evaluation_examples(request)


class _Attempt:
    """One worker attempt: a real session over a real control runtime."""

    def __init__(
        self,
        root: Path,
        run_directory: Path,
        events: list,
        *,
        baseline: int = 0,
        satisfied: bool = False,
        perturb=None,
        interrupt_before: int | None = None,
    ) -> None:
        self.controller = FakeController(
            invocation=armed_invocation(root, run_directory, baseline=baseline),
            step_zero_eval_gate_required=True,
            step_zero_eval_gate_satisfied=satisfied,
            attempt_baseline_optimizer_step=baseline,
        )
        self.session = WorkerSession(
            load_worker_bootstrap(bootstrap_document()), connector=self.controller
        )
        invocation = self.session.start()
        self.runtime = controls_from_invocation(self.session, invocation)
        self.controls = _Recorder(
            self.runtime,
            events,
            perturb=perturb,
            interrupt_before=interrupt_before,
        )

    @property
    def published_eval_examples(self) -> tuple:
        """Every eval-examples artifact this attempt actually put on the wire."""

        return tuple(
            message
            for message in self.controller.received
            if message.WhichOneof("message") == "artifact"
            and message.artifact.kind == wire.ARTIFACT_KIND_EVAL_EXAMPLES
        )

    def close(self) -> None:
        self.session.close()


def _run(harness, attempt: _Attempt, run_directory: Path, *, resume=None, attempt_id):
    return run_hf_multimodal_sft(
        invocation=SimpleNamespace(
            attempt_id=attempt_id,
            observability=FINAL_EVALUATION_OBSERVABILITY,
        ),
        components=harness.Components(),
        run_directory=run_directory,
        controls=attempt.controls,
        observability=harness.Observability(),
        step_profiler=SimpleNamespace(input_wait=nullcontext, step=lambda _step: None),
        resume_directory=resume,
        resume_parent_artifact_ids=() if resume is None else ("checkpoint-1",),
        resume_checkpoint_manifest_digest=(
            None if resume is None else "sha256:" + "1" * 64
        ),
        device="cpu",
    )


@pytest.fixture
def observed_mutations():
    """Every optimizer mutation the process performs, in order.

    Shares one list with the recorder's boundary crossings, so what is asserted
    is an interleaving rather than two independent counts.
    """

    events: list[tuple[str, int]] = []
    handle = register_optimizer_step_pre_hook(
        lambda optimizer, args, kwargs: events.append(("mutation", -1))
    )
    try:
        yield events
    finally:
        handle.remove()


@pytest.fixture
def interrupted(tmp_path: Path, observed_mutations):
    """A first attempt crashed before step 2, leaving a real step-1 checkpoint.

    The checkpoint is published by the production `CheckpointPublisher` against
    the production session, so the directory the replacement attempt restores
    from is one this stack actually produced.
    """

    run_directory = tmp_path / "run"
    run_directory.mkdir()
    harness = _engine_harness(
        tmp_path, schedule_configuration=RESUMABLE_CADENCE, maximum_steps=3
    )
    attempt = _Attempt(tmp_path, run_directory, observed_mutations, interrupt_before=2)
    try:
        with pytest.raises(RuntimeError, match="crash before optimizer step"):
            _run(harness, attempt, run_directory, attempt_id="attempt-1")
        trained = [
            source
            for step, source in attempt.controls.checkpoint_sources
            if step == 1
        ]
        assert trained, "the interrupted attempt published no trained checkpoint"
        assert attempt.controls.published_example_steps == [0], (
            "the fresh attempt did not publish its baseline evidence at zero"
        )
    finally:
        attempt.close()
    observed_mutations.clear()
    # One harness across both attempts, as `test_hf_multimodal_sft_mutation_boundary`
    # does: the replacement attempt restores its weights from the checkpoint, so
    # what carries over is the frozen dataset and the object graph, not state.
    return SimpleNamespace(
        root=tmp_path,
        run_directory=run_directory,
        resume_directory=trained[0],
        harness=harness,
    )


def _replacement(interrupted, observed_mutations, **keywords) -> _Attempt:
    return _Attempt(
        interrupted.root,
        interrupted.run_directory,
        observed_mutations,
        **keywords,
    )


# --------------------------------------------------------------------------
# Recovered from replayed evidence — the positive control
# --------------------------------------------------------------------------


def test_a_reconnect_recovers_gate_satisfaction_without_republishing(
    interrupted, observed_mutations
) -> None:
    """The claim criterion 4 makes, and the control for every refusal below.

    The controller's Welcome carries `step_zero_eval_gate_satisfied` recovered
    from its journal — `durable_attempt_baseline_eval_gate_satisfied` in
    `trainvm/src/eval_examples_contract.cpp` reads the attempt's baseline
    checkpoint, scalar and eval-examples records and replays the verdict at
    `service.cpp:3468-3479`. The worker must accept it without minting a second
    artifact for evidence that is already durable, and must still cross the
    boundary before every mutation.

    Without this case, every refusal below is satisfied by a gate stuck shut.
    """

    attempt = _replacement(
        interrupted, observed_mutations, baseline=1, satisfied=True
    )
    try:
        step = _run(
            interrupted.harness,
            attempt,
            interrupted.run_directory,
            resume=interrupted.resume_directory,
            attempt_id="attempt-2",
        )
        assert step == 3
        assert attempt.controls.published_example_steps == [], (
            "durable baseline evidence was published a second time"
        )
        assert attempt.published_eval_examples == (), (
            "a replayed gate still put an eval-examples artifact on the wire"
        )
        # Recovering satisfaction never licenses skipping the boundary.
        assert observed_mutations == [
            ("boundary", 2),
            ("mutation", -1),
            ("boundary", 3),
            ("mutation", -1),
        ]
    finally:
        attempt.close()


def test_a_replacement_attempt_publishes_its_baseline_at_the_gated_step(
    interrupted, observed_mutations
) -> None:
    """Nothing durable to replay, so this attempt owes evidence — at step 1.

    The second positive control, and the one that makes the wrong-step case
    below load-bearing: it is the real `session.py` latch, keyed on
    `optimizer_step == attempt_baseline_optimizer_step`, that opens the gate
    here. A trainer that keyed its baseline to a literal zero would publish a
    perfectly valid artifact that this latch ignores.
    """

    attempt = _replacement(
        interrupted, observed_mutations, baseline=1, satisfied=False
    )
    try:
        assert attempt.runtime.attempt_baseline_optimizer_step == 1
        assert not attempt.runtime.step_zero_eval_gate_satisfied

        step = _run(
            interrupted.harness,
            attempt,
            interrupted.run_directory,
            resume=interrupted.resume_directory,
            attempt_id="attempt-2",
        )

        assert step == 3
        assert attempt.controls.published_example_steps == [1], (
            "the replacement attempt did not owe its baseline at its own step"
        )
        assert attempt.runtime.step_zero_eval_gate_satisfied
        assert observed_mutations == [
            ("boundary", 2),
            ("mutation", -1),
            ("boundary", 3),
            ("mutation", -1),
        ]
    finally:
        attempt.close()


# --------------------------------------------------------------------------
# Defective replacement evidence
# --------------------------------------------------------------------------


def test_empty_replacement_evidence_cannot_move_the_mutation_sentinel(
    interrupted, observed_mutations
) -> None:
    """A well-formed replacement artifact carrying no examples.

    The double cannot show this: it accepts any request. The real publisher
    refuses before anything reaches the controller, so the gate never latches
    and the boundary refuses the first mutation of the replacement attempt.
    """

    attempt = _replacement(
        interrupted,
        observed_mutations,
        baseline=1,
        satisfied=False,
        perturb=lambda request: dataclasses.replace(request, examples=()),
    )
    try:
        with pytest.raises(EvalExamplesError, match="example count is outside its bound"):
            _run(
                interrupted.harness,
                attempt,
                interrupted.run_directory,
                resume=interrupted.resume_directory,
                attempt_id="attempt-2",
            )
        assert not attempt.runtime.step_zero_eval_gate_satisfied
        assert observed_mutations == []
        # The run aborted on the refused publication, and an abort alone would
        # leave the parameters unmutated even with the pre-mutation guard
        # deleted. So cross the boundary the loop would have crossed next: the
        # attempt still owes evidence, and production still says no.
        with pytest.raises(WorkerControlError, match=BLOCKED):
            attempt.runtime.pre_optimizer_step(2, lambda *_: None)
    finally:
        attempt.close()


def test_malformed_replacement_evidence_cannot_move_the_mutation_sentinel(
    interrupted, observed_mutations
) -> None:
    """The count and the provenance are right; one evidence part lies about its kind.

    The engine's caption evidence carries a `structured` part whose payload is
    the frozen manifest provenance. Claiming `text` while carrying that payload
    is malformed evidence, not a lesser sin than an absent artifact.
    """

    def malform(request):
        example = request.examples[0]
        part = dataclasses.replace(example.input[-1], kind="text", text="caption")
        broken = dataclasses.replace(
            example, input=(*example.input[:-1], part)
        )
        return dataclasses.replace(request, examples=(broken, *request.examples[1:]))

    attempt = _replacement(
        interrupted,
        observed_mutations,
        baseline=1,
        satisfied=False,
        perturb=malform,
    )
    try:
        with pytest.raises(EvalExamplesError, match="text part has incompatible fields"):
            _run(
                interrupted.harness,
                attempt,
                interrupted.run_directory,
                resume=interrupted.resume_directory,
                attempt_id="attempt-2",
            )
        assert not attempt.runtime.step_zero_eval_gate_satisfied
        assert observed_mutations == []
        # The run aborted on the refused publication, and an abort alone would
        # leave the parameters unmutated even with the pre-mutation guard
        # deleted. So cross the boundary the loop would have crossed next: the
        # attempt still owes evidence, and production still says no.
        with pytest.raises(WorkerControlError, match=BLOCKED):
            attempt.runtime.pre_optimizer_step(2, lambda *_: None)
    finally:
        attempt.close()


def test_replacement_evidence_at_the_previous_attempts_step_cannot_move_the_sentinel(
    interrupted, observed_mutations
) -> None:
    """The literal-zero bug, shown against the real latch.

    This is the one case where the publication SUCCEEDS and the gate still
    stays shut: an artifact at step 0 is valid, non-empty and well-formed, the
    publisher accepts it and the controller acknowledges it. It satisfies
    nothing, because the latch keys on
    `optimizer_step == attempt_baseline_optimizer_step` and this replacement
    attempt is gated at 1 — the step it will never reach again.

    `8f0da3e` fixed exactly this in `src/rwkv_lab/mage_flow_pretrain.py`, where
    the baseline had been a literal `0`. Against the harness double the same
    publication opens the gate, so the double could not have caught it.
    """

    attempt = _replacement(
        interrupted,
        observed_mutations,
        baseline=1,
        satisfied=False,
        perturb=lambda request: dataclasses.replace(request, optimizer_step=0),
    )
    try:
        with pytest.raises(WorkerControlError, match=BLOCKED):
            _run(
                interrupted.harness,
                attempt,
                interrupted.run_directory,
                resume=interrupted.resume_directory,
                attempt_id="attempt-2",
            )
        published = attempt.published_eval_examples
        assert published, "the wrong-step artifact was not itself accepted"
        assert published[0].artifact.optimizer_step == 0
        assert not attempt.runtime.step_zero_eval_gate_satisfied
        # The crossing was attempted and refused. Unlike the two cases above,
        # which never reach the loop, this run gets as far as the boundary — and
        # stops there, with no mutation after it.
        assert observed_mutations == [("boundary", 2)]
    finally:
        attempt.close()


def test_a_replacement_attempt_gated_at_a_step_it_cannot_reach_refuses_to_train(
    interrupted, observed_mutations
) -> None:
    """Say so, rather than deadlocking at the first crossing with nothing to read.

    The controller gates this attempt at zero — its invocation carries no
    resume authority, so `attempt_baseline_optimizer_step` is zero — while the
    worker restored a step-1 checkpoint. Any evidence it published would be
    evidence the gate ignores.
    """

    attempt = _replacement(
        interrupted, observed_mutations, baseline=0, satisfied=False
    )
    try:
        assert attempt.runtime.attempt_baseline_optimizer_step == 0
        with pytest.raises(HFMultimodalSFTError, match="does not gate"):
            _run(
                interrupted.harness,
                attempt,
                interrupted.run_directory,
                resume=interrupted.resume_directory,
                attempt_id="attempt-2",
            )
        assert attempt.controls.published_example_steps == []
        assert observed_mutations == []
    finally:
        attempt.close()
