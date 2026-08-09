"""The four defects criterion 4 names, each shown not to move the sentinel.

`card-986f974e` requires that **missing, malformed, empty or wrong-step**
attempt-baseline evidence prevents the mutation sentinel from changing. Three
things about how that is asserted here are deliberate.

**Production controls, not a double.** Every existing criterion-4 test on an
armed route drives a hand-written `Controls` object: the HF engine harness'
(`tests/test_hf_multimodal_sft_engine.py`) `publish_evaluation_examples`
accepts *any* request and sets `step_zero_eval_gate_satisfied = True`
unconditionally, and `_LoopControls` in `tests/test_rwkv_mutation_sentinel.py`
re-implements the refusal it is testing. A double that cannot reject an empty
artifact cannot demonstrate that an empty artifact is rejected. These tests use
the real `WorkerSession`, the real `WorkerControlRuntime` built by
`controls_from_invocation`, and the real `EvalExamplesPublisher`, so the
condition each case binds is the one that ships.

**Four defects, four distinct production conditions.** They are separate tests
because they redden separately — deleting the emptiness bound leaves the
malformed and wrong-step cases green, and vice versa. See the module's
`MUTATION MATRIX` note below.

**Messages, not just types.** Every refusal here would raise `EvalExamplesError`
or `WorkerControlError` for half a dozen unrelated reasons, so a bare
`pytest.raises(EvalExamplesError)` stays green when the condition under test is
deleted and a neighbouring one catches the same input. Each case matches the
sentence its own condition emits.

Each case ends the same way and that ending is the actual claim: the sentinel's
journal is empty and `observed_mutations == 0`. A refusal that raised but left
the parameters mutated would satisfy every `pytest.raises` above and fail here.

MUTATION MATRIX — which production edit reddens which case, measured:

  `eval_examples.py` `not 0 < len(request.examples)`   -> empty only
  `eval_examples.py` `eval text part has incompatible` -> malformed only
  `session.py` `optimizer_step == ...baseline...`      -> wrong-step only
  `controls.py` `step_zero_eval_gate_required and not` -> all four

The last row is honest rather than embarrassing: one shared guard is what
refuses the mutation in every case, and the four evidence *defects* are caught
by four different conditions upstream of it. Only the shared guard is shared.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from test_trainvm_worker_documents import bootstrap_document, invocation_document

from rwkv_lab.rwkv_pretrain import perform_rwkv_optimizer_step
from rwkv_lab.trainvm_worker import (
    CheckpointPublisher,
    EvalEvidencePart,
    EvalExample,
    EvalExamplesPublicationRequest,
    OptimizerMutationSentinel,
    WorkerControlError,
    WorkerSession,
    controls_from_invocation,
    load_worker_bootstrap,
)
from rwkv_lab.trainvm_worker._canonical import canonical_dumps, sha256_digest
from rwkv_lab.trainvm_worker.eval_examples import EvalExamplesError

# Imported rather than re-declared: a second fake controller would be a second
# opinion about the wire protocol, and the two would drift.
from test_trainvm_worker_session import FakeController, resume_authority

BLOCKED = "optimizer mutation is blocked until durable attempt-baseline"


def armed_invocation(run_directory: Path, *, baseline: int = 0) -> bytes:
    """An invocation whose eval-examples declaration satisfies all three conjuncts.

    `required`, `type` and `schema` are what
    `invocation_requires_step_zero_eval_gate` reads
    (`trainvm/src/eval_examples_contract.cpp:390`); `immutability` and
    `fingerprint` are what the worker's own `EvalExamplesPublisher` additionally
    demands before it will publish. A declaration carrying one set and not the
    other is the seam that left the HF family declaring the port yet inert, so
    this fixture carries both.
    """

    # The baseline is not the controller's to assert unilaterally: the session
    # refuses a Welcome whose `attempt_baseline_optimizer_step` disagrees with
    # the invocation's immutable resume authority ("WorkerWelcome attempt
    # baseline disagrees with invocation"). A non-zero baseline therefore has
    # to be minted the way a replacement attempt's is — and only the v2
    # invocation shape carries a `resume` field at all.
    document = json.loads(
        invocation_document(resume=resume_authority(baseline) if baseline else None)
    )
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
                "name": name,
                "type": "gauge",
                "unit": "ratio",
                "step_domain": "optimizer_step",
                "aggregation": "last",
            }
            for name in ("perplexity", "validation_loss")
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
                "required": True,
            },
        },
        "eval_examples": {
            "logical_name": "eval_examples",
            "declaration": {
                "type": "eval_examples",
                "schema": "rwkv-lab.eval-examples.v1",
                "immutability": "append_only",
                "fingerprint": "manifest_sha256",
                "required": True,
            },
        },
    }
    return canonical_dumps(
        {**document, "invocation_digest": sha256_digest(canonical_dumps(document))}
    )


class _Armed:
    """A live worker session whose controller has armed the step-zero gate."""

    def __init__(self, run_directory: Path, *, baseline: int = 0) -> None:
        self.run_directory = run_directory
        self.baseline = baseline
        self.controller = FakeController(
            invocation=armed_invocation(run_directory, baseline=baseline),
            step_zero_eval_gate_required=True,
            attempt_baseline_optimizer_step=baseline,
        )
        self.session = WorkerSession(
            load_worker_bootstrap(bootstrap_document()), connector=self.controller
        )
        invocation = self.session.start()
        self.controls = controls_from_invocation(self.session, invocation)
        self.sentinel = OptimizerMutationSentinel()

    def publish_baseline_scalars(self, step: int) -> None:
        for name, value in (("validation_loss", 4.0), ("perplexity", 54.598)):
            self.session.metric(
                name,
                value,
                unit="ratio",
                step_domain="optimizer_step",
                step=step,
                wait=False,
            )

    def publish_checkpoint(self, step: int):
        source = self.run_directory / f"checkpoint-{step}"
        source.mkdir()
        (source / "state.pt").write_bytes(b"baseline-state")
        return CheckpointPublisher(self.session).publish(
            source,
            optimizer_step=step,
            resume_grade="terminal_checkpoint",
            state_components=("model", "optimizer", "rng_torch"),
        )

    def request(self, checkpoint, *, step: int, examples) -> EvalExamplesPublicationRequest:
        return EvalExamplesPublicationRequest(
            output_name="eval_examples",
            optimizer_step=step,
            series_id="rwkv-token-predictions",
            identity_field="id",
            identities_digest="sha256:" + "1" * 64,
            selector_digest="sha256:" + "2" * 64,
            evaluator_component_digest="sha256:" + "3" * 64,
            metric_names=("perplexity", "validation_loss"),
            checkpoint_artifact_id=checkpoint.artifact_id,
            checkpoint_manifest_digest=checkpoint.manifest_sha256,
            policy_digest="sha256:" + "4" * 64,
            examples=examples,
            parent_artifact_ids=(checkpoint.artifact_id,),
        )

    def attempt_guarded_mutation(self) -> None:
        """Exactly what a trainer does: cross the boundary, then mutate.

        Routed through `perform_rwkv_optimizer_step` rather than calling the
        controls directly, so a boundary that stopped refusing would have to
        also get past the sentinel to reach the parameters.
        """

        parameter = torch.nn.Parameter(torch.zeros(4))
        parameter.grad = torch.ones(4)
        optimizer = torch.optim.AdamW([parameter], lr=0.1)
        with self.sentinel.installed():
            perform_rwkv_optimizer_step(
                optimizer,
                self.controls,
                next_step=self.baseline + 1,
                control_applier=lambda *_: None,
                sentinel=self.sentinel,
            )

    def assert_nothing_mutated(self) -> None:
        assert self.sentinel.observed_mutations == 0
        assert self.sentinel.journal == ()
        assert not self.session.step_zero_eval_gate_satisfied


def _example(part: EvalEvidencePart | None = None) -> EvalExample:
    return EvalExample(
        example_id="rwkv-heldout",
        heldout_item_id="heldout-1",
        heldout_item_digest="sha256:" + "5" * 64,
        input=(part or EvalEvidencePart(kind="text", text="token_ids: 3 4"),),
        target=(EvalEvidencePart(kind="text", text="token_id: 5"),),
        prediction=(EvalEvidencePart(kind="text", text="token_id: 7"),),
    )


@pytest.fixture
def armed(tmp_path: Path) -> _Armed:
    session = _Armed(tmp_path)
    yield session
    session.session.close()


# --------------------------------------------------------------------------
# 1. Missing
# --------------------------------------------------------------------------


def test_missing_evidence_cannot_move_the_mutation_sentinel(armed: _Armed) -> None:
    """No publication at all. The commonest shape: a route wired but not armed."""

    with pytest.raises(WorkerControlError, match=BLOCKED):
        armed.attempt_guarded_mutation()
    armed.assert_nothing_mutated()


# --------------------------------------------------------------------------
# 2. Empty
# --------------------------------------------------------------------------


def test_empty_evidence_cannot_move_the_mutation_sentinel(armed: _Armed) -> None:
    """A well-formed artifact carrying no examples.

    This is the case a double cannot show. The publication is refused before
    anything reaches the controller, so the gate never latches and the boundary
    refuses for the same reason a missing artifact does — but by a different
    condition, `0 < len(request.examples)`.
    """

    armed.publish_baseline_scalars(0)
    checkpoint = armed.publish_checkpoint(0)
    with pytest.raises(EvalExamplesError, match="example count is outside its bound"):
        armed.controls.publish_evaluation_examples(
            armed.request(checkpoint, step=0, examples=())
        )
    with pytest.raises(WorkerControlError, match=BLOCKED):
        armed.attempt_guarded_mutation()
    armed.assert_nothing_mutated()


# --------------------------------------------------------------------------
# 3. Malformed
# --------------------------------------------------------------------------


def test_malformed_evidence_cannot_move_the_mutation_sentinel(armed: _Armed) -> None:
    """A non-empty artifact whose typed evidence does not typecheck.

    The count is right and the provenance is right; one evidence part claims to
    be text while carrying a structured payload. Modality-appropriate typed
    input/target/prediction is exactly what the card asks the artifact to
    carry, so a part that lies about its kind is malformed evidence and not a
    lesser sin than an absent one.
    """

    armed.publish_baseline_scalars(0)
    checkpoint = armed.publish_checkpoint(0)
    malformed = EvalEvidencePart(kind="text", text="token_ids: 3 4", value={"a": 1})
    with pytest.raises(EvalExamplesError, match="text part has incompatible fields"):
        armed.controls.publish_evaluation_examples(
            armed.request(checkpoint, step=0, examples=(_example(malformed),))
        )
    with pytest.raises(WorkerControlError, match=BLOCKED):
        armed.attempt_guarded_mutation()
    armed.assert_nothing_mutated()


# --------------------------------------------------------------------------
# 4. Wrong step
# --------------------------------------------------------------------------


def test_wrong_step_evidence_cannot_move_the_mutation_sentinel(tmp_path: Path) -> None:
    """The one case where publication SUCCEEDS and the gate still stays shut.

    A replacement attempt is gated at its resume checkpoint's step, not at
    zero. Evidence published at zero is a valid, non-empty, well-formed
    `rwkv-lab.eval-examples.v1` artifact — the publisher accepts it and the
    controller acknowledges it. It satisfies nothing, because the gate latches
    on `optimizer_step == attempt_baseline_optimizer_step` and this is the
    step the attempt will never reach again.

    A test that only ever fed refused publications would pass with that
    equality deleted; this one is the reason the equality is load-bearing.
    """

    armed = _Armed(tmp_path, baseline=7)
    try:
        armed.publish_baseline_scalars(7)
        checkpoint = armed.publish_checkpoint(0)
        published = armed.controls.publish_evaluation_examples(
            armed.request(checkpoint, step=0, examples=(_example(),))
        )
        assert published.artifact_id, "the wrong-step artifact was itself accepted"
        assert armed.session.acknowledged_worker_sequence >= published.worker_sequence

        with pytest.raises(WorkerControlError, match=BLOCKED):
            armed.attempt_guarded_mutation()
        armed.assert_nothing_mutated()
    finally:
        armed.session.close()


# --------------------------------------------------------------------------
# The positive control
# --------------------------------------------------------------------------


def test_complete_evidence_at_the_gated_step_authorizes_exactly_one_mutation(
    tmp_path: Path,
) -> None:
    """Without this, all four refusals above are satisfied by a gate stuck shut.

    Run at a non-zero baseline so the passing path is the resumed-attempt
    keying rather than a literal zero that happens to coincide with it.
    """

    armed = _Armed(tmp_path, baseline=7)
    try:
        armed.publish_baseline_scalars(7)
        checkpoint = armed.publish_checkpoint(7)
        armed.controls.publish_evaluation_examples(
            armed.request(checkpoint, step=7, examples=(_example(),))
        )
        assert armed.session.step_zero_eval_gate_satisfied

        armed.attempt_guarded_mutation()
        assert armed.sentinel.journal == (("boundary", 8), ("mutation", 8))
        assert armed.sentinel.observed_mutations == 1
    finally:
        armed.session.close()
