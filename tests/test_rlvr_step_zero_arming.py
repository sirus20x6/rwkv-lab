"""RLVR's newly-armed step-zero gate, driven through the production stack.

`card-21d583f6` says 18 stateful routes cannot arm because they declare no
`eval_examples` output port. RLVR is the first of them to be armed, and these
tests are the evidence that the arming is real rather than declarative.

Three things about how that is asserted are deliberate.

**The declaration under test is the shipped one.** `shipped_artifacts` reads the
artifact declarations out of `docs/experiment-vm/examples/rlvr-candidate.json`
rather than restating them. A fixture that hand-builds what the shipped
document declares is two opinions about one contract, and the seam this route
had to cross is exactly a disagreement between two such fields: the *port's*
`required` (in `rwkv_lab_worker_contract.cpp`) and the *artifact's* `required`
(in the composition document) are independent, and only the second arms the
controller. So the document is the input here, and if it stops satisfying the
publisher these tests say so.

**Production controls, not a double.** `tests/test_rlvr_mutation_boundary.py`
drives a hand-written `FakeControls` whose `pre_optimizer_step` re-implements
the refusal it tests; that is right for the ordering property it owns and wrong
for this one. These tests use the real `WorkerSession`, the real
`WorkerControlRuntime` built by `controls_from_invocation`, and the real
`EvalExamplesPublisher`, reached through the real `rlvr_train.run`. The
`FakeController` is the *wire peer*, not the component under test — the same
arrangement `tests/test_step_zero_gate_refusals.py` established.

**Messages, not types.** Every refusal below would raise `ValueError` or
`EvalExamplesError` for half a dozen unrelated reasons — a missing task pool, a
malformed digest, an unsupported optimizer. Each case matches the sentence its
own condition emits, so deleting that condition reddens the case rather than
letting a neighbour catch the same input.

WHAT THIS FILE DOES NOT PROVE. The controller-side
`validate_eval_examples_gate_provenance` (`trainvm/src/eval_examples_contract.cpp`)
additionally requires a prior durable *declared scalar* and a prior durable
*checkpoint artifact* at the manifest's own step. Those are C++ conditions
evaluated by the real service, and a Python `FakeController` does not run them.
What is shown here is that the worker emits its evidence in the order those
conditions require — checkpoint, then examples, both before the first mutation
— not that the service then accepts it.
"""

from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

import pytest
import torch
from test_trainvm_worker_documents import bootstrap_document, invocation_document
from test_trainvm_worker_session import FakeController, resume_authority
from torch import nn
from torch.optim.optimizer import register_optimizer_step_pre_hook

from rwkv_lab import rlvr_train
from rwkv_lab.rlvr_train import Rollout
from rwkv_lab.trainvm_adapters.rlvr import RLVRHeldoutEvalPolicy
from rwkv_lab.trainvm_worker import (
    WorkerSession,
    controls_from_invocation,
    load_worker_bootstrap,
)
from rwkv_lab.trainvm_worker._canonical import canonical_dumps, sha256_digest
from rwkv_lab.trainvm_worker.eval_examples import EvalExamplesError

SHIPPED = (
    Path(__file__).resolve().parents[1]
    / "docs/experiment-vm/examples/rlvr-candidate.json"
)
DIGEST = "sha256:" + "b" * 64

# Distinguishes "run with no evaluation policy at all" from "run with the
# fixture's default", which `None` cannot do because `None` is what the trainer
# receives in the first case.
DEFAULT_POLICY = object()


def shipped_artifacts() -> dict[str, dict]:
    """The two artifact declarations the shipped RLVR composition publishes.

    Keyed by the *output port* name, which is how the worker's `publishes` map
    is keyed, rather than by the logical artifact name the document uses.
    """

    document = json.loads(SHIPPED.read_text(encoding="utf-8"))
    specification = document["spec"]
    node = specification["workflow"]["nodes"]["rlvr_candidate"]
    return {
        output: dict(specification["artifacts"][logical])
        for output, logical in node["publishes"].items()
    }


def armed_invocation(
    run_directory: Path, *, publishes: dict | None = None, baseline: int = 0
) -> bytes:
    # A non-zero baseline is not the controller's to assert unilaterally: the
    # session refuses a Welcome whose `attempt_baseline_optimizer_step`
    # disagrees with the invocation's immutable resume authority, so a
    # replacement attempt's baseline has to be minted the way a real one is.
    document = json.loads(
        invocation_document(resume=resume_authority(baseline) if baseline else None)
    )
    document.pop("invocation_digest")
    # RLVR v1 declares no live controls and refuses a run that arrives with
    # any, so the shared fixture's learning-rate control is dropped rather than
    # worked around inside the trainer.
    document["controls"] = {}
    document["workspace"] = {
        "run_directory": str(run_directory),
        "allowed_read_roots": [str(run_directory)],
        "allowed_write_roots": [str(run_directory)],
    }
    declarations = shipped_artifacts() if publishes is None else publishes
    document["publishes"] = {
        output: {"logical_name": output, "declaration": declaration}
        for output, declaration in declarations.items()
    }
    return canonical_dumps(
        {**document, "invocation_digest": sha256_digest(canonical_dumps(document))}
    )


def eval_policy(**overrides) -> RLVRHeldoutEvalPolicy:
    values: dict[str, object] = {
        "identity_field": "id",
        "evaluator_component_digest": DIGEST,
        "metric_names": ("eval.pass_at_k", "eval.reward"),
        "artifact_renderer_digest": DIGEST,
        "qualitative_sample_digest": DIGEST,
        "sample_count": 2,
        "decode": (("max_new_tokens", 4), ("top_p", 1.0)),
    }
    values.update(overrides)
    return RLVRHeldoutEvalPolicy(**values)  # type: ignore[arg-type]


def published_manifests(run_directory: Path) -> list[dict]:
    root = run_directory / "trainvm_artifacts" / "eval_examples" / "revisions"
    return [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(root.rglob("manifest.json"))
    ]


class ToyLM(nn.Module):
    """The smallest model the real scoring and update paths accept."""

    def __init__(self, vocab: int = 16, width: int = 8) -> None:
        super().__init__()
        self.emb = nn.Embedding(vocab, width)
        self.head = nn.Linear(width, vocab, bias=False)

    def forward(self, ids):
        return self.head(self.emb(ids))


class ToyVocab:
    def encode(self, text: str) -> list[int]:
        return [2 + (ord(character) % 8) for character in text[:6]] or [2]

    def decode(self, ids) -> str:
        return "".join(chr(97 + (int(i) % 8)) for i in ids)


@pytest.fixture
def mutations():
    """Every optimizer mutation in the process, however it was reached.

    A process-global pre-hook rather than a counter kept by the trainer: an
    alternate update path, a second optimizer, or a fused step all appear here,
    and none of them appear in a number the trainer maintains about itself.
    """

    recorded: list[int] = []
    handle = register_optimizer_step_pre_hook(
        lambda optimizer, args, kwargs: recorded.append(1)
    )
    try:
        yield recorded
    finally:
        handle.remove()


@pytest.fixture
def armed(tmp_path, monkeypatch):
    """The real `rlvr_train.run` against a real session, controls and publisher.

    Only generation and verification are replaced — both are slow and neither
    mutates parameters — so the baseline held-out evaluation, the checkpoint
    publication, the eval-examples publication and both mutation paths are the
    production ones.
    """

    from rwkv_lab import generate as generate_module

    model = ToyLM()
    source = tmp_path / "source.pt"
    source.write_bytes(b"toy-rlvr-checkpoint")
    state: dict[str, object] = {"blob": {"arch": {}}}
    monkeypatch.setattr(
        generate_module,
        "build_from_ckpt",
        lambda *a, **k: (model, dict(state["blob"])),  # type: ignore[arg-type]
    )
    monkeypatch.setattr(generate_module, "WorldVocab", lambda *a, **k: ToyVocab())

    responses: dict[str, str] = {}

    def fake_generate(
        _model, _vocab, tasks, *, group_size, return_stats=False, **kwargs
    ):
        rollouts = [
            Rollout(
                f"{task.id}:{member}",
                task,
                [2, 3],
                [4 + member, 1],
                responses.get(task.id, responses.get("*", f"answer-{member}")),
            )
            for task in tasks
            for member in range(group_size)
        ]
        stats = {"tokens": 2 * len(rollouts), "truncated": 0}
        return (rollouts, stats) if return_stats else rollouts

    def fake_verify(rollouts, **kwargs):
        return (
            torch.tensor([float(index % 2) for index in range(len(rollouts))]),
            [{"source": "fake"} for _ in rollouts],
        )

    monkeypatch.setattr(rlvr_train, "generate_rollouts", fake_generate)
    monkeypatch.setattr(rlvr_train, "verify_rollouts", fake_verify)

    sessions: list[WorkerSession] = []

    def invoke(
        *,
        baseline: int = 0,
        policy: object = DEFAULT_POLICY,
        publishes: dict | None = None,
        source_blob: dict | None = None,
        **overrides,
    ):
        if source_blob is not None:
            state["blob"] = source_blob
        run_directory = tmp_path / f"run-{len(sessions)}"
        run_directory.mkdir()
        controller = FakeController(
            invocation=armed_invocation(
                run_directory, publishes=publishes, baseline=baseline
            ),
            step_zero_eval_gate_required=True,
            attempt_baseline_optimizer_step=baseline,
        )
        session = WorkerSession(
            load_worker_bootstrap(bootstrap_document()), connector=controller
        )
        sessions.append(session)
        invocation = session.start()
        controls = controls_from_invocation(session, invocation)
        arguments = {
            "ckpt": str(source), "resume": "", "out": str(run_directory / "out"),
            "tasks": "", "heldout_tasks": "", "algorithm": "gspo", "steps": 1,
            "prompts_per_step": 2, "group_size": 2, "epochs": 1, "max_new": 4,
            "rollout_engine": "auto", "rollout_devices": "", "temperature": 1.0,
            "eval_temperature": 0.0, "top_p": 1.0, "top_k": 0, "stop_token": 1,
            "lr": 1e-3, "weight_decay": 0.0, "optimizer": "adamw", "warmup": 1,
            "grad_clip": 1.0, "clip_low": 0.2, "clip_high": 0.2, "kl_coef": 0.0,
            "reference": "none", "reference_ckpt": "", "train_tasks": 6,
            "eval_tasks": 3, "difficulty": 1, "curriculum_stages": "",
            "sft_steps": 0, "sft_batch_size": 2, "sft_lr": 1e-3,
            "preflight_prompts": 0, "min_preflight_reward": 0.0,
            "max_preflight_reward": 1.0, "min_preflight_active_groups": 0,
            "eval_every": 0, "eval_prompts": 2, "eval_group_size": 1,
            "min_heldout_delta": 0.0, "confidence": 0.95, "bootstrap_samples": 0,
            "require_confidence": False, "max_family_regression": 1.0,
            "max_rollout_tokens": 0, "max_train_seconds": 0, "save_every": 0,
            "verifier_command": (), "verifier_timeout": 5.0, "log_samples": 0,
            "seed": 5, "device": "cpu", "use_ema": False, "vocab": str(source),
        }
        arguments.update(overrides)
        result = rlvr_train.run(
            Namespace(**arguments),
            worker_controls=controls,
            worker_eval_examples=(
                eval_policy() if policy is DEFAULT_POLICY else policy
            ),
        )
        return session, result, run_directory

    invoke.responses = responses
    try:
        yield invoke
    finally:
        for session in sessions:
            session.close()


# --------------------------------------------------------------------------
# The positive control
# --------------------------------------------------------------------------


def test_complete_baseline_evidence_authorizes_the_run_to_mutate(
    armed, mutations
) -> None:
    """Without this, every refusal below is satisfied by a gate stuck shut.

    Asserts the whole chain rather than any link of it: the publication
    happened, exactly once, at the gated step, carrying non-empty typed
    evidence; the session's gate latched; and only then did the optimizer
    mutate.
    """

    session, result, run_directory = armed()

    assert session.step_zero_eval_gate_satisfied
    manifests = published_manifests(run_directory)
    assert len(manifests) == 1, "exactly one attempt-baseline artifact"
    assert manifests[0]["optimizer_step"] == 0
    assert manifests[0]["examples"], "the artifact is not empty"
    assert result["status"] == "complete"
    assert mutations, "the run mutated after the gate opened"


@pytest.mark.parametrize(
    "field",
    [
        "evaluator_component_digest",
        "artifact_renderer_digest",
        "qualitative_sample_digest",
    ],
)
def test_the_policy_refuses_provenance_that_is_not_a_digest(field: str) -> None:
    """A malformed digest is caught where it is built, not where it is compared.

    Each of these three reaches the published manifest, and the controller
    compares the evaluator one against the resolved component. A value that is
    not a `sha256:` digest at all would fail there too — as a provenance
    disagreement thousands of lines from the composition slot that produced it.
    Refusing in the constructor names the field instead.

    Added because the mutation matrix for this file found this validation
    unbound: deleting it left every other test green.
    """

    with pytest.raises(ValueError, match=f"{field} must be a sha256: digest"):
        eval_policy(**{field: "not-a-digest"})


def test_the_shipped_composition_declares_what_the_publisher_demands() -> None:
    """The document under `examples/` is what everything above reads.

    `require_artifact_contract` ties only the port's *type* and *schema*; the
    worker's `EvalExamplesPublisher` additionally demands `append_only` and
    `manifest_sha256`; the controller's arming predicate reads a third field,
    the artifact's own `required`. Nothing ties the three together, which is why
    the HF family declared its port and stayed inert, so this asserts the
    shipped declaration satisfies all of them at once.
    """

    declaration = shipped_artifacts()["eval_examples"]
    assert declaration["required"] is True
    assert declaration["type"] == "eval_examples"
    assert declaration["schema"] == "rwkv-lab.eval-examples.v1"
    assert declaration["immutability"] == "append_only"
    assert declaration["fingerprint"] == "manifest_sha256"


@pytest.mark.parametrize(
    "field,value",
    [
        ("type", "report"),
        ("schema", "rwkv-lab.eval-gallery.v2"),
        ("immutability", "immutable"),
        ("fingerprint", "content_sha256"),
    ],
)
def test_a_declaration_the_publisher_rejects_publishes_nothing(
    armed, mutations, field, value
) -> None:
    """Each of the publisher's four fields is load-bearing on its own.

    Parametrised rather than folded into one case that mutates four fields at
    once: a single case asserting "one of these four is checked" stays green
    once three of them have stopped being checked.
    """

    declarations = shipped_artifacts()
    declarations["eval_examples"][field] = value
    with pytest.raises(
        EvalExamplesError, match="output is not a declared eval-examples v1 artifact"
    ):
        armed(publishes=declarations)
    assert mutations == []


def test_the_published_evidence_carries_the_resolved_evaluator_provenance(
    armed,
) -> None:
    """The manifest's evaluator fields are the ones the controller cross-checks.

    `validate_eval_examples_gate_provenance` requires
    `evaluator.component_digest` to equal the resolved component's
    `descriptor_digest`, and the metric-name set to equal its configured
    metrics. Both reach the trainer through the policy the handler builds off
    the resolved composition, so this asserts the trainer forwards them
    unaltered rather than composing provenance of its own.
    """

    policy = eval_policy(evaluator_component_digest="sha256:" + "c" * 64)
    _, _, run_directory = armed(policy=policy)
    manifest = published_manifests(run_directory)[0]

    assert manifest["evaluator"]["component_digest"] == "sha256:" + "c" * 64
    assert manifest["evaluator"]["metric_names"] == ["eval.pass_at_k", "eval.reward"]
    assert len(manifest["examples"]) == policy.sample_count
    for example in manifest["examples"]:
        assert example["input"] and example["target"] and example["prediction"]


def test_one_example_is_published_for_each_held_out_task(armed) -> None:
    """A held-out task carries a whole rollout group, and identities are unique.

    `EvalExamplesPublisher.publish` refuses a duplicate `heldout_item_id`, so
    the group has to collapse to a single example or nothing publishes at all.
    Running with a group size above one is what makes that collapse observable
    rather than incidental.
    """

    _, _, run_directory = armed(eval_group_size=2, eval_prompts=2)
    manifest = published_manifests(run_directory)[0]
    identities = [example["heldout_item_id"] for example in manifest["examples"]]

    assert identities and len(identities) == len(set(identities))


# --------------------------------------------------------------------------
# 1. Missing — no authority to publish with
# --------------------------------------------------------------------------


def test_a_worker_with_no_evaluation_policy_refuses_before_mutating(
    armed, mutations
) -> None:
    """The gate is required and the worker was handed nothing to satisfy it.

    Before this change that was RLVR's only behaviour, for an authority-side
    reason rather than a missing evidence source: the profile declared no
    `eval_examples` output, so the publication was unreachable however the
    worker was configured. The refusal is kept for the case where the policy
    really is absent, and it fires here rather than thousands of rollout tokens
    later at the first crossing.
    """

    with pytest.raises(ValueError, match="no held-out evaluation policy"):
        armed(policy=None)
    assert mutations == []


# --------------------------------------------------------------------------
# 2. Wrong step — evidence the attempt will never be gated at
# --------------------------------------------------------------------------


def test_a_baseline_the_controller_does_not_gate_refuses_before_mutating(
    armed, mutations
) -> None:
    """A replacement attempt is gated at its resume step, not at zero.

    The trainer starts at step zero here while the controller gates at seven,
    so any evidence it published would name a step this attempt never reaches
    while the boundary refused every mutation waiting for it. Keying to a
    literal `0` is the bug `8f0da3e` fixed in `mage_flow_pretrain.py`; this
    asserts RLVR reads the controller instead of assuming.
    """

    with pytest.raises(
        ValueError, match="resumed at a step the controller does not gate"
    ):
        armed(baseline=7)
    assert mutations == []


# --------------------------------------------------------------------------
# 3. Malformed — evidence that would have to be altered to fit
# --------------------------------------------------------------------------


def test_evidence_too_large_to_publish_is_refused_not_truncated(
    armed, mutations
) -> None:
    """A rollout longer than the evidence bound fails the run closed.

    Truncating would produce a structurally perfect artifact whose prediction
    is not what the policy emitted — valid to every check the publisher makes
    and false to the only reader who matters. The bound refuses instead, and
    names the task so the operator can see which one.
    """

    # Keyed by "*" rather than by a task id: this route generates its own
    # arithmetic task pool when no manifest is given, so a hard-coded id would
    # silently match nothing and the test would pass by not exercising anything.
    armed.responses["*"] = "x" * (rlvr_train.MAXIMUM_EVIDENCE_CHARACTERS + 1)
    with pytest.raises(ValueError, match="is longer than"):
        armed()
    assert mutations == []


# --------------------------------------------------------------------------
# 4. Empty — a summary is not evidence
# --------------------------------------------------------------------------


def test_a_stored_summary_carrying_no_rollouts_cannot_satisfy_the_gate(
    armed, mutations, tmp_path
) -> None:
    """Held-out *scalars* are not held-out *evidence*.

    A resumed RLVR checkpoint carries `baseline_heldout`, the reward summary,
    but not the rollouts it was computed from — so there is no prediction to
    publish and no way to build a non-empty artifact. The publisher's
    `0 < len(request.examples)` bound would refuse the empty one anyway; the
    trainer says the more useful thing first, which is *why* it is empty.
    """

    stored = {
        "arch": {},
        "rlvr_step": 0,
        "rlvr": {
            "baseline_heldout": {
                "reward": 0.5,
                "pass_at_k": 0.5,
                "task_rewards": {"t2": 0.5},
            }
        },
    }
    with pytest.raises(ValueError, match="carries no rollouts"):
        armed(source_blob=stored, resume=str(tmp_path / "source.pt"))
    assert mutations == []
