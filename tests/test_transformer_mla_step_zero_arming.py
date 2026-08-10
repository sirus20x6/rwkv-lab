"""The eight Transformer MLA routes' newly-armed step-zero eval gate.

`card-21d583f6` records that eighteen stateful routes could not arm because
their adapter contracts declared no `eval_examples` output port. This family is
eight of those eighteen, and they arm together because they share one authoring
declaration and one composition contract.

Four things about how that is asserted are deliberate.

**The declaration under test is the shipped one.** `shipped_specs` reads the
template documents out of
`docs/experiment-vm/examples/transformer-mla.recipe-profiles.v1.json` rather
than restating them. A fixture that hand-builds what the shipped catalog
declares is two opinions about one contract, and the seam this family had to
cross is exactly a disagreement between two such fields: the *port's* `required`
(in `rwkv_lab_worker_contract.cpp`) and the *artifact's* `required` (in the
composition document) are independent, and only the second arms the controller.

**Five conditions, not three.** `invocation_requires_step_zero_eval_gate` is
three conjuncts -- `required`, `type`, `schema`. `EvalExamplesPublisher`
separately demands `immutability: append_only` and `fingerprint:
manifest_sha256`, and *nothing checks that those agree*. A document satisfying
the arming predicate but not the publisher arms the controller and then cannot
publish: the route deadlocks at its first crossing and says nothing about why,
which is strictly worse than never arming. `ci_step_zero_arming_gate.py` reads
the three; this file reads all five.

**The refusal had to change shape, and that is the interesting regression.**
Before this family could publish, an unsatisfied gate was terminal and
`refuse_ungated_attempt_baseline` refused it unconditionally, which was right.
The moment the trainer can publish, that same unconditional refusal rejects
every *armed fresh attempt* before it reaches the publication -- the trainer
deadlocking itself over evidence it was about to produce. So the refusal is now
scoped by whether anything downstream can publish, and both sides of that scope
are asserted here.

**Messages, not types.** Every refusal below raises `ValueError`, and would for
half a dozen unrelated reasons. Each case matches the sentence its own condition
emits, so deleting that condition reddens the case rather than letting a
neighbour catch the same input.

WHAT THIS FILE DOES NOT PROVE. It does not run a training step, and it does not
run the controller. `validate_eval_examples_gate_provenance`
(`trainvm/src/eval_examples_contract.cpp`) additionally requires a prior durable
declared scalar and a prior durable checkpoint artifact at the manifest's own
step; those are C++ conditions evaluated by the real service. What is shown here
is that the declaration satisfies every condition the worker-side publisher
imposes, and that the trainer emits its evidence in the order the service
requires -- not that a live service then accepted it.
"""

from __future__ import annotations

import ast
import json
import pathlib
from types import SimpleNamespace

import pytest

from rwkv_lab.train_mla import (
    refuse_ungated_attempt_baseline,
    select_heldout_windows,
)
from rwkv_lab.trainvm_adapters.transformer_mla import (
    PROFILE_ADAPTERS,
    TransformerMLAEvalPolicy,
)
from rwkv_lab.trainvm_worker.controls import WorkerControlRuntime
from rwkv_lab.trainvm_worker.eval_examples import EVAL_EXAMPLES_SCHEMA

REPOSITORY = pathlib.Path(__file__).resolve().parent.parent
CATALOG = (
    REPOSITORY
    / "docs/experiment-vm/examples/transformer-mla.recipe-profiles.v1.json"
)
HANDLERS = REPOSITORY / "src/rwkv_lab/trainvm_adapters/handlers.py"
TRAINER = REPOSITORY / "src/rwkv_lab/train_mla.py"

# The eight contracts registered by the loop in `rwkv_lab_worker_contract.cpp`.
# Derived from `PROFILE_ADAPTERS` rather than restated, so a ninth topology
# added there without a shipped composition reddens `test_every_registered_mla
# _topology_ships_an_arming_composition` instead of passing unnoticed.
CONTRACTS = frozenset(
    "rwkv_lab." + adapter.removeprefix("rwkv-lab.").replace("-", "_") + ".v1.Train"
    for adapter in PROFILE_ADAPTERS.values()
)


def shipped_specs() -> dict[str, dict]:
    """Every shipped MLA template document, keyed by the contract it invokes."""

    catalog = json.loads(CATALOG.read_text(encoding="utf-8"))
    specs: dict[str, dict] = {}
    for recipe in catalog["recipes"]:
        spec = recipe["template_document"]["spec"]
        invoke = spec["workflow"]["nodes"]["train"]["invoke"]
        component = spec["components"][invoke["component"]]
        contract = component["operations"][invoke["operation"]]["contract"]
        assert contract not in specs, f"{contract} is shipped twice"
        specs[contract] = spec
    return specs


def _runtime(**session) -> WorkerControlRuntime:
    """A real control runtime over a session stub with the gate fields set."""

    return WorkerControlRuntime(
        SimpleNamespace(
            **{
                "step_zero_eval_gate_required": False,
                "step_zero_eval_gate_satisfied": False,
                "attempt_baseline_optimizer_step": 0,
                "poll_commands": lambda: (),
                **session,
            }
        ),
        {},
        0,
    )


def _digest(character: str) -> str:
    return "sha256:" + character * 64


def _policy(**overrides) -> TransformerMLAEvalPolicy:
    values = {
        "identity_field": "window_offset",
        "evaluator_component_digest": _digest("a"),
        "metric_names": ("eval.loss", "eval.perplexity"),
        "artifact_renderer_digest": _digest("b"),
        "qualitative_sample_digest": _digest("c"),
        "sample_count": 8,
    }
    values.update(overrides)
    return TransformerMLAEvalPolicy(**values)


# --------------------------------------------------------------------------
# The shipped declaration
# --------------------------------------------------------------------------


def test_every_registered_mla_topology_ships_an_arming_composition() -> None:
    """All eight, not one worked example and seven that still cannot arm.

    Arming is per contract. A single document arms the one contract it invokes
    and leaves the other seven exactly where they were, which is the state this
    card was dispatched to end.
    """

    assert set(shipped_specs()) == CONTRACTS


@pytest.mark.parametrize("contract", sorted(CONTRACTS))
def test_the_shipped_composition_declares_what_the_publisher_demands(
    contract: str,
) -> None:
    """All five conditions, from the two places that impose them.

    Three arm the controller; two more decide whether the worker can then
    publish. Satisfying only the first three is the deadlock this card was
    warned about, and it is invisible to the arming gate.
    """

    spec = shipped_specs()[contract]
    node = spec["workflow"]["nodes"]["train"]
    logical = node["publishes"]["eval_examples"]
    declaration = spec["artifacts"][logical]
    # `invocation_requires_step_zero_eval_gate`, trainvm/src/eval_examples_contract.cpp
    assert declaration["required"] is True
    assert declaration["type"] == "eval_examples"
    assert declaration["schema"] == EVAL_EXAMPLES_SCHEMA
    # `EvalExamplesPublisher.publish`, src/rwkv_lab/trainvm_worker/eval_examples.py
    assert declaration["immutability"] == "append_only"
    assert declaration["fingerprint"] == "manifest_sha256"


@pytest.mark.parametrize("contract", sorted(CONTRACTS))
def test_the_shipped_composition_binds_evidence_to_a_checkpoint(
    contract: str,
) -> None:
    """Eval-examples is checkpoint-bound by construction.

    The publisher carries a checkpoint artifact id and manifest digest, and the
    controller refuses a manifest with no prior durable checkpoint at its own
    step. A document publishing examples and no checkpoint arms a route that
    cannot satisfy its own gate.
    """

    spec = shipped_specs()[contract]
    logical = spec["workflow"]["nodes"]["train"]["publishes"]["checkpoint"]
    assert spec["artifacts"][logical]["type"] == "checkpoint"


@pytest.mark.parametrize("contract", sorted(CONTRACTS))
def test_the_shipped_composition_declares_the_evaluator_provenance(
    contract: str,
) -> None:
    """The four evaluation slots the handler reads unconditionally.

    `_transformer_mla` reads `evaluator`, `qualitative_samples` and
    `artifact_renderer` out of the *resolved* composition with no `get`, because
    `transformer_mla_composition()` declares all of them. A document that
    omitted one would raise a `KeyError` from inside the handler rather than a
    diagnosis, so the shipped documents are checked to carry them.
    """

    components = shipped_specs()[contract]["workflow"]["nodes"]["train"]["invoke"][
        "training"
    ]["components"]
    for slot in ("artifact_renderer", "evaluation_schedule", "evaluator",
                 "qualitative_samples"):
        assert slot in components, f"{contract} omits the {slot} slot"
    metrics = components["evaluator"]["configuration"]["metrics"]
    assert metrics and metrics == sorted(set(metrics))


# --------------------------------------------------------------------------
# The composition-derived policy
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "field",
    [
        "evaluator_component_digest",
        "artifact_renderer_digest",
        "qualitative_sample_digest",
    ],
)
@pytest.mark.parametrize(
    "value",
    [
        "not-a-digest",
        # Right length, right alphabet, wrong algorithm prefix. A validator
        # checking only length and hex accepts this, and the manifest then
        # carries provenance the controller cannot interpret.
        "sha512:" + "a" * 64,
        # Right length and prefix, one character outside the hex alphabet.
        "sha256:" + "a" * 63 + "z",
    ],
)
def test_the_policy_refuses_provenance_that_is_not_a_digest(
    field: str, value: str
) -> None:
    with pytest.raises(ValueError, match="must be a sha256: digest"):
        _policy(**{field: value})


@pytest.mark.parametrize(
    "metrics", [(), ("eval.perplexity", "eval.loss"), ("eval.loss", "eval.loss")]
)
def test_the_policy_refuses_metric_names_it_cannot_compare(metrics) -> None:
    """Empty, unsorted, or duplicated.

    The controller cross-checks the manifest's metric names against the ones
    the resolved evaluator declares. An unordered or duplicated tuple makes two
    equivalent compositions produce different manifests, so the shape is pinned
    rather than merely non-empty.
    """

    with pytest.raises(ValueError, match="sorted and unique"):
        _policy(metric_names=metrics)


def test_the_policy_refuses_a_sample_count_outside_the_manifest_bound() -> None:
    with pytest.raises(ValueError, match="sample_count"):
        _policy(sample_count=513)


# --------------------------------------------------------------------------
# The frozen held-out selection
# --------------------------------------------------------------------------


def test_the_held_out_selection_is_a_pure_function_of_range_and_count() -> None:
    """Two attempts over the same corpus draw the same windows.

    That is what makes a replacement attempt's evidence comparable with the
    attempt it replaced. A selection seeded from `cfg.seed` -- which the
    training stream folds `start_step` into -- would move on resume.
    """

    first = select_heldout_windows(1_000, 1_000_000, 2048, 8)
    assert first == select_heldout_windows(1_000, 1_000_000, 2048, 8)
    assert len(first) == 8
    assert first[0] == 1_000


def test_the_held_out_windows_are_disjoint_and_inside_the_range() -> None:
    offsets = select_heldout_windows(0, 100_000, 1023, 8)
    width = 1024
    assert all(
        following - preceding >= width
        for preceding, following in zip(offsets, offsets[1:])
    )
    assert offsets[-1] + width <= 100_000


def test_a_range_too_small_for_the_selection_is_refused_not_overlapped() -> None:
    """Refuse rather than silently return overlapping or short windows.

    Overlapping windows would publish the same evidence several times under
    different identities, and a short one would run past the evaluation range
    into training tokens -- held-out evidence that is not held out.
    """

    with pytest.raises(ValueError, match="disjoint"):
        select_heldout_windows(0, 4096, 2048, 8)


# --------------------------------------------------------------------------
# The refusal, rescoped
# --------------------------------------------------------------------------


def test_an_armed_attempt_that_can_publish_is_not_refused_before_publishing() -> None:
    """The regression this change exists to avoid.

    An armed fresh attempt arrives with the gate required and unsatisfied --
    that is what "armed" means before the evidence exists. Refusing it here
    would deadlock the route on evidence it was three statements away from
    publishing, which is worse than never arming at all.
    """

    controls = _runtime(
        attempt_baseline_optimizer_step=0,
        step_zero_eval_gate_required=True,
        step_zero_eval_gate_satisfied=False,
    )
    refuse_ungated_attempt_baseline(
        controls, 0, can_publish_baseline_evidence=True
    )


def test_an_armed_attempt_with_nothing_to_publish_with_is_still_refused() -> None:
    """The other side of the same scope.

    A run given no eval-examples policy will never publish, so the gate is
    terminal and the diagnosis belongs here rather than at the first crossing
    after a full model load. The message names the missing policy, so this case
    cannot be satisfied by the neighbouring `does not gate` refusal.
    """

    controls = _runtime(
        attempt_baseline_optimizer_step=4096,
        step_zero_eval_gate_required=True,
        step_zero_eval_gate_satisfied=False,
    )
    with pytest.raises(ValueError) as refusal:
        refuse_ungated_attempt_baseline(
            controls, 4096, can_publish_baseline_evidence=False
        )
    assert "no eval-examples policy" in str(refusal.value)


def test_a_baseline_the_controller_does_not_gate_is_refused_even_when_armed() -> None:
    """Being able to publish does not license starting at the wrong step.

    The two conditions are independent, and the ability to publish must not
    weaken the one that has nothing to do with publication.
    """

    controls = _runtime(
        attempt_baseline_optimizer_step=4096,
        step_zero_eval_gate_required=True,
        step_zero_eval_gate_satisfied=False,
    )
    with pytest.raises(ValueError) as refusal:
        refuse_ungated_attempt_baseline(
            controls, 2048, can_publish_baseline_evidence=True
        )
    assert "does not gate" in str(refusal.value)


# --------------------------------------------------------------------------
# The producer wiring
# --------------------------------------------------------------------------


def _function(path: pathlib.Path, name: str) -> ast.FunctionDef:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{path}: no function named {name}")


def test_the_handler_hands_the_trainer_a_composition_derived_policy() -> None:
    """Read from the resolved composition, and actually delivered.

    Two failures this catches are different: building the policy and dropping
    it on the floor arms a route whose trainer has nothing to publish with, and
    building it from the authored document rather than the resolved composition
    makes the manifest disagree with what the registry resolved.
    """

    handler = _function(HANDLERS, "_transformer_mla")
    source = ast.unparse(handler)
    assert "TransformerMLAEvalPolicy(" in source
    assert "worker_eval_examples=eval_policy" in source
    assert "components.composition.components['evaluator'].descriptor_digest" in source


def test_the_trainer_publishes_its_baseline_evidence_before_it_mutates() -> None:
    """Publication precedes every optimizer mutation in `train`.

    Ordering is the whole property. Publishing after a mutation satisfies the
    controller for evidence that no longer describes the parameters it was
    gathered from.
    """

    train = _function(TRAINER, "train")
    publications = [
        node.lineno
        for node in ast.walk(train)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "publish_attempt_baseline_examples"
    ]
    mutations = [
        node.lineno
        for node in ast.walk(train)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "step"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id.startswith("optimizer")
    ]
    assert publications, "`train` never publishes attempt-baseline evidence"
    assert mutations, "the mutation detector found nothing to order against"
    assert max(publications) < min(mutations)


def test_the_baseline_publication_is_keyed_to_the_controller_not_to_zero() -> None:
    """A literal zero deadlocks every replacement attempt.

    It is the bug `mage_flow_pretrain.py` shipped: evidence owed at a step the
    resumed attempt will never reach again, while `pre_optimizer_step` refuses
    every mutation until it exists. A branch keyed to zero that never fires
    looks exactly like a branch with nothing to do.
    """

    assignments = [
        node.value
        for node in ast.walk(_function(TRAINER, "train"))
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "gate_baseline"
            for target in node.targets
        )
        and not (isinstance(node.value, ast.Constant) and node.value.value is None)
    ]
    assert assignments, "`train` never decides a gate baseline"
    # Every non-`None` assignment must read the controller. A single literal
    # among them is the bug, so this is `all`, not `any` -- and asserting the
    # string appears somewhere in `train` would pass with the literal in place,
    # because the reconnect branch mentions the same attribute.
    for value in assignments:
        assert isinstance(value, ast.Attribute), (
            f"gate_baseline is assigned a {type(value).__name__} at line "
            f"{value.lineno}, not a controller reading"
        )
        assert value.attr == "attempt_baseline_optimizer_step"
