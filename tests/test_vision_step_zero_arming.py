"""The four vision routes' newly-armed step-zero eval gate.

Four separate assertions, because arming needs four separate things and doing
only the first is worse than doing none of them -- a route that arms its
controller-side gate with no producer behind it deadlocks at its first
`pre_optimizer_step` crossing and says nothing about why:

1. the `eval_examples` output port on each adapter contract, in
   `trainvm/src/rwkv_lab_worker_contract.cpp` (checked here through the
   generated pin, so this suite needs no compiler);
2. a shipped composition document that declares the artifact with all five
   conditions -- the three the arming predicate reads and the two
   `EvalExamplesPublisher` demands separately, which the arming predicate does
   not check;
3. a producer that publishes a step-0 scalar baseline, a baseline checkpoint,
   and non-empty eval-examples evidence keyed to
   `controls.attempt_baseline_optimizer_step` and never to a literal zero;
4. evaluator provenance read from the *resolved* composition.

No test in this repository executes `train()` in any of the four vision
trainers, and none can cheaply: each needs staged MoonViT/SigLIP2/DINOv2/SAM
caches on disk, a frozen caption stack, and `torch.autocast("cuda")`. That is
stated here rather than implied, because a structural suite that reads like a
behavioural one is worse than an honest one. What is behavioural here is the
part that can be: the policy dataclass, the frozen selection, the refusal, the
evidence builders, and the composition accessor they all depend on.
"""

from __future__ import annotations

import ast
import json
import pathlib

import pytest

from rwkv_lab.trainvm_adapters.vision_eval import VisionEvalPolicy
from rwkv_lab.vision_step_zero import (
    attempt_baseline_gate,
    bounded_text,
    caption_eval_examples,
    reconstruction_eval_examples,
    refuse_ungated_attempt_baseline,
    select_heldout_indices,
    selection_digest,
)

REPOSITORY = pathlib.Path(__file__).resolve().parents[1]
EXAMPLES = REPOSITORY / "docs/experiment-vm/examples"
PIN = REPOSITORY / "docs/experiment-vm/step-zero-arming.v1.json"
SOURCE = REPOSITORY / "src/rwkv_lab"

EVAL_EXAMPLES_SCHEMA = "rwkv-lab.eval-examples.v1"

# The four contracts and the document shipped to arm each. Written out rather
# than discovered, so a fifth vision route added without a composition fails
# `test_every_registered_vision_route_ships_an_arming_composition` instead of
# passing unnoticed.
ARMING = {
    "rwkv_lab.vision_frozen_adapter.v1.Train": (
        "vision-representation-ab.json",
        "moonvit_arm",
    ),
    "rwkv_lab.vision_native_head.v1.Train": ("vision-native-head.json", "train"),
    "rwkv_lab.vision_rwkv_student.v1.Train": ("vision-rwkv-student.json", "train"),
    "rwkv_lab.vision_teacher_compressor.v1.Train": (
        "vision-teacher-compressor.json",
        "train",
    ),
}

# Which trainer produces each route's evidence, and the family string its
# refusal carries. The family string is what a deadlocked operator reads.
PRODUCERS = {
    "rwkv_lab.vision_frozen_adapter.v1.Train": (
        "vision_train.py",
        "Vision frozen adapter",
    ),
    "rwkv_lab.vision_native_head.v1.Train": (
        "vision_native_train.py",
        "Vision native head",
    ),
    "rwkv_lab.vision_rwkv_student.v1.Train": (
        "vision_rwkv_student_train.py",
        "Vision RWKV student",
    ),
    "rwkv_lab.vision_teacher_compressor.v1.Train": (
        "vision_teacher_compressor.py",
        "Vision teacher compressor",
    ),
}


def _digest(character: str) -> str:
    return "sha256:" + character * 64


def _policy(**overrides) -> VisionEvalPolicy:
    values = {
        "identity_field": "image",
        "evaluator_component_digest": _digest("a"),
        "metric_names": ("eval.loss", "eval.perplexity"),
        "artifact_renderer_digest": _digest("b"),
        "qualitative_sample_digest": _digest("c"),
        "sample_count": 8,
    }
    values.update(overrides)
    return VisionEvalPolicy(**values)


class _Controls:
    """A controls double with no refusal logic of its own.

    Deliberately dumb. A double that re-implements the condition under test
    passes whatever the test asserts; this one only reports the three values the
    production refusal reads.
    """

    def __init__(self, *, baseline: int, required: bool, satisfied: bool) -> None:
        self.attempt_baseline_optimizer_step = baseline
        self.step_zero_eval_gate_required = required
        self.step_zero_eval_gate_satisfied = satisfied


def _document(name: str) -> dict:
    return json.loads((EXAMPLES / name).read_text(encoding="utf-8"))


def _function(path: pathlib.Path, name: str) -> ast.FunctionDef:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{path.name} declares no function {name}")


# --------------------------------------------------------------------------
# The port
# --------------------------------------------------------------------------


@pytest.mark.parametrize("contract", sorted(ARMING))
def test_the_adapter_declares_a_required_eval_examples_port(contract: str) -> None:
    pin = json.loads(PIN.read_text(encoding="utf-8"))
    entry = next(
        profile for profile in pin["profiles"] if profile["contract"] == contract
    )
    port = entry["eval_examples_output"]
    assert port is not None, f"{contract} declares no eval_examples output port"
    assert port["output_name"] == "eval_examples"
    assert port["required"] is True
    assert port["artifact_schema"] == EVAL_EXAMPLES_SCHEMA
    # Checkpoint-bound by construction: `EvalExamplesPublicationRequest`
    # requires a checkpoint artifact id and manifest digest.
    assert entry["checkpoint_outputs"], f"{contract} declares no checkpoint output"


# --------------------------------------------------------------------------
# The shipped declaration
# --------------------------------------------------------------------------


@pytest.mark.parametrize("contract", sorted(ARMING))
def test_every_registered_vision_route_ships_an_arming_composition(
    contract: str,
) -> None:
    name, node_name = ARMING[contract]
    document = _document(name)
    node = document["spec"]["workflow"]["nodes"][node_name]
    assert node["invoke"]["component"]
    component = document["spec"]["components"][node["invoke"]["component"]]
    assert component["operations"][node["invoke"]["operation"]]["contract"] == contract


@pytest.mark.parametrize("contract", sorted(ARMING))
def test_the_shipped_composition_declares_what_the_publisher_demands(
    contract: str,
) -> None:
    """All five conditions, from the shipped document rather than a fixture.

    The arming predicate reads the first three. `EvalExamplesPublisher.__init__`
    reads `type` and `schema` too and *additionally* demands `append_only` and
    `manifest_sha256`, which the arming gate does not check -- so a document
    satisfying only the predicate arms the controller and then cannot publish.
    """

    name, node_name = ARMING[contract]
    document = _document(name)
    logical = document["spec"]["workflow"]["nodes"][node_name]["publishes"][
        "eval_examples"
    ]
    declaration = document["spec"]["artifacts"][logical]
    assert declaration["required"] is True
    assert declaration["type"] == "eval_examples"
    assert declaration["schema"] == EVAL_EXAMPLES_SCHEMA
    assert declaration["immutability"] == "append_only"
    assert declaration["fingerprint"] == "manifest_sha256"


@pytest.mark.parametrize("contract", sorted(ARMING))
def test_the_shipped_composition_binds_evidence_to_a_checkpoint(
    contract: str,
) -> None:
    name, node_name = ARMING[contract]
    document = _document(name)
    publishes = document["spec"]["workflow"]["nodes"][node_name]["publishes"]
    assert "checkpoint" in publishes, (
        f"{contract} publishes eval-examples with no checkpoint beside it; the "
        "artifact is checkpoint-bound by construction"
    )
    checkpoint = document["spec"]["artifacts"][publishes["checkpoint"]]
    assert checkpoint["type"] == "checkpoint"


@pytest.mark.parametrize("contract", sorted(ARMING))
def test_the_shipped_composition_declares_the_evaluator_provenance(
    contract: str,
) -> None:
    """Everything `_vision_eval_policy` reads, and the metrics it will publish.

    `validate_eval_examples_gate_provenance` cross-checks the manifest's
    evaluator digest and metric names against the resolved composition, and
    `publish_if_declared` silently drops a metric the plan does not declare --
    so a composition naming a metric the observability plan omits arms a gate it
    can never satisfy.
    """

    name, node_name = ARMING[contract]
    document = _document(name)
    components = document["spec"]["workflow"]["nodes"][node_name]["invoke"]["training"][
        "components"
    ]
    for slot in (
        "artifact_renderer",
        "evaluation_schedule",
        "evaluator",
        "qualitative_samples",
    ):
        assert slot in components, f"{contract} omits the {slot} slot"
    metrics = tuple(components["evaluator"]["configuration"]["metrics"])
    assert metrics == tuple(sorted(set(metrics))), (
        f"{contract} evaluator metrics are unsorted or duplicated; "
        "VisionEvalPolicy refuses that"
    )
    declared = {
        entry["name"]: entry for entry in document["spec"]["observability"]["metrics"]
    }
    for metric in metrics:
        assert metric in declared, (
            f"{contract} evaluates {metric} but the observability plan does not "
            "declare it, so publish_if_declared would drop it"
        )
        assert declared[metric]["step_domain"] == "optimizer_step"
    assert components["qualitative_samples"]["configuration"]["sample_count"] >= 1


def test_the_compressor_does_not_claim_a_caption_it_cannot_produce() -> None:
    """The one vision route with no text anywhere in its objective.

    `rwkv_lab_worker_contract.cpp` withholds `caption_triplet` from it for that
    reason, so a document choosing that renderer would be refused by the
    registry -- but the failure would arrive at dispatch. Say it here.
    """

    name, node_name = ARMING["rwkv_lab.vision_teacher_compressor.v1.Train"]
    components = _document(name)["spec"]["workflow"]["nodes"][node_name]["invoke"][
        "training"
    ]["components"]
    assert components["artifact_renderer"]["key"]["name"] == "evidence_envelope"


# --------------------------------------------------------------------------
# The composition-derived policy
# --------------------------------------------------------------------------


def test_the_policy_accepts_what_a_resolved_composition_supplies() -> None:
    policy = _policy()
    assert policy.sample_count == 8
    assert policy.metric_names == ("eval.loss", "eval.perplexity")


@pytest.mark.parametrize(
    "field",
    [
        "evaluator_component_digest",
        "artifact_renderer_digest",
        "qualitative_sample_digest",
    ],
)
def test_the_policy_refuses_provenance_that_is_not_a_digest(field: str) -> None:
    with pytest.raises(ValueError):
        _policy(**{field: "sha256:not-a-digest"})
    with pytest.raises(ValueError):
        _policy(**{field: _digest("a")[:-1]})


@pytest.mark.parametrize(
    "metrics",
    [(), ("eval.perplexity", "eval.loss"), ("eval.loss", "eval.loss"), ("",)],
)
def test_the_policy_refuses_metric_names_it_cannot_compare(metrics) -> None:
    with pytest.raises(ValueError):
        _policy(metric_names=metrics)


@pytest.mark.parametrize("count", [0, 513, True, "8"])
def test_the_policy_refuses_a_sample_count_outside_the_manifest_bound(count) -> None:
    with pytest.raises(ValueError):
        _policy(sample_count=count)


def test_the_policy_refuses_an_empty_identity_field() -> None:
    with pytest.raises(ValueError):
        _policy(identity_field="")


# --------------------------------------------------------------------------
# The frozen held-out selection
# --------------------------------------------------------------------------


def test_the_held_out_selection_is_a_pure_function_of_population_and_count() -> None:
    assert select_heldout_indices(100, 8) == select_heldout_indices(100, 8)
    assert select_heldout_indices(100, 8) != select_heldout_indices(200, 8)
    assert select_heldout_indices(100, 8) != select_heldout_indices(100, 7)
    # The stride is integer, so a population that grows without crossing a
    # stride boundary leaves the selection where it was. That is a property
    # rather than a defect -- a manifest gaining one row must not move an
    # already-published held-out identity -- and stating it here stops a later
    # reader from "fixing" it into a hash of the population size.
    assert select_heldout_indices(100, 8) == select_heldout_indices(101, 8)


def test_the_held_out_rows_are_distinct_and_inside_the_population() -> None:
    chosen = select_heldout_indices(97, 8)
    assert len(chosen) == len(set(chosen)) == 8
    assert all(0 <= index < 97 for index in chosen)


def test_a_population_too_small_for_the_selection_is_refused_not_repeated() -> None:
    with pytest.raises(ValueError):
        select_heldout_indices(4, 8)
    with pytest.raises(ValueError):
        select_heldout_indices(8, 0)


def test_the_selection_digest_is_stable_and_order_sensitive() -> None:
    assert selection_digest(["a", "b"]) == selection_digest(["a", "b"])
    assert selection_digest(["a", "b"]) != selection_digest(["b", "a"])
    assert selection_digest({"a": 1, "b": 2}) == selection_digest({"b": 2, "a": 1})
    assert selection_digest(["a"]).startswith("sha256:")
    assert len(selection_digest(["a"])) == 71


# --------------------------------------------------------------------------
# The refusal
# --------------------------------------------------------------------------


def test_an_armed_attempt_that_can_publish_is_not_refused_before_publishing() -> None:
    """The trap this refusal has to avoid.

    Before these routes could publish, an unsatisfied gate was terminal. The
    moment they can, an unconditional refusal rejects every armed *fresh*
    attempt before it reaches the publication it was about to make.
    """

    refuse_ungated_attempt_baseline(
        _Controls(baseline=0, required=True, satisfied=False),
        0,
        family="Vision native head",
        can_publish_baseline_evidence=True,
    )


def test_an_armed_attempt_with_nothing_to_publish_with_is_still_refused() -> None:
    with pytest.raises(ValueError, match="no eval-examples policy"):
        refuse_ungated_attempt_baseline(
            _Controls(baseline=0, required=True, satisfied=False),
            0,
            family="Vision native head",
            can_publish_baseline_evidence=False,
        )


def test_a_resume_the_controller_does_not_gate_is_refused_either_way() -> None:
    for publishable in (False, True):
        with pytest.raises(ValueError, match="does not gate"):
            refuse_ungated_attempt_baseline(
                _Controls(baseline=400, required=True, satisfied=False),
                0,
                family="Vision RWKV student",
                can_publish_baseline_evidence=publishable,
            )


def test_a_satisfied_gate_and_a_standalone_run_are_both_admitted() -> None:
    refuse_ungated_attempt_baseline(
        _Controls(baseline=7, required=True, satisfied=True), 7, family="x"
    )
    refuse_ungated_attempt_baseline(None, 12, family="x")


def test_the_gate_step_is_the_controllers_baseline_and_never_a_literal_zero() -> None:
    assert attempt_baseline_gate(None) is None
    assert (
        attempt_baseline_gate(_Controls(baseline=400, required=False, satisfied=False))
        is None
    )
    # Already durable: a reconnecting or replacement worker must not republish.
    assert (
        attempt_baseline_gate(_Controls(baseline=400, required=True, satisfied=True))
        is None
    )
    assert (
        attempt_baseline_gate(_Controls(baseline=400, required=True, satisfied=False))
        == 400
    )


# --------------------------------------------------------------------------
# The evidence
# --------------------------------------------------------------------------


def test_caption_evidence_carries_the_triple_and_unique_identities() -> None:
    examples = caption_eval_examples(
        [
            ("/images/a.png", "describe", "a cat", "a dog"),
            ("/images/b.png", "describe", "a tree", "a tree"),
        ],
        optimizer_step=400,
    )
    assert len(examples) == 2
    assert len({example.example_id for example in examples}) == 2
    assert len({example.heldout_item_id for example in examples}) == 2
    first = examples[0]
    assert first.heldout_item_id == "/images/a.png"
    assert "step:400" in first.example_id
    assert first.target[0].text == "a cat"
    assert first.prediction[0].text == "a dog"
    assert first.heldout_item_digest.startswith("sha256:")


def test_caption_evidence_identifies_the_question_not_the_answer() -> None:
    """The item digest must not move when the model's output moves.

    Otherwise two attempts of one run carry different held-out identities and
    stop being comparable, which is the whole reason the selection is frozen.
    """

    left = caption_eval_examples([("i", "p", "ref", "one")], optimizer_step=0)
    right = caption_eval_examples([("i", "p", "ref", "two")], optimizer_step=0)
    assert left[0].heldout_item_digest == right[0].heldout_item_digest
    assert left[0].heldout_item_id == right[0].heldout_item_id


def test_reconstruction_evidence_is_structured_rather_than_invented_text() -> None:
    examples = reconstruction_eval_examples(
        [("cache-a.pt", {"teacher_norm": 1.5}, {"latent_norm": 1.4, "loss": 0.2})],
        optimizer_step=250,
        schema="rwkv-lab.vision-teacher-reconstruction.v1",
    )
    assert len(examples) == 1
    example = examples[0]
    assert example.target[0].kind == "structured"
    assert example.prediction[0].kind == "structured"
    assert example.target[0].schema == "rwkv-lab.vision-teacher-reconstruction.v1"
    assert example.prediction[0].value["loss"] == pytest.approx(0.2)
    assert all(part.text is None for part in example.target)


@pytest.mark.parametrize("step", [-1, True, "0"])
def test_evidence_refuses_a_step_that_is_not_a_step(step) -> None:
    with pytest.raises(ValueError):
        caption_eval_examples([("i", "p", "t", "p")], optimizer_step=step)
    with pytest.raises(ValueError):
        reconstruction_eval_examples(
            [("i", {"a": 1.0}, {"b": 2.0})], optimizer_step=step, schema="s"
        )


def test_bounded_text_fits_the_manifest_budget_and_the_publishers_charset() -> None:
    assert bounded_text("a\nb\r\nc") == "a b c"
    assert bounded_text("") == "(empty)"
    assert bounded_text("   ") == "(empty)"
    long = bounded_text("x" * 5000)
    assert len(long) == 512
    assert long.endswith("…")


# --------------------------------------------------------------------------
# The producer wiring
# --------------------------------------------------------------------------


@pytest.mark.parametrize("contract", sorted(PRODUCERS))
def test_the_trainer_refuses_an_ungated_baseline_before_it_mutates(
    contract: str,
) -> None:
    """Anchored on the trainer's own `train`, not on any call anywhere in it.

    A file-wide `"refuse_ungated_attempt_baseline" in source` check is satisfied
    by an import line, and an anchor that walks the module finds a call in a
    helper. This walks `train`'s body specifically.
    """

    module, family = PRODUCERS[contract]
    train = _function(SOURCE / module, "train")
    calls = [
        node
        for node in ast.walk(train)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "refuse_ungated_attempt_baseline"
    ]
    assert len(calls) == 1, (
        f"{module}:train must call refuse_ungated_attempt_baseline exactly "
        f"once, found {len(calls)}"
    )
    keywords = {keyword.arg: keyword.value for keyword in calls[0].keywords}
    assert "family" in keywords
    assert ast.literal_eval(keywords["family"]) == family
    scope = keywords["can_publish_baseline_evidence"]
    assert isinstance(scope, ast.Compare), (
        "the refusal must be scoped by whether anything downstream can publish; "
        "an unconditional refusal deadlocks every armed fresh attempt"
    )
    assert isinstance(scope.left, ast.Name)
    assert scope.left.id == "worker_eval_examples"


@pytest.mark.parametrize("contract", sorted(PRODUCERS))
def test_the_trainer_publishes_its_baseline_evidence(contract: str) -> None:
    module, _ = PRODUCERS[contract]
    train = _function(SOURCE / module, "train")
    calls = [
        node
        for node in ast.walk(train)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "publish_attempt_baseline_evidence"
    ]
    assert len(calls) == 1, (
        f"{module}:train must publish attempt-baseline evidence exactly once, "
        f"found {len(calls)}"
    )
    keywords = {keyword.arg for keyword in calls[0].keywords}
    for required in (
        "worker_controls",
        "policy",
        "baseline",
        "series_id",
        "identities",
        "selector",
        "examples",
        "stage_checkpoint",
    ):
        assert required in keywords, f"{module} omits {required}"


@pytest.mark.parametrize("contract", sorted(PRODUCERS))
def test_the_baseline_publication_is_keyed_to_the_controller_not_to_zero(
    contract: str,
) -> None:
    """The bug `mage_flow_pretrain.py` shipped and 8f0da3e fixed.

    All four of these routes resume, so a publication keyed to a literal zero
    would owe evidence at a step a replacement attempt never reaches again while
    `pre_optimizer_step` refuses every mutation until it exists.
    """

    module, _ = PRODUCERS[contract]
    train = _function(SOURCE / module, "train")
    call = next(
        node
        for node in ast.walk(train)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "publish_attempt_baseline_evidence"
    )
    baseline = next(
        keyword.value for keyword in call.keywords if keyword.arg == "baseline"
    )
    assert isinstance(baseline, ast.Name) and baseline.id == "gate_baseline", (
        f"{module} keys its baseline publication to something other than the "
        "controller-derived gate step"
    )
    assigned = [
        node
        for node in ast.walk(train)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "gate_baseline"
            for target in node.targets
        )
    ]
    sources = [ast.dump(node.value) for node in assigned]
    assert any("attempt_baseline_gate" in source for source in sources), (
        f"{module} does not derive gate_baseline from attempt_baseline_gate"
    )
    assert not any(
        isinstance(node.value, ast.Constant) and node.value.value == 0
        for node in assigned
    ), f"{module} assigns a literal zero to gate_baseline"


@pytest.mark.parametrize(
    "handler",
    [
        "_vision_teacher_compressor",
        "_vision_frozen_adapter",
        "_vision_native_head",
        "_vision_rwkv_student",
    ],
)
def test_the_handler_hands_the_trainer_a_composition_derived_policy(
    handler: str,
) -> None:
    """Anchored on the individual handler, not on the module.

    Four handlers live in one file, so a module-wide search proves only that
    *some* handler passes a policy -- and a run of PR #219's own harness found
    exactly that failure mode, an anchor that hit a neighbouring family's
    handler.
    """

    handlers = SOURCE / "trainvm_adapters/handlers.py"
    function = _function(handlers, handler)
    calls = [
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Call)
        and any(
            keyword.arg == "worker_eval_examples"
            and isinstance(keyword.value, ast.Call)
            and isinstance(keyword.value.func, ast.Name)
            and keyword.value.func.id == "_vision_eval_policy"
            for keyword in node.keywords
        )
    ]
    assert len(calls) == 1, (
        f"{handler} must hand its trainer exactly one composition-derived "
        f"eval-examples policy, found {len(calls)}"
    )


def test_the_policy_helper_reads_the_resolved_composition_unconditionally() -> None:
    helper = _function(SOURCE / "trainvm_adapters/handlers.py", "_vision_eval_policy")
    assert not [
        node for node in ast.walk(helper) if isinstance(node, (ast.If, ast.IfExp))
    ], (
        "_vision_eval_policy must not be conditional: a route that silently "
        "skipped the policy would arm its own composition's gate and then be "
        "unable to publish the evidence that composition demands"
    )
    attributes = {
        node.attr for node in ast.walk(helper) if isinstance(node, ast.Attribute)
    }
    assert "descriptor_digest" in attributes
    assert {"evaluator", "qualitative_samples"} <= {
        node.func.attr
        for node in ast.walk(helper)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
