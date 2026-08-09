"""Ordering and coverage for the Transformer MLA family's optimizer boundary.

``tests/test_step_zero_interception_enumeration.py`` can only see that
``src/rwkv_lab/train_mla.py`` *names* ``pre_optimizer_step`` somewhere. It is a
static reader with one question -- "does a boundary call appear in a producer
this profile resolves" -- and it credits the file, not the site. All eight
``rwkv-lab.transformer-mla*`` profiles share this one producer, so a boundary
placed before one of its three optimizer mutations and after the other two
would satisfy that reader for all eight at once. That is the specific way this
route can lie, and it is what this file exists to prevent.

Why these tests are structural where the HF ones are not
-------------------------------------------------------
``tests/test_hf_multimodal_sft_mutation_boundary.py`` drives the real engine
and asserts an observed interleaving. That is the stronger instrument and it is
the one to prefer. It is not available here: no test in this repository
executes ``train_mla.train()`` and none can cheaply -- it wants a converted MLA
patch directory on disk, a memory-mapped token corpus, a HuggingFace backbone
and CUDA. Writing that harness is real work; a harness assembled in a hurry
would be the thing under test rather than the trainer.

So the structural half below asserts the strongest claim available without one:
that in the trainer's own syntax tree the crossing **dominates and precedes**
*every* optimizer mutation, inside an installed sentinel, keyed to a step the
loop computes, declaring a count that covers all three receivers. The runtime
half then exercises the composition the trainer builds -- a real
``OptimizerMutationSentinel``, a real ``WorkerControlRuntime`` boundary, real
``torch`` optimizers and a real ``LRScheduler`` -- for the properties syntax
cannot reach: that an extra mutation is refused, that a miscounted crossing
fails closed in both directions, that a scheduler step is not a parameter
mutation, and that a refusing boundary leaves the sentinel disarmed.

The AST helpers mirror ``tests/test_vision_mutation_boundary.py`` deliberately
rather than being imported from it -- test modules are not importable from each
other under this repository's layout. They differ in one substantive way: every
"the mutation" singular there is a set here, because this trainer has three
mutation receivers rather than one.
"""

from __future__ import annotations

import ast
import pathlib
from types import SimpleNamespace

import pytest
import torch

from rwkv_lab.train_mla import refuse_ungated_attempt_baseline
from rwkv_lab.trainvm_worker import MutationSentinelError, OptimizerMutationSentinel
from rwkv_lab.trainvm_worker.controls import WorkerControlError, WorkerControlRuntime

REPOSITORY = pathlib.Path(__file__).resolve().parent.parent
TRAINER = "src/rwkv_lab/train_mla.py"

BOUNDARY = "pre_optimizer_step"

# The three receivers whose `.step()` mutates parameters. `optimizer` is always
# constructed; `optimizer_host` is the engram route's second optimizer-category
# component (its big sparse host embedding tables); `optimizer_aux` is the
# standalone Prodigy branch. `trainvm_scheduler` is deliberately absent -- see
# `test_a_learning_rate_schedule_step_is_not_a_parameter_mutation`.
MUTATION_RECEIVERS = ("optimizer", "optimizer_host", "optimizer_aux")

# Every `.step()` receiver in the module, mutating or not, each classified once.
# Asserted as a set below so a fourth optimizer added later cannot slip past a
# count that was written when there were three.
#
#   trainvm_scheduler    LRScheduler; rewrites param_group["lr"]
#   trainvm_weight_decay ConstantWeightDecaySchedule; a plain dataclass
#   plateau_ctrl         _PlateauLRController, defined in this module; returns
#                        an LR multiplier from an eval metric
#   worker_step_profiler WorkerStepProfiler; telemetry
#
# None of the four is a `torch.optim.Optimizer`, so none reaches the
# process-global pre-hook, and none moves a parameter.
NON_MUTATING_RECEIVERS = (
    "trainvm_scheduler",
    "trainvm_weight_decay",
    "plateau_ctrl",
    "worker_step_profiler",
)
STEP_RECEIVERS = MUTATION_RECEIVERS + NON_MUTATING_RECEIVERS


def _tree() -> ast.Module:
    return ast.parse((REPOSITORY / TRAINER).read_text())


def _statement_paths(tree: ast.Module) -> dict[int, tuple[tuple[str, int], ...]]:
    """Position of every statement as a path of ``(body field, index)`` pairs.

    Line numbers cannot answer "does this precede that": a call on an earlier
    line can sit in a branch the mutation's branch never runs. A path built from
    the body lists themselves can, because two statements are comparable only
    where they share an enclosing body.
    """

    paths: dict[int, tuple[tuple[str, int], ...]] = {}

    def visit(node: ast.AST, prefix: tuple[tuple[str, int], ...]) -> None:
        for field, value in ast.iter_fields(node):
            if isinstance(value, list):
                for index, item in enumerate(value):
                    if isinstance(item, ast.stmt):
                        path = prefix + ((field, index),)
                        paths[id(item)] = path
                        visit(item, path)
                    elif isinstance(item, ast.AST):
                        visit(item, prefix)
            elif isinstance(value, ast.AST):
                visit(value, prefix)

    visit(tree, ())
    return paths


def _enclosing_statements(tree: ast.Module) -> dict[int, ast.stmt]:
    """Map every node to the statement it belongs to."""

    enclosing: dict[int, ast.stmt] = {}

    def visit(node: ast.AST, statement: ast.stmt | None) -> None:
        for child in ast.iter_child_nodes(node):
            current = child if isinstance(child, ast.stmt) else statement
            if current is not None:
                enclosing[id(child)] = current
            visit(child, current)

    visit(tree, None)
    return enclosing


def _calls(tree: ast.Module) -> list[ast.Call]:
    return [node for node in ast.walk(tree) if isinstance(node, ast.Call)]


def _attribute_calls(tree: ast.Module, attribute: str) -> list[ast.Call]:
    return [
        call
        for call in _calls(tree)
        if isinstance(call.func, ast.Attribute) and call.func.attr == attribute
    ]


def _step_calls(tree: ast.Module) -> list[ast.Call]:
    return [
        call
        for call in _attribute_calls(tree, "step")
        if isinstance(call.func.value, ast.Name)
    ]


def _mutations(tree: ast.Module) -> list[ast.Call]:
    return [
        call
        for call in _step_calls(tree)
        if call.func.value.id in MUTATION_RECEIVERS
    ]


def _crossing_helpers(tree: ast.Module) -> list[ast.FunctionDef]:
    """Nested functions whose body arms a sentinel."""

    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        and any(
            isinstance(call.func, ast.Attribute) and call.func.attr == "cross"
            for call in _calls(node)
        )
        and node.name != "train"
    ]


def _crossings(tree: ast.Module) -> list[ast.Call]:
    """Call *sites* that cross the boundary, not the helper that implements it.

    The trainer routes its crossing through a local ``cross_mutation_boundary``
    so the sentinel instance and the boundary lambda are written once. A first
    version of this helper counted that function's own ``sentinel.cross(...)``
    as a crossing -- and because the definition sits lexically above the loop,
    it "preceded" every mutation whether or not anything called it. Deleting
    the call site, moving it below the mutations, and replacing it with a
    direct controller call then all left the ordering tests green. The
    definition is not a crossing; invoking it is.

    Which spelling the trainer uses at the site is still style: a direct
    ``sentinel.cross(...)`` in the loop counts too.
    """

    helpers = _crossing_helpers(tree)
    helper_names = {helper.name for helper in helpers}
    inside_helper = {
        id(call) for helper in helpers for call in _calls(helper)
    }
    return [
        call
        for call in _calls(tree)
        if id(call) not in inside_helper
        and (
            (isinstance(call.func, ast.Attribute) and call.func.attr == "cross")
            or (isinstance(call.func, ast.Name) and call.func.id in helper_names)
        )
    ]


def _precedes(
    paths: dict[int, tuple[tuple[str, int], ...]],
    enclosing: dict[int, ast.stmt],
    earlier: ast.Call,
    later: ast.Call,
) -> bool:
    """True when ``earlier``'s statement runs before ``later``'s on every path."""

    first = paths[id(enclosing[id(earlier)])]
    second = paths[id(enclosing[id(later)])]
    for left, right in zip(first, second):
        if left == right:
            continue
        # Divergence in different body lists (a `try` body versus its handler,
        # say) is not an ordering at all, so refuse to call it one.
        return left[0] == right[0] and left[1] < right[1]
    # One statement encloses the other, which is also not a sibling ordering.
    return False


def _enclosing_installed_names(tree: ast.Module, target: ast.Call) -> set[str]:
    """Names whose ``installed()`` block lexically contains ``target``."""

    names: set[str] = set()

    def visit(node: ast.AST, active: frozenset[str]) -> None:
        if node is target:
            names.update(active)
            return
        if isinstance(node, ast.With):
            entered = {
                item.context_expr.func.value.id
                for item in node.items
                if isinstance(item.context_expr, ast.Call)
                and isinstance(item.context_expr.func, ast.Attribute)
                and item.context_expr.func.attr == "installed"
                and isinstance(item.context_expr.func.value, ast.Name)
            }
            if entered:
                for statement in node.body:
                    visit(statement, active | entered)
                for other in ast.iter_child_nodes(node):
                    if other not in node.body:
                        visit(other, active)
                return
        for child in ast.iter_child_nodes(node):
            visit(child, active)

    visit(tree, frozenset())
    return names


# --------------------------------------------------------------------------
# The structural half: where the crossing sits relative to the mutations
# --------------------------------------------------------------------------


def test_the_loop_steps_exactly_the_receivers_this_file_reasons_about() -> None:
    """The inventory every claim below is stated over.

    A fourth optimizer added later would be guarded by nothing this file
    asserts and would make the counting reasoning quietly wrong rather than
    loudly so. Asserted as a set rather than a count so the failure names the
    receiver nobody classified.
    """

    receivers = {call.func.value.id for call in _step_calls(_tree())}
    assert receivers == set(STEP_RECEIVERS), (
        f"{TRAINER} steps {sorted(receivers)}; this file reasons about "
        f"{sorted(STEP_RECEIVERS)}. Classify the difference as a parameter "
        "mutation or not before assuming the boundary still covers it."
    )


def test_every_optimizer_mutation_is_enclosed_by_an_installed_sentinel() -> None:
    """A sentinel installed beside the loop guards nothing inside it.

    ``OptimizerMutationSentinel`` observes through a process-global pre-hook
    that only exists while ``installed()`` is entered. A mutation reached
    outside that block is unobserved, so enclosure is the property, not the
    presence of the call.
    """

    tree = _tree()
    mutations = _mutations(tree)
    assert mutations, f"{TRAINER} has no optimizer mutation to guard"
    unguarded = [
        call.lineno
        for call in mutations
        if not _enclosing_installed_names(tree, call)
    ]
    assert not unguarded, (
        f"{TRAINER}: optimizer mutations at {unguarded} are not inside any "
        "`installed()` sentinel block, so no pre-hook is watching them"
    )


def test_a_crossing_precedes_every_optimizer_mutation() -> None:
    """The whole point: a crossing that follows a mutation is a notification.

    This trainer previously called ``worker_controls.optimizer_step(step)``
    after all three updates. That call can refuse, but only once the parameters
    it was meant to protect have already moved. Checked per mutation rather
    than once, because the enumeration credits the file and would be satisfied
    by a crossing that precedes only the first of the three.
    """

    tree = _tree()
    paths = _statement_paths(tree)
    enclosing = _enclosing_statements(tree)
    crossings = _crossings(tree)
    assert crossings, f"{TRAINER} never crosses a sentinel before mutating"
    unpreceded = [
        call.lineno
        for call in _mutations(tree)
        if not any(
            _precedes(paths, enclosing, crossing, call) for crossing in crossings
        )
    ]
    assert not unpreceded, (
        f"{TRAINER}: no sentinel crossing precedes the mutations at "
        f"{unpreceded}; crossings are at {[c.lineno for c in crossings]}"
    )


def test_the_crossing_is_on_the_sentinel_that_encloses_the_mutations() -> None:
    """Two sentinels would let a crossing arm an object nothing observes.

    Separated from the enclosure and ordering checks because it fails for a
    different reason than either: the order can be right and the block can be
    right while the token is armed on the wrong instance.
    """

    tree = _tree()
    mutations = _mutations(tree)
    installed = set()
    for call in mutations:
        installed |= _enclosing_installed_names(tree, call)
    # A site that arms directly names its sentinel; a site that calls the
    # helper arms whatever the helper's own `.cross` names.
    armed = {
        call.func.value.id
        for call in _crossings(tree)
        if isinstance(call.func, ast.Attribute)
        and isinstance(call.func.value, ast.Name)
    }
    helper_names = {helper.name for helper in _crossing_helpers(tree)}
    if any(
        isinstance(call.func, ast.Name) and call.func.id in helper_names
        for call in _crossings(tree)
    ):
        for helper in _crossing_helpers(tree):
            armed |= {
                call.func.value.id
                for call in _calls(helper)
                if isinstance(call.func, ast.Attribute)
                and call.func.attr == "cross"
                and isinstance(call.func.value, ast.Name)
            }
    assert armed & installed, (
        f"{TRAINER}: the crossing arms {sorted(armed)} but the mutations are "
        f"observed by {sorted(installed)}"
    )


def test_the_crossing_names_a_computed_step_not_a_constant() -> None:
    """A literal step is the bug PR #130 found, in its other direction.

    ``pre_optimizer_step`` refuses any step at or below the attempt baseline,
    which is zero for a fresh attempt and the resume checkpoint's step for a
    replacement one. A crossing hard-coded to 1 therefore works on every fresh
    run and fails closed on every resumed one -- and all eight of these
    profiles resume. The step must come from the loop.
    """

    tree = _tree()
    paths = _statement_paths(tree)
    enclosing = _enclosing_statements(tree)
    mutations = _mutations(tree)
    checked = 0
    for crossing in _crossings(tree):
        if not any(
            _precedes(paths, enclosing, crossing, call) for call in mutations
        ):
            continue
        argument = crossing.args[0]
        assert not isinstance(argument, ast.Constant), (
            f"{TRAINER}: the crossing at line {crossing.lineno} names the "
            f"constant step {argument.value!r} rather than one the loop computed"
        )
        checked += 1
    assert checked, f"{TRAINER}: no crossing precedes any mutation"


def test_the_crossing_declares_a_count_covering_every_optional_optimizer() -> None:
    """One crossing authorizes a count, and the count is what can go stale.

    ``optimizer_host`` and ``optimizer_aux`` are conditional, so the number of
    mutations per training step is 1, 2 or 3. A crossing that declares a
    literal 1 orders the first update correctly and leaves the other two to
    fail closed mid-step -- which is safe but breaks the engram and Prodigy
    routes outright. The declared count has to be derived from the same names
    the mutations are guarded by.
    """

    tree = _tree()
    paths = _statement_paths(tree)
    enclosing = _enclosing_statements(tree)
    mutations = _mutations(tree)
    optional = {"optimizer_host", "optimizer_aux"} & {
        call.func.value.id for call in mutations
    }
    preceding = [
        crossing
        for crossing in _crossings(tree)
        if any(_precedes(paths, enclosing, crossing, call) for call in mutations)
    ]
    assert preceding, f"{TRAINER}: no crossing precedes any mutation"
    declared = [
        next(
            (word.value for word in crossing.keywords if word.arg == "mutations"),
            None,
        )
        for crossing in preceding
    ]
    constant = [
        crossing.lineno
        for crossing, count in zip(preceding, declared)
        if count is None or isinstance(count, ast.Constant)
    ]
    assert not constant, (
        f"{TRAINER}: the crossings at {constant} declare a constant mutation "
        f"count for a step that makes up to {len(mutations)} of them, so the "
        "count cannot track which optional optimizers this run built"
    )
    # At least one of them must derive the count from the very names whose
    # `.step()` it authorizes. Stated over the set rather than over a
    # particular call site so that routing the crossing through a helper, or
    # not, is style rather than something this test decides.
    derived = [
        count
        for count in declared
        if optional
        <= {node.id for node in ast.walk(count) if isinstance(node, ast.Name)}
    ]
    assert derived, (
        f"{TRAINER}: no crossing declares a count mentioning {sorted(optional)}, "
        "whose `.step()` it is nonetheless expected to authorize"
    )


def test_the_crossing_carries_the_controller_boundary() -> None:
    """An unarmed crossing satisfies the sentinel while asking nobody.

    ``cross(next_step, None)`` is legal -- it is what a standalone CLI run does
    -- and it orders the mutations correctly while never consulting the
    controller. Under TrainVM authority the crossing has to reach
    ``WorkerControlRuntime.pre_optimizer_step``, or the gate is decorative.
    """

    tree = _tree()
    reaches_controller = any(
        isinstance(call.func, ast.Attribute) and call.func.attr == BOUNDARY
        for call in _calls(tree)
    )
    assert reaches_controller, (
        f"{TRAINER} never calls WorkerControlRuntime.{BOUNDARY}, so its "
        "crossing arms the sentinel without asking the controller anything"
    )
    # And the call must be reachable from the crossing helper rather than
    # sitting in some unrelated corner of a 2600-line module.
    helper = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        and node.name == "cross_mutation_boundary"
    )
    assert any(
        isinstance(call.func, ast.Attribute) and call.func.attr == BOUNDARY
        for call in _calls(helper)
    ), (
        f"{TRAINER}: `cross_mutation_boundary` arms the sentinel without "
        f"reaching {BOUNDARY}"
    )


def test_no_post_mutation_control_notification_remains() -> None:
    """The call that used to sit after the update must be gone, not doubled.

    Leaving it beside the new crossing would apply the same safe point twice
    per step and, worse, would leave a reader with two plausible answers to
    "where is this trainer's boundary" -- one of which is the wrong side of the
    mutations.
    """

    stale = [
        call.lineno
        for call in _attribute_calls(_tree(), "optimizer_step")
        if isinstance(call.func.value, ast.Name)
        and call.func.value.id == "worker_controls"
    ]
    assert not stale, (
        f"{TRAINER} still calls worker_controls.optimizer_step at {stale}; the "
        "pre-mutation crossing replaces it rather than joining it"
    )


def test_the_trainer_refuses_an_ungated_baseline_before_it_mutates() -> None:
    """A refusal nothing calls is a refusal that never happens.

    The runtime tests below drive ``refuse_ungated_attempt_baseline`` directly,
    so every one of them stays green if ``train()`` stops calling it. This is
    the check that does not: the call must exist inside ``train`` and must
    precede every optimizer mutation.
    """

    tree = _tree()
    entry = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "train"
    )
    guards = [
        call
        for call in _calls(entry)
        if isinstance(call.func, ast.Name)
        and call.func.id == "refuse_ungated_attempt_baseline"
    ]
    assert guards, (
        f"{TRAINER}: `train` never calls `refuse_ungated_attempt_baseline`, so "
        "a resume the controller does not gate reaches the loop"
    )
    paths = _statement_paths(tree)
    enclosing = _enclosing_statements(tree)
    for mutation in _mutations(tree):
        assert any(
            _precedes(paths, enclosing, guard, mutation) for guard in guards
        ), (
            f"{TRAINER}: the baseline refusal does not precede the mutation at "
            f"line {mutation.lineno}"
        )


# --------------------------------------------------------------------------
# The runtime half: the composition the trainer builds, not its syntax
# --------------------------------------------------------------------------


def _runtime(**session) -> WorkerControlRuntime:
    """A real control runtime over a session stub with the gate fields set."""

    return WorkerControlRuntime(
        SimpleNamespace(
            **{
                "step_zero_eval_gate_required": False,
                "step_zero_eval_gate_satisfied": False,
                "attempt_baseline_optimizer_step": 0,
                # No controller is attached, so no command ever arrives and the
                # safe point applies an empty patch set. What these tests
                # exercise is the gate decision; a fake command stream would
                # only test the fake.
                "poll_commands": lambda: (),
                **session,
            }
        ),
        {},
        0,
    )


def _optimizer() -> torch.optim.Optimizer:
    parameter = torch.nn.Parameter(torch.zeros(2))
    parameter.grad = torch.zeros(2)
    return torch.optim.SGD([parameter], lr=0.0)


def test_one_crossing_authorizes_exactly_the_declared_mutations() -> None:
    """Three optimizers, one training step, one word to the controller."""

    sentinel = OptimizerMutationSentinel()
    optimizers = [_optimizer() for _ in range(3)]
    with sentinel.installed():
        sentinel.cross(1, mutations=3)
        for optimizer in optimizers:
            optimizer.step()
    assert sentinel.journal == (
        ("boundary", 1),
        ("mutation", 1),
        ("mutation", 1),
        ("mutation", 1),
    )


def test_a_mutation_beyond_the_declared_count_is_refused() -> None:
    """The fused/alternate-path case, at runtime rather than in syntax.

    A fourth optimizer -- one this module never constructed, or a fused update
    added later -- reaches the parameters with nothing left armed.
    """

    sentinel = OptimizerMutationSentinel()
    optimizers = [_optimizer() for _ in range(4)]
    with sentinel.installed():
        sentinel.cross(1, mutations=3)
        for optimizer in optimizers[:3]:
            optimizer.step()
        with pytest.raises(MutationSentinelError) as refusal:
            optimizers[3].step()
    assert "without first crossing" in str(refusal.value)


def test_declaring_more_mutations_than_happen_fails_at_the_next_crossing() -> None:
    """The other direction of the same miscount, and it must not pass silently.

    A count that over-declares would otherwise let the *next* step's mutation
    ride on this step's leftover authorization -- a mutation the controller
    was never asked about, arriving one step late.
    """

    sentinel = OptimizerMutationSentinel()
    optimizer = _optimizer()
    with sentinel.installed():
        sentinel.cross(1, mutations=2)
        optimizer.step()
        with pytest.raises(MutationSentinelError) as refusal:
            sentinel.cross(2, mutations=2)
    assert "still has authorized" in str(refusal.value)


@pytest.mark.parametrize("count", [0, -1, True, 1.0, "2"])
def test_a_crossing_that_authorizes_no_mutation_is_refused(count) -> None:
    """The count is caller-supplied, so it is validated like the step is.

    ``mutations=0`` would arm a token nothing can consume and disarm it before
    the first update, which is a crossing that quietly authorizes nothing;
    ``True`` is an ``int`` in Python and would silently mean one. Both are
    caller mistakes that must fail loudly rather than pass as a count.
    """

    sentinel = OptimizerMutationSentinel()
    with pytest.raises(MutationSentinelError) as refusal:
        sentinel.cross(1, mutations=count)
    assert "at least one optimizer mutation" in str(refusal.value)


def test_a_learning_rate_schedule_step_is_not_a_parameter_mutation() -> None:
    """Evidence for excluding ``trainvm_scheduler`` from the declared count.

    The trainer steps four objects in a row and only three of them mutate
    parameters. This is the decision recorded as a test rather than as a
    comment: an ``LRScheduler.step()`` rewrites ``param_group["lr"]`` and never
    enters ``Optimizer.step``, so it does not reach the process-global pre-hook
    at all. Had it counted, every declared count in the trainer would be one
    short and every step would fail closed.
    """

    sentinel = OptimizerMutationSentinel()
    optimizer = _optimizer()
    schedule = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _step: 1.0)
    with sentinel.installed():
        sentinel.cross(1, mutations=1)
        optimizer.step()
        # Nothing is armed now. A scheduler step that counted as a mutation
        # would raise here; the assertion is that it does not.
        schedule.step()
    assert sentinel.observed_mutations == 1
    assert sentinel.journal == (("boundary", 1), ("mutation", 1))


def test_a_refusing_boundary_leaves_the_sentinel_disarmed() -> None:
    """Why the crossing routes through the sentinel instead of calling direct.

    ``cross`` arms only after the boundary returns, so a controller refusal
    means the update that would have followed finds nothing armed and fails
    closed rather than proceeding on an unanswered question.
    """

    controls = _runtime(
        step_zero_eval_gate_required=True,
        step_zero_eval_gate_satisfied=False,
    )
    sentinel = OptimizerMutationSentinel()
    optimizer = _optimizer()
    with sentinel.installed():
        with pytest.raises(WorkerControlError) as refusal:
            sentinel.cross(
                1,
                lambda step: controls.pre_optimizer_step(step, lambda *_: None),
                mutations=1,
            )
        assert "durable attempt-baseline" in str(refusal.value)
        with pytest.raises(MutationSentinelError):
            optimizer.step()
    assert sentinel.observed_mutations == 0


# --------------------------------------------------------------------------
# The resume half: the gate meeting journal/checkpoint evidence
# --------------------------------------------------------------------------


def test_a_resume_the_controller_does_not_gate_is_refused() -> None:
    """A replacement attempt whose loop starts somewhere the gate does not."""

    controls = _runtime(attempt_baseline_optimizer_step=4096)
    with pytest.raises(ValueError) as refusal:
        refuse_ungated_attempt_baseline(controls, 2048)
    assert "does not gate" in str(refusal.value)
    assert "2048 != 4096" in str(refusal.value)


def test_the_baseline_is_read_and_not_assumed_to_be_zero() -> None:
    """The literal-zero bug 8f0da3e fixed, asserted from the resumed side.

    A check written against a literal ``0`` accepts a fresh attempt and refuses
    every replacement one, and its branch looks identical to a branch with
    nothing to do. This is the case that distinguishes them: a resume at 4096
    that the controller gates at 4096 is admissible.
    """

    controls = _runtime(attempt_baseline_optimizer_step=4096)
    refuse_ungated_attempt_baseline(controls, 4096)


def test_a_required_and_unsatisfied_gate_is_refused_before_the_loop() -> None:
    """Fail closed at the top rather than at the first crossing.

    The crossing would refuse this anyway. Refusing here means the diagnosis is
    the missing evidence rather than a refused step arriving after a full model
    load and a baseline evaluation.
    """

    controls = _runtime(
        attempt_baseline_optimizer_step=4096,
        step_zero_eval_gate_required=True,
        step_zero_eval_gate_satisfied=False,
    )
    with pytest.raises(ValueError) as refusal:
        refuse_ungated_attempt_baseline(controls, 4096)
    assert "cannot mutate" in str(refusal.value)
    assert "4096" in str(refusal.value)


def test_a_reconnect_whose_evidence_the_controller_replayed_proceeds() -> None:
    """The card's recover-without-permitting-an-unguarded-update requirement.

    A controller that replayed durable journal evidence for this exact attempt
    reports the gate satisfied, and the trainer proceeds -- to the crossing,
    which still happens on every step. Satisfaction removes the refusal, never
    the boundary.
    """

    controls = _runtime(
        attempt_baseline_optimizer_step=4096,
        step_zero_eval_gate_required=True,
        step_zero_eval_gate_satisfied=True,
    )
    refuse_ungated_attempt_baseline(controls, 4096)


def test_a_standalone_run_without_a_controller_is_unaffected() -> None:
    """The CLI path has no controller to disagree with."""

    refuse_ungated_attempt_baseline(None, 4096)
