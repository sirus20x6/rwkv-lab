"""Ordering, not presence, for the four vision trainers' optimizer boundary.

``tests/test_step_zero_interception_enumeration.py`` can only see that these
trainers *name* ``pre_optimizer_step`` somewhere: it is a static reader that
asks one question, "does a boundary call appear in a producer this profile
resolves". That claim is satisfied by a call placed after the mutation it was
meant to guard, which is precisely the shape all four of these trainers had
before this change -- a post-mutation ``worker_controls.optimizer_step(step)``
that could only ever notify.

Why these tests are structural where the HF ones are not
-------------------------------------------------------
``tests/test_hf_multimodal_sft_mutation_boundary.py`` drives the real engine
and asserts an observed interleaving of crossings and mutations. That is the
stronger instrument and it is the one to prefer. It is not available here: no
test in this repository executes ``train()`` in any of these four modules, and
none can cheaply. Each needs staged MoonViT/SigLIP2/DINOv2/SAM feature caches
on disk, a frozen caption stack, and ``torch.autocast("cuda")``; the existing
vision suites all test extracted pieces -- models, samplers, loss functions,
cache codecs -- and never the loop. Writing that harness is real work and it is
filed rather than faked, because a harness assembled in a hurry would be the
thing under test rather than the trainer.

So what is asserted below is the strongest claim available without one: that in
each trainer's own syntax tree the crossing **dominates and precedes** the
mutation, inside an installed sentinel, keyed to a step the loop computes. That
is weaker than an observed interleaving and is not pretended otherwise. What
makes it more than a second copy of the enumeration is that the enumeration
cannot express any of it: it has no notion of order, of enclosure, or of what
the crossing was handed.

The runtime half is at the bottom. It exercises the composition these trainers
actually build -- a real ``OptimizerMutationSentinel`` crossed with a real
``WorkerControlRuntime`` boundary -- and shows that a refusal leaves the
sentinel disarmed so the mutation that would have followed fails closed. That
is the property the structural tests cannot reach, and the reason the trainers
route their crossing through the sentinel rather than calling the controller
directly.
"""

from __future__ import annotations

import ast
import pathlib
from types import SimpleNamespace

import pytest
import torch

from rwkv_lab.trainvm_worker import MutationSentinelError, OptimizerMutationSentinel
from rwkv_lab.trainvm_worker.controls import WorkerControlError, WorkerControlRuntime

REPOSITORY = pathlib.Path(__file__).resolve().parent.parent

# The four profiles this file covers, named by their producer. Kept as the
# module path rather than the contract because everything below reads source;
# the contract-to-producer mapping is the enumeration's job, not this file's.
TRAINERS = (
    "src/rwkv_lab/vision_teacher_compressor.py",
    "src/rwkv_lab/vision_train.py",
    "src/rwkv_lab/vision_native_train.py",
    "src/rwkv_lab/vision_rwkv_student_train.py",
)

BOUNDARY = "pre_optimizer_step"
MUTATION_RECEIVER = "optimizer"


def _tree(trainer: str) -> ast.Module:
    return ast.parse((REPOSITORY / trainer).read_text())


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


def _mutations(tree: ast.Module) -> list[ast.Call]:
    return [
        call
        for call in _attribute_calls(tree, "step")
        if isinstance(call.func.value, ast.Name)
        and call.func.value.id == MUTATION_RECEIVER
    ]


def _crossings(tree: ast.Module) -> list[ast.Call]:
    return _attribute_calls(tree, "cross")


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
        current = active
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
            visit(child, current)

    visit(tree, frozenset())
    return names


@pytest.fixture(params=TRAINERS, ids=lambda path: pathlib.Path(path).stem)
def trainer(request) -> str:
    return request.param


def test_each_trainer_has_exactly_one_optimizer_mutation(trainer: str) -> None:
    """The count every claim below is stated over.

    The card said one mutation site each and this checks it rather than
    inheriting it: a second `optimizer.step()` added later would be guarded by
    nothing this file asserts, and would make the single-crossing reasoning
    below quietly wrong rather than loudly so.
    """

    mutations = _mutations(_tree(trainer))
    assert len(mutations) == 1, (
        f"{trainer} has {len(mutations)} optimizer mutations at lines "
        f"{[call.lineno for call in mutations]}; each needs its own preceding "
        "boundary crossing and this file assumes one"
    )


def test_the_mutation_is_enclosed_by_an_installed_sentinel(trainer: str) -> None:
    """A sentinel installed beside the loop guards nothing inside it.

    ``OptimizerMutationSentinel`` observes through a process-global pre-hook
    that only exists while ``installed()`` is entered. A mutation reached
    outside that block is unobserved, so enclosure is the property, not the
    presence of the call.
    """

    tree = _tree(trainer)
    (mutation,) = _mutations(tree)
    enclosing = _enclosing_installed_names(tree, mutation)
    assert enclosing, (
        f"{trainer}: the optimizer mutation at line {mutation.lineno} is not "
        "inside any `installed()` sentinel block, so no pre-hook is watching it"
    )


def test_the_crossing_precedes_the_mutation(trainer: str) -> None:
    """The whole point: a crossing that follows the mutation is a notification.

    All four of these trainers previously called
    ``worker_controls.optimizer_step(step)`` after the update. That call can
    refuse, but only once the parameters it was meant to protect have already
    moved.
    """

    tree = _tree(trainer)
    (mutation,) = _mutations(tree)
    crossings = _crossings(tree)
    paths = _statement_paths(tree)
    enclosing = _enclosing_statements(tree)
    assert crossings, f"{trainer} never crosses a sentinel before mutating"
    preceding = [
        crossing
        for crossing in crossings
        if _precedes(paths, enclosing, crossing, mutation)
    ]
    assert preceding, (
        f"{trainer}: no sentinel crossing precedes the mutation at line "
        f"{mutation.lineno}; crossings are at "
        f"{[crossing.lineno for crossing in crossings]}"
    )


def test_the_crossing_is_on_the_same_sentinel_that_encloses_the_mutation(
    trainer: str,
) -> None:
    """Two sentinels would let a crossing arm an object nothing observes.

    Separated from the enclosure and ordering checks because it fails for a
    different reason than either: the order can be right and the block can be
    right while the token is armed on the wrong instance.
    """

    tree = _tree(trainer)
    (mutation,) = _mutations(tree)
    paths = _statement_paths(tree)
    enclosing = _enclosing_statements(tree)
    installed = _enclosing_installed_names(tree, mutation)
    crossed = {
        crossing.func.value.id
        for crossing in _crossings(tree)
        if isinstance(crossing.func, ast.Attribute)
        and isinstance(crossing.func.value, ast.Name)
        and _precedes(paths, enclosing, crossing, mutation)
    }
    assert crossed & installed, (
        f"{trainer}: the crossing arms {sorted(crossed)} but the mutation is "
        f"observed by {sorted(installed)}"
    )


def test_the_crossing_names_a_computed_step_not_a_constant(trainer: str) -> None:
    """A literal step is the bug PR #130 found, in its other direction.

    ``pre_optimizer_step`` refuses any step at or below the attempt baseline,
    which is zero for a fresh attempt and the resume checkpoint's step for a
    replacement one. A crossing hard-coded to 1 therefore works on every fresh
    run and fails closed on every resumed one -- and a crossing hard-coded to 0
    fails always. The step must come from the loop.
    """

    tree = _tree(trainer)
    (mutation,) = _mutations(tree)
    paths = _statement_paths(tree)
    enclosing = _enclosing_statements(tree)
    for crossing in _crossings(tree):
        if not _precedes(paths, enclosing, crossing, mutation):
            continue
        argument = crossing.args[0]
        assert not isinstance(argument, ast.Constant), (
            f"{trainer}: the crossing at line {crossing.lineno} names the "
            f"constant step {argument.value!r} rather than one the loop computed"
        )
        return
    pytest.fail(f"{trainer}: no crossing precedes the mutation")


def test_the_crossing_carries_the_controller_boundary(trainer: str) -> None:
    """An unarmed crossing satisfies the sentinel while asking nobody.

    ``cross(next_step, None)`` is legal -- it is what a standalone run does --
    and it orders the mutation correctly while never consulting the
    controller. Under TrainVM authority the second argument has to reach
    ``WorkerControlRuntime.pre_optimizer_step``, or the gate is decorative.
    """

    tree = _tree(trainer)
    (mutation,) = _mutations(tree)
    paths = _statement_paths(tree)
    enclosing = _enclosing_statements(tree)
    crossing = next(
        candidate
        for candidate in _crossings(tree)
        if _precedes(paths, enclosing, candidate, mutation)
    )
    assert len(crossing.args) == 2, (
        f"{trainer}: the crossing at line {crossing.lineno} passes no boundary"
    )
    boundary = crossing.args[1]
    assert isinstance(boundary, ast.Name), (
        f"{trainer}: the crossing's boundary argument is not a named callable"
    )
    # The name must be bound, somewhere in the module, to something that calls
    # the controller. Following the binding rather than trusting the name is
    # the difference between this and asserting a spelling.
    reaches_controller = any(
        isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == boundary.id
            for target in node.targets
        )
        and any(
            isinstance(call.func, ast.Attribute) and call.func.attr == BOUNDARY
            for call in ast.walk(node)
            if isinstance(call, ast.Call)
        )
        for node in ast.walk(tree)
    )
    assert reaches_controller, (
        f"{trainer}: {boundary.id!r} is never bound to a call reaching "
        f"WorkerControlRuntime.{BOUNDARY}"
    )


def test_no_post_mutation_control_notification_remains(trainer: str) -> None:
    """The call that used to sit after the update must be gone, not doubled.

    Leaving it beside the new crossing would apply the same safe point twice
    per step and, worse, would leave a reader with two plausible answers to
    "where is this trainer's boundary" -- one of which is the wrong side of the
    mutation.
    """

    tree = _tree(trainer)
    stale = [
        call.lineno
        for call in _attribute_calls(tree, "optimizer_step")
        if isinstance(call.func.value, ast.Name)
        and call.func.value.id == "worker_controls"
    ]
    assert not stale, (
        f"{trainer} still calls worker_controls.optimizer_step at {stale}; the "
        "pre-mutation crossing replaces it rather than joining it"
    )


# --------------------------------------------------------------------------
# The runtime half: the composition the trainers build, not their syntax
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
                # safe point applies an empty patch set. That is deliberate:
                # what these tests exercise is the gate decision, and a fake
                # command stream would only test the fake.
                "poll_commands": lambda: (),
                **session,
            }
        ),
        {},
        0,
    )


def _parameter() -> torch.nn.Parameter:
    parameter = torch.nn.Parameter(torch.zeros(4))
    parameter.grad = torch.ones(4)
    return parameter


def _boundary(controls: WorkerControlRuntime):
    """Exactly the callable the four trainers build."""

    return lambda next_step: controls.pre_optimizer_step(next_step, lambda *_: None)


def test_a_refused_boundary_leaves_the_mutation_failing_closed() -> None:
    """A controller refusal must stop the update, not merely be recorded.

    This is why the crossing goes through the sentinel instead of calling the
    controller directly: `cross` arms its token only *after* the boundary
    returns, so a raise leaves nothing armed and the `step()` that follows
    cannot proceed even if the trainer swallowed the refusal.
    """

    controls = _runtime(
        step_zero_eval_gate_required=True, step_zero_eval_gate_satisfied=False
    )
    parameter = _parameter()
    optimizer = torch.optim.SGD([parameter], lr=0.1)
    sentinel = OptimizerMutationSentinel()
    with sentinel.installed():
        with pytest.raises(WorkerControlError, match="durable attempt-baseline"):
            sentinel.cross(1, _boundary(controls))
        with pytest.raises(MutationSentinelError, match="mandatory pre-optimizer-step"):
            optimizer.step()
    assert sentinel.observed_mutations == 0
    assert torch.equal(parameter.detach(), torch.zeros(4))


def test_a_satisfied_gate_admits_exactly_one_mutation_per_crossing() -> None:
    """The armed token is single use, so a fused second update fails closed."""

    controls = _runtime(
        step_zero_eval_gate_required=True, step_zero_eval_gate_satisfied=True
    )
    optimizer = torch.optim.SGD([_parameter()], lr=0.1)
    sentinel = OptimizerMutationSentinel()
    with sentinel.installed():
        sentinel.cross(1, _boundary(controls))
        optimizer.step()
        with pytest.raises(MutationSentinelError, match="mandatory pre-optimizer-step"):
            optimizer.step()
    assert sentinel.journal == (("boundary", 1), ("mutation", 1))


def test_a_resumed_attempt_must_cross_at_its_own_next_step() -> None:
    """Why the crossing may not be a constant, stated at runtime.

    A replacement attempt resuming at 5500 is gated there. The trainers cross
    at ``step + 1``, which is 5501 and is accepted. A trainer that crossed at a
    literal 1 -- the shape that reads as "the first step of this run" -- is
    refused, which is the resumed-run half of the failure PR #130 found on the
    publication side.
    """

    controls = _runtime(
        step_zero_eval_gate_required=True,
        step_zero_eval_gate_satisfied=True,
        attempt_baseline_optimizer_step=5500,
    )
    boundary = _boundary(controls)
    boundary(5501)
    for constant in (0, 1):
        with pytest.raises(WorkerControlError, match="immutable attempt baseline"):
            boundary(constant)
