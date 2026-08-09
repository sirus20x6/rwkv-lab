"""The pretrained-RWKV optimizer route crosses the boundary before it mutates.

`tests/test_step_zero_interception_enumeration.py` already asserts that this
profile reaches `pre_optimizer_step` at all -- it walks the registry and fails
any mutating profile that does not, and its allowlist can only shrink. That
enumeration answers *whether*, from a name. It cannot answer *where*, and
"calls the boundary somewhere in the module" is satisfied by a call after the
mutation, which guards nothing.

So this file asserts the two properties the enumeration cannot see:

- the crossing precedes `optimizer.step()` in the same statement list, and
  names `step + 1` rather than the step just finished;
- `refuse_ungated_attempt_baseline` compares against the controller's attempt
  baseline rather than a literal zero.

The AST reader deliberately looks at the *loop body*, not at `train` as a
whole. `train` defines a nested `cross_mutation_boundary` helper whose own body
contains a `cross` call, so a whole-function walk finds a boundary call even
after the real call site at the loop level is deleted -- a helper's internal
call counted as a call site is exactly how another agent's harness reported a
green mutation here.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest

from rwkv_lab import rwkv_optimizer_finetune

BOUNDARY = "cross_mutation_boundary"
MUTATION = "step"
MUTATION_RECEIVER = "optimizer"


def _train_function() -> ast.FunctionDef:
    source = Path(inspect.getsourcefile(rwkv_optimizer_finetune)).read_text(
        encoding="utf-8"
    )
    module = ast.parse(source)
    for node in module.body:
        if isinstance(node, ast.FunctionDef) and node.name == "train":
            return node
    raise AssertionError("rwkv_optimizer_finetune defines no train()")


def _training_loop_body() -> list[ast.stmt]:
    """The statement list that contains the optimizer mutation.

    Found by locating the mutation rather than by assuming a nesting depth, so
    the reader survives the loop gaining or losing a `with` wrapper.
    """

    for node in ast.walk(_train_function()):
        if not isinstance(node, ast.While):
            continue
        for statements in (node.body,):
            if any(_is_mutation(statement) for statement in statements):
                return statements
    raise AssertionError(
        "no while-loop body in train() contains an optimizer mutation; the "
        "reader below would report vacuously on any question asked of it"
    )


def _is_mutation(statement: ast.stmt) -> bool:
    return (
        isinstance(statement, ast.Expr)
        and isinstance(statement.value, ast.Call)
        and isinstance(statement.value.func, ast.Attribute)
        and statement.value.func.attr == MUTATION
        and isinstance(statement.value.func.value, ast.Name)
        and statement.value.func.value.id == MUTATION_RECEIVER
    )


def _is_boundary(statement: ast.stmt) -> bool:
    return (
        isinstance(statement, ast.Expr)
        and isinstance(statement.value, ast.Call)
        and isinstance(statement.value.func, ast.Name)
        and statement.value.func.id == BOUNDARY
    )


def test_the_reader_finds_exactly_one_mutation_and_one_crossing() -> None:
    """Guard the readers themselves before asking them anything.

    Every assertion below is of the form "index of A is less than index of B".
    If either list were empty the remaining tests would raise rather than
    report, and if either held two entries the ordering claim would be about an
    arbitrary one of them.
    """

    body = _training_loop_body()
    assert sum(1 for statement in body if _is_mutation(statement)) == 1
    assert sum(1 for statement in body if _is_boundary(statement)) == 1


def test_train_calls_the_attempt_baseline_guard() -> None:
    """The behavioural tests below drive the guard directly, which proves it
    works and not that anything uses it.

    Deleting the call from ``train`` left every other test in this file green.
    The walk is over ``train``'s body only; ``refuse_ungated_attempt_baseline``
    is a module-level function, so its own definition is not inside the walked
    node and cannot be miscounted as a call site -- the failure mode that made
    an earlier harness report this mutation as caught when it was not.
    """

    train = _train_function()
    calls = [
        node
        for node in ast.walk(train)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "refuse_ungated_attempt_baseline"
    ]
    assert len(calls) == 1, (
        "rwkv_optimizer_finetune.train must call "
        f"refuse_ungated_attempt_baseline exactly once, found {len(calls)}"
    )
    # It must be handed the restored step, not a constant: passing 0 would
    # reproduce the literal-zero defect from outside the guard instead of in it.
    argument = calls[0].args[1]
    assert isinstance(argument, ast.Name) and argument.id == "step", (
        f"the guard is called with {ast.dump(argument)}, not the restored step"
    )


def test_the_training_loop_runs_under_the_installed_sentinel() -> None:
    """The crossing arms a token; the installed pre-hook is what spends it.

    Without ``installed()`` the sentinel observes nothing, so a fused or
    alternate update path that never reaches ``cross_mutation_boundary`` would
    mutate unguarded and the arming would be theatre.
    """

    train = _train_function()
    guarded = [
        item
        for node in ast.walk(train)
        if isinstance(node, ast.With)
        for item in node.items
        if isinstance(item.context_expr, ast.Call)
        and isinstance(item.context_expr.func, ast.Attribute)
        and item.context_expr.func.attr == "installed"
        and any(
            isinstance(inner, ast.While) and any(
                _is_mutation(statement) for statement in inner.body
            )
            for inner in ast.walk(node)
        )
    ]
    assert guarded, (
        "the optimizer-mutating loop does not run inside "
        "mutation_sentinel.installed()"
    )


def test_the_crossing_precedes_the_mutation_it_guards() -> None:
    body = _training_loop_body()
    boundary = next(i for i, s in enumerate(body) if _is_boundary(s))
    mutation = next(i for i, s in enumerate(body) if _is_mutation(s))
    assert boundary < mutation, (
        "rwkv_optimizer_finetune.train crosses the controller boundary after "
        "optimizer.step(), which guards nothing"
    )


def test_the_crossing_names_the_step_about_to_run() -> None:
    """``step + 1`` -- never a constant, never the step just finished.

    The sentinel refuses a non-positive step and the controller refuses any
    step at or below the attempt baseline, so a crossing naming the completed
    step fails on the first iteration of a resumed attempt. This reddens only
    on the argument, so it discriminates from the ordering test above.
    """

    body = _training_loop_body()
    crossing = next(s for s in body if _is_boundary(s))
    argument = crossing.value.args[0]
    assert isinstance(argument, ast.BinOp) and isinstance(argument.op, ast.Add), (
        f"the crossing passes {ast.dump(argument)}, not the next step"
    )
    assert isinstance(argument.left, ast.Name) and argument.left.id == "step"
    assert isinstance(argument.right, ast.Constant) and argument.right.value == 1


def _controls(*, baseline: int, required: bool = False, satisfied: bool = False):
    return SimpleNamespace(
        step_zero_eval_gate_required=required,
        step_zero_eval_gate_satisfied=satisfied,
        attempt_baseline_optimizer_step=baseline,
    )


def test_the_guard_refuses_a_resume_the_controller_does_not_gate() -> None:
    """Behavioural: the comparison is against the baseline, not a literal zero.

    A replacement attempt resumes at its checkpoint's step and is gated there.
    Keying this to zero -- the defect ``mage_flow_pretrain.py`` shipped -- would
    accept a resume the controller refuses at every crossing, and the run would
    stall with nothing saying why.

    Both directions are asserted. A literal-zero implementation passes the
    fresh-attempt case and fails only the replacement one, so a test that
    checked a single direction would report green on the bug.
    """

    refuse = rwkv_optimizer_finetune.refuse_ungated_attempt_baseline

    refuse(_controls(baseline=0), 0)
    refuse(_controls(baseline=5500), 5500)

    # A literal zero would accept this: the controller gates 5500, the trainer
    # restored 0, so no crossing it can make is ever legal.
    with pytest.raises(ValueError, match="does not gate"):
        refuse(_controls(baseline=5500), 0)
    # ... and the mirror image, which a literal zero would reject.
    with pytest.raises(ValueError, match="does not gate"):
        refuse(_controls(baseline=0), 5500)


def test_the_guard_checks_the_baseline_even_when_no_eval_gate_is_required() -> None:
    """This route can never arm the step-zero gate, so the check must not hide
    behind it.

    `docs/experiment-vm/STEP_ZERO_ARMING.md` records that this contract
    declares no `eval_examples` output port, so no composition may publish one
    and `step_zero_eval_gate_required` is always false here. Ordering the
    baseline comparison after an early return on that flag -- which is how the
    Qwen AO3 guard is written, correctly, because that route can arm -- would
    make this branch permanently dead, and a branch that never fires looks
    exactly like a branch with nothing to do.
    """

    refuse = rwkv_optimizer_finetune.refuse_ungated_attempt_baseline

    with pytest.raises(ValueError, match="does not gate"):
        refuse(_controls(baseline=5500, required=False), 0)
    with pytest.raises(ValueError, match="does not gate"):
        refuse(_controls(baseline=5500, required=True, satisfied=True), 0)


def test_the_guard_is_inert_where_there_is_nothing_to_refuse() -> None:
    """A standalone run has no controller, and a satisfied gate owes nothing."""

    refuse = rwkv_optimizer_finetune.refuse_ungated_attempt_baseline

    refuse(None, 4200)
    refuse(_controls(baseline=5500, required=False), 5500)
    refuse(_controls(baseline=5500, required=True, satisfied=True), 5500)

    # Gated, unsatisfied, and correctly positioned is still a refusal: the
    # evidence the controller requires does not exist for this attempt.
    with pytest.raises(ValueError, match="requires durable attempt-baseline"):
        refuse(_controls(baseline=5500, required=True, satisfied=False), 5500)
