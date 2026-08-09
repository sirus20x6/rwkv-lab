"""Ordering and resume keying for the Qwen AO3 and RWKV post-training boundaries.

``tests/test_step_zero_interception_enumeration.py`` can only see that these two
trainers *name* ``pre_optimizer_step`` somewhere: it is a static presence check,
and it is satisfied by a call placed after the mutation it was meant to guard --
which is exactly the state ``posttrain_train.py`` was in before this suite
existed. What it could not see is the *order*.

Why the ordering claims here are structural
-------------------------------------------
The reference suite for this boundary,
``tests/test_hf_multimodal_sft_mutation_boundary.py``, drives the real engine
over doubles and asserts an interleaving of boundary crossings and mutations in
one list. That instrument is not available to these two routes: the AO3 trainer
loads a quantized 35B Qwen MoE through ``transformers`` and requires CUDA
(``torch.device(f"cuda:{config.cuda_index}")`` is unconditional), and the
post-training trainer builds a real RWKV model from a checkpoint blob. Standing
up either inside a CPU-only pytest would mean a double so large that what it
tested would be the double.

So the ordering is asserted against the source these trainers actually execute,
via its AST, and the *behavioural* claims are asserted against real functions:
``refuse_ungated_attempt_baseline`` in both modules, which is where the resume
keying lives. The mechanism itself -- that an armed crossing is consumed by one
mutation and an unarmed one raises -- is already proven behaviourally against a
real ``torch`` optimizer in ``tests/test_rwkv_mutation_sentinel.py``. Restating
it here would be a second literal for a fact that file already states.

What each test would catch is recorded beside it, because a suite whose tests
all redden together is one assertion wearing several costumes.
"""

from __future__ import annotations

import ast
import inspect
import pathlib
from types import SimpleNamespace

import pytest

from rwkv_lab import posttrain_train, qwen_ao3_cpt

# The two routes this suite owns, each named by the enumeration contract whose
# UNMAPPED_INTERCEPTION entry this work removed, and the function inside the
# module that performs the mutation.
TRAINERS = {
    "rwkv_lab.qwen_ao3.v1.Train": (qwen_ao3_cpt, "train"),
    "rwkv_lab.rwkv_posttraining.v1.Train": (posttrain_train, "train"),
}


def _mutating_function(module: object, name: str) -> ast.FunctionDef:
    source = pathlib.Path(inspect.getsourcefile(module)).read_text()
    for node in ast.parse(source).body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{module.__name__} has no {name}()")


def _calls(tree: ast.AST, attribute: str) -> list[ast.Call]:
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == attribute
    ]


def _parents(tree: ast.AST) -> dict[ast.AST, ast.AST]:
    parents: dict[ast.AST, ast.AST] = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[child] = node
    return parents


def _ancestors(node: ast.AST, parents: dict[ast.AST, ast.AST]) -> list[ast.AST]:
    chain: list[ast.AST] = []
    while node in parents:
        node = parents[node]
        chain.append(node)
    return chain


@pytest.mark.parametrize("contract", sorted(TRAINERS))
def test_the_boundary_crossing_precedes_the_optimizer_mutation(contract: str) -> None:
    """The load-bearing ordering claim: cross first, then mutate.

    Reddens if a crossing is deleted, or moved below ``optimizer.step()`` --
    the precise defect the post-training route carried, where the only
    controller call sat after the update and so could refuse nothing.
    """

    module, name = TRAINERS[contract]
    function = _mutating_function(module, name)
    mutations = [
        call
        for call in _calls(function, "step")
        if isinstance(call.func.value, ast.Name)
        and call.func.value.id in {"optimizer", "opt", "optim"}
    ]
    crossings = _calls(function, "cross")
    assert len(mutations) == 1, (
        f"{contract} no longer has exactly one optimizer mutation: "
        f"{[node.lineno for node in mutations]}; a new one needs its own crossing"
    )
    assert len(crossings) == 1, (
        f"{contract} has {len(crossings)} sentinel crossings, expected one"
    )
    assert crossings[0].lineno < mutations[0].lineno, (
        f"{contract} crosses its boundary at line {crossings[0].lineno}, below "
        f"the mutation at line {mutations[0].lineno}: a boundary that cannot "
        "refuse before the parameters move is a notification"
    )


@pytest.mark.parametrize("contract", sorted(TRAINERS))
def test_the_mutation_runs_inside_an_installed_sentinel(contract: str) -> None:
    """The mutation must be under ``OptimizerMutationSentinel.installed()``.

    Reddens if the ``with`` block is removed or narrowed so the mutation falls
    outside it -- which is what would let a fused update, a second optimizer,
    or a reward-head optimizer reach the parameters without a crossing. It does
    NOT redden when a crossing is merely misplaced, which is the previous
    test's job.
    """

    module, name = TRAINERS[contract]
    function = _mutating_function(module, name)
    parents = _parents(function)
    sentinels = {
        target.id
        for node in ast.walk(function)
        if isinstance(node, ast.Assign)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name)
        and node.value.func.id == "OptimizerMutationSentinel"
        for target in node.targets
        if isinstance(target, ast.Name)
    }
    assert sentinels, f"{contract} constructs no OptimizerMutationSentinel"
    mutations = [
        call
        for call in _calls(function, "step")
        if isinstance(call.func.value, ast.Name)
        and call.func.value.id in {"optimizer", "opt", "optim"}
    ]
    for mutation in mutations:
        guarded = [
            block
            for block in _ancestors(mutation, parents)
            if isinstance(block, ast.With)
            and any(
                isinstance(item.context_expr, ast.Call)
                and isinstance(item.context_expr.func, ast.Attribute)
                and item.context_expr.func.attr == "installed"
                and isinstance(item.context_expr.func.value, ast.Name)
                and item.context_expr.func.value.id in sentinels
                for item in block.items
            )
        ]
        assert guarded, (
            f"{contract} mutates at line {mutation.lineno} outside any installed "
            "mutation sentinel, so an alternate update path fails open"
        )


@pytest.mark.parametrize("contract", sorted(TRAINERS))
def test_the_crossing_calls_the_pre_mutation_entry_point(contract: str) -> None:
    """The boundary handed to the sentinel must be ``pre_optimizer_step``.

    ``WorkerControlRuntime`` exposes both names for the same safe point, and
    only one of them says which side of the mutation it belongs on. Reddens if
    the crossing is rewired to ``optimizer_step``, or if a post-mutation
    ``optimizer_step`` notification is reintroduced beside the crossing --
    these calls were *moved*, not duplicated, and two call sites is how the
    ordering silently regresses to the old shape.
    """

    module, name = TRAINERS[contract]
    function = _mutating_function(module, name)
    assert _calls(function, "pre_optimizer_step"), (
        f"{contract} never calls pre_optimizer_step"
    )
    stale = [
        call.lineno
        for call in _calls(function, "optimizer_step")
        if isinstance(call.func.value, ast.Name)
        and call.func.value.id == "worker_controls"
    ]
    assert not stale, (
        f"{contract} still calls worker_controls.optimizer_step at {stale}; the "
        "crossing replaces that call rather than joining it"
    )


@pytest.mark.parametrize("contract", sorted(TRAINERS))
def test_the_crossing_names_the_step_about_to_run(contract: str) -> None:
    """``cross(step + 1)`` -- never a constant, never the completed step.

    The sentinel refuses a non-positive step and the controller refuses any
    step at or below the attempt baseline, so a crossing that names the step
    just *finished* fails on the first iteration of a resumed attempt. Reddens
    only on the argument, so it discriminates from the ordering tests above.
    """

    module, name = TRAINERS[contract]
    function = _mutating_function(module, name)
    crossing = _calls(function, "cross")[0]
    argument = crossing.args[0]
    assert isinstance(argument, ast.BinOp) and isinstance(argument.op, ast.Add), (
        f"{contract} crosses with {ast.dump(argument)}, not the next step"
    )
    assert isinstance(argument.left, ast.Name) and argument.left.id == "step"
    assert (
        isinstance(argument.right, ast.Constant) and argument.right.value == 1
    ), f"{contract} does not cross for the step about to run"


def test_qwen_ao3_refuses_a_resume_the_controller_does_not_gate() -> None:
    """Behavioural: the AO3 guard compares against the attempt baseline.

    A replacement attempt resumes at its checkpoint's step and is gated there.
    Keying this to a literal zero -- the defect ``mage_flow_pretrain.py``
    shipped -- would accept a resume the controller will refuse at every
    crossing, and the run would stall with nothing saying why.
    """

    refuse = qwen_ao3_cpt.refuse_ungated_attempt_baseline

    def controls(baseline=0):
        return SimpleNamespace(
            step_zero_eval_gate_required=True,
            step_zero_eval_gate_satisfied=False,
            attempt_baseline_optimizer_step=baseline,
        )

    # A fresh attempt at zero, and a replacement attempt at its own baseline.
    refuse(controls(), 0)
    refuse(controls(5500), 5500)

    # A literal zero would accept this: the controller gates 5500 and the
    # trainer restored 0, so no crossing it ever makes can be legal.
    with pytest.raises(ValueError, match="does not gate"):
        refuse(controls(5500), 0)
    # ... and the mirror image, which a literal zero would reject.
    with pytest.raises(ValueError, match="does not gate"):
        refuse(controls(), 5500)


def test_qwen_ao3_leaves_ungated_and_already_satisfied_attempts_alone() -> None:
    """The guard is inert where there is nothing to gate.

    A standalone command-line run has no controller at all, an attempt whose
    recipe declares no required eval-examples artifact is not gated, and a
    reconnect whose evidence the controller already replayed owes nothing.
    None of those may be turned into a refusal -- and none of them licenses
    skipping the crossing, which the ordering tests above assert separately.
    """

    refuse = qwen_ao3_cpt.refuse_ungated_attempt_baseline
    refuse(None, 4200)
    refuse(
        SimpleNamespace(
            step_zero_eval_gate_required=False,
            step_zero_eval_gate_satisfied=False,
            attempt_baseline_optimizer_step=5500,
        ),
        0,
    )
    refuse(
        SimpleNamespace(
            step_zero_eval_gate_required=True,
            step_zero_eval_gate_satisfied=True,
            attempt_baseline_optimizer_step=5500,
        ),
        0,
    )


def test_posttraining_refuses_a_baseline_its_restart_only_loop_cannot_reach() -> None:
    """Behavioural: restart-only means the baseline must be zero, and is checked.

    The handler rejects resume state, so every attempt's loop starts at zero
    and the first crossing names one. A non-zero baseline is unreachable, and
    the controller would refuse every crossing forever. Reddens if the check is
    dropped on the reasoning that the value "is always zero here".
    """

    refuse = posttrain_train.refuse_ungated_attempt_baseline

    refuse(None)
    refuse(
        SimpleNamespace(
            step_zero_eval_gate_required=True,
            step_zero_eval_gate_satisfied=False,
            attempt_baseline_optimizer_step=0,
        )
    )
    refuse(
        SimpleNamespace(
            step_zero_eval_gate_required=False,
            step_zero_eval_gate_satisfied=False,
            attempt_baseline_optimizer_step=64,
        )
    )
    with pytest.raises(ValueError, match="restart-only"):
        refuse(
            SimpleNamespace(
                step_zero_eval_gate_required=True,
                step_zero_eval_gate_satisfied=False,
                attempt_baseline_optimizer_step=64,
            )
        )
