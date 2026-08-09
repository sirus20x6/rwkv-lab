"""Enumerate every registered training operation and map it to a producer and
a pre-mutation interception site.

This is acceptance item 1 of the step-zero-evidence card, and only item 1. It
answers one question that nothing in the repository answered before: *how big
is the gap?* The card asserts that every optimizer mutation reachable from the
authority's stateful profiles must cross
``WorkerControlRuntime.pre_optimizer_step`` immediately before mutating. Before
this file, the only way to find out how many profiles actually do that was to
read twenty trainers by hand.

What this file deliberately does NOT do
---------------------------------------
It checks **presence**, not ordering. Whether the boundary sits before the
mutation, survives a fused update path, or can be bypassed by a second
optimizer instance is acceptance item 2, and it already has a mechanism --
``rwkv_lab.trainvm_worker.mutation_sentinel.OptimizerMutationSentinel`` -- with
its own suite in ``tests/test_rwkv_mutation_sentinel.py``. Restating any of
that here would be a second literal for a fact those files already state.

Why the enumeration is registry-driven
--------------------------------------
The profile set is read out of ``handlers._HANDLERS``, the closed dispatch
table the worker actually resolves an invocation against. It is not a list
maintained here. A hand-written list of twenty profiles would be a second
literal stating a fact the registry already states, and it would go stale in
the one direction that matters: a profile added to the registry and forgotten
here would be silently unenumerated, which is precisely the gap this file
exists to measure.

That registry is the authority's, not a Python-side approximation of it.
``trainvm/tests/verify_rwkv_lab_worker_contract.py`` asserts key-for-key
equality between ``supported_adapter_keys()`` and the native adapter registry
built by ``trainvm/src/rwkv_lab_worker_contract.cpp``, and that check runs in
the native CI job. So enumerating ``_HANDLERS`` enumerates the authority's
operations, with drift already gated elsewhere.

What could not be derived, and what blocks it
---------------------------------------------
The native registry declares ``lifecycle.stateful`` per profile. The Python
side does not carry that flag anywhere, and the hosted Python jobs cannot build
``trainvm`` (it needs GCC 16 with ``-freflection``), so a pytest cannot read the
declaration in the job where this suite runs.

The response is *not* to filter the enumeration by a locally inferred notion of
statefulness. Inferring it -- from ``operation == "train"``, say -- would be
exactly the hand-maintained second literal this file refuses, and any filter is
a place where a hard profile can quietly drop out of the count. So the
enumeration covers **every registered operation** and lets the
mutation-site detector decide what needs a boundary: an operation that
constructs no optimizer has nothing to intercept and says so by name. Breadth
is safer than a filter, because it cannot skip.

``_native_stateful_profiles`` closes the loop opportunistically: when a trainvm
binary resolves in this checkout, the enumeration's own classification is
cross-checked against the native declaration. That follows the precedent set by
``tests/trainvm_binary.py`` -- the core assertion never depends on a build, and
the build-dependent part strengthens rather than gates.
"""

from __future__ import annotations

import ast
import inspect
import json
import pathlib
import subprocess
from dataclasses import dataclass

import pytest

from rwkv_lab.trainvm_adapters import handlers

from trainvm_binary import resolve_trainvm

REPOSITORY = pathlib.Path(__file__).resolve().parent.parent
SOURCE_ROOT = REPOSITORY / "src"

# The controller-facing call that is the boundary. Named once, here, because
# every other reference to it in this file is derived from this constant.
BOUNDARY = "pre_optimizer_step"

# A mutation is `<something ending in an optimizer-ish name>.step()`. The
# receiver is matched on its rightmost identifier so that both a bare
# `optimizer.step()` and an attribute chain like `stack.optimizer.step()`
# count -- the latter is how the HF multimodal SFT trainer mutates, and a
# Name-only detector reports that trainer as having no mutation at all, which
# is the most dangerous possible false negative for this card.
#
# The detector is deliberately conservative rather than exhaustive: it finds
# the *named* optimizer mutations. Fused and alternate update paths that never
# spell an optimizer are acceptance item 2's problem, and the process-global
# pre-hook in OptimizerMutationSentinel is the mechanism that catches them at
# runtime. A static detector cannot, and pretending otherwise here would put a
# false floor under the count.
MUTATION_RECEIVERS = frozenset({"optimizer", "optimizers", "opt", "optim"})


# Profiles with a real optimizer mutation and no pre-mutation boundary.
#
# This is a countdown, not a configuration. It can only shrink, and every
# removal is a visible commit. A reporting-only test would be wallpaper within
# a week; an entry here is a named debt with an owner.
#
# It started at nineteen of twenty stateful profiles, which is worth stating
# plainly rather than burying: scratch-RWKV was the only route that reached the
# boundary when this file landed. It is fourteen now -- scratch-RWKV, HF
# multimodal SFT and the four vision routes cross it. An allowlist that begins
# by covering almost everything is a
# weak instrument -- it cannot fail until someone removes an entry -- and the
# argument for keeping it anyway is that the alternative fails *every* run from
# day one, which teaches people to ignore the suite rather than to fix it. The
# stale-entry check below is what gives it teeth in the meantime: the moment a
# trainer gains a boundary, this list must shrink or CI goes red.
UNMAPPED_INTERCEPTION: dict[str, str] = {
    "rwkv_lab.mageflow_appearance_expert.v1.Train":
        "mage_flow_expert_train.py mutates without reaching the boundary",
    "rwkv_lab.mageflow_full_backbone.v1.Train":
        "mage_flow_pretrain.py reads attempt_baseline_optimizer_step to key its "
        "baseline work but never calls the boundary before mutating",
    "rwkv_lab.mageflow_terminal_expert.v1.Train":
        "mage_flow_terminal_train.py mutates without reaching the boundary",
    "rwkv_lab.rwkv_rlvr.v1.Train":
        "rlvr_train.py has two optimizer mutations, neither guarded",
    "rwkv_lab.transformer_mla.v1.Train":
        "train_mla.py mutates without reaching the boundary",
    "rwkv_lab.transformer_mla_mtp.v1.Train":
        "train_mla.py mutates without reaching the boundary",
    "rwkv_lab.transformer_mla_mutor.v1.Train":
        "train_mla.py mutates without reaching the boundary",
    "rwkv_lab.transformer_mla_fsp.v1.Train":
        "train_mla.py mutates without reaching the boundary",
    "rwkv_lab.transformer_mla_parallel.v1.Train":
        "train_mla.py mutates without reaching the boundary",
    "rwkv_lab.transformer_mla_rwkv8.v1.Train":
        "train_mla.py mutates without reaching the boundary",
    "rwkv_lab.transformer_mla_engram.v1.Train":
        "train_mla.py mutates without reaching the boundary; this route also "
        "holds two optimizer-category components in different slots",
    "rwkv_lab.transformer_mla_full_backbone.v1.Train":
        "train_mla.py mutates without reaching the boundary",
}

# Operations that resolve no optimizer mutation at all, so there is nothing for
# a pre-mutation boundary to precede. This is a different claim from the list
# above and is kept separate on purpose: "has no mutation" is a property of the
# operation, "has a mutation nobody guards" is a debt.
NO_MUTATION: dict[str, str] = {
    "rwkv_lab.scalar_metric_decision.v1.Decide":
        "the registry's only non-stateful operation: it compares two immutable "
        "scalar results and constructs no optimizer",
}


@dataclass(frozen=True)
class RegisteredOperation:
    """One row of the enumeration, attributable by name when it fails."""

    adapter: str
    version: str
    operation: str
    contract: str
    handler: str
    producers: tuple[str, ...]
    mutation_sites: tuple[str, ...]
    interception_sites: tuple[str, ...]

    def __str__(self) -> str:
        return self.contract


def _module_path(module: str) -> pathlib.Path | None:
    direct = SOURCE_ROOT / (module.replace(".", "/") + ".py")
    if direct.exists():
        return direct
    package = SOURCE_ROOT / module.replace(".", "/") / "__init__.py"
    return package if package.exists() else None


def _absolute_module(node: ast.ImportFrom, package: str) -> str | None:
    """Resolve an ``ImportFrom`` to a dotted module, relative imports included.

    Handlers import their trainer *inside* the handler body -- the scratch-RWKV
    handler does `from rwkv_lab.rwkv_pretrain import main as train` -- and the
    HF multimodal handler uses a relative `from .hf_multimodal_sft import ...`.
    An absolute-only resolver silently drops the relative one, so that trainer
    would enumerate with no producer and never reach the mutation check.
    """
    if node.level == 0:
        return node.module
    parts = package.split(".")
    base = parts[: len(parts) - (node.level - 1)] if node.level > 1 else parts
    return ".".join(base + ([node.module] if node.module else []))


def _receiver_identifier(node: ast.expr) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _mutation_lines(tree: ast.AST) -> tuple[int, ...]:
    return tuple(
        call.lineno
        for call in ast.walk(tree)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Attribute)
        and call.func.attr == "step"
        and _receiver_identifier(call.func.value) in MUTATION_RECEIVERS
    )


def _boundary_lines(tree: ast.AST) -> tuple[int, ...]:
    return tuple(
        call.lineno
        for call in ast.walk(tree)
        if isinstance(call, ast.Call)
        and (
            (isinstance(call.func, ast.Attribute) and call.func.attr == BOUNDARY)
            or (isinstance(call.func, ast.Name) and call.func.id == BOUNDARY)
        )
    )


def _handler_definitions() -> dict[str, ast.FunctionDef]:
    source = pathlib.Path(inspect.getsourcefile(handlers)).read_text()
    return {
        node.name: node
        for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef)
    }


def _enumerate() -> tuple[RegisteredOperation, ...]:
    definitions = _handler_definitions()
    package = handlers.__name__.rsplit(".", 1)[0]
    rows: list[RegisteredOperation] = []
    for key, handler in handlers._HANDLERS.items():
        adapter, version, operation, contract = key
        definition = definitions[handler.__name__]
        modules = sorted(
            {
                resolved
                for node in ast.walk(definition)
                if isinstance(node, ast.ImportFrom)
                and (resolved := _absolute_module(node, package))
                and resolved.startswith("rwkv_lab")
            }
        )
        producers: list[str] = []
        mutations: list[str] = []
        boundaries: list[str] = []
        for module in modules:
            path = _module_path(module)
            if path is None:
                continue
            tree = ast.parse(path.read_text())
            relative = path.relative_to(REPOSITORY)
            mutation_lines = _mutation_lines(tree)
            boundary_lines = _boundary_lines(tree)
            if not mutation_lines and not boundary_lines:
                continue
            producers.append(module)
            mutations += [f"{relative}:{line}" for line in mutation_lines]
            boundaries += [f"{relative}:{line}" for line in boundary_lines]
        rows.append(
            RegisteredOperation(
                adapter=adapter,
                version=version,
                operation=operation,
                contract=contract,
                handler=handler.__name__,
                producers=tuple(producers),
                mutation_sites=tuple(mutations),
                interception_sites=tuple(boundaries),
            )
        )
    return tuple(sorted(rows, key=lambda row: row.contract))


OPERATIONS = _enumerate()


def _identifier(row: RegisteredOperation) -> str:
    return row.contract


def test_the_registry_is_not_empty() -> None:
    """A resolver that silently returns nothing would make every row below vacuous.

    This is the failure the enumeration is least able to notice on its own: a
    parametrised suite over an empty sequence reports zero failures, and a
    reader sees a green check. The repository has already shipped a gate that
    printed PASSED *because* its check failed; this asserts the floor.
    """

    assert len(OPERATIONS) >= 20, (
        "the adapter handler registry resolved fewer operations than the "
        f"authority declares stateful alone: {len(OPERATIONS)}"
    )


@pytest.mark.parametrize("row", OPERATIONS, ids=_identifier)
def test_every_registered_operation_resolves_a_real_producer(
    row: RegisteredOperation,
) -> None:
    """Each operation must name a trainer module that exists on disk.

    A handler that dispatches to nothing -- or to a module the enumeration
    cannot resolve -- makes every later claim about that profile unfounded.
    """

    if row.contract in NO_MUTATION:
        assert not row.mutation_sites, (
            f"{row.contract} is recorded as constructing no optimizer, but the "
            f"enumeration found mutations at {list(row.mutation_sites)}; remove "
            "it from NO_MUTATION and give it a real classification"
        )
        return
    assert row.producers, (
        f"{row.contract} resolves no producer module under src/. Its handler "
        f"{row.handler}() imports no rwkv_lab trainer that mutates an optimizer "
        "or crosses the boundary, so nothing maps this profile to real code."
    )


@pytest.mark.parametrize("row", OPERATIONS, ids=_identifier)
def test_every_mutating_operation_reaches_the_pre_mutation_boundary(
    row: RegisteredOperation,
) -> None:
    """A producer that mutates must cross the boundary, or be a named debt.

    An unmapped profile fails rather than reports. The allowlist above is the
    escape hatch, and it is greppable, one line per profile, with a reason.
    """

    if not row.mutation_sites:
        assert row.contract in NO_MUTATION, (
            f"{row.contract} resolves no optimizer mutation. Either its producer "
            "is not being found -- which would make its clean result meaningless "
            "-- or it genuinely constructs no optimizer and belongs in NO_MUTATION "
            "with a reason."
        )
        return
    if row.interception_sites:
        return
    assert row.contract in UNMAPPED_INTERCEPTION, (
        f"{row.contract} mutates at {list(row.mutation_sites)} and never calls "
        f"{BOUNDARY}. Add the boundary immediately before the mutation, or record "
        "this profile in UNMAPPED_INTERCEPTION with a one-line reason."
    )


@pytest.mark.parametrize("row", OPERATIONS, ids=_identifier)
def test_an_interception_site_is_production_code(row: RegisteredOperation) -> None:
    """A site only tests call does not count.

    This repository has twice found a fully tested mechanism with zero
    production callers. The boundary is exercised heavily under ``tests/`` --
    ``test_rwkv_mutation_sentinel.py`` alone accounts for most references to it
    -- so "the codebase mentions the boundary N times" is not evidence that any
    trainer crosses it. Every site this enumeration credits must live under
    ``src/``.
    """

    for site in row.interception_sites:
        assert site.startswith("src/"), (
            f"{row.contract} credits an interception site outside src/: {site}"
        )


def test_the_allowlist_holds_no_stale_entries() -> None:
    """The allowlist may only shrink.

    This is what stops it becoming configuration. A profile that gains a
    boundary must be removed from the list in the same change, and a profile
    that no longer exists must not linger as an unearned excuse.
    """

    by_contract = {row.contract: row for row in OPERATIONS}
    unknown = sorted(set(UNMAPPED_INTERCEPTION) - set(by_contract))
    assert not unknown, (
        f"UNMAPPED_INTERCEPTION names profiles the registry no longer has: {unknown}"
    )
    fixed = sorted(
        contract
        for contract in UNMAPPED_INTERCEPTION
        if by_contract[contract].interception_sites
    )
    assert not fixed, (
        "these profiles now reach the boundary and must be removed from "
        f"UNMAPPED_INTERCEPTION: {fixed}"
    )
    unknown_clean = sorted(set(NO_MUTATION) - set(by_contract))
    assert not unknown_clean, (
        f"NO_MUTATION names profiles the registry no longer has: {unknown_clean}"
    )


def test_every_operation_is_classified_exactly_once() -> None:
    """No profile may be both a named debt and exempt.

    Overlap would let a profile satisfy the suite from whichever list happened
    to be consulted first, which is how a check comes to pass for a reason
    nobody intended.
    """

    overlap = sorted(set(UNMAPPED_INTERCEPTION) & set(NO_MUTATION))
    assert not overlap, f"profiles classified twice: {overlap}"


def _native_profiles() -> list[dict] | None:
    """The authority's own registry, when a binary of this checkout exists.

    ``resolve_trainvm`` raises ``TrainvmBinaryError`` when ``TRAINVM_BINARY``
    names something unusable, and that exception is deliberately allowed to
    propagate: an explicitly requested binary that cannot be honoured must fail
    rather than degrade into the skip below.
    """

    binary, _ = resolve_trainvm()
    if binary is None:
        return None
    fingerprint = "sha256:" + "a" * 64
    output = subprocess.check_output(
        [binary, "inspect-rwkv-lab-worker", fingerprint], text=True
    )
    return json.loads(output)["adapter_registry"]["profiles"]


def test_the_enumeration_agrees_with_the_native_stateful_declaration() -> None:
    """Cross-check the local classification against ``lifecycle.stateful``.

    Only the native registry declares statefulness, and the hosted Python jobs
    cannot build it, so this strengthens the suite where a build exists rather
    than gating it -- the same contract ``tests/trainvm_binary.py`` documents
    for the qualification graders. The enumeration above is complete without
    it; what this adds is proof that "constructs no optimizer" and "the
    authority calls it non-stateful" pick out the same profile.
    """

    profiles = _native_profiles()
    if profiles is None:
        pytest.skip("no trainvm binary in this checkout; enumeration ran without it")
    native_stateful = {
        profile["key"]["contract"]
        for profile in profiles
        if profile.get("lifecycle", {}).get("stateful")
    }
    if not native_stateful:
        pytest.skip("this trainvm binary does not publish lifecycle declarations")
    locally_stateless = {row.contract for row in OPERATIONS if not row.mutation_sites}
    assert not (native_stateful & locally_stateless), (
        "the authority calls these profiles stateful but the enumeration found "
        "no optimizer mutation in them, so their clean result is unfounded: "
        f"{sorted(native_stateful & locally_stateless)}"
    )
