"""Static contract checks between adapters, the worker runtimes, and their doubles.

`WorkerControlRuntime.publish_artifact` shipped importing `ArtifactPublisher`
from `.artifact`, a name that module has never exported, so every call raised
`ImportError` before it reached the session. It survived three days and several
pull requests with a green suite, because the only route that calls it --
HF multimodal SFT publishing its `test_eval` bundle -- is exercised in tests
through a hand-written controls double, and the double's `publish_artifact`
returns a `SimpleNamespace` without importing anything.

The general statement is the point: **every method a controls double stubs is a
production path no test has executed**, so the union of what the doubles stub is
the untested surface. Coverage of that surface by real end-to-end tests is
expensive and arrives route by route. These checks are the cheap half, and they
are exactly the shape of defect the doubles hide:

* a deferred `from .x import Y` inside the worker package naming something `x`
  does not export -- the shipped defect, one import away from being caught;
* a `<runtime>.<name>(...)` call site naming a member the runtime does not have,
  or passing an argument shape its signature would refuse;
* a double that accepts a call the real object would reject, so the test proves
  nothing about production.

The checks cover six runtimes -- `WorkerControlRuntime`, `WorkerSession`,
`WorkerObservability`, `WorkerExecutionPhases`, `WorkerPublicationRuntime` and
`WorkerStepProfiler` -- because the double census is the same code for each and
the session doubles are the largest in the suite.

Each candidate double is attributed to the runtime it agrees with *most*, rather
than to every runtime it happens to overlap. That is not a tidiness preference:
this module's own `_Transport` is a session double, and the session supplies
three attributes the control runtime also exposes (`step_zero_eval_gate_required`
and friends are read off the session by `getattr`), so a first revision counted
it as a fourth *controls* double and propped up the very census that guards the
parametrisation from emptying. Only the mutation table showed it. Argmax
attribution fixes that generally; a tie is reported rather than resolved.

The third check runs in one direction on purpose. A double that is *stricter*
than its runtime -- the HF engine's `Observability` requires the `phase` that
`WorkerObservability.optimizer_step` defaults, and omits the `sample_weight` and
`labels` that `publish_if_declared` defaults -- refuses calls six other trainers
write, and that is not a defect: those trainers use the real object, and a test
that handed this double such a call would fail loudly. The dangerous direction is
the other one, where a double *accepts* what the runtime would refuse, because
that green test is the whole failure this module exists for. So the rule is that
a double must be no more permissive than its runtime: it must declare every
parameter the runtime requires, and must not answer to a parameter name the
runtime does not have.

None of this executes the runtimes, so none of it replaces a production-controls
test. It fails in the commit that introduces the disagreement, which is the
property the shipped defect needed and did not have.
"""

from __future__ import annotations

import ast
import collections
import importlib
import inspect
import pathlib
import sys
from dataclasses import dataclass

import pytest

REPOSITORY = pathlib.Path(__file__).resolve().parent.parent
SOURCE = REPOSITORY / "src"
if str(SOURCE) not in sys.path:  # pragma: no cover - import bootstrap
    sys.path.insert(0, str(SOURCE))

import rwkv_lab.trainvm_worker as worker
from rwkv_lab.trainvm_worker.controls import (
    WorkerControlError,
    WorkerControlRuntime,
    WorkerResourcesReleasedPause,
)
from rwkv_lab.trainvm_worker.session import (
    CommandKind,
    LifecycleDisposition,
    WorkerCommand,
)

WORKER_PACKAGE = SOURCE / "rwkv_lab" / "trainvm_worker"
RWKV_LAB = SOURCE / "rwkv_lab"

RUNTIMES = {
    name: getattr(worker, name)
    for name in (
        "WorkerControlRuntime",
        "WorkerSession",
        "WorkerObservability",
        "WorkerExecutionPhases",
        "WorkerPublicationRuntime",
        "WorkerStepProfiler",
    )
}

RUNTIME_API = {
    runtime: frozenset(name for name in dir(cls) if not name.startswith("_"))
    for runtime, cls in RUNTIMES.items()
}

# The names an adapter binds a `WorkerControlRuntime` to without annotating it.
# `hf_multimodal_sft.py` annotates its parameter `Any`, so an annotation is not
# usable as the filter there -- and a name that is *assigned* in the same
# function is something else: `adapter_recursive.py` builds a local `controls`
# dictionary out of a proposal document and would otherwise be read as a runtime.
# Every other runtime below *is* annotated at its call sites, so this fallback
# stays a two-name exception rather than growing.
CONTROL_RECEIVERS = ("controls", "worker_controls")


def _module_name(path: pathlib.Path) -> str:
    parts = path.relative_to(SOURCE).with_suffix("").parts
    name = ".".join(parts)
    return name[: -len(".__init__")] if name.endswith(".__init__") else name


def _package_of(path: pathlib.Path) -> str:
    name = _module_name(path)
    if path.name == "__init__.py":
        return name
    return name.rsplit(".", 1)[0]


def _resolve_relative(package: str, node: ast.ImportFrom) -> str:
    base = package
    for _ in range(node.level - 1):
        base = base.rsplit(".", 1)[0]
    return f"{base}.{node.module}" if node.module else base


def _python_files(root: pathlib.Path) -> list[pathlib.Path]:
    return sorted(path for path in root.rglob("*.py") if "__pycache__" not in path.parts)


def _parse(path: pathlib.Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), str(path))


def _relative_imports(path: pathlib.Path) -> list[tuple[int, str, str]]:
    """Every `from .x import Y` in one file, module level or function local."""

    package = _package_of(path)
    found: list[tuple[int, str, str]] = []
    for node in ast.walk(_parse(path)):
        if not isinstance(node, ast.ImportFrom) or not node.level:
            continue
        target = _resolve_relative(package, node)
        for alias in node.names:
            if alias.name != "*":
                found.append((node.lineno, target, alias.name))
    return found


def test_worker_package_relative_imports_name_real_exports() -> None:
    """The shipped defect, stated as a check.

    A deferred import inside a method body is invisible until that method runs,
    and the runtime uses them deliberately to keep the scalar-control module
    free of checkpoint and artifact mechanics. That is fine as long as something
    other than production traffic proves the names exist.
    """

    broken: list[str] = []
    checked = 0
    for path in _python_files(WORKER_PACKAGE):
        for lineno, target, name in _relative_imports(path):
            checked += 1
            try:
                module = importlib.import_module(target)
            except Exception as error:  # pragma: no cover - reported, not raised
                broken.append(
                    f"{path.relative_to(REPOSITORY)}:{lineno}: "
                    f"cannot import {target}: {error!r}"
                )
                continue
            if hasattr(module, name):
                continue
            try:
                importlib.import_module(f"{target}.{name}")
            except Exception:
                broken.append(
                    f"{path.relative_to(REPOSITORY)}:{lineno}: "
                    f"{target} does not export {name}"
                )
    assert checked, "no relative imports were inspected"
    assert not broken, "worker package imports a name its module does not export:\n" + (
        "\n".join(broken)
    )


@dataclass(frozen=True)
class _CallSite:
    runtime: str
    location: str
    lineno: int
    member: str
    positional: int
    keywords: tuple[str, ...]
    unpacked: bool


def _annotated_runtimes(node: ast.expr | None) -> set[str]:
    """The runtime names an annotation mentions, through `X | None` and strings."""

    if node is None:
        return set()
    if isinstance(node, ast.Name):
        return {node.id} & RUNTIMES.keys()
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return {node.value.split("|")[0].strip()} & RUNTIMES.keys()
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        return _annotated_runtimes(node.left) | _annotated_runtimes(node.right)
    return set()


def _call_shape(node: ast.Call) -> tuple[int, tuple[str, ...], bool]:
    unpacked = any(isinstance(argument, ast.Starred) for argument in node.args) or any(
        keyword.arg is None for keyword in node.keywords
    )
    return (
        len(node.args),
        tuple(keyword.arg for keyword in node.keywords if keyword.arg),
        unpacked,
    )


def _runtime_call_sites() -> list[_CallSite]:
    """Every production call on something known to be one of the runtimes.

    Three ways of knowing, in one pass. A parameter annotated with the runtime
    covers five of the six. A parameter *named* `controls`/`worker_controls`
    covers the sixth, whose only unannotated binding is the HF engine's `Any`.
    And `self._session.<name>(...)` inside the worker package covers
    `WorkerSession`, whose callers are the publishers themselves: they annotate
    the constructor parameter with a local structural Protocol rather than with
    `WorkerSession`, so nothing else would find those 25 calls -- and they are
    what makes the session doubles' agreement check non-vacuous.
    """

    sites: dict[tuple[str, str, int, int], _CallSite] = {}

    def record(runtime: str, path: pathlib.Path, node: ast.Call, member: str) -> None:
        positional, keywords, unpacked = _call_shape(node)
        location = str(path.relative_to(REPOSITORY))
        sites[(runtime, location, node.lineno, node.col_offset)] = _CallSite(
            runtime, location, node.lineno, member, positional, keywords, unpacked
        )

    for path in _python_files(RWKV_LAB):
        tree = _parse(path)
        for scope in ast.walk(tree):
            if not isinstance(scope, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            arguments = scope.args
            bound: dict[str, str] = {}
            for argument in (
                *arguments.posonlyargs,
                *arguments.args,
                *arguments.kwonlyargs,
            ):
                if argument.arg in CONTROL_RECEIVERS:
                    bound[argument.arg] = "WorkerControlRuntime"
                for runtime in _annotated_runtimes(argument.annotation):
                    bound[argument.arg] = runtime
            if not bound:
                continue
            rebound = {
                target.id
                for node in ast.walk(scope)
                if isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign))
                for target in (
                    node.targets if isinstance(node, ast.Assign) else [node.target]
                )
                if isinstance(target, ast.Name)
            }
            for node in ast.walk(scope):
                if not isinstance(node, ast.Call):
                    continue
                function = node.func
                if not isinstance(function, ast.Attribute):
                    continue
                if not isinstance(function.value, ast.Name):
                    continue
                runtime = bound.get(function.value.id)
                if runtime is None or function.value.id in rebound:
                    continue
                record(runtime, path, node, function.attr)
        if WORKER_PACKAGE not in path.parents:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            function = node.func
            if not isinstance(function, ast.Attribute):
                continue
            receiver = function.value
            if (
                isinstance(receiver, ast.Attribute)
                and receiver.attr == "_session"
                and isinstance(receiver.value, ast.Name)
                and receiver.value.id == "self"
            ):
                record("WorkerSession", path, node, function.attr)
    return sorted(
        sites.values(), key=lambda site: (site.runtime, site.location, site.lineno)
    )


RUNTIME_CALL_SITES = _runtime_call_sites()

# Every runtime must contribute call sites, or its half of the contract is
# vacuous and reads as passing. `WorkerPublicationRuntime` contributes exactly
# one -- `handlers.py`'s MageFlow live-eval publisher -- which is the number, not
# a rounding of it.
EXPECTED_CALL_SITES = {
    "WorkerControlRuntime": 120,
    "WorkerObservability": 30,
    "WorkerSession": 20,
    "WorkerExecutionPhases": 12,
    "WorkerStepProfiler": 10,
    "WorkerPublicationRuntime": 1,
}


def test_every_runtime_contributes_call_sites() -> None:
    """Guard the discovery, not the code under it.

    A binder that finds nothing agrees with a binder that finds everything and
    is satisfied. These floors are set below the counts observed when the check
    landed so ordinary deletions do not trip them, and high enough that a
    discovery mode silently ceasing to match does.
    """

    counts = collections.Counter(site.runtime for site in RUNTIME_CALL_SITES)
    assert {
        runtime: counts[runtime] >= floor for runtime, floor in EXPECTED_CALL_SITES.items()
    } == dict.fromkeys(EXPECTED_CALL_SITES, True), dict(counts)


def test_call_sites_bind_against_the_real_runtime() -> None:
    """Every `<runtime>.<name>(...)` a caller writes must be a real call.

    A double answers to whatever the caller invents, so a renamed or re-signed
    runtime method leaves its callers green. Binding the recorded argument shape
    against `inspect.signature` costs nothing and refuses both.
    """

    problems: list[str] = []
    for site in RUNTIME_CALL_SITES:
        attribute = inspect.getattr_static(RUNTIMES[site.runtime], site.member, None)
        if attribute is None:
            problems.append(
                f"{site.location}:{site.lineno}: {site.runtime} has no {site.member}"
            )
            continue
        if isinstance(attribute, property):
            problems.append(
                f"{site.location}:{site.lineno}: {site.runtime}.{site.member} is a "
                "property, called as a method"
            )
            continue
        if site.unpacked:
            continue
        signature = inspect.signature(attribute)
        try:
            signature.bind(
                None, *[None] * site.positional, **dict.fromkeys(site.keywords)
            )
        except TypeError as error:
            problems.append(
                f"{site.location}:{site.lineno}: {site.runtime}.{site.member}"
                f"{signature} refuses {site.positional} positional + "
                f"{sorted(site.keywords)}: {error}"
            )
    assert not problems, "a call site does not match its runtime:\n" + "\n".join(
        problems
    )


PROPERTY_DECORATORS = frozenset({"property", "cached_property"})


def _is_property(node: ast.AST) -> bool:
    """A `@property` reads as an attribute however it is spelled.

    `_Channel` in the execution-phase suite declares
    `execution_phase_cancellation` with `@property`, matching the session's own
    property. Reading the `def` and stopping there classifies it as a method and
    reports a mismatch that is not there.
    """

    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return False
    for decorator in node.decorator_list:
        target = decorator.func if isinstance(decorator, ast.Call) else decorator
        name = target.attr if isinstance(target, ast.Attribute) else getattr(
            target, "id", ""
        )
        if name in PROPERTY_DECORATORS:
            return True
    return False


@dataclass(frozen=True)
class _Double:
    location: str
    lineno: int
    name: str
    runtime: str
    stubbed: tuple[tuple[str, ast.AST], ...]
    scores: tuple[tuple[str, int], ...]

    @property
    def label(self) -> str:
        module = pathlib.Path(self.location).stem.removeprefix("test_")
        return f"{self.name}-{self.runtime.removeprefix('Worker')}-{module}"


def _class_members(node: ast.ClassDef) -> dict[str, ast.AST]:
    members: dict[str, ast.AST] = {}
    for statement in node.body:
        if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
            members[statement.name] = statement
        elif isinstance(statement, ast.Assign):
            for target in statement.targets:
                if isinstance(target, ast.Name):
                    members[target.id] = statement
        elif isinstance(statement, ast.AnnAssign) and isinstance(
            statement.target, ast.Name
        ):
            members[statement.target.id] = statement
    return members


def _double_classes() -> tuple[list[_Double], list[str]]:
    """Test classes that stand in for one of the worker runtimes.

    A double is recognised by agreement rather than by name -- `Controls`,
    `_LoopControls`, `_Recorder`, `_PublishingControls`, `FakeSession`,
    `EmptySession`, `_Channel`, `_Transport` and `Observability` share no
    convention beyond implementing at least two members some runtime has -- and
    attributed to whichever runtime it agrees with most. A tie is not resolved;
    it is reported, because a class that matches two runtimes equally is either
    a genuinely ambiguous double or a sign the runtimes have converged, and
    both want a human.
    """

    doubles: list[_Double] = []
    ambiguous: list[str] = []
    for path in _python_files(pathlib.Path(__file__).resolve().parent):
        for node in ast.walk(_parse(path)):
            if not isinstance(node, ast.ClassDef):
                continue
            members = _class_members(node)
            scores = {
                runtime: len(members.keys() & api)
                for runtime, api in RUNTIME_API.items()
            }
            best = max(scores.values())
            if best < 2:
                continue
            location = str(path.relative_to(REPOSITORY))
            winners = sorted(
                runtime for runtime, score in scores.items() if score == best
            )
            if len(winners) > 1:
                ambiguous.append(
                    f"{location}:{node.lineno}: {node.name} matches "
                    f"{winners} equally ({best} members each)"
                )
                continue
            api = RUNTIME_API[winners[0]]
            doubles.append(
                _Double(
                    location=location,
                    lineno=node.lineno,
                    name=node.name,
                    runtime=winners[0],
                    stubbed=tuple(
                        (name, member)
                        for name, member in sorted(members.items())
                        if name in api
                    ),
                    scores=tuple(sorted(scores.items())),
                )
            )
    return doubles, ambiguous


RUNTIME_DOUBLES, AMBIGUOUS_DOUBLES = _double_classes()

# Floors, again set below the counts observed when the check landed. The census
# exists because the parametrisation below is vacuous without it, and a vacuous
# parametrisation reports as a pass.
EXPECTED_DOUBLES = {
    "WorkerControlRuntime": 4,
    "WorkerSession": 6,
    "WorkerObservability": 1,
}


def test_runtime_doubles_were_found() -> None:
    counts = collections.Counter(double.runtime for double in RUNTIME_DOUBLES)
    assert {
        runtime: counts[runtime] >= floor for runtime, floor in EXPECTED_DOUBLES.items()
    } == dict.fromkeys(EXPECTED_DOUBLES, True), [
        (double.location, double.name, double.runtime) for double in RUNTIME_DOUBLES
    ]


def test_no_double_matches_two_runtimes_equally() -> None:
    assert not AMBIGUOUS_DOUBLES, "\n".join(AMBIGUOUS_DOUBLES)


# A `**kwargs` catch-all cannot express "this keyword is required", so the
# permissiveness check below cannot conclude anything about these members: each
# accepts a call that omits a keyword its runtime demands, and no signature
# comparison can tell an intentional pass-through from an accident. That is a
# genuine blind spot in the cheap check, so it is enumerated rather than
# implicit -- a new one has to be added deliberately, and adding one is the
# moment to ask whether that member wants a real test instead.
#
# Two of the nine forward to the real object (`_Recorder`, `_Channel`); the rest
# are no-op stubs whose suites never exercise the omitted keyword.
CATCH_ALL_MEMBERS = {
    ("tests/test_hf_multimodal_sft_resume_gate.py", "_Recorder", "publish_policy_checkpoint"),
    ("tests/test_mage_flow_expert_train.py", "_EmptyControlSession", "acknowledge_controls"),
    ("tests/test_trainvm_mageflow_controls.py", "EmptySession", "acknowledge_controls"),
    ("tests/test_trainvm_observability.py", "FakeSession", "heartbeat"),
    ("tests/test_trainvm_observability.py", "FakeSession", "metric"),
    ("tests/test_trainvm_qwen_controls.py", "EmptySession", "acknowledge_controls"),
    ("tests/test_trainvm_worker_execution_phases.py", "_Channel", "execution_phase_receipt"),
    ("tests/test_worker_controls_contract.py", "_Transport", "acknowledge_checkpoint"),
    ("tests/test_worker_controls_contract.py", "_Transport", "acknowledge_controls"),
}


def test_catch_all_double_members_are_enumerated() -> None:
    """Keep the blind spot counted rather than implicit."""

    found = {
        (double.location, double.name, member)
        for double in RUNTIME_DOUBLES
        for member, node in double.stubbed
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.args.kwarg is not None
    }
    assert found == CATCH_ALL_MEMBERS, {
        "unlisted": sorted(found - CATCH_ALL_MEMBERS),
        "listed but gone": sorted(CATCH_ALL_MEMBERS - found),
    }


@pytest.mark.parametrize(
    "double", RUNTIME_DOUBLES, ids=[double.label for double in RUNTIME_DOUBLES]
)
def test_double_is_no_more_permissive_than_its_runtime(double: _Double) -> None:
    """A double must not accept a call the real object would refuse.

    That is the direction that matters. A stricter double refuses calls other
    routes write and its own tests never make, which is not a defect -- see the
    module docstring for the `Observability` case that proves the point. A
    *laxer* double is the failure this module exists for: the suite goes green
    on a call the runtime would have rejected.

    So: every parameter the runtime requires must be declared, and every name
    the double answers to must exist on the runtime.
    """

    runtime = RUNTIMES[double.runtime]
    problems: list[str] = []
    for member, node in double.stubbed:
        real = inspect.getattr_static(runtime, member, None)
        if real is None:  # pragma: no cover - selection guarantees membership
            continue
        double_is_attribute = not isinstance(
            node, (ast.FunctionDef, ast.AsyncFunctionDef)
        ) or _is_property(node)
        real_is_attribute = isinstance(real, property) or not callable(real)
        if double_is_attribute != real_is_attribute:
            problems.append(
                f"{member}: double is "
                f"{'an attribute' if double_is_attribute else 'a method'} while "
                f"{double.runtime} is "
                f"{'an attribute' if real_is_attribute else 'a method'}"
            )
            continue
        if double_is_attribute:
            continue
        problems.extend(
            f"{member}: {problem}"
            for problem in _permissiveness(inspect.signature(real), node)
        )
    assert not problems, (
        f"{double.name} at {double.location}:{double.lineno} stands in for "
        f"{double.runtime} and is more permissive than it:\n" + "\n".join(problems)
    )


def _permissiveness(
    signature: inspect.Signature, node: ast.FunctionDef | ast.AsyncFunctionDef
) -> list[str]:
    """Where the double accepts calls the runtime's signature would refuse.

    Positional parameters are compared by arity and keyword parameters by name,
    which is not a shortcut -- it is how callers reach them. Several doubles
    rename an unused positional parameter to `_step` or `_applier` to say so;
    every caller passes those positionally, so the name is unobservable and
    flagging it would be a convention complaint dressed up as a contract
    violation. A keyword name, by contrast, is exactly what the caller types.
    """

    arguments = node.args
    problems: list[str] = []
    double_keywords = {argument.arg for argument in arguments.kwonlyargs}
    double_positional = [
        *arguments.posonlyargs,
        *arguments.args,
    ]
    if double_positional and double_positional[0].arg == "self":
        double_positional = double_positional[1:]
    double_required_positional = len(double_positional) - len(arguments.defaults)
    # Declaring a required parameter is not enough: a default makes it optional
    # again, and the double then accepts the call that omits it while the runtime
    # refuses. `publish_if_declared(self, name, value, step=0)` is exactly that.
    optional_names = {
        argument.arg
        for argument in double_positional[
            len(double_positional) - len(arguments.defaults) :
        ]
    } | {
        argument.arg
        for argument, default in zip(arguments.kwonlyargs, arguments.kw_defaults)
        if default is not None
    }

    runtime_positional = [
        parameter
        for name, parameter in signature.parameters.items()
        if name != "self"
        and parameter.kind
        in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        )
    ]
    runtime_required_positional = sum(
        parameter.default is inspect.Parameter.empty for parameter in runtime_positional
    )
    if (
        arguments.vararg is None
        and double_required_positional < runtime_required_positional
    ):
        problems.append(
            f"the runtime requires {runtime_required_positional} positional "
            f"argument(s) and the double requires {double_required_positional}, "
            "so a caller that omits one passes here and fails in production"
        )

    runtime_keyword_capable = {
        name
        for name, parameter in signature.parameters.items()
        if name != "self"
        and parameter.kind
        in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        )
    }
    accepts_any_keyword = any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
    for name, parameter in signature.parameters.items():
        if (
            parameter.kind is not inspect.Parameter.KEYWORD_ONLY
            or parameter.default is not inspect.Parameter.empty
        ):
            continue
        declared = name in double_keywords or any(
            argument.arg == name for argument in double_positional
        )
        if declared and name not in optional_names:
            continue
        if not declared and arguments.kwarg is not None:
            # A `**kwargs` double cannot express "required", so this check
            # cannot conclude for one. That is a stated limit rather than a
            # silent pass -- see `test_delegating_doubles_are_named`.
            continue
        problems.append(
            f"the runtime requires the keyword {name!r} and the double "
            + (
                "gives it a default"
                if declared
                else "does not declare it"
            )
            + ", so a caller that omits it passes here and fails in production"
        )
    if not accepts_any_keyword:
        for name in sorted(double_keywords - runtime_keyword_capable):
            problems.append(
                f"the double answers to the keyword {name!r}, which the runtime "
                "has no parameter for, so a caller that passes it passes here "
                "and fails in production"
            )
    return problems


class _Transport:
    """The session wire, not the object under test.

    `WorkerControlRuntime` talks to a controller through this Protocol, so a
    test that drives the real runtime still has to supply one. Everything it
    records is an acknowledgement the runtime decided to send; it decides
    nothing itself, which is the property the doubles this module inspects do
    not have.
    """

    attempt_baseline_optimizer_step = 0
    step_zero_eval_gate_required = False
    step_zero_eval_gate_satisfied = False

    def __init__(self, *commands: WorkerCommand) -> None:
        self.commands = list(commands)
        self.lifecycle: list[tuple[str, LifecycleDisposition, int, str]] = []

    def poll_commands(self, maximum: int | None = None) -> tuple[WorkerCommand, ...]:
        taken = tuple(self.commands if maximum is None else self.commands[:maximum])
        del self.commands[: len(taken)]
        return taken

    def acknowledge_controls(self, command, disposition, **keywords) -> int:
        raise AssertionError(command)

    def acknowledge_checkpoint(self, command, disposition, **keywords) -> int:
        raise AssertionError(command)

    def acknowledge_lifecycle(
        self,
        command,
        disposition,
        *,
        optimizer_step: int = 0,
        artifact_id: str = "",
        diagnostics=(),
        wait: bool = True,
    ) -> int:
        self.lifecycle.append(
            (command.command_id, disposition, optimizer_step, artifact_id)
        )
        return len(self.lifecycle)

    def heartbeat(
        self,
        optimizer_step: int,
        phase: str,
        *,
        execution_phase: object | None = None,
        wait: bool = False,
    ) -> int:
        raise AssertionError(phase)


# Each publisher refuses anything but its own frozen request type, and a double
# enforces none of it: `Controls.publish_artifact` in the engine suite accepts
# whatever it is handed. Coverage of the whole worker package over the suite
# shows every one of these refusals unexecuted, so an adapter that assembled the
# wrong request would have been told so only in production.
UNTYPED_REFUSALS = (
    "publish_policy_checkpoint",
    "publish_evaluation_examples",
    "publish_artifact",
    "publish_final_evaluation",
)


@pytest.mark.parametrize("member", UNTYPED_REFUSALS)
def test_real_runtime_refuses_an_untyped_publication_request(member: str) -> None:
    runtime = WorkerControlRuntime(_Transport(), {}, 0)
    with pytest.raises(WorkerControlError):
        getattr(runtime, member)({"source_directory": "/tmp", "output_name": "x"})


def test_real_runtime_refuses_an_untyped_evaluation_gallery() -> None:
    """The gallery refuses on either half of its typed evidence."""

    runtime = WorkerControlRuntime(_Transport(), {}, 0)
    with pytest.raises(WorkerControlError):
        runtime.publish_evaluation_gallery({"output_name": "x"}, checkpoint=object())


def test_publish_requested_checkpoint_directory_reaches_the_real_runtime() -> None:
    """The convenience wrapper nine trainers call and no test executed.

    Coverage over the whole suite records both of its statements unexecuted
    while `mage_flow_*`, `qwen_ao3_cpt`, `train_mla`, `vision_*` and
    `vision_rwkv_student_train` call it in production. A resource-releasing
    pause that does not ask for a checkpoint first is the one path through it
    that publishes nothing, so it exercises the wrapper -- the deferred import
    and the typed request it assembles -- without a filesystem.
    """

    pause = WorkerCommand(
        7,
        "pause-7",
        CommandKind.PAUSE,
        checkpoint_first=False,
        release_resources=True,
    )
    transport = _Transport(pause)
    runtime = WorkerControlRuntime(transport, {}, 0)
    with pytest.raises(WorkerResourcesReleasedPause):
        runtime.publish_requested_checkpoint_directory(
            "/nonexistent-source",
            optimizer_step=11,
            resume_grade="compatible",
            state_components=("model",),
        )
    assert transport.lifecycle == [("pause-7", LifecycleDisposition.APPLIED, 0, "")]
