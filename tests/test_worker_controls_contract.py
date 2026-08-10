"""Static contract checks between adapters, the worker runtime, and its doubles.

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
* a `controls.<name>(...)` call site naming a member `WorkerControlRuntime` does
  not have, or passing an argument shape its signature would refuse;
* a double that disagrees with the runtime about either -- accepting a call the
  real object would reject, so the test proves nothing about production.

None of this executes the runtime, so none of it replaces a production-controls
test. It fails in the commit that introduces the disagreement, which is the
property the shipped defect needed and did not have.
"""

from __future__ import annotations

import ast
import importlib
import inspect
import pathlib
import sys

import pytest

REPOSITORY = pathlib.Path(__file__).resolve().parent.parent
SOURCE = REPOSITORY / "src"
if str(SOURCE) not in sys.path:  # pragma: no cover - import bootstrap
    sys.path.insert(0, str(SOURCE))

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

# The names an adapter binds a `WorkerControlRuntime` to. The runtime arrives as
# a parameter -- `hf_multimodal_sft.py` annotates it `Any`, so an annotation is
# not usable as the filter -- and a name that is *assigned* in the same function
# is something else: `adapter_recursive.py` builds a local `controls` dictionary
# out of a proposal document and would otherwise be read as a runtime.
CONTROL_RECEIVERS = ("controls", "worker_controls")

RUNTIME_API = frozenset(
    name for name in dir(WorkerControlRuntime) if not name.startswith("_")
)


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


def _control_call_sites() -> list[tuple[str, int, str, int, tuple[str, ...], bool]]:
    """(location, line, member, positional count, keywords, unpacked) per call."""

    sites: dict[tuple[str, int, int], tuple[str, int, str, int, tuple[str, ...], bool]]
    sites = {}
    for path in _python_files(RWKV_LAB):
        tree = _parse(path)
        for scope in ast.walk(tree):
            if not isinstance(scope, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            arguments = scope.args
            declared = {
                argument.arg
                for argument in (
                    *arguments.posonlyargs,
                    *arguments.args,
                    *arguments.kwonlyargs,
                )
                if argument.arg in CONTROL_RECEIVERS
            }
            if not declared:
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
            receivers = declared - rebound
            if not receivers:
                continue
            for node in ast.walk(scope):
                if not isinstance(node, ast.Call):
                    continue
                function = node.func
                if not isinstance(function, ast.Attribute):
                    continue
                if not isinstance(function.value, ast.Name):
                    continue
                if function.value.id not in receivers:
                    continue
                unpacked = any(
                    isinstance(argument, ast.Starred) for argument in node.args
                ) or any(keyword.arg is None for keyword in node.keywords)
                key = (str(path.relative_to(REPOSITORY)), node.lineno, node.col_offset)
                sites[key] = (
                    str(path.relative_to(REPOSITORY)),
                    node.lineno,
                    function.attr,
                    len(node.args),
                    tuple(keyword.arg for keyword in node.keywords if keyword.arg),
                    unpacked,
                )
    return sorted(sites.values())


CONTROL_CALL_SITES = _control_call_sites()


def test_control_call_sites_bind_against_the_real_runtime() -> None:
    """Every `controls.<name>(...)` a trainer writes must be a real call.

    A double answers to whatever the caller invents, so a renamed or re-signed
    runtime method leaves its callers green. Binding the recorded argument shape
    against `inspect.signature` costs nothing and refuses both.
    """

    assert CONTROL_CALL_SITES, "no worker-control call sites were found"
    problems: list[str] = []
    for location, lineno, member, positional, keywords, unpacked in CONTROL_CALL_SITES:
        attribute = inspect.getattr_static(WorkerControlRuntime, member, None)
        if attribute is None:
            problems.append(
                f"{location}:{lineno}: WorkerControlRuntime has no {member}"
            )
            continue
        if isinstance(attribute, property):
            problems.append(
                f"{location}:{lineno}: {member} is a property, called as a method"
            )
            continue
        if unpacked:
            continue
        signature = inspect.signature(attribute)
        try:
            signature.bind(None, *[None] * positional, **dict.fromkeys(keywords))
        except TypeError as error:
            problems.append(
                f"{location}:{lineno}: {member}{signature} refuses "
                f"{positional} positional + {sorted(keywords)}: {error}"
            )
    assert not problems, "worker-control call site does not match the runtime:\n" + (
        "\n".join(problems)
    )


def _double_classes() -> list[tuple[str, int, str, dict[str, ast.AST]]]:
    """Test classes that stand in for `WorkerControlRuntime`.

    A double is recognised by agreement rather than by name: `Controls`,
    `_LoopControls`, `_Recorder` and `_PublishingControls` share no convention
    beyond implementing at least two members the runtime has.
    """

    doubles: list[tuple[str, int, str, dict[str, ast.AST]]] = []
    here = pathlib.Path(__file__).resolve()
    for path in _python_files(here.parent):
        if path == here:
            # This module's own `_Transport` is a *session* double, and the
            # session Protocol shares three attribute names with the runtime
            # (`step_zero_eval_gate_required` and friends are read off the
            # session by `getattr`). Counting it would inflate the census the
            # check below guards.
            continue
        for node in ast.walk(_parse(path)):
            if not isinstance(node, ast.ClassDef):
                continue
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
            stubbed = {
                name: member
                for name, member in members.items()
                if name in RUNTIME_API
            }
            if len(stubbed) >= 2:
                doubles.append(
                    (str(path.relative_to(REPOSITORY)), node.lineno, node.name, stubbed)
                )
    return doubles


CONTROLS_DOUBLES = _double_classes()


def test_controls_doubles_were_found() -> None:
    """Guard the discovery itself.

    Both checks below are vacuous if nothing is recognised as a double, and a
    vacuous check is indistinguishable from a passing one in a test report.
    """

    assert len(CONTROLS_DOUBLES) >= 4, [
        (location, name) for location, _, name, _ in CONTROLS_DOUBLES
    ]


@pytest.mark.parametrize(
    "location,lineno,name",
    [(location, lineno, name) for location, lineno, name, _ in CONTROLS_DOUBLES],
    ids=[f"{name}" for _, _, name, _ in CONTROLS_DOUBLES],
)
def test_controls_double_agrees_with_the_runtime(
    location: str, lineno: int, name: str
) -> None:
    """A double must answer every production call the way the runtime would.

    Not identically -- a double exists to observe and to fail on demand -- but
    with a signature that accepts exactly the call sites production writes. A
    double that accepts a call the runtime refuses is the failure this card is
    about, one step earlier than an unresolvable import: the test passes and the
    real object would have raised.
    """

    stubbed = next(
        members
        for candidate, line, _, members in CONTROLS_DOUBLES
        if candidate == location and line == lineno
    )
    problems: list[str] = []
    for member, node in sorted(stubbed.items()):
        runtime_attribute = inspect.getattr_static(WorkerControlRuntime, member, None)
        if runtime_attribute is None:
            # `_double_classes` only selects members the runtime has, so this is
            # unreachable unless the runtime changed under a cached module. The
            # call-site check above owns that report; do not duplicate it here.
            continue
        callable_double = isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        callable_runtime = not isinstance(runtime_attribute, property) and callable(
            runtime_attribute
        )
        if callable_double != callable_runtime:
            problems.append(
                f"{member}: double is "
                f"{'a method' if callable_double else 'an attribute'} while the "
                f"runtime is {'a method' if callable_runtime else 'an attribute'}"
            )
            continue
        if not callable_double:
            continue
        for site, site_line, site_member, positional, keywords, unpacked in (
            CONTROL_CALL_SITES
        ):
            if site_member != member or unpacked:
                continue
            try:
                _bind_double(node, positional, keywords)
            except TypeError as error:
                problems.append(
                    f"{member}: refuses the call at {site}:{site_line} "
                    f"({positional} positional + {sorted(keywords)}) "
                    f"that the runtime accepts: {error}"
                )
    assert not problems, f"{name} at {location}:{lineno} disagrees with the runtime:\n" + (
        "\n".join(problems)
    )


def _bind_double(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    positional: int,
    keywords: tuple[str, ...],
) -> None:
    """Bind a call shape against a double's declared parameters."""

    arguments = node.args
    parameters = [
        inspect.Parameter(argument.arg, inspect.Parameter.POSITIONAL_ONLY)
        for argument in arguments.posonlyargs
    ]
    parameters += [
        inspect.Parameter(argument.arg, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        for argument in arguments.args
    ]
    if arguments.vararg is not None:
        parameters.append(
            inspect.Parameter(arguments.vararg.arg, inspect.Parameter.VAR_POSITIONAL)
        )
    parameters += [
        inspect.Parameter(argument.arg, inspect.Parameter.KEYWORD_ONLY)
        for argument in arguments.kwonlyargs
    ]
    if arguments.kwarg is not None:
        parameters.append(
            inspect.Parameter(arguments.kwarg.arg, inspect.Parameter.VAR_KEYWORD)
        )
    defaults = len(arguments.defaults)
    if defaults:
        positional_names = [
            parameter
            for parameter in parameters
            if parameter.kind
            in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            )
        ]
        for index, parameter in enumerate(positional_names[-defaults:]):
            place = parameters.index(parameter)
            parameters[place] = parameter.replace(default=None)
    for argument, default in zip(arguments.kwonlyargs, arguments.kw_defaults):
        if default is None:
            continue
        place = next(
            index
            for index, parameter in enumerate(parameters)
            if parameter.name == argument.arg
            and parameter.kind is inspect.Parameter.KEYWORD_ONLY
        )
        parameters[place] = parameters[place].replace(default=None)
    inspect.Signature(parameters).bind(
        *[None] * (positional + 1), **dict.fromkeys(keywords)
    )


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

    def heartbeat(self, optimizer_step: int, phase: str, *, wait: bool = False) -> int:
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
