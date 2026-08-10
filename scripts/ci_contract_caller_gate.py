#!/usr/bin/env python3
"""Fail when an adapter the native registry advertises has no production caller.

This is the Python half of ``scripts/ci_unwired_module_gate.py``, scoped to the
one population where "nothing in production reaches this" is a defect rather
than the intended state.

Why the population is the contract and not ``__all__``
------------------------------------------------------
The obvious Python analogue of the native gate -- a symbol exported from an
``__init__.py`` whose every importer is under ``tests/`` -- was implemented and
measured against ``origin/main`` at ``5bdf539`` before being dropped
(card-d198cc09):

    symbol level, rwkv_lab.trainvm_worker      79 of 117 exports unwired
    symbol level, rwkv_lab.trainvm_adapters    17 of  18 exports unwired
    module level, everything under src/        69 of 231 modules unwired

The instrument was not broken -- there are 14 real production importers of
``rwkv_lab.trainvm_worker`` under ``src/`` and it found them. Those 165 are
mostly not defects, for two structural reasons:

- ``rwkv_lab.trainvm_worker`` is a **published SDK**
  (docs/experiment-vm/PYTHON_WORKER_SDK.md). Its exports exist to be imported
  from outside this repository, so "no in-tree importer" is the intended state
  for much of it.
- ``src/rwkv_lab`` is a **research lever library**. ``coconut.py``,
  ``hedgehog.py``, ``llm_jepa.py`` and sixty others being test-only is the tree
  working as designed, not rot.

Shipping that gate would have cost a ~96-entry allowlist, which is an instrument
tuned until the number looks comfortable -- precisely the defect class the card
that asked for the gate was filed about. A smaller allowlist is not the fix; a
different question is.

The question this gate asks instead
-----------------------------------
The symbols the **native adapter registry** names are different. If one has no
production caller, an adapter the registry advertises to every composition
document cannot actually run, and nothing today says so. That population is 22
routes, it is already defined and pinned, and it needs no allowlist.

Three things are checked, and the first exists so the other two cannot be
quietly measured against a short list:

1. Every adapter contract in ``docs/experiment-vm/step-zero-arming.v1.json``
   -- regenerated from ``trainvm inspect-rwkv-lab-worker``, so it *is* the
   native registry -- resolves to a handler function defined in
   ``handlers.py``, and ``_HANDLERS`` advertises no contract the registry does
   not. Drift in either direction means the gate cannot know which handler
   serves an advertised route, so it refuses rather than reporting a smaller
   population.
2. Every handler so named is reachable from the worker's production entrypoint.
3. Every module in the contract's implementation package is reachable from that
   same entrypoint.

Reading ``_HANDLERS`` statically is a known trap, and it is not evaded by care
---------------------------------------------------------------------------
``ci_step_zero_arming_gate.py`` records why it does not do this: ``_HANDLERS``
ends in a ``**`` unpacking of a dict comprehension over ``PROFILE_ADAPTERS``,
so eight of the contracts exist as no string literal at all, "and an AST reader
silently reports thirteen". Silently is the whole problem -- a gate over 13 of
22 routes passes for the same reason a gate over 22 does.

So this does not read the literal, it **evaluates** it: the assignment's AST is
rewritten so each handler reference becomes its own name as a string, and the
result is evaluated by Python with ``PROFILE_ADAPTERS`` supplied from
``transformer_mla.py``. The comprehension runs, all 22 keys appear, and a
construct the rewriter cannot handle raises rather than disappearing. Check 1
is the second half of that guarantee: if the evaluation ever loses a route, the
pin still names it and the gate fails.

No torch is imported, so this runs in the seconds-fast schema job beside the
other gates rather than in the CPU job.

What counts as production
-------------------------
The graph reads ``src/rwkv_lab/`` and nothing else. The installed library holds
no test file -- every suite lives in ``tests/`` or ``trainvm/tests/`` -- so
production is a property of what is parsed rather than of a filter, the same
construction the native gate uses and for the same reason: a substring match on
"test" is how a previous count of this question went wrong.

The graph is the whole library while the *population* is the contract package,
and those two scopes are deliberately different. Restricting the graph as well
was the first version of this gate and it reported a false positive on its
first run: ``mageflow_phases`` is reached from ``handlers.py`` only by going out
to ``rwkv_lab.mage_flow_expert_train`` and back in through a deferred import, so
a graph over the worker packages alone called it unreached while the worker
process reaches it on every MageFlow run.

The root is not hardcoded. It is the ``trainvm-worker`` console script in
``pyproject.toml``, which is what the built worker artifact actually invokes
(``scripts/build_trainvm_worker_artifact.py`` writes that same import). A gate
rooted at a stale entrypoint would answer confidently about a process nobody
runs.

Why there is no allowlist
-------------------------
Because the honest count is zero. Every advertised route resolves, every handler
is reachable, and all 20 implementation modules are reached. An allowlist is
what a gate needs when its population contains legitimate exceptions; this one
does not, and adding an empty one would invite the first exception to be written
down instead of fixed. If a route ever needs one, that is a card, not a config
entry.

Two consequences worth stating, because they are how this gate will first
surprise someone:

- Reachability is over imports and name references, so it is as coarse as the
  native gate's include graph: a module that ``handlers.py`` still imports but
  no longer calls into reads as wired. That is the floor this instrument
  measures to, not an oversight. Catching the finer shape -- an exported symbol
  raised and caught inside one module and referenced by no handler, which is
  what ``WorkerExecutionPhaseCancelled`` did -- is candidate 3 on card-d198cc09
  and is a different check.
- ``__init__.py`` is excluded from check 3 by name. It is the package's
  re-export surface and the entrypoint reaches its siblings directly, so it is
  unreachable by construction and always would be.

Usage:
    python scripts/ci_contract_caller_gate.py [--repository .]
"""

from __future__ import annotations

import argparse
import ast
import json
import pathlib
import sys
import tomllib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from scripts.gate_verdict import verdict_line  # noqa: E402

PIN = "docs/experiment-vm/step-zero-arming.v1.json"
PACKAGE_ROOT = "src"
CONTRACT_PACKAGE = "rwkv_lab.trainvm_adapters"
WORKER_PACKAGE = "rwkv_lab.trainvm_worker"
HANDLERS = f"{CONTRACT_PACKAGE}.handlers"
TRANSFORMER_MLA = f"{CONTRACT_PACKAGE}.transformer_mla"
CONSOLE_SCRIPT = "trainvm-worker"

MODULE_NODE = "<module>"


class GateError(RuntimeError):
    """A condition under which the gate cannot answer, and must not guess."""


def module_name(path: pathlib.Path, root: pathlib.Path) -> str:
    relative = path.relative_to(root).with_suffix("")
    parts = list(relative.parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


class Module:
    """One parsed production module: what it defines and what it imports."""

    def __init__(self, name: str, path: pathlib.Path) -> None:
        self.name = name
        self.path = path
        self.tree = ast.parse(path.read_text(encoding="utf-8"))
        self.package = name if path.name == "__init__.py" else name.rpartition(".")[0]
        self.definitions: dict[str, ast.AST] = {}
        self.aliases: dict[str, tuple[str, str]] = {}
        self._scan()

    def _absolute(self, node: ast.ImportFrom) -> str | None:
        if not node.level:
            return node.module
        parts = self.package.split(".")
        climbed = parts[: len(parts) - (node.level - 1)] if node.level > 1 else parts
        return ".".join(climbed + ([node.module] if node.module else []))

    def _scan(self) -> None:
        for node in self.tree.body:
            if isinstance(
                node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
            ):
                self.definitions[node.name] = node
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        self.definitions[target.id] = node
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                self.definitions[node.target.id] = node
        # Imports are collected from the whole tree, not just the top level: a
        # deferred import inside a function is still a production reference,
        # and several modules import that way to keep torch off the import path.
        for node in ast.walk(self.tree):
            if isinstance(node, ast.ImportFrom):
                module = self._absolute(node)
                if module:
                    for alias in node.names:
                        self.aliases[alias.asname or alias.name] = (module, alias.name)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    local = alias.asname or alias.name.split(".")[0]
                    self.aliases[local] = (alias.name, MODULE_NODE)


Node = tuple[str, str]


class ReferenceGraph:
    """Production references between (module, symbol) nodes in two packages."""

    def __init__(self, modules: dict[str, Module]) -> None:
        self.modules = modules
        self.edges: dict[Node, set[Node]] = {}
        for module in modules.values():
            self._add_module_edges(module)

    def _resolve(self, module: Module, name: str) -> Node | None:
        if name in module.aliases:
            target, symbol = module.aliases[name]
            if target not in self.modules:
                return None
            if symbol == MODULE_NODE:
                return (target, MODULE_NODE)
            imported = self.modules[target]
            if symbol in imported.definitions:
                return (target, symbol)
            # A name re-exported by a package __init__ resolves to wherever
            # that __init__ imported it from, not to the __init__ itself.
            if symbol in imported.aliases:
                onward, onward_symbol = imported.aliases[symbol]
                if onward in self.modules:
                    return (onward, onward_symbol)
            submodule = f"{target}.{symbol}"
            if submodule in self.modules:
                return (submodule, MODULE_NODE)
            return None
        if name in module.definitions:
            return (module.name, name)
        return None

    def _references(self, node: ast.AST) -> set[str]:
        names: set[str] = set()
        for child in ast.walk(node):
            if isinstance(child, ast.Name):
                names.add(child.id)
            elif isinstance(child, ast.Attribute):
                names.add(child.attr)
        return names

    def _add_module_edges(self, module: Module) -> None:
        importing = self.edges.setdefault((module.name, MODULE_NODE), set())
        for local in module.aliases:
            target = self._resolve(module, local)
            if target is not None:
                importing.add(target)
        for symbol, definition in module.definitions.items():
            source = (module.name, symbol)
            reached = self.edges.setdefault(source, set())
            # Reaching a symbol reaches its module: importing it runs the
            # module body, which is what a module-level check asks about.
            reached.add((module.name, MODULE_NODE))
            for name in self._references(definition):
                target = self._resolve(module, name)
                if target is not None and target != source:
                    reached.add(target)

    def reachable(self, root: Node) -> set[Node]:
        seen: set[Node] = set()
        pending = [root]
        while pending:
            current = pending.pop()
            if current in seen:
                continue
            seen.add(current)
            pending.extend(self.edges.get(current, ()))
        return seen


def load_modules(repository: pathlib.Path) -> dict[str, Module]:
    """Parse the whole installed library, not just the packages being checked.

    Scoping the *population* is the point of this gate; scoping the *graph* is
    a bug, and it produced a false positive on the first run.
    ``mageflow_phases`` is reached from ``handlers.py`` only by going out to
    ``rwkv_lab.mage_flow_expert_train`` and back in through a deferred import,
    so a graph over the two worker packages alone reported it unreached when
    the worker process reaches it on every MageFlow run.
    """
    root = repository / PACKAGE_ROOT
    library = root / "rwkv_lab"
    if not library.is_dir():
        raise GateError(f"no rwkv_lab package under {PACKAGE_ROOT}/")
    modules: dict[str, Module] = {}
    for path in sorted(library.rglob("*.py")):
        name = module_name(path, root)
        try:
            modules[name] = Module(name, path)
        except SyntaxError as error:
            raise GateError(f"{name} does not parse: {error}") from error
    for package in (CONTRACT_PACKAGE, WORKER_PACKAGE):
        if package not in modules:
            raise GateError(f"{package} is not a package under {PACKAGE_ROOT}/")
    return modules


def literal_assignment(module: Module, name: str) -> ast.AST:
    for node in module.tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name
            for target in node.targets
        ):
            return node.value
        # `_HANDLERS: Mapping[AdapterKey, Handler] = {...}` is an annotated
        # assignment, which is a different node type carrying the same literal.
        if (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == name
            and node.value is not None
        ):
            return node.value
    raise GateError(f"{module.name} defines no {name}")


class _HandlersAsNames(ast.NodeTransformer):
    """Rewrite handler references to their own names, leaving the rest alone.

    Every ``Name`` load in the ``_HANDLERS`` literal is either a handler
    function or one of the two names the comprehension binds. Turning the first
    kind into a string constant makes the whole assignment evaluable with no
    imports, so Python's own evaluator expands the ``**`` comprehension the AST
    reader in ci_step_zero_arming_gate.py could not.
    """

    def __init__(self, bound: frozenset[str]) -> None:
        self.bound = bound

    def visit_Name(self, node: ast.Name) -> ast.AST:  # noqa: N802
        if isinstance(node.ctx, ast.Load) and node.id not in self.bound:
            return ast.copy_location(ast.Constant(node.id), node)
        return node


def dispatch_table(modules: dict[str, Module]) -> dict[tuple[str, ...], str]:
    """Return the adapter key -> handler function name mapping in _HANDLERS."""
    handlers = modules[HANDLERS]
    profiles = literal_assignment(modules[TRANSFORMER_MLA], "PROFILE_ADAPTERS")
    namespace = {"PROFILE_ADAPTERS": ast.literal_eval(profiles)}
    literal = literal_assignment(handlers, "_HANDLERS")
    rewritten = _HandlersAsNames(frozenset(namespace) | {"profile", "adapter"}).visit(
        ast.parse(ast.unparse(literal), mode="eval")
    )
    ast.fix_missing_locations(rewritten)
    try:
        table = eval(compile(rewritten, "<_HANDLERS>", "eval"), namespace)  # noqa: S307
    except Exception as error:  # pragma: no cover - refuses rather than guesses
        raise GateError(f"_HANDLERS could not be evaluated: {error}") from error
    if not isinstance(table, dict) or not table:
        raise GateError("_HANDLERS did not evaluate to a non-empty mapping")
    # The shape is asserted rather than assumed. A key that is not a 4-tuple of
    # strings would still index without error -- `key[0]` and `key[3]` of a
    # string are characters -- and the gate would go on to compare nonsense
    # against the pin. Refusing here says which half is wrong.
    for key, handler in table.items():
        if (
            not isinstance(key, tuple)
            or len(key) != 4
            or not all(isinstance(part, str) for part in key)
            or not isinstance(handler, str)
        ):
            raise GateError(f"_HANDLERS entry {key!r} is not an adapter key")
    return dict(table)


def advertised(repository: pathlib.Path) -> set[tuple[str, str]]:
    """Return (adapter, contract) for every route the native registry names."""
    path = repository / PIN
    if not path.is_file():
        raise GateError(f"missing {PIN}")
    document = json.loads(path.read_text(encoding="utf-8"))
    profiles = document.get("profiles")
    if not isinstance(profiles, list) or not profiles:
        raise GateError(f"{PIN} declares no profiles")
    return {(entry["adapter"], entry["contract"]) for entry in profiles}


def entrypoint_root(repository: pathlib.Path) -> Node:
    """The worker's production entrypoint, read from the console script."""
    path = repository / "pyproject.toml"
    if not path.is_file():
        raise GateError("missing pyproject.toml")
    scripts = tomllib.loads(path.read_text(encoding="utf-8")).get("project", {}).get(
        "scripts", {}
    )
    target = scripts.get(CONSOLE_SCRIPT)
    if not isinstance(target, str) or ":" not in target:
        raise GateError(
            f"pyproject.toml declares no {CONSOLE_SCRIPT} console script, so "
            "the worker's production entrypoint is unknown"
        )
    module, _, function = target.partition(":")
    return (module, function)


def analyse(repository: pathlib.Path) -> tuple[list[str], str]:
    """Return the problems found and the population summary to print."""
    modules = load_modules(repository)
    graph = ReferenceGraph(modules)
    root = entrypoint_root(repository)
    if root[0] not in modules or root[1] not in modules[root[0]].definitions:
        raise GateError(
            f"the {CONSOLE_SCRIPT} console script names {root[0]}:{root[1]}, "
            "which is not a definition in this tree"
        )
    reachable = graph.reachable(root)

    table = dispatch_table(modules)
    routes = {(key[0], key[3]) for key in table}
    registry = advertised(repository)

    problems: list[str] = []
    for adapter, contract in sorted(registry - routes):
        problems.append(
            f"UNSERVED: the native registry advertises {contract} "
            f"({adapter}) and _HANDLERS dispatches no handler for it"
        )
    for adapter, contract in sorted(routes - registry):
        problems.append(
            f"UNADVERTISED: _HANDLERS dispatches {contract} ({adapter}), which "
            f"{PIN} does not name; the gate cannot scope itself to a contract "
            "surface it disagrees with"
        )

    handlers = modules[HANDLERS]
    unwired_handlers = []
    for key, handler in sorted(table.items()):
        if handler not in handlers.definitions:
            problems.append(
                f"MISSING HANDLER: {key[3]} dispatches to {handler}, which "
                f"{HANDLERS} does not define"
            )
        elif (HANDLERS, handler) not in reachable:
            unwired_handlers.append((key[3], handler))
    for contract, handler in unwired_handlers:
        problems.append(
            f"UNWIRED HANDLER: {contract} dispatches to {handler}, which "
            f"nothing reaches from {root[0]}:{root[1]}; the registry "
            "advertises a route the worker process cannot run"
        )

    implementation = sorted(
        name
        for name, module in modules.items()
        if name.startswith(f"{CONTRACT_PACKAGE}.")
        and module.path.name != "__init__.py"
    )
    unwired_modules = [
        name for name in implementation if (name, MODULE_NODE) not in reachable
    ]
    for name in unwired_modules:
        problems.append(
            f"UNWIRED MODULE: {name} implements part of the advertised contract "
            f"surface and is reached by nothing from {root[0]}:{root[1]}"
        )

    summary = population_summary(
        registry, table, len(unwired_handlers), implementation, unwired_modules, root
    )
    return problems, summary


def population_summary(
    registry: set[tuple[str, str]],
    table: dict[tuple[str, ...], str],
    unwired_handlers: int,
    implementation: list[str],
    unwired_modules: list[str],
    root: Node,
) -> str:
    """Say which population each number counts, in the line that gets quoted.

    Four counts over three populations, because they answer different
    questions and a single "N unwired" would hide which one moved:

    - routes the native registry advertises, and how many of them dispatch to a
      handler nothing reaches. This is the number the gate exists for, and it
      is zero on a passing run -- which is exactly when one number under two
      names is indistinguishable from the truth;
    - distinct handler functions those routes share (22 routes, 15 handlers:
      eight Transformer-MLA profiles are one handler), so a reader is not
      surprised that the two counts differ;
    - implementation modules in the contract package, and how many are
      unreached.
    """
    routes = len(registry)
    handlers = len(set(table.values()))
    modules = len(implementation)
    return (
        f"{routes} advertised {'route' if routes == 1 else 'routes'} over "
        f"{handlers} distinct {'handler' if handlers == 1 else 'handlers'}, "
        f"rooted at {root[0]}:{root[1]}; {unwired_handlers} unwired, "
        f"{len(unwired_modules)} of {modules} implementation modules unreached"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repository",
        default=str(pathlib.Path(__file__).resolve().parent.parent),
        help="repository root to analyse",
    )
    arguments = parser.parse_args()
    repository = pathlib.Path(arguments.repository).resolve()

    try:
        problems, summary = analyse(repository)
    except GateError as error:
        print(f"FAIL: {error}")
        print(verdict_line("contract caller gate", [str(error)], "0 routes analysed"))
        return 1

    for problem in problems:
        print(problem)
    print(verdict_line("contract caller gate", problems, summary))
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
