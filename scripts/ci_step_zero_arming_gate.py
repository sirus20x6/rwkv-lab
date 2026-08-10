#!/usr/bin/env python3
"""Fail when the step-zero arming table stops describing the repository.

`docs/experiment-vm/STEP_ZERO_ARMING.md` answers one question for every
registered adapter contract: *can this route arm the universal step-zero eval
gate today, and if not, what is missing?*

That question has already been paid for twice. Two agents in a row were
dispatched to arm the gate and discovered, mid-implementation, that their route
structurally could not: one route's family had no way to publish a valid
step-zero artifact, and another declared no checkpoint output at all while
eval-examples is checkpoint-bound by construction. Both refusals were correct.
Both were expensive, because the blocking fact is cheap to determine up front
and nothing wrote it down.

A table that is true today and unchecked tomorrow is a worse instrument than no
table, because the next agent believes it. This repository already has that
defect on record: a prose count that drifted from 145 to 155 with nothing
holding it, in a sentence whose other number was right because *that* one was a
compiled array length. So the table is generated and compared, never
hand-maintained.

What arms the gate, exactly
---------------------------
`trainvm/src/service.cpp` sets `step_zero_eval_gate_required` from
`invocation_requires_step_zero_eval_gate(invocation->publishes)`, and that
predicate (`trainvm/src/eval_examples_contract.cpp`) is three conjuncts over
each publication's `declaration`:

    declaration->value("required", false) &&
    declaration->value("type", ...) == "eval_examples" &&
    declaration->value("schema", ...) == kEvalExamplesSchema

`publishes` is built in `trainvm/src/adapter_invocation.cpp` as, per entry of
the workflow node's `publishes` map, `spec.artifacts[logical_name]`. So the
declaration the predicate reads is an **artifact declaration in a composition
document**, reached through a node that publishes it.

Two things are therefore needed, and the table separates them because they fail
for different reasons and are fixed in different files:

1. The adapter contract must declare an `eval_examples` output port.
   `trainvm/src/adapter_registry.cpp` refuses a node that "publishes undeclared
   operation output", and `require_artifact_contract` forces the published
   artifact's type and schema to match the port. Without the port, no
   composition can publish eval-examples at all.
2. A shipped composition document must declare the artifact with all three
   conjuncts and publish it from the invoking node. Note the seam: the port's
   `required` and the artifact's `required` are *different fields*, and
   `require_artifact_contract` does not tie them. A required port forces the
   node to publish something; only the artifact's own `required: true` arms the
   controller. `run_authoring.cpp` says as much in a comment beside the HF
   family -- "a declared-but-optional publication leaves it inert, which is the
   exact state this family was in".

Arming is not publishing
------------------------
The three conjuncts arm the controller. They do not make the route able to
satisfy what it has just been armed for. `EvalExamplesPublisher.__init__`
(`src/rwkv_lab/trainvm_worker/eval_examples.py`) rejects the very same
declaration unless it *also* carries `immutability: append_only` and
`fingerprint: manifest_sha256` -- and `adapter_invocation.cpp` hands the worker
`encode_json(spec.artifacts[logical_name])`, so those two fields come from the
same artifact object the predicate reads. Both default to a value the publisher
refuses (`Immutability immutability{}` is `immutable`, `Fingerprint
fingerprint{}` is `sha256`, in `trainvm/include/trainvm/model.hpp`), so omission
is not a neutral state.

A document satisfying only the arming three therefore arms the controller and
then cannot publish: the route deadlocks at its first `pre_optimizer_step`
crossing, with no diagnostic, which is strictly worse than never arming --
an inert gate is merely absent, a deadlocked one stops the run. PR #204 paid
for this the expensive way on RLVR. So the table reports that population
separately: `no -- arms, then the publisher refuses` is neither "armed" nor the
same condition as "blocked" (no port, arms nothing), and the two are fixed in
different files.

The three conjuncts are C++ and the two publisher requirements are Python.
Neither is restated here. The C++ side is a `required`/`type`/`schema` triple
this gate already states twice and compares against the pin; the Python side is
**read out of `eval_examples.py` itself** by `publisher_requirements()`, which
recovers `{key: value}` from the rejection condition's own AST. What forces the
two halves to agree is that the extraction must recover `type` and `schema`
equal to this gate's constants -- so a parse that finds the wrong node, or a
publisher that moves off `rwkv-lab.eval-examples.v1`, reddens the gate instead
of quietly checking fewer conditions.

Two traps this gate is built to avoid
-------------------------------------
Both produce a confident wrong answer from a token grep for `eval_examples`.

- `output_name="eval_examples"` in a worker's publish call is a *runtime*
  publication, not a declaration. `hf_multimodal_sft.py` has one. The gate reads
  authoring, so a worker's call says nothing about arming.
- `eval_examples: int = 64` in the vision trainer configs is a *count* -- how
  many examples to evaluate. Three vision routes declare it. A grep-derived
  table reports all three as arming and is wrong.

Neither trap can reach this gate, because nothing here reads Python trainer
source. The registry half comes from a pin regenerated from the built registry
(`scripts/print_step_zero_arming_pin.py`) and the composition half comes from
parsed JSON documents.

Two sides, from two places
--------------------------
The table in the Markdown document is read as text. The truth is computed from
the pin and the composition documents. A gate that recomputed both sides from
the same place would always agree with itself and measure nothing.

Why here and not in the native suite
------------------------------------
The pull request that changes an arming answer is usually a JSON edit to a
recipe-profile catalog, which is the one least likely to be waiting on an
~8 minute GCC 16 build the PR-tier classifier can skip. The registry half is
pinned precisely so this can run with no compiler, and ctest `step_zero_arming_pin`
checks the pin against a real build where one exists.

Usage:
    python scripts/ci_step_zero_arming_gate.py [--repository .] [--write]
"""

from __future__ import annotations

import argparse
import ast
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from scripts.gate_verdict import verdict_line  # noqa: E402

DOCUMENT = "docs/experiment-vm/STEP_ZERO_ARMING.md"
PIN = "docs/experiment-vm/step-zero-arming.v1.json"
EXAMPLES = "docs/experiment-vm/examples"
PUBLISHER = "src/rwkv_lab/trainvm_worker/eval_examples.py"
PUBLISHER_CLASS = "EvalExamplesPublisher"

BEGIN = "<!-- BEGIN GENERATED ARMING TABLE -->"
END = "<!-- END GENERATED ARMING TABLE -->"

HEADERS = (
    "route",
    "stateful",
    "evaluator slot?",
    "checkpoint output?",
    "eval_examples output port (required+type+schema)?",
    "can arm today?",
    "what is missing",
)

# `kEvalExamplesSchema` in trainvm/include/trainvm/eval_examples_contract.hpp.
# The pin carries the value the built registry used; this is the value the
# *predicate* compares against, and they are only the same string while nobody
# has changed one of them, which is why both are stated and compared.
EVAL_EXAMPLES_SCHEMA = "rwkv-lab.eval-examples.v1"
EVAL_EXAMPLES_TYPE = "eval_examples"

# The "what is missing" cell for a route some of whose compositions arm and
# some of which do not. Shared rather than repeated because the summary counts
# that population and a second copy of the prefix would drift out of agreement
# with the cell it is meant to detect.
PARTIALLY_ARMED = "armed, but not everywhere: "

# The "can arm today?" cell, and the "what is missing" prefix, for a route whose
# only arming compositions satisfy the predicate and not the publisher. Both are
# constants because the summary counts this population off the cell, exactly as
# it does for PARTIALLY_ARMED, and a second copy of either string would drift
# out of agreement with the cell it is meant to detect.
DEADLOCK_CELL = "no — arms, then the publisher refuses"
DEADLOCKS = "arms the controller, then the publisher refuses the same declaration: "


def publisher_requirements(repository: pathlib.Path) -> tuple[dict[str, str], list[str]]:
    """What `EvalExamplesPublisher.__init__` demands of the declaration.

    Read out of `eval_examples.py` rather than restated. The publisher rejects
    with a single `if` whose disjuncts are `declaration.get(<key>) != <value>`,
    and this recovers that `{key: value}` map from the source's own AST. A third
    statement of `append_only`/`manifest_sha256` here would be one more fact in
    two places with nothing forcing agreement -- the defect this gate was filed
    for, one level up.

    Importing the module instead was rejected: it imports `trainvm.v1.trainvm_pb2`
    at module scope and raises a `RuntimeError` naming the `trainvm-worker` extra
    when that is absent, which the seconds-fast schema job does not install. A
    static read needs nothing.

    The extraction is checked rather than trusted. `rows()` requires the returned
    map to agree with this gate's own `type`/`schema` constants, which the
    publisher states too -- so a parse that matched the wrong `if`, or returned
    nothing, cannot present itself as "no further conditions".
    """
    path = repository / PUBLISHER
    try:
        source = path.read_text(encoding="utf-8")
    except OSError as error:
        return {}, [f"{PUBLISHER}: cannot be read ({error})"]
    try:
        tree = ast.parse(source)
    except SyntaxError as error:
        return {}, [f"{PUBLISHER}: is not parseable Python ({error})"]

    classes = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.ClassDef) and node.name == PUBLISHER_CLASS
    ]
    if len(classes) != 1:
        return {}, [
            (
                f"{PUBLISHER}: declares {len(classes)} classes named "
                f"{PUBLISHER_CLASS!r}, this gate reads exactly one"
            )
        ]

    found: dict[str, str] = {}
    for node in ast.walk(classes[0]):
        if not isinstance(node, ast.Compare) or len(node.ops) != 1:
            continue
        if not isinstance(node.ops[0], ast.NotEq):
            continue
        call = node.left
        if (
            not isinstance(call, ast.Call)
            or not isinstance(call.func, ast.Attribute)
            or call.func.attr != "get"
            or not isinstance(call.func.value, ast.Name)
            or call.func.value.id != "declaration"
            or len(call.args) != 1
        ):
            continue
        key, value = call.args[0], node.comparators[0]
        if not isinstance(key, ast.Constant) or not isinstance(key.value, str):
            continue
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            found[key.value] = value.value
        elif isinstance(value, ast.Name):
            # `declaration.get("schema") != EVAL_EXAMPLES_SCHEMA` -- resolve the
            # module constant, so renaming the literal into a name does not
            # silently drop a condition.
            for assignment in tree.body:
                if (
                    isinstance(assignment, ast.Assign)
                    and any(
                        isinstance(target, ast.Name) and target.id == value.id
                        for target in assignment.targets
                    )
                    and isinstance(assignment.value, ast.Constant)
                    and isinstance(assignment.value.value, str)
                ):
                    found[key.value] = assignment.value.value
    if not found:
        return {}, [
            (
                f"{PUBLISHER}: no `declaration.get(...) != ...` comparison was "
                f"found in {PUBLISHER_CLASS}, so this gate cannot tell what the "
                "publisher demands"
            )
        ]
    return found, []


def dictionaries(value: object):
    """Every dict anywhere in a parsed document.

    Recipe catalogs nest a whole experiment document under
    `recipes[].template_document`, so a top-level-only reader finds none of
    them -- and those are exactly the documents that arm today.
    """
    if isinstance(value, dict):
        yield value
        for nested in value.values():
            yield from dictionaries(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from dictionaries(nested)


def is_spec(value: object) -> bool:
    return (
        isinstance(value, dict)
        and isinstance(value.get("components"), dict)
        and isinstance(value.get("workflow"), dict)
        and isinstance(value.get("artifacts"), dict)
    )


def specs(document: object):
    """Every experiment spec in a document, with an identity that survives.

    Identity is the enclosing document's `metadata.name`, not the file, because
    a recipe catalog holds several template documents against the same adapter
    contract. Keyed by file alone, one recipe losing its arming declaration is
    invisible whenever a sibling recipe in the same file still arms -- which is
    exactly the drift this gate exists to catch, and it is the shape that made
    an earlier draft of this check pass all three single-conjunct mutations.
    """
    named: dict[int, str] = {}
    for node in dictionaries(document):
        if is_spec(node.get("spec")):
            metadata = node.get("metadata")
            name = metadata.get("name") if isinstance(metadata, dict) else None
            named[id(node["spec"])] = str(name) if name else ""
    ordinal = 0
    for node in dictionaries(document):
        if not is_spec(node):
            continue
        ordinal += 1
        # A bare spec with no enclosing document still yields, so a
        # differently-shaped document cannot silently contribute nothing.
        yield (named.get(id(node)) or f"#{ordinal}"), node


def contract_of(spec: dict, invoke: dict) -> str | None:
    component = spec["components"].get(invoke.get("component"))
    if not isinstance(component, dict):
        return None
    operations = component.get("operations")
    if not isinstance(operations, dict):
        return None
    operation = operations.get(invoke.get("operation"))
    if not isinstance(operation, dict):
        return None
    contract = operation.get("contract")
    return contract if isinstance(contract, str) else None


def publication_verdict(
    artifact: object, requirements: dict[str, str]
) -> tuple[str, str]:
    """Evaluate all five conditions, and say which of them failed.

    Returns one of three states, because the two ways of failing are different
    conditions with different fixes and collapsing them is how the previous
    version of this gate misled a reader:

    - ``armed``   -- predicate and publisher both satisfied;
    - ``deadlock`` -- predicate satisfied, publisher will refuse it;
    - ``inert``   -- the predicate is not satisfied, so nothing is armed.

    Reporting the failing condition is not decoration. A check that tests one
    conjunct while asserting five passes every mutation of the other four, and
    the only way to tell from the outside is whether it can name them.
    """
    if not isinstance(artifact, dict):
        return "inert", "the published logical artifact is not declared"
    reasons: list[str] = []
    if artifact.get("type") != EVAL_EXAMPLES_TYPE:
        reasons.append(f"type is {artifact.get('type')!r}, not {EVAL_EXAMPLES_TYPE!r}")
    if artifact.get("schema") != EVAL_EXAMPLES_SCHEMA:
        reasons.append(
            f"schema is {artifact.get('schema')!r}, not {EVAL_EXAMPLES_SCHEMA!r}"
        )
    if artifact.get("required", False) is not True:
        reasons.append(
            f"required is {artifact.get('required', False)!r}, so the publication "
            "is declared but inert"
        )
    if reasons:
        return "inert", "; ".join(reasons)
    # The predicate holds. Whether the worker can then publish is a separate
    # question over the same object, and `type`/`schema` are deliberately
    # re-evaluated from the publisher's own map: if it has moved off the value
    # the predicate accepts, `rows()` has already reddened, and skipping them
    # here would leave the table describing a route that arms and cannot publish.
    for key, value in sorted(requirements.items()):
        if artifact.get(key) != value:
            reasons.append(
                f"{key} is {artifact.get(key)!r}, and the publisher "
                f"({PUBLISHER}) refuses anything but {value!r}"
            )
    if reasons:
        return "deadlock", "; ".join(reasons)
    return "armed", ""


def composition_evidence(
    repository: pathlib.Path, requirements: dict[str, str]
) -> tuple[dict, dict, dict, list[str]]:
    """Per contract: which shipped documents arm it, and why the others do not.

    Every JSON document under `docs/experiment-vm/examples/` is read, recursively.
    A malformed one is a problem rather than a skip: a document that stops
    parsing silently removes whatever it used to prove.
    """
    arming: dict[str, set[str]] = {}
    deadlocking: dict[str, list[str]] = {}
    rejected: dict[str, list[str]] = {}
    problems: list[str] = []
    root = repository / EXAMPLES
    for path in sorted(root.rglob("*.json")):
        label = str(path.relative_to(repository / "docs/experiment-vm"))
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            problems.append(f"{label}: is not parseable JSON ({error})")
            continue
        for identity, spec in specs(document):
            composition = f"{label}[{identity}]"
            nodes = spec["workflow"].get("nodes")
            if not isinstance(nodes, dict):
                continue
            for node_name, node in nodes.items():
                if not isinstance(node, dict):
                    continue
                invoke = node.get("invoke")
                if not isinstance(invoke, dict):
                    continue
                contract = contract_of(spec, invoke)
                if contract is None:
                    continue
                publishes = node.get("publishes")
                if not isinstance(publishes, dict):
                    continue
                for output, logical in publishes.items():
                    artifact = spec["artifacts"].get(logical)
                    named = output == EVAL_EXAMPLES_TYPE or (
                        isinstance(artifact, dict)
                        and artifact.get("type") == EVAL_EXAMPLES_TYPE
                    )
                    if not named:
                        continue
                    state, reason = publication_verdict(artifact, requirements)
                    detail = f"{composition} node {node_name}: {reason}"
                    if state == "armed":
                        arming.setdefault(contract, set()).add(composition)
                    elif state == "deadlock":
                        deadlocking.setdefault(contract, []).append(detail)
                    else:
                        rejected.setdefault(contract, []).append(detail)
    return arming, deadlocking, rejected, problems


def missing(
    entry: dict, armed: set[str], deadlocking: list[str], rejected: list[str]
) -> str:
    shortfall = sorted(deadlocking) + sorted(rejected)
    if armed:
        if shortfall:
            # Some compositions arm and others do not. Saying only "armed"
            # would let a sibling recipe lose its declaration behind a
            # neighbour that kept it.
            return PARTIALLY_ARMED + "; ".join(shortfall)
        return "nothing — armed"
    if deadlocking:
        # Its own cell text, before the stateful check and before the port
        # reasons: the port is present and a shipped composition does publish
        # through it, so every sentence below would be false here. This is the
        # harmful state -- the controller demands evidence the worker is
        # structurally unable to supply -- and it must not read as "blocked".
        return DEADLOCKS + "; ".join(sorted(deadlocking))
    if not entry.get("stateful"):
        # A non-stateful operation mutates nothing, so there is no first
        # optimizer step for the gate to precede. Saying "it declares no
        # eval_examples port" would be true and misleading: the port is not
        # what is missing, the mutation is.
        return "nothing — not stateful, so no optimizer mutation to gate"
    port = entry.get("eval_examples_output")
    parts: list[str] = []
    if port is None:
        parts.append(
            "the adapter declares no eval_examples output port, so no "
            "composition may publish one (adapter_registry.cpp rejects an "
            "undeclared operation output)"
        )
    else:
        if not port.get("required"):
            parts.append("the eval_examples output port is optional")
        if port.get("artifact_schema") != EVAL_EXAMPLES_SCHEMA:
            parts.append(
                f"the port pins schema {port.get('artifact_schema')!r}, which the "
                "predicate does not accept"
            )
    if not entry.get("checkpoint_outputs"):
        parts.append(
            "and no checkpoint output, while eval-examples is checkpoint-bound"
        )
    if rejected:
        parts.append("shipped compositions fall short: " + "; ".join(sorted(rejected)))
    elif port is not None:
        parts.append("no shipped composition document publishes it")
    return "; ".join(parts) if parts else "no shipped composition document publishes it"


def rows(repository: pathlib.Path) -> tuple[list[tuple[str, ...]], list[str]]:
    pin = json.loads((repository / PIN).read_text(encoding="utf-8"))
    problems: list[str] = []
    if pin.get("eval_examples_schema") != EVAL_EXAMPLES_SCHEMA:
        problems.append(
            f"{PIN}: pins eval_examples_schema "
            f"{pin.get('eval_examples_schema')!r}, this gate evaluates the "
            f"predicate against {EVAL_EXAMPLES_SCHEMA!r}"
        )
    requirements, publisher_problems = publisher_requirements(repository)
    problems += publisher_problems
    # What forces the two authorities to agree. The publisher states `type` and
    # `schema` as well as the two conditions only it states, so a disagreement
    # with the constants this gate evaluates the predicate against means one of
    # them moved -- and every arming answer below was computed against a
    # declaration the other side would refuse. It also validates the extraction
    # itself: an AST read that matched the wrong node returns a map missing
    # these, and would otherwise present as "the publisher demands nothing".
    for key, expected in (
        ("type", EVAL_EXAMPLES_TYPE),
        ("schema", EVAL_EXAMPLES_SCHEMA),
    ):
        if not publisher_problems and requirements.get(key) != expected:
            problems.append(
                f"{PUBLISHER}: {PUBLISHER_CLASS} demands {key} "
                f"{requirements.get(key)!r}, this gate evaluates the predicate "
                f"against {expected!r}"
            )
    arming, deadlocking, rejected, document_problems = composition_evidence(
        repository, requirements
    )
    problems += document_problems

    built: list[tuple[str, ...]] = []
    for entry in sorted(pin.get("profiles", []), key=lambda item: item["contract"]):
        contract = entry["contract"]
        armed = arming.get(contract, set())
        port = entry.get("eval_examples_output")
        port_cell = "no"
        if port is not None:
            port_cell = (
                "yes"
                if port.get("required")
                and port.get("artifact_schema") == EVAL_EXAMPLES_SCHEMA
                else "declared, not required"
            )
        built.append(
            (
                f"`{contract}`",
                "yes" if entry.get("stateful") else "no",
                "yes" if entry.get("evaluator_slots") else "no",
                "yes" if entry.get("checkpoint_outputs") else "no",
                port_cell,
                ("yes — " + ", ".join(sorted(armed)))
                if armed
                else (DEADLOCK_CELL if deadlocking.get(contract) else "no"),
                missing(
                    entry,
                    armed,
                    deadlocking.get(contract, []),
                    rejected.get(contract, []),
                ),
            )
        )
    return built, problems


def registry_parity_note() -> str:
    """Why this gate does not compare the pin against the worker's dispatch table.

    It would be the obvious check: the pin is regenerated from the native
    registry, `verify_rwkv_lab_worker_contract.py` asserts that registry is
    key-for-key `handlers._HANDLERS`, so comparing the pin against `_HANDLERS`
    would catch a route added to the authority and forgotten in the pin.

    It cannot run here. `rwkv_lab.trainvm_adapters.handlers` imports torch
    transitively and the schema job installs `.[test]` without it; adding torch
    to a seconds-fast job to read a couple of dozen strings is the wrong trade.
    Reading the literal statically does not work either -- `_HANDLERS` ends in a
    `**` unpacking of a dict comprehension over `PROFILE_ADAPTERS`, so the
    Transformer MLA family does not exist as string literals at all, and an AST
    reader silently reports only the explicit keys.

    No counts here on purpose. This paragraph used to name three, and by the
    time anyone checked, two of them were wrong and nothing had noticed. A count
    restated in prose beside the thing that computes it is a second copy that
    only ever drifts -- the same defect this file's own verdict line was fixed
    for. Restating the corrected numbers here would just restart the clock, so
    the argument is left in the form that stays true: the static read is
    systematically short by however large that family currently is, the gate's
    verdict line prints the live figure, and the two tests named below are what
    actually hold the parity.

    So the check lives in the two jobs that can answer it, and neither is
    optional on a pull request:

    - `tests/test_step_zero_arming_gate.py::test_the_pin_names_exactly_the_dispatched_routes`
      imports the registry in the CPU job, which installs torch and runs on
      every pull request;
    - ctest `step_zero_arming_pin` compares the pin field by field against a
      real build, and adding a route means editing
      `trainvm/src/rwkv_lab_worker_contract.cpp`, which is exactly what makes
      `native-change-scope` schedule the native job.

    This function exists so that reasoning is attached to the gate rather than
    lost in a commit message, and returns the sentence the verdict prints.
    """
    return (
        "route-set parity is checked by pytest and ctest, not here (see "
        "registry_parity_note)"
    )


def population_summary(built: list[tuple[str, ...]]) -> str:
    """Say which population each number counts, in the line that gets quoted.

    This sentence used to read `{armed} able to arm today`, over a variable that
    counts routes a shipped composition **already arms**. Those are different
    claims: "able to arm today" is a capability that has not been exercised, so
    the number reads as a backlog. It was read that way, and "0 routes are
    armed, 2 possible today" travelled from here into three cards and an agent
    dispatch sent to arm a route that had been armed end to end for some time.
    The investigation to discover that was the whole cost.

    The generated table was never wrong -- its cell for both routes says
    `nothing — armed`, and `missing()` reasons carefully about the partial case.
    But a verdict line is read far more often than the document it summarises,
    so an ambiguity here propagates in a way the same ambiguity in a table cell
    does not.

    The two populations are genuinely distinct and both are worth printing:
    arming needs an adapter port **and** a composition that publishes through
    it, and the port half can be in place with nothing publishing it. Today that
    second population is empty, which is exactly when a single number under two
    names is indistinguishable from either -- and exactly why it is printed.

    The five counts partition the routes, in this order, so they always sum to
    the total and a reader can check that they do:

    - armed: some shipped composition satisfies all five conditions;
    - would deadlock: a shipped composition satisfies the three predicate
      conjuncts and none satisfies the publisher, so the route arms and then
      cannot publish. Printed even at zero, for the same reason the armable
      count is: a population that is absent today is indistinguishable from one
      that is not counted, and this one is the harmful direction;
    - armable but unarmed: the adapter's port is usable, no composition arms it;
    - blocked: stateful, and the port is missing or unusable;
    - not stateful: nothing mutates, so there is no first optimizer step to gate.
    """
    total = len(built)
    stateful = sum(1 for row in built if row[1] == "yes")
    armed = sum(1 for row in built if row[5].startswith("yes"))
    # Counted off the cell, like `partial` below, so the line cannot claim a
    # different deadlock set than the table shows.
    deadlock = sum(1 for row in built if row[5] == DEADLOCK_CELL)
    # Counted off the cell `missing()` writes, so the summary cannot claim a
    # different partial set than the table shows.
    partial = sum(1 for row in built if row[6].startswith(PARTIALLY_ARMED))
    armable = sum(1 for row in built if row[5] == "no" and row[4] == "yes")
    blocked = sum(
        1 for row in built if row[5] == "no" and row[4] != "yes" and row[1] == "yes"
    )
    inert = total - armed - deadlock - armable - blocked

    armed_clause = f"{armed} armed today by a shipped composition"
    if partial:
        # Collapsing this would let a sibling recipe lose its declaration behind
        # a neighbour that kept it, in the one line most likely to be read alone.
        armed_clause += f", {partial} of those armed in some compositions and not others"
    return (
        f"{total} registered {'route' if total == 1 else 'routes'} "
        f"({stateful} stateful); {armed_clause}, {deadlock} would deadlock "
        f"(arms the controller, the publisher refuses the declaration), "
        f"{armable} armable but unarmed "
        f"(port in place, no shipped composition arms it), {blocked} blocked, "
        f"{inert} not stateful"
    )


def render(built: list[tuple[str, ...]]) -> str:
    lines = [
        "| " + " | ".join(HEADERS) + " |",
        "| " + " | ".join("---" for _ in HEADERS) + " |",
    ]
    lines += ["| " + " | ".join(cell for cell in row) + " |" for row in built]
    return "\n".join(lines)


def parse_table(document: str) -> tuple[list[tuple[str, ...]], list[str]]:
    if BEGIN not in document or END not in document:
        return [], [f"{DOCUMENT}: the generated table markers are missing"]
    body = document.split(BEGIN, 1)[1].split(END, 1)[0].strip()
    parsed: list[tuple[str, ...]] = []
    problems: list[str] = []
    for index, line in enumerate(body.splitlines()):
        line = line.strip()
        if not line.startswith("|"):
            continue
        cells = tuple(cell.strip() for cell in line.strip("|").split("|"))
        if index == 0:
            if cells != HEADERS:
                problems.append(
                    f"{DOCUMENT}: the table header is {list(cells)}, the gate "
                    f"emits {list(HEADERS)}"
                )
            continue
        if all(set(cell) <= {"-", ":"} for cell in cells):
            continue
        parsed.append(cells)
    return parsed, problems


def compare(
    written: list[tuple[str, ...]], built: list[tuple[str, ...]]
) -> list[str]:
    problems: list[str] = []
    by_route = {row[0]: row for row in written}
    truth = {row[0]: row for row in built}
    for route in sorted(set(truth) - set(by_route)):
        problems.append(
            f"{DOCUMENT}: states no row for {route}, which the registry declares"
        )
    for route in sorted(set(by_route) - set(truth)):
        problems.append(
            f"{DOCUMENT}: states a row for {route}, which the registry does not "
            "declare"
        )
    for route in sorted(set(by_route) & set(truth)):
        for column, header in enumerate(HEADERS):
            if column == 0:
                continue
            stated = by_route[route][column] if column < len(by_route[route]) else ""
            correct = truth[route][column]
            if stated != correct:
                problems.append(
                    f"{DOCUMENT}: {route} column {header!r} says {stated!r}, the "
                    f"repository holds {correct!r}"
                )
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repository",
        default=str(pathlib.Path(__file__).resolve().parent.parent),
        help="repository root to analyse",
    )
    parser.add_argument(
        "--write", action="store_true", help="splice the computed table into the document"
    )
    arguments = parser.parse_args()
    repository = pathlib.Path(arguments.repository).resolve()
    path = repository / DOCUMENT

    built, problems = rows(repository)
    document = path.read_text(encoding="utf-8")
    written, table_problems = parse_table(document)
    problems += table_problems
    if not table_problems:
        problems += compare(written, built)

    for problem in problems:
        print(f"FAIL: {problem}")

    if arguments.write and BEGIN in document and END in document:
        head, rest = document.split(BEGIN, 1)
        _, tail = rest.split(END, 1)
        path.write_text(
            f"{head}{BEGIN}\n\n{render(built)}\n\n{END}{tail}", encoding="utf-8"
        )
        print(f"WROTE: {len(built)} rows into {DOCUMENT}")
        # Still a failure when there was drift. --write is a fixer, not a way to
        # make CI green: the run that had to correct the document is the run
        # that should be red, and the next one passes.

    print(
        verdict_line(
            "step-zero arming gate",
            problems,
            f"{population_summary(built)}; checked against the pinned registry "
            f"and every composition document under {EXAMPLES}; "
            f"{registry_parity_note()}",
        )
    )
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
