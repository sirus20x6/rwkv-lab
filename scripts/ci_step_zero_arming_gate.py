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
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from scripts.gate_verdict import verdict_line  # noqa: E402

DOCUMENT = "docs/experiment-vm/STEP_ZERO_ARMING.md"
PIN = "docs/experiment-vm/step-zero-arming.v1.json"
EXAMPLES = "docs/experiment-vm/examples"

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


def publication_verdict(artifact: object) -> tuple[bool, str]:
    """Evaluate the three conjuncts, and say which one failed.

    Reporting the failing conjunct is not decoration. A check that tests one
    conjunct while asserting three passes every mutation of the other two, and
    the only way to tell from the outside is whether it can name them.
    """
    if not isinstance(artifact, dict):
        return False, "the published logical artifact is not declared"
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
    return (not reasons), "; ".join(reasons)


def composition_evidence(repository: pathlib.Path) -> tuple[dict, dict, list[str]]:
    """Per contract: which shipped documents arm it, and why the others do not.

    Every JSON document under `docs/experiment-vm/examples/` is read, recursively.
    A malformed one is a problem rather than a skip: a document that stops
    parsing silently removes whatever it used to prove.
    """
    arming: dict[str, set[str]] = {}
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
                    armed, reason = publication_verdict(artifact)
                    if armed:
                        arming.setdefault(contract, set()).add(composition)
                    else:
                        rejected.setdefault(contract, []).append(
                            f"{composition} node {node_name}: {reason}"
                        )
    return arming, rejected, problems


def missing(entry: dict, armed: set[str], rejected: list[str]) -> str:
    if armed:
        if rejected:
            # Some compositions arm and others do not. Saying only "armed"
            # would let a sibling recipe lose its declaration behind a
            # neighbour that kept it.
            return "armed, but not everywhere: " + "; ".join(sorted(rejected))
        return "nothing — armed"
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
    arming, rejected, document_problems = composition_evidence(repository)
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
                ("yes — " + ", ".join(sorted(armed))) if armed else "no",
                missing(entry, armed, rejected.get(contract, [])),
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
    to a seconds-fast job to read twenty-one strings is the wrong trade. Reading
    the literal statically does not work either -- `_HANDLERS` ends in a `**`
    unpacking of a dict comprehension over `PROFILE_ADAPTERS`, so eight of the
    twenty-one contracts do not exist as string literals at all, and an AST
    reader silently reports thirteen.

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

    armed = sum(1 for row in built if row[5] != "no")
    stateful = sum(1 for row in built if row[1] == "yes")
    print(
        verdict_line(
            "step-zero arming gate",
            problems,
            f"{len(built)} registered routes ({stateful} stateful), {armed} able "
            "to arm today, checked against the pinned registry and every "
            f"composition document under {EXAMPLES}; {registry_parity_note()}",
        )
    )
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
