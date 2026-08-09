#!/usr/bin/env python3
"""Validate the checked-in example experiment documents against the schema.

The C++ compiler is the authority on what a plan means, but it needs GCC 16 and
so cannot run on a hosted CI runner. This is the portable half: it proves the
example documents still satisfy experiment-v1.schema.json, which is the
contract the dashboard editor and every external producer code against.

It also fails when the coverage fixtures under examples/coverage/ regain a key
that would carry host or executable authority -- the same forbidden-key rule
the native coverage_fixture_tests enforces, kept runnable without a build.

## Why the recipe-profile catalogs are in scope

This gate used to glob `examples/*.json` non-recursively and validate whatever
carried a `spec`, which came to exactly three documents. It printed
`PASSED -- 3 experiment documents validated` and that read as "the examples are
clean". It was not a statement about the examples; it was a statement about
three of them.

`docs/experiment-vm/examples/hf-multimodal-sft.recipe-profiles.v1.json`
declares an artifact `{"type": "eval_examples", ...}`. `ArtifactType::eval_examples`
exists in the native model and `trainvm/src/eval_examples_contract.cpp` requires
that declaration verbatim -- but the schema's artifact `type` enum did not list
it until PR #157. Every experiment document materialised from that profile was
therefore refused by this gate, while this gate said PASSED, because the
document that exercised the disagreement was outside the set it looked at. A
recipe profile is a template *for* experiment documents; validating the
documents while skipping the templates they are minted from is a green light for
a question nobody asked.

So the scope is now stated over the tree rather than enumerated: every `*.json`
under `examples/`, recursively, is classified by its `api_version` and either
validated, unpacked and validated, or reported by name as unvalidated. A
hand-maintained list of catalogs would be the same defect one level up.

## Why contract resolution is here too

The schema constrains `contract` as a string. It has no way to know which
contract strings a registry declares, so an example could name a contract that
has never existed and stay green forever -- and two did:
`examples/mageflow-cache-resume.json` and `examples/rwkv-scratch-topologies.json`
both invoke a bare `mageflow` component whose operations name
`rwkv_lab.mageflow.v1.{Train,PrepareCacheSpan,CacheEncoders}`. The registry
declares the MageFlow family as `rwkv_lab.mageflow_{appearance_expert,
terminal_expert,full_backbone}.v1.Train`. An example that names a contract no
registry declares is a worked example that cannot work, and its failure mode is
the bad one: nothing reports it, so it sits in `examples/` looking
authoritative, and the first person to copy it as a template inherits a broken
id and debugs their own edit.

So every non-builtin invocation is resolved against the registry pin, and the
unresolvable ones are enumerated with a stated reason in
`unresolved-contract-exclusions.v1.json`.

Two things this deliberately does NOT do:

- It does not grep for contract ids in native source. `rwkv_lab_worker_contract.cpp`
  assembles ids from parts, so a literal search for `"rwkv_lab.mageflow..."`
  finds nothing there and that zero is not evidence of absence. The authority is
  the generated pin.
- It does not hand-maintain a list of valid ids. The pin is regenerated from a
  real build by `scripts/print_step_zero_arming_pin.py`; keeping it honest is
  ctest `step_zero_arming_pin` and
  `tests/test_step_zero_arming_gate.py::test_the_pin_names_exactly_the_dispatched_routes`,
  neither of which is optional on a pull request. This gate consumes it.
"""

from __future__ import annotations

import json
import pathlib
import sys

import jsonschema

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from scripts.gate_verdict import verdict_line  # noqa: E402

# The spec/invocation recognisers are shared with the arming gate rather than
# reimplemented. `specs()` keys each spec by its enclosing `metadata.name`, not
# by filename, which is what stops one recipe in a multi-recipe catalog masking
# a broken sibling in the same file.
from scripts.ci_step_zero_arming_gate import specs  # noqa: E402

DOCS = pathlib.Path("docs/experiment-vm")
SCHEMA = DOCS / "experiment-v1.schema.json"
EXAMPLES = DOCS / "examples"
# Regenerated from a real build; see the module docstring for what keeps it
# honest. `profiles[].contract` is the set of adapter contracts the shipped
# rwkv-lab worker registry declares.
REGISTRY_PIN = DOCS / "step-zero-arming.v1.json"
EXCLUSIONS = DOCS / "unresolved-contract-exclusions.v1.json"

# The two api_version constants this gate knows how to check. Anything else is
# reported by name in the summary rather than passed over in silence.
EXPERIMENT_API = "trainvm.rwkv-lab/v1alpha1"
RECIPE_PROFILES_API = "trainvm.recipe-profiles/v1"

# A catalog is recognised by BOTH its filename and its api_version, and the two
# must agree. Recognising it by api_version alone would let a catalog fall out
# of scope by editing one string; recognising it by filename alone would miss a
# catalog that is named differently. Disagreement is a failure, not a skip.
RECIPE_PROFILE_GLOB = "*.recipe-profiles.v1.json"

# Mirrors trainvm/tests/coverage_fixture_tests.cpp. An evidence fixture must
# never carry the authority to name an executable, argv, or environment.
FORBIDDEN_KEYS = {
    "arguments", "argv", "code_fingerprint", "code_path", "command",
    "environment", "env", "executable", "executable_fingerprint",
    "executable_path", "host", "host_launch", "host_launches",
    "host_launch_profile", "host_profile", "host_profile_digest",
    "host_registry_digest", "interpreter", "module", "public_arguments",
    "required_capabilities", "source_roots", "trusted_roots",
    "working_directory",
}
FORBIDDEN_PREFIXES = (
    "#!", "/bin/", "/sbin/", "/usr/bin/", "/usr/local/bin/",
    "python -m ", "python3 -m ", "sh -c ", "bash -c ",
)


def walk(value, path=""):
    """Yield (json_pointer, key, value) for every mapping entry."""
    if isinstance(value, dict):
        for key, child in value.items():
            yield f"{path}/{key}", key, child
            yield from walk(child, f"{path}/{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from walk(child, f"{path}/{index}")


def check_no_executable_authority(path: pathlib.Path) -> list[str]:
    document = json.loads(path.read_text())
    problems = []
    for pointer, key, value in walk(document):
        if key in FORBIDDEN_KEYS:
            problems.append(f"{path}:{pointer} uses reserved key '{key}'")
        if isinstance(value, str) and value.startswith(FORBIDDEN_PREFIXES):
            problems.append(f"{path}:{pointer} embeds executable authority")
    return problems


def api_version_of(document: object) -> str | None:
    """The document's api_version under either spelling, or None."""
    if not isinstance(document, dict):
        return None
    for key in ("api_version", "apiVersion"):
        value = document.get(key)
        if isinstance(value, str):
            return value
    return None


def schema_errors(validator, label: str, instance: object) -> list[str]:
    """Every schema violation in `instance`, rendered against `label`."""
    problems = []
    for error in sorted(validator.iter_errors(instance),
                        key=lambda e: list(e.path)):
        pointer = "/".join(str(part) for part in error.path)
        problems.append(f"{label}:/{pointer} {error.message}")
    return problems


def template_documents(path: pathlib.Path, catalog: object):
    """Yield (label, template_document) for each recipe in a catalog.

    A profile is not itself an Experiment document -- it wraps one. The wrapper
    carries the recipe key and description; `template_document` is the thing the
    authoring service mints experiment documents from, so it is the thing the
    schema has to accept.
    """
    if not isinstance(catalog, dict):
        return
    recipes = catalog.get("recipes")
    if not isinstance(recipes, list):
        return
    for index, recipe in enumerate(recipes):
        if not isinstance(recipe, dict):
            continue
        key = recipe.get("key") if isinstance(recipe.get("key"), dict) else {}
        name = key.get("name", f"#{index}")
        version = key.get("version", "?")
        yield f"{path}[{name}@{version}]", recipe.get("template_document")


def invocation_sites(spec: dict):
    """Yield (site, invocation) for every place a spec names component+operation.

    Both places count. `spec.execution` names the component and operation the
    run executes, and `spec.workflow.nodes[*].invoke` names one per node. The
    two mageflow documents declare the unresolvable `mageflow` component in
    both, so a walk over only one of them would report half the sites and read
    as a smaller problem than it is.
    """
    execution = spec.get("execution")
    if isinstance(execution, dict) and isinstance(execution.get("component"), str):
        yield "execution", execution
    nodes = spec.get("workflow", {}).get("nodes")
    if isinstance(nodes, dict):
        for name in sorted(nodes):
            node = nodes[name]
            if not isinstance(node, dict):
                continue
            invoke = node.get("invoke")
            if isinstance(invoke, dict):
                yield f"node {name}", invoke


def resolve_site(spec: dict, invoke: dict) -> tuple[str | None, str | None, str]:
    """Return (contract, problem, runtime) for one invocation site.

    A missing component, a missing operation and a missing contract are three
    different defects and are reported as three different sentences; folding
    them into one None would let a typo'd component name be reported as an
    unresolvable contract, which sends the reader to the wrong file.
    """
    component_key = invoke.get("component")
    component = spec.get("components", {}).get(component_key)
    if not isinstance(component, dict):
        return None, (
            f"invokes component {component_key!r}, which the document does not "
            f"declare under spec.components"
        ), ""
    runtime = component.get("runtime")
    runtime = runtime if isinstance(runtime, str) else ""
    operation_key = invoke.get("operation")
    operations = component.get("operations")
    operation = operations.get(operation_key) if isinstance(operations, dict) else None
    if not isinstance(operation, dict):
        return None, (
            f"invokes operation {operation_key!r} on component "
            f"{component_key!r}, which declares no such operation"
        ), runtime
    contract = operation.get("contract")
    if not isinstance(contract, str) or not contract:
        return None, (
            f"component {component_key!r} operation {operation_key!r} declares "
            f"no contract string"
        ), runtime
    return contract, None, runtime


def load_exclusions() -> tuple[dict[str, dict], list[str]]:
    """The recorded unresolvable contracts, keyed by contract id.

    Keyed by contract rather than by document on purpose. A document-keyed
    exemption would let a document that is excused for one bad id acquire a
    second one silently; and the precedent to avoid is closer than that -- an
    earlier gate in this repository keyed its lookup by filename and passed
    every mutation, because one catalog held two recipes against one contract
    and the healthy sibling masked the broken one.
    """
    if not EXCLUSIONS.exists():
        return {}, [f"missing {EXCLUSIONS}; every example contract would be excused"]
    catalog = json.loads(EXCLUSIONS.read_text())
    problems: list[str] = []
    by_contract: dict[str, dict] = {}
    for entry in catalog.get("exclusions", []):
        contract = entry.get("contract")
        if not isinstance(contract, str) or not contract:
            problems.append(f"{EXCLUSIONS}: an exclusion declares no contract")
            continue
        if contract in by_contract:
            problems.append(f"{EXCLUSIONS}: {contract} is listed twice")
            continue
        if not str(entry.get("reason", "")).strip():
            problems.append(f"{EXCLUSIONS}: {contract} is excluded with no reason")
        documents = entry.get("documents")
        if not isinstance(documents, list) or not documents:
            problems.append(
                f"{EXCLUSIONS}: {contract} names no documents, so nothing bounds "
                f"which documents the exemption covers")
            documents = []
        by_contract[contract] = {**entry, "documents": set(documents)}
    return by_contract, problems


def check_contract_resolution() -> tuple[list[str], dict[str, int]]:
    """Resolve every example invocation against the registry pin.

    Returns (problems, counts). The counts are printed whether or not the gate
    passes: a silent scope collapse -- the glob stops matching, `api_version` is
    renamed, a walk quietly finds no nodes -- shows up as a dropped count rather
    than as continued green.
    """
    problems: list[str] = []
    if not REGISTRY_PIN.exists():
        return [f"missing registry pin {REGISTRY_PIN}"], {}
    pin = json.loads(REGISTRY_PIN.read_text())
    registry = {
        entry["contract"]
        for entry in pin.get("profiles", [])
        if isinstance(entry, dict) and isinstance(entry.get("contract"), str)
    }
    if not registry:
        problems.append(
            f"{REGISTRY_PIN}: declares no contracts, so every example would "
            f"resolve against an empty registry")

    excluded, exclusion_problems = load_exclusions()
    problems += exclusion_problems

    counts = {
        "documents": 0,
        "specs": 0,
        "sites": 0,
        "resolved": 0,
        "builtin": 0,
        "excluded": 0,
        "registry": len(registry),
    }
    used: set[str] = set()

    for path in sorted(EXAMPLES.rglob("*.json")):
        counts["documents"] += 1
        label = str(path.relative_to(DOCS))
        try:
            document = json.loads(path.read_text())
        except json.JSONDecodeError as error:
            problems.append(f"{label}: is not parseable JSON ({error})")
            continue
        for identity, spec in specs(document):
            counts["specs"] += 1
            composition = f"{label}[{identity}]"
            for site, invoke in invocation_sites(spec):
                counts["sites"] += 1
                contract, problem, runtime = resolve_site(spec, invoke)
                if problem is not None:
                    problems.append(f"{composition} {site} {problem}")
                    continue
                if runtime == "builtin":
                    # The builtin core profiles are compiled into
                    # trainvm/src/adapter_registry.cpp and published in no
                    # generated artifact, so there is nothing here to resolve
                    # them against. Counted, not silently dropped.
                    counts["builtin"] += 1
                    continue
                if contract in registry:
                    counts["resolved"] += 1
                    continue
                entry = excluded.get(contract)
                if entry is None:
                    problems.append(
                        f"{composition} {site} names contract {contract!r}, "
                        f"which the registry pin {REGISTRY_PIN} does not "
                        f"declare and {EXCLUSIONS} does not record")
                    continue
                used.add(contract)
                counts["excluded"] += 1
                if label not in entry["documents"]:
                    problems.append(
                        f"{composition} {site} names excluded contract "
                        f"{contract!r}, but {EXCLUSIONS} records that exemption "
                        f"only for {sorted(entry['documents'])}; a document that "
                        f"copies an unresolvable id does not inherit its excuse")

    for contract in sorted(set(excluded) - used):
        if contract in registry:
            problems.append(
                f"{EXCLUSIONS}: {contract} is excluded but the registry pin now "
                f"declares it; delete the entry")
        else:
            problems.append(
                f"{EXCLUSIONS}: {contract} is excluded but no example document "
                f"invokes it; delete the entry")

    if counts["sites"] and not counts["resolved"]:
        problems.append(
            "no example invocation resolved against the registry pin, so this "
            "check proved nothing")
    if not counts["sites"]:
        problems.append(
            "no example document declares an invocation, so the contract "
            "resolution check walked nothing")

    return problems, counts


def main() -> int:
    if not SCHEMA.exists():
        raise SystemExit(f"missing schema: {SCHEMA}")
    schema = json.loads(SCHEMA.read_text())
    validator_class = jsonschema.validators.validator_for(schema)
    validator_class.check_schema(schema)
    validator = validator_class(schema)

    # Recursive, so examples/coverage/ is in scope too. Those fixtures were
    # only ever checked for forbidden keys; nobody had asked whether they
    # satisfy the schema.
    documents = sorted(EXAMPLES.rglob("*.json"))
    if not documents:
        raise SystemExit(f"no example documents under {EXAMPLES}")
    catalog_paths = set(EXAMPLES.rglob(RECIPE_PROFILE_GLOB))

    failures: list[str] = []
    validated = 0
    profiles = 0
    catalogs = 0
    # api_version -> paths this gate has no schema for. Named in the summary so
    # the verdict line says what it did NOT check.
    unvalidated: dict[str, list[str]] = {}

    for path in documents:
        document = json.loads(path.read_text())
        api_version = api_version_of(document)
        is_catalog_name = path in catalog_paths
        is_catalog_api = api_version == RECIPE_PROFILES_API

        if is_catalog_name != is_catalog_api:
            failures.append(
                f"{path} is named like a recipe-profile catalog but declares "
                f"api_version {api_version!r}, or the reverse; a catalog must "
                f"be recognisable both ways or it can leave this gate's scope "
                f"by editing one string")
            continue

        if is_catalog_api:
            catalogs += 1
            before = profiles
            for label, template in template_documents(path, document):
                if template is None:
                    failures.append(
                        f"{label} has no template_document; a recipe that "
                        f"mints no document cannot be checked against the "
                        f"schema, and a catalog of them would pass vacuously")
                    continue
                failures.extend(schema_errors(validator, label, template))
                profiles += 1
            if profiles == before:
                failures.append(
                    f"{path} contributed no template document to the schema "
                    f"gate")
            continue

        if api_version == EXPERIMENT_API and "spec" in document:
            failures.extend(schema_errors(validator, str(path), document))
            validated += 1
            continue

        # Everything else -- the training-component registry, the hostd daemon
        # snapshot, recipe instances, qualification evidence, the caption
        # parity contract -- has no JSON Schema in this repository. Most are
        # read by a native loader and one only by a Python test, so this line
        # deliberately claims nothing about what does check them; it states
        # only that THIS gate did not. Naming them is the point: an unnamed
        # skip is what made the old summary read as a clean bill of health.
        unvalidated.setdefault(api_version or "(no api_version)", []).append(
            path.name)

    fixtures = sorted((EXAMPLES / "coverage").glob("*.json"))
    for path in fixtures:
        failures.extend(check_no_executable_authority(path))

    # Vacuity is a failure like any other, so it belongs in the list rather
    # than after the summary. It used to print its FAIL line *below* the tally
    # and return early, which made this gate the one place where the last line
    # did state a verdict -- inconsistently with its own passing output.
    if validated == 0:
        failures.append("no experiment document was validated")
    if catalogs == 0:
        failures.append(
            f"no {RECIPE_PROFILE_GLOB} catalog was found under {EXAMPLES}; "
            f"the profiles are why this gate missed the eval_examples drift, "
            f"so their absence is a failure rather than a smaller run")

    resolution_problems, counts = check_contract_resolution()
    failures += resolution_problems

    for failure in failures:
        print(f"FAIL: {failure}")

    print(
        "NOT CONTRACT-RESOLVED: "
        f"{counts.get('builtin', 0)} builtin-runtime invocations (the "
        "trainvm.core profile set is compiled into "
        "trainvm/src/adapter_registry.cpp and published in no generated "
        "artifact) -- and for every other invocation this gate compares the "
        "contract id only, never the adapter name, version or runtime, "
        f"because {REGISTRY_PIN.name} publishes only the contract. Documents "
        f"outside {EXAMPLES} are out of scope, and whether the pin still "
        "matches a real build is checked by ctest step_zero_arming_pin and "
        "tests/test_step_zero_arming_gate.py, not here.")

    for api_version in sorted(unvalidated):
        names = ", ".join(sorted(unvalidated[api_version]))
        print(f"NOT SCHEMA-CHECKED: {api_version} ({names}) -- "
              f"experiment-v1.schema.json does not describe this api_version")

    skipped = sum(len(names) for names in unvalidated.values())
    print(verdict_line(
        "schema gate",
        failures,
        f"{validated} experiment documents and {profiles} recipe-profile "
        f"template documents from {catalogs} catalogs validated against "
        f"{SCHEMA.name}, {len(fixtures)} coverage fixtures checked for "
        f"executable authority, {skipped} example documents NOT schema-checked "
        f"(listed above); "
        f"{counts.get('sites', 0)} component/operation invocations across "
        f"{counts.get('specs', 0)} specs in {counts.get('documents', 0)} "
        f"example documents resolved against {counts.get('registry', 0)} "
        f"pinned registry contracts — {counts.get('resolved', 0)} resolved, "
        f"{counts.get('excluded', 0)} recorded in {EXCLUSIONS.name}, "
        f"{counts.get('builtin', 0)} builtin",
    ))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
