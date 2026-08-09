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
"""

from __future__ import annotations

import json
import pathlib
import sys

import jsonschema

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from scripts.gate_verdict import verdict_line  # noqa: E402

DOCS = pathlib.Path("docs/experiment-vm")
SCHEMA = DOCS / "experiment-v1.schema.json"
EXAMPLES = DOCS / "examples"

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

    for failure in failures:
        print(f"FAIL: {failure}")

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
        f"(listed above)",
    ))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
