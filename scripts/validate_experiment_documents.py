#!/usr/bin/env python3
"""Validate the checked-in example experiment documents against the schema.

The C++ compiler is the authority on what a plan means, but it needs GCC 16 and
so cannot run on a hosted CI runner. This is the portable half: it proves the
example documents still satisfy experiment-v1.schema.json, which is the
contract the dashboard editor and every external producer code against.

It also fails when the coverage fixtures under examples/coverage/ regain a key
that would carry host or executable authority -- the same forbidden-key rule
the native coverage_fixture_tests enforces, kept runnable without a build.
"""

from __future__ import annotations

import json
import pathlib
import sys

import jsonschema

DOCS = pathlib.Path("docs/experiment-vm")
SCHEMA = DOCS / "experiment-v1.schema.json"
EXAMPLES = DOCS / "examples"

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


def main() -> int:
    if not SCHEMA.exists():
        raise SystemExit(f"missing schema: {SCHEMA}")
    schema = json.loads(SCHEMA.read_text())
    validator_class = jsonschema.validators.validator_for(schema)
    validator_class.check_schema(schema)
    validator = validator_class(schema)

    documents = sorted(EXAMPLES.glob("*.json"))
    if not documents:
        raise SystemExit(f"no example documents under {EXAMPLES}")

    failures: list[str] = []
    validated = 0
    for path in documents:
        document = json.loads(path.read_text())
        # Only experiment documents carry apiVersion/kind; the registry and
        # daemon examples are validated by their own native loaders.
        if document.get("apiVersion") is None and document.get(
                "api_version") is None:
            continue
        if "spec" not in document:
            continue
        errors = sorted(validator.iter_errors(document), key=lambda e: e.path)
        for error in errors:
            pointer = "/".join(str(part) for part in error.path)
            failures.append(f"{path}:/{pointer} {error.message}")
        validated += 1

    fixtures = sorted((EXAMPLES / "coverage").glob("*.json"))
    for path in fixtures:
        failures.extend(check_no_executable_authority(path))

    for failure in failures:
        print(f"FAIL: {failure}")
    print(
        f"schema gate: {validated} experiment documents validated, "
        f"{len(fixtures)} coverage fixtures checked for executable authority"
    )
    if validated == 0:
        print("FAIL: no experiment document was validated")
        return 1
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
