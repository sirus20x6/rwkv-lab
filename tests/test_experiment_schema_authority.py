"""The JSON Schema restates rules the native authority already enforces.

`docs/experiment-vm/experiment-v1.schema.json` is hand-maintained and shares no
mechanical link with the reflected C++ structs it describes. It is the surface
external authors and the dashboard editor validate against; `trainvm/src/
document.cpp` and the headers under `trainvm/include/trainvm/` are what the
authority actually enforces. Two independent statements of one fact, with
nothing forcing them to agree, is the defect this repository keeps re-finding.

It had already drifted, in three places, all silent because no checked-in
document exercised them:

- `TrainingComponentCategory` has 28 categories and the schema listed 26. A
  document selecting an `activation_memory` or `generation_policy` component was
  refused by the schema gate before the native authority ever saw it.
- `ArtifactType` has `eval_examples` and the schema did not, even though the
  shipped `hf-multimodal-sft` recipe profiles emit exactly that artifact type
  and `eval_examples_contract.cpp` requires the declaration to be present.
- Neither direction of either was reported by anything.

## Why a pin and not generation

The repository already answers this exact question with a pinning test:
`test_worker_accepts_every_category_the_shipped_registry_uses` pins the worker's
`_CATEGORIES`. Adding a third mechanism for a third copy of one vocabulary would
itself be the defect these checks are about, so this extends the established
pattern instead.

Generating the schema would need a schema-emission subcommand, which moves
`main.cpp`'s `classification_surface_digest` — the whole file is its
classification surface, so any edit there does — and taxes every future touch
with the four-step pin refresh in `CLAUDE.md`. It also could not reproduce the
descriptions, examples and `if`/`then` prose the schema is hand-written to
carry, short of annotating every struct.

The regex and length bounds below cannot have exactly one home either: the C++
cannot read the schema at compile time and JSON Schema cannot reference a C++
constant. Short of generation, a check that fails when the two copies disagree
is the honest fix.

## Every assertion here is two-directional

A rule stated in the schema but not in the native source is exactly as broken as
the reverse, and it is the direction a hand-edit produces.
"""

from __future__ import annotations

import json
import pathlib
import re

REPOSITORY = pathlib.Path(__file__).resolve().parent.parent
SCHEMA_PATH = REPOSITORY / "docs/experiment-vm/experiment-v1.schema.json"
DOCUMENT_SOURCE = REPOSITORY / "trainvm/src/document.cpp"
HEADER_DIRECTORY = REPOSITORY / "trainvm/include/trainvm"

# Schema JSON pointer -> the C++ scoped enum stating the same vocabulary.
# Every entry is checked in both directions.
PINNED_ENUMS = {
    "/$defs/training_component_key/properties/category": "TrainingComponentCategory",
    "/$defs/artifact/properties/type": "ArtifactType",
    "/$defs/artifact/properties/immutability": "Immutability",
    "/$defs/artifact/properties/fingerprint": "Fingerprint",
    "/$defs/parameter/properties/type": "ParameterType",
    "/$defs/component/properties/runtime": "ComponentRuntime",
    "/$defs/control/properties/type": "ControlType",
    "/$defs/control/properties/apply": "ApplyPoint",
    "/$defs/metric/properties/type": "MetricType",
    "/$defs/metric/properties/step_domain": "StepDomain",
    "/$defs/metric/properties/aggregation": "Aggregation",
    "/$defs/node/properties/idempotency": "Idempotency",
    "/$defs/node/properties/effect": "Effect",
    "/$defs/retry/properties/backoff": "Backoff",
    "/$defs/loop_guard/properties/direction": "ProgressDirection",
    "/$defs/recovery/properties/reconcile": "ReconcilePolicy",
    "/$defs/recovery/properties/orphan_policy": "OrphanPolicy",
    "/$defs/resources/properties/accelerators/properties/vendor": "AcceleratorVendor",
    "/$defs/gpu_trace_capture/properties/backend": "ProfilerBackend",
    "/$defs/gpu_trace_capture/properties/activities/items": "ProfilerActivity",
    "/$defs/input_content_root/properties/kind": "ContentRootKind",
    "/$defs/post_training_arm/properties/kind": "PostTrainingArmKind",
    "/$defs/post_training_arm/properties/bounds/items/properties/kind": "RunBoundKind",
    "/$defs/post_training_arm/properties/reproducibility_claim": "ReproducibilityClaim",
}

# The wire name differs from the C++ enumerator, on purpose and in one place:
# `enum` is a keyword, so ControlType spells it `enumeration` and
# reflection_json.hpp's enum_from_string/enum_to_string translate it. Encoded
# here rather than inferred, so that removing the translation fails this test.
WIRE_NAME_ALIASES = {("ControlType", "enumeration"): "enum"}

# Vocabularies stated as a std::set<std::string> literal in document.cpp rather
# than as a scoped enum. Schema pointer -> the C++ variable name.
PINNED_STRING_SETS = {
    "/$defs/comparison/properties/operator": "operators",
    "/$defs/binding/oneOf/4/properties/context": "contexts",
}

# Schema enums that intentionally state a *subset* of a native vocabulary.
# Pointer -> enum name. Checked as a subset, which still catches a typo or a
# name the authority has dropped.
SUBSET_ENUMS = {
    # The if/then clause that forbids torch-only knobs on the sampling backends.
    "/$defs/gpu_trace_capture/allOf/2/if/properties/backend": "ProfilerBackend",
}


def _schema() -> dict:
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


def _schema_string_enums() -> dict[str, list[str]]:
    """Every all-string `enum` in the schema, keyed by JSON pointer."""
    found: dict[str, list[str]] = {}

    def walk(node: object, pointer: str = "") -> None:
        if isinstance(node, dict):
            values = node.get("enum")
            if isinstance(values, list) and all(
                isinstance(entry, str) for entry in values
            ):
                found[pointer] = values
            for key, child in node.items():
                walk(child, f"{pointer}/{key}")
        elif isinstance(node, list):
            for index, child in enumerate(node):
                walk(child, f"{pointer}/{index}")

    walk(_schema())
    return found


def _native_enums() -> dict[str, list[str]]:
    """Every scoped enum declared under trainvm/include/trainvm, in order."""
    enums: dict[str, list[str]] = {}
    for header in sorted(HEADER_DIRECTORY.glob("*.hpp")):
        source = header.read_text(encoding="utf-8")
        for match in re.finditer(
            r"enum class (\w+)\s*(?::[^{]*)?\{(.*?)\}\s*;", source, re.DOTALL
        ):
            name, body = match.group(1), match.group(2)
            assert "=" not in body, (
                f"{name} in {header.name} assigns explicit enumerator values; "
                f"this parse would report the assignment as part of the name")
            enumerators = [entry.strip() for entry in body.split(",")]
            enums[name] = [entry for entry in enumerators if entry]
    return enums


def _wire_names(enum: str, enumerators: list[str]) -> list[str]:
    return [WIRE_NAME_ALIASES.get((enum, name), name) for name in enumerators]


def _document_string_set(name: str) -> list[str]:
    source = DOCUMENT_SOURCE.read_text(encoding="utf-8")
    match = re.search(
        r"std::set<std::string>\s+" + re.escape(name) + r"\s*\{(.*?)\}\s*;",
        source,
        re.DOTALL,
    )
    assert match is not None, (
        f"`{name}` is no longer a std::set<std::string> literal in "
        f"document.cpp; this pin cannot read the rule it pins against, so fix "
        f"the parse rather than deleting the check")
    return re.findall(r'"([^"]*)"', match.group(1))


def test_every_pinned_schema_enum_matches_its_native_enum() -> None:
    """Both directions, and in declaration order.

    Order equality is deliberate: it also catches a duplicate entry, and it
    keeps the two files diffable side by side, which is how a human notices a
    drift before this test does.
    """
    schema_enums = _schema_string_enums()
    native_enums = _native_enums()
    problems = []

    for pointer, enum_name in PINNED_ENUMS.items():
        assert pointer in schema_enums, (
            f"{pointer} no longer names an enum in the schema; the pin against "
            f"{enum_name} has lost its target")
        assert enum_name in native_enums, (
            f"{enum_name} is no longer declared under {HEADER_DIRECTORY.name}/; "
            f"the pin for {pointer} has lost its authority")

        expected = _wire_names(enum_name, native_enums[enum_name])
        actual = schema_enums[pointer]
        missing = [name for name in expected if name not in actual]
        extra = [name for name in actual if name not in expected]
        if missing:
            problems.append(
                f"{pointer} refuses values {enum_name} admits: {missing} — a "
                f"document naming one is rejected by the schema gate before "
                f"the native authority ever sees it")
        if extra:
            problems.append(
                f"{pointer} admits values {enum_name} does not have: {extra} — "
                f"a document the schema accepts would be rejected by the "
                f"authority as corrupt")
        if not missing and not extra and actual != expected:
            problems.append(
                f"{pointer} has {enum_name}'s values in a different order or "
                f"with duplicates: {actual} vs {expected}")

    assert not problems, "\n".join(problems)


def test_every_schema_enum_mirroring_a_native_enum_is_pinned() -> None:
    """A new mirrored vocabulary must join the table, not appear beside it.

    This cannot catch an unpinned enum that already disagrees — it recognises a
    mirror by its values, so a drifted one no longer looks like one. It catches
    the case that actually occurs: someone adds a schema enum for a C++ enum and
    does not think to pin it.
    """
    native_enums = _native_enums()
    by_values = {
        frozenset(_wire_names(name, values)): name
        for name, values in native_enums.items()
    }
    unpinned = []
    for pointer, values in _schema_string_enums().items():
        if pointer in PINNED_ENUMS or pointer in SUBSET_ENUMS:
            continue
        match = by_values.get(frozenset(values))
        if match is not None:
            unpinned.append(f"{pointer} mirrors {match}")
    assert not unpinned, (
        "these schema enums restate a native enum without being pinned to it; "
        f"add them to PINNED_ENUMS: {unpinned}")


def test_subset_schema_enums_stay_within_their_native_enum() -> None:
    schema_enums = _schema_string_enums()
    native_enums = _native_enums()
    for pointer, enum_name in SUBSET_ENUMS.items():
        assert pointer in schema_enums, f"{pointer} no longer names an enum"
        expected = set(_wire_names(enum_name, native_enums[enum_name]))
        actual = set(schema_enums[pointer])
        assert actual < expected, (
            f"{pointer} is documented as a proper subset of {enum_name} but "
            f"names {sorted(actual - expected)}, which {enum_name} does not "
            f"have")


def test_pinned_string_set_vocabularies_match_document_cpp() -> None:
    schema_enums = _schema_string_enums()
    for pointer, variable in PINNED_STRING_SETS.items():
        expected = set(_document_string_set(variable))
        actual = set(schema_enums[pointer])
        assert actual == expected, (
            f"{pointer} and document.cpp's `{variable}` disagree; refused by "
            f"the schema but accepted by the authority: "
            f"{sorted(expected - actual)}; accepted by the schema but refused "
            f"by the authority: {sorted(actual - expected)}")


def _native_identifier_rule() -> tuple[str, int]:
    """Return (regex, max length) as `document.cpp` states them."""
    source = DOCUMENT_SOURCE.read_text(encoding="utf-8")

    pattern = re.search(r'const std::regex kIdentifier\("([^"]*)"\);', source)
    assert pattern is not None, (
        "kIdentifier is no longer a single-literal std::regex in document.cpp; "
        "this pin cannot read the rule it is supposed to pin against")
    literal = pattern.group(1)
    assert "\\" not in literal, (
        "kIdentifier's literal now contains a backslash escape, so comparing "
        "the raw C++ literal against the JSON Schema pattern would compare two "
        "different strings")

    bound = re.search(
        r"bool is_identifier\(const std::string& value\) \{\s*"
        r"return value\.size\(\) <= (\d+)U",
        source,
    )
    assert bound is not None, (
        "is_identifier no longer opens with a literal size bound in "
        "document.cpp")
    return literal, int(bound.group(1))


def test_the_schema_identifier_matches_the_native_identifier_rule() -> None:
    """The regex and its 128-byte bound appear verbatim in both files."""
    native_pattern, native_max = _native_identifier_rule()
    identifier = _schema()["$defs"]["identifier"]

    assert identifier["pattern"] == native_pattern, (
        f"the schema's identifier pattern {identifier['pattern']!r} is not the "
        f"rule document.cpp enforces ({native_pattern!r}); one of the two "
        f"surfaces is accepting names the other refuses")
    assert identifier["maxLength"] == native_max, (
        f"the schema caps identifiers at {identifier['maxLength']} bytes and "
        f"document.cpp at {native_max}")


def test_the_schema_event_field_pattern_matches_the_native_one() -> None:
    source = DOCUMENT_SOURCE.read_text(encoding="utf-8")
    pattern = re.search(r'const std::regex kEventField\("([^"]*)"\);', source)
    assert pattern is not None, (
        "kEventField is no longer a single-literal std::regex in document.cpp")
    native = pattern.group(1)

    schema = _schema()
    stated = (
        ("/$defs/comparison/properties/field",
         schema["$defs"]["comparison"]["properties"]["field"]),
        ("/$defs/loop_guard/properties/progress_field",
         schema["$defs"]["loop_guard"]["properties"]["progress_field"]),
    )
    for pointer, node in stated:
        assert node["pattern"] == native, (
            f"{pointer} states {node['pattern']!r} but document.cpp's "
            f"kEventField is {native!r}")


def test_the_schema_component_version_bound_matches_the_native_one() -> None:
    source = DOCUMENT_SOURCE.read_text(encoding="utf-8")
    bound = re.search(r"selection\.key\.version\.size\(\) > (\d+)U", source)
    assert bound is not None, (
        "the training component version bound is no longer a literal in "
        "document.cpp")
    native_max = int(bound.group(1))

    version = _schema()["$defs"]["training_component_key"]["properties"][
        "version"]
    assert version["maxLength"] == native_max, (
        f"the schema caps component versions at {version['maxLength']} bytes "
        f"and document.cpp at {native_max}")
    assert version["minLength"] == 1, (
        "document.cpp rejects an empty component version; the schema must too")


def test_documents_using_the_recovered_vocabulary_validate() -> None:
    """The concrete bugs, stated as the behaviour they broke.

    Until the enums were completed each of these was refused by
    scripts/validate_experiment_documents.py before the authority saw it.
    """
    import jsonschema

    schema = _schema()

    def probe(ref: str, instance: object) -> None:
        jsonschema.validate(
            instance,
            {"$schema": schema["$schema"], "$defs": schema["$defs"],
             "$ref": ref},
        )

    for category in ("activation_memory", "generation_policy"):
        probe("#/$defs/training_component_key",
              {"category": category, "name": "example", "version": "1"})
    probe(
        "#/$defs/artifact",
        {
            "type": "eval_examples",
            "immutability": "immutable",
            "fingerprint": "manifest_sha256",
        },
    )
