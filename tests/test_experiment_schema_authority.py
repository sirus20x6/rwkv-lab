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


# --- The gate's scope, not just the schema's content ---------------------
#
# The three drifts above were fixed. They survived as long as they did because
# scripts/validate_experiment_documents.py looked at three example documents and
# none of the recipe-profile catalogs, then printed PASSED. The schema and the
# authority disagreed and the check that would have noticed was pointed
# somewhere else, so the checks below are about where the gate looks rather than
# about what the schema says.

import shutil  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402

GATE = REPOSITORY / "scripts/validate_experiment_documents.py"
EXAMPLES = REPOSITORY / "docs/experiment-vm/examples"
RECIPE_PROFILE_GLOB = "*.recipe-profiles.v1.json"


def _run_gate(root: pathlib.Path) -> subprocess.CompletedProcess:
    """Run the real gate with `root` as its working directory.

    The gate resolves docs/experiment-vm relative to the cwd but imports
    scripts.gate_verdict relative to its own location, so a copied docs tree is
    enough to exercise it against mutated inputs without touching the
    repository.
    """
    return subprocess.run(
        [sys.executable, str(GATE)], cwd=root,
        capture_output=True, text=True, check=False)


def _sandbox(tmp_path: pathlib.Path) -> pathlib.Path:
    root = tmp_path / "repository"
    (root / "docs").mkdir(parents=True)
    shutil.copytree(REPOSITORY / "docs/experiment-vm",
                    root / "docs/experiment-vm")
    return root


def test_the_gate_validates_every_recipe_profile_catalog_on_disk() -> None:
    """Scope stated over the tree, not over a list.

    A hand-maintained list of catalogs would drift exactly as the schema enums
    did, one level up: a new catalog would be added, nobody would remember the
    list, and the gate would keep printing a confident PASSED about a set that
    no longer matched the directory. So the property asserted is set equality
    between what the tree holds and what the gate says it read.
    """
    on_disk = sorted(EXAMPLES.rglob(RECIPE_PROFILE_GLOB))
    assert on_disk, (
        f"no {RECIPE_PROFILE_GLOB} under {EXAMPLES}; this test would pass "
        f"vacuously")

    completed = _run_gate(REPOSITORY)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    summary = completed.stdout.strip().splitlines()[-1]

    counted = re.search(r"from (\d+) catalogs", summary)
    assert counted is not None, (
        f"the gate's summary no longer states how many catalogs it read: "
        f"{summary}")
    assert int(counted.group(1)) == len(on_disk), (
        f"{len(on_disk)} recipe-profile catalogs are on disk but the gate "
        f"read {counted.group(1)}: {summary}")


def test_the_gate_summary_names_what_it_did_not_check() -> None:
    """`PASSED -- 3 documents` used to read as "the examples are clean".

    It was a statement about three of them. The example directory also holds
    recipe instances, a component registry, a daemon snapshot, qualification
    evidence and a caption parity contract, none of which have a JSON Schema
    here. Leaving them unmentioned is what made the old summary misleading, so
    the gate has to say they were skipped and how many.

    What the gate must NOT do is assert what covers them instead. An earlier
    draft of this line said "validated by its own native loader", which is true
    of five of the six api_versions -- the parity contract is read only by
    tests/test_qwen_caption_declarative_parity.py. A gate that volunteers an
    unchecked reassurance about coverage elsewhere is the same defect in
    miniature, so the line states only that this gate did not look.
    """
    completed = _run_gate(REPOSITORY)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    lines = completed.stdout.strip().splitlines()

    skipped = [line for line in lines if line.startswith("NOT SCHEMA-CHECKED:")]
    assert skipped, (
        "the example directory holds documents this gate has no schema for; "
        "the gate reported none of them")
    assert re.search(r"NOT schema-checked", lines[-1]), (
        f"the verdict line does not state that anything went unchecked: "
        f"{lines[-1]}")


def test_removing_eval_examples_from_the_schema_turns_the_gate_red(
        tmp_path: pathlib.Path) -> None:
    """The exact historical drift, replayed against the current gate.

    `eval_examples` was absent from the artifact type enum while the shipped
    hf-multimodal-sft profiles emitted it, and the gate said PASSED for as long
    as that lasted. Reintroducing the omission must now fail, and must fail
    naming the profiles -- if it only failed on some other document the gate
    would be red for the wrong reason and the drift could return.
    """
    root = _sandbox(tmp_path)
    schema_path = root / "docs/experiment-vm/experiment-v1.schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    enum = schema["$defs"]["artifact"]["properties"]["type"]["enum"]
    assert "eval_examples" in enum, (
        "the enum no longer carries eval_examples; this test cannot replay the "
        "drift it exists for")
    enum.remove("eval_examples")
    schema_path.write_text(json.dumps(schema, indent=2) + "\n",
                           encoding="utf-8")

    completed = _run_gate(root)
    assert completed.returncode != 0, (
        f"the gate stayed green with eval_examples removed from the schema, "
        f"which is the state that shipped:\n{completed.stdout}")

    blamed = [line for line in completed.stdout.splitlines()
              if line.startswith("FAIL:")
              and RECIPE_PROFILE_GLOB.replace("*", "") in line]
    assert blamed, (
        f"the gate failed but not on a recipe-profile catalog, so it is not "
        f"the profiles that caught this:\n{completed.stdout}")


def test_a_profile_emitting_an_undeclared_artifact_type_turns_the_gate_red(
        tmp_path: pathlib.Path) -> None:
    """The other direction: a new catalog is in scope the moment it exists.

    Nothing has to be registered anywhere for this to be checked, which is the
    difference between a scope derived from the tree and an enumeration someone
    has to remember to update.
    """
    root = _sandbox(tmp_path)
    examples = root / "docs/experiment-vm/examples"
    source = next(iter(sorted(examples.glob(RECIPE_PROFILE_GLOB))))
    catalog = json.loads(source.read_text(encoding="utf-8"))
    recipe = catalog["recipes"][0]
    recipe["key"] = {"name": "probe_undeclared_type", "version": "1"}
    artifacts = recipe["template_document"]["spec"]["artifacts"]
    artifacts[next(iter(artifacts))]["type"] = "telemetry_stream"
    (examples / "probe.recipe-profiles.v1.json").write_text(
        json.dumps({"api_version": "trainvm.recipe-profiles/v1",
                    "recipes": [recipe]}, indent=2) + "\n",
        encoding="utf-8")

    completed = _run_gate(root)
    assert completed.returncode != 0, (
        f"a profile emitting an artifact type the schema does not declare was "
        f"accepted:\n{completed.stdout}")
    assert "telemetry_stream" in completed.stdout, (
        f"the gate failed without naming the undeclared type:\n"
        f"{completed.stdout}")


def test_a_catalog_cannot_leave_the_gate_by_renaming_its_api_version(
        tmp_path: pathlib.Path) -> None:
    """Recognition is by filename AND api_version, and they must agree.

    Either signal alone is escapable: keyed only on api_version, editing one
    string drops a catalog out of scope silently; keyed only on the filename, a
    catalog named anything else is never seen. Disagreement between the two is
    therefore a failure rather than a skip.
    """
    root = _sandbox(tmp_path)
    examples = root / "docs/experiment-vm/examples"
    target = next(iter(sorted(examples.glob(RECIPE_PROFILE_GLOB))))
    catalog = json.loads(target.read_text(encoding="utf-8"))
    catalog["api_version"] = "trainvm.something-else/v1"
    target.write_text(json.dumps(catalog, indent=2) + "\n", encoding="utf-8")

    completed = _run_gate(root)
    assert completed.returncode != 0, (
        f"a recipe-profile catalog left this gate's scope by editing one "
        f"string, and the gate stayed green:\n{completed.stdout}")
