"""The step-zero arming gate must go red for each condition independently.

`scripts/ci_step_zero_arming_gate.py` asserts a five-condition property of every
eval-examples publication a shipped composition declares: the three conjuncts of
the arming predicate (`required`, `type`, `schema`, from
`trainvm/src/eval_examples_contract.cpp`) and the two the publisher separately
demands of the same declaration (`immutability`, `fingerprint`, from
`src/rwkv_lab/trainvm_worker/eval_examples.py`). A check that tests one while
asserting five passes every mutation of the other four, and from the outside it
is indistinguishable from a correct one: it is green today, green after the
mutation, and green when a real edit breaks arming.

So each condition is mutated separately here and the gate must fail for each.
That is the only evidence that the conjunction is real. The gate checked three
of the five until `card-20510814`; the two publisher tests below are the
regression that could not have been observed before, since satisfying only the
three arms the controller and then deadlocks it.

Two failure modes of this file itself, both already observed while writing it:

- **A mutation that does not apply reads as PASSED.** An earlier shell version
  of this battery mis-quoted two of the three mutations; the gate printed
  PASSED for both, and the transcript looked like proof that the check does not
  bite. Every mutation below therefore asserts that it changed the document.
- **File-level identity hides sibling drift.** An earlier draft of the gate
  keyed arming compositions by filename. `rwkv-lm.recipe-profiles.v1.json` holds
  two recipes against the same contract, so breaking one left the cell
  unchanged and all three mutations passed. The gate now names each composition
  by its document `metadata.name`, and `test_a_sibling_recipe_cannot_hide_a_broken_one`
  pins that specific behaviour.

The gate is run as a subprocess against a copied repository rather than
imported, because its verdict is an exit code and a printed line and those are
what CI reads.
"""

from __future__ import annotations

import json
import os
import pathlib
import re
import shutil
import subprocess
import sys

import pytest

REPOSITORY = pathlib.Path(__file__).resolve().parent.parent
GATE = "scripts/ci_step_zero_arming_gate.py"
CATALOG = "docs/experiment-vm/examples/rwkv-lm.recipe-profiles.v1.json"
PIN = "docs/experiment-vm/step-zero-arming.v1.json"
DOCUMENT = "docs/experiment-vm/STEP_ZERO_ARMING.md"
PUBLISHER = "src/rwkv_lab/trainvm_worker/eval_examples.py"

# The recipe whose declaration is mutated, and the contract it arms. Named
# rather than indexed so a reordered catalog fails loudly instead of quietly
# mutating a different recipe.
RECIPE = "rwkv_lm_scratch"
CONTRACT = "rwkv_lab.rwkv_scratch.v1.Train"

# The same markers the gate itself slices on (`ci_step_zero_arming_gate.py`
# BEGIN/END). Restated rather than imported because the gate is deliberately run
# as a subprocess here, not imported -- see this module's docstring.
TABLE_BEGIN = "<!-- BEGIN GENERATED ARMING TABLE -->"
TABLE_END = "<!-- END GENERATED ARMING TABLE -->"


def _generated_table(repository: pathlib.Path) -> tuple[list[str], list[list[str]]]:
    """The generated table's header and route rows, sliced between its markers.

    An earlier version selected rows from the WHOLE document by the prefix
    "| `", which is two defects wearing one line. Any second Markdown table in
    the prose feeds it rows, and a row shorter than the widest index raises a
    bare `IndexError` from a test whose name is about a verdict line -- an
    investigation that ends nowhere near the edit that caused it. The gate has
    always sliced on these markers; only the test did not.
    """
    document = (repository / DOCUMENT).read_text(encoding="utf-8")
    assert TABLE_BEGIN in document and TABLE_END in document, (
        f"{DOCUMENT} is missing its generated-table markers, so this test "
        f"cannot tell the generated table from prose")
    body = document.split(TABLE_BEGIN, 1)[1].split(TABLE_END, 1)[0]

    lines = [line.strip() for line in body.splitlines() if line.strip().startswith("|")]
    assert len(lines) >= 3, (
        f"{DOCUMENT}: the generated table has {len(lines)} lines between its "
        f"markers; a header, a separator and at least one route are required")

    def cells(line: str) -> list[str]:
        return [cell.strip() for cell in line.strip().strip("|").split("|")]

    header = cells(lines[0])
    rows = [cells(line) for line in lines[2:]]
    for row in rows:
        assert len(row) == len(header), (
            f"{DOCUMENT}: a table row has {len(row)} cells against a "
            f"{len(header)}-cell header, so column positions cannot be trusted:"
            f"\n{row}")
    assert rows, f"{DOCUMENT} holds no route rows, so this test proves nothing"
    return header, rows


def _column(header: list[str], name: str) -> int:
    """Resolve a column by its header text, never by position.

    Positional indices are how this file's counts quietly start measuring a
    different column: the table has gained a column before, and an index that
    still parses reads exactly like one that is right.
    """
    matches = [i for i, cell in enumerate(header) if cell.startswith(name)]
    assert len(matches) == 1, (
        f"{DOCUMENT}: expected exactly one column starting {name!r}, found "
        f"{len(matches)} in {header}")
    return matches[0]



def _copy(tmp_path: pathlib.Path) -> pathlib.Path:
    """A minimal copy of the repository: the gate, its inputs, and the package.

    Copying the whole checkout would drag in build trees measured in gigabytes.
    """
    root = tmp_path / "repository"
    (root / "scripts").mkdir(parents=True)
    (root / "docs/experiment-vm").mkdir(parents=True)
    # The publisher source is an input, not decoration: the gate reads the two
    # conditions only it states out of this file's AST rather than restating
    # them, so a copy without it measures three conditions and looks fine.
    (root / PUBLISHER).parent.mkdir(parents=True)
    shutil.copy2(REPOSITORY / PUBLISHER, root / PUBLISHER)
    for name in (GATE, "scripts/gate_verdict.py"):
        shutil.copy2(REPOSITORY / name, root / name)
    (root / "scripts/__init__.py").touch()
    for name in (PIN, DOCUMENT):
        shutil.copy2(REPOSITORY / name, root / name)
    shutil.copytree(
        REPOSITORY / "docs/experiment-vm/examples", root / "docs/experiment-vm/examples"
    )
    return root


def _run(root: pathlib.Path) -> subprocess.CompletedProcess[str]:
    # `rwkv_lab` is imported from this checkout, not from the copy: the copy
    # holds the documents under test, and the worker registry it is compared
    # against is the real one. The schema CI job installs the package, so the
    # gate finds it there without help.
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(REPOSITORY / "src"), environment.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    return subprocess.run(
        [sys.executable, str(root / GATE), "--repository", str(root)],
        capture_output=True,
        text=True,
        check=False,
        env=environment,
    )


def _catalog(root: pathlib.Path) -> dict:
    return json.loads((root / CATALOG).read_text(encoding="utf-8"))


def _write(root: pathlib.Path, catalog: dict) -> None:
    (root / CATALOG).write_text(json.dumps(catalog, indent=2), encoding="utf-8")


def _recipe(catalog: dict, name: str) -> dict:
    for recipe in catalog["recipes"]:
        if recipe["key"]["name"] == name:
            return recipe
    raise AssertionError(f"{CATALOG} has no recipe named {name}")


@pytest.fixture
def repository(tmp_path: pathlib.Path) -> pathlib.Path:
    return _copy(tmp_path)


def test_the_gate_passes_on_an_unmutated_copy(repository: pathlib.Path) -> None:
    """The floor. Every mutation below is only evidence against this baseline.

    Without it, a gate that failed for an unrelated reason -- a missing input in
    the copy, say -- would make every mutation test pass while proving nothing.
    """

    result = _run(repository)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "step-zero arming gate: PASSED" in result.stdout


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("required", False),
        ("type", "report"),
        ("schema", "rwkv-lab.eval-examples.v2"),
    ],
    ids=["required", "type", "schema"],
)
def test_each_conjunct_reddens_the_gate(
    repository: pathlib.Path, field: str, value: object
) -> None:
    catalog = _catalog(repository)
    artifact = _recipe(catalog, RECIPE)["template_document"]["spec"]["artifacts"][
        "eval_examples"
    ]
    assert artifact[field] != value, (
        f"the mutation is a no-op: {field} already is {value!r}, so a PASSED "
        "verdict below would prove nothing"
    )
    artifact[field] = value
    _write(repository, catalog)

    result = _run(repository)
    assert result.returncode == 1, (
        f"mutating {field} left the gate green:\n{result.stdout}{result.stderr}"
    )
    assert CONTRACT in result.stdout, (
        f"the gate failed without naming {CONTRACT}, so a reader cannot tell "
        f"which route drifted:\n{result.stdout}"
    )


def test_a_sibling_recipe_cannot_hide_a_broken_one(repository: pathlib.Path) -> None:
    """Two recipes in one file arm the same contract; breaking one must show.

    This is the exact false negative an earlier draft had. It is pinned
    separately from the conjunct battery because it is a property of how
    compositions are identified, not of the predicate.
    """

    catalog = _catalog(repository)
    names = [
        recipe["key"]["name"]
        for recipe in catalog["recipes"]
        if "eval_examples"
        in recipe["template_document"]["spec"]["artifacts"]
    ]
    assert len(names) > 1, (
        f"{CATALOG} no longer holds two arming recipes, so this test cannot "
        f"detect the failure it exists for; it holds {names}"
    )
    _recipe(catalog, RECIPE)["template_document"]["spec"]["artifacts"][
        "eval_examples"
    ]["required"] = False
    _write(repository, catalog)

    result = _run(repository)
    assert result.returncode == 1, result.stdout + result.stderr
    assert RECIPE.replace("_", "-") in result.stdout.replace("_", "-"), (
        f"the gate did not name the recipe that broke:\n{result.stdout}"
    )


def test_the_pin_names_exactly_the_dispatched_routes() -> None:
    """A route the worker dispatches must have a row, and nothing else may.

    This lives here rather than in the gate because
    `rwkv_lab.trainvm_adapters.handlers` imports torch transitively and the
    schema job installs `.[test]` without it. This job installs torch and runs
    on every pull request, so nothing is skipped by the split; the gate's
    `registry_parity_note` records the reasoning at the other end.

    Reading the literal statically was tried and rejected: `_HANDLERS` ends in a
    `**` unpacking of a dict comprehension over `PROFILE_ADAPTERS`, so eight of
    the twenty-one contracts are not string literals anywhere in the file and an
    AST reader reports thirteen while looking correct.

    ctest `step_zero_arming_pin` is the other half -- it compares the same pin
    field by field against a real build, which is the only thing that can check
    the evaluator/checkpoint/port values themselves.
    """

    from rwkv_lab.trainvm_adapters.handlers import supported_adapter_keys

    pin = json.loads((REPOSITORY / PIN).read_text(encoding="utf-8"))
    pinned = {entry["contract"] for entry in pin["profiles"]}
    dispatched = {key[3] for key in supported_adapter_keys()}
    assert pinned == dispatched, (
        "the pin and the worker dispatch table disagree; regenerate with "
        "scripts/print_step_zero_arming_pin.py --write. "
        f"dispatched only: {sorted(dispatched - pinned)}; "
        f"pinned only: {sorted(pinned - dispatched)}"
    )


def _population(summary: str, label: str) -> int:
    """Read one population out of the verdict line.

    The counts are the gate's own answer, so a test asserting a delta has to
    take the baseline from the gate rather than from a constant that goes stale
    the next time a family arms.
    """

    match = re.search(rf"(\d+) {re.escape(label)}", summary)
    assert match is not None, f"the verdict line names no {label!r}: {summary}"
    return int(match.group(1))


def _summary(stdout: str) -> str:
    """The verdict line, which is the part of the output that gets quoted."""
    lines = [line for line in stdout.splitlines() if "step-zero arming gate:" in line]
    assert len(lines) == 1, f"expected exactly one verdict line:\n{stdout}"
    return lines[0]


def test_the_verdict_line_says_armed_not_able_to_arm(repository: pathlib.Path) -> None:
    """The number that gets quoted must not read as remaining work.

    Asserted on the message rather than the exit code on purpose: the defect
    this pins is a correct computation described loosely, and the gate was green
    throughout. Only a wording assertion can catch it coming back.

    The counts are read out of the table rather than hardcoded, so a route
    arming for the first time turns this test's expectation green with it
    instead of reddening a test that has nothing to do with the change.
    """

    result = _run(repository)
    assert result.returncode == 0, result.stdout + result.stderr
    summary = _summary(result.stdout)

    header, rows = _generated_table(repository)
    can_arm = _column(header, "can arm today?")
    has_port = _column(header, "eval_examples output port")
    stateful = _column(header, "stateful")

    armed = sum(1 for row in rows if row[can_arm].startswith("yes"))
    armable = sum(
        1 for row in rows if row[can_arm] == "no" and row[has_port] == "yes")
    blocked = sum(
        1 for row in rows
        if row[can_arm] == "no" and row[has_port] != "yes" and row[stateful] == "yes"
    )

    assert "able to arm today" not in summary, (
        "the verdict describes a count of already-armed routes as a capability, "
        f"which reads as a backlog of unarmed ones:\n{summary}"
    )
    assert f"{armed} armed today by a shipped composition" in summary, summary
    assert f"{armable} armable but unarmed" in summary, summary
    assert f"{blocked} blocked" in summary, summary
    assert f"{len(rows)} registered routes" in summary, summary
    assert "not stateful" in summary, summary


def test_the_verdict_line_names_partial_arming(repository: pathlib.Path) -> None:
    """A route armed by one composition and not its sibling must say so.

    `missing()` already refuses to write a bare "armed" for this case. The
    summary collapsing it would put the looser claim in the line that is read
    most often, which is the whole failure this file's wording tests exist for.
    """

    catalog = _catalog(repository)
    _recipe(catalog, RECIPE)["template_document"]["spec"]["artifacts"][
        "eval_examples"
    ]["required"] = False
    _write(repository, catalog)

    summary = _summary(_run(repository).stdout)
    assert "1 of those armed in some compositions and not others" in summary, summary


@pytest.mark.parametrize(
    ("count", "expected"),
    [
        (
            0,
            "0 registered routes (0 stateful); 0 armed today by a shipped "
            "composition, 0 would deadlock (arms the controller, the publisher "
            "refuses the declaration), 0 armable but unarmed (port in place, no "
            "shipped composition arms it), 0 blocked, 0 not stateful",
        ),
        (
            1,
            "1 registered route (1 stateful); 1 armed today by a shipped "
            "composition, 0 would deadlock (arms the controller, the publisher "
            "refuses the declaration), 0 armable but unarmed (port in place, no "
            "shipped composition arms it), 0 blocked, 0 not stateful",
        ),
        (
            3,
            "3 registered routes (3 stateful); 3 armed today by a shipped "
            "composition, 0 would deadlock (arms the controller, the publisher "
            "refuses the declaration), 0 armable but unarmed (port in place, no "
            "shipped composition arms it), 0 blocked, 0 not stateful",
        ),
    ],
    ids=["none", "one", "several"],
)
def test_the_verdict_line_reads_correctly_at_every_count(
    count: int, expected: str
) -> None:
    """Singular and plural, at 0, 1 and n.

    "1 routes armed" is the shape of error that makes a reader distrust the
    whole line, and the counts here move whenever a route arms, so the sentence
    has to survive its own arithmetic.
    """

    sys.path.insert(0, str(REPOSITORY))
    from scripts.ci_step_zero_arming_gate import population_summary

    armed_row = ("`x`", "yes", "yes", "yes", "yes", "yes — somewhere", "nothing — armed")
    assert population_summary([armed_row] * count) == expected


def test_a_wrong_schema_constant_in_the_pin_reddens(repository: pathlib.Path) -> None:
    """The pin records the schema the built registry used.

    If it disagrees with the constant this gate evaluates the predicate
    against, one of the two moved and the table's arming answers were computed
    against a schema the controller does not accept.
    """

    pin = json.loads((repository / PIN).read_text(encoding="utf-8"))
    pin["eval_examples_schema"] = "rwkv-lab.eval-examples.v9"
    (repository / PIN).write_text(json.dumps(pin, indent=2), encoding="utf-8")

    result = _run(repository)
    assert result.returncode == 1, result.stdout + result.stderr
    assert "eval_examples_schema" in result.stdout, result.stdout


@pytest.mark.parametrize(
    ("field", "value"),
    [("immutability", "immutable"), ("fingerprint", "sha256")],
    ids=["immutability", "fingerprint"],
)
def test_a_publisher_condition_reddens_the_gate_as_a_deadlock(
    repository: pathlib.Path, field: str, value: str
) -> None:
    """The two conditions the predicate does not check, and their own verdict.

    `EvalExamplesPublisher.__init__` refuses a declaration whose `immutability`
    is not `append_only` or whose `fingerprint` is not `manifest_sha256`, over
    the same object the arming predicate reads. A composition satisfying only
    the predicate arms the controller and then cannot publish -- the route
    deadlocks at its first `pre_optimizer_step` crossing with no diagnostic,
    which is worse than never arming.

    Both mutated values are the enum defaults in `trainvm/include/trainvm/model.hpp`
    (`Immutability immutability{}`, `Fingerprint fingerprint{}`), so this is the
    state a composition reaches by *omitting* the field, not an exotic one.

    The assertion is not merely "red". A gate that folded this into "blocked" or
    into "no shipped composition arms it" would be red here and still misleading,
    since the port exists and a composition does publish through it, so the cell
    text and the summary population are asserted too.
    """

    # The populations this mutation is expected to move, measured on the
    # unmutated tree rather than restated. Hardcoding them made this test fail
    # the day an unrelated family armed, which is the rot `registry_parity_note`
    # and card-ac588eda are about: a test that restates a count is a second
    # authority for it.
    baseline = _summary(_run(repository).stdout)
    armed_before = _population(baseline, "armed today by a shipped composition")
    blocked_before = _population(baseline, "blocked")

    catalog = _catalog(repository)
    broken = 0
    for recipe in catalog["recipes"]:
        artifacts = recipe["template_document"]["spec"]["artifacts"]
        if "eval_examples" not in artifacts:
            continue
        assert artifacts["eval_examples"][field] != value, (
            f"the mutation is a no-op: {field} already is {value!r}, so a red "
            "verdict below would prove nothing"
        )
        artifacts["eval_examples"][field] = value
        broken += 1
    # Every arming recipe in the catalog, so the route as a whole reaches the
    # deadlock state. Breaking one of two would leave it armed by its sibling,
    # which is a different population and is pinned separately below.
    assert broken > 1, (
        f"{CATALOG} no longer holds two arming recipes, so this test cannot "
        "reach the whole-route deadlock state it exists for"
    )
    _write(repository, catalog)

    result = _run(repository)
    assert result.returncode == 1, (
        f"mutating {field} left the gate green, so it checks only the three "
        f"arming conjuncts:\n{result.stdout}{result.stderr}"
    )
    assert "the publisher" in result.stdout and field in result.stdout, (
        f"the gate failed without naming the publisher condition that broke:\n"
        f"{result.stdout}"
    )
    summary = _summary(result.stdout)
    assert "1 would deadlock" in summary, (
        "the deadlock population is not counted separately, so a reader cannot "
        f"tell it from a route that arms nothing:\n{result.stdout}"
    )
    assert (
        f"{armed_before - 1} armed today by a shipped composition" in summary
    ), summary
    assert f"{blocked_before} blocked" in summary, (
        "a deadlocking route was counted as blocked; blocked means the port is "
        f"missing or unusable and nothing arms:\n{summary}"
    )
    assert CONTRACT in result.stdout, result.stdout


def test_a_sibling_that_still_arms_is_not_a_deadlocked_route(
    repository: pathlib.Path,
) -> None:
    """One recipe deadlocking while its sibling arms is neither state cleanly.

    The route does arm, through the sibling, so counting it as a deadlock would
    overstate the harm; saying only "armed" would let the broken recipe hide
    behind the neighbour that kept its declaration -- the failure
    `test_a_sibling_recipe_cannot_hide_a_broken_one` exists for, reached through
    a publisher condition rather than a conjunct.
    """

    catalog = _catalog(repository)
    _recipe(catalog, RECIPE)["template_document"]["spec"]["artifacts"][
        "eval_examples"
    ]["immutability"] = "immutable"
    _write(repository, catalog)

    result = _run(repository)
    assert result.returncode == 1, result.stdout + result.stderr
    summary = _summary(result.stdout)
    assert "0 would deadlock" in summary, summary
    assert "1 of those armed in some compositions and not others" in summary, summary
    assert "the publisher" in result.stdout, result.stdout


def test_the_gate_reads_the_publisher_conditions_from_the_publisher(
    repository: pathlib.Path,
) -> None:
    """Not restated here, and not restated in the gate either.

    The gate recovers `{key: value}` from `EvalExamplesPublisher`'s own
    rejection condition. This drives that extractor over the real source and
    requires the four conditions it states, so a parse that silently matched
    nothing -- which would present as "the publisher demands nothing" and check
    three conditions again -- fails here rather than in production.
    """

    sys.path.insert(0, str(REPOSITORY))
    from scripts.ci_step_zero_arming_gate import (
        EVAL_EXAMPLES_SCHEMA,
        EVAL_EXAMPLES_TYPE,
        publisher_requirements,
    )

    requirements, problems = publisher_requirements(REPOSITORY)
    assert not problems, problems
    assert requirements == {
        "type": EVAL_EXAMPLES_TYPE,
        "schema": EVAL_EXAMPLES_SCHEMA,
        "immutability": "append_only",
        "fingerprint": "manifest_sha256",
    }, requirements


def test_a_publisher_that_moved_off_the_predicate_schema_reddens(
    repository: pathlib.Path,
) -> None:
    """What forces the two authorities to agree.

    The three conjuncts are C++ and the two extra conditions are Python. The
    only overlap is `type` and `schema`, which both sides state, so that overlap
    is what a disagreement can be detected on: if the publisher stops accepting
    the schema the predicate arms on, every arming answer in the table describes
    a declaration the worker would refuse.

    It doubles as the extractor's own validity check -- a read that matched the
    wrong node returns a map without these keys and fails here.

    Asserted on the *message*, not on the exit code, and that is load-bearing:
    with the cross-check removed the gate is still red, because every shipped
    declaration then reads as a deadlock and the table drifts. A test that
    accepted any red would pass with the cross-check deleted -- it was observed
    surviving exactly that mutation -- while a reader was told eleven routes had
    started deadlocking rather than that the two authorities disagree.
    """

    source = (repository / PUBLISHER).read_text(encoding="utf-8")
    mutated = source.replace(
        'EVAL_EXAMPLES_SCHEMA = "rwkv-lab.eval-examples.v1"',
        'EVAL_EXAMPLES_SCHEMA = "rwkv-lab.eval-examples.v9"',
    )
    assert mutated != source, "the publisher no longer states the schema constant"
    (repository / PUBLISHER).write_text(mutated, encoding="utf-8")

    result = _run(repository)
    assert result.returncode == 1, result.stdout + result.stderr
    assert "demands schema" in result.stdout, (
        "the gate went red without saying the publisher and the predicate "
        f"disagree about the schema:\n{result.stdout}"
    )
    assert PUBLISHER in result.stdout, result.stdout


def test_a_missing_publisher_source_is_a_failure_not_a_skip(
    repository: pathlib.Path,
) -> None:
    """A gate that cannot read its input must say so, not check less.

    Silently falling back to the three conjuncts is the exact defect this change
    removes, and an unreadable file is the cheapest way back into it.
    """

    (repository / PUBLISHER).unlink()

    result = _run(repository)
    assert result.returncode == 1, result.stdout + result.stderr
    assert PUBLISHER in result.stdout, result.stdout
