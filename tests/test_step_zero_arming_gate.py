"""The step-zero arming gate must go red for each conjunct independently.

`scripts/ci_step_zero_arming_gate.py` asserts a three-conjunct property --
`required` and `type` and `schema` -- of every eval-examples publication a
shipped composition declares. A check that tests one conjunct while asserting
three passes every mutation of the other two, and from the outside it is
indistinguishable from a correct one: it is green today, green after the
mutation, and green when a real edit breaks arming.

So each conjunct is mutated separately here and the gate must fail for each.
That is the only evidence that the conjunction is real.

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
import shutil
import subprocess
import sys

import pytest

REPOSITORY = pathlib.Path(__file__).resolve().parent.parent
GATE = "scripts/ci_step_zero_arming_gate.py"
CATALOG = "docs/experiment-vm/examples/rwkv-lm.recipe-profiles.v1.json"
PIN = "docs/experiment-vm/step-zero-arming.v1.json"
DOCUMENT = "docs/experiment-vm/STEP_ZERO_ARMING.md"

# The recipe whose declaration is mutated, and the contract it arms. Named
# rather than indexed so a reordered catalog fails loudly instead of quietly
# mutating a different recipe.
RECIPE = "rwkv_lm_scratch"
CONTRACT = "rwkv_lab.rwkv_scratch.v1.Train"


def _copy(tmp_path: pathlib.Path) -> pathlib.Path:
    """A minimal copy of the repository: the gate, its inputs, and the package.

    Copying the whole checkout would drag in build trees measured in gigabytes.
    """
    root = tmp_path / "repository"
    (root / "scripts").mkdir(parents=True)
    (root / "docs/experiment-vm").mkdir(parents=True)
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
