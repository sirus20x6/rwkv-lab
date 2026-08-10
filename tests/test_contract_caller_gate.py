"""The contract caller gate must go red for each of its checks independently.

`scripts/ci_contract_caller_gate.py` asserts that every adapter route the
native registry advertises reaches production code: the route dispatches to a
handler, the handler is reachable from the worker's entrypoint, and the modules
implementing the contract are reachable from it too. All three are zero-defect
today, which is the state in which a gate is least trustworthy -- it is green
whether or not it measures anything. So each check is broken separately here
against a copy of the real tree, and the gate must fail for each.

The mutations are real refactors, not synthetic ones:

- deleting a route from `_HANDLERS` is what happens when a handler is retired
  and the native registry is not told;
- replacing an implementation module's import with a local stand-in is what a
  half-finished extraction leaves behind, and it is the Python form of the
  `conversion_publication` defect the native gate was written for;
- routing dispatch through an empty table is what a rewritten `execute_invocation`
  would do if it stopped consulting `_HANDLERS`.

`test_a_new_unreferenced_symbol_is_not_a_finding` is a control, and it is
designed to catch nothing rather than observed to: the gate's reachability is
over imports and name references, so a symbol added to an already-imported
module cannot change any edge it reads. It pins the instrument's floor -- this
gate does not see symbol-level rot, which is candidate 3 on card-d198cc09 and a
different check -- so that a later reader does not mistake its silence there
for a claim.

The gate is run as a subprocess against a copied repository rather than
imported, because its verdict is an exit code and a printed line and those are
what CI reads.
"""

from __future__ import annotations

import pathlib
import re
import shutil
import subprocess
import sys

import pytest

REPOSITORY = pathlib.Path(__file__).resolve().parent.parent
GATE = "scripts/ci_contract_caller_gate.py"
PIN = "docs/experiment-vm/step-zero-arming.v1.json"
HANDLERS = "src/rwkv_lab/trainvm_adapters/handlers.py"

# The route deleted to break the registry/dispatch correspondence. Named rather
# than indexed so a reordered table fails loudly instead of quietly mutating a
# different route.
RETIRED_ROUTE = '''    (
        "rwkv-lab.rwkv-rlvr",
        "1.0.0",
        "train",
        "rwkv_lab.rwkv_rlvr.v1.Train",
    ): _rlvr,
'''
RLVR_IMPORT = "from .rlvr import RLVRHeldoutEvalPolicy, RLVRTrainConfig\n"
DISPATCH = "        handler = _HANDLERS[key]\n"


def _copy(tmp_path: pathlib.Path) -> pathlib.Path:
    """A minimal copy: the gate, its inputs, and the library it reads.

    Copying the whole checkout would drag in build trees measured in gigabytes.
    """
    root = tmp_path / "repository"
    (root / "scripts").mkdir(parents=True)
    (root / "docs/experiment-vm").mkdir(parents=True)
    for name in (GATE, "scripts/gate_verdict.py"):
        shutil.copy2(REPOSITORY / name, root / name)
    (root / "scripts/__init__.py").touch()
    shutil.copy2(REPOSITORY / PIN, root / PIN)
    shutil.copy2(REPOSITORY / "pyproject.toml", root / "pyproject.toml")
    shutil.copytree(REPOSITORY / "src/rwkv_lab", root / "src/rwkv_lab")
    return root


def _run(root: pathlib.Path) -> subprocess.CompletedProcess[str]:
    # No PYTHONPATH: this gate reads source, it never imports the package, and
    # that is what lets it run in the schema job without torch. If it ever
    # gained an import, this call would find the copy's own tree anyway.
    return subprocess.run(
        [sys.executable, str(root / GATE), "--repository", str(root)],
        capture_output=True,
        text=True,
        check=False,
    )


def _edit(root: pathlib.Path, relative: str, before: str, after: str) -> None:
    """Replace `before` with `after`, refusing a mutation that did not apply.

    A mutation that silently fails to apply reads as a passing gate, which is
    indistinguishable from a gate that does not bite.
    """
    path = root / relative
    text = path.read_text(encoding="utf-8")
    assert text.count(before) == 1, f"{relative} does not contain the text to mutate"
    path.write_text(text.replace(before, after), encoding="utf-8")


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
    assert "contract caller gate: PASSED" in result.stdout


def test_a_retired_route_the_registry_still_advertises_reddens_the_gate(
    repository: pathlib.Path,
) -> None:
    _edit(repository, HANDLERS, RETIRED_ROUTE, "")

    result = _run(repository)

    assert result.returncode == 1, result.stdout + result.stderr
    assert "UNSERVED: the native registry advertises rwkv_lab.rwkv_rlvr.v1.Train" in (
        result.stdout
    )
    assert "contract caller gate: FAILED" in result.stdout


def test_an_implementation_module_losing_its_last_importer_reddens_the_gate(
    repository: pathlib.Path,
) -> None:
    """The `conversion_publication` shape, in Python.

    `handlers.py` holds the only production import of `trainvm_adapters.rlvr`.
    Replacing it with local stand-ins is what a half-finished extraction leaves:
    the module is still there, still tested, and the RLVR route the registry
    advertises no longer reaches a line of it.
    """

    _edit(
        repository,
        HANDLERS,
        RLVR_IMPORT,
        "RLVRHeldoutEvalPolicy = RLVRTrainConfig = object\n",
    )

    result = _run(repository)

    assert result.returncode == 1, result.stdout + result.stderr
    assert (
        "UNWIRED MODULE: rwkv_lab.trainvm_adapters.rlvr implements part of the "
        "advertised contract surface"
    ) in result.stdout


def test_dispatch_that_stops_consulting_the_table_reddens_the_gate(
    repository: pathlib.Path,
) -> None:
    """Every route still advertised, every handler still defined, none callable.

    This is the check the other two cannot make: the registry and `_HANDLERS`
    agree, so check 1 is silent, and every implementation module is still
    imported, so check 3 is silent.
    """

    _edit(repository, HANDLERS, DISPATCH, "        handler = {}[key]\n")

    result = _run(repository)

    assert result.returncode == 1, result.stdout + result.stderr
    assert "UNWIRED HANDLER: rwkv_lab.rwkv_rlvr.v1.Train dispatches to _rlvr" in (
        result.stdout
    )
    # The point is that check 3 stays silent -- zero modules unreached -- not
    # how many modules there are. Restating the total made this test red the
    # day an unrelated adapter module was added, which is the rot the
    # `registry_parity_note` in the arming gate exists to avoid.
    assert re.search(
        r"0 of \d+ implementation modules unreached", result.stdout
    ), result.stdout


def test_an_unknown_entrypoint_refuses_rather_than_reporting_zero(
    repository: pathlib.Path,
) -> None:
    """A gate that cannot find its root must not answer about an empty graph.

    With no root, every reachability question has the same answer, and the
    honest one is "this instrument is not measuring anything".
    """

    _edit(
        repository,
        "pyproject.toml",
        'trainvm-worker = "rwkv_lab.trainvm_adapters.entrypoint:main"',
        'trainvm-worker = "rwkv_lab.trainvm_adapters.entrypoint:absent"',
    )

    result = _run(repository)

    assert result.returncode == 1, result.stdout + result.stderr
    assert "which is not a definition in this tree" in result.stdout
    assert "0 routes analysed" in result.stdout


def test_a_new_unreferenced_symbol_is_not_a_finding(
    repository: pathlib.Path,
) -> None:
    """The control, designed to catch nothing.

    The gate's edges are imports and name references. A function added to a
    module that is already imported adds a node no edge points at and changes
    no edge the gate reads, so it cannot move the verdict -- and the verdict
    line must be identical, not merely still green, or the control would be
    measuring something after all.
    """

    before = _run(repository)
    path = repository / "src/rwkv_lab/trainvm_adapters/rlvr.py"
    path.write_text(
        path.read_text(encoding="utf-8")
        + "\n\ndef unreferenced_by_anything() -> None:\n    return None\n",
        encoding="utf-8",
    )

    after = _run(repository)

    assert after.returncode == 0, after.stdout + after.stderr
    assert after.stdout == before.stdout
