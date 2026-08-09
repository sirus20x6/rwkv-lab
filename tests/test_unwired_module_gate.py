"""The unwired-module gate must actually fire, in every direction it claims.

``scripts/ci_unwired_module_gate.py`` fails when a native module is reached by
no translation unit under ``trainvm/src/`` other than its own implementation.
It exists because four such modules — fully implemented, fully tested, green —
shipped in this repository in a single session, and a passing suite never said
anything used them.

A gate for that failure mode has to be held to its own standard. This
repository has shipped an integrity gate that printed PASSED *because* its
check failed, so these tests assert the gate fires rather than assuming it. The
proofs were run by hand once when the gate landed; they live here so they keep
being run, because a one-time demonstration decays into a claim.

The most important test here is ``test_a_module_reached_only_transitively_is_wired``.
The cheap form of this check counts direct includes, and on this tree that form
reports 34 unwired headers where the transitive traversal reports 5 — 29 false
positives, because headers here include other headers (``service.hpp`` includes
13). A gate wrong five times out of six on its first run is one nobody trusts
again, so the traversal is the load-bearing part and it gets the regression
test.
"""

from __future__ import annotations

import json
import pathlib

import pytest

from scripts.ci_unwired_module_gate import (
    API_VERSION,
    AUTHORITY,
    DECLARATION,
    HEADER_ROOT,
    allowed,
    declaration_problems,
    staleness,
    unwired_modules,
)

REPOSITORY = pathlib.Path(__file__).resolve().parent.parent


@pytest.fixture
def declaration() -> dict:
    return json.loads((REPOSITORY / DECLARATION).read_text(encoding="utf-8"))


def build(root: pathlib.Path, headers: dict[str, str],
          production: dict[str, str], tests: dict[str, str]) -> pathlib.Path:
    """Materialise a miniature trainvm tree and return its repository root."""
    for relative, text in (
        *((f"{HEADER_ROOT}/{name}", body) for name, body in headers.items()),
        *((f"trainvm/src/{name}", body) for name, body in production.items()),
        *((f"trainvm/tests/{name}", body) for name, body in tests.items()),
    ):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    (root / "trainvm/benchmarks").mkdir(parents=True, exist_ok=True)
    return root


def include(header: str) -> str:
    return f'#include "{header}"\n'


# --- the checked-in state -------------------------------------------------


def test_the_checked_in_declaration_explains_every_unwired_module(declaration):
    """The gate is green on main, and green for the stated reasons."""
    reasons, problems = allowed(declaration)
    assert declaration_problems(declaration) == []
    assert problems == []

    unwired, _ = unwired_modules(REPOSITORY)
    assert sorted(unwired) == sorted(reasons), (
        "the allowlist and the unwired set have diverged; run "
        "python scripts/ci_unwired_module_gate.py"
    )
    now_wired, missing = staleness(reasons, unwired, REPOSITORY / HEADER_ROOT)
    assert now_wired == []
    assert missing == []


def test_authority_is_evidence_only(declaration):
    """The file records scope. It must not read as granting any."""
    assert declaration["authority"] == AUTHORITY
    assert declaration["api_version"] == API_VERSION

    declaration["authority"] = "ci_execution"
    assert declaration_problems(declaration) != []


def test_the_known_open_instance_is_still_named(declaration):
    """conversion_publication is the instance that motivated the card.

    If it ever gains a production caller this test does not need to change --
    the staleness check fails first and tells whoever wired it to drop the
    entry. This asserts only that the gate did not lose sight of it.
    """
    reasons, _ = allowed(declaration)
    assert "trainvm/conversion_publication.hpp" in reasons


# --- the traversal --------------------------------------------------------


def test_a_module_with_no_production_includer_is_reported(tmp_path):
    root = build(
        tmp_path,
        headers={"trainvm/lonely.hpp": "#pragma once\n"},
        production={"lonely.cpp": include("trainvm/lonely.hpp")},
        tests={"lonely_tests.cpp": include("trainvm/lonely.hpp")},
    )
    unwired, tally = unwired_modules(root)
    assert unwired == ["trainvm/lonely.hpp"]
    assert tally == {"headers": 1, "production": 1, "tests": 1}


def test_a_module_reached_only_transitively_is_wired(tmp_path):
    """The false positive that makes a direct-include count unusable here.

    ``deep.hpp`` is included by no .cpp at all. It is reached only through
    ``middle.hpp``, which a production translation unit includes. A gate that
    counted direct includes would report it, and would have reported 29 such
    headers on this tree.
    """
    root = build(
        tmp_path,
        headers={
            "trainvm/deep.hpp": "#pragma once\n",
            "trainvm/middle.hpp": "#pragma once\n" + include("trainvm/deep.hpp"),
        },
        production={"caller.cpp": include("trainvm/middle.hpp")},
        tests={},
    )
    unwired, _ = unwired_modules(root)
    assert unwired == []


def test_a_modules_own_source_is_not_its_includer(tmp_path):
    """Otherwise nothing is ever unwired and the gate is decorative."""
    root = build(
        tmp_path,
        headers={"trainvm/solo.hpp": "#pragma once\n"},
        production={"solo.cpp": include("trainvm/solo.hpp")},
        tests={},
    )
    unwired, _ = unwired_modules(root)
    assert unwired == ["trainvm/solo.hpp"]


def test_a_test_only_includer_does_not_count_as_production(tmp_path):
    """The entire point: a suite exercising a module is not a caller of it."""
    root = build(
        tmp_path,
        headers={"trainvm/tested.hpp": "#pragma once\n"},
        production={},
        tests={"tested_tests.cpp": include("trainvm/tested.hpp")},
    )
    unwired, _ = unwired_modules(root)
    assert unwired == ["trainvm/tested.hpp"]


def test_a_production_includer_clears_the_module(tmp_path):
    root = build(
        tmp_path,
        headers={"trainvm/used.hpp": "#pragma once\n"},
        production={
            "used.cpp": include("trainvm/used.hpp"),
            "consumer.cpp": include("trainvm/used.hpp"),
        },
        tests={},
    )
    unwired, _ = unwired_modules(root)
    assert unwired == []


def test_a_system_include_cannot_wire_a_module(tmp_path):
    """Angle-bracket includes name no module in this tree and must not match."""
    root = build(
        tmp_path,
        headers={"trainvm/quiet.hpp": "#pragma once\n"},
        production={
            "quiet.cpp": include("trainvm/quiet.hpp"),
            "elsewhere.cpp": "#include <trainvm/quiet.hpp>\n",
        },
        tests={},
    )
    unwired, _ = unwired_modules(root)
    assert unwired == ["trainvm/quiet.hpp"]


# --- the allowlist and its teeth ------------------------------------------


def test_an_exclusion_without_a_stated_reason_fails():
    """A reason short enough to be a label is not a reason."""
    _, problems = allowed({
        "exclusions": [{"module": "trainvm/x.hpp", "reason": "later"}],
    })
    assert problems == ["trainvm/x.hpp: reason must be a stated sentence"]


def test_an_empty_allowlist_fails():
    _, problems = allowed({"exclusions": []})
    assert problems == ["exclusions must be a non-empty list"]


def test_an_entry_whose_module_became_wired_fails(tmp_path):
    """The countdown. An excuse that stopped being true must be removed."""
    header_root = tmp_path / HEADER_ROOT
    (header_root / "trainvm").mkdir(parents=True)
    (header_root / "trainvm/wired.hpp").write_text("#pragma once\n")

    now_wired, missing = staleness(
        {"trainvm/wired.hpp": "a stated reason that no longer holds"},
        unwired=[],
        header_root=header_root,
    )
    assert now_wired == ["trainvm/wired.hpp"]
    assert missing == []


def test_an_entry_whose_module_no_longer_exists_fails(tmp_path):
    now_wired, missing = staleness(
        {"trainvm/deleted.hpp": "a stated reason for a module that is gone"},
        unwired=[],
        header_root=tmp_path / HEADER_ROOT,
    )
    assert now_wired == []
    assert missing == ["trainvm/deleted.hpp"]
