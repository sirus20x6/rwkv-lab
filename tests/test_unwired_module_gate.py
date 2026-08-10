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
import re
import subprocess
import sys

import pytest

from scripts.ci_unwired_module_gate import (
    API_VERSION,
    AUTHORITY,
    DECLARATION,
    HEADER_ROOT,
    allowed,
    declaration_problems,
    population_summary,
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


# --- the words the verdict line prints ------------------------------------
#
# These assert the MESSAGE, never the exit code. The defect they pin is a
# correct computation described loosely: the gate was green before the wording
# was fixed and is green after, so an exit-code assertion passes against both
# spellings and proves nothing.


def _verdict(stdout: str) -> str:
    lines = [line for line in stdout.splitlines()
             if line.startswith("unwired module gate:")]
    assert len(lines) == 1, f"expected exactly one verdict line:\n{stdout}"
    return lines[0]


def test_the_verdict_line_counts_unexplained_modules_not_all_unwired(declaration):
    """The number a reader acts on is the unexplained one, and it is zero.

    The line used to read `{len(unwired)} unwired, {len(reasons)} explained
    exclusions` -- every unwired header beside the allowlist that explains them,
    which `staleness()` forces to be a subset. Today those are the same five
    modules twice, reading as a five-module backlog standing beside five
    excuses. The unexplained count never appeared.

    The expected counts are read out of the gate's own analysis rather than
    hardcoded, so wiring a module or adding an allowlist entry moves this test
    with the change instead of reddening it.
    """
    result = subprocess.run(
        [sys.executable, "scripts/ci_unwired_module_gate.py"],
        capture_output=True, text=True, cwd=REPOSITORY, check=False,
    )
    verdict = _verdict(result.stdout)

    reasons, _ = allowed(declaration)
    unwired, tally = unwired_modules(REPOSITORY)
    undeclared = [header for header in unwired if header not in reasons]

    assert "explained exclusions" not in verdict, (
        "the verdict counts every unwired header beside the allowlist that "
        f"explains it, which reads as a backlog of unfixed modules:\n{verdict}")
    assert f"{len(undeclared)} unwired and unexplained" in verdict, verdict
    assert (f"{len(unwired) - len(undeclared)} unwired but explained in "
            f"{DECLARATION}") in verdict, verdict
    assert f"{tally['headers'] - len(unwired)} wired" in verdict, verdict


def test_the_verdict_lines_header_counts_sum_to_the_total():
    """A reader must be able to check the arithmetic, so it has to hold.

    Parsed out of the printed sentence rather than recomputed, because the
    claim under test is about the sentence.
    """
    result = subprocess.run(
        [sys.executable, "scripts/ci_unwired_module_gate.py"],
        capture_output=True, text=True, cwd=REPOSITORY, check=False,
    )
    verdict = _verdict(result.stdout)
    matched = re.search(
        r"(\d+) headers? over .*?; (\d+) unwired and unexplained, "
        r"(\d+) unwired but explained in \S+, (\d+) wired", verdict)
    assert matched is not None, verdict
    total, unexplained, explained, wired = (int(g) for g in matched.groups())
    assert unexplained + explained + wired == total, verdict


def test_the_verdict_line_separates_populations_that_were_one_number():
    """The regression proof: this sentence would have caught the original slip.

    Two unwired headers, one of them explained. The old sentence said
    "2 unwired, 1 explained exclusions" -- two populations of different sizes,
    neither of them the one a reader wants. The fixed sentence names the
    unexplained module as its own count, and it is not the unwired total.
    """
    summary = population_summary(
        {"headers": 3, "production": 2, "tests": 1},
        unwired=["trainvm/a.hpp", "trainvm/b.hpp"],
        undeclared=["trainvm/a.hpp"],
        reasons={"trainvm/b.hpp": "a stated reason for leaving this unwired"},
        stale=0,
    )
    assert "1 unwired and unexplained" in summary, summary
    assert f"1 unwired but explained in {DECLARATION}" in summary, summary
    assert "1 wired" in summary, summary
    assert "2 unwired" not in summary, summary


def test_a_stale_allowlist_entry_is_reported_as_its_own_population():
    """The allowlist is a countdown, so an entry that stopped applying counts."""
    summary = population_summary(
        {"headers": 1, "production": 1, "tests": 0},
        unwired=[],
        undeclared=[],
        reasons={"trainvm/gone.hpp": "a stated reason that no longer holds"},
        stale=1,
    )
    assert "1 allowlist entry, 1 no longer applicable" in summary, summary


@pytest.mark.parametrize(
    ("headers", "production", "tests", "expected"),
    [
        (
            0, 0, 0,
            "0 headers over 0 production and 0 test translation units; "
            "0 unwired and unexplained, 0 unwired but explained in "
            f"{DECLARATION}, 0 wired; 0 allowlist entries, 0 no longer applicable",
        ),
        (
            1, 1, 0,
            "1 header over 1 production and 0 test translation units; "
            "0 unwired and unexplained, 0 unwired but explained in "
            f"{DECLARATION}, 1 wired; 0 allowlist entries, 0 no longer applicable",
        ),
        (
            4, 2, 2,
            "4 headers over 2 production and 2 test translation units; "
            "0 unwired and unexplained, 0 unwired but explained in "
            f"{DECLARATION}, 4 wired; 0 allowlist entries, 0 no longer applicable",
        ),
    ],
    ids=["none", "one", "several"],
)
def test_the_verdict_line_reads_correctly_at_every_count(
    headers, production, tests, expected
):
    """Singular and plural, at 0, 1 and n.

    "1 headers" is the shape of error that makes a reader distrust the whole
    line, and every count here moves as the tree grows.
    """
    assert population_summary(
        {"headers": headers, "production": production, "tests": tests},
        unwired=[], undeclared=[], reasons={}, stale=0,
    ) == expected
