"""The sweep checker's value is entirely in what it refuses.

A checker that answers "reclaimable" when it does not know is worse than no
checker, because a sweep would then delete on its authority. So the cases that
matter are the ones with no marker, a corrupt marker, and a mixed set — not
the happy path.
"""

from __future__ import annotations

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from scripts.worktree_sweep_check import (  # noqa: E402
    MARKER,
    OWNED,
    RECLAIMABLE,
    UNKNOWN,
    classify,
    main,
    read_marker,
    report,
    write_marker,
)


def tree(tmp_path: pathlib.Path, name: str, agent: str | None = None) -> pathlib.Path:
    path = tmp_path / name
    path.mkdir()
    if agent is not None:
        write_marker(path, agent, "2026-08-10T04:00:00Z")
    return path


def test_an_unmarked_worktree_is_unknown_and_never_reclaimable(tmp_path):
    """The property the whole design rests on: silence means refuse.

    This is what makes the checker safe to adopt before anything writes
    markers — every worktree is refused, so the sweep does nothing.
    """
    verdict, detail = classify(tree(tmp_path, "bare"), live={"anyone"})
    assert verdict == UNKNOWN
    assert MARKER in detail


def test_a_corrupt_marker_is_unknown_rather_than_an_exception(tmp_path):
    """A checker that crashes on a bad marker strands the sweep entirely."""
    for payload in ("not json at all", "[]", '{"agent": ""}', '{"agent": 7}', "{}"):
        path = tree(tmp_path, f"t{abs(hash(payload)) % 10**6}")
        (path / MARKER).write_text(payload, encoding="utf-8")
        assert read_marker(path) is None, payload
        assert classify(path, live=set())[0] == UNKNOWN, payload


def test_a_marker_naming_a_live_agent_is_owned(tmp_path):
    verdict, detail = classify(tree(tmp_path, "busy", "reviewer"),
                               live={"reviewer", "builder"})
    assert verdict == OWNED
    assert "reviewer" in detail


def test_a_marker_naming_an_absent_agent_is_reclaimable(tmp_path):
    """Fails safe on a crash: a dead agent leaves the roster, so its tree
    becomes reclaimable without the dead process cleaning anything up."""
    verdict, _ = classify(tree(tmp_path, "stale", "ghost"), live={"reviewer"})
    assert verdict == RECLAIMABLE


def test_an_empty_live_roster_does_not_make_everything_reclaimable(tmp_path):
    """An unmarked tree stays refused even when no agent is live.

    The tempting shortcut — "nobody is running, so sweep everything" — is
    exactly how a worktree holding uncommitted work gets removed.
    """
    assert classify(tree(tmp_path, "bare"), live=set())[0] == UNKNOWN
    assert classify(tree(tmp_path, "marked", "ghost"), live=set())[0] == RECLAIMABLE


def test_the_run_refuses_unless_every_tree_is_reclaimable(tmp_path):
    reclaimable = tree(tmp_path, "a", "ghost")
    owned = tree(tmp_path, "b", "reviewer")
    bare = tree(tmp_path, "c")

    assert main(["--live", "reviewer", str(reclaimable)]) == 0
    assert main(["--live", "reviewer", str(reclaimable), str(owned)]) == 1
    assert main(["--live", "reviewer", str(reclaimable), str(bare)]) == 1


def test_examining_nothing_is_not_success(tmp_path):
    """Zero worktrees must not report PASSED — a sweep gating on the exit
    code would otherwise treat "found nothing" as "everything is safe"."""
    assert main(["--live", "reviewer", "--repository", str(tmp_path)]) == 1
    assert "0 worktrees examined" in report([])[-1]


def test_the_summary_states_the_population_and_the_split(tmp_path):
    rows = [(tree(tmp_path, "a", "ghost"), RECLAIMABLE, ""),
            (tree(tmp_path, "b", "reviewer"), OWNED, "")]
    line = report(rows)[-1]
    assert "2 worktrees examined" in line
    assert "1 reclaimable" in line and "1 owned" in line
    assert "REFUSED" in line


def test_claiming_writes_a_marker_that_reads_back(tmp_path):
    path = tree(tmp_path, "fresh")
    assert main(["--claim", "reviewer", "--claimed-at", "2026-08-10T04:00:00Z",
                 str(path)]) == 0
    assert read_marker(path) == "reviewer"
    payload = json.loads((path / MARKER).read_text(encoding="utf-8"))
    assert payload["claimed_at"] == "2026-08-10T04:00:00Z"


def test_claiming_without_a_target_is_an_error_not_a_silent_success(tmp_path):
    assert main(["--claim", "reviewer"]) == 2
