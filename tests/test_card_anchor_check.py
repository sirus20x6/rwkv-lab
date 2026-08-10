"""The anchor checker must not be able to report a card as healthy silently.

`scripts/card_anchor_check.py` exists because a card's "measured, do not
re-derive" block has no expiry. The way a tool like it fails is not by being
wrong -- it is by matching nothing and printing a clean line, which reads
exactly like a corpus with no problems in it. Most of what is asserted here is
that separation: a card with no recognisable anchors and a card whose anchors
all hold must produce visibly different output, and the summary must carry the
recognition count so a parse that stopped matching is visible in the one line a
reader sees.

Everything runs against a git repository built in `tmp_path`, so no assertion
here depends on what `origin/main` says today. The proof that it fires on the
real stale cards is in the pull request, against the real board; a test pinned
to live card bodies would go red for reasons that are not about this code.
"""

from __future__ import annotations

import json
import pathlib
import subprocess
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from scripts.card_anchor_check import (  # noqa: E402
    NEVER_ANYWHERE,
    locate_missing,
)

REPOSITORY = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = REPOSITORY / "scripts" / "card_anchor_check.py"


def git(repository: pathlib.Path, *args: str, when: str | None = None) -> None:
    env = None
    if when is not None:
        import os
        env = dict(os.environ, GIT_AUTHOR_DATE=when, GIT_COMMITTER_DATE=when)
    subprocess.run(["git", "-C", str(repository), *args], check=True,
                   capture_output=True, text=True, env=env)


@pytest.fixture
def tree(tmp_path):
    """A small repository with a `main` branch and one committed source file."""
    repository = tmp_path / "tree"
    repository.mkdir()
    git(repository, "init", "-q", "-b", "main")
    git(repository, "config", "user.email", "t@example.invalid")
    git(repository, "config", "user.name", "test")
    (repository / "src").mkdir()
    (repository / "src" / "ledger.cpp").write_text(
        "\n".join(f"line {n}" for n in range(1, 41)) + "\n", encoding="utf-8")
    (repository / ".gitignore").write_text("build/\n", encoding="utf-8")
    git(repository, "add", "-A")
    git(repository, "commit", "-qm", "initial", when="2026-01-01T00:00:00Z")
    return repository


def run(tree, cards, *extra):
    board = tree.parent / "board.json"
    board.write_text(json.dumps({"cards": cards}), encoding="utf-8")
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--board", str(board),
         "--repository", str(tree), "--rev", "main", *extra],
        capture_output=True, text=True, check=False, cwd=REPOSITORY)


def card(card_id, body, created_at="2026-06-01T00:00:00Z", title="a card"):
    return {"id": card_id, "title": title, "body": body,
            "created_at": created_at}


def test_a_cited_path_that_does_not_exist_is_reported_and_fails(tree):
    completed = run(tree, [card("card-gone", "See `src/vanished.cpp` for it.")])
    assert "MISSING" in completed.stdout
    assert "src/vanished.cpp" in completed.stdout
    assert "STALE" in completed.stdout
    assert completed.returncode == 1
    assert "FAILED" in completed.stdout.strip().splitlines()[-1]


def test_a_card_whose_anchors_all_hold_passes(tree):
    completed = run(tree, [card("card-ok", "Look at `src/ledger.cpp:12`.")])
    assert "ANCHORS HOLD" in completed.stdout
    assert completed.returncode == 0
    assert "PASSED" in completed.stdout.strip().splitlines()[-1]


def test_no_anchors_and_all_anchors_holding_are_different_reports(tree):
    """The failure this tool would otherwise have: silence read as health.

    A parse that recognises nothing in a card body must not produce the report
    a card whose every anchor resolved produces. Both exit zero, so the exit
    code cannot carry this distinction -- the text has to.
    """
    nothing = run(tree, [card("card-prose", "This card names no files at all.")])
    everything = run(tree, [card("card-ok", "See `src/ledger.cpp` for it.")])

    assert nothing.returncode == everything.returncode == 0
    assert "NO ANCHORS FOUND" in nothing.stdout
    assert "NO ANCHORS" not in everything.stdout
    assert "ANCHORS HOLD" in everything.stdout
    assert "ANCHORS HOLD" not in nothing.stdout


def test_the_summary_states_how_many_cards_could_not_be_checked(tree):
    """The recognition rate belongs in the line most likely to be read alone.

    Without it, a regex that stopped matching prints the same PASSED as a clean
    corpus. With it, "2 cards had none" is on the same line as the verdict.
    """
    completed = run(tree, [
        card("card-prose", "No files here."),
        card("card-prose-two", "Nor here."),
        card("card-ok", "See `src/ledger.cpp`."),
    ])
    last = completed.stdout.strip().splitlines()[-1]
    assert "checked 1 anchors across 1 of 3 cards" in last
    assert "2 cards had none" in last


def test_a_parse_that_recognised_nothing_would_fail_this(tree):
    """A direct bite on the regex, independent of any verdict wording.

    If `ANCHOR` stops matching, every other assertion here still passes on the
    cards that expect a clean result, and only this one goes red. It is the
    control for the whole file.
    """
    sys.path.insert(0, str(REPOSITORY))
    from scripts.card_anchor_check import parse_anchors

    body = ("`trainvm/include/trainvm/host_ledger.hpp:102` and "
            "`host_ledger.hpp:79` and plain scripts/run.sh and "
            "`src/rwkv_lab/qwen_caption_finetune.py`")
    cited = {a.cited for a in parse_anchors(body)}
    assert cited == {
        "trainvm/include/trainvm/host_ledger.hpp:102",
        "host_ledger.hpp:79",
        "scripts/run.sh",
        "src/rwkv_lab/qwen_caption_finetune.py",
    }


def test_a_line_past_the_end_of_the_file_is_reported(tree):
    completed = run(tree, [card("card-eof", "See `src/ledger.cpp:900`.")])
    assert "PAST EOF" in completed.stdout
    assert completed.returncode == 1


def test_a_path_that_changed_after_the_card_was_written_is_reported(tree):
    """Drift is reported and does not fail: a commit may not touch the claim.

    This is the half that would have caught card-74b81095, whose cited paths
    all still existed -- PR #201 had rewritten them.
    """
    (tree / "src" / "ledger.cpp").write_text("changed\n", encoding="utf-8")
    git(tree, "commit", "-qam", "rewrite the ledger",
        when="2026-07-01T00:00:00Z")

    completed = run(tree, [card("card-drift", "See `src/ledger.cpp`.",
                                created_at="2026-06-01T00:00:00Z")])
    assert "DRIFTED" in completed.stdout
    assert "rewrite the ledger" in completed.stdout
    assert completed.returncode == 0


def test_a_path_that_changed_before_the_card_was_written_is_not_drift(tree):
    """The other direction, so the drift check is not simply always on."""
    (tree / "src" / "ledger.cpp").write_text("changed\n", encoding="utf-8")
    git(tree, "commit", "-qam", "rewrite the ledger",
        when="2026-02-01T00:00:00Z")

    completed = run(tree, [card("card-fresh", "See `src/ledger.cpp`.",
                                created_at="2026-06-01T00:00:00Z")])
    assert "DRIFTED" not in completed.stdout
    assert "ANCHORS HOLD" in completed.stdout


def test_an_ambiguous_basename_is_unresolved_rather_than_missing(tree):
    """Two files share the name, so the card named neither.

    Reporting this as a missing path would be false, and picking one and
    checking it would answer a question the card did not ask. It is counted as
    unresolvable in the summary so the gap is visible rather than absorbed.
    """
    (tree / "other").mkdir()
    (tree / "other" / "ledger.cpp").write_text("x\n", encoding="utf-8")
    git(tree, "add", "-A")
    git(tree, "commit", "-qm", "second ledger", when="2026-01-02T00:00:00Z")

    completed = run(tree, [card("card-ambiguous", "See `ledger.cpp`.")])
    assert "unresolved" in completed.stdout
    assert "MISSING" not in completed.stdout
    assert "NOTHING CHECKABLE" in completed.stdout
    assert "1 anchors were unresolvable" in completed.stdout
    assert completed.returncode == 0


def test_a_template_path_is_skipped_rather_than_reported_missing(tree):
    completed = run(tree, [card("card-template",
                                "written to `eval/step_XXXXXXXX.json`")])
    assert "template, not a path" in completed.stdout
    assert "MISSING" not in completed.stdout
    assert completed.returncode == 0


def test_a_gitignored_artifact_is_skipped_rather_than_reported_missing(tree):
    completed = run(tree, [card("card-artifact", "reads `build/out.json`")])
    assert "gitignored" in completed.stdout
    assert "MISSING" not in completed.stdout
    assert completed.returncode == 0


def test_an_unknown_card_id_refuses_rather_than_checking_the_remainder(tree):
    """Silently checking fewer cards than asked is the same defect one level up."""
    completed = run(tree, [card("card-ok", "See `src/ledger.cpp`.")],
                    "--card", "card-not-here")
    assert completed.returncode != 0
    assert "card-not-here" in completed.stderr


def test_selecting_one_card_checks_only_that_card(tree):
    completed = run(tree, [card("card-ok", "See `src/ledger.cpp`."),
                           card("card-gone", "See `src/vanished.cpp`.")],
                    "--card", "card-ok")
    assert "card-gone" not in completed.stdout
    assert completed.returncode == 0


# --- locate_missing: where does an absent path actually live? --------------


def _repo(tmp_path):
    repo = tmp_path / "r"
    repo.mkdir()
    git(repo, "init", "-q", "-b", "main")
    git(repo, "config", "user.email", "t@example.com")
    git(repo, "config", "user.name", "t")
    (repo / "kept.py").write_text("x = 1\n")
    git(repo, "add", "kept.py")
    git(repo, "commit", "-qm", "base")
    return repo


def test_a_path_that_exists_nowhere_is_named_as_such(tmp_path):
    repo = _repo(tmp_path)
    assert locate_missing(repo, "no/such/file.py", "main") == NEVER_ANYWHERE


def test_a_path_only_on_a_branch_is_reported_as_branch_only(tmp_path):
    """The verdict that misleads: the card reads as trunk, the work is not."""
    repo = _repo(tmp_path)
    git(repo, "checkout", "-q", "-b", "side")
    (repo / "sideways.py").write_text("y = 2\n")
    git(repo, "add", "sideways.py")
    git(repo, "commit", "-qm", "add sideways")
    git(repo, "checkout", "-q", "main")

    note = locate_missing(repo, "sideways.py", "main")
    assert note.startswith("on a branch, never on main")
    assert "add sideways" in note


def test_a_deleted_path_is_reported_as_having_been_on_the_revision(tmp_path):
    """`git log <rev> -- <path>` finds a deleted path; a tree test does not.

    Without this case a renamed file reads as 'never landed', which is the
    absence trap arriving through a filename instead of a sha.
    """
    repo = _repo(tmp_path)
    (repo / "gone.py").write_text("z = 3\n")
    git(repo, "add", "gone.py")
    git(repo, "commit", "-qm", "add gone")
    git(repo, "rm", "-q", "gone.py")
    git(repo, "commit", "-qm", "remove gone")

    note = locate_missing(repo, "gone.py", "main")
    assert note.startswith("was on main and is not now")


def test_the_revision_is_checked_before_any_branch(tmp_path):
    """A path satisfying BOTH queries must report the revision's answer.

    Deleted from main and still alive on a branch: `git log main -- p` finds
    the deletion and `git log --all -- p` finds the branch. The revision wins,
    so the card is told its work landed and was renamed, not that it never
    landed.

    This case does NOT catch an inverted lookup order — under inversion it
    returns the same verdict. Mutation testing established that; the test that
    catches inversion is the branch-only one above, which flips to was-on-main.
    Both are kept because they fail to different mutations.
    """
    repo = _repo(tmp_path)
    (repo / "both.py").write_text("a = 1\n")
    git(repo, "add", "both.py")
    git(repo, "commit", "-qm", "add both")
    git(repo, "checkout", "-q", "-b", "keeper")
    git(repo, "checkout", "-q", "main")
    git(repo, "rm", "-q", "both.py")
    git(repo, "commit", "-qm", "remove both from main")

    note = locate_missing(repo, "both.py", "main")
    assert note.startswith("was on main and is not now"), note
