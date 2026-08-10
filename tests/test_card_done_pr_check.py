"""The done-PR checker must separate a gap from a false claim.

The interesting cases are not "did it find the missing URL". They are the two
ways a recorded URL can be worse than an absent one -- naming something that
is not a pull request, and naming a pull request that never merged -- and the
requirement that a board it recognises nothing in cannot read as a clean bill
of health.
"""

from __future__ import annotations

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from scripts.card_done_pr_check import (  # noqa: E402
    CITES_MERGED,
    MALFORMED,
    NOT_MERGED,
    NO_PR,
    check_card,
    is_done,
    suggest,
    title_overlap,
    title_tokens,
    main,
    pull_request_number,
    report_lines,
)

MERGED_URL = "https://github.com/sirus20x6/rwkv-lab/pull/245"


def done_card(identifier: str, url: str | None = None, title: str = "a card",
              **extra) -> dict:
    card = {"id": identifier, "title": title, "claim_status": "done"}
    if url is not None:
        card["result_pr_url"] = url
    card.update(extra)
    return card


def test_both_spellings_of_done_are_honoured():
    """Neither field is authoritative alone, so both mark a card done."""
    assert is_done({"claim_status": "done"})
    assert is_done({"status": "completed"})
    assert not is_done({"claim_status": "claimed"})
    assert not is_done({})


def test_a_done_card_with_no_url_is_reported_but_does_not_fail_the_run():
    """37 cards are in this state; a permanently red check gets ignored."""
    report = check_card(done_card("card-1"), lookup=None)
    assert report.verdict == NO_PR
    assert report.detail == "no result_pr_url"


def test_a_url_that_is_not_a_pull_request_is_malformed_not_accepted():
    """Worse than an absent URL: it reads as evidence and is not.

    A commit URL is the tempting substitution -- it looks like proof the work
    landed, and under squash merging the branch-local sha it names is exactly
    the thing that proves nothing.
    """
    for url in (
        "https://github.com/sirus20x6/rwkv-lab/commit/917caa75",
        "https://github.com/sirus20x6/rwkv-lab/tree/card/some-branch",
        "https://github.com/sirus20x6/rwkv-lab/pull/",
        "see PR 245",
        "245",
    ):
        report = check_card(done_card("card-1", url), lookup=None)
        assert report.verdict == MALFORMED, url


def test_a_pull_request_url_yields_its_number_through_trailing_noise():
    assert pull_request_number(MERGED_URL) == 245
    assert pull_request_number(MERGED_URL + "/files") == 245
    assert pull_request_number(MERGED_URL + "#issuecomment-1") == 245
    assert pull_request_number("  " + MERGED_URL + "  ") == 245


def test_a_cited_pull_request_that_never_merged_fails():
    """The failure the rule exists to catch: card says done, evidence disagrees."""
    for state in ("OPEN", "CLOSED", "UNKNOWN"):
        report = check_card(done_card("card-1", MERGED_URL),
                            lookup=lambda _number, s=state: s)
        assert report.verdict == NOT_MERGED, state
        assert state in report.detail


def test_a_cited_pull_request_that_merged_passes():
    report = check_card(done_card("card-1", MERGED_URL),
                        lookup=lambda _number: "MERGED")
    assert report.verdict == CITES_MERGED


def test_the_summary_distinguishes_an_empty_population_from_a_clean_one():
    """A board with no recognised Done cards must not read as all-clear.

    This is the failure mode the summary line exists to prevent: zero findings
    over zero cards and zero findings over 168 cards are the same sentence
    unless the population is stated.
    """
    empty = report_lines([], total_cards=40)[-1]
    assert "0 done of 40 cards" in empty
    assert "PASSED" in empty

    clean = report_lines(
        [check_card(done_card("card-1", MERGED_URL),
                    lookup=lambda _n: "MERGED")],
        total_cards=40)[-1]
    assert "1 done of 40 cards" in clean
    assert clean != empty


def test_the_run_exits_non_zero_only_for_defects_not_for_gaps(tmp_path):
    """A gap is a countdown; a false claim is a failure. Different exit codes."""
    board = tmp_path / "board.json"

    board.write_text(json.dumps({"cards": [
        done_card("card-1"),                      # NO PR -- a gap
        done_card("card-2", MERGED_URL),          # cited, unverified
        {"id": "card-3", "claim_status": "claimed"},
    ]}))
    assert main(["--board", str(board)]) == 0

    board.write_text(json.dumps({"cards": [
        done_card("card-1", "https://github.com/o/r/commit/deadbeef"),
    ]}))
    assert main(["--board", str(board)]) == 1

    board.write_text(json.dumps({"cards": [
        done_card("card-1", MERGED_URL),
    ]}))
    assert main(["--board", str(board)], lookup=lambda _n: "OPEN") == 1
    assert main(["--board", str(board)], lookup=lambda _n: "MERGED") == 0


def test_a_bare_list_payload_is_accepted(tmp_path):
    """kanban responses appear both as {"cards": [...]} and as a bare list."""
    board = tmp_path / "board.json"
    board.write_text(json.dumps([done_card("card-1", MERGED_URL)]))
    assert main(["--board", str(board)], lookup=lambda _n: "MERGED") == 0


# --- suggestion mode -------------------------------------------------------


def pull(number: int, title: str) -> dict:
    return {"number": number, "title": title}


def test_stopwords_alone_never_produce_a_match():
    """Without this the method collapses into noise.

    Two titles that share only "the", "of" and "for" describe nothing in
    common. If those counted, every card would match every PR weakly and the
    ranking would be meaningless — the same worthlessness measured for timing
    correlation, which was wrong for two of the three cards it matched.
    """
    assert title_overlap("The state of the run", "For the sake of it") == 0.0
    assert title_tokens("the a an of to for") == frozenset()


def test_an_identical_title_scores_one_and_an_unrelated_one_scores_zero():
    assert title_overlap("Remap the vision keys", "Remap the vision keys") == 1.0
    assert title_overlap("Remap the vision keys", "Bound the skip count") == 0.0


def test_the_runner_up_is_reported_so_a_crowded_field_is_visible():
    """A 0.44 with a 0.40 rival is one of a crowd; a 0.56 with 0.08 is not.

    Reporting only the winner hides the difference a reader needs, and this is
    the field that made the real board's weakest candidate identifiable.
    """
    pulls = [pull(1, "Serve registries from opt so updates need no root"),
             pull(2, "Serve registries from opt so updates need less root"),
             pull(3, "Something entirely different about kernels")]
    hit = suggest({"title": "Move registries out of etc so updates need no root"},
                  pulls, minimum=0.1)
    assert hit["number"] == 1
    assert hit["runner_up"] > 0.3          # the near-duplicate is visible
    assert hit["runner_up"] < hit["score"]


def test_a_match_below_the_floor_is_not_proposed():
    pulls = [pull(1, "Remap the Qwen vision checkpoint keys at load")]
    card = {"title": "Bound the runtime skip count"}
    assert suggest(card, pulls, minimum=0.30) is None


def test_a_single_candidate_reports_a_zero_runner_up_rather_than_failing():
    hit = suggest({"title": "Remap vision keys"},
                  [pull(1, "Remap vision keys")], minimum=0.1)
    assert hit["runner_up"] == 0.0


def test_suggest_skips_cards_that_already_record_a_pull_request(tmp_path, capsys):
    board = tmp_path / "board.json"
    board.write_text(json.dumps({"cards": [
        done_card("card-has", MERGED_URL, title="Remap vision keys"),
        done_card("card-none", title="Remap vision keys"),
    ]}))
    pulls = tmp_path / "pulls.json"
    pulls.write_text(json.dumps([pull(96, "Remap vision keys")]))

    assert main(["--board", str(board), "--suggest", str(pulls)]) == 0
    out = capsys.readouterr().out
    assert "card-none" in out
    assert "card-has" not in out
    assert "1 proposed for 1 cards recording no PR, of 2 done" in out


def test_suggest_exits_zero_even_when_it_proposes(tmp_path):
    """Advisory, deliberately. A suggestion that could fail a run would
    eventually be applied unread, which is the failure this check exists to
    stop — a result_pr_url that reads as evidence and is not."""
    board = tmp_path / "board.json"
    board.write_text(json.dumps({"cards": [done_card("card-1", title="Remap vision keys")]}))
    pulls = tmp_path / "pulls.json"
    pulls.write_text(json.dumps([pull(96, "Remap vision keys")]))
    assert main(["--board", str(board), "--suggest", str(pulls)]) == 0
