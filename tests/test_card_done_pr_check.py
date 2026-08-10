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
    main,
    pull_request_number,
    report_lines,
)

MERGED_URL = "https://github.com/sirus20x6/rwkv-lab/pull/245"


def done_card(identifier: str, url: str | None = None, **extra) -> dict:
    card = {"id": identifier, "title": "a card", "claim_status": "done"}
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
