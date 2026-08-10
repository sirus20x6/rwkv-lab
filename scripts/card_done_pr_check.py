#!/usr/bin/env python3
"""Check that Done cards cite a pull request that actually merged.

`CLAUDE.md` states the rule this enforces: a card in Done is a promise that
`git grep` against `origin/main` will find the thing, and under squash merging
the merged-PR number is the only decisive evidence that it landed. Every
git-native alternative fails here -- a squashed branch head is never an
ancestor of main, `git cherry` reported 10 of 10 commits absent when 9 were
present, and an "ahead" count stays non-zero forever.

That rule has been a prose convention, and prose conventions lose. Measured
across this project's five boards on 2026-08-10: **37 of 168 Done cards record
no `result_pr_url` at all**. An earlier count put it at 56 of 126, so the
convention is being followed more often than it was and is still missed about
a fifth of the time.

Three failures are worth separating, because they need different responses:

- **NO PR RECORDED** -- Done with no `result_pr_url`. The work may well be on
  main; nothing here says it is not. What is lost is the only cheap way to
  check, so the next reader pays for an investigation. Reported, and counted
  as a countdown rather than failing the run: it is a backlog of 37, and a
  check that is permanently red is one everybody learns to ignore.
- **MALFORMED** -- a `result_pr_url` that is not a pull-request URL. This is
  worse than an absent one, because it reads as evidence and is not.
- **NOT MERGED** -- a `result_pr_url` naming a pull request that is open, or
  closed without merging. This is the failure the rule exists to catch: the
  card says done, the evidence it cites says otherwise.

The last two exit non-zero. They are defects in the record itself rather than
gaps in it.

This reads a kanban board, so it cannot be a CI gate -- nothing in a pull
request can see a card. It is a review-time tool, the same shape and for the
same reason as `card_anchor_check.py`.

Usage:

    # save a kanban_get_board response to board.json first
    python scripts/card_done_pr_check.py --board board.json
    python scripts/card_done_pr_check.py --board board.json --verify-merged
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import subprocess
import sys
from dataclasses import dataclass

# A card is Done when the board says so. Both spellings appear in board
# payloads and neither is authoritative on its own, so both are honoured.
DONE_MARKERS = (("claim_status", "done"), ("status", "completed"))

PULL_REQUEST_URL = re.compile(
    r"^https://github\.com/[^/\s]+/[^/\s]+/pull/(\d+)(?:[/?#].*)?$")

NO_PR = "NO PR RECORDED"
MALFORMED = "MALFORMED"
NOT_MERGED = "NOT MERGED"
CITES_MERGED = "CITES A MERGED PR"
UNVERIFIED = "CITES A PR (not verified)"


@dataclass(frozen=True)
class CardReport:
    identifier: str
    title: str
    verdict: str
    detail: str


def is_done(card: dict) -> bool:
    return any(card.get(field) == value for field, value in DONE_MARKERS)


def pull_request_number(url: str) -> int | None:
    """The PR number a result URL names, or None if it names something else.

    Deliberately strict. A checker that accepts anything containing digits
    would report a branch URL or a commit URL as good evidence, which is the
    exact substitution this is meant to catch.
    """
    match = PULL_REQUEST_URL.match(url.strip())
    return int(match.group(1)) if match else None


def merged_state(number: int, repository: str | None) -> str:
    """Ask GitHub whether a pull request merged. Returns its state string."""
    command = ["gh", "pr", "view", str(number), "--json", "state",
               "-q", ".state"]
    if repository:
        command.extend(["--repo", repository])
    finished = subprocess.run(command, capture_output=True, text=True)
    if finished.returncode != 0:
        return "UNKNOWN"
    return finished.stdout.strip() or "UNKNOWN"


def check_card(card: dict, lookup) -> CardReport:
    identifier = card.get("id", "<no id>")
    title = (card.get("title") or "").strip()
    url = (card.get("result_pr_url") or "").strip()

    if not url:
        return CardReport(identifier, title, NO_PR, "no result_pr_url")

    number = pull_request_number(url)
    if number is None:
        return CardReport(identifier, title, MALFORMED,
                          f"{url!r} is not a pull-request URL")

    if lookup is None:
        return CardReport(identifier, title, UNVERIFIED, f"#{number}")

    state = lookup(number)
    if state == "MERGED":
        return CardReport(identifier, title, CITES_MERGED, f"#{number}")
    return CardReport(identifier, title, NOT_MERGED, f"#{number} is {state}")


# Words that carry no signal when matching a card title to a pull-request
# title. Without these, two unrelated titles sharing "the" and "for" score a
# spurious overlap, and the whole method degrades into the timing correlation
# that was measured and rejected -- see below.
STOPWORDS = frozenset(
    "the a an of to for in on and or is are be that this it its with by from "
    "no not so as at".split())


def title_tokens(title: str) -> frozenset[str]:
    return frozenset(
        word for word in re.findall(r"[a-z0-9_]+", title.lower())
        if word not in STOPWORDS and len(word) > 2)


def title_overlap(card_title: str, pull_title: str) -> float:
    """Jaccard overlap of the two titles' significant words."""
    left, right = title_tokens(card_title), title_tokens(pull_title)
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def suggest(card: dict, pulls: list[dict], minimum: float) -> dict | None:
    """The best-matching pull request for a card, with its runner-up.

    The runner-up is returned because it is what makes a suggestion readable.
    A 0.44 match whose nearest rival scores 0.40 is one of a crowd; a 0.56
    whose rival scores 0.08 is isolated. Reporting only the winner hides
    exactly the difference a reader needs.

    Measured on this board 2026-08-10: at a 0.30 floor this proposes five
    candidates for 20 unmatched cards, two of which independently reproduce
    matches found earlier by searching pull-request bodies. Timing correlation
    over the same population was wrong for two of the three cards it matched
    uniquely, which is why proximity in time is not used here at all.
    """
    scored = sorted(
        ((title_overlap(card.get("title") or "", p.get("title") or ""), p)
         for p in pulls),
        key=lambda row: row[0], reverse=True)
    if not scored or scored[0][0] < minimum:
        return None
    runner_up = scored[1][0] if len(scored) > 1 else 0.0
    return {"score": scored[0][0], "runner_up": runner_up,
            "number": scored[0][1].get("number"),
            "title": scored[0][1].get("title")}


def load_cards(source: str) -> list[dict]:
    payload = json.loads(pathlib.Path(source).read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return payload
    return payload.get("cards", [])


def report_lines(reports: list[CardReport], total_cards: int) -> list[str]:
    lines: list[str] = []
    by_verdict: dict[str, list[CardReport]] = {}
    for report in reports:
        by_verdict.setdefault(report.verdict, []).append(report)

    for verdict in (NOT_MERGED, MALFORMED, NO_PR):
        entries = by_verdict.get(verdict, [])
        if not entries:
            continue
        lines.append(f"{verdict} ({len(entries)}):")
        for entry in entries:
            lines.append(f"  {entry.identifier}  {entry.title[:64]}")
            lines.append(f"      {entry.detail}")

    good = len(by_verdict.get(CITES_MERGED, []))
    unverified = len(by_verdict.get(UNVERIFIED, []))
    failures = len(by_verdict.get(NOT_MERGED, [])) + \
        len(by_verdict.get(MALFORMED, []))

    # The summary states the population as well as the findings. A board whose
    # cards this tool did not recognise as Done produces zero findings, which
    # would otherwise read exactly like a board where every card is correct.
    lines.append(
        f"card done-PR check: {'FAILED' if failures else 'PASSED'} — "
        f"{len(reports)} done of {total_cards} cards; "
        f"{good} cite a merged PR, {unverified} cite a PR unverified, "
        f"{len(by_verdict.get(NO_PR, []))} record none, "
        f"{len(by_verdict.get(MALFORMED, []))} malformed, "
        f"{len(by_verdict.get(NOT_MERGED, []))} not merged")
    return lines


def main(argv: list[str] | None = None, lookup=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--board", required=True,
                        help="path to a saved kanban_get_board response")
    parser.add_argument("--verify-merged", action="store_true",
                        help="ask GitHub whether each cited PR merged")
    parser.add_argument("--repo", default=None,
                        help="owner/name passed to gh, when not inferable")
    parser.add_argument("--suggest", metavar="PULLS_JSON",
                        help="propose a PR for each card recording none, by "
                             "title overlap against a saved "
                             "`gh pr list --state merged --json number,title`")
    parser.add_argument("--min-overlap", type=float, default=0.30,
                        help="floor for --suggest (default 0.30)")
    arguments = parser.parse_args(argv)

    cards = load_cards(arguments.board)
    done = [card for card in cards if is_done(card)]

    if arguments.suggest:
        pulls = json.loads(
            pathlib.Path(arguments.suggest).read_text(encoding="utf-8"))
        without = [c for c in done if not (c.get("result_pr_url") or "").strip()]
        proposed = 0
        for card in without:
            hit = suggest(card, pulls, arguments.min_overlap)
            if hit is None:
                continue
            proposed += 1
            print(f"{hit['score']:.2f} (next {hit['runner_up']:.2f})  "
                  f"{card.get('id')}")
            print(f"      card: {(card.get('title') or '')[:76]}")
            print(f"      PR #{hit['number']}: {(hit['title'] or '')[:70]}")
        # Advisory only: these are candidates for a reader to confirm against
        # the diff, never evidence. Exit status stays 0 so nothing gates on a
        # guess -- a suggestion that could fail a run would eventually be
        # applied unread, which is the failure this whole check exists to stop.
        print(f"card done-PR suggestions: {proposed} proposed for "
              f"{len(without)} cards recording no PR, of {len(done)} done; "
              f"confirm each against the diff before recording it")
        return 0


    if lookup is None and arguments.verify_merged:
        def lookup(number: int) -> str:
            return merged_state(number, arguments.repo)

    reports = [check_card(card, lookup) for card in done]
    for line in report_lines(reports, len(cards)):
        print(line)

    return 1 if any(r.verdict in (NOT_MERGED, MALFORMED) for r in reports) else 0


if __name__ == "__main__":
    sys.exit(main())
