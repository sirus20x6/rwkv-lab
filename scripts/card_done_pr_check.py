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
    arguments = parser.parse_args(argv)

    cards = load_cards(arguments.board)
    done = [card for card in cards if is_done(card)]

    if lookup is None and arguments.verify_merged:
        def lookup(number: int) -> str:
            return merged_state(number, arguments.repo)

    reports = [check_card(card, lookup) for card in done]
    for line in report_lines(reports, len(cards)):
        print(line)

    return 1 if any(r.verdict in (NOT_MERGED, MALFORMED) for r in reports) else 0


if __name__ == "__main__":
    sys.exit(main())
