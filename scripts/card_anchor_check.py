#!/usr/bin/env python3
"""Check a kanban card's cited file anchors against a revision, at dispatch time.

`card-74b81095` carried a block headed "measured on origin/main, do not
re-derive" with four file-and-line-cited claims. All four were true when
written and all four were false five hours later, when PR #201 merged. Nothing
on the card said what it had been measured against, so a reader obeying "do not
re-derive" worked from a description of a tree that no longer existed.

The convention answer -- cite the sha you measured at -- reaches only cards
written after the convention. The 200+ already filed will never be
back-annotated, and they are the ones that go stale. So this reads what a card
already says. Every card that cites `path/to/file.cpp:123` has, without
intending to, recorded a claim that can be resolved against a revision.

Two questions are asked of each anchor, and they are deliberately not the same
question:

**Does the cited path exist on the revision?** This is decisive and it is the
only thing that sets a non-zero exit. A card naming a path that is not there
either describes a branch that was read as trunk, or describes a file it wants
created -- and the tool cannot tell those apart, so it reports and the reader
decides. Two cards on the board today name `src/rwkv_lab/qwen_caption_finetune.py`,
which `git log origin/main` shows has never existed.

**Did the cited paths change after the card was written?** `created_at` is the
measurement time every card already carries, so this needs nothing added to any
card. It does not prove a claim is stale -- a commit touching the file may not
touch the claim -- so it never fails the run. It is printed because it is the
one command the reader would otherwise have to construct themselves, and
because it is what would have caught `card-74b81095`: six of its nine cited
paths moved between authoring and dispatch.

WHAT THIS DOES NOT DO, measured rather than assumed. Checking that a cited line
still *says* what the card claims -- `host_ledger.hpp:102` still describing
`BundleRequestResult` as `{status, grant, outcome_digest, replayed}` -- was
built and rejected. Binding backticked identifiers on a card's line to the
anchor on that line and requiring the identifier within +/-3 lines of the cited
line reported 46 of 70 checkable anchors as moved. Most were prose adjacency,
not drift: a sentence naming four symbols and one path binds all four to it. A
check that fires on two thirds of its input carries about as much information
as one that fires on none, and it would have made every card look stale, which
is how a real signal gets ignored. The line-number check kept here is the weak
half of that idea -- a cited line past the end of the file -- which is honest
but toothless: it fires on zero of the 200 line-carrying anchors on the board
today. Its count is printed so that silence is visible rather than implied.

Scope: card bodies only, not comments. A comment is usually the correction to a
body, so anchors there are current by construction and would dilute the report.
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
import re
import subprocess
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from scripts.gate_verdict import verdict_line  # noqa: E402

LABEL = "card anchor check"

# Extensions a source path in a card body plausibly ends in. An explicit list
# rather than "a dotted token" -- prose is full of `foo.bar` that is a method
# call, and every one of those would resolve to nothing and be reported as a
# missing path, which is the failure mode this whole script exists to avoid.
EXTENSIONS = (
    "py|cpp|hpp|cc|h|go|json|md|proto|sh|yaml|yml|ts|tsx|js|toml|cmake|sql|rs"
)

# A path token, optionally followed by :line or :line-line.
#
# The leading lookbehind refuses a match that continues a longer token, so
# `.../mageflow-cache-resume.json` yields nothing rather than yielding the
# basename and reporting the ellipsis as a missing file.
ANCHOR = re.compile(
    r"(?<![\w/.-])"
    r"((?:[A-Za-z0-9][\w.-]*/)*[A-Za-z0-9][\w.-]*\.(?:" + EXTENSIONS + r"))"
    r"(?::(\d+)(?:-(\d+))?)?"
    r"(?![\w])"
)

# Shapes that are a template rather than a path. `eval_samples/step_XXXXXXXX.json`
# names a family of files and none of them; resolving it and reporting it
# missing would be a true statement about a question nobody asked.
PLACEHOLDER = re.compile(r"XXX|\$\{|\$[A-Z]|<[a-z]|\*")


class Anchor:
    """One `path` or `path:line` citation, with where it resolved."""

    def __init__(self, path: str, line: int | None, end: int | None):
        self.path = path
        self.line = line
        self.end = end
        self.resolved: str | None = None
        self.state = "unchecked"
        self.note = ""

    @property
    def cited(self) -> str:
        if self.line is None:
            return self.path
        if self.end is not None and self.end != self.line:
            return f"{self.path}:{self.line}-{self.end}"
        return f"{self.path}:{self.line}"


def parse_anchors(body: str) -> list[Anchor]:
    """Every path-shaped citation in a card body, in order, deduplicated.

    Deduplicated on the full citation rather than the path: `service.cpp:400`
    and `service.cpp:4999` are two claims about one file and both are worth
    resolving, while the same string twice is one claim stated twice.
    """
    seen: dict[str, Anchor] = {}
    for path, line, end in ANCHOR.findall(body):
        anchor = Anchor(path, int(line) if line else None,
                        int(end) if end else (int(line) if line else None))
        seen.setdefault(anchor.cited, anchor)
    return list(seen.values())


def git(repository: pathlib.Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(repository), *args],
                          capture_output=True, text=True, check=False)


NEVER_ANYWHERE = "never on any ref — a proposal, or an example in prose"
BRANCH_ONLY = "on a branch, never on {rev}"
WAS_ON_REV = "was on {rev} and is not now — renamed or deleted"


def locate_missing(repository: pathlib.Path, path: str, rev: str) -> str:
    """Say where a path that is absent from ``rev`` does live.

    A bare MISSING is three different situations wearing one label, and they
    need opposite responses. Measured over the 54 missing paths on this
    project's boards: 25 exist only on a branch, 27 exist nowhere, 2 were on
    main and were renamed away.

    - **Branch-only** is the one that misleads. The card reads as though it
      describes trunk and describes unlanded work; seven such paths are a
      single production-qualification toolchain that never landed.
    - **Never anywhere** is usually benign — an illustrative path in a
      sentence, or a file the card proposes to create. No tool separates
      those two; a reader must.
    - **Was on the revision** means the card is right about the work and
      wrong about the name. It would otherwise read as "this never landed",
      which is the absence trap this repository already warns about, arriving
      through a filename instead of a sha.

    Order matters, and specifically the revision is asked first. ``git log
    <rev> -- <path>`` finds a *deleted* path, which an ls-tree membership
    test does not — that is what separates a renamed file from branch-only
    work. Asking ``--all`` first reports **branch-only paths as was-on-rev**,
    because ``--all`` includes the revision; 25 of the 27 findings here would
    have inverted. The branch-only case is what catches that mistake, not the
    deleted case, whose answer is the same under either order.
    """
    on_rev = git(repository, "log", rev, "--oneline", "-1", "--", path)
    if on_rev.returncode == 0 and on_rev.stdout.strip():
        return WAS_ON_REV.format(rev=rev) + f"; last {on_rev.stdout.strip()}"
    anywhere = git(repository, "log", "--all", "--oneline", "-1", "--", path)
    if anywhere.returncode == 0 and anywhere.stdout.strip():
        return (BRANCH_ONLY.format(rev=rev)
                + f"; see {anywhere.stdout.strip()}")
    return NEVER_ANYWHERE


def revision_files(repository: pathlib.Path, rev: str) -> list[str]:
    completed = git(repository, "ls-tree", "-r", "--name-only", rev)
    if completed.returncode != 0:
        raise SystemExit(
            f"{LABEL}: cannot read {rev} in {repository}: "
            f"{completed.stderr.strip()}")
    return completed.stdout.splitlines()


class Revision:
    """The file list of one revision, with the lookups the resolver needs."""

    def __init__(self, repository: pathlib.Path, rev: str):
        self.repository = repository
        self.rev = rev
        self.files = revision_files(repository, rev)
        self.exact = set(self.files)
        self.by_name: dict[str, list[str]] = collections.defaultdict(list)
        for path in self.files:
            self.by_name[path.rsplit("/", 1)[-1]].append(path)
        self._lengths: dict[str, int] = {}

    def resolve(self, cited: str) -> tuple[str | None, str]:
        """Map a cited path onto a real path, or explain why it did not.

        Cards abbreviate: `host_ledger.hpp:79` after naming the full path once.
        A unique basename or unique suffix match recovers those. An ambiguous
        one is reported as ambiguous rather than guessed -- picking the first of
        four `main.cpp` and checking it would answer a question about a file the
        card never named.
        """
        if cited in self.exact:
            return cited, "exact"
        if "/" not in cited:
            candidates = self.by_name.get(cited, [])
        else:
            candidates = [f for f in self.files if f.endswith("/" + cited)]
        if len(candidates) == 1:
            return candidates[0], "resolved by suffix"
        if candidates:
            return None, f"ambiguous: {len(candidates)} files match"
        return None, "no such path"

    def length(self, path: str) -> int:
        if path not in self._lengths:
            blob = git(self.repository, "show", f"{self.rev}:{path}").stdout
            self._lengths[path] = len(blob.splitlines())
        return self._lengths[path]

    def commits_since(self, path: str, since: str) -> list[str]:
        completed = git(self.repository, "log", "--oneline", f"--since={since}",
                        self.rev, "--", path)
        return completed.stdout.strip().splitlines()

    def is_ignored(self, path: str) -> bool:
        """True for a path .gitignore excludes -- a build or evidence artifact.

        `evidence/acceptance.json` and `compile_commands.json` are cited by
        cards and are correctly absent from every revision. Reporting them as
        missing paths would be five wrong rows in a report whose whole value is
        that its rows are worth reading.
        """
        return git(self.repository, "check-ignore", "-q", "--", path).returncode == 0


class CardReport:
    def __init__(self, card_id: str, title: str):
        self.card_id = card_id
        self.title = title
        self.anchors: list[Anchor] = []
        self.drift: list[tuple[str, list[str]]] = []
        self.since = ""

    @property
    def missing(self) -> list[Anchor]:
        return [a for a in self.anchors if a.state in ("missing", "past end of file")]

    @property
    def checked(self) -> list[Anchor]:
        return [a for a in self.anchors
                if a.state not in ("skipped", "unresolved")]

    @property
    def unresolved(self) -> list[Anchor]:
        return [a for a in self.anchors if a.state == "unresolved"]

    @property
    def verdict(self) -> str:
        """The three states this tool must never conflate.

        NO ANCHORS is the one that matters. A checker whose parse recognises
        nothing in a card reports that card as healthy unless "nothing was
        checked" is a visibly different answer from "everything held".
        """
        if not self.checked:
            return "NO ANCHORS" if not self.anchors else "NOTHING CHECKABLE"
        if self.missing:
            return "STALE"
        if self.drift:
            return "DRIFTED"
        return "ANCHORS HOLD"


def check_card(card: dict, revision: Revision) -> CardReport:
    report = CardReport(card.get("id", "<no id>"), card.get("title", ""))
    report.since = card.get("created_at", "")
    report.anchors = parse_anchors(card.get("body", ""))

    drifted: dict[str, list[str]] = {}
    for anchor in report.anchors:
        if PLACEHOLDER.search(anchor.path):
            anchor.state, anchor.note = "skipped", "template, not a path"
            continue
        resolved, why = revision.resolve(anchor.path)
        if resolved is None:
            if why.startswith("ambiguous"):
                # Not a stale claim and not a checked one. `train_mla.py` names
                # two files on main; picking one would answer a question the
                # card did not ask, and calling it missing would be false.
                anchor.state, anchor.note = "unresolved", why
                continue
            if revision.is_ignored(anchor.path):
                anchor.state = "skipped"
                anchor.note = "gitignored: a generated artifact, absent by design"
                continue
            anchor.state, anchor.note = "missing", why
            continue
        anchor.resolved = resolved
        if anchor.end is not None and anchor.end > revision.length(resolved):
            anchor.state = "past end of file"
            anchor.note = f"file is {revision.length(resolved)} lines"
            continue
        anchor.state = "holds"
        anchor.note = "" if resolved == anchor.path else f"{why} to {resolved}"
        if report.since and resolved not in drifted:
            commits = revision.commits_since(resolved, report.since)
            if commits:
                drifted[resolved] = commits
    report.drift = sorted(drifted.items())
    return report


def print_report(report: CardReport, rev: str,
                 repository: pathlib.Path | None = None) -> None:
    print(f"\n{report.card_id}  {report.title}")
    if report.verdict in ("NO ANCHORS", "NOTHING CHECKABLE"):
        if report.verdict == "NO ANCHORS":
            print("  NO ANCHORS FOUND — this card cites no file path this tool "
                  "recognises, so nothing was checked. That is not the same as "
                  "'the card is accurate'.")
        else:
            print(f"  NOTHING CHECKABLE — this card cites "
                  f"{len(report.anchors)} path-shaped token(s), none of which "
                  f"could be resolved to one file. Nothing was checked.")
        for anchor in report.anchors:
            if anchor.state in ("skipped", "unresolved"):
                print(f"    {anchor.state:9s}{anchor.cited}  ({anchor.note})")
        return

    held = [a for a in report.anchors if a.state == "holds"]
    summary = (f"{len(held)} of {len(report.checked)} checked anchors resolve "
               f"on {rev}")
    if report.verdict == "DRIFTED":
        summary += (f", but {len(report.drift)} of them changed after the card "
                    f"was written")
    print(f"  {report.verdict} — {summary}")
    for anchor in report.anchors:
        if anchor.state == "missing":
            print(f"    MISSING  {anchor.cited}  ({anchor.note} on {rev})")
            if repository is not None:
                print(f"             {locate_missing(repository, anchor.path, rev)}")
        elif anchor.state == "past end of file":
            print(f"    PAST EOF {anchor.cited}  ({anchor.note})")
        elif anchor.state in ("skipped", "unresolved"):
            print(f"    {anchor.state:9s}{anchor.cited}  ({anchor.note})")
        elif anchor.note:
            print(f"    holds    {anchor.cited}  ({anchor.note})")
    if report.drift:
        print(f"  cited paths that changed after this card was written "
              f"({report.since}) — the measurement may not survive them:")
        for path, commits in report.drift:
            print(f"    {path}  {len(commits)} commit(s)")
            for commit in commits[:3]:
                print(f"      {commit}")
            if len(commits) > 3:
                print(f"      … {len(commits) - 3} more")


def load_cards(source: str) -> list[dict]:
    text = sys.stdin.read() if source == "-" else pathlib.Path(source).read_text(
        encoding="utf-8")
    document = json.loads(text)
    if isinstance(document, list):
        return document
    if isinstance(document, dict) and isinstance(document.get("cards"), list):
        return document["cards"]
    raise SystemExit(
        f"{LABEL}: {source} is neither a list of cards nor a kanban_get_board "
        f"document with a 'cards' array")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Produce the input with the kanban_get_board MCP tool and save "
               "its JSON, then pass it here with --card <id> for the card you "
               "are about to dispatch.")
    parser.add_argument("--board", required=True,
                        help="kanban_get_board JSON, or '-' for stdin")
    parser.add_argument("--card", action="append", default=[],
                        help="restrict to this card id (repeatable)")
    parser.add_argument("--repository", default=".",
                        help="repository to resolve anchors in")
    parser.add_argument("--rev", default="origin/main",
                        help="revision to resolve against (default origin/main)")
    args = parser.parse_args(argv)

    cards = load_cards(args.board)
    if args.card:
        wanted = set(args.card)
        found = {c.get("id") for c in cards}
        unknown = sorted(wanted - found)
        if unknown:
            raise SystemExit(
                f"{LABEL}: no card in the input has id(s) {unknown}. Checking "
                f"the remainder would report a subset as if it were the whole "
                f"request.")
        cards = [c for c in cards if c.get("id") in wanted]
    if not cards:
        raise SystemExit(f"{LABEL}: the input holds no cards")

    revision = Revision(pathlib.Path(args.repository).resolve(), args.rev)

    reports = [check_card(card, revision) for card in cards]
    problems: list[str] = []
    anchors_checked = 0
    cards_without = 0
    unresolved = 0
    past_eof = 0
    for report in reports:
        print_report(report, args.rev, revision.repository)
        anchors_checked += len(report.checked)
        unresolved += len(report.unresolved)
        if report.verdict in ("NO ANCHORS", "NOTHING CHECKABLE"):
            cards_without += 1
        past_eof += sum(1 for a in report.anchors if a.state == "past end of file")
        for anchor in report.missing:
            problems.append(f"{report.card_id} {anchor.cited}")

    # The recognition rate is the verdict line's job as much as the outcome is.
    # A parse that matched nothing would otherwise print the same reassuring
    # PASSED as a parse that matched everything and found it sound.
    detail = (f"checked {anchors_checked} anchors across "
              f"{len(reports) - cards_without} of {len(reports)} cards, "
              f"{cards_without} cards had none this tool could check, "
              f"{unresolved} anchors were unresolvable, {past_eof} cited a "
              f"line past the end of its file")
    print()
    print(verdict_line(LABEL, problems, detail))
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
