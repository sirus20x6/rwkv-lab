#!/usr/bin/env python3
"""Answer which worktrees a sweep may remove, refusing when it cannot tell.

Two worktrees were deleted out from under working agents on 2026-08-09, one
of them mid-build, and a third was one command away. `CLAUDE.md` records the
diagnosis under "Layout does NOT tell you who owns a worktree": every signal
available is ambiguous in the direction that causes harm. A clean tree looks
identical whether an agent just committed or finished an hour ago; commit age
says nothing; and the directory path carries no ownership field at all.

`card-1bf6bb54` sketched a pid lock, or `flock`, releasing on process death.
**That does not work here, and the reason is worth stating so nobody builds
it.** In-process subagents share the parent `claude` process, so a pid-keyed
lock marks every worktree as owned by the same live pid — it never reclaims
anything while the session lives, which is precisely the "silently rots into
every worktree looks live" failure that card warns against. `flock` inherits
the same problem: there is no per-agent process to hold the descriptor.

What does discriminate is the **agent roster**, which the harness knows and a
shell script does not. So ownership is recorded at creation, in a marker file
inside the worktree, and liveness is supplied by the caller from `ListAgents`:

    python scripts/worktree_sweep_check.py --claim reviewer .claude/worktrees/x
    python scripts/worktree_sweep_check.py --live reviewer --live builder

Three verdicts, and the third is the point:

- **RECLAIMABLE** — the marker names an agent that is not live. Safe to remove.
- **OWNED** — the marker names a live agent. Leave it alone.
- **UNKNOWN** — no marker, or an unreadable one. **Refused, not swept.**

`UNKNOWN` is why this is safe to adopt before anything writes markers: on a
tree with no markers every worktree is refused and the sweep does nothing.
Adoption makes it *more* permissive, one worktree at a time, which is the
correct direction for a mechanism whose failure costs an agent's work.

It also fails safe on a crash without the agent doing anything at exit: a dead
agent leaves the roster, so its marker stops matching and its tree becomes
reclaimable on the next sweep. Nothing has to be cleaned up by the process
that is gone — which was the second requirement on that card.

Exit status is 0 only when **every** examined worktree is reclaimable, so a
sweep can gate on it wholesale; read the per-tree rows to sweep selectively.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys

MARKER = ".claude-agent-owner"

RECLAIMABLE = "RECLAIMABLE"
OWNED = "OWNED"
UNKNOWN = "UNKNOWN"


def nested_worktrees(repository: pathlib.Path) -> list[pathlib.Path]:
    """The worktrees under this checkout's own `.claude/worktrees/`.

    Filtered deliberately: `git worktree list` in this repository lists two
    repositories' worktrees, because the checkout shares a git dir with
    /thearray/git/moe-mla. Entries belonging to the live checkout are nobody's
    to sweep from here.
    """
    listing = subprocess.run(
        ["git", "-C", str(repository), "worktree", "list", "--porcelain"],
        capture_output=True, text=True)
    root = (repository / ".claude" / "worktrees").resolve()
    found = []
    for line in listing.stdout.splitlines():
        if not line.startswith("worktree "):
            continue
        path = pathlib.Path(line[len("worktree "):]).resolve()
        if root in path.parents:
            found.append(path)
    return sorted(found)


def read_marker(worktree: pathlib.Path) -> str | None:
    """The agent named by this worktree's marker, or None if there is none.

    An unreadable or malformed marker returns None rather than raising: the
    caller treats that as UNKNOWN, which refuses. A checker that crashes on a
    corrupt marker would strand a sweep entirely.
    """
    path = worktree / MARKER
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    agent = payload.get("agent") if isinstance(payload, dict) else None
    return agent if isinstance(agent, str) and agent.strip() else None


def write_marker(worktree: pathlib.Path, agent: str, stamp: str) -> pathlib.Path:
    path = worktree / MARKER
    path.write_text(
        json.dumps({"agent": agent, "claimed_at": stamp}, indent=2) + "\n",
        encoding="utf-8")
    return path


def classify(worktree: pathlib.Path, live: set[str]) -> tuple[str, str]:
    agent = read_marker(worktree)
    if agent is None:
        return UNKNOWN, f"no readable {MARKER}; refusing rather than guessing"
    if agent in live:
        return OWNED, f"claimed by {agent}, which is live"
    return RECLAIMABLE, f"claimed by {agent}, which is not in the live roster"


def report(rows: list[tuple[pathlib.Path, str, str]]) -> list[str]:
    lines = []
    for path, verdict, detail in rows:
        lines.append(f"{verdict:12} {path.name}")
        lines.append(f"             {detail}")
    counts = {v: sum(1 for _, verdict, _ in rows if verdict == v)
              for v in (RECLAIMABLE, OWNED, UNKNOWN)}
    # The population is stated because zero findings over zero worktrees and
    # zero findings over fifty are otherwise the same sentence.
    lines.append(
        f"worktree sweep check: "
        f"{'PASSED' if rows and counts[OWNED] == 0 and counts[UNKNOWN] == 0 else 'REFUSED'} — "
        f"{len(rows)} worktrees examined; {counts[RECLAIMABLE]} reclaimable, "
        f"{counts[OWNED]} owned by a live agent, {counts[UNKNOWN]} unknown")
    return lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repository", default=".")
    parser.add_argument("--live", action="append", default=[],
                        help="name of a live agent, from ListAgents; repeatable")
    parser.add_argument("--claim", metavar="AGENT",
                        help="write an ownership marker instead of checking")
    parser.add_argument("--claimed-at", default="unrecorded",
                        help="timestamp written with --claim")
    parser.add_argument("worktrees", nargs="*",
                        help="paths to examine; default is every nested worktree")
    arguments = parser.parse_args(argv)

    repository = pathlib.Path(arguments.repository).resolve()

    if arguments.claim:
        if not arguments.worktrees:
            print("--claim needs the worktree to mark", file=sys.stderr)
            return 2
        for entry in arguments.worktrees:
            path = write_marker(pathlib.Path(entry).resolve(),
                                arguments.claim, arguments.claimed_at)
            print(f"claimed {path}")
        return 0

    targets = ([pathlib.Path(w).resolve() for w in arguments.worktrees]
               or nested_worktrees(repository))
    live = {name for name in arguments.live if name}
    rows = [(path, *classify(path, live)) for path in targets]
    for line in report(rows):
        print(line)
    return 0 if rows and all(v == RECLAIMABLE for _, v, _ in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
