# Working in this repository

Several agents work this repository at once, against a shared kanban board.
These are the conventions that keep that from going wrong. They are here rather
than in a shell comment because tooling has already started depending on them.

## Worktrees live under `.claude/worktrees/`

Create the worktree for a card nested inside the checkout:

```bash
git worktree add -b card/<slug> .claude/worktrees/<slug> origin/main
```

Not as a sibling of the repository. Both work as far as git is concerned, and
this checkout has historically had both shapes at once — roughly 26 siblings
named `/thearray/git/moe-mla-card-<slug>` alongside the nested ones. One
location is chosen so `git worktree list` output is predictable and so a stale
worktree is visible in one place rather than scattered through the parent
directory.

Branch from `origin/main`, not from a feature branch, unless the card says
otherwise. Run `git fetch origin` first: `origin/main` is a local ref and is
only as current as your last fetch. Do not branch from `HEAD` — the primary
checkout is not necessarily on main, and cannot always be, because git allows a
branch in one worktree at a time and a sibling checkout may hold main.

### The same fact makes the primary checkout unsafe to READ

The paragraph above is about where you branch from. The sharper hazard is what
you learn by opening a file there, and it has already produced a confident,
specific, false finding.

Measured 2026-08-10: `/thearray/git/moe-mla-dashboard-vm` is on
`dashboard/declarative-vm-fsm` — a branch whose pull request merged long ago —
**248 commits behind `origin/main`**, with 150 untracked files. It is not
neglect and it is not fixable by switching: `/thearray/git/moe-mla-card-inputpipe`
holds `main`, and git allows a branch in one worktree at a time. Whichever
worktree loses that race is stale by construction, and this one lost it.

So a file read from the primary checkout describes a world that is a quarter of
a thousand commits gone. What that cost, concretely: `WorkerToController` in
`src/trainvm/v1/trainvm_pb2.py` there has nine fields and no `runtime_evidence`,
and constructing one raises `ValueError: Protocol message WorkerToController has
no "runtime_evidence" field`. On `origin/main` that field exists, numbered 10.
The conclusion drawn from the stale read — that `publish_runtime_evidence` can
never work — was wrong, specific, and would have sent someone to rewrite a
working transport.

**Read through the ref, or from a worktree you created from it:**

```bash
git show origin/main:<path>          # a single file
git grep -n '<pattern>' origin/main -- <paths>
```

`git grep` accepts a revision and most greps do not, which is why the habit has
to be deliberate: `grep -rn foo src/` in the primary checkout answers a question
about a retired branch and looks identical to an answer about main.

What caught it was not care, it was **disagreement**: the card being worked said
tests call that method and CI is green, so "the call always raises" could not be
true. A lone confident reading of one file has no error detection in it. When
something you read implies a shipped path is broken, prefer the hypothesis that
you read the wrong tree.

There is deliberately no guard. A hook cannot be landed by a pull request here
(`core.hooksPath` is user-global, outside every repository), and the tempting
predicate — refuse when `HEAD` is not main — would fire in every legitimate card
worktree, which is all of them. The affordable instrument is the habit above.

### This checkout is a worktree of `/thearray/git/moe-mla`. Do not use `EnterWorktree`

`/thearray/git/moe-mla-dashboard-vm` is not a sibling clone of
`/thearray/git/moe-mla`. It is a worktree of it, and the two names read as
independent checkouts while sharing one git dir:

```bash
git rev-parse --git-dir         # /thearray/git/moe-mla/.git/worktrees/moe-mla-dashboard-vm
git rev-parse --git-common-dir  # /thearray/git/moe-mla/.git
```

So anything resolving "the repository root" resolves to `/thearray/git/moe-mla`
— the live checkout every dispatch says never to create, move or delete anything
under. `EnterWorktree` does exactly that: called from here it creates
`/thearray/git/moe-mla/.claude/worktrees/<name>`, reports success, and reveals
the path only after it exists. **Do not use it in this repository.** Use the
explicit form with an absolute path, and confirm where you landed before your
first edit:

```bash
git -C /thearray/git/moe-mla-dashboard-vm worktree add \
    /thearray/git/moe-mla-dashboard-vm/.claude/worktrees/<name> -b card/<slug> origin/main
pwd   # must start with /thearray/git/moe-mla-dashboard-vm/
```

This happened on 2026-08-09. `ExitWorktree(action: "remove")` cleaned it up
completely — directory gone, absent from `git worktree list`, no stray branch —
so the incident is recoverable. Nothing tells you it needs recovering.

**The same shared git dir makes `git worktree list` a listing of two
repositories.** Entries under `/thearray/git/moe-mla/.claude/worktrees/` belong
to the live checkout and are out of scope for any sweep from here — as of
writing `qwen36-vision-key-audit`, `qwen36-vision-remap`,
`recipe-metrics-artifact` and `registries-out-of-etc`, all created 2026-08-07.
Filter the list before you act on it, rather than reading paths out of the raw
output:

```bash
git worktree list | grep '^/thearray/git/moe-mla-dashboard-vm/\.claude/worktrees/'
```

That filter also excludes the legitimate siblings named at the top of this
section — `/thearray/git/moe-mla-card-*`, `-parity-integration`,
`-dashboard-vm-wt-*`. They are worktrees of the same git dir and are nobody's to
sweep either.

There is deliberately no automated guard here, and the alternatives were weighed
rather than skipped:

- **A hook cannot be landed by a pull request.** `core.hooksPath` points at
  `/home/sirus/.config/git/hooks` — user-global, outside every repository. A
  hook committed here would never run, and installing one globally would refuse
  commits in `/thearray/git/moe-mla` itself, the checkout it exists to protect.
- **The obvious predicate is false.** "Refuse when the working tree is not under
  `moe-mla-dashboard-vm`" rejects `/thearray/git/moe-mla-card-*`,
  `/thearray/git/moe-mla-parity-integration` and the live checkout, all of which
  commit routinely through the same git dir. A guard that is wrong most of the
  time gets bypassed, and a bypassed guard is worse than none.
- **Commit and push time are both too late.** Creating the directory is the
  violation; a refusal at commit arrives once it already exists. CI cannot see
  it at all — a local worktree creation leaves nothing in the pushed tree for a
  gate to inspect.

The second-order hazard, a sweep reaching into the live checkout, is the one
worth protecting because `rm` is not recoverable. What it needs is not a guard
but a correct list: a sweep is an agent typing a command, and a check that runs
after that protects nothing. That is what the filter above is for, on the same
principle as the untracked-files command below — hand over the command, not the
warning.

### Before you remove a worktree, look for untracked files

A clean `git status` is not evidence that a worktree holds nothing. It reports
tracked files. The things most worth not losing — a handoff document, a locked
experiment document, a generated specification — arrive as untracked files and
never appear there.

```bash
git -C <wt> status --porcelain --untracked-files=all | grep '^??' | grep -v build
```

Run that on every worktree before removing it, and act on any output. "Branch
merged and tree clean" is a *weaker* claim than it sounds: it means nothing is
tracked-and-uncommitted, which is a different statement.

This is written down because it nearly went wrong. A sweep on 2026-08-09 removed
sixty worktrees, and the rule it started with would have deleted a 269-line
live-launch handoff recording an explicit user instruction; it survived only
because that tree happened to be dirty for an unrelated reason. Adding the check
partway through caught two more trees holding eighteen untracked files between
them, including four content-pinned locked documents that existed on no ref at
all. They are preserved in `docs/experiment-vm/PRESERVED_WORKTREE_ARTIFACTS.md`.

The signal is noisy in both directions — most dirty trees hold only stray build
directories — so the check is a prompt to read, not a rule that decides. Read
what the files are before concluding they are disposable, and if you conclude
they are, record the reason somewhere that outlives the worktree.

### If you do branch from unmerged work, you inherit its blockers

The escape clause above is real — building on a colleague's branch to avoid
duplicating it is sometimes right. But a commit travels between branches while
the pull request it belongs to stays behind, and everything that says "not yet"
is attached to the pull request: the draft flag, the review, the "do not merge
until X" paragraph. Your merge carries the code and leaves all of that.

This is not hypothetical. PR #92 was a draft whose body listed two conditions
that had to hold before it landed. Its commit was rebased onto another card's
branch, that branch merged main, and the work shipped in PR #99 with one
condition still unmet — it is unmet today. Neither agent did anything wrong;
the second one had no reason to read the first one's PR body.

So, if your branch contains commits you did not write:

- Enumerate them in your PR body and re-assert their blockers as your own. You
  are the one merging them.
- Better, do not rely on prose at all. **A blocker that matters belongs in a
  failing check.** If something must not ship until a condition holds, write a
  test that fails while it does not hold. A sentence in a PR description cannot
  stop a merge; a red check can, and it survives being rebased onto someone
  else's branch, which is the whole problem.

There is deliberately no automated guard here. A merge-base age check would fire
on legitimately long-lived branches, and a check for "contains another PR's
commits" is unreliable once a rebase has rewritten them. Encoding the specific
blocker as a test is the affordable mechanism, and it is the one that would have
stopped PR #99.

## Layout does NOT tell you who owns a worktree

This matters because it is easy to assume otherwise, and something already did.

The kanban autopilot Stop hook needs to answer "is one of *my* pull requests
still in CI?", so it does not nag an agent to start new work while it is waiting
on a build, and does not fall silent whenever any other agent has a PR open. It
tried to answer that by filtering `git worktree list` to paths under
`.claude/worktrees/`, on the assumption that only one agent used that location.

That assumption is false. Multiple agents now create worktrees there, so the
filter selects other agents' branches too. A hook that picks one such branch can
go quiet for a build it is not waiting on, and can fire while its own build is
still running.

**A directory path is not an ownership signal.** If you write tooling that needs
to distinguish your work from another agent's, it needs something that actually
identifies the agent — every agent pushes as the same GitHub user, so PR author
does not work either. This is genuinely unsolved here; do not paper over it with
a path filter and a hopeful comment.

### Before acting on another agent's workspace, ask the agent

Recovering a dead agent's work is right and valuable — four agents died on a
session limit on 2026-08-09 and their finished, green pull requests were the
highest-value work on the board that hour. What is not valuable is getting the
diagnosis wrong, and the diagnosis is harder than it looks.

**An agent's task-output file mtime is not a liveness signal.** It records when
the agent last *reported*, not when it last *worked*. An agent inside an 8–14
minute native build, or a five-minute pytest, writes nothing to it. An hour of
silence is two or three ordinary cycles back to back.

On 2026-08-10 a coordinator concluded an agent had died from three signals:

- its output file had not been written in sixty minutes;
- no `cmake --build`, `ninja` or `ctest` process was running;
- its branch had **zero commits**, with a worktree of uncommitted work.

  (The coordinator recorded that work as "985 insertions", from `git diff
  --stat`. The merged commit is 25 files and 3236 insertions: `git diff` does
  not count **untracked** files, and five of the new sources were untracked. A
  size read off `git diff` in a tree with new files is a subset presented as a
  total — the same shape as everything else in this section.)

It committed that work, rebased, opened a pull request and merged it. **The
agent was alive**, committed on top nine minutes later, and had a pytest running
in that worktree throughout.

Each signal is individually reasonable and all three are weak in the same
direction, which is what made them feel like corroboration:

- **No build process** was one sample of a gap that is seconds long between a
  build finishing and the next starting.
- **Zero commits** is what an agent that commits at the end looks like. Nothing
  here asks agents to commit early, so that is the expected shape, not a
  symptom.
- The output file had *already* been written down as not-a-progress-signal, and
  was used anyway because the other two agreed with it.

Three weak signals agreeing is one signal counted three times.

**The check that answers it is to send the agent a message and wait.** A live
agent replies; a dead one does not. It costs one round trip, and it is the only
check that asks the agent rather than inferring from its exhaust. If you want a
cheap pre-filter first, these beat all three signals above:

```bash
pgrep -af '<worktree-name>'                        # live processes in that tree
find <worktree> -newermt '-10 minutes' -type f \
     -not -path '*/.git/*' | head                  # a build writes objects
```

Both catch a working agent that has committed nothing and reported nothing.

Do not reach for a longer timeout instead. The native build is 8–14 minutes and
a full suite is five, so any threshold generous enough to avoid false positives
leaves a genuinely dead agent's work idle for an hour. Timeouts are the wrong
instrument here.

**What the failure costs, so the check is worth its round trip:** merging a live
agent's branch deletes its remote (`delete_branch_on_merge`), so its next push
fails with an error that does not say "this was merged". It cannot distinguish a
merged branch from a lost one, and neither can you — see the ancestry section
below for why every git-native check that would settle it silently does not work
here. Nothing was lost in the 2026-08-10 case, and that was timing rather than
process: the agent's own follow-up commit happened to ride the same merge.

This is the same family as two other hazards already recorded — a sweep removing
a worktree while an agent is working in it, and two agents editing one worktree
at once. All three are a coordinator acting on a live agent's workspace from a
bad liveness read, and all three are answered by asking first.

## Before you start a card: read its comments, and read the docs it names

Two failures cost real work on 2026-08-09, both the same shape — acting on a
description of the world instead of the world.

**A card body is a snapshot; its comments are the current state.** `card-4f3f56a2`
was dispatched against its full five-item scope while roughly 80% of it had
already merged. The body still described the original problem, three days stale.
The comments recorded that two PRs had landed and that the card's own disposition
had been *retracted*. Everything needed to avoid the wasted start was on the card
— just not in the part that gets read first. Read the comments, oldest to newest,
before claiming or delegating.

**A card's "measured, do not re-derive" block is the most dangerous part of it.**
Good cards save the next reader an investigation by recording what they measured
— and the better the block, the more it discourages checking. It is an
instruction to trust a measurement whose age you cannot see.

Four cards were found stale in this way in a single night, and one of them was
being *dispatched at the time*:

- `card-74b81095` carried four claims under "measured on origin/main, do not
  re-derive". **All four were wrong.** PR #201 had landed the half they
  described, under the same parent card, roughly three hours before the card was
  handed to anyone. The cost was bounded only because the first file the reader
  opened contradicted claim one; a less central stale claim would have had them
  re-implement a live mechanism and land it green beside the original, which is
  the "superseded" shape this file warns about elsewhere.
- `card-f027ef06`'s stated scope was "add a `WorkerRuntimeEvidence` arm to the
  oneof and a `service.cpp` handler". Both had shipped; the proto arm is field
  10 and the handler is at `service.cpp:4999`.
- `card-b2f647ee` and `card-9b360158` both describe replacing and retiring
  `src/rwkv_lab/qwen_caption_finetune.py`. `git log origin/main -- <that path>`
  returns nothing: it has never existed on main. The implementation is real and
  unmerged, so the cards are accurate about a *branch* while reading as though
  they described trunk.

So:

- **When you write a measured block, cite the sha you measured at.** "Measured
  on `origin/main` at `35e62bb6`" lets the next reader run `git log
  35e62bb6..origin/main --oneline | wc -l` and decide. "Measured on
  `origin/main`" is a claim about a moment, presented as a standing fact.
- **When you read one, re-verify the claims your work depends on** — not all of
  them, the load-bearing ones. It is cheap: these are greps.
- **"Do not re-derive" means "do not redo the reasoning", never "do not check
  the facts".** The reasoning is usually still good when the numbers have moved;
  that is the common case, and it is why the block is worth writing at all.
- **Never write a pin consequence you have not run.** "This needs a new 166th
  disposition entry, which moves `entries().size() == 165U` and the
  `reviewed_classification_digest`" appeared on a card and was wrong in both
  halves: PR #208 added `src/rwkv_lab/rwkv_optimizer_finetune.py` with **no**
  disposition entry, the catalog still holds 165, and every pin check stayed
  green. Following it would have added an entry for a file that exists only in
  this repository — a dangling path in the legacy tree, landing green and
  breaking whoever first sets `TRAINVM_LEGACY_SOURCE_ROOT`. The section below
  on adding a script explains why. The check costs seconds and needs no
  compiler:

  ```bash
  python scripts/ci_compatibility_pin_gate.py
  for c in docs/experiment-vm/source-dispositions.*.v1.json; do
    python scripts/print_disposition_digests.py "$c" --check
  done
  ```

  Run it *before* the sentence goes on the card, and the claim is either true
  or it is not written. This is worth calling out separately from the rule
  above because a pin claim reads as mechanical fact rather than as a
  measurement — nobody re-checks an assertion about a digest — and the same
  sentence will otherwise recur on every card that adds a file under
  `scripts/` or `src/rwkv_lab/`.

Suspect staleness hardest when a card names a *file path* — paths move and get
superseded — and when its parent card has other children, because a sibling may
have landed the half you are reading about.

### One command answers "do the cited paths still hold": `card_anchor_check.py`

The rule above is a convention, and conventions here have a poor record — the
merged-PR reference that 37 of 168 Done cards still ignore is the standing
example, and `card_done_pr_check.py` below is what recounts it.
Worse, "cite the sha" only reaches cards written after it, and the 200+ already
filed are the ones that will go stale. So there is also a mechanical check, and
it needs nothing added to any card:

```bash
# save the kanban_get_board JSON first, then, for the card you are dispatching:
python scripts/card_anchor_check.py --board board.json --card card-74b81095
```

It resolves every `path` and `path:line` a card body cites, against
`origin/main`, and asks two different questions:

- **Does the path exist?** Decisive, and the only thing that exits non-zero. It
  fires today on `card-b2f647ee`, which names
  `src/rwkv_lab/qwen_caption_finetune.py` — a path that has never existed on
  main — and on 32 other cards.
- **Did a cited path change after the card was written?** `created_at` is the
  measurement time every card already carries. This never fails the run, because
  a commit touching the file need not touch the claim. It is the half that
  catches `card-74b81095`: six of its nine cited paths moved between authoring
  and dispatch, and the report names PR #201 against each — the commit that
  falsified all four claims.

Four card states are printed and they are deliberately not interchangeable:
`STALE`, `DRIFTED`, `ANCHORS HOLD`, and `NO ANCHORS FOUND` / `NOTHING
CHECKABLE`. The last pair is the point. A checker that recognises nothing in a
card body reports it as healthy, so "nothing was checked" has to look different
from "everything held" — and the summary line carries the recognition rate for
the same reason. Measured across all 353 cards on this project's five boards:
**994 anchors checked across 224 cards; 129 cards cite no path it can check.**

**What it does not do, measured rather than assumed.** It does not check that a
cited line still *says* what the card claims. That was built and rejected:
binding the backticked identifiers on a card's line to the anchor on that line,
and requiring the identifier within ±3 lines of the cited line, reported 46 of
70 checkable anchors as moved — mostly prose adjacency, since a sentence naming
four symbols binds all four to the one path it also names. A check that fires on
two thirds of its input is as uninformative as one that fires on none, and it
would have taught everyone to ignore the report. So the content half of the
card's ask is unmet, and `DRIFTED` is what stands in for it: it tells you which
commits to read, not whether the sentence survived them. The weak remnant is a
cited line past the end of its file, which fires on **zero** of the 200
line-carrying anchors on the board today; that count is printed on the summary
line rather than left implied, so its silence is visible.

It reads the board, so it is not a CI gate — nothing in a pull request can see a
card. It is a dispatch-time step, run by whoever is about to hand a card over,
and it is a report rather than a decision: a card that legitimately proposes to
*create* a file cites a path that does not exist, and no tool can tell that from
a card describing a branch as though it were trunk. Read the rows.

**Read the document you are about to add to.** Three cards were filed proposing to
write down rules that were already written down: two claimed this file did not
state the base branch for new worktrees (it did, and had for some time), and one
raised a question `docs/experiment-vm/SOURCE_DISPOSITIONS.md` already answered in
its first paragraph. Each cost a full investigation to discover that nothing was
wrong. Grep the docs for the thing you are about to assert is missing.

The general form: **a finding is not real until it is checked against the thing
itself.** That is the same rule the repository already applies to its own
artifacts — a receipt names the commit it ran against, a catalog pins per-file
hashes rather than trusting a revision string — and it applies to cards and
documentation too.

## Finishing a card

- Claim before working. The board's claim state lags reality, so also check
  `git worktree list` for a branch with recent commits before starting.
- Commit, open a PR against `main`, wait for green CI, and **merge**. Work that
  is not merged is not done, whatever the card says.
- Merge with `gh pr merge <n> --squash`, and **not** `--delete-branch`. Delete
  the local branch by hand afterwards; the remote one is already gone, because
  `delete_branch_on_merge` is set on the repository.

  `--delete-branch` checks out `main` to do its local cleanup, and `main` is
  normally held by a sibling worktree — `/thearray/git/moe-mla-card-inputpipe`
  at the time of writing — so git refuses:

  ```
  fatal: 'main' is already used by worktree at '/thearray/git/moe-mla-card-inputpipe'
  ```

  **The merge has already succeeded when that prints.** The message reads as a
  failed merge, and the tempting responses both cost: re-running the merge finds
  the PR already merged and produces a different confusing error, and concluding
  the work did not land sends you off diagnosing a sync problem that does not
  exist — which is exactly what ten minutes on PR #132 were spent on, from a
  different cause. The precondition here is permanent, not a race, so every
  agent meets it eventually.
- Close the card against its own done-when, item by item. If you landed less
  than it asked for, say so and leave the card open rather than closing it.
- File a card whenever you notice something worth fixing. A finding that is not
  on the board is lost when the session ends.

### Done means merged to main, and a Done card must cite the PR that merged it

The board's Done column is not a private note about how far you got. Other
cards are written against it: `card-5020d5ca` instructed its reader to model a
new cache on "the safe owner-only inode digest cache the runtime-closure
materializer already has", because the card for that cache sat in Done. It was
not on main. It was on `integration/parity-candidate` and a handful of card
branches, and the reader spent an investigation discovering there was nothing to
read. So:

- **Move a card to Done only when its work is on `origin/main`.** Green CI on an
  open PR is not Done. Merged into an integration branch is not Done. Working
  perfectly in your worktree is certainly not Done.
- **Record `result_pr_url` when you complete it.** Under squash merging that PR
  number is the only decisive evidence the work landed — see the ancestry
  section above for why every git-native check silently fails here. A completion
  comment citing a branch-local sha is not evidence and will mislead the next
  reader exactly as it did this time.
- If the work is real but stopped short of main, say where it lives — in a
  comment, naming the branch and the commits — and leave the card out of Done.
  "Implemented at `<sha>`" is a true sentence that reads as completion; it is
  the specific phrasing that caused this.

A card in Done is a promise that `git grep` against `origin/main` will find the
thing. Anything weaker belongs in a comment.

**One command checks it: `card_done_pr_check.py`.** Like `card_anchor_check.py`
it reads a saved board rather than the repository, so it is a review-time tool
and not a CI gate — nothing in a pull request can see a card.

```bash
python scripts/card_done_pr_check.py --board board.json
python scripts/card_done_pr_check.py --board board.json --verify-merged
```

It separates three failures that need different responses, which is the whole
point of running it rather than eyeballing the column:

- **NO PR RECORDED** — Done with no `result_pr_url`. Reported, but does not
  fail the run: 37 cards are in this state and a permanently red check is one
  everybody learns to ignore. The work is probably on main; what is lost is the
  cheap way to confirm it.
- **MALFORMED** — a `result_pr_url` that is not a pull-request URL. **Worse
  than an absent one, because it reads as evidence.** Its first run against a
  real board found one: a card recording
  `https://github.com/sirus20x6/rwkv-lab/tree/dashboard/declarative-vm-fsm`,
  a *branch* URL, and pointing at the stale branch this file warns about at the
  top. A commit URL is the same trap — under squash merging the sha it names is
  exactly the thing that proves nothing.
- **NOT MERGED** — a cited pull request that is open, or closed unmerged. The
  card says done and the evidence it cites disagrees.

The last two exit non-zero; they are defects in the record rather than gaps in
it. The summary line states the population as well as the findings, because
zero findings over zero recognised cards and zero findings over 168 are
otherwise the same sentence.

Measured across this project's five boards on 2026-08-10: **37 of 168 Done
cards record no `result_pr_url`.** An earlier count put it at 56 of 126, so the
convention is followed more often than it was and is still missed about a fifth
of the time — recompute rather than quoting either number.

### Check mergeability BEFORE you read the checks

```bash
gh pr view <n> --json mergeStateStatus   # DIRTY means conflicting
```

`pull_request` workflows run against the `refs/pull/N/merge` ref, and GitHub
cannot compute that ref while the PR conflicts with its base. So a conflicting
PR gets **no workflow run at all** — not queued, not failed, absent.
`gh pr checks <n>` reports `no checks reported on the '<branch>' branch`, and
`gh api .../actions/runs?branch=<branch>` returns `total_count: 0`.

That is indistinguishable from CI not having started, and it costs in two
different ways depending on how you read it. Waiting for green means waiting
forever for a build that will never exist — 25 minutes and three pushes on one
card, including a close/reopen attempted as a re-trigger, which also did
nothing. Worse, a script that counts failing checks sees **zero failed, zero
running** and concludes the PR is ready. Absence of checks is not a state most
logic models, so it inherits whatever the default branch happens to be, and
"nothing is failing" is the tempting default.

Hence the ordering: **if `mergeStateStatus` is DIRTY, the check counts carry no
information and must not be interpreted at all.** Rebase, push, and a run
appears within about thirty seconds. Any tooling that reads check state has to
short-circuit on mergeability first; documenting only "no checks may mean
conflicting" still lets a classifier reach the wrong conclusion.

Expect this often here rather than rarely. Several agents work this board at
once and main can move every few minutes, and **every** PR that touches a
classified source conflicts on the same regenerated pin files — two PRs need
not touch the same code, only any classified source. At one point on
2026-08-09 all six open PRs were simultaneously green and conflicting, none of
them blocked by anything in its own change.

### Then check which head the verdict was computed against

`mergeStateStatus` is not the only way check state can describe a commit other
than the one you are about to merge. `gh pr view <n> --json statusCheckRollup`
can return a conclusion computed against a **different head** than the PR
currently points at, and the response carries no marker saying so.

```bash
gh pr view <n> --json headRefOid -q .headRefOid
git rev-parse origin/<branch>
```

**If those two disagree, the check counts describe the other commit and must not
be interpreted at all** — the same rule as DIRTY, for the same reason. Re-query;
it settles in seconds to minutes.

It goes wrong in both directions, and both were observed on 2026-08-09. PR #140
reported `CLEAN, running=0, failed=0` while the last commit on its branch was
fourteen seconds old; fourteen seconds later the same query returned
`headRefOid 828d3526, UNSTABLE, running=7`. The first reading described the head
from *before* the agent's push, and merging on it would have landed a commit
whose CI never ran — silently, because nothing afterwards distinguishes that
from a normal merge. In the other direction, PR #132 served a stale head for
about ten minutes after it was squash-merged: `git ls-remote` showed the ref
current the whole time, only `gh pr close` revealed the PR had already merged,
and an agent spent those minutes diagnosing a sync problem that did not exist.

Do not address this by polling harder or sleeping longer. The window was
fourteen seconds in one case and ten minutes in the other, so no timeout is the
right instrument; the OID comparison is exact and costs one command. Any tooling
that merges on check state has to compare the OID first. A cheap secondary
signal for a human reader: an "all checks complete" verdict on a branch whose
last commit is seconds old is suspect on its face, because CI cannot have
finished that fast.

### This repository squash-merges, so ancestry cannot answer "did this land"

Squash is the only strategy the repository allows. Set 2026-08-09:

```
allow_squash_merge   = true
allow_merge_commit   = false
allow_rebase_merge   = false
delete_branch_on_merge = true
```

Before that all three were enabled and nothing selected between them, so the
history is **mixed**: of the last 25 commits on main, 22 have one parent and 3
have two. Existing merge commits are not going anywhere; the setting only fixes
the shape of everything from here on.

Squash was chosen because it was already the de facto rule — six of seven PRs on
the evening this was noticed, and the overwhelming majority since — and because
it keeps main linear and readable at this PR volume. The cost is real and worth
stating plainly: **a squashed branch head is never an ancestor of main**, by
construction. That cost is paid either way; what the setting buys is that it is
now paid *uniformly*.

That uniformity is the entire point, and it is worth more than it sounds.
"Ancestry never works here" is a rule you can remember and apply safely.
"Ancestry works for some PRs and not others, with nothing on the card saying
which" is a rule nobody applies correctly — and it fails in the harmful
direction. Running `git merge-base --is-ancestor` on one of the three
merge-committed PRs returns ANCESTOR, correctly. That true positive teaches
confidence in a method that then returns a confident, wrong "absent" on the next
squashed PR you try it on. A method that is consistently useless is safer than
one that is occasionally right.

**`git cherry` and patch-id comparison cannot decide whether work reached main** —
`git cherry` reported all 10 commits of a branch absent when 9 were present.

**`git merge-base --is-ancestor <sha> main` is usable in exactly one direction.**
A **yes** is decisive: nothing makes a commit that never landed an ancestor of
main. A **no** carries zero information, because a squashed branch head is never
an ancestor by construction. So it is a free positive — try it first and fall
through to a content check when it says no. Never quote a negative as evidence
of absence.

That refinement came from the 239-card sweep: the instruction it was given said
flatly never to use ancestry, and it worked out the asymmetry itself and applied
it correctly to 173 citations. The blunter rule was wrong in the cheap direction,
which is the direction worth fixing.

Use instead, in order of strength:

1. **A merged PR number** — `gh pr view <n> --json state`. Decisive under any
   strategy, needs no judgement. When a card records completion it should cite
   this, or the merge commit on main, never a branch-local sha.
2. **A distinctive identifier from the diff, grepped against `origin/main`.**
   Take the identifier from `git show <sha> -- <path> | grep '^+'`, never from a
   card's prose description of the change — prose paraphrases, and grepping an
   invented name returns a clean-looking zero.
   Guard the result by also grepping a subsystem-level term that would survive a
   refactor. If the neighbourhood is present in quantity and your identifier is
   zero, the zero means something. If the neighbourhood count has *grown* past
   what the commit itself contained, the work was **superseded under another
   name** — porting it would duplicate a live mechanism, not restore a lost one.
3. **Added file paths** — only for commits that add files, and it fails
   optimistically: one commit had all five of its files on main and none of its
   mechanism.

Before porting anything the second check says is missing, ask a third question:
**is that file still the right place?** `git log origin/main --oneline -- <path>`
returning nothing means the path has **never existed on main in its entire
history** — not that it was deleted, which is the natural reading and the wrong
one. `card-88d6b252` cited a commit touching three files; two were a Python
qualification runner that main never carried, because main serves that path from
a Go handler instead. Its one real change was already on main in stronger form,
under a named constant. A file that was never there cannot be where the fix
belongs, and a commit can be simultaneously absent, unportable, and already
done.

The cost of not having this written down was five days: three cards recorded
work as complete, citing shas, that had never reached main, and nobody noticed
because the obvious check silently does not work here.

### A branch that is "ahead" is usually stale, not stranded

The checks above answer "did this commit land". This one answers the question
that usually comes first and is easier to get wrong: **is there anything on
this branch worth landing at all?**

`git rev-list --count origin/main..<branch>` is the tempting instrument and it
is worthless here, for the same reason ancestry is. A branch that merged main
and then had its own work land through a squashed pull request reports commits
ahead **forever**. "ahead=5" is not five pieces of work; it can be three merge
commits and two commits that shipped weeks ago.

`git diff origin/main...<branch>` — **three dots** — is worse, because it looks
like a diff against main and is not. It diffs from the *merge-base*, so on a
branch that last merged main long ago it describes a tree that no longer
exists, and it does so in exactly the format you would use to review a change.

**Use two dots, and read the deletion count before anything else:**

```bash
git diff origin/main <branch> --shortstat
```

A branch with real unlanded work shows insertions and few deletions. A stale
one shows a number like these, all measured on 2026-08-10:

| branch | reported "ahead" | `git diff origin/main <branch>` |
| --- | --- | --- |
| `integrate` | 5 | 184 files, 2,787 insertions, **36,948 deletions** |
| `card/qwen36-declarative-migration` | 1 | 408 files, 4,961 insertions, **91,130 deletions** |

Neither had anything unlanded. Both commits on `integrate` were already on main
— one as `eac8fc4c` via PR #169, the other byte-identical, confirmed with `git
diff --quiet origin/main HEAD -- <path>`. `card/qwen36-declarative-migration`'s
single commit shipped through PR #99. **Merging either would have reverted tens
of thousands of lines of main**, cleanly, including most of the gate scripts
listed earlier in this file. That is the `1c91acd` hazard reached from a
different direction: the branch is not carrying work, it is carrying an old
tree, and git will apply it without complaint.

Three related instruments failed in the same hour, all in the flattering
direction:

- **A same-path existence check**, defeated by a moved file.
  `hf-multimodal-sft.recipe-profiles.v1.json` reads as absent at
  `docs/experiment-vm/` and is present at `docs/experiment-vm/examples/`.
  Search the basename across the tree, not the path.
- **An identifier grep with no neighbourhood guard.**
  `transformer_mla_evaluation_slots` is genuinely absent from main — and so is
  every other `*_evaluation_slots` symbol, so the zero was guaranteed before
  the question was asked. The guard the section above prescribes is not
  optional; when the neighbourhood count is **zero**, your identifier proved
  nothing at all.
- Reading the code settled it: main wraps that composition in
  `with_evaluation_suite()`, a shared helper **ten** compositions use. The
  commit added four slots inline for one family; main generalised them across
  ten. Superseded under another name, with the neighbourhood grown past what
  the commit contained.

Four "stranded work" findings were filed from count-based instruments that
night and **three were false**. The one that was real looked identical from the
outside. So do not treat the two-dot diff as a formality to skip when a count
already told you what you expected to hear — the count is what will be wrong.

### Before you cherry-pick anything a card calls "stranded"

A sweep of all 239 cards on 2026-08-09 found 173 citing a sha: 125 present on
main, 19 stranded, 16 partially present, and **13 superseded** — the identifier
absent from main while its subsystem is present and *larger*. Superseded is the
dangerous verdict, because the port applies cleanly, CI stays green, and a
weaker mechanism lands beside the working one.

Two numbers from that sweep are worth carrying:

- An automated line-matching pass triaged all 173. Hand review of the 44
  ambiguous ones **overturned 31 — 70% — in both directions.** Line-matching
  triages; it cannot conclude.
- 56 of 126 Done cards recorded no merged-PR reference at the time of that
  sweep — 37 of 168 today — so for those the citation is the only trail and
  it is the unreliable kind. `card_done_pr_check.py` reports the current set.

**`1c91acd` must never be cherry-picked.** It is cited on two cards that both
read as done. It sets, on `deploy/trainvm-hostd.service`:

```ini
CapabilityBoundingSet=CAP_BPF CAP_DAC_OVERRIDE CAP_DAC_READ_SEARCH CAP_KILL \
  CAP_PERFMON CAP_SETGID CAP_SETUID CAP_SYS_ADMIN CAP_SYS_RESOURCE
AmbientCapabilities=(the same nine)
```

Main deliberately carries both as **empty**, with `NoNewPrivileges=yes` and the
comment *"Nothing left needs a privilege transition, and this makes the daemon's
own guarantee match the one it demands of its workers."* Restoring that commit
would re-privilege a daemon that was intentionally stripped — and it ships its
own passing test (`test_hostd_can_drop_worker_credentials_inside_its_capability_bound`)
which asserts those capabilities are *present*, so the regression arrives green.

That is the general shape to fear: **a superseded commit brings its own tests,
and those tests encode the old invariant.** A green suite after a port proves
the port is self-consistent, not that it is wanted. Before porting anything that
touches privileges, sandboxing, or a security boundary, diff the *current* file
against the commit and read what main says about why it looks the way it does.

`card-ab464dc6` holds the full do-not-port hazard list; `card-2ae29669` tracks
making the merged-PR reference structural instead of a prose convention.

## Running the gates locally

```bash
python scripts/ci_coverage_gate.py -m "not gpu"
python scripts/ci_unwired_module_gate.py
python scripts/ci_contract_caller_gate.py
python scripts/ci_catalog_doc_counts_gate.py
python scripts/ci_step_zero_arming_gate.py
python scripts/ci_gpu_observation_gate.py
python scripts/ci_native_host_path_gate.py
python scripts/ci_experiment_adapter_gate.py
python scripts/validate_benchmark_matrix.py
python scripts/validate_experiment_documents.py
python scripts/validate_native_ci_exclusions.py
python scripts/ci_compatibility_pin_gate.py
for c in docs/experiment-vm/source-dispositions.*.v1.json; do
  python scripts/print_disposition_digests.py "$c" --check
done
```

Each ends with a line stating `PASSED` or `FAILED`. Read that line — the older
form printed a neutral tally that looked identical either way, and a PR was
pushed red because its output was truncated to exactly that line.

### None of these resolves a training composition

Every gate above answers a question about a **document**. If you changed a
training composition, none of them has told you it works, and two of them are
close enough to the question to feel like they did.

The four relationship validators live at `training_component_registry.cpp:847`:

```cpp
validate_model_trainability_relationships(resolved);
validate_data_pipeline_relationships(resolved);
validate_evaluation_checkpoint_relationships(resolved);
validate_optimizer_decay_relationships(resolved);
```

They run inside `resolve_composition`, and `resolve_composition` is reached from
exactly one production site — `trainvm/src/service.cpp:3457`, the run-authoring
service — plus four test files. `recipe inspect` is not among its callers and
neither is `ci_step_zero_arming_gate.py`. **The only thing that runs those four
validators outside the service is ctest.**

So `recipe inspect` printing `VALID` means the catalog document is well-formed.
The arming gate printing `PASSED` means a shipped composition names this route.
Neither means the composition resolves, and a composition that does not resolve
fails at authoring time, which is after everything cheap has already gone green.

This is not a hypothetical ordering of concerns. On 2026-08-10 a `model_loader`
slot was reported as landed and measured-to-work on exactly those two green
readings; the native suite refused it, because
`validate_model_trainability_relationships` rejects a model loader with no
trainability policy beside it. The claim had to be retracted. Worse, the
tempting way to make it pass was to declare `trainability.full.v1` — false for
that family, which sets `requires_grad` per profile inside `train_mla.py`. **A
cheap green reading pointed straight at a decorative declaration**, which is the
hazard this file warns about in three other places.

The check is to build and run the suite that resolves it:

```bash
ctest --test-dir trainvm/build -R \
  'training_component_registry_tests|rwkv_lab_worker_contract_tests|adapter_invocation_tests|author_run_service_tests'
```

Those are the four suites that call `resolve_composition`, and they are four
separately registered ctest targets rather than one — a selector naming only
the first two runs half of them and still prints a confident green, which is
the same failure one level down. `ctest` ends with `N tests passed ... out of
N` — read that N and confirm it is four before believing the result, because a
selector that matches fewer suites than you meant looks identical to one that
matches all of them.

That costs a GCC 16 build, which is the reason it gets skipped. Skip it for a
catalog typo if you like; do not skip it for a change to what a composition
*contains*. The general rule this instance belongs to: a gate tells you what
the gate checks, and the distance between that and what you want to know is
invisible in its output — every one of them prints the same `PASSED`.

### Mutation testing: the baseline row is not a formality

There is no mutation-testing harness in this repository — the practice is to
write one per card: edit a source, run the tests, restore it, repeat. Whatever
you write, run the unmutated tree first and **read the baseline row before
anything else**. A red baseline means the tree is not what you think it is, and
every red count printed after it is measured against a reference you do not
understand; the interesting rows are not interesting, they are noise.

That row is the only instrument that catches a corrupted working tree, and it
has done it twice in one night. One agent's failure parser was reporting
`red=0` for every mutation, which a broken harness and a perfect test suite
produce identically — the baseline distinguished them. Another had had an
earlier harness run killed mid-mutation, leaving a mutated file on disk that the
pin refresh above then certified with every gate green; the next run's red
baseline is what surfaced it. If the baseline is red, stop, clean the tree, and
re-establish it before interpreting a single mutation result.

Making harnesses signal-safe — trapping SIGTERM/SIGINT to restore the file — was
weighed and deliberately not adopted. It narrows the window without closing it
(SIGKILL and the OOM killer take no handler), and a harness advertising a
restore invites exactly the trust the baseline check exists to withhold. Treat
restore as best-effort; the baseline row is the check.

### Adding a native module now costs a caller or a stated reason

`ci_unwired_module_gate.py` fails when a header under `trainvm/include/trainvm/`
is reached by no translation unit in `trainvm/src/` other than its own `.cpp`.
So a new module lands red until something in production includes it — directly
or through another header, the traversal is transitive — or until
`docs/experiment-vm/unwired-module-exclusions.v1.json` says why not.

That is deliberate friction and it is affordable, because landing the module
ahead of its consumer is often right. What is not affordable is doing it
silently: four modules shipped fully implemented, fully tested and green with
zero production callers, and every one was found by a card sweep months later.
An allowlist entry costs one sentence and turns that into a recorded decision.

The allowlist is a countdown. An entry whose module later gains a production
includer **fails**, so it can only shrink — write the entry expecting to delete
it.

### The Python gate asks the same question of the adapter contract only

`ci_contract_caller_gate.py` fails when a route the native adapter registry
advertises dispatches to a handler nothing reaches from the worker's
entrypoint, or when a module in `src/rwkv_lab/trainvm_adapters/` is reached by
nothing from it. It has **no allowlist**, because the honest count over that
population is zero.

Do not widen it to `__all__`. That was measured — 79 of 117 `trainvm_worker`
exports and 17 of 18 `trainvm_adapters` exports have no in-tree importer — and
it is not a backlog: `trainvm_worker` is a published SDK whose consumers sit
outside this repository, and `src/rwkv_lab` is a lever library where a
test-only lever is the tree working as designed. A gate over that surface needs
a ~96-entry allowlist, which is an instrument tuned until the number looks
comfortable. card-d198cc09 holds the measurement and the two candidate
questions that are still unexplored.

## Content pins, and how to refresh them

Editing a classified source moves a content digest, and several are pinned in
more than one place. None of them should be recomputed by hand.

There are **four** pin sites, and they are not next to each other. Refresh them
in this order — later ones are computed from the values you just wrote, so out
of order you will pin a digest of a digest that is about to change:

**Before step 1, confirm the tree holds only what you meant to pin.** `git status
--porcelain` and `git diff origin/main`, read rather than skimmed, cost seconds
and are the only opportunity you get. A refresh pins the bytes that are on disk
when it runs, so anything else sitting in the tree gets pinned too, and the
result is indistinguishable from a correct one: a digest over stray content and
a digest over the content you meant are both 64 hex characters, and nothing in
this repository can look at either and tell you which it has. The refresh will
succeed, `validate-catalog` will report valid, `ci_compatibility_pin_gate.py`
will pass and the disposition `--check` loop will pass — all of them, because
every one of them asks "do these bytes match this digest?", and they do.

The route that produced this was an in-place mutation harness — anything that
edits a source, runs the tests, and restores it. One was killed by a tool
timeout mid-mutation, leaving `src/rwkv_lab/trainvm_adapters/handlers.py`
carrying the mutation that was live at the kill. `git status` showed the file
modified, which it already legitimately was from the card's own work, so nothing
looked wrong; the four steps then regenerated
`compatibility-workflows.v1.json`, `source-dispositions.rwkv-lab.v1.json` and
`kReviewedCatalogDigest` as digests of mutated source, and every gate agreed.
**Never regenerate pins in a tree that an in-place harness has run in without
first re-establishing that it is clean** — `git checkout` the paths the harness
touched, or do the refresh in a fresh worktree. This is the same failure as a
hand-resolved pin conflict below ("a plausible-looking digest that matches
nothing"), reached from the other direction.

```bash
# 1. Build first. `trainvm/build/trainvm` is gitignored, so whatever is sitting
#    there is whatever your last build left — `git pull` never refreshes it.
cmake -S trainvm -B trainvm/build -G Ninja -DCMAKE_BUILD_TYPE=Debug
cmake --build trainvm/build -j "$(nproc)" --target trainvm

# 2. The compatibility catalog's per-source pins and its classification
#    surface digest, plus any empty classification surface. --write splices
#    them back in; nothing else in the document moves. There are 155 pins, so
#    do not hand-edit them.
trainvm/build/trainvm print-catalog-digests \
  docs/experiment-vm/compatibility-workflows.v1.json . --write

# 3. kReviewedCatalogDigest, in trainvm/src/compatibility_catalog.cpp. It is
#    computed FROM the values step 2 just wrote, so it only settles after it.
#    This command fails on purpose and names the value to pin.
trainvm/build/trainvm validate-catalog \
  docs/experiment-vm/compatibility-workflows.v1.json .

# 4. The source-disposition catalogs: the per-source hashes, and only those.
#    The tree digest is no longer stored anywhere; it is computed from entries
#    at load time. A catalog that still declares one is rejected.
for c in docs/experiment-vm/source-dispositions.*.v1.json; do
  python scripts/print_disposition_digests.py "$c" --write
done
```

**You do not need any of this to find out whether you owe a refresh.** Both
catalogs' per-source pins have a seconds-fast check that needs no compiler, and
both run in CI's schema job ahead of the native build:

```bash
python scripts/ci_compatibility_pin_gate.py          # compatibility-workflows
for c in docs/experiment-vm/source-dispositions.*.v1.json; do
  python scripts/print_disposition_digests.py "$c" --check
done
```

`ci_compatibility_pin_gate.py` names the drifted path in the same sentence the
binary uses (`compatibility source <path> does not match its pinned bytes`), so
the same grep finds either report. It checks the per-source pins **only** —
neither the tree digest nor `classification_surface_digest` is recomputed in
Python, because a mirror of a C++ fold is a second implementation to keep in
agreement, and step 2 above is still what regenerates all three. Running it
tells you whether you owe a refresh; it does not perform one, and a green run is
not a substitute for the four steps when you have edited a classified source.

Step 4 usually stops there now. The two digests pinned in
`trainvm/tests/source_disposition_catalog_tests.cpp` cover the *classification*
with every `source_sha256` erased, so ordinary source edits no longer move them
— that is deliberate, and it is why two pull requests editing different files
in the same scope no longer collide. If one of them does move, something in a
catalog's review content changed: read the diff before repinning. Rebuild and
run `ctest --test-dir trainvm/build -R source_disposition_catalog`; it prints
each computed digest beside its name, which is how you pin them without
guessing.

### Pin conflicts on a rebase are mechanical — never resolve them by hand

This used to be constant. Every PR touching a classified source rewrote the
same single-line pins, so two PRs conflicted whether or not their work
overlapped: `source_tree_digest` in each catalog, and the whole-document digests
pinned in `source_disposition_catalog_tests.cpp`. Both are gone — the tree
digest is derived rather than stored, and the native pins now cover the
classification with the per-file hashes erased. What is left in the catalogs is
the `entries` list, where two changes touch two different objects and git merges
them without help.

`compatibility-workflows.v1.json` no longer stores a whole-tree digest either.
Its byte binding is `source_digests`, one object per referenced path, so two
pull requests editing two different sources rewrite two different objects and
git merges them. A catalog that still declares a `source_tree_digest` is
refused by name — that is what a hand-resolved rebase produces.

What is left that a merge can still collide on is `kReviewedCatalogDigest` in
`compatibility_catalog.cpp`, and it only moves when a *classification* moves.
When it does conflict, do **not** resolve that hunk by editing it or by picking
a side. A pin is a digest of content: whichever side you
choose was computed against a tree that no longer exists, so both sides are
wrong after the merge, and the result is a plausible-looking digest that
matches nothing.

Take the base version and regenerate:

```bash
git checkout --theirs docs/experiment-vm/compatibility-workflows.v1.json \
                      docs/experiment-vm/source-dispositions.*.v1.json
# then re-run the four numbered steps above
```

(During a rebase `--theirs` is the commit being replayed and `--ours` is the
new base; check which you have before trusting either.) The pinned digests in
`trainvm/tests/source_disposition_catalog_tests.cpp` are recovered the same way
— take one side, then let the ctest print the two correct values beside their
names.

A conflict in `compatibility-workflows.v1.json` itself should now be rare. If
you get one, read it rather than regenerating past it: two changes to the same
entry, or to the same source's pin, is a real overlap.

There is deliberately no single `refresh-all` script yet. Writing one is filed.

**Do not skip the build**, and do not assume a checkout that already has
`trainvm/build/trainvm` has a current one. This is a real trap, not a caution:
the binary in the primary checkout was months old and predated
`print-catalog-digests` entirely, so it answered with a usage dump — which reads
as "you typed the command wrong", not "your binary is old". A slightly newer
stale binary is worse: it knows the subcommand, refuses the catalog's schema,
and sends you off debugging a catalog that is fine. Either way the digest you
are chasing never appears. The build takes several minutes and needs GCC 16
with `-freflection`; that is expected.

Two things to know before you accept a regenerated value:

The compatibility catalog's per-source pins moving is mechanical — regenerate
them with `--write` and move on. (Neither artifact stores a tree digest now;
both derive it from per-file hashes at load time.)
But if
`classification_surface_digest` also moves, that is a **real** signal: something
changed a file's entrypoint, argument surface, or checkpoint/resume call sites.
Read the catalog entry before bumping `kReviewedCatalogDigest`; that bump is the
review the gate exists to force. It stayed put across an eight-commit port and
moved for a one-module one, so it does discriminate.

That discrimination is **Python-only**, which is not obvious and will otherwise
surprise you mid-card. The extractor keeps only classification-bearing lines
from a Python source; for every other referenced file — `trainvm/src/main.cpp`
is one — the surface is the whole file. So a comment or a help string in
`main.cpp` moves `classification_surface_digest` exactly as far as a new
subcommand would, and you will be bumping `kReviewedCatalogDigest` for it. That
is a property of the extractor, not a signal about your change; say which it was
in the commit message so the next reader does not have to re-derive it.

`trainvm/tests/source_disposition_catalog_tests.cpp` holds **two** pins, scripts
and RWKV. A regex on the first `sha256` in that file changes the wrong one. The
test catches it only because it prints each computed digest beside its name.

That file also runs the only check comparing
`scripts/print_disposition_digests.py` against the C++ it mirrors: it drives
both over the same entries and requires the same answer. Nothing else compares
them, so if you edit the Python fold, the native suite must run. It will —
`classify_native_ci_changes.py` lists that script under `FULL_FILES` for exactly
this reason, because `scripts/` otherwise selects the catalog tier and never
reaches ctest.

The native suite needs GCC 16 with `-freflection`:

```bash
cmake -S trainvm -B trainvm/build -G Ninja -DCMAKE_BUILD_TYPE=Debug
cmake --build trainvm/build -j "$(nproc)"
ctest --test-dir trainvm/build -j 4 --output-on-failure
```

`trainvm/build/` is gitignored; other build directory names are not, and
`git add -A` will happily commit several gigabytes of object files.

### Adding a script to `scripts/` costs nothing today

This file used to claim the opposite here — that `scripts/` is an exhaustively
enumerated disposition scope, so *adding* a script to it trips the gate and
moves three pinned counts in `source_disposition_catalog_tests.cpp`. Paying that
is still welcome, but nothing requires it, and believing otherwise has already
made agents restructure work around a gate that was not there.

The enumeration is a property of the **legacy** `/thearray/git/moe-mla` checkout
at the revision each catalog pins, not of this repository. Measured against
`origin/main` at `3d39122`:

- `source-dispositions.scripts.v1.json` pins `git-sha1:bae678b9`. Its entries
  were *exactly* the top-level `scripts/*.{py,sh}` files in the legacy tree at
  that revision — set equality, zero dangling entries — when that was last
  measured at 128 entries. It now holds **129**, so one entry has been added
  since and the set-equality claim against the legacy tree has not been
  re-checked. The same scope in this repository holds **160** files, so
  **31 are unclassified**, among them `scripts/acceptance.sh` and
  `scripts/generate_trainvm_proto.sh`.
- `source-dispositions.rwkv-lab.v1.json` has the same shape: **165** entries
  against **173** files in scope here, so **8 are unclassified**.
- `source-dispositions.dashboard.v1.json` pins a different revision and is
  complete for this tree.

Those figures were 128/146/18 and 165/171/6 when this section was written, and
nothing reports the drift — so recompute rather than quoting them. Both scopes
are declared in the documents themselves (`source_scope`, and note
`recursive: false` for both, so subdirectories are out of scope and a recursive
count overstates the gap badly):

```bash
python - <<'PY'
import json, pathlib
for name in ("scripts", "rwkv-lab", "dashboard"):
    d = json.loads(pathlib.Path(
        f"docs/experiment-vm/source-dispositions.{name}.v1.json").read_text())
    scope = d["source_scope"]; root = pathlib.Path(scope["prefix"])
    exts = set(scope["extensions"])
    walk = root.rglob("*") if scope.get("recursive") else root.glob("*")
    files = [f for f in walk if f.suffix in exts and f.is_file()]
    print(name, "entries", len(d["entries"]), "files", len(files),
          "unclassified", len(files) - len(d["entries"]))
PY
```

**The scripts gap grew from 18 to 31 while one entry was added.** That is the
direction to watch: the unclassified count is not a backlog draining down, it
rises every time a card adds a script, which is most of them.

Nothing reports those gaps. `print_disposition_digests.py --check` only walks the
entries the document lists and hashes the files they name, so a file with *no*
entry is invisible to it. The C++ loader does enumerate the scope and reject
missing paths (`enumerate_scope` in `source_disposition_catalog.cpp`), but only
when `load_file` is handed a repository root — and
`source_disposition_catalog_tests.cpp` guards that call behind
`getenv("TRAINVM_LEGACY_SOURCE_ROOT")`, which is unset in CI. Confirmed from the
other side as well: an uncatalogued script dropped into `scripts/` left all three
`--check` runs, `ctest -R source_disposition_catalog`, and `validate-catalog`
green.

So **what binds in this repository is the content pins, not the enumeration.**
Editing a classified source moves its digest and you must refresh the pins as
above. Adding a new source is unclassified and ungated. The two halves point at
different trees, and you can see the seam: 127 of the 128 scripts pins equal the
legacy blob at `bae678b9`, while `scripts/supervisor_night.sh`'s equals this
repository's copy — because the pins get regenerated here and the entry list
never does.

Closing the gap is real work rather than an oversight to tidy up in passing. One
catalog cannot be complete for two trees at once, so an entry for a file that
exists only here would break the `TRAINVM_LEGACY_SOURCE_ROOT` path the moment
anyone sets it; it needs the live check parameterised by which root it validates,
plus 24 honest classifications. That is filed as its own card. Until it lands, do
not budget for a scripts-enumeration tax, and do not rely on the gate to catch an
unclassified script.

## Do not `pip install -e .` on the training host

`pyproject.toml` declares `transformers>=4.57`. The host runs **4.52.4** — below
its own package's floor. That is not drift to tidy up in passing, and running
the install that would "fix" it changes model behaviour.

`fla/models/utils.py` ends like this, byte-identically in flash-linear-attention
**0.4.1 and 0.5.2** (both wheels read, not inferred):

```python
_TF_VERSION = transformers.__version__
_NEED_NEW   = "4.53.3"
...
if version.parse(_TF_VERSION) > version.parse(_NEED_NEW):
    class Cache(FLACache): ...        # layers-based, per-layer token tally
else:
    class Cache(LegacyFLACache): ...  # self.states, scalar _seen_tokens
```

So **`transformers` 4.53.3 is a behavioural switch inside a dependency the
models use**, and the host sits one minor version below it. `pip install -e
'.[...]'` honours the declaration, upgrades past 4.53.3, and from that moment
`fla.models.utils.Cache` is a different class with different state semantics —
in production, on the machine that runs training. Nothing warns and nothing
fails; the install command that looks like routine maintenance is the one that
flips it.

This also explains a discrepancy that otherwise looks like flakiness: the
`fla`-marked test in `tests/test_vision_loop.py` **passes here and fails in
CI**, on the same commit. CI installs from the declaration, lands above the
threshold, and gets the other class. The host passes *because* it is below the
floor. Neither environment is misconfigured relative to itself.

**The version to check is `transformers`, not `flash-linear-attention`.** The
fla version does not enter into it — the threshold constant is the same in both
releases — and an afternoon was spent pinning fla versions against a symptom
that fla does not control.

```bash
python -c "import transformers, fla.models.utils as u; \
print(transformers.__version__, u.Cache.__mro__[1].__name__)"
```

That prints the version and which implementation you actually got —
`LegacyFLACache` or `FLACache`. Run it before concluding anything about cache
behaviour. Do not try to answer it by grepping fla's source for `class Cache`:
the declaration is indented inside that `if`, so an anchored grep returns
nothing and reads as "the symbol does not exist", which is a confident zero that
was never capable of returning anything else.

Whether the declaration should move — the standing proposal is
`transformers>=4.52,<4.53.3`, the ceiling being the load-bearing half — is
recorded on `card-2757f59e` and is the user's call, because it governs the
environment they train in. Until it is made, **leave the host's `transformers`
alone**, and install into a throwaway virtualenv if you need a different one.

## C++ diagnostics: trust them only after you have configured a build

The editor's C++ diagnostics come from clangd, which is **clang**. The build is
**GCC 16 with `-freflection`**. They are different compilers, so what clangd
tells you about this tree is an approximation, and it has already been wrong in
the most expensive way: confident, specific errors on source that compiles
clean and passes ctest. An agent that "fixes" one of those damages working code.

**Configure a build first, in every new worktree:**

```bash
cmake -S trainvm -B trainvm/build -G Ninja -DCMAKE_BUILD_TYPE=Debug
```

Configuring is enough — it writes `trainvm/build/compile_commands.json`, which
is what clangd reads. You do not have to finish the compile to get a usable
index. Until that file exists clangd has no include path and no `-std`, falls
back to bare C++17, and emits a wall of pure fiction: `'trainvm/....hpp' file
not found`, `no member named 'ranges' in namespace 'std'`, `use of undeclared
identifier 'nlohmann'`, and finally implicit-int fallout such as `'ostream'
(aka 'int')` and `cannot initialize return object of type 'int' with an rvalue
of type 'std::nullptr_t'`. Measured on `trainvm/src/run_authoring.cpp`: 100
diagnostics, none of them real. **`compile_commands.json` is gitignored and per
worktree — a fresh worktree always starts in this state.**

`trainvm/.clangd` handles the rest: it strips `-freflection` from the recorded
GCC command (clang's driver hard-errors on unknown `-f` flags), supplies
`-std=c++26 -I../include` as a fallback so an unconfigured tree is merely
incomplete rather than actively lying, and turns off clangd's include-cleaner,
which cannot see a header that is reached only through `std::meta` or `^^T` and
so proposes deleting includes the GCC build needs. That file configures the
indexer only.
Do not "fix" an indexer complaint by editing `trainvm/CMakeLists.txt` — the
build is correct and the flags there are load-bearing.

### The one error that is expected and must not be "fixed"

```
in included file: no member named 'meta' in namespace 'std'
```

reported against the first `#include` line of the ~46 translation units that
reach `trainvm/include/trainvm/reflection_json.hpp`, directly or through
`trainvm/document.hpp`.

This is real and unfixable here. clangd 22 does not implement P2996 static
reflection at all — it has no `-freflection`, no `__cpp_reflection`, and its
`<meta>` defines nothing — so `std::meta::identifier_of`, `^^T`, and
`std::define_static_array` cannot resolve for it. GCC 16 compiles them fine.

It is left visible rather than suppressed on purpose: clangd can only suppress
by diagnostic name, and `no_member` is exactly the name a genuine
"no member named X" bug would carry, so silencing it would cost more than it
saves. **Do not delete the `#include` to make it go away, and do not rewrite
reflection code to avoid it.** Verify any suspected C++ defect against
`cmake --build trainvm/build` before believing it.

After configuring, each of `run_authoring.cpp`, `training_component_registry.cpp`,
`recipe_profile_tests.cpp` and `training_component_registry_tests.cpp` reports
that single diagnostic and nothing else. Check a file yourself with
`clangd --check=<path>`; anything beyond that one line is either a real defect
or a sign that the build directory went stale — reconfigure before concluding
which.
