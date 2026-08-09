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
- Close the card against its own done-when, item by item. If you landed less
  than it asked for, say so and leave the card open rather than closing it.
- File a card whenever you notice something worth fixing. A finding that is not
  on the board is lost when the session ends.

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

**So do not use any of these to decide whether work reached main:**

- `git merge-base --is-ancestor <branch> main`
- `git cherry` — it reported all 10 commits of a branch absent when 9 were present
- patch-id comparison

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
- 56 of 126 Done cards record no merged-PR reference at all, so for those the
  citation is the only trail and it is the unreliable kind.

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
python scripts/validate_benchmark_matrix.py
python scripts/validate_experiment_documents.py
python scripts/validate_native_ci_exclusions.py
for c in docs/experiment-vm/source-dispositions.*.v1.json; do
  python scripts/print_disposition_digests.py "$c" --check
done
```

Each ends with a line stating `PASSED` or `FAILED`. Read that line — the older
form printed a neutral tally that looked identical either way, and a PR was
pushed red because its output was truncated to exactly that line.

## Content pins, and how to refresh them

Editing a classified source moves a content digest, and several are pinned in
more than one place. None of them should be recomputed by hand.

There are **four** pin sites, and they are not next to each other. Refresh them
in this order — later ones are computed from the values you just wrote, so out
of order you will pin a digest of a digest that is about to change:

```bash
# 1. Build first. `trainvm/build/trainvm` is gitignored, so whatever is sitting
#    there is whatever your last build left — `git pull` never refreshes it.
cmake -S trainvm -B trainvm/build -G Ninja -DCMAKE_BUILD_TYPE=Debug
cmake --build trainvm/build -j "$(nproc)" --target trainvm

# 2. The compatibility catalog's two digests, plus any empty classification
#    surface. Paste both into docs/experiment-vm/compatibility-workflows.v1.json.
trainvm/build/trainvm print-catalog-digests \
  docs/experiment-vm/compatibility-workflows.v1.json .

# 3. kReviewedCatalogDigest, in trainvm/src/compatibility_catalog.cpp. It is
#    computed FROM the value you just pasted, so it only settles after step 2.
#    This command fails on purpose and names the value to pin.
trainvm/build/trainvm validate-catalog \
  docs/experiment-vm/compatibility-workflows.v1.json .

# 4. The source-disposition catalogs: per-source hashes and the tree digest.
for c in docs/experiment-vm/source-dispositions.*.v1.json; do
  python scripts/print_disposition_digests.py "$c" --write
done
```

If step 4 changed the scripts or RWKV catalog, rebuild and run
`ctest --test-dir trainvm/build -R source_disposition_catalog` — it prints each
computed digest beside its name, which is how you pin the fifth and sixth values
in `trainvm/tests/source_disposition_catalog_tests.cpp` without guessing.

### Pin conflicts on a rebase are mechanical — never resolve them by hand

Because every PR touching a classified source rewrites the same pin files, a
rebase conflicts in them constantly. Do **not** resolve those hunks by editing
them or by picking a side. A pin is a digest of content: whichever side you
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

Two things to know before you paste a new value in:

`source_tree_digest` moving is mechanical — regenerate it and move on. But if
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

- `source-dispositions.scripts.v1.json` pins `git-sha1:bae678b9`, and its 128
  entries are *exactly* the 128 top-level `scripts/*.{py,sh}` files in the legacy
  tree at that revision — set equality, and zero dangling entries. The same
  scope in this repository holds 146 files, so **18 are unclassified**, among
  them `scripts/acceptance.sh` and `scripts/generate_trainvm_proto.sh`.
- `source-dispositions.rwkv-lab.v1.json` has the same shape: 165 entries against
  171 files in scope here, so 6 are unclassified.
- `source-dispositions.dashboard.v1.json` pins a different revision and is
  complete for this tree.

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
