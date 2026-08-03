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
otherwise.

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

## Finishing a card

- Claim before working. The board's claim state lags reality, so also check
  `git worktree list` for a branch with recent commits before starting.
- Commit, open a PR against `main`, wait for green CI, and **merge**. Work that
  is not merged is not done, whatever the card says.
- Close the card against its own done-when, item by item. If you landed less
  than it asked for, say so and leave the card open rather than closing it.
- File a card whenever you notice something worth fixing. A finding that is not
  on the board is lost when the session ends.

## Running the gates locally

```bash
python scripts/ci_coverage_gate.py -m "not gpu"
python scripts/validate_benchmark_matrix.py
python scripts/validate_experiment_documents.py
python scripts/validate_native_ci_exclusions.py
```

Each ends with a line stating `PASSED` or `FAILED`. Read that line — the older
form printed a neutral tally that looked identical either way, and a PR was
pushed red because its output was truncated to exactly that line.

The native suite needs GCC 16 with `-freflection`:

```bash
cmake -S trainvm -B trainvm/build -G Ninja -DCMAKE_BUILD_TYPE=Debug
cmake --build trainvm/build -j "$(nproc)"
ctest --test-dir trainvm/build -j 4 --output-on-failure
```

`trainvm/build/` is gitignored; other build directory names are not, and
`git add -A` will happily commit several gigabytes of object files.
