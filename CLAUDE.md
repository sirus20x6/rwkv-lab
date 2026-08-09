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

```bash
# The compatibility catalog: both digests, plus any empty classification surface.
trainvm/build/trainvm print-catalog-digests \
  docs/experiment-vm/compatibility-workflows.v1.json .

# The source-disposition catalogs: per-source hashes and the tree digest.
python scripts/print_disposition_digests.py <catalog> --write
```

Two things to know before you paste a new value in:

`source_tree_digest` moving is mechanical — regenerate it and move on. But if
`classification_surface_digest` also moves, that is a **real** signal: something
changed a file's entrypoint, argument surface, or checkpoint/resume call sites.
Read the catalog entry before bumping `kReviewedCatalogDigest`; that bump is the
review the gate exists to force. It stayed put across an eight-commit port and
moved for a one-module one, so it does discriminate.

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
