# Acceptance and compatibility CI

CI has to answer one question honestly: *did everything that should have run,
run?* A green badge that covers a shrinking fraction of the suite is worse than
a red one, because it is trusted.

## What changed and why

The previous workflow executed a hand-maintained list of 36 test files. The
repository has 119. Two of the 36 named files -- `test_loop_invariants.py` and
`test_loop_new_features.py` -- contain no test functions at all, so CI had been
running them as no-ops while they appeared covered. Nothing in the workflow
could detect either problem: a list does not know what it omits.

The suite was not broken, only unobserved. Running the full non-GPU suite
locally at the time of this change: 1024 passed, 7 skipped, 0 failures.

There was also no native, Go, schema, or compatibility-catalog coverage of any
kind, so every TrainVM C++ change merged unverified by CI.

## Jobs

| Job | Runner | What it proves |
|---|---|---|
| `coverage-gate` | hosted | Every `tests/test_*.py` contributes tests to the CPU run, or is an explained exclusion |
| `python-cpu` (3.11, 3.12) | hosted | The **whole** `-m "not gpu"` suite, not a named subset |
| `schema-golden` | hosted | Example experiment documents satisfy `experiment-v1.schema.json`; coverage fixtures carry no executable authority |
| `go` | hosted | Dashboard builds, vets, and tests |
| `native-change-scope` | hosted | Changed paths select `full`, `catalog`, or `none`; unknown paths fail closed to `full` |
| `native` | hosted | Full `cmake` + `ctest`, or CLI-only catalog/adapter-contract validation, according to the scope receipt |
| `gpu` | self-hosted | The `-m gpu` suite, serialized across the whole repository |

Each job uploads its junit XML or log as a build artifact, so a run's evidence
outlives its log retention.

## Coverage gate

`scripts/ci_coverage_gate.py` collects the real CI invocation and fails when a
test file contributes nothing and is not listed in
`tests/coverage_exclusions.json` with a reason. It also fails on a stale
exclusion naming a file that no longer exists, so the exclusion list cannot rot
in the other direction.

Nine files are currently excluded: five are module-level `pytest.mark.gpu`
suites that run in the GPU job, and four are standalone diagnostic or benchmark
programs with no test functions.

Adding a test file requires nothing. Adding a file that contributes no tests
requires saying why.

## The native and GPU jobs are gated, deliberately

TrainVM needs GCC 16 with `-freflection` (C++26 P2996) plus protobuf, gRPC,
yaml-cpp, nlohmann_json, OpenSSL and SQLite3. The GPU suites need a real
accelerator.

The compiler is **not** the obstacle, contrary to an earlier claim here. The
official `gcc:16` image (16.1.0, Debian trixie) exists and accepts
`-freflection`; this was verified, not assumed. What blocks a hosted native job
is the dependency pair:

- Debian trixie ships `gRPCConfig.cmake` but no `protobuf-config.cmake`, because
  its protobuf 3.21 is autotools-built. `find_package(Protobuf CONFIG REQUIRED)`
  therefore fails on stock `gcc:16` plus apt.
- Building protobuf 3.21.12 from source alongside Debian's gRPC 1.51 gets past
  `find_package` and then fails at generate time with a missing ALIAS target,
  because the two halves no longer agree on their protobuf targets.

That image now exists: `.github/docker/trainvm-ci.Dockerfile`. **The native
job builds it and runs the suite inside it on an ordinary hosted runner — no
self-hosted runner, no gate.** Only the GPU job stays gated, because it needs a
real accelerator.

The job builds the image rather than pulling a published one. A registry copy
has to be published, made visible to this repository, and kept in step with the
Dockerfile; each of those is a way for CI to drift from the source it claims to
test. Buildx layer caching makes the cost a first-run one.

### Change-scoped native tiers

The original hosted native job cost about 8 minutes 10 seconds even when a
pull request changed only Python or documentation: roughly 65 seconds restored
and loaded the cached 3.46 GiB toolchain image, then roughly 416 seconds built
all native targets and ran ctest. The image cache was already working; full
compilation was the dominant cost.

`scripts/classify_native_ci_changes.py` now chooses the smallest honest tier:

- `full` builds every target and runs the declared ctest set. Any C/C++/CMake,
  TrainVM, native fixture, toolchain, workflow, or unrecognized path selects
  this tier.
- `catalog` builds the real `trainvm` CLI, validates the compatibility catalog
  against the checkout, and crosses the native/Python adapter and runtime-
  requirement contracts. Python, dashboard, script, and documentation source
  trees select this tier because their bytes may be catalog inputs.
- `none` is limited to paths whose behavior is exercised by the Python, schema,
  or Go jobs. It still produces a scope receipt; it is not a missing job.

Pushes to `main` always use `full`. A missing diff base, malformed path, new
top-level tree, or unknown file type also selects `full`, so extending the
repository cannot silently narrow native coverage. Focused tests pin these
decisions. Each native run uploads its selected mode, elapsed seconds, and
ccache statistics; those receipts are the source of before/after claims.

Fail-closed has to survive the plumbing, not just the classifier, and two of the
ways it could have leaked are worth naming because neither is visible from the
Python:

- The classification step runs under `set -euo pipefail`. GitHub invokes a `run`
  block with `bash -e` and *not* `-o pipefail`, so a crashed classifier piped
  into `tee` would have exited 0, written no `mode=` line, and left the job
  output an empty string. Empty compares unequal to `none`, so the native job
  would have run — but as the *catalog* tier, quietly dropping ctest because of
  a Python traceback nothing was watching. The step now also asserts the emitted
  mode is one of the three known values.
- Every comparison that decides to do **less** work is written against the
  narrow tier (`!= catalog`) rather than the wide one (`== full`). The two are
  identical while the mode is one of three known strings and opposite the moment
  it is not. `test_workflow_spells_every_narrowing_decision_against_the_narrow_tier`
  fails if either is rewritten to the positive form, so the property is a red
  check rather than a comment.

The compiler cache is separate from BuildKit because project compilation
happens in `docker run`, after the image exists. It is capped at 1 GiB, keyed
by the pinned toolchain image, mounted at a stable `/build` base, and never
substitutes for parity or ctest gates.

Building it surfaced three more blockers beyond the protobuf/gRPC pair, none of
which were guessable from the outside:

- Debian's CMake 3.31 `FindSQLite3` does not define the `SQLite3::SQLite3`
  imported target the build links against. `find_package(SQLite3 REQUIRED)`
  still *succeeds*, so the failure surfaces much later as a missing ALIAS
  target. The image pins CMake 4.4.0.
- The journal's auxiliary-path authority refuses to run below SQLite 3.53.3,
  which is newer than Debian ships, so the hostd and ledger suites failed
  outright. SQLite is built from source.
- The authority reads `/etc/machine-id` as its host identity, and container
  images do not ship one, so `trainvm_tests` aborted before its first
  assertion. The image generates a stable per-image id.

Every version is pinned with a verified checksum, and the Dockerfile ends in a
smoke test asserting each part of the contract — reflection compiles, both
CMake config packages exist, `SQLite3::SQLite3` resolves, SQLite clears the
floor, protoc/grpc_cpp_plugin/Go/Python/torch are present. A toolchain image
that silently loses one of those is worse than no image, because the native job
would go red for a reason unrelated to the change under test.

### What the native job does not cover

78 of 82 suites run. Four are excluded, each for a stated reason:

| suite | why |
|---|---|
| `hostd_linux_session_authority_tests` | needs `openat2` to pin the session procfs; a container cannot provide it |
| `host_resources_tests` | asserts pinned inventory/occupancy digests built from a real host |
| `trainvm_dashboard_live_e2e` | needs an authority directory whose complete ancestry is unwritable by others |
| `rwkv_lab_worker_artifact` | builds a sealed worker artifact from the real runtime closure |

All four still run in `scripts/acceptance.sh` on a real host. Their exact set
and reasons are pinned by `native-ci-exclusions.v1.json`; the hosted job cannot
silently add another exclusion.

The job was verified to fail for the right reason: breaking one native
assertion turns it red (ctest exit 8), so it is a gate rather than decoration.

Only the GPU job is gated by `TRAINVM_SELF_HOSTED`, because it needs a physical
accelerator. The native tiers run on ordinary hosted Linux runners inside the
pinned toolchain image.

## GPU concurrency

The GPU job's concurrency group is `gpu-tests` with `cancel-in-progress: false`,
and is intentionally **not** keyed by ref. Two branches must not run CUDA
suites against the same device simultaneously and then blame each other's
allocator failures. Not cancelling means a running GPU suite finishes rather
than being killed mid-device.

## Local acceptance

`scripts/acceptance.sh` runs the same matrix locally, including the native
suites CI cannot host:

```bash
scripts/acceptance.sh                 # full non-GPU acceptance
scripts/acceptance.sh --gpu           # also the GPU-marked suite
```

It writes `evidence/acceptance.json`: a per-suite pass/fail/skip receipt
stamped with the exact commit and whether the worktree was dirty, so a result
can never be attributed to source it did not run against. Skips are recorded
with their reason rather than omitted, because a silently missing suite is the
failure this whole document is about.

`evidence/` is generated output and is git-ignored in full. Nothing under it
belongs in a commit, and that follows directly from the stamping rule above: a
receipt records the commit it *ran against*, but a committed receipt is served
to every reader at whatever commit they have checked out. The two diverge the
moment anyone commits again, and the receipt then makes a confident, precise,
wrong claim — which is worse than having no receipt, because it reads as
evidence. This was not hypothetical: `evidence/acceptance.json` and
`evidence/python-cpu.xml` were committed by accident in #43/#44 and sat in the
tree for six days describing a 2026-08-03 run of a *dirty* worktree. They have
been removed. The same rule covers whole qualification trees: one GPU
qualification run left 187 files in the working tree for three days. The
existing `*.pt`/`*.safetensors`/`*.bin` rules happened to cover its checkpoint
payloads, but 150 files — receipts, configs, manifests, logs — were still
stageable by any `git add -A`. Ignoring the directory by shape is what makes
that independent of whether extension rules keep coincidentally covering it.

Durable evidence therefore lives in three places, none of them the source tree:
CI build artifacts for the raw run output; the pull request body for what a
reviewer needs to judge the change; and the kanban card for anything a later
reader must be able to find without knowing which PR to look in. When a
qualification receipt proves something a card is blocked on, quote its decisive
fields onto the card rather than committing the file.

The portable `authoring-matrix` suite validates
`no-code-authoring-matrix.v1.json`. That document is a coverage declaration,
not evidence that the recipes already run: it freezes the required Qwen
multimodal, transformer, RWKV, and MageFlow variation axes; the forbidden
source/build/deployment mutations; and the evidence an eventual sealed
qualification receipt must contain. The release gate is complete only after
the declared variants have produced those receipts. A green matrix validator
alone must never be reported as no-code authoring qualification.

Measured evidence uses `trainvm.no-code-authoring-qualification/v1` and is
checked with `scripts/validate_no_code_authoring_receipt.py`. The validator
requires every declared variant, the authority expansion digest, and an exact
instance-provenanced recipe-field-to-target binding whose target appears in the
compiled-plan diff. This avoids comparing friendly authoring names with invented
document paths. It also requires a successful preflight before any accelerator lease, immediate
step-zero dashboard registration, nonempty modality examples, bounded optimizer
work, checkpoint publication, pause/resume proof, and matching before/after
digests for repository source, adapter dispatch, sealed worker deployment,
dashboard source, and service configuration. It also requires an unsupported
request probe to stop at `new_implementation_required` without acquiring a
lease. Until such a complete receipt exists, this capability remains pending.
