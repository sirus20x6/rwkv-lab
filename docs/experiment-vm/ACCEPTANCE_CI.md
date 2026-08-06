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
| `native` | self-hosted | `cmake` + full `ctest` + `trainvm validate-catalog` |
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

57 of 61 suites run. Four are excluded, each for a stated reason:

| suite | why |
|---|---|
| `hostd_linux_session_authority_tests` | needs `openat2` to pin the session procfs; a container cannot provide it |
| `host_resources_tests` | asserts pinned inventory/occupancy digests built from a real host |
| `trainvm_dashboard_live_e2e` | not yet diagnosed |
| `rwkv_lab_worker_artifact` | not yet diagnosed |

The last two are excluded so the job is usable today, and tracked as a
follow-up rather than quietly dropped. A permanently red job teaches everyone
to ignore it, which is worse than not having one. All four still run in
`scripts/acceptance.sh` on a real host.

The job was verified to fail for the right reason: breaking one native
assertion turns it red (ctest exit 8), so it is a gate rather than decoration.

Both jobs therefore target self-hosted runners and stay off until the
repository variable `TRAINVM_SELF_HOSTED` is set to `true`. This is a
deliberate choice over emulating them: a native job that cannot build the
authority would report green while proving nothing, which is the exact failure
mode this document exists to prevent.

To enable them, register a runner with labels `self-hosted,linux,trainvm` (and
`self-hosted,linux,gpu` for the GPU job), then set the variable.

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
