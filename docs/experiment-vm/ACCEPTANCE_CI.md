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
yaml-cpp, nlohmann_json, OpenSSL and SQLite3. No GitHub-hosted runner image
provides that toolchain, and the GPU suites need a real accelerator.

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
