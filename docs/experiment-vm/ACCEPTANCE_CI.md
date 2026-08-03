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

`scripts/acceptance.sh` runs the developer matrix locally, including the native
suites CI cannot host:

```bash
scripts/acceptance.sh                 # full non-GPU acceptance
scripts/acceptance.sh --gpu           # also the GPU-marked suite
```

It writes `evidence/acceptance.json`: a per-suite pass/fail/skip receipt
stamped with the exact commit and whether the worktree was dirty, so a result
can never be attributed to source it did not run against. Skips are recorded
with their reason rather than omitted, because a silently missing suite is the
failure this whole document is about. A required skip closes `gate_open` and
returns status 3. The receipt calls its scope `non_gpu` or `gpu_unit`: even the
latter is developer acceptance, not a claim that hostd launched a production
trainer.

## Production parity gate

The production gate is deliberately separate from `pytest -m gpu`. Kernel and
adapter tests can pass without traversing hostd, without publishing a real
checkpoint, and without proving that the dashboard can read a trainer's live
outputs. `scripts/verify_trainvm_production_acceptance.py` verifies one offline,
content-bound evidence bundle described by
`production-acceptance-v1.schema.json`:

```bash
scripts/verify_trainvm_production_acceptance.py \
  evidence/production/bundle.json \
  --receipt evidence/production/receipt.json
```

The gate opens only when all of the following evidence agrees on one clean
source commit:

- the complete `gpu_unit` developer acceptance matrix passed;
- the destructive, privileged 16-point hostd crash qualification gate opened;
- the live dashboard observed an admitted, unpoisoned hostd authority with a
  verified ledger and process launch enabled;
- native journal-chain verification passed after the runs stopped;
- MageFlow, RWKV, and transformer runs all completed through the durable host
  resource and process sagas, emitted finite live train and eval metrics,
  published verified checkpoints, resumed from a durable attempt, and
  published non-empty accelerator trace profiles;
- the MageFlow evidence additionally contains a checkpoint-bound side-by-side
  eval gallery; and
- every run has a valid reproducibility capsule bound to its run and plan.

Every referenced JSON document is a relative, non-symlink regular file whose
exact SHA-256 appears in the bundle. A missing family, skipped GPU suite,
worker launched outside hostd, truncated required evidence, invalid checkpoint,
empty profile, or changed evidence byte rejects the claim. Passing developer
acceptance alone must never be described as production parity.

The normal path creates those three runs with the qualification controller. It
accepts only real registered family adapters—not `coverage.*` fixtures—and
preflights each document for a checkpoint-publishing training node, checkpoint
recovery, a declared eval metric, and a bounded in-process accelerator trace.
MageFlow must additionally declare its generated/original gallery. The
controller runs the families sequentially, waits for the selected optimizer
step, requests a checkpoint-first resource-releasing pause, proves the hostd
process and fence counts returned to zero, resumes, and requires optimizer and
metric progress before terminal completion:

Do not hand-author those three documents for a deployment. The strict
`production-qualification-inputs-v1.schema.json` contract names the local
MageFlow base/manifests/image roots, RWKV model/token stream, transformer
model/patch/token stream, and the authority-owned run root. The source-only
materializer validates every input path and emits three schema-valid real
experiments plus the exact root sets required for content locking:

```json
{
  "api_version": "trainvm.production-qualification-inputs/v1",
  "workspace_root": "/absolute/moe-mla-deployment",
  "run_root": "/absolute/runs/production-qualification",
  "concurrency_key": "production-qualification",
  "mageflow": {
    "model_path": "/absolute/Mage-Flow-Base",
    "train_manifest": "/absolute/mageflow/train.jsonl",
    "eval_manifest": "/absolute/mageflow/eval.jsonl",
    "image_roots": ["/absolute/mageflow/images"]
  },
  "rwkv": {
    "model_path": "/absolute/rwkv-model.pth",
    "data_path": "/absolute/rwkv-tokens.bin"
  },
  "transformer": {
    "model_dir": "/absolute/transformer-base",
    "patch_dir": "/absolute/mla-patch",
    "tokens_bin": "/absolute/transformer-tokens.bin",
    "total_tokens_in_bin": 1000000
  }
}
```

```bash
scripts/materialize_trainvm_production_qualification.py \
  /absolute/deployment-inputs.json \
  /absolute/new/qualification-documents

for family in mageflow rwkv transformer; do
  trainvm lock-input-content \
    /absolute/new/qualification-documents/$family.json \
    /absolute/new/qualification-documents/$family.input-roots.json \
    > /absolute/new/qualification-documents/$family.locked.json
done
```

The materializer never hashes inputs itself, compiles a plan, names an
executable, or launches a worker. The native authoring command is the sole
producer of content identities. It uses the full-backbone MageFlow profile, so
production control-plane qualification needs the local Mage base and real
train/eval manifests but does not depend on a prior expert checkpoint. The
transformer MLA profile still requires an actual compatible base-model tree,
patch tree, and packed uint32 token stream; missing assets fail before any
document is written. The output directory is new-only and contains a
`materialization.json` handoff with document digests and pause steps.

Submit only the native-locked documents:

```bash
scripts/run_trainvm_production_qualification.py \
  --dashboard-url http://127.0.0.1:9124 \
  --experiment mageflow=/absolute/new/qualification-documents/mageflow.locked.json \
  --experiment rwkv=/absolute/new/qualification-documents/rwkv.locked.json \
  --experiment transformer=/absolute/new/qualification-documents/transformer.locked.json \
  --pause-after-step mageflow=2 \
  --pause-after-step rwkv=2 \
  --pause-after-step transformer=2 \
  --output evidence/production/live
```

The controller submits the documents through the dashboard's native compile
and immutable submission APIs. It never derives or launches a Python command
line. It leaves a failed run intact for diagnosis rather than silently
cancelling or replacing it, and it calls the live capture only after all three
runs have completed their resume cycle.

Capture remains available as a separate first phase when the runs were launched
interactively. Journal verification must not be raced against a running
authority. While the dashboard still serves three completed qualification
runs, capture its immutable views:

```bash
scripts/capture_trainvm_production_evidence.py live \
  --dashboard-url http://127.0.0.1:9124 \
  --run mageflow=MAGEFLOW_RUN_ID \
  --run rwkv=RWKV_RUN_ID \
  --run transformer=TRANSFORMER_RUN_ID \
  --output evidence/production/live
```

The capture reads only the loopback dashboard and refuses a run that changes
during capture. It pages the complete terminal timeline, metrics, and artifact
views, selects the newest MageFlow gallery detail, writes every document with
mode 0600 beneath a new staging directory, and publishes the directory only
after all three families are coherent.

Then stop the TrainVM authority, run `trainvm journal verify` against the exact
journal, and run the privileged crash matrix separately against a disposable
workspace. The crash command really forks and SIGKILLs processes; its workspace
and delegated cgroup must never be a live deployment path. Once those JSON
receipts and `scripts/acceptance.sh --gpu` exist, seal and verify the capture:

```bash
scripts/capture_trainvm_production_evidence.py finalize \
  --capture evidence/production/live/live-capture.json \
  --acceptance evidence/acceptance.json \
  --hostd-crash evidence/hostd-crash.json \
  --journal-verification evidence/journal-verification.json \
  --receipt evidence/production/receipt.json
```

Finalization copies the three external receipts without rewriting their bytes,
constructs `bundle.json`, runs the offline verifier, and publishes a receipt
only if the production gate opens. A failed verification removes the tentative
bundle and cannot leave a passing receipt behind.
