# TrainVM native control plane

This directory contains the first executable slice of the declarative experiment runtime described
in [`docs/experiment-vm/ARCHITECTURE.md`](../docs/experiment-vm/ARCHITECTURE.md).

Implemented now:

- GCC C++26 reflection-based strict JSON decoding and canonical encoding;
- reflected enum parsing and schema-name introspection;
- semantic graph, reference, artifact-availability, cycle, control, and recovery validation;
- stable canonical plan hashing with SHA-256;
- generated Protobuf/gRPC C++ protocol types;
- an append-only SQLite/WAL event journal with idempotent event IDs;
- atomic multi-event transactions for a causing event and its derived VM transition;
- monotonic run, plan, and worker sequence checks;
- a SHA-256 journal hash chain;
- deterministic run-projection replay that refuses a corrupted journal;
- a deterministic process-free FSM reducer with structured predicates, terminal states, bounded
  visits, monotonic loop progress, and stale-attempt rejection;
- a durable controller that atomically commits each worker cause, derived transition, and next-node
  or terminal observation, then reconstructs and verifies execution state after restart;
- a scripted fake worker adapter used for restart/resume, retry-idempotency, plan-mismatch, and
  transaction-rollback tests;
- durable exclusive resource leases with explicit expiry, owner-checked renewal/release, and
  monotonic fencing tokens that invalidate stale controllers after takeover;
- durable per-attempt dispatch intents and completion receipts, with idempotent worker re-execution
  and atomic receipt/cause/FSM commits closing the controller-crash ambiguity window;
- reflected live-control validation with atomic multi-value patches, declared safe-point selection,
  pause requirements, optimistic run/control revisions, idempotent command keys, and durable
  applied/rejected/restart-required acknowledgements;
- content-addressed compiled-plan persistence, verified plan recovery, and fail-closed schema
  migration policy;
- a native gRPC command authority over a permission-restricted Unix socket, with an exclusive
  journal-owner lock, typed control results, serialized optimistic updates, and fenced worker acks;
- strict in-memory JSON/YAML experiment submission, canonical recompilation by the authority,
  journal-bound idempotency identities, atomic queued run creation, and deterministic queued-run
  recovery without launching work;
- a process-free queue reconciliation boundary that atomically acquires a fenced workspace lease,
  completes the declarative builtin resource-admission node, and advances to the real worker node
  while remaining unassigned and observed-acquiring;
- durable worker launch tickets binding node/attempt, nonce, adapter version, trusted code
  fingerprint, required capabilities, and workspace fence; exact WorkerHello acceptance atomically
  journals readiness, observed-running, and node entry; every managed transition into another
  external node returns to unassigned observed-acquiring and requires a distinct launch/hello;
- live-fence validation in the same transaction as worker-backed dispatch preparation and result
  completion, so an expired, released, or superseded worker cannot mutate the FSM;
- a bounded `WorkerControl.Connect` stream that requires `WorkerHello` first, persists readiness
  and dispatch before `WorkerWelcome`, admits one exact fenced result, persists its transition and
  receipt before acknowledging it, rejects duplicate live streams, and replays lost receipts;
- typed `trainvm.core` artifact-validation and resource-release execution that cannot enter through
  generic worker/simulation hooks, with atomic builtin result, transition, dispatch receipt, and
  immutable lease-release evidence;
- an authority-owned exact adapter registry that binds adapter/version/runtime/operation/contract,
  effect, idempotency, trusted code fingerprint, and required worker capabilities before mutation;
- a restart-safe launch-authorization reconciler that resumes partial resource admission, converges
  concurrent/repeated steps on one fenced launch intent, and fails closed on registry drift;
- a separate immutable host-launch registry and deterministic `worker.launch_bound` receipt that
  bind the exact active operation to a host/boot, versioned public argv, verified source evidence,
  and sealed executable/code bytes opened beneath trusted Linux `openat2` dirfds; this resolver is
  process-free and structurally excludes dynamic credentials;
- a strongly typed authority-time sampler that separates display-only wall time from Linux
  `CLOCK_BOOTTIME`, latches one boot UUID, and fails closed on boot-time regression;
- `validate`, `plan`, `simulate`, and journal inspection/replay CLI commands.

This code does not yet launch or control a trainer. The launch-authorization reconciler first
persists a fenced `worker.launch_requested` protocol intent; the host resolver can then persist a
deterministic, non-secret binding while retaining sealed file descriptors, but neither is sufficient
to spawn or signal an OS process. The next boundary is a single-threaded guarded launcher helper,
boot-time lease renewal, host-wide resource locks, cgroup cleanup, process-instance credentials, and
durable spawn/exit receipts. Real MageFlow process ownership follows only after those fault-injection
tests pass.

The typed clock is a migration foundation, not current lease authority: the v4 lease rows and APIs
still use scalar wall timestamps. They must be migrated to boot-scoped v5 rows, with unreleased v4
leases quarantined, before the startup gate may enable spawning or renewal.

The sealed payload hashes cover only the copied executable/interpreter and adapter artifact. They do
not yet claim a reproducible dynamic-library, Python standard-library, or import closure; real Python
launch remains disabled until an isolated bootstrap and runtime-closure policy are enforced.

## Toolchain

Host launch resolution requires Linux 6.3 or newer and matching UAPI headers
(`openat2`, `MFD_EXEC`, `MFD_NOEXEC_SEAL`, and `F_SEAL_EXEC`). Runtime security
policy must permit `openat2` and `memfd_create`; unsupported or blocked hosts
fail closed before a launch can be bound.

TrainVM intentionally requires GCC 16 or newer. CMake rejects other compilers. It builds with
`-std=c++26 -freflection` and uses reflection to remove handwritten field-registration code. Its
persisted formats remain the checked-in JSON Schema and Protobuf definitions.

Required native libraries:

- nlohmann-json
- SQLite 3
- OpenSSL
- yaml-cpp
- Protobuf and `protoc`
- gRPC and `grpc_cpp_plugin`
- CMake and Ninja

## Build and test

```bash
cmake -S trainvm -B trainvm/build -G Ninja -DCMAKE_BUILD_TYPE=Debug
cmake --build trainvm/build -j
ctest --test-dir trainvm/build --output-on-failure
```

Validate and inspect the reference plan:

```bash
trainvm/build/trainvm validate \
  docs/experiment-vm/examples/mageflow-cache-resume.json
trainvm/build/trainvm plan \
  docs/experiment-vm/examples/mageflow-cache-resume.json
trainvm/build/trainvm compile < \
  docs/experiment-vm/examples/mageflow-cache-resume.json
trainvm/build/trainvm serve --journal /tmp/trainvm.db --socket /tmp/trainvm.sock \
  --registry /etc/trainvm/adapters.json \
  --host-launch-registry /etc/trainvm/host-launches.json
trainvm/build/trainvm simulate \
  docs/experiment-vm/examples/mageflow-cache-resume.json \
  docs/experiment-vm/examples/mageflow-cache-resume.events.jsonl
```

The reference plan currently has the golden identity
`783d2860b51374138e7352d39607cb07254c3b774f9d776946a6f2b5e6ad468c`. A deliberate canonical
format change must update the golden test and supply a plan-schema migration rationale.
`compile` is the bounded dashboard authoring boundary: it reads one JSON document from stdin and
returns either structured diagnostics or the native compiler's canonical plan and content hash.
`serve` is the stateful mutation boundary. The dashboard connects to its Unix socket through gRPC;
it never opens the journal writable. Startup requires bounded `trainvm.adapters/v1` and
`trainvm.host-launches/v1` registry documents, decoded strictly and retained immutably for the
daemon lifetime. An explicitly supplied host registry with no profiles is the launch-disabled
configuration; missing or invalid host authority fails startup. `SubmitExperiment`
validates every exact adapter profile before validation succeeds or a run is created, and supports
idempotent queued
creation; the live-control `CommandRun` variant is also implemented. The dashboard freezes ambiguous
submissions and retries their exact body and key. The process-free worker launch/readiness core is
implemented, as is the initial single-result WorkerControl stream. Process launch/reconciliation,
heartbeats and metrics, and pause/resume remain subsequent milestones. A worker launch ticket is a
protocol authorization only until it is paired with a trusted descriptor digest, resolved launch
specification, host identity, and durable process receipt.

Secret-marked parameters are restricted to versioned opaque references of the form
`secret://provider/name#version`; raw secret values are rejected before canonical plan persistence.
The host-launch registry contains only fixed `public_arguments`. Future resolved credentials will
travel over sealed descriptors and will not enter argv, the adapter lock, launch binding, journal,
or diagnostics.

## Journal CLI

```bash
trainvm/build/trainvm journal init /tmp/trainvm.db
trainvm/build/trainvm journal verify /tmp/trainvm.db
trainvm/build/trainvm journal replay /tmp/trainvm.db
trainvm/build/trainvm journal show /tmp/trainvm.db RUN_ID
```

The CLI is diagnostic scaffolding. Runtime mutations are available only through typed controller
operations; there is intentionally no raw event-append command on an authority journal.
