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
- `validate`, `plan`, `simulate`, and journal inspection/replay CLI commands.

This code does not yet launch or control a trainer. The next implementation boundary is the
WorkerControl stream and reconciler: map the existing Protobuf WorkerHello into the verified native
readiness transaction, issue an explicit welcome receipt, and then own an adapter process. Real
MageFlow process ownership follows only after that service boundary and its fault-injection tests
pass.

## Toolchain

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
trainvm/build/trainvm serve --journal /tmp/trainvm.db --socket /tmp/trainvm.sock
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
it never opens the journal writable. `SubmitExperiment` supports validation and idempotent queued
creation; the live-control `CommandRun` variant is also implemented. The dashboard freezes ambiguous
submissions and retries their exact body and key. The process-free worker launch/readiness core is
implemented; WorkerControl streaming, pause/resume, and lifecycle reconciliation remain subsequent
milestones.

## Journal CLI

```bash
trainvm/build/trainvm journal init /tmp/trainvm.db
trainvm/build/trainvm journal verify /tmp/trainvm.db
trainvm/build/trainvm journal replay /tmp/trainvm.db
trainvm/build/trainvm journal show /tmp/trainvm.db RUN_ID
```

The CLI is diagnostic scaffolding. Runtime mutations are available only through typed controller
operations; there is intentionally no raw event-append command on an authority journal.
