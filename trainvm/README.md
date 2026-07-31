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
- `validate`, `plan`, `simulate`, and journal inspection/replay CLI commands.

This code does not yet launch or control a trainer. The next implementation boundary is the FSM
reconciler service loop, resource leases, and process-free fault injection around dispatch receipts;
real MageFlow process ownership follows only after those boundaries pass.

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
trainvm/build/trainvm simulate \
  docs/experiment-vm/examples/mageflow-cache-resume.json \
  docs/experiment-vm/examples/mageflow-cache-resume.events.jsonl
```

The reference plan currently has the golden identity
`783d2860b51374138e7352d39607cb07254c3b774f9d776946a6f2b5e6ad468c`. A deliberate canonical
format change must update the golden test and supply a plan-schema migration rationale.

## Journal CLI

```bash
trainvm/build/trainvm journal init /tmp/trainvm.db
trainvm/build/trainvm journal append /tmp/trainvm.db event.json
trainvm/build/trainvm journal verify /tmp/trainvm.db
trainvm/build/trainvm journal replay /tmp/trainvm.db
trainvm/build/trainvm journal show /tmp/trainvm.db RUN_ID
```

`event.json` follows the fields in `trainvm::Event`. The CLI is diagnostic scaffolding; the daemon
will receive the equivalent generated Protobuf messages and append them through the same journal API.
