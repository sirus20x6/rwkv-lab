# TrainVM native control plane

Legacy migration coverage is tracked by the strict, evidence-only
[source disposition catalogs](../docs/experiment-vm/SOURCE_DISPOSITIONS.md).

This directory contains the first executable slice of the declarative experiment runtime described
in [`docs/experiment-vm/ARCHITECTURE.md`](../docs/experiment-vm/ARCHITECTURE.md).

Implemented now:

- GCC C++26 reflection-based strict JSON decoding and canonical encoding;
- native bounded experiment analysis: paired bootstrap/sign-flip statistics, Holm and sequential
  alpha-spending decisions, Pareto selection, and a strictly read-only typed SQLite snapshot of
  both normalized campaigns and legacy aggregate results;
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
  and dispatch before `WorkerWelcome`, durably ingests replay-safe heartbeats, scalar metrics,
  artifact manifests, and ordered control acknowledgements without advancing the FSM, delivers
  pending live-control patches by controller sequence, and persists the terminal transition and
  receipt before acknowledging it; duplicate live streams are rejected and lost acknowledgements
  or receipts replay exactly;
- a matching Python worker SDK with sealed bootstrap and immutable invocation decoders, checked-in
  protobuf bindings, strict bidirectional stream sequencing, typed telemetry/artifact/control APIs,
  adapter-owned safe-point polling, and cross-runtime canonical JSON golden digests; see
  [`PYTHON_WORKER_SDK.md`](../docs/experiment-vm/PYTHON_WORKER_SDK.md);
- typed `trainvm.core` artifact-validation and resource-release execution that cannot enter through
  generic worker/simulation hooks, with atomic builtin result, transition, dispatch receipt, and
  immutable lease-release evidence;
- an authority-owned exact adapter registry that binds adapter/version/runtime/operation/contract,
  effect, idempotency, trusted code fingerprint, required worker capabilities, and a reflected
  operation lifecycle/resume grade before mutation;
- an independent, authority-owned training-component registry for optimizers, parameter routing,
  learning-rate and weight-decay schedules, activations, normalization, objectives, precision,
  clipping, accumulation, curricula, and metric reducers; exact keys, reflected scalar contracts,
  model-family compatibility, checkpoint-state grades, explicit schedule step domains, and worker
  capabilities are resolved into a content-addressed per-node composition and submission lock;
- a reflected compatibility-workflow catalog and executable validation gate spanning RWKV,
  transformer/MLA, vision, MageFlow/diffusion, conversion/distillation, post-training, RLVR,
  external trainers, data/cache, and evaluation/profile/export, bound to the exact reviewed source
  bytes while structurally granting no adapter or host execution authority;
- a restart-safe launch-authorization reconciler that resumes partial resource admission, converges
  concurrent/repeated steps on one fenced launch intent, and fails closed on registry drift;
- a separate immutable host-launch registry and deterministic `worker.launch_bound` receipt that
  bind the exact active operation to a host/boot, versioned public argv, verified source evidence,
  and sealed executable/code bytes opened beneath trusted Linux `openat2` dirfds; this resolver is
  process-free and structurally excludes dynamic credentials;
- a strongly typed authority-time sampler that separates display-only wall time from Linux
  `CLOCK_BOOTTIME`, latches one boot UUID, and fails closed on boot-time regression;
- typed host inventory, probe/context evidence, topology policies, occupancy snapshots, stable
  accelerator/partition/mutex identities, deterministic bundle selection, and parent/partition
  conflict closure; inventory is never treated as proof that a resource is free;
- an authority-bound SQLite host ledger with immutable typed hash-chain evidence, exact projection
  closure, persistent per-resource generations, atomic all-or-none grant/release CAS, durable busy
  and exact replay outcomes, stale-inventory revocation, and prior-boot blocking; its retained
  filesystem authority pins the database/lock inode, protects SQLite auxiliary names, and reports a
  filesystem boundary only—not the broader host enforcement grade;
- shared strict host-saga codecs for sealed release requests and grant/busy/release results, so
  journal replay and the future mutating hostd transport consume one canonical shape and reject
  unknown fields, forged digests, or status/payload contradictions;
- journal schema v7 boot-scoped lease authority across acquisition, renewal, release, readiness,
  dispatch, control acknowledgement, and host binding; renewal atomically advances mutable expiry
  and appends an immutable exact-input receipt, while v4 wall-clock rows migrate as quarantined
  `legacy-wall/v1` evidence and can never satisfy active authority;
- exact journal-schema and metadata attestation with transactional v4 migration, plus a
  descriptor-resolved authority namespace that rejects unsafe SQLite aliases and permanently
  poisons a live Journal if its directory, database, or lock identity moves;
- model-family-neutral, operation-scoped declarations for compile, disposable warmup,
  qualification, bounded accelerator tracing, and CPU/I/O placement, with no arbitrary command or
  environment channel;
- a bounded, versioned filesystem `AF_UNIX` `SOCK_SEQPACKET` status transport with canonical JSON,
  SHA-256 payload framing, exact correlation, `SO_PEERCRED` plus per-packet `SCM_CREDENTIALS`,
  deadline-bounded I/O, descriptor-delegation rejection, protected endpoint identity, and truthful
  sealed/auditing/admitting/poisoned lifecycle reporting;
- additive host-ledger v2 startup-audit evidence with exact v1 migration, canonical bounded reports,
  configured-policy admission, historical inventory/occupancy reconstruction, predecessor-chain
  proof, atomic report/projection/receipt commit, strict replay closure, and post-commit re-read;
- an additive host-ledger v3 admission epoch that atomically finalizes an exact current audit and
  occupancy, keeps policy-enabled grants sealed beforehand, binds every later request to the active
  epoch, preserves release-only cleanup while startup is blocked, and exposes a read-only exact
  outcome reconciler that can recover an immutable grant or busy receipt across broker epochs
  without admitting missing work or mutating occupancy/generations;
- an additive host-ledger v4 process-authority chain that leaves the v1 resource evidence
  byte-for-byte unchanged, commits an exact active-grant/cgroup/executable launch intent before any
  child may exist, binds a stopped child's boot/PID/starttime/cgroup/executable identity to that
  intent, and provides atomic rollback plus exact post-commit lost-reply replay for both boundaries;
- an additive host-ledger v5 terminal-process receipt that requires exact pidfd wait identity,
  twice-empty allocation-cgroup evidence, and a complete host-side accelerator-context audit;
  spawned allocations are structurally unreleasable until this receipt exists, while v4 histories
  migrate additively only when they contain no unsafe released/nonterminal process;
- a process-free, single-use journal-fence challenge verifier binding socket peer process instance,
  host/boot/broker, pinned-journal claims, controller generation, logical fence, nonce, and bounded
  boottime lifetime, with per-peer quotas and no bearer-capability interpretation of decoded data;
- a canonical hostd mutation-envelope contract that binds one open journal/controller claim to the
  issued single-use challenge, echoed response, exactly attributed grant/reconcile/release/process
  payload, sealed command digest, and operation-compatible reply, now dispatched over a bounded
  one-command `SOCK_SEQPACKET` session with exact correlation and per-packet credentials; process
  prepare alone may carry an exact role-ordered sealed executable, optional code, and opened working
  directory through `SCM_RIGHTS`, while every other packet still rejects descriptor delegation;
- per-concurrency-scope durable hostd controller heads, generations, event identities, and retained
  controller IDs, so takeover invalidates only the affected logical resource scope while aliased,
  rolled-back, or legacy-global controller metadata fails closed;
- Linux production challenge primitives using `getrandom`, boot-bound `CLOCK_BOOTTIME`, strict
  mount/PID/cgroup/time namespace identity, `SO_PEERCRED` plus `SO_PEERPIDFD`, and pinned-proc process
  identity, together with a read-only pinned-Journal attestor rooted in hash-chained controller,
  acquisition, renewal, release, and current-fence authority events;
- a strict Linux service-role authority that pins procfs, cgroup-v2, and every configured service
  cgroup, then double-samples unified membership and proc starttime while binding exact UID/GID,
  service identity, and grant/release role without accepting any request-provided role;
- strict Linux process-launch primitives that pin a protected cgroup-v2 root, deterministically
  create or reopen one empty allocation cgroup, validate immutable launch descriptors, and use
  `clone3(CLONE_INTO_CGROUP|CLONE_PIDFD)` with a private pre-exec gate; PID starttime and unified
  membership are double-attested before the v4 spawn receipt, while every error closes the gate and
  kills/reaps only through the pidfd; a hostd-owned supervisor retains that stopped identity across
  request connections, replays exact prepare/commit/finalize commands, releases the pre-exec gate
  once, and keeps the launch until terminal v5 evidence is durable;
- a read-only Linux NVIDIA inventory collector that pins procfs/sysfs/devfs roots, dynamically loads
  and grades NVML evidence, double-samples bounded device/MIG/display/context state, retains process
  instance observations, proves device-node mappings, and makes incomplete, torn, stale, or
  insufficiently trusted evidence ineligible;
- a canonical, explicitly untrusted compiler/JIT namespace-claim format covering
  adapter/code/executable fingerprints, compute compatibility and optional placement, host ABI,
  driver/runtime closure, compiler configuration, and compile inputs. Cache reuse remains disabled
  until an authority builder verifies those receipts and an immutable cache-publication protocol
  prevents poisoning;
- `validate`, `plan`, `simulate`, and journal inspection/replay CLI commands.

The launch-authorization reconciler first persists a fenced `worker.launch_requested` protocol
intent; the host resolver then persists a deterministic, non-secret descriptor binding while
retaining the sealed files. The guarded host daemon exposes exact-replay prepare, commit, and
finalize process commands backed by the stopped-child cgroup/pidfd supervisor. The TrainVM service
does not yet drive that daemon process surface end to end, so production trainer launch remains
disabled at the orchestration boundary. Production launch preparation requires an exact, live
durable host grant and binds its request identity, receipt digest, and physical fences into the
worker ticket and resolved launch identity; process-free unit fixtures can opt into a visibly
test-only legacy mode.

The filesystem `SOCK_SEQPACKET` boundary has separate status and mutation servers. The mutation
exchange now binds the accepted socket process instance to a single-use journal challenge, obtains
service access only from an injected host-owned identity authority, samples grant/release time only
on the server, dispatches request/reconcile/release through `HostGrantCoordinator`, and disconnects
the scoped coordinator session on every exit path. Duplicate requests and read-only recovery return
the exact persisted result; stale fences and cross-scope payloads fail before ledger mutation.
Deterministic transport checkpoints cover challenge delivery, command receipt, verification,
coordinator connection, durable dispatch, and reply delivery; pre-dispatch interruption leaves no
outcome, while a post-commit lost reply converges to the exact replay and remains releasable.
Current end-to-end transport tests use the visibly cooperative enforcement grade. The strict
service-cgroup authority and strict namespace/socket-pidfd primitives are implemented, but
production mutation admission remains disabled until daemon bootstrap wires their externally
guarded policies together; the dashboard/service also does not yet route all admission/release work
through this endpoint. The remaining process-ownership work is wiring TrainVM to the guarded daemon
surface, completing daemon-restart recovery of retained launches, enabling sealed Python
code-descriptor invocation, and qualifying the end-to-end crash windows before real trainer
ownership is enabled.
MIG evidence is collected and attributed per instance, but grants remain disabled: the generic
conflict selector intentionally blocks a child while its full-device parent is nonselectable, until
a partition-aware enforcement policy proves that scheduling relationship end to end.

MageFlow is only the first recovery fixture. The runtime is not a MageFlow-specific launcher: RWKV,
transformer, vision/multimodal, conversion, distillation, post-training, RLVR, external-trainer, and
qualification workflows use the same lifecycle and evidence protocols through distinct registered
adapter operations. The coverage and optimization inventories live in
[`WORKFLOW_COVERAGE.md`](../docs/experiment-vm/WORKFLOW_COVERAGE.md) and
[`PERFORMANCE_ROADMAP.md`](../docs/experiment-vm/PERFORMANCE_ROADMAP.md).

The v7 renewal authority includes a manually tickable coordinator bounded to 256 exact targets that
samples authority time separately for each target and permanently stops on clock or receipt
failure. Renewal receipts bind acquisition identity, prior and new expiry, the equal boot/wall
timeout delta, and a continuous per-fence history; conflicting inserts, replacements, updates, and
deletes are rejected. It deliberately owns no thread or timer: service scheduling,
host-wide resource/orphan checks, cgroup cleanup, process-instance credentials, and durable
spawn/exit receipts remain mandatory startup gates before spawning is enabled. Exact renewal replay
requires the same expected expiry, authority-time sample, and timeout; after restart a coordinator
tracks the journal's current active lease instead of retrying stale pre-renewal state.

The journal namespace guard closes cooperating-process split authority and rejects unsafe SQLite
side-file aliases at every SQL boundary, but stock SQLite still opens WAL/SHM/rollback files by
pathname. A hostile same-UID process can race that open unless the authority directory is part of
the trusted host boundary or a controlled VFS is used. The abstract-socket namespace fence is also
network-namespace-local. These residuals are explicit host-isolation/lock-broker design inputs, not
authorization to enable worker spawning.

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
trainvm/build/trainvm validate-catalog \
  "$PWD/docs/experiment-vm/compatibility-workflows.v1.json" "$PWD"
trainvm/build/trainvm inspect-training-components \
  "$PWD/docs/experiment-vm/examples/training-components.v1.json"
trainvm/build/trainvm inspect-registry "$PWD/experiments.db" \
  --task recall:16 --metric acc --baseline baseline --limit 20
trainvm/build/trainvm serve --journal /tmp/trainvm.db --socket /tmp/trainvm.sock \
  --registry /etc/trainvm/adapters.json \
  --host-launch-registry /etc/trainvm/host-launches.json \
  --training-component-registry \
  "$PWD/docs/experiment-vm/examples/empty-training-components.json"
trainvm/build/trainvm simulate \
  docs/experiment-vm/examples/mageflow-cache-resume.json \
  docs/experiment-vm/examples/mageflow-cache-resume.events.jsonl
```

The empty training-component registry keeps composition disabled. The checked-in
`training-components.v1.json` contains real cross-family runtime-backed descriptors for inspection
and adapter qualification; a production daemon should enable it only with worker profiles that
advertise the catalog's exact capabilities.

The reference plan currently has the golden identity
`d9874d50706cb8b13f3803258bde08f2175bfb4869eae860aa47994d151e901e`. This deliberate canonical
plan-schema migration adds typed cross-family lifecycle/profiling and CPU/I/O policy declarations
to the reference fixture. Any further deliberate canonical
format change must update the golden test and supply a plan-schema migration rationale.
`compile` is the bounded dashboard authoring boundary: it reads one JSON document from stdin and
returns either structured diagnostics or the native compiler's canonical plan and content hash.
`serve` is the stateful mutation boundary. The dashboard connects to its Unix socket through gRPC;
it never opens the journal writable. Startup requires bounded `trainvm.adapters/v2` and
`trainvm.host-launches/v1` registry documents, decoded strictly and retained immutably for the
daemon lifetime. An explicitly supplied host registry with no profiles is the launch-disabled
configuration; missing or invalid host authority fails startup. `SubmitExperiment`
validates every exact adapter profile before validation succeeds or a run is created, and supports
idempotent queued
creation; the live-control `CommandRun` variant is also implemented. The dashboard freezes ambiguous
submissions and retries their exact body and key. The worker stream now carries durable telemetry,
artifacts, live-control delivery/acknowledgement, and a terminal result. Before Welcome, TrainVM
freezes a canonical, content-addressed operation invocation containing resolved public inputs,
effective controls, authorized artifact manifests, output declarations, execution policy, and exact
adapter identity; reconnects receive those same bytes. The Linux host authority has a stopped-child
cgroup/pidfd launcher, durable process intent/spawn/exit receipts, and guarded daemon
prepare/commit/finalize commands. Service-to-daemon launch integration and pause/resume remain
subsequent milestones. A worker launch ticket is a protocol authorization only until it is paired
with a trusted descriptor digest, resolved launch specification, host identity, and durable process
receipt.

Secret-marked parameters are restricted to versioned opaque references of the form
`secret://provider/name#version`; raw secret values are rejected before canonical plan persistence.
The host-launch registry contains only fixed `public_arguments`. Future resolved credentials will
travel over sealed descriptors and will not enter argv, the adapter lock, launch binding, journal,
or diagnostics.

Every external profile in `trainvm.adapters/v2` contains the following closed
authority-owned shape (values shown are illustrative, not a registered legacy
launcher):

```json
{
  "lifecycle": {
    "stateful": true,
    "graceful_stop": true,
    "checkpoint_now": true,
    "pause_keep_resources": true,
    "pause_release_resources": true,
    "compile": true,
    "warmup": true,
    "qualify": true,
    "profile": true,
    "resume_grade": "exact"
  }
}
```

`resume_grade` is one of `none`, `restart_only`, `terminal_checkpoint`,
`compatible`, or `exact`. Stateless profiles must use `none` and cannot declare
checkpoint or pause controls. Stateful profiles must have `process` effect.
`terminal_checkpoint` cannot claim a mid-run checkpoint or resource-releasing
pause; resource-releasing pause requires checkpoint-now and a `compatible` or
`exact` grade. A plan requesting exact recovery is rejected unless every
reachable stateful process operation is graded `exact`, and no reachable
process operation may be `at_most_once`. Reachable stateful operations must
also implement graceful stop and the declared pause resource policy. Typed
compile, warmup, qualification, and profiling phases must target an operation
actually invoked by a reachable workflow node.

The canonical plan adapter lock is `trainvm.adapter-lock/v2`. Lifecycle fields
are part of its digest; legacy v1 locks are rejected rather than silently
upgraded.

## Journal CLI

```bash
trainvm/build/trainvm journal init /tmp/trainvm.db
trainvm/build/trainvm journal verify /tmp/trainvm.db
trainvm/build/trainvm journal replay /tmp/trainvm.db
trainvm/build/trainvm journal show /tmp/trainvm.db RUN_ID
```

The CLI is diagnostic scaffolding. Runtime mutations are available only through typed controller
operations; there is intentionally no raw event-append command on an authority journal.
