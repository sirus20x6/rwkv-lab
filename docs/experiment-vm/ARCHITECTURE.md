# Declarative experiment VM architecture

Status: proposed architecture for `dashboard/declarative-vm-fsm`

Scope: training orchestration, live control, recovery, telemetry, and dashboard integration

The cross-family performance, profiling, and optimization plan is maintained in
[`PERFORMANCE_ROADMAP.md`](PERFORMANCE_ROADMAP.md). It covers MageFlow and other flow/diffusion
workers, RWKV, transformers, vision/multimodal training, fine-tuning, distillation, post-training,
and qualification; TrainVM itself remains model-family-neutral.

The P0 physical-resource, guarded-launch, and orphan-recovery boundary is specified in
[`HOST_RESOURCE_AUTHORITY.md`](HOST_RESOURCE_AUTHORITY.md). Journal-local leases remain logical run
authority; host-wide accelerator and process authority belongs to the shared host daemon described
there.

The model-family-neutral authority contract for acquisition, inventories, dataset materialization,
cache qualification, human review, eval galleries, export, promotion, and retention is specified in
[`DATA_EVAL_ARTIFACT_AUTHORITY.md`](DATA_EVAL_ARTIFACT_AUTHORITY.md). Paths and worker-produced bytes
remain staging evidence until that contract's immutable publication transaction succeeds.

The original repository's reviewed workflow surface is pinned by the source-bound, explicitly
non-authoritative catalog described in
[`COMPATIBILITY_CATALOG.md`](COMPATIBILITY_CATALOG.md). It is a regression gate for model-family and
operation coverage, never a substitute for adapter or host execution authority.

## Decision

Build a compiled C++ control-plane daemon, **TrainVM**, that owns the lifecycle of every
experiment. Experiments become versioned declarative documents compiled into an immutable,
typed execution plan. Python remains the worker language for PyTorch, model construction, and
other tensor-heavy code. The existing Go dashboard remains the web/UI layer during the migration
and talks to TrainVM through a generated protocol instead of spawning or signalling trainers.

The durable system boundary is schema-first, while the TrainVM implementation is intentionally
GCC-reflection-first:

- Protobuf defines commands, events, worker messages, and dashboard queries.
- JSON Schema defines the human-authored experiment document.
- Generated C++/Go/Python types provide the cross-language contract.
- C++26 reflection drives struct decoding, strict unknown-field checks, enum conversion, descriptor
  generation, and internal operation registration inside the C++ process.
- Persisted data never depends on a compiler-specific reflection representation.

GCC 16 in this workspace compiles the P2996 reflection syntax with
`-std=c++26 -freflection`, consistent with the [GCC C++ status table](https://gcc.gnu.org/projects/cxx-status.html).
[Clang's C++ status table](https://clang.llvm.org/cxx_status) still marks P2996 unsupported. TrainVM
therefore standardizes on GCC 16+, `-std=c++26`, and `-freflection`; supporting Clang is not a goal.
JSON Schema and Protobuf remain explicit so compiler upgrades cannot silently change stored plans or
wire messages.

## Why the current boundary needs to change

The dashboard is already useful and should not be discarded. It has Go handlers, Datastar/SSE,
Pixi visualizations, SQLite rollups, eval galleries, and process telemetry. The problem is that the
control plane is spread across layers:

- dashboard handlers validate fields and construct Python argument vectors;
- the Go server launches detached processes and sends Unix signals;
- shell and Python supervisors encode waits, retries, handoffs, and restart loops;
- trainers append partially standardized JSONL and sometimes write `status.json`;
- live controls use shared SQLite rows directly between Go and Python;
- workflow state is inferred from PIDs, files, exit codes, and log contents.

This makes a new experiment family require UI, validation, launch, state, and rendering code in
several places. It also makes crash recovery dependent on the particular supervisor that happened
to launch the run.

TrainVM makes one process authoritative for desired state and transition history. Trainers report
observations; the dashboard issues commands; neither independently decides workflow state.

## System shape

```text
 experiment document          dashboard browser
   JSON or YAML                     |
        |                        SSE/HTTP
        v                           v
 +----------------+       +-------------------+
 | plan compiler  |       | existing Go UI    |
 | C++ typed IR   |<----->| protocol client   |
 +-------+--------+ gRPC  +-------------------+
         |
         v
 +--------------------------------------------------+
 | TrainVM daemon                                  |
 | command validation | FSM | leases | recovery    |
 | event journal      | projections | artifact CAS |
 +-----------+----------------------+---------------+
             | local worker protocol|
             v                      v
   +------------------+    +------------------+
   | Python/PyTorch   |    | compiled/builtin |
   | model worker     |    | operations       |
   +------------------+    +------------------+
```

The first version is local-machine-first. gRPC runs over a Unix-domain socket with filesystem
permissions; the Go server relays the event stream to browsers as SSE. A TCP listener can be added
later without changing message types.

## Ownership rules

TrainVM is the only writer of lifecycle state, commands, resource leases, and the event journal.
The Go dashboard becomes a client for mutations. Python workers never open the dashboard database.

| Concern | Owner | Notes |
|---|---|---|
| experiment definition and revisions | TrainVM | content-addressed after validation |
| desired run state | TrainVM | revisioned commands with idempotency keys |
| observed worker state | worker, recorded by TrainVM | heartbeat and structured events |
| logical run/workspace lease | TrainVM | one durable boot-scoped lease per concurrency key |
| physical GPU/process grant | host resource daemon | host-wide across every journal and service |
| model and tensor operations | Python worker | PyTorch remains where it is valuable |
| event and command journal | TrainVM | append-only SQLite/WAL initially |
| query projections | TrainVM | rebuildable from journal |
| browser rendering | Go dashboard | schema-driven controls and panels |
| trainer logs | worker artifact | diagnostic, not a control protocol |

This is a single-host single-writer design. Multi-host scheduling is intentionally deferred, but
lease and worker identities include a host ID so adding it does not invalidate persisted data.

## Experiment document and compiled plan

The checked-in schema is `experiment-v1.schema.json`. A document contains:

1. metadata and labels;
2. typed parameters and named artifact contracts;
3. resource and concurrency requirements;
4. registered components and operations;
5. a graph of nodes with typed inputs, outputs, transitions, and retry policy;
6. a live-control catalog including when each update may apply;
7. observability and retention declarations;
8. exact-resume, orphan, and reconciliation policy.
9. optional model-family-neutral compile, warmup, qualification, and bounded GPU-trace phases;
10. bounded CPU affinity, cgroup weight, thread, preprocessing-worker, and nice policies.

Experiments do not contain shell, Python, Jinja, or an unrestricted expression language. Values are
either literals or structured references to parameters, controls, artifacts, prior node outputs, or
run context. Conditions are a small typed predicate tree over event fields.

Lifecycle and profiling declarations select only typed VM features and registered profiler
backends. They never carry commands or environment maps. Adapter capability resolution decides
whether a MageFlow, RWKV, transformer, or future worker can implement a requested phase; the shared
plan vocabulary does not encode model-family-specific launch behavior.

Compile and warmup are worker-side execution phases, not authority-side simulations. The immutable
worker invocation is lowered into digest-bound `WorkerExecutionPhaseRequest` values in
`WorkerWelcome`. A worker can answer only those requests and returns a fenced
`WorkerExecutionPhaseReceipt` carrying disposition, exact completed-step count, timestamps,
diagnostics, and before/after fingerprints of every trajectory-affecting state component. A
successful or skipped phase must restore the identical state fingerprint; failed phases retain
their possibly changed state evidence and cannot masquerade as completion. Each receipt is a
replay-safe, dashboard-visible journal observation and does not advance the workflow FSM.

Before a run exists, the plan compiler performs:

- JSON Schema validation and schema-version migration;
- component and operation lookup in the installed registry;
- input/output type checking against operation descriptors;
- reference resolution and graph reachability checks;
- resource feasibility and path-policy checks;
- CPU/I/O policy conflict checks and bounded profiler schedule/artifact validation;
- cycle analysis (every cycle needs a visit bound and monotonic progress value);
- artifact producer/consumer and immutability checks;
- live-control type, target, range, and application-point validation;
- resume-capability checks against each stateful operation's declared grade; plans requesting exact
  resume reject stop-only, restart-only, or incomplete-state workers;
- construction of a canonical plan and SHA-256 plan identity.

The CLI must expose `trainvm validate`, `trainvm plan`, and `trainvm diff`. `plan` shows resolved
components, resources, graph order, possible side effects, mutable fields, and restart boundaries
without starting a process.

### Component registry

An experiment names a registered adapter, such as `rwkv-lab.mageflow.v1`, rather than a Python
module and a hand-built set of arguments. An adapter descriptor declares:

- operation names and versioned input/output message types;
- the allowlisted executable, Python module, or builtin implementation;
- argument/environment mapping where a legacy CLI is still used;
- emitted event types, safe points, checkpoint protocol, and capabilities;
- side-effect and path policy;
- compatibility and migration rules.

Lifecycle authority is attached to an exact operation profile rather than to a
component, executable name, or experiment. `trainvm.adapters/v2` uses the
reflected `OperationLifecycleCapabilities` object: a closed `stateful` bit,
booleans for graceful stop, checkpoint-now, retained-resource pause,
resource-releasing pause, compile, warmup, qualification, and profiling, plus
the reflected resume grade `none | restart_only | terminal_checkpoint |
compatible | exact`. Stateless operations always use grade `none` and cannot
claim checkpoint or pause state. A stateful declaration is legal only for a
`process` effect. Resource-releasing pause requires an on-demand checkpoint and
a compatible or exact grade; `terminal_checkpoint` has no mid-run safe point
and cannot declare checkpoint-now or resource-releasing pause. Exact resume
additionally requires checkpoint-now.

When `spec.recovery.exact_resume` is true, registry validation walks the
reachable workflow and rejects every stateful process operation whose
authority-owned grade is not `exact`. Stateless process nodes such as
idempotent cache builders do not have to pretend to serialize training state,
but reachable `at_most_once` process operations are incompatible with exact
recovery. Every reachable stateful operation must support graceful stop and the
plan's retained-resource or resource-releasing pause policy; a pause-required
control without an explicit resource policy requires at least one safe pause
protocol. The same registry check gates requested compile, warmup, qualify, and
profile phases against an operation actually invoked by a reachable workflow
node. These declarations describe protocol support only; they do not
themselves authorize a legacy launcher.

An operation that consumes shared training components additionally owns one
reflected `TrainingCompositionContract`: an exact model-family identity and a
closed map from semantic slot name to `TrainingComponentCategory`. Every node
using that operation must provide exactly those slots and categories. An
operation without the contract rejects any attached composition. This keeps a
field from appearing declarative while being ignored by the selected trainer,
and moves missing/extra/wrong-family failures ahead of leases and worker
launches.

The same native registry publishes the canonical `trainvm.operations/v1`
authoring document through the exact `trainvm.operations@1.0.0` descriptor
selector. Each exact adapter/version/runtime/operation/contract key carries its
effect, idempotency, lifecycle, capabilities, optional training composition,
and mandatory bounded input/output maps. Port value types are closed; published
outputs are artifacts and may narrow both artifact type and schema. Registry
construction rejects omitted declarations, oversized surfaces, or primitive
outputs. Plan validation then rejects missing required or unknown ports,
incompatible literal/reference types, artifact type/schema disagreement, and
undeclared or omitted publications before any resource lease or worker launch.
Descriptors describe the protocol the current handler actually implements;
local trainer files are not advertised as outputs until the worker protocol
publishes them.

Adding a trainer family requires one adapter and its tests. Creating another experiment from
registered operations requires no new Go handler, HTML form, subprocess code, or Python supervisor.
The dashboard generates its editor and live controls from descriptors.

### Training-component registry

Trainer families do not own private copies of common algorithm switches. A workflow process node
may select a bounded `training` composition whose slots point to exact versioned entries in the
independent `trainvm.training-components/v1` authority registry. Categories cover optimizer,
parameter router, learning-rate and weight-decay schedule, activation, normalization, objective,
precision/scaling, gradient clipping and accumulation, curriculum, and metric reducer. Slot names
remain explicit so a topology can use several optimizers or schedules without hiding ownership in
positional conventions.

Each entry declares a reflected flat configuration contract, compatible model families, backend
kind, authority-owned symbolic implementation identity, required worker capabilities, checkpoint
state shape and grade, and—only for schedule-like components—an exact step domain. Resolution
rejects unknown or ill-typed fields, applies declared defaults, and canonicalizes both descriptors
and values. It then freezes the resolved per-node compositions in a content-addressed submission
lock. The same resolved object is included in the immutable worker invocation; experiments never
supply import paths, argv, environment variables, or implementation code.

Static trainer inputs use reflected `input_content_roots` rather than pathname identity. The native
authority measures a bounded descriptor-relative Merkle tree and freezes its path, kind, file count,
byte count, and root digest in the plan. The Python worker independently remeasures every declared
root before importing a family trainer and confines every ordinary adapter read to those verified
roots. Symlinks and special nodes are forbidden. Controller-published checkpoint artifacts retain
their separate canonical object-manifest verifier because replacement checkpoints are selected by
durable runtime lineage rather than the original static plan.

Templates keep the operator-readable path list in a closed reflected
`trainvm.input-content-root-set/v1` document. `trainvm lock-input-content` is the native authoring
boundary: it measures those paths, canonicalizes and injects the identities, and recompiles the
whole experiment before emitting a runnable snapshot. Measurement remains outside the pure compiler,
and neither the template nor the root-set file is mutated.

Required capabilities from all selected components are unioned with the adapter operation's
capabilities before the launch intent is committed. The immutable `trainvm.host-launches/v4`
profile for the exact sealed code bytes independently declares its provided capabilities; host
resolution proves the required set is a subset and freezes both sets in
`trainvm.resolved-launch/v4`. The bootstrap carries the provided set rather than echoing the
request. A worker therefore cannot accept an optimizer or fused kernel it did not independently
advertise. The same profile binds the exact argv index replaced by sealed code fd 3, allowing
`python -I /proc/self/fd/3` while rejecting an out-of-range or native code slot. It also binds the
adapter-specific bootstrap-runtime-closure digest verified by the packaged worker before
third-party imports. A reflected native requirements contract declares each registered adapter's
root Python distributions, so deployment tooling discovers MageFlow, transformer, and RWKV runtime
closure policy from the same C++ catalog that owns adapter identity;
resolved-launch replay and cache authority require that same digest rather than trusting a probe to
choose its own closure. Components may only attach to external process
operations, and an exact-resume plan rejects any selected stateful component whose state grade is
merely compatible. Optimizer state, schedule/curriculum cursors, parameter-routing identity, and
precision/scaler state become part of the later checkpoint manifest contract rather than ad hoc
trainer files.

The immutable registry is exposed as the exact descriptor provider
`trainvm.training-components@1.0.0`. `GetDescriptor` returns the canonical registry document and its
registry SHA-256. The Go bridge recomputes that digest before returning the document to the browser.
The experiment editor uses the reflected field contracts to compose node/family/slot selections;
there is no optimizer-, schedule-, or activation-specific dashboard handler.
The implementation and extension rules are specified in
[`TRAINING_COMPONENTS.md`](TRAINING_COMPONENTS.md).

### Immutable evaluation galleries

The dashboard discovers qualitative evaluations only from complete `artifact.published` events for
the versioned `rwkv-lab.eval-gallery.v2` manifest. The scrubber orders revisions by declared step and
journal sequence, preserving multiple attempts at the same step, and follows the newest revision
until the operator pins an older one. It does not poll or enumerate run directories. Generated and
target/source images are shown side by side when the manifest declares a pair.

The HTTP projection revalidates the published manifest's SHA-256 on every metadata or image request,
strictly decodes its bounded schema, checks run/node/attempt lineage, and confines `file:` locators to
configured image roots. Image bytes are streamed from the same opened file descriptor only after
their declared SHA-256 has been verified. URIs are location hints; journal identity, artifact
fingerprints, and per-object hashes remain the authority.

## VM and finite-state semantics

The plan graph is a durable hierarchical state machine. The top-level run states are:

```text
Draft -> Validated -> Queued -> Acquiring -> Running <-> Pausing -> Paused
                                      |          |
                                      |          +-> Cancelling -> Cancelled
                                      +-> Completing -> Completed
                                      +-> Recovering -> Running
                                      +-> Failing -> Failed
```

`Running` contains the current plan node and worker attempt. Node transitions occur only after the
causing event and the resulting state change have committed in one SQLite transaction. External
effects use an intent/receipt protocol:

1. commit an operation intent with command ID;
2. perform or reconcile the effect;
3. commit its receipt and emitted outputs;
4. advance the node.

Builtin `trainvm.core` operations are the supervisor's typed executors. Alongside resource
admission, artifact validation, and resource release, `qualify_cache` gates a published cache
artifact: the supervisor resolves qualification evidence through an authority-owned seam, runs the
qualification decision itself, and commits `cache.qualified` or `cache.rejected` as a managed
builtin receipt. A plan must route both verdicts and must declare the `qualify` execution phase,
so the adapter operation had to advertise qualification support before the gate can appear. The
executor never releases the lease it runs under, and neither the experiment document nor the worker
supplies the verdict.

Operations declare one idempotency class:

- `replay_safe`: repeating the same command ID is harmless;
- `receipt_required`: reconcile an existing receipt before retrying;
- `at_most_once`: automatic retry is forbidden after dispatch uncertainty.

Cycles are supported for training/eval restarts and iterative campaigns. Each cyclic node declares
`max_visits` and a progress field that must move monotonically. This replaces magic exit-code loops
and prevents silent restart storms.

### Desired versus observed state

The run record stores both:

- **desired:** running, paused, checkpointed, cancelled, plus control revision;
- **observed:** worker connection, PID, attempt, phase, step, heartbeat, checkpoint, resources.

The reconciler continuously converges observed state toward desired state. A PID is evidence, not
identity. Worker identity is `(run_id, node_id, attempt_id, launch_nonce)` and every message carries
that tuple and a monotonic sequence number.

### Pause, checkpoint, resume, and cancel

Signals become an adapter implementation detail. The VM issues typed commands and waits for a typed
acknowledgement at a declared safe point.

- Pause means checkpoint if policy requires it, release or retain resources as declared, then ack.
- Checkpoint is a transaction: write temporary data, fsync as appropriate, validate, atomically
  publish a manifest, then emit `CheckpointCommitted`.
- Resume includes the checkpoint fingerprint, plan revision, data cursor, RNG state, optimizer state,
  and compatibility decision.
- Cancel is graceful by default and escalates only according to an explicit timeout policy.

Every one of these verbs is admitted by exactly one declared adapter capability and is never
inferred from signals, files, worker output, or the absence of a refusal. That decision is
`admit_lifecycle_control` in `trainvm/lifecycle_admission.hpp`; the gRPC surface renders its stable
refusal codes (`cancel.unsupported_by_operation`, `checkpoint.unsupported_by_operation`,
`lifecycle.unsupported_by_operation`) rather than deciding for itself.

`lifecycle_equivalence_tests` drives the whole matrix — every registered adapter times every verb
times both the checkpoint-first and plain form — and compares each admission against the
declaration restated independently of the gate, so a gate that ever answers from something other
than the declared capability disagrees with the table. It also pins the honesty rules: only an
`exact` grade preserves a trajectory, `compatible` resumes from a checkpoint without claiming one,
and a `terminal_checkpoint` operation (today the scratch RWKV trainer) refuses pause, resume, and
checkpoint-now in every form while keeping graceful cancellation. The registry's own coherence
rules are exercised adversarially: a profile claiming a resource-releasing pause without
checkpoint-now or without a resumable grade, a stateless profile claiming any of these, an
`exact` profile without checkpoint-now, and a `terminal_checkpoint` profile with checkpoint-now are
each refused at construction.

Duplicate and stale commands are covered against the real controller and journal: an exact repeat
of a control patch, checkpoint-now, or cancel replays the single durable command, the same
idempotency key with any changed field (assignments, expected revision, author, cancel reason,
graceful timeout) is refused without forking durable history, and a reconnecting controller
rebuilds an identical execution state after every attempt, accepted or refused.

## Live mutation model

“Updatable in real time” cannot mean mutating arbitrary history. Every change is a versioned command
with an expected run revision, author, reason, idempotency key, and audit record.

Controls declare one application class:

| Class | Example | Worker acknowledgement |
|---|---|---|
| `immediate` | logging verbosity | current loop boundary |
| `next_microbatch` | caption dropout | microbatch index |
| `next_optimizer_step` | learning rate, loss weight | optimizer step |
| `next_eval` | eval sample count | eval revision |
| `next_checkpoint` | retention policy | checkpoint revision |
| `restart` | precision/backend needing reinit | new attempt ID |
| immutable | model topology already instantiated | rejected |

Multi-control patches apply atomically. The worker validates the whole patch, applies it at the named
safe point, and acks the effective values and step. Rejection leaves the previous revision active.
The dashboard shows requested, accepted, effective, and superseded values separately.

The current protocol recompiles a proposed document and shows a semantic diff fenced to the journal,
selected run revision, current plan hash, and validated proposed plan hash. It does not mutate an
active plan in place. A changed document is submitted as a new queued run whose `run.created` event
records the exact parent run ID, revision, and plan hash. Exact retries replay that child identity;
stale parent identities fail closed. Completed nodes and artifact bindings remain immutable. A future
protocol may adopt edits limited to nodes that have not begun, but that is not claimed today.

## Events, metrics, and artifacts

Logs are retained for diagnosis but are not parsed to decide state. The required event envelope is:

```text
event_id, run_id, plan_revision, node_id, attempt_id, worker_sequence,
event_type, event_version, wall_time, monotonic_time, step, payload
```

Core event families are lifecycle, heartbeat, progress, metric, control ack/reject, artifact,
checkpoint, resource, warning, and failure. Unknown versioned payloads remain preserved even when a
projection does not understand them.

Metrics carry a name, numeric type, unit, step domain, aggregation semantics, labels, and sample
weight. This prevents treating images/second, optimizer steps, batches, and examples as the same
axis. High-rate samples may be compacted only after immutable rollups are committed.

Artifacts are named contracts, not paths found by convention. Each publication includes type,
producer, URI, size, content hash or declared fingerprint strategy, schema version, completeness,
and lineage. Eval galleries pair generated and target images through the same manifest and remain
addressable by step for the existing scrubber.

## Storage

SQLite in WAL mode is sufficient for the first single-host runtime. Use a normalized append-only
journal plus rebuildable projections:

- `experiments`: source document, canonical plan, schema and plan hashes;
- `runs`: desired/observed summary and current revision;
- `events`: immutable ordered envelopes;
- `commands`: idempotency key, expected revision, disposition, result;
- `node_attempts`: dispatch and terminal receipts;
- `leases`: resource/concurrency ownership and expiry;
- `artifacts`: immutable manifests and lineage;
- `control_revisions`: requested and effective patches;
- `metric_samples` and `metric_rollups`;
- `run_projection`, `timeline_projection`, and dashboard-specific read models.

The event journal is never edited during normal operation. Projection schema can evolve freely
because it is reproducible. Backups use SQLite's online backup mechanism and include the experiment
source and adapter-lock manifest.

## Worker protocol

The Python package should shrink to a small worker SDK around existing trainers. It provides:

- registration and handshake with capability and code fingerprints;
- strict decoding of the immutable, content-addressed operation invocation supplied in Welcome;
- a command thread or safe-point poll that never touches the dashboard DB;
- typed event, metric, and artifact publication;
- atomic control-patch delivery and acknowledgement;
- heartbeat and cancellation handling;
- exact-checkpoint manifest helpers.

Tensor code, dataset code that depends on ML libraries, and checkpoint serialization remain Python.
Process supervision, polling, retries, locks, JSON status publication, argument assembly, and campaign
state move to C++.

Compiled builtins should handle path validation, hashing, manifest joins, file inventory, resource
sampling, process supervision, artifact checks, and deterministic graph execution. GPU kernels remain
in CUDA/Triton/PyTorch unless a measured bottleneck justifies native code.

The first algorithm migration is also native: the bounded experiment-analysis library implements
paired effect/interval/permutation statistics, Holm and sequential alpha-spending decisions, and
Pareto selection. Its registry adapter opens existing `experiments.db` files strictly read-only and
returns one typed SQLite snapshot across normalized campaigns or the legacy latest-result fallback.
It does not yet own result writes, reproducibility-capsule capture, or campaign mutation; those stay
in Python until they can publish typed events through TrainVM rather than share a writable database.

## Dashboard behavior

The Go dashboard consumes descriptors and read models rather than embedding experiment-family logic.
The redesigned experiment view includes:

- schema-generated create/edit form with inline type and compatibility errors;
- resolved-plan preview and semantic diff before launch or live revision;
- graph and timeline views showing current node, attempt, transition, and wait reason;
- desired/observed state and heartbeat/resource ownership;
- revision-fenced checkpoint, pause, GPU-releasing pause, resume, and graceful-cancel actions whose
  ambiguous outcomes can only replay the exact serialized idempotent intent;
- controls grouped by application point, with pending/effective/rejected revisions;
- metrics selected from their descriptors rather than hard-coded field lists;
- artifact and checkpoint lineage;
- eval evolution scrubber with generated/target side-by-side pairs;
- replay mode that renders any historical event position without changing the run.

The current SSE and Pixi implementation can render these projections. Rewriting it in C++ would add
risk without improving orchestration correctness, so that is not a prerequisite.

## Safety and invariants

TrainVM fails closed on these invariants:

1. one active lease for a concurrency key;
2. one authoritative attempt for a node;
3. no transition without its causing committed event;
4. no artifact is complete until its manifest validates;
5. no exact resume unless required state and fingerprints match;
6. no mutable control outside its declared range/application point;
7. no unbounded workflow cycle;
8. no arbitrary shell evaluation in experiment documents;
9. no path write outside adapter-declared roots;
10. no secret value persisted in a document or event—only a secret reference.
11. lease authority uses boot-scoped monotonic time plus boot identity; display wall time never
    authorizes work, and legacy wall-clock leases cannot be adopted as active.

All command endpoints require an idempotency key and optimistic run revision. Destructive operations
are explicit protocol variants, never stringly typed action names.

## Migration plan

### Phase 0 — contracts and replay fixtures

- Land the experiment schema, representative documents, protocol definitions, and golden event logs.
- Define adapter descriptors for MageFlow, vision training, generic LM experiments, and core builtins.
- Capture current dashboard JSONL/status/control behavior as compatibility fixtures.

Exit: the compiler rejects invalid graphs and produces stable plan hashes for the fixtures.

### Phase 1 — C++ runtime in shadow mode

- Implement journal, projections, plan compiler, FSM, and read-only Go client.
- Observe existing runs and translate their events without owning processes.
- Prove deterministic replay by deleting and rebuilding projections in tests.

Exit: shadow state agrees with representative current runs and survives forced daemon restarts.

### Phase 2 — own representative adapter workflows

- Implement the Python worker SDK and representative MageFlow, RWKV, and transformer adapters.
- Move cache-span handoff, cache validation, resume protocol/evidence, pause, and control delivery
  into TrainVM while preserving honest stop-only/non-resumable adapter grades.
- Keep the existing Go UI, adding the VM timeline and desired/observed state.

Exit: representative MageFlow, RWKV, and transformer workflows are each represented by one document
and can recover after a daemon or worker kill at every node boundary.

### Phase 3 — clean-process eval and experiment campaigns

- Migrate planned eval restarts, sequential/parallel A/B arms, queues, RLVR, post-training, and
  qualification into graph templates.
- Generate dashboard forms from operation/control schemas.
- Retire family-specific subprocess construction from Go.

Exit: a new composition of registered operations needs only a document and no dashboard code.

### Phase 4 — make TrainVM authoritative

- Route all commands through TrainVM and remove Python access to `trainboard.db`.
- Rebuild dashboard projections from the event journal; retain legacy log ingestion only for old runs.
- Preserve immutable selected-run semantic diffs and lineage-recorded forks; extend historical
  replay beyond the durable workflow/timeline and eval/profile scrubbers where useful.

Exit: no active run depends on PID discovery, status-file polling, or a shell supervisor for recovery.

### Phase 5 — hardening and reflection stabilization

- fault injection at every external-effect boundary;
- property tests for transitions and control revisions;
- compatibility/migration tests across adapter and document versions;
- benchmark event ingestion and metric compaction;
- extend reflection-backed registration where it removes code without changing storage, and retain
  compile-time assertions that detect compiler/library behavior changes.

## Required test strategy

- schema corpus: valid and invalid documents with exact diagnostics;
- compiler golden tests: canonical plan and hash stability;
- model/property tests: no illegal transition, lease overlap, or unbounded cycle;
- replay tests: identical projections after any restart/event prefix;
- fault injection: crash before dispatch, after dispatch, before receipt, and after receipt;
- adapter contract tests with fake workers before real GPUs are involved;
- exact-resume tests for adapters declaring that capability, comparing data cursor, RNG, optimizer,
  and checkpoint hashes; negative tests prove non-resumable adapters reject resume;
- control race tests for concurrent edits and stale revisions;
- end-to-end Go/C++/Python protocol compatibility tests;
- dashboard tests driven from descriptors, not experiment names.

## Explicit non-goals for the first release

- replacing PyTorch training code with C++;
- a general-purpose bytecode VM or arbitrary user scripting engine;
- Kubernetes or multi-host scheduling;
- dynamic mutation of already executed graph history;
- supporting a non-GCC TrainVM toolchain;
- making C++26 reflection a persisted-format dependency;
- rewriting the working Go/Datastar/Pixi frontend before the control plane is proven.
