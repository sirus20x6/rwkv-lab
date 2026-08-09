# GPU schedule authority

Status: design contract for a statically checked GPU schedule IR; no implementation, no generated
CUDA, no device execution

Scope: the trusted boundary between an authored GPU schedule and anything that may run on a device
— the IR, its validation invariants, its rejection receipts, its version rules, and where it meets
experiment declarations, hostd, and the qualification authority

This contract complements [`ARCHITECTURE.md`](ARCHITECTURE.md),
[`HOST_RESOURCE_AUTHORITY.md`](HOST_RESOURCE_AUTHORITY.md), and
[`DATA_EVAL_ARTIFACT_AUTHORITY.md`](DATA_EVAL_ARTIFACT_AUTHORITY.md). It is the design a later card
implements. It is written against the source-pinned
[AutoMegaKernel static-validator audit](AUTOMEGAKERNEL_STATIC_VALIDATOR_AUDIT.md), which is
authority over the upstream paper wherever the two disagree, because the audit was run against the
pinned code at `a514bbc20a03bbf698a17443f8f14a27a617fc10` and the paper was not.

Nothing here authorizes a GPU launch. A schedule that this authority accepts has been proven safe
to *lower*; whether it is fast, or even correct in its numerics, is the qualification authority's
question and is answered by different documents with different digests.

## Decision

Treat a GPU schedule as a versioned, content-addressed, declarative document that is an untrusted
claim until a native validator accepts it, and give that validator no way to reach a device. The
authority is a pure function from a schedule document plus a target capability record to a
validation receipt. It opens no CUDA context, loads no driver, allocates no device memory, and
reads no file that the receipt does not name.

The design separates four concepts that the upstream system collapses:

1. **the schedule IR** — what tasks exist, what they read and write, and how they synchronize;
2. **the validation authority** — the proofs that make lowering safe, and the receipt that records
   exactly which proofs were discharged;
3. **the generator** — the CUDA lowering that consumes an accepted schedule, out of scope here;
4. **qualification evidence** — measured correctness and performance, which is
   [`trainvm.cache-qualification-evidence/v1`](examples/qualification-evidence.qualified.v1.json)
   and is never produced by this authority.

A schedule file on disk, a passing upstream test, an accepted schedule from a previous target, a
plausible opcode name, or a number in a paper is evidence only. None of them independently
authorizes lowering, launch, promotion, or reuse.

### Why this is not "port the upstream validator"

The audit found the upstream validator's headline contract — "never raises on a malformed program,
always returns a result" — false at the pin: sparse buffer and task IDs raise `IndexError`, a
negative task ID raises `ValueError`, and *sparse or negative counter IDs are accepted* even though
the runtime uses counter IDs as array indices. It also found that above `_PROVENANCE_MAX_TASKS =
8000` the transitive read-after-write and write-after-write proofs are skipped with a warning and
the program is still accepted, and that unknown configuration keys are silently dropped for
additive compatibility. Each of those is a fail-open, and each is inverted below rather than
inherited. The algorithms and the counterexample catalog are worth adapting; the contract around
them is not.

## Authority split

| Concern | Authority | Other roles |
|---|---|---|
| schedule document content and identity | TrainVM schedule authority | an author or a search agent proposes; nothing else writes |
| opcode domain, arity, parameter keys, and ABI caps | compiled reflected descriptor in TrainVM | the generator and any search space read it; they do not restate it |
| structural, synchronization, and hazard proofs | TrainVM schedule authority | no proof may be delegated to the generator or to a test |
| target capability record | hostd fenced selected resource | the schedule declares a required capability floor; it never names a device |
| lowering to CUDA and launch configuration | generator (future, out of scope) | consumes only an accepted schedule plus its receipt |
| device execution, occupancy, and watchdog | hostd grant plus generator | never reached from validation |
| measured correctness and performance | qualification authority | a validation receipt is not evidence and is never promoted into one |
| upstream numbers and paper claims | none | recorded as source notes in the audit; not qualification |

The Go dashboard is a protocol client here as everywhere. It may render a schedule, its receipt, and
its diagnostics. It does not validate, lower, launch, or decide that a schedule is reusable.

## Trusted inputs

Exactly five, and the receipt names all of them:

1. **The schedule document.** `api_version: trainvm.gpu-schedule/v1`, decoded by reflected C++ with
   strict unknown-field checks. Untrusted until accepted.
2. **The compiled opcode descriptor.** A reflected C++ table — opcode numbering, input/output arity,
   required and permitted parameter keys with their marshalled types and semantic ranges, write-
   extent obligation, and reference-oracle availability. It is compiled in, not loaded, and is
   pinned by `kReviewedOpRegistryDigest` (below).
3. **The target capability record.** SM count, per-SM shared memory, per-block opt-in shared-memory
   cap, register file, maximum threads per SM and per block, cooperative-launch availability, and
   compute capability. Sourced from hostd's fenced selected resource, never from a static table
   keyed by a marketing name.
4. **The experiment declaration** that references the schedule by digest, if the schedule is being
   validated in the context of a plan rather than standalone.
5. **The ABI version compiled into TrainVM**, as a constant, for the major-version check.

Anything else is not an input. In particular: no environment variables, no file path conventions, no
`meta` free-form map contents, no upstream result JSON, and no previously issued receipt for a
different schedule digest. The `meta` map exists for human labels and is excluded from every proof
and from the canonical digest input, so it cannot smuggle a decision.

## Trusted outputs

One document type, `trainvm.gpu-schedule-validation/v1`, and nothing else. It is immutable and
content-addressed, it is the only artifact that may unlock lowering, and it carries:

| Field | Meaning |
|---|---|
| `api_version` | `trainvm.gpu-schedule-validation/v1` |
| `authority` | `schedule_validation_only`; a receipt cannot grant execution or qualification |
| `verdict` | `accepted` or `rejected`; there is no third value and no partial credit |
| `schedule_digest` | sha256 of the canonical schedule; the schedule's only identity |
| `ir_version`, `abi_version` | the exact versions the proofs were discharged against |
| `op_registry_digest` | the reviewed opcode descriptor digest in force |
| `target_binding` | the hostd resource identity and capability record the proofs used |
| `proved_invariants` | the ordered list of invariant IDs actually discharged, by ID |
| `unproven_obligations` | named conditions this authority cannot decide (see below) |
| `diagnostics` | `{severity, code, path, message}` records, JSON pointer paths into the schedule |
| `stats` | task/buffer/counter/edge counts and byte totals, informational only |

`proved_invariants` is the load-bearing field and it is why `verdict: accepted` cannot be read as
"safe" in the abstract. It says which proofs ran. An implementer must emit it from the same table
that drives the checks, so a check that is skipped cannot silently look discharged — the failure
mode this replaces is the upstream warning that says provenance was skipped while the result still
reports `ok=True`.

Validation writes no other file, mutates nothing, and never returns a program object. A rejected
schedule produces a receipt and nothing else; there is no "rejected but here is the lowered code"
path.

Both documents carry an `authority` enum, following every other `*.v1.json` in this directory —
`compatibility_evidence_only` on the compatibility catalog and the source dispositions,
`benchmark_fixture_evidence_only` on the benchmark matrix, `ci_scope_evidence_only` on the CI
exclusions. The value is checked on load and a wrong one is a rejection, which is how a document
states the ceiling on what it is allowed to mean.

A schedule is a declaration and never a launcher. `scripts/validate_benchmark_matrix.py` enforces
the same separation on benchmark fixtures with a `FORBIDDEN_KEYS` set covering argv, environment,
executable identity, interpreter, working directory, and any measured result; the schedule document
carries the identical prohibition. A key naming a command, an environment map, a filesystem path, or
a measured number is `schedule.forbidden_key`, not a field this authority ignores.

## The IR

### Document shape

```
trainvm.gpu-schedule/v1
  authority          "schedule_declaration_only"
  ir_version         "1.0"        major must equal the compiled IR major
  abi_version        "1.0"        major must equal the compiled ABI major
  meta               {string: string}   labels only; excluded from digest and proofs
  required_target    capability floor, not a device name
  buffers[]          name, kind, dtype, shape, space, optional source key
  counters[]         init, note
  tasks[]            op, inputs[], outputs[], write_extents[], out_counter, waits[],
                     params{}, placement
  pages              optional: buffer_to_page, pages[]
  config             tiling, fusion, placement policy, pipelining depth, page policy,
                     threads_per_block, dynamic smem bytes
```

### Identity is position, not a value

Every entity is identified by its **array position** in the document. There is no `id` field on a
buffer, counter, task, or page. A reference is an index into the corresponding array, and the decode
step checks `0 <= index < size` before any structure is built.

This is deliberate and it is the single largest departure from upstream. The audit's first
fail-closed gap is a family of bugs — sparse IDs, duplicate IDs, negative IDs, IDs that are set
members during validation but array indices at runtime — and every member of that family exists only
because an ID is a value that can disagree with a position. Removing the field removes the family;
there is nothing left to canonicalize, deduplicate, or bounds-check twice. The cost is that a
document cannot be edited by deleting an array element without renumbering references, which is a
property of a generated, digested artifact rather than a hand-edited one, and is acceptable.

The receipt still needs a stable human-facing name, so `buffers[i].name` and `tasks[i].label` remain
— as labels, unique-checked so a diagnostic cannot be ambiguous, and never as references.

### Write extents are declared, not inferred

Every task declares `write_extents`: for each output buffer, a list of half-open `[lo, hi)` intervals,
one per dimension of that buffer, covering exactly the elements the task writes.

Upstream infers the written region from opcode parameters, and only for `GEMV_TILE` and `GEMM_TILE`;
every other opcode is treated as writing the whole buffer. The audit records this as needing "an
opcode-independent region algebra". Declaring the extent gives one: disjointness is per-dimension
interval intersection over a closed representation, independent of opcode semantics, and it works
for a fused or future opcode without teaching the validator that opcode's tiling parameters.

The obligation moves rather than disappearing: the generator must write exactly the declared extent
and no more. That is recorded in `unproven_obligations` as `generator.write_extent_exact` and is
discharged by a generator conformance test, not by this authority. Stating it is the point — an
accepted schedule that a lying generator lowers is unsafe, and the receipt says so in the field
whose job is to say so.

### Placement

`tasks[i].placement` is `unassigned`, or `{sm: n, queue_position: k}`. `queue_position` is explicit
rather than implied by task-array order. Upstream derives the per-SM queue order from the position
of the task in the global task list, which means reordering the task array for any reason — a search
mutation, a canonicalization pass, a serializer — silently changes the synchronization semantics of
an already-validated program. An explicit, per-SM-dense `queue_position` makes that impossible, and
makes the queue order visible in the digest.

### Closed domains

`kind`, `dtype`, `space`, `op`, `page policy`, and `placement policy` are C++ enums, decoded by the
reflected `enum.unknown` path. A string outside the enum is a rejection, not a default, not a
passthrough, and not a warning.

## Validation invariants

Grouped into phases. **Phases run in order and a phase that produces any error stops the run**, so a
later phase never indexes structure an earlier phase did not prove. Every invariant has an ID, which
is what `proved_invariants` lists, and a diagnostic code, which is what a rejection carries.

### Phase A — decode and identity

| ID | Invariant | Code |
|---|---|---|
| A1 | `api_version` is the exact supported constant | `api_version.unsupported` |
| A2 | `ir_version` major equals the compiled IR major | `schedule.ir_version` |
| A3 | `abi_version` major equals the compiled ABI major | `schedule.abi_version` |
| A4 | every key is a known reflected field | `field.unknown` |
| A5 | every non-`std::optional` field is present | `field.required` |
| A6 | every enum-typed string is in its closed domain | `enum.unknown` |
| A7 | every buffer/counter/task/page reference is in range | `schedule.reference.out_of_range` |
| A8 | buffer names and task labels are unique | `schedule.label.duplicate` |
| A9 | shapes are non-empty, positive, rank within `ABI_MAX_RANK`, and `numel` does not overflow 64 bits | `schedule.shape.invalid` |

A4 is the house rule, not a new one: `field.unknown` is emitted by the shared reflected decoder at
`trainvm/include/trainvm/reflection_json.hpp:135`, and the accompanying rule that a reflected member
is omissible **only** if it is `std::optional` — a default initializer does not make it optional — is
stated in [`ARCHITECTURE.md`](ARCHITECTURE.md) and produces `field.required` at
`reflection_json.hpp:154`. This directly inverts the audit's fifth gap, upstream's `_filter_known`
additive-compatibility drop, under which two different author intents decode to the same program.

### Phase B — static bounds

| ID | Invariant | Code |
|---|---|---|
| B1 | opcode is registered in the compiled descriptor | `schedule.opcode.unregistered` |
| B2 | opcode has a reference oracle; unimplemented opcodes are refused | `schedule.opcode.unimplemented` |
| B3 | input/output counts are within the descriptor's arity | `schedule.arity` |
| B4 | inputs, outputs, and waits are within `ABI_MAX_INPUTS`/`OUTPUTS`/`WAITS` | `schedule.abi.capacity` |
| B5 | every required parameter key is present | `schedule.param.missing` |
| B6 | every present parameter key is known to the descriptor | `schedule.param.unknown` |
| B7 | every parameter value marshals to its declared ABI type and range | `schedule.param.type` |
| B8 | every parameter value is within its declared **semantic** bounds | `schedule.param.semantic` |
| B9 | `write_extents` covers exactly the outputs, is in bounds, and each interval is non-empty | `schedule.extent.invalid` |
| B10 | `threads_per_block` and dynamic shared memory are within the descriptor's permitted set | `schedule.config.launch` |

B6 is an error where upstream emits a warning — an unknown parameter key upstream is reported as
"will not marshal to the device" and the program is still accepted, which is the same silent-drop
failure as A4 one level down.

B8 is the audit's third gap. Type-correct `int32` is not semantically safe: a GEMV with `K <= 0`, an
attention window with `kv_len` exceeding the KV buffer's extent, a RoPE `head_dim` that is odd, a
`pos` beyond the cache, an `n_off + N_tile` past the output buffer. The descriptor carries a bound
per parameter per opcode — a constant, a shape-derived expression over the task's own buffers, or
both — and B8 evaluates it. An opcode whose descriptor declares no bound for a parameter it requires
does not compile; that is a static assertion on the descriptor, not a runtime check.

B10 is the audit's fourth gap, the `threads_per_block=512` disagreement between schema, search, CPU
evaluation, and the CUDA loader. There is one reflected descriptor and it is the single source for
the validator, the search space, and the generator's accepted set: a value the loader cannot run is
not representable in the descriptor, so it cannot be proposed and cannot be accepted here.

The fourth surface, the JSON Schema, is the unresolved half of that requirement and must not be
glossed. This repository has **no schema-emission subcommand**; `experiment-v1.schema.json` is
hand-maintained and shares no mechanical link with the reflected structs, to the point that the
identifier regex and its `maxLength: 128` appear verbatim in both the schema and `document.cpp`. A
hand-written schedule schema would recreate exactly the drift B10 exists to prevent, one document
later. The implementer therefore has two admissible options and may not pick a third: emit the
schedule schema from the reflected descriptor — the pieces exist as `reflected_field_names`,
`json_field_name`, and `enumerators_of`, and nothing wires them to an emitter yet — or ship no
schedule JSON Schema at all and let the reflected decoder be the only checker, accepting that the
portable CI half cannot validate a schedule. Duplicating the contract by hand is the one outcome
this design rejects.

### Phase C — counters and waits

| ID | Invariant | Code |
|---|---|---|
| C1 | no task waits on its own `out_counter` | `schedule.wait.self` |
| C2 | every wait threshold is a positive static integer | `schedule.wait.threshold` |
| C3 | every waited counter has at least one producer | `schedule.wait.unsatisfiable` |
| C4 | threshold does not exceed the producer count | `schedule.wait.unsatisfiable` |
| C5 | a counter with more than one producer is waited on only as an all-of-N join | `schedule.wait.partial_join` |
| C6 | every counter's `init` is zero | `schedule.counter.init` |

C5 is upstream's best invariant and is adopted verbatim in substance: a counter only counts, so a
partial wait on a multi-producer counter is a first-k-of-N race in which the wrong producers can
satisfy the wait. Its two counterexamples — a partial wait on a shared counter, and two producers
with one partially-waiting consumer — port directly as C5 regressions.

C6 is narrower than upstream, which allows a non-zero `init`. A non-zero initial value means a wait
can be satisfied by initialization rather than by a producer, which makes C3 and C4 unsound as
written. A schedule that needs pre-satisfied waits is expressible with an explicit producer task, so
the general case buys nothing and costs a proof. See open question OQ3 for the multi-pass case.

### Phase D — ordering

| ID | Invariant | Code |
|---|---|---|
| D1 | the producer→consumer graph is acyclic | `schedule.graph.cycle` |
| D2 | per-SM `queue_position` values are dense, zero-based, and unique within each SM | `schedule.queue.malformed` |
| D3 | every assigned SM index is within the bound target's SM count | `schedule.placement.sm_range` |
| D4 | placement is all-or-nothing: either every task is assigned or none is | `schedule.placement.partial` |
| D5 | **the queue-augmented graph is acyclic** | `schedule.queue.deadlock` |

The edge set for D1 is the upstream one: for every wait, an edge from each producer of the waited
counter to the waiting task, self-edges retained so a self-wait also fails D1. Kahn's algorithm,
deterministic lowest-index-first, and a bounded cycle witness produced by an **iterative** DFS — the
upstream `test_huge_cycle_does_not_crash_validator` regression exists because a recursive witness
crashed the validator on a large cyclic schedule, and a validator that crashes is a validator that
did not reject.

D5 is stronger than anything upstream has, and it is the invariant that makes "wait satisfiability
under per-SM reordering" decidable rather than an open question. Upstream checks only that for a
dependency edge whose endpoints share an SM, the producer precedes the consumer in that SM's queue.
That catches a two-cycle and nothing else. It does not catch this:

```
SM 0 queue:  t0 (waits on counter A)   t1 (produces counter B)
SM 1 queue:  t2 (waits on counter B)   t3 (produces counter A)
```

The dependency graph is `t3 → t0` and `t1 → t2`, which is acyclic, so D1 passes. No dependency edge
has both endpoints on one SM, so the upstream per-SM check passes. The schedule deadlocks: `t0`
blocks the head of SM 0's serial queue, so `t1` never runs, so `t2` never unblocks, so `t3` never
runs, so `t0` never unblocks.

The correct invariant is acyclicity of the graph augmented with the queue's own ordering edges:

> Let `E_dep` be the producer→consumer edges and `E_queue` be, for every SM, the edge from the task
> at `queue_position k` to the task at `queue_position k+1`. The schedule is deadlock-free under
> serial per-SM queues if and only if `E_dep ∪ E_queue` is acyclic.

Sufficiency is the argument above run backwards: an acyclic union has a topological order, and
executing in that order never asks a task to wait on something behind it in its own queue.
Necessity holds under the strict serial-queue model, where an SM runs exactly one task at a time in
queue order. It may not hold once an SM can hold more than one task in flight — see OQ2 — and the
authority takes the conservative side, so a schedule that only a more permissive execution model
would run is rejected rather than accepted.

D5 subsumes the upstream same-SM check, which is the special case where a cycle has length two.

### Phase E — hazards

| ID | Invariant | Code |
|---|---|---|
| E1 | every read of a written buffer has **all** of that buffer's other writers ordered before it | `schedule.hazard.raw` |
| E2 | every buffer with multiple writers has either provably disjoint extents under one join counter, or a total happens-before order among its writers | `schedule.hazard.waw` |
| E3 | every KV-cache read that this pass also writes obeys E1 | `schedule.hazard.kv` |
| E4 | on-chip buffers are used on exactly one SM | `schedule.placement.onchip_cross_sm` |
| E5 | buffers sharing a physical page have fully ordered live ranges | `schedule.page.alias` |
| E6 | the graph is small enough for E1–E3 and E5 to be proven exactly | `schedule.scale.unproven` |

E1 is the transitive happens-before provenance check, and buffer-level reachability is not enough:
"some writer wrote it" lets a partial multi-writer read through. The proof is an ancestor bitset per
task computed in topological order, `anc[t] = ⋃_{p ∈ pred(t)} (anc[p] ∪ {p})`, and a read of buffer
`b` by task `t` is race-free iff every writer of `b` other than `t` is set in `anc[t]`. Read-only
kinds — weights, constants, model inputs — need no producer edge. A KV cache that this pass never
writes is prior-step state and is readable.

E2 adopts the upstream disjoint-tiles-under-one-counter proof with the declared extents of B9
substituted for opcode-inferred regions, so it is opcode-independent. The alternative proof, a total
happens-before order among the writers, is unchanged. The diagnostic names a concrete offending
writer pair and which of the two proofs failed, because "there is a race somewhere in this buffer"
is not actionable on a schedule with thousands of tasks.

E4 and E5 exist upstream but the audit found **no direct CPU regression** for either and marked both
"do not adopt yet". They are specified here and gated on their own tests: E4 requires a positive
same-SM case, a negative cross-SM case, and an unassigned-placement case, and E5 requires read and
write live-range counterexamples and an alias-clobber counterexample, before either may appear in
`proved_invariants`. Until those tests exist the invariant is not implemented and a schedule that
uses on-chip spaces or page aliasing is rejected by E6's sibling rule below.

**E6 is where this design most sharply diverges from upstream, and it is the audit's second gap.**
Above `_PROVENANCE_MAX_TASKS = 8000` upstream *skips* E1, E2, and E5, emits a warning, and can still
return `ok=True`. TrainVM may not turn a safety proof into a warning based on graph size. E6 is
therefore a rejection: if the task count exceeds `kMaxProvenanceTasks`, the verdict is `rejected`
with `schedule.scale.unproven` and the receipt says which proofs could not be discharged. The bound
is set by the ancestor-bitset memory cost — `V` tasks need `V·ceil(V/8)` bytes, which is about 8 MB
at 8192 tasks and about 512 MB at 65536 — so `kMaxProvenanceTasks` is a compiled constant with the
arithmetic written beside it, and raising it is a review. Designing a proof that scales past it is
OQ1; until that lands, a large schedule fails closed and is not runnable, which is the intended
outcome.

The same rule applies to any invariant that is not implemented: an unimplemented safety check is a
rejection of every schedule that would need it, never a silent pass. Concretely, the validator holds
a table of `(invariant ID → implemented, required-for-feature)` and rejects with
`schedule.invariant.unimplemented` when a schedule uses a feature whose invariant is not in force.

### Phase F — target and resources

| ID | Invariant | Code |
|---|---|---|
| F1 | a target capability record is bound; validation without one is refused | `schedule.target.unbound` |
| F2 | the bound record satisfies the schedule's `required_target` floor | `schedule.target.insufficient` |
| F3 | total scratch bytes fit the target's memory | `schedule.resource.memory` |
| F4 | per-block dynamic shared memory is within the target's opt-in cap | `schedule.resource.smem` |
| F5 | the launch configuration is cooperatively launchable on the target if the schedule requires it | `schedule.resource.cooperative` |
| F6 | the bound record's identity matches the fenced hostd resource named in the receipt | `schedule.target.identity` |
| F7 | the bound resource is a full device, not a partition, until OQ5 is resolved | `schedule.target.partition_unsupported` |

F1 is the audit's sixth gap. Upstream's target table is static data and its `meta["gpu"]` mismatch is
only a warning; the CUDA build meanwhile derives code generation from the live device, so nothing
proves the record used for validation describes the device that will run. Here there is no static
registry: the record comes from hostd's validated inventory and selected-resource fence, and F6
binds it. A receipt is valid only for the target identity it names; a different device means
re-validation, not reuse.

F2's `required_target` is a floor expressed in capabilities — minimum SM count, minimum opt-in
shared memory, minimum compute capability, cooperative-launch required or not. A schedule never
names a device. This is what makes a schedule portable without making it optimistic.

### Phase G — completion

| ID | Invariant | Code |
|---|---|---|
| G1 | every output buffer is produced by some task | `schedule.output.unreachable` |
| G2 | every task is reachable and contributes to some output | `schedule.task.dead` |
| G3 | every declared buffer is used | `schedule.buffer.unused` |

G2 and G3 have no upstream counterpart. They are here because an unused entity in an accepted,
digested artifact is either a lowering bug or an author error, and because this repository already
rejects unused configuration in experiment documents rather than ignoring it.

## Rejection receipts

A receipt is the only way a verdict leaves the authority, and it uses the diagnostic shape this
repository already has: `trainvm/include/trainvm/reflection_json.hpp:22` defines
`Diagnostic{severity, code, path, message}`, the pointer path is built by `child_path` at
`reflection_json.hpp:54`, and `trainvm::diagnostics_json` renders it. The schedule authority adds
codes to that vocabulary; it does not add a second diagnostic type.

Rules:

1. **Every rejection names a path.** A JSON pointer into the schedule document —
   `/tasks/417/waits/2/threshold`, `/buffers/9/shape/1` — not a task label and not a sentence.

   These are unambiguous for free, which is worth noting because it is not generally true here.
   `child_path` performs no RFC 6901 escaping — `~` does not become `~0`, `/` does not become `~1` —
   and it interpolates map keys raw, so a document with an exotic key can produce a pointer that
   does not resolve to what it names. The schedule IR indexes every entity by integer position and
   has exactly one map-typed field, `meta`, whose contents are excluded from every proof. So every
   pointer a *proof* emits is built from array indices and fixed field names and needs no escaping.
   A decode-phase `field.unknown` on a `meta` key is the one place the general caveat still applies.
2. **Every rejection names a code** from the tables above. Codes are stable identifiers; message
   text is not, and nothing may match on it.

   The codes here differ from the existing ones in one respect that is deliberate. In
   `trainvm/src/document.cpp` a diagnostic code is a bare string literal at its emit site — there is
   no enum, table, or list, roughly 98 codes exist, and only about 14 are named by any test, so
   renaming one is silent. The schedule authority's codes are instead entries in the same compiled
   invariant table that produces `proved_invariants`, because that table has to exist anyway for the
   receipt to be honest, and a code that is not in it cannot be emitted. This is not a criticism of
   the existing surface; it is that a safety authority whose receipt enumerates its own proofs gets
   the registry for free and should use it.
3. **A witness is bounded.** A cycle witness is a bounded node list; a race witness is the offending
   pair plus the buffer. Neither may grow with graph size beyond a compiled cap.
4. **`warning` severity is forbidden for any safety condition.** A proof either discharges or the
   verdict is `rejected`. Warnings exist only for non-safety facts such as a label mismatch. This is
   the rule that upstream's provenance skip violates.
5. **Nothing raises.** A malformed document produces a receipt with `verdict: rejected`, not an
   exception, not a stack trace, and not a partially built program. The upstream contract said this
   and the audit disproved it at the pin; a fuzz corpus over the wire format is the regression that
   keeps it true here, not a docstring.
6. **A rejection is complete.** All errors in the failing phase are reported, not the first. Phases
   after a failing phase are reported as not attempted, in `proved_invariants` by absence and in the
   `not_attempted` list by name, so "no error from Phase E" never reads as "Phase E passed".

### Exit codes

| Code | Meaning |
|---|---|
| 0 | schedule accepted; receipt on stdout |
| 2 | document malformed — Phase A failed; diagnostics on stderr |
| 3 | document well-formed, schedule rejected; receipt with diagnostics on stdout |
| 64 | usage error |

Separating 2 from 3 matters because they mean different things to an automated author loop: 2 is
"you produced something that is not a schedule", 3 is "you produced a schedule that is unsafe, and
here are the invariant IDs that failed". A search loop can act on 3 and must stop on 2. The 3 is
taken from `qualify-evidence`, which documents the same reasoning at `trainvm/src/main.cpp:297` —
"a rejection is a normal, reportable outcome and must be distinguishable from a broken document".

The 2 is taken from `trainvm validate` (`main.cpp:104`), and it is worth saying why rather than
inheriting it silently: `qualify-evidence` uses **1** for a malformed input, not 2, so the two
existing commands disagree about how a broken document exits. Since a schedule is a document that
this authority compiles, `validate`'s 2 is the closer precedent and is what a wrapper distinguishing
"unreadable" from "unsafe" should key on. The disagreement itself is a real inconsistency in the
CLI and is filed separately rather than being resolved here by fiat.

## Compatibility and version rules

1. **`api_version` is a constant, checked by equality.** No range, no prefix match. This is how
   `experiment-v1.schema.json` declares `api_version` (`{"const": "trainvm.rwkv-lab/v1alpha1"}`) and
   how the roughly forty `api_version` checks across `trainvm/src` all work. There is no version
   negotiation and no migration anywhere in TrainVM — schema-version migration is listed as deferred
   in [`ARCHITECTURE.md`](ARCHITECTURE.md) and is not implemented — so a version bump is a new
   suffix and a hard rejection of everything older. The `authority` enum is checked the same way and
   for the same reason.
2. **`ir_version` and `abi_version` are compared on major.** A major mismatch is a rejection with
   `schedule.ir_version` / `schedule.abi_version`. A minor difference is permitted only in the
   additive direction the reflected decoder already enforces — that is, a newer minor may not
   introduce a field an older reader would have to ignore, because an older reader rejects unknown
   fields. In practice this means a new field is a new major unless every reader is rebuilt
   together, and that is the intended friction.
3. **Opcode numbering is append-only.** An opcode's numeric value is ABI and is never reused or
   renumbered. Removing an opcode means retiring its number, not compacting the enum.
4. **The opcode descriptor is digest-reviewed.** `kReviewedOpRegistryDigest` follows
   `kReviewedCatalogDigest` in `trainvm/src/compatibility_catalog.cpp:44` exactly: the descriptor's
   canonical form is hashed at construction, compared to a compiled constant, and a mismatch throws
   with both digests named. Changing an opcode's arity, parameter set, semantic bounds, or write-
   extent obligation moves that digest, and moving it is the review. This is the mechanism that
   keeps the schema, the validator, the search space, and the generator from drifting apart the way
   upstream's four `threads_per_block` surfaces did.
5. **The schedule digest is the identity.** Canonical serialization is the reflected round-trip
   dumped with sorted keys and no insignificant whitespace, `meta` excluded, hashed with
   `trainvm::sha256_hex`. A path is never an identity, matching the artifact-authority rule that a
   published artifact has one canonical manifest digest and never changes in place.
6. **A receipt binds to one `(schedule_digest, op_registry_digest, ir_version, abi_version,
   target_binding)` tuple.** Any change in any component invalidates it. There is no partial reuse
   and no "same schedule, different GPU" carry-over.

## Integration points

### Experiment declarations

A new optional `spec` section:

```json
"gpu_schedule": {
  "schedule_digest": "sha256:...",
  "validation_receipt_digest": "sha256:...",
  "required_target": { ... },
  "workload_class": "serving"
}
```

Every field is required in the reflected struct unless it is `std::optional`; the section as a whole
is `std::optional` so existing documents are unaffected. The plan compiler adds these checks to the
list it already performs before a run exists:

- the referenced receipt exists, its `verdict` is `accepted`, and its `schedule_digest` equals the
  declared one — `gpu_schedule.receipt_missing`, `gpu_schedule.receipt_rejected`,
  `gpu_schedule.digest_mismatch`;
- the receipt's `op_registry_digest` equals the compiled one — `gpu_schedule.registry_drift`;
- `required_target` is satisfied by the plan's resource request — `gpu_schedule.target_infeasible`;
- `workload_class` is one this repository has a validated path for — see the non-goal below —
  `gpu_schedule.workload_unsupported`.

Compilation is accelerator-passive and stays so. None of these checks opens a device.

### Preflight

The accelerator-passive training preflight is where the schedule and the live host meet without
touching the GPU. It already combines a compiled plan with digest-bound family and runtime evidence
and fails closed on missing adapter probe evidence; the schedule authority adds the target-capability
half of F1/F2/F6 there, against hostd's passive inventory. A schedule whose `required_target` no
passive-visible resource satisfies fails preflight, before any grant is requested.

### hostd

F6 binds the receipt's `target_binding` to hostd's `ResourceFence`, which already carries exactly
the four things needed — `resource`, `generation`, `inventory_digest`, and `topology_digest` — plus
the `host_id` and `boot_id` on the `HostInventoryReceipt` it came from. A receipt cannot outlive the
fence it was issued under, and `detect_bundle_degradation`'s existing `host_or_boot_changed` and
`topology_changed` signals invalidate it. The audit's sixth gap is exactly this: upstream validates
against a static record and generates against the live device.

The resource identity is the whole bundle digest and not one token, which is the rule
`HOST_RESOURCE_AUTHORITY.md` already states — "the complete bundle digest, not one selected token,
is the process resource identity" — so a schedule validated against one member of a bundle is not
validated against the bundle.

**This integration is currently blocked and that is a finding, not a detail.** TrainVM's Linux NVIDIA
inventory (`trainvm/include/trainvm/linux_nvidia_inventory.hpp`) carries UUID, PCI address, total
memory, MIG state, device nodes, and display state — and no compute capability, SM count, shared
memory per SM, opt-in shared-memory cap, register file size, or cooperative-launch flag. Every one
of those is required by Phase F. The capability record therefore has no trusted source today, and
this design does not invent one: adding those fields to the fenced inventory is prerequisite work,
filed as its own card.

### Qualification authority

A validation receipt is not evidence. It says a schedule is safe to lower; it says nothing about
whether the lowered kernel computes the right answer or runs faster than the baseline. Those are
`trainvm.cache-qualification-evidence/v1` questions, graded by the existing gate with its parity,
throughput, and memory-regression fields, and answered by measurement on a granted device.

The dependency runs one way and only one way: **an accepted validation receipt is a precondition for
producing qualification evidence, and qualification evidence is never a substitute for a receipt.**
A fast, correct, well-measured kernel from an unvalidated schedule is not promotable. The two
documents have different `api_version`s, different digests, and different exit codes so that no
tooling can confuse them.

Two existing conventions carry over unchanged. A validation receipt is generated evidence about the
repository at a commit, so it belongs under the gitignored `/evidence/` root and is never committed
— `tests/test_evidence_is_not_committed.py` enforces that, on the reasoning that a committed receipt
is served to every reader at whatever commit they have checked out and so diverges on the next
commit, which is worse than no receipt because it reads as evidence and is precise about the wrong
thing. And measurement stays out of the declaration: `scripts/run_benchmark_fixture.py` emits
evidence and decides nothing, because "reimplementing the thresholds here is exactly the drift the
split exists to prevent". A schedule tool that graded its own schedule would be the same mistake.

### Source dispositions

No upstream code is copied by this design and none should be copied by its implementation; the
algorithms are re-derived in typed C++ from the invariant statements above, which is what the audit's
adoption table requires ("Port counterexamples/algorithms into typed C++; do not import the Python
validator as authority"). If upstream code is ever vendored, its MIT notice is preserved and the
exact commit is bound in a source-disposition catalog. The audit's pinned counterexample list is the
starting test corpus, and each ported counterexample cites the upstream test name it came from.

## Open questions

These are unresolved. They are listed rather than answered because a plausible paragraph here would
be a guess that an implementer would read as a decision.

**OQ1 — an exact hazard proof that scales past `kMaxProvenanceTasks`.** E6 rejects instead of
skipping, which is correct and fails closed, but it also means a large schedule is simply not
runnable. The ancestor-bitset proof is `O(V·E/64)` time and `O(V²/8)` bytes. Reachability indexing
by chain decomposition, 2-hop labelling, or interval labelling on the transitive reduction would all
plausibly work, and the choice interacts with the witness requirement — a labelling scheme that
answers "is `u` before `v`" in constant time does not necessarily produce the concrete offending
writer pair E2's diagnostic requires. Unresolved: which structure, what its worst case is on a real
lowered decoder graph, and whether it can produce witnesses. Until this is answered the bound stands
and large schedules are rejected.

**OQ2 — the liveness model when an SM holds more than one task in flight.** D5 is proven sufficient
and, under a strictly serial per-SM queue, necessary. The upstream VM's `pipelining_depth` prefetches
ahead, and a persistent kernel may have several resident blocks per SM, in which case the queue is
not strictly serial and D5 may reject schedules that would run. The conservative side is taken
deliberately, but the question of what the exact invariant is under a bounded-in-flight-window
execution model — probably a windowed variant where `E_queue` edges span the window rather than
adjacent positions — is not answered here. It cannot be answered without the generator's execution
model, which is a later card. Anyone relaxing D5 must state the window explicitly in the IR and
re-derive the necessity argument; relaxing it by "it seemed too strict" is a fail-open.

**OQ3 — counter semantics across more than one pass.** The IR is single-pass: counters start at zero
(C6), increase monotonically, and are never decremented, so a validated schedule is a proof about
one forward pass. A persistent decode loop, a multi-token path, or any schedule replayed across
iterations needs counters reset between passes, and the reset itself is a synchronization event with
its own hazards — a task from pass `n+1` must not observe a counter value from pass `n`. Whether
this is expressed as a validated reset barrier task, as a generation-tagged counter, or by declaring
the schedule single-shot and making the loop the generator's problem, is undecided. Until it is,
`required_target` and `workload_class` are per-pass and a looped schedule is out of contract.

**OQ4 — memory-ordering obligations the IR cannot discharge.** Every invariant above is about the
happens-before *graph*. None of them proves that the generated CUDA makes a counter increment
release-ordered and a wait acquire-ordered, that writes to a buffer are visible at the scope the
consumer reads at, or that a page reused across SMs is flushed rather than left in an L1 that another
SM cannot see. Those are real race conditions that a graph-correct schedule can still exhibit. The
design's answer is to name them in `unproven_obligations` — `generator.release_on_increment`,
`generator.acquire_on_wait`, `generator.visibility_scope`, `generator.write_extent_exact` — so an
accepted receipt states precisely what its acceptance does not cover. What it does **not** do is
specify how those obligations get discharged, because that requires the generator's memory model.
An accepted schedule plus a generator with no conformance suite is not safe, and this document
cannot make it so.

**OQ5 — MIG.** The SM index space under a MIG compute instance is not the physical device's, and
TrainVM's inventory already models partitions as first-class. Whether `placement.sm` is validated
against the partition's SM count or the parent device's, and whether a schedule may be validated
against a parent and run on a child, is unanswered. Phase F currently assumes a full-device target,
and F7 rejects a partition outright until this is resolved.

**OQ6 — whether a schedule gets a JSON Schema at all.** B10 requires one descriptor to drive every
surface, and this repository's fourth surface — the portable JSON Schema that hosted CI validates
against — has no generator, so `experiment-v1.schema.json` is hand-maintained beside the reflected
structs it restates. The two admissible options are named under B10: emit the schedule schema from
the reflected descriptor, or ship none and accept that hosted CI cannot check a schedule document.
Which one is right depends on how much a schedule needs to be validated by something other than the
GCC-16 binary, and that is not decided here. What is decided is that hand-writing a third copy of
the contract is not an option.

## Non-goals

- **A training megakernel.** The audited system is inference-forward: no backward graph, no gradient
  synchronization, no optimizer update, no checkpoint trajectory. This IR inherits that boundary. It
  can describe a serving or preprocessing schedule; it cannot describe a training step, and it must
  not be inserted into a training path because its forward validator is useful.
- **A model importer.** The upstream importer targets a bias-free dense RMSNorm SiLU Llama-style
  decoder with GQA and rejects MoE, MLA, projection bias, non-SiLU activations, and scaled RoPE.
  RWKV, MageFlow, and multimodal encoders are outside it. Nothing in this design imports a model.
- **A performance claim.** No number in the upstream repository or paper is qualification evidence
  here, including the 7,160-schedule zero-false-accept result, whose driver the audit found absent
  from the OSS tree.
- **The generator.** No CUDA is specified, designed, or implied by this document beyond the
  obligations register of OQ4.
