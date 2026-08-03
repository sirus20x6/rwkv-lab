# Cross-family training performance roadmap

Status: implementation roadmap for `dashboard/declarative-vm-fsm`

Scope: shared runtime, MageFlow and other diffusion/flow trainers, RWKV trainers, transformer
trainers, multimodal/vision training, fine-tuning, distillation, post-training, and qualification
campaigns

## Boundary

TrainVM is a model-family-neutral experiment control plane. It does not encode a MageFlow training
loop, an RWKV architecture, or a transformer architecture. Those are versioned adapters with typed
capabilities. The common runtime owns lifecycle, resources, profiling, metrics, artifacts, resume
protocol/evidence, and qualification. Exact resume exists only when an adapter serializes
every semantic state element and passes equivalence tests; the control plane cannot manufacture
missing trainer state.

Performance work is split into three layers:

1. **Shared execution capabilities** apply to any compatible adapter: compile and kernel warmup,
   immutable cache namespaces, CPU/I/O placement, asynchronous input staging, bounded profiling,
   memory telemetry, and performance qualification.
2. **Runtime-specific capabilities** are selected by an adapter: PyTorch compile, CUDA graphs,
   Triton or native kernels, FP8 recipes, distributed communication overlap, and compiled optimizer
   steps.
3. **Model/algorithm capabilities** are opt-in experiment variants: Muon-family optimizers,
   attention-window curricula, multi-token prediction, architectural skips, loss fusion, or altered
   parameter schedules. They are never silently enabled as generic “speed” flags.

Dataset, cache, evaluation, benchmark, gallery, export, promotion, and retention evidence follows
the publication and qualification contract in
[`DATA_EVAL_ARTIFACT_AUTHORITY.md`](DATA_EVAL_ARTIFACT_AUTHORITY.md). In particular, cache presence is
not cache integrity, mutable gallery staging is not historical evidence, and optimization promotion
requires an isolated benchmark map plus a machine-readable parity decision.

An experiment declares capabilities, constraints, and expected evidence. It never embeds arbitrary
commands, environment variables, Python, or compiler flags. Existing environment-selected paths,
runtimes, and credentials migrate to typed host-profile, artifact, and secret references whose
resolved public identities are hashed and recorded while secret values remain non-persistent.

## Composable training-component architecture

Keep training algorithms separated by semantic responsibility so a new experiment selects and
composes registered components instead of adding another trainer-wide conditional. The shared
contract covers optimizers, parameter-group routing, learning-rate and weight-decay schedules,
activation functions, normalization, losses/objectives, precision/scaling policy, gradient clipping
and accumulation, data curricula, checkpoint state, and metric reducers. Model-family adapters own
only the topology-specific integration that cannot be expressed by those common contracts.

Each component has an exact versioned key, reflected configuration schema, compatibility predicate,
state schema, lifecycle hooks, and qualification requirements. Construction is registry-driven;
string switches spread across MageFlow, RWKV, transformer, vision, and post-training loops are not
an extension mechanism. Optimizer state and every schedule phase/cursor are part of checkpoint and
resume identity. Parameter routing is explicit and exhaustive, with overlap and unclaimed-parameter
checks. A schedule consumes a declared step domain (microbatch, optimizer step, token, sample, or
wall-time) and cannot infer one from a trainer's local counter.

Implementation boundaries:

- Small control/configuration interfaces and registries live in strongly typed C++ where that
  improves validation, composition, reflection-driven descriptors, and static checking.
- Tensor implementations remain in the runtime best suited to them—normally PyTorch/Triton/CUDA,
  with qualified native extensions where measurement justifies one—behind the same typed contract.
- Activations, losses, and optimizer kernels expose a reference implementation and parity fixtures;
  fused or compiled variants are interchangeable qualified backends, not different experiment
  semantics.
- Learning-rate, decay, accumulation, and curriculum schedules are pure, deterministic state
  machines with golden trajectories and checkpoint round-trip tests.
- Dashboard editors and comparison views are generated from component descriptors, so adding a
  registered optimizer or activation does not require a new Go handler or bespoke form.

Keep the source tree equally explicit. The common training-component namespace has one module per
category (`optimizers`, `parameter_routing`, `lr_schedules`, `weight_decay_schedules`, `activations`,
`normalization`, `objectives`, `precision`, `gradient_policy`, and `curricula`), plus only shared
protocol/configuration types at its root. Family adapters may depend on those modules; common
modules must never import a MageFlow, RWKV, transformer, or vision trainer. CLI parsing and training
loops consume a resolved composition rather than construct components ad hoc. Backend-specific
kernels live below their semantic component, and tests mirror the same categories with shared
contract suites plus backend parity fixtures. This directory/dependency rule is an enforceable
architecture check, not merely a naming convention.

Roadmap deliverables are a common component schema/registry, family compatibility predicates,
parameter-routing audits, state serialization contracts, cross-runtime golden tests, and migration
of existing duplicated trainer switches. Promotion requires representative end-to-end quality and
throughput evidence; tidy separation alone does not imply that two families share a mathematically
valid implementation.

Separation is complete only when every training loop receives one immutable resolved composition
and the following responsibilities have no family-local duplicate:

| Responsibility | Owned interface | Persisted/reproducible state | Required contract tests |
|---|---|---|---|
| Optimizer update | optimizer factory + update/state protocol | exact tensor groups, hyperparameters, implementation key, moment/master-weight state | reference update, state round-trip, resumed trajectory, backend parity |
| Parameter ownership and relative rates | exhaustive parameter router | ordered group membership, exclusions, freeze state, per-group LR/decay multipliers | no overlap, no unclaimed trainable tensor, stable aliases/order, exact audit replay |
| Learning rate and weight decay | independent schedule state machines | step domain, phase/cursor, base value, warmup/cooldown boundaries | golden trajectory, boundary values, resume at every phase transition |
| Activation and normalization | topology installation contract + tensor backend | exact component/version/config at every installation point | forward/backward parity, dtype/shape coverage, fused/reference equivalence |
| Objective/loss | objective protocol over declared model outputs and batch fields | reduction, weighting, masks, auxiliary terms, implementation key | value/gradient parity, empty/masked edge cases, distributed reduction parity |
| Precision and scaling | precision/scaler policy | parameter/compute/accumulator dtypes, scaler state, overflow history | overflow/recovery trajectory, checkpoint resume, backend capability rejection |
| Gradient policy | accumulation, clipping, synchronization protocol | accumulation cursor, clip mode/threshold, synchronization domain | boundary/update-count trajectory, distributed parity, resume mid-accumulation |
| Curriculum/data phase | deterministic curriculum state machine | sample/token domain, bucket/phase cursor, frozen dataset identity | phase-boundary trajectory, restart membership/order, cache-identity checks |

The architecture check rejects dependency edges from a common component back into a family trainer,
direct optimizer/schedule/activation construction inside a training loop, and descriptor entries
without a consuming runtime factory. Migration is not done while legacy switches remain reachable
for a supported workflow: they must either be removed or isolated behind an explicitly versioned
compatibility adapter with deprecation evidence.

### Code-organization acceptance gate

Component separation is a release requirement, not optional refactoring. Each semantic category
owns its configuration type, validation, factory/runtime implementation, checkpoint-state codec,
and contract tests in a category-specific module. In particular, optimizer algorithms, parameter
groups and relative learning-rate routing, learning-rate schedules, weight-decay schedules,
activation functions, normalization, objectives, precision/scaling, gradient policy, and curricula
must remain independently selectable and independently testable. A category may expose a small
shared protocol, but it may not become a miscellaneous training-utilities module.

CI must enforce the following rules as the remaining categories migrate:

- common component modules never import MageFlow, RWKV, transformer, vision, or post-training
  adapters;
- family adapters contain installation points and topology mapping only, not optimizer or schedule
  math, generic activation implementations, or duplicated scalar validation;
- every registered descriptor resolves to exactly one consumed runtime factory, checkpoint schema,
  and contract-test suite; unused or presentation-only registry entries fail validation;
- adding or replacing an optimizer, LR schedule, activation, or precision policy requires no new
  dashboard endpoint, family-wide switch statement, or unrelated component edit;
- source and tests mirror the same category layout, with dependency-graph, state round-trip,
  reference-parity, and exact-resume checks run in the normal test target;
- resolved experiment compositions record exact component/version/config identities, so a tidy
  source layout is also preserved in run lineage and reproducibility evidence.

The gate is complete when the architecture test scans every supported worker path, legacy direct
construction is unreachable, and a representative component can be swapped declaratively without
editing its trainer loop.

## Qualification contract

Every optimization is admitted through a baseline-versus-candidate qualification node. A receipt
records the adapter/code, model, dataset, bucket geometry, precision, accelerator, driver/runtime,
compiler configuration, dependency closure, seed set, and checkpoint identities. Cache reuse
requires an exact fingerprint match.

Evidence is selected by operation effect class rather than assuming every operation has gradients or
an optimizer:

- training kernels require output, gradient, optimizer-update, state, and resumed-trajectory parity;
- serving kernels require output/state/determinism parity but no invented backward requirement;
- cache builders and preprocessors require content, shape, ordering, and manifest parity;
- schedulers/resource policies require ownership, trajectory, resume, and end-to-end throughput
  evidence;
- peak allocated/reserved memory and host pinned-memory bounds;
- cold compile time, warmup time, steady-state throughput, and tail step latency;
- representative shape-bucket coverage, including a transition between curriculum stages;
- model-family quality gates (loss/perplexity, image eval suite, reward, or task metric);
- an immutable trace or benchmark artifact and a machine-readable promotion decision.

Algorithmic changes additionally require paired seeds and a quality/non-regression decision.
Single-run speedrun results are hypotheses, not production defaults.

### Throughput basis, and why an input-pipeline candidate needs a different one

`steady_state_step_seconds` measures the training step and deliberately excludes time blocked on
input. That is the correct basis for a kernel candidate, whose effect is entirely inside the step,
and the wrong one for an input-pipeline candidate, whose effect is entirely outside it.

A prefetching loader moves cost rather than removing it: the training thread blocks for less time,
while the producer's own CPU work contends with compute and inflates the step. Scored on step time
alone, a real end-to-end win is recorded as a regression. Measured on the AO3 fixture at
`seq512xbatch8`, six steps, one seed:

| basis | baseline | candidate (8 workers, depth 2) | ratio |
| --- | --- | --- | --- |
| input wait (total) | 1.3985 s | 0.1290 s | 10.8x less blocking |
| end to end, input + compute | 9.459 steps/s | 17.260 steps/s | 1.82x |
| training step only | 24.627 steps/s | 15.851 steps/s | 0.64x |

Both bases are real measurements of different things, so `run_benchmark_fixture.py` publishes both
in every receipt, along with a `throughput_basis` field naming the one the evidence used:
`training_step_only` by default, `end_to_end_including_input_wait` for `--candidate prefetch`. The
basis is stated rather than switched silently, because a candidate that gets to choose its own
metric can always find one that flatters it.

The claim this permits is narrow: the same batches, in the same order, arrive with less blocking.
`ordering_parity` and `content_parity` are what hold it to that, and they are measured from
per-step batch digests published by both arms — not asserted. A loader that reorders batches, drops
one, or resumes on the wrong cursor fails those gates however good its throughput looks. Absent
digests report `false`, because unmeasured is not the same as equal.

### Consumer-Blackwell attention candidate

Track [SecondNatureComputing/flash-attn-4-sm120](https://huggingface.co/SecondNatureComputing/flash-attn-4-sm120)
at revision `60117041e10fcc6f19882afd274318c755a5ef6e` as an optional SM120/SM121
attention capability, not a replacement default. It requires CUDA 12.8 or newer, dispatches
consumer Blackwell without the SM100 `tcgen05`/TMEM path, and supports training backward,
dropout, block-sparse attention, and paged KV. Its SM120 contract must reject head dimensions over
128 and must not claim split-KV support.

Qualify it in an isolated dependency closure through `kernels.get_kernel`, with the repository
revision in the runtime fingerprint. Cover BF16 and FP16 forward/backward, MHA and GQA, causal and
non-causal execution, every exact MageFlow/transformer shape bucket, checkpoint resume, cold JIT
latency, peak memory, and steady-state update throughput. Compare against the current FA2 winner;
the package's own dense-GQA measurements report FA4 roughly 4–10 percent slower than FA2 at
512–4096 tokens on SM121a, so feature availability alone cannot authorize promotion.

## Declarative execution and profiling (P0)

- Make `compile`, `warmup`, `qualify`, `train`, `eval`, and `profile` explicit graph phases. Warm a
  bounded adapter-declared shape set and schedule transitions in a disposable worker, destroy it,
  then launch a fresh timed worker from the same checkpoint. In-process state restoration is not a
  safe substitute for resetting compile wrappers, FP8 conversions, process groups, CUDA graphs, or
  allocator state.
- Namespace compiled/JIT artifacts by the exact adapter profile, code and executable hashes,
  compute-device identity/architecture and optional topology, driver, sorted runtime dependency closure (for example CUDA,
  PyTorch, Triton, and FA4), compiler configuration, and a canonical compile-input manifest covering
  model graph/topology, bounded shapes, dtypes/precision, runtime options, and any embedded constant
  or checkpoint fingerprints. A mismatch produces a cold namespace rather than an optimistic cache
  hit.
- Treat a namespace document as an untrusted claim. The separate authority builder now recomputes
  it from the exact adapter and host-launch registries, sealed invocation/launch, validated host
  inventory and selected-resource fences, a bounded runtime-probe interface, and a digest-only
  compile-input manifest. It emits a self-bound authority receipt, including the exact run, node,
  attempt, concurrency scope, lease, and fencing token, and makes any registry, launch, device,
  runtime, topology, shape, precision, option, constant, or checkpoint change cold.
- Keep cache qualification, publication, and adoption as distinct authorities. The artifact
  authority now checks the live journal fence before and after potentially long publication and
  qualification work; obtains correctness, trajectory, quality, shape-transition, unprofiled
  throughput, and memory evidence only through a trusted evidence-source interface; and requires a
  second live-fence and byte verification around adoption. The Linux immutable store pins allowed
  source and publication roots, traverses beneath descriptors without following symlinks or mount
  crossings, rejects nonregular/changing/unbounded input, hashes a canonical manifest, fsyncs and
  atomically promotes a read-only content-addressed tree, and re-hashes every declared file while
  rejecting writable or undeclared content. The production lease implementation reads the exact
  active boot-scoped journal lease. Separate Linux production readers now obtain runtime closure
  and qualification evidence from canonical immutable receipts beneath constructor-pinned,
  authority-owned roots; they reject symlink/mount escapes, changing, writable, multiply-linked,
  oversized, noncanonical, or differently bound receipts, and qualification is reread on adoption.
  The paired authority publisher writes canonical typed observations through private temporary
  files, fsyncs bytes and modes, promotes with no-replace atomic rename, replays only identical
  history, and rejects identity conflicts. Runtime reuse remains disabled until a sealed worker
  evidence transport feeds that publisher and the service graph wires namespace derivation,
  publication, and adoption into worker launch; caller-provided evidence never authorizes reuse.
- Add typed CPU/I/O policy: CPU set, CPU weight, I/O weight, worker count, OpenMP thread count, and
  nice level. Record effective policy and throttling as evidence. The non-root worker runtime now
  parses the sealed policy in one category-owned module, narrows and reattests CPU affinity, applies
  tensor-library thread limits, exports the bounded preprocessing-worker hint, and verifies its
  effective nice level before importing a family trainer. Hostd now owns the complementary kernel
  layer: it canonicalizes the same policy once, writes and re-reads cpuset/CPU/I/O controls through
  pinned cgroup descriptors, sets and double-attests process nice before dropping credentials, and
  binds compiled intent plus effective installation into v3 launch/spawn receipts. Restart adoption
  rechecks those controls and the live proc nice value before recovering authority.
- Add bounded `torch`, Nsight Systems, and Nsight Compute trace phases with warmup/skip/capture
  limits. Publish trace, summary, kernel table, launch count, GPU-active ratio, input-stall time,
  allocator pressure, and step-range metadata as dashboard artifacts.
- Give trace artifacts explicit sensitivity and access policy. Raw traces can contain prompts,
  image paths, filenames, tensor shapes, and private/adult dataset identity; summaries may be
  publishable even when raw traces are restricted. Profiled timing is never used as unprofiled
  qualification timing because instrumentation perturbs latency.
- Compare end-to-end steps, not isolated kernels. Qualification includes data wait, host-to-device
  copies, forward/backward, optimizer, communication, checkpoint interference, and evaluation.
- Expose compiler/cache/warmup state and trace artifacts generically in the dashboard; adapters may
  add family-specific panels without changing lifecycle code.

### LibTorch C++ profiling parity

Use the analysis workflow in
[PyTorch Profiling 101 with Modded-NanoGPT](https://blog.underfit.ai/profiling-101-nanogpt) as the
cross-runtime acceptance reference, while implementing the worker integration through LibTorch,
Kineto/RecordFunction where supported, NVTX, and NVIDIA's external profilers. The required outcome
is equivalent evidence, not dependence on the Python `torch.profiler` wrapper.

- Give every LibTorch worker the same declarative wait/warmup/capture step state machine as the
  Python Torch worker. The training loop, not an external timer, advances the schedule at exact
  optimizer-step boundaries.
- Add low-overhead C++ NVTX domains and RAII ranges for input wait, host-to-device transfer,
  forward, backward, optimizer, communication, evaluation, and checkpoint publication. Nsight
  Systems remains the stable whole-process collector because it observes LibTorch's ATen, cuBLAS,
  cuDNN, custom CUDA, runtime, and stream activity without a Python dependency.
- Add an optional in-process Kineto/RecordFunction collector when the pinned LibTorch distribution
  exposes the required C++ profiler API. Treat that API as version-locked rather than assuming the
  Python profiler's compatibility guarantees apply to C++.
- Normalize both collectors into `trainvm.gpu-trace.v1`: exact run/node/attempt and optimizer-step
  interval, GPU-active ratio, input-stall time, launch count, allocator pressure, communication
  overlap, bounded operator/kernel tables, raw-trace digest, and explicit instrumentation overhead.
- Preserve raw trace sensitivity and publication rules. Python stacks and TorchInductor/Triton
  kernel names are optional evidence, not schema requirements; pure LibTorch traces instead retain
  C++ symbols, ATen operations, NVTX ranges, and CUDA kernel identities.
- Emit Perfetto/Chrome-trace-compatible flow events that preserve CPU operator -> CUDA runtime ->
  GPU kernel causality. When the collector exposes them, retain bounded tensor shape/stride and
  source-symbol metadata so an operator can connect a slow kernel to its model geometry and exact
  C++ call site; stack capture remains a separately declared high-overhead mode.
- Derive a machine-readable temporal breakdown from each completed window: compute, communication,
  input/host wait, and unattributed GPU idle time; top operators and kernels by self/total time and
  call count; launch latency; and communication-compute overlap. The dashboard must link each
  summary row back to the corresponding trace interval rather than presenting an untraceable
  aggregate.
- Qualify the implementation with matching unprofiled, NVTX-only, Kineto, and Nsight windows so the
  dashboard can show profiler overhead and never mistake instrumented timing for production
  throughput.
- Acceptance requires one bounded LibTorch training fixture whose trace can be opened directly in
  Perfetto, whose optimizer-step and phase ranges agree with independently counted runtime events,
  and whose synthetic input stall and communication overlap are classified within declared error
  tolerances. A C++-only worker must pass without importing Python or TensorBoard.

The bounded Torch path is implemented end to end for the native MageFlow appearance/terminal and
Qwen adapters. One shared optimizer-step hook drives the declared wait/warmup/capture schedule; it
freezes a Chrome trace and bounded operator summary into `trainvm.gpu-trace.v1`, binds the exact
worker invocation and optimizer-step interval, marks timing as instrumented and the raw trace as
restricted, and publishes it through the worker artifact protocol. The dashboard verifies the
manifest before rendering summaries and hashes the raw trace only on an explicit download. It does
not fetch multi-gigabyte or potentially sensitive traces during polling. Nsight Systems and Nsight
Compute remain separate host-launch profiles: an in-process worker must not pretend that selecting
their enum has wrapped the process. The Torch summary now unions overlapping CUDA event intervals
over the captured optimizer-step wall window, reports GPU-active ratio, normalizes accelerator
launch count per step in the dashboard, and records baseline/peak allocated and reserved memory.
When a run has multiple compatible trace windows, the dashboard selects the oldest rich summary as
the default comparison baseline and lets the operator change it. It shows normalized wall-time and
launch-count deltas, GPU-active percentage-point change, and peak-allocation change; it refuses to
compare windows whose node, backend, schedule, activities, profiler options, or declared warmup
overlap classification differ. Trace cards also show the run's declared compile, warmup, and
qualification state; warmup overlap is explicitly unknown when warmup is enabled without a declared
step count, rather than being reported as steady state.

The external-profiler launch path now binds the collector, fixed argv, capture declaration,
run/node/attempt, worker executable, runtime closure, and output identity into the resolved host
launch. Hostd delegates the collector and a sealed worker authority separately, and the stopped
launcher exposes only fixed descriptors 3-6 with an empty environment. NCU executes as an immutable
sealed copy. NSYS is instead digest-verified and inode-pinned beneath a trusted, non-worker-writable
root because its CLI intentionally refuses anonymous executable images while discovering its
installed support tree from `/proc/self/exe`. CPU-only qualification verifies both empty-environment
startup and that NSYS preserves the worker authority and target descriptors through its wrapper.

The shared profiler protocol also exposes an explicit input-wait context and iterable wrapper. The
native MageFlow appearance/terminal and Qwen paths place it around their actual prefetched-batch or
packed-row acquisition, so a complete capture publishes measured input-stall time and ratio. The
profiler never infers input stall from GPU gaps, which would mislabel CPU, synchronization, or
communication work as data wait; adapters without an explicit boundary omit the field.

Native checkpoint publication now has the same family-neutral boundary: handlers return typed
checkpoint publication requests, and the worker freezes completed MageFlow appearance,
MageFlow terminal/TREAD, or Qwen AO3 state into immutable, per-file-hashed, canonical tree
revisions before emitting the terminal event. The snapshot is independent of trainer retention,
binds producer/step/resume-grade/state inventory and parent lineage, and rejects symlink,
nonregular, outside-workspace, changing-source, or mutated-replay trees. This closes terminal
artifact visibility and durable handoff for the three native adapters. On-demand safe-point command
handling now has a shared ordered/atomic worker runtime, and scratch RWKV consumes its microbatch,
optimizer, evaluation, and checkpoint boundaries while persisting the effective control snapshot.
Both MageFlow routes also consume live learning-rate, eval-cadence, and caption-dropout controls,
rebase group rates without resetting schedule phase, and verify their control snapshots before
mutating resume state. Qwen's independent immutable-config overlay now does the same for learning
rate and eval cadence while preserving its PowerCool ratio. Lifecycle pause/checkpoint/cancel and
stronger content binding for exact resume remain separate lifecycle work.
The dashboard verifies the snapshot envelope and renders its optimizer step, resume grade, state
inventory, file/byte counts, tree digest, and parent lineage in the generic artifact stream.

The guarded GPU launch path is implemented: a root-owned hostd with explicitly non-root worker
credentials exposes process mutation only after startup recovery/admission, and TrainVM drives the
prepare -> durable journal receipt -> exec commit saga through the sealed host-process client.
Boot-scoped grants, exact GPU-device BPF fences, CPU/I/O controls, stopped-child identity, pidfds,
spawn/exit receipts, and restart adoption are bound and reattested. This is implementation status,
not deployment qualification: privileged real-host crash-window and driver-context tests remain a
release gate, and configurations that cannot prove strict launch capability fail closed.

Resource admission lowering is a separate pure module: experiment accelerator requirements become
one deterministic host bundle request, independently of adapter and trainer code. CPU-only external
workers stay disabled until a typed host process-slot resource is available; they must not borrow a
GPU or an unrelated mutex merely to satisfy the launcher protocol.

## Modded NanoGPT adaptation matrix

Reviewed source: [KellerJordan/modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt),
commit `bc1b58e83fa499c5df268bd6c8b98701273b96e7` (2026-07-27).

The superseding source-bound classification, per-technique compatibility/state/runtime evidence,
dispositions, and proposed follow-up cards are maintained in
[`MODDED_NANOGPT_INVENTORY.md`](MODDED_NANOGPT_INVENTORY.md).

## Remaining cross-family roadmap

### P0 — safe execution foundation

The implementation contract and adversarial gates for host-wide allocation, guarded launch, and
startup orphan recovery are defined in
[`HOST_RESOURCE_AUTHORITY.md`](HOST_RESOURCE_AUTHORITY.md).

- Boot-scoped authority time and journal migration with legacy lease quarantine.
- Centralized active-lease validation, renewal receipts, fencing-token checks, and host-wide resource
  locks.
- Startup orphan/resource audit before any process authority can be granted.
  The production configured auditor now constructs exact clock/head/occupancy evidence and blocks
  retained fences. The host ledger now exposes a bounded, integrity-checked read-only recovery view
  joining each active process intent to its exact grant and optional unclosed spawn receipt; durable
  process recovery now has a read-only pidfd/proc/executable/cgroup-inode double-attestation probe.
  A one-shot recovery set retains each exact pidfd for a single supervisor transfer; bounded policy
  can terminate and reconcile exact live processes, while independently gated conclusive-nonlive
  repair closes PID-absent or identity-superseded records only after exact cgroup and accelerator
  context closure. A bounded wake-driven startup FSM now keeps the one-shot admission audit behind
  complete process/release convergence, and the shared socket has one status/mutation listener
  router. One strict reflected daemon document now compiles the ledger, inventory, cgroup, socket,
  transport, service-role, challenge, restart-recovery, and startup-audit policies without
  authority defaults scattered across the executable. The foreground daemon assembly now proves
  live namespace/service-cgroup/cgroup-root/boot/inventory/journal identities before host-ledger
  initialization, drives bounded startup recovery/audit, and transactionally binds the unified
  endpoint only after admission. It exposes grant/release, stopped-child process launch, and
  restart reconciliation. Exact default-deny cgroup-device BPF programs and non-root worker
  credentials are now bound into durable launch/spawn evidence and reattested on recovery.
  Real-host crash qualification remains before training admission can rely on the complete path.
  The prerequisite v2 inventory capability model is implemented: exact assigned and shared NVIDIA
  driver nodes are topology-bound, capability drift degrades the resource, and unmapped partitions
  fail launch eligibility rather than borrowing their parent node.
- The stock-SQLite auxiliary pathname design is resolved with a declared trusted authority
  directory and one shared filesystem-authority implementation. Strict mode requires a dedicated
  non-root/non-`nobody` UID, mode 0700, `openat2` beneath/no-symlink resolution, pinned singleton
  inodes, and owner UID/GID attestation; journal exclusive WAL eliminates `-shm`. A controlled VFS
  was rejected: a hostile process sharing the authority UID can write the main database directly,
  so a VFS inside the authority process cannot close that trust-boundary failure. The
  network-namespace-local abstract socket remains only a cooperating-process namespace fence, not
  the host-wide GPU/resource fence.
- Extend the implemented wake-driven, restart-scanning service supervisor (admission, launch,
  terminal process/resource release, and exact lease renewal) with typed executors for
  compile/warmup/qualification nodes and dashboard-visible supervisor health.
  The qualification executor is implemented. A `qualify_cache` node is an exact builtin
  `trainvm.core` operation whose topology requires both a `cache.qualified` and a
  `cache.rejected` transition and an enabled `/spec/execution/qualify` phase, so a rejected
  candidate can never fall through as qualified. The supervisor resolves evidence through an
  authority-owned seam, re-runs the implemented qualification gate itself, and commits the verdict
  as a managed builtin receipt; a node without a configured evidence source fails closed, and a
  node whose evidence has not been published yet stays pending across wakes instead of inventing a
  verdict. The gate cannot release the fence it runs under, the journal refuses a self-contradictory
  verdict, and replay revalidates the verdict against its own receipt. Throughput and peak-memory
  gates are part of that same qualification receipt, so `benchmark` is not a separate executor.
  Compile and warmup now have a typed worker phase protocol. The authority lowers their immutable
  declarations into digest-bound Welcome requests and accepts only fenced receipts with exact step,
  timing, diagnostic, and before/after trajectory-state evidence. Successful or skipped phases must
  prove state restoration, recovery-safe repeats remain separate receipts, and the generic dashboard
  renders the receipt history. Scratch RWKV now consumes the requests, triggers lazy compilation,
  counts exact disposable warmup workloads, restores RNG/gradients, and content-hashes the complete
  model/optimizer trajectory around both phases. The MageFlow appearance route now also lowers the
  phase declaration into VAE/regional compilation and real-shaped disposable forwards/backwards,
  with complete tensor, schedule, control, RNG, and cursor proofs. The terminal/TREAD route shares
  that bridge and covers its configured REPA, immiscible-flow, weighted/directional, and loop
  objectives. Remaining RWKV paths, transformer, and vision-loop adoption plus privileged CUDA
  qualification remain before these phases are production-qualified.
- Fingerprinted cache namespaces; typed CPU/I/O policy lowering and recovery attestation are
  implemented, with privileged real-host qualification remaining.
- Declarative bounded Torch GPU profiling and dashboard trace artifacts are implemented; qualified
  Nsight launch profiles and richer cross-backend summaries remain.

### P1 — runtime speed qualification

- Build a reusable adapter capability manifest covering compile, CUDA graphs, FP8, attention/kernel
  backend, fused optimizer/loss, input pipeline, distributed execution, and an honest resume grade.
- Introduce the composable training-component registry for optimizers, parameter routing, learning
  rates/decay, activations, normalization, objectives, precision, clipping/accumulation, and
  curricula; migrate MageFlow, RWKV, and transformer workers away from duplicated string switches.
  The typed C++ registry, exact composition/submission locks, capability-augmented launches, native
  descriptor RPC, descriptor-generated dashboard composer, category-separated tensor runtime,
  typed worker-composition bridge, concrete MageFlow optimizer/schedule factories, and exhaustive
  appearance/terminal expert parameter routers are implemented. Global-norm gradient clipping is
  now a fourth physically independent tensor-runtime category with its own typed descriptor and is
  consumed by the MageFlow appearance/terminal, Qwen, and non-distributed scratch-RWKV paths,
  including the terminal loop-gate group as a separate slot.
  Qwen/RWKV now also pair a v2 no-decay AdamW mechanic with an independently resolved constant
  optimizer-step weight-decay schedule; the v1 combined descriptors remain compatibility-only.
  Both MageFlow expert trainers and
  the Qwen AO3 transformer continuation and scratch-RWKV PowerCool paths now consume verified
  worker slots and bind composition identity into resume state. RWKV retains a pure schedule state
  machine because its optimizer topology can change during training. The fixed fd-4 Python runner
  now dispatches a closed set of typed MageFlow/Qwen contracts, accepts only invocation-frozen
  inline configuration, confines every top-level model/data/checkpoint/cache/output path to
  authority workspace roots, and reports a receipted terminal event. Recursive immutable identity
  The baseline non-distributed scratch-RWKV path now has a closed typed v1 adapter configuration,
  path-authority lowering, terminal-checkpoint publication, sealed scalar metrics/heartbeats, and
  the shared bounded step profiler; research topology levers remain excluded until their state is
  represented by later adapter versions. Recursive immutable identity for paths referenced inside
  data/model manifests, trainer safe-point controls, broader RWKV/MLA/Engram variants, additional
  component families, and removal of the remaining legacy family switches remain.
- Keep implementation boundaries aligned with component ownership: optimizer algorithms,
  parameter routing, LR schedules, weight-decay schedules, activations, normalization, objectives,
  precision/scaling, gradient policy, and curricula live in separate source/test modules. Shared
  modules may not import family trainers; family adapters compose them through typed registry
  contracts. Add a component only when a runtime path consumes it, so the registry cannot advertise
  decorative configuration that training silently ignores. The Python runtime now physically
  separates optimizer mechanics, LR schedules, parameter routing, gradient clipping, weight-decay
  schedules, scratch-RWKV's fixed step-boundary gradient accumulation, and its linear-head token
  objective, BF16/FP32 precision policy, ChannelMix activation choice, LayerNorm construction, and
  optimizer-step context-length curriculum behind the stable `training_components` facade;
  category dependency checks prevent regressions. Distributed synchronization and mid-update
  accumulation remain for their first real adapter migrations, not as empty scaffolding.
- Make component state and schedule domains explicit in checkpoint manifests, with exhaustive
  parameter ownership and exact-resume trajectory tests.
- Add representative benchmark fixtures for MageFlow/flow, RWKV LM, transformer LM, vision/RWKV,
  fine-tuning, distillation, and post-training rather than one synthetic GEMM.
  The fixture matrix is declared in [`benchmark-matrix.v1.json`](benchmark-matrix.v1.json) and
  validated by `scripts/validate_benchmark_matrix.py`. It covers ten families across
  representative shape buckets and declares, per fixture, its effect class, portability, quality
  gate, and whether it exercises a curriculum-stage transition. The document declares what must be
  measured; it never carries argv, environment, an executable identity, or a measured result, and
  the validator rejects all of those.
  The effect class selects parity evidence exactly as the qualification contract requires, and
  each fixture restates its class's evidence so a hand-edit is a failure rather than drift: a
  serving fixture cannot acquire a gradient-parity claim, and a training fixture cannot drop
  resumed-trajectory parity to look qualified. The evidence vocabulary is pinned to the parity
  booleans actually implemented on `CacheQualificationEvidence`, so the matrix cannot require a
  dimension nothing computes. `portable` means the baseline is meaningful without an accelerator;
  a portable fixture that requires one is a validator error. This distinction was previously named
  here but nowhere defined.
  `trainvm qualify-evidence` reads a `trainvm.cache-qualification-evidence/v1`
  document on stdin, runs the implemented gate, and prints the receipt. Its exit
  status is the verdict: 0 qualified, 3 rejected with attributable reasons, 1 for
  a malformed document, so a rejection is a reportable outcome rather than
  indistinguishable from a crash. A benchmark runner measures; this decides. That
  boundary is what stops a runner reimplementing the thresholds and quietly
  disagreeing with the `qualify_cache` node that actually admits an optimization.
  The fixture runner executes cold, warmup, and timed phases in fresh processes and emits evidence
  for that existing qualification path. Portable fixtures run with accelerator visibility masked,
  so their receipts prove the CPU path was measured even on an accelerator host. Accelerator
  fixtures require a real CUDA device and fail closed when any timed cell lacks a device name,
  capability, or nonzero CUDA allocator peak. Accelerator fixtures and the existing portable LM
  fixtures still synthesize tensors in process and label their input waits
  `synthetic_in_process`. `rwkv.ao3-real-input` is the real-input
  exception: each measured step reads complete indexed AO3 source files of at least 128 KiB,
  decodes UTF-8, tokenizes the entire decoded text with the RWKV World `ztok` tokenizer, and then
  buckets IDs for the portable LM step. Its receipt publishes document-set size, source bytes,
  decoded characters, encoded tokens, and the deterministic selection digest; it fails instead of
  falling back when the corpus index, corpus root, or tokenizer is unavailable. Before every
  accelerator cell the runner bounds
  resident compute-process memory and records the process residency, total and used device memory,
  utilization, and applied allowance in the benchmark-run receipt, making ambient compositor
  conditions auditable. Benchmark remains not a separate executor — throughput and peak-memory
  gates belong to the existing qualification receipt rather than a parallel decision path. A native
  reflected loader with a pinned digest, matching the compatibility-catalog pattern, still belongs
  with authority integration; until a qualification node consumes the matrix this remains fixture
  evidence, not authority.
  The runner's default `--candidate eager` remains a neutral eager self-comparison.
  `--candidate compile` instead runs the eager baseline and a `torch.compile(dynamic=False)`
  candidate in separate fresh processes for every cold, warmup, and timed phase. The benchmark-run
  receipt publishes each arm's compilation-included first-step cost, warm timed throughput, peak
  memory, and deterministic final-loss/gradient-norm fingerprints; these diagnostics stay outside
  the exact `CacheQualificationEvidence` schema. Output and gradient parity use relative tolerance
  `1e-4` plus absolute tolerance `1e-6`. This allows ordinary float32 reduction-order noise while
  rejecting a materially different step, and the receipt records both absolute and relative
  observed deviations. Those measured booleans feed the existing native authority, so no speedup
  can qualify when either fingerprint fails parity. A compiled workload can also exercise an
  explicit out-of-set fallback bucket; shape mismatch may recompile or graph-break, but must return
  a measurable step whose loss and gradient fingerprint match eager at the same tolerance.
  `--compile-mode reduce-overhead` is exposed for device experiments, but it is not the default and
  carries no CUDA-graph qualification by itself. `reduce-overhead` enables CUDA graphs only on a
  compatible CUDA execution; until a device run demonstrates capture/replay, mismatch fallback,
  and parity, CUDA graphs remain explicitly unqualified.
- Audit existing optimizations against the qualification contract and publish portable versus
  machine-native receipts.
- Implement the broadly reusable Modded NanoGPT candidates in the matrix where an existing
  capability has a measured gap.

### P2 — guarded launcher and complete workflow coverage

- Add privilege-minimized process credentials, cgroups, sealed descriptors, pidfd supervision, and
  durable spawn/exit receipts.
- Register supported executable operations and effect-audited graph templates from the original
  `moe-mla` worktree. Do not register importable research oracles, shell supervisors, or design-only
  documents merely because they exist.
- Restore dashboard lifecycle, metrics, artifacts, eval galleries, generated/target side-by-side
  history, and scrubber from generic descriptors.

### P3 — research campaigns

- Run paired, multi-seed ablations for model/algorithm changes from Modded NanoGPT and other sources.
- Promote only candidates that improve the relevant family quality target at equal data/compute or
  reduce end-to-end time at equal quality.
- Keep rejected candidates and their evidence addressable; promotion changes an adapter profile,
  never historical runs.

## Completion criteria

The performance roadmap is complete when every supported workflow can be declared without a new
dashboard handler, every enabled optimization has immutable machine-readable evidence, a trace can
be captured and compared from the dashboard without editing trainer code, cache reuse is exact and
auditable, pause/resume preserves training semantics, and model-family-specific techniques cannot be
enabled on an incompatible adapter. Pause/resume equivalence applies only to adapters that declare
resumability; stop-only and non-resumable operations remain valid when represented honestly.
