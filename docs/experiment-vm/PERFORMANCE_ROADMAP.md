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
- Treat the current namespace document as an untrusted claim. Before reuse, an authority-owned
  builder must recompute it from the adapter registry, sealed launch, bounded runtime/host/device
  probes, and canonical compile manifests. Publication additionally requires the active owner/fence,
  symlink-safe paths, atomic immutable promotion, and an artifact-tree digest receipt; a claim digest
  alone never authorizes reuse.
- Add typed CPU/I/O policy: CPU set, CPU weight, I/O weight, worker count, OpenMP thread count, and
  nice level. Record effective policy and throttling as evidence.
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

No P0 item authorizes TrainVM to start a GPU process. Launch remains disabled until durable
boot-scoped leases, host resource fencing, orphan checks, sealed launch identity, and spawn/exit
receipts are complete.

## Modded NanoGPT adaptation matrix

Reviewed source: [KellerJordan/modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt),
commit `bc1b58e83fa499c5df268bd6c8b98701273b96e7` (2026-07-27). Its primary benchmark is a small GPT
trained on eight H100s to a fixed FineWeb validation-loss target. That makes its results valuable
leads, but not universal results for our single-GPU, RWKV, flow, vision, or post-training workloads.

### Shared or broadly reusable

| Technique | Dashboard/adapter treatment | Priority |
|---|---|---|
| Explicit compile plus disposable multi-shape warmup | Shared phase; warm a bounded declared boundary-shape set, destroy the warmup worker, and start timed work fresh | P0 |
| Static full-graph compilation | Runtime capability, shape-bucketed; retain eager fallback only when declared | P0 |
| Asynchronous batch fetch/index and pinned transfer | Shared input-pipeline capability with deterministic cursor and bounded queues | P0 |
| Fused optimizer steps and scalar hyperparameters that do not recompile graphs | Runtime capability; exact optimizer-state schema required | P1 |
| BF16 parameters with FP32 optimizer state | Precision capability; qualify per parameter class and checkpoint round trip | P1 |
| FP8 projection matmuls and fused quantization | Shape/hardware allowlist; compare FA4/torchao/custom kernels rather than assuming one backend | P1 |
| Fused activation/projection and loss kernels | Adapter kernel capability, parity first; retain trace evidence | P1 |
| Parameter-bank layout and transposed weights | Adapter checkpoint-layout version with conversion and resume tests | P2 |
| Communication/optimizer overlap and reduce-scatter ordering | Distributed-only capability; inactive on one GPU | P2 |

The repository already contains analogous pieces—pinned prefetch and nonblocking transfer, PyTorch
compile/megakernel paths, FA4 and FP8 experiments, fused losses/channel-mix kernels, FSDP prefetch,
and `SpectralMuon`. The first work is to expose and qualify those consistently, then close measured
gaps; it is not to add duplicate flags to every trainer.

### Transformer and language-model experiment capabilities

| Technique | Adaptation rule | Priority |
|---|---|---|
| Muon/NorMuon and Polar Express | Extend the common optimizer capability; explicit parameter routing keeps embeddings, scalar/vector gates, norms, and biases on an auxiliary optimizer | P1 |
| Cautious weight decay tied to LR | Promote the existing `SpectralMuon` option into typed schedules and compare against ordinary decoupled decay | P1 |
| Less-frequent auxiliary Adam updates / selective accumulation | Typed per-parameter cadence; preserve effective sample weighting and resume phase | P1 |
| Batch-size and maximum-sequence curricula | Generic schedule segments with token-normalized metrics and warmed boundary shapes | P1 |
| Document-aligned batch starts and maximum-document packing | Transformer data-packing capability with deterministic cursor; compare utilization and quality against the current packer | P1 |
| FP8 LM head and MLP up projection | Transformer kernel capability with loss/logit parity and architecture-specific scaling | P1 |
| Fused softcapped cross entropy | Optional transformer loss capability; softcap itself is algorithmic, while fusion can qualify independently | P1 |
| Flash Attention 3, long/short windows, window warmup, YaRN | Transformer attention experiment; do not apply to RWKV or flow blocks by name | P2 |
| Multi-token prediction | Reuse existing MTP/L-MTP/JTP support behind typed loss and schedule capabilities | P2 |
| Accumulate selected embedding/head gradients on a different cadence | Typed parameter-group update/accumulation schedule with equal-token and resume-phase checks | P2 |
| Tied embedding/head followed by scheduled split | Checkpoint-topology transition requiring optimizer-state conversion receipt | P2 |
| Rotary embeddings, QK norm, ReLU-squared, partial key offset, paired heads | Architecture experiments requiring fresh quality campaigns; existing equivalents are reused, not cloned | P3 |
| Zero-initialized projections and muP-like scaling | Fresh-initialization experiment only; cannot be retrofitted into continuation runs | P3 |
| Embedding-to-block, block-to-block, value-embedding, MUDD/U-Net-style and gated skips | Architecture experiments, not runtime optimizations; opt-in and adapter-versioned | P3 |
| Shared activation input for later attention layers and exponential residual decay | Transformer topology experiment with activation-memory and quality ablations | P3 |
| Bigram hash embeddings, smear/one-token lookback, sparse attention gates | Research capabilities only after isolated ablations and scale tests | P3 |
| Learnable XSA and lightweight dynamically composable MHA | Transformer attention research; requires clean implementation and scale tests before speed qualification | P3 |

For RWKV, only mathematically compatible capabilities are candidates. Optimizer routing, fused loss,
precision, compilation, input staging, and schedule machinery can be shared. Transformer attention
windows, value embeddings, QK-specific changes, and GPT residual topology must not be projected onto
RWKV by analogy. RWKV-specific fused recurrent kernels and state handling remain separate adapter
capabilities with the same evidence contract.

For diffusion/flow/image/video trainers, reusable candidates are compilation/warmup, static shape
buckets, input staging, FP8 projection qualification, fused optimizers, and profiling. Token-only
architectural or language-loss changes do not apply. Image-size/aspect buckets and generated-image
eval suites replace sequence-length and perplexity gates.

## Remaining cross-family roadmap

### P0 — safe execution foundation

The implementation contract and adversarial gates for host-wide allocation, guarded launch, and
startup orphan recovery are defined in
[`HOST_RESOURCE_AUTHORITY.md`](HOST_RESOURCE_AUTHORITY.md).

- Boot-scoped authority time and journal migration with legacy lease quarantine.
- Centralized active-lease validation, renewal receipts, fencing-token checks, and host-wide resource
  locks.
- Startup orphan/resource audit before any process authority can be granted.
- Resolve the remaining stock-SQLite auxiliary pathname race in the declared host threat model:
  trusted isolated authority directory, controlled VFS, or host-global lock broker. Do not treat
  the network-namespace-local abstract socket as the host-wide GPU/resource fence.
- Production reconciliation coordinator for compile/warmup/qualification nodes.
- Fingerprinted cache namespaces and typed CPU/I/O policy.
- Declarative bounded GPU profiling and dashboard trace artifacts.

### P1 — runtime speed qualification

- Build a reusable adapter capability manifest covering compile, CUDA graphs, FP8, attention/kernel
  backend, fused optimizer/loss, input pipeline, distributed execution, and an honest resume grade.
- Add representative benchmark fixtures for MageFlow/flow, RWKV LM, transformer LM, vision/RWKV,
  fine-tuning, distillation, and post-training rather than one synthetic GEMM.
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
