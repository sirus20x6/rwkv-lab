# Modded NanoGPT candidate inventory and compatibility map

Status: source-bound classification for `dashboard/declarative-vm-fsm`.

Reviewed source: [KellerJordan/modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt),
commit `bc1b58e83fa499c5df268bd6c8b98701273b96e7` (2026-07-27).

This document supersedes the adaptation matrix formerly embedded in
[`PERFORMANCE_ROADMAP.md`](PERFORMANCE_ROADMAP.md). It classifies techniques; it does not import an
upstream training recipe. The reviewed program is a fixed, small GPT-2-scale FineWeb speedrun on
eight H100s to a target validation loss. Its schedules, shapes, hardware balance, and optimizer
hyperparameters are leads for controlled experiments, not defaults for single-GPU, RWKV,
flow/diffusion, vision, conversion, or post-training workloads.

The gate order is correctness/parity, determinism, resume/state, and equal quality before any speed
claim. A candidate remains a measured gap until it passes those gates on its target adapter and the
adapter's complete runtime closure. Throughput evidence from a different model family, shape,
precision, accelerator, or token budget does not satisfy the gate.

## Classification rules

`adopted-existing` means an equivalent mechanism is present in this repository; it does not mean
that every adapter exposes it through the typed component registry or that it has been qualified on
every accelerator. `candidate-measured-gap` means the technique is mathematically applicable to at
least one supported family but still needs an explicit card and the ordered gates above. `rejected`
requires a concrete incompatibility and is not shorthand for low priority.

The model-family applicability fields use these abbreviations:

- **T/MLA:** transformer and multi-head latent-attention training;
- **R:** RWKV scratch and pretrained/continuation training, distinguished where topology matters;
- **V/MM:** vision and multimodal training;
- **M/D:** MageFlow and other diffusion/flow training;
- **C/D:** conversion and distillation;
- **P/RLVR:** post-training, preference optimization, and RLVR.

The component-slot names are the typed categories defined in
[`TRAINING_COMPONENTS.md`](TRAINING_COMPONENTS.md): `optimizer`, `parameter router`,
`learning-rate schedule`, `weight-decay schedule`, `activation`, `normalization`, `objective`,
`precision/scaling`, `gradient clipping`, `gradient accumulation`, `curriculum`, and
`metric reducer`. “None exists yet” means the concern must not be smuggled into a nearby category.

Muon LR units require special treatment. `GuardedMuonClip` consumes the external MuonClip update
with its `0.4*sqrt(max_dim)` corrected-RMS amplifier
(`src/rwkv_lab/muon_helpers.py:6-9`, `src/rwkv_lab/train_mla.py:603-617`). At width 4096 the factor
is about 25.6, and repo evidence records 25–60x excessive effective steps when vanilla-Muon LRs
were reused (`src/rwkv_lab/train_mla.py:605-611`). Raw modded-nanogpt Muon LRs are therefore not
transferable to MuonClip or GuardedMuonClip. Every optimizer comparison must match effective update
RMS, state precision, parameter routing, token count, and schedule phase before comparing speed or
quality.

## Prior-matrix coverage

Every row from the superseded matrix is carried forward below. Split rows retain the source row's
priority.

| Prior matrix row | Priority | Inventory ids |
|---|---:|---|
| Explicit compile plus disposable multi-shape warmup | P0 | `mng.compile-disposable-prewarm` |
| Static full-graph compilation | P0 | `mng.compile-static-fullgraph` |
| Asynchronous batch fetch/index and pinned transfer | P0 | `mng.async-pinned-batch-fetch` |
| Fused optimizer steps and scalar hyperparameters that do not recompile graphs | P1 | `mng.fused-optimizer-step` |
| BF16 parameters with FP32 optimizer state | P1 | `mng.bf16-parameters-fp32-state` |
| FP8 projection matmuls and fused quantization | P1 | `mng.fp8-projection-matmuls` |
| Fused activation/projection and loss kernels | P1 | `mng.fused-activation-projection`, `mng.fused-loss-kernel` |
| Parameter-bank layout and transposed weights | P2 | `mng.parameter-bank-transposed-layout` |
| Communication/optimizer overlap and reduce-scatter ordering | P2 | `mng.communication-optimizer-overlap`, `mng.backward-prefetch-overlap` |
| Muon/NorMuon and Polar Express | P1 | `mng.muon-newton-schulz`, `mng.normuon-polar-express`, `mng.adam-muon-parameter-split` |
| Cautious weight decay tied to LR | P1 | `mng.cautious-weight-decay` |
| Less-frequent auxiliary Adam updates / selective accumulation | P1 | `mng.aux-adam-update-cadence` |
| Batch-size and maximum-sequence curricula | P1 | `mng.batch-size-ramp`, `mng.sequence-length-curriculum` |
| Document-aligned batch starts and maximum-document packing | P1 | `mng.document-aligned-token-packing`, `mng.document-boundary-masking` |
| FP8 LM head and MLP up projection | P1 | `mng.fp8-lm-head`, `mng.fp8-mlp-up-projection` |
| Fused softcapped cross entropy | P1 | `mng.logit-softcap-cross-entropy` |
| Flash Attention 3, long/short windows, window warmup, YaRN | P2 | `mng.flash-attention-3`, `mng.windowed-block-causal-attention`, `mng.flexattention`, `mng.yarn-rotary-scaling` |
| Multi-token prediction | P2 | `mng.multi-token-prediction` |
| Accumulate selected embedding/head gradients on a different cadence | P2 | `mng.selected-gradient-cadence` |
| Tied embedding/head followed by scheduled split | P2 | `mng.scheduled-untied-embeddings` |
| Rotary embeddings, QK norm, ReLU-squared, partial key offset, paired heads | P3 | `mng.rotary-embeddings`, `mng.qk-norm`, `mng.relu-squared`, `mng.partial-key-offset-paired-heads` |
| Zero-initialized projections and muP-like scaling | P3 | `mng.zero-init-projections`, `mng.mup-fan-in-scaling` |
| Embedding-to-block, block-to-block, value-embedding, MUDD/U-Net-style and gated skips | P3 | `mng.embedding-to-block-skip`, `mng.block-to-block-skip`, `mng.value-embedding-lambda-residual`, `mng.mudd-unet-cross-block-skip`, `mng.gated-skip-connections` |
| Shared activation input for later attention layers and exponential residual decay | P3 | `mng.shared-attention-activation-input`, `mng.exponential-residual-decay` |
| Bigram hash embeddings, smear/one-token lookback, sparse attention gates | P3 | `mng.bigram-hash-embeddings`, `mng.smear-one-token-lookback`, `mng.sparse-attention-gates` |
| Learnable XSA and lightweight dynamically composable MHA | P3 | `mng.learnable-xsa`, `mng.dynamic-composable-mha` |

## Compile techniques

### `mng.compile-disposable-prewarm`

- **id:** `mng.compile-disposable-prewarm`
- **technique:** Compile a bounded set of declared boundary shapes in a disposable warmup phase. Prior matrix priority: P0.
- **model family applicability:** T/MLA, R, V/MM, M/D, C/D, and P/RLVR can use it when their adapter declares stable shape buckets; dynamic control flow or unbounded ragged shapes must remain eager or use separately qualified buckets.
- **operation:** schedule
- **component slot:** none exists yet; this is an adapter operation capability, not a tensor component.
- **state / resume impact:** Compiled artifacts are not checkpoint state. Warmup must restore RNG and must not advance optimizer, scheduler, dataloader, scaler, or model buffers; otherwise an `exact` run is silently downgraded. A disposable worker also prevents allocator/compiler state from contaminating timed work.
- **kernel / runtime requirement:** `torch.compile` and backend caches must be inside the per-adapter runtime closure; the worker must declare every warmed shape and eager fallback policy.
- **existing equivalent in this repo:** `src/rwkv_lab/rwkv_pretrain.py:1831-1849` prewarms curriculum shapes and saves RNG, but it runs in the training worker rather than destroying a disposable worker.
- **disposition:** `candidate-measured-gap`
- **rationale:** Multi-shape prewarm exists, but the prior matrix's disposable-worker boundary is not implemented. Qualification must prove parity, RNG neutrality, clean timed-worker startup, memory bounds, and equal quality before a speed claim.
- **follow-up card:** “Isolate compile prewarm in disposable workers” — 04 Performance & Kernel Qualification.

### `mng.compile-static-fullgraph`

- **id:** `mng.compile-static-fullgraph`
- **technique:** Require static full-graph compilation for declared shapes. Prior matrix priority: P0.
- **model family applicability:** T/MLA, R, V/MM, M/D, C/D, and P/RLVR apply only where a complete forward/backward boundary is static; recurrent state, variable masks, and evaluator branches need their own graphs or an explicit eager declaration.
- **operation:** model construction
- **component slot:** none exists yet; compile is an operation capability.
- **state / resume impact:** No checkpoint state is added. Graph selection, shape bucket, backend, and fallback policy are run identity; resuming under a different graph contract is at most `compatible` until parity is re-established.
- **kernel / runtime requirement:** PyTorch Dynamo/Inductor `torch.compile(dynamic=False, fullgraph=True)` inside the adapter runtime closure.
- **existing equivalent in this repo:** `src/rwkv_lab/rwkv_pretrain.py:884-889` exposes the fail-closed flag and `src/rwkv_lab/rwkv_pretrain.py:1723-1727` applies the compile boundary.
- **disposition:** `adopted-existing`
- **rationale:** The mechanism and fail-closed full-graph contract already exist. Other adapters may adopt it only after the standard parity, determinism, resume, and equal-quality gates.
- **follow-up card:** none.

## Language objective, data, attention, and related distributed techniques

### `mng.document-aligned-token-packing`

- **id:** `mng.document-aligned-token-packing`
- **technique:** Sample within documents, pack whole documents into maximum-length rows, mask padding, and choose batch/bucket probabilities by real-token count. Prior matrix priority: P1.
- **model family applicability:** T/MLA and R apply to language corpora; V/MM can use sample/token buckets but not text rules by name; M/D uses aspect/frame buckets; C/D must align teacher/student rows; P/RLVR must keep conversation or preference groups atomic.
- **operation:** dataloader
- **component slot:** none exists yet.
- **state / resume impact:** Exact resume requires source identity, packing manifest, bucket probabilities, authoritative sampler RNG/cursor, and consumed-row position. Prefetched rows must commit the cursor only on consumption.
- **kernel / runtime requirement:** CPU/NumPy packing plus pinned transfer or GPU indexing in the adapter closure; no custom GPU kernel.
- **existing equivalent in this repo:** `src/rwkv_lab/build_corpus.py:116-185` performs whole-document best-fit packing with explicit long-document chunking; `src/rwkv_lab/rwkv_pretrain.py:1441-1467` uses real-token-weighted buckets and reciprocal batches; `src/rwkv_lab/rwkv_pretrain.py:1479-1507` samples within documents.
- **disposition:** `adopted-existing`
- **rationale:** The repo has deterministic semantic packing and step-time-aware CPU/GPU sampling paths. This does not imply equivalence to an upstream corpus order or permit attention to cross document boundaries.
- **follow-up card:** none.

### `mng.document-boundary-masking`

- **id:** `mng.document-boundary-masking`
- **technique:** Prevent attention or recurrent state from crossing packed-document boundaries while retaining packed-row utilization. Prior matrix priority: P1.
- **model family applicability:** T/MLA needs a block-diagonal causal attention mask; R needs reset masks for matrix state, token shifts, and rotary positions; V/MM applies only to concatenated independent samples; M/D generally does not concatenate independent denoising examples; C/D and P/RLVR must preserve teacher/conversation/preference boundaries.
- **operation:** dataloader
- **component slot:** none exists yet.
- **state / resume impact:** Boundary IDs/masks are deterministic batch data and add no checkpoint tensor, but packing cursor and boundary convention are run identity. Omitting the mask after resume changes examples and downgrades exactness.
- **kernel / runtime requirement:** T/MLA needs an SDPA/FlexAttention/other kernel that accepts block-causal masks; R reset-mask support must land in its recurrent kernel closure.
- **existing equivalent in this repo:** `src/rwkv_lab/rwkv8_deltanet.py:699-706` resets token shift at boundaries and `src/rwkv_lab/rwkv8_deltanet.py:762-771` resets RoPE positions; `src/rwkv_lab/rwkv_pretrain.py:317-328` validates reset-mask packing. No transformer block-diagonal document mask was found, and `src/rwkv_lab/lookahead_module.py:25` explicitly says its auxiliary objective does not apply document-boundary masking.
- **disposition:** `candidate-measured-gap`
- **rationale:** RWKV has a family-correct equivalent, but the upstream transformer technique is not implemented for T/MLA packing. The gap must prove mask correctness, deterministic packing/resume, equal quality, memory, and then utilization/step time.
- **follow-up card:** “Add deterministic transformer document masks and packing cursors” — 02 Trainer Adapters & Components.

### `mng.fp8-lm-head`

- **id:** `mng.fp8-lm-head`
- **technique:** Run the vocabulary projection in FP8 while preserving loss and gradient semantics. Prior matrix priority: P1.
- **model family applicability:** T/MLA, R, C/D, and P/RLVR apply to large vocabulary heads; V/MM applies to token decoders; M/D does not apply without a token head.
- **operation:** precision
- **component slot:** precision/scaling
- **state / resume impact:** The current TorchAO path retains master weights and does not add authoritative checkpoint state. A stateful scaling recipe must checkpoint its scale history for exact resume; tied/scheduled-split heads are currently incompatible with this route.
- **kernel / runtime requirement:** TorchAO rowwise FP8, aligned dimensions, FP8-capable GPU, optional `torch.compile`, and fused or bounded CE, all inside the adapter closure.
- **existing equivalent in this repo:** `src/rwkv_lab/rwkv_pretrain.py:554-592` includes the head with a rowwise high-precision-gradient recipe; `src/rwkv_lab/fused_ce.py:136-142` calls the module so Float8Linear remains active.
- **disposition:** `adopted-existing`
- **rationale:** The head route exists with explicit exclusions and fallback. Loss/logit, gradient, resume, equal-quality, memory, and throughput evidence remain device- and vocabulary-specific.
- **follow-up card:** none.

### `mng.fp8-mlp-up-projection`

- **id:** `mng.fp8-mlp-up-projection`
- **technique:** Cache or fuse FP8 quantization for the first MLP/ChannelMix projection and use BF16-compatible backward. Prior matrix priority: P1.
- **model family applicability:** R applies to native ChannelMix; T/MLA, V/MM, and M/D apply only to matching projection/activation equations; C/D and P/RLVR inherit the target model.
- **operation:** precision
- **component slot:** precision/scaling
- **state / resume impact:** Cached FP8 weights are derived from authoritative live weights after each optimizer step, so they need not be checkpointed if deterministically refreshed. Persisting stale cache bytes or scale history without a codec would downgrade exactness.
- **kernel / runtime requirement:** Custom autograd, optional Triton, `_scaled_mm`, E4M3 support, and compute capability 9+ for the scaled path, all in the adapter closure.
- **existing equivalent in this repo:** `src/rwkv_lab/fused_channelmix.py:144-171` quantizes and applies the cached FP8 up weight; `src/rwkv_lab/fused_channelmix.py:174-235` retains BF16 backward; `src/rwkv_lab/rwkv_pretrain.py:2051-2052` refreshes after optimizer steps.
- **disposition:** `adopted-existing`
- **rationale:** The custom-autograd FP8-up mechanism exists for the exact RWKV equation. It is not portable to SwiGLU/GELU blocks without a new parity-qualified kernel.
- **follow-up card:** none.

### `mng.logit-softcap-cross-entropy`

- **id:** `mng.logit-softcap-cross-entropy`
- **technique:** Apply a tanh/logit softcap before cross entropy, optionally fused with the vocabulary projection and CE. Prior matrix priority: P1.
- **model family applicability:** T/MLA, R, C/D, P/RLVR, and token-producing V/MM apply as an objective change; M/D does not apply to continuous flow losses. Distillation must specify whether teacher, student, or both distributions are capped.
- **operation:** loss
- **component slot:** objective
- **state / resume impact:** Stateless, but cap value, placement, reduction, and precision are objective identity. Enabling it on resume changes the loss and is at most compatible.
- **kernel / runtime requirement:** A portable PyTorch tanh reference first; optional Triton/custom-autograd fused CE must be inside the adapter closure.
- **existing equivalent in this repo:** `src/rwkv_lab/fused_ce.py:116-153` provides fused/bounded CE but no logit cap. `src/rwkv_lab/looped_rwkv.py:353-367` uses tanh capping for loop gates, which is not a loss equivalent.
- **disposition:** `candidate-measured-gap`
- **rationale:** Fusion and softcapping must be separated: the existing CE kernel is adopted, while the algorithmic cap needs loss/gradient parity, determinism, resume identity, equal quality, and then kernel timing.
- **follow-up card:** “Add a typed softcapped LM objective and qualified kernel” — 02 Trainer Adapters & Components.

### `mng.windowed-block-causal-attention`

- **id:** `mng.windowed-block-causal-attention`
- **technique:** Mix long/short sliding windows and block-causal attention, with an optimizer-step window-size warmup schedule. Prior matrix priority: P2.
- **model family applicability:** T/MLA applies directly; V/MM may use spatial/temporal windows with different masks; R token mixing does not port because recurrence is not QK attention; M/D applies only to attention blocks, not flow equations; C/D needs teacher/student context equivalence; P/RLVR inherits transformer attention and conversation masks.
- **operation:** model construction
- **component slot:** curriculum for window schedule; no attention-topology component exists yet.
- **state / resume impact:** Window pattern is model semantics; schedule phase must derive from or checkpoint the optimizer cursor. KV/cache layout and mask convention are checkpoint/run identity even without new trainable tensors; changing any of them prevents `exact` resume.
- **kernel / runtime requirement:** Window/block-causal capable attention backend, shape buckets, and mask compiler inside the adapter closure; no RWKV kernel reuse.
- **existing equivalent in this repo:** No training implementation found. `src/rwkv_lab/mla_module.py:285-291` uses ordinary SDPA causal/mask attention, while `src/rwkv_lab/rosa.py:29-30` limits a separate module to already-windowed softmax layers.
- **disposition:** `candidate-measured-gap`
- **rationale:** This is an attention algorithm and schedule, not a generic runtime flag. It requires mask parity, determinism, schedule resume, equal quality at full context, memory, and then step-time evidence.
- **follow-up card:** “Qualify transformer window and block-causal curricula” — 04 Performance & Kernel Qualification.

### `mng.flexattention`

- **id:** `mng.flexattention`
- **technique:** Express score modification and block masks through PyTorch FlexAttention. Prior matrix priority: P2, split from the attention-kernel row.
- **model family applicability:** T/MLA and attention-based V/MM/M/D apply; R token mixing does not; C/D and P/RLVR inherit the target attention masks and must preserve boundary semantics.
- **operation:** model construction
- **component slot:** none exists yet.
- **state / resume impact:** Stateless kernel selection, but score-mod and block-mask functions are model/data identity. Changing them across resume is not exact.
- **kernel / runtime requirement:** A PyTorch build with FlexAttention, compiled block-mask support, and supported GPU architecture in the adapter runtime closure; eager/reference fallback is required for parity.
- **existing equivalent in this repo:** No `flex_attention`, `FlexAttention`, or `BlockMask` implementation was found; `src/rwkv_lab/mla_module.py:285-291` uses SDPA.
- **disposition:** `candidate-measured-gap`
- **rationale:** FlexAttention is useful only after exact score/mask semantics and runtime support are proven. It must not be used to approximate document or window masks silently.
- **follow-up card:** “Qualify FlexAttention mask backends per adapter” — 04 Performance & Kernel Qualification.

### `mng.flash-attention-3`

- **id:** `mng.flash-attention-3`
- **technique:** Use Flash Attention 3 for supported attention shapes. Prior matrix priority: P2.
- **model family applicability:** T/MLA and attention blocks in V/MM/M/D apply; R recurrence does not; C/D and P/RLVR inherit supported target layers.
- **operation:** model construction
- **component slot:** none exists yet.
- **state / resume impact:** No persistent state; backend, mask semantics, accumulation precision, and dropout RNG behavior are run identity. A backend switch needs numerical and resumed-trajectory parity before `exact` can be claimed.
- **kernel / runtime requirement:** FA3 package/build and supported Hopper-class CUDA closure; shape/mask allowlists and an SDPA reference fallback belong in the adapter closure.
- **existing equivalent in this repo:** `src/rwkv_lab/mla_module.py:285-291` uses PyTorch SDPA. The repo's flash-attn Triton CE in `src/rwkv_lab/fused_ce.py:17-48` is a loss kernel, not Flash Attention 3.
- **disposition:** `candidate-measured-gap`
- **rationale:** No FA3 attention implementation was found. Qualification must precede any claim based on the upstream H100 result.
- **follow-up card:** “Qualify Flash Attention 3 against SDPA” — 04 Performance & Kernel Qualification.

### `mng.yarn-rotary-scaling`

- **id:** `mng.yarn-rotary-scaling`
- **technique:** Apply YaRN-style rotary scaling for changed context lengths. Prior matrix priority: P2.
- **model family applicability:** T/MLA with RoPE applies; attention-free R does not port by analogy, while the repo's special RAD-RWKV rotary graft would need a separate derivation; V/MM applies only to rotary token axes; M/D only to rotary attention blocks; C/D/P/RLVR require checkpoint/config compatibility.
- **operation:** model construction
- **component slot:** none exists yet.
- **state / resume impact:** Usually no extra tensor state, but rotary base/scaling and context regime are model identity. Changing them during continuation without a declared schedule is not exact.
- **kernel / runtime requirement:** RoPE-capable attention kernels with the same scaling convention, inside the adapter closure.
- **existing equivalent in this repo:** Standard/partial RoPE exists in `src/rwkv_lab/mla_module.py:30-32` and `src/rwkv_lab/mla_module.py:238-248`; no YaRN implementation was found.
- **disposition:** `candidate-measured-gap`
- **rationale:** Ordinary RoPE is not evidence for YaRN equivalence. Context extrapolation quality must be established before runtime measurement.
- **follow-up card:** “Qualify YaRN context scaling for transformer adapters” — 02 Trainer Adapters & Components.

### `mng.multi-token-prediction`

- **id:** `mng.multi-token-prediction`
- **technique:** Add auxiliary heads/layers that predict multiple future tokens. Prior matrix priority: P2.
- **model family applicability:** T/MLA and R apply; V/MM applies to autoregressive token decoders; M/D does not apply to continuous flow objectives; C/D can distill auxiliary targets; P/RLVR usually inherits a pretrained MTP topology and must decide whether auxiliary loss remains active.
- **operation:** loss
- **component slot:** objective
- **state / resume impact:** Adds model parameters and optimizer state; loss weights/cooldown and target offsets are run identity. Exact resume requires head state, optimizer moments, topology, and schedule cursor.
- **kernel / runtime requirement:** Ordinary PyTorch/HF layers plus bounded/fused CE in the adapter closure; no mandatory custom kernel.
- **existing equivalent in this repo:** `src/rwkv_lab/mtp_module.py:41-118` implements Qwen MTP and `src/rwkv_lab/lookahead_module.py:216-249` implements L-MTP heads/losses.
- **disposition:** `adopted-existing`
- **rationale:** Several MTP variants already exist and must be reused rather than cloned. A typed objective entry and family-specific quality evidence may still be needed before production registry exposure.
- **follow-up card:** none.

### `mng.selected-gradient-cadence`

- **id:** `mng.selected-gradient-cadence`
- **technique:** Accumulate embedding/head gradients and update them on a different cadence from matrix parameters. Prior matrix priority: P2.
- **model family applicability:** T/MLA, R, token-producing V/MM, C/D, and P/RLVR apply; M/D applies only to analogous selected parameter groups, not by embedding/head names.
- **operation:** parameter routing
- **component slot:** parameter router plus gradient accumulation; no composed per-group cadence component exists yet.
- **state / resume impact:** Selected-group accumulators, counts, optimizer moments, group ownership, and phase are exact state. Missing any pending gradient accumulator downgrades exactness.
- **kernel / runtime requirement:** Foreach optimizer operations; distributed variants also require deterministic group-specific synchronization in the adapter closure.
- **existing equivalent in this repo:** `src/rwkv_lab/rwkv_pretrain.py:510-525` routes embedding/head/norm tensors to fallback Adam, and `src/rwkv_lab/spectral_muon.py:554-618` applies the checkpointed Adam cadence to that group.
- **disposition:** `adopted-existing`
- **rationale:** For the repo's Muon route, selected vocabulary/head-like parameters already receive the independent accumulated cadence. Other optimizers must not grow a duplicate trainer-local schedule.
- **follow-up card:** none.

### `mng.tail-ema-readout`

- **id:** `mng.tail-ema-readout`
- **technique:** Start an FP32 EMA late, exclude embeddings, and partially blend it only for evaluation. Prior matrix priority: P2, absorbed from existing modded-nanogpt-provenance levers.
- **model family applicability:** T/MLA, R, V/MM, M/D, C/D, and P/RLVR can use model averaging, but parameter exclusions and evaluation semantics are family-specific; sharded models need a distributed codec.
- **operation:** schedule
- **component slot:** metric reducer is not sufficient; no model-averaging/readout component exists yet.
- **state / resume impact:** FP32 shadows, update count, start step, horizon/decay, blend, and parameter-name mapping are mandatory exact checkpoint state. Re-seeding on resume changes evaluation and future EMA.
- **kernel / runtime requirement:** PyTorch foreach lerp and temporary eval swaps; sharded support would belong in the adapter closure.
- **existing equivalent in this repo:** `src/rwkv_lab/optimizer_speedups.py:41-150` implements and checkpoints TailEMA; `src/rwkv_lab/rwkv_pretrain.py:1521-1534` constructs it and `src/rwkv_lab/rwkv_pretrain.py:2140-2141` saves it.
- **disposition:** `adopted-existing`
- **rationale:** The stateful evaluation readout is already implemented for non-FSDP scratch RWKV. Its state contract prevents treating it as a stateless metric option.
- **follow-up card:** none.

### `mng.lmtp-tail-cooldown`

- **id:** `mng.lmtp-tail-cooldown`
- **technique:** Linearly remove L-MTP auxiliary loss/compute during the training tail. Prior matrix priority: P2, inherited from MTP and existing provenance levers.
- **model family applicability:** T/MLA and R with L-MTP apply; token V/MM may adapt it; M/D does not; C/D and P/RLVR apply only when that auxiliary objective is active.
- **operation:** schedule
- **component slot:** objective; no objective-weight schedule slot exists yet.
- **state / resume impact:** The immutable start fraction and restored optimizer-step cursor determine phase; auxiliary-head model/optimizer state remains required. Resetting the multiplier on resume downgrades exactness.
- **kernel / runtime requirement:** No custom kernel; ordinary objective code remains inside the adapter runtime closure, and reduced tail compute is meaningful only if the adapter actually bypasses zero-weight work.
- **existing equivalent in this repo:** `src/rwkv_lab/optimizer_speedups.py:18-27` implements the tail multiplier and `src/rwkv_lab/rwkv_pretrain.py:1909-1914` applies it to L-MTP.
- **disposition:** `adopted-existing`
- **rationale:** The schedule exists. It must retain equal-token and final-quality comparisons because changing auxiliary compute also changes the optimization objective.
- **follow-up card:** none.

### `mng.sparse-embedding-communication`

- **id:** `mng.sparse-embedding-communication`
- **technique:** Keep embedding-like tables replicated and all-gather only touched gradient rows. Prior matrix priority: P2, inherited from selected embedding/head communication and existing provenance levers.
- **model family applicability:** T/MLA, R, token V/MM, C/D, and P/RLVR apply to sparse tables; M/D applies only if it has analogous sparse conditioning tables.
- **operation:** distributed sync
- **component slot:** none exists yet.
- **state / resume impact:** Parameters and optimizer state remain replicated; touched-row hints are per-step data. Exactness requires unioning hints with actual nonzero rows and identical averaging, plus checkpointing the replicated optimizer state.
- **kernel / runtime requirement:** PyTorch distributed all-gather in the adapter closure; payload bounds and world-size behavior require qualification.
- **existing equivalent in this repo:** `src/rwkv_lab/distributed.py:120-177` gathers touched indices and FP32 row gradients safely; `src/rwkv_lab/rwkv_pretrain.py:1621-1654` excludes selected tables from FSDP and activates the path.
- **disposition:** `adopted-existing`
- **rationale:** The exact sparse-row communication mechanism exists and defensively includes real nonzero rows. Benefit depends on vocabulary touch rate and world size.
- **follow-up card:** none.

### `mng.backward-prefetch-overlap`

- **id:** `mng.backward-prefetch-overlap`
- **technique:** Prefetch the next sharded blocks during forward and reverse-order predecessors during backward. Prior matrix priority: P2.
- **model family applicability:** T/MLA, R, V/MM, M/D, C/D, and P/RLVR apply only to static distributed block traversal; dynamic MoE/routing requires different dependency discovery.
- **operation:** distributed sync
- **component slot:** none exists yet.
- **state / resume impact:** No new checkpoint state; traversal/prefetch depth and sharding topology are runtime identity. It preserves exactness only if collective/reduction ordering remains deterministic.
- **kernel / runtime requirement:** PyTorch composable FSDP2 prefetch APIs in the per-adapter runtime closure.
- **existing equivalent in this repo:** `src/rwkv_lab/distributed.py:87-110` installs adjacent forward and backward prefetch from static block order; `src/rwkv_lab/rwkv_pretrain.py:1643-1654` consumes the configured depth.
- **disposition:** `adopted-existing`
- **rationale:** Explicit backward prefetch is already present. Full optimizer/reduce-scatter overlap remains separately classified as a gap.
- **follow-up card:** none.

## Shared runtime, precision, and kernel techniques

### `mng.async-pinned-batch-fetch`

- **id:** `mng.async-pinned-batch-fetch`
- **technique:** Fetch/index the next batch asynchronously, pin it, and use nonblocking host-to-device transfer. Prior matrix priority: P0.
- **model family applicability:** T/MLA, R, V/MM, M/D, C/D, and P/RLVR apply when CPU decode/index work is measurable; GPU-resident data and stateful online sampling need a different cursor boundary.
- **operation:** dataloader
- **component slot:** none exists yet.
- **state / resume impact:** Queue contents need not be serialized if producer RNG is private and commits only on consumption. Advancing the authoritative sampler when a future is submitted would downgrade `exact`; the current implementation avoids that.
- **kernel / runtime requirement:** Pinned host memory, a bounded CPU executor, and nonblocking transfer inside the adapter runtime closure; no custom GPU kernel.
- **existing equivalent in this repo:** `src/rwkv_lab/training_speedups.py:42-84` clones and commits NumPy RNG on consumption; `src/rwkv_lab/rwkv_pretrain.py:1776-1788` performs pinned prefetch and nonblocking transfer.
- **disposition:** `adopted-existing`
- **rationale:** The repo implementation states and enforces the exact-resume cursor rule. Step-time benefit remains workload-specific and must be measured after correctness gates.
- **follow-up card:** none.

### `mng.fused-optimizer-step`

- **id:** `mng.fused-optimizer-step`
- **technique:** Fuse optimizer tensor updates while keeping scalar hyperparameter changes outside compiled model graphs. Prior matrix priority: P1.
- **model family applicability:** T/MLA, R, V/MM, M/D, C/D, and P/RLVR apply because this changes optimizer execution, not model equations; sparse, sharded, or host-offloaded groups need backend-specific qualification.
- **operation:** optimizer step
- **component slot:** optimizer
- **state / resume impact:** Optimizer moments, per-group step counters, and routing identity remain exact checkpoint state. Changing fused/foreach mechanics is a component-identity change and may be only `compatible` unless resumed-trajectory parity is demonstrated.
- **kernel / runtime requirement:** PyTorch fused/foreach optimizer support in the per-adapter runtime closure; custom optimizer kernels, if selected, must be capability-bound.
- **existing equivalent in this repo:** `src/rwkv_lab/training_runtime/optimizers.py:25-65` types fused/foreach selection; `src/rwkv_lab/rwkv_pretrain.py:539-550` selects fused CUDA AdamW; `src/rwkv_lab/spectral_muon.py:591-614` uses foreach Adam updates.
- **disposition:** `adopted-existing`
- **rationale:** Fused AdamW and foreach fallback-Adam updates exist with explicit state. No speed claim transfers across optimizer, device, or parameter-group topology.
- **follow-up card:** none.

### `mng.bf16-parameters-fp32-state`

- **id:** `mng.bf16-parameters-fp32-state`
- **technique:** Keep live model parameters and compute in BF16 while retaining FP32 optimizer masters and moments. Prior matrix priority: P1.
- **model family applicability:** T/MLA, R, V/MM, M/D, C/D, and P/RLVR apply subject to operator support; frozen teacher/reference models may use an independent precision policy.
- **operation:** precision
- **component slot:** precision/scaling and optimizer
- **state / resume impact:** FP32 masters and moments are mandatory exact checkpoint state. Restoring only BF16 live weights loses update history and downgrades resume; scaler state is absent because the registered BF16 policy disables gradient scaling.
- **kernel / runtime requirement:** BF16-capable PyTorch/device runtime; no custom kernel. The precision policy and FP32-master optimizer must both be in the adapter closure.
- **existing equivalent in this repo:** `src/rwkv_lab/training_runtime/precision.py:13-95` defines BF16 compute/FP32 reduction; `src/rwkv_lab/training_optimizers.py:12-56` builds FP32 masters and `src/rwkv_lab/training_optimizers.py:150-186` checkpoints them.
- **disposition:** `adopted-existing`
- **rationale:** The typed precision policy and exact master-state codec exist. Adoption elsewhere requires checkpoint round-trip and resumed-trajectory evidence.
- **follow-up card:** none.

### `mng.fp8-projection-matmuls`

- **id:** `mng.fp8-projection-matmuls`
- **technique:** Execute eligible projection matmuls in FP8 with fused or amortized quantization. Prior matrix priority: P1.
- **model family applicability:** T/MLA, R, V/MM, M/D, C/D, and P/RLVR can use projection-local FP8; small, irregular, zero-init, embedding, or numerically sensitive projections may be excluded by shape and quality allowlists.
- **operation:** precision
- **component slot:** precision/scaling
- **state / resume impact:** The current TorchAO route retains BF16/FP32 master weights, so model and optimizer checkpoint schemas are unchanged. Any delayed-scaling backend adds amax/scale history that must be checkpointed for `exact`; omitting it silently downgrades resume.
- **kernel / runtime requirement:** TorchAO Float8Linear and FP8 tensor cores (Hopper/Blackwell in the current path), normally with `torch.compile`; any TransformerEngine or custom-autograd alternative belongs inside the adapter runtime closure.
- **existing equivalent in this repo:** `src/rwkv_lab/rwkv_pretrain.py:554-592` converts aligned linears to TorchAO FP8 while preserving masters; `src/rwkv_lab/fused_channelmix.py:134-171` provides a custom scaled-matmul path on compute capability 9+.
- **disposition:** `adopted-existing`
- **rationale:** General projection FP8 and cached quantization paths exist. Each shape/hardware combination still needs parity, determinism, checkpoint, quality, memory, and then throughput qualification.
- **follow-up card:** none.

### `mng.fused-activation-projection`

- **id:** `mng.fused-activation-projection`
- **technique:** Fuse a projection, activation, and following projection across one training block. Prior matrix priority: P1.
- **model family applicability:** R applies directly at squared-ReLU ChannelMix sites; T/MLA and V/MM apply only to matching FFN equations; M/D often uses GELU/SwiGLU and cannot reuse this kernel by name; C/D and P/RLVR inherit the selected model topology.
- **operation:** model construction
- **component slot:** activation for the equation; no fused-block kernel slot exists yet.
- **state / resume impact:** No additional persistent state in BF16 mode. Cached FP8 copies are derived after each optimizer step and must not be treated as authoritative checkpoint state; activation semantics are model identity, and stale/authoritative caches prevent `exact` resume.
- **kernel / runtime requirement:** Custom autograd plus optional Triton and FP8 `_scaled_mm`; shape constraints and the Triton package must be in the adapter closure.
- **existing equivalent in this repo:** `src/rwkv_lab/fused_channelmix.py:1-7` states the fused training boundary, `src/rwkv_lab/fused_channelmix.py:174-235` implements custom autograd, and `src/rwkv_lab/fused_channelmix.py:257-280` begins parity/speed qualification.
- **disposition:** `adopted-existing`
- **rationale:** The exact Linear→ReLU²→Linear RWKV block is implemented, with portable fallback. It is not a generic license to substitute ReLU² for another family's activation.
- **follow-up card:** none.

### `mng.fused-loss-kernel`

- **id:** `mng.fused-loss-kernel`
- **technique:** Fuse or memory-bound the vocabulary projection and cross-entropy loss. Prior matrix priority: P1.
- **model family applicability:** T/MLA, R, C/D, and P/RLVR apply to token CE; V/MM applies to token heads but not image-only objectives; M/D does not apply to flow-matching losses.
- **operation:** loss
- **component slot:** objective
- **state / resume impact:** Stateless; reduction semantics, ignore-index behavior, precision, and backend identity are part of the objective contract. Changing them without parity evidence makes a resume merely `compatible` even though no tensor state is added.
- **kernel / runtime requirement:** Optional flash-attn Triton CE inside the adapter closure, with a chunked PyTorch fallback.
- **existing equivalent in this repo:** `src/rwkv_lab/fused_ce.py:1-6` defines the shared boundary, `src/rwkv_lab/fused_ce.py:116-153` fuses or chunks LM-head CE, and `src/rwkv_lab/training_runtime/objectives.py:47-87` exposes it as a stateless typed objective.
- **disposition:** `adopted-existing`
- **rationale:** Fused and bounded CE paths already exist and preserve masked-mean semantics. Softcapping is classified separately because it changes the objective.
- **follow-up card:** none.

### `mng.parameter-bank-transposed-layout`

- **id:** `mng.parameter-bank-transposed-layout`
- **technique:** Store repeated weights in parameter banks and/or a kernel-preferred transposed layout. Prior matrix priority: P2.
- **model family applicability:** T/MLA, R, V/MM, and M/D apply only to repeated same-shape projections with stable traversal; C/D must define conversion receipts; P/RLVR can inherit a qualified topology but cannot mutate a pretrained layout implicitly.
- **operation:** model construction
- **component slot:** none exists yet.
- **state / resume impact:** This changes model keys, strides/layout, optimizer parameter identity, and possibly sharding. Exact resume requires a versioned checkpoint-layout conversion plus optimizer-state remap; loading only numerical weights is `compatible`, not `exact`.
- **kernel / runtime requirement:** No single required backend; any banked/custom kernel and layout converter must be sealed in the adapter runtime closure.
- **existing equivalent in this repo:** No general equivalent found. `src/rwkv_lab/fused_channelmix.py:161-170` materializes a transposed FP8 operand locally, but it does not change authoritative parameter layout.
- **disposition:** `candidate-measured-gap`
- **rationale:** The layout can be useful only after checkpoint topology, sharding, and optimizer-state conversion are specified. It must pass state round-trip and equal-quality gates before memory or step-time claims.
- **follow-up card:** “Qualify versioned parameter-bank checkpoint layouts” — 04 Performance & Kernel Qualification.

### `mng.communication-optimizer-overlap`

- **id:** `mng.communication-optimizer-overlap`
- **technique:** Order reduce-scatter/all-gather and optimizer work so communication overlaps useful computation. Prior matrix priority: P2.
- **model family applicability:** T/MLA, R, V/MM, M/D, C/D, and P/RLVR apply only in distributed adapters; it is inactive and should add no machinery on one GPU.
- **operation:** distributed sync
- **component slot:** none exists yet.
- **state / resume impact:** Collective ordering adds no logical state, but sharded optimizer state and world/topology identity must round-trip. A changed ordering must preserve deterministic reductions or the run cannot claim `exact` numerical continuation.
- **kernel / runtime requirement:** Distributed PyTorch/FSDP or backend-specific collectives inside the adapter closure; streams/events and reduce-scatter ordering require accelerator qualification.
- **existing equivalent in this repo:** `src/rwkv_lab/distributed.py:64-83` installs FSDP2 and `src/rwkv_lab/distributed.py:87-110` configures explicit forward/backward all-gather prefetch. No explicit optimizer/reduce-scatter overlap controller was found.
- **disposition:** `candidate-measured-gap`
- **rationale:** Adjacent-block prefetch exists, but the full prior-row claim includes optimizer overlap and reduce-scatter ordering that repo evidence does not establish. Correctness, deterministic reduction, sharded resume, equal quality, and only then overlap benefit must be measured.
- **follow-up card:** “Qualify distributed optimizer and reduce-scatter overlap” — 04 Performance & Kernel Qualification.

## Optimizer and schedule techniques

### `mng.muon-newton-schulz`

- **id:** `mng.muon-newton-schulz`
- **technique:** Apply momentum followed by Newton–Schulz polar orthogonalization to eligible matrix updates. Prior matrix priority: P1.
- **model family applicability:** T/MLA, R scratch and pretrained, V/MM, M/D, C/D, and P/RLVR can route suitable dense 2-D projections to Muon; embeddings, heads, norms, biases, scalars, zero-init gates, sparse tables, and some expert tensors require explicit exclusion.
- **operation:** optimizer step
- **component slot:** optimizer
- **state / resume impact:** Momentum, per-parameter step state, routing identity, NS configuration, and auxiliary optimizer state are required for `exact`. Repo `SpectralMuon` restores floating optimizer state as FP32 to avoid silent BF16 requantization.
- **kernel / runtime requirement:** BF16/FP32 PyTorch matmuls; optional compiled/batched paths are separate capabilities within the adapter closure. No upstream LR may be copied across the repo's MuonClip RMS amplifier.
- **existing equivalent in this repo:** `src/rwkv_lab/spectral_muon.py:90-116` implements NS polar orthogonalization; `src/rwkv_lab/spectral_muon.py:213-285` defines the optimizer and FP32 restore path.
- **disposition:** `adopted-existing`
- **rationale:** A configurable Muon-family optimizer is already present. Qualification must compare effective update RMS and routing, not raw LR, and must pass parity, determinism, resume, and equal-quality gates before speed.
- **follow-up card:** none.

### `mng.batched-newton-schulz`

- **id:** `mng.batched-newton-schulz`
- **technique:** Bucket same-shaped matrices and execute their Newton–Schulz iterations as batched GEMMs, optionally as one static compiled graph. Prior matrix priority: P1, inherited from fused optimizer execution and Muon.
- **model family applicability:** T/MLA, R, V/MM, M/D, C/D, and P/RLVR apply when their routed Muon matrices share shapes; heterogeneous, sparse, tiled, or advanced Muon variants fall back.
- **operation:** optimizer step
- **component slot:** optimizer
- **state / resume impact:** No new mathematical state beyond Muon momentum; bucket construction must be deterministic from parameter identity. Compile caches are disposable, and changing bucket order must not change checkpoint mapping or reduction order for an `exact` claim.
- **kernel / runtime requirement:** Batched PyTorch GEMMs and optional `torch.compile(dynamic=False, fullgraph=True)` in the adapter runtime closure.
- **existing equivalent in this repo:** `src/rwkv_lab/spectral_muon.py:119-137` implements batched NS, `src/rwkv_lab/spectral_muon.py:329-338` buckets by device/dtype/shape, and `src/rwkv_lab/spectral_muon.py:377-419` compiles and applies the batched update.
- **disposition:** `adopted-existing`
- **rationale:** The modded-nanogpt-derived batched/compiled path is already implemented with explicit fallbacks. Its benefit remains shape- and hardware-dependent.
- **follow-up card:** none.

### `mng.normuon-polar-express`

- **id:** `mng.normuon-polar-express`
- **technique:** Alternative normalized/polar Muon formulations represented upstream by NorMuon and Polar Express. Prior matrix priority: P1.
- **model family applicability:** T/MLA, R, V/MM, M/D, C/D, and P/RLVR have the same matrix-routing applicability as Muon, but every formulation changes update geometry and requires a new quality campaign.
- **operation:** optimizer step
- **component slot:** optimizer
- **state / resume impact:** Formulation-specific moments, iterations, and routing must be checkpointed and bound to optimizer identity. Loading their state into `SpectralMuon` without an explicit conversion is not exact.
- **kernel / runtime requirement:** Formulation-specific matrix kernels, potentially `torch.compile`; every dependency must be in the adapter closure.
- **existing equivalent in this repo:** `src/rwkv_lab/spectral_muon.py:213-245` exposes multiple Muon transforms, but no repo symbol or implementation identified as NorMuon or Polar Express was found.
- **disposition:** `candidate-measured-gap`
- **rationale:** Nearby Muon variants are not evidence of mathematical equivalence. A clean implementation must first match a reference update and resumed trajectory, then equal quality, before kernel timing.
- **follow-up card:** “Qualify NorMuon and Polar Express update geometry” — 04 Performance & Kernel Qualification.

### `mng.adam-muon-parameter-split`

- **id:** `mng.adam-muon-parameter-split`
- **technique:** Route eligible dense matrices to Muon while embeddings, output head, norms, biases, scalar/vector gates, and unsafe matrices use Adam. Prior matrix priority: P1.
- **model family applicability:** T/MLA, R scratch/pretrained, V/MM, M/D, C/D, and P/RLVR apply, but each adapter must define semantic roles rather than infer safety solely from rank; MoE and sparse/host parameters require additional groups.
- **operation:** parameter routing
- **component slot:** parameter router
- **state / resume impact:** Exhaustive parameter ownership, group order, LRs, optimizer identities, and all group state are exact-resume identity. A changed route silently changes optimization and is at most `compatible` without a conversion receipt.
- **kernel / runtime requirement:** No additional kernel; both optimizer implementations and the router must be in the adapter closure.
- **existing equivalent in this repo:** `src/rwkv_lab/rwkv_pretrain.py:494-525` routes matrices to `SpectralMuon` and embeddings/head/norm/Engram elsewhere; `src/rwkv_lab/train_mla.py:619-628` explicitly excludes embeddings and LM heads; `src/rwkv_lab/muon_helpers.py:15-27` provides curated routing to MuonClip.
- **disposition:** `adopted-existing`
- **rationale:** The split exists in scratch RWKV and MLA/conversion paths. The typed registry does not yet advertise Muon, but that catalog gap does not justify a duplicate optimizer implementation.
- **follow-up card:** none.

### `mng.cautious-weight-decay`

- **id:** `mng.cautious-weight-decay`
- **technique:** Apply LR-coupled weight decay only where the optimizer update already moves a coordinate toward zero. Prior matrix priority: P1.
- **model family applicability:** T/MLA, R, V/MM, M/D, C/D, and P/RLVR can use it on qualified Muon groups; scale-invariant norms and excluded parameter classes still require explicit decay routing.
- **operation:** optimizer step
- **component slot:** weight-decay schedule for policy plus optimizer for coordinate masking; the current typed decay slot does not encode the mask.
- **state / resume impact:** Stateless beyond optimizer state, but decay policy and live LR are update semantics and run identity. Switching ordinary and cautious decay mid-resume is not `exact`.
- **kernel / runtime requirement:** PyTorch elementwise operations in the optimizer closure; fused implementations require their own capability.
- **existing equivalent in this repo:** `src/rwkv_lab/spectral_muon.py:486-514` implements sign-agreeing decay after radius handling; `TRAINING_LEVERS.md:84` exposes `--sm-cautious-wd`.
- **disposition:** `adopted-existing`
- **rationale:** The option exists and is off by default. It still needs workload-specific equal-quality evidence before any performance interpretation.
- **follow-up card:** none.

### `mng.muon-row-radial-conditioning`

- **id:** `mng.muon-row-radial-conditioning`
- **technique:** Apply a per-output-row update floor, outward radial brake, and optional finite-step radius pin after Muon orthogonalization. Prior matrix priority: P1, inherited from the cautious-decay/Track-3 optimizer row.
- **model family applicability:** T/MLA, R, V/MM, M/D, C/D, and P/RLVR apply only to matrix parameters for which row geometry and radial scale have the intended meaning; embeddings, zero-init gates, and sparse tables are excluded.
- **operation:** optimizer step
- **component slot:** optimizer
- **state / resume impact:** No extra tensors beyond Muon state, but all three scalars and their ordering relative to decay are optimizer identity. A changed setting prevents an `exact` trajectory claim.
- **kernel / runtime requirement:** PyTorch reductions/elementwise math in the optimizer closure; no mandatory custom kernel.
- **existing equivalent in this repo:** `src/rwkv_lab/spectral_muon.py:473-514` implements all three operations; `TRAINING_LEVERS.md:82-84` documents their flags.
- **disposition:** `adopted-existing`
- **rationale:** These modded-nanogpt-derived postconditioners already exist. They alter optimization geometry and require quality sweeps rather than being enabled as runtime-only speed flags.
- **follow-up card:** none.

### `mng.aux-adam-update-cadence`

- **id:** `mng.aux-adam-update-cadence`
- **technique:** Accumulate fallback-Adam gradients and update that parameter group every N Muon steps. Prior matrix priority: P1.
- **model family applicability:** T/MLA, R, V/MM, M/D, C/D, and P/RLVR apply when a Muon/Adam split exists; effective sample weighting and stale auxiliary parameters must be validated per objective.
- **operation:** optimizer step
- **component slot:** optimizer; no independent per-group cadence component exists yet.
- **state / resume impact:** Gradient accumulators, counts, Adam moments, and optimizer-step phase are mandatory for `exact`. Dropping the pending accumulator or reconstructing phase from the wrong cursor silently downgrades resume.
- **kernel / runtime requirement:** Foreach Adam operations in the optimizer closure; no custom kernel.
- **existing equivalent in this repo:** `src/rwkv_lab/spectral_muon.py:293-346` persists cadence in optimizer groups, and `src/rwkv_lab/spectral_muon.py:554-618` accumulates, averages, updates, and clears fallback gradients; `TRAINING_LEVERS.md:85` states that state and phase are checkpointed.
- **disposition:** `adopted-existing`
- **rationale:** The exact mechanism exists. Comparisons must hold tokens, gradient averaging, Muon steps, resume phase, and equal quality constant.
- **follow-up card:** none.

### `mng.batch-size-ramp`

- **id:** `mng.batch-size-ramp`
- **technique:** Increase effective batch size over optimizer steps. Prior matrix priority: P1.
- **model family applicability:** T/MLA, R, V/MM, M/D, C/D, and P/RLVR can use an example-, token-, or pixel-normalized ramp; preference groups and diffusion aspect buckets require domain-specific atomic units.
- **operation:** schedule
- **component slot:** curriculum
- **state / resume impact:** The optimizer-step cursor and batch/token accounting must restore exactly; changing microbatch count can also change RNG consumption and reduction order. Deriving the phase from elapsed wall time would downgrade `exact`.
- **kernel / runtime requirement:** No custom kernel; new boundary shapes may require compile prewarm within the adapter closure.
- **existing equivalent in this repo:** `src/rwkv_lab/training_runtime/curricula.py:134-173` changes batch reciprocally with context to preserve a token budget, but no independent monotonic batch-size ramp was found.
- **disposition:** `candidate-measured-gap`
- **rationale:** The existing reciprocal context curriculum is not the upstream batch ramp. A typed schedule must define its unit, accumulation interaction, cursor, equal-token comparison, and quality gate.
- **follow-up card:** “Add token-normalized batch-size ramp components” — 02 Trainer Adapters & Components.

### `mng.sequence-length-curriculum`

- **id:** `mng.sequence-length-curriculum`
- **technique:** Increase maximum sequence length in declared optimizer-step stages while holding tokens per update approximately constant. Prior matrix priority: P1.
- **model family applicability:** T/MLA and R scratch apply directly; R pretrained applies only if recurrent/reset semantics and position behavior match; V/MM may adapt the concept to token/image buckets; M/D uses image/aspect curricula instead; C/D and P/RLVR require aligned teacher/group rows.
- **operation:** schedule
- **component slot:** curriculum
- **state / resume impact:** Phase derives from the restored optimizer-step cursor and immutable horizon; no additional state is needed. A changed horizon/stage list changes run identity and cannot claim exact continuation.
- **kernel / runtime requirement:** No custom kernel beyond the adapter closure; every boundary shape must be prewarmed if compiled.
- **existing equivalent in this repo:** `src/rwkv_lab/training_runtime/curricula.py:107-131` defines the stateless typed curriculum and `src/rwkv_lab/training_runtime/curricula.py:164-173` derives reciprocal batch size; `src/rwkv_lab/rwkv_pretrain.py:1884-1895` consumes it.
- **disposition:** `adopted-existing`
- **rationale:** The typed optimizer-step curriculum exists and preserves token budget. It does not establish transfer of the upstream stage fractions or quality outcome.
- **follow-up card:** none.

### `mng.lr-momentum-trapezoid`

- **id:** `mng.lr-momentum-trapezoid`
- **technique:** Coordinate LR warmup, momentum warmup, plateau, and trapezoidal warmdown over optimizer steps. Prior matrix priority: P1, newly split from the schedule/curriculum treatment.
- **model family applicability:** T/MLA, R, V/MM, M/D, C/D, and P/RLVR can use typed step schedules; short conversion and RLVR runs need separately tuned horizons and cannot inherit speedrun fractions.
- **operation:** schedule
- **component slot:** learning-rate schedule; no momentum-schedule component exists yet.
- **state / resume impact:** LR and momentum phases must restore from the optimizer-step cursor or a checkpointed scheduler cursor. A missing momentum phase/cursor silently makes resume only `compatible`.
- **kernel / runtime requirement:** No custom kernel; scalar updates must stay outside compiled model graphs and inside the adapter closure.
- **existing equivalent in this repo:** `src/rwkv_lab/training_runtime/schedules.py:116-167` provides typed linear-warmup/cosine and PowerCool schedules; no coordinated momentum warmup or trapezoidal schedule was found.
- **disposition:** `candidate-measured-gap`
- **rationale:** Existing LR schedules are not equivalent to the complete upstream LR/momentum trapezoid. The card must first specify step domains, resume phase, effective update RMS, equal-token comparison, and equal quality.
- **follow-up card:** “Add resumable LR and momentum trapezoid schedules” — 02 Trainer Adapters & Components.

### `mng.powercool-warmdown`

- **id:** `mng.powercool-warmdown`
- **technique:** Use linear warmup, optional plateau, then a power-law cooldown with explicit exponent and floor. Prior matrix priority: P1, absorbed from existing modded-NanoGPT-provenance levers.
- **model family applicability:** T/MLA, R, V/MM, M/D, C/D, and P/RLVR can use it as a generic optimizer-step LR schedule, but exponent and cooldown fraction are regime-specific.
- **operation:** schedule
- **component slot:** learning-rate schedule
- **state / resume impact:** The restored optimizer-step/scheduler cursor and immutable horizon determine phase; scheduler state is checkpoint state where a worker uses `LambdaLR`. A reset cursor downgrades `exact`.
- **kernel / runtime requirement:** No custom kernel; scalar LR mutation stays in the adapter runtime closure.
- **existing equivalent in this repo:** `src/rwkv_lab/powercool.py:22-64` implements the pure schedule; `src/rwkv_lab/training_runtime/schedules.py:60-113` types it and `src/rwkv_lab/training_runtime/schedules.py:134-167` builds the registered scheduler.
- **disposition:** `adopted-existing`
- **rationale:** The schedule exists with explicit non-canonical hyperparameters. Its presence does not validate any upstream exponent or transfer quality.
- **follow-up card:** none.

### `mng.mup-fan-in-scaling`

- **id:** `mng.mup-fan-in-scaling`
- **technique:** Use muP-like initialization and per-parameter Adam LR scaling derived from fan-in and role. Prior matrix priority: P3.
- **model family applicability:** T/MLA, R scratch, V/MM, and M/D fresh initialization can apply family-specific parametrization; R pretrained, C/D continuation, and P/RLVR cannot retroactively change initialization and may use LR grouping only with compatibility evidence.
- **operation:** parameter routing
- **component slot:** parameter router; initialization has no typed component slot yet.
- **state / resume impact:** Initialization changes model state; LR multipliers and role manifest are routing identity. Exact resume requires the same group topology and multipliers but no extra dynamic state.
- **kernel / runtime requirement:** No custom kernel or additional adapter-runtime-closure dependency.
- **existing equivalent in this repo:** `src/rwkv_lab/u_mup.py:70-105` implements unit/fan-in initialization and an audit manifest; `src/rwkv_lab/u_mup.py:108-150` derives role-aware LR multipliers and optimizer groups.
- **disposition:** `adopted-existing`
- **rationale:** The repo has a documented practical u-muP subset. It is a fresh-initialization/scale-transfer experiment, not a runtime optimization or a permissible pretrained checkpoint rewrite.
- **follow-up card:** none.

### `mng.rms-matched-update-scaling`

- **id:** `mng.rms-matched-update-scaling`
- **technique:** Normalize or cap per-parameter effective update size relative to parameter RMS so optimizer groups use comparable update units. Prior matrix priority: P1, inherited from Muon routing.
- **model family applicability:** T/MLA, R, V/MM, M/D, C/D, and P/RLVR can use audited update scaling; zero-init tensors need a nonzero reference rule because RMS-relative caps otherwise create an absorbing state.
- **operation:** optimizer step
- **component slot:** optimizer
- **state / resume impact:** Stateless if scaling derives from current weights/update, but cap/rule and routing are optimizer identity. Any running RMS statistics would become mandatory checkpoint state; omitting them prevents `exact` resume.
- **kernel / runtime requirement:** PyTorch reductions in the optimizer closure; a fused kernel is optional and separately qualified.
- **existing equivalent in this repo:** `src/rwkv_lab/muon_helpers.py:95-106` caps Muon updates by `lr*RMS(update)/RMS(parameter)` and `src/rwkv_lab/muon_helpers.py:124-132` does the same for Adam; `src/rwkv_lab/train_mla.py:603-617` records the LR-unit reason.
- **disposition:** `adopted-existing`
- **rationale:** GuardedMuonClip implements RMS-relative caps, while u-muP implements fan-in LR groups. Raw upstream LRs remain non-transferable, and zero-init gates must be excluded or use a separate rule.
- **follow-up card:** none.

### `mng.raw-upstream-lr-transfer`

- **id:** `mng.raw-upstream-lr-transfer`
- **technique:** Reuse modded-nanogpt's numerical optimizer LRs unchanged in this repo. Prior matrix priority: none; this is the explicit non-reusable recipe element.
- **model family applicability:** T/MLA, R, V/MM, M/D, C/D, and P/RLVR are all incompatible with blind numerical transfer because parameter scales, routing, optimizer equations, token budgets, and schedules differ; MuonClip is a confirmed failure case.
- **operation:** optimizer step
- **component slot:** optimizer
- **state / resume impact:** No new state, but it changes every update and therefore cannot preserve an exact trajectory.
- **kernel / runtime requirement:** No additional adapter-runtime-closure dependency; the incompatibility is mathematical, not a missing runtime.
- **existing equivalent in this repo:** `src/rwkv_lab/train_mla.py:605-611` records that MuonClip's RMS amplifier made vanilla-Muon effective steps 25-60x too high and caused divergent runs.
- **disposition:** `rejected`
- **rationale:** MuonClip multiplies the orthogonalized update by `0.4*sqrt(max_dim)`, so the same raw LR is not the same algorithmic step. LR candidates must be calibrated by effective update RMS after routing and pass correctness, determinism, resume, and equal-quality gates.
- **follow-up card:** none.

## Model-topology techniques

### `mng.scheduled-untied-embeddings`

- **id:** `mng.scheduled-untied-embeddings`
- **technique:** Use independent input embedding and output-head weights, optionally tying them initially and splitting at a scheduled step with cloned optimizer moments. Prior matrix priority: P2.
- **model family applicability:** T/MLA and R scratch apply; R pretrained, C/D, and P/RLVR may use the checkpoint's existing topology but cannot schedule surgery unless conversion is defined; token V/MM can apply; M/D does not without a token head.
- **operation:** model construction
- **component slot:** none exists yet; the transition also depends on parameter routing.
- **state / resume impact:** Independent weights add model state. A scheduled split changes parameter identity and optimizer topology; exact resume requires the tied/split phase, split cursor, new head tensor, cloned moments, group order, and topology metadata.
- **kernel / runtime requirement:** No custom kernel; FSDP2 and FP8-head paths need topology-aware support inside their closures.
- **existing equivalent in this repo:** `src/rwkv_lab/rwkv_pretrain.py:287-297` constructs untied weights by default; `src/rwkv_lab/optimizer_speedups.py:153-198` ties and splits while cloning optimizer state; `src/rwkv_lab/rwkv_pretrain.py:1896-1908` performs the scheduled transition.
- **disposition:** `adopted-existing`
- **rationale:** Both ordinary untied embeddings and the checkpoint-sensitive scheduled split exist. Current code explicitly rejects FSDP2, FP8 head, and pretrained initialization for the transition (`src/rwkv_lab/rwkv_pretrain.py:1002-1007`).
- **follow-up card:** none.

### `mng.rotary-embeddings`

- **id:** `mng.rotary-embeddings`
- **technique:** Apply rotary position embeddings to Q/K attention channels. Prior matrix priority: P3.
- **model family applicability:** T/MLA applies directly; attention-based V/MM/M/D may use 1-D/2-D rotary; R token mixing has no Q/K and does not port by analogy, although a separately derived RAD-RWKV conversion graft rotates receptance/write-key; C/D and P/RLVR must preserve the pretrained rotary convention.
- **operation:** model construction
- **component slot:** none exists yet.
- **state / resume impact:** Usually no trainable state, but theta, rotary fraction, axis/layout, and position reset convention are model identity. A changed convention is not exact resume.
- **kernel / runtime requirement:** Rotary-compatible attention kernels inside the adapter closure; no standalone custom kernel is mandatory.
- **existing equivalent in this repo:** `src/rwkv_lab/mla_module.py:30-32` applies RoPE and `src/rwkv_lab/mla_module.py:238-248` integrates it into MLA; the non-equivalent conversion-specific RWKV graft is documented at `src/rwkv_lab/rwkv8_deltanet.py:357-370`.
- **disposition:** `adopted-existing`
- **rationale:** Standard attention RoPE already exists in MLA. It must not be projected onto native RWKV recurrence merely because a conversion-specific positional graft also exists.
- **follow-up card:** none.

### `mng.qk-norm`

- **id:** `mng.qk-norm`
- **technique:** Normalize per-head Q and K before rotary/attention. Prior matrix priority: P3.
- **model family applicability:** T/MLA applies; attention-based V/MM/M/D may apply; R token mixing lacks Q/K and does not port; C/D and P/RLVR must preserve source norm placement and weights.
- **operation:** model construction
- **component slot:** normalization
- **state / resume impact:** Affine Q/K norm weights add model state; norm placement, epsilon, and layout are checkpoint identity. Synthesizing missing norms for continuation is not exact.
- **kernel / runtime requirement:** Ordinary PyTorch RMSNorm or a parity-qualified fused attention kernel in the adapter closure.
- **existing equivalent in this repo:** `src/rwkv_lab/mla_module.py:102-107` constructs per-head Q/K RMSNorm and `src/rwkv_lab/mla_module.py:232-248` applies it before RoPE; `src/rwkv_lab/svd_init.py:299-313` preserves source norm weights during conversion.
- **disposition:** `adopted-existing`
- **rationale:** The MLA implementation matches explicit full-head QK-normalization semantics. It is attention-specific and has no native RWKV ChannelMix/TimeMix port.
- **follow-up card:** none.

### `mng.relu-squared`

- **id:** `mng.relu-squared`
- **technique:** Use squared ReLU as the feed-forward activation. Prior matrix priority: P3.
- **model family applicability:** R ChannelMix applies directly; T/MLA and V/MM apply only in fresh or matching FFN topologies; M/D commonly uses another activation and cannot substitute it as a kernel-only change; C/D/P/RLVR must retain checkpoint architecture.
- **operation:** model construction
- **component slot:** activation
- **state / resume impact:** Stateless, but activation is model semantics. Switching activation on resume is not compatible with exact continuation even if parameter shapes match.
- **kernel / runtime requirement:** Portable PyTorch or the repo's custom-autograd/Triton fused ChannelMix path inside the closure.
- **existing equivalent in this repo:** `src/rwkv_lab/training_runtime/activations.py:13-49` registers stateless squared ReLU; `src/rwkv_lab/rwkv8_deltanet.py:89-105` installs and guards it.
- **disposition:** `adopted-existing`
- **rationale:** Squared ReLU is a real typed RWKV activation. Other families require fresh equal-quality campaigns rather than silent substitution.
- **follow-up card:** none.

### `mng.partial-key-offset-paired-heads`

- **id:** `mng.partial-key-offset-paired-heads`
- **technique:** Offset selected key channels and pair attention heads with coordinated roles. Prior matrix priority: P3.
- **model family applicability:** T/MLA scratch is the direct target; attention-based V/MM/M/D may adapt it with topology-specific semantics; R has no attention keys/heads; C/D and P/RLVR cannot retrofit it without a checkpoint conversion and fresh quality evidence.
- **operation:** model construction
- **component slot:** none exists yet.
- **state / resume impact:** Adds or reinterprets model parameters/head topology and therefore changes checkpoint identity, cache layout, and possibly optimizer routing. Exact continuation from an ordinary attention checkpoint is unavailable without an explicit lossless mapping.
- **kernel / runtime requirement:** Attention kernels must support the exact head/key layout; any custom kernel belongs in the adapter closure.
- **existing equivalent in this repo:** No partial-key-offset or paired-head implementation was found. `src/rwkv_lab/mla_module.py:70-73` supports multiple rotary KV heads, but that is not the same head pairing rule.
- **disposition:** `candidate-measured-gap`
- **rationale:** This is a fresh architecture experiment, not an attention backend optimization. It needs reference forward/gradient tests, deterministic construction, checkpoint round-trip, equal quality, and only then kernel qualification.
- **follow-up card:** “Ablate partial-key offset and paired-head topology” — 02 Trainer Adapters & Components.

### `mng.zero-init-projections`

- **id:** `mng.zero-init-projections`
- **technique:** Zero-initialize selected residual/output projections so a new branch is an exact or near no-op at construction. Prior matrix priority: P3.
- **model family applicability:** T/MLA, R scratch and adapters, V/MM, and M/D fresh branches apply; C/D can use it for newly added residual adapters; P/RLVR can add adapters but must not zero existing pretrained projections.
- **operation:** model construction
- **component slot:** none exists yet.
- **state / resume impact:** No state beyond model weights, but initialization is irreversible run identity. It cannot be retrofitted into a continuation run or changed under `exact` resume; zero-init tensors must avoid RMS-relative optimizer rules that trap them at zero.
- **kernel / runtime requirement:** No custom kernel or additional adapter-runtime-closure dependency.
- **existing equivalent in this repo:** `src/rwkv_lab/rwkv8_deltanet.py:312-314` documents zero-init output semantics and `src/rwkv_lab/rwkv8_deltanet.py:471-474` applies it; `src/rwkv_lab/looped_rwkv.py:188-199` zero-initializes residual gates/offsets.
- **disposition:** `adopted-existing`
- **rationale:** Init-preserving zero projections are widely used in repo-native adapters. They remain fresh-construction choices and must be routed safely under MuonClip.
- **follow-up card:** none.

### `mng.embedding-to-block-skip`

- **id:** `mng.embedding-to-block-skip`
- **technique:** Inject token embeddings into later transformer blocks through explicit skip paths. Prior matrix priority: P3.
- **model family applicability:** T/MLA scratch applies; R has family-specific DeepEmbed/embedding residuals but not the upstream transformer path; token V/MM may apply; M/D does not use token embeddings by default; C/D/P/RLVR require checkpoint/topology conversion.
- **operation:** model construction
- **component slot:** none exists yet.
- **state / resume impact:** Trainable projections/gates add model and optimizer state; injection sites and source embedding identity are checkpoint topology. Exact resume requires all of them.
- **kernel / runtime requirement:** Ordinary PyTorch; fused implementations must be adapter-local and closure-bound.
- **existing equivalent in this repo:** `src/rwkv_lab/rwkv_pretrain.py:335-339` supplies the raw embedding for RWKV DeepEmbed folding, but no equivalent modded-nanogpt transformer embedding-to-block skip was found.
- **disposition:** `candidate-measured-gap`
- **rationale:** The RWKV mechanism is an analogy, not a reusable transformer implementation. A fresh topology campaign must establish parity of the disabled state, resume, and equal quality.
- **follow-up card:** “Ablate transformer embedding and block skip topologies” — 02 Trainer Adapters & Components.

### `mng.block-to-block-skip`

- **id:** `mng.block-to-block-skip`
- **technique:** Feed selected earlier block outputs directly into later blocks. Prior matrix priority: P3.
- **model family applicability:** T/MLA scratch and some V/MM/M/D U-shaped stacks apply; R recurrence already has distinct residual/state paths and cannot reuse transformer indices by analogy; C/D/P/RLVR require topology-aware conversion.
- **operation:** model construction
- **component slot:** none exists yet.
- **state / resume impact:** Learned projections/gates add state; even parameter-free adds change graph semantics and activation memory. Source/target block mapping is checkpoint/run identity and must match for `exact` resume.
- **kernel / runtime requirement:** No mandatory custom kernel; activation retention may materially increase memory and must be qualified inside the closure.
- **existing equivalent in this repo:** No general transformer block-to-block skip was found. `src/rwkv_lab/mage_flow_terminal_experts.py:1534-1555` maps residual-expert checkpoints into terminal blocks, but that is initialization transfer, not an activation skip.
- **disposition:** `candidate-measured-gap`
- **rationale:** The topology needs an exact disabled-state reference, activation-memory accounting, checkpoint schema, and a fresh equal-quality campaign.
- **follow-up card:** “Ablate transformer embedding and block skip topologies” — 02 Trainer Adapters & Components.

### `mng.value-embedding-lambda-residual`

- **id:** `mng.value-embedding-lambda-residual`
- **technique:** Reuse a value embedding across layers and combine it through learned lambda residual connections. Prior matrix priority: P3.
- **model family applicability:** T/MLA scratch applies; R has a native cross-layer `v_first` recurrence-specific value residual but cannot reuse the transformer equation; attention V/MM/M/D may adapt it; C/D/P/RLVR require source-compatible topology.
- **operation:** model construction
- **component slot:** none exists yet.
- **state / resume impact:** Lambda/gate parameters add model and optimizer state; the shared value source and layer mapping are topology identity. Exact resume requires those tensors and routing.
- **kernel / runtime requirement:** Ordinary PyTorch or a topology-specific fused kernel in the adapter closure.
- **existing equivalent in this repo:** `src/rwkv_lab/rwkv8_deltanet.py:741-754` implements RWKV's native `v_first` cross-layer residual, explicitly a different recurrent equation; no modded-nanogpt lambda residual was found.
- **disposition:** `candidate-measured-gap`
- **rationale:** The repo analog proves the general idea is useful but not mathematical portability. A transformer implementation needs independent forward/gradient and quality evidence.
- **follow-up card:** “Ablate value-embedding and lambda residuals” — 02 Trainer Adapters & Components.

### `mng.mudd-unet-cross-block-skip`

- **id:** `mng.mudd-unet-cross-block-skip`
- **technique:** Add U-Net/MUDD-style symmetric cross-block skip connections. Prior matrix priority: P3.
- **model family applicability:** T/MLA scratch and U-shaped V/MM/M/D architectures apply; R sequential recurrence does not map symmetric attention-block pairs directly; C/D and P/RLVR require explicit topology conversion.
- **operation:** model construction
- **component slot:** none exists yet.
- **state / resume impact:** Skip projections/norms add model/optimizer state and retain earlier activations; block pairing is checkpoint identity. Pretrained continuation is not exact without a no-op-preserving insertion and declared conversion.
- **kernel / runtime requirement:** No mandatory custom kernel; activation-memory/lifetime and compile behavior belong in the adapter closure.
- **existing equivalent in this repo:** No MUDD or U-Net-style language-model cross-block skip implementation was found; `src/rwkv_lab/looped_rwkv.py:39-55` instead defines recurrent hyper-connection lanes within one wrapped block.
- **disposition:** `candidate-measured-gap`
- **rationale:** This is a topology/activation-memory experiment, not a runtime optimization. It needs fresh quality and memory evidence before timing.
- **follow-up card:** “Ablate MUDD/U-Net cross-block skips” — 02 Trainer Adapters & Components.

### `mng.gated-skip-connections`

- **id:** `mng.gated-skip-connections`
- **technique:** Modulate added skip paths with learned, preferably no-op-initialized gates. Prior matrix priority: P3.
- **model family applicability:** T/MLA, R, V/MM, and M/D can use topology-specific gates; C/D and P/RLVR may add no-op adapters but cannot reinterpret existing residuals silently.
- **operation:** model construction
- **component slot:** none exists yet.
- **state / resume impact:** Gate tensors and optimizer moments are exact state; gate granularity, cap, source/target mapping, and FP32 growth policy are topology/routing identity.
- **kernel / runtime requirement:** Ordinary PyTorch inside the adapter runtime closure; tiny zero-init gates may need FP32 even with BF16 model compute.
- **existing equivalent in this repo:** `src/rwkv_lab/looped_rwkv.py:14-24` defines multiple gated residual granularities, `src/rwkv_lab/looped_rwkv.py:188-199` initializes them, and `src/rwkv_lab/looped_rwkv.py:353-367` applies optional tanh capping.
- **disposition:** `adopted-existing`
- **rationale:** A family-specific gated-skip mechanism already exists for looped RWKV. Transformer/MUDD adoption still needs topology-specific work and must not clone the RWKV gate wholesale.
- **follow-up card:** none.

### `mng.shared-attention-activation-input`

- **id:** `mng.shared-attention-activation-input`
- **technique:** Feed a shared earlier activation into multiple later attention layers. Prior matrix priority: P3.
- **model family applicability:** T/MLA scratch is the direct target; attention V/MM/M/D may apply; R token mixing has no later attention layers; C/D/P/RLVR require topology and checkpoint conversion.
- **operation:** model construction
- **component slot:** none exists yet.
- **state / resume impact:** Parameter-free sharing still changes forward semantics and activation lifetime; learned transforms add model/optimizer state. Source layer and fan-out are run identity and must match for `exact` resume.
- **kernel / runtime requirement:** No mandatory custom kernel; activation retention and compile graph must fit the adapter closure.
- **existing equivalent in this repo:** No matching transformer activation fan-out was found. `src/rwkv_lab/megakernel_linear.py:293` supports a shared one-token input within a kernel boundary, not across attention layers.
- **disposition:** `candidate-measured-gap`
- **rationale:** This must be evaluated as a fresh topology with activation-memory and equal-quality gates, not as a launch-count optimization.
- **follow-up card:** “Ablate shared-input and residual-decay transformer topology” — 02 Trainer Adapters & Components.

### `mng.exponential-residual-decay`

- **id:** `mng.exponential-residual-decay`
- **technique:** Apply depth-dependent exponential decay to residual contributions. Prior matrix priority: P3.
- **model family applicability:** T/MLA scratch and some V/MM/M/D stacks apply; R has separate learned recurrent decay equations and cannot import transformer residual decay; C/D/P/RLVR require checkpoint-compatible scaling.
- **operation:** model construction
- **component slot:** none exists yet.
- **state / resume impact:** Fixed decay adds no tensors but changes model semantics; learned decay adds model/optimizer state. Depth convention and coefficients are run identity and must match for `exact` resume.
- **kernel / runtime requirement:** No custom kernel or additional adapter-runtime-closure dependency.
- **existing equivalent in this repo:** No matching transformer residual-decay implementation was found; RWKV's recurrent decay at `src/rwkv_lab/rwkv8_deltanet.py:733-737` governs state retention and is not equivalent.
- **disposition:** `candidate-measured-gap`
- **rationale:** The apparently similar RWKV decay acts on recurrent state, not residual-stream depth. A fresh transformer quality campaign is required.
- **follow-up card:** “Ablate shared-input and residual-decay transformer topology” — 02 Trainer Adapters & Components.

### `mng.bigram-hash-embeddings`

- **id:** `mng.bigram-hash-embeddings`
- **technique:** Add hash-indexed bigram embeddings to the token representation. Prior matrix priority: P3.
- **model family applicability:** T/MLA and R language models apply; token V/MM may apply to text streams; M/D only through text conditioning; C/D needs tokenizer/hash agreement; P/RLVR must preserve conversation token boundaries.
- **operation:** model construction
- **component slot:** none exists yet.
- **state / resume impact:** Embedding tables add model/optimizer state; hash function, vocabulary, collision policy, boundary handling, and injection sites are checkpoint identity and mandatory for `exact` resume.
- **kernel / runtime requirement:** Embedding lookup, optional sparse gradients/communication, and deterministic hashing inside the adapter closure.
- **existing equivalent in this repo:** `src/rwkv_lab/train_mla.py:222-231` integrates a broader hashed N-gram Engram, but no isolated modded-nanogpt bigram embedding implementation was found.
- **disposition:** `candidate-measured-gap`
- **rationale:** Engram is not an equivalent small bigram embedding and carries different memory/gating semantics. An isolated implementation needs collision, boundary, memory, and scale-quality ablations.
- **follow-up card:** “Ablate bigram hash embeddings independently of Engram” — 02 Trainer Adapters & Components.

### `mng.smear-one-token-lookback`

- **id:** `mng.smear-one-token-lookback`
- **technique:** Mix each token with a learned or fixed one-token lookback (“smear”). Prior matrix priority: P3.
- **model family applicability:** T/MLA scratch applies; R already has token shift but the upstream mix is not the recurrent equation; token V/MM may apply; M/D does not by name; C/D/P/RLVR require checkpoint/topology compatibility and boundary resets.
- **operation:** model construction
- **component slot:** none exists yet.
- **state / resume impact:** Learned mixing adds model/optimizer state; even fixed mixing changes model semantics. Packed-document and recurrent chunk boundaries must restore the previous-token/reset convention exactly.
- **kernel / runtime requirement:** Ordinary tensor shift/mix or a fused topology kernel inside the closure.
- **existing equivalent in this repo:** `src/rwkv_lab/rwkv8_deltanet.py:689-712` implements boundary-aware RWKV token shift, but no transformer smear implementation was found.
- **disposition:** `candidate-measured-gap`
- **rationale:** RWKV token shift is family-native and not evidence for the transformer smear equation. The candidate needs boundary and chunk parity plus fresh quality tests.
- **follow-up card:** “Ablate transformer smear and sparse attention gates” — 02 Trainer Adapters & Components.

### `mng.sparse-attention-gates`

- **id:** `mng.sparse-attention-gates`
- **technique:** Learn sparse gates that select or suppress attention interactions. Prior matrix priority: P3.
- **model family applicability:** T/MLA and attention V/MM/M/D apply; R token mixing does not; C/D can learn gates from teacher evidence; P/RLVR inherits attention and must preserve causal/conversation masks.
- **operation:** model construction
- **component slot:** none exists yet.
- **state / resume impact:** Gate parameters/thresholds and optimizer state are exact state; selection/tie-breaking and sparsity schedule must be deterministic and cursor-bound.
- **kernel / runtime requirement:** Sparse-mask-capable attention kernel, potentially FlexAttention/Triton, in the adapter closure; dense fallback is needed for parity.
- **existing equivalent in this repo:** No matching sparse attention-gate implementation was found; `src/rwkv_lab/mla_module.py:285-291` uses dense SDPA with a supplied mask.
- **disposition:** `candidate-measured-gap`
- **rationale:** Quality/sparsity behavior must be isolated before a sparse kernel can claim speed. Mask correctness and deterministic selection precede throughput.
- **follow-up card:** “Ablate transformer smear and sparse attention gates” — 02 Trainer Adapters & Components.

### `mng.learnable-xsa`

- **id:** `mng.learnable-xsa`
- **technique:** Apply Exclusive Self Attention by removing the attention-output component parallel to each value vector, with learnable integration upstream. Prior matrix priority: P3.
- **model family applicability:** T/MLA and attention V/MM/M/D apply; R has no softmax attention output/value-head geometry; C/D and P/RLVR require topology-compatible continuation.
- **operation:** model construction
- **component slot:** none exists yet.
- **state / resume impact:** The current projection is stateless, while a learnable variant would add model/optimizer state. XSA enablement and placement are model identity and must match for `exact` resume.
- **kernel / runtime requirement:** PyTorch normalization/projection or a fused attention epilogue in the adapter closure.
- **existing equivalent in this repo:** `src/rwkv_lab/mla_module.py:294-298` implements XSA projection and `src/rwkv_lab/train_mla.py:251-254` exposes it for MLA.
- **disposition:** `adopted-existing`
- **rationale:** An XSA equivalent exists, although repo evidence does not show that its current form includes every learnable upstream detail. It remains an architecture/quality experiment, not a speed default.
- **follow-up card:** none.

### `mng.dynamic-composable-mha`

- **id:** `mng.dynamic-composable-mha`
- **technique:** Compose lightweight attention-head modules dynamically. Prior matrix priority: P3.
- **model family applicability:** T/MLA and attention V/MM/M/D scratch models apply; R does not; C/D and P/RLVR need a fixed exported topology or explicit dynamic state contract.
- **operation:** model construction
- **component slot:** none exists yet.
- **state / resume impact:** Modules/router parameters and optimizer state must be checkpointed; dynamic choices, RNG/tie-breaking, and routing schedule are exact state. Undeclared runtime composition prevents exact resume and stable compile graphs.
- **kernel / runtime requirement:** Dynamic routing plus attention kernels; bounded routes/shape buckets and all code must be sealed in the adapter closure.
- **existing equivalent in this repo:** No lightweight dynamically composable MHA implementation was found; `src/rwkv_lab/mla_module.py:80-107` constructs a fixed set of attention projections and norms.
- **disposition:** `candidate-measured-gap`
- **rationale:** The candidate needs a closed topology/routing contract, deterministic replay, checkpoint codec, scale-quality tests, and only then kernel qualification.
- **follow-up card:** “Specify deterministic dynamically composable MHA” — 02 Trainer Adapters & Components.

### `mng.distributed-allgather-muon`

- **id:** `mng.distributed-allgather-muon`
- **technique:** Bucket parameters across ranks, all-gather enough matrix/update data for Muon orthogonalization, and synchronize the resulting updates efficiently. Prior matrix priority: P2, inherited from Muon and communication overlap.
- **model family applicability:** T/MLA, R, V/MM, M/D, C/D, and P/RLVR apply only to distributed dense Muon groups; one-GPU runs must bypass it, and sparse/host-offloaded groups remain on their auxiliary optimizer.
- **operation:** distributed sync
- **component slot:** optimizer; no distributed-sync component exists yet.
- **state / resume impact:** Muon momentum and auxiliary state must shard/gather with stable parameter IDs; bucket order, padding, world size, and optimizer-step phase are exact identity. A world-size change is at best compatible unless a state reshard receipt proves equivalence.
- **kernel / runtime requirement:** Distributed all-gather/reduce-scatter, deterministic parameter bucketing, batched NS, and overlap streams in the adapter runtime closure.
- **existing equivalent in this repo:** `src/rwkv_lab/spectral_muon.py:329-419` has single-process same-shape bucketing; `src/rwkv_lab/distributed.py:145-168` has all-gather for sparse rows, but no distributed Muon all-gather implementation was found.
- **disposition:** `candidate-measured-gap`
- **rationale:** Neither local Muon bucketing nor sparse-row communication establishes the required distributed optimizer algorithm. Reference update parity, deterministic collectives, reshard/resume, equal quality, memory, and then overlap/throughput are required.
- **follow-up card:** “Implement and qualify distributed bucketed Muon” — 04 Performance & Kernel Qualification.

## Disposition summary

| id | disposition |
|---|---|
| `mng.compile-disposable-prewarm` | `candidate-measured-gap` |
| `mng.compile-static-fullgraph` | `adopted-existing` |
| `mng.async-pinned-batch-fetch` | `adopted-existing` |
| `mng.fused-optimizer-step` | `adopted-existing` |
| `mng.bf16-parameters-fp32-state` | `adopted-existing` |
| `mng.fp8-projection-matmuls` | `adopted-existing` |
| `mng.fused-activation-projection` | `adopted-existing` |
| `mng.fused-loss-kernel` | `adopted-existing` |
| `mng.parameter-bank-transposed-layout` | `candidate-measured-gap` |
| `mng.communication-optimizer-overlap` | `candidate-measured-gap` |
| `mng.muon-newton-schulz` | `adopted-existing` |
| `mng.batched-newton-schulz` | `adopted-existing` |
| `mng.normuon-polar-express` | `candidate-measured-gap` |
| `mng.adam-muon-parameter-split` | `adopted-existing` |
| `mng.cautious-weight-decay` | `adopted-existing` |
| `mng.muon-row-radial-conditioning` | `adopted-existing` |
| `mng.aux-adam-update-cadence` | `adopted-existing` |
| `mng.batch-size-ramp` | `candidate-measured-gap` |
| `mng.sequence-length-curriculum` | `adopted-existing` |
| `mng.lr-momentum-trapezoid` | `candidate-measured-gap` |
| `mng.powercool-warmdown` | `adopted-existing` |
| `mng.mup-fan-in-scaling` | `adopted-existing` |
| `mng.rms-matched-update-scaling` | `adopted-existing` |
| `mng.raw-upstream-lr-transfer` | `rejected` |
| `mng.document-aligned-token-packing` | `adopted-existing` |
| `mng.document-boundary-masking` | `candidate-measured-gap` |
| `mng.fp8-lm-head` | `adopted-existing` |
| `mng.fp8-mlp-up-projection` | `adopted-existing` |
| `mng.logit-softcap-cross-entropy` | `candidate-measured-gap` |
| `mng.windowed-block-causal-attention` | `candidate-measured-gap` |
| `mng.flexattention` | `candidate-measured-gap` |
| `mng.flash-attention-3` | `candidate-measured-gap` |
| `mng.yarn-rotary-scaling` | `candidate-measured-gap` |
| `mng.multi-token-prediction` | `adopted-existing` |
| `mng.selected-gradient-cadence` | `adopted-existing` |
| `mng.tail-ema-readout` | `adopted-existing` |
| `mng.lmtp-tail-cooldown` | `adopted-existing` |
| `mng.sparse-embedding-communication` | `adopted-existing` |
| `mng.backward-prefetch-overlap` | `adopted-existing` |
| `mng.scheduled-untied-embeddings` | `adopted-existing` |
| `mng.rotary-embeddings` | `adopted-existing` |
| `mng.qk-norm` | `adopted-existing` |
| `mng.relu-squared` | `adopted-existing` |
| `mng.partial-key-offset-paired-heads` | `candidate-measured-gap` |
| `mng.zero-init-projections` | `adopted-existing` |
| `mng.embedding-to-block-skip` | `candidate-measured-gap` |
| `mng.block-to-block-skip` | `candidate-measured-gap` |
| `mng.value-embedding-lambda-residual` | `candidate-measured-gap` |
| `mng.mudd-unet-cross-block-skip` | `candidate-measured-gap` |
| `mng.gated-skip-connections` | `adopted-existing` |
| `mng.shared-attention-activation-input` | `candidate-measured-gap` |
| `mng.exponential-residual-decay` | `candidate-measured-gap` |
| `mng.bigram-hash-embeddings` | `candidate-measured-gap` |
| `mng.smear-one-token-lookback` | `candidate-measured-gap` |
| `mng.sparse-attention-gates` | `candidate-measured-gap` |
| `mng.learnable-xsa` | `adopted-existing` |
| `mng.dynamic-composable-mha` | `candidate-measured-gap` |
| `mng.distributed-allgather-muon` | `candidate-measured-gap` |

Totals: 58 techniques; 33 `adopted-existing`, 24 `candidate-measured-gap`, and 1
`rejected`.

## Rejected with reason

- `mng.raw-upstream-lr-transfer`: rejected because MuonClip/GuardedMuonClip multiplies the
  orthogonalized update by `0.4*sqrt(max_dim)`. Repo evidence records 25–60x excessive effective
  steps and divergent runs when a vanilla-Muon LR was reused. The rejection follows that
  mathematical unit mismatch.

Family-local incompatibilities do not reject otherwise reusable techniques. In particular, QK
norm, standard RoPE, windowed attention, FlexAttention, FA3, XSA, and attention gates do not port to
RWKV token mixing; language-token objectives do not port to flow matching; and fresh-initialization
or topology-surgery techniques do not port to pretrained continuation without a conversion receipt.
Those restrictions are recorded in each applicability field.

## Proposed follow-up cards

No cards are created by this inventory. The 24 measured gaps map to these proposed cards:

| Target board | Proposed card | Covered ids |
|---|---|---|
| 02 Trainer Adapters & Components | Add token-normalized batch-size ramp components | `mng.batch-size-ramp` |
| 02 Trainer Adapters & Components | Add resumable LR and momentum trapezoid schedules | `mng.lr-momentum-trapezoid` |
| 02 Trainer Adapters & Components | Add deterministic transformer document masks and packing cursors | `mng.document-boundary-masking` |
| 02 Trainer Adapters & Components | Add a typed softcapped LM objective and qualified kernel | `mng.logit-softcap-cross-entropy` |
| 02 Trainer Adapters & Components | Qualify YaRN context scaling for transformer adapters | `mng.yarn-rotary-scaling` |
| 02 Trainer Adapters & Components | Ablate partial-key offset and paired-head topology | `mng.partial-key-offset-paired-heads` |
| 02 Trainer Adapters & Components | Ablate transformer embedding and block skip topologies | `mng.embedding-to-block-skip`, `mng.block-to-block-skip` |
| 02 Trainer Adapters & Components | Ablate value-embedding and lambda residuals | `mng.value-embedding-lambda-residual` |
| 02 Trainer Adapters & Components | Ablate MUDD/U-Net cross-block skips | `mng.mudd-unet-cross-block-skip` |
| 02 Trainer Adapters & Components | Ablate shared-input and residual-decay transformer topology | `mng.shared-attention-activation-input`, `mng.exponential-residual-decay` |
| 02 Trainer Adapters & Components | Ablate bigram hash embeddings independently of Engram | `mng.bigram-hash-embeddings` |
| 02 Trainer Adapters & Components | Ablate transformer smear and sparse attention gates | `mng.smear-one-token-lookback`, `mng.sparse-attention-gates` |
| 02 Trainer Adapters & Components | Specify deterministic dynamically composable MHA | `mng.dynamic-composable-mha` |
| 04 Performance & Kernel Qualification | Isolate compile prewarm in disposable workers | `mng.compile-disposable-prewarm` |
| 04 Performance & Kernel Qualification | Qualify versioned parameter-bank checkpoint layouts | `mng.parameter-bank-transposed-layout` |
| 04 Performance & Kernel Qualification | Qualify distributed optimizer and reduce-scatter overlap | `mng.communication-optimizer-overlap` |
| 04 Performance & Kernel Qualification | Qualify NorMuon and Polar Express update geometry | `mng.normuon-polar-express` |
| 04 Performance & Kernel Qualification | Qualify transformer window and block-causal curricula | `mng.windowed-block-causal-attention` |
| 04 Performance & Kernel Qualification | Qualify FlexAttention mask backends per adapter | `mng.flexattention` |
| 04 Performance & Kernel Qualification | Qualify Flash Attention 3 against SDPA | `mng.flash-attention-3` |
| 04 Performance & Kernel Qualification | Implement and qualify distributed bucketed Muon | `mng.distributed-allgather-muon` |

No measured gap currently requires a standalone card on 01 Runtime & Host Authority, 03 Dashboard &
Observability, or 05 Legacy Parity/QA/Cutover. Runtime-closure binding is a gate on the listed
qualification cards, observability is evidence produced by those campaigns, and cutover is not
authorized by an inventory.

## Revisions to the prior matrix

The source priorities are unchanged. The following revisions make previously grouped or ambiguous
claims explicit:

- The P0 compile-prewarm row remains a measured gap because current RWKV prewarm is RNG-neutral and
  multi-shape but is not run in a disposable worker. Static full-graph compilation itself is
  `adopted-existing`.
- The P2 communication row is split. Explicit FSDP forward/backward prefetch is
  `adopted-existing`; optimizer overlap and reduce-scatter ordering remain a measured gap.
- The P1 fused-loss row is split. Generic fused/bounded cross entropy is `adopted-existing`, while
  algorithmic logit softcapping is a measured gap.
- The P1 document-packing row is split. Whole-document packing, within-document sampling, padding
  masks, token-weighted bucket selection, and exact-resume-safe prefetch are `adopted-existing`.
  Transformer block-diagonal document attention masking remains a measured gap; RWKV reset masking
  is only a family-specific equivalent.
- The P1 Muon row is split. Newton–Schulz Muon, same-shape batching, the Adam/Muon parameter split,
  auxiliary Adam cadence, cautious decay, and row/radial conditioning are `adopted-existing`.
  NorMuon/Polar Express and distributed all-gather Muon are not evidenced by the current repo and
  remain measured gaps.
- The P2 attention row is split into FA3, window/block-causal scheduling, FlexAttention, and YaRN so
  backend work cannot obscure algorithmic mask or rotary changes.
- The P3 architecture rows are split by checkpoint topology and family applicability. Existing MLA
  RoPE/QK norm/XSA, RWKV ReLU2/zero-init/gated residual/value-residual analogs, and u-muP are cited;
  analogs are not treated as mathematical implementations of different transformer equations.
- Existing modded-nanogpt-provenance levers that were only named in README/training-lever prose are
  now explicit inventory records: Tail EMA, LMTP tail cooldown, sparse embedding communication,
  batched Newton–Schulz, row/radial conditioning, PowerCool, and scheduled embedding/head splitting.
- Blind transfer of raw upstream Muon LR values is explicitly rejected because repo MuonClip units
  differ by the confirmed corrected-RMS amplifier. The prior matrix did not state this incompatibility.

## Unverified claims

- Network access and GPU work were intentionally not used. The pinned upstream tree was not
  re-fetched during this classification; upstream-specific names and grouping rely on the prior
  reviewed matrix, the repository's provenance comments, and the card's required technique list.
- The exact pinned-commit equations and state choices for NorMuon, Polar Express, partial key
  offset, paired heads, MUDD, lambda residuals, sparse attention gates, dynamic MHA, window warmup,
  block-causal masks, and distributed Muon could not be confirmed from a local upstream checkout.
  They therefore remain measured gaps rather than asserted equivalents.
- The exact relationship between this pinned modded-nanogpt revision and the repo's PowerCool
  implementation is not locally verifiable. `src/rwkv_lab/powercool.py:3-6` attributes the public
  schedule name to an OpenAI NanoGPT speedrun report and explicitly says its private exponent and
  hyperparameters are unknown.
- Current installed-version and hardware support for FlexAttention, FA3, TorchAO FP8 recipes,
  Triton tensor descriptors, `_scaled_mm`, and TransformerEngine was not exercised. Runtime and SM
  requirements above are compatibility constraints, not qualification results.
- No benchmark, training launch, or quality campaign was run. All candidate speed, memory, and
  equal-quality outcomes are unmeasured; adopted implementations are source-evidenced, not newly
  performance-qualified by this document.
- The repo XSA path confirms the projection operation, but whether it matches every learnable detail
  of the pinned upstream XSA variant is unverified.
