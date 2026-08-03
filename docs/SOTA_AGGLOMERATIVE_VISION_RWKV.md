# SOTA Agglomerative Vision-RWKV

**Status:** superseded as the deployment plan, retained as the native-RWKV
research contingency, 2026-07-16  
**Working name:** AV-RWKV-g  
**Supersedes:** the fixed `128 x 1024 -> 64 x 2560` compressor/student design
in `MULTI_TEACHER_VISION_DISTILLATION.md`.

The deployment plan changed after inspecting NVIDIA's RADIO1D-H checkpoint.
It already distills SigLIP2-g, DINOv3-7B, and SAM3, emits an ordered variable
budget of global tokens, and its 2,560-wide output exactly matches the frozen
RWKV-7 2.9B text model. The current authoritative integration contract is
[RADIO1D_RWKV_ARCHITECTURE.md](RADIO1D_RWKV_ARCHITECTURE.md). Nothing in this
document authorizes reviving the discarded 54M compressor or 64/128-token
bottleneck.

## Mission

Distill MoonViT, SigLIP2, DINOv2, and SAM into one deployment-time vision
foundation model. The student must retain the complementary information of the
whole teacher stack while representing information shared by several teachers
only once. It must support global semantic, vision-language, fine spatial,
correspondence, segmentation, OCR, and high-resolution use cases.

This is not primarily a caption-adapter project. Captioning is one downstream
test. The old frozen caption RWKV's 64-token input is not an architectural
constraint on the vision foundation model.

"SOTA" is an empirical target, not a label we assign in advance. A model is not
called SOTA until it passes the benchmark gates in this document against its
individual teachers and current agglomerative models such as C-RADIOv4.

## Non-negotiable invariants

1. **No destructive teacher pooling before distillation.** Every dense teacher
   loss uses the teacher's native token grid and geometry.
2. **No fixed 64- or 128-token foundation bottleneck.** The foundation output
   is a variable-resolution dense pyramid. A separate, reversible output-time
   token packer serves VLMs at configurable budgets.
3. **One deployable student.** Teacher towers and teacher-specific loss heads
   are training-only. The deployed backbone does not need MoonViT, SigLIP2,
   DINOv2, SAM, or an intermediate Transformer compressor.
4. **Shared information is represented once.** Teachers supervise one common
   spatial backbone. We do not concatenate four independent token sequences at
   deployment.
5. **Teacher-specific information remains recoverable.** Frozen evaluation
   probes must reconstruct every teacher's native dense and summary features
   from the shared student representation.
6. **Resolution robustness is trained, not assumed.** All teachers supervise
   all student resolution regimes, and scale equivariance is measured.
7. **SAM is never bilinearly resized as a target.** Student features are
   decoded at SAM's grid coordinates; SAM crops/mosaics preserve its native
   feature statistics.
8. **No expensive run begins without a small overfit gate, cache-integrity
   gate, and exact-resume test.**

## Why the discarded design was insufficient

The discarded cache reduced SigLIP2, DINOv2, and SAM to 128 aligned positions
and reduced three MoonViT stages to 128 positions each. A 54M compressor then
produced 128 canonical tokens, and the student reduced those to 64 caption
tokens. SAM alone natively produces a 64 x 64 grid (4,096 tokens). That design
could optimize a caption interface, but it could not plausibly replace the
teacher stack for dense vision.

The corrected design follows the successful agglomerative-model pattern in
[AM-RADIO](https://arxiv.org/abs/2312.06709),
[RADIOv2.5](https://arxiv.org/abs/2412.07679), and
[PHI-S](https://arxiv.org/abs/2410.01680), while testing whether a native 2-D
RWKV backbone can improve the quality/compute frontier. RADIOv2.5 shows that
multi-resolution training, PHI-S teacher balancing, SAM-preserving mosaics,
and output-time feature-based token merging are material improvements. The
current [C-RADIO repository](https://github.com/NVlabs/RADIO) is a required
baseline and implementation reference, not a source of unverified claims.

## Native teacher contracts

All tensors are stored in BF16 or losslessly recoverable FP16 with original
grid metadata. A cache entry includes the exact image bytes fingerprint,
teacher weights fingerprint, preprocessing fingerprint, source dimensions,
resize/pad/crop transform, token coordinates, validity mask, and dtype.

| Teacher | Native student target | Summary target | Notes |
|---|---:|---:|---|
| MoonViT | taps 8/17/26 as `[L, Hm, Wm, 4, 1152]` | pooled/projected Moon summary | Preserve the four 2x2 subtoken vectors; do not flatten them into one lossy average. Full-image and optional quadrant views retain coordinate transforms. |
| SigLIP2 So400m | `[32, 32, 1152]` at its native 512 input | native pooled image embedding | Preserve text-aligned semantics and use its frozen text tower for zero-shot evaluation. |
| DINOv2 Base | `[37, 37, 768]` at native 518 input | class token `[768]` | Keep class and spatial tokens separate. |
| SAM ViT-B | `[64, 64, 256]` at native 1024 input | spatially pooled diagnostic only | Preserve the complete dense image embedding. Do not substitute decoded masks for encoder features. |

Teacher upgrades can be added behind versioned adapters, but changing a model
or preprocessing fingerprint creates a new cache schema version. It must never
silently reuse old entries.

## Student architecture: AV-RWKV-g

### Capacity target

The full model target is approximately **1.4-1.8B trainable parameters**. This
is deliberately in the same capacity class as the 1.1B RADIOv2.5-g rather than
a small adapter. We start with an approximately 100M architecture proxy to
prove data flow and losses, then scale the identical topology. The proxy is a
validation tool, not the deliverable.

The exact full configuration is selected by measured memory and throughput,
but reducing spatial output or teacher fidelity to make the model fit is not
allowed. Activation checkpointing, BF16, fused RWKV kernels, optimizer-state
offload, and gradient accumulation are execution choices.

### Hierarchical 2-D RWKV backbone

The student consumes aspect-preserving, dynamically sized images. Padding is
explicitly masked.

```text
pixels
  -> anti-aliased convolutional stem
  -> stride-4 high-frequency feature map
  -> stride-8  Vision-RWKV stage
  -> stride-16 Vision-RWKV stage
  -> stride-32 Vision-RWKV semantic stage
  -> top-down nonlinear feature-pyramid neck
  -> dense shared pyramid {P8, P16, P32} + summary bank
```

Each Vision-RWKV stage mixes information with row-forward, row-backward,
column-forward, and column-backward recurrent orders, local depthwise spatial
mixing, 2-D rotary/coordinate features, and learned scale/aspect conditioning.
Scan order rotates by block. Cross-scale fusion occurs at every stage boundary
and in the top-down neck so thin boundaries are not expected to survive only
through a stride-16 or stride-32 stream.

The backbone returns:

- `P8`: high-resolution boundary, texture, small-object, and OCR detail;
- `P16`: primary general-purpose dense representation;
- `P32`: global semantic/context representation;
- `S`: 16 learned summary tokens plus a masked global pooled vector;
- intermediate stage features for nonlinear DPT-style dense probes.

The canonical representation is this pyramid, not a fixed token array.

### Training-only teacher adapters

Every teacher has two adapters:

- a nonlinear dense adapter that samples/decodes the student pyramid directly
  at that teacher's native token coordinates;
- a summary adapter for its image-level embedding or class token.

MoonViT receives separate adapters for its three depths and four subtokens.
SAM's adapter decodes at exactly 64 x 64 after applying the recorded geometric
transform. Teacher adapters are discarded after training except for frozen
benchmark probes.

The student is successful only if these adapters recover all teachers from the
same shared backbone output. Merely learning one fused target is insufficient.

## Removing overlap without removing detail

Teacher overlap is removed in representation space, not by prematurely
discarding spatial tokens:

1. All teacher grids are mapped into a common continuous image-coordinate
   system while their native targets remain unchanged.
2. PHI-S standardizes each teacher distribution so high-variance SAM channels
   cannot dominate SigLIP2 or DINOv2.
3. A single shared pyramid is supervised by all teacher adapters on the same
   images. Concepts present in several teachers therefore share backbone
   channels and spatial states rather than becoming duplicated token streams.
4. Adapter bottlenecks and a cross-teacher effective-rank audit measure how
   many shared degrees of freedom are actually required. The shared width is
   reduced only when held-out reconstruction retains at least 99.5% of the
   standardized teacher variance and downstream gates remain green.
5. Teacher-residual probes measure information unique to each teacher. A
   common/private diagnostic decomposition is allowed during training, but the
   deployed representation contains the common union, not four routed experts.
6. Feature dropout removes random teachers or teacher regions during training
   so no single adapter becomes the de facto canonical space.

This is deduplication by shared explanation: one student feature can explain
the corresponding MoonViT, SigLIP2, DINOv2, and SAM features. We do not define
overlap as simple cosine similarity between raw teacher channels.

## Adaptive VLM token packer

The foundation pyramid remains dense. Downstream consumers may request an
adaptive token budget, initially 256, 512, or 1,024 tokens. There is no 64-token
default.

The packer applies output-time feature-based bipartite merging with:

- spatially strided target selection so every region remains represented;
- similarity computed on normalized student features;
- token mass, centroid, bounding extent, and source indices retained;
- an exact discrete unmerge map;
- protected tokens for OCR edges, high-frequency boundaries, rare semantic
  regions, and summary tokens;
- a reconstruction loss through every frozen teacher adapter after unmerging.

The budget is selected by measured reconstruction curves. The default is the
smallest budget whose unmerged student features preserve at least 99% of the
uncompressed adapter fidelity on the validation suite. If that requires 1,024
tokens, the default is 1,024. The old text RWKV must adapt to this interface;
the foundation model is not degraded to satisfy its old 64-token bridge.

## Distillation objectives

For teacher `t`, let `A_t(F(x))` be the student adapter prediction and
`PHI_t(T_t(x))` the PHI-S-standardized native teacher feature.

The primary objective contains:

1. native dense PHI-S MSE for every teacher and MoonViT tap;
2. summary-token cosine loss for SigLIP2, DINOv2, and MoonViT;
3. spatial relation/correspondence loss on pairwise token similarities;
4. cross-image relation loss so neighborhood and retrieval structure survive;
5. multi-resolution scale-equivariance consistency;
6. cross-crop and full-image coordinate consistency;
7. token-packer unmerge reconstruction through all teacher adapters;
8. masked-feature prediction and student self-distillation from an EMA copy;
9. variance/covariance anti-collapse regularization;
10. optional supervised/pseudo-supervised OCR, detection, masks, depth, and
    caption objectives only after the representation gates pass.

Teacher weights are controlled by PHI-S plus dynamic fidelity balancing based
on normalized unexplained variance. Raw loss magnitudes never determine the
balance. All teachers are presented on the same optimizer step when feasible;
sequential teacher micro-passes may accumulate gradients for memory, but they
must correspond to the same image batch.

## Multi-resolution and SAM training

The final training distribution includes simultaneous low and high resolution
branches. A representative schedule is:

1. 256-384 short-side warmup with all teachers;
2. 384-512 general training with all teachers;
3. simultaneous 384 and 768/1024 branches;
4. variable-aspect high-resolution finish with OCR/document oversampling.

For fixed-resolution teachers, student predictions are decoded at the teacher
coordinates. SAM uses native 1024 canvases and crop-aware mosaics; its target
features are never interpolated. Resolution and aspect distributions are
balanced so the student cannot switch from a DINO-like mode at low resolution
to a SAM-like mode at high resolution.

## Cache v2

The old `*_128_*` caches are not valid for this project. They remain archived
only for reproducibility and must never pass a v2 preflight.

Cache v2 is sharded and append-only:

```text
native-v2/<teacher fingerprint>/<manifest fingerprint>/<shard>/
  index.jsonl
  tensors-00000.safetensors
  tensors-00001.safetensors
  receipt.json
```

Each row records tensor names, native shapes, geometry, validity, and hashes.
Writers use temporary files plus atomic rename. Receipts include completed
sample IDs, permitting interruption and exact resume. Validation randomly
recomputes teacher outputs and requires shape, finite-value, coordinate, and
numeric round-trip checks.

Native caches are large. Work proceeds in trainable slices: cache a slice,
validate it, train/replay it, then retain or migrate it according to available
disk. Cache pressure never justifies 128-token pre-pooling.

## Training stages and gates

### Stage 0: statistics and architecture proxy

- Estimate PHI-S transforms and teacher effective ranks on at least 50k diverse
  images.
- Train the approximately 100M topology proxy.
- Overfit 128 images until every teacher's normalized reconstruction improves
  by at least 100x over its initialization and source coordinates round-trip.
- Prove exact resume reproduces sample order and optimizer state.

### Stage 1: full AV-RWKV-g representation distillation

- Initialize the full 1.4-1.8B student only after Stage 0 passes.
- Train all native dense and summary teacher objectives.
- Keep caption/text RWKV weights out of this stage.
- Save durable latest and independently retained best checkpoints for the
  geometric mean of held-out teacher fidelity.

### Stage 2: adaptive token packing

- Freeze or use a low learning rate for the backbone.
- Train 256/512/1,024-budget token packing and unmerge reconstruction.
- Choose the default budget from quality curves, not convenience.

### Stage 3: downstream alignment

- Pair the student with the 2.9B RWKV using the selected adaptive token
  contract.
- Train a new projector/resampler and then optionally unfreeze the upper vision
  blocks at low learning rate.
- Caption loss is introduced here, after vision fidelity has been established.

## SOTA acceptance suite

Every reported result records input resolution, token budget, frozen/unfrozen
status, parameter count, and throughput. Required comparisons include each
individual teacher, the literal four-teacher stack where applicable,
RADIOv2.5-g, and C-RADIOv4-H.

### Representation fidelity

- PHI-S normalized MSE and fidelity `teacher variance / residual variance` per
  teacher, tap, and resolution;
- linear CKA and neighborhood overlap;
- token-relation and cross-image retrieval preservation;
- adapter reconstruction after 256/512/1,024-token pack/unpack;
- scale-equivariance variance from 256 through 1024 resolution.

### Global and vision-language

- SigLIP2 text-aligned zero-shot and retrieval;
- ImageNet zero-shot and k-NN probes;
- TextVQA, ChartQA, DocVQA, InfoVQA, OCRBench;
- caption grounding and hallucination tests, not only teacher-forced PPL.

### Dense and spatial

- ADE20K and Pascal/VOC semantic segmentation probes;
- COCO instance segmentation using the SAM-compatible adapter;
- depth, surface normal, semantic correspondence, and multi-view probes;
- boundary and small-object metrics;
- OCR localization and reading at multiple scales.

### Claim gate

The model may be called a successful stack replacement only when:

1. no teacher-specific probe has a catastrophic regression;
2. the geometric mean of normalized teacher fidelity beats the strongest
   single-teacher baseline;
3. dense, global, and VLM benchmark groups each match or exceed the relevant
   strongest teacher within a predeclared tolerance;
4. the student beats the literal stack's practical quality/compute Pareto after
   including token packing;
5. results reproduce from a clean exact-resume checkpoint.

"SOTA" additionally requires beating the current public agglomerative baseline
on a predeclared aggregate, with no hidden reduction in input resolution or
token count.

## Immediate implementation order

1. Implement native-v2 cache payloads, geometry, fingerprints, receipts, and
   strict validators.
2. Change MoonViT extraction to preserve native grids/taps/views.
3. Cache native SigLIP2, DINOv2, and full 64 x 64 SAM embeddings.
4. Implement PHI-S statistics/calibration and reversible transforms.
5. Implement the approximately 100M hierarchical AV-RWKV proxy and native
   teacher adapters.
6. Pass the 128-image overfit, same-image multi-teacher, and exact-resume gates.
7. Measure effective rank and select the full model width/depth within the
   1.4-1.8B envelope.
8. Build validated cache slices and begin full Stage 1 training.

No old `*_128_*` cache, 54M canonical compressor, 64-token caption head, or
caption PPL checkpoint selector is permitted in this pipeline.
