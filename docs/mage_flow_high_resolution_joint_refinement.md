# Mage-Flow Stage 4A: high-resolution joint refinement

> Historical plan: its neutral-route preservation and continued expert updates
> do not describe the current recovery stage. The active contract is
> [mage_flow_shared_tail_recovery.md](mage_flow_shared_tail_recovery.md): both
> learned experts stay frozen, every sample selects exactly one expert, and
> only original Mage-Flow blocks 8–11 adapt first.

## Decision

The next stage continues the completed 512-equivalent appearance experts on
the exact same photo and animation datasets, raises native resolution in
qualified rungs, and lets a small late portion of the shared backbone learn.
It is a continuation from the final Stage 2 expert checkpoint, but it starts a
fresh optimizer and learning-rate schedule because the trainable parameter set
and image geometry change.

This is not a full-backbone run and it does not cast persistent weights or
optimizer state to FP8. FP8 is restricted to qualified MLP matrix
multiplications. Attention, normalization, modulation, the VAE, the flow
objective, gradient reduction, and checkpoint weights remain BF16 or FP32.

The live 512-equivalent run is not modified by this design.

## Immutable parent and data

Parent:

```text
runs/mage_flow_expert/checkpoint-00016000
```

The stage must refuse to start until the parent is complete and passes its
unified evaluation. It reuses the current content, captions, domain labels,
and train/evaluation separation:

```text
train: 19,223 images
  photo:      3,500
  animation: 15,723

eval: 256 images
  photo:        128
  animation:    128

train manifest SHA-256:
bd0012f28469506896c5a7e8e999bbfb7d799d1ee0919ddded9cb77c1890c2c5

eval manifest SHA-256:
eb54598a15fee202f27714bb4a4373a7a5563314edc4a64229c8ce73158f03d0
```

Rung manifests are derived only by changing training geometry and, above one
megapixel, selecting source-resolution-eligible rows. Image IDs, captions,
conditioning, and source paths must be byte-for-byte identical to their
parent rows. No new image or caption source is introduced.

Never enlarge an image beyond its decoded native dimensions. Dimensions
remain multiples of 16, preserve aspect ratio, have a maximum side of 2048,
and retain the existing 4:1 aspect-ratio limit.

Current source-resolution eligibility is:

| Rung | Source-pixel threshold | Photo train | Animation train | Photo eval | Animation eval |
|---|---:|---:|---:|---:|---:|
| 1024-equivalent | none; cap low-resolution rows at native size | 3,500 | 15,723 | 128 | 128 |
| 1536-equivalent | 2,359,296 | 2,196 | 8,138 | 71 | 75 |
| 2048-equivalent | 4,194,304 | 1,679 | 4,917 | 56 | 46 |

The full 128+128 evaluation set is still evaluated at every unified
evaluation. Rows below the current rung's source threshold remain at their
largest non-upscaled geometry; they are not removed from evaluation.

## Trainable boundary

Continue training both existing residual experts:

```text
blocks 4-11 photo residual FFNs
blocks 4-11 animation residual FFNs
all expert route scales
```

Also unfreeze only:

```text
transformer_blocks.10.img_mlp.shared_ffn
transformer_blocks.11.img_mlp.shared_ffn
proj_out
```

This adds approximately 151,419,008 trainable shared parameters, 3.68% of the
released 4,115,745,408-parameter backbone. The text encoder, VAE, early
backbone, text stream, modulation layers, all attention projections, and
blocks 0-9 shared image FFNs remain frozen.

The experts use a `1.0x` learning-rate multiplier. The selected shared
parameters and output projection use `0.1x`. Expert dropout is 12.5% on
captioned domain rows. A dropped row uses the general route and therefore
updates only the selected shared parameters. Photo and animation
microbatches are paired in every optimizer update so shared gradients never
come from a one-domain update.

Because this dataset has no general anchor rows, shared-weight promotion is
conditional on neutral-route evaluation. This stage must not progressively
unfreeze more shared layers.

## Resolution and precision curriculum

Every microbatch contains one image. Gradient accumulation is two: one photo
microbatch and one animation microbatch per optimizer update. This bounds
activation memory while preserving exact 50/50 domain exposure.

| Rung | Updates | Geometry | Maximum image tokens | Compute | Expert LR | Shared LR |
|---|---:|---|---:|---|---:|---:|
| A | 1,500 | 1024-equivalent, no upscale | 4,096 | hybrid FP8/BF16 | 3e-5 | 3e-6 |
| B | 750 | 1536-equivalent, eligible train rows | 9,216 | hybrid FP8/BF16 | 2e-5 | 2e-6 |
| C | 250 | 2048-equivalent, eligible train rows | 16,384 | hybrid FP8/BF16 | 1e-5 | 1e-6 |
| D | 500 | 60% 1024, 30% 1536, 10% 2048 | 16,384 | BF16 | 5e-6 | 5e-7 |

“Equivalent” is a pixel budget, not a forced square resize. Native aspect
ratios are retained. The 2048 rung therefore includes 2048x2048 for square
sources and smaller-area shapes when the 2048 maximum side is reached.

Each rung starts from the preceding rung's weights but creates a new
optimizer/scheduler state. Rung A has 100 warmup updates. Later rungs have 25
warmup updates. Cosine decay ends at 20% of each rung's initial learning rate.

Rung D is the numerical-quality polish. Use BF16 rather than IEEE FP16 because
BF16 retains substantially more exponent range. It mixes resolutions so the
final checkpoint does not specialize exclusively to the most expensive
bucket.

## FP8 boundary

FP8 conversion is allowlisted, not recursive:

```text
expert fc1/fc2 GEMMs
shared image-FFN GEMMs
```

Use dynamically scaled E4M3 for forward activations and weights and E5M2
where the qualified backward implementation requires it. Retain FP32 master
expert weights, FP32 optimizer moments, FP32 clipping/norm calculations, and
BF16 checkpoint exports. Do not quantize:

```text
attention Q/K/V or attention output projections
FlashAttention inputs
normalization or modulation
img_in, txt_in, norm_out, or proj_out
Mage-VAE or Qwen3-VL
flow targets or loss
```

The installed Mage environment does not currently contain a qualified FP8
training library. Rung A cannot launch until the chosen implementation passes
BF16/FP8 forward, loss, gradient, checkpoint-resume, and zero-route parity
tests on this exact Blackwell GPU.

## Memory contract

The target GPU has 97,887 MiB. A rung is launchable only after a disposable
qualification process performs ten complete optimizer updates, both domains,
a scalar evaluation batch through all three routes, checkpoint save/resume,
and one gallery decode.

The process must record:

```text
peak allocated VRAM <= 80 GiB
peak reserved VRAM  <= 86 GiB
```

The margin is intentional: evaluation, compilation, allocator fragmentation,
and checkpoint transitions must not turn a successful microbenchmark into a
long-run OOM.

Memory measures used by the stage:

1. cache immutable Qwen conditioning outputs and unload Qwen3-VL during
   optimization;
2. keep activation checkpointing enabled for every transformer block;
3. use microbatch one at every resolution;
4. move the VAE decoder off GPU outside the unified gallery phase;
5. keep only the VAE encoder resident during training;
6. use CPU-offloaded Adam moments if the resident optimizer violates the
   80/86 GiB limits;
7. if needed for rung C only, enable pinned-CPU saved-tensor offload.

There is no automatic resolution downgrade. If a rung still exceeds either
limit after its specified offload tier, that rung is refused and the previous
rung becomes the candidate final checkpoint.

## Evaluation and promotion

Unified evaluation happens at the parent and at the end of every rung. Each
event consumes all 128 photo and all 128 animation rows and forwards every row
through general, photo, and animation routes. The dashboard gallery retains
four photo and four animation examples, each beside its original target.
Gallery generation stays at 1024-equivalent for direct visual comparability;
the scalar flow evaluation uses each rung's non-upscaled geometry.

A rung is promotable only when:

```text
all losses and gradients are finite
memory contract passes
checkpoint exact-resume passes
neutral-route aggregate loss regresses no more than 2% from the Stage 2 parent
neither domain's neutral-route loss regresses more than 3%
each specialist route beats its neutral route on its own domain
cross-domain routing does not become the best route for either domain
```

FP8 rungs additionally require, on a fixed paired probe:

```text
prediction cosine similarity to BF16 >= 0.995
trainable-gradient cosine similarity to BF16 >= 0.99
relative loss difference <= 1%
```

Failure leaves the preceding checkpoint untouched and visible in the
dashboard. Promotion never overwrites the Stage 2 parent.

## Checkpoint contract

Every rung checkpoint contains both experts, a BF16 shared-weight delta for
the explicitly selected shared parameters, optimizer/scheduler state, RNG and
data position, FP8 scaling state where applicable, and hashes of the parent
checkpoint and derived rung manifests. The inference export is:

```text
mageflow-shared-stage4a-delta.safetensors
mageflow-photo-expert.safetensors
mageflow-animation-expert.safetensors
```

Loading the shared delta is explicit. Omitting it reconstructs the unchanged
released Mage-Flow neutral route plus the Stage 2 experts.

## Required implementation before launch

1. Add deterministic no-upscale resolution-rung manifest derivation.
2. Add selected-shared parameter groups and differential learning rates.
3. Add paired-domain accumulation and 12.5% expert dropout.
4. Add allowlisted FP8 MLP training with FP32 masters.
5. Add immutable conditioning caches and component residency control.
6. Add the disposable memory/parity qualifier and fail-closed rung launcher.
7. Extend checkpoint lineage so a fresh optimizer can load parent expert
   weights while recording the immutable Stage 2 checkpoint hash.
8. Keep the existing unified 128+128 route matrix and target/generated
   side-by-side dashboard contract.
