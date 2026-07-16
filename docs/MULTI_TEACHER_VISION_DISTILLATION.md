# Multi-Teacher Vision Compression and RWKV-Native Student

## Objective

Replace the frozen MoonViT, SigLIP2, DINOv2, and SAM ensemble with one compact
vision model that emits visual embeddings in the exact format expected by the
2.9B RWKV captioner.

The intended deployment contract is:

```text
input:   image pixels
output:  [batch, 128 visual tokens, 2560 channels]
dtype:   bfloat16
meaning: continuous visual pseudo-token embeddings, not text-token IDs
```

The output can be inserted directly into RWKV's embedding sequence. The final
deployment model should not require the four teacher towers, the teacher
compressor, a separate MoonViT projector, fusion residuals, or deep-vision
adapters.

MoonViT does not have a text tokenizer. It patchifies images into continuous
features. RWKV retains its 65,536-entry text tokenizer for captions while the
vision model supplies continuous embeddings with RWKV's 2,560-channel width.

## Motivation

The four teachers provide complementary information:

- MoonViT: the representation used by Kimi's multimodal path, including useful
  intermediate-layer features.
- SigLIP2 So400m: language-aligned semantics, retrieval, and localization.
- DINOv2: general visual structure, dense correspondence, and appearance.
- SAM: object boundaries and dense spatial organization.

Running all four models at inference is expensive. Training four separate RWKV
replacements would also preserve their incompatibilities. Instead, first learn
one coherent latent representation from the frozen teacher committee, then
train one image model to predict that representation.

This is related to agglomerative multi-teacher distillation in
[AM-RADIO](https://arxiv.org/abs/2312.06709) and its token-compression and
teacher-balancing improvements in
[RADIOv2.5](https://arxiv.org/abs/2412.07679). The explicit intermediate latent
codec proposed here separates teacher reconciliation from image-encoder
training and makes the expensive teacher outputs reusable.

## Existing Teacher Cache Contracts

The next-stage cache is independent of trainable adapters:

```text
MoonViT: [3 stages, 128 tokens, 4 subpatches, 1152 channels]
Fusion:  [128 tokens, 2176 channels]
SAM dense supplemental: [256 channels, 64 rows, 64 columns]
```

The fusion channels can be separated as:

```text
SigLIP2 So400m: [0:1152]       -> 1152 channels
DINOv2 Base:    [1152:1920]    ->  768 channels
SAM ViT-B:      [1920:2176]    ->  256 channels
```

The MoonViT cache retains layers 8, 17, and 26. A merged MoonViT patch group
contains `4 × 1152 = 4608` values before projection.

The 256-channel SAM tensor is the native dense image-encoder embedding grid,
not a decoded segmentation mask. It is cached separately so the compressor can
learn dense correspondence without the destructive 1-D pooling used by the
caption-prefix fusion cache. Actual prompted or automatic masks can be added as
RLE pseudo-labels later if segmentation output, rather than spatial structure,
becomes a required student capability.

## Proposed Architecture

```text
MoonViT stages ─┐
SigLIP2 ────────┤
DINOv2 ─────────┼─> teacher normalizers ─> latent compressor ─> canonical Z
SAM ────────────┘                              │
                                              ├─> teacher reconstruction heads
                                              └─> frozen/low-LR RWKV captioner

image pixels ─> small spatial Vision-RWKV student ─> canonical Z ─> RWKV
```

### Teacher-specific input stems

Each teacher needs its own normalization and input projection. Teacher and
MoonViT-stage identity embeddings prevent the compressor from confusing
incompatible feature spaces. Spatial coordinates and source aspect ratio
should accompany the tokens so differently shaped images do not become
geometrically ambiguous after pooling.

### Canonical latent compressor

A small Perceiver-style or RWKV-style resampler should cross-aggregate the
teacher streams into a fixed latent array. The initial conservative target is:

```text
canonical core: [batch, 128, 1024]
```

This retains the current 128-token spatial budget while reducing channel
width. A learned output expansion can produce `[batch, 128, 2560]` for the
captioner. Ablations can later test 96 or 64 tokens, but the first version
should prioritize retention over maximum compression.

The compressor should be approximately 40–100M parameters, not another
foundation-sized vision tower.

### Reconstruction heads

Temporary teacher-specific decoder heads reconstruct:

- all three cached MoonViT stages;
- SigLIP2 semantic tokens;
- DINOv2 structural tokens;
- SAM spatial tokens.

These heads make the bottleneck preserve each teacher's distinctive
information rather than collapsing toward the easiest caption-semantic signal.
They are training-only components and are discarded at deployment.

### Deployable image student

The second model consumes pixels and predicts the frozen canonical latent. A
reasonable initial student is a bidirectional/multi-directional spatial
Vision-RWKV with:

- MoonViT-compatible aspect-preserving patchification;
- internal width between 768 and 1,152;
- 12–27 spatial RWKV blocks, depending on the parameter target;
- a native final output of 128 tokens by 2,560 channels.

The final 2,560-channel layer is part of the vision model's native interface,
not a separately deployed compatibility adapter. Making every internal layer
2,560-wide is unnecessary and would make the student much larger.

## Training Phases

### Phase 0: Raw teacher shards

1. Cache MoonViT and aligned fusion features in bounded shards.
2. The first shard contains 40,000 training images plus all 384 stable eval
   images.
3. Preserve SAM's native `256 × 64 × 64` image embeddings for the first shard.
4. Do not delete raw teacher features until the compressor architecture and
   latent schema are frozen and validated.

### Phase 1: Train the teacher compressor

Train entirely from cached features; the image towers do not need to run.

Recommended objectives:

- per-teacher cosine and normalized feature reconstruction;
- relational or Gram-matrix loss to preserve patch geometry;
- variance/covariance regularization to prevent latent collapse;
- caption cross-entropy through RWKV;
- optional RWKV hidden-state and caption-logit distillation;
- random teacher dropout so the latent is robust to missing or noisy teachers.

Teacher losses must be normalized and monitored separately. SigLIP2 is likely
to dominate caption semantics unless DINOv2 and SAM receive explicit spatial
and relational objectives.

Split the first shard into compressor train and validation sets. Once its
architecture, losses, and latent statistics are stable, freeze and fingerprint
the compressor and canonical schema.

### Phase 2: Produce canonical latent caches

Run the frozen compressor over each raw teacher shard. Validate every output
for shape, dtype, finite values, and matching compressor fingerprint.

Approximate BF16 storage for 400,384 images:

| Representation | Approximate size |
|---|---:|
| Current raw teacher caches | 1.49 TiB |
| `128 × 1024` canonical core | 98 GiB |
| `128 × 2560` RWKV-native output | 244 GiB |

The compact 1,024-channel core is about 15 times smaller than the raw cache.
The student can learn the final 2,560-channel expansion as part of its own
output path.

Raw shards may be removed only after the canonical cache has been validated,
backed up if required, and the compressor has been declared immutable.

### Phase 3: Train the image-to-latent student

Train the student from image pixels to canonical latents without running any
teacher towers. Use:

- canonical latent regression;
- decoded teacher reconstruction through the frozen Phase-1 heads;
- RWKV intermediate hidden-state matching;
- caption-logit distillation;
- ordinary caption cross-entropy;
- multi-resolution and aspect-ratio-balanced batches.

The reconstruction and caption losses should remain active after latent loss
converges so the student does not merely reproduce easy low-frequency latent
structure.

### Phase 4: Consolidate adapters into RWKV

Use the completed multimodal model as a teacher while progressively removing
the current internal vision adapters:

1. Feed the student's native `[B, 128, 2560]` tokens directly to RWKV.
2. Unfreeze RWKV at a low learning rate.
3. Anneal deep-vision, layer-matching, and fusion-adapter contributions toward
   zero.
4. Match teacher logits and selected hidden states while retaining caption
   cross-entropy.
5. Remove the zeroed adapters and continue a short stabilization phase.
6. Merge any mathematically mergeable low-rank RWKV updates into base weights.

The result retains a small vision model plus the RWKV captioner, without the
four original teachers or internal vision-adapter stack.

## Why Direct Checkpoint Conversion Is Not the Plan

Transformer attention weights cannot be algebraically mapped into RWKV spatial
recurrence weights. Some compatible patch embeddings, normalizations, or MLP
weights may provide initialization, but the spatial mixer must learn through
distillation. A direct block swap followed by ordinary fine-tuning would likely
destroy much of the teachers' pretrained knowledge.

A fully 1,152-wide RWKV language model would be a new, much smaller language
model and would not preserve the current 2.9B checkpoint. A fully 4,608-wide
RWKV would be much larger. The native-output student avoids both extremes.

## Principal Risks and Required Telemetry

- **Teacher imbalance:** track every teacher loss and gradient contribution.
- **Latent collapse:** track per-channel variance, covariance, rank, and token
  diversity.
- **Spatial loss after pooling:** train the SAM reconstruction head against the
  supplemental native 64-by-64 grid, retain aspect/grid metadata, and evaluate
  dense correspondence separately from caption quality.
- **Resolution mode shift:** train and evaluate at multiple aspect ratios and
  resolutions.
- **Schema drift:** fingerprint compressor weights, token count, width,
  normalization, teacher identities, and cache source metadata.
- **Caption-only overfitting:** keep teacher reconstruction validation even if
  caption perplexity improves.
- **Premature raw-cache deletion:** retain raw shards until the canonical codec
  is locked and reproducible.

Evaluation should include teacher reconstruction similarity, image-text
retrieval, spatial correspondence, grounding accuracy, caption likelihood,
free-running caption quality, and human review of hallucination and omitted
details.

## Current Decision

The immediate cache handoff remains:

1. stop multi-view MoonViT at 40,000 cached training images;
2. add the 384 non-adult eval images;
3. build SigLIP2 So400m, DINOv2, and SAM features for the same shard;
4. use that shard to prototype the multi-teacher compressor;
5. freeze the latent contract before compressing or deleting later raw shards.
