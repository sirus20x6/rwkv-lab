# Mage-Flow Base appearance-expert training

`rwkv_lab.mage_flow_expert_train` is the Stage 2 trainer for the photo and
animation residual experts. It is intentionally separate from
`mage_flow_pretrain.py`, which updates every transformer weight in the editing
checkpoint.

The trainer is pinned to:

- `microsoft/Mage-Flow-Base`
- revision `59a9cfd58cf6ecef28245852c6bdace3f12428a2`
- Microsoft Mage source
  `ef932e2cc3e94bb026d937a6cffae65492adc0fb`

Run `scripts/cache_mage_flow_base.sh` once before planning a run.

## Training contract

The released transformer, Mage-VAE, and Qwen3-VL encoder are frozen. The
trainer wraps the final eight of twelve image FFNs by default and optimizes
only:

```text
photo residual FFNs + photo scales
animation residual FFNs + animation scales
```

The initial configuration uses:

```text
resolution contract:       512 native-resolution pixel budget
specialized blocks:        blocks 4–11 (final 66.7%)
expert parameter target:   15% of the released backbone per domain
expert hidden width:       12,544
microbatch size:           2 homogeneous-domain images
gradient accumulation:     4
expert learning rate:      1e-4
weight decay:              0.01
caption CFG dropout:       10%
activation checkpointing:  enabled
```

Expert parameters and AdamW moments remain FP32. Transformer forwards use BF16
autocast, giving BF16 tensor-core matmuls without reducing optimizer-state
precision. Final inference experts are exported as BF16.

Against the released 4,115,745,408-parameter transformer, 128-wide alignment
produces 616,687,624 parameters per domain: 14.9836% of the backbone. Both
experts add 1,233,375,248 parameters. The shared backbone is therefore 76.94%
of the combined architecture, within the design's 75–90% sharing target.

The first four transformer blocks remain entirely shared. Blocks 4–11 retain
their released image FFNs and add a photo or animation residual branch with
shape:

```text
3072 → 12544 → 3072
         GELU
```

The first expert-only stage assigns zero training weight to `general` rows and
uses no domain dropout. With the backbone frozen, either choice would disable
all trainable paths and waste the batch. General anchors and domain dropout
become useful during Stage 4, when selected shared layers are unfrozen.

## Conditioning dispatch

Captioned rows normally use their prepared `conditioning_text`. Ten percent
are stochastically replaced with the official Mage null prompt, a single
space, to preserve classifier-free guidance.

Uncaptioned rows always encode:

```text
<uncaptioned-image>
```

They are never eligible for CFG dropout. A semantic `<cfg-null>` record is
dispatched to the official single-space null path and is never tokenized
literally. Since the text encoder and shared backbone are frozen, uncaptioned
steps can update only their active visual expert.

## Plan and launch

After preparing train and optional evaluation manifests:

```bash
PYTHONPATH=src python -m rwkv_lab.mage_flow_expert_train plan \
  --train-manifest /path/train.jsonl \
  --eval-manifest /path/eval.jsonl \
  --run-dir runs/mage_flow_expert_plan \
  --output-dir runs/mage_flow_expert
```

The default plan uses `--expert-parameter-fraction 0.15`. The hidden width is
derived from the exact loaded backbone size rather than maintained as an
independent tuning value.

Planning validates:

- model and source pins;
- explicit domain labels;
- exact duplicate and cross-domain leakage audits;
- the 15% default uncaptioned ceiling;
- availability of every positively weighted domain;
- all optimizer, architecture, and checkpoint settings.

At model startup, a second preflight inspects every transformer parameter. It
aborts if any non-expert parameter is trainable, any expert parameter is
accidentally frozen, a fresh expert output projection is nonzero, or a wrapper
does not begin on the neutral route.

It writes:

```text
runs/mage_flow_expert_plan/
├── train_config.json
├── preparation_receipt.json
└── launch.sh
```

Launch only after the Stage 0 baseline and full-size zero-output parity check:

```bash
runs/mage_flow_expert_plan/launch.sh
```

The initial implementation is deliberately single-GPU. This avoids
different-domain routes selecting different unused parameters across data
parallel ranks. Sequence or data parallelism can be qualified after the
single-GPU prototype.

## Batching

Every microbatch has one hard route. Photo and animation microbatches are
scheduled from their configured weights and cycle round-robin through sources
inside each domain. Sampling can repeat examples from a small source rather
than allowing a much larger source to dominate.

Gradient accumulation may combine photo and animation microbatches in one
optimizer step. Metrics record the number of microbatches from each route.
The route context remains active through `backward()` because activation
checkpointing recomputes transformer blocks during backward.

## Evaluation

Evaluation uses fixed flow noise and no caption dropout. For every data domain
available in the evaluation manifest, it computes the complete route matrix:

```text
general data   × general/photo/animation routes
photo data     × general/photo/animation routes
animation data × general/photo/animation routes
```

The scheduled evaluation is one unified phase at a single training
`global_step`. By default it exhausts every row in the evaluation manifest:
all held-out photo and animation images are evaluated together, and each image
is forwarded through all three routes. The event logs exact per-domain and
total example counts so partial coverage cannot look like a complete eval.
Configuration validation requires an evaluation domain for every weighted
expert and rejects exact image-ID overlap with the training manifest.

That same phase also publishes a deterministic trainboard gallery with four
held-out photo prompts routed through the photo expert and four held-out
animation prompts routed through the animation expert. Each generated image is
paired with its held-out target. Gallery generation uses 30 rectified-flow
steps at CFG 5 and has no independent cadence: a gallery and the complete
128+128 scalar route matrix always share the same global step.

This supplies the primary expert targets, neutral baselines, and contamination
negative controls. Visual benchmark generation remains a Stage 0/quality-gate
operation and is not silently substituted with training loss.

## Checkpoints and resume

Each atomic checkpoint contains:

```text
checkpoint-NNNNNNNN/
├── mageflow-photo-expert.safetensors
├── mageflow-animation-expert.safetensors
├── trainer_state.pt
└── checkpoint.json
```

The local trainer state includes AdamW, scheduler, Python RNG, CPU and CUDA RNG,
and the exact next epoch/batch position. A fingerprint covers the complete
optimization configuration and train/evaluation manifest contents. Resume is
rejected if that contract changes.

Set `resume_from` in the generated `train_config.json` to an intact checkpoint
directory. At completion the output root also contains inference-oriented:

```text
mageflow-photo-expert.safetensors       # BF16, no shared weights
mageflow-animation-expert.safetensors   # BF16, no shared weights
```

## Verification

The non-GPU suite is:

```bash
PYTHONPATH=src pytest -q \
  tests/test_mage_flow_adaptation.py \
  tests/test_mage_flow_expert_train.py \
  tests/test_mage_flow_pretrain.py
```

It covers conditioning dispatch, data audits, homogeneous scheduling,
zero-output parity, FP32 trainable isolation, exact checkpoint round trips,
BF16 modular exports, planning, and the existing flow objective.
