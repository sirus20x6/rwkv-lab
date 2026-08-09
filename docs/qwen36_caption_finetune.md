# Qwen3.6 caption-distillation run

This run implements the contract in
`/thearray/git/datasets/captiontest/QWEN36_CAPTION_FINETUNE_HANDOFF.md`.
It is prepared but must not be submitted until the operator approves GPU use.

## Frozen inputs

- Base: `/thearray/git/ob/text-generation-webui/models/Qwen3.6-35B-A3B-heretic`
- Dataset: `/thearray/git/datasets/captiontest/qwen36-caption-finetune-v1`
- Dataset digest: `4869008d3853b22c83fcf67ec02f9fb22b331203c58efad8a6d2e7b9d2760347`
- Split counts: 11,909 train, 680 validation, 674 test
- Fixed validation generations: 100 images, 20 from each source subset
- Blinded pixel audit: 200 test images, 40 from each source subset
- LoRA target digest: `b7fff9ea6e4ca3d3d805b2080b6b8a92bb5bc72ada6ea7e84004305831a60b4f`

The split is assigned only from the SHA-256 filename stem. Exact content cannot
cross splits. Dataset preparation fails on count drift, missing images, empty
captions, duplicates, or changed split bytes.

## First arm

The base checkpoint stays BF16. The vision tower, visual merger/projector,
router, fused 256-expert banks, and MTP head are not trained. Rank-256 LoRA with
alpha 512 and dropout 0.05 covers exactly 311 checkpoint-enumerated modules:
language linear-attention projections, full-attention projections, shared-expert
projections, and `lm_head`. No suffix-only target rule can accidentally capture
a visual module.

The sealed rank-256 target set contains 402,751,488 adapter parameters. Its
BF16 parameter payload is about 768 MiB; PEFT's stability-oriented FP32 adapter
weights, gradients, and Adam moments require roughly 6 GiB before allocator
overhead. The 90 GiB free-VRAM launch gate remains in force.

Images retain aspect ratio under one preprocessing policy with 65,536 minimum
and 1,048,576 maximum pixel area. The checkpoint processor and chat template
own all image tokens. System, user, expanded image, and padding tokens are
masked; loss covers only the assistant caption and normal assistant terminator.

The initial schedule is one epoch: 11,909 microbatches, accumulation 16, 745
optimizer steps, 22 warmup steps, then cosine decay from 2e-5 to 2e-6. Validation
loss and the fixed 100-image generation set run at steps 250, 500, and 745.
Dashboard checkpoint and pause requests are honored at optimizer-step boundaries;
the resulting checkpoint has compatible-resume status. Live hyperparameter changes
remain disabled for this qualified first arm.

## Evaluation order

The worker first generates all 674 test captions with the untouched BF16 model,
using deterministic decoding. Only then does it install LoRA. After the fixed
one-epoch decision, it evaluates the final adapter on the same 674 rows with
identical prompt, processor bounds, and decoding. Raw responses, timings, and
per-image failures are append-only and resumable.

The terminal worker creates a blinded three-way 200-image audit containing the
untouched baseline, adapter result, and stored teacher caption. Candidate labels
are deterministically shuffled per image; the answer key is separate. Test data
never participates in optimization or validation checkpoint selection.

## Prepared definitions

- Standalone resolved config: `experiments/qwen36_caption_distill_lora_v1.json`
- Declarative TrainVM graph: `experiments/qwen36_caption_distill_lora_v1.trainvm.json`
- Content roots: `experiments/qwen36_caption_distill_lora_v1.input-roots.json`
- Audited targets: `experiments/qwen36_caption_distill_lora_v1.targets.json`

After this branch lands and the worker deployment is accepted, lock the large
input roots once, validate the locked document, and submit through the dashboard:

```bash
trainvm lock-input-content \
  experiments/qwen36_caption_distill_lora_v1.trainvm.json \
  experiments/qwen36_caption_distill_lora_v1.input-roots.json \
  > experiments/qwen36_caption_distill_lora_v1.locked.json

trainvm validate experiments/qwen36_caption_distill_lora_v1.locked.json
```

Do not run the standalone `train` action alongside TrainVM; the dashboard run
is the authority for GPU ownership, checkpoint publication, metrics, and resume.
