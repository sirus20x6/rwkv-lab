# Mage-Flow-Edit full continued pre-training

This integration updates every weight in the 4B NR-MMDiT backbone of
`microsoft/Mage-Flow-Edit-Base`. It is full-backbone training, not LoRA. The
Mage-VAE and Qwen3-VL-4B conditioning encoder remain frozen, matching
Microsoft's published training contract.

Image/caption pairs are generation examples: the target latent is noised and
conditioned only on its caption. Do not turn them into source=target identity
edits. Microsoft trained the editor with a balanced edit/generation mixture in
stage 1 and a 2:1 edit/generation mixture in stage 2; generation pairs preserve
the inherited visual prior but do not replace real `(source, instruction,
target)` triples when the goal is to improve editing itself.

The implementation uses:

- the released checkpoint's scheduler-compatible velocity target `epsilon - z`
  at `z_t = (1-t)z + t*epsilon`;
- the official `mage-flow` prompt template, Mage-VAE encoder, Qwen3-VL packed
  conditioning path, frame-aware RoPE, and FlashAttention varlen model;
- native aspect ratios with no crop or distortion, approximately a 1024x1024
  pixel budget, and token-budgeted heterogeneous packs;
- BF16 full weights, activation checkpointing, AdamW, cosine decay, prompt
  dropout for CFG preservation, DeepSpeed ZeRO-2, and CPU-offloaded optimizer
  states;
- rank-local deterministic packing plus model, optimizer, scheduler, RNG, and
  data-position checkpoints.

## Prepare

The Mage-Flow dependency stack conflicts with this repository's older
Transformers environment, so it deliberately lives in a separate Python 3.11
environment.

```bash
scripts/bootstrap_mage_flow_pretrain.sh
scripts/prepare_mage_flow_edit_pretrain.sh
```

The bootstrap pins:

- Microsoft Mage source commit
  `ef932e2cc3e94bb026d937a6cffae65492adc0fb`
- model snapshot
  `8654a7bc0283ab2946385230b5b2eb944e0b76ea`
- PyTorch 2.13, Transformers 5.5, Diffusers 0.38, DeepSpeed 0.19.3, and
  FlashAttention 2.8.3.

It also prebuilds DeepSpeed CPUAdam and downloads the pinned model snapshot into
the repository's ignored `.hf_cache`. Set `MAGE_FLOW_DOWNLOAD_MODEL=0` only
when preparing an offline worker whose cache will be populated separately.

Preparation scans and decodes every image, applies EXIF orientation, rejects
missing/corrupt/non-caption rows and aspect ratios above 4:1, removes duplicates
and truncated captions, and writes a receipt beside each canonical manifest.
The default source is the append-only Qwen caption file under
`/workspace/git/datasets/private_web/web_forum/subweb_forums`. Preparation freezes the
earliest valid prefix at a single EOF, then performs a deterministic seeded
split targeting 5000 training images and 128 held-out images. Each generated
manifest is a durable snapshot. Override the locations with
`MAGE_FLOW_REDDIT_ROOT`, `MAGE_FLOW_CAPTION_SOURCE`, and
`MAGE_FLOW_ARTIFACT_SOURCE`.

For the Web Forum quick test, artifact screening is strict: unscanned images and
any image with a watermark or censorship flag are excluded. The preparer
requests 5000 train rows plus 128 eval rows, but safely uses the largest
available clean training count rounded down to a multiple of eight while the
scanner catches up.

## Launch

```bash
runs/mage_flow_edit_web_forum_cpt_plan/launch.sh
```

The default 10240-token pack measures and includes both image and Qwen text
tokens; it normally holds two 1024-square generation examples. With gradient
accumulation 4, one optimizer step sees about eight images per GPU. On the
96 GB RTX PRO 6000, start with this setting. Increase
`packed_sequence_tokens` only after measuring peak memory with the real model.
The Web Forum quick test is exactly one epoch. With two images per pack and four
packs per optimizer step, the launcher computes `train_rows / 8` steps from
the prepared clean manifest: 625 steps at 5000 images. It uses a reduced
`2e-6` learning rate, 40 warmup steps, evaluation every 100 steps over 32
held-out packs, and checkpoints every 100 steps.

The lower learning rate and single epoch are intentional: 5000 images are a
small corpus relative to a 4B backbone, and a multi-epoch `1e-5` run would
heavily overfit and erode the inherited editing prior.

To resume, set `resume_from` in `train_config.json` to an intact
`checkpoint-NNNNNNNN` directory. The checkpoint includes full optimizer and RNG
state plus the exact next epoch/pack position for every rank.

At successful completion, `hf_transformer/` contains the BF16
`diffusion_pytorch_model.safetensors` and transformer config. Replace the
`transformer/` directory in a snapshot of the pinned Base model with this
directory; the frozen VAE, text encoder, scheduler, and `model_index.json` stay
from the Base snapshot.

## Data contract

Input JSONL may use:

```json
{"image": "relative/or/absolute/path.jpg", "text": "A detailed caption", "task": "caption"}
```

`caption` is also accepted in place of `text`. Relative paths resolve against
`--data-root`, not the manifest directory. Structured mask, bbox, and OCR rows
are intentionally excluded from this image-generation run.

The canonical output stores absolute paths, source and train geometry, latent
token count, and a stable image identity. Keep evaluation images in a separate
manifest and pass it with `--exclude-manifest` while preparing training data.

## Quality boundaries

- No pooling, latent downsampling, low-rank adapters, quantized model weights,
  or resolution buckets are used.
- Images are resized only to the same native-aspect pixel-budget regime
  described in the paper; they are never center-cropped.
- The paper states uniform `t` in the rectified-flow objective, so that is the
  default. It prints the velocity with the opposite sign from the released
  Diffusers scheduler path; the trainer follows the functioning checkpoint and
  official inference/inversion code (`epsilon-z`). Shifted-uniform and
  logit-normal sampling are available only as explicit experiments.
- Caption-only continued pre-training can improve domain generation and retain
  editor robustness. Improving edit adherence requires adding high-quality
  edit triples in a later extension; identity pairs are not a substitute.
