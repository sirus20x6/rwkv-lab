# Mage-Flow adaptation foundation

This is the CPU-testable foundation for Stages 0–2 of
`mageflow_staged_plan.md`. It is intentionally separate from
`mage_flow_pretrain.py`, which updates the full Mage-Flow-Edit backbone.

The adaptation path is pinned to:

- model: `microsoft/Mage-Flow-Base`
- model revision: `59a9cfd58cf6ecef28245852c6bdace3f12428a2`
- Microsoft Mage source revision:
  `ef932e2cc3e94bb026d937a6cffae65492adc0fb`

No model download, inference, VAE encoding, or GPU training occurs in the
foundation commands.

Cache and validate the pinned Base snapshot without initializing CUDA:

```bash
scripts/cache_mage_flow_base.sh
```

## Stage 0 contracts

Write the immutable v1 generation, editing, and VAE evaluation contract:

```bash
PYTHONPATH=src python -m rwkv_lab.mage_flow_adaptation \
  write-benchmark \
  --output runs/mage_flow_adaptation/benchmark_v1.json
```

The suite fixes prompts, seeds, dimensions, model revisions, VAE categories,
and required metrics. Editing cases name reference fixtures that must be
provided before the editing baseline can run.

`inspect_safetensors_layout()` reads tensor keys, shapes, and dtypes without
materializing model tensors on a GPU. `checkpoint_key_report()` reports all
missing and unexpected keys before a checkpoint is loaded.

`vae_reconstruction_report()` consumes normalized decoded tensors and computes
PSNR, local SSIM, Sobel edge error, normalized OCR character accuracy, and
frame-to-frame reconstruction jitter on CPU. LPIPS is accepted as an optional
callable so its implementation and weights can remain isolated in the Mage
environment.

The GPU phase still needs to record deterministic output hashes, peak VRAM,
latency, throughput, variable-aspect behavior, and the actual conditioning
tensor layouts.

## Stage 1 data contract

Every input JSONL row must explicitly declare one of:

```json
{"image": "photo.jpg", "domain": "photo", "caption": "A detailed caption.", "source": "dataset-a"}
{"image": "drawing.png", "domain": "animation", "dense_caption": "A detailed caption.", "source": "dataset-b"}
{"image": "anchor.jpg", "domain": "general", "human_caption": "A curated caption.", "source": "anchor"}
```

Prepare a native-resolution 512-pixel-budget manifest with:

```bash
PYTHONPATH=src python -m rwkv_lab.mage_flow_adaptation \
  prepare-domain \
  --input raw.jsonl \
  --output prepared.jsonl \
  --data-root /absolute/image/root
```

The preparer:

- requires explicit `general`, `photo`, or `animation` labels;
- applies the curated/human, dense automatic, generic, short automatic,
  tags/entities, and uncaptioned hierarchy;
- hashes image contents and removes exact duplicates;
- writes native geometry divisible by Mage-VAE's 16× spatial compression;
- records caption provenance and the permitted update scope;
- rejects uncaptioned `general` rows because they have no visual expert to
  receive an expert-only update.

Missing captions use `<uncaptioned-image>`. The true classifier-free-guidance
case uses `<cfg-null>` and is a different conditioning kind. They must never be
collapsed by a tokenizer or collator.

Audit an existing manifest with:

```bash
PYTHONPATH=src python -m rwkv_lab.mage_flow_adaptation \
  audit-domain \
  --manifest prepared.jsonl \
  --output prepared.audit.json
```

The audit fails on cross-domain image leakage, duplicate identities, invalid
domains, or more than 15% uncaptioned rows by default. It also reports
cross-domain caption reuse and any source contributing more than 80% of a
domain.

`homogeneous_domain_batches()` implements the initial 45% photo, 45%
animation, and 10% general mixture. Every microbatch has exactly one hard route
and samples sources round-robin within that domain.

## Stage 2 expert contract

`inject_appearance_experts()` wraps `img_mlp` in blocks 4–11 by default. It
derives a tensor-core-aligned hidden width of 12,544 from an explicit target of
15% of the released backbone per domain.

The wrapper computes:

```text
shared image FFN(input) + active-domain residual expert(input)
```

The result remains under the released block's existing adaptive image-FFN
gate. Attention, text FFNs, normalization, and output projection are untouched.
The residual expert's output projection is exactly zero-initialized, while its
scale starts at one. Therefore:

- `general` does not execute an expert;
- `photo` starts exactly equal to the released route;
- `animation` starts exactly equal to the released route;
- the output branch can begin learning on the first optimizer step.

Each expert contains 616,687,624 parameters, or 14.9836% of the backbone after
alignment. With both experts installed, 76.94% of the combined architecture is
the shared released backbone.

Use `freeze_for_expert_training()` before constructing the optimizer.
`controller.parameters()` then exposes only expert and expert-scale
parameters.

Photo and animation experts are persisted independently:

```python
controller = inject_appearance_experts(transformer)
freeze_for_expert_training(transformer, controller)

save_appearance_expert(
    controller,
    "photo",
    Path("mageflow-photo-expert.safetensors"),
)
```

Each file pins the base model and revision and contains no shared or
other-domain tensors. Strict loading reports every missing and unexpected key.

## Verification

Run the CPU suite:

```bash
PYTHONPATH=src pytest -q \
  tests/test_mage_flow_adaptation.py \
  tests/test_mage_flow_pretrain.py
```

The tests cover deterministic benchmark generation, checkpoint inspection,
caption semantics, exact duplicate handling, leakage audits, homogeneous
source balancing, zero-output equivalence, frozen parameters, and modular
expert checkpoint round trips. The injection seam has also been instantiated
against a small official `MageFlow` model using the pinned external source.

## Remaining GPU gates

Before expert training:

1. Materialize editing/reference fixtures and domain-specific VAE samples.
2. Run the official Base baseline and record output hashes and system metrics.
3. Run VAE reconstruction metrics after installing LPIPS and selecting an OCR
   engine in the isolated Mage environment; the other metrics are built in.
4. Verify packed Qwen3-VL and Mage-VAE layouts from the real checkpoint.
5. Run full-size zero-delta parity for all three routes.

Only then should the first 512-resolution expert-only run begin.

The corresponding trainer and launch contract are documented in
`docs/mage_flow_expert_training.md`.
