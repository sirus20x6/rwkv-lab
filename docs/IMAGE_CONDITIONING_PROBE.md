# Proving a caption model reads its images

## Why a caption loss is not evidence

A caption-distillation run reports loss against a reference caption. That number
stays entirely plausible when the vision tower is broken, because a language
model good enough to distil is also good enough to produce a fluent caption from
the prompt, the length prior, and the token statistics of the target.

That is not hypothetical here. The Qwen3.6 caption audit found that
`Qwen3.6-35B-A3B-heretic` nests its ViT at `model.language_model.visual.*` while
`Qwen3_5MoeForConditionalGeneration` looks for `model.visual.*` and declares no
`_checkpoint_conversion_mapping`. All 333 pretrained vision tensors therefore
loaded as *missing* and were freshly initialised, the load still reported
success, and `qwen36-caption-distill-lora-r256-v2` trained a rank-256 LoRA on
top of a random image encoder with an unremarkable loss curve.

Two changes closed the hole in the loader: PR #95 (`required_tensor_families`
and `ignorable_unexpected_prefixes`, fail-closed by family name) and PR #96 (the
declared `model.language_model.visual.=>model.visual.` remap). Both are gates on
*names*. Neither can tell you that the tensors which arrived are the ones that
matter, or that the pixels they encode reach the caption. This document covers
the measurement that can.

## The trap: sensitivity is not comprehension

The obvious test — feed different images, check the outputs differ — does not
detect the bug it appears to test for.

A randomly initialised frozen ViT is still a *function of its input*. Perturb
the image and its outputs move, so a probe that only asked "did the output
change?" passes a completely broken model. The v2 run would have passed such a
probe on every example.

So `rwkv_lab.image_conditioning_probe` reports two independent quantities and
refuses to collapse them:

| Metric | Question | Fails when |
| --- | --- | --- |
| `responsiveness` | Do pixels reach the computation at all? | The vision path is disconnected, images are dropped, or image tokens never enter the sequence. |
| `discrimination` | Does the model read pixels *for content*? | The encoder carries no semantics — the random-ViT case. |

`discrimination` is the load-bearing one. It scores each image against its own
caption and against other images' captions, and asks whether the model prefers
its own. Chance is zero margin; a working encoder is positive. This is the same
signal `vision_train.py` already computes as `ocr_image_conditioning_nll_delta`
via `rotate_batch`, generalised and given a verdict.

A third quantity, `repeat_determinism`, is a control rather than a metric. If
identical pixels do not reproduce an identical score, every other number is
noise, so the verdict short-circuits and says only that.

## Image variants

| Variant | What it is | What it buys |
| --- | --- | --- |
| `original` | the dataset's image | reference condition |
| `repeated` | the same pixels again | determinism control; **not** a perturbation |
| `shuffled` | the image's own tiles under a seeded permutation | identical colour histogram, destroyed layout — so "photographs are beige" earns no credit |
| `blank` | uniform mid-grey, same size | the furthest valid image from the original |

## Running it

The decision logic is pure and has no torch dependency, so it is tested on CPU
against stubs — including a pixel-blind stub and a *pixel-responsive but
semantically blind* stub, both of which the suite requires to FAIL. That second
one is the random-ViT case, and it is why the probe cannot be satisfied by
sensitivity alone.

```bash
python scripts/probe_image_conditioning.py \
  --model-dir <checkpoint> \
  --dataset <caption split>.jsonl \
  --output docs/evidence/image-conditioning-<name>.json
```

The script measures the same loaded model twice:

- `pretrained-vision` — the checkpoint as it loads, remap applied.
- `randomized-vision` — the identical model with its vision tower re-initialised
  in place, reproducing the audited defect exactly.

Reporting only the first would leave open the possibility that the probe passes
everything. The pair is the evidence: the first must pass and the second must
fail. Re-initialising in place rather than loading a second 70 GiB copy keeps
the control cheap enough that there is no excuse for omitting it.

The script refuses to start when the target device has less free VRAM than the
declared floor, because this host also runs training. It never writes to the
checkpoint directory.

## Measured result

See `docs/evidence/image-conditioning-qwen36-caption.json` for the full record,
including per-example variant scores.

It lives under `docs/` rather than `/evidence/`, which #101 gitignored. The rule
there is right and this is not an exception to it: `/evidence/` holds receipts
**about this repository at a commit**, and a committed one goes stale the moment
the next commit lands, making a confident wrong claim. This file is a
measurement of an **external model artifact**. It does not describe the repo, so
it does not decay with it.

That only holds if the artifact is identified by something better than a path,
because `...step000745-merged-bf16` names a *merged* checkpoint — the kind that
gets regenerated in place. So the document binds itself to what it measured:

- `model_content_identity` in three layers — the weight map's digest, a digest
  over every shard's safetensors header (tensor names, dtypes, shapes, offsets),
  and per-shard content digests lifted from the publisher's `merge-receipt.json`.
  Each records what it covers; the third is the only one that catches weights
  edited in place without a shape change, and it says so when unavailable rather
  than going quiet.
- `generated_at`, and the `repository` commit the probe ran from with a
  `dirty_worktree` flag.
- `runtime`: torch, transformers, python, CUDA, and the device name.
- `shuffle_seed` at top level, so the perturbation is reproducible across runs
  and not merely deterministic within one.

Re-running the probe at a later commit reproduced every figure below to four
decimal places, so the numbers are a property of the model rather than of one
process.

Subject: `Qwen3.6-35B-A3B-heretic-caption-step000745-merged-bf16` — the merged
corrected-v3 caption model. Five validation images (deterministic: the frozen
`validation-fixed-100.jsonl` split sorted by content hash, first five), captions
truncated to 60 words, `max_pixels` 200704, bf16 on one idle device. The load
was exact: 1026 tensors, 0 missing, 0 unexpected, remap applied.

| | `pretrained-vision` | `randomized-vision` (control) |
| --- | --- | --- |
| own-caption margin | **+1.1243 nats** | +0.0262 nats |
| own-caption accuracy | 5/5 | 4/5 |
| weakest perturbation response | 0.3538 nats | 0.0083 nats |
| repeat determinism drift | 0.0 | 0.0 |
| **verdict** | **passes** | **fails, as required** |

The margin separates by a factor of 43. On the first example the pretrained
model scores its own caption at 1.5329 nats against 2.3143 for other images'
captions; blanking the image costs it 0.45 nats and shuffling the tiles 0.35.
The same model with a random ViT scores 2.0358 against 2.1350 — and blanking the
image moves it by 0.038 nats, which is to say almost not at all.

Two details are worth keeping.

The control still gets 4/5 own-caption preferences. Accuracy alone would have
looked like a near-pass; only the margin exposes it. That is why the verdict
requires both, and it is the concrete reason not to simplify this to one number.

The control also failed *responsiveness*, not just discrimination. A random
frozen ViT turns out to pass so little signal that the caption likelihood barely
moves when the image is replaced with grey. So the earlier concern that a random
encoder might still leak useful image information does not survive measurement
here: at this scale it leaks almost none.

Reproduce with `pytest -m gpu tests/test_image_conditioning_probe.py`, which
skips with a reason when the device, the checkpoint, or the free VRAM is
missing. CI never runs it — the `gpu` job is gated on the unset
`TRAINVM_SELF_HOSTED` repository variable — so the committed evidence file is
the record.

## Disposition

**The shipped corrected-v3 caption adapter is valid as merged. No retraining,
and no re-evaluation under a different loader.**

The measurement above is on the merged artifact itself, so this is a statement
about the thing in use rather than an inference from the loader's behaviour.

The history is worth stating plainly, because the disposition on this changed
twice and the reasoning matters more than the conclusion:

- `qwen36-caption-distill-lora-r256-v2` **did** train against a randomly
  initialised ViT. Any number recorded against that run — including its
  `baseline-validation-step-000000.json` loss of 2.1979 — describes a model with
  a random vision tower and is not comparable across the fix.
- `qwen36-caption-distill-lora-r256-corrected-v3` is the retrain. Its
  `base-load-receipt.json` attests 0 missing, 0 unexpected, `exact: true`, 333
  vision tensors, `vision_namespace: "legacy-remapped"`. An intermediate
  RETRAIN disposition recorded against v2 was retracted once that was found: the
  retrain had already happened.
- The evidence here closes the remaining gap. The attestation proved the right
  *tensors* loaded; this proves the merged model *uses* them.

One operational note that outlives this card. The merged model re-publishes its
ViT at `model.language_model.visual.*`, the same legacy nesting as the base. So
anything loading the merged caption model through the declarative
`hf_multimodal_sft` path needs the PR #96 remap too, exactly as the base does.
The probe script derives the remap from the sealed index rather than hardcoding
it, which is why it loaded the merged model exactly; a configuration that omits
the remap will now fail closed on `required_tensor_families` rather than
silently randomising 333 tensors again.
