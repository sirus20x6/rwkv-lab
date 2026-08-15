# Unlabeled Local Image Deduplication

The local private corpus at `/workspace/datasets/private_web` is an image-and-video
tree. The first vision-distillation pass intentionally considers still-image
files only. Videos and PDFs are ignored rather than decoded or frame-sampled.

The pipeline has four resumable stages backed by one SQLite database:

1. Inventory image paths and metadata without decoding images.
2. Hash only byte-size collision groups to remove exact copies efficiently.
3. Decode remaining images, calculate 256-bit perceptual and color hashes, and
   cluster every committed batch immediately.
4. Publish atomic `*.partial.jsonl` snapshots during hashing, then keep the
   highest-resolution representative from each conservative near-duplicate
   group and atomically export the final JSONL manifest.

Paths with uncertain-age indicators are excluded during inventory and are not
opened by the hashing phases. Images whose shortest side is below 256 pixels,
corrupt images, videos, and PDFs are also excluded from the output manifest.

## Full run

The corpus, database, and manifest paths have project-specific defaults, so a
complete resumable run can be started from the repository root with:

```bash
.venv/bin/python scripts/build_unlabeled_image_manifest.py
```

The default hash phase is incremental and crash-safe. It assigns exact and
perceptual duplicates as each batch commits, resumes clustering hashes written
by older interrupted runs, and refreshes
`curated_vision/local_private_web_unlabeled_dedup.partial.jsonl` every 100,000 newly
clustered images. Change that cadence with `--publish-every`; use
`--no-incremental` only when explicitly deferring all clustering. The partial
filename is intentional: cache automation treats the canonical `.jsonl` as a
completed, deterministically normalized manifest.

Run individual phases so expensive work can be scheduled independently:

```bash
PYTHONPATH=src .venv/bin/python scripts/build_unlabeled_image_manifest.py \
  --root /workspace/datasets/private_web \
  --db /workspace/downloads/cache/moe-mla/local_private_web_image_dedup.sqlite \
  --manifest curated_vision/local_private_web_unlabeled_dedup.jsonl \
  --phase inventory --workers 8

PYTHONPATH=src .venv/bin/python scripts/build_unlabeled_image_manifest.py \
  --root /workspace/datasets/private_web \
  --db /workspace/downloads/cache/moe-mla/local_private_web_image_dedup.sqlite \
  --manifest curated_vision/local_private_web_unlabeled_dedup.jsonl \
  --phase exact --workers 8

PYTHONPATH=src .venv/bin/python scripts/build_unlabeled_image_manifest.py \
  --root /workspace/datasets/private_web \
  --db /workspace/downloads/cache/moe-mla/local_private_web_image_dedup.sqlite \
  --manifest curated_vision/local_private_web_unlabeled_dedup.jsonl \
  --phase hash --workers 8

PYTHONPATH=src .venv/bin/python scripts/build_unlabeled_image_manifest.py \
  --root /workspace/datasets/private_web \
  --db /workspace/downloads/cache/moe-mla/local_private_web_image_dedup.sqlite \
  --manifest curated_vision/local_private_web_unlabeled_dedup.jsonl \
  --phase cluster --phash-distance 4 --min-side 256

PYTHONPATH=src .venv/bin/python scripts/build_unlabeled_image_manifest.py \
  --root /workspace/datasets/private_web \
  --db /workspace/downloads/cache/moe-mla/local_private_web_image_dedup.sqlite \
  --manifest curated_vision/local_private_web_unlabeled_dedup.jsonl \
  --phase export
```

The perceptual threshold defaults to four differing bits out of 256 and also
requires a close color hash. This is deliberately conservative: it removes
resized or recompressed copies without treating visually related photographs
from the same shoot as duplicates.

The exported rows contain absolute image paths and image metadata but no text
caption. They are inputs for frozen-teacher feature production and latent
distillation, not caption cross-entropy targets.

## Videos later

Video should use a separate scene-aware sampler. It should detect shot changes,
select a small number of sharp keyframes per shot, and then pass those keyframes
through the same image deduplicator. Fixed-interval frame extraction would
overweight long clips with thousands of nearly identical frames.
