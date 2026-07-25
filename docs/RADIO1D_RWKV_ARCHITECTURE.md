# RADIO1D-H + RWKV-7 2.9B Vision Architecture

**Status:** authoritative implementation contract, 2026-07-16  
**Vision encoder:** `nvidia/C-RADIOv4-1D-H`  
**Text model:** `rwkv7-g1h-2.9b-20260710-ctx10240.pth`

## Decision

Use RADIO1D-H directly as the frozen vision foundation model and align its
ordered global tokens to the frozen RWKV-7 2.9B language model. RADIO1D-H has
already agglomerated the complementary supervision of SigLIP2-g, DINOv3-7B,
and SAM3. Its global output width is 2,560, exactly matching RWKV's embedding
width, so there is no width-changing projector and no learned token
resampler.

MoonViT is not concatenated by default. It can return later as a measured
teacher or ablation only if it adds information that RADIO1D-H cannot recover.

## Non-negotiable visual contract

1. Each 512 x 512 tile requests all **256 RADIO1D global tokens**.
2. Those 256 tokens are never pooled, merged, or shortened before RWKV.
3. A multi-tile image includes one letterboxed global thumbnail followed by
   detail tiles.
4. Tiling is driven by source-image pixel area and aspect ratio. The default
   ceiling is **48 detail tiles plus one thumbnail**, or 12,544 visual tokens.
5. RWKV's 10,240 training context is a monitoring reference, not a hard
   truncation boundary. Examples above it are retained and report their
   estimated total sequence length.
6. Small images are not duplicated merely to consume a budget. A source with
   one tile's worth of pixels uses one full-image tile.
7. The cache records normalized source boxes, tile role, grid coordinates,
   exact encoder revision, preprocessing fingerprint, and source hash.
8. Variable-length visual prefixes are padded only at batch assembly. Padding
   positions must be masked and never enter a loss.

At the default maximum the visual sequence is:

```text
1 thumbnail + 48 details = 49 tiles
49 * 256 = 12,544 visual tokens
```

Ordinary images use less. For example, a 1920 x 1080 image naturally calls for
about eight detail tiles plus the thumbnail (2,304 visual tokens), while a 4K
image calls for about 28-32 details plus the thumbnail (7,424-8,448 visual
tokens), depending on the aspect-preserving grid.

## Tiling policy

- Tile canvas: 512 x 512, aspect-preserving letterbox; never stretch pixels.
- Detail-grid selection: search integer grids up to the detail ceiling and
  jointly minimize aspect-ratio error and distance from
  `ceil(source_pixels / 512^2)`.
- Detail crops: cover the complete source in row-major order with 12.5%
  overlap, including image edges.
- Thumbnail: the complete image, first in the sequence, for every multi-detail
  example. A one-tile image uses only the complete image once.
- Metadata: normalized half-open `(x0, y0, x1, y1)`, role, tile index, grid row
  and column, and selected grid dimensions.

The ceiling is a configuration knob, not a representational limit. Very large
images may use a larger ceiling when memory and batching allow it.

## RADIO output contract

For every tile, retain `global_tokens` from RADIO1D-H:

```text
[tile, 256, 2560]
```

Do not cache its ten internal prefix tokens (four class tokens and six
registers), decoder output, a mean pool, or the older RADIO spatial sequence as
the RWKV input. Cache BF16 global tokens and a validity mask. The cache key must
include the checkpoint revision, requested token count, input-conditioning
configuration, tile geometry, source image hash, and code schema.

## RWKV alignment bridge

The bridge preserves both width and token count:

```text
z = LayerNorm(radio_token)
z = z + tanh(gate) * Up(GELU(Down(z)))
z += role + tile_index + within_tile_index + Fourier(source_box)
```

- `Down/Up` is a rank-256 residual adapter, initialized as an exact no-op.
- The gate is initialized to zero.
- Learned embeddings identify thumbnail/detail role, tile order, and each of
  the 256 ordered RADIO token ranks.
- A small Fourier box encoder supplies global position and scale for each tile.
- No module maps 2,560 to another width or maps 256 tokens to fewer tokens.

Initial training freezes RADIO1D-H and RWKV and trains only this bridge plus
task/grounding heads. The complete aligned prefix is also re-injected at RWKV
layers **8, 16, and 24** through independent rank-256, zero-initialized residual
adapters. Reinjection changes only visual-token states; later caption tokens
consume the refreshed recurrent state causally. It never pools the prefix and
never injects padded visual positions. Later phases may unfreeze the final
RADIO blocks and selected RWKV layers only after the frozen alignment beats the
MoonViT baseline on caption, OCR, grounding, and hallucination evaluations.

## Batching and context

Batch examples by visual-token buckets, not image count. Concatenate each
example's valid visual prefix and pad to the largest visual length in that
microbatch. The trainer reports:

- detail tiles, total tiles, and visual tokens per example;
- visual padding fraction;
- text tokens and total effective sequence length;
- fraction above 10,240, without truncating those examples;
- tokens/second and source pixels/second.

Caption targets must never be shortened just to fit the old training context.
Memory pressure is handled with length bucketing, microbatching, gradient
accumulation, activation checkpointing, and RWKV's linear recurrent execution.
Training microbatches use one exact tile count per bucket: padding in the middle
of a recurrent sequence is not assumed to be a state no-op.

On the RTX PRO 6000 Blackwell qualification machine, the pruned 1.140B encoder
occupies 2.12 GiB of BF16 weights. Warm 512px throughput rose from 75 tiles/s at
batch 1 to 173 tiles/s at batch 8, with only 2.41 GiB peak allocation. The cache
builder therefore defaults to eight tiles per encoder forward.

## Qualification gates

Before a long run:

1. Tiler tests prove full-image coverage, correct ordering, no stretching, and
   deterministic grids.
2. Encoder smoke test proves exactly `[tiles, 256, 2560]` finite BF16 output.
3. Cache round-trip proves source/revision/preprocessing fingerprints and exact
   resume.
4. Bridge test proves token count preservation and zeroed padding.
5. A 32-example overfit test proves the bridge learns while both foundations
   stay bitwise frozen.
6. Side-by-side evaluation compares this stack with the best prior MoonViT
   checkpoint using identical images and decoding.

## Explicitly forbidden regressions

- The frozen 54M multi-teacher compressor.
- A fixed 64- or 128-token canonical head.
- Mean-pooling RADIO tokens.
- Treating 10,240 as an automatic truncation boundary.
- Stretching non-square source images into square tiles.
- Selecting checkpoints only by teacher-forced caption perplexity.
