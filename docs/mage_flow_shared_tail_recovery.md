# Mage-Flow shared-tail recovery stage

This stage adapts the released Mage-Flow backbone around the photo and
animation experts already learned at checkpoint 29,012. It is a continuation,
not a new expert initialization.

## Routing contract

Every training, scalar-evaluation, and generated-gallery sample is hard routed
through exactly one appearance expert:

- photo sample → photo expert
- animation sample → animation expert

The neutral/general bypass is not part of this stage's evaluation or deployment
contract. Expert interpolation is also disabled.

## Trainable scope

Mage-Flow-Base has 12 transformer blocks. The recovery stage uses:

| Component | State |
|---|---|
| Mage VAE | frozen |
| Qwen text encoder | frozen |
| Photo expert, blocks 4–11 | frozen at checkpoint 29,012 |
| Animation expert, blocks 4–11 | frozen at checkpoint 29,012 |
| Original Mage-Flow blocks 0–7 | frozen |
| Original Mage-Flow blocks 8–11 | trainable |

Only original, non-expert parameters in blocks 8–11 enter the optimizer. The
learned experts are loaded unchanged and are never zeroed or identity
initialized.

## Continuation semantics

Parent:

`runs/mage_flow_midjourney_v6_continuation_30pct_mixedres/checkpoint-00029012`

The continuation restores:

- both expert weight sets;
- global step 29,012;
- epoch 0 and batch position 4,096;
- RNG state.

It deliberately creates a fresh optimizer and cosine schedule because the
trainable parameter set changed. The first recovery window is 2,000 updates at
`1e-6`, with checkpoints and unified photo/animation evaluation every 250
updates.

## Advancement rule

Do not unfreeze earlier backbone blocks on a fixed step count alone. Inspect the
paired held-out target/generated galleries at each 250-step checkpoint. Expand
the trainable shared region only after both routed domains consistently produce
coherent composition and recognizable objects without a severe loss spike.

The appearance experts remain frozen until an explicit operator decision.
When they are unfrozen, use separate optimizer parameter groups:

- photo and animation experts: `1.0×` the stage learning rate;
- original Mage-Flow blocks 8–11: `0.5×` the expert learning rate;
- all other original backbone blocks: frozen until separately approved.

Unfreezing the experts must not reset them, change the hard routing contract,
or discard the shared-tail optimizer/checkpoint lineage.
