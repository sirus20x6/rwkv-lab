# MageFlow TREAD + Learned Factored Looping

This implementation extends the repository's custom terminal-expert model. It
does not modify the separate Mage source checkout.

## Executed architecture

The combined training preset executes:

1. Original MageFlow backbone blocks 0–1 on every packed token.
2. TREAD removes 50% of image tokens independently per packed sample.
3. Active image tokens and every text token execute original blocks 2–10.
4. TREAD restores bypassed tokens at their exact original packed positions.
5. Original backbone block 11 reconciles the complete sequence.
6. Exactly one resident photo or animation expert executes its three blocks.
7. The unchanged MageFlow output head emits one velocity prediction.

No original backbone block is replaced. TREAD only changes which image tokens
receive the middle blocks' computation.

## Learned recurrent depth

All 12 original backbone blocks and all three resident-expert blocks are
wrapped by independent zero-initialized loop controls. Each block always
executes its normal first pass. It may execute up to two additional refinement
passes for the default maximum depth of three.

For layer `l`, refinement pass `i`, and channel `c`, the effective gate is:

```text
gate[l,i,c] =
    head_weight[l,i,head(c)] * (1 + channel_delta[l,i,c])
```

The gate is softly bounded to `[-0.25, 0.25]`. Every head weight starts at
zero, so all added passes are exact no-ops at installation. The channel delta
starts at zero, giving the same coarse-to-fine learning behavior used by the
RWKV experiments: head factors learn first, then channel deltas specialize
their contribution.

Each layer also has:

- a learned loop-index conditioning offset;
- per-pass, per-token/head PonderNet halting probabilities;
- a KL-to-geometric ponder loss encouraging useful shorter depth.

The maximum loop count is configured; the effective loop rate is learned
independently by every layer, head, and channel. The 24 attention heads and
MLP expansion channels are arranged into four executable factors. Adjacent
active factors are coalesced into one GEMM/attention call; inactive factors do
not execute their QKV, attention, or MLP slices. With all factors active, the
sliced contributions reconstruct the original dense block numerically.

Soft learned gates execute every factor during training so each factor receives
gradients. At inference, each factor whose largest effective gate remains below
`1e-4` is omitted independently. A whole refinement is omitted only when all
four factors are inactive.

## Flow invariants

The noisy latent, prompt conditioning, and rectified-flow timestep remain fixed
through every internal refinement. Internal passes update only hidden states.
Only the final output advances the external flow solver.

Text tokens are never removed by TREAD. Bypassed image tokens preserve their
sample, packed position, RoPE association, and identity-gradient route.

## Configuration

The low-level `TreadLoopConfig()` is baseline-off. With no controller
installed, the custom terminal-expert model remains unchanged. Training uses
`TreadLoopConfig.combined_training_preset()`, enabling token-only TREAD and
learned loops on both backbone and selected expert.

The checked-in preset is
`experiments/mageflow_tread_factored_looping.json`.

## Training scope

- Selected domain expert blocks and their loop controls: full expert learning
  rate.
- Backbone learned-loop controls and the configured final backbone fraction:
  half expert learning rate.
- VAE-REPA projection: configured REPA rate.

The corrected terminal trainer takes its primary VAE-REPA representation after
backbone block 1, before TREAD extraction. It uses the deterministic cached VAE
posterior mean as the target while retaining posterior sampling for the flow
path. Student tokens are RMS-normalized before projection; Smooth-L1 is reduced
per image, smoothly capped per example, and then averaged so native resolution
does not change example weight. REPA gradients are excluded from learned loop
gates structurally because the tap precedes the recurrent route. The dashboard
log includes raw/capped REPA loss and hidden, projected, and target activation
ranges.

The LightningDiT-compatible continuation samples flow times with a
logit-normal distribution. It also adds channel-wise velocity cosine error,
averaged independently over each packed native-resolution image. That
directional term uses the same normalized flow Min-SNR example weights as the
primary velocity MSE and is enabled conservatively at weight 0.1. Training and
evaluation logs expose raw and weighted direction loss plus sampled timestep
range and mean.

The subsequent Lightning-block migration converts all 12 backbone blocks and
all three resident expert blocks together. Each image/text GELU MLP becomes a
parameter-matched SwiGLU with a two-thirds hidden width, including the
physically sliced factored-loop execution path. Four affine-free LayerNorms per
block and the final output normalization become learnable FP32-statistics
RMSNorms. Legacy expert and shared-backbone checkpoints are translated
explicitly: the strongest dense MLP neurons seed the SwiGLU gate/down
projections, the value branch starts as a constant-one path, and new RMS scales
start at one. The architecture migration intentionally resets Adam moments
while preserving the scheduler position and all model, REPA, and learned-loop
weights.

Every improving `eval/primary_loss` checkpoint is hardlinked into `best/` before
rolling retention. The atomic `best/best.json` manifest identifies the retained
checkpoint and its evaluation loss.
- Earlier original backbone weights, input/output heads, VAE, and text encoder:
  frozen.
- Inactive domain expert: separate checkpoint and not GPU-resident.

Expert checkpoints include their domain-specific learned-loop controls.
Shared-backbone checkpoints include backbone loop controls and trainable
backbone weights.

## Alternating expert residency

The original deterministic 500–1,000-step residency schedule established the
fresh-epoch domain ratio. The current balanced run uses strict optimizer-step
alternation instead:

- counts the frozen manifest by domain;
- alternates photo and animation after every complete optimizer update;
- keeps all accumulation microbatches for one update on the same route;
- starts a fresh deterministic data epoch for the new architecture.

For the 97,819-row continuation manifest, the ratio is 56,527 animation to
41,292 photo images (57.787% / 42.213%). With microbatch 1 and eight-way
gradient accumulation, one fresh pass is 12,228 optimizer steps. The checked-in
schedule assigns 7,067 animation and 5,161 photo steps across 18 windows.

Both experts' FP32 master weights and Adam moments remain CUDA-resident, while
one BF16 execution slot is resident in the model. At a boundary, same-device
tensor ownership is exchanged and the incoming FP32 master is copied into that
slot. Shared-backbone and REPA optimizer state is never swapped or reset. Only
one BF16 dense expert execution copy remains GPU-resident.

The exact schedule is
`experiments/mageflow_terminal_alternating_epoch_schedule.json`.

## Initialization and metrics

Unlike the discarded block-diagonal replacement, this architecture preserves
all trained dense blocks. Zero loop gates make the looped model bit-identical
to the existing terminal-expert model when TREAD is disabled.

At released dimensions, the complete MageFlow base has 4,115,745,408
parameters, one resident dense expert has 1,019,493,888, and all 15 learned
loop adapters together add only 2,397,600 parameters. The active
backbone-plus-one-expert model is therefore 5,137,636,896 parameters.

Logged metrics include active/bypassed image tokens, all 12 executed backbone
blocks, selected expert depth, maximum and expected learned loop depth, gate
RMS, Ponder loss, loop auxiliary loss, CUDA path time, optimizer time,
examples per second, peak VRAM, primary flow loss, REPA, Immiscible assignment,
flow-compatible loss weighting, velocity-direction error, and timestep
sampling statistics.
