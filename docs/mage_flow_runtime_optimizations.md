# Mage-Flow runtime optimization branch

This branch accelerates the existing routed Mage-Flow objective without
introducing a representation-alignment loss. REPA/VAE-REPA work is deliberately
separate so convergence changes can be compared with runtime changes.

All new behavior is opt-in. Existing configurations retain FlashAttention-2,
full-block activation checkpointing, eager DiT blocks, BF16 linears, and live
Qwen/VAE encoding.

The same fields are wired into both the legacy routed-continuation trainer and
the exclusive single-resident terminal-expert trainer. Regional compilation
includes the 12 backbone blocks and the three terminal-expert blocks.

## Qualified environment

The isolated environment is:

```text
/workspace/git/moe-mla/.venv-mage-flow-fa4
```

It is a clone of `.venv-mage-flow` with:

```text
flash-attn-4==4.0.0b24
nvidia-cutlass-dsl==4.6.0.dev0
torchao==0.17.0
```

CUDA 13.3, PyTorch 2.13.0+cu130, and the RTX PRO 6000 Blackwell (SM120)
passed:

- FA4 BF16 forward and backward with 128-dimensional heads;
- finite Q/K/V gradients;
- TorchAO tensorwise FP8 linear forward and backward;
- FP32 FP8 master parameters under BF16 autocast;
- compiled FP8 execution through TorchInductor.

Transformer Engine 2.17's `MXFP8BlockScaling` was also tested, but its runtime
currently rejects compute capability 12.0+ for the required GEMM layouts.
MXFP8 is therefore not installed or exposed as a training option. This is a
dependency capability gap, not a silent fallback to ordinary FP8.

The original `.venv-mage-flow` was not modified.

## Configuration

The next run can enable the runtime stack with:

```json
{
  "attention_backend": "flash2",
  "activation_checkpointing_mode": "none",
  "compile_transformer_blocks": false,
  "compile_transformer_mode": "default",
  "compile_transformer_dynamic": false,
  "float8_training": false,
  "float8_recipe": "tensorwise",
  "encoder_cache_dir": "/workspace/git/moe-mla/caches/mage_flow_encoders",
  "encoder_cache_mode": "read_only",
  "offload_cached_encoders": true
}
```

The generated launcher automatically selects `.venv-mage-flow-fa4` whenever
FA4 or FP8 is enabled. Every selection and its resolved module/block allowlist
is written into `run_contract.json`.

The qualified terminal trainer uses checkpointable FP32 master parameters and
FP32 AdamW moments while leaving executable model weights in BF16. This keeps
FlashAttention inputs supported without rounding optimizer history through
BF16 on every update or resume.

### Activation checkpointing

Modes are:

| Mode | Behavior |
|---|---|
| `full` | Released behavior: checkpoint all 12 blocks |
| `trainable` | Checkpoint only blocks containing trainable parameters |
| `selective` | Trainable blocks only; retain GEMM results and recompute cheaper ops |
| `none` | Retain all activations |

Start the fixed-geometry benchmark with `none`. If it violates the VRAM
qualification limit, try `selective`, then `trainable`. Do not infer the winner
from allocated memory alone; record reserved memory and full update time.

### Regional compilation

Compilation is applied in place to each repeated transformer block after
checkpoint restoration and FP8 conversion. This preserves state-dict names
while allowing a small graph family for varying image and caption lengths.
First-use compilation is expected and must be excluded from steady-state
throughput measurements.

Regional compilation remains disabled for the current run. On the fixed
1024-square terminal-expert probe it was slower than eager execution, and
dynamic shapes are incompatible with Mage's packed-attention buffer
allocation. Any future compile experiment must use static-per-bucket graphs.

### FP8

TorchAO conversion is allowlisted to trainable image-FFN `Linear` modules with
dimensions divisible by 16. Attention, text FFNs, modulation, normalization,
input/output projections, the VAE, and Qwen remain BF16/FP32. FP8 requires
regional compilation.

FP8 is not enabled by default. Promotion requires a BF16 comparison covering
loss drift, prediction cosine similarity, gradient cosine similarity,
checkpoint reload, and held-out gallery output.

### Frozen-encoder cache

The cache stores:

- final packed Qwen hidden states for each exact conditioning string;
- the CFG null string as a separate entry;
- VAE posterior mean and log-variance for each image and training geometry.

It never caches a sampled latent. A new posterior sample is drawn on every
training visit, preserving the existing stochastic objective. Cache keys
include model revision, image path, file size, modification time, and geometry.
Entries use atomically published safetensors.

Build or resume the cache while the GPU is available:

```bash
MAGE_FLOW_VENV=/workspace/git/moe-mla/.venv-mage-flow-fa4 \
  /workspace/git/moe-mla/.venv-mage-flow-fa4/bin/python \
  -m rwkv_lab.mage_flow_expert_train cache-encoders \
  --config /absolute/path/to/train_config.json
```

Use `encoder_cache_mode: "read_write"` for the build. After
`cache_build_receipt.json` reports complete coverage, change it to
`"read_only"` and enable `offload_cached_encoders`. A read-only run refuses to
start on any missing text or VAE-moment entry.

## Benchmark order

Use one immutable checkpoint, one fixed 1024-equivalent probe, identical
samples/seeds, and at least ten post-warmup optimizer updates:

1. FA2 + full checkpointing + eager BF16 baseline.
2. FA4 only.
3. FA4 + checkpoint mode `none`.
4. If needed, compare `selective` and `trainable`.
5. Add regional compilation.
6. Compare FP8 separately against the best BF16 setting.
7. Add the complete read-only encoder cache and offload frozen encoders.

Do not combine a REPA objective in these measurements. Runtime improvements
and fewer-updates-to-quality are separate axes.

The July 30 qualification selected FA2 eager execution. FA4 and regional
compilation did not improve this workload. Encoder caching and disabling
activation checkpointing were the only individually positive runtime changes;
the hot single-entry result is an upper bound until repeated with a shuffled
production-shaped cache benchmark.
