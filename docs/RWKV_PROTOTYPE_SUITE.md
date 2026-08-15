# Declarative RWKV prototype suite

This suite exposes four experimental architecture families through the normal
declarative configuration entry point:

- RWKV-8 with DeepEmbed gating;
- RWKV-7 with entropy-driven dynamic BLT ChannelMix;
- RWKV-8 with KAN spline ChannelMix;
- ROSA+BLT and latent-thinking proof components.

The initial implementations and proof tests were developed in the public
`CodeDoes/rwkv-lab` fork. They were reconstructed onto the current upstream
history, audited, and integrated with the shared configuration runner. The
integration deliberately excludes generated dependency locks and unrelated
content from the fork.

## Run the smoke suite

With an editable install:

```bash
python -m rwkv_lab.config run experiments/rwkv_prototype_suite.yaml
```

From a source checkout without installation:

```bash
PYTHONPATH=src python -m rwkv_lab.config run experiments/rwkv_prototype_suite.yaml
```

The checked-in suite runs deterministic CPU-reference probes. Each task records
parameter count, output shape, latency, and logit RMS. The combined result is
written to `runs/rwkv-prototype-suite/summary.json`.

## Configuration

Every task requires a unique filesystem-safe `name` and a `kind`:

```yaml
name: local architecture study
prototype:
  output_dir: runs/local-architecture-study
  tasks:
    - name: kan-rwkv
      kind: model_probe
      architecture: kan_rwkv
      model:
        vocab_size: 256
        d_model: 32
        n_layers: 2
        head_size: 16
        grid_size: 5
      batch: 2
      sequence_length: 32
```

Supported `model_probe` architectures are `rwkv8`, `kan_rwkv`, `blt_rwkv7`,
and `rosa_blt`. RWKV time-mix initialization requires at least two layers, and
the model width must be divisible by the head size.

Two heavier task kinds expose the proof pipelines:

- `blt_comparison` forwards its `args` to
  `run_comparative_validation()` and produces a per-task JSON report.
- `standard_benchmark` forwards its `args` to `run_benchmark()` and compares
  RWKV-8, ROSA+BLT, and latent thinking under one workload.

Both accept scale controls such as `epochs`, `train_rows`, `sequence_length`,
model dimensions, and benchmark iteration counts. Use small workloads for CI;
performance claims should use a separately recorded full-scale configuration.

## Audit corrections

Upstream integration fixed three issues discovered by running the forked code
against the current core:

- run-length compression now advances after encoding a compressed run instead
  of looping forever;
- declarative validation rejects invalid one-layer RWKV prototypes before
  construction;
- CPU probes explicitly select the deterministic Python reference kernel, so
  an installed FLA/Triton stack cannot receive CPU tensors accidentally.
- RWKV-8 and KAN-RWKV DeepEmbed gates now use a trainable identity
  initialization; the model-wide initializer previously randomized the output
  gate, while zeroing both low-rank factors would have prevented learning.

The focused architecture suite is covered by `tests/test_prototype_suite.py`
and the imported model-specific tests.

## Real-corpus pretraining

The architectures are also first-class `rwkv_pretrain` models. They use the
same corpus sampler, fixed step-0 validation, optimizer stack, learning-rate
schedules, gradient accumulation, checkpoints, exact resume state, JSONL
telemetry, campaign registry, and dashboard ingestion as the baseline model.

Use the checked-in equal-budget comparison:

```bash
python -m rwkv_lab.config run experiments/rwkv_architecture_pretrain.example.yaml
```

That configuration builds one document-aware UTF-8 byte corpus and compares
`rwkv7`, `rwkv8`, `blt_rwkv7`, `rosa_blt`, and `kan_rwkv` on identical data,
seeds, model dimensions, and budgets. Generated corpora, checkpoints, and run
artifacts remain ignored under `models/cache/` and `runs/`.

The relevant model fields are:

```yaml
model:
  architecture: rwkv8       # rwkv7 | rwkv8 | blt_rwkv7 | rosa_blt | kan_rwkv
  vocab_size: 256
  d_model: 128
  n_layers: 4
  head_size: 32
```

BLT models must use `data.encoding: bytes` and `vocab_size: 256`. Their entropy
heads are optimized with `blt_entropy_weight`; training and evaluation records
also report entropy loss, mean predicted entropy, and average dynamic patch
length. RWKV-8 and KAN-RWKV may instead use the ordinary World-token corpus and
65,536-token vocabulary.

Checkpoints record the architecture, vocabulary, dimensions, DeepEmbed shape,
BLT patch controls, KAN spline controls, and ROSA memory size. Resume refuses a
topology mismatch before loading weights. Resumed runs append to `train.jsonl`
so dashboard history is preserved rather than replaced.

The imported architectures initially exclude levers whose semantics have not
been qualified on their blocks, including recurrent-depth wrappers, Engram,
state-offset tuning, online memory, and routing-free MoE. Unsupported
combinations fail during argument validation instead of silently degrading to
the baseline.

Declarative campaign trials are written as direct children of `runs/` using
`campaign-ID--arm--seed-NNNN` names. This matches trainboard's ingestion
contract, so each arm has an ordinary live loss/throughput graph in addition to
its campaign-registry comparison.
