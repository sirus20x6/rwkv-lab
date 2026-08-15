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

The focused architecture suite is covered by `tests/test_prototype_suite.py`
and the imported model-specific tests.
