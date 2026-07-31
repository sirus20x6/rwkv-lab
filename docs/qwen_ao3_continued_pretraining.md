# Qwen3.6 AO3 continued pretraining

This pipeline prepares the immutable AO3 top-five-percent selection and trains
the local `Qwen3.6-35B-A3B-heretic` checkpoint as a text-only causal language
model. Rejected text is never copied into logs.

## Why the trainer is hybrid QLoRA

The checkpoint contains about 64.6 GiB of BF16 language weights. Sixty GiB is
stored in fused three-dimensional MoE expert parameters rather than
`nn.Linear` modules, so the standard bitsandbytes four-bit loader does not
quantize it. Full AdamW training has a storage floor around 387 GiB before
activations and cannot fit on this 96 GiB GPU. FP8 autocast would accelerate
eligible matrix multiplies but normally retains high-precision weights and
optimizer state, so it does not solve that fit problem.

The configured run therefore uses:

- frozen BF16 fused experts with rank-4 per-expert rsLoRA;
- NF4 double-quantized ordinary language-model linears;
- rank-64 rsLoRA on DeltaNet, attention, shared-expert, embedding, and output
  projections;
- rank-8 rsLoRA on every router;
- BF16 compute, non-reentrant activation checkpointing, FlashAttention, FLA
  Gated DeltaNet kernels, fused vocabulary cross entropy, and PowerCool;
- a router load-balancing loss to avoid domain-driven expert collapse.

The expert adapters use a local grouped-GEMM implementation. PEFT's generic
3-D `target_parameters` path constructs a complete dense `B @ A` delta for
every expert bank on every forward; on this checkpoint that transient would be
roughly the size of the 60 GiB expert base. The local path applies both low-rank
factors directly to the routed token/expert groups, preserves exact zero-delta
parity with Transformers' native grouped implementation, and never
materializes a dense expert delta.

The static estimate is approximately 62.6 GiB resident base, 5.2 GiB adapter
training state, and 87.8 GiB peak with activation/runtime reserves. The trainer
refuses to start unless at least 92 GiB VRAM is free and verifies all 80 expert
base tensors and every expert/router adapter before its first update.

## Data preparation

The original directory remains untouched:

```text
/thearray/downloads/completed/ao3/ao3_filthiest_top5pct/
```

The historical selector checked only one AO3 warning spelling. The preparer
normalizes both `Archive Warning` and `Archive Warnings`, applies metadata and
bounded prose checks, emits deterministic split shards, and records only
hashed rejected IDs plus reason codes.

```bash
bash scripts/bootstrap_ao3_cpt.sh

.venv-ao3-cpt/bin/python scripts/prepare_ao3_cpt.py
bash scripts/prepare_qwen_ao3_cpt.sh
```

The second command tokenizes every accepted document without truncation,
stores `uint32` token streams and document offsets, and performs a seeded
exactly-once pack into 8,192-token rows. Each document already ends in the
model-config raw EOS (`<|endoftext|>`, ID 248044), rather than the chat
template's `<|im_end|>`. Only the final tail shorter than one context is omitted;
the trainer uses a smaller final accumulation when needed so every packed row
participates in the epoch.

## Training

The generated launcher lives in:

```text
runs/qwen36_ao3_cpt_r64_e4/launch.sh
```

Stop other GPU consumers first; the current llama.cpp server occupies roughly
45 GiB and intentionally causes the trainer's free-VRAM preflight to fail.
Then launch:

```bash
bash runs/qwen36_ao3_cpt_r64_e4/launch.sh
```

The first run performs a 512-token forward/backward qualification before the
baseline evaluation. Checkpoints contain PEFT adapter weights, optimizer/RNG
state, a separate grouped-expert adapter tensor file, the exact packed-row
cursor, metrics, and source manifest fingerprints. SIGINT or SIGTERM requests
a save after the current optimizer step. Relaunching the same `launch.sh`
automatically resumes `latest.json`; a completed run exits without loading the
model again.
