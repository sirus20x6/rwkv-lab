# trainboard — RWKV-Lab training dashboard v2.0

GPU-accelerated, real-time training dashboard for the RWKV-Lab conversion project.
**Stack:** Go + SQLite + Datastar + Pixi.js. Successor to `../dashboard/` (FastAPI + Chart.js).

## Run

```bash
go -C /thearray/git/moe-mla/dashboard run ./cmd/trainboard
# open http://127.0.0.1:9124
```

Reads `/thearray/git/moe-mla/runs/`. Ingests all `train.jsonl` logs + system telemetry into a local
SQLite DB (`trainboard.db`). GPU-light — safe to run alongside live training.
The one-second shared snapshot reads conversion quality from ingestion-time rollups plus one batched
codec query; it does not fan out KPI/count queries per layer. Run-sidecar and campaign/dataset discovery
are cached, so multiple panels and browser tabs do not repeatedly walk the 1TB-scale `runs/` tree.
The health detector likewise fetches every live run's bounded training window and latest rollup in
two batch queries per scan instead of resolving IDs, statistics, and PPL independently per process.

The experiment builder exposes trainer-native P0/P1 comparison arms for
[u-μP](https://arxiv.org/abs/2407.17465), [Titans](https://arxiv.org/abs/2501.00663) /
[MIRAS](https://arxiv.org/abs/2504.13173) / [ATLAS](https://arxiv.org/abs/2505.23735) /
[Nested Learning](https://arxiv.org/abs/2512.24695) online memory, and
[NVFP4](https://arxiv.org/abs/2509.25149) with optional
[TetraJet-v2](https://arxiv.org/abs/2510.27527) randomized Hadamard transforms. NVFP4 arms include
the fake-quant correctness oracle and a fail-closed native Blackwell Transformer Engine arm that must
pass parity and throughput qualification. These arms are scratch-LM-only; the builder rejects
u-μP+Muon and NVFP4+FP8 combinations. Its advanced systems controls also expose opt-in FSDP2
multi-process launch, activation checkpointing, CPU offload, learning-rate schedules, and the
parity-tested compiled [ARO-Sinkhorn](https://arxiv.org/abs/2602.09006) tensor path while preserving
the exact-resumable optimizer fallback.

The dedicated **verifiable-reward training** panel launches equal-budget, paired-seed campaigns and
reads their versioned `campaign.json` evidence directly from `runs/`. It compares the sequence-level
importance ratios from [GSPO](https://arxiv.org/abs/2507.18071), the unbiased group-relative objective
from [Dr.GRPO](https://arxiv.org/abs/2503.20783), and the asymmetric clipping/dynamic sampling ideas
from [DAPO](https://arxiv.org/abs/2503.14476). The launcher exposes
[DeepSeek-R1](https://arxiv.org/abs/2501.12948)-style cold-start SFT and curriculum/preflight controls,
[RWKV-7](https://arxiv.org/abs/2503.14456) recurrent batching with a semantics-safe fallback, paired
[bootstrap](https://doi.org/10.1214/aos/1176344552) confidence, task-family regression limits, and hard
rollout/time budgets. Results report parent and candidate held-out reward, across-seed dispersion,
RL/SFT updates, preflight passes, budget usage, and promotion eligibility. The same panel discovers
`loop.json` lineage from [Absolute Zero](https://arxiv.org/abs/2505.03335)-inspired bounded recursive
iterations. It launches model-side training only; proposal commands remain CLI-only and generated-code
verification remains outside this repository in Adamaton's sandboxed verifier process.

Advanced RLVR controls expose rollout/evaluation sampling, optimizer and warmup settings, reference
checkpoints, bootstrap group counts, SFT phase sizing, preflight thresholds, active-group safeguards,
and checkpoint/log cadence. Every value is validated by the server and translated into the
allowlisted campaign command; arbitrary verifier commands deliberately remain CLI-only.

The **post-training** panel validates the versioned SFT/preference/feedback/PRM/RLVR
JSONL contract (including structured tool calls), shows rendered variants and their trainable spans,
and compares two saved checkpoints
under an identical prompt, seed, temperature, and generation budget. It can merge repository-confined
sources into immutable content-addressed versions under `datasets/versions/`; duplicate content and
cross-split leakage are refused. An explicit human choice may be
appended to `datasets/trainboard_preferences.jsonl` as training data. Its allowlisted launcher runs
equal-token, paired exploration seeds and fresh confirmation seeds through
`rwkv_lab.posttrain_campaign`; result cards show phase-specific deltas/confidence intervals, promotion
eligibility, and adapter-first recursive lineage. It exposes qualified reset-mask packing, fail-closed
automatic TorchAO NF4 qualification (with an explicit portable correctness oracle), device slots,
per-attempt timeouts, and retries; campaign state resumes
completed command-identical arms after interruption. The panel cannot write hidden evaluation data, run
an Adamaton proposal command, promote/merge a checkpoint, or publish a model. The underlying
training methods are [LoRA](https://arxiv.org/abs/2106.09685),
[QLoRA](https://arxiv.org/abs/2305.14314),
[DPO](https://arxiv.org/abs/2305.18290), [KTO](https://arxiv.org/abs/2402.01306),
[ORPO](https://arxiv.org/abs/2403.07691), and [SimPO](https://arxiv.org/abs/2405.14734); paper links
are also kept next to the Python implementations. Process-reward training and calibration follow
[Let's Verify Step by Step](https://arxiv.org/abs/2305.20050); recursive adapter lineage follows the
bounded proposer/solver direction of [Absolute Zero](https://arxiv.org/abs/2505.03335) while keeping
promotion independent.

The **production qualification** panel launches the allowlisted kernel/serving qualification runner
and reads its versioned receipts from `runs/`. It exposes device, checkpoint/prompt, repetition and
generation budgets, plus baseline speed and memory regression limits. The latest receipt reports the
environment, available and adopted backends, parity, speedup, memory, and the baseline regression
gate. This is the UI counterpart to the fail-closed parity-before-speed policy used for native
[NVFP4](https://arxiv.org/abs/2509.25149), TorchAO NF4, and compiled serving paths; it does not install
a backend, promote a checkpoint, or publish an artifact. With a compatible checkpoint it also
compiles and qualifies the native RWKV [Megakernels](https://github.com/HazyResearch/Megakernels/tree/throughput)-
inspired backend: a fused Triton DPLR transition inside an Inductor-autotuned, CUDA-Graph-replayed
decode plan. The UI exposes fast compilation or full autotuning and reports exact-token parity,
warm speedup, CUDA launches before/after, cold compile cost, and whether the plan was adopted.

The **research capability inventory** exposes the readiness and entry point of the community-derived
reference paths: Recursal `balance_state`, rwkv-rlhf/OpenMOSE state adapters,
[AUXStar/RWKV-Server](https://github.com/AUXStar/RWKV-Server)-inspired state paging,
[stable triangular delta inversion](https://arxiv.org/abs/2605.21325),
[offline sleep consolidation](https://arxiv.org/abs/2605.26099),
[Reasoning Cache](https://arxiv.org/abs/2602.03773),
[B³D-RWKV](https://arxiv.org/abs/2605.25969), and
[HiLS-Attention](https://arxiv.org/abs/2607.02980). The panel labels portable oracles separately
from dashboard launchers and qualified production paths. It also inventories the deterministic
[decoding evaluation](https://arxiv.org/abs/2402.06925) matrix, launchable
[State-offset Tuning](https://arxiv.org/abs/2503.03499), routed state banks, byte-aware and
[SuperBPE](https://arxiv.org/abs/2503.13423) experiment boundaries, promotion-gated overnight
adapter consolidation, and typed allowlisted decoding policies. Community-only proposals are
explicitly labeled as such rather than presented as established paper results.

## Why v2

- Real SQLite datastore (v1 re-parsed JSONL on every request).
- Exposes **every** logged signal (v1 showed ~6 of ~20 fields): SMT/DMT conversion losses, guard
  saturation, d_prodigy, kl, per-horizon loss/ppl, skipped steps.
- Incremental Pixi append + Datastar signal streaming → smooth realtime (v1 re-rendered every 2 s).
- Interactive model-architecture strip, GPU telemetry history, compare overlay.
- Confirm-gated control actions: stop / checkpoint-now / launch / notes & tags.

See `CLAUDE.md` (instructions), `STACK.md` (API cheat-sheet), `DATA_MODEL.md` (schema + field catalog).

## Status

Built in phases (see `/home/sirus/.claude/plans/inherited-munching-anchor.md`):
1. scaffold ✅ · 2. DB+ingester · 3. sysmon+Datastar shell · 4. Pixi charts · 5. control · 6. instrumentation
