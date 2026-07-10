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

The experiment builder exposes trainer-native P0/P1 comparison arms for
[u-μP](https://arxiv.org/abs/2407.17465), [Titans](https://arxiv.org/abs/2501.00663) /
[MIRAS](https://arxiv.org/abs/2504.13173) / [ATLAS](https://arxiv.org/abs/2505.23735) /
[Nested Learning](https://arxiv.org/abs/2512.24695) online memory, and simulated
[NVFP4](https://arxiv.org/abs/2509.25149) with optional
[TetraJet-v2](https://arxiv.org/abs/2510.27527) randomized Hadamard transforms. These arms are
scratch-LM-only; the builder rejects u-μP+Muon and NVFP4+FP8 combinations.

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

The **post-training** panel validates the versioned SFT/preference/feedback/PRM/RLVR
JSONL contract (including structured tool calls), shows rendered variants and their trainable spans,
and compares two saved checkpoints
under an identical prompt, seed, temperature, and generation budget. It can merge repository-confined
sources into immutable content-addressed versions under `datasets/versions/`; duplicate content and
cross-split leakage are refused. An explicit human choice may be
appended to `datasets/trainboard_preferences.jsonl` as training data. Its allowlisted launcher runs
equal-token, paired exploration seeds and fresh confirmation seeds through
`rwkv_lab.posttrain_campaign`; result cards show phase-specific deltas/confidence intervals, promotion
eligibility, and adapter-first recursive lineage. It exposes qualified reset-mask packing, automatic
portable/TorchAO NF4 selection, device slots, per-attempt timeouts, and retries; campaign state resumes
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
