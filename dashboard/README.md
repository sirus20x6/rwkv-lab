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
from [DAPO](https://arxiv.org/abs/2503.14476). Results report frozen-parent and candidate held-out
reward, across-seed dispersion, updates actually applied, and promotion eligibility. The panel launches
model-side training only; generated-code verification remains outside this repository in Adamaton's
sandboxed verifier process.

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
