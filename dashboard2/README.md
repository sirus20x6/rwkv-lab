# trainboard — RWKV-Lab training dashboard v2.0

GPU-accelerated, real-time training dashboard for the RWKV-Lab conversion project.
**Stack:** Go + SQLite + Datastar + Pixi.js. Successor to `../dashboard/` (FastAPI + Chart.js).

## Run

```bash
go -C /thearray/git/moe-mla/dashboard2 run ./cmd/trainboard
# open http://127.0.0.1:9124
```

Reads `/thearray/git/moe-mla/runs/`. Ingests all `train.jsonl` logs + system telemetry into a local
SQLite DB (`trainboard.db`). GPU-light — safe to run alongside live training.

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
