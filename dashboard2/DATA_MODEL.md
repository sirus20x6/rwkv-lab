# DATA_MODEL.md — SQLite schema + JSONL field catalog

## Source of truth

Each run is a directory `/thearray/git/moe-mla/runs/<name>/` containing:
- `train.jsonl` — newline-delimited events, one per line, field `kind ∈ {train, eval, checkpoint}`.
- `step_NNNNNN/config.json` — sidecar TrainConfig dump (architecture panel reads this).
- `step_NNNNNN/ckpt.pt` — weights (we only stat its size/mtime).
- `loop_rw.json` — (optional) LoopedRWKV residual-weight stats.
Global: `runs/_baseline.json` — original-model eval reference (ppl/loss/top1) drawn as a fixed line.

Current corpus (2026-06-30): ~105 run dirs, 175 K train / 1.6 K eval / 219 checkpoint events.

## JSONL field catalog

### kind="train" (every `log_every`, default 10)
Producers: `train_mla.py:2159`, `convert_train.py:416`.

| field | always? | source | meaning |
|-------|---------|--------|---------|
| step, loss, lr | yes | both | step counter, avg CE over accum, scheduled LR |
| gnorm | when grad_clip on | both | clipped grad L2 |
| tok_per_sec | yes | both | throughput |
| skipped | sometimes | train_mla | spike-guard/non-finite step flag (junk loss → drop in charts) |
| kl | sometimes | train_mla | KL to teacher (distill) |
| guard_muon_sat/_total, guard_adam_sat/_total | GuardedMuonClip | train_mla | RMS-cap saturation counts (muon vs adam side) |
| d_prodigy | prodigy_aux | train_mla | Prodigy auto step-size D |
| lm_ce | convert | convert_train | student backbone→lm_head CE |
| block | convert | convert_train | student-vs-teacher block-output MSE |
| smt_mem, smt_update_pen | convert | convert_train | state-memory-transition loss + update penalty |
| dmt_mem, dmt_state_rms | convert | convert_train | dynamic-memory-trajectory loss + state RMS (stability) |

### kind="eval" (every `eval_every`, default 500)
Producer: `train_mla.py:1777`, `convert_train.py:434`.

| field | meaning |
|-------|---------|
| step, loss, ppl, top1_acc, top5_acc | backbone (h=1) metrics; ppl=exp(loss) |
| h{2,3,4}_{loss,ppl,top1,top5} | chained-MTP horizons (only when MTP installed + horizons>1) |
| rop_mult, rop_dropped | reduce-on-plateau LR multiplier + drop flag (when ROP active) |
| tokens | eval tokens scored |

### kind="checkpoint"
`{step}` (+ optional `reason:"interrupt"`). Sidecar `config.json` has the full TrainConfig.

**Design rule:** known fields → typed columns; everything else (the conditional long tail above) →
`extra_json`. New instrumentation fields (Phase 6) land in `extra_json` automatically, no migration.

## SQLite schema (internal/db/schema.go, WAL)

```sql
CREATE TABLE runs (
  id INTEGER PRIMARY KEY, name TEXT UNIQUE, path TEXT,
  created_ts REAL, last_update_ts REAL, status TEXT,        -- healthy|stalling|cold|no_log
  max_steps INTEGER, config_json TEXT,                       -- latest sidecar config
  notes TEXT DEFAULT '', tags_json TEXT DEFAULT '[]'         -- user annotations (control action)
);
CREATE TABLE train_events (
  run_id INTEGER, step INTEGER,
  loss REAL, lr REAL, gnorm REAL, tok_per_sec REAL, skipped INTEGER DEFAULT 0,
  extra_json TEXT, ts REAL,
  PRIMARY KEY (run_id, step)
);
CREATE TABLE eval_events (
  run_id INTEGER, step INTEGER,
  loss REAL, ppl REAL, top1 REAL, top5 REAL,
  extra_json TEXT, ts REAL,                                  -- horizons + rop live in extra_json
  PRIMARY KEY (run_id, step)
);
CREATE TABLE checkpoints (
  run_id INTEGER, step INTEGER, reason TEXT, size_bytes INTEGER, mtime REAL,
  PRIMARY KEY (run_id, step)
);
CREATE TABLE system_samples (                                -- time-series for sparklines
  ts REAL PRIMARY KEY, gpu_json TEXT, cpu_pct REAL, ram_pct REAL, disk_pct REAL, loadavg REAL
);
CREATE TABLE ingest_cursors (                                -- offset-based incremental tail
  path TEXT PRIMARY KEY, offset INTEGER, size INTEGER, mtime REAL
);
CREATE TABLE annotations (run_id INTEGER, step INTEGER, ts REAL, text TEXT);
CREATE TABLE actions (                                       -- control-action audit log
  ts REAL, kind TEXT, run_id INTEGER, args_json TEXT, result TEXT, pid INTEGER
);
CREATE INDEX idx_train_run_step ON train_events(run_id, step);
CREATE INDEX idx_eval_run_step  ON eval_events(run_id, step);
```

## Ingest algorithm (idempotent)

For each `runs/*/train.jsonl`: look up `ingest_cursors[path]`. If file `size` < stored offset (truncation/
rotation) reset offset=0. Seek to offset, read to EOF line-by-line; for each parseable line route by `kind`
into the right table via `INSERT … ON CONFLICT(run_id,step) DO UPDATE`. Persist new `(offset,size,mtime)`.
Sanitize NaN/Inf → NULL before insert (JSON can't carry them; mirrors v1 `_finite`). Bump the run's
`last_update_ts` and its **version token = max(step)**, which the SSE tick publishes to drive Pixi append.
