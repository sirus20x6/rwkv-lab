// Package db owns the SQLite datastore: schema, migrations, and typed
// read/write helpers. Driver is modernc.org/sqlite (pure Go, no cgo) registered
// under the name "sqlite". WAL mode + a single writer connection keep the
// ingester and sampler from tripping over each other.
package db

import (
	"database/sql"
	"fmt"

	_ "modernc.org/sqlite"
)

// DB wraps *sql.DB with trainboard-specific helpers.
type DB struct {
	*sql.DB
}

const schemaDDL = `
CREATE TABLE IF NOT EXISTS runs (
  id             INTEGER PRIMARY KEY,
  name           TEXT UNIQUE,
  path           TEXT,
  created_ts     REAL,
  last_update_ts REAL,
  status         TEXT,
  max_steps      INTEGER,
  config_json    TEXT,
  notes          TEXT DEFAULT '',
  tags_json      TEXT DEFAULT '[]'
);
CREATE TABLE IF NOT EXISTS train_events (
  run_id      INTEGER NOT NULL,
  step        INTEGER NOT NULL,
  loss        REAL,
  lr          REAL,
  gnorm       REAL,
  tok_per_sec REAL,
  skipped     INTEGER DEFAULT 0,
  extra_json  TEXT,
  ts          REAL,
  PRIMARY KEY (run_id, step)
) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS eval_events (
  run_id     INTEGER NOT NULL,
  step       INTEGER NOT NULL,
  loss       REAL,
  ppl        REAL,
  top1       REAL,
  top5       REAL,
  extra_json TEXT,
  ts         REAL,
  PRIMARY KEY (run_id, step)
) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS checkpoints (
  run_id     INTEGER NOT NULL,
  step       INTEGER NOT NULL,
  reason     TEXT,
  size_bytes INTEGER,
  mtime      REAL,
  PRIMARY KEY (run_id, step)
) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS system_samples (
  ts       REAL PRIMARY KEY,
  gpu_json TEXT,
  cpu_pct  REAL,
  ram_pct  REAL,
  disk_pct REAL,
  loadavg  REAL
);
CREATE TABLE IF NOT EXISTS ingest_cursors (
  path   TEXT PRIMARY KEY,
  offset INTEGER,
  size   INTEGER,
  mtime  REAL
);
CREATE TABLE IF NOT EXISTS annotations (
  run_id INTEGER, step INTEGER, ts REAL, text TEXT
);
CREATE TABLE IF NOT EXISTS actions (
  ts REAL, kind TEXT, run_id INTEGER, args_json TEXT, result TEXT, pid INTEGER
);

-- Live hyperparameter overrides. The dashboard writes desired (run_name,key)->
-- value in one tx (bumping generation); the instrumented trainer polls, applies,
-- and writes back applied_step/applied_ts as an ACID ack. desired != applied =
-- "pending".
CREATE TABLE IF NOT EXISTS run_controls (
  run_name     TEXT NOT NULL,
  key          TEXT NOT NULL,
  value        REAL,
  generation   INTEGER NOT NULL DEFAULT 0,
  requested_ts REAL,
  applied_step INTEGER,
  applied_ts   REAL,
  PRIMARY KEY (run_name, key)
);

-- Divergence / health alerts raised by the detector goroutine.
CREATE TABLE IF NOT EXISTS alerts (
  id           INTEGER PRIMARY KEY AUTOINCREMENT,
  ts           REAL,
  run_name     TEXT,
  kind         TEXT,      -- codec_collapse|gnorm_spike|nan_rate|ppl_regress|throughput_drop|stall
  severity     TEXT,      -- warn|critical
  message      TEXT,
  step         INTEGER,
  acknowledged INTEGER DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_alerts_ts ON alerts(ts);

-- GPU launch queue: enqueued training runs auto-started when the GPU frees.
CREATE TABLE IF NOT EXISTS launch_queue (
  id          INTEGER PRIMARY KEY AUTOINCREMENT,
  enqueued_ts REAL,
  script      TEXT,
  args        TEXT,
  status      TEXT DEFAULT 'queued',  -- queued|running|done|failed|canceled
  priority    INTEGER DEFAULT 0,
  pid         INTEGER,
  started_ts  REAL,
  finished_ts REAL,
  log_path    TEXT
);

-- Narrow secondary indexes for the per-second summary aggregates. The event
-- tables are WITHOUT ROWID, so scanning "the table" reads whole rows including
-- extra_json blobs; these key-only indexes keep count()/max(step)/min(ppl)
-- GROUP BY scans to a few MB regardless of how big extra_json payloads grow.
CREATE INDEX IF NOT EXISTS idx_train_run_step ON train_events(run_id, step);
CREATE INDEX IF NOT EXISTS idx_eval_run_step_ppl ON eval_events(run_id, step, ppl, top1);

-- Layer library provenance: which run/checkpoint produced each accepted L*.pt.
CREATE TABLE IF NOT EXISTS layer_lib (
  layer       INTEGER PRIMARY KEY,
  run_name    TEXT,
  src_step    INTEGER,
  lib_path    TEXT,
  ppl         REAL,
  codec_rel   REAL,
  accepted_ts REAL
);
`

// Open opens (creating if needed) the SQLite database, applies WAL pragmas, and
// runs migrations. A single open connection serializes all writes — simplest
// correct model for one ingester + one sampler + quick handler reads.
func Open(path string) (*DB, error) {
	dsn := fmt.Sprintf("file:%s?_pragma=busy_timeout(5000)&_pragma=journal_mode(WAL)&_pragma=synchronous(NORMAL)&_pragma=foreign_keys(ON)", path)
	sdb, err := sql.Open("sqlite", dsn)
	if err != nil {
		return nil, fmt.Errorf("open sqlite: %w", err)
	}
	// One connection = fully serialized access, no SQLITE_BUSY churn. Handler
	// reads are fast (indexed by run_id, step), so this is fine for localhost.
	sdb.SetMaxOpenConns(1)
	d := &DB{sdb}
	if err := d.migrate(); err != nil {
		_ = sdb.Close()
		return nil, err
	}
	return d, nil
}

func (d *DB) migrate() error {
	if _, err := d.Exec(schemaDDL); err != nil {
		return fmt.Errorf("migrate: %w", err)
	}
	return nil
}
