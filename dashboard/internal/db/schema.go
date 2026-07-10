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
CREATE TABLE IF NOT EXISTS run_rollups (
  run_id            INTEGER PRIMARY KEY,
  n_train           INTEGER NOT NULL DEFAULT 0,
  n_eval            INTEGER NOT NULL DEFAULT 0,
  n_ckpt            INTEGER NOT NULL DEFAULT 0,
  latest_train_step INTEGER,
  latest_train_loss REAL,
  latest_eval_step  INTEGER,
  latest_eval_ppl   REAL,
  latest_eval_top1  REAL,
  best_ppl          REAL,
  best_top1         REAL,
  has_horizons      INTEGER NOT NULL DEFAULT 0,
  initialized       INTEGER NOT NULL DEFAULT 1,
  FOREIGN KEY(run_id) REFERENCES runs(id) ON DELETE CASCADE
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

CREATE TRIGGER IF NOT EXISTS rollup_run_insert AFTER INSERT ON runs BEGIN
  INSERT OR IGNORE INTO run_rollups(run_id) VALUES(NEW.id);
END;
CREATE TRIGGER IF NOT EXISTS rollup_train_insert AFTER INSERT ON train_events BEGIN
  INSERT INTO run_rollups(run_id,n_train,latest_train_step,latest_train_loss)
  VALUES(NEW.run_id,1,NEW.step,NEW.loss)
  ON CONFLICT(run_id) DO UPDATE SET
    n_train=n_train+1,
    latest_train_step=CASE WHEN latest_train_step IS NULL OR NEW.step>=latest_train_step THEN NEW.step ELSE latest_train_step END,
    latest_train_loss=CASE WHEN latest_train_step IS NULL OR NEW.step>=latest_train_step THEN NEW.loss ELSE latest_train_loss END;
END;
CREATE TRIGGER IF NOT EXISTS rollup_train_update AFTER UPDATE ON train_events BEGIN
  UPDATE run_rollups SET latest_train_loss=NEW.loss
  WHERE run_id=NEW.run_id AND latest_train_step=NEW.step;
END;
CREATE TRIGGER IF NOT EXISTS rollup_eval_insert AFTER INSERT ON eval_events BEGIN
  INSERT INTO run_rollups(run_id,n_eval,latest_eval_step,latest_eval_ppl,latest_eval_top1,best_ppl,best_top1,has_horizons)
  VALUES(NEW.run_id,1,NEW.step,NEW.ppl,NEW.top1,NEW.ppl,NEW.top1,
         CASE WHEN json_extract(NEW.extra_json,'$.h4_top1') IS NOT NULL THEN 1 ELSE 0 END)
  ON CONFLICT(run_id) DO UPDATE SET
    n_eval=n_eval+1,
    latest_eval_step=CASE WHEN latest_eval_step IS NULL OR NEW.step>=latest_eval_step THEN NEW.step ELSE latest_eval_step END,
    latest_eval_ppl=CASE WHEN latest_eval_step IS NULL OR NEW.step>=latest_eval_step THEN NEW.ppl ELSE latest_eval_ppl END,
    latest_eval_top1=CASE WHEN latest_eval_step IS NULL OR NEW.step>=latest_eval_step THEN NEW.top1 ELSE latest_eval_top1 END,
    best_ppl=CASE WHEN NEW.ppl IS NULL THEN best_ppl WHEN best_ppl IS NULL OR NEW.ppl<best_ppl THEN NEW.ppl ELSE best_ppl END,
    best_top1=CASE WHEN NEW.top1 IS NULL THEN best_top1 WHEN best_top1 IS NULL OR NEW.top1>best_top1 THEN NEW.top1 ELSE best_top1 END,
    has_horizons=MAX(has_horizons,CASE WHEN json_extract(NEW.extra_json,'$.h4_top1') IS NOT NULL THEN 1 ELSE 0 END);
END;
CREATE TRIGGER IF NOT EXISTS rollup_eval_update AFTER UPDATE ON eval_events BEGIN
  UPDATE run_rollups SET
    latest_eval_ppl=CASE WHEN latest_eval_step=NEW.step THEN NEW.ppl ELSE latest_eval_ppl END,
    latest_eval_top1=CASE WHEN latest_eval_step=NEW.step THEN NEW.top1 ELSE latest_eval_top1 END,
    best_ppl=(SELECT min(ppl) FROM eval_events WHERE run_id=NEW.run_id),
    best_top1=(SELECT max(top1) FROM eval_events WHERE run_id=NEW.run_id),
    has_horizons=MAX(has_horizons,CASE WHEN json_extract(NEW.extra_json,'$.h4_top1') IS NOT NULL THEN 1 ELSE 0 END)
  WHERE run_id=NEW.run_id;
END;
CREATE TRIGGER IF NOT EXISTS rollup_ckpt_insert AFTER INSERT ON checkpoints BEGIN
  INSERT INTO run_rollups(run_id,n_ckpt) VALUES(NEW.run_id,1)
  ON CONFLICT(run_id) DO UPDATE SET n_ckpt=n_ckpt+1;
END;
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
	// Existing databases get one backfill. Thereafter the transactional triggers
	// keep this O(runs) summary current without per-second event-history scans.
	if _, err := d.Exec(`INSERT OR IGNORE INTO run_rollups(run_id,initialized)
		SELECT id,0 FROM runs`); err != nil {
		return fmt.Errorf("rollup rows: %w", err)
	}
	if _, err := d.Exec(`UPDATE run_rollups SET
		n_train=(SELECT count(*) FROM train_events WHERE run_id=run_rollups.run_id),
		n_eval=(SELECT count(*) FROM eval_events WHERE run_id=run_rollups.run_id),
		n_ckpt=(SELECT count(*) FROM checkpoints WHERE run_id=run_rollups.run_id),
		latest_train_step=(SELECT max(step) FROM train_events WHERE run_id=run_rollups.run_id),
		latest_train_loss=(SELECT loss FROM train_events WHERE run_id=run_rollups.run_id ORDER BY step DESC LIMIT 1),
		latest_eval_step=(SELECT max(step) FROM eval_events WHERE run_id=run_rollups.run_id),
		latest_eval_ppl=(SELECT ppl FROM eval_events WHERE run_id=run_rollups.run_id ORDER BY step DESC LIMIT 1),
		latest_eval_top1=(SELECT top1 FROM eval_events WHERE run_id=run_rollups.run_id ORDER BY step DESC LIMIT 1),
		best_ppl=(SELECT min(ppl) FROM eval_events WHERE run_id=run_rollups.run_id),
		best_top1=(SELECT max(top1) FROM eval_events WHERE run_id=run_rollups.run_id),
		has_horizons=EXISTS(SELECT 1 FROM eval_events WHERE run_id=run_rollups.run_id
		  AND json_extract(extra_json,'$.h4_top1') IS NOT NULL), initialized=1
		WHERE initialized=0`); err != nil {
		return fmt.Errorf("backfill rollups: %w", err)
	}
	return nil
}
