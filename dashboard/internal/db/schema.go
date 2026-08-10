// Package db owns the SQLite datastore: schema, migrations, and typed
// read/write helpers. Driver is modernc.org/sqlite (pure Go, no cgo) registered
// under the name "sqlite". WAL mode keeps readers pooled while a logical
// writer token keeps the ingester and sampler from tripping over each other.
package db

import (
	"context"
	"database/sql"
	"fmt"
	"sync"

	_ "modernc.org/sqlite"
)

// DB wraps *sql.DB with trainboard-specific helpers.
//
// SQLite permits exactly one writer per database regardless of how many
// connections the pool holds, and this package used to state that the
// application only ever has one without anything making it true. It was not:
// the ingester's transaction and the 1 Hz sysmon sample are separate
// goroutines, so a 698,554-event rebuild held a write transaction for minutes
// while the sampler tried to write on a second pooled connection. The DSN's
// busy_timeout does not rescue that — it converts an immediate SQLITE_BUSY
// into a five-second stall and then fails anyway.
//
// writer is that missing mechanism: a one-token semaphore every durable write
// passes through. Holding it blocks (Exec, beginWrite) for writes that must
// land, and is skipped without waiting (TryExec) for samples whose next value
// supersedes them. Reads never take it, so the pool keeps serving the live UI
// underneath a long ingest.
//
// pool is unexported and no longer embedded, which is what makes the token
// unavoidable rather than merely conventional. While *sql.DB was embedded the
// token held only because every author remembered it: any package with a
// *db.DB could write d.DB.Exec(...) or d.DB.Begin() and contend with ingest
// exactly as before, which is the same unenforced-convention shape as the
// defect the token was added to fix. Unexporting makes that call site
// inexpressible outside this package instead of merely discouraged; the
// forwarders below re-offer the pooled read surface callers actually used, and
// deliberately do not re-offer Exec, Begin, BeginTx or Prepare.
type DB struct {
	pool   *sql.DB
	writer chan struct{}
}

// Query, QueryRow and their context forms forward to the pool untouched:
// readers must NOT take the writer token, because WAL exists precisely so the
// live UI keeps being served underneath a multi-minute ingest transaction.
// They are hand-written forwarders rather than promotion because promotion is
// all-or-nothing — it would carry Exec, Begin, BeginTx and Prepare with it.
func (d *DB) Query(query string, args ...any) (*sql.Rows, error) {
	return d.pool.Query(query, args...)
}

func (d *DB) QueryContext(ctx context.Context, query string, args ...any) (*sql.Rows, error) {
	return d.pool.QueryContext(ctx, query, args...)
}

func (d *DB) QueryRow(query string, args ...any) *sql.Row {
	return d.pool.QueryRow(query, args...)
}

func (d *DB) QueryRowContext(ctx context.Context, query string, args ...any) *sql.Row {
	return d.pool.QueryRowContext(ctx, query, args...)
}

// Close and Ping are lifecycle, not writes, and every caller of Open needs
// them.
func (d *DB) Close() error { return d.pool.Close() }

func (d *DB) Ping() error { return d.pool.Ping() }

// Stats reports pool statistics. It is a diagnostic: the pool must keep more
// than one connection so reads survive a long ingest, and a test asserts that.
func (d *DB) Stats() sql.DBStats { return d.pool.Stats() }

// acquireWriter waits for the logical writer token.
func (d *DB) acquireWriter() { d.writer <- struct{}{} }

// tryAcquireWriter takes the token if it is free, and reports whether it did.
// It never waits: a caller holding a replaceable value would rather drop it
// than queue behind a multi-minute ingest.
func (d *DB) tryAcquireWriter() bool {
	select {
	case d.writer <- struct{}{}:
		return true
	default:
		return false
	}
}

// releaseWriter returns the token. Every acquisition must reach this, on error
// paths included: a token leaked by a failed transaction wedges every
// subsequent dashboard write permanently, which is a worse failure than the
// SQLITE_BUSY it replaced.
func (d *DB) releaseWriter() { <-d.writer }

// Exec shadows the promoted sql.DB.Exec so every standalone durable write in
// this package is serialized against ingest without each call site having to
// remember. Shadowing rather than renaming is deliberate: a caller that
// forgets a new name still compiles and still contends, and this defect was
// caused by exactly that kind of unenforced convention.
func (d *DB) Exec(query string, args ...any) (sql.Result, error) {
	d.acquireWriter()
	defer d.releaseWriter()
	return d.pool.Exec(query, args...)
}

// TryExec performs a write only if no other writer holds the token, and
// reports whether it did. It is for rows whose next sample replaces them
// entirely — losing one is invisible, whereas blocking the sampler behind an
// ingest stops live telemetry, which is the thing this whole change protects.
func (d *DB) TryExec(query string, args ...any) (persisted bool, err error) {
	if !d.tryAcquireWriter() {
		return false, nil
	}
	defer d.releaseWriter()
	_, err = d.pool.Exec(query, args...)
	return true, err
}

// writeTx is a transaction that owns the logical writer token for its
// lifetime. Commit and Rollback both release it, and release is idempotent so
// the customary "defer tx.Rollback()" after a successful Commit stays a no-op.
type writeTx struct {
	*sql.Tx
	db       *DB
	released sync.Once
}

// beginWrite takes the writer token, then opens the transaction. Ordering
// matters: taking the token second would let two goroutines both hold open
// SQLite write transactions, which is the condition being removed.
func (d *DB) beginWrite() (*writeTx, error) {
	d.acquireWriter()
	tx, err := d.pool.Begin()
	if err != nil {
		d.releaseWriter()
		return nil, err
	}
	return &writeTx{Tx: tx, db: d}, nil
}

func (w *writeTx) release() { w.released.Do(w.db.releaseWriter) }

// Commit releases the writer token whether or not the commit succeeded.
func (w *writeTx) Commit() error {
	err := w.Tx.Commit()
	w.release()
	return err
}

// Rollback releases the writer token whether or not the rollback succeeded.
func (w *writeTx) Rollback() error {
	err := w.Tx.Rollback()
	w.release()
	return err
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
  tags_json      TEXT DEFAULT '[]',
  event_generation INTEGER NOT NULL DEFAULT 0
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
  path      TEXT PRIMARY KEY,
  offset    INTEGER,
  size      INTEGER,
  mtime     REAL,
  tail_hash TEXT DEFAULT '',
  file_id   TEXT DEFAULT ''
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
CREATE INDEX IF NOT EXISTS idx_train_codec_rel ON train_events(run_id, step DESC)
  WHERE json_extract(extra_json,'$.codec_rel') IS NOT NULL;

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

// Open opens (creating if needed) the SQLite database and applies WAL pragmas.
// Writes are serialized by the writer token above, not by the pool; the extra
// connections exist so WAL readers keep serving the dashboard while a large
// initial scan or rewritten-log replay holds a write transaction.
func Open(path string) (*DB, error) {
	dsn := fmt.Sprintf("file:%s?_pragma=busy_timeout(5000)&_pragma=journal_mode(WAL)&_pragma=synchronous(NORMAL)&_pragma=foreign_keys(ON)", path)
	sdb, err := sql.Open("sqlite", dsn)
	if err != nil {
		return nil, fmt.Errorf("open sqlite: %w", err)
	}
	// Do not collapse this to one connection: database/sql would then queue every
	// handler behind the ingester's transaction even though WAL permits those
	// reads. Four is enough for the one writer plus concurrent live/series/SSE
	// reads without creating an unbounded localhost connection pool.
	sdb.SetMaxOpenConns(4)
	sdb.SetMaxIdleConns(4)
	d := &DB{pool: sdb, writer: make(chan struct{}, 1)}
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
	// CREATE TABLE IF NOT EXISTS cannot add columns to an existing dashboard.
	// A cursor tail fingerprint detects rewrite-and-regrow between 1 Hz polls.
	columns, err := d.Query(`PRAGMA table_info(ingest_cursors)`)
	if err != nil {
		return fmt.Errorf("cursor columns: %w", err)
	}
	hasTailHash, hasFileID := false, false
	for columns.Next() {
		var cid, notnull, pk int
		var name, kind string
		var defaultValue any
		if err := columns.Scan(&cid, &name, &kind, &notnull, &defaultValue, &pk); err != nil {
			columns.Close()
			return fmt.Errorf("cursor column: %w", err)
		}
		if name == "tail_hash" {
			hasTailHash = true
		} else if name == "file_id" {
			hasFileID = true
		}
	}
	if err := columns.Close(); err != nil {
		return fmt.Errorf("cursor columns close: %w", err)
	}
	// A mid-iteration error would end the loop early and misread a column as
	// missing, turning the guarded ALTER into a duplicate-column failure.
	if err := columns.Err(); err != nil {
		return fmt.Errorf("cursor columns iterate: %w", err)
	}
	if !hasTailHash {
		if _, err := d.Exec(`ALTER TABLE ingest_cursors ADD COLUMN tail_hash TEXT DEFAULT ''`); err != nil {
			return fmt.Errorf("add cursor tail hash: %w", err)
		}
	}
	if !hasFileID {
		if _, err := d.Exec(`ALTER TABLE ingest_cursors ADD COLUMN file_id TEXT DEFAULT ''`); err != nil {
			return fmt.Errorf("add cursor file id: %w", err)
		}
	}
	// A maximum step cannot identify every rewrite: a recovered trainer may
	// regrow to the old tip between browser polls. Persist an explicit event
	// generation so clients can replace same-tip rewritten history.
	runColumns, err := d.Query(`PRAGMA table_info(runs)`)
	if err != nil {
		return fmt.Errorf("run columns: %w", err)
	}
	hasEventGeneration := false
	for runColumns.Next() {
		var cid, notnull, pk int
		var name, kind string
		var defaultValue any
		if err := runColumns.Scan(
			&cid, &name, &kind, &notnull, &defaultValue, &pk); err != nil {
			runColumns.Close()
			return fmt.Errorf("run column: %w", err)
		}
		if name == "event_generation" {
			hasEventGeneration = true
		}
	}
	if err := runColumns.Close(); err != nil {
		return fmt.Errorf("run columns close: %w", err)
	}
	if err := runColumns.Err(); err != nil {
		return fmt.Errorf("run columns iterate: %w", err)
	}
	if !hasEventGeneration {
		if _, err := d.Exec(`ALTER TABLE runs ADD COLUMN event_generation INTEGER NOT NULL DEFAULT 0`); err != nil {
			return fmt.Errorf("add run event generation: %w", err)
		}
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
