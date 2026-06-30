package db

import (
	"database/sql"
	"fmt"
)

// ---- row types (nil *float64 → SQL NULL) ----

type TrainRow struct {
	Step      int64
	Loss      *float64
	LR        *float64
	Gnorm     *float64
	TokPerSec *float64
	Skipped   bool
	Extra     string // JSON of leftover fields ("" → NULL)
	TS        float64
}

type EvalRow struct {
	Step  int64
	Loss  *float64
	PPL   *float64
	Top1  *float64
	Top5  *float64
	Extra string
	TS    float64
}

type CkptRow struct {
	Step      int64
	Reason    string
	SizeBytes int64
	Mtime     float64
}

// Cursor tracks how far we've ingested a single train.jsonl file.
type Cursor struct {
	Offset int64
	Size   int64
	Mtime  float64
}

// ---- run identity ----

// EnsureRun inserts the run if absent and returns its id.
func (d *DB) EnsureRun(name, path string, createdTs float64) (int64, error) {
	if _, err := d.Exec(
		`INSERT INTO runs(name, path, created_ts, last_update_ts, status)
		 VALUES(?,?,?,?,'') ON CONFLICT(name) DO NOTHING`,
		name, path, createdTs, createdTs); err != nil {
		return 0, fmt.Errorf("ensure run: %w", err)
	}
	var id int64
	if err := d.QueryRow(`SELECT id FROM runs WHERE name=?`, name).Scan(&id); err != nil {
		return 0, fmt.Errorf("run id: %w", err)
	}
	return id, nil
}

// TouchRun updates a run's last-seen timestamp.
func (d *DB) TouchRun(id int64, lastUpdateTs float64) error {
	_, err := d.Exec(`UPDATE runs SET last_update_ts=? WHERE id=?`, lastUpdateTs, id)
	return err
}

// RunID looks up a run id by name. ok=false if the run is unknown.
func (d *DB) RunID(name string) (int64, bool, error) {
	var id int64
	err := d.QueryRow(`SELECT id FROM runs WHERE name=?`, name).Scan(&id)
	if err == sql.ErrNoRows {
		return 0, false, nil
	}
	if err != nil {
		return 0, false, err
	}
	return id, true, nil
}

// ---- ingest cursors ----

// GetCursor returns the stored cursor for a path (zero value if none).
func (d *DB) GetCursor(path string) (Cursor, error) {
	var c Cursor
	err := d.QueryRow(`SELECT offset, size, mtime FROM ingest_cursors WHERE path=?`, path).
		Scan(&c.Offset, &c.Size, &c.Mtime)
	if err == sql.ErrNoRows {
		return Cursor{}, nil
	}
	return c, err
}

// SaveCursor upserts the cursor for a path.
func (d *DB) SaveCursor(path string, c Cursor) error {
	_, err := d.Exec(
		`INSERT INTO ingest_cursors(path, offset, size, mtime) VALUES(?,?,?,?)
		 ON CONFLICT(path) DO UPDATE SET offset=excluded.offset, size=excluded.size, mtime=excluded.mtime`,
		path, c.Offset, c.Size, c.Mtime)
	return err
}

// ---- batched event ingest (one tx per file scan) ----

const trainUpsert = `INSERT INTO train_events(run_id,step,loss,lr,gnorm,tok_per_sec,skipped,extra_json,ts)
VALUES(?,?,?,?,?,?,?,?,?)
ON CONFLICT(run_id,step) DO UPDATE SET loss=excluded.loss, lr=excluded.lr, gnorm=excluded.gnorm,
  tok_per_sec=excluded.tok_per_sec, skipped=excluded.skipped, extra_json=excluded.extra_json, ts=excluded.ts`

const evalUpsert = `INSERT INTO eval_events(run_id,step,loss,ppl,top1,top5,extra_json,ts)
VALUES(?,?,?,?,?,?,?,?)
ON CONFLICT(run_id,step) DO UPDATE SET loss=excluded.loss, ppl=excluded.ppl, top1=excluded.top1,
  top5=excluded.top5, extra_json=excluded.extra_json, ts=excluded.ts`

const ckptUpsert = `INSERT INTO checkpoints(run_id,step,reason,size_bytes,mtime)
VALUES(?,?,?,?,?)
ON CONFLICT(run_id,step) DO UPDATE SET reason=excluded.reason, size_bytes=excluded.size_bytes, mtime=excluded.mtime`

// IngestBatch holds a transaction + prepared upserts for one file scan.
type IngestBatch struct {
	tx        *sql.Tx
	trainStmt *sql.Stmt
	evalStmt  *sql.Stmt
	ckptStmt  *sql.Stmt
}

// Begin opens a batch. Call Commit or Rollback exactly once.
func (d *DB) Begin() (*IngestBatch, error) {
	tx, err := d.DB.Begin()
	if err != nil {
		return nil, err
	}
	b := &IngestBatch{tx: tx}
	for _, p := range []struct {
		sql string
		dst **sql.Stmt
	}{{trainUpsert, &b.trainStmt}, {evalUpsert, &b.evalStmt}, {ckptUpsert, &b.ckptStmt}} {
		st, err := tx.Prepare(p.sql)
		if err != nil {
			_ = tx.Rollback()
			return nil, fmt.Errorf("prepare: %w", err)
		}
		*p.dst = st
	}
	return b, nil
}

func nullIfEmpty(s string) any {
	if s == "" {
		return nil
	}
	return s
}

func (b *IngestBatch) Train(runID int64, r TrainRow) error {
	skip := 0
	if r.Skipped {
		skip = 1
	}
	_, err := b.trainStmt.Exec(runID, r.Step, r.Loss, r.LR, r.Gnorm, r.TokPerSec, skip, nullIfEmpty(r.Extra), r.TS)
	return err
}

func (b *IngestBatch) Eval(runID int64, r EvalRow) error {
	_, err := b.evalStmt.Exec(runID, r.Step, r.Loss, r.PPL, r.Top1, r.Top5, nullIfEmpty(r.Extra), r.TS)
	return err
}

func (b *IngestBatch) Checkpoint(runID int64, r CkptRow) error {
	_, err := b.ckptStmt.Exec(runID, r.Step, nullIfEmpty(r.Reason), r.SizeBytes, r.Mtime)
	return err
}

// Commit finalizes the batch.
func (b *IngestBatch) Commit() error { return b.tx.Commit() }

// Rollback aborts the batch (safe to call after Commit — it just errors).
func (b *IngestBatch) Rollback() { _ = b.tx.Rollback() }

// ---- read helpers (verification + summaries) ----

// EventCounts returns (train, eval, checkpoint) row counts for a run.
func (d *DB) EventCounts(runID int64) (nTrain, nEval, nCkpt int, err error) {
	q := func(table string) (int, error) {
		var n int
		e := d.QueryRow(fmt.Sprintf(`SELECT count(*) FROM %s WHERE run_id=?`, table), runID).Scan(&n)
		return n, e
	}
	if nTrain, err = q("train_events"); err != nil {
		return
	}
	if nEval, err = q("eval_events"); err != nil {
		return
	}
	nCkpt, err = q("checkpoints")
	return
}

// ---- control actions ----

// LogAction appends a row to the audit table. runID<=0 means "no run".
func (d *DB) LogAction(ts float64, kind, runName, argsJSON, result string, pid int) {
	var rid any
	if id, ok, _ := d.RunID(runName); ok {
		rid = id
	}
	_, _ = d.Exec(`INSERT INTO actions(ts,kind,run_id,args_json,result,pid) VALUES(?,?,?,?,?,?)`,
		ts, kind, rid, argsJSON, result, pid)
}

// SetNotes updates a run's free-text notes.
func (d *DB) SetNotes(name, notes string) error {
	_, err := d.Exec(`UPDATE runs SET notes=? WHERE name=?`, notes, name)
	return err
}

// SetTags updates a run's tags (JSON array string).
func (d *DB) SetTags(name, tagsJSON string) error {
	_, err := d.Exec(`UPDATE runs SET tags_json=? WHERE name=?`, tagsJSON, name)
	return err
}

// RunMeta returns a run's notes + tags_json (defaults if unset).
func (d *DB) RunMeta(name string) (notes, tagsJSON string) {
	tagsJSON = "[]"
	_ = d.QueryRow(`SELECT COALESCE(notes,''), COALESCE(tags_json,'[]') FROM runs WHERE name=?`, name).
		Scan(&notes, &tagsJSON)
	return notes, tagsJSON
}

// MaxTrainStep returns the highest ingested train step (the version token), or 0.
func (d *DB) MaxTrainStep(runID int64) (int64, error) {
	var v sql.NullInt64
	err := d.QueryRow(`SELECT max(step) FROM train_events WHERE run_id=?`, runID).Scan(&v)
	if err != nil {
		return 0, err
	}
	return v.Int64, nil
}
