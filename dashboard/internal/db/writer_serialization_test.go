package db

import (
	"path/filepath"
	"sync"
	"testing"
	"time"
)

// openWriterTestDB opens a scratch database in the test's own directory.
func openWriterTestDB(t *testing.T) *DB {
	t.Helper()
	d, err := Open(filepath.Join(t.TempDir(), "writer.db"))
	if err != nil {
		t.Fatalf("open: %v", err)
	}
	t.Cleanup(func() { _ = d.Close() })
	return d
}

// openIngestBatch starts a run and an ingest transaction holding rows, leaving
// the transaction open. The caller commits or rolls it back.
func openIngestBatch(t *testing.T, d *DB, name string) (int64, *IngestBatch) {
	t.Helper()
	runID, err := d.EnsureRun(name, "/tmp/"+name, 1)
	if err != nil {
		t.Fatalf("ensure run: %v", err)
	}
	batch, err := d.Begin()
	if err != nil {
		t.Fatalf("begin: %v", err)
	}
	for step := int64(0); step < 200; step++ {
		if err := batch.Train(runID, TrainRow{Step: step, TS: float64(step)}); err != nil {
			batch.Rollback()
			t.Fatalf("train row: %v", err)
		}
	}
	return runID, batch
}

// A long ingest transaction is the production case: the 698,554-event rebuild
// held one for far longer than the DSN's five-second busy timeout, so a second
// connection attempting a durable write did not merely wait — it failed with
// SQLITE_BUSY. A durable write therefore has to queue behind the transaction
// rather than contend with it, which is also the proof that no second SQLite
// writer is attempted while ingest is in progress.
func TestDurableWriteWaitsForLongIngestTransactionInsteadOfFailingBusy(t *testing.T) {
	d := openWriterTestDB(t)
	runID, batch := openIngestBatch(t, d, "busy-run")

	// Longer than busy_timeout(5000): a contending connection has exhausted its
	// retries before the transaction commits, so a pooled second writer cannot
	// pass this by waiting.
	const ingestHold = 6 * time.Second

	var wg sync.WaitGroup
	var standaloneErr error
	var finished time.Time
	wg.Add(1)
	go func() {
		defer wg.Done()
		_, standaloneErr = d.Exec(
			`INSERT INTO actions(ts,kind,run_id,args_json,result,pid) VALUES(?,?,?,?,?,?)`,
			1.0, "notes", runID, "{}", "ok", 0)
		finished = time.Now()
	}()

	// Let the standalone write reach the datastore before the hold starts.
	time.Sleep(200 * time.Millisecond)
	time.Sleep(ingestHold)
	committed := time.Now()
	if err := batch.CommitAndPublish(runID, 2, "/tmp/busy-run/train.jsonl", Cursor{Offset: 1}); err != nil {
		t.Fatalf("commit and publish: %v", err)
	}
	wg.Wait()

	if standaloneErr != nil {
		t.Fatalf("durable write during ingest failed instead of queueing: %v", standaloneErr)
	}
	if finished.Before(committed) {
		t.Fatalf("durable write completed at %v, before ingest committed at %v: "+
			"a second SQLite writer ran concurrently with ingest", finished, committed)
	}
	var actions int
	if err := d.QueryRow(`SELECT count(*) FROM actions`).Scan(&actions); err != nil {
		t.Fatalf("count actions: %v", err)
	}
	if actions != 1 {
		t.Fatalf("actions rows = %d, want 1", actions)
	}
}

// Reads must not be serialized with writes: stalling them behind a long ingest
// is the failure this change exists to avoid, and WAL permits them.
func TestReadsStayServableWhileIngestHoldsTheWriter(t *testing.T) {
	d := openWriterTestDB(t)
	runID, batch := openIngestBatch(t, d, "reader-run")
	defer batch.Rollback()

	done := make(chan error, 1)
	go func() {
		var n int
		done <- d.QueryRow(`SELECT count(*) FROM train_events WHERE run_id=?`, runID).Scan(&n)
	}()
	select {
	case err := <-done:
		if err != nil {
			t.Fatalf("read during ingest: %v", err)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("read blocked behind the ingest transaction")
	}
}

// A writer waiting its turn must not be holding a pooled connection while it
// waits. The pool is four connections and the reads that keep the live UI
// moving come out of the same pool, so queued writers that each sit on an open
// transaction starve exactly the readers this change promised to preserve.
// That is why the token is taken before the transaction is opened rather than
// after: SQLite defers a write transaction's lock until the first write, so
// the wrong order looks correct until enough writers queue to exhaust the pool.
func TestQueuedWritersDoNotConsumePoolConnectionsWhileWaiting(t *testing.T) {
	d := openWriterTestDB(t)
	_, batch := openIngestBatch(t, d, "pool-run")

	// More waiters than the pool has connections.
	const waiters = 6
	var wg sync.WaitGroup
	for i := 0; i < waiters; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			if err := d.SetControls("pool-run", map[string]float64{"lr": float64(i)}, float64(i)); err != nil {
				t.Errorf("queued control write: %v", err)
			}
		}(i)
	}
	// Let every waiter reach the datastore.
	time.Sleep(300 * time.Millisecond)

	read := make(chan error, 1)
	go func() {
		var n int
		read <- d.QueryRow(`SELECT count(*) FROM runs`).Scan(&n)
	}()
	select {
	case err := <-read:
		if err != nil {
			t.Fatalf("read while writers queued: %v", err)
		}
	case <-time.After(3 * time.Second):
		t.Fatal("read starved: queued writers are holding pool connections while waiting")
	}

	batch.Rollback()
	wg.Wait()
}

// A replaceable sample must be dropped immediately rather than queued: the next
// tick supersedes it, and waiting would stop the sampler for the length of the
// ingest.
func TestReplaceableWriteSkipsWithoutWaitingDuringIngest(t *testing.T) {
	d := openWriterTestDB(t)
	_, batch := openIngestBatch(t, d, "skip-run")
	defer batch.Rollback()

	type result struct {
		persisted bool
		err       error
		took      time.Duration
	}
	done := make(chan result, 1)
	go func() {
		start := time.Now()
		persisted, err := d.TryExec(
			`INSERT OR REPLACE INTO system_samples(ts,gpu_json,cpu_pct,ram_pct,disk_pct,loadavg)
			 VALUES(?,?,?,?,?,?)`, 1.0, "[]", 1.0, 2.0, 3.0, 4.0)
		done <- result{persisted, err, time.Since(start)}
	}()

	select {
	case r := <-done:
		if r.err != nil {
			t.Fatalf("replaceable write errored: %v", r.err)
		}
		if r.persisted {
			t.Fatal("replaceable write persisted during ingest: it took the writer alongside the transaction")
		}
		if r.took > 500*time.Millisecond {
			t.Fatalf("replaceable write waited %v for the writer; it must skip without blocking", r.took)
		}
	case <-time.After(3 * time.Second):
		t.Fatal("replaceable write blocked behind the ingest transaction")
	}

	var samples int
	if err := d.QueryRow(`SELECT count(*) FROM system_samples`).Scan(&samples); err != nil {
		t.Fatalf("count samples: %v", err)
	}
	if samples != 0 {
		t.Fatalf("system_samples rows = %d, want 0 (the sample was skipped)", samples)
	}
}

// Once the transaction is gone the sampler must be able to persist again — a
// skip is for the duration of the contention, not permanent.
func TestReplaceableWritePersistsOnceTheWriterIsFree(t *testing.T) {
	d := openWriterTestDB(t)
	persisted, err := d.TryExec(
		`INSERT OR REPLACE INTO system_samples(ts,gpu_json,cpu_pct,ram_pct,disk_pct,loadavg)
		 VALUES(?,?,?,?,?,?)`, 2.0, "[]", 1.0, 2.0, 3.0, 4.0)
	if err != nil {
		t.Fatalf("replaceable write: %v", err)
	}
	if !persisted {
		t.Fatal("replaceable write skipped with no writer contending")
	}
	var samples int
	if err := d.QueryRow(`SELECT count(*) FROM system_samples`).Scan(&samples); err != nil {
		t.Fatalf("count samples: %v", err)
	}
	if samples != 1 {
		t.Fatalf("system_samples rows = %d, want 1", samples)
	}
}

// The expensive version of this bug: a writer token that a failed transaction
// never returns turns one transient ingest error into a dashboard that can
// never write again. Every exit from a write transaction must release it.
func TestFailedCommitAndPublishReleasesTheWriter(t *testing.T) {
	d := openWriterTestDB(t)
	runID, batch := openIngestBatch(t, d, "rollback-run")

	// publishCursorTx rejects an unknown run id, so the commit path fails after
	// the batch has already taken the writer.
	err := batch.CommitAndPublish(runID+9999, 2, "/tmp/rollback-run/train.jsonl", Cursor{Offset: 1})
	if err == nil {
		t.Fatal("expected CommitAndPublish to fail for an unknown run id")
	}

	assertWriterFree(t, d, "after a failed CommitAndPublish")

	// The failed generation must also have left no events behind.
	var events int
	if err := d.QueryRow(`SELECT count(*) FROM train_events`).Scan(&events); err != nil {
		t.Fatalf("count events: %v", err)
	}
	if events != 0 {
		t.Fatalf("train_events rows = %d after a rolled-back batch, want 0", events)
	}
}

func TestRollbackReleasesTheWriter(t *testing.T) {
	d := openWriterTestDB(t)
	_, batch := openIngestBatch(t, d, "explicit-rollback-run")
	batch.Rollback()
	assertWriterFree(t, d, "after an explicit Rollback")
}

func TestCommitReleasesTheWriter(t *testing.T) {
	d := openWriterTestDB(t)
	_, batch := openIngestBatch(t, d, "commit-run")
	if err := batch.Commit(); err != nil {
		t.Fatalf("commit: %v", err)
	}
	assertWriterFree(t, d, "after Commit")
}

// A transactional control write releases the writer too — it is a separate
// entry point into the same token.
func TestControlWriteReleasesTheWriter(t *testing.T) {
	d := openWriterTestDB(t)
	if err := d.SetControls("ctl-run", map[string]float64{"lr": 1e-4}, 5); err != nil {
		t.Fatalf("set controls: %v", err)
	}
	assertWriterFree(t, d, "after SetControls")
}

// assertWriterFree proves the token is available by taking it the only way a
// replaceable write can: without waiting.
func assertWriterFree(t *testing.T, d *DB, when string) {
	t.Helper()
	done := make(chan bool, 1)
	go func() {
		persisted, err := d.TryExec(
			`INSERT OR REPLACE INTO system_samples(ts,gpu_json,cpu_pct,ram_pct,disk_pct,loadavg)
			 VALUES(?,?,?,?,?,?)`, 3.0, "[]", 1.0, 2.0, 3.0, 4.0)
		if err != nil {
			t.Errorf("write %s: %v", when, err)
		}
		done <- persisted
	}()
	select {
	case persisted := <-done:
		if !persisted {
			t.Fatalf("writer still held %s: the dashboard would be permanently unable to write", when)
		}
	case <-time.After(5 * time.Second):
		t.Fatalf("write %s never returned", when)
	}
}
