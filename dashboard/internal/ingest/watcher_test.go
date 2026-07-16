package ingest

import (
	"fmt"
	"os"
	"path/filepath"
	"testing"
	"time"

	"trainboard/internal/db"
)

func TestRewriteAndRegrowBeyondCursorTriggersFullReingest(t *testing.T) {
	root := t.TempDir()
	runDir := filepath.Join(root, "vision")
	if err := os.MkdirAll(runDir, 0o755); err != nil {
		t.Fatal(err)
	}
	logPath := filepath.Join(runDir, "train.jsonl")
	initial := []byte("{\"kind\":\"train\",\"step\":1,\"loss\":1}\n" +
		"{\"kind\":\"train\",\"step\":2,\"loss\":2}\n")
	if err := os.WriteFile(logPath, initial, 0o600); err != nil {
		t.Fatal(err)
	}
	database, err := db.Open(filepath.Join(root, "dashboard.db"))
	if err != nil {
		t.Fatal(err)
	}
	defer database.Close()
	ingester := New(database, root, 0)
	if n, err := ingester.ScanOnce(); err != nil || n != 2 {
		t.Fatalf("initial scan n=%d err=%v", n, err)
	}
	rewritten := make([]byte, 0, len(initial)*4)
	for step := 10; step < 20; step++ {
		rewritten = fmt.Appendf(rewritten,
			"{\"kind\":\"train\",\"step\":%d,\"loss\":%d}\n", step, step)
	}
	if len(rewritten) <= len(initial) {
		t.Fatal("test rewrite did not regrow beyond the previous cursor")
	}
	if err := os.WriteFile(logPath, rewritten, 0o600); err != nil {
		t.Fatal(err)
	}
	if n, err := ingester.ScanOnce(); err != nil || n != 10 {
		t.Fatalf("rewrite scan n=%d err=%v", n, err)
	}

	var count, minimum, maximum int
	if err := database.QueryRow(`SELECT count(*), min(step), max(step)
		FROM train_events`).Scan(&count, &minimum, &maximum); err != nil {
		t.Fatal(err)
	}
	if count != 10 || minimum != 10 || maximum != 19 {
		t.Fatalf("stale/missing rows after rewrite: count=%d range=%d..%d",
			count, minimum, maximum)
	}
}

func TestTruncateToEmptyCommitsNewCursorGeneration(t *testing.T) {
	root := t.TempDir()
	runDir := filepath.Join(root, "vision")
	if err := os.MkdirAll(runDir, 0o755); err != nil {
		t.Fatal(err)
	}
	logPath := filepath.Join(runDir, "train.jsonl")
	if err := os.WriteFile(logPath,
		[]byte("{\"kind\":\"train\",\"step\":1,\"loss\":1}\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	database, err := db.Open(filepath.Join(root, "dashboard.db"))
	if err != nil {
		t.Fatal(err)
	}
	defer database.Close()
	ingester := New(database, root, 0)
	if n, err := ingester.ScanOnce(); err != nil || n != 1 {
		t.Fatalf("initial scan n=%d err=%v", n, err)
	}
	var beforeRevision float64
	if err := database.QueryRow(`SELECT last_update_ts FROM runs WHERE name='vision'`).
		Scan(&beforeRevision); err != nil {
		t.Fatal(err)
	}
	if err := os.Truncate(logPath, 0); err != nil {
		t.Fatal(err)
	}
	if n, err := ingester.ScanOnce(); err != nil || n != 0 {
		t.Fatalf("truncate scan n=%d err=%v", n, err)
	}
	cur, err := database.GetCursor(logPath)
	if err != nil {
		t.Fatal(err)
	}
	if cur.Offset != 0 || cur.Size != 0 || cur.FileID == "" {
		t.Fatalf("empty generation cursor was not committed: %+v", cur)
	}
	var count int
	if err := database.QueryRow(`SELECT count(*) FROM train_events`).Scan(&count); err != nil {
		t.Fatal(err)
	}
	if count != 0 {
		t.Fatalf("truncated run retained %d stale events", count)
	}
	var afterRevision float64
	if err := database.QueryRow(`SELECT last_update_ts FROM runs WHERE name='vision'`).
		Scan(&afterRevision); err != nil {
		t.Fatal(err)
	}
	if !(afterRevision > beforeRevision) {
		t.Fatalf("empty rewrite was not published: %f -> %f", beforeRevision, afterRevision)
	}
}

func TestSameTipRewriteAdvancesEventGeneration(t *testing.T) {
	root := t.TempDir()
	runDir := filepath.Join(root, "vision")
	if err := os.MkdirAll(runDir, 0o755); err != nil {
		t.Fatal(err)
	}
	logPath := filepath.Join(runDir, "train.jsonl")
	initial := []byte("{\"kind\":\"train\",\"step\":1,\"loss\":1}\n" +
		"{\"kind\":\"train\",\"step\":2,\"loss\":2}\n")
	if err := os.WriteFile(logPath, initial, 0o600); err != nil {
		t.Fatal(err)
	}
	database, err := db.Open(filepath.Join(root, "dashboard.db"))
	if err != nil {
		t.Fatal(err)
	}
	defer database.Close()
	ingester := New(database, root, 0)
	if n, err := ingester.ScanOnce(); err != nil || n != 2 {
		t.Fatalf("initial scan n=%d err=%v", n, err)
	}
	var beforeRevision float64
	if err := database.QueryRow(`SELECT last_update_ts FROM runs WHERE name='vision'`).
		Scan(&beforeRevision); err != nil {
		t.Fatal(err)
	}

	// Keep the row count and maximum step unchanged while replacing values.
	// A tip-only browser cursor cannot distinguish these generations.
	rewritten := []byte("{\"kind\":\"train\",\"step\":1,\"loss\":8}\n" +
		"{\"kind\":\"train\",\"step\":2,\"loss\":9}\n")
	if err := os.WriteFile(logPath, rewritten, 0o600); err != nil {
		t.Fatal(err)
	}
	if n, err := ingester.ScanOnce(); err != nil || n != 2 {
		t.Fatalf("rewrite scan n=%d err=%v", n, err)
	}

	var generation, maximum, rollupCount, rollupTip int
	var afterRevision float64
	var loss float64
	if err := database.QueryRow(`SELECT r.event_generation,
		(SELECT max(step) FROM train_events WHERE run_id=r.id),
		(SELECT loss FROM train_events WHERE run_id=r.id AND step=2),
		(SELECT n_train FROM run_rollups WHERE run_id=r.id),
		(SELECT latest_train_step FROM run_rollups WHERE run_id=r.id),
		r.last_update_ts
		FROM runs r WHERE name='vision'`).Scan(
		&generation, &maximum, &loss, &rollupCount, &rollupTip,
		&afterRevision); err != nil {
		t.Fatal(err)
	}
	if generation != 1 || maximum != 2 || loss != 9 ||
		rollupCount != 2 || rollupTip != 2 || !(afterRevision > beforeRevision) {
		t.Fatalf("same-tip rewrite not atomically published: generation=%d max=%d "+
			"loss=%g rollup=%d/%d revision=%f->%f",
			generation, maximum, loss, rollupCount, rollupTip,
			beforeRevision, afterRevision)
	}
	cursor, err := database.GetCursor(logPath)
	if err != nil {
		t.Fatal(err)
	}
	if cursor.Offset != int64(len(rewritten)) || cursor.TailHash == "" || cursor.FileID == "" {
		t.Fatalf("replacement cursor did not commit with reset generation: %+v", cursor)
	}
}

func TestMigratedCursorDoesNotBlessOfflineRewrite(t *testing.T) {
	root := t.TempDir()
	runDir := filepath.Join(root, "vision")
	if err := os.MkdirAll(runDir, 0o755); err != nil {
		t.Fatal(err)
	}
	logPath := filepath.Join(runDir, "train.jsonl")
	initial := []byte("{\"kind\":\"train\",\"step\":1,\"loss\":1}\n" +
		"{\"kind\":\"train\",\"step\":2,\"loss\":2}\n")
	if err := os.WriteFile(logPath, initial, 0o600); err != nil {
		t.Fatal(err)
	}
	database, err := db.Open(filepath.Join(root, "dashboard.db"))
	if err != nil {
		t.Fatal(err)
	}
	defer database.Close()
	ingester := New(database, root, 0)
	if n, err := ingester.ScanOnce(); err != nil || n != 2 {
		t.Fatalf("initial scan n=%d err=%v", n, err)
	}
	// Simulate an ingest_cursors row created by the schema before file identity
	// and tail hashes existed.
	if _, err := database.Exec(`UPDATE ingest_cursors SET tail_hash='',file_id='' WHERE path=?`,
		logPath); err != nil {
		t.Fatal(err)
	}
	rewritten := []byte("{\"kind\":\"train\",\"step\":8,\"loss\":8}\n" +
		"{\"kind\":\"train\",\"step\":9,\"loss\":9}\n")
	if err := os.WriteFile(logPath, rewritten, 0o600); err != nil {
		t.Fatal(err)
	}
	future := time.Now().Add(2 * time.Second)
	if err := os.Chtimes(logPath, future, future); err != nil {
		t.Fatal(err)
	}
	if n, err := ingester.ScanOnce(); err != nil || n != 2 {
		t.Fatalf("migrated rewrite scan n=%d err=%v", n, err)
	}
	var count, minimum, maximum, generation int
	if err := database.QueryRow(`SELECT count(*),min(step),max(step),
		(SELECT event_generation FROM runs WHERE name='vision') FROM train_events`).
		Scan(&count, &minimum, &maximum, &generation); err != nil {
		t.Fatal(err)
	}
	if count != 2 || minimum != 8 || maximum != 9 || generation != 1 {
		t.Fatalf("offline rewrite mixed generations: count=%d range=%d..%d generation=%d",
			count, minimum, maximum, generation)
	}
}
