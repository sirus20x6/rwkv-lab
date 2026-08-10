package sysmon

import (
	"path/filepath"
	"testing"
	"time"

	"trainboard/internal/db"
)

// A sample dropped for lack of the datastore writer must still reach the live
// UI: the snapshot is published in memory before persistence is attempted, and
// persistence returning without writing is a normal outcome, not an error.
func TestSamplerKeepsLiveSnapshotWhenPersistenceIsSkipped(t *testing.T) {
	database, err := db.Open(filepath.Join(t.TempDir(), "sysmon.db"))
	if err != nil {
		t.Fatalf("open: %v", err)
	}
	defer database.Close()

	// Hold the writer the way a large ingest does.
	runID, err := database.EnsureRun("sysmon-run", "/tmp/sysmon-run", 1)
	if err != nil {
		t.Fatalf("ensure run: %v", err)
	}
	batch, err := database.Begin()
	if err != nil {
		t.Fatalf("begin: %v", err)
	}
	if err := batch.Train(runID, db.TrainRow{Step: 1, TS: 1}); err != nil {
		batch.Rollback()
		t.Fatalf("train row: %v", err)
	}
	defer batch.Rollback()

	s := New(database, t.TempDir(), t.TempDir(), time.Second)
	done := make(chan struct{})
	go func() {
		s.sampleOnce()
		close(done)
	}()
	select {
	case <-done:
	case <-time.After(15 * time.Second):
		t.Fatal("sampler blocked behind the ingest transaction instead of skipping persistence")
	}

	if got := s.Latest().TS; got == 0 {
		t.Fatal("sampler dropped the in-memory snapshot along with the skipped row")
	}
	if got := s.SkippedSamples(); got != 1 {
		t.Fatalf("skipped samples = %d, want 1", got)
	}
	var rows int
	if err := database.QueryRow(`SELECT count(*) FROM system_samples`).Scan(&rows); err != nil {
		t.Fatalf("count samples: %v", err)
	}
	if rows != 0 {
		t.Fatalf("system_samples rows = %d during ingest, want 0", rows)
	}
}

// With nothing contending, the same path persists and counts no skip.
func TestSamplerPersistsWhenTheWriterIsFree(t *testing.T) {
	database, err := db.Open(filepath.Join(t.TempDir(), "sysmon.db"))
	if err != nil {
		t.Fatalf("open: %v", err)
	}
	defer database.Close()

	s := New(database, t.TempDir(), t.TempDir(), time.Second)
	s.sampleOnce()

	if got := s.SkippedSamples(); got != 0 {
		t.Fatalf("skipped samples = %d with no contention, want 0", got)
	}
	var rows int
	if err := database.QueryRow(`SELECT count(*) FROM system_samples`).Scan(&rows); err != nil {
		t.Fatalf("count samples: %v", err)
	}
	if rows != 1 {
		t.Fatalf("system_samples rows = %d, want 1", rows)
	}
}
