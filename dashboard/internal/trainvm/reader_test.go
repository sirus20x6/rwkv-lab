package trainvm

import (
	"context"
	"database/sql"
	"path/filepath"
	"testing"
)

func fixture(t *testing.T) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "trainvm.db")
	db, err := sql.Open("sqlite", path)
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()
	ddl := `
		CREATE TABLE run_projection (
		  run_id TEXT PRIMARY KEY, experiment_name TEXT, plan_hash TEXT,
		  desired_state TEXT, observed_state TEXT, current_node_id TEXT,
		  current_attempt_id TEXT, run_revision INTEGER, optimizer_step INTEGER,
		  last_heartbeat_ns INTEGER, last_event_sequence INTEGER, failure_summary TEXT
		);
		CREATE TABLE events (
		  journal_sequence INTEGER PRIMARY KEY, event_id TEXT, run_id TEXT,
		  run_revision INTEGER, plan_revision INTEGER, node_id TEXT, attempt_id TEXT,
		  worker_sequence INTEGER, event_type TEXT, event_version INTEGER,
		  wall_time_ns INTEGER, monotonic_time_ns INTEGER, optimizer_step INTEGER,
		  payload_json TEXT
		);
		INSERT INTO run_projection VALUES
		  ('run-a','mageflow','hash-a','running','running','train','train@1',3,120,0,7,''),
		  ('run-b','mageflow','hash-b','running','completed','','',5,500,0,12,'');
		INSERT INTO events VALUES
		  (7,'a-7','run-a',3,1,'train','train@1',1,'worker.heartbeat',1,10,11,120,'{"loss":1.5}'),
		  (8,'a-8','run-a',3,1,'train','train@1',2,'worker.completed',1,12,13,125,'{"reason":"done"}'),
		  (12,'b-12','run-b',5,1,'','','0','run.observed_state_changed',1,14,15,NULL,'{"state":"completed"}');`
	if _, err := db.Exec(ddl); err != nil {
		t.Fatalf("create fixture: %v", err)
	}
	return path
}

func TestReadOnlyRunProjectionAndTimeline(t *testing.T) {
	reader, err := Open(fixture(t))
	if err != nil {
		t.Fatal(err)
	}
	defer reader.Close()
	ctx := context.Background()
	runs, err := reader.Runs(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if len(runs) != 2 || runs[0].RunID != "run-b" || runs[1].CurrentAttemptID != "train@1" {
		t.Fatalf("unexpected runs: %#v", runs)
	}
	run, found, err := reader.Run(ctx, "run-a")
	if err != nil || !found || run.RunRevision != 3 || run.OptimizerStep != 120 {
		t.Fatalf("unexpected run: %#v found=%v err=%v", run, found, err)
	}
	if _, found, err := reader.Run(ctx, "missing"); err != nil || found {
		t.Fatalf("missing run found=%v err=%v", found, err)
	}
	events, err := reader.Timeline(ctx, "run-a", 7, 10)
	if err != nil {
		t.Fatal(err)
	}
	if len(events) != 1 || events[0].EventID != "a-8" || events[0].OptimizerStep == nil ||
		*events[0].OptimizerStep != 125 || string(events[0].Payload) != `{"reason":"done"}` {
		t.Fatalf("unexpected timeline: %#v", events)
	}
	if _, err := reader.db.Exec(`DELETE FROM events`); err == nil {
		t.Fatal("read-only TrainVM connection unexpectedly accepted a write")
	}
}

func TestOpenMissingJournalFails(t *testing.T) {
	if reader, err := Open(filepath.Join(t.TempDir(), "missing.db")); err == nil {
		reader.Close()
		t.Fatal("opening a missing read-only journal unexpectedly succeeded")
	}
}
