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
		CREATE TABLE journal_meta (key TEXT PRIMARY KEY, value TEXT);
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
		CREATE TABLE compiled_plans (
		  plan_hash TEXT PRIMARY KEY, experiment_name TEXT, canonical_plan_json TEXT
		);
		CREATE TABLE control_commands (
		  command_id TEXT PRIMARY KEY, run_id TEXT, control_revision INTEGER,
		  apply_point TEXT, assignments_json TEXT, author TEXT, reason TEXT,
		  status TEXT, effective_step INTEGER, effective_values_json TEXT,
		  diagnostics_json TEXT
		);
		INSERT INTO run_projection VALUES
		  ('run-a','mageflow','3db87fc6e885535afe28e5639a1512f00aafbf75d1dae317655c10a2aec00a12','running','running','train','train@1',3,120,0,7,''),
		  ('run-b','mageflow','hash-b','running','completed','','',5,500,0,12,'');
		INSERT INTO events VALUES
		  (7,'a-7','run-a',3,1,'train','train@1',1,'worker.heartbeat',1,10,11,120,'{"loss":1.5}'),
		  (8,'a-8','run-a',3,1,'train','train@1',2,'worker.completed',1,12,13,125,'{"reason":"done"}'),
		  (12,'b-12','run-b',5,1,'','','0','run.observed_state_changed',1,14,15,NULL,'{"state":"completed"}');
		INSERT INTO compiled_plans VALUES
		  ('3db87fc6e885535afe28e5639a1512f00aafbf75d1dae317655c10a2aec00a12','mageflow','{"spec":{"controls":{"catalog":{"learning_rate":{"type":"number","default":0.001,"minimum":0,"maximum":1,"apply":"next_optimizer_step","mutable_after_start":true}}}}}');
		INSERT INTO control_commands VALUES
		  ('control-1','run-a',1,'next_optimizer_step','{"learning_rate":0.0005}','operator','tune','applied',125,'{"learning_rate":0.0005}','[]'),
		  ('control-2','run-a',2,'next_optimizer_step','{"learning_rate":0.00025}','operator','tune again','requested',NULL,'{}','[]');
		INSERT INTO journal_meta VALUES ('journal_id','0123456789abcdef0123456789abcdef');`
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
	if identity, err := reader.JournalID(ctx); err != nil || identity != "0123456789abcdef0123456789abcdef" {
		t.Fatalf("unexpected journal identity %q: %v", identity, err)
	}
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
	controls, found, err := reader.Controls(ctx, "run-a")
	if err != nil || !found || controls.LatestRequestedRevision != 2 ||
		controls.LatestEffectiveRevision != 1 || controls.EffectiveValues["learning_rate"] != 0.0005 ||
		len(controls.Commands) != 2 || controls.Commands[0].Status != "requested" {
		t.Fatalf("unexpected control view: %#v found=%v err=%v", controls, found, err)
	}
}

func TestOpenMissingJournalFails(t *testing.T) {
	if reader, err := Open(filepath.Join(t.TempDir(), "missing.db")); err == nil {
		reader.Close()
		t.Fatal("opening a missing read-only journal unexpectedly succeeded")
	}
}

func TestControlsRejectsTamperedPersistedPlan(t *testing.T) {
	path := fixture(t)
	db, err := sql.Open("sqlite", path)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := db.Exec(`UPDATE compiled_plans SET canonical_plan_json=canonical_plan_json || ' '`); err != nil {
		db.Close()
		t.Fatal(err)
	}
	if err := db.Close(); err != nil {
		t.Fatal(err)
	}
	reader, err := Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer reader.Close()
	if _, _, err := reader.Controls(context.Background(), "run-a"); err == nil {
		t.Fatal("tampered persisted plan unexpectedly reached the dashboard")
	}
}
