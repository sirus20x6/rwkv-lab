package trainvm

import (
	"context"
	"database/sql"
	"encoding/json"
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
	plan, found, err := reader.CompiledPlan(ctx, "run-a")
	if err != nil || !found || plan.JournalID != "0123456789abcdef0123456789abcdef" ||
		plan.RunID != "run-a" || plan.RunRevision != 3 ||
		plan.PlanHash != "3db87fc6e885535afe28e5639a1512f00aafbf75d1dae317655c10a2aec00a12" ||
		!json.Valid(plan.CanonicalPlan) {
		t.Fatalf("unexpected compiled plan: %#v found=%v err=%v", plan, found, err)
	}
	if _, found, err := reader.CompiledPlan(ctx, "missing"); err != nil || found {
		t.Fatalf("missing compiled plan found=%v err=%v", found, err)
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

func TestQueuedAndAcquiringRunsPreserveEmptyAssignmentAndLeaseTimeline(t *testing.T) {
	path := fixture(t)
	db, err := sql.Open("sqlite", path)
	if err != nil {
		t.Fatal(err)
	}
	_, err = db.Exec(`
		INSERT INTO run_projection VALUES
		  ('run-queued','mageflow','hash-q','queued','queued','','',1,0,0,20,''),
		  ('run-acquiring','mageflow','hash-c','running','acquiring','','',3,0,0,23,''),
		  ('run-ready','mageflow','hash-r','running','running','train_to_boundary','train_to_boundary@1',5,0,0,27,'');
		INSERT INTO events VALUES
		  (20,'run-queued:created','run-queued',1,1,'','',0,'run.created',1,20,0,NULL,
		   '{"desired_state":"queued","observed_state":"queued"}'),
		  (21,'run-acquiring:lease-desired','run-acquiring',2,1,'','',0,'run.desired_state_changed',1,21,0,NULL,
		   '{"state":"running"}'),
		  (22,'run-acquiring:lease-acquired','run-acquiring',2,1,'','',0,'resource.lease_acquired',1,22,0,NULL,
		   '{"concurrency_key":"local-gpu-training","lease_id":"lease-1","fencing_token":7}'),
		  (23,'run-acquiring:acquiring','run-acquiring',3,1,'','',0,'run.observed_state_changed',1,23,0,NULL,
		   '{"state":"acquiring","cause_event_id":"run-acquiring:lease-acquired"}'),
		  (24,'run-ready:launch','run-ready',4,1,'train_to_boundary','train_to_boundary@1',0,'worker.launch_requested',1,24,0,NULL,
		   '{"launch_nonce":"nonce-1","fencing_token":7}'),
		  (25,'run-ready:ready','run-ready',4,1,'train_to_boundary','train_to_boundary@1',0,'worker.ready',1,25,0,NULL,
		   '{"cause_event_id":"run-ready:launch","launch_nonce":"nonce-1","fencing_token":7}'),
		  (26,'run-ready:running','run-ready',5,1,'train_to_boundary','train_to_boundary@1',0,'run.observed_state_changed',1,26,0,NULL,
		   '{"state":"running","cause_event_id":"run-ready:ready"}'),
		  (27,'run-ready:entered','run-ready',5,1,'train_to_boundary','train_to_boundary@1',0,'node.entered',1,27,0,NULL,
		   '{"component":"mageflow","operation":"train"}');`)
	if err != nil {
		db.Close()
		t.Fatal(err)
	}
	db.Close()
	reader, err := Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer reader.Close()
	ctx := context.Background()
	queued, found, err := reader.Run(ctx, "run-queued")
	if err != nil || !found || queued.DesiredState != "queued" ||
		queued.ObservedState != "queued" || queued.CurrentNodeID != "" ||
		queued.CurrentAttemptID != "" || queued.RunRevision != 1 {
		t.Fatalf("unexpected queued projection: %#v found=%v err=%v", queued, found, err)
	}
	acquiring, found, err := reader.Run(ctx, "run-acquiring")
	if err != nil || !found || acquiring.DesiredState != "running" ||
		acquiring.ObservedState != "acquiring" || acquiring.CurrentNodeID != "" ||
		acquiring.CurrentAttemptID != "" || acquiring.RunRevision != 3 ||
		acquiring.LastEventSeq != 23 {
		t.Fatalf("unexpected acquiring projection: %#v found=%v err=%v", acquiring, found, err)
	}
	events, err := reader.Timeline(ctx, "run-acquiring", 20, 10)
	if err != nil || len(events) != 3 || events[1].EventType != "resource.lease_acquired" ||
		string(events[1].Payload) != `{"concurrency_key":"local-gpu-training","lease_id":"lease-1","fencing_token":7}` {
		t.Fatalf("unexpected acquisition timeline: %#v err=%v", events, err)
	}
	ready, found, err := reader.Run(ctx, "run-ready")
	if err != nil || !found || ready.DesiredState != "running" ||
		ready.ObservedState != "running" || ready.CurrentNodeID != "train_to_boundary" ||
		ready.CurrentAttemptID != "train_to_boundary@1" || ready.RunRevision != 5 ||
		ready.LastEventSeq != 27 {
		t.Fatalf("unexpected ready projection: %#v found=%v err=%v", ready, found, err)
	}
	readyEvents, err := reader.Timeline(ctx, "run-ready", 20, 10)
	if err != nil || len(readyEvents) != 4 || readyEvents[0].EventType != "worker.launch_requested" ||
		readyEvents[1].EventType != "worker.ready" ||
		readyEvents[2].EventType != "run.observed_state_changed" ||
		readyEvents[3].EventType != "node.entered" {
		t.Fatalf("unexpected readiness timeline: %#v err=%v", readyEvents, err)
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
	if _, _, err := reader.CompiledPlan(context.Background(), "run-a"); err == nil {
		t.Fatal("tampered persisted plan unexpectedly reached the workflow graph")
	}
}
