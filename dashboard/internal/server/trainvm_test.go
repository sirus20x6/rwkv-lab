package server

import (
	"database/sql"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"testing"

	trainvmstore "trainboard/internal/trainvm"
)

func trainVMFixture(t *testing.T) *trainvmstore.Reader {
	t.Helper()
	path := filepath.Join(t.TempDir(), "trainvm.db")
	db, err := sql.Open("sqlite", path)
	if err != nil {
		t.Fatal(err)
	}
	_, err = db.Exec(`
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
		  ('vm-run','mageflow','abc','running','running','train','train@1',2,50,0,3,'');
		INSERT INTO events VALUES
		  (3,'result','vm-run',2,1,'train','train@1',1,'worker.heartbeat',1,1,1,50,'{"loss":2.0}');`)
	if err != nil {
		t.Fatalf("create TrainVM server fixture: %v", err)
	}
	db.Close()
	reader, err := trainvmstore.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { reader.Close() })
	return reader
}

func TestTrainVMReadModelEndpoints(t *testing.T) {
	srv := New(Config{TrainVM: trainVMFixture(t)})

	request := httptest.NewRequest(http.MethodGet, "/api/trainvm/runs", nil)
	response := httptest.NewRecorder()
	srv.Handler().ServeHTTP(response, request)
	if response.Code != http.StatusOK {
		t.Fatalf("runs status=%d body=%s", response.Code, response.Body.String())
	}
	var listing struct {
		Enabled bool               `json:"enabled"`
		Runs    []trainvmstore.Run `json:"runs"`
	}
	if err := json.Unmarshal(response.Body.Bytes(), &listing); err != nil || !listing.Enabled ||
		len(listing.Runs) != 1 || listing.Runs[0].RunID != "vm-run" {
		t.Fatalf("unexpected listing: %#v err=%v", listing, err)
	}

	request = httptest.NewRequest(http.MethodGet, "/api/trainvm/runs/vm-run", nil)
	response = httptest.NewRecorder()
	srv.Handler().ServeHTTP(response, request)
	if response.Code != http.StatusOK || !json.Valid(response.Body.Bytes()) {
		t.Fatalf("run status=%d body=%s", response.Code, response.Body.String())
	}

	request = httptest.NewRequest(http.MethodGet, "/api/trainvm/runs/vm-run/timeline?after=2&limit=10", nil)
	response = httptest.NewRecorder()
	srv.Handler().ServeHTTP(response, request)
	var events []trainvmstore.Event
	if err := json.Unmarshal(response.Body.Bytes(), &events); err != nil || len(events) != 1 ||
		events[0].EventID != "result" {
		t.Fatalf("unexpected timeline: %#v err=%v body=%s", events, err, response.Body.String())
	}
}

func TestTrainVMDisabledIsExplicit(t *testing.T) {
	srv := New(Config{})
	request := httptest.NewRequest(http.MethodGet, "/api/trainvm/runs", nil)
	response := httptest.NewRecorder()
	srv.Handler().ServeHTTP(response, request)
	if response.Code != http.StatusOK || response.Body.String() != "{\"enabled\":false,\"runs\":[]}\n" {
		t.Fatalf("unexpected disabled response: status=%d body=%s", response.Code, response.Body.String())
	}
}
