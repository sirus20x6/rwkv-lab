package server

import (
	"context"
	"database/sql"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	trainvmstore "trainboard/internal/trainvm"

	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

type fakeTrainVMCommander struct {
	request          trainvmstore.ControlRequest
	result           trainvmstore.ControlResult
	submission       trainvmstore.SubmissionRequest
	submissionResult trainvmstore.SubmissionResult
	err              error
}

func (f *fakeTrainVMCommander) SubmitExperiment(_ context.Context,
	request trainvmstore.SubmissionRequest) (trainvmstore.SubmissionResult, error) {
	f.submission = request
	return f.submissionResult, f.err
}

type unreachableTrainVMCommander struct{ fakeTrainVMCommander }

func (*unreachableTrainVMCommander) Reachable(context.Context) bool { return false }

func (f *fakeTrainVMCommander) RequestControls(_ context.Context,
	request trainvmstore.ControlRequest) (trainvmstore.ControlResult, error) {
	f.request = request
	return f.result, f.err
}

func newTrainVMControlRequest(body string) *http.Request {
	request := httptest.NewRequest(http.MethodPost, "/api/trainvm/runs/vm-run/controls",
		strings.NewReader(body))
	request.Host = "127.0.0.1:9124"
	return request
}

func trainVMFixture(t *testing.T) *trainvmstore.Reader {
	t.Helper()
	path := filepath.Join(t.TempDir(), "trainvm.db")
	db, err := sql.Open("sqlite", path)
	if err != nil {
		t.Fatal(err)
	}
	_, err = db.Exec(`
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
		  ('vm-run','mageflow','baa59aa47fcce31100a77393fcaeb04265bbfc3af2235c62a65ba2006225811a','running','running','train','train@1',2,50,0,3,'');
		INSERT INTO events VALUES
		  (3,'result','vm-run',2,1,'train','train@1',1,'worker.heartbeat',1,1,1,50,'{"loss":2.0}');
		INSERT INTO compiled_plans VALUES
		  ('baa59aa47fcce31100a77393fcaeb04265bbfc3af2235c62a65ba2006225811a','mageflow','{"spec":{"controls":{"catalog":{"learning_rate":{"type":"number","default":0.001,"apply":"next_optimizer_step","mutable_after_start":true}}}}}');
		INSERT INTO journal_meta VALUES ('journal_id','0123456789abcdef0123456789abcdef');`)
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

func trainVMAuthoringFixture(t *testing.T) *trainvmstore.Authoring {
	t.Helper()
	directory := t.TempDir()
	schema := filepath.Join(directory, "schema.json")
	example := filepath.Join(directory, "example.json")
	binary := filepath.Join(directory, "trainvm")
	if err := os.WriteFile(schema, []byte(`{"title":"TrainVM"}`), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(example, []byte(`{"kind":"Experiment"}`), 0o600); err != nil {
		t.Fatal(err)
	}
	script := "#!/bin/sh\ncat >/dev/null\nprintf '%s\\n' '{\"valid\":true,\"plan_hash\":\"native-hash\",\"canonical_plan\":{\"kind\":\"Experiment\"}}'\n"
	if err := os.WriteFile(binary, []byte(script), 0o700); err != nil {
		t.Fatal(err)
	}
	return &trainvmstore.Authoring{BinaryPath: binary, SchemaPath: schema, ExamplePath: example}
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
		Enabled   bool               `json:"enabled"`
		JournalID string             `json:"journal_id"`
		Runs      []trainvmstore.Run `json:"runs"`
	}
	if err := json.Unmarshal(response.Body.Bytes(), &listing); err != nil || !listing.Enabled ||
		listing.JournalID != "0123456789abcdef0123456789abcdef" ||
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
	request = httptest.NewRequest(http.MethodGet, "/api/trainvm/runs/vm-run/controls", nil)
	response = httptest.NewRecorder()
	srv.Handler().ServeHTTP(response, request)
	if response.Code != http.StatusOK || !strings.Contains(response.Body.String(), `"learning_rate"`) {
		t.Fatalf("controls status=%d body=%s", response.Code, response.Body.String())
	}
}

func TestTrainVMDisabledIsExplicit(t *testing.T) {
	srv := New(Config{})
	request := httptest.NewRequest(http.MethodGet, "/api/trainvm/runs", nil)
	response := httptest.NewRecorder()
	srv.Handler().ServeHTTP(response, request)
	if response.Code != http.StatusOK ||
		response.Body.String() != "{\"commands_enabled\":false,\"enabled\":false,\"runs\":[]}\n" {
		t.Fatalf("unexpected disabled response: status=%d body=%s", response.Code, response.Body.String())
	}
}

func TestTrainVMRunListingReportsAuthorityReachability(t *testing.T) {
	srv := New(Config{TrainVM: trainVMFixture(t), Commander: &unreachableTrainVMCommander{}})
	request := httptest.NewRequest(http.MethodGet, "/api/trainvm/runs", nil)
	response := httptest.NewRecorder()
	srv.Handler().ServeHTTP(response, request)
	if response.Code != http.StatusOK ||
		!strings.Contains(response.Body.String(), `"commands_enabled":false`) {
		t.Fatalf("unexpected authority readiness response: status=%d body=%s",
			response.Code, response.Body.String())
	}
}

func TestTrainVMAuthoringEndpoints(t *testing.T) {
	srv := New(Config{Authoring: trainVMAuthoringFixture(t)})
	for path, expected := range map[string]string{
		"/api/trainvm/schema":  `"title":"TrainVM"`,
		"/api/trainvm/example": `"kind":"Experiment"`,
	} {
		request := httptest.NewRequest(http.MethodGet, path, nil)
		response := httptest.NewRecorder()
		srv.Handler().ServeHTTP(response, request)
		if response.Code != http.StatusOK || !strings.Contains(response.Body.String(), expected) {
			t.Fatalf("%s status=%d body=%s", path, response.Code, response.Body.String())
		}
	}
	request := httptest.NewRequest(http.MethodPost, "/api/trainvm/compile",
		strings.NewReader(`{"kind":"Experiment"}`))
	response := httptest.NewRecorder()
	srv.Handler().ServeHTTP(response, request)
	if response.Code != http.StatusOK || !strings.Contains(response.Body.String(), `"plan_hash":"native-hash"`) {
		t.Fatalf("compile status=%d body=%s", response.Code, response.Body.String())
	}
	request = httptest.NewRequest(http.MethodPost, "/api/trainvm/compile", strings.NewReader(`{`))
	response = httptest.NewRecorder()
	srv.Handler().ServeHTTP(response, request)
	if response.Code != http.StatusBadRequest {
		t.Fatalf("invalid compile status=%d body=%s", response.Code, response.Body.String())
	}
}

func TestTrainVMCompileMergesAuthorityAdapterPreview(t *testing.T) {
	commander := &fakeTrainVMCommander{submissionResult: trainvmstore.SubmissionResult{
		PlanHash:             "native-hash",
		CanonicalPlan:        `{"kind":"Experiment"}`,
		AdapterLockDigest:    "sha256:adapter-lock",
		CanonicalAdapterLock: `{"api_version":"trainvm.adapter-lock/v1","profiles":[]}`,
		Diagnostics: []trainvmstore.ControlDiagnostic{{
			Severity: "ERROR", Code: "adapter.registry", Path: "/spec/components",
			Message: "profile is not authorized",
		}},
	}}
	srv := New(Config{
		Authoring: trainVMAuthoringFixture(t), Commander: commander,
		TrainVM: trainVMFixture(t),
	})
	request := httptest.NewRequest(http.MethodPost, "/api/trainvm/compile",
		strings.NewReader(`{"kind":"Experiment"}`))
	response := httptest.NewRecorder()
	srv.Handler().ServeHTTP(response, request)
	if response.Code != http.StatusOK ||
		commander.submission.SourceDocument != `{"kind":"Experiment"}` ||
		commander.submission.SourceFormat != "json" || commander.submission.CreateRun ||
		commander.submission.ExpectedJournalID != "0123456789abcdef0123456789abcdef" ||
		!strings.Contains(response.Body.String(), `"adapter_lock_digest":"sha256:adapter-lock"`) ||
		!strings.Contains(response.Body.String(), `"canonical_adapter_lock":"{\"api_version\":`) ||
		!strings.Contains(response.Body.String(), `"valid":false`) ||
		!strings.Contains(response.Body.String(), `"code":"adapter.registry"`) {
		t.Fatalf("authority adapter preview was not merged: status=%d request=%#v body=%s",
			response.Code, commander.submission, response.Body.String())
	}
}

func TestTrainVMCompileRejectsCompilerIdentitySkew(t *testing.T) {
	for name, result := range map[string]trainvmstore.SubmissionResult{
		"plan hash": {
			PlanHash: "authority-hash", CanonicalPlan: `{"kind":"Experiment"}`,
		},
		"canonical plan": {
			PlanHash: "native-hash", CanonicalPlan: `{"kind":"Different"}`,
		},
	} {
		t.Run(name, func(t *testing.T) {
			commander := &fakeTrainVMCommander{submissionResult: result}
			srv := New(Config{
				Authoring: trainVMAuthoringFixture(t), Commander: commander,
				TrainVM: trainVMFixture(t),
			})
			request := httptest.NewRequest(http.MethodPost, "/api/trainvm/compile",
				strings.NewReader(`{"kind":"Experiment"}`))
			response := httptest.NewRecorder()
			srv.Handler().ServeHTTP(response, request)
			if response.Code != http.StatusBadGateway ||
				!strings.Contains(response.Body.String(), "compiler previews disagree") {
				t.Fatalf("compiler skew was not rejected: status=%d body=%s",
					response.Code, response.Body.String())
			}
		})
	}
}

func TestTrainVMControlEndpointUsesNativeCommander(t *testing.T) {
	commander := &fakeTrainVMCommander{result: trainvmstore.ControlResult{
		Disposition: "ACCEPTED", CommandID: "control-1", ControlRevision: 3,
		ApplyPoint: "NEXT_OPTIMIZER_STEP", Status: "REQUESTED",
	}}
	srv := New(Config{Commander: commander, TrainVM: trainVMFixture(t)})
	request := newTrainVMControlRequest(`{
			"expected_run_revision":7,
			"expected_control_revision":2,
			"idempotency_key":"tab-intent",
			"reason":"reduce learning rate",
			"assignments":{"learning_rate":0.00001,"eval_every":250}
		}`)
	request.Header.Set("Content-Type", "application/json")
	response := httptest.NewRecorder()
	srv.Handler().ServeHTTP(response, request)
	if response.Code != http.StatusAccepted || commander.request.RunID != "vm-run" ||
		commander.request.ExpectedJournalID != "0123456789abcdef0123456789abcdef" ||
		commander.request.ExpectedPlanHash != "baa59aa47fcce31100a77393fcaeb04265bbfc3af2235c62a65ba2006225811a" ||
		commander.request.ExpectedRunRevision != 7 ||
		commander.request.ExpectedControlRevision != 2 ||
		commander.request.Author != "dashboard" || commander.request.IdempotencyKey != "tab-intent" {
		t.Fatalf("unexpected command forwarding: status=%d request=%#v body=%s",
			response.Code, commander.request, response.Body.String())
	}
	if _, ok := commander.request.Assignments["eval_every"].(json.Number); !ok {
		t.Fatalf("JSON number type was lost: %#v", commander.request.Assignments)
	}
}

func TestTrainVMSubmissionEndpointUsesNativeCommander(t *testing.T) {
	commander := &fakeTrainVMCommander{submissionResult: trainvmstore.SubmissionResult{
		PlanHash: "native-plan", AdapterLockDigest: "native-lock", Run: &trainvmstore.RunIdentity{
			RunID: "run-new", Revision: 1, PlanHash: "native-plan",
		},
	}}
	srv := New(Config{Commander: commander, TrainVM: trainVMFixture(t)})
	request := httptest.NewRequest(http.MethodPost, "/api/trainvm/experiments",
		strings.NewReader(`{"source_document":"{\"kind\":\"Experiment\"}","source_format":"json","create_run":true,"idempotency_key":"submit-1","expected_journal_id":"0123456789abcdef0123456789abcdef","expected_plan_hash":"native-plan","expected_adapter_lock_digest":"native-lock","reason":"launch test"}`))
	request.Host = "127.0.0.1:9124"
	request.Header.Set("Content-Type", "application/json")
	response := httptest.NewRecorder()
	srv.Handler().ServeHTTP(response, request)
	if response.Code != http.StatusAccepted ||
		commander.submission.SourceDocument != `{"kind":"Experiment"}` ||
		commander.submission.SourceFormat != "json" || !commander.submission.CreateRun ||
		commander.submission.IdempotencyKey != "submit-1" || commander.submission.Author != "dashboard" ||
		commander.submission.ExpectedJournalID != "0123456789abcdef0123456789abcdef" ||
		commander.submission.ExpectedPlanHash != "native-plan" ||
		commander.submission.ExpectedAdapterLockDigest != "native-lock" ||
		!strings.Contains(response.Body.String(), `"run_id":"run-new"`) {
		t.Fatalf("unexpected submission forwarding: status=%d request=%#v body=%s",
			response.Code, commander.submission, response.Body.String())
	}
}

func TestTrainVMSubmissionRejectsStaleAuthorityIdentity(t *testing.T) {
	commander := &fakeTrainVMCommander{}
	srv := New(Config{Commander: commander, TrainVM: trainVMFixture(t)})
	request := httptest.NewRequest(http.MethodPost, "/api/trainvm/experiments",
		strings.NewReader(`{"source_document":"{}","source_format":"json","create_run":true,"idempotency_key":"submit-1","expected_journal_id":"stale-journal","expected_plan_hash":"native-plan","expected_adapter_lock_digest":"native-lock","reason":"launch test"}`))
	request.Host = "127.0.0.1:9124"
	request.Header.Set("Content-Type", "application/json")
	response := httptest.NewRecorder()
	srv.Handler().ServeHTTP(response, request)
	if response.Code != http.StatusConflict || commander.submission.SourceDocument != "" {
		t.Fatalf("stale submission reached authority: status=%d request=%#v body=%s",
			response.Code, commander.submission, response.Body.String())
	}
}

func TestTrainVMSubmissionWithoutCreatedRunIsUnprocessable(t *testing.T) {
	commander := &fakeTrainVMCommander{submissionResult: trainvmstore.SubmissionResult{
		PlanHash: "native-plan", AdapterLockDigest: "native-lock",
		Diagnostics: []trainvmstore.ControlDiagnostic{{
			Severity: "ERROR", Code: "experiment.invalid", Message: "draft rejected",
		}},
	}}
	srv := New(Config{Commander: commander, TrainVM: trainVMFixture(t)})
	request := httptest.NewRequest(http.MethodPost, "/api/trainvm/experiments",
		strings.NewReader(`{"source_document":"{}","source_format":"json","create_run":true,"idempotency_key":"submit-1","expected_journal_id":"0123456789abcdef0123456789abcdef","expected_plan_hash":"native-plan","expected_adapter_lock_digest":"native-lock","reason":"launch test"}`))
	request.Host = "127.0.0.1:9124"
	request.Header.Set("Content-Type", "application/json")
	response := httptest.NewRecorder()
	srv.Handler().ServeHTTP(response, request)
	if response.Code != http.StatusUnprocessableEntity ||
		!strings.Contains(response.Body.String(), `"code":"experiment.invalid"`) {
		t.Fatalf("invalid native result was not preserved: status=%d body=%s",
			response.Code, response.Body.String())
	}
}

func TestTrainVMSubmissionRejectsMismatchedPreviewResult(t *testing.T) {
	commander := &fakeTrainVMCommander{submissionResult: trainvmstore.SubmissionResult{
		PlanHash: "different-plan", AdapterLockDigest: "native-lock",
		Run: &trainvmstore.RunIdentity{RunID: "run-new", Revision: 1, PlanHash: "different-plan"},
	}}
	srv := New(Config{Commander: commander, TrainVM: trainVMFixture(t)})
	request := httptest.NewRequest(http.MethodPost, "/api/trainvm/experiments",
		strings.NewReader(`{"source_document":"{}","source_format":"json","create_run":true,"idempotency_key":"submit-1","expected_journal_id":"0123456789abcdef0123456789abcdef","expected_plan_hash":"native-plan","expected_adapter_lock_digest":"native-lock","reason":"launch test"}`))
	request.Host = "127.0.0.1:9124"
	request.Header.Set("Content-Type", "application/json")
	response := httptest.NewRecorder()
	srv.Handler().ServeHTTP(response, request)
	if response.Code != http.StatusBadGateway {
		t.Fatalf("mismatched native identity accepted: status=%d body=%s",
			response.Code, response.Body.String())
	}
}

func TestTrainVMSubmissionEndpointSharesStrictMutationBoundary(t *testing.T) {
	commander := &fakeTrainVMCommander{}
	srv := New(Config{Commander: commander, TrainVM: trainVMFixture(t)})
	for name, test := range map[string]struct {
		body        string
		contentType string
		host        string
		expected    int
	}{
		"unknown field": {body: `{"mystery":true}`, contentType: "application/json", host: "127.0.0.1", expected: http.StatusBadRequest},
		"trailing JSON": {body: `{}` + ` {}`, contentType: "application/json", host: "127.0.0.1", expected: http.StatusBadRequest},
		"wrong type":    {body: `{}`, contentType: "text/plain", host: "127.0.0.1", expected: http.StatusUnsupportedMediaType},
		"rebound host":  {body: `{}`, contentType: "application/json", host: "attacker.invalid", expected: http.StatusForbidden},
	} {
		t.Run(name, func(t *testing.T) {
			request := httptest.NewRequest(http.MethodPost, "/api/trainvm/experiments", strings.NewReader(test.body))
			request.Host = test.host
			request.Header.Set("Content-Type", test.contentType)
			response := httptest.NewRecorder()
			srv.Handler().ServeHTTP(response, request)
			if response.Code != test.expected {
				t.Fatalf("status=%d expected=%d body=%s", response.Code, test.expected, response.Body.String())
			}
		})
	}
}

func TestTrainVMControlEndpointRejectsUnsafeHTTPInput(t *testing.T) {
	commander := &fakeTrainVMCommander{result: trainvmstore.ControlResult{Disposition: "ACCEPTED"}}
	srv := New(Config{Commander: commander, TrainVM: trainVMFixture(t)})
	for name, body := range map[string]string{
		"unknown field": `{"expected_run_revision":1,"mystery":true}`,
		"trailing JSON": `{"expected_run_revision":1} {}`,
		"oversize":      `{"reason":"` + strings.Repeat("x", trainVMCommandLimit) + `"}`,
	} {
		t.Run(name, func(t *testing.T) {
			request := newTrainVMControlRequest(body)
			request.Header.Set("Content-Type", "application/json")
			response := httptest.NewRecorder()
			srv.Handler().ServeHTTP(response, request)
			expected := http.StatusBadRequest
			if name == "oversize" {
				expected = http.StatusRequestEntityTooLarge
			}
			if response.Code != expected {
				t.Fatalf("status=%d body=%s", response.Code, response.Body.String())
			}
		})
	}
}

func TestTrainVMControlEndpointEnforcesBrowserRequestBoundary(t *testing.T) {
	requestBody := `{"expected_run_revision":1,"expected_control_revision":0,"idempotency_key":"key","reason":"test","assignments":{"rate":1}}`
	for name, test := range map[string]struct {
		configure func(*http.Request)
		expected  int
	}{
		"missing content type": {configure: func(*http.Request) {}, expected: http.StatusUnsupportedMediaType},
		"foreign origin": {configure: func(r *http.Request) {
			r.Header.Set("Content-Type", "application/json")
			r.Header.Set("Origin", "https://attacker.invalid")
		}, expected: http.StatusForbidden},
		"cross-scheme origin": {configure: func(r *http.Request) {
			r.Header.Set("Content-Type", "application/json")
			r.Header.Set("Origin", "https://127.0.0.1:9124")
		}, expected: http.StatusForbidden},
		"rebound non-loopback host": {configure: func(r *http.Request) {
			r.Host = "attacker.invalid"
			r.Header.Set("Content-Type", "application/json")
			r.Header.Set("Origin", "http://attacker.invalid")
			r.Header.Set("Sec-Fetch-Site", "same-origin")
		}, expected: http.StatusForbidden},
		"cross-site fetch": {configure: func(r *http.Request) {
			r.Header.Set("Content-Type", "application/json")
			r.Header.Set("Sec-Fetch-Site", "cross-site")
		}, expected: http.StatusForbidden},
		"same origin": {configure: func(r *http.Request) {
			r.Header.Set("Content-Type", "application/json; charset=utf-8")
			r.Header.Set("Origin", "http://127.0.0.1:9124")
			r.Header.Set("Sec-Fetch-Site", "same-origin")
		}, expected: http.StatusAccepted},
	} {
		t.Run(name, func(t *testing.T) {
			commander := &fakeTrainVMCommander{result: trainvmstore.ControlResult{Disposition: "ACCEPTED"}}
			srv := New(Config{Commander: commander, TrainVM: trainVMFixture(t)})
			request := newTrainVMControlRequest(requestBody)
			test.configure(request)
			response := httptest.NewRecorder()
			srv.Handler().ServeHTTP(response, request)
			if response.Code != test.expected {
				t.Fatalf("status=%d expected=%d body=%s", response.Code, test.expected, response.Body.String())
			}
		})
	}
}

func TestTrainVMControlEndpointStatusMapping(t *testing.T) {
	for name, test := range map[string]struct {
		commander trainvmstore.Commander
		expected  int
	}{
		"disabled":    {commander: nil, expected: http.StatusServiceUnavailable},
		"unavailable": {commander: &fakeTrainVMCommander{err: status.Error(codes.Unavailable, "offline")}, expected: http.StatusServiceUnavailable},
		"validation":  {commander: &fakeTrainVMCommander{err: &trainvmstore.ValidationError{Message: "bad request"}}, expected: http.StatusBadRequest},
		"conflict":    {commander: &fakeTrainVMCommander{result: trainvmstore.ControlResult{Disposition: "CONFLICT"}}, expected: http.StatusConflict},
		"rejected":    {commander: &fakeTrainVMCommander{result: trainvmstore.ControlResult{Disposition: "REJECTED"}}, expected: http.StatusUnprocessableEntity},
	} {
		t.Run(name, func(t *testing.T) {
			config := Config{Commander: test.commander}
			if test.commander != nil {
				config.TrainVM = trainVMFixture(t)
			}
			srv := New(config)
			request := newTrainVMControlRequest(`{"expected_run_revision":1,"expected_control_revision":0,"idempotency_key":"key","reason":"test","assignments":{"rate":1}}`)
			request.Header.Set("Content-Type", "application/json")
			response := httptest.NewRecorder()
			srv.Handler().ServeHTTP(response, request)
			if response.Code != test.expected {
				t.Fatalf("status=%d expected=%d body=%s", response.Code, test.expected, response.Body.String())
			}
		})
	}
}
