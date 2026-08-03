package server

import (
	"context"
	"crypto/sha256"
	"database/sql"
	"encoding/json"
	"fmt"
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
	actionRequest    trainvmstore.RunActionRequest
	actionResult     trainvmstore.RunActionResult
	submission       trainvmstore.SubmissionRequest
	submissionResult trainvmstore.SubmissionResult
	planDiff         trainvmstore.PlanDiffRequest
	planDiffResult   trainvmstore.PlanDiffResult
	descriptor       trainvmstore.DescriptorRequest
	descriptorResult trainvmstore.DescriptorResult
	hostAuthority    trainvmstore.HostAuthorityStatus
	err              error
}

func (f *fakeTrainVMCommander) GetHostAuthorityStatus(_ context.Context) (trainvmstore.HostAuthorityStatus, error) {
	return f.hostAuthority, f.err
}

func (f *fakeTrainVMCommander) DiffPlan(_ context.Context,
	request trainvmstore.PlanDiffRequest) (trainvmstore.PlanDiffResult, error) {
	f.planDiff = request
	return f.planDiffResult, f.err
}

func (f *fakeTrainVMCommander) GetDescriptor(_ context.Context,
	request trainvmstore.DescriptorRequest) (trainvmstore.DescriptorResult, error) {
	f.descriptor = request
	return f.descriptorResult, f.err
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

func (f *fakeTrainVMCommander) RequestRunAction(_ context.Context,
	request trainvmstore.RunActionRequest) (trainvmstore.RunActionResult, error) {
	f.actionRequest = request
	return f.actionResult, f.err
}

func newTrainVMControlRequest(body string) *http.Request {
	request := httptest.NewRequest(http.MethodPost, "/api/trainvm/runs/vm-run/controls",
		strings.NewReader(body))
	request.Host = "127.0.0.1:9124"
	return request
}

func TestTrainVMHostAuthorityEndpointUsesOnlyNativeCommanderEvidence(t *testing.T) {
	expected := trainvmstore.HostAuthorityStatus{
		APIVersion:   "trainvm.hostd-authority-status/v1",
		Coordinator:  trainvmstore.HostdCoordinatorStatus{Lifecycle: "admitting", HostID: "host-1"},
		StartupPhase: "admitting", LedgerVerified: true,
		ActiveFences:         []trainvmstore.HostResourceFenceStatus{},
		ActiveProcesses:      []trainvmstore.HostdProcessAuthorityStatus{},
		ProcessLaunchEnabled: true, MutationEnabled: true,
	}
	commander := &fakeTrainVMCommander{hostAuthority: expected}
	response := httptest.NewRecorder()
	request := httptest.NewRequest(http.MethodGet, "/api/trainvm/host-authority", nil)
	New(Config{Commander: commander}).Handler().ServeHTTP(response, request)
	if response.Code != http.StatusOK || response.Header().Get("Cache-Control") != "no-store" {
		t.Fatalf("unexpected host authority response: status=%d headers=%v body=%s", response.Code, response.Header(), response.Body.String())
	}
	var observed trainvmstore.HostAuthorityStatus
	if err := json.Unmarshal(response.Body.Bytes(), &observed); err != nil ||
		observed.APIVersion != expected.APIVersion || !observed.MutationEnabled ||
		observed.Coordinator.HostID != "host-1" {
		t.Fatalf("unexpected host authority payload: %#v err=%v", observed, err)
	}

	unconfigured := httptest.NewRecorder()
	New(Config{}).Handler().ServeHTTP(unconfigured,
		httptest.NewRequest(http.MethodGet, "/api/trainvm/host-authority", nil))
	if unconfigured.Code != http.StatusServiceUnavailable {
		t.Fatalf("unconfigured host authority endpoint returned %d", unconfigured.Code)
	}

	commander.err = status.Error(codes.FailedPrecondition, "hostd status source is absent")
	failedPrecondition := httptest.NewRecorder()
	New(Config{Commander: commander}).Handler().ServeHTTP(failedPrecondition,
		httptest.NewRequest(http.MethodGet, "/api/trainvm/host-authority", nil))
	if failedPrecondition.Code != http.StatusServiceUnavailable {
		t.Fatalf("missing hostd status source returned %d, want 503", failedPrecondition.Code)
	}
}

func newTrainVMActionRequest(body string) *http.Request {
	request := httptest.NewRequest(http.MethodPost, "/api/trainvm/runs/vm-run/actions",
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
		  ('vm-run','mageflow','3ca70ff811ace5137501dfe841bf8d1914ecd9083e9e6f68c521aaedf90d5b0d','running','running','train','train@1',2,50,1,5,'');
		INSERT INTO events VALUES
		  (3,'result','vm-run',2,1,'train','train@1',1,'worker.heartbeat',1,1,1,50,'{"phase":"train","observed_at_ns":1}'),
		  (4,'metric','vm-run',2,1,'train','train@1',2,'metric.sampled',1,2,2,51,'{"name":"loss","value":1.5,"unit":"loss","step_domain":"optimizer_step","step":51,"sample_weight":1,"labels":{"route":"animation"}}'),
		  (5,'artifact','vm-run',2,1,'train','train@1',3,'artifact.published',1,3,3,NULL,'{"artifact_id":"gallery-51","logical_name":"eval/gallery","kind":"image_gallery","schema":"trainvm.eval-gallery/v1","uri":"file:///sealed/gallery-51","size_bytes":4096,"fingerprint_algorithm":"sha256","fingerprint":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","complete":true,"producer_node_id":"train","producer_attempt_id":"train@1","parent_artifact_ids":[],"published_at_ns":3}');
		INSERT INTO compiled_plans VALUES
		  ('3ca70ff811ace5137501dfe841bf8d1914ecd9083e9e6f68c521aaedf90d5b0d','mageflow','{"spec":{"controls":{"catalog":{"learning_rate":{"type":"number","default":0.001,"apply":"next_optimizer_step","mutable_after_start":true}}},"observability":{"heartbeat_seconds":5,"metrics":[{"name":"loss","type":"gauge","unit":"loss","step_domain":"optimizer_step","aggregation":"mean"}],"retain_raw_metrics_days":30}}}');
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
	return trainVMAuthoringFixtureWithPlan(t, `{"kind":"Experiment"}`)
}

func trainVMAuthoringFixtureWithPlan(t *testing.T, canonicalPlan string) *trainvmstore.Authoring {
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
	script := "#!/bin/sh\ncat >/dev/null\nprintf '%s\\n' '{\"valid\":true,\"plan_hash\":\"native-hash\",\"canonical_plan\":" + canonicalPlan + "}'\n"
	if err := os.WriteFile(binary, []byte(script), 0o700); err != nil {
		t.Fatal(err)
	}
	return &trainvmstore.Authoring{BinaryPath: binary, SchemaPath: schema, ExamplePath: example}
}

func canonicalTestJSON(t *testing.T, value any) string {
	t.Helper()
	encoded, err := json.Marshal(value)
	if err != nil {
		t.Fatal(err)
	}
	return string(encoded)
}

func operationDescriptorFixture(adapter string) map[string]any {
	return map[string]any{
		"key": map[string]any{
			"adapter": adapter, "version": "1.0.0", "runtime": "python_worker",
			"operation": "train", "contract": "trainvm.test.Train",
		},
		"effect": "process", "idempotency": "receipt_required",
		"code_fingerprint":      "sha256:" + strings.Repeat("a", 64),
		"required_capabilities": []any{"trainer.test.v1"},
		"lifecycle": map[string]any{
			"stateful": true, "graceful_stop": true, "checkpoint_now": true,
			"pause_keep_resources": true, "pause_release_resources": true,
			"compile": false, "warmup": false, "qualify": false, "profile": false,
			"resume_grade": "exact",
		},
		"authoring": map[string]any{
			"inputs": map[string]any{
				"dataset": map[string]any{
					"type": "artifact", "required": true,
					"artifact_type": "dataset", "artifact_schema": "trainvm.dataset/v1",
				},
			},
			"outputs": map[string]any{
				"checkpoint": map[string]any{
					"type": "artifact", "required": true, "artifact_type": "checkpoint",
				},
			},
		},
		"training_composition": map[string]any{
			"model_family": "rwkv", "slots": map[string]any{"optimizer": "optimizer"},
		},
	}
}

func trainingComponentDescriptorFixture() map[string]any {
	return map[string]any{
		"key": map[string]any{
			"category": "optimizer", "name": "test_adamw", "version": "1.0.0",
		},
		"backend": "python", "implementation": "trainer.optimizer.test_adamw.v1",
		"model_families":        []any{"rwkv"},
		"required_capabilities": []any{"optimizer.test_adamw.v1"},
		"configuration": []any{
			map[string]any{
				"name": "learning_rate", "type": "number", "required": true,
				"minimum": 0.0, "maximum": 1.0,
			},
			map[string]any{
				"name": "weight_decay", "type": "number", "required": false,
				"default": 0.01, "minimum": 0.0, "maximum": 10.0,
			},
		},
		"state": []any{map[string]any{
			"name": "parameter_state_manifest", "type": "string", "required": true,
		}},
		"state_grade": "exact", "reference_implementation": true,
	}
}

func coherentTrainingPreviewFixture(t *testing.T) (string, string) {
	t.Helper()
	descriptor := trainingComponentDescriptorFixture()
	descriptorJSON := canonicalTestJSON(t, descriptor)
	registryDigest := "sha256:" + strings.Repeat("b", 64)
	key := descriptor["key"]
	requestedConfiguration := map[string]any{"learning_rate": 0.1}
	resolvedConfiguration := map[string]any{"learning_rate": 0.1, "weight_decay": 0.01}
	plan := map[string]any{
		"spec": map[string]any{
			"workflow": map[string]any{
				"nodes": map[string]any{
					"train": map[string]any{
						"invoke": map[string]any{
							"training": map[string]any{
								"model_family": "rwkv",
								"components": map[string]any{
									"optimizer": map[string]any{
										"key": key, "configuration": requestedConfiguration,
									},
								},
							},
						},
					},
				},
			},
		},
	}
	resolvedComponents := map[string]any{
		"optimizer": map[string]any{
			"configuration": resolvedConfiguration,
			"descriptor":    descriptor,
			"descriptor_digest": fmt.Sprintf("sha256:%x",
				sha256.Sum256([]byte(descriptorJSON))),
		},
	}
	compositionBody := map[string]any{
		"api_version": "trainvm.resolved-training-composition/v1",
		"components":  resolvedComponents, "model_family": "rwkv",
		"registry_digest": registryDigest,
	}
	compositionJSON := canonicalTestJSON(t, compositionBody)
	composition := map[string]any{}
	for key, value := range compositionBody {
		composition[key] = value
	}
	composition["composition_digest"] = fmt.Sprintf("sha256:%x",
		sha256.Sum256([]byte(compositionJSON)))
	lock := map[string]any{
		"api_version":     "trainvm.training-component-lock/v1",
		"nodes":           map[string]any{"train": composition},
		"registry_digest": registryDigest,
	}
	return canonicalTestJSON(t, plan), canonicalTestJSON(t, lock)
}

func mutateTestJSONObject(t *testing.T, source string, mutate func(map[string]any)) string {
	t.Helper()
	var document map[string]any
	if err := json.Unmarshal([]byte(source), &document); err != nil {
		t.Fatal(err)
	}
	mutate(document)
	return canonicalTestJSON(t, document)
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

	request = httptest.NewRequest(http.MethodGet, "/api/trainvm/runs/vm-run/plan", nil)
	response = httptest.NewRecorder()
	srv.Handler().ServeHTTP(response, request)
	var plan trainvmstore.CompiledPlanView
	if err := json.Unmarshal(response.Body.Bytes(), &plan); err != nil ||
		response.Code != http.StatusOK || plan.RunID != "vm-run" || plan.RunRevision != 2 ||
		plan.PlanHash != "3ca70ff811ace5137501dfe841bf8d1914ecd9083e9e6f68c521aaedf90d5b0d" ||
		!json.Valid(plan.CanonicalPlan) {
		t.Fatalf("unexpected compiled plan: %#v err=%v body=%s", plan, err, response.Body.String())
	}

	request = httptest.NewRequest(http.MethodGet, "/api/trainvm/runs/vm-run/timeline?after=2&limit=1", nil)
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
	request = httptest.NewRequest(http.MethodGet, "/api/trainvm/runs/vm-run/metrics?after=3&limit=10", nil)
	response = httptest.NewRecorder()
	srv.Handler().ServeHTTP(response, request)
	var metrics []trainvmstore.MetricPoint
	if err := json.Unmarshal(response.Body.Bytes(), &metrics); err != nil ||
		response.Code != http.StatusOK || len(metrics) != 1 ||
		metrics[0].Name != "loss" || metrics[0].Step != 51 {
		t.Fatalf("unexpected metrics: %#v err=%v body=%s", metrics, err, response.Body.String())
	}
	request = httptest.NewRequest(http.MethodGet, "/api/trainvm/runs/vm-run/artifacts?after=3&limit=10", nil)
	response = httptest.NewRecorder()
	srv.Handler().ServeHTTP(response, request)
	var artifacts []trainvmstore.ObservableArtifact
	if err := json.Unmarshal(response.Body.Bytes(), &artifacts); err != nil ||
		response.Code != http.StatusOK || len(artifacts) != 1 ||
		artifacts[0].ArtifactID != "gallery-51" || artifacts[0].Kind != "image_gallery" ||
		strings.Contains(response.Body.String(), "file:///") {
		t.Fatalf("unexpected artifacts: %#v err=%v body=%s", artifacts, err, response.Body.String())
	}
	request = httptest.NewRequest(http.MethodGet,
		"/api/trainvm/runs/vm-run/observability?after=0&limit=10", nil)
	response = httptest.NewRecorder()
	srv.Handler().ServeHTTP(response, request)
	var snapshot trainvmstore.TelemetrySnapshot
	if err := json.Unmarshal(response.Body.Bytes(), &snapshot); err != nil ||
		response.Code != http.StatusOK || !snapshot.CaughtUp || snapshot.ReplayPending ||
		snapshot.TargetSequence != 5 || snapshot.NextSequence != 5 ||
		len(snapshot.Heartbeats) != 1 || snapshot.Heartbeats[0].Phase != "train" ||
		len(snapshot.Metrics) != 1 || snapshot.Metrics[0].Type != "gauge" ||
		len(snapshot.Artifacts) != 1 || snapshot.Artifacts[0].ArtifactID != "gallery-51" ||
		strings.Contains(response.Body.String(), "file:///") {
		t.Fatalf("unexpected observability snapshot: %#v err=%v body=%s",
			snapshot, err, response.Body.String())
	}
	request = httptest.NewRequest(http.MethodGet,
		"/api/trainvm/runs/vm-run/capsule", nil)
	response = httptest.NewRecorder()
	srv.Handler().ServeHTTP(response, request)
	var capsule trainvmstore.ReproducibilityCapsule
	if err := json.Unmarshal(response.Body.Bytes(), &capsule); err != nil ||
		response.Code != http.StatusOK ||
		capsule.Body.APIVersion != "trainvm.reproducibility-capsule/v1" ||
		capsule.Body.ThroughSequence != 5 || capsule.Body.PlanHash != plan.PlanHash ||
		len(capsule.Body.LatestMetrics) != 1 || len(capsule.Body.Artifacts) != 1 ||
		strings.Contains(response.Body.String(), "file:///") ||
		response.Header().Get("ETag") != `"`+capsule.CapsuleDigest+`"` ||
		!strings.Contains(response.Header().Get("Content-Disposition"),
			"trainvm-vm-run-capsule.json") {
		t.Fatalf("unexpected reproducibility capsule: %#v err=%v headers=%v body=%s",
			capsule, err, response.Header(), response.Body.String())
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

func TestTrainVMTrainingComponentsComeFromNativeAuthority(t *testing.T) {
	document := `{"api_version":"trainvm.training-components/v1","components":[]}`
	digest := fmt.Sprintf("sha256:%x", sha256.Sum256([]byte(document)))
	commander := &fakeTrainVMCommander{descriptorResult: trainvmstore.DescriptorResult{
		SchemaJSON: document,
		SchemaHash: digest,
	}}
	srv := New(Config{Commander: commander})
	request := httptest.NewRequest(http.MethodGet, "/api/trainvm/training-components", nil)
	response := httptest.NewRecorder()
	srv.Handler().ServeHTTP(response, request)
	if response.Code != http.StatusOK ||
		commander.descriptor.Provider != "trainvm.training-components" ||
		commander.descriptor.Version != "1.0.0" ||
		!strings.Contains(response.Body.String(), `"schema_hash":"`+digest+`"`) ||
		!strings.Contains(response.Body.String(), `"api_version":"trainvm.training-components/v1"`) {
		t.Fatalf("training-component descriptor status=%d request=%#v body=%s",
			response.Code, commander.descriptor, response.Body.String())
	}

	commander.descriptorResult.SchemaHash = "sha256:" + strings.Repeat("0", 64)
	response = httptest.NewRecorder()
	srv.Handler().ServeHTTP(response, request)
	if response.Code != http.StatusBadGateway {
		t.Fatalf("mismatched descriptor identity accepted: status=%d body=%s",
			response.Code, response.Body.String())
	}
}

func TestTrainVMTrainingComponentDescriptorNativeInvariants(t *testing.T) {
	serve := func(t *testing.T, document map[string]any) *httptest.ResponseRecorder {
		t.Helper()
		encoded := canonicalTestJSON(t, document)
		commander := &fakeTrainVMCommander{descriptorResult: trainvmstore.DescriptorResult{
			SchemaJSON: encoded,
			SchemaHash: fmt.Sprintf("sha256:%x", sha256.Sum256([]byte(encoded))),
		}}
		response := httptest.NewRecorder()
		New(Config{Commander: commander}).Handler().ServeHTTP(response,
			httptest.NewRequest(http.MethodGet, "/api/trainvm/training-components", nil))
		return response
	}
	valid := map[string]any{
		"api_version": "trainvm.training-components/v1",
		"components":  []any{trainingComponentDescriptorFixture()},
	}
	if response := serve(t, valid); response.Code != http.StatusOK {
		t.Fatalf("valid native-shape training descriptor rejected: status=%d body=%s",
			response.Code, response.Body.String())
	}
	mutations := map[string]func(map[string]any){
		"wildcard family mixed with exact family": func(component map[string]any) {
			component["model_families"] = []any{"*", "rwkv"}
		},
		"nonnumeric field carries bounds": func(component map[string]any) {
			component["configuration"].([]any)[0].(map[string]any)["type"] = "boolean"
		},
		"inverted numeric bounds": func(component map[string]any) {
			field := component["configuration"].([]any)[0].(map[string]any)
			field["minimum"], field["maximum"] = 2.0, 1.0
		},
		"wrong typed default": func(component map[string]any) {
			component["configuration"].([]any)[0].(map[string]any)["default"] = "fast"
		},
		"default outside numeric bounds": func(component map[string]any) {
			component["configuration"].([]any)[0].(map[string]any)["default"] = 2.0
		},
		"integer default encoded as number": func(component map[string]any) {
			field := component["configuration"].([]any)[0].(map[string]any)
			field["type"], field["default"] = "integer", 1.5
		},
		"enum omits values": func(component map[string]any) {
			field := component["configuration"].([]any)[0].(map[string]any)
			field["type"] = "enumeration"
			delete(field, "minimum")
			delete(field, "maximum")
		},
		"enum values not canonical": func(component map[string]any) {
			field := component["configuration"].([]any)[0].(map[string]any)
			field["type"], field["values"] = "enumeration", []any{"z", "a"}
			delete(field, "minimum")
			delete(field, "maximum")
		},
		"enum default outside values": func(component map[string]any) {
			field := component["configuration"].([]any)[0].(map[string]any)
			field["type"], field["values"], field["default"] = "enumeration", []any{"a", "b"}, "c"
			delete(field, "minimum")
			delete(field, "maximum")
		},
		"oversized string default": func(component map[string]any) {
			field := component["configuration"].([]any)[0].(map[string]any)
			field["type"], field["default"] = "string", strings.Repeat("x", 4097)
			delete(field, "minimum")
			delete(field, "maximum")
		},
		"state carries configuration default": func(component map[string]any) {
			component["state"].([]any)[0].(map[string]any)["default"] = "state.json"
		},
		"stateless grade carries state": func(component map[string]any) {
			component["state_grade"] = "stateless"
		},
		"schedule category omits step domain": func(component map[string]any) {
			component["key"].(map[string]any)["category"] = "gradient_accumulation"
		},
		"nonschedule category declares step domain": func(component map[string]any) {
			component["step_domain"] = "optimizer_step"
		},
		"CUDA extension claims reference implementation": func(component map[string]any) {
			component["backend"] = "cuda_extension"
		},
	}
	for name, mutate := range mutations {
		t.Run(name, func(t *testing.T) {
			component := trainingComponentDescriptorFixture()
			mutate(component)
			document := map[string]any{
				"api_version": "trainvm.training-components/v1",
				"components":  []any{component},
			}
			response := serve(t, document)
			if response.Code != http.StatusBadGateway {
				t.Fatalf("invalid training descriptor accepted: status=%d body=%s",
					response.Code, response.Body.String())
			}
		})
	}
}

func TestTrainVMOperationsComeFromNativeAuthority(t *testing.T) {
	document := canonicalTestJSON(t, map[string]any{
		"api_version": "trainvm.operations/v1",
		"operations":  []any{operationDescriptorFixture("rwkv.posttrain")},
	})
	digest := fmt.Sprintf("sha256:%x", sha256.Sum256([]byte(document)))
	commander := &fakeTrainVMCommander{descriptorResult: trainvmstore.DescriptorResult{
		SchemaJSON: document, SchemaHash: digest,
	}}
	srv := New(Config{Commander: commander})
	request := httptest.NewRequest(http.MethodGet, "/api/trainvm/operations", nil)
	response := httptest.NewRecorder()
	srv.Handler().ServeHTTP(response, request)
	if response.Code != http.StatusOK ||
		commander.descriptor.Provider != "trainvm.operations" ||
		commander.descriptor.Version != "1.0.0" ||
		response.Header().Get("Cache-Control") != "no-store" ||
		!strings.Contains(response.Body.String(), `"api_version":"trainvm.operations/v1"`) ||
		!strings.Contains(response.Body.String(), `"schema_hash":"`+digest+`"`) {
		t.Fatalf("operation descriptor status=%d request=%#v body=%s",
			response.Code, commander.descriptor, response.Body.String())
	}
}

func TestTrainVMDescriptorBoundaryFailsClosed(t *testing.T) {
	valid := map[string]any{
		"api_version": "trainvm.operations/v1",
		"operations":  []any{operationDescriptorFixture("rwkv.posttrain")},
	}
	duplicate := map[string]any{
		"api_version": "trainvm.operations/v1",
		"operations": []any{
			operationDescriptorFixture("rwkv.posttrain"),
			operationDescriptorFixture("rwkv.posttrain"),
		},
	}
	outOfOrder := map[string]any{
		"api_version": "trainvm.operations/v1",
		"operations": []any{
			operationDescriptorFixture("z.adapter"),
			operationDescriptorFixture("a.adapter"),
		},
	}
	wrongType := map[string]any{
		"api_version": "trainvm.operations/v1",
		"operations": []any{func() map[string]any {
			profile := operationDescriptorFixture("rwkv.posttrain")
			profile["effect"] = true
			return profile
		}()},
	}
	nonArtifactOutput := map[string]any{
		"api_version": "trainvm.operations/v1",
		"operations": []any{func() map[string]any {
			profile := operationDescriptorFixture("rwkv.posttrain")
			authoring := profile["authoring"].(map[string]any)
			outputs := authoring["outputs"].(map[string]any)
			outputs["checkpoint"] = map[string]any{"type": "string", "required": true}
			return profile
		}()},
	}
	extraEnvelope := map[string]any{
		"api_version": "trainvm.operations/v1", "operations": []any{}, "trusted": true,
	}
	validDocument := canonicalTestJSON(t, valid)
	tests := map[string]struct {
		document string
		digest   string
	}{
		"malformed JSON":  {document: "{"},
		"oversized JSON":  {document: `{"api_version":"trainvm.operations/v1","operations":[],"padding":"` + strings.Repeat("x", trainVMDescriptorLimit) + `"}`},
		"duplicate field": {document: `{"api_version":"trainvm.operations/v1","api_version":"trainvm.operations/v1","operations":[]}`},
		"wrong api version": {document: canonicalTestJSON(t, map[string]any{
			"api_version": "trainvm.operations/v2", "operations": []any{},
		})},
		"wrong hash":                  {document: validDocument, digest: "sha256:" + strings.Repeat("0", 64)},
		"noncanonical JSON":           {document: "{\"operations\":[], \"api_version\":\"trainvm.operations/v1\"}"},
		"extra envelope field":        {document: canonicalTestJSON(t, extraEnvelope)},
		"wrong basic field type":      {document: canonicalTestJSON(t, wrongType)},
		"non-artifact output":         {document: canonicalTestJSON(t, nonArtifactOutput)},
		"duplicate identity":          {document: canonicalTestJSON(t, duplicate)},
		"noncanonical identity order": {document: canonicalTestJSON(t, outOfOrder)},
	}
	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			digest := test.digest
			if digest == "" {
				digest = fmt.Sprintf("sha256:%x", sha256.Sum256([]byte(test.document)))
			}
			commander := &fakeTrainVMCommander{descriptorResult: trainvmstore.DescriptorResult{
				SchemaJSON: test.document, SchemaHash: digest,
			}}
			srv := New(Config{Commander: commander})
			request := httptest.NewRequest(http.MethodGet, "/api/trainvm/operations", nil)
			response := httptest.NewRecorder()
			srv.Handler().ServeHTTP(response, request)
			if response.Code != http.StatusBadGateway ||
				response.Header().Get("Cache-Control") != "no-store" {
				t.Fatalf("invalid descriptor accepted: status=%d body=%s",
					response.Code, response.Body.String())
			}
		})
	}
}

func TestTrainVMCompileMergesAuthorityAdapterPreview(t *testing.T) {
	trainingLock := `{"api_version":"trainvm.training-component-lock/v1","nodes":{},"registry_digest":"sha256:` + strings.Repeat("b", 64) + `"}`
	commander := &fakeTrainVMCommander{submissionResult: trainvmstore.SubmissionResult{
		PlanHash:             "native-hash",
		CanonicalPlan:        `{"kind":"Experiment"}`,
		AdapterLockDigest:    "sha256:adapter-lock",
		CanonicalAdapterLock: `{"api_version":"trainvm.adapter-lock/v1","profiles":[]}`,
		TrainingComponentLockDigest: fmt.Sprintf("sha256:%x",
			sha256.Sum256([]byte(trainingLock))),
		CanonicalTrainingComponentLock: trainingLock,
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
		!strings.Contains(response.Body.String(), `"training_component_lock_digest":"sha256:`) ||
		!strings.Contains(response.Body.String(), `"canonical_training_component_lock":"{\"api_version\":`) ||
		!strings.Contains(response.Body.String(), `"valid":false`) ||
		!strings.Contains(response.Body.String(), `"code":"adapter.registry"`) {
		t.Fatalf("authority adapter preview was not merged: status=%d request=%#v body=%s",
			response.Code, commander.submission, response.Body.String())
	}
}

func TestTrainVMCompileRejectsIncoherentTrainingComponentPreview(t *testing.T) {
	trainingPlan := `{"spec":{"workflow":{"nodes":{"train":{"invoke":{"training":{"components":{},"model_family":"rwkv"}}}}}}}`
	validLock := `{"api_version":"trainvm.training-component-lock/v1","nodes":{},"registry_digest":"sha256:` + strings.Repeat("b", 64) + `"}`
	tests := map[string]struct {
		plan   string
		lock   string
		digest string
	}{
		"digest mismatch": {
			plan: `{"kind":"Experiment"}`, lock: validLock,
			digest: "sha256:" + strings.Repeat("0", 64),
		},
		"malformed canonical lock": {
			plan: `{"kind":"Experiment"}`, lock: "{",
			digest: fmt.Sprintf("sha256:%x", sha256.Sum256([]byte("{"))),
		},
		"training plan missing lock": {plan: trainingPlan},
	}
	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			commander := &fakeTrainVMCommander{submissionResult: trainvmstore.SubmissionResult{
				PlanHash: "native-hash", CanonicalPlan: test.plan,
				TrainingComponentLockDigest:    test.digest,
				CanonicalTrainingComponentLock: test.lock,
			}}
			srv := New(Config{
				Authoring: trainVMAuthoringFixtureWithPlan(t, test.plan),
				Commander: commander, TrainVM: trainVMFixture(t),
			})
			request := httptest.NewRequest(http.MethodPost, "/api/trainvm/compile",
				strings.NewReader(`{"kind":"Experiment"}`))
			response := httptest.NewRecorder()
			srv.Handler().ServeHTTP(response, request)
			if response.Code != http.StatusBadGateway {
				t.Fatalf("incoherent training lock accepted: status=%d body=%s",
					response.Code, response.Body.String())
			}
		})
	}
}

func TestTrainVMCompileAcceptsCoherentRequiredTrainingComponentPreview(t *testing.T) {
	trainingPlan, trainingLock := coherentTrainingPreviewFixture(t)
	commander := &fakeTrainVMCommander{submissionResult: trainvmstore.SubmissionResult{
		PlanHash:      "native-hash",
		CanonicalPlan: trainingPlan,
		TrainingComponentLockDigest: fmt.Sprintf("sha256:%x",
			sha256.Sum256([]byte(trainingLock))),
		CanonicalTrainingComponentLock: trainingLock,
	}}
	srv := New(Config{
		Authoring: trainVMAuthoringFixtureWithPlan(t, trainingPlan),
		Commander: commander, TrainVM: trainVMFixture(t),
	})
	request := httptest.NewRequest(http.MethodPost, "/api/trainvm/compile",
		strings.NewReader(`{"kind":"Experiment"}`))
	response := httptest.NewRecorder()
	srv.Handler().ServeHTTP(response, request)
	if response.Code != http.StatusOK ||
		!strings.Contains(response.Body.String(), `"valid":true`) ||
		!strings.Contains(response.Body.String(), `"training_component_lock_digest":"sha256:`) {
		t.Fatalf("coherent required training lock was rejected: status=%d body=%s",
			response.Code, response.Body.String())
	}
}

func TestTrainVMCompileRejectsTrainingLockPlanSkew(t *testing.T) {
	trainingPlan, validLock := coherentTrainingPreviewFixture(t)
	mutations := map[string]func(map[string]any){
		"extra lock envelope field": func(lock map[string]any) {
			lock["untrusted"] = true
		},
		"malformed registry digest": func(lock map[string]any) {
			lock["registry_digest"] = "sha256:NOT-CANONICAL"
		},
		"extra training node": func(lock map[string]any) {
			lock["nodes"].(map[string]any)["extra"] = map[string]any{}
		},
		"missing training node": func(lock map[string]any) {
			delete(lock["nodes"].(map[string]any), "train")
		},
		"node registry differs from envelope": func(lock map[string]any) {
			node := lock["nodes"].(map[string]any)["train"].(map[string]any)
			node["registry_digest"] = "sha256:" + strings.Repeat("c", 64)
		},
		"model family differs from plan": func(lock map[string]any) {
			node := lock["nodes"].(map[string]any)["train"].(map[string]any)
			node["model_family"] = "transformer"
		},
		"component slot differs from plan": func(lock map[string]any) {
			node := lock["nodes"].(map[string]any)["train"].(map[string]any)
			components := node["components"].(map[string]any)
			components["objective"] = components["optimizer"]
		},
		"descriptor key differs from plan": func(lock map[string]any) {
			node := lock["nodes"].(map[string]any)["train"].(map[string]any)
			component := node["components"].(map[string]any)["optimizer"].(map[string]any)
			descriptor := component["descriptor"].(map[string]any)
			descriptor["key"].(map[string]any)["name"] = "different_optimizer"
		},
		"configuration differs from plan": func(lock map[string]any) {
			node := lock["nodes"].(map[string]any)["train"].(map[string]any)
			component := node["components"].(map[string]any)["optimizer"].(map[string]any)
			component["configuration"].(map[string]any)["learning_rate"] = 0.2
		},
		"resolved configuration adds unknown field": func(lock map[string]any) {
			node := lock["nodes"].(map[string]any)["train"].(map[string]any)
			component := node["components"].(map[string]any)["optimizer"].(map[string]any)
			component["configuration"].(map[string]any)["authority_extra"] = true
		},
		"resolved configuration changes descriptor default": func(lock map[string]any) {
			node := lock["nodes"].(map[string]any)["train"].(map[string]any)
			component := node["components"].(map[string]any)["optimizer"].(map[string]any)
			component["configuration"].(map[string]any)["weight_decay"] = 0.02
		},
	}
	for name, mutate := range mutations {
		t.Run(name, func(t *testing.T) {
			lock := mutateTestJSONObject(t, validLock, mutate)
			commander := &fakeTrainVMCommander{submissionResult: trainvmstore.SubmissionResult{
				PlanHash:      "native-hash",
				CanonicalPlan: trainingPlan,
				TrainingComponentLockDigest: fmt.Sprintf("sha256:%x",
					sha256.Sum256([]byte(lock))),
				CanonicalTrainingComponentLock: lock,
			}}
			srv := New(Config{
				Authoring: trainVMAuthoringFixtureWithPlan(t, trainingPlan),
				Commander: commander, TrainVM: trainVMFixture(t),
			})
			request := httptest.NewRequest(http.MethodPost, "/api/trainvm/compile",
				strings.NewReader(`{"kind":"Experiment"}`))
			response := httptest.NewRecorder()
			srv.Handler().ServeHTTP(response, request)
			if response.Code != http.StatusBadGateway {
				t.Fatalf("plan-skewed training lock accepted: status=%d body=%s",
					response.Code, response.Body.String())
			}
		})
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
		commander.request.ExpectedPlanHash != "3ca70ff811ace5137501dfe841bf8d1914ecd9083e9e6f68c521aaedf90d5b0d" ||
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

func TestTrainVMLifecycleEndpointUsesNativeCommander(t *testing.T) {
	commander := &fakeTrainVMCommander{actionResult: trainvmstore.RunActionResult{
		Disposition: "ACCEPTED", Action: "pause", CommandID: "pause-1",
		ControllerSequence: 12, Status: "REQUESTED", CheckpointFirst: true,
		ReleaseResources: true,
	}}
	srv := New(Config{Commander: commander, TrainVM: trainVMFixture(t)})
	request := newTrainVMActionRequest(`{
		"expected_run_revision":7,
		"idempotency_key":"tab-action-intent",
		"reason":"release GPU for interactive work",
		"action":"pause",
		"checkpoint_first":true,
		"release_resources":true,
		"graceful_timeout_seconds":0
	}`)
	request.Header.Set("Content-Type", "application/json")
	response := httptest.NewRecorder()
	srv.Handler().ServeHTTP(response, request)
	got := commander.actionRequest
	if response.Code != http.StatusAccepted || got.RunID != "vm-run" ||
		got.ExpectedJournalID != "0123456789abcdef0123456789abcdef" ||
		got.ExpectedPlanHash != "3ca70ff811ace5137501dfe841bf8d1914ecd9083e9e6f68c521aaedf90d5b0d" ||
		got.ExpectedRunRevision != 7 || got.IdempotencyKey != "tab-action-intent" ||
		got.Author != "dashboard" || got.Action != "pause" || !got.CheckpointFirst || !got.ReleaseResources ||
		!strings.Contains(response.Body.String(), `"command_id":"pause-1"`) {
		t.Fatalf("unexpected action forwarding: status=%d request=%#v body=%s",
			response.Code, got, response.Body.String())
	}
}

func TestTrainVMLifecycleEndpointSharesStrictMutationBoundary(t *testing.T) {
	for name, test := range map[string]struct {
		body        string
		contentType string
		host        string
		origin      string
		expected    int
	}{
		"unknown field":  {body: `{"mystery":true}`, contentType: "application/json", host: "127.0.0.1", expected: http.StatusBadRequest},
		"trailing JSON":  {body: `{}` + ` {}`, contentType: "application/json", host: "127.0.0.1", expected: http.StatusBadRequest},
		"wrong type":     {body: `{}`, contentType: "text/plain", host: "127.0.0.1", expected: http.StatusUnsupportedMediaType},
		"rebound host":   {body: `{}`, contentType: "application/json", host: "attacker.invalid", expected: http.StatusForbidden},
		"foreign origin": {body: `{}`, contentType: "application/json", host: "127.0.0.1:9124", origin: "https://attacker.invalid", expected: http.StatusForbidden},
	} {
		t.Run(name, func(t *testing.T) {
			commander := &fakeTrainVMCommander{}
			srv := New(Config{Commander: commander, TrainVM: trainVMFixture(t)})
			request := newTrainVMActionRequest(test.body)
			request.Host = test.host
			request.Header.Set("Content-Type", test.contentType)
			if test.origin != "" {
				request.Header.Set("Origin", test.origin)
			}
			response := httptest.NewRecorder()
			srv.Handler().ServeHTTP(response, request)
			if response.Code != test.expected || commander.actionRequest.RunID != "" {
				t.Fatalf("status=%d expected=%d forwarded=%#v body=%s",
					response.Code, test.expected, commander.actionRequest, response.Body.String())
			}
		})
	}
}

func TestTrainVMLifecycleEndpointStatusMapping(t *testing.T) {
	for name, test := range map[string]struct {
		commander trainvmstore.Commander
		expected  int
	}{
		"disabled":    {commander: nil, expected: http.StatusServiceUnavailable},
		"unavailable": {commander: &fakeTrainVMCommander{err: status.Error(codes.Unavailable, "offline")}, expected: http.StatusServiceUnavailable},
		"validation":  {commander: &fakeTrainVMCommander{err: &trainvmstore.ValidationError{Message: "bad request"}}, expected: http.StatusBadRequest},
		"accepted":    {commander: &fakeTrainVMCommander{actionResult: trainvmstore.RunActionResult{Disposition: "ACCEPTED", Action: "resume"}}, expected: http.StatusAccepted},
		"replayed":    {commander: &fakeTrainVMCommander{actionResult: trainvmstore.RunActionResult{Disposition: "ALREADY_APPLIED", Action: "resume"}}, expected: http.StatusOK},
		"conflict":    {commander: &fakeTrainVMCommander{actionResult: trainvmstore.RunActionResult{Disposition: "CONFLICT", Action: "resume"}}, expected: http.StatusConflict},
		"rejected":    {commander: &fakeTrainVMCommander{actionResult: trainvmstore.RunActionResult{Disposition: "REJECTED", Action: "resume"}}, expected: http.StatusUnprocessableEntity},
	} {
		t.Run(name, func(t *testing.T) {
			config := Config{Commander: test.commander}
			if test.commander != nil {
				config.TrainVM = trainVMFixture(t)
			}
			srv := New(config)
			request := newTrainVMActionRequest(`{"expected_run_revision":7,"idempotency_key":"action-1","reason":"continue","action":"resume"}`)
			request.Header.Set("Content-Type", "application/json")
			response := httptest.NewRecorder()
			srv.Handler().ServeHTTP(response, request)
			if response.Code != test.expected {
				t.Fatalf("status=%d expected=%d body=%s", response.Code, test.expected, response.Body.String())
			}
		})
	}
}

func TestTrainVMSubmissionEndpointUsesNativeCommander(t *testing.T) {
	commander := &fakeTrainVMCommander{submissionResult: trainvmstore.SubmissionResult{
		PlanHash: "native-plan", AdapterLockDigest: "native-lock",
		TrainingComponentLockDigest: "training-lock", Run: &trainvmstore.RunIdentity{
			RunID: "run-new", Revision: 1, PlanHash: "native-plan",
		},
	}}
	srv := New(Config{Commander: commander, TrainVM: trainVMFixture(t)})
	request := httptest.NewRequest(http.MethodPost, "/api/trainvm/experiments",
		strings.NewReader(`{"source_document":"{\"kind\":\"Experiment\"}","source_format":"json","create_run":true,"idempotency_key":"submit-1","expected_journal_id":"0123456789abcdef0123456789abcdef","expected_plan_hash":"native-plan","expected_adapter_lock_digest":"native-lock","expected_training_component_lock_digest":"training-lock","reason":"launch test","forked_from_run_id":"vm-run","expected_parent_run_revision":2,"expected_parent_plan_hash":"3ca70ff811ace5137501dfe841bf8d1914ecd9083e9e6f68c521aaedf90d5b0d"}`))
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
		commander.submission.ExpectedTrainingComponentLockDigest != "training-lock" ||
		commander.submission.ForkedFromRunID != "vm-run" ||
		commander.submission.ExpectedParentRunRevision != 2 ||
		commander.submission.ExpectedParentPlanHash != "3ca70ff811ace5137501dfe841bf8d1914ecd9083e9e6f68c521aaedf90d5b0d" ||
		!strings.Contains(response.Body.String(), `"run_id":"run-new"`) {
		t.Fatalf("unexpected submission forwarding: status=%d request=%#v body=%s",
			response.Code, commander.submission, response.Body.String())
	}
}

func TestTrainVMPlanDiffEndpointFencesSelectedRunAndProposedPlan(t *testing.T) {
	commander := &fakeTrainVMCommander{planDiffResult: trainvmstore.PlanDiffResult{
		ProposedPlanHash: "proposed-plan", SemanticDiff: json.RawMessage(`[{"op":"replace","path":"/metadata/name","value":"next"}]`),
		Diagnostics: []trainvmstore.ControlDiagnostic{{Severity: "WARNING", Code: "plan.adoption_requires_new_run"}},
	}}
	srv := New(Config{Commander: commander, TrainVM: trainVMFixture(t)})
	request := httptest.NewRequest(http.MethodPost, "/api/trainvm/runs/vm-run/diff",
		strings.NewReader(`{"expected_run_revision":2,"proposed_source_document":"{\"kind\":\"Experiment\"}","source_format":"json","expected_journal_id":"0123456789abcdef0123456789abcdef","expected_current_plan_hash":"3ca70ff811ace5137501dfe841bf8d1914ecd9083e9e6f68c521aaedf90d5b0d","expected_proposed_plan_hash":"proposed-plan"}`))
	request.Host = "127.0.0.1:9124"
	request.Header.Set("Content-Type", "application/json")
	response := httptest.NewRecorder()
	srv.Handler().ServeHTTP(response, request)
	if response.Code != http.StatusOK || commander.planDiff.RunID != "vm-run" ||
		commander.planDiff.ExpectedRunRevision != 2 ||
		commander.planDiff.ExpectedJournalID != "0123456789abcdef0123456789abcdef" ||
		commander.planDiff.ExpectedCurrentPlanHash != "3ca70ff811ace5137501dfe841bf8d1914ecd9083e9e6f68c521aaedf90d5b0d" ||
		commander.planDiff.ExpectedProposedPlanHash != "proposed-plan" ||
		!strings.Contains(response.Body.String(), `"semantic_diff":[{"op":"replace"`) {
		t.Fatalf("unexpected plan diff forwarding: status=%d request=%#v body=%s",
			response.Code, commander.planDiff, response.Body.String())
	}
}

func TestTrainVMPlanDiffEndpointRejectsStaleSelectedRun(t *testing.T) {
	commander := &fakeTrainVMCommander{}
	srv := New(Config{Commander: commander, TrainVM: trainVMFixture(t)})
	request := httptest.NewRequest(http.MethodPost, "/api/trainvm/runs/vm-run/diff",
		strings.NewReader(`{"expected_run_revision":1,"proposed_source_document":"{}","source_format":"json","expected_journal_id":"0123456789abcdef0123456789abcdef","expected_current_plan_hash":"3ca70ff811ace5137501dfe841bf8d1914ecd9083e9e6f68c521aaedf90d5b0d","expected_proposed_plan_hash":"proposed-plan"}`))
	request.Host = "127.0.0.1:9124"
	request.Header.Set("Content-Type", "application/json")
	response := httptest.NewRecorder()
	srv.Handler().ServeHTTP(response, request)
	if response.Code != http.StatusConflict || commander.planDiff.RunID != "" {
		t.Fatalf("stale plan diff reached authority: status=%d request=%#v body=%s",
			response.Code, commander.planDiff, response.Body.String())
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
