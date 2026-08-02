package server

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	trainvmstore "trainboard/internal/trainvm"
)

type profileReadModel struct {
	trainvmstore.ReadModel
	events            []trainvmstore.Event
	compiledPlan      trainvmstore.CompiledPlanView
	compiledPlanFound bool
	compiledPlanErr   error
}

func (f *profileReadModel) Events(_ context.Context, query trainvmstore.EventQuery) ([]trainvmstore.Event, error) {
	result := make([]trainvmstore.Event, 0, len(f.events))
	for _, event := range f.events {
		if event.RunID != query.RunID || event.Sequence <= query.After {
			continue
		}
		matched := len(query.EventTypes) == 0
		for _, eventType := range query.EventTypes {
			matched = matched || event.EventType == eventType
		}
		if matched {
			result = append(result, event)
		}
		if query.Limit > 0 && len(result) >= query.Limit {
			break
		}
	}
	return result, nil
}

func (f *profileReadModel) CompiledPlan(context.Context, string) (trainvmstore.CompiledPlanView, bool, error) {
	return f.compiledPlan, f.compiledPlanFound, f.compiledPlanErr
}

func prefixedTestSHA256(data []byte) string {
	digest := sha256.Sum256(data)
	return "sha256:" + hex.EncodeToString(digest[:])
}

func trainVMGPUTraceFixture(t *testing.T) (*Server, string, string) {
	t.Helper()
	root := t.TempDir()
	revision := filepath.Join(root, "gpu-trace-1")
	if err := os.Mkdir(revision, 0o700); err != nil {
		t.Fatal(err)
	}
	trace := []byte(`{"traceEvents":[{"name":"step"}]}`)
	tracePath := filepath.Join(revision, "trace.json")
	if err := os.WriteFile(tracePath, trace, 0o400); err != nil {
		t.Fatal(err)
	}
	body := map[string]any{
		"activities": []string{"cpu", "accelerator"}, "api_version": trainVMGPUTraceSchema,
		"attempt_id": "train@1", "backend": "torch", "capture_steps": uint64(3),
		"first_optimizer_step": uint64(12), "instrumented_timing": true,
		"invocation_digest":   prefixedTestSHA256([]byte("invocation")),
		"last_optimizer_step": uint64(14), "node_id": "train",
		"options": map[string]bool{"profile_memory": true, "record_shapes": true, "with_stack": false},
		"run_id":  "vm-run", "sensitivity": "restricted", "skip_steps": uint64(1),
		"summary": map[string]any{
			"accelerator_time_us": 19.5, "cpu_time_us": 8.25, "kernel_or_operator_count": int64(1),
			"accelerator_launch_count": uint64(17), "captured_step_wall_time_us": 25.0,
			"gpu_active_ratio": 0.6, "gpu_active_time_us": 15.0,
			"input_stall_ratio": 0.2, "input_stall_time_us": 5.0,
			"allocator_baseline_allocated_bytes": uint64(100),
			"allocator_baseline_reserved_bytes":  uint64(200),
			"allocator_peak_allocated_bytes":     uint64(150),
			"allocator_peak_reserved_bytes":      uint64(250),
			"top_operators": []map[string]any{{
				"accelerator_time_us": 19.5, "calls": int64(3), "cpu_time_us": 8.25, "name": "train_step",
			}},
		},
		"trace_sha256": prefixedTestSHA256(trace), "trace_size_bytes": uint64(len(trace)),
		"warmup_steps": uint64(2),
	}
	canonicalBody, err := json.Marshal(body)
	if err != nil {
		t.Fatal(err)
	}
	body["canonical_manifest_digest"] = prefixedTestSHA256(canonicalBody)
	manifest, err := json.Marshal(body)
	if err != nil {
		t.Fatal(err)
	}
	manifestPath := filepath.Join(revision, "manifest.json")
	if err := os.WriteFile(manifestPath, manifest, 0o400); err != nil {
		t.Fatal(err)
	}
	payload, err := json.Marshal(map[string]any{
		"artifact_id": "gpu-trace-1", "logical_name": "gpu_trace", "kind": "opaque",
		"schema": trainVMGPUTraceSchema, "uri": testFileURI(manifestPath),
		"size_bytes": len(manifest) + len(trace), "fingerprint_algorithm": "adapter",
		"fingerprint": prefixedTestSHA256(manifest), "complete": true,
		"producer_node_id": "train", "producer_attempt_id": "train@1",
		"parent_artifact_ids": []string{}, "published_at_ns": int64(99),
	})
	if err != nil {
		t.Fatal(err)
	}
	reader := &profileReadModel{events: []trainvmstore.Event{{
		Sequence: 22, EventID: "profile-event", RunID: "vm-run", NodeID: "train", AttemptID: "train@1",
		WorkerSequence: 7, EventType: "artifact.published", EventVersion: 1, WallTimeNS: 99,
		Payload: payload,
	}}}
	return New(Config{TrainVM: reader, ImageRoots: []string{root}}), manifestPath, tracePath
}

func TestTrainVMGPUTraceSummaryAndExplicitVerifiedDownload(t *testing.T) {
	srv, _, _ := trainVMGPUTraceFixture(t)
	request := httptest.NewRequest(http.MethodGet, "/api/trainvm/runs/vm-run/profiles", nil)
	response := httptest.NewRecorder()
	srv.Handler().ServeHTTP(response, request)
	if response.Code != http.StatusOK || !strings.Contains(response.Body.String(), `"artifact_id":"gpu-trace-1"`) ||
		!strings.Contains(response.Body.String(), `"first_optimizer_step":12`) ||
		!strings.Contains(response.Body.String(), `"accelerator_launch_count":17`) ||
		!strings.Contains(response.Body.String(), `"gpu_active_ratio":0.6`) ||
		!strings.Contains(response.Body.String(), `"input_stall_ratio":0.2`) ||
		!strings.Contains(response.Body.String(), `"sensitivity":"restricted"`) {
		t.Fatalf("profile list status=%d body=%s", response.Code, response.Body.String())
	}
	var profiles []trainVMGPUTraceSummary
	if err := json.Unmarshal(response.Body.Bytes(), &profiles); err != nil || len(profiles) != 1 {
		t.Fatalf("decode profiles: %#v err=%v", profiles, err)
	}
	request = httptest.NewRequest(http.MethodGet, profiles[0].TraceDownloadURL, nil)
	response = httptest.NewRecorder()
	srv.Handler().ServeHTTP(response, request)
	if response.Code != http.StatusOK || response.Header().Get("Cache-Control") != "private, no-store" ||
		response.Header().Get("Content-Disposition") == "" || !strings.Contains(response.Body.String(), "traceEvents") {
		t.Fatalf("trace download status=%d headers=%v body=%s", response.Code, response.Header(), response.Body.String())
	}
}

func TestTrainVMGPUTraceCorrelatesDeclaredExecutionPhases(t *testing.T) {
	boolPointer := func(value bool) *bool { return &value }
	uint64Pointer := func(value uint64) *uint64 { return &value }
	tests := []struct {
		name     string
		plan     json.RawMessage
		found    bool
		planErr  error
		expected *trainVMGPUTraceExecutionPhases
	}{
		{
			name:  "capture fully after warmup",
			plan:  json.RawMessage(`{"spec":{"execution":{"compile":{"enabled":true},"warmup":{"enabled":true,"steps":8},"qualify":{"enabled":true,"steps":4}}}}`),
			found: true,
			expected: &trainVMGPUTraceExecutionPhases{
				CompileEnabled: boolPointer(true), WarmupEnabled: boolPointer(true),
				WarmupStepsDeclared: uint64Pointer(8), QualifyEnabled: boolPointer(true),
				QualifySteps: uint64Pointer(4), OverlapsWarmup: boolPointer(false),
			},
		},
		{
			name:  "capture starts inside warmup",
			plan:  json.RawMessage(`{"spec":{"execution":{"compile":{"enabled":false},"warmup":{"enabled":true,"steps":13},"qualify":{"enabled":false}}}}`),
			found: true,
			expected: &trainVMGPUTraceExecutionPhases{
				CompileEnabled: boolPointer(false), WarmupEnabled: boolPointer(true),
				WarmupStepsDeclared: uint64Pointer(13), QualifyEnabled: boolPointer(false),
				OverlapsWarmup: boolPointer(true),
			},
		},
		{
			name:  "warmup enabled without steps is unknown",
			plan:  json.RawMessage(`{"spec":{"execution":{"warmup":{"enabled":true}}}}`),
			found: true,
			expected: &trainVMGPUTraceExecutionPhases{
				WarmupEnabled: boolPointer(true), OverlapsWarmup: nil,
			},
		},
		{name: "compiled plan not found"},
		{name: "compiled plan read fails", found: true, planErr: errors.New("compiled plan unavailable")},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			srv, _, _ := trainVMGPUTraceFixture(t)
			reader := srv.trainvm.(*profileReadModel)
			reader.compiledPlan = trainvmstore.CompiledPlanView{
				RunID: "vm-run", CanonicalPlan: test.plan,
			}
			reader.compiledPlanFound = test.found
			reader.compiledPlanErr = test.planErr

			request := httptest.NewRequest(http.MethodGet, "/api/trainvm/runs/vm-run/profiles", nil)
			response := httptest.NewRecorder()
			srv.Handler().ServeHTTP(response, request)
			var profiles []trainVMGPUTraceSummary
			if response.Code != http.StatusOK {
				t.Fatalf("profile list status=%d body=%s", response.Code, response.Body.String())
			}
			if err := json.Unmarshal(response.Body.Bytes(), &profiles); err != nil || len(profiles) != 1 {
				t.Fatalf("decode profile list: %#v err=%v", profiles, err)
			}
			assertExecutionPhases(t, profiles[0].ExecutionPhases, test.expected)
			if test.expected == nil && strings.Contains(response.Body.String(), `"execution_phases"`) {
				t.Fatalf("unavailable plan should omit execution phases: %s", response.Body.String())
			}
			if test.expected != nil && test.expected.OverlapsWarmup == nil &&
				!strings.Contains(response.Body.String(), `"overlaps_warmup":null`) {
				t.Fatalf("unknown overlap should be explicit JSON null: %s", response.Body.String())
			}

			request = httptest.NewRequest(http.MethodGet, "/api/trainvm/runs/vm-run/profiles/gpu-trace-1", nil)
			response = httptest.NewRecorder()
			srv.Handler().ServeHTTP(response, request)
			var profile trainVMGPUTraceSummary
			if response.Code != http.StatusOK {
				t.Fatalf("profile detail status=%d body=%s", response.Code, response.Body.String())
			}
			if err := json.Unmarshal(response.Body.Bytes(), &profile); err != nil {
				t.Fatalf("decode profile detail: err=%v", err)
			}
			assertExecutionPhases(t, profile.ExecutionPhases, test.expected)
		})
	}
}

func assertExecutionPhases(t *testing.T, actual, expected *trainVMGPUTraceExecutionPhases) {
	t.Helper()
	actualJSON, err := json.Marshal(actual)
	if err != nil {
		t.Fatal(err)
	}
	expectedJSON, err := json.Marshal(expected)
	if err != nil {
		t.Fatal(err)
	}
	if string(actualJSON) != string(expectedJSON) {
		t.Fatalf("execution phases got=%s want=%s", actualJSON, expectedJSON)
	}
}

func TestTrainVMGPUTraceRejectsManifestAndRawTraceMutation(t *testing.T) {
	srv, manifestPath, tracePath := trainVMGPUTraceFixture(t)
	if err := os.Chmod(manifestPath, 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(manifestPath, []byte(`{"api_version":"tampered"}`), 0o400); err != nil {
		t.Fatal(err)
	}
	request := httptest.NewRequest(http.MethodGet, "/api/trainvm/runs/vm-run/profiles", nil)
	response := httptest.NewRecorder()
	srv.Handler().ServeHTTP(response, request)
	if response.Code != http.StatusBadGateway || !strings.Contains(response.Body.String(), "mismatch") {
		t.Fatalf("mutated manifest status=%d body=%s", response.Code, response.Body.String())
	}

	srv, _, tracePath = trainVMGPUTraceFixture(t)
	request = httptest.NewRequest(http.MethodGet, "/api/trainvm/runs/vm-run/profiles", nil)
	response = httptest.NewRecorder()
	srv.Handler().ServeHTTP(response, request)
	var profiles []trainVMGPUTraceSummary
	if err := json.Unmarshal(response.Body.Bytes(), &profiles); err != nil || len(profiles) != 1 {
		t.Fatalf("decode profiles: status=%d err=%v", response.Code, err)
	}
	if err := os.Chmod(tracePath, 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(tracePath, []byte(`{"traceEvents":[]}`), 0o400); err != nil {
		t.Fatal(err)
	}
	request = httptest.NewRequest(http.MethodGet, profiles[0].TraceDownloadURL, nil)
	response = httptest.NewRecorder()
	srv.Handler().ServeHTTP(response, request)
	if response.Code != http.StatusBadGateway && response.Code != http.StatusConflict {
		t.Fatalf("mutated raw trace status=%d body=%s", response.Code, response.Body.String())
	}
}

func TestRichGPUTraceSummaryIsAllOrNothingAndInternallyConsistent(t *testing.T) {
	if !validRichGPUTraceSummary(trainVMGPUTraceSummaryValues{}) {
		t.Fatal("legacy summary without rich metrics should remain readable")
	}
	launches := uint64(4)
	wall := 20.0
	active := 10.0
	ratio := 0.5
	baselineAllocated := uint64(100)
	baselineReserved := uint64(200)
	peakAllocated := uint64(150)
	peakReserved := uint64(250)
	summary := trainVMGPUTraceSummaryValues{
		AcceleratorLaunchCount:          &launches,
		CapturedStepWallTimeUS:          &wall,
		GPUActiveTimeUS:                 &active,
		GPUActiveRatio:                  &ratio,
		AllocatorBaselineAllocatedBytes: &baselineAllocated,
		AllocatorBaselineReservedBytes:  &baselineReserved,
		AllocatorPeakAllocatedBytes:     &peakAllocated,
		AllocatorPeakReservedBytes:      &peakReserved,
	}
	if !validRichGPUTraceSummary(summary) {
		t.Fatal("consistent rich summary was rejected")
	}
	inputRatio := 0.1
	summary.InputStallRatio = &inputRatio
	if validRichGPUTraceSummary(summary) {
		t.Fatal("partial input-stall summary was accepted")
	}
	inputTime := 2.0
	summary.InputStallTimeUS = &inputTime
	if !validRichGPUTraceSummary(summary) {
		t.Fatal("consistent input-stall summary was rejected")
	}
	summary.InputStallRatio = nil
	summary.InputStallTimeUS = nil
	summary.GPUActiveRatio = nil
	if validRichGPUTraceSummary(summary) {
		t.Fatal("partial rich summary was accepted")
	}
	summary.GPUActiveRatio = &ratio
	ratio = 0.4
	if validRichGPUTraceSummary(summary) {
		t.Fatal("active ratio inconsistent with active and wall time was accepted")
	}
}
