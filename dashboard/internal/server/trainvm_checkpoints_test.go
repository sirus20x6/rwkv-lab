package server

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"testing"

	trainvmstore "trainboard/internal/trainvm"
)

type checkpointReadModel struct {
	trainvmstore.ReadModel
	events []trainvmstore.Event
	query  trainvmstore.EventQuery
}

func (f *checkpointReadModel) Run(_ context.Context, runID string) (trainvmstore.Run, bool, error) {
	var last uint64
	for _, event := range f.events {
		if event.RunID == runID && event.Sequence > last {
			last = event.Sequence
		}
	}
	return trainvmstore.Run{RunID: runID, LastEventSeq: last}, last != 0, nil
}

func (f *checkpointReadModel) Events(_ context.Context, query trainvmstore.EventQuery) ([]trainvmstore.Event, error) {
	f.query = query
	result := make([]trainvmstore.Event, 0, len(f.events))
	for _, event := range f.events {
		if event.RunID != query.RunID || event.Sequence <= query.After ||
			(query.Through != 0 && event.Sequence > query.Through) {
			continue
		}
		result = append(result, event)
	}
	sort.Slice(result, func(i, j int) bool {
		if query.NewestFirst {
			return result[i].Sequence > result[j].Sequence
		}
		return result[i].Sequence < result[j].Sequence
	})
	if query.Limit > 0 && len(result) > query.Limit {
		result = result[:query.Limit]
	}
	return result, nil
}

func trainVMCheckpointFixture(t *testing.T) (*Server, string) {
	t.Helper()
	root := t.TempDir()
	revision := filepath.Join(root, "checkpoint-artifact-42")
	if err := os.MkdirAll(filepath.Join(revision, "payload", "adapter"), 0o700); err != nil {
		t.Fatal(err)
	}
	objects := []map[string]any{
		{"relative_path": "adapter/weights.safetensors", "sha256": prefixedTestSHA256([]byte("weights")), "size_bytes": uint64(7)},
		{"relative_path": "trainer-state.pt", "sha256": prefixedTestSHA256([]byte("trainer")), "size_bytes": uint64(7)},
	}
	body := map[string]any{
		"schema":            trainVMCheckpointEnvelopeSchema,
		"checkpoint_schema": "rwkv-lab.mageflow-checkpoint.v1",
		"producer":          map[string]string{"run_id": "vm-run", "node_id": "train", "attempt_id": "train@1"},
		"optimizer_step":    uint64(42), "resume_grade": "compatible",
		"state_components": []string{
			"component_composition", "control_revision", "curriculum", "data_cursor",
			"expert_routing", "gradient_scaler", "lr_schedule", "model", "optimizer",
			"optimizer_groups", "parameter_routing", "plateau_state", "rng_accelerator",
			"rng_numpy", "rng_python", "rng_torch", "topology", "weight_decay_schedule",
		},
		"payload_directory": "payload", "file_count": uint64(2), "payload_size_bytes": uint64(14),
		"objects": objects, "parent_artifact_ids": []string{"base-1"},
	}
	canonicalBody, err := json.Marshal(body)
	if err != nil {
		t.Fatal(err)
	}
	body["canonical_tree_digest"] = prefixedTestSHA256(canonicalBody)
	manifest, err := json.Marshal(body)
	if err != nil {
		t.Fatal(err)
	}
	manifestPath := filepath.Join(revision, "manifest.json")
	if err := os.WriteFile(manifestPath, manifest, 0o400); err != nil {
		t.Fatal(err)
	}
	payload, err := json.Marshal(map[string]any{
		"artifact_id": "checkpoint-artifact-42", "logical_name": "checkpoint", "kind": "checkpoint",
		"schema": "rwkv-lab.mageflow-checkpoint.v1", "uri": testFileURI(manifestPath),
		"size_bytes": len(manifest) + 14, "fingerprint_algorithm": "manifest_sha256",
		"fingerprint": prefixedTestSHA256(manifest), "complete": true,
		"producer_node_id": "train", "producer_attempt_id": "train@1",
		"parent_artifact_ids": []string{"base-1"}, "published_at_ns": int64(100),
	})
	if err != nil {
		t.Fatal(err)
	}
	reader := &checkpointReadModel{events: []trainvmstore.Event{{
		Sequence: 28, EventID: "checkpoint-event", RunID: "vm-run", NodeID: "train", AttemptID: "train@1",
		WorkerSequence: 9, EventType: "artifact.published", EventVersion: 1, WallTimeNS: 100,
		Payload: payload,
	}}}
	return New(Config{TrainVM: reader, ImageRoots: []string{root}}), manifestPath
}

func TestTrainVMCheckpointHistoryShowsVerifiedStateSummary(t *testing.T) {
	srv, _ := trainVMCheckpointFixture(t)
	request := httptest.NewRequest(http.MethodGet, "/api/trainvm/runs/vm-run/checkpoints?limit=25", nil)
	response := httptest.NewRecorder()
	srv.Handler().ServeHTTP(response, request)
	if response.Code != http.StatusOK ||
		!strings.Contains(response.Body.String(), `"artifact_id":"checkpoint-artifact-42"`) ||
		!strings.Contains(response.Body.String(), `"optimizer_step":42`) ||
		!strings.Contains(response.Body.String(), `"resume_grade":"compatible"`) ||
		!strings.Contains(response.Body.String(), `"optimizer_groups"`) ||
		!strings.Contains(response.Body.String(), `"plateau_state"`) ||
		!strings.Contains(response.Body.String(), `"file_count":2`) {
		t.Fatalf("checkpoint list status=%d body=%s", response.Code, response.Body.String())
	}
}

func TestTrainVMCheckpointHistoryUsesBoundedNewestArtifactTail(t *testing.T) {
	srv, _ := trainVMCheckpointFixture(t)
	reader := srv.trainvm.(*checkpointReadModel)
	request := httptest.NewRequest(http.MethodGet, "/api/trainvm/runs/vm-run/checkpoints?limit=7", nil)
	response := httptest.NewRecorder()
	srv.Handler().ServeHTTP(response, request)
	if response.Code != http.StatusOK || reader.query.Limit != 7 ||
		!reader.query.NewestFirst || reader.query.Through != 28 ||
		len(reader.query.EventTypes) != 1 || reader.query.EventTypes[0] != "artifact.published" {
		t.Fatalf("checkpoint tail was not bounded newest-first: status=%d query=%#v", response.Code, reader.query)
	}

	request = httptest.NewRequest(http.MethodGet, "/api/trainvm/runs/vm-run/checkpoints?limit=1001", nil)
	response = httptest.NewRecorder()
	srv.Handler().ServeHTTP(response, request)
	if response.Code != http.StatusBadRequest {
		t.Fatalf("oversized checkpoint tail status=%d body=%s", response.Code, response.Body.String())
	}
}

func TestTrainVMCheckpointQuarantinesManifestMutationWithoutPoisoningHistory(t *testing.T) {
	srv, manifestPath := trainVMCheckpointFixture(t)
	if err := os.Chmod(manifestPath, 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(manifestPath, []byte(`{"schema":"tampered"}`), 0o400); err != nil {
		t.Fatal(err)
	}
	request := httptest.NewRequest(http.MethodGet, "/api/trainvm/runs/vm-run/checkpoints", nil)
	response := httptest.NewRecorder()
	srv.Handler().ServeHTTP(response, request)
	if response.Code != http.StatusOK ||
		!strings.Contains(response.Body.String(), `"valid":false`) ||
		!strings.Contains(response.Body.String(), `"validation_error":"manifest unavailable or failed immutable verification"`) ||
		!strings.Contains(response.Body.String(), `"artifact_id":"checkpoint-artifact-42"`) {
		t.Fatalf("mutated checkpoint status=%d body=%s", response.Code, response.Body.String())
	}
}
