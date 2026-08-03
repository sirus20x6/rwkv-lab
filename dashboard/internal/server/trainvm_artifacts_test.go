package server

import (
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"path/filepath"
	"testing"

	trainvmstore "trainboard/internal/trainvm"
)

func TestReadVerifiedArtifactRestrictsRootsSizeAndContentIdentity(t *testing.T) {
	root := t.TempDir()
	content := []byte("immutable report\n")
	path := filepath.Join(root, "report.json")
	if err := os.WriteFile(path, content, 0o600); err != nil {
		t.Fatal(err)
	}
	digest := fmt.Sprintf("%x", sha256.Sum256(content))
	artifact := trainvmstore.PublishedArtifact{
		ArtifactID: "report-1", URI: (&url.URL{Scheme: "file", Path: filepath.ToSlash(path)}).String(),
		SizeBytes: uint64(len(content)), FingerprintAlgorithm: "sha256", Fingerprint: digest,
	}
	srv := New(Config{ImageRoots: []string{root}})
	loaded, info, actual, err := srv.readVerifiedArtifact(artifact)
	if err != nil {
		t.Fatal(err)
	}
	if string(loaded) != string(content) || info.Size() != int64(len(content)) || actual != digest {
		t.Fatalf("unexpected verified artifact: bytes=%q info=%#v digest=%q", loaded, info, actual)
	}

	tampered := artifact
	tampered.Fingerprint = fmt.Sprintf("%064x", 1)
	if _, _, _, err := srv.readVerifiedArtifact(tampered); err == nil {
		t.Fatal("tampered artifact fingerprint unexpectedly served")
	}
	outside := filepath.Join(t.TempDir(), "outside.json")
	if err := os.WriteFile(outside, content, 0o600); err != nil {
		t.Fatal(err)
	}
	artifact.URI = (&url.URL{Scheme: "file", Path: filepath.ToSlash(outside)}).String()
	if _, _, _, err := srv.readVerifiedArtifact(artifact); err == nil {
		t.Fatal("artifact outside configured roots unexpectedly served")
	}
	link := filepath.Join(root, "escaped.json")
	if err := os.Symlink(outside, link); err != nil {
		t.Fatal(err)
	}
	artifact.URI = (&url.URL{Scheme: "file", Path: filepath.ToSlash(link)}).String()
	if _, _, _, err := srv.readVerifiedArtifact(artifact); err == nil {
		t.Fatal("artifact symlink escaping configured roots unexpectedly served")
	}
}

func TestArtifactContentRouteRequiresSnapshotTokenAndServesVerifiedBytes(t *testing.T) {
	root := t.TempDir()
	content := []byte("bounded artifact\n")
	path := filepath.Join(root, "report.txt")
	if err := os.WriteFile(path, content, 0o600); err != nil {
		t.Fatal(err)
	}
	digest := fmt.Sprintf("%x", sha256.Sum256(content))
	payload, err := json.Marshal(map[string]any{
		"artifact_id": "report-1", "logical_name": "evaluation/report", "kind": "report",
		"schema": "trainvm.report.v1", "uri": testFileURI(path), "size_bytes": len(content),
		"fingerprint_algorithm": "sha256", "fingerprint": digest, "complete": true,
		"producer_node_id": "eval", "producer_attempt_id": "eval@1",
		"parent_artifact_ids": []string{}, "published_at_ns": int64(10),
	})
	if err != nil {
		t.Fatal(err)
	}
	reader := &checkpointReadModel{events: []trainvmstore.Event{{
		Sequence: 4, EventID: "artifact", RunID: "run-1", NodeID: "eval", AttemptID: "eval@1",
		WorkerSequence: 1, EventType: "artifact.published", EventVersion: 1,
		WallTimeNS: 10, Payload: payload,
	}}}
	srv := New(Config{TrainVM: reader, ImageRoots: []string{root}})
	request := httptest.NewRequest(http.MethodGet,
		"/api/trainvm/runs/run-1/artifacts/report-1/content?v="+digest+"&s=4", nil)
	response := httptest.NewRecorder()
	srv.Handler().ServeHTTP(response, request)
	if response.Code != http.StatusOK || response.Body.String() != string(content) ||
		response.Header().Get("ETag") != `"sha256:`+digest+`"` ||
		response.Header().Get("Content-Disposition") == "" {
		t.Fatalf("verified content status=%d headers=%v body=%q",
			response.Code, response.Header(), response.Body.String())
	}

	request = httptest.NewRequest(http.MethodGet,
		"/api/trainvm/runs/run-1/artifacts/report-1/content?v="+fmt.Sprintf("%064x", 1)+"&s=4", nil)
	response = httptest.NewRecorder()
	srv.Handler().ServeHTTP(response, request)
	if response.Code != http.StatusConflict {
		t.Fatalf("stale artifact token status=%d body=%s", response.Code, response.Body.String())
	}

	request = httptest.NewRequest(http.MethodGet,
		"/api/trainvm/runs/run-1/artifacts/report-1/content?v="+digest, nil)
	response = httptest.NewRecorder()
	srv.Handler().ServeHTTP(response, request)
	if response.Code != http.StatusBadRequest {
		t.Fatalf("missing artifact sequence status=%d body=%s", response.Code, response.Body.String())
	}
}

func TestArtifactContentLookupIsAnchoredToExactPublicationSequence(t *testing.T) {
	payload, err := json.Marshal(map[string]any{
		"artifact_id": "report-new", "logical_name": "evaluation/report", "kind": "report",
		"schema": "trainvm.report.v1", "uri": "file:///unused", "size_bytes": 1,
		"fingerprint_algorithm": "sha256", "fingerprint": fmt.Sprintf("%064x", 1), "complete": true,
		"producer_node_id": "eval", "producer_attempt_id": "eval@1",
		"parent_artifact_ids": []string{}, "published_at_ns": int64(10),
	})
	if err != nil {
		t.Fatal(err)
	}
	reader := &checkpointReadModel{events: []trainvmstore.Event{{
		Sequence: 20_001, EventID: "artifact-new", RunID: "run-1", NodeID: "eval",
		AttemptID: "eval@1", WorkerSequence: 1, EventType: "artifact.published",
		EventVersion: 1, WallTimeNS: 10, Payload: payload,
	}}}
	srv := New(Config{TrainVM: reader})
	request := httptest.NewRequest(http.MethodGet,
		"/api/trainvm/runs/run-1/artifacts/report-new/content?v="+fmt.Sprintf("%064x", 1)+"&s=20000", nil)
	request.SetPathValue("run", "run-1")
	if _, found, err := srv.publishedArtifactByID(request, "report-new", 20_001); err != nil || !found {
		t.Fatalf("exact high-sequence artifact lookup failed: found=%v err=%v", found, err)
	}
	if _, found, err := srv.publishedArtifactByID(request, "report-new", 20_000); err != nil || found {
		t.Fatalf("wrong publication sequence resolved an artifact: found=%v err=%v", found, err)
	}
}
