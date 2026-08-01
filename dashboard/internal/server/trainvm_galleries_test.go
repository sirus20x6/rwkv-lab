package server

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"testing"

	trainvmstore "trainboard/internal/trainvm"
)

type galleryReadModel struct {
	trainvmstore.ReadModel
	events []trainvmstore.Event
}

func (f *galleryReadModel) Events(_ context.Context, query trainvmstore.EventQuery) ([]trainvmstore.Event, error) {
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

func testSHA256(data []byte) string {
	digest := sha256.Sum256(data)
	return hex.EncodeToString(digest[:])
}

func testFileURI(path string) string {
	return (&url.URL{Scheme: "file", Path: filepath.ToSlash(path)}).String()
}

func trainVMGalleryFixture(t *testing.T) (*Server, string, string) {
	t.Helper()
	root := t.TempDir()
	generatedPath := filepath.Join(root, "generated.png")
	targetPath := filepath.Join(root, "target.png")
	pngHeader := []byte{0x89, 'P', 'N', 'G', '\r', '\n', 0x1a, '\n'}
	generated := append(append([]byte{}, pngHeader...), []byte("generated-image-bytes")...)
	target := append(append([]byte{}, pngHeader...), []byte("target-image-bytes")...)
	if err := os.WriteFile(generatedPath, generated, 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(targetPath, target, 0o600); err != nil {
		t.Fatal(err)
	}
	manifest := trainVMGalleryManifest{
		APIVersion: trainVMGallerySchema, RunID: "vm-run", NodeID: "eval", AttemptID: "eval@2",
		Step: 75, StepDomain: "optimizer_step",
		CheckpointManifestDigest: strings.Repeat("c", 64),
		EvaluatorProfileDigest:   strings.Repeat("e", 64),
		UsePolicyDigest:          strings.Repeat("a", 64),
		CanonicalManifestDigest:  strings.Repeat("d", 64),
		Items: []trainVMGalleryItem{{
			ItemID: "heldout-7", HeldoutItemID: "photo-7",
			HeldoutManifestDigest:   strings.Repeat("a", 64),
			PromptOrConditionDigest: strings.Repeat("b", 64), Seed: 123,
			GeneratedObjectSHA256: testSHA256(generated), GeneratedObjectURI: testFileURI(generatedPath),
			TargetObjectSHA256: testSHA256(target), TargetObjectURI: testFileURI(targetPath),
			SamplingAttributes: map[string]string{"route": "photo", "steps": "30"},
		}},
	}
	manifestData, err := json.Marshal(manifest)
	if err != nil {
		t.Fatal(err)
	}
	manifestPath := filepath.Join(root, "gallery.json")
	if err := os.WriteFile(manifestPath, manifestData, 0o600); err != nil {
		t.Fatal(err)
	}
	payload, err := json.Marshal(map[string]any{
		"artifact_id": "gallery-75", "logical_name": "eval/gallery", "kind": "image_gallery",
		"schema": trainVMGallerySchema, "uri": testFileURI(manifestPath), "size_bytes": len(manifestData),
		"fingerprint_algorithm": "manifest_sha256", "fingerprint": testSHA256(manifestData), "complete": true,
		"producer_node_id": "eval", "producer_attempt_id": "eval@2",
		"parent_artifact_ids": []string{"checkpoint-75"}, "published_at_ns": int64(99),
	})
	if err != nil {
		t.Fatal(err)
	}
	reader := &galleryReadModel{events: []trainvmstore.Event{{
		Sequence: 12, EventID: "gallery-event", RunID: "vm-run", NodeID: "eval", AttemptID: "eval@2",
		WorkerSequence: 4, EventType: "artifact.published", EventVersion: 1, WallTimeNS: 99,
		Payload: payload,
	}}}
	return New(Config{TrainVM: reader, ImageRoots: []string{root}}), generatedPath, manifestPath
}

func TestTrainVMGalleryHistoryAndVerifiedSideBySideImages(t *testing.T) {
	srv, generatedPath, _ := trainVMGalleryFixture(t)
	request := httptest.NewRequest(http.MethodGet, "/api/trainvm/runs/vm-run/galleries", nil)
	response := httptest.NewRecorder()
	srv.Handler().ServeHTTP(response, request)
	if response.Code != http.StatusOK || !strings.Contains(response.Body.String(), `"artifact_id":"gallery-75"`) ||
		!strings.Contains(response.Body.String(), `"item_count":1`) {
		t.Fatalf("gallery history status=%d body=%s", response.Code, response.Body.String())
	}

	request = httptest.NewRequest(http.MethodGet, "/api/trainvm/runs/vm-run/galleries/gallery-75", nil)
	response = httptest.NewRecorder()
	srv.Handler().ServeHTTP(response, request)
	if response.Code != http.StatusOK || !strings.Contains(response.Body.String(), `"generated_image_url"`) ||
		!strings.Contains(response.Body.String(), `"target_image_url"`) {
		t.Fatalf("gallery status=%d body=%s", response.Code, response.Body.String())
	}
	var gallery trainVMGalleryResponse
	if err := json.Unmarshal(response.Body.Bytes(), &gallery); err != nil || len(gallery.Items) != 1 {
		t.Fatalf("decode gallery: %#v err=%v", gallery, err)
	}

	request = httptest.NewRequest(http.MethodGet, gallery.Items[0].GeneratedImageURL, nil)
	response = httptest.NewRecorder()
	srv.Handler().ServeHTTP(response, request)
	if response.Code != http.StatusOK || !strings.HasSuffix(response.Body.String(), "generated-image-bytes") ||
		response.Header().Get("Cache-Control") != "private, immutable, max-age=31536000" ||
		response.Header().Get("Content-Type") != "image/png" ||
		response.Header().Get("X-Content-Type-Options") != "nosniff" {
		t.Fatalf("generated image status=%d headers=%v body=%q", response.Code, response.Header(), response.Body.String())
	}

	if err := os.WriteFile(generatedPath, []byte("mutated"), 0o600); err != nil {
		t.Fatal(err)
	}
	request = httptest.NewRequest(http.MethodGet, gallery.Items[0].GeneratedImageURL, nil)
	response = httptest.NewRecorder()
	srv.Handler().ServeHTTP(response, request)
	if response.Code != http.StatusConflict {
		t.Fatalf("mutated image was served: status=%d body=%s", response.Code, response.Body.String())
	}
}

func TestTrainVMGalleryRejectsMutatedManifest(t *testing.T) {
	srv, _, manifestPath := trainVMGalleryFixture(t)
	if err := os.WriteFile(manifestPath, []byte(`{"api_version":"tampered"}`), 0o600); err != nil {
		t.Fatal(err)
	}
	request := httptest.NewRequest(http.MethodGet, "/api/trainvm/runs/vm-run/galleries/gallery-75", nil)
	response := httptest.NewRecorder()
	srv.Handler().ServeHTTP(response, request)
	if response.Code != http.StatusBadGateway || !strings.Contains(response.Body.String(), "mismatch") {
		t.Fatalf("mutated manifest status=%d body=%s", response.Code, response.Body.String())
	}
}

func TestTrainVMGalleryNeverServesAnArtifactPathOutsideConfiguredRoots(t *testing.T) {
	srv, _, manifestPath := trainVMGalleryFixture(t)
	outsidePath := filepath.Join(t.TempDir(), "outside.png")
	outsideData := []byte("outside-image")
	if err := os.WriteFile(outsidePath, outsideData, 0o600); err != nil {
		t.Fatal(err)
	}
	manifestData, err := os.ReadFile(manifestPath)
	if err != nil {
		t.Fatal(err)
	}
	var manifest trainVMGalleryManifest
	if err := json.Unmarshal(manifestData, &manifest); err != nil {
		t.Fatal(err)
	}
	manifest.Items[0].GeneratedObjectURI = testFileURI(outsidePath)
	manifest.Items[0].GeneratedObjectSHA256 = testSHA256(outsideData)
	manifestData, err = json.Marshal(manifest)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(manifestPath, manifestData, 0o600); err != nil {
		t.Fatal(err)
	}
	reader := srv.trainvm.(*galleryReadModel)
	var payload map[string]any
	if err := json.Unmarshal(reader.events[0].Payload, &payload); err != nil {
		t.Fatal(err)
	}
	payload["fingerprint"] = testSHA256(manifestData)
	payload["size_bytes"] = len(manifestData)
	reader.events[0].Payload, err = json.Marshal(payload)
	if err != nil {
		t.Fatal(err)
	}

	request := httptest.NewRequest(http.MethodGet, "/api/trainvm/runs/vm-run/galleries/gallery-75", nil)
	response := httptest.NewRecorder()
	srv.Handler().ServeHTTP(response, request)
	var gallery trainVMGalleryResponse
	if err := json.Unmarshal(response.Body.Bytes(), &gallery); err != nil || len(gallery.Items) != 1 {
		t.Fatalf("gallery status=%d body=%s err=%v", response.Code, response.Body.String(), err)
	}
	request = httptest.NewRequest(http.MethodGet, gallery.Items[0].GeneratedImageURL, nil)
	response = httptest.NewRecorder()
	srv.Handler().ServeHTTP(response, request)
	if response.Code != http.StatusForbidden {
		t.Fatalf("outside image was served: status=%d body=%s", response.Code, response.Body.String())
	}
}
