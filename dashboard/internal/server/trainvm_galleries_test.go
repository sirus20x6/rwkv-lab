package server

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
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
	events       []trainvmstore.Event
	run          trainvmstore.Run
	compiledPlan trainvmstore.CompiledPlanView
	journalID    string
	journalReads int
	runReads     int
	planReads    int
	raceMode     string
}

func (f *galleryReadModel) JournalID(context.Context) (string, error) {
	f.journalReads++
	return f.journalID, nil
}

func (f *galleryReadModel) Events(_ context.Context, query trainvmstore.EventQuery) ([]trainvmstore.Event, error) {
	result := make([]trainvmstore.Event, 0, len(f.events))
	for _, event := range f.events {
		if event.RunID != query.RunID || event.Sequence <= query.After ||
			(query.Through > 0 && event.Sequence > query.Through) {
			continue
		}
		matched := len(query.EventTypes) == 0
		for _, eventType := range query.EventTypes {
			matched = matched || event.EventType == eventType
		}
		if matched {
			result = append(result, event)
		}
	}
	if query.NewestFirst {
		for left, right := 0, len(result)-1; left < right; left, right = left+1, right-1 {
			result[left], result[right] = result[right], result[left]
		}
	}
	if query.Limit > 0 && len(result) > query.Limit {
		result = result[:query.Limit]
	}
	return result, nil
}

func (f *galleryReadModel) Run(_ context.Context, runID string) (trainvmstore.Run, bool, error) {
	f.runReads++
	return f.run, f.run.RunID == runID, nil
}

func (f *galleryReadModel) CompiledPlan(
	_ context.Context, runID string,
) (trainvmstore.CompiledPlanView, bool, error) {
	f.planReads++
	if f.raceMode == "once" && f.planReads == 1 {
		raced := f.compiledPlan
		raced.RunRevision++
		return raced, raced.RunID == runID, nil
	}
	if f.raceMode == "always" {
		raced := f.compiledPlan
		raced.RunRevision++
		return raced, raced.RunID == runID, nil
	}
	return f.compiledPlan, f.compiledPlan.RunID == runID, nil
}

func testSHA256(data []byte) string {
	digest := sha256.Sum256(data)
	return hex.EncodeToString(digest[:])
}

func testFileURI(path string) string {
	return (&url.URL{Scheme: "file", Path: filepath.ToSlash(path)}).String()
}

func sealTestGalleryManifest(t *testing.T, manifest trainVMGalleryManifest) []byte {
	t.Helper()
	manifest.CanonicalManifestDigest = strings.Repeat("0", 64)
	encoded, err := json.Marshal(manifest)
	if err != nil {
		t.Fatal(err)
	}
	canonical, err := canonicalDigestWithoutField(encoded, "canonical_manifest_digest")
	if err != nil {
		t.Fatal(err)
	}
	manifest.CanonicalManifestDigest = "sha256:" + canonical
	encoded, err = json.Marshal(manifest)
	if err != nil {
		t.Fatal(err)
	}
	return encoded
}

func replaceGalleryManifest(t *testing.T, srv *Server, manifestPath string, manifest trainVMGalleryManifest) {
	t.Helper()
	manifestData := sealTestGalleryManifest(t, manifest)
	replaceRawGalleryManifest(t, srv, manifestPath, manifestData)
}

func replaceRawGalleryManifest(t *testing.T, srv *Server, manifestPath string, manifestData []byte) {
	t.Helper()
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
	encoded, err := json.Marshal(payload)
	if err != nil {
		t.Fatal(err)
	}
	reader.events[0].Payload = encoded
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
	manifestData := sealTestGalleryManifest(t, manifest)
	manifestPath := filepath.Join(root, "gallery.json")
	if err := os.WriteFile(manifestPath, manifestData, 0o600); err != nil {
		t.Fatal(err)
	}
	payload, err := json.Marshal(map[string]any{
		"artifact_id": "gallery-75", "logical_name": "eval_gallery", "kind": "image_gallery",
		"schema": trainVMGallerySchema, "uri": testFileURI(manifestPath), "size_bytes": len(manifestData),
		"fingerprint_algorithm": "manifest_sha256", "fingerprint": testSHA256(manifestData), "complete": true,
		"producer_node_id": "eval", "producer_attempt_id": "eval@2",
		"parent_artifact_ids": []string{"checkpoint-75"}, "published_at_ns": int64(99),
	})
	if err != nil {
		t.Fatal(err)
	}
	compiledPlan := json.RawMessage(`{"spec":{"observability":{"heartbeat_seconds":10,"metrics":[],"retain_raw_metrics_days":30,"eval_gallery_artifact":"eval_gallery"}}}`)
	const journalID = "0123456789abcdef0123456789abcdef"
	const planHash = "gallery-plan-hash"
	reader := &galleryReadModel{
		journalID: journalID,
		run: trainvmstore.Run{
			RunID: "vm-run", RunRevision: 2, PlanHash: planHash, LastEventSeq: 12,
		},
		compiledPlan: trainvmstore.CompiledPlanView{
			JournalID: journalID, RunID: "vm-run", RunRevision: 2,
			PlanHash: planHash, CanonicalPlan: compiledPlan,
		},
		events: []trainvmstore.Event{{
			Sequence: 12, EventID: "gallery-event", RunID: "vm-run", NodeID: "eval", AttemptID: "eval@2",
			WorkerSequence: 4, EventType: "artifact.published", EventVersion: 1, WallTimeNS: 99,
			Payload: payload,
		}}}
	return New(Config{TrainVM: reader, ImageRoots: []string{root}}), generatedPath, manifestPath
}

func TestTrainVMGalleryHistoryUsesTheDeclaredLogicalArtifact(t *testing.T) {
	srv, _, _ := trainVMGalleryFixture(t)
	reader := srv.trainvm.(*galleryReadModel)
	var payload map[string]any
	if err := json.Unmarshal(reader.events[0].Payload, &payload); err != nil {
		t.Fatal(err)
	}
	payload["logical_name"] = "some/other/gallery"
	encoded, err := json.Marshal(payload)
	if err != nil {
		t.Fatal(err)
	}
	reader.events[0].Payload = encoded

	request := httptest.NewRequest(http.MethodGet, "/api/trainvm/runs/vm-run/galleries", nil)
	response := httptest.NewRecorder()
	srv.Handler().ServeHTTP(response, request)
	var history trainVMGalleryHistoryResponse
	if err := json.Unmarshal(response.Body.Bytes(), &history); err != nil ||
		response.Code != http.StatusOK || len(history.Galleries) != 0 || history.HistoryTruncated {
		t.Fatalf("undeclared gallery leaked into history: status=%d body=%s", response.Code, response.Body.String())
	}
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
	replaceGalleryManifest(t, srv, manifestPath, manifest)

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

func TestTrainVMGalleryCaptureRetriesOnePlanRaceAndFailsClosedOnAContinuingRace(t *testing.T) {
	for _, test := range []struct {
		name       string
		raceMode   string
		wantStatus int
		wantReads  int
	}{
		{name: "settles", raceMode: "once", wantStatus: http.StatusOK, wantReads: 2},
		{name: "continues", raceMode: "always", wantStatus: http.StatusBadGateway, wantReads: 2},
	} {
		t.Run(test.name, func(t *testing.T) {
			srv, _, _ := trainVMGalleryFixture(t)
			reader := srv.trainvm.(*galleryReadModel)
			reader.raceMode = test.raceMode
			request := httptest.NewRequest(http.MethodGet, "/api/trainvm/runs/vm-run/galleries", nil)
			response := httptest.NewRecorder()
			srv.Handler().ServeHTTP(response, request)
			if response.Code != test.wantStatus || reader.planReads != test.wantReads {
				t.Fatalf("race status=%d plan_reads=%d body=%s",
					response.Code, reader.planReads, response.Body.String())
			}
		})
	}
}

func TestTrainVMGalleryHistoryReportsArtifactTailTruncation(t *testing.T) {
	srv, _, _ := trainVMGalleryFixture(t)
	reader := srv.trainvm.(*galleryReadModel)
	reader.events = nil
	for sequence := 1; sequence <= trainVMGalleryMaxEvents+1; sequence++ {
		payload, err := json.Marshal(map[string]any{
			"artifact_id": fmt.Sprintf("report-%d", sequence), "logical_name": "report",
			"kind": "report", "schema": "test.report.v1", "uri": "file:///sealed/report",
			"size_bytes": 1, "fingerprint_algorithm": "sha256",
			"fingerprint": strings.Repeat("a", 64), "complete": true,
			"producer_node_id": "eval", "producer_attempt_id": "eval@2",
			"parent_artifact_ids": []string{}, "published_at_ns": int64(sequence),
		})
		if err != nil {
			t.Fatal(err)
		}
		reader.events = append(reader.events, trainvmstore.Event{
			Sequence: uint64(sequence), EventID: fmt.Sprintf("report-event-%d", sequence),
			RunID: "vm-run", NodeID: "eval", AttemptID: "eval@2",
			WorkerSequence: uint64(sequence), EventType: "artifact.published",
			EventVersion: 1, WallTimeNS: int64(sequence), Payload: payload,
		})
	}
	reader.run.LastEventSeq = trainVMGalleryMaxEvents + 1
	request := httptest.NewRequest(http.MethodGet, "/api/trainvm/runs/vm-run/galleries", nil)
	response := httptest.NewRecorder()
	srv.Handler().ServeHTTP(response, request)
	var history trainVMGalleryHistoryResponse
	if err := json.Unmarshal(response.Body.Bytes(), &history); err != nil ||
		response.Code != http.StatusOK || !history.HistoryTruncated || len(history.Galleries) != 0 {
		t.Fatalf("truncated gallery history status=%d body=%s", response.Code, response.Body.String())
	}
}

func TestTrainVMGalleryHistoryEnforcesRevisionAndAggregateManifestBounds(t *testing.T) {
	srv, _, _ := trainVMGalleryFixture(t)
	reader := srv.trainvm.(*galleryReadModel)
	base := reader.events[0]
	reader.events = nil
	for sequence := 1; sequence <= trainVMGalleryMaxRevisions+1; sequence++ {
		var payload map[string]any
		if err := json.Unmarshal(base.Payload, &payload); err != nil {
			t.Fatal(err)
		}
		payload["artifact_id"] = fmt.Sprintf("gallery-%d", sequence)
		payload["published_at_ns"] = int64(sequence)
		encoded, err := json.Marshal(payload)
		if err != nil {
			t.Fatal(err)
		}
		reader.events = append(reader.events, trainvmstore.Event{
			Sequence: uint64(sequence), EventID: fmt.Sprintf("gallery-event-%d", sequence),
			RunID: "vm-run", NodeID: "eval", AttemptID: "eval@2",
			WorkerSequence: uint64(sequence), EventType: "artifact.published",
			EventVersion: 1, WallTimeNS: int64(sequence), Payload: encoded,
		})
	}
	reader.run.LastEventSeq = trainVMGalleryMaxRevisions + 1
	request := httptest.NewRequest(http.MethodGet, "/api/trainvm/runs/vm-run/galleries", nil)
	response := httptest.NewRecorder()
	srv.Handler().ServeHTTP(response, request)
	var history trainVMGalleryHistoryResponse
	if err := json.Unmarshal(response.Body.Bytes(), &history); err != nil ||
		response.Code != http.StatusOK || !history.HistoryTruncated ||
		len(history.Galleries) != trainVMGalleryMaxRevisions {
		t.Fatalf("revision bound status=%d body=%s", response.Code, response.Body.String())
	}

	oversized := make([]trainvmstore.PublishedArtifact, 9)
	for index := range oversized {
		oversized[index].SizeBytes = trainVMGalleryMaxBytes
	}
	bounded, truncated, err := boundGalleryProjection(oversized)
	if err != nil || !truncated || len(bounded) != 8 {
		t.Fatalf("aggregate manifest budget was not enforced: count=%d truncated=%t err=%v",
			len(bounded), truncated, err)
	}
}

func TestTrainVMGalleryVerifiesCanonicalManifestDigestAndStepDomain(t *testing.T) {
	for _, test := range []struct {
		name   string
		mutate func(*trainVMGalleryManifest)
		reseal bool
	}{
		{
			name: "canonical digest",
			mutate: func(manifest *trainVMGalleryManifest) {
				manifest.CanonicalManifestDigest = "sha256:" + strings.Repeat("f", 64)
			},
		},
		{
			name: "step domain",
			mutate: func(manifest *trainVMGalleryManifest) {
				manifest.StepDomain = "trainer_guess"
			},
			reseal: true,
		},
	} {
		t.Run(test.name, func(t *testing.T) {
			srv, _, manifestPath := trainVMGalleryFixture(t)
			data, err := os.ReadFile(manifestPath)
			if err != nil {
				t.Fatal(err)
			}
			var manifest trainVMGalleryManifest
			if err := json.Unmarshal(data, &manifest); err != nil {
				t.Fatal(err)
			}
			test.mutate(&manifest)
			if test.reseal {
				replaceGalleryManifest(t, srv, manifestPath, manifest)
			} else {
				data, err = json.Marshal(manifest)
				if err != nil {
					t.Fatal(err)
				}
				replaceRawGalleryManifest(t, srv, manifestPath, data)
			}
			request := httptest.NewRequest(http.MethodGet, "/api/trainvm/runs/vm-run/galleries", nil)
			response := httptest.NewRecorder()
			srv.Handler().ServeHTTP(response, request)
			if response.Code != http.StatusBadGateway ||
				!strings.Contains(response.Body.String(), "identity or cardinality") {
				t.Fatalf("invalid manifest status=%d body=%s", response.Code, response.Body.String())
			}
		})
	}
}

func TestTrainVMGallerySummaryOrderingIsDeterministic(t *testing.T) {
	summaries := []trainVMGallerySummary{
		{ArtifactID: "sample-8", StepDomain: "sample", Step: 8, Sequence: 40},
		{ArtifactID: "optimizer-late", StepDomain: "optimizer_step", Step: 10, Sequence: 30},
		{ArtifactID: "optimizer-first-attempt", StepDomain: "optimizer_step", Step: 10, Sequence: 20},
		{ArtifactID: "optimizer-earlier-step", StepDomain: "optimizer_step", Step: 5, Sequence: 10},
	}
	sortGallerySummaries(summaries)
	got := []string{
		summaries[0].ArtifactID, summaries[1].ArtifactID,
		summaries[2].ArtifactID, summaries[3].ArtifactID,
	}
	want := []string{
		"optimizer-earlier-step", "optimizer-first-attempt", "optimizer-late", "sample-8",
	}
	if fmt.Sprint(got) != fmt.Sprint(want) {
		t.Fatalf("gallery ordering got %v want %v", got, want)
	}
}
