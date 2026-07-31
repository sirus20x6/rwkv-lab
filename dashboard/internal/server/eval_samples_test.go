package server

import (
	"encoding/json"
	"fmt"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestEvalSampleJSONAndImageEndpoints(t *testing.T) {
	runs := t.TempDir()
	runDir := filepath.Join(runs, "vision", "eval_samples")
	if err := os.MkdirAll(runDir, 0o755); err != nil {
		t.Fatal(err)
	}
	image := filepath.Join(runs, "heldout.jpg")
	if err := os.WriteFile(image, []byte("fake-jpeg"), 0o644); err != nil {
		t.Fatal(err)
	}
	targetImage := filepath.Join(runs, "target.jpg")
	if err := os.WriteFile(targetImage, []byte("target-jpeg"), 0o644); err != nil {
		t.Fatal(err)
	}
	artifact := evalSampleArtifact{Step: 100, PPL: 7.5, Decoding: "greedy", MaxNew: 64,
		OCRGeneration: map[string]float64{
			"normalized_cer": 0.4, "wer": 0.6, "eod_rate": 0.5,
		},
		StructuredGeneration: map[string]float64{
			"box_iou": 0.25, "mask_dice": 0.5, "precision_at_50": 0.2,
		},
		Items: []evalSampleItem{{Image: image, TargetImage: targetImage,
			Prompt:    "Describe this image:\n",
			Reference: "cat; box=[100,200,700,800]", Caption: "cat; box=[120,210,690,790]",
			Source: "eval", StructuredHead: &evalStructuredHead{
				Format: "native_feature_grid",
				Instances: []evalStructuredHeadInstance{{
					Query: 2, Objectness: 0.75,
					BoxXYXY:   []float64{0.1, 0.2, 0.7, 0.8},
					MaskShape: []int{30, 40}, MaskSpans: "0:1-2",
				}},
			}}}}
	data, _ := json.Marshal(artifact)
	if err := os.WriteFile(filepath.Join(runDir, "step_00000100.json"), data, 0o644); err != nil {
		t.Fatal(err)
	}
	s := &Server{cfg: Config{RunsDir: runs}}

	req := httptest.NewRequest("GET", "/api/runs/vision/eval-samples/100", nil)
	req.SetPathValue("name", "vision")
	req.SetPathValue("step", "100")
	w := httptest.NewRecorder()
	s.handleEvalSamples(w, req)
	if w.Code != 200 || !strings.Contains(w.Body.String(), `"image_url"`) ||
		!strings.Contains(w.Body.String(), `"target_image_url"`) ||
		!strings.Contains(w.Body.String(), `?v=`) ||
		!strings.Contains(w.Body.String(), `"caption":"cat; box=[120,210,690,790]"`) ||
		!strings.Contains(w.Body.String(), `"target_boxes":[{"label":"cat","x1":100,"y1":200,"x2":700,"y2":800}]`) ||
		!strings.Contains(w.Body.String(), `"predicted_boxes":[{"label":"cat","x1":120,"y1":210,"x2":690,"y2":790}]`) ||
		!strings.Contains(w.Body.String(), `"head_boxes":[{"x1":100,"y1":200,"x2":699,"y2":799}]`) ||
		!strings.Contains(w.Body.String(), `"structured_head":{"format":"native_feature_grid"`) ||
		!strings.Contains(w.Body.String(), `"structured_generation":{"box_iou":0.25`) ||
		!strings.Contains(w.Body.String(), `"ocr_generation":{"eod_rate":0.5`) ||
		!strings.Contains(w.Body.String(), `"complete":true`) ||
		!strings.Contains(w.Body.String(), `"prompt":"Describe this image:\n"`) {
		t.Fatalf("unexpected response %d: %s", w.Code, w.Body.String())
	}
	if got := w.Header().Get("Cache-Control"); got != "no-store" {
		t.Fatalf("mutable eval artifact cache policy = %q", got)
	}

	token := evalSampleImageToken(artifact, 0)
	req = httptest.NewRequest("GET", "/api/runs/vision/eval-samples/100/image/0?v="+token, nil)
	req.SetPathValue("name", "vision")
	req.SetPathValue("step", "100")
	req.SetPathValue("index", "0")
	w = httptest.NewRecorder()
	s.handleEvalSampleImage(w, req)
	if w.Code != 200 || w.Body.String() != "fake-jpeg" {
		t.Fatalf("unexpected image response %d: %q", w.Code, w.Body.String())
	}
	if got := w.Header().Get("Cache-Control"); got != "private, no-store" {
		t.Fatalf("same-step replacement image cache policy = %q", got)
	}

	targetToken := evalSampleTargetImageToken(artifact, 0)
	req = httptest.NewRequest("GET",
		"/api/runs/vision/eval-samples/100/target-image/0?v="+targetToken, nil)
	req.SetPathValue("name", "vision")
	req.SetPathValue("step", "100")
	req.SetPathValue("index", "0")
	w = httptest.NewRecorder()
	s.handleEvalSampleTargetImage(w, req)
	if w.Code != 200 || w.Body.String() != "target-jpeg" {
		t.Fatalf("unexpected target image response %d: %q", w.Code, w.Body.String())
	}

	// If the trainer atomically replaces this same-step artifact between the
	// JSON and image requests, the old card must never receive the new index's
	// image under its stable path.
	artifact.PPL = 6.25
	data, _ = json.Marshal(artifact)
	if err := os.WriteFile(filepath.Join(runDir, "step_00000100.json"), data, 0o644); err != nil {
		t.Fatal(err)
	}
	req = httptest.NewRequest("GET", "/api/runs/vision/eval-samples/100/image/0?v="+token, nil)
	req.SetPathValue("name", "vision")
	req.SetPathValue("step", "100")
	req.SetPathValue("index", "0")
	w = httptest.NewRecorder()
	s.handleEvalSampleImage(w, req)
	if w.Code != 409 {
		t.Fatalf("old image generation token returned %d after artifact replacement", w.Code)
	}
}

func TestParseNormalizedBoxesRejectsMisleadingInvalidPredictions(t *testing.T) {
	boxes := parseNormalizedBoxes(strings.Join([]string{
		"person; box=[006,020,900,998]; mask16=1:1-2",
		"instance 2; box=[400,300,200,500]",  // inverted
		"instance 3; box=[000,000,000,000]",  // zero area
		"instance 4; box=[100,100,1000,900]", // outside contract
		"box=[010,011,012,013]",              // label is optional
	}, "\n"))
	if len(boxes) != 2 {
		t.Fatalf("parsed boxes = %#v", boxes)
	}
	if boxes[0] != (evalBox{Label: "person", X1: 6, Y1: 20, X2: 900, Y2: 998}) {
		t.Fatalf("first box = %#v", boxes[0])
	}
	if boxes[1] != (evalBox{X1: 10, Y1: 11, X2: 12, Y2: 13}) {
		t.Fatalf("unlabelled box = %#v", boxes[1])
	}
}

func TestNegativeStructuredReferenceIsExplicitlyLabeled(t *testing.T) {
	prompt := `Segment every visible instance matching the concept "puppet". ` +
		"Return `none` when the concept is absent:"
	if !isNegativeStructuredReference(prompt, "none") {
		t.Fatal("structured absence was not recognized as a negative reference")
	}
	if isNegativeStructuredReference("Transcribe all visible text:", "none") {
		t.Fatal("non-structured literal none was mislabeled as a negative reference")
	}
	if isNegativeStructuredReference(prompt, "instance 1; box=[1,2,3,4]") {
		t.Fatal("positive structured target was mislabeled as negative")
	}
}

// TestNegativeStructuredReferenceStatesTheCrossRepoContract documents the
// coupling this package has to the trainer's prompt templates, and pins that a
// prose reword on the Python side cannot silently revert every negative to a
// plain "reference" label showing a bare `none`.
//
// CONTRACT (this repo, not editable from here):
//
//	scripts/build_captioning_first_mix.py CONCEPT_SAM_PROMPT reads
//	  'Segment every visible instance matching the concept "{concept}". Return
//	   one line per instance with normalized box [x1,y1,x2,y2] and 16x16 mask
//	   row spans. Return `none` when the concept is absent:'
//
// Only the machine-readable half of that -- the box/mask output contract -- is
// load-bearing here; the lead-in is recognized as a convenience.
func TestNegativeStructuredReferenceStatesTheCrossRepoContract(t *testing.T) {
	const conceptSAMPrompt = `Segment every visible instance matching the concept "puppet". ` +
		"Return one line per instance with normalized box [x1,y1,x2,y2] and " +
		"16x16 mask row spans. Return `none` when the concept is absent:\n"
	if !isNegativeStructuredReference(conceptSAMPrompt, "none") {
		t.Fatal("the shipped CONCEPT_SAM_PROMPT is no longer recognized")
	}
	// A reword of the lead-in prose must not break recognition: the output
	// contract still identifies the template.
	reworded := "Find and outline each instance of the concept \"puppet\". " +
		"Return one line per instance with normalized box [x1,y1,x2,y2] and " +
		"16x16 mask row spans. Return `none` when the concept is absent:\n"
	if !isNegativeStructuredReference(reworded, "NONE") {
		t.Fatal("a reworded structured prompt silently lost its negative label")
	}
	// Detection templates other than concept segmentation are covered too --
	// the old prefix-only test left every one of them unlabeled.
	detection := "List every traffic sign as `label; box=[x1,y1,x2,y2]`, " +
		"or `none` if there are no signs:\n"
	if !isNegativeStructuredReference(detection, "none") {
		t.Fatal("a non-segmentation structured template stayed unlabeled")
	}
	// A caption task whose model happened to answer "none" is NOT a structured
	// negative; mislabeling it would assert an absence the task never asked about.
	if isNegativeStructuredReference("Describe this image accurately:\n", "none") {
		t.Fatal("a captioning reference was mislabeled as a structured negative")
	}
}

func TestEvalSampleImageRejectsPathsOutsideAllowedRoots(t *testing.T) {
	runs := t.TempDir()
	runDir := filepath.Join(runs, "vision", "eval_samples")
	if err := os.MkdirAll(runDir, 0o755); err != nil {
		t.Fatal(err)
	}
	outside := t.TempDir() // not under RunsDir/RepoRoot/ImageRoots
	secret := filepath.Join(outside, "secret.txt")
	if err := os.WriteFile(secret, []byte("top-secret"), 0o644); err != nil {
		t.Fatal(err)
	}
	s := &Server{cfg: Config{RunsDir: runs}}
	for _, image := range []string{"/etc/passwd", secret, "relative/path.jpg",
		filepath.Join(runs, "..", filepath.Base(outside), "secret.txt")} {
		artifact := evalSampleArtifact{Step: 100, PPL: 7.5,
			Items: []evalSampleItem{{Image: image}}}
		data, _ := json.Marshal(artifact)
		if err := os.WriteFile(filepath.Join(runDir, "step_00000100.json"), data, 0o644); err != nil {
			t.Fatal(err)
		}
		token := evalSampleImageToken(artifact, 0)
		req := httptest.NewRequest("GET", "/api/runs/vision/eval-samples/100/image/0?v="+token, nil)
		req.SetPathValue("name", "vision")
		req.SetPathValue("step", "100")
		req.SetPathValue("index", "0")
		w := httptest.NewRecorder()
		s.handleEvalSampleImage(w, req)
		if w.Code != 403 {
			t.Fatalf("artifact image %q served with status %d: %q", image, w.Code, w.Body.String())
		}
	}

	// A symlink inside the allowed root pointing outside must also be rejected.
	link := filepath.Join(runs, "escape.jpg")
	if err := os.Symlink(secret, link); err != nil {
		t.Fatal(err)
	}
	artifact := evalSampleArtifact{Step: 100, PPL: 7.5,
		Items: []evalSampleItem{{Image: link}}}
	data, _ := json.Marshal(artifact)
	if err := os.WriteFile(filepath.Join(runDir, "step_00000100.json"), data, 0o644); err != nil {
		t.Fatal(err)
	}
	req := httptest.NewRequest("GET", "/api/runs/vision/eval-samples/100/image/0?v="+
		evalSampleImageToken(artifact, 0), nil)
	req.SetPathValue("name", "vision")
	req.SetPathValue("step", "100")
	req.SetPathValue("index", "0")
	w := httptest.NewRecorder()
	s.handleEvalSampleImage(w, req)
	if w.Code != 403 {
		t.Fatalf("symlink escape served with status %d: %q", w.Code, w.Body.String())
	}
}

func TestEvalSamplesRejectParentDirectoryRunName(t *testing.T) {
	s := &Server{cfg: Config{RunsDir: t.TempDir()}}
	req := httptest.NewRequest("GET", "/api/runs/../eval-samples/100", nil)
	req.SetPathValue("name", "..")
	req.SetPathValue("step", "100")
	w := httptest.NewRecorder()
	s.handleEvalSamples(w, req)
	if w.Code != 400 {
		t.Fatalf("parent-directory run name returned %d", w.Code)
	}
}

func TestLatestEvalSamplesSelectsNewestArtifactAtOrBeforeStep(t *testing.T) {
	runs := t.TempDir()
	dir := filepath.Join(runs, "vision", "eval_samples")
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	for _, step := range []int64{21000, 22000, 23000} {
		artifact := evalSampleArtifact{Step: step, PPL: float64(step),
			Items: []evalSampleItem{{Caption: "caption"}}}
		data, _ := json.Marshal(artifact)
		path := filepath.Join(dir, fmt.Sprintf("step_%08d.json", step))
		if err := os.WriteFile(path, data, 0o644); err != nil {
			t.Fatal(err)
		}
	}
	s := &Server{cfg: Config{RunsDir: runs}}
	req := httptest.NewRequest("GET", "/api/runs/vision/eval-samples/latest?at_step=22500", nil)
	req.SetPathValue("name", "vision")
	w := httptest.NewRecorder()
	s.handleLatestEvalSamples(w, req)
	if w.Code != 200 || !strings.Contains(w.Body.String(), `"step":22000`) {
		t.Fatalf("latest response %d: %s", w.Code, w.Body.String())
	}
}

func TestEvalSampleIndexReturnsAscendingDiscreteSteps(t *testing.T) {
	runs := t.TempDir()
	dir := filepath.Join(runs, "vision", "eval_samples")
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	for _, name := range []string{
		"step_00023000.json",
		"step_00021000.json",
		"notes.txt",
		"step_invalid.json",
		"step_00022000.json",
	} {
		if err := os.WriteFile(filepath.Join(dir, name), []byte("{}"), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	s := &Server{cfg: Config{RunsDir: runs}}
	req := httptest.NewRequest("GET", "/api/runs/vision/eval-samples", nil)
	req.SetPathValue("name", "vision")
	w := httptest.NewRecorder()
	s.handleEvalSampleIndex(w, req)
	if w.Code != 200 {
		t.Fatalf("index response %d: %s", w.Code, w.Body.String())
	}
	var response struct {
		Steps     []int64 `json:"steps"`
		Count     int     `json:"count"`
		FirstStep int64   `json:"first_step"`
		LastStep  int64   `json:"last_step"`
	}
	if err := json.Unmarshal(w.Body.Bytes(), &response); err != nil {
		t.Fatal(err)
	}
	if fmt.Sprint(response.Steps) != "[21000 22000 23000]" ||
		response.Count != 3 || response.FirstStep != 21000 || response.LastStep != 23000 {
		t.Fatalf("unexpected index: %#v", response)
	}
	if got := w.Header().Get("Cache-Control"); got != "no-store" {
		t.Fatalf("eval index cache policy = %q", got)
	}
}

// TestLatestEvalSamplesIndexSeesNewSnapshots guards the cached directory
// listing added for the 2s gallery retry loop: the cache keys on the
// directory's mtime, which moves whenever the trainer publishes a snapshot, so
// a new artifact must never be hidden behind a stale index.
func TestLatestEvalSamplesIndexSeesNewSnapshots(t *testing.T) {
	runs := t.TempDir()
	dir := filepath.Join(runs, "vision", "eval_samples")
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	write := func(step int64) {
		artifact := evalSampleArtifact{Step: step, PPL: float64(step),
			Items: []evalSampleItem{{Caption: "caption"}}}
		data, _ := json.Marshal(artifact)
		path := filepath.Join(dir, fmt.Sprintf("step_%08d.json", step))
		if err := os.WriteFile(path, data, 0o644); err != nil {
			t.Fatal(err)
		}
	}
	s := &Server{cfg: Config{RunsDir: runs}, evalIndex: map[string]*evalSampleIndex{}}
	write(100)
	req := httptest.NewRequest("GET", "/api/runs/vision/eval-samples/latest", nil)
	req.SetPathValue("name", "vision")
	w := httptest.NewRecorder()
	s.handleLatestEvalSamples(w, req)
	if w.Code != 200 || !strings.Contains(w.Body.String(), `"step":100`) {
		t.Fatalf("first listing %d: %s", w.Code, w.Body.String())
	}
	// A second request inside the TTL is served from the cache.
	w = httptest.NewRecorder()
	s.handleLatestEvalSamples(w, req)
	if w.Code != 200 || !strings.Contains(w.Body.String(), `"step":100`) {
		t.Fatalf("cached listing %d: %s", w.Code, w.Body.String())
	}
	// Publishing bumps the directory mtime, which must invalidate the cache
	// regardless of the TTL.
	write(200)
	if err := os.Chtimes(dir, time.Now().Add(time.Second), time.Now().Add(time.Second)); err != nil {
		t.Fatal(err)
	}
	w = httptest.NewRecorder()
	s.handleLatestEvalSamples(w, req)
	if w.Code != 200 || !strings.Contains(w.Body.String(), `"step":200`) {
		t.Fatalf("new snapshot hidden behind a stale index %d: %s", w.Code, w.Body.String())
	}
}
