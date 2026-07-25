package server

import (
	"encoding/json"
	"fmt"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
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
	artifact := evalSampleArtifact{Step: 100, PPL: 7.5, Decoding: "greedy", MaxNew: 64,
		Items: []evalSampleItem{{Image: image, Prompt: "Describe this image:\n",
			Reference: "cat; box=[100,200,700,800]", Caption: "cat; box=[120,210,690,790]",
			Source: "eval"}}}
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
		!strings.Contains(w.Body.String(), `?v=`) ||
		!strings.Contains(w.Body.String(), `"caption":"cat; box=[120,210,690,790]"`) ||
		!strings.Contains(w.Body.String(), `"target_boxes":[{"label":"cat","x1":100,"y1":200,"x2":700,"y2":800}]`) ||
		!strings.Contains(w.Body.String(), `"predicted_boxes":[{"label":"cat","x1":120,"y1":210,"x2":690,"y2":790}]`) ||
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
