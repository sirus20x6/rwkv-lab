package server

import (
	"encoding/json"
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
			Reference: "reference", Caption: "generated", Source: "eval"}}}
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
		!strings.Contains(w.Body.String(), `"caption":"generated"`) ||
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
