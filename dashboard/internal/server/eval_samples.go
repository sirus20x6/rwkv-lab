package server

import (
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"strconv"
	"strings"
)

type evalSampleItem struct {
	Image        string `json:"image"`
	Prompt       string `json:"prompt"`
	Reference    string `json:"reference"`
	Caption      string `json:"caption"`
	Tokens       int    `json:"tokens"`
	StoppedAtEOD bool   `json:"stopped_at_eod"`
	Source       string `json:"source"`
}

type evalSampleArtifact struct {
	Step            int64            `json:"step"`
	PPL             float64          `json:"ppl"`
	Decoding        string           `json:"decoding"`
	MaxNew          int              `json:"max_new"`
	Complete        *bool            `json:"complete,omitempty"`
	GenerationSteps int              `json:"generation_steps,omitempty"`
	Items           []evalSampleItem `json:"items"`
}

type evalSampleResponseItem struct {
	Prompt       string `json:"prompt"`
	Reference    string `json:"reference"`
	Caption      string `json:"caption"`
	Tokens       int    `json:"tokens"`
	StoppedAtEOD bool   `json:"stopped_at_eod"`
	Source       string `json:"source"`
	ImageURL     string `json:"image_url"`
}

func evalSampleImageToken(artifact evalSampleArtifact, index int) string {
	if index < 0 || index >= len(artifact.Items) {
		return ""
	}
	digest := sha256.Sum256([]byte(fmt.Sprintf("%d\x00%.17g\x00%s",
		artifact.Step, artifact.PPL, artifact.Items[index].Image)))
	return fmt.Sprintf("%x", digest[:16])
}

func (s *Server) evalSamplePath(name, rawStep string) (string, int64, error) {
	if name == "" || name == "." || name == ".." || name != filepath.Base(name) ||
		strings.ContainsAny(name, `/\\`) {
		return "", 0, fmt.Errorf("invalid run name")
	}
	step, err := strconv.ParseInt(rawStep, 10, 64)
	if err != nil || step < 0 {
		return "", 0, fmt.Errorf("invalid eval step")
	}
	return filepath.Join(s.cfg.RunsDir, name, "eval_samples",
		fmt.Sprintf("step_%08d.json", step)), step, nil
}

func readEvalSample(path string) (evalSampleArtifact, error) {
	var artifact evalSampleArtifact
	data, err := os.ReadFile(path)
	if err != nil {
		return artifact, err
	}
	err = json.Unmarshal(data, &artifact)
	return artifact, err
}

func (s *Server) handleEvalSamples(w http.ResponseWriter, r *http.Request) {
	// The trainer atomically rewrites this document every few generated tokens.
	// Reusing a cached incomplete response would make the card appear stuck.
	w.Header().Set("Cache-Control", "no-store")
	name, rawStep := r.PathValue("name"), r.PathValue("step")
	path, step, err := s.evalSamplePath(name, rawStep)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	artifact, err := readEvalSample(path)
	if os.IsNotExist(err) {
		http.Error(w, "no qualitative snapshot was recorded for this eval", http.StatusNotFound)
		return
	}
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	items := make([]evalSampleResponseItem, len(artifact.Items))
	for i, item := range artifact.Items {
		items[i] = evalSampleResponseItem{
			Prompt: item.Prompt, Reference: item.Reference,
			Caption: item.Caption, Tokens: item.Tokens,
			StoppedAtEOD: item.StoppedAtEOD, Source: item.Source,
			ImageURL: fmt.Sprintf("/api/runs/%s/eval-samples/%d/image/%d?v=%s",
				url.PathEscape(name), step, i, evalSampleImageToken(artifact, i)),
		}
	}
	w.Header().Set("Content-Type", "application/json")
	complete := true // artifacts written before resumable generation predate this field
	if artifact.Complete != nil {
		complete = *artifact.Complete
	}
	_ = json.NewEncoder(w).Encode(map[string]any{
		"step": artifact.Step, "ppl": artifact.PPL, "decoding": artifact.Decoding,
		"max_new": artifact.MaxNew, "complete": complete,
		"generation_steps": artifact.GenerationSteps, "items": items,
	})
}

func (s *Server) handleEvalSampleImage(w http.ResponseWriter, r *http.Request) {
	// The route is stable across checkpoint recovery, but its artifact generation
	// is not. Always revalidate the generation token below.
	w.Header().Set("Cache-Control", "private, no-store")
	path, _, err := s.evalSamplePath(r.PathValue("name"), r.PathValue("step"))
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	index, err := strconv.Atoi(r.PathValue("index"))
	if err != nil || index < 0 {
		http.Error(w, "invalid image index", http.StatusBadRequest)
		return
	}
	artifact, err := readEvalSample(path)
	if err != nil {
		http.Error(w, "qualitative snapshot not found", http.StatusNotFound)
		return
	}
	if index >= len(artifact.Items) {
		http.Error(w, "image index out of range", http.StatusNotFound)
		return
	}
	if token := r.URL.Query().Get("v"); token == "" ||
		token != evalSampleImageToken(artifact, index) {
		http.Error(w, "eval snapshot changed; refresh the card", http.StatusConflict)
		return
	}
	image := artifact.Items[index].Image
	if info, statErr := os.Stat(image); statErr != nil || !info.Mode().IsRegular() {
		http.Error(w, "image is no longer available", http.StatusNotFound)
		return
	}
	http.ServeFile(w, r, image)
}
