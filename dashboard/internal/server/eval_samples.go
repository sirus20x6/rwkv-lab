package server

import (
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
)

var normalizedBoxPattern = regexp.MustCompile(
	`(?i)box\s*=\s*\[\s*([+-]?\d+)\s*,\s*([+-]?\d+)\s*,\s*([+-]?\d+)\s*,\s*([+-]?\d+)\s*\]`)

type evalBox struct {
	Label string `json:"label,omitempty"`
	X1    int    `json:"x1"`
	Y1    int    `json:"y1"`
	X2    int    `json:"x2"`
	Y2    int    `json:"y2"`
}

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
	Prompt         string    `json:"prompt"`
	Reference      string    `json:"reference"`
	Caption        string    `json:"caption"`
	Tokens         int       `json:"tokens"`
	StoppedAtEOD   bool      `json:"stopped_at_eod"`
	Source         string    `json:"source"`
	ImageURL       string    `json:"image_url"`
	TargetBoxes    []evalBox `json:"target_boxes,omitempty"`
	PredictedBoxes []evalBox `json:"predicted_boxes,omitempty"`
}

// parseNormalizedBoxes extracts the captioning-first 0..999 box contract.
// Invalid, inverted, and zero-area model generations are omitted rather than
// clamped: drawing a fabricated edge box would be more misleading than showing
// a zero prediction count in the gallery legend.
func parseNormalizedBoxes(text string) []evalBox {
	matches := normalizedBoxPattern.FindAllStringSubmatchIndex(text, -1)
	boxes := make([]evalBox, 0, len(matches))
	for _, match := range matches {
		coordinates := [4]int{}
		valid := true
		for i := range coordinates {
			value, err := strconv.Atoi(text[match[2+2*i]:match[3+2*i]])
			if err != nil || value < 0 || value > 999 {
				valid = false
				break
			}
			coordinates[i] = value
		}
		if !valid || coordinates[2] <= coordinates[0] || coordinates[3] <= coordinates[1] {
			continue
		}
		lineStart := strings.LastIndex(text[:match[0]], "\n") + 1
		label := strings.TrimSpace(strings.TrimSuffix(
			strings.TrimSpace(text[lineStart:match[0]]), ";"))
		boxes = append(boxes, evalBox{Label: label,
			X1: coordinates[0], Y1: coordinates[1],
			X2: coordinates[2], Y2: coordinates[3]})
	}
	return boxes
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

func (s *Server) writeEvalSampleResponse(w http.ResponseWriter, name string,
	step int64, artifact evalSampleArtifact) {
	items := make([]evalSampleResponseItem, len(artifact.Items))
	for i, item := range artifact.Items {
		items[i] = evalSampleResponseItem{
			Prompt: item.Prompt, Reference: item.Reference,
			Caption: item.Caption, Tokens: item.Tokens,
			StoppedAtEOD: item.StoppedAtEOD, Source: item.Source,
			TargetBoxes:    parseNormalizedBoxes(item.Reference),
			PredictedBoxes: parseNormalizedBoxes(item.Caption),
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
	s.writeEvalSampleResponse(w, name, step, artifact)
}

// handleLatestEvalSamples resolves the newest usable qualitative artifact at
// or before at_step. Scalar evals and qualitative evals intentionally have
// independent cadences, so the newest PPL marker often has no same-step JSON.
func (s *Server) handleLatestEvalSamples(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Cache-Control", "no-store")
	name := r.PathValue("name")
	if _, _, err := s.evalSamplePath(name, "0"); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	maxStep := int64(^uint64(0) >> 1)
	if raw := r.URL.Query().Get("at_step"); raw != "" {
		parsed, err := strconv.ParseInt(raw, 10, 64)
		if err != nil || parsed < 0 {
			http.Error(w, "invalid at_step", http.StatusBadRequest)
			return
		}
		maxStep = parsed
	}
	entries, err := os.ReadDir(filepath.Join(s.cfg.RunsDir, name, "eval_samples"))
	if os.IsNotExist(err) {
		http.Error(w, "no qualitative snapshots were recorded for this run", http.StatusNotFound)
		return
	}
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	var bestStep int64 = -1
	var bestPath string
	for _, entry := range entries {
		if entry.IsDir() {
			continue
		}
		namePart := entry.Name()
		if !strings.HasPrefix(namePart, "step_") || !strings.HasSuffix(namePart, ".json") {
			continue
		}
		rawStep := strings.TrimSuffix(strings.TrimPrefix(namePart, "step_"), ".json")
		step, err := strconv.ParseInt(rawStep, 10, 64)
		if err == nil && step <= maxStep && step > bestStep {
			bestStep = step
			bestPath = filepath.Join(s.cfg.RunsDir, name, "eval_samples", namePart)
		}
	}
	if bestStep < 0 {
		http.Error(w, "no qualitative snapshot exists at or before this eval", http.StatusNotFound)
		return
	}
	artifact, err := readEvalSample(bestPath)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	s.writeEvalSampleResponse(w, name, bestStep, artifact)
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
	// The artifact JSON is trainer-written but lives on disk; never let a
	// tampered/garbage path turn this endpoint into an arbitrary file read.
	image, ok := s.resolveEvalImage(artifact.Items[index].Image)
	if !ok {
		http.Error(w, "image path is outside the allowed data roots", http.StatusForbidden)
		return
	}
	if info, statErr := os.Stat(image); statErr != nil || !info.Mode().IsRegular() {
		http.Error(w, "image is no longer available", http.StatusNotFound)
		return
	}
	http.ServeFile(w, r, image)
}

// resolveEvalImage resolves an artifact-listed image path (following symlinks)
// and requires the result to live inside one of the allowed roots
// (cfg.ImageRoots, defaulting to the runs dir and repo root). Returns the
// resolved path to serve, so a post-check symlink swap cannot redirect it.
func (s *Server) resolveEvalImage(image string) (string, bool) {
	if image == "" || !filepath.IsAbs(image) {
		return "", false
	}
	resolved, err := filepath.EvalSymlinks(filepath.Clean(image))
	if err != nil {
		return "", false
	}
	roots := s.cfg.ImageRoots
	if len(roots) == 0 {
		roots = []string{s.cfg.RunsDir, s.cfg.RepoRoot}
	}
	for _, root := range roots {
		if root == "" {
			continue
		}
		if resolvedRoot, err := filepath.EvalSymlinks(filepath.Clean(root)); err == nil {
			root = resolvedRoot
		}
		if rel, err := filepath.Rel(root, resolved); err == nil && rel != ".." &&
			!strings.HasPrefix(rel, ".."+string(filepath.Separator)) {
			return resolved, true
		}
	}
	return "", false
}
