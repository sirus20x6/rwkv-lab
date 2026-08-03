package server

import (
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"math"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"time"
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
	Image          string              `json:"image"`
	TargetImage    string              `json:"target_image,omitempty"`
	Prompt         string              `json:"prompt"`
	Reference      string              `json:"reference"`
	Caption        string              `json:"caption"`
	Tokens         int                 `json:"tokens"`
	StoppedAtEOD   bool                `json:"stopped_at_eod"`
	Source         string              `json:"source"`
	StructuredHead *evalStructuredHead `json:"structured_head,omitempty"`
}

type evalStructuredHeadInstance struct {
	Query      int       `json:"query"`
	Objectness float64   `json:"objectness"`
	BoxXYXY    []float64 `json:"box_xyxy"`
	MaskShape  []int     `json:"mask_shape"`
	MaskSpans  string    `json:"mask_spans"`
}

type evalStructuredHead struct {
	Format    string                       `json:"format"`
	Instances []evalStructuredHeadInstance `json:"instances"`
}

type evalSampleArtifact struct {
	Step                 int64              `json:"step"`
	PPL                  float64            `json:"ppl"`
	EvalKind             string             `json:"eval_kind,omitempty"`
	Decoding             string             `json:"decoding"`
	MaxNew               int                `json:"max_new"`
	Complete             *bool              `json:"complete,omitempty"`
	GenerationSteps      int                `json:"generation_steps,omitempty"`
	OCRGeneration        map[string]float64 `json:"ocr_generation,omitempty"`
	StructuredGeneration map[string]float64 `json:"structured_generation,omitempty"`
	Items                []evalSampleItem   `json:"items"`
}

type evalSampleResponseItem struct {
	Prompt            string              `json:"prompt"`
	Reference         string              `json:"reference"`
	ReferenceNegative bool                `json:"reference_negative"`
	Caption           string              `json:"caption"`
	Tokens            int                 `json:"tokens"`
	StoppedAtEOD      bool                `json:"stopped_at_eod"`
	Source            string              `json:"source"`
	ImageURL          string              `json:"image_url"`
	TargetImageURL    string              `json:"target_image_url,omitempty"`
	TargetBoxes       []evalBox           `json:"target_boxes,omitempty"`
	PredictedBoxes    []evalBox           `json:"predicted_boxes,omitempty"`
	HeadBoxes         []evalBox           `json:"head_boxes,omitempty"`
	StructuredHead    *evalStructuredHead `json:"structured_head,omitempty"`
}

func structuredHeadBoxes(head *evalStructuredHead) []evalBox {
	if head == nil {
		return nil
	}
	boxes := make([]evalBox, 0, len(head.Instances))
	for _, instance := range head.Instances {
		if len(instance.BoxXYXY) != 4 {
			continue
		}
		values := [4]int{}
		valid := true
		for i, value := range instance.BoxXYXY {
			if math.IsNaN(value) || math.IsInf(value, 0) || value < 0 || value > 1 {
				valid = false
				break
			}
			values[i] = int(math.Round(value * 999))
		}
		if !valid || values[2] <= values[0] || values[3] <= values[1] {
			continue
		}
		boxes = append(boxes, evalBox{
			X1: values[0], Y1: values[1], X2: values[2], Y2: values[3]})
	}
	return boxes
}

// conceptSegmentationPromptLead is the opening prose of CONCEPT_SAM_PROMPT in
// scripts/build_captioning_first_mix.py. It is a CROSS-REPO CONTRACT: nothing
// in the build script links to this file, so a reword there would silently
// stop matching here. It is therefore only the LAST resort — recognition below
// leads with the machine-readable output contract, which a prose reword does
// not touch, and TestNegativeStructuredReferenceStatesTheCrossRepoContract
// pins both halves.
const conceptSegmentationPromptLead = "Segment every visible instance matching the concept"

// structuredPromptMarkers are the fragments of the structured/detection output
// contract that a template must state to be answerable with `none`. Matching on
// these rather than on the prompt's prose means (a) a reworded lead-in keeps
// working and (b) detection templates other than concept segmentation — which
// the old prefix test ignored entirely — are covered too.
var structuredPromptMarkers = []string{
	"normalized box [x1,y1,x2,y2]",
	"mask row spans",
	"box=[",
	"mask16=",
	strings.ToLower(conceptSegmentationPromptLead),
}

// isStructuredPrompt reports whether a prompt asks for the structured
// box/mask output contract, for which a bare `none` is a real negative answer
// rather than a caption that happens to read "none".
func isStructuredPrompt(prompt string) bool {
	folded := strings.ToLower(prompt)
	for _, marker := range structuredPromptMarkers {
		if strings.Contains(folded, marker) {
			return true
		}
	}
	return false
}

func isNegativeStructuredReference(prompt, reference string) bool {
	return strings.EqualFold(strings.TrimSpace(reference), "none") &&
		isStructuredPrompt(prompt)
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
	return evalSamplePathToken(artifact, index, artifact.Items[index].Image)
}

func evalSampleTargetImageToken(artifact evalSampleArtifact, index int) string {
	if index < 0 || index >= len(artifact.Items) {
		return ""
	}
	return evalSamplePathToken(artifact, index, artifact.Items[index].TargetImage)
}

func evalSamplePathToken(artifact evalSampleArtifact, index int, image string) string {
	fileVersion := ""
	if info, err := os.Stat(image); err == nil {
		fileVersion = fmt.Sprintf("%d\x00%d", info.Size(), info.ModTime().UnixNano())
	}
	digest := sha256.Sum256([]byte(fmt.Sprintf("%d\x00%.17g\x00%s\x00%s",
		artifact.Step, artifact.PPL, image, fileVersion)))
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
			ReferenceNegative: isNegativeStructuredReference(
				item.Prompt, item.Reference),
			TargetBoxes:    parseNormalizedBoxes(item.Reference),
			PredictedBoxes: parseNormalizedBoxes(item.Caption),
			HeadBoxes:      structuredHeadBoxes(item.StructuredHead),
			StructuredHead: item.StructuredHead,
			ImageURL: fmt.Sprintf("/api/runs/%s/eval-samples/%d/image/%d?v=%s",
				url.PathEscape(name), step, i, evalSampleImageToken(artifact, i)),
		}
		if item.TargetImage != "" {
			items[i].TargetImageURL = fmt.Sprintf(
				"/api/runs/%s/eval-samples/%d/target-image/%d?v=%s",
				url.PathEscape(name), step, i,
				evalSampleTargetImageToken(artifact, i))
		}
	}
	w.Header().Set("Content-Type", "application/json")
	complete := true // artifacts written before resumable generation predate this field
	if artifact.Complete != nil {
		complete = *artifact.Complete
	}
	_ = json.NewEncoder(w).Encode(map[string]any{
		"step": artifact.Step, "ppl": artifact.PPL, "decoding": artifact.Decoding,
		"eval_kind": artifact.EvalKind,
		"max_new":   artifact.MaxNew, "complete": complete,
		"generation_steps":      artifact.GenerationSteps,
		"ocr_generation":        artifact.OCRGeneration,
		"structured_generation": artifact.StructuredGeneration,
		"items":                 items,
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

// evalSampleIndexTTL bounds how often one run's eval_samples directory is
// re-listed. app.js retries this endpoint every 2s while it waits for a
// gallery, and the directory accumulates one JSON per eval step over a long
// run, so an unbounded ReadDir per request grows without limit.
const evalSampleIndexTTL = 2 * time.Second

// evalSampleIndex is one run's cached, ascending list of snapshot steps.
type evalSampleIndex struct {
	steps   []int64
	modTime time.Time
	fetched time.Time
}

// evalSampleSteps returns the snapshot steps for a run, reusing a recent
// listing when the directory's mtime is unchanged. Directory mtime moves on
// every create/rename, which is exactly how the trainer publishes snapshots,
// so a stale index cannot hide a new one for longer than the TTL.
func (s *Server) evalSampleSteps(name string) ([]int64, error) {
	dir := filepath.Join(s.cfg.RunsDir, name, "eval_samples")
	info, err := os.Stat(dir)
	if err != nil {
		return nil, err
	}
	now := time.Now()
	s.evalIndexMu.Lock()
	cached, ok := s.evalIndex[name]
	s.evalIndexMu.Unlock()
	if ok && cached.modTime.Equal(info.ModTime()) &&
		now.Sub(cached.fetched) < evalSampleIndexTTL {
		return cached.steps, nil
	}

	entries, err := os.ReadDir(dir)
	if err != nil {
		return nil, err
	}
	steps := make([]int64, 0, len(entries))
	for _, entry := range entries {
		if entry.IsDir() {
			continue
		}
		namePart := entry.Name()
		if !strings.HasPrefix(namePart, "step_") || !strings.HasSuffix(namePart, ".json") {
			continue
		}
		rawStep := strings.TrimSuffix(strings.TrimPrefix(namePart, "step_"), ".json")
		if step, err := strconv.ParseInt(rawStep, 10, 64); err == nil {
			steps = append(steps, step)
		}
	}
	sort.Slice(steps, func(i, j int) bool { return steps[i] < steps[j] })

	s.evalIndexMu.Lock()
	if s.evalIndex == nil {
		s.evalIndex = map[string]*evalSampleIndex{}
	}
	s.evalIndex[name] = &evalSampleIndex{
		steps: steps, modTime: info.ModTime(), fetched: now,
	}
	s.evalIndexMu.Unlock()
	return steps, nil
}

// handleEvalSampleIndex exposes the discrete qualitative checkpoints used by
// the gallery scrubber. The slider indexes this ascending list rather than
// pretending that every training step has an artifact.
func (s *Server) handleEvalSampleIndex(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Cache-Control", "no-store")
	name := r.PathValue("name")
	if _, _, err := s.evalSamplePath(name, "0"); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	steps, err := s.evalSampleSteps(name)
	if os.IsNotExist(err) {
		http.Error(w, "no qualitative snapshots were recorded for this run", http.StatusNotFound)
		return
	}
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	firstStep, lastStep := int64(0), int64(0)
	if len(steps) > 0 {
		firstStep, lastStep = steps[0], steps[len(steps)-1]
	}
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(map[string]any{
		"steps": steps, "count": len(steps),
		"first_step": firstStep, "last_step": lastStep,
	})
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
	steps, err := s.evalSampleSteps(name)
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
	// steps is ascending: the last entry at or before maxStep is the newest.
	if i := sort.Search(len(steps), func(i int) bool { return steps[i] > maxStep }); i > 0 {
		bestStep = steps[i-1]
		bestPath = filepath.Join(s.cfg.RunsDir, name, "eval_samples",
			fmt.Sprintf("step_%08d.json", bestStep))
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
	s.serveEvalSampleImage(w, r, false)
}

func (s *Server) handleEvalSampleTargetImage(w http.ResponseWriter, r *http.Request) {
	s.serveEvalSampleImage(w, r, true)
}

func (s *Server) serveEvalSampleImage(
	w http.ResponseWriter, r *http.Request, target bool,
) {
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
	imagePath := artifact.Items[index].Image
	expectedToken := evalSampleImageToken(artifact, index)
	if target {
		imagePath = artifact.Items[index].TargetImage
		expectedToken = evalSampleTargetImageToken(artifact, index)
	}
	if token := r.URL.Query().Get("v"); token == "" || token != expectedToken {
		http.Error(w, "eval snapshot changed; refresh the card", http.StatusConflict)
		return
	}
	// The artifact JSON is trainer-written but lives on disk; never let a
	// tampered/garbage path turn this endpoint into an arbitrary file read.
	image, ok := s.resolveEvalImage(imagePath)
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
