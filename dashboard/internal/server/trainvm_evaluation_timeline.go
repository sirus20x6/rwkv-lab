package server

import (
	"encoding/json"
	"errors"
	"net/http"
	"os"
	"sort"

	trainvmstore "trainboard/internal/trainvm"
)

// The evaluation timeline exists because a run's evidence arrives on several
// independent cadences and used to be rendered as if it were one. A scalar
// probe, a full validation read, a caption gallery and the final audit are
// different milestones with different costs; showing them as undifferentiated
// "eval" points is what let a 745-step run go two thirds of the way with no
// baseline and nobody noticing. Each milestone here keeps its own kind, its own
// value, and the checkpoint and frozen evaluation manifest it was read against.
const (
	evaluationTimelineMaxMetrics    = 20_000
	evaluationScalarFullMetric      = "eval.loss"
	evaluationScalarProbeMetric     = "eval.probe_loss"
	evaluationProbeExamplesMetric   = "eval.probe_examples"
	evaluationFinalAuditMetric      = "eval.test_loss"
	evaluationCadenceRevisionMetric = "eval.cadence_revision"
)

var evaluationPlannedMetrics = map[string]string{
	"eval.planned_scalar_full_milestones":  "scalar_full",
	"eval.planned_scalar_probe_milestones": "scalar_probe",
	"eval.planned_qualitative_milestones":  "qualitative",
}

type evaluationMilestone struct {
	Step                     uint64   `json:"step"`
	Kinds                    []string `json:"kinds"`
	ScalarLoss               *float64 `json:"scalar_loss,omitempty"`
	ProbeLoss                *float64 `json:"probe_loss,omitempty"`
	ProbeExamples            *float64 `json:"probe_examples,omitempty"`
	TestLoss                 *float64 `json:"test_loss,omitempty"`
	GalleryArtifactID        string   `json:"gallery_artifact_id,omitempty"`
	CheckpointManifestDigest string   `json:"checkpoint_manifest_digest,omitempty"`
	EvalManifestDigest       string   `json:"eval_manifest_digest,omitempty"`
}

type evaluationTimelineResponse struct {
	Milestones       []evaluationMilestone `json:"milestones"`
	Planned          map[string]uint64     `json:"planned"`
	Observed         map[string]uint64     `json:"observed"`
	CadenceRevision  uint64                `json:"cadence_revision"`
	HistoryTruncated bool                  `json:"history_truncated"`
}

// evaluationMetricFloat accepts every numeric shape a decoded scalar can arrive in. A
// metric whose value is not numeric is reported as absent rather than zero: a
// missing loss and a loss of zero must not render identically.
func evaluationMetricFloat(value any) (float64, bool) {
	switch typed := value.(type) {
	case float64:
		return typed, true
	case float32:
		return float64(typed), true
	case int64:
		return float64(typed), true
	case uint64:
		return float64(typed), true
	case int:
		return float64(typed), true
	case json.Number:
		parsed, err := typed.Float64()
		return parsed, err == nil
	}
	return 0, false
}

func addEvaluationKind(milestone *evaluationMilestone, kind string) {
	for _, existing := range milestone.Kinds {
		if existing == kind {
			return
		}
	}
	milestone.Kinds = append(milestone.Kinds, kind)
}

func buildEvaluationTimeline(
	galleries []trainVMGallerySummary,
	metrics []trainvmstore.MetricPoint,
	historyTruncated bool,
) evaluationTimelineResponse {
	milestones := map[uint64]*evaluationMilestone{}
	at := func(step uint64) *evaluationMilestone {
		milestone, ok := milestones[step]
		if !ok {
			milestone = &evaluationMilestone{Step: step, Kinds: []string{}}
			milestones[step] = milestone
		}
		return milestone
	}
	response := evaluationTimelineResponse{
		Planned:          map[string]uint64{},
		Observed:         map[string]uint64{},
		HistoryTruncated: historyTruncated,
	}

	for _, gallery := range galleries {
		milestone := at(gallery.Step)
		addEvaluationKind(milestone, "qualitative")
		milestone.GalleryArtifactID = gallery.ArtifactID
		milestone.CheckpointManifestDigest = gallery.CheckpointManifestDigest
		milestone.EvalManifestDigest = gallery.EvaluatorProfileDigest
	}

	for _, metric := range metrics {
		if metric.StepDomain != "optimizer_step" {
			continue
		}
		if kind, planned := evaluationPlannedMetrics[metric.Name]; planned {
			if value, ok := evaluationMetricFloat(metric.Value); ok && value >= 0 {
				response.Planned[kind] = uint64(value)
			}
			continue
		}
		if metric.Name == evaluationCadenceRevisionMetric {
			if value, ok := evaluationMetricFloat(metric.Value); ok && value >= 0 {
				response.CadenceRevision = uint64(value)
			}
			continue
		}
		value, ok := evaluationMetricFloat(metric.Value)
		if !ok {
			continue
		}
		switch metric.Name {
		case evaluationScalarFullMetric:
			milestone := at(metric.Step)
			addEvaluationKind(milestone, "scalar_full")
			scalar := value
			milestone.ScalarLoss = &scalar
		case evaluationScalarProbeMetric:
			milestone := at(metric.Step)
			addEvaluationKind(milestone, "scalar_probe")
			probe := value
			milestone.ProbeLoss = &probe
		case evaluationProbeExamplesMetric:
			examples := value
			at(metric.Step).ProbeExamples = &examples
		case evaluationFinalAuditMetric:
			milestone := at(metric.Step)
			addEvaluationKind(milestone, "final_audit")
			test := value
			milestone.TestLoss = &test
		}
	}

	response.Milestones = make([]evaluationMilestone, 0, len(milestones))
	for _, milestone := range milestones {
		if len(milestone.Kinds) == 0 {
			continue
		}
		sort.Strings(milestone.Kinds)
		for _, kind := range milestone.Kinds {
			response.Observed[kind]++
		}
		response.Milestones = append(response.Milestones, *milestone)
	}
	sort.Slice(response.Milestones, func(i, j int) bool {
		return response.Milestones[i].Step < response.Milestones[j].Step
	})
	return response
}

func (s *Server) handleTrainVMEvaluationTimeline(w http.ResponseWriter, r *http.Request) {
	if s.trainvm == nil {
		http.Error(w, "TrainVM authority is not configured", http.StatusServiceUnavailable)
		return
	}
	runID := r.PathValue("run")
	artifacts, historyTruncated, err := s.publishedGalleries(r.Context(), runID)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			http.Error(w, "no such TrainVM run or persisted plan", http.StatusNotFound)
			return
		}
		if errors.Is(err, errTrainVMGalleryHistoryBound) {
			http.Error(w, err.Error(), http.StatusUnprocessableEntity)
			return
		}
		writeTrainVMAuthorityError(w, err)
		return
	}
	artifacts, budgetTruncated, err := boundGalleryProjection(artifacts)
	if err != nil {
		http.Error(w, err.Error(), http.StatusUnprocessableEntity)
		return
	}
	galleries := make([]trainVMGallerySummary, 0, len(artifacts))
	for _, artifact := range artifacts {
		manifest, err := s.loadGallery(artifact)
		if err != nil {
			http.Error(w, err.Error(), http.StatusBadGateway)
			return
		}
		galleries = append(galleries, gallerySummary(artifact, manifest))
	}
	metrics, err := trainvmstore.Metrics(
		r.Context(), s.trainvm, runID, 0, evaluationTimelineMaxMetrics,
	)
	if err != nil {
		writeTrainVMAuthorityError(w, err)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Cache-Control", "no-store")
	_ = json.NewEncoder(w).Encode(
		buildEvaluationTimeline(galleries, metrics, historyTruncated || budgetTruncated),
	)
}
