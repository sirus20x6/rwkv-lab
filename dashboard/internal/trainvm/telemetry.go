package trainvm

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"sort"
	"strconv"
	"strings"
)

type EventQuery struct {
	RunID                 string
	After                 uint64
	Through               uint64
	Limit                 int
	EventTypes            []string
	NewestFirst           bool
	NewestPerMetricSeries bool
}

const maximumLiveMetricSeries = 512

func normalizeEventQuery(input EventQuery) (EventQuery, error) {
	input.RunID = strings.TrimSpace(input.RunID)
	if input.RunID == "" || len(input.RunID) > 256 {
		return EventQuery{}, &ValidationError{Message: "bounded run ID is required"}
	}
	if input.Limit <= 0 || input.Limit > 1_000 {
		input.Limit = 250
	}
	if input.Through != 0 && input.After >= input.Through {
		return EventQuery{}, &ValidationError{Message: "event upper fence must exceed the after cursor"}
	}
	if input.NewestFirst && input.Through == 0 {
		return EventQuery{}, &ValidationError{Message: "newest-first event scans require an upper fence"}
	}
	if input.NewestPerMetricSeries &&
		(input.Through == 0 || input.NewestFirst || len(input.EventTypes) != 1 ||
			input.EventTypes[0] != "metric.sampled") {
		return EventQuery{}, &ValidationError{Message: "newest-per-series scans require an upper fence and only metric.sampled"}
	}
	if len(input.EventTypes) > 64 {
		return EventQuery{}, &ValidationError{Message: "event-type filter exceeds 64 values"}
	}
	unique := make(map[string]struct{}, len(input.EventTypes))
	for index, eventType := range input.EventTypes {
		eventType = strings.TrimSpace(eventType)
		if eventType == "" || len(eventType) > 256 {
			return EventQuery{}, &ValidationError{Message: "event-type filter is malformed"}
		}
		if _, exists := unique[eventType]; exists {
			return EventQuery{}, &ValidationError{Message: "event-type filters must be unique"}
		}
		unique[eventType] = struct{}{}
		input.EventTypes[index] = eventType
	}
	sort.Strings(input.EventTypes)
	return input, nil
}

type MetricPoint struct {
	Sequence       uint64            `json:"sequence"`
	RunID          string            `json:"run_id"`
	NodeID         string            `json:"node_id"`
	AttemptID      string            `json:"attempt_id"`
	Name           string            `json:"name"`
	Type           string            `json:"type"`
	Value          any               `json:"value"`
	Unit           string            `json:"unit"`
	StepDomain     string            `json:"step_domain"`
	Aggregation    string            `json:"aggregation"`
	Description    string            `json:"description,omitempty"`
	Step           uint64            `json:"step"`
	SampleWeight   float64           `json:"sample_weight"`
	Labels         map[string]string `json:"labels"`
	ObservedAtNS   int64             `json:"observed_at_ns"`
	WorkerSequence uint64            `json:"worker_sequence"`
	OptimizerStep  *uint64           `json:"optimizer_step,omitempty"`
}

type MetricDescriptor struct {
	Name        string `json:"name"`
	Type        string `json:"type"`
	Unit        string `json:"unit"`
	StepDomain  string `json:"step_domain"`
	Aggregation string `json:"aggregation"`
	Description string `json:"description,omitempty"`
}

type ObservabilityDeclaration struct {
	HeartbeatSeconds     uint64             `json:"heartbeat_seconds"`
	Metrics              []MetricDescriptor `json:"metrics"`
	RetainRawMetricsDays uint64             `json:"retain_raw_metrics_days"`
	EvalGalleryArtifact  string             `json:"eval_gallery_artifact,omitempty"`
	LogArtifact          string             `json:"log_artifact,omitempty"`
}

func (o ObservabilityDeclaration) Metric(name string) (MetricDescriptor, bool) {
	for _, descriptor := range o.Metrics {
		if descriptor.Name == name {
			return descriptor, true
		}
	}
	return MetricDescriptor{}, false
}

func ObservabilityFromCompiledPlan(plan CompiledPlanView) (ObservabilityDeclaration, error) {
	var document struct {
		Spec struct {
			Observability json.RawMessage `json:"observability"`
		} `json:"spec"`
	}
	if err := json.Unmarshal(plan.CanonicalPlan, &document); err != nil ||
		len(document.Spec.Observability) == 0 {
		return ObservabilityDeclaration{}, fmt.Errorf("TrainVM compiled plan has no observability declaration")
	}
	var raw struct {
		HeartbeatSeconds     uint64            `json:"heartbeat_seconds"`
		Metrics              []json.RawMessage `json:"metrics"`
		RetainRawMetricsDays uint64            `json:"retain_raw_metrics_days"`
		EvalGalleryArtifact  string            `json:"eval_gallery_artifact,omitempty"`
		LogArtifact          string            `json:"log_artifact,omitempty"`
	}
	if err := strictPayload(document.Spec.Observability, &raw); err != nil ||
		raw.HeartbeatSeconds < 1 || raw.HeartbeatSeconds > 300 ||
		raw.RetainRawMetricsDays < 1 || len(raw.Metrics) > 256 ||
		!validBoundedText(raw.EvalGalleryArtifact, 256, true) ||
		!validBoundedText(raw.LogArtifact, 256, true) {
		return ObservabilityDeclaration{}, fmt.Errorf("TrainVM observability declaration is malformed")
	}
	result := ObservabilityDeclaration{
		HeartbeatSeconds: raw.HeartbeatSeconds, RetainRawMetricsDays: raw.RetainRawMetricsDays,
		EvalGalleryArtifact: raw.EvalGalleryArtifact, LogArtifact: raw.LogArtifact,
		Metrics: make([]MetricDescriptor, 0, len(raw.Metrics)),
	}
	seen := make(map[string]struct{}, len(raw.Metrics))
	for _, encoded := range raw.Metrics {
		var descriptor MetricDescriptor
		if err := strictPayload(encoded, &descriptor); err != nil ||
			!validBoundedText(descriptor.Name, 256, false) ||
			!validBoundedText(descriptor.Unit, 128, false) ||
			!validBoundedText(descriptor.Description, 2048, true) ||
			!validMetricType(descriptor.Type) ||
			!validStepDomain(descriptor.StepDomain) ||
			!validAggregation(descriptor.Aggregation) {
			return ObservabilityDeclaration{}, fmt.Errorf("TrainVM metric declaration is malformed")
		}
		if _, duplicate := seen[descriptor.Name]; duplicate {
			return ObservabilityDeclaration{}, fmt.Errorf("TrainVM metric declaration %q is duplicated", descriptor.Name)
		}
		seen[descriptor.Name] = struct{}{}
		result.Metrics = append(result.Metrics, descriptor)
	}
	return result, nil
}

type WorkerHeartbeatPoint struct {
	Sequence  uint64 `json:"sequence"`
	RunID     string `json:"run_id"`
	NodeID    string `json:"node_id"`
	AttemptID string `json:"attempt_id"`
	Phase     string `json:"phase"`
	// ExecutionPhase is "compile", "warmup", or empty outside a phase. Unlike
	// Phase, which is a free-text operator label, the authority accepted this
	// only against the attempt's immutable phase requests.
	ExecutionPhase string `json:"execution_phase"`
	OptimizerStep  uint64 `json:"optimizer_step"`
	ObservedAtNS   int64  `json:"observed_at_ns"`
	AcceptedAtNS   int64  `json:"accepted_at_ns"`
	WorkerSequence uint64 `json:"worker_sequence"`
}

type ExecutionPhaseDiagnostic struct {
	Severity     string `json:"severity"`
	Code         string `json:"code"`
	DocumentPath string `json:"document_path"`
	Message      string `json:"message"`
	Help         string `json:"help"`
}

type ExecutionPhaseReceiptPoint struct {
	Sequence               uint64                     `json:"sequence"`
	RunID                  string                     `json:"run_id"`
	NodeID                 string                     `json:"node_id"`
	AttemptID              string                     `json:"attempt_id"`
	WorkerSequence         uint64                     `json:"worker_sequence"`
	Phase                  string                     `json:"phase"`
	Enabled                bool                       `json:"enabled"`
	RequestedSteps         uint64                     `json:"requested_steps"`
	RequestDigest          string                     `json:"request_digest"`
	Disposition            string                     `json:"disposition"`
	StepsExecuted          uint64                     `json:"steps_executed"`
	StateFingerprintBefore string                     `json:"state_fingerprint_before"`
	StateFingerprintAfter  string                     `json:"state_fingerprint_after"`
	StartedAtNS            int64                      `json:"started_at_ns"`
	CompletedAtNS          int64                      `json:"completed_at_ns"`
	Diagnostics            []ExecutionPhaseDiagnostic `json:"diagnostics"`
}

type TelemetrySnapshot struct {
	JournalID       string                       `json:"journal_id"`
	Run             Run                          `json:"run"`
	Observability   ObservabilityDeclaration     `json:"observability"`
	AfterSequence   uint64                       `json:"after_sequence"`
	TargetSequence  uint64                       `json:"target_sequence"`
	NextSequence    uint64                       `json:"next_sequence"`
	ReplayPending   bool                         `json:"replay_pending"`
	CaughtUp        bool                         `json:"caught_up"`
	Heartbeats      []WorkerHeartbeatPoint       `json:"heartbeats"`
	Metrics         []MetricPoint                `json:"metrics"`
	Artifacts       []ObservableArtifact         `json:"artifacts"`
	ExecutionPhases []ExecutionPhaseReceiptPoint `json:"execution_phases"`
}

type PublishedArtifact struct {
	Sequence             uint64   `json:"sequence"`
	RunID                string   `json:"run_id"`
	ArtifactID           string   `json:"artifact_id"`
	LogicalName          string   `json:"logical_name"`
	Kind                 string   `json:"kind"`
	Schema               string   `json:"schema"`
	URI                  string   `json:"uri"`
	SizeBytes            uint64   `json:"size_bytes"`
	FingerprintAlgorithm string   `json:"fingerprint_algorithm"`
	Fingerprint          string   `json:"fingerprint"`
	Complete             bool     `json:"complete"`
	ProducerNodeID       string   `json:"producer_node_id"`
	ProducerAttemptID    string   `json:"producer_attempt_id"`
	ParentArtifactIDs    []string `json:"parent_artifact_ids"`
	PublishedAtNS        int64    `json:"published_at_ns"`
	WorkerSequence       uint64   `json:"worker_sequence"`
}

// ObservableArtifact omits the worker-authored storage URI. Browser clients
// operate on immutable artifact identities and verified action URLs; they do
// not need host filesystem topology.
type ObservableArtifact struct {
	Sequence             uint64   `json:"sequence"`
	RunID                string   `json:"run_id"`
	ArtifactID           string   `json:"artifact_id"`
	LogicalName          string   `json:"logical_name"`
	Kind                 string   `json:"kind"`
	Schema               string   `json:"schema"`
	SizeBytes            uint64   `json:"size_bytes"`
	FingerprintAlgorithm string   `json:"fingerprint_algorithm"`
	Fingerprint          string   `json:"fingerprint"`
	Complete             bool     `json:"complete"`
	ProducerNodeID       string   `json:"producer_node_id"`
	ProducerAttemptID    string   `json:"producer_attempt_id"`
	ParentArtifactIDs    []string `json:"parent_artifact_ids"`
	PublishedAtNS        int64    `json:"published_at_ns"`
	WorkerSequence       uint64   `json:"worker_sequence"`
}

// RedactArtifact removes the worker-authored storage locator before data
// crosses into a browser-facing projection.
func RedactArtifact(artifact PublishedArtifact) ObservableArtifact {
	return ObservableArtifact{
		Sequence: artifact.Sequence, RunID: artifact.RunID,
		ArtifactID: artifact.ArtifactID, LogicalName: artifact.LogicalName,
		Kind: artifact.Kind, Schema: artifact.Schema, SizeBytes: artifact.SizeBytes,
		FingerprintAlgorithm: artifact.FingerprintAlgorithm,
		Fingerprint:          artifact.Fingerprint, Complete: artifact.Complete,
		ProducerNodeID:    artifact.ProducerNodeID,
		ProducerAttemptID: artifact.ProducerAttemptID,
		ParentArtifactIDs: append([]string(nil), artifact.ParentArtifactIDs...),
		PublishedAtNS:     artifact.PublishedAtNS, WorkerSequence: artifact.WorkerSequence,
	}
}

func strictPayload(raw json.RawMessage, output any) error {
	decoder := json.NewDecoder(bytes.NewReader(raw))
	decoder.DisallowUnknownFields()
	decoder.UseNumber()
	if err := decoder.Decode(output); err != nil {
		return err
	}
	var trailing any
	if err := decoder.Decode(&trailing); err != io.EOF {
		return fmt.Errorf("payload contains trailing JSON")
	}
	return nil
}

func validBoundedText(value string, maximum int, allowEmpty bool) bool {
	return (allowEmpty || value != "") && len(value) <= maximum &&
		!strings.ContainsRune(value, '\x00')
}

func validMetricType(value string) bool {
	switch value {
	case "counter", "gauge", "histogram":
		return true
	default:
		return false
	}
}

func validStepDomain(value string) bool {
	switch value {
	case "microbatch", "optimizer_step", "sample", "token", "epoch", "wall_time":
		return true
	default:
		return false
	}
}

func validAggregation(value string) bool {
	switch value {
	case "last", "sum", "mean", "weighted_mean", "min", "max", "histogram":
		return true
	default:
		return false
	}
}

func validMetricScalar(value any) bool {
	switch typed := value.(type) {
	case json.Number:
		parsed, err := strconv.ParseFloat(string(typed), 64)
		return err == nil && !math.IsInf(parsed, 0) && !math.IsNaN(parsed)
	case bool:
		return true
	case string:
		return len([]byte(typed)) <= 4096 && !strings.ContainsRune(typed, '\x00')
	default:
		return false
	}
}

func heartbeatFromEvent(event Event) (WorkerHeartbeatPoint, error) {
	var payload struct {
		Phase          string `json:"phase"`
		ExecutionPhase string `json:"execution_phase"`
		ObservedAtNS   int64  `json:"observed_at_ns"`
	}
	if err := strictPayload(event.Payload, &payload); err != nil ||
		event.EventType != "worker.heartbeat" || event.Sequence == 0 ||
		event.WorkerSequence == 0 || event.OptimizerStep == nil ||
		!validBoundedText(event.RunID, 256, false) ||
		!validBoundedText(event.NodeID, 256, false) ||
		!validBoundedText(event.AttemptID, 256, false) ||
		!validBoundedText(payload.Phase, 128, false) ||
		(payload.ExecutionPhase != "" && payload.ExecutionPhase != "compile" &&
			payload.ExecutionPhase != "warmup") ||
		// The authority refuses a label that claims a phase the typed field
		// does not name. Re-check it here rather than trust the producer, so a
		// journal written by an older or patched authority cannot render a
		// compile the run never ran.
		((payload.Phase == "compile" || payload.Phase == "warmup") &&
			payload.Phase != payload.ExecutionPhase) ||
		payload.ObservedAtNS < 0 || event.WallTimeNS < 0 {
		return WorkerHeartbeatPoint{}, fmt.Errorf("TrainVM heartbeat event %q has a malformed payload", event.EventID)
	}
	return WorkerHeartbeatPoint{
		Sequence: event.Sequence, RunID: event.RunID, NodeID: event.NodeID,
		AttemptID: event.AttemptID, Phase: payload.Phase,
		ExecutionPhase: payload.ExecutionPhase,
		OptimizerStep:  *event.OptimizerStep, ObservedAtNS: payload.ObservedAtNS,
		AcceptedAtNS: event.WallTimeNS, WorkerSequence: event.WorkerSequence,
	}, nil
}

func validSHA256Digest(value string) bool {
	if len(value) != 71 || !strings.HasPrefix(value, "sha256:") {
		return false
	}
	for _, character := range value[7:] {
		if (character < '0' || character > '9') &&
			(character < 'a' || character > 'f') {
			return false
		}
	}
	return true
}

func executionPhaseReceiptFromEvent(event Event) (ExecutionPhaseReceiptPoint, error) {
	var payload struct {
		Phase                  string                     `json:"phase"`
		Enabled                bool                       `json:"enabled"`
		RequestedSteps         uint64                     `json:"requested_steps"`
		RequestDigest          string                     `json:"request_digest"`
		Disposition            string                     `json:"disposition"`
		StepsExecuted          uint64                     `json:"steps_executed"`
		StateFingerprintBefore string                     `json:"state_fingerprint_before"`
		StateFingerprintAfter  string                     `json:"state_fingerprint_after"`
		StartedAtNS            int64                      `json:"started_at_ns"`
		CompletedAtNS          int64                      `json:"completed_at_ns"`
		Diagnostics            []ExecutionPhaseDiagnostic `json:"diagnostics"`
	}
	if err := strictPayload(event.Payload, &payload); err != nil ||
		event.EventType != "worker.execution_phase_receipted" ||
		event.Sequence == 0 || event.WorkerSequence == 0 ||
		event.OptimizerStep != nil || event.WallTimeNS != payload.CompletedAtNS ||
		!validBoundedText(event.RunID, 256, false) ||
		!validBoundedText(event.NodeID, 256, false) ||
		!validBoundedText(event.AttemptID, 256, false) ||
		(payload.Phase != "compile" && payload.Phase != "warmup") ||
		(payload.Disposition != "completed" && payload.Disposition != "skipped" &&
			payload.Disposition != "failed" && payload.Disposition != "cancelled") ||
		!validSHA256Digest(payload.RequestDigest) ||
		!validSHA256Digest(payload.StateFingerprintBefore) ||
		!validSHA256Digest(payload.StateFingerprintAfter) ||
		payload.StartedAtNS < 0 || payload.CompletedAtNS < payload.StartedAtNS ||
		payload.Diagnostics == nil || len(payload.Diagnostics) > 64 ||
		(payload.Enabled && payload.Disposition == "skipped") ||
		(!payload.Enabled && payload.Disposition != "skipped") ||
		(payload.Phase == "compile" && payload.RequestedSteps != 0) ||
		(payload.Disposition == "completed" && payload.StepsExecuted != payload.RequestedSteps) ||
		(payload.Disposition == "skipped" && payload.StepsExecuted != 0) ||
		// A cancellation shares the failure bounds — it stopped part-way, so a
		// partial step count is legal and diagnostics must say what stopped it
		// — but not the failure exemption below: cancelling an enabled phase
		// still has to leave the trajectory exactly where it found it.
		((payload.Disposition == "failed" || payload.Disposition == "cancelled") &&
			(payload.StepsExecuted > payload.RequestedSteps || len(payload.Diagnostics) == 0)) ||
		(payload.Disposition == "cancelled" && !payload.Enabled) ||
		(payload.Disposition != "failed" &&
			payload.StateFingerprintBefore != payload.StateFingerprintAfter) {
		return ExecutionPhaseReceiptPoint{}, fmt.Errorf(
			"TrainVM execution-phase event %q has a malformed payload", event.EventID)
	}
	for _, diagnostic := range payload.Diagnostics {
		if (diagnostic.Severity != "info" && diagnostic.Severity != "warning" &&
			diagnostic.Severity != "error") ||
			!validBoundedText(diagnostic.Code, 256, false) ||
			!validBoundedText(diagnostic.Message, 4096, false) ||
			len(diagnostic.DocumentPath) > 1024 || len(diagnostic.Help) > 4096 {
			return ExecutionPhaseReceiptPoint{}, fmt.Errorf(
				"TrainVM execution-phase event %q has malformed diagnostics", event.EventID)
		}
	}
	return ExecutionPhaseReceiptPoint{
		Sequence: event.Sequence, RunID: event.RunID, NodeID: event.NodeID,
		AttemptID: event.AttemptID, WorkerSequence: event.WorkerSequence,
		Phase: payload.Phase, Enabled: payload.Enabled,
		RequestedSteps: payload.RequestedSteps, RequestDigest: payload.RequestDigest,
		Disposition: payload.Disposition, StepsExecuted: payload.StepsExecuted,
		StateFingerprintBefore: payload.StateFingerprintBefore,
		StateFingerprintAfter:  payload.StateFingerprintAfter,
		StartedAtNS:            payload.StartedAtNS, CompletedAtNS: payload.CompletedAtNS,
		Diagnostics: payload.Diagnostics,
	}, nil
}

func metricFromEvent(event Event, declared *MetricDescriptor) (MetricPoint, error) {
	var payload struct {
		Name         string            `json:"name"`
		Value        any               `json:"value"`
		Unit         string            `json:"unit"`
		StepDomain   string            `json:"step_domain"`
		Step         uint64            `json:"step"`
		SampleWeight float64           `json:"sample_weight"`
		Labels       map[string]string `json:"labels"`
	}
	if err := strictPayload(event.Payload, &payload); err != nil ||
		event.EventType != "metric.sampled" || event.Sequence == 0 ||
		event.WorkerSequence == 0 ||
		!validBoundedText(event.RunID, 256, false) ||
		!validBoundedText(event.NodeID, 256, false) ||
		!validBoundedText(event.AttemptID, 256, false) ||
		!validBoundedText(payload.Name, 256, false) ||
		!validBoundedText(payload.Unit, 128, false) ||
		!validStepDomain(payload.StepDomain) || !validMetricScalar(payload.Value) ||
		payload.Labels == nil || len(payload.Labels) > 16 ||
		math.IsInf(payload.SampleWeight, 0) || math.IsNaN(payload.SampleWeight) ||
		payload.SampleWeight <= 0 || event.WallTimeNS < 0 {
		return MetricPoint{}, fmt.Errorf("TrainVM metric event %q has a malformed payload", event.EventID)
	}
	for key, value := range payload.Labels {
		if !validBoundedText(key, 128, false) || !validBoundedText(value, 256, false) {
			return MetricPoint{}, fmt.Errorf("TrainVM metric event %q has malformed labels", event.EventID)
		}
	}
	if (payload.StepDomain == "optimizer_step" &&
		(event.OptimizerStep == nil || *event.OptimizerStep != payload.Step)) ||
		(payload.StepDomain != "optimizer_step" && event.OptimizerStep != nil) {
		return MetricPoint{}, fmt.Errorf("TrainVM metric event %q has incoherent step semantics", event.EventID)
	}
	point := MetricPoint{
		Sequence: event.Sequence, RunID: event.RunID, NodeID: event.NodeID,
		AttemptID: event.AttemptID, Name: payload.Name, Value: payload.Value,
		Unit: payload.Unit, StepDomain: payload.StepDomain, Step: payload.Step,
		SampleWeight: payload.SampleWeight, Labels: payload.Labels,
		ObservedAtNS: event.WallTimeNS, WorkerSequence: event.WorkerSequence,
		OptimizerStep: event.OptimizerStep,
	}
	if declared != nil {
		if declared.Name != payload.Name || declared.Unit != payload.Unit ||
			declared.StepDomain != payload.StepDomain {
			return MetricPoint{}, fmt.Errorf("TrainVM metric event %q disagrees with its declaration", event.EventID)
		}
		point.Type, point.Aggregation, point.Description =
			declared.Type, declared.Aggregation, declared.Description
	}
	return point, nil
}

func artifactFromEvent(event Event) (PublishedArtifact, error) {
	var payload struct {
		ArtifactID           string   `json:"artifact_id"`
		LogicalName          string   `json:"logical_name"`
		Kind                 string   `json:"kind"`
		Schema               string   `json:"schema"`
		URI                  string   `json:"uri"`
		SizeBytes            uint64   `json:"size_bytes"`
		FingerprintAlgorithm string   `json:"fingerprint_algorithm"`
		Fingerprint          string   `json:"fingerprint"`
		Complete             bool     `json:"complete"`
		ProducerNodeID       string   `json:"producer_node_id"`
		ProducerAttemptID    string   `json:"producer_attempt_id"`
		ParentArtifactIDs    []string `json:"parent_artifact_ids"`
		PublishedAtNS        int64    `json:"published_at_ns"`
	}
	// External profilers wrap the worker process and finish their report only
	// after the worker exits. Those artifacts are controller-authored sequence
	// zero events; every other publication remains worker-sequenced.
	if err := strictPayload(event.Payload, &payload); err != nil ||
		event.EventType != "artifact.published" || event.Sequence == 0 ||
		(event.WorkerSequence == 0 && !(payload.Kind == "opaque" && payload.Schema == "trainvm.gpu-trace.v1")) ||
		event.OptimizerStep != nil ||
		payload.ArtifactID == "" || payload.LogicalName == "" ||
		payload.Kind == "" || payload.URI == "" ||
		payload.FingerprintAlgorithm == "" || payload.Fingerprint == "" ||
		!payload.Complete || payload.ProducerNodeID != event.NodeID ||
		payload.ProducerAttemptID != event.AttemptID ||
		payload.ParentArtifactIDs == nil || payload.PublishedAtNS < 0 ||
		payload.PublishedAtNS != event.WallTimeNS {
		return PublishedArtifact{}, fmt.Errorf("TrainVM artifact event %q has a malformed payload", event.EventID)
	}
	parents := make(map[string]struct{}, len(payload.ParentArtifactIDs))
	for _, parent := range payload.ParentArtifactIDs {
		if parent == "" || parent == payload.ArtifactID {
			return PublishedArtifact{}, fmt.Errorf("TrainVM artifact event %q has malformed lineage", event.EventID)
		}
		if _, duplicate := parents[parent]; duplicate {
			return PublishedArtifact{}, fmt.Errorf("TrainVM artifact event %q has duplicate lineage", event.EventID)
		}
		parents[parent] = struct{}{}
	}
	return PublishedArtifact{
		Sequence: event.Sequence, RunID: event.RunID,
		ArtifactID: payload.ArtifactID, LogicalName: payload.LogicalName,
		Kind: payload.Kind, Schema: payload.Schema, URI: payload.URI,
		SizeBytes: payload.SizeBytes, FingerprintAlgorithm: payload.FingerprintAlgorithm,
		Fingerprint: payload.Fingerprint, Complete: payload.Complete,
		ProducerNodeID:    payload.ProducerNodeID,
		ProducerAttemptID: payload.ProducerAttemptID,
		ParentArtifactIDs: payload.ParentArtifactIDs,
		PublishedAtNS:     payload.PublishedAtNS,
		WorkerSequence:    event.WorkerSequence,
	}, nil
}

func Metrics(ctx context.Context, reader ReadModel, runID string, after uint64, limit int) ([]MetricPoint, error) {
	events, err := reader.Events(ctx, EventQuery{
		RunID: runID, After: after, Limit: limit,
		EventTypes: []string{"metric.sampled"},
	})
	if err != nil {
		return nil, err
	}
	result := make([]MetricPoint, 0, len(events))
	for _, event := range events {
		point, err := metricFromEvent(event, nil)
		if err != nil {
			return nil, err
		}
		result = append(result, point)
	}
	return result, nil
}

func Artifacts(ctx context.Context, reader ReadModel, runID string, after uint64, limit int) ([]PublishedArtifact, error) {
	events, err := reader.Events(ctx, EventQuery{
		RunID: runID, After: after, Limit: limit,
		EventTypes: []string{"artifact.published"},
	})
	if err != nil {
		return nil, err
	}
	result := make([]PublishedArtifact, 0, len(events))
	for _, event := range events {
		artifact, err := artifactFromEvent(event)
		if err != nil {
			return nil, err
		}
		result = append(result, artifact)
	}
	return result, nil
}

// RecentArtifacts returns a bounded newest-first tail captured against one
// durable run prefix. It is intended for specialized dashboard projections
// that verify manifests without scanning an entire long-running history.
func RecentArtifacts(
	ctx context.Context, reader ReadModel, runID string, limit int,
) ([]PublishedArtifact, bool, error) {
	if limit < 1 || limit > 1_000 {
		return nil, false, &ValidationError{Message: "recent artifact limit must be from 1 through 1000"}
	}
	run, found, err := reader.Run(ctx, runID)
	if err != nil || !found {
		return nil, found, err
	}
	if run.LastEventSeq == 0 {
		return []PublishedArtifact{}, true, nil
	}
	result, _, err := RecentArtifactsThrough(
		ctx, reader, runID, run.LastEventSeq, limit,
	)
	if err != nil {
		return nil, true, err
	}
	return result, true, nil
}

// RecentArtifactsThrough returns a newest-first artifact tail from one caller-
// captured durable run prefix. Truncated is true only when an older artifact
// exists beyond the returned tail; callers must never present that tail as a
// complete history.
func RecentArtifactsThrough(
	ctx context.Context, reader ReadModel, runID string, through uint64, limit int,
) ([]PublishedArtifact, bool, error) {
	if limit < 1 || limit > 1_000 {
		return nil, false, &ValidationError{Message: "recent artifact limit must be from 1 through 1000"}
	}
	if through == 0 {
		return []PublishedArtifact{}, false, nil
	}
	events, err := reader.Events(ctx, EventQuery{
		RunID: runID, After: 0, Through: through, Limit: limit,
		EventTypes: []string{"artifact.published"}, NewestFirst: true,
	})
	if err != nil {
		return nil, false, err
	}
	if len(events) > limit {
		return nil, false, fmt.Errorf("TrainVM authority exceeded the bounded artifact tail")
	}
	result := make([]PublishedArtifact, 0, len(events))
	var previous uint64
	for index, event := range events {
		if event.RunID != runID || event.Sequence == 0 || event.Sequence > through ||
			(index > 0 && event.Sequence >= previous) {
			return nil, false, fmt.Errorf("TrainVM authority returned an incoherent artifact tail")
		}
		previous = event.Sequence
		artifact, parseErr := artifactFromEvent(event)
		if parseErr != nil {
			return nil, false, parseErr
		}
		result = append(result, artifact)
	}
	truncated := false
	if len(events) == limit && events[len(events)-1].Sequence > 1 {
		older, olderErr := reader.Events(ctx, EventQuery{
			RunID: runID, After: 0, Through: events[len(events)-1].Sequence - 1,
			Limit: 1, EventTypes: []string{"artifact.published"}, NewestFirst: true,
		})
		if olderErr != nil {
			return nil, false, olderErr
		}
		if len(older) > 1 {
			return nil, false, fmt.Errorf("TrainVM authority exceeded the artifact truncation sentinel")
		}
		if len(older) == 1 {
			if older[0].RunID != runID || older[0].Sequence == 0 ||
				older[0].Sequence >= events[len(events)-1].Sequence {
				return nil, false, fmt.Errorf("TrainVM authority returned an incoherent artifact truncation sentinel")
			}
			truncated = true
		}
	}
	return result, truncated, nil
}

// CaptureRunPlanPrefix obtains one coherent journal/run/compiled-plan identity.
// The reads are retried once so a concurrent plan adoption can settle; a
// continuing race fails closed rather than mixing declarations and events.
func CaptureRunPlanPrefix(
	ctx context.Context, reader ReadModel, runID string,
) (string, Run, CompiledPlanView, bool, error) {
	for attempt := 0; attempt < 2; attempt++ {
		journalBefore, err := reader.JournalID(ctx)
		if err != nil {
			return "", Run{}, CompiledPlanView{}, false, err
		}
		run, found, err := reader.Run(ctx, runID)
		if err != nil || !found {
			return "", Run{}, CompiledPlanView{}, found, err
		}
		plan, found, err := reader.CompiledPlan(ctx, runID)
		if err != nil || !found {
			return "", Run{}, CompiledPlanView{}, found, err
		}
		journalAfter, err := reader.JournalID(ctx)
		if err != nil {
			return "", Run{}, CompiledPlanView{}, false, err
		}
		if journalBefore == journalAfter && journalBefore == plan.JournalID &&
			run.RunID == runID && plan.RunID == runID &&
			run.RunRevision == plan.RunRevision && run.PlanHash == plan.PlanHash {
			return journalBefore, run, plan, true, nil
		}
	}
	return "", Run{}, CompiledPlanView{}, false,
		fmt.Errorf("TrainVM run changed while its plan-bound artifact prefix was being captured")
}

var telemetryEventTypes = []string{
	"artifact.published",
	"metric.sampled",
	"worker.execution_phase_receipted",
	"worker.heartbeat",
}

func telemetryHeader(ctx context.Context, reader ReadModel, runID string) (
	string, Run, CompiledPlanView, bool, error,
) {
	for attempt := 0; attempt < 2; attempt++ {
		journalID, err := reader.JournalID(ctx)
		if err != nil {
			return "", Run{}, CompiledPlanView{}, false, err
		}
		run, found, err := reader.Run(ctx, runID)
		if err != nil || !found {
			return "", Run{}, CompiledPlanView{}, found, err
		}
		plan, found, err := reader.CompiledPlan(ctx, runID)
		if err != nil || !found {
			return "", Run{}, CompiledPlanView{}, found, err
		}
		if journalID == plan.JournalID && run.RunID == runID && plan.RunID == runID &&
			run.RunRevision == plan.RunRevision && run.PlanHash == plan.PlanHash {
			return journalID, run, plan, true, nil
		}
	}
	return "", Run{}, CompiledPlanView{}, false,
		fmt.Errorf("TrainVM run changed while its telemetry snapshot was being captured")
}

func bindMetricDeclaration(event Event, declaration ObservabilityDeclaration) (MetricPoint, error) {
	point, err := metricFromEvent(event, nil)
	if err != nil {
		return MetricPoint{}, err
	}
	descriptor, found := declaration.Metric(point.Name)
	if !found {
		return MetricPoint{}, fmt.Errorf(
			"TrainVM metric event %q is not declared by the compiled plan", event.EventID)
	}
	if descriptor.Unit != point.Unit || descriptor.StepDomain != point.StepDomain {
		return MetricPoint{}, fmt.Errorf(
			"TrainVM metric event %q disagrees with its declaration", event.EventID)
	}
	point.Type = descriptor.Type
	point.Aggregation = descriptor.Aggregation
	point.Description = descriptor.Description
	return point, nil
}

// ProjectTelemetrySnapshot reads all generic telemetry families against one
// durable run prefix. NextSequence is the only cursor a caller needs to carry:
// when ReplayPending is false, it advances to the captured target even if that
// prefix ended in non-telemetry events.
func ProjectTelemetrySnapshot(
	ctx context.Context, reader ReadModel, runID string, after uint64, limit int,
) (TelemetrySnapshot, bool, error) {
	runID = strings.TrimSpace(runID)
	if runID == "" || len(runID) > 256 {
		return TelemetrySnapshot{}, false, &ValidationError{Message: "bounded run ID is required"}
	}
	if limit <= 0 || limit > 1_000 {
		limit = 250
	}

	journalID, run, plan, found, err := telemetryHeader(ctx, reader, runID)
	if err != nil || !found {
		return TelemetrySnapshot{}, found, err
	}
	if after > run.LastEventSeq {
		return TelemetrySnapshot{}, true,
			&ValidationError{Message: "telemetry cursor is ahead of the durable run prefix"}
	}
	declaration, err := ObservabilityFromCompiledPlan(plan)
	if err != nil {
		return TelemetrySnapshot{}, true, err
	}
	confirmedJournalID, err := reader.JournalID(ctx)
	if err != nil {
		return TelemetrySnapshot{}, true, err
	}
	if confirmedJournalID != journalID {
		return TelemetrySnapshot{}, true,
			fmt.Errorf("TrainVM journal changed while its telemetry snapshot was being captured")
	}
	snapshot := TelemetrySnapshot{
		JournalID: journalID, Run: run, Observability: declaration,
		AfterSequence: after, TargetSequence: run.LastEventSeq, NextSequence: after,
		Heartbeats:      make([]WorkerHeartbeatPoint, 0),
		Metrics:         make([]MetricPoint, 0),
		Artifacts:       make([]ObservableArtifact, 0),
		ExecutionPhases: make([]ExecutionPhaseReceiptPoint, 0),
	}
	if after == run.LastEventSeq {
		snapshot.NextSequence = run.LastEventSeq
		snapshot.CaughtUp = true
		return snapshot, true, nil
	}

	coldLoad := after == 0
	var events []Event
	if coldLoad {
		// A combined tail can be starved by a high-rate family (usually metrics
		// or heartbeats) and hide the latest artifact or phase. Read three
		// independently bounded tails against the same captured upper fence,
		// then merge them into one ordered snapshot and browser cursor.
		for _, query := range []struct {
			eventType             string
			limit                 int
			newestPerMetricSeries bool
		}{
			{eventType: "worker.heartbeat", limit: 1},
			{eventType: "metric.sampled", limit: maximumLiveMetricSeries + 1,
				newestPerMetricSeries: true},
			{eventType: "artifact.published", limit: limit},
			{eventType: "worker.execution_phase_receipted", limit: 64},
		} {
			page, queryErr := reader.Events(ctx, EventQuery{
				RunID: runID, After: 0, Through: run.LastEventSeq, Limit: query.limit,
				EventTypes:            []string{query.eventType},
				NewestFirst:           !query.newestPerMetricSeries,
				NewestPerMetricSeries: query.newestPerMetricSeries,
			})
			if queryErr != nil {
				return TelemetrySnapshot{}, true, queryErr
			}
			if len(page) > query.limit {
				return TelemetrySnapshot{}, true,
					fmt.Errorf("TrainVM authority exceeded the bounded telemetry tail")
			}
			if query.newestPerMetricSeries && len(page) > maximumLiveMetricSeries {
				return TelemetrySnapshot{}, true,
					fmt.Errorf("TrainVM run exceeds the %d-series live metric bound", maximumLiveMetricSeries)
			}
			events = append(events, page...)
		}
		sort.Slice(events, func(left, right int) bool {
			return events[left].Sequence < events[right].Sequence
		})
	} else {
		// Reserve one record as a replay sentinel while remaining inside the
		// authority's bounded 1,000-event page contract.
		if limit >= 1_000 {
			limit = 999
		}
		queryLimit := limit + 1
		events, err = reader.Events(ctx, EventQuery{
			RunID: runID, After: after, Through: run.LastEventSeq, Limit: queryLimit,
			EventTypes: append([]string(nil), telemetryEventTypes...),
		})
		if err != nil {
			return TelemetrySnapshot{}, true, err
		}
		if len(events) > queryLimit {
			return TelemetrySnapshot{}, true,
				fmt.Errorf("TrainVM authority exceeded the bounded telemetry page")
		}
	}
	previous := after
	for _, event := range events {
		validType := event.EventType == "worker.heartbeat" ||
			event.EventType == "metric.sampled" || event.EventType == "artifact.published" ||
			event.EventType == "worker.execution_phase_receipted"
		if !validType || event.RunID != runID || event.Sequence <= previous ||
			event.Sequence > run.LastEventSeq {
			return TelemetrySnapshot{}, true,
				fmt.Errorf("TrainVM authority returned an incoherent telemetry event page")
		}
		previous = event.Sequence
	}
	pending := !coldLoad && len(events) > limit
	if pending {
		events = events[:limit]
	}
	for _, event := range events {
		switch event.EventType {
		case "worker.heartbeat":
			point, parseErr := heartbeatFromEvent(event)
			if parseErr != nil {
				return TelemetrySnapshot{}, true, parseErr
			}
			snapshot.Heartbeats = append(snapshot.Heartbeats, point)
		case "metric.sampled":
			point, parseErr := bindMetricDeclaration(event, declaration)
			if parseErr != nil {
				return TelemetrySnapshot{}, true, parseErr
			}
			snapshot.Metrics = append(snapshot.Metrics, point)
		case "artifact.published":
			artifact, parseErr := artifactFromEvent(event)
			if parseErr != nil {
				return TelemetrySnapshot{}, true, parseErr
			}
			snapshot.Artifacts = append(snapshot.Artifacts, RedactArtifact(artifact))
		case "worker.execution_phase_receipted":
			phase, parseErr := executionPhaseReceiptFromEvent(event)
			if parseErr != nil {
				return TelemetrySnapshot{}, true, parseErr
			}
			snapshot.ExecutionPhases = append(snapshot.ExecutionPhases, phase)
		default:
			return TelemetrySnapshot{}, true,
				fmt.Errorf("TrainVM authority returned an unrequested telemetry event type")
		}
	}

	if pending {
		snapshot.NextSequence = events[len(events)-1].Sequence
		snapshot.ReplayPending = true
	} else {
		snapshot.NextSequence = run.LastEventSeq
		snapshot.CaughtUp = true
	}
	finalJournalID, err := reader.JournalID(ctx)
	if err != nil {
		return TelemetrySnapshot{}, true, err
	}
	if finalJournalID != journalID {
		return TelemetrySnapshot{}, true,
			fmt.Errorf("TrainVM journal changed while its telemetry events were being captured")
	}
	return snapshot, true, nil
}
