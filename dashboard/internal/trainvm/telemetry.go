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
	RunID      string
	After      uint64
	Limit      int
	EventTypes []string
}

func normalizeEventQuery(input EventQuery) (EventQuery, error) {
	input.RunID = strings.TrimSpace(input.RunID)
	if input.RunID == "" || len(input.RunID) > 256 {
		return EventQuery{}, &ValidationError{Message: "bounded run ID is required"}
	}
	if input.Limit <= 0 || input.Limit > 1_000 {
		input.Limit = 250
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
	Value          any               `json:"value"`
	Unit           string            `json:"unit"`
	StepDomain     string            `json:"step_domain"`
	Step           uint64            `json:"step"`
	SampleWeight   float64           `json:"sample_weight"`
	Labels         map[string]string `json:"labels"`
	ObservedAtNS   int64             `json:"observed_at_ns"`
	WorkerSequence uint64            `json:"worker_sequence"`
	OptimizerStep  *uint64           `json:"optimizer_step,omitempty"`
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

func validMetricScalar(value any) bool {
	switch typed := value.(type) {
	case json.Number:
		parsed, err := strconv.ParseFloat(string(typed), 64)
		return err == nil && !math.IsInf(parsed, 0) && !math.IsNaN(parsed)
	case string, bool:
		return true
	default:
		return false
	}
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
			event.EventType != "metric.sampled" || payload.Name == "" ||
			payload.StepDomain == "" || !validMetricScalar(payload.Value) ||
			payload.Labels == nil || event.OptimizerStep == nil ||
			*event.OptimizerStep != payload.Step ||
			math.IsInf(payload.SampleWeight, 0) || math.IsNaN(payload.SampleWeight) ||
			payload.SampleWeight <= 0 {
			return nil, fmt.Errorf("TrainVM metric event %q has a malformed payload", event.EventID)
		}
		result = append(result, MetricPoint{
			Sequence: event.Sequence, RunID: event.RunID, NodeID: event.NodeID,
			AttemptID: event.AttemptID, Name: payload.Name, Value: payload.Value,
			Unit: payload.Unit, StepDomain: payload.StepDomain, Step: payload.Step,
			SampleWeight: payload.SampleWeight, Labels: payload.Labels,
			ObservedAtNS: event.WallTimeNS, WorkerSequence: event.WorkerSequence,
			OptimizerStep: event.OptimizerStep,
		})
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
		if err := strictPayload(event.Payload, &payload); err != nil ||
			event.EventType != "artifact.published" || payload.ArtifactID == "" ||
			payload.LogicalName == "" ||
			payload.Kind == "" || payload.URI == "" ||
			payload.FingerprintAlgorithm == "" || payload.Fingerprint == "" ||
			!payload.Complete || payload.ProducerNodeID != event.NodeID ||
			payload.ProducerAttemptID != event.AttemptID ||
			payload.ParentArtifactIDs == nil || payload.PublishedAtNS < 0 ||
			payload.PublishedAtNS != event.WallTimeNS {
			return nil, fmt.Errorf("TrainVM artifact event %q has a malformed payload", event.EventID)
		}
		parents := make(map[string]struct{}, len(payload.ParentArtifactIDs))
		for _, parent := range payload.ParentArtifactIDs {
			if parent == "" || parent == payload.ArtifactID {
				return nil, fmt.Errorf("TrainVM artifact event %q has malformed lineage", event.EventID)
			}
			if _, duplicate := parents[parent]; duplicate {
				return nil, fmt.Errorf("TrainVM artifact event %q has duplicate lineage", event.EventID)
			}
			parents[parent] = struct{}{}
		}
		result = append(result, PublishedArtifact{
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
		})
	}
	return result, nil
}
